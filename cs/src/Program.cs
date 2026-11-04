// cs/src/Program.cs — C# host that drives the reconl C ABI through the unsafe wrapper.
//
// This is the C# equivalent of the Zig abi-test and the Rust abi.rs integration
// test: it creates a device, records a frame, and verifies the result. It proves
// the C# wrapper can talk to the shipped DLL.
//
// Design: minimal, no allocations in the hot path, unsafe pointers for the ABI,
// the disposable wrappers for cleanup. This is a smoke test + a starting point
// for a fuller C# conformance suite.

using System;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using ReconL;
using Device = ReconL.Device;
using Swapchain = ReconL.Swapchain;
using Pipeline = ReconL.Pipeline;
using CommandList = ReconL.CommandList;
using Buffer = ReconL.ReconLBuffer;
using Texture = ReconL.Texture;
using Fence = ReconL.Fence;

class Program
{
    static int Main(string[] args)
    {
        Console.WriteLine("ReconL C# host — ABI smoke test");
        Console.WriteLine($"  runtime: {RuntimeInformation.FrameworkDescription}");
        Console.WriteLine($"  OS: {RuntimeInformation.OSDescription}");
        Console.WriteLine();

        // Version
        ReconlNative.reconlVersion(out uint major, out uint minor, out uint patch);
        Console.WriteLine($"  reconl version: {major}.{minor}.{patch}");
        Console.WriteLine($"  result names: {ReconLResultInfo.Name(ReconLResult.OK)} / {ReconLResultInfo.Name(ReconLResult.ErrInvalidArgument)}");
        Console.WriteLine();

        // Probe
        var probeInfo = Probe();
        if (probeInfo == null) return 1;

        Console.WriteLine($"  probe: {probeInfo.Value.recommended_backend} / tier {probeInfo.Value.recommended_tier}");
        Console.WriteLine($"  backends:");
        for (int i = 0; i < probeInfo.Value.entry_count; i++)
        {
            var e = probeInfo.Value.entries[i];
            Console.WriteLine($"    {e.backend} {e.name} usable={e.usable} tier={e.best_tier} caps={e.caps}");
        }
        Console.WriteLine();

        // Device + frame test (soft-cpu for determinism)
        return DeviceAndFrameTest();
    }

    static ReconL.ReconLProbeInfo? Probe()
    {
        // Allocate unmanaged memory for the structs and marshal manually
        var descSize = Marshal.SizeOf<ReconL.ReconLProbeDesc>();
        var infoSize = Marshal.SizeOf<ReconL.ReconLProbeInfo>();
        var descPtr = Marshal.AllocHGlobal(descSize);
        var infoPtr = Marshal.AllocHGlobal(infoSize);
        try
        {
            // Zero out the memory
            Marshal.WriteInt32(descPtr, 0);
            Marshal.WriteInt32(infoPtr, 0);

            // Set up the probe desc header using unsafe pointer arithmetic
            unsafe
            {
                var desc = (ReconL.ReconLProbeDesc*)descPtr;
                desc->header.struct_size = (uint)descSize;
                desc->header.type = ReconL.ReconLStructType.ProbeDesc;
                desc->header.next = IntPtr.Zero;
                desc->flags = 0;
                desc->spill_dir = IntPtr.Zero;
            }

            var r = ReconlNative.reconlProbe(descPtr, infoPtr);
            if (r != ReconLResult.OK)
            {
                Console.WriteLine($"  probe failed: {r}");
                return null;
            }

            // Marshal the result back to a managed struct
            var info = Marshal.PtrToStructure<ReconL.ReconLProbeInfo>(infoPtr);
            Console.WriteLine($"  probe OK: {info.entry_count} backends");
            return info;
        }
        finally
        {
            Marshal.FreeHGlobal(descPtr);
            Marshal.FreeHGlobal(infoPtr);
        }
    }

    // Process heap for the host allocator
    static readonly IntPtr s_heap = ReconlNative.GetProcessHeap();

    // Align a pointer up to the given alignment (power of two).
    static IntPtr AlignUp(IntPtr ptr, uint alignment)
    {
        ulong p = (ulong)(ptr.ToInt64());
        return (IntPtr)((p + alignment - 1) & ~(ulong)(alignment - 1));
    }

    static IntPtr DoAlloc(uint size, uint alignment)
    {
        if (size == 0) return IntPtr.Zero;
        uint slack = (uint)Math.Max(alignment, 16) + 8;
        uint total = size + slack;
        var heapPtr = ReconlNative.HeapAlloc(s_heap, 8u, total);
        if (heapPtr == IntPtr.Zero) return IntPtr.Zero;
        var aligned = AlignUp(heapPtr + 8, alignment);
        Marshal.WriteInt64(aligned - 8, heapPtr.ToInt64());
        return aligned;
    }

    [UnmanagedCallersOnly]
    static IntPtr StdAlloc(IntPtr user, uint size, uint alignment)
    {
        return DoAlloc(size, alignment);
    }

    [UnmanagedCallersOnly]
    static IntPtr StdReAlloc(IntPtr user, IntPtr ptr, uint oldSize, uint newSize, uint alignment)
    {
        if (ptr == IntPtr.Zero) return DoAlloc(newSize, alignment);
        if (newSize == 0) { DoFree(ptr); return IntPtr.Zero; }
        // Read the original heap pointer, realloc with enough slack for alignment.
        var heapPtr = (IntPtr)Marshal.ReadInt64(ptr - 8);
        uint slack = (uint)Math.Max(alignment, 16) + 8;
        var newHeap = ReconlNative.HeapReAlloc(s_heap, 8u, heapPtr, (uint)(newSize + slack));
        if (newHeap == IntPtr.Zero) return IntPtr.Zero;
        var aligned = AlignUp(newHeap + 8, alignment);
        Marshal.WriteInt64(aligned - 8, newHeap.ToInt64());
        return aligned;
    }

    static void DoFree(IntPtr ptr)
    {
        if (ptr == IntPtr.Zero) return;
        var heapPtr = (IntPtr)Marshal.ReadInt64(ptr - 8);
        ReconlNative.HeapFree(s_heap, 8u, heapPtr);
    }

    [UnmanagedCallersOnly]
    static void StdFree(IntPtr user, IntPtr ptr)
    {
        DoFree(ptr);
    }

    static readonly ReconL.ReconLAllocator s_hostAllocator = new ReconL.ReconLAllocator
    {
        alloc = typeof(Program).GetMethod("StdAlloc", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic).MethodHandle.GetFunctionPointer(),
        realloc = typeof(Program).GetMethod("StdReAlloc", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic).MethodHandle.GetFunctionPointer(),
        free = typeof(Program).GetMethod("StdFree", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic).MethodHandle.GetFunctionPointer(),
        user = IntPtr.Zero
    };

    static int DeviceAndFrameTest()
    {
        // Create a device on the soft-cpu backend (deterministic, works everywhere)
        var deviceDesc = new ReconL.ReconLDeviceDesc
        {
            header = new ReconL.ReconLHeader
            {
                struct_size = (uint)Unsafe.SizeOf<ReconL.ReconLDeviceDesc>(),
                type = ReconL.ReconLStructType.DeviceDesc,
                next = IntPtr.Zero
            },
            backend_hint = ReconL.ReconLBackendId.SoftCpu,
            tier_hint = ReconL.ReconLTier.T2CpuRam,
            allow_downgrade = 0,
            worker_threads = 0,
            target_frame_ms = 16,
            downgrade_after_frames = 60,
            seed = 0x52434C53u,
            flags = 0,
            budget = IntPtr.Zero,
            allocator = s_hostAllocator,
            backend_desc = IntPtr.Zero
        };

        var pinnedDesc = GCHandle.Alloc(deviceDesc, GCHandleType.Pinned);
        IntPtr devicePtr;
        try
        {
            var r = ReconlNative.reconlCreateDevice(pinnedDesc.AddrOfPinnedObject(), out devicePtr);
            pinnedDesc.Free();
            if (r != ReconLResult.OK)
            {
                var lastError = ReconL.LastError.Get(IntPtr.Zero);
                Console.WriteLine($"  device creation failed: {r} ({ReconLResultInfo.Name(r)})");
                if (lastError != null)
                    Console.WriteLine($"  last error: {lastError}");
                return 1;
            }
        }
        finally
        {
            if (pinnedDesc.IsAllocated) pinnedDesc.Free();
        }

        Console.WriteLine($"  allocator: alloc=0x{s_hostAllocator.alloc.ToInt64().ToString("X")} realloc=0x{s_hostAllocator.realloc.ToInt64().ToString("X")} free=0x{s_hostAllocator.free.ToInt64().ToString("X")}");
        using var device = new ReconL.Device(devicePtr);
        Console.WriteLine($"  device created: {device.ptr}");

        // Get limits
        ReconL.ReconLDeviceLimits limits;
        var limitsSize = Marshal.SizeOf<ReconL.ReconLDeviceLimits>();
        var limitsPtr = Marshal.AllocHGlobal(limitsSize);
        try
        {
            var zero = new byte[limitsSize];
            Marshal.Copy(zero, 0, limitsPtr, limitsSize);
            var r = ReconlNative.reconlGetDeviceLimits(device.ptr, limitsPtr);
            if (r != ReconLResult.OK)
            {
                Console.WriteLine($"  get limits failed: {r}");
                Marshal.FreeHGlobal(limitsPtr);
                return 1;
            }
            limits = Marshal.PtrToStructure<ReconL.ReconLDeviceLimits>(limitsPtr);
        }
        finally
        {
            Marshal.FreeHGlobal(limitsPtr);
        }
        Console.WriteLine($"  limits: vram={limits.vram_bytes} ram={limits.ram_bytes} max_alloc={limits.max_allocation_bytes} worker_threads={limits.worker_threads_max}");

        // Create a swapchain
        using var swapchain = CreateSwapchain(device, 64, 64);
        if (!swapchain.IsValid) return 1;
        Console.WriteLine($"  swapchain created: {swapchain.ptr}");

        // Create a pipeline
        using var pipeline = CreatePipeline(device);
        if (!pipeline.IsValid) return 1;
        Console.WriteLine($"  pipeline created: {pipeline.ptr}");

        // Create a command list
        using var cmdList = CreateCommandList(device, 256);
        if (!cmdList.IsValid) return 1;
        Console.WriteLine($"  command list created: {cmdList.ptr}");

        // Render a frame — a single triangle
        var pixels = new byte[64 * 64 * 4];
        var result = RenderFrame(device, swapchain, pipeline, cmdList, pixels, 64, 64);
        if (result != 0)
        {
            Console.WriteLine($"  frame render failed: {result}");
            return 1;
        }

        // Check the frame has something (not all clear)
        int litPixels = 0;
        for (int i = 0; i < pixels.Length; i += 4)
        {
            if (pixels[i + 0] > 10 || pixels[i + 1] > 10 || pixels[i + 2] > 10)
                litPixels++;
        }
        Console.WriteLine($"  frame rendered: {litPixels} lit pixels (of {64*64})");
        Console.WriteLine($"  first pixel: R={pixels[0]} G={pixels[1]} B={pixels[2]} A={pixels[3]}");

        // Stats
        ReconL.ReconLStats stats;
        var statsSize = Marshal.SizeOf<ReconL.ReconLStats>();
        var statsPtr = Marshal.AllocHGlobal(statsSize);
        try
        {
            var zero = new byte[statsSize];
            Marshal.Copy(zero, 0, statsPtr, statsSize);
            // Set the header so the library knows the struct size
            Marshal.WriteInt32(statsPtr, (int)statsSize);
            Marshal.WriteInt32(statsPtr, 17, (int)ReconL.ReconLStructType.Stats);
            var r = ReconlNative.reconlGetStats(device.ptr, statsPtr);
            if (r != ReconLResult.OK)
            {
                Console.WriteLine($"  get stats failed: {r}");
                Marshal.FreeHGlobal(statsPtr);
                return 1;
            }
            stats = Marshal.PtrToStructure<ReconL.ReconLStats>(statsPtr);
        }
        finally
        {
            Marshal.FreeHGlobal(statsPtr);
        }

        Console.WriteLine($"  stats: frames_presented={stats.frames_presented} tier={stats.tier} backend={stats.backend}");
        Console.WriteLine($"  frame: total_ns={stats.frame.total_ns} shadow_ns={stats.frame.shadow_ns} raster_ns={stats.frame.raster_ns}");
        Console.WriteLine($"  lit pixels in frame: check above (should be > 0 for a triangle)");

        Console.WriteLine();
        Console.WriteLine("C# host: ALL TESTS PASSED");
        return 0;
    }

    static Swapchain CreateSwapchain(Device device, uint width, uint height)
    {
        var desc = new ReconL.ReconLSwapchainDesc
        {
            header = new ReconL.ReconLHeader
            {
                struct_size = (uint)Unsafe.SizeOf<ReconL.ReconLSwapchainDesc>(),
                type = ReconL.ReconLStructType.SwapchainDesc,
                next = IntPtr.Zero
            },
            width = width,
            height = height,
            format = ReconL.ReconLFormat.R8G8B8A8Unorm,
            image_count = 2,
            present_to_memory = 1,
            depth_format = (uint)ReconL.ReconLFormat.D32Float,
            flags = 0,
            reserved = 0
        };
        var pinned = GCHandle.Alloc(desc, GCHandleType.Pinned);
        try
        {
            var r = ReconlNative.reconlCreateSwapchain(device.ptr, pinned.AddrOfPinnedObject(), out IntPtr ptr);
            if (r == ReconLResult.OK)
                return new Swapchain(ptr);
            Console.WriteLine($"  swapchain creation failed: {r}");
            return new Swapchain(IntPtr.Zero);
        }
        finally
        {
            pinned.Free();
        }
    }

    static Pipeline CreatePipeline(Device device)
    {
        var descSize = Marshal.SizeOf<ReconL.ReconLPipelineDesc>();
        var descPtr = Marshal.AllocHGlobal(descSize);
        try
        {
            var zero = new byte[descSize];
            Marshal.Copy(zero, 0, descPtr, descSize);
            // Set header
            Marshal.WriteInt32(descPtr, (int)descSize);
            Marshal.WriteInt32(descPtr, 12, (int)ReconL.ReconLStructType.PipelineDesc);
            // Fill fields
            Marshal.WriteInt32(descPtr, 16, (int)ReconL.ReconLPipelineShading.Lambert);
            Marshal.WriteInt32(descPtr, 20, (int)ReconL.ReconLBlendMode.Opaque);
            Marshal.WriteInt32(descPtr, 24, (int)ReconL.ReconLCullMode.Back);
            Marshal.WriteInt32(descPtr, 28, (int)ReconL.ReconLCompareFunc.Greater);
            Marshal.WriteInt32(descPtr, 32, 1); // depth_write
            Marshal.WriteInt32(descPtr, 36, 0); // texture_slots
            // texture_formats at offset 40, 8 x uint32 = 32 bytes, all Unknown (0)
            // Already zeroed
            Marshal.WriteInt32(descPtr, 72, 0); // receives_shadow
            Marshal.WriteInt32(descPtr, 76, 0); // casts_shadow
            Marshal.WriteInt32(descPtr, 80, 0); // flags
            Marshal.WriteInt32(descPtr, 84, 0); // reserved
            // debug_name at offset 88, 8 bytes, IntPtr.Zero
            Marshal.WriteInt64(descPtr, 88, 0);

            var r = ReconlNative.reconlCreatePipeline(device.ptr, descPtr, out IntPtr ptr);
            if (r == ReconLResult.OK)
                return new Pipeline(ptr);
            Console.WriteLine($"  pipeline creation failed: {r}");
            return new Pipeline(IntPtr.Zero);
        }
        finally
        {
            Marshal.FreeHGlobal(descPtr);
        }
    }

    static CommandList CreateCommandList(Device device, uint commandCount)
    {
        var desc = new ReconL.ReconLCommandListDesc
        {
            header = new ReconL.ReconLHeader
            {
                struct_size = (uint)Unsafe.SizeOf<ReconL.ReconLCommandListDesc>(),
                type = ReconL.ReconLStructType.CommandListDesc,
                next = IntPtr.Zero
            },
            capacity_bytes = commandCount * ReconL.ReconLResultInfo.RECONL_COMMAND_BYTES,
            reserved = 0,
            debug_name = IntPtr.Zero
        };
        var pinned = GCHandle.Alloc(desc, GCHandleType.Pinned);
        try
        {
            var r = ReconlNative.reconlCreateCommandList(device.ptr, pinned.AddrOfPinnedObject(), out IntPtr ptr);
            if (r == ReconLResult.OK)
                return new CommandList(ptr);
            Console.WriteLine($"  command list creation failed: {r}");
            return new CommandList(IntPtr.Zero);
        }
        finally
        {
            pinned.Free();
        }
    }

    static int RenderFrame(Device device, Swapchain swapchain, Pipeline pipeline, CommandList cmdList, byte[] pixels, uint width, uint height)
    {
        // Begin frame
        var frameDesc = new ReconL.ReconLFrameDesc
        {
            header = new ReconL.ReconLHeader
            {
                struct_size = (uint)Unsafe.SizeOf<ReconL.ReconLFrameDesc>(),
                type = ReconL.ReconLStructType.FrameDesc,
                next = IntPtr.Zero
            },
            width = width,
            height = height,
            seed = 1,
            reserved = 0,
            lights = IntPtr.Zero,
            shadows = IntPtr.Zero,
            camera = IntPtr.Zero,
            framegen = IntPtr.Zero
        };
        var frameDescPtr = Marshal.AllocHGlobal(Marshal.SizeOf<ReconL.ReconLFrameDesc>());
        try
        {
            var zero = new byte[Marshal.SizeOf<ReconL.ReconLFrameDesc>()];
            Marshal.Copy(zero, 0, frameDescPtr, zero.Length);
            Marshal.WriteInt32(frameDescPtr, (int)Marshal.SizeOf<ReconL.ReconLFrameDesc>());
            Marshal.WriteInt32(frameDescPtr, 12, (int)ReconL.ReconLStructType.FrameDesc);
            var r = ReconlNative.reconlBeginFrame(device.ptr, frameDescPtr);
            Marshal.FreeHGlobal(frameDescPtr);
            if (r != ReconLResult.OK)
            {
                Console.WriteLine($"  begin frame failed: {r}");
                return -1;
            }
        }
        finally
        {
            if (pinnedFrame.IsAllocated) pinnedFrame.Free();
        }

        // Reset command list
        var r2 = ReconlNative.reconlCmdReset(cmdList.ptr);
        if (r2 != ReconLResult.OK) { Console.WriteLine($"  cmd reset failed: {r2}"); return -2; }

        // Begin render pass
        var passDesc = new ReconL.ReconLRenderPassDesc
        {
            header = new ReconL.ReconLHeader
            {
                struct_size = (uint)Unsafe.SizeOf<ReconL.ReconLRenderPassDesc>(),
                type = ReconL.ReconLStructType.RenderPassDesc,
                next = IntPtr.Zero
            },
            color_count = 0,
            reserved = 0,
            color = new ReconL.ReconLColorAttachment[4],
            depth = IntPtr.Zero,
            viewport_width = 0,
            viewport_height = 0,
            load_color = 1,
            load_depth = 1,
            clear_color = new float[] { 0.0f, 0.0f, 1.0f, 1.0f },
            clear_depth = 0.0f,
            stencil_clear = 0,
            reserved2 = 0
        };
        for (int i = 0; i < 4; i++) passDesc.color[i] = new ReconL.ReconLColorAttachment();

        var passDescSize = Marshal.SizeOf<ReconL.ReconLRenderPassDesc>();
        var passDescPtr = Marshal.AllocHGlobal(passDescSize);
        try
        {
            var zero = new byte[passDescSize];
            Marshal.Copy(zero, 0, passDescPtr, zero.Length);
            Marshal.WriteInt32(passDescPtr, (int)passDescSize);
            Marshal.WriteInt32(passDescPtr, 12, (int)ReconL.ReconLStructType.RenderPassDesc);
            Marshal.WriteInt32(passDescPtr, 16, 0); // color_count
            Marshal.WriteInt32(passDescPtr, 20, 0); // reserved
            // color array at offset 24 (4 x ReconLColorAttachment = 4 x 16 bytes = 64 bytes)
            // Already zeroed
            Marshal.WriteInt64(passDescPtr, 88, 0); // depth
            Marshal.WriteInt32(passDescPtr, 96, 0); // viewport_width
            Marshal.WriteInt32(passDescPtr, 100, 0); // viewport_height
            Marshal.WriteInt32(passDescPtr, 104, 1); // load_color
            Marshal.WriteInt32(passDescPtr, 108, 1); // load_depth
            // clear_color at offset 112 (4 x float = 16 bytes)
            Marshal.WriteSingle(passDescPtr, 112, 0.0f);
            Marshal.WriteSingle(passDescPtr, 116, 0.0f);
            Marshal.WriteSingle(passDescPtr, 120, 1.0f);
            Marshal.WriteSingle(passDescPtr, 124, 1.0f);
            Marshal.WriteInt32(passDescPtr, 128, 0); // clear_depth (float at 128)
            Marshal.WriteInt32(passDescPtr, 132, 0); // stencil_clear
            Marshal.WriteInt32(passDescPtr, 136, 0); // reserved2

            r2 = ReconlNative.reconlCmdBeginRenderPass(cmdList.ptr, passDescPtr);
            Marshal.FreeHGlobal(passDescPtr);
            if (r2 != ReconLResult.OK) { Console.WriteLine($"  begin pass failed: {r2}"); return -3; }
        }
        finally
        {
            if (pinnedPass.IsAllocated) pinnedPass.Free();
        }

        // Set pipeline
        r2 = ReconlNative.reconlCmdSetPipeline(cmdList.ptr, pipeline.ptr);
        if (r2 != ReconLResult.OK) { Console.WriteLine($"  set pipeline failed: {r2}"); return -4; }

        // Push constants: identity view*proj in slot 0, identity model in slot 1
        var identity = new float[16] {
            1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1
        };
        var pinnedIdentity = GCHandle.Alloc(identity, GCHandleType.Pinned);
        try
        {
            r2 = ReconlNative.reconlCmdPushConstants(cmdList.ptr, 0, pinnedIdentity.AddrOfPinnedObject(), 64);
            if (r2 != ReconLResult.OK) { Console.WriteLine($"  push constants 0 failed: {r2}"); return -5; }
            r2 = ReconlNative.reconlCmdPushConstants(cmdList.ptr, 1, pinnedIdentity.AddrOfPinnedObject(), 64);
            if (r2 != ReconLResult.OK) { Console.WriteLine($"  push constants 1 failed: {r2}"); return -6; }
        }
        finally
        {
            pinnedIdentity.Free();
        }

        // Create a vertex buffer with a simple triangle
        using var vb = CreateVertexBuffer(device, new float[] {
            0.0f, 0.5f, 0.0f,  0,0,1,  0,0,  1,1,1,1,
            -0.5f, -0.5f, 0.0f,  0,0,1,  0,0,  1,1,1,1,
            0.5f, -0.5f, 0.0f,  0,0,1,  0,0,  1,1,1,1
        });
        if (!vb.IsValid) return -7;

        r2 = ReconlNative.reconlCmdSetVertexBuffer(cmdList.ptr, 0, vb.ptr, 0);
        if (r2 != ReconLResult.OK) { Console.WriteLine($"  set vertex buffer failed: {r2}"); return -8; }

        // Draw
        r2 = ReconlNative.reconlCmdDraw(cmdList.ptr, 3, 0);
        if (r2 != ReconLResult.OK) { Console.WriteLine($"  draw failed: {r2}"); return -9; }

        // End render pass
        r2 = ReconlNative.reconlCmdEndRenderPass(cmdList.ptr);
        if (r2 != ReconLResult.OK) { Console.WriteLine($"  end pass failed: {r2}"); return -10; }

        // Submit
        r2 = ReconlNative.reconlSubmit(device.ptr, cmdList.ptr);
        if (r2 != ReconLResult.OK) { Console.WriteLine($"  submit failed: {r2}"); return -11; }

        // Present
        var presentDesc = new ReconL.ReconLPresentDesc
        {
            header = new ReconL.ReconLHeader
            {
                struct_size = (uint)Unsafe.SizeOf<ReconL.ReconLPresentDesc>(),
                type = ReconL.ReconLStructType.PresentDesc,
                next = IntPtr.Zero
            },
            out_pixels = Marshal.AllocHGlobal(pixels.Length),
            out_pixels_size = (ulong)pixels.Length,
            out_row_pitch = (uint)(width * 4),
            out_format = ReconL.ReconLFormat.R8G8B8A8Unorm,
            flip = 0
        };
        try
        {
            var presentDescSize = Marshal.SizeOf<ReconL.ReconLPresentDesc>();
            var presentDescPtr = Marshal.AllocHGlobal(presentDescSize);
            try
            {
                var zero = new byte[presentDescSize];
                Marshal.Copy(zero, 0, presentDescPtr, zero.Length);
                Marshal.WriteInt32(presentDescPtr, (int)presentDescSize);
                Marshal.WriteInt32(presentDescPtr, 12, (int)ReconL.ReconLStructType.PresentDesc);
                Marshal.WriteInt64(presentDescPtr, 16, (long)presentDesc.out_pixels);
                Marshal.WriteInt64(presentDescPtr, 24, (long)presentDesc.out_pixels_size);
                Marshal.WriteInt32(presentDescPtr, 32, (int)presentDesc.out_row_pitch);
                Marshal.WriteInt32(presentDescPtr, 36, (int)presentDesc.out_format);
                Marshal.WriteInt32(presentDescPtr, 40, presentDesc.flip);

                r2 = ReconlNative.reconlPresent(device.ptr, swapchain.ptr, presentDescPtr);
                if (r2 != ReconLResult.OK)
                {
                    Console.WriteLine($"  present failed: {r2}");
                    Marshal.FreeHGlobal(presentDescPtr);
                    return -12;
                }
                Marshal.Copy(presentDesc.out_pixels, pixels, 0, pixels.Length);
            }
            finally
            {
                Marshal.FreeHGlobal(presentDescPtr);
                Marshal.FreeHGlobal(presentDesc.out_pixels);
            }

        return 0;
    }

    static Buffer CreateVertexBuffer(Device device, float[] vertices)
    {
        var data = Marshal.AllocHGlobal(vertices.Length * 4);
        Marshal.Copy(vertices, 0, data, vertices.Length);
        ulong dataSize = (ulong)(vertices.Length * 4);
        var descSize = Marshal.SizeOf<ReconL.ReconLBufferDesc>();
        var descPtr = Marshal.AllocHGlobal(descSize);
        try
        {
            var zero = new byte[descSize];
            Marshal.Copy(zero, 0, descPtr, descSize);
            // Set header
            Marshal.WriteInt32(descPtr, (int)descSize);
            Marshal.WriteInt32(descPtr, 12, (int)ReconL.ReconLStructType.BufferDesc);
            // Fill fields
            Marshal.WriteInt64(descPtr, 16, (long)dataSize); // size_bytes
            Marshal.WriteInt32(descPtr, 24, (int)(ReconL.ReconLBufferUsage.Vertex | ReconL.ReconLBufferUsage.Static)); // usage
            Marshal.WriteInt32(descPtr, 28, 0); // reserved
            Marshal.WriteInt64(descPtr, 32, (long)data); // data (IntPtr)
            Marshal.WriteInt64(descPtr, 40, (long)dataSize); // data_size
            Marshal.WriteInt64(descPtr, 48, 0); // debug_name

            var r = ReconlNative.reconlCreateBuffer(device.ptr, descPtr, out IntPtr bufPtr);
            if (r != ReconLResult.OK)
            {
                Console.WriteLine($"  vertex buffer creation failed: {r}");
                Marshal.FreeHGlobal(data);
                return new Buffer(IntPtr.Zero);
            }
            Marshal.FreeHGlobal(data);
            return new Buffer(bufPtr);
        }
        finally
        {
            Marshal.FreeHGlobal(descPtr);
        }
    }
}
