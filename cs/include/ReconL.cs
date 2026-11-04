// ReconL.cs — C# unsafe wrapper over the reconl.h C ABI.
//
// Design: direct DllImport to reconl.dll/libreconl.so with exactly the C ABI
// signatures. No marshaling layers, no GC heap in the hot path, no SafeHandle
// overhead in the frame loop.
//
// Handles are IntPtr. The wrapper provides both raw-pointer access and a
// disposable wrapper. Neither allocates in the hot path.

using System;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;

namespace ReconL
{
    // ---- result codes ----
    public enum ReconLResult : int
    {
        OK = 0,
        ErrInvalidArgument = -1,
        ErrOutOfMemory = -2,
        ErrNotSupported = -3,
        ErrBackendUnavailable = -4,
        ErrBudgetExceeded = -5,
        ErrDeviceLost = -6,
        ErrInvalidHandle = -7,
        ErrStructSize = -8,
        ErrWrongStructType = -9,
        ErrAbiVersion = -10,
        ErrFrameInProgress = -11,
        ErrNoFrame = -12,
        ErrNotReady = -13,
        ErrIO = -14,
        ErrCorruptCache = -15,
        ErrDegraded = -16,
        ErrPanic = -17,
        ErrEmptyFrame = -18
    }

    public static class ReconLResultInfo
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static string Name(ReconLResult r)
        {
            var p = ReconlNative.reconlResultName(r);
            return p == IntPtr.Zero ? "unknown" : Marshal.PtrToStringAnsi(p);
        }

        public const uint RECONL_COMMAND_BYTES = 64;
    }

    // ---- struct types ----
    public enum ReconLStructType : uint
    {
        None = 0,
        Allocator = 1,
        ProbeDesc = 2,
        ProbeInfo = 3,
        MemoryBudget = 4,
        DeviceLimits = 5,
        DeviceDesc = 6,
        BufferDesc = 7,
        TextureDesc = 8,
        SamplerDesc = 9,
        PipelineDesc = 10,
        CommandListDesc = 11,
        RenderPassDesc = 12,
        SwapchainDesc = 13,
        FrameDesc = 14,
        PresentDesc = 15,
        ShadowConfig = 16,
        Stats = 17,
        Light = 18,
        LightList = 19,
        ErrorInfo = 20,
        Camera = 21,
        FrameGen = 22,
        SoftCpuDesc = 64,
        NullDesc = 65,
        D3D11Desc = 66,
        AdapterInfo = 67,
        ComputeDesc = 68
    }

    public enum ReconLBackendId : uint
    {
        None = 0,
        SoftCpu = 1,
        Null = 2,
        D3D11 = 3,
        D3D12 = 4,
        Vulkan = 5,
        GL = 6,
        Metal = 7,
        WebGPU = 8,
        WasmWebGL2 = 9,
        GpuCompute = 10
    }

    public enum ReconLAdapterType : uint
    {
        Unknown = 0,
        Integrated = 1,
        Discrete = 2
    }

    public enum ReconLAdapterPreference : uint
    {
        Auto = 0,
        Integrated = 1,
        Discrete = 2,
        Index = 3,
        Luid = 4
    }

    public enum ReconLTier : uint
    {
        T0GpuDiscrete = 0,
        T1GpuShared = 1,
        T2CpuRam = 2,
        T3CpuThrifty = 3,
        T4OutOfCore = 4
    }

    public enum ReconLTierReason : uint
    {
        StartupProbe = 0,
        HostRequest = 1,
        AllocationOverBudget = 2,
        FrameTimeOverTarget = 3,
        DeviceRemoved = 4,
        DeviceLost = 5,
        MemoryPressure = 6,
        DiskCacheFull = 7,
        Build = 8,
        NoGpuApi = 9,
        Recovery = 10
    }

    [Flags]
    public enum ReconLCaps : uint
    {
        None = 0,
        Textures = 1u << 0,
        Mipmaps = 1u << 1,
        Shadows = 1u << 2,
        Pcf5x5 = 1u << 3,
        PcssLite = 1u << 4,
        DiskSpill = 1u << 5,
        Multithread = 1u << 6,
        OutOfCore = 1u << 7,
        CachedCascade = 1u << 8,
        SimdSse2 = 1u << 9,
        SimdAvx2 = 1u << 10,
        SimdNeon = 1u << 11,
        SimdWasm128 = 1u << 12,
        Compute = 1u << 13,
        PresentToMemory = 1u << 14
    }

    [Flags]
    public enum ReconLAllowDowngrade : uint
    {
        None = 0,
        Tier = 1u << 0,
        Shadows = 1u << 1,
        Resolution = 1u << 2,
        Disk = 1u << 3,
        FreezeCache = 1u << 4,
        All = 0x1Fu
    }

    public enum ReconLLogLevel : uint
    {
        Off = 0,
        Error = 1,
        Warn = 2,
        Info = 3,
        Debug = 4,
        Trace = 5
    }

    public enum ReconLFormat : uint
    {
        Unknown = 0,
        R8G8B8A8Unorm = 1,
        B8G8R8A8Unorm = 2,
        R8G8B8A8Srgb = 3,
        R32Float = 4,
        R32G32Float = 5,
        R32G32B32Float = 6,
        R32G32B32A32Float = 7,
        D32Float = 8
    }

    [Flags]
    public enum ReconLBufferUsage : uint
    {
        Vertex = 1u << 0,
        Index = 1u << 1,
        Uniform = 1u << 2,
        Upload = 1u << 3,
        Readback = 1u << 4,
        Static = 1u << 5,
        Dynamic = 1u << 6
    }

    [Flags]
    public enum ReconLTextureUsage : uint
    {
        Sampled = 1u << 0,
        RenderTarget = 1u << 1,
        DepthStencil = 1u << 2,
        ShadowMap = 1u << 3,
        HostReadback = 1u << 4,
        MipMapped = 1u << 5,
        Static = 1u << 6
    }

    public enum ReconLIndexFormat : uint
    {
        Uint16 = 0,
        Uint32 = 1
    }

    public enum ReconLSamplerFilter : uint
    {
        Nearest = 0,
        Linear = 1,
        NearestMipLinear = 2,
        LinearMipLinear = 3
    }

    public enum ReconLSamplerWrap : uint
    {
        Repeat = 0,
        Clamp = 1,
        Mirror = 2
    }

    public enum ReconLBlendMode : uint
    {
        Opaque = 0,
        Alpha = 1,
        Additive = 2,
        Multiply = 3
    }

    public enum ReconLCompareFunc : uint
    {
        Less = 0,
        Greater = 1
    }

    public enum ReconLCullMode : uint
    {
        None = 0,
        Back = 1,
        Front = 2
    }

    public enum ReconLLightType : uint
    {
        Directional = 0,
        Spot = 1,
        Point = 2
    }

    public enum ReconLShadowFilter : uint
    {
        Hard = 0,
        Pcf3x3 = 1,
        Pcf5x5 = 2,
        PcssLite = 3
    }

    public enum ReconLFrameState : uint
    {
        Idle = 0,
        Open = 1,
        Submitted = 2,
        Presented = 3
    }

    public enum ReconLShadowEvent : uint
    {
        None = 0,
        CascadeDropped = 1,
        MapBudgetClamped = 2,
        CacheHit = 3,
        CacheMiss = 4,
        CacheCorrupt = 5,
        FallbackUnshadowed = 6,
        FilterDowngraded = 7,
        FrozenCascade = 8
    }

    // ---- base header (renamed from 'base' to avoid C# keyword conflict) ----
    [StructLayout(LayoutKind.Sequential)]
    public struct ReconLHeader
    {
        public uint struct_size;
        public ReconLStructType type;
        public IntPtr next;
    }

    // ---- allocator ----
    [StructLayout(LayoutKind.Sequential)]
    public struct ReconLAllocator
    {
        public IntPtr alloc;
        public IntPtr realloc;
        public IntPtr free;
        public IntPtr user;
    }

    // ---- probe ----
    [StructLayout(LayoutKind.Sequential, Size = 260)]
    public struct ReconLProbeDesc
    {
        public ReconLHeader header;
        public uint flags;
        public IntPtr spill_dir;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ReconLBackendProbe
    {
        public ReconLBackendId backend;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 64)]
        public string name;
        public int usable;
        public uint caps;
        public ReconLTier best_tier;
        public ulong vram_bytes;
        public ulong ram_bytes;
        public uint max_cascades;
        public ulong shadow_texel_budget;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 64)]
        public string device_name;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 192)]
        public string note;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ReconLProbeInfo
    {
        public ReconLHeader header;
        public uint entry_count;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
        public ReconLBackendProbe[] entries;
        public ReconLTier recommended_tier;
        public uint recommended_backend;
    }

    [StructLayout(LayoutKind.Sequential, Size = 260)]
    public struct ReconLAdapterInfo
    {
        public ReconLHeader header;
        public ReconLBackendId backend;
        public ReconLAdapterType adapter_type;
        public uint usable;
        public uint vendor_id;
        public uint device_id;
        public uint reserved;
        public ulong adapter_luid;
        public ulong dedicated_video_memory;
        public ulong shared_system_memory;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 64)]
        public string name;
    }

    // ---- limits ----
    [StructLayout(LayoutKind.Sequential, Size = 256)]
    public struct ReconLDeviceLimits
    {
        public ReconLHeader header;
        public ulong vram_bytes;
        public ulong ram_bytes;
        public ulong disk_bytes;
        public ulong max_allocation_bytes;
        public ulong shadow_texel_budget_bytes;
        public uint worker_threads_max;
        public uint tile_size_min;
        public uint max_cascades;
        public uint max_lights;
        public uint caps;
        public ReconLBackendId backend;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 64)]
        public string device_name;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 64)]
        public string driver;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ReconLMemoryBudget
    {
        public ReconLHeader header;
        public ulong vram_cap_bytes;
        public ulong ram_cap_bytes;
        public ulong disk_cap_bytes;
        public uint allow_disk_spill;
        public uint reserved;
        public IntPtr spill_dir;
    }

    // ---- device ----
    [StructLayout(LayoutKind.Sequential)]
    public struct ReconLDeviceDesc
    {
        public ReconLHeader header;
        public ReconLBackendId backend_hint;
        public ReconLTier tier_hint;
        public uint allow_downgrade;
        public uint worker_threads;
        public uint target_frame_ms;
        public uint downgrade_after_frames;
        public uint seed;
        public uint flags;
        public IntPtr budget;
        public ReconLAllocator allocator;
        public IntPtr backend_desc;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ReconLSwapchainDesc
    {
        public ReconLHeader header;
        public uint width;
        public uint height;
        public ReconLFormat format;
        public uint image_count;
        public uint present_to_memory;
        public uint depth_format;
        public uint flags;
        public uint reserved;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ReconLBufferDesc
    {
        public ReconLHeader header;
        public ulong size_bytes;
        public uint usage;
        public uint reserved;
        public IntPtr data;
        public ulong data_size;
        public IntPtr debug_name;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ReconLTextureDesc
    {
        public ReconLHeader header;
        public uint width;
        public uint height;
        public uint mip_levels;
        public uint array_layers;
        public ReconLFormat format;
        public uint usage;
        public uint reserved;
        public IntPtr debug_name;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ReconLTextureLevel
    {
        public ReconLHeader header;
        public uint mip;
        public uint layer;
        public uint row_pitch;
        public uint row_count;
        public IntPtr data;
        public ulong data_size;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ReconLSamplerDesc
    {
        public ReconLHeader header;
        public ReconLSamplerFilter filter;
        public ReconLSamplerWrap wrap_u;
        public ReconLSamplerWrap wrap_v;
        public ReconLSamplerWrap wrap_w;
        public uint max_anisotropy;
        public float mip_lod_bias;
        public uint reserved;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ReconLVertex
    {
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public float[] position;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public float[] normal;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public float[] uv;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 4)]
        public float[] color;
    }

    public enum ReconLPipelineShading : uint
    {
        Unlit = 0,
        Lambert = 1,
        Textured = 2,
        TexturedLambert = 3
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ReconLPipelineDesc
    {
        public ReconLHeader header;
        public ReconLPipelineShading shading;
        public ReconLBlendMode blend;
        public ReconLCullMode cull;
        public ReconLCompareFunc depth_compare;
        public uint depth_write;
        public uint texture_slots;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
        public ReconLFormat[] texture_formats;
        public uint receives_shadow;
        public uint casts_shadow;
        public uint flags;
        public uint reserved;
        public IntPtr debug_name;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ReconLLight
    {
        public ReconLHeader header;
        public ReconLLightType type;
        public uint cast_shadow;
        public uint shadow_priority;
        public uint shadow_quality;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public float[] position;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public float[] direction;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public float[] color;
        public float intensity;
        public float range;
        public float cone_inner_deg;
        public float cone_outer_deg;
        public uint reserved;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ReconLLightList
    {
        public ReconLHeader header;
        public uint count;
        public uint reserved;
        public IntPtr lights;
    }

    [StructLayout(LayoutKind.Sequential, Size = 120)]
    public struct ReconLShadowConfig
    {
        public ReconLHeader header;
        public uint enabled;
        public uint cascade_count;
        public ulong texel_budget_bytes;
        public ReconLShadowFilter filter;
        public float max_distance;
        public float blend_band;
        public float normal_bias;
        public float depth_bias;
        public float slope_bias;
        public float resolution_scale;
        public uint allow_disk_cache;
        public uint freeze_static_cascade;
        public uint refresh_interval_frames;
        public uint reserved;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ReconLShadowStats
    {
        public ReconLHeader header;
        public uint cascades_active;
        public uint map_width;
        public uint map_height;
        public uint map_bytes;
        public ReconLShadowFilter filter_active;
        public ReconLShadowFilter filter_requested;
        public ulong shadow_pass_ns;
        public ulong cascade_fit_ns;
        public ulong cache_bytes_read;
        public ulong cache_bytes_hit;
        public uint cache_hits;
        public uint cache_misses;
        public uint cache_corrupt;
        public uint fail_safe_unshadowed;
        public uint frozen_cascades;
        public uint shadowed_lights;
        public uint point_lights_capped;
    }

    [StructLayout(LayoutKind.Sequential, Size = 120)]
    public struct ReconLMemoryStats
    {
        public ReconLHeader header;
        public ulong ram_resident_bytes;
        public ulong ram_budget_bytes;
        public ulong ram_peak_bytes;
        public ulong spill_resident_bytes;
        public ulong spill_disk_bytes;
        public ulong spill_disk_cap_bytes;
        public ulong spill_evicted_bytes;
        public ulong spill_cache_bytes;
        public uint spill_entries;
        public uint spill_evictions;
        public uint spill_compactions;
        public uint spill_recovered_entries;
        public uint spill_errors;
        public uint host_alloc_calls;
        public ulong host_alloc_bytes;
        public uint host_free_calls;
        public uint reserved;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ReconLFrameTiming
    {
        public ReconLHeader header;
        public ulong frame_index;
        public ulong total_ns;
        public ulong shadow_ns;
        public ulong raster_ns;
        public ulong bin_ns;
        public ulong upload_ns;
        public ulong spill_wait_ns;
        public ulong min_ns;
        public ulong avg_ns;
        public ulong max_ns;
        public uint tiles_total;
        public uint tiles_rendered;
        public uint triangles_in;
        public uint triangles_binned;
        public uint triangles_culled;
        public uint pixels_shaded;
        public uint worker_threads;
        public float resolution_scale;
        public uint reserved;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ReconLDowngrade
    {
        public ReconLTier from;
        public ReconLTier to;
        public ReconLTierReason reason;
        public ulong frame_index;
        public ulong at_ns;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 192)]
        public string detail;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ReconLStatsDesc
    {
        public ReconLHeader header;
        public uint include_downgrades;
        public uint reserved;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ReconLFrameGenStats
    {
        public uint ready;
        public uint generated;
        public ulong generated_ns;
        public float last_ahead;
        public uint reserved;
        public uint reserved2;
    }

    [StructLayout(LayoutKind.Sequential, Size = 400)]
    public struct ReconLStats
    {
        public ReconLHeader header;
        public ReconLBackendId backend;
        public ReconLTier tier;
        public ReconLTierReason tier_reason;
        public uint tier_locked;
        public uint frames_presented;
        public uint frames_dropped;
        public uint downgrade_count;
        public uint downgrade_capacity;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
        public ReconLDowngrade[] downgrades;
        public ReconLMemoryStats memory;
        public ReconLShadowStats shadows;
        public ReconLFrameTiming frame;
        public uint caps;
        public uint failures;
        public uint safe_path_events;
        public uint audit_divergences;
        public ReconLResult last_result;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 192)]
        public string tier_reason_text;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 64)]
        public string device_name;
        public ReconLFrameGenStats framegen;
    }

    [StructLayout(LayoutKind.Sequential, Size = 400)]
    public struct ReconLErrorInfo
    {
        public ReconLHeader header;
        public ReconLResult result;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 192)]
        public string message;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 260)]
        public string file;
        public uint line;
        public uint function_name_index;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 64)]
        public string function;
    }

    // handles — IntPtr
    public struct ReconLDevicePtr { public IntPtr ptr; }
    public struct ReconLBufferPtr { public IntPtr ptr; }
    public struct ReconLTexturePtr { public IntPtr ptr; }
    public struct ReconLPipelinePtr { public IntPtr ptr; }
    public struct ReconLSwapchainPtr { public IntPtr ptr; }
    public struct ReconLCommandListPtr { public IntPtr ptr; }
    public struct ReconLFencePtr { public IntPtr ptr; }

    // ---- command list ----
    [StructLayout(LayoutKind.Sequential)]
    public struct ReconLCommandListDesc
    {
        public ReconLHeader header;
        public uint capacity_bytes;
        public uint reserved;
        public IntPtr debug_name;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ReconLColorAttachment
    {
        public IntPtr texture;
        public IntPtr resolve;
        public uint mip;
        public uint layer;
    }

    [StructLayout(LayoutKind.Sequential, Size = 256)]
    public struct ReconLRenderPassDesc
    {
        public ReconLHeader header;
        public uint color_count;
        public uint reserved;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 4)]
        public ReconLColorAttachment[] color;
        public IntPtr depth;
        public uint viewport_width;
        public uint viewport_height;
        public uint load_color;
        public uint load_depth;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 4)]
        public float[] clear_color;
        public float clear_depth;
        public uint stencil_clear;
        public uint reserved2;
    }

    // ---- camera ----
    [StructLayout(LayoutKind.Sequential, Size = 72)]
    public struct ReconLCamera
    {
        public ReconLHeader header;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
        public float[] view;
        public float fov_y_deg;
        public float near;
        public float far;
        public float reserved;
    }

    // ---- frame gen ----
    [StructLayout(LayoutKind.Sequential)]
    public struct ReconLFrameGenDesc
    {
        public ReconLHeader header;
        public uint enabled;
        public uint reserved;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ReconLFrameDesc
    {
        public ReconLHeader header;
        public uint width;
        public uint height;
        public uint seed;
        public uint reserved;
        public IntPtr lights;
        public IntPtr shadows;
        public IntPtr camera;
        public IntPtr framegen;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ReconLPresentDesc
    {
        public ReconLHeader header;
        public IntPtr out_pixels;
        public ulong out_pixels_size;
        public uint out_row_pitch;
        public ReconLFormat out_format;
        public uint flip;
    }

    // ---- native imports ----
    internal static class ReconlNative
    {
        private const string LibName = "reconl";

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void reconlVersion(out uint major, out uint minor, out uint patch);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern IntPtr reconlVersionString();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern IntPtr reconlResultName(ReconLResult r);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern IntPtr reconlTierName(ReconLTier tier);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern IntPtr reconlBackendName(ReconLBackendId backend);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void reconlSetLogLevel(ReconLLogLevel level);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLLogLevel reconlGetLogLevel();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void reconlSetLogSink(IntPtr sink, IntPtr user);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlProbe(IntPtr desc, IntPtr out_info);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlEnumerateAdapters(
            ReconLBackendId backend, IntPtr adapters, uint capacity, out uint out_count);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlCreateDevice(
            IntPtr desc, out IntPtr out_device);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlGetDeviceLimits(
            IntPtr device, IntPtr out_limits);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlGetMemoryStats(
            IntPtr device, IntPtr out_stats);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlCreateBuffer(
            IntPtr device, IntPtr desc, out IntPtr out_buffer);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlWriteBuffer(
            IntPtr device, IntPtr buffer, ulong offset, IntPtr data, ulong size);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlCreateTexture(
            IntPtr device, IntPtr desc, out IntPtr out_texture);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlWriteTexture(
            IntPtr device, IntPtr texture, IntPtr level);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlReadTexture(
            IntPtr device, IntPtr texture, uint mip, uint layer,
            IntPtr out_ptr, ulong out_size, uint out_row_pitch);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlGenerateMips(IntPtr device, IntPtr texture);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlCreatePipeline(
            IntPtr device, IntPtr desc, out IntPtr out_pipeline);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlCreateSwapchain(
            IntPtr device, IntPtr desc, out IntPtr out_swapchain);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlCreateCommandList(
            IntPtr device, IntPtr desc, out IntPtr out_commands);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlCreateFence(
            IntPtr device, uint signaled, out IntPtr out_fence);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlWaitFence(
            IntPtr device, IntPtr fence, ulong timeout_ns);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlConfigureShadows(
            IntPtr device, IntPtr config);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlBumpWorldRevision(
            IntPtr device, uint flags);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlGetStats(
            IntPtr device, IntPtr out_stats);

        // System allocator for the host
        [DllImport("kernel32.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern IntPtr GetProcessHeap();

        [DllImport("kernel32.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern IntPtr HeapAlloc(IntPtr heap, uint flags, uint size);

        [DllImport("kernel32.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern IntPtr HeapReAlloc(IntPtr heap, uint flags, IntPtr ptr, uint size);

        [DllImport("kernel32.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern bool HeapFree(IntPtr heap, uint flags, IntPtr ptr);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlGetLastError(IntPtr device, IntPtr out_info);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlCmdReset(IntPtr list);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlCmdBeginRenderPass(IntPtr list, IntPtr desc);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlCmdEndRenderPass(IntPtr list);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlCmdSetPipeline(IntPtr list, IntPtr pipeline);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlCmdSetVertexBuffer(
            IntPtr list, uint stream, IntPtr buffer, ulong offset);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlCmdSetIndexBuffer(
            IntPtr list, IntPtr buffer, ulong offset, ReconLIndexFormat format);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlCmdSetTexture(
            IntPtr list, uint slot, IntPtr texture, IntPtr sampler);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlCmdPushConstants(
            IntPtr list, uint slot, IntPtr data, uint size_bytes);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlCmdDraw(
            IntPtr list, uint vertex_count, uint first_vertex);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlCmdDrawIndexed(
            IntPtr list, uint index_count, uint first_index, int vertex_offset);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern uint reconlCmdCount(IntPtr list);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlBeginFrame(IntPtr device, IntPtr frame_desc);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlSubmit(IntPtr device, IntPtr commands);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlPresent(
            IntPtr device, IntPtr swapchain, IntPtr present_desc);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlPresentGenerated(
            IntPtr device, IntPtr swapchain, IntPtr present_desc);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ReconLResult reconlRelease(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern uint reconlRetain(IntPtr handle);
    }

    // ---- thin handle wrappers ----
    public unsafe struct Device : IDisposable
    {
        public IntPtr ptr;
        private bool disposed;
        public Device(IntPtr p) { ptr = p; disposed = false; }
        public bool IsValid => ptr != IntPtr.Zero;
        public void Dispose()
        {
            if (!disposed && ptr != IntPtr.Zero)
            {
                ReconlNative.reconlRelease(ptr);
                ptr = IntPtr.Zero;
                disposed = true;
            }
        }
        public void Reset() { Dispose(); }
    }

    public unsafe struct ReconLBuffer : IDisposable
    {
        public IntPtr ptr;
        private bool disposed;
        public ReconLBuffer(IntPtr p) { ptr = p; disposed = false; }
        public bool IsValid => ptr != IntPtr.Zero;
        public void Dispose()
        {
            if (!disposed && ptr != IntPtr.Zero)
            {
                ReconlNative.reconlRelease(ptr);
                ptr = IntPtr.Zero;
                disposed = true;
            }
        }
    }

    public unsafe struct Texture : IDisposable
    {
        public IntPtr ptr;
        private bool disposed;
        public Texture(IntPtr p) { ptr = p; disposed = false; }
        public bool IsValid => ptr != IntPtr.Zero;
        public void Dispose()
        {
            if (!disposed && ptr != IntPtr.Zero)
            {
                ReconlNative.reconlRelease(ptr);
                ptr = IntPtr.Zero;
                disposed = true;
            }
        }
    }

    public unsafe struct Pipeline : IDisposable
    {
        public IntPtr ptr;
        private bool disposed;
        public Pipeline(IntPtr p) { ptr = p; disposed = false; }
        public bool IsValid => ptr != IntPtr.Zero;
        public void Dispose()
        {
            if (!disposed && ptr != IntPtr.Zero)
            {
                ReconlNative.reconlRelease(ptr);
                ptr = IntPtr.Zero;
                disposed = true;
            }
        }
    }

    public unsafe struct Swapchain : IDisposable
    {
        public IntPtr ptr;
        private bool disposed;
        public Swapchain(IntPtr p) { ptr = p; disposed = false; }
        public bool IsValid => ptr != IntPtr.Zero;
        public void Dispose()
        {
            if (!disposed && ptr != IntPtr.Zero)
            {
                ReconlNative.reconlRelease(ptr);
                ptr = IntPtr.Zero;
                disposed = true;
            }
        }
    }

    public unsafe struct CommandList : IDisposable
    {
        public IntPtr ptr;
        private bool disposed;
        public CommandList(IntPtr p) { ptr = p; disposed = false; }
        public bool IsValid => ptr != IntPtr.Zero;
        public void Dispose()
        {
            if (!disposed && ptr != IntPtr.Zero)
            {
                ReconlNative.reconlRelease(ptr);
                ptr = IntPtr.Zero;
                disposed = true;
            }
        }
    }

    public unsafe struct Fence : IDisposable
    {
        public IntPtr ptr;
        private bool disposed;
        public Fence(IntPtr p) { ptr = p; disposed = false; }
        public bool IsValid => ptr != IntPtr.Zero;
        public void Dispose()
        {
            if (!disposed && ptr != IntPtr.Zero)
            {
                ReconlNative.reconlRelease(ptr);
                ptr = IntPtr.Zero;
                disposed = true;
            }
        }
    }

    // ---- result helper ----
    public static class Result
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsOk(ReconLResult r) => r == ReconLResult.OK;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Check(ReconLResult r)
        {
            if (r != ReconLResult.OK)
                throw new ReconLException(r);
        }
    }

    public class ReconLException : Exception
    {
        public ReconLResult Result { get; }
        public ReconLException(ReconLResult r) : base($"ReconL error {r}: {ReconLResultInfo.Name(r)}")
        {
            Result = r;
        }
    }

    /// <summary>
    /// Gets the last error recorded by a device (or the process-wide last error
    /// when device is IntPtr.Zero), matching what the host tools print.
    /// </summary>
    public static class LastError
    {
        public static string Get(IntPtr device)
        {
            // ReconLErrorInfo has fixed-size string fields; allocate unmanaged
            // memory and marshal manually.
            var size = Marshal.SizeOf<ReconLErrorInfo>();
            var ptr = Marshal.AllocHGlobal(size);
            try
            {
                // Zero the header so the library fills the whole struct
                var zero = new byte[size];
                Marshal.Copy(zero, 0, ptr, size);
                var r = ReconlNative.reconlGetLastError(device, ptr);
                if (r != ReconLResult.OK)
                    return $"get-last-error failed: {r}";
                var info = Marshal.PtrToStructure<ReconLErrorInfo>(ptr);
                if (info.result == ReconLResult.OK || string.IsNullOrEmpty(info.message))
                    return null;
                return $"{info.message} ({info.result})";
            }
            finally
            {
                Marshal.FreeHGlobal(ptr);
            }
        }
    }
}
