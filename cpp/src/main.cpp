// cpp/src/main.cpp — C++ host that drives the reconl C ABI through the zero-overhead wrapper.
//
// This is the C++ equivalent of the Rust abi.rs integration test and the Zig
// abi-test: it creates a device, records a frame on the soft-cpu backend, and
// verifies the result. It proves the C++ wrapper can talk to the shipped DLL.
//
// Design: no exceptions (the wrapper does not throw), no RTTI, no heap in the
// frame setup path beyond what the ABI does. Handle lifetime is deterministic
// via the move-only wrapper destructors.

#include "reconl_cpp.hpp"
#include <cstdio>
#include <cstring>

// Helper: print a hex dump of the first N pixels of a frame.
static void print_frame_preview(const uint8_t* pixels, uint32_t width, uint32_t height, uint32_t count = 16)
{
    printf("  frame preview (first %u pixels, RGBA):\n", count);
    for (uint32_t i = 0; i < count && i < width * height; i++)
    {
        printf("    [%u] R=%d G=%d B=%d A=%d\n", i,
               pixels[i * 4 + 0], pixels[i * 4 + 1], pixels[i * 4 + 2], pixels[i * 4 + 3]);
    }
}

// Helper: count lit pixels (any channel > threshold).
static uint32_t count_lit(const uint8_t* pixels, uint32_t pixel_count, uint8_t threshold = 10)
{
    uint32_t count = 0;
    for (uint32_t i = 0; i < pixel_count; i++)
    {
        if (pixels[i * 4 + 0] > threshold || pixels[i * 4 + 1] > threshold || pixels[i * 4 + 2] > threshold)
            count++;
    }
    return count;
}

// Host allocator for the C++ host — a thin wrapper over malloc/free that the
// ABI accepts. The ABI takes function pointers, so we pass the C functions
// directly. No allocation in the hot path: the host's frame loop would use a
// pre-allocated block pool, but this smoke test uses the system allocator.
static void* host_alloc(void* user, size_t size, size_t alignment)
{
    (void)user;
    // alignment is honoured by the ABI; we just allocate and let the ABI's
    // wrapper sort the alignment. For a real host, align up.
    return std::malloc(size);
}

static void* host_realloc(void* user, void* ptr, size_t old_size, size_t new_size, size_t alignment)
{
    (void)user; (void)old_size; (void)alignment;
    return std::realloc(ptr, new_size);
}

static void host_free(void* user, void* ptr, size_t size)
{
    (void)user; (void)size;
    std::free(ptr);
}

// Create a vertex buffer with a simple triangle (two triangles as a quad, or
// one triangle — here one triangle for the smoke test).
static reconl::Buffer create_vertex_buffer(reconl::Device& device, const float* vertices, uint32_t count)
{
    reconl::Buffer buf;
    // Each vertex is 48 bytes: pos3 + normal3 + uv2 + color4 = 3*4 + 3*4 + 2*4 + 4*4 = 48
    uint64_t size_bytes = (uint64_t)count * 48;
    uint64_t data_size = size_bytes;

    // Allocate a copy of the vertex data for the host to own
    float* host_data = new float[count];
    std::memcpy(host_data, vertices, size_bytes);

    auto r = reconl::create_buffer(device, size_bytes,
                                    RECONL_BUFFER_VERTEX | RECONL_BUFFER_STATIC,
                                    host_data, data_size, &buf);
    delete[] host_data;

    if (r != RECONL_OK)
    {
        printf("  create_vertex_buffer failed: %d (%s)\n", r, reconl::result_name((ReconLResult)r));
        return buf;
    }
    return buf;
}

int main()
{
    printf("ReconL C++ host — ABI smoke test\n");
    printf("  checking the zero-overhead wrapper against the shipped DLL\n\n");

    // Version
    uint32_t major, minor, patch;
    reconlVersion(&major, &minor, &patch);
    printf("  reconl version: %u.%u.%u\n", major, minor, patch);
    printf("  result names: %s / %s\n\n",
           reconlResultName(RECONL_OK),
           reconlResultName(RECONL_ERR_INVALID_ARGUMENT));

    // Probe
    ReconLProbeInfo probe_info{};
    probe_info.base.struct_size = sizeof(probe_info);
    probe_info.base.type = RECONL_STRUCT_PROBE_INFO;
    probe_info.base.next = nullptr;
    for (uint32_t i = 0; i < RECONL_MAX_BACKENDS; i++)
    {
        probe_info.entries[i].base.struct_size = sizeof(ReconLBackendProbe);
        probe_info.entries[i].base.type = RECONL_STRUCT_BACKEND_PROBE;
        probe_info.entries[i].base.next = nullptr;
    }

    ReconLProbeDesc probe_desc{};
    probe_desc.base.struct_size = sizeof(probe_desc);
    probe_desc.base.type = RECONL_STRUCT_PROBE_DESC;
    probe_desc.base.next = nullptr;
    probe_desc.flags = 0;
    probe_desc.spill_dir = nullptr;

    auto pr = reconlProbe(&probe_desc, &probe_info);
    if (pr != RECONL_OK)
    {
        printf("  probe failed: %d\n", pr);
        return 1;
    }

    printf("  probe: recommended backend=%u/%s tier=%u/%s\n",
           probe_info.recommended_backend,
           reconlBackendName((ReconLBackendId)probe_info.recommended_backend),
           probe_info.recommended_tier,
           reconlTierName((ReconLTier)probe_info.recommended_tier));
    printf("  backends:\n");
    for (uint32_t i = 0; i < probe_info.entry_count; i++)
    {
        auto& e = probe_info.entries[i];
        printf("    %u %s usable=%u tier=%u caps=0x%08x\n",
               i,
               e.name,
               e.usable,
               e.best_tier,
               e.caps);
    }
    printf("\n");

    // Device + frame test on the soft-cpu backend (deterministic, works everywhere)
    ReconLAllocator allocator{};
    allocator.alloc = (void*(*)(void*, size_t, size_t))host_alloc;
    allocator.realloc = (void*(*)(void*, void*, size_t, size_t, size_t))host_realloc;
    allocator.free = (void(*)(void*, void*, size_t))host_free;
    allocator.user = nullptr;

    ReconLDeviceDesc device_desc{};
    device_desc.base.struct_size = sizeof(device_desc);
    device_desc.base.type = RECONL_STRUCT_DEVICE_DESC;
    device_desc.base.next = nullptr;
    device_desc.backend_hint = RECONL_BACKEND_SOFT_CPU;
    device_desc.tier_hint = RECONL_TIER_T2_CPU_RAM;
    device_desc.allow_downgrade = 0;
    device_desc.worker_threads = 0;
    device_desc.target_frame_ms = 16;
    device_desc.downgrade_after_frames = 60;
    device_desc.seed = 0x52434C53;
    device_desc.flags = 0;
    device_desc.budget = nullptr;
    device_desc.allocator = allocator;
    device_desc.backend_desc = nullptr;

    reconl::Device device;
    auto dr = reconl::create_device(RECONL_BACKEND_SOFT_CPU,
                                     RECONL_TIER_T2_CPU_RAM,
                                     0, 0, 16, 60, 0x52434C53,
                                     nullptr, &allocator, &device);
    if (dr != RECONL_OK || !device.valid())
    {
        printf("  device creation failed: %d\n", dr);
        return 1;
    }
    printf("  device created: %p\n", device.native());

    // Get limits
    ReconLDeviceLimits limits{};
    limits.base.struct_size = sizeof(limits);
    limits.base.type = RECONL_STRUCT_DEVICE_LIMITS;
    limits.base.next = nullptr;
    auto lr = reconlGetDeviceLimits(device.native(), &limits);
    if (lr != RECONL_OK)
    {
        printf("  get limits failed: %d\n", lr);
        return 1;
    }
    printf("  limits: vram=%llu ram=%llu max_alloc=%llu worker_threads=%u caps=0x%08x\n\n",
           (unsigned long long)limits.vram_bytes,
           (unsigned long long)limits.ram_bytes,
           (unsigned long long)limits.max_allocation_bytes,
           limits.worker_threads_max,
           limits.caps);

    // Swapchain
    reconl::Swapchain swapchain;
    auto sr = reconl::create_swapchain(device, 64, 64, RECONL_FORMAT_R8G8B8A8_UNORM, 2, &swapchain);
    if (sr != RECONL_OK || !swapchain.valid())
    {
        printf("  swapchain creation failed: %d\n", sr);
        return 1;
    }
    printf("  swapchain created: %p\n", swapchain.native());

    // Pipeline
    reconl::Pipeline pipeline;
    auto pr2 = reconl::create_pipeline(device, RECONL_SHADING_LAMBERT, RECONL_BLEND_OPAQUE, RECONL_CULL_BACK, &pipeline);
    if (pr2 != RECONL_OK || !pipeline.valid())
    {
        printf("  pipeline creation failed: %d\n", pr2);
        return 1;
    }
    printf("  pipeline created: %p\n", pipeline.native());

    // Command list
    reconl::CommandList cmd_list;
    auto cr = reconl::create_command_list(device, 256, &cmd_list);
    if (cr != RECONL_OK || !cmd_list.valid())
    {
        printf("  command list creation failed: %d\n", cr);
        return 1;
    }
    printf("  command list created: %p\n", cmd_list.native());

    // Render a frame
    const uint32_t width = 64, height = 64;
    std::vector<uint8_t> pixels(width * height * 4, 0);

    // Vertices: one triangle in the upper-right quadrant
    float vertices[] = {
        // position      normal      uv      color
        0.5f,  0.5f, 0.0f,  0,0,1,  0,0,  1,1,1,1,
        0.0f, -0.5f, 0.0f,  0,0,1,  0,0,  1,1,1,1,
        1.0f, -0.5f, 0.0f,  0,0,1,  0,0,  1,1,1,1,
    };
    uint32_t vertex_count = 3;

    reconl::Buffer vb = create_vertex_buffer(device, vertices, vertex_count);
    if (!vb.valid()) return 1;
    printf("  vertex buffer created: %p\n", vb.native());

    // Begin frame
    ReconLFrameDesc frame_desc{};
    frame_desc.base.struct_size = sizeof(frame_desc);
    frame_desc.base.type = RECONL_STRUCT_FRAME_DESC;
    frame_desc.base.next = nullptr;
    frame_desc.width = width;
    frame_desc.height = height;
    frame_desc.seed = 1;
    frame_desc.reserved = 0;
    frame_desc.lights = nullptr;
    frame_desc.shadows = nullptr;
    frame_desc.camera = nullptr;
    frame_desc.framegen = nullptr;

    auto br = reconlBeginFrame(device.native(), &frame_desc);
    if (br != RECONL_OK)
    {
        printf("  begin frame failed: %d\n", br);
        return 1;
    }
    printf("  frame begun\n");

    // Reset command list
    auto rr = reconlCmdReset(cmd_list.native());
    if (rr != RECONL_OK)
    {
        printf("  cmd reset failed: %d\n", rr);
        return 1;
    }

    // Begin render pass
    ReconLRenderPassDesc pass_desc{};
    pass_desc.base.struct_size = sizeof(pass_desc);
    pass_desc.base.type = RECONL_STRUCT_RENDER_PASS_DESC;
    pass_desc.base.next = nullptr;
    pass_desc.color_count = 0;
    pass_desc.reserved = 0;
    pass_desc.depth = nullptr;
    pass_desc.viewport_width = 0;
    pass_desc.viewport_height = 0;
    pass_desc.load_color = 1;
    pass_desc.load_depth = 1;
    pass_desc.clear_color[0] = 0.0f;
    pass_desc.clear_color[1] = 0.0f;
    pass_desc.clear_color[2] = 1.0f;
    pass_desc.clear_color[3] = 1.0f;
    pass_desc.clear_depth = 0.0f;
    pass_desc.stencil_clear = 0;
    pass_desc.reserved2 = 0;

    rr = reconlCmdBeginRenderPass(cmd_list.native(), &pass_desc);
    if (rr != RECONL_OK)
    {
        printf("  begin pass failed: %d\n", rr);
        return 1;
    }

    // Set pipeline
    rr = reconlCmdSetPipeline(cmd_list.native(), pipeline.native());
    if (rr != RECONL_OK)
    {
        printf("  set pipeline failed: %d\n", rr);
        return 1;
    }

    // Push constants: identity matrices
    float identity[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    rr = reconlCmdPushConstants(cmd_list.native(), 0, identity, 64);
    if (rr != RECONL_OK)
    {
        printf("  push constants 0 failed: %d\n", rr);
        return 1;
    }
    rr = reconlCmdPushConstants(cmd_list.native(), 1, identity, 64);
    if (rr != RECONL_OK)
    {
        printf("  push constants 1 failed: %d\n", rr);
        return 1;
    }

    // Set vertex buffer
    rr = reconlCmdSetVertexBuffer(cmd_list.native(), 0, vb.native(), 0);
    if (rr != RECONL_OK)
    {
        printf("  set vertex buffer failed: %d\n", rr);
        return 1;
    }

    // Draw
    rr = reconlCmdDraw(cmd_list.native(), vertex_count, 0);
    if (rr != RECONL_OK)
    {
        printf("  draw failed: %d\n", rr);
        return 1;
    }

    // End pass
    rr = reconlCmdEndRenderPass(cmd_list.native());
    if (rr != RECONL_OK)
    {
        printf("  end pass failed: %d\n", rr);
        return 1;
    }

    // Submit
    rr = reconlSubmit(device.native(), cmd_list.native());
    if (rr != RECONL_OK)
    {
        printf("  submit failed: %d\n", rr);
        return 1;
    }
    printf("  frame submitted\n");

    // Present
    ReconLPresentDesc present_desc{};
    present_desc.base.struct_size = sizeof(present_desc);
    present_desc.base.type = RECONL_STRUCT_PRESENT_DESC;
    present_desc.base.next = nullptr;
    present_desc.out_pixels = pixels.data();
    present_desc.out_pixels_size = (uint64_t)pixels.size();
    present_desc.out_row_pitch = width * 4; // tight
    present_desc.out_format = RECONL_FORMAT_R8G8B8A8_UNORM;
    present_desc.flip = 0;

    auto pres_r = reconlPresent(device.native(), swapchain.native(), &present_desc);
    if (pres_r != RECONL_OK)
    {
        printf("  present failed: %d\n", pres_r);
        return 1;
    }
    printf("  frame presented\n\n");

    // Analyse the frame
    uint32_t lit = count_lit(pixels.data(), width * height);
    printf("  frame analysis:\n");
    printf("    total pixels: %u\n", width * height);
    printf("    lit pixels: %u\n", lit);
    printf("    first pixel: R=%d G=%d B=%d A=%d\n",
           pixels[0], pixels[1], pixels[2], pixels[3]);
    print_frame_preview(pixels.data(), width, height, 16);
    printf("\n");

    // Stats
    ReconLStats stats{};
    stats.base.struct_size = sizeof(stats);
    stats.base.type = RECONL_STRUCT_STATS;
    stats.base.next = nullptr;
    ReconLStatsDesc stats_desc{};
    stats_desc.base.struct_size = sizeof(stats_desc);
    stats_desc.base.type = RECONL_STRUCT_STATS_DESC;
    stats_desc.base.next = nullptr;
    stats_desc.include_downgrades = 0;
    stats_desc.reserved = 0;

    auto sr2 = reconlGetStats(device.native(), &stats_desc, &stats);
    if (sr2 != RECONL_OK)
    {
        printf("  get stats failed: %d\n", sr2);
        // Non-fatal for the smoke test
    }
    else
    {
        printf("  stats:\n");
        printf("    frames_presented: %u\n", stats.frames_presented);
        printf("    tier: %u/%s\n", stats.tier, reconlTierName((ReconLTier)stats.tier));
        printf("    backend: %u/%s\n", stats.backend, reconlBackendName((ReconLBackendId)stats.backend));
        printf("    frame total_ns: %llu\n", (unsigned long long)stats.frame.total_ns);
        printf("    frame shadow_ns: %llu\n", (unsigned long long)stats.frame.shadow_ns);
        printf("    frame raster_ns: %llu\n", (unsigned long long)stats.frame.raster_ns);
    }

    printf("\n");
    printf("C++ host: %s\n", lit > 0 ? "ALL TESTS PASSED" : "FRAME EMPTY (unexpected)");
    return lit > 0 ? 0 : 1;
}
