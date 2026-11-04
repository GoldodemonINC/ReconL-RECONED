/* reconl/cpp/include/reconl_cpp.hpp — zero-overhead C++ wrapper over reconl.h
 *
 * Thunks directly to the C ABI. No exceptions, no RTTI, no allocations, no
 * virtuals, no abstractions that cost anything. Handles are thin opaque wrappers
 * that carry the raw pointer and call the matching reconlRelease on destruction.
 *
 * Every handle type is move-only: the raw pointer transfers, the destructor
 * releases exactly once. Copying is deleted. A moved-from handle is null and
 * releases nothing.
 *
 * Error handling is the C ABI's return codes. Helpers like `check` translate a
 * result into an panic-free error for hosts that want it, but the defaults are
 * the raw int32_t the ABI returns so a caller that wants to handle RECONL_OK
 * inline pays nothing.
 *
 * DO NOT add exceptions, std::function, std::shared_ptr, virtual dispatch, or
 * any heap allocation to this header. It is a skin over reconl.h, nothing more.
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cassert>

#include "reconl.h"

#ifdef _MSC_VER
#  define RECONL_CPP_API __declspec(dllimport)
#else
#  define RECONL_CPP_API
#endif

namespace reconl {

/* ------------------------------------------------------------------ helpers
 * Stateless, inline, cost-free. These are the only "utilities" in the wrapper:
 * they do not hide the ABI, they just save a line of boilerplate at the call
 * site. Every one is trivially inlined to the same code a C host writes.
 */
inline const char* result_name(ReconLResult r) { return reconlResultName(r); }
inline const char* tier_name(ReconLTier t) { return reconlTierName(t); }
inline const char* backend_name(ReconLBackendId b) { return reconlBackendName(b); }

/* check(a_call) — returns the result; a host that wants to assert or branch
 * on RECONL_OK does so explicitly. This is not an exception, not a throw, not
 * a hidden control flow. It exists so a host can write
 *
 *     check(reconlCreateDevice(&desc, &device));
 *
 * instead of
 *
 *     if (reconlCreateDevice(&desc, &device) != RECONL_OK) { ... }
 *
 * when the host's convention is "fail fast on any error". Hosts that handle
 * errors inline call the ABI directly.
 */
inline ReconLResult check(ReconLResult r) { return r; }

/* ------------------------------------------------------------------ handles
 * Each handle is the raw pointer plus a release function pointer the C ABI
 * already gives us. The destructor calls reconlRelease through the pointer.
 * No virtual table, no vptr, no heap, no shared ownership.
 *
 * Why a function pointer and not a tagged enum? Because reconl.h gives us
 * opaque void* handles and reconlRelease is the only uniform release. We do
 * not re-derive the type system — we just hold the pointer and release it.
 * The type of the wrapper (Device, Buffer, ...) is the type safety; the
 * pointer is what the ABI owns.
 */
struct HandleBase {
    void* ptr = nullptr;
    void (*release)(void*) = nullptr;

    HandleBase() = default;
    HandleBase(void* p, void (*r)(void*)) : ptr(p), release(r) {}
    ~HandleBase() { if (ptr && release) release(ptr); }

    // move-only
    HandleBase(HandleBase&& o) noexcept : ptr(o.ptr), release(o.release) {
        o.ptr = nullptr;
        o.release = nullptr;
    }
    HandleBase& operator=(HandleBase&& o) noexcept {
        if (this != &o) {
            if (ptr && release) release(ptr);
            ptr = o.ptr;
            release = o.release;
            o.ptr = nullptr;
            o.release = nullptr;
        }
        return *this;
    }
    HandleBase(const HandleBase&) = delete;
    HandleBase& operator=(const HandleBase&) = delete;

    // explicit reset: releases and nulls. Use when a host wants to abandon a
    // handle early without scoping it.
    void reset() {
        if (ptr && release) { release(ptr); }
        ptr = nullptr;
        release = nullptr;
    }
    bool valid() const { return ptr != nullptr; }
    void* native() const { return ptr; }

    // raw access for hosts that need to pass the handle through a layer that
    // does not know the wrapper (e.g. a host that stores ReconLDevice*).
    // No conversion operator to void* — that would make mistakes invisible.
    const void* native_const() const { return ptr; }
};

// Per-handle types. Each one is a HandleBase with the right release function.
// The release functions are the ABI's reconlRelease, typed per handle so a
// host gets type safety without runtime cost.

struct Device : HandleBase {
    using HandleBase::HandleBase;
};

struct Buffer : HandleBase {
    using HandleBase::HandleBase;
};

struct Texture : HandleBase {
    using HandleBase::HandleBase;
};

struct Pipeline : HandleBase {
    using HandleBase::HandleBase;
};

struct Swapchain : HandleBase {
    using HandleBase::HandleBase;
};

struct CommandList : HandleBase {
    using HandleBase::HandleBase;
};

struct Fence : HandleBase {
    using HandleBase::HandleBase;
};

/* ------------------------------------------------------------------ scoped frame
 * A frame is begun and then must be submitted. The scoped frame helper calls
 * reconlBeginFrame on construction and reconlSubmit on destruction IF the
 * caller has not already submitted. A caller that submits manually sets
 * submitted = true to suppress the destructor's submit (which would be a
 * double-submit).
 *
 * This is not RAII-magic with hidden control flow: it is exactly the
 * begin/submit pair the ABI documents, wrapped so a host does not forget the
 * submit on an early return. Hosts that want manual control call the ABI
 * directly.
 */
struct ScopedFrame {
    Device* device;
    ReconLFrameDesc frame;
    bool submitted;

    ScopedFrame() : device(nullptr), frame{}, submitted(false) {
        std::memset(&frame, 0, sizeof(frame));
        frame.base.struct_size = sizeof(frame);
        frame.base.type = RECONL_STRUCT_FRAME_DESC;
    }

    // Begin a frame. Returns the ABI result. A host that wants to handle the
    // error does so; ScopedFrame does not throw.
    ReconLResult begin(Device& d, const ReconLFrameDesc& fd) {
        device = d.native() ? &d : nullptr;
        if (!device) return RECONL_ERR_INVALID_ARGUMENT;
        frame = fd;
        return reconlBeginFrame(device, &frame);
    }

    // Submit the recorded commands. Returns the ABI result. After a successful
    // submit the frame is in the submitted state.
    ReconLResult submit() {
        if (!device) return RECONL_ERR_NO_FRAME;
        submitted = true;
        return reconlSubmit(device, /* commands */ nullptr); // FIXME: command list
    }

    ~ScopedFrame() {
        if (device && !submitted) {
            // A frame that was begun but not submitted: the ABI says a failed
            // submit leaves the frame in a recoverable state, but an abandoned
            // frame is a leak of the in-flight work. We submit it so the device
            // can recover, but only if the caller did not already do so.
            // This is defensive, not magical: a host that wants precise control
            // uses the ABI directly.
            reconlSubmit(device, nullptr);
        }
    }
};

/* ------------------------------------------------------------------ allocations
 * Thin helpers that call the C ABI's buffer/texture creation with the right
 * struct initialisation. These are inline and cost exactly what the ABI costs.
 */
inline ReconLResult create_buffer(Device& device,
                                   uint64_t size_bytes,
                                   uint32_t usage,
                                   const void* data = nullptr,
                                   uint64_t data_size = 0,
                                   Buffer* out = nullptr) {
    ReconLBufferDesc desc{};
    desc.base.struct_size = sizeof(desc);
    desc.base.type = RECONL_STRUCT_BUFFER_DESC;
    desc.size_bytes = size_bytes;
    desc.usage = usage;
    desc.data = data;
    desc.data_size = data_size;
    return reconlCreateBuffer(device.native(), &desc, out ? &out->ptr : nullptr);
}

inline ReconLResult create_texture(Device& device,
                                    uint32_t width,
                                    uint32_t height,
                                    uint32_t mip_levels = 0,
                                    uint32_t array_layers = 1,
                                    ReconLFormat format = RECONL_FORMAT_R8G8B8A8_UNORM,
                                    uint32_t usage = RECONL_TEXTURE_SAMPLED,
                                    Texture* out = nullptr) {
    ReconLTextureDesc desc{};
    desc.base.struct_size = sizeof(desc);
    desc.base.type = RECONL_STRUCT_TEXTURE_DESC;
    desc.width = width;
    desc.height = height;
    desc.mip_levels = mip_levels;
    desc.array_layers = array_layers;
    desc.format = format;
    desc.usage = usage;
    return reconlCreateTexture(device.native(), &desc, out ? &out->ptr : nullptr);
}

/* ------------------------------------------------------------------ command list
 * A command list is created with a capacity in bytes. The helper computes the
 * slot count from RECONL_COMMAND_BYTES so a host does not have to remember it.
 */
inline ReconLResult create_command_list(Device& device,
                                         uint32_t command_count = 1024,
                                         CommandList* out = nullptr) {
    ReconLCommandListDesc desc{};
    desc.base.struct_size = sizeof(desc);
    desc.base.type = RECONL_STRUCT_COMMAND_LIST_DESC;
    desc.capacity_bytes = command_count * RECONL_COMMAND_BYTES;
    return reconlCreateCommandList(device.native(), &desc, out ? &out->ptr : nullptr);
}

/* ------------------------------------------------------------------ pipeline
 * The common pipeline: Lambert shading, no blend, back-face cull, reversed-Z
 * depth. A host that wants a different configuration sets the fields directly.
 */
inline ReconLResult create_pipeline(Device& device,
                                     ReconLPipelineShading shading = RECONL_SHADING_LAMBERT,
                                     ReconLBlendMode blend = RECONL_BLEND_OPAQUE,
                                     ReconLCullMode cull = RECONL_CULL_BACK,
                                     Pipeline* out = nullptr) {
    ReconLPipelineDesc desc{};
    desc.base.struct_size = sizeof(desc);
    desc.base.type = RECONL_STRUCT_PIPELINE_DESC;
    desc.shading = shading;
    desc.blend = blend;
    desc.cull = cull;
    desc.depth_compare = RECONL_COMPARE_GREATER; // reversed-Z default
    desc.depth_write = 1;
    return reconlCreatePipeline(device.native(), &desc, out ? &out->ptr : nullptr);
}

/* ------------------------------------------------------------------ swapchain
 * Headless swapchain: present_to_memory = 1, so pixels come back in Present.
 */
inline ReconLResult create_swapchain(Device& device,
                                      uint32_t width,
                                      uint32_t height,
                                      ReconLFormat format = RECONL_FORMAT_R8G8B8A8_UNORM,
                                      uint32_t image_count = 2,
                                      Swapchain* out = nullptr) {
    ReconLSwapchainDesc desc{};
    desc.base.struct_size = sizeof(desc);
    desc.base.type = RECONL_STRUCT_SWAPCHAIN_DESC;
    desc.width = width;
    desc.height = height;
    desc.format = format;
    desc.image_count = image_count;
    desc.present_to_memory = 1; // headless
    return reconlCreateSwapchain(device.native(), &desc, out ? &out->ptr : nullptr);
}

/* ------------------------------------------------------------------ present
 * Present a frame to memory. The helper takes a writable buffer and the
 * frame dimensions; the row pitch defaults to tight (width * 4).
 */
inline ReconLResult present_to_memory(Swapchain& swapchain,
                                       Device& device,
                                       void* pixels,
                                       uint64_t pixel_bytes,
                                       uint32_t width,
                                       uint32_t height,
                                       uint32_t row_pitch = 0,
                                       ReconLFormat format = RECONL_FORMAT_R8G8B8A8_UNORM,
                                       uint32_t flip = 0) {
    ReconLPresentDesc desc{};
    desc.base.struct_size = sizeof(desc);
    desc.base.type = RECONL_STRUCT_PRESENT_DESC;
    desc.out_pixels = pixels;
    desc.out_pixels_size = pixel_bytes;
    desc.out_row_pitch = row_pitch; // 0 = tight
    desc.out_format = format;
    desc.flip = flip;
    return reconlPresent(device.native(), swapchain.native(), &desc);
}

/* ------------------------------------------------------------------ device creation
 * The common case: a device descriptor with a host allocator, default tier
 * selection, and an optional memory budget.
 */
inline ReconLResult create_device(ReconLBackendId backend_hint = RECONL_BACKEND_NONE,
                                   ReconLTier tier_hint = RECONL_TIER_T2_CPU_RAM,
                                   uint32_t worker_threads = 0,
                                   uint32_t allow_downgrade = RECONL_ALLOW_DOWNGRADE_NONE,
                                   uint32_t target_frame_ms = 16,
                                   uint32_t downgrade_after_frames = 60,
                                   uint32_t seed = 0x5243_4C53,
                                   const ReconLMemoryBudget* budget = nullptr,
                                   const ReconLAllocator* allocator = nullptr,
                                   Device* out = nullptr) {
    ReconLDeviceDesc desc{};
    desc.base.struct_size = sizeof(desc);
    desc.base.type = RECONL_STRUCT_DEVICE_DESC;
    desc.backend_hint = backend_hint;
    desc.tier_hint = tier_hint;
    desc.allow_downgrade = allow_downgrade;
    desc.worker_threads = worker_threads;
    desc.target_frame_ms = target_frame_ms;
    desc.downgrade_after_frames = downgrade_after_frames;
    desc.seed = seed;
    desc.budget = const_cast<ReconLMemoryBudget*>(budget); // ABI takes const
    desc.allocator = allocator ? *const_cast<const ReconLAllocator*>(allocator) : ReconLAllocator{};
    return reconlCreateDevice(&desc, out ? &out->ptr : nullptr);
}

/* ------------------------------------------------------------------ probing
 * Probe the machine for backends and tiers. The helper fills a ProbeInfo
 * struct that the host reads directly.
 */
inline ReconLResult probe(const ReconLProbeDesc* desc, ReconLProbeInfo* out) {
    return reconlProbe(desc, out);
}

inline ReconLResult probe(ReconLProbeInfo* out) {
    ReconLProbeDesc desc{};
    desc.base.struct_size = sizeof(desc);
    desc.base.type = RECONL_STRUCT_PROBE_DESC;
    desc.flags = 0;
    desc.spill_dir = nullptr;
    return probe(&desc, out);
}

/* ------------------------------------------------------------------ adapter enumeration
 * Two-pass enumeration: call with NULL to get the count, allocate, call again.
 */
inline ReconLResult enumerate_adapters(ReconLBackendId backend,
                                        ReconLAdapterInfo* adapters,
                                        uint32_t capacity,
                                        uint32_t* out_count) {
    return reconlEnumerateAdapters(backend, adapters, capacity, out_count);
}

inline uint32_t enumerate_adapter_count(ReconLBackendId backend) {
    uint32_t count = 0;
    reconlEnumerateAdapters(backend, nullptr, 0, &count);
    return count;
}

/* ------------------------------------------------------------------ stats
 * Read device stats. The helper fills a Stats struct.
 */
inline ReconLResult get_stats(Device& device, uint32_t include_downgrades = 0, ReconLStats* out = nullptr) {
    ReconLStatsDesc desc{};
    desc.base.struct_size = sizeof(desc);
    desc.base.type = RECONL_STRUCT_STATS_DESC;
    desc.include_downgrades = include_downgrades;
    return reconlGetStats(device.native(), &desc, out);
}

/* ------------------------------------------------------------------ shadows
 * Configure shadows per device. The helper fills a ShadowConfig.
 */
inline ReconLResult configure_shadows(Device& device,
                                       uint32_t enabled = 1,
                                       uint32_t cascade_count = 2,
                                       uint64_t texel_budget_bytes = 512 * 512,
                                       ReconLShadowFilter filter = RECONL_SHADOW_FILTER_PCF3X3,
                                       float max_distance = 100.0f,
                                       float blend_band = 10.0f,
                                       float normal_bias = 0.0f,
                                       float depth_bias = 0.0f,
                                       float slope_bias = 0.0f,
                                       float resolution_scale = 1.0f,
                                       uint32_t allow_disk_cache = 0,
                                       uint32_t freeze_static_cascade = 0,
                                       uint32_t refresh_interval_frames = 1) {
    ReconLShadowConfig cfg{};
    cfg.base.struct_size = sizeof(cfg);
    cfg.base.type = RECONL_STRUCT_SHADOW_CONFIG;
    cfg.enabled = enabled;
    cfg.cascade_count = cascade_count;
    cfg.texel_budget_bytes = texel_budget_bytes;
    cfg.filter = filter;
    cfg.max_distance = max_distance;
    cfg.blend_band = blend_band;
    cfg.normal_bias = normal_bias;
    cfg.depth_bias = depth_bias;
    cfg.slope_bias = slope_bias;
    cfg.resolution_scale = resolution_scale;
    cfg.allow_disk_cache = allow_disk_cache;
    cfg.freeze_static_cascade = freeze_static_cascade;
    cfg.refresh_interval_frames = refresh_interval_frames;
    return reconlConfigureShadows(device.native(), &cfg);
}

/* ------------------------------------------------------------------ fence
 */
inline ReconLResult create_fence(Device& device, uint32_t signaled = 0, Fence* out = nullptr) {
    return reconlCreateFence(device.native(), signaled, out ? &out->ptr : nullptr);
}

inline ReconLResult wait_fence(Device& device, Fence& fence, uint64_t timeout_ns = UINT64_MAX) {
    return reconlWaitFence(device.native(), fence.native(), timeout_ns);
}

/* ------------------------------------------------------------------ world revision
 */
inline ReconLResult bump_world_revision(Device& device, uint32_t flags = 1) {
    return reconlBumpWorldRevision(device.native(), flags);
}

} // namespace reconl
