//! The Vulkan implementation of the ReconL hardware backend.
//!
//! # Frame path
//!
//! The same three passes the d3d11 backend runs, against the same contract:
//!
//! 1. **Shadow pass.** `reconl-shadow`'s `fit_cascades` runs on the CPU (the
//!    identical code the reference rasteriser uses), then every `casts_shadow`
//!    draw is rendered depth-only into a `D32_SFLOAT` map with the cascade's
//!    light·model matrix pushed as a constant, front faces culled, and no
//!    fixed-function depth bias - the reference applies bias at lookup.
//! 2. **Colour pass.** The ported pixel shader in [`shaders`]: cascade scan,
//!    crossfade, filter dispatch, lighting, and the four blend modes as four
//!    pipeline states.
//! 3. **Readback.** The colour image is copied into a HOST_VISIBLE buffer and
//!    mapped, then laid into the host's row layout.
//!
//! # Why there is no swapchain
//!
//! The benchmark surface presents *to memory*: the host wants the pixels back on
//! the CPU. A swapchain hands images to a window system, which is not what is
//! being asked for, so this backend renders into an offscreen colour image and
//! copies it out - the same shape the d3d11 backend uses with a render target and
//! a staging texture. That removes every surface, present mode and semaphore
//! from the path: one queue, one command buffer, one fence.
//!
//! # Shaders
//!
//! GLSL sources in [`shaders`], compiled to SPIR-V at device creation by the
//! Vulkan SDK's `glslc` (see [`render::init_device`]). A missing SDK is a
//! backend-unavailable error naming `VULKAN_SDK`, not a silent fallback.

mod render;
mod shaders;

use ash::vk;
use ash::Entry;
use reconl_contract::FrameInput;
use reconl_core::alloc::HostAlloc;
use reconl_core::error::{Code, Error, Result};
use reconl_core::stats::{Counters, FrameNumbers, ShadowCounters};
use reconl_core::tier::{caps, filter_caps, Backend, ShadowPlan, Tier, TierReason};
use reconl_raster::Vertex;

/// The vertex stride the input layout assumes: position, normal, uv, colour.
pub const VERTEX_STRIDE: u32 = std::mem::size_of::<Vertex>() as u32;

/// Everything a host can say to this backend at creation.
#[derive(Clone, Debug)]
pub struct VulkanConfig {
    pub tier: Tier,
    pub resolution_scale: f32,
    pub target_frame_ms: u32,
    /// Cascade split lambda, shared with the reference and d3d11 backends so the
    /// three fit the same cascades: `0` uniform, `1` logarithmic.
    pub split_lambda: f32,
    pub shadow: reconl_contract::ShadowRequest,
}

impl Default for VulkanConfig {
    fn default() -> Self {
        Self {
            tier: Tier::GpuShared,
            resolution_scale: 1.0,
            target_frame_ms: 16,
            split_lambda: 0.75,
            shadow: reconl_contract::ShadowRequest::default(),
        }
    }
}

/// What a frame left behind, in the shape the FFI's own snapshot uses.
#[derive(Clone, Copy, Default)]
pub struct VulkanSnapshot {
    pub counters: Counters,
    pub shadows: ShadowCounters,
    pub frame: FrameNumbers,
    pub color_checksum: u64,
}

/// One GPU the loader reported, for `reconlEnumerateAdapters`.
#[derive(Clone)]
pub struct AdapterInfo {
    pub description: String,
    pub adapter_luid: u64,
    pub vendor_id: u32,
    pub device_id: u32,
    pub dedicated_video_memory: u64,
    pub shared_system_memory: u64,
    pub feature_level_11: bool,
    pub integrated: bool,
}

/// A live Vulkan device and everything it owns.
///
/// The fields are private to this crate: `render` builds the value and the FFI
/// drives it through the methods below, so no caller can hold a `vk::Image`
/// without the device that made it.
pub struct VulkanDevice {
    entry: Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family: u32,

    vs_module: vk::ShaderModule,
    ps_module: vk::ShaderModule,
    vs_depth_module: vk::ShaderModule,

    pipeline_layout: vk::PipelineLayout,
    /// One pipeline per `BLEND_*` mode: Vulkan bakes blend state into the
    /// pipeline where d3d11 keeps it in a state object.
    /// Colour pipelines, indexed by cull mode then blend mode: cull mode is
    /// baked into a pipeline, and d3d11 keeps three rasteriser states, so the
    /// pair is what gets bound.
    graphics_pipelines: [[vk::Pipeline; 4]; 3],
    /// Running write cursors into the vertex and index upload buffers.
    ///
    /// Every draw in a frame is recorded into one command buffer before anything
    /// is submitted, so a draw that copied its vertices to offset 0 would
    /// overwrite the previous draw's and both would read the last one's geometry.
    /// The cursors append instead, and are reset once per frame.
    upload_vertex: std::cell::Cell<u64>,
    upload_index: std::cell::Cell<u64>,
    depth_pipeline: vk::Pipeline,

    main_pass: vk::RenderPass,
    shadow_pass: vk::RenderPass,

    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,

    point_sampler: vk::Sampler,
    cmp_sampler: vk::Sampler,
    shadow_sampler: vk::Sampler,

    lights: render::MappedBuffer,
    shadow_constants: render::MappedBuffer,
    vertex: render::MappedBuffer,
    index: render::MappedBuffer,
    vertex_capacity: u32,
    index_capacity: u32,

    white: render::Image,

    color: Option<render::Image>,
    depth: Option<render::Image>,
    /// `(map edge, cascade count, image)` of the current cascade set.
    /// Width, height, cascades, image. A map is not square: a host sizing it
    /// from the frame's aspect ratio asks for one that is not.
    shadow_map: Option<(u32, u32, u32, render::Image)>,

    framebuffer: vk::Framebuffer,
    shadow_framebuffers: Vec<vk::Framebuffer>,
    /// One view per cascade, each addressing a single array layer of the shadow
    /// image: a framebuffer names exactly one layer, so the depth pass is begun
    /// once per cascade.
    shadow_layer_views: Vec<vk::ImageView>,
    /// Trivial fragment shader for the depth pass; see `PS_DEPTH_GLSL`.
    ps_depth_module: vk::ShaderModule,
    readback: Option<render::MappedBuffer>,

    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,

    width: u32,
    height: u32,

    counters: Counters,
    shadows: ShadowCounters,
    frame: FrameNumbers,
    color_checksum: u64,

    config: VulkanConfig,
    tier: Tier,
    tier_reason: TierReason,
    caps: u32,
    /// The adapter's own name, kept as an owned string because `device_name()`
    /// returns a borrow and the properties struct it came from is long gone.
    device_name_text: String,
}

impl VulkanDevice {
    pub fn new(_alloc: HostAlloc, _budget: std::sync::Arc<reconl_core::budget::Budget>, config: VulkanConfig) -> Result<Self> {
        render::init_device(&config)
    }

    pub fn backend(&self) -> Backend {
        Backend::Vulkan
    }

    pub fn tier(&self) -> Tier {
        self.tier
    }

    pub fn tier_reason(&self) -> TierReason {
        self.tier_reason
    }

    pub fn caps(&self) -> u32 {
        self.caps
    }

    pub fn device_name(&self) -> &str {
        &self.device_name_text
    }

    pub fn driver(&self) -> &'static str {
        "vulkan"
    }

    pub fn frame_size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn shadow_counters(&self) -> ShadowCounters {
        self.shadows
    }

    pub fn counters(&self) -> Counters {
        self.counters
    }

    pub fn snapshot(&self) -> VulkanSnapshot {
        VulkanSnapshot {
            counters: self.counters,
            shadows: self.shadows,
            frame: self.frame,
            color_checksum: self.color_checksum,
        }
    }

    pub fn color_checksum(&self) -> u64 {
        self.color_checksum
    }

    /// Bytes this device is accountable for. Vulkan allocates through
    /// `vkAllocateMemory`, not the host allocator, so what is reported is the
    /// colour, depth and cascade sets' own `MemoryRequirements.size`.
    pub fn resident_bytes(&self) -> u64 {
        self.color.as_ref().map(|i| i.bytes).unwrap_or(0)
            + self.depth.as_ref().map(|i| i.bytes).unwrap_or(0)
            + self.shadow_map.as_ref().map(|(_, _, _, i)| i.bytes).unwrap_or(0)
            + self.readback.as_ref().map(|b| b.bytes).unwrap_or(0)
    }

    pub fn relabel(&mut self, to: Tier, reason: TierReason) {
        self.tier = to;
        self.tier_reason = reason;
        self.counters.frames_since_tier_change = 0;
        // A new tier caps cascades and the filter differently, so the cascade set
        // is rebuilt by the next `ensure_maps`.
        self.shadow_map = None;
    }

    /// Vulkan has no `GetDeviceRemovedReason`; a lost device surfaces as
    /// `VK_ERROR_DEVICE_LOST` on the call that saw it, which `render` classifies.
    pub fn device_removed(&self) -> bool {
        false
    }

    /// Reserves the frame's targets between frames, so the frame itself grows
    /// nothing.
    pub fn prepare_frame(&mut self, width: u32, height: u32) -> Result<()> {
        render::prepare_frame(self, width, height)
    }

    /// Allocates the cascade set the tier's plan decided on.
    pub fn ensure_maps(&mut self, plan: &ShadowPlan) -> Result<()> {
        render::ensure_maps(self, plan, plan.map_bytes)
    }

    pub fn render(&mut self, input: &FrameInput<'_>) -> Result<FrameNumbers> {
        render::render_frame(self, input)
    }

    pub fn read_frame_into(&mut self, out: &mut [u8], row_bytes: u32, pad: u32) -> Result<()> {
        render::read_frame(self, out, row_bytes, pad)
    }

    /// Depth readback for frame generation. Not implemented in this first cut:
    /// the d3d11 backend reads a depth staging texture, and the Vulkan equivalent
    /// needs its own depth image in `TRANSFER_SRC` plus a second staging buffer.
    /// Reported rather than faked, so a host asking for generated frames gets a
    /// code instead of a frame warped from zeroes.
    pub fn depth_into(&mut self, _out: &mut [f32]) -> Result<()> {
        Err(Error::fmt_at(
            Code::NotSupported,
            file!(),
            line!(),
            "reconl-backend-vulkan",
            format_args!("depth readback is not implemented on the Vulkan backend yet"),
        ))
    }

    pub fn on_frame_end(&mut self) -> Result<()> {
        Ok(())
    }
}

impl Drop for VulkanDevice {
    /// Releases every object this device owns.
    ///
    /// The `vk::Device` and `vk::Instance` handles themselves are left to the
    /// loader: destroying them here would run before the `Image` and
    /// `MappedBuffer` fields' own `Drop` impls (fields drop after this body),
    /// and those impls need a live device to free their memory. Letting the
    /// loader reclaim the two handles at process exit is the safe order; every
    /// allocation this backend made is still freed explicitly, below.
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            // Free every GPU allocation while the device is still alive. The
            // `Image`/`MappedBuffer` values have no `Drop` of their own for
            // exactly this reason: a field's `Drop` runs after this body, and the
            // device would already be torn down.
            if let Some(img) = self.readback.take() {
                img.destroy(&self.device);
            }
            if let Some(img) = self.color.take() {
                img.destroy(&self.device);
            }
            if let Some(img) = self.depth.take() {
                img.destroy(&self.device);
            }
            if let Some((_, _, _, img)) = self.shadow_map.take() {
                img.destroy(&self.device);
            }
            self.white.destroy(&self.device);
            self.lights.destroy(&self.device);
            self.shadow_constants.destroy(&self.device);
            self.vertex.destroy(&self.device);
            self.index.destroy(&self.device);
            if self.framebuffer != vk::Framebuffer::null() {
                self.device.destroy_framebuffer(self.framebuffer, None);
            }
            for fb in self.shadow_framebuffers.drain(..) {
                self.device.destroy_framebuffer(fb, None);
            }
            for view in self.shadow_layer_views.drain(..) {
                self.device.destroy_image_view(view, None);
            }
            self.device.destroy_shader_module(self.ps_depth_module, None);
            self.device.destroy_sampler(self.point_sampler, None);
            self.device.destroy_sampler(self.cmp_sampler, None);
            self.device.destroy_sampler(self.shadow_sampler, None);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            for set in self.graphics_pipelines {
                for p in set {
                    self.device.destroy_pipeline(p, None);
                }
            }
            self.device.destroy_pipeline(self.depth_pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_shader_module(self.vs_module, None);
            self.device.destroy_shader_module(self.ps_module, None);
            self.device.destroy_shader_module(self.vs_depth_module, None);
            self.device.destroy_render_pass(self.main_pass, None);
            self.device.destroy_render_pass(self.shadow_pass, None);
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
            let _ = &self.entry;
            let _ = &self.instance;
        }
    }
}

// ---------------------------------------------------------------------------
// Probe
// ---------------------------------------------------------------------------

/// Whether a Vulkan loader and at least one physical device are present.
///
/// Checked for real rather than assumed: a machine with `vulkan-1.dll` but no
/// driver must not be reported as a GPU this build can render on.
pub fn hardware_available() -> bool {
    match probe_adapters() {
        Ok(adapters) => !adapters.is_empty(),
        Err(_) => false,
    }
}

pub fn probe_adapters() -> Result<Vec<AdapterInfo>> {
    let entry = unsafe { Entry::load() }.map_err(|e| {
        Error::fmt_at(
            Code::BackendUnavailable,
            file!(),
            line!(),
            "reconl-backend-vulkan",
            format_args!("Vulkan loader not available: {e}"),
        )
    })?;

    let app = std::ffi::CString::new("ReconL").unwrap();
    let engine = std::ffi::CString::new("ReconL").unwrap();
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app)
        .application_version(1)
        .engine_name(&engine)
        .engine_version(1)
        .api_version(vk::make_api_version(0, 1, 1, 0));
    let info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = unsafe { entry.create_instance(&info, None) }.map_err(|e| {
        Error::fmt_at(
            Code::BackendUnavailable,
            file!(),
            line!(),
            "reconl-backend-vulkan",
            format_args!("create instance: {e:?}"),
        )
    })?;

    let devices = unsafe { instance.enumerate_physical_devices() }.map_err(|e| {
        Error::fmt_at(
            Code::BackendUnavailable,
            file!(),
            line!(),
            "reconl-backend-vulkan",
            format_args!("enumerate physical devices: {e:?}"),
        )
    })?;

    let mut out = Vec::with_capacity(devices.len());
    for pd in devices {
        let props = unsafe { instance.get_physical_device_properties(pd) };
        let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
        let has_graphics = families
            .iter()
            .any(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS));
        out.push(AdapterInfo {
            description: render::device_name_string(&props.device_name),
            // Vulkan exposes no LUID through core 1.1; a host selects by index.
            adapter_luid: 0,
            vendor_id: props.vendor_id,
            device_id: props.device_id,
            // `maxMemoryAllocationCount` is a count, not bytes: reporting it as
            // memory would be a fabricated number. A driver that answers the
            // memory-budget extension fills this in later; until then it is 0
            // and the tier ladder works from the budget the host declared.
            dedicated_video_memory: 0,
            shared_system_memory: 0,
            feature_level_11: has_graphics,
            integrated: props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU,
        });
    }

    unsafe { instance.destroy_instance(None) };
    Ok(out)
}

/// The capability bits a Vulkan tier advertises.
///
/// Compared with the d3d11 backend's `caps_for`: the same texture, mip, shadow,
/// compute and present bits, with the filter bits read back from the tier's own
/// clamp so a host can only ask for a filter the tier will run.
pub fn caps_for(tier: Tier) -> u32 {
    caps::TEXTURES
        | caps::MIPMAPS
        | caps::SHADOWS
        | caps::COMPUTE
        | caps::PRESENT_TO_MEMORY
        | filter_caps(tier)
}
