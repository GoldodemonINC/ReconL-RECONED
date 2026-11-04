//! The Vulkan render path: device creation, the offscreen targets, both
//! pipelines, and the frame that draws the shadow pass, the colour pass and the
//! readback.
//!
//! # Why there is no swapchain
//!
//! The ABI's benchmark surface presents *to memory*: the host wants the pixels
//! back on the CPU. A swapchain exists to hand images to a window system, which
//! is not what is being asked for, so this backend renders into an offscreen
//! colour image and copies it into a HOST_VISIBLE buffer - the same shape the
//! d3d11 backend uses with a render target and a staging texture. Every surface,
//! present mode and semaphore drops out of the path: one queue, one command
//! buffer, one fence.

use crate::shaders;
use crate::{VulkanConfig, VulkanDevice, VERTEX_STRIDE};

use ash::vk;
use ash::Entry;
use reconl_contract::FrameInput;
use reconl_core::error::{Code, Error, Result};
use reconl_core::stats::{Counters, FrameNumbers, ShadowCounters};
use reconl_core::tier::{shadow_plan, ShadowFilter, ShadowPlan, TierReason};
use reconl_raster::checksum_bytes;
use reconl_raster::shade::LightSet;
use reconl_raster::{DrawItem, ShaderRef, Vertex};
use reconl_shadow as shadow;

use std::ffi::CString;
use std::ptr;

/// Width of the shadow penumbra in world units.
///
/// A shadow's softness is a property of the light and the scene, not of the map
/// resolution: a sun of finite angular size casts a fringe that widens with the
/// distance between caster and receiver. Holding the fringe in world units and
/// converting it per cascade is what keeps the near and far cascades looking
/// like one light instead of two different filters.
const SHADOW_PENUMBRA_WORLD: f32 = 0.035;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// The adapter name as a Rust string.
///
/// `VkPhysicalDeviceProperties::deviceName` is a fixed `[i8; 256]`, NUL-padded
/// and Latin-1-ish, not UTF-16 - converting it as UTF-16 is what produced a
/// mismatched-type error and, before that, a name full of replacement glyphs.
pub(crate) fn device_name_string(raw: &[i8]) -> String {
    let len = raw.iter().position(|&c| c == 0).unwrap_or(raw.len());
    let bytes: Vec<u8> = raw[..len].iter().map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn verr(label: &str, e: vk::Result) -> Error {
    Error::fmt_at(
        Code::BackendUnavailable,
        file!(),
        line!(),
        "reconl-backend-vulkan",
        format_args!("{label}: {e:?}"),
    )
}

fn vfail(label: &str, detail: impl std::fmt::Display) -> Error {
    Error::fmt_at(
        Code::BackendUnavailable,
        file!(),
        line!(),
        "reconl-backend-vulkan",
        format_args!("{label}: {detail}"),
    )
}

// ---------------------------------------------------------------------------
// GLSL -> SPIR-V
// ---------------------------------------------------------------------------

/// Compiles one GLSL source to SPIR-V with the Vulkan SDK's `glslc`.
///
/// Done at device creation rather than at build time so consuming this crate does
/// not require the SDK on the build machine. A missing SDK is reported as an
/// error naming `VULKAN_SDK`, which is the one thing a host can fix.
fn compile_glsl(name: &str, source: &str, stage: &str) -> Result<Vec<u32>> {
    use std::io::Write;
    use std::process::Command;

    let sdk = std::env::var("VULKAN_SDK").map_err(|_| {
        vfail(
            "shader compile",
            format!("VULKAN_SDK is not set; {name} needs glslc from the Vulkan SDK"),
        )
    })?;
    let glslc = std::path::Path::new(&sdk).join("bin").join("glslc.exe");
    if !glslc.exists() {
        return Err(vfail(
            "shader compile",
            format!("{} does not exist", glslc.display()),
        ));
    }

    let dir = std::env::temp_dir().join("reconl-vulkan");
    std::fs::create_dir_all(&dir).map_err(|e| vfail("shader compile", e))?;
    let glsl_path = dir.join(format!("{name}.{stage}.glsl"));
    let spv_path = dir.join(format!("{name}.{stage}.spv"));

    {
        let mut f = std::fs::File::create(&glsl_path).map_err(|e| vfail("shader compile", e))?;
        f.write_all(source.as_bytes())
            .map_err(|e| vfail("shader compile", e))?;
    }

    let out = Command::new(&glslc)
        .arg(format!("-fshader-stage={stage}"))
        .arg("--target-env=vulkan1.1")
        .arg("-O")
        .arg("-o")
        .arg(&spv_path)
        .arg(&glsl_path)
        .output()
        .map_err(|e| vfail("shader compile", e))?;

    if !out.status.success() {
        return Err(vfail(
            "shader compile",
            format!("glslc {name}: {}", String::from_utf8_lossy(&out.stderr)),
        ));
    }

    let bytes = std::fs::read(&spv_path).map_err(|e| vfail("shader compile", e))?;
    let _ = std::fs::remove_file(&glsl_path);
    let _ = std::fs::remove_file(&spv_path);

    if bytes.len() % 4 != 0 {
        return Err(vfail("shader compile", "SPIR-V is not a multiple of 4 bytes"));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn shader_module(
    device: &ash::Device,
    name: &str,
    source: &str,
    stage: &str,
) -> Result<vk::ShaderModule> {
    let words = compile_glsl(name, source, stage)?;
    let info = vk::ShaderModuleCreateInfo::default().code(&words);
    unsafe { device.create_shader_module(&info, None) }.map_err(|e| verr("create shader module", e))
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

fn memory_type(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    bits: u32,
    wanted: vk::MemoryPropertyFlags,
) -> Result<u32> {
    let props = unsafe { instance.get_physical_device_memory_properties(physical) };
    for i in 0..props.memory_type_count {
        if bits & (1 << i) != 0 && props.memory_types[i as usize].property_flags.contains(wanted) {
            return Ok(i);
        }
    }
    Err(vfail(
        "memory type",
        format!("no type with {wanted:?} among {bits:#x}"),
    ))
}

/// An image and the memory behind it, freed when the value drops.
pub struct Image {
    pub device: vk::Device,
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub memory: vk::DeviceMemory,
    pub bytes: u64,
    pub edge: u32,
    pub layers: u32,
}

impl Image {
    fn new(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        device: &ash::Device,
        width: u32,
        height: u32,
        layers: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        aspect: vk::ImageAspectFlags,
        view_type: vk::ImageViewType,
    ) -> Result<Self> {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&info, None) }.map_err(|e| verr("create image", e))?;

        let req = unsafe { device.get_image_memory_requirements(image) };
        let index = memory_type(instance, physical, req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(index);
        let memory = unsafe { device.allocate_memory(&alloc, None) }
            .map_err(|e| verr("allocate image memory", e))?;
        unsafe { device.bind_image_memory(image, memory, 0) }
            .map_err(|e| verr("bind image memory", e))?;

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(view_type)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(aspect)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(layers),
            );
        let view = unsafe { device.create_image_view(&view_info, None) }
            .map_err(|e| verr("create image view", e))?;

        Ok(Self {
            device: device.handle(),
            image,
            view,
            memory,
            bytes: req.size,
            edge: width,
            layers,
        })
    }
}

impl Image {
    /// Frees the view, the image and its memory, in that order.
    ///
    /// Called by `VulkanDevice::drop` rather than from a `Drop` impl: the device
    /// that owns these must still be alive when they are freed, and a field's
    /// own `Drop` runs after the owner's body has finished.
    pub(crate) fn destroy(&self, device: &ash::Device) {
        unsafe {
            if self.view != vk::ImageView::null() {
                device.destroy_image_view(self.view, None);
            }
            if self.image != vk::Image::null() {
                device.destroy_image(self.image, None);
            }
            if self.memory != vk::DeviceMemory::null() {
                device.free_memory(self.memory, None);
            }
        }
    }
}

/// A HOST_VISIBLE buffer that stays mapped for its life.
pub struct MappedBuffer {
    pub device: vk::Device,
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub ptr: *mut u8,
    pub bytes: u64,
}

impl MappedBuffer {
    fn new(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        device: &ash::Device,
        bytes: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        let info = vk::BufferCreateInfo::default()
            .size(bytes.max(4))
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.create_buffer(&info, None) }.map_err(|e| verr("create buffer", e))?;
        let req = unsafe { device.get_buffer_memory_requirements(buffer) };
        // Prefer a *cached* host-visible heap for anything the host reads back.
        // HOST_VISIBLE | HOST_COHERENT alone resolves to the write-combined
        // heap, and reading 8 MiB out of that costs ~24 ms a frame on Intel
        // integrated - the entire present phase. Cached unified memory reads at
        // RAM speed, which is what the D3D11 backend's staging texture gets.
        let host_visible = vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT
            | vk::MemoryPropertyFlags::HOST_CACHED;
        let index = memory_type(instance, physical, req.memory_type_bits, host_visible)
            .or_else(|_| {
                memory_type(
                    instance,
                    physical,
                    req.memory_type_bits,
                    vk::MemoryPropertyFlags::HOST_VISIBLE
                        | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
            })?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(index);
        let memory = unsafe { device.allocate_memory(&alloc, None) }
            .map_err(|e| verr("allocate buffer memory", e))?;
        unsafe { device.bind_buffer_memory(buffer, memory, 0) }
            .map_err(|e| verr("bind buffer memory", e))?;
        let ptr = unsafe { device.map_memory(memory, 0, req.size, vk::MemoryMapFlags::empty()) }
            .map_err(|e| verr("map buffer memory", e))? as *mut u8;
        Ok(Self { device: device.handle(), buffer, memory, ptr, bytes: req.size })
    }
}

impl MappedBuffer {
    /// Unmaps the memory, frees it, then destroys the buffer. Called from
    /// `VulkanDevice::drop` while the device is still alive.
    pub(crate) fn destroy(&self, device: &ash::Device) {
        unsafe {
            if self.memory != vk::DeviceMemory::null() {
                device.unmap_memory(self.memory);
                device.free_memory(self.memory, None);
            }
            if self.buffer != vk::Buffer::null() {
                device.destroy_buffer(self.buffer, None);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Constant-buffer layouts
// ---------------------------------------------------------------------------

/// The push-constant block every draw sets. 96 bytes, inside the 128 every
/// Vulkan implementation guarantees.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct DrawPush {
    pub transform: [f32; 16],
    pub flags: [f32; 4],
    pub flags1: [f32; 4],
}

/// The lights, in `reconl-raster`'s element order. 656 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LightsUbo {
    pub light_a: [[f32; 4]; 16],
    pub light_b: [[f32; 4]; 16],
    pub light_c: [[f32; 4]; 16],
    pub light_d: [[f32; 4]; 16],
    pub ambient_count: [f32; 4],
}

/// The whole shadow lookup, one buffer so the cascade scan is a dynamic index.
/// 368 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ShadowUbo {
    pub view_z_row: [f32; 4],
    pub params_a: [f32; 4],
    pub params_b: [f32; 4],
    /// x map width, y map height. The set is not necessarily square, so a tap's
    /// texel step needs both axes.
    pub params_c: [f32; 4],
    pub cascade_mvp: [[f32; 16]; shadow::MAX_CASCADES],
    pub cascade_params: [[f32; 4]; shadow::MAX_CASCADES],
}

impl Default for ShadowUbo {
    fn default() -> Self {
        Self {
            view_z_row: [0.0; 4],
            params_a: [0.0; 4],
            params_b: [1.0, 1.0, 0.0, 1.0],
            params_c: [0.0; 4],
        cascade_mvp: [[0.0; 16]; shadow::MAX_CASCADES],
            cascade_params: [[0.0; 4]; shadow::MAX_CASCADES],
        }
    }
}

/// The light array in `reconl-raster`'s element order - the same packing the
/// d3d11 backend's `lights_cb` builds.
fn lights_ubo(lights: &LightSet) -> LightsUbo {
    let mut cb = LightsUbo {
        light_a: [[0.0; 4]; 16],
        light_b: [[0.0; 4]; 16],
        light_c: [[0.0; 4]; 16],
        light_d: [[0.0; 4]; 16],
        ambient_count: [
            lights.ambient[0],
            lights.ambient[1],
            lights.ambient[2],
            lights.count as f32,
        ],
    };
    for (slot, light) in lights.lights.iter().take(lights.count as usize).enumerate() {
        let Some(light) = light else { continue };
        cb.light_a[slot] = [
            light.position[0],
            light.position[1],
            light.position[2],
            light.kind as f32,
        ];
        cb.light_b[slot] = [
            light.direction[0],
            light.direction[1],
            light.direction[2],
            light.intensity,
        ];
        cb.light_c[slot] = [light.color[0], light.color[1], light.color[2], light.range];
        cb.light_d[slot] = [
            light.cos_inner,
            light.cos_outer,
            if light.cast_shadow { 1.0 } else { 0.0 },
            0.0,
        ];
    }
    cb
}

fn draw_triangles(draw: &DrawItem<'_>) -> u64 {
    let count = match draw.indices {
        Some(indices) => indices.len(),
        None => draw.vertices.len(),
    };
    (count / 3) as u64
}

fn count_shadowed_lights(lights: &LightSet) -> u32 {
    lights
        .lights
        .iter()
        .take(lights.count as usize)
        .flatten()
        .filter(|l| l.cast_shadow)
        .count() as u32
}

// ---------------------------------------------------------------------------
// Device creation
// ---------------------------------------------------------------------------

pub fn init_device(config: &VulkanConfig) -> Result<VulkanDevice> {
    let entry = unsafe { Entry::load() }.map_err(|e| vfail("load vulkan", e))?;

    let app = CString::new("ReconL").unwrap();
    let engine = CString::new("ReconL").unwrap();
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app)
        .application_version(1)
        .engine_name(&engine)
        .engine_version(1)
        .api_version(vk::make_api_version(0, 1, 1, 0));
    let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = unsafe { entry.create_instance(&instance_info, None) }
        .map_err(|e| verr("create instance", e))?;

    let devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(|e| verr("enumerate physical devices", e))?;
    if devices.is_empty() {
        return Err(vfail("physical device", "no Vulkan device found"));
    }

    // Prefer an integrated device so the tier matches what d3d11 picks on this
    // hardware; any usable device is accepted otherwise.
    let mut physical_device = devices[0];
    for &candidate in &devices {
        let p = unsafe { instance.get_physical_device_properties(candidate) };
        if p.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU {
            physical_device = candidate;
            break;
        }
    }
    let props = unsafe { instance.get_physical_device_properties(physical_device) };
    let device_name_text = device_name_string(&props.device_name);

    let families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let family = families
        .iter()
        .position(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .ok_or_else(|| vfail("queue family", "no graphics queue family"))? as u32;

    let priority = [1.0f32];
    let queue_info = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(family)
        .queue_priorities(&priority)];
    let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_info);
    let device = unsafe { instance.create_device(physical_device, &device_info, None) }
        .map_err(|e| verr("create device", e))?;
    let queue = unsafe { device.get_device_queue(family, 0) };

    // ---- render passes ----------------------------------------------------
    let (main_pass, shadow_pass) = create_render_passes(&device)?;

    // ---- pipelines --------------------------------------------------------
    let vs_module = shader_module(&device, "reconl_vs", shaders::VS_GLSL, "vertex")?;
    let ps_module = shader_module(&device, "reconl_ps", shaders::PS_GLSL, "fragment")?;
    let ps_depth_module =
        shader_module(&device, "reconl_depth_ps", shaders::PS_DEPTH_GLSL, "fragment")?;
    let vs_depth_module =
        shader_module(&device, "reconl_vs_depth", shaders::VS_DEPTH_GLSL, "vertex")?;

    let push_range = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(std::mem::size_of::<DrawPush>() as u32)];
    let set_layout = descriptor_set_layout(&device)?;
    let set_layouts = [set_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_range);
    let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
        .map_err(|e| verr("create pipeline layout", e))?;

    let mut graphics_pipelines = [[vk::Pipeline::null(); 4]; 3];
    for cull in 0..3u32 {
        for mode in 0..4u32 {
            graphics_pipelines[cull as usize][mode as usize] = create_graphics_pipeline(
                &device,
                main_pass,
                pipeline_layout,
                vs_module,
                ps_module,
                mode,
                cull,
            )?;
        }
    }
    let depth_pipeline =
        create_depth_pipeline(&device, shadow_pass, pipeline_layout, vs_depth_module, ps_depth_module)?;

    // ---- descriptors and samplers -----------------------------------------
    let descriptor_pool = create_descriptor_pool(&device)?;
    let descriptor_set = allocate_descriptor_set(&device, descriptor_pool, set_layout)?;
    let (point_sampler, cmp_sampler) = create_samplers(&device)?;
    let shadow_sampler = create_shadow_sampler(&device, false)?;

    let white = Image::new(
        &instance,
        physical_device,
        &device,
        1,
        1,
        1,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::SAMPLED,
        vk::ImageAspectFlags::COLOR,
        vk::ImageViewType::TYPE_2D,
    )?;

    // ---- constant and upload buffers --------------------------------------
    let lights = MappedBuffer::new(
        &instance,
        physical_device,
        &device,
        std::mem::size_of::<LightsUbo>() as u64,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
    )?;
    let shadow_constants = MappedBuffer::new(
        &instance,
        physical_device,
        &device,
        std::mem::size_of::<ShadowUbo>() as u64,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
    )?;

    let vertex_capacity = 1u32 << 16;
    let vertex = MappedBuffer::new(
        &instance,
        physical_device,
        &device,
        u64::from(vertex_capacity) * u64::from(VERTEX_STRIDE),
        vk::BufferUsageFlags::VERTEX_BUFFER,
    )?;
    let index_capacity = 1u32 << 17;
    let index = MappedBuffer::new(
        &instance,
        physical_device,
        &device,
        u64::from(index_capacity) * 4,
        vk::BufferUsageFlags::INDEX_BUFFER,
    )?;

    // ---- command recording ------------------------------------------------
    let pool_info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(family);
    let command_pool = unsafe { device.create_command_pool(&pool_info, None) }
        .map_err(|e| verr("create command pool", e))?;
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let command_buffer = unsafe { device.allocate_command_buffers(&alloc_info) }
        .map_err(|e| verr("allocate command buffer", e))?[0];

    let fence = unsafe {
        device.create_fence(
            &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
            None,
        )
    }
    .map_err(|e| verr("create fence", e))?;

    let mut dev = VulkanDevice {
        entry,
        instance,
        physical_device,
        device,
        queue,
        queue_family: family,
        vs_module,
        ps_module,
        vs_depth_module,
        pipeline_layout,
        graphics_pipelines,
        upload_vertex: std::cell::Cell::new(0),
        upload_index: std::cell::Cell::new(0),
        depth_pipeline,
        main_pass,
        shadow_pass,
        descriptor_set_layout: set_layout,
        descriptor_pool,
        descriptor_set,
        point_sampler,
        cmp_sampler,
        shadow_sampler,
        lights,
        shadow_constants,
        vertex,
        index,
        vertex_capacity,
        index_capacity,
        white,
        color: None,
        depth: None,
        shadow_map: None,
        framebuffer: vk::Framebuffer::null(),
        shadow_framebuffers: Vec::new(),
        shadow_layer_views: Vec::new(),
        ps_depth_module,
        readback: None,
        command_pool,
        command_buffer,
        fence,
        width: 0,
        height: 0,
        counters: Counters::default(),
        shadows: ShadowCounters::default(),
        frame: FrameNumbers::default(),
        color_checksum: 0,
        config: config.clone(),
        tier: config.tier,
        tier_reason: TierReason::HostRequest,
        caps: crate::caps_for(config.tier),
        device_name_text,
    };

    dev.write_descriptors()?;
    Ok(dev)
}

// ---------------------------------------------------------------------------
// Render passes
// ---------------------------------------------------------------------------

fn create_render_passes(device: &ash::Device) -> Result<(vk::RenderPass, vk::RenderPass)> {
    // Main pass: colour with a TRANSFER_SRC final layout so the readback copy is
    // legal without a second barrier, plus the depth attachment.
    let attachments = [
        vk::AttachmentDescription::default()
            .format(vk::Format::R8G8B8A8_UNORM)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
        vk::AttachmentDescription::default()
            .format(vk::Format::D32_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
    ];
    let color_ref = [vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
    let depth_ref = vk::AttachmentReference::default()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let subpass = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_ref)
        .depth_stencil_attachment(&depth_ref)];
    let deps = [vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .src_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        )
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        )];
    let info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpass)
        .dependencies(&deps);
    let main_pass = unsafe { device.create_render_pass(&info, None) }
        .map_err(|e| verr("create render pass", e))?;

    // Shadow pass: depth only. Final layout SHADER_READ_ONLY so the colour pass
    // samples it.
    let shadow_attachments = [vk::AttachmentDescription::default()
        .format(vk::Format::D32_SFLOAT)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
    // The subpass has to *name* the depth attachment. Declaring it on the render
    // pass and not referencing it here is legal and silently renders nothing: the
    // map keeps its clear value, every comparison then reads a stored depth of
    // 0.0, and the whole frame comes out unshadowed.
    let shadow_depth_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let shadow_subpass = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .depth_stencil_attachment(&shadow_depth_ref)];
    let shadow_deps = [
        // Earlier reads of the map finish before this pass writes it.
        vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
            .dst_stage_mask(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .src_access_mask(vk::AccessFlags::SHADER_READ)
            .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE),
        // And this pass's writes are visible to the colour pass's taps.
        vk::SubpassDependency::default()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::LATE_FRAGMENT_TESTS)
            .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
            .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ),
    ];
    let shadow_info = vk::RenderPassCreateInfo::default()
        .attachments(&shadow_attachments)
        .subpasses(&shadow_subpass)
        .dependencies(&shadow_deps);
    let shadow_pass = unsafe { device.create_render_pass(&shadow_info, None) }
        .map_err(|e| verr("create shadow render pass", e))?;

    Ok((main_pass, shadow_pass))
}

// ---------------------------------------------------------------------------
// Pipelines
// ---------------------------------------------------------------------------

fn vertex_input() -> (
    Vec<vk::VertexInputBindingDescription>,
    Vec<vk::VertexInputAttributeDescription>,
) {
    // The same offsets the d3d11 input layout declares.
    let bindings = vec![vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(VERTEX_STRIDE)
        .input_rate(vk::VertexInputRate::VERTEX)];
    let attrs = vec![
        vk::VertexInputAttributeDescription::default()
            .location(0)
            .binding(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .location(1)
            .binding(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(12),
        vk::VertexInputAttributeDescription::default()
            .location(2)
            .binding(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(24),
        vk::VertexInputAttributeDescription::default()
            .location(3)
            .binding(0)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(32),
    ];
    (bindings, attrs)
}

fn color_blend_attachment(mode: u32) -> vk::PipelineColorBlendAttachmentState {
    let base = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA);
    match mode {
        1 => base
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD),
        2 => base
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE)
            .alpha_blend_op(vk::BlendOp::ADD),
        3 => base
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::DST_COLOR)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ZERO)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE)
            .alpha_blend_op(vk::BlendOp::ADD),
        _ => base.blend_enable(false),
    }
}

fn create_graphics_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vs: vk::ShaderModule,
    ps: vk::ShaderModule,
    blend_mode: u32,
    cull: u32,
) -> Result<vk::Pipeline> {
    let entry = CString::new("main").unwrap();
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vs)
            .name(&entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(ps)
            .name(&entry),
    ];
    let (bindings, attrs) = vertex_input();
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attrs);
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    // CLOCKWISE front faces with the vertex shader's Y flip gives the same
    // front/back verdict the d3d11 backend gets from `FrontCounterClockwise:
    // TRUE`. Cull mode itself comes from `pipeline.cull` - the reference scene
    // does not use one winding for everything, and culling the wrong side drops
    // whole draws rather than shading them wrongly.
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(cull_flags(cull))
        .front_face(vk::FrontFace::CLOCKWISE)
        .line_width(1.0)
        .depth_bias_enable(false);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::GREATER) // reversed-Z
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);
    let attachment = [color_blend_attachment(blend_mode)];
    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachment);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);

    let pipelines =
        unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) }
            .map_err(|(_, e)| verr("create graphics pipeline", e))?;
    Ok(pipelines[0])
}

fn create_depth_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vs: vk::ShaderModule,
    ps: vk::ShaderModule,
) -> Result<vk::Pipeline> {
    let entry = CString::new("main").unwrap();
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vs)
            .name(&entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(ps)
            .name(&entry),
    ];
    let (bindings, attrs) = vertex_input();
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attrs);
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    // No culling in the depth pass. Back-face culling here dropped every caster:
    // the light's view-projection flips the winding relative to the camera's, so
    // the faces that are front-facing to the light are wound the other way and
    // `CULL_BACK` discarded all of them. The map then kept its clear value, every
    // comparison read a stored depth of 0.0, and the frame rendered fully lit with
    // no error anywhere. Rendering both sides is also what a depth-only pass wants:
    // a single-sided caster leaks light through its open side.
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::CLOCKWISE)
        .line_width(1.0)
        // Bias lives in the pixel shader, applied at lookup, so the depth this
        // pass writes stays unbiased and the golden does not move.
        .depth_bias_enable(false);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::GREATER)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default();
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);

    let pipelines =
        unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) }
            .map_err(|(_, e)| verr("create depth pipeline", e))?;
    Ok(pipelines[0])
}

// ---------------------------------------------------------------------------
// Descriptors and samplers
// ---------------------------------------------------------------------------

fn descriptor_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
    let sampler = |binding: u32| {
        vk::DescriptorSetLayoutBinding::default()
            .binding(binding)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
    };
    let uniform = |binding: u32| {
        vk::DescriptorSetLayoutBinding::default()
            .binding(binding)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
    };
    let bindings = [sampler(0), sampler(1), sampler(2), uniform(3), uniform(4)];
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&info, None) }
        .map_err(|e| verr("create descriptor set layout", e))
}

fn create_descriptor_pool(device: &ash::Device) -> Result<vk::DescriptorPool> {
    let sizes = [
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(8),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(8),
    ];
    let info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(8)
        .pool_sizes(&sizes);
    unsafe { device.create_descriptor_pool(&info, None) }
        .map_err(|e| verr("create descriptor pool", e))
}

fn allocate_descriptor_set(
    device: &ash::Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
) -> Result<vk::DescriptorSet> {
    let layouts = [layout];
    let info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    let sets = unsafe { device.allocate_descriptor_sets(&info) }
        .map_err(|e| verr("allocate descriptor set", e))?;
    Ok(sets[0])
}

fn create_samplers(device: &ash::Device) -> Result<(vk::Sampler, vk::Sampler)> {
    let nearest = || {
        vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_lod(0.0)
    };
    let point = unsafe { device.create_sampler(&nearest(), None) }
        .map_err(|e| verr("create point sampler", e))?;
    // Reversed-Z comparison: lit when the reference depth is >= the stored one,
    // which is `GREATER_OR_EQUAL` in `[0,1]` depth.
    //
    // LINEAR here is the hardware PCF path: a filtered comparison returns the
    // bilinear blend of four texel comparisons in one fetch, so a four-tap
    // kernel covers sixteen texels. NEAREST turned every tap into a single
    // binary sample, which is why a 3x3 box at 2048 showed the texel grid as
    // stair steps.
    let cmp = unsafe {
        device.create_sampler(
            &nearest()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .compare_enable(true)
                .compare_op(vk::CompareOp::GREATER_OR_EQUAL),
            None,
        )
    }
    .map_err(|e| verr("create compare sampler", e))?;
    Ok((point, cmp))
}

fn create_shadow_sampler(device: &ash::Device, compare: bool) -> Result<vk::Sampler> {
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::NEAREST)
        .min_filter(vk::Filter::NEAREST)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .compare_enable(compare)
        .compare_op(vk::CompareOp::GREATER_OR_EQUAL)
        .max_lod(0.0);
    unsafe { device.create_sampler(&info, None) }.map_err(|e| verr("create shadow sampler", e))
}

// ---------------------------------------------------------------------------
// Frame preparation
// ---------------------------------------------------------------------------

pub fn prepare_frame(dev: &mut VulkanDevice, width: u32, height: u32) -> Result<()> {
    let scale = dev.config.resolution_scale.clamp(0.05, 1.0);
    let w = (((width as f32) * scale).round() as u32).clamp(1, width.max(1));
    let h = (((height as f32) * scale).round() as u32).clamp(1, height.max(1));
    if dev.width == w && dev.height == h && dev.color.is_some() {
        return Ok(());
    }
    dev.width = w;
    dev.height = h;

    // Drop the old targets before asking for the new ones: a tier that changed
    // size must not peak at both.
    dev.color = None;
    dev.depth = None;
    dev.readback = None;
    if dev.framebuffer != vk::Framebuffer::null() {
        unsafe { dev.device.destroy_framebuffer(dev.framebuffer, None) };
        dev.framebuffer = vk::Framebuffer::null();
    }

    let color = Image::new(
        &dev.instance,
        dev.physical_device,
        &dev.device,
        w,
        h,
        1,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::ImageAspectFlags::COLOR,
        vk::ImageViewType::TYPE_2D,
    )?;
    let depth = Image::new(
        &dev.instance,
        dev.physical_device,
        &dev.device,
        w,
        h,
        1,
        vk::Format::D32_SFLOAT,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        vk::ImageAspectFlags::DEPTH,
        vk::ImageViewType::TYPE_2D,
    )?;
    let readback = MappedBuffer::new(
        &dev.instance,
        dev.physical_device,
        &dev.device,
        u64::from(w) * u64::from(h) * 4,
        vk::BufferUsageFlags::TRANSFER_DST,
    )?;

    let attachments = [color.view, depth.view];
    let info = vk::FramebufferCreateInfo::default()
        .render_pass(dev.main_pass)
        .attachments(&attachments)
        .width(w)
        .height(h)
        .layers(1);
    dev.framebuffer = unsafe { dev.device.create_framebuffer(&info, None) }
        .map_err(|e| verr("create framebuffer", e))?;

    dev.color = Some(color);
    dev.depth = Some(depth);
    dev.readback = Some(readback);
    Ok(())
}

pub fn ensure_maps(dev: &mut VulkanDevice, plan: &ShadowPlan, requested_bytes: u64) -> Result<()> {
    let cascades = plan.cascades.max(1);

    // Two ways to size the set.
    //
    // The plan's way - square, power-of-two, budget scaled by tier and split
    // across cascades - is what the reference tier and the golden were produced
    // with, so it stays the default.
    //
    // The exact way sizes a single map to the *requested* budget at the frame's own
    // aspect ratio, with no tier scale, no split across cascades and no rounding to
    // a power of two. That is how a host asks for shadows at a fixed fraction of
    // the frame: 14,745,600 bytes on a 6144x3456 frame is exactly 2560x1440.
    let exact = std::env::var("RECONL_VULKAN_SHADOW_FROM_BUDGET")
        .map(|v| v != "0")
        .unwrap_or(false);
    let (width, height) = if exact {
        let texels = (requested_bytes / 4).max(1) as f64;
        let aspect = (dev.width.max(1) as f64) / (dev.height.max(1) as f64);
        let w = (texels * aspect).sqrt().max(1.0).min(4096.0) as u32;
        let h = (texels / aspect).sqrt().max(1.0).min(4096.0) as u32;
        (w, h)
    } else {
        (plan.map_size, plan.map_size)
    };

    if let Some((existing_w, existing_h, existing_cascades, _)) = &dev.shadow_map {
        if *existing_w == width && *existing_h == height && *existing_cascades == cascades {
            return Ok(());
        }
    }

    for fb in dev.shadow_framebuffers.drain(..) {
        unsafe { dev.device.destroy_framebuffer(fb, None) };
    }
    for view in dev.shadow_layer_views.drain(..) {
        unsafe { dev.device.destroy_image_view(view, None) };
    }
    if let Some((_, _, _, old)) = dev.shadow_map.take() {
        old.destroy(&dev.device);
    }

    // One array layer per cascade. The shader indexes the layer with the
    // cascade's `map_index`, which is what lets the cascade scan sample a
    // different map per tap instead of reading cascade 0 for all of them.
    let image = Image::new(
        &dev.instance,
        dev.physical_device,
        &dev.device,
        width,
        height,
        cascades,
        vk::Format::D32_SFLOAT,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        vk::ImageAspectFlags::DEPTH,
        vk::ImageViewType::TYPE_2D_ARRAY,
    )?;

    // A framebuffer names exactly one layer, so each cascade gets its own view
    // and its own framebuffer; the depth pass is begun once per cascade.
    for layer in 0..cascades {
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image.image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(layer)
                    .layer_count(1),
            );
        let view = unsafe { dev.device.create_image_view(&view_info, None) }
            .map_err(|e| verr("create shadow layer view", e))?;
        dev.shadow_layer_views.push(view);
        let attachments = [view];
        let fb_info = vk::FramebufferCreateInfo::default()
            .render_pass(dev.shadow_pass)
            .attachments(&attachments)
            .width(width)
            .height(height)
            .layers(1);
        let fb = unsafe { dev.device.create_framebuffer(&fb_info, None) }
            .map_err(|e| verr("create shadow framebuffer", e))?;
        dev.shadow_framebuffers.push(fb);
    }
    dev.shadow_map = Some((width, height, cascades, image));

    dev.write_descriptors()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Descriptor writes
// ---------------------------------------------------------------------------

impl VulkanDevice {
    pub(crate) fn write_descriptors(&mut self) -> Result<()> {
        let white_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(self.white.view)
            .sampler(self.point_sampler)];

        // With no cascade set yet the white texture goes into both shadow slots:
        // the shader's fail-safe returns 1.0 (fully lit) outside the map, so an
        // unshadowed frame is what such a device draws.
        let shadow_view = self
            .shadow_map
            .as_ref()
            .map(|(_, _, _, img)| img.view)
            .unwrap_or(self.white.view);
        let shadow_cmp = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(shadow_view)
            .sampler(self.cmp_sampler)];
        let shadow_raw = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(shadow_view)
            .sampler(self.point_sampler)];

        let lights_info = [vk::DescriptorBufferInfo::default()
            .buffer(self.lights.buffer)
            .offset(0)
            .range(std::mem::size_of::<LightsUbo>() as u64)];
        let shadow_info = [vk::DescriptorBufferInfo::default()
            .buffer(self.shadow_constants.buffer)
            .offset(0)
            .range(std::mem::size_of::<ShadowUbo>() as u64)];

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&white_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&shadow_cmp),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&shadow_raw),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&lights_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&shadow_info),
        ];
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The frame
// ---------------------------------------------------------------------------

pub fn render_frame(dev: &mut VulkanDevice, input: &FrameInput<'_>) -> Result<FrameNumbers> {
    // One cursor pair for the whole frame: the shadow pass and the colour pass
    // are recorded into the same command buffer, so they append to each other
    // rather than overwriting.
    dev.upload_vertex.set(0);
    dev.upload_index.set(0);

    use std::time::Instant;

    let frame_start = Instant::now();
    prepare_frame(dev, input.width, input.height)?;

    let plan = if input.shadow.enabled && (dev.caps & reconl_core::tier::caps::SHADOWS) != 0 {
        Some(shadow_plan(
            dev.tier,
            input.shadow.cascades,
            input.shadow.texel_budget_bytes,
            input.shadow.filter,
            dev.caps,
        ))
    } else {
        None
    };

    // ---- shadow pass: fit, then record -----------------------------------
    let shadow_start = Instant::now();
    let mut fit_ns = 0u64;
    let mut cascades_rendered = 0u32;
    let mut shadowed_triangles = 0u64;
    let fits = if let Some(p) = &plan {
        ensure_maps(dev, p, input.shadow.texel_budget_bytes)?;
        let (width, height) = dev
            .shadow_map
            .as_ref()
            .map(|(w, h, _, _)| (*w, *h))
            .unwrap_or((p.map_size, p.map_size));
        let size = width.max(height);
        let fit_start = Instant::now();
        let fits = shadow::fit_cascades(&shadow::FitInput {
            camera_view: input.camera_view,
            fov_y_deg: input.fov_y_deg,
            aspect: input.aspect,
            near: input.near,
            light_dir: input.light_dir,
            cascade_count: p.cascades,
            max_distance: input.shadow.max_distance,
            split_lambda: dev.config.split_lambda,
            map_size: size,
            snap: true,
        });
        fit_ns += fit_start.elapsed().as_nanos() as u64;

        dev.shadows.map_width = width;
        dev.shadows.map_height = height;
        // What the set actually occupies, not what the plan asked for: in
        // budget-exact mode those differ, and a host reading this is sizing its
        // own resources from it.
        dev.shadows.map_bytes = u64::from(width) * u64::from(height) * 4 * u64::from(p.cascades);
        dev.shadows.cascades_active = p.cascades;
        dev.shadows.filter_active = p.filter;
        dev.shadows.filter_requested = p.filter_requested;
        dev.shadows.filter_taps = p.filter.taps();
        dev.shadows.shadowed_lights = count_shadowed_lights(&input.lights);
        fits
    } else {
        dev.shadows = ShadowCounters::default();
        shadow::CascadeSet::empty()
    };

    // Upload the lights and the shadow lookup before recording: both are
    // per-frame, not per-draw.
    unsafe {
        let lights = lights_ubo(&input.lights);
        ptr::copy_nonoverlapping(
            &lights as *const LightsUbo as *const u8,
            dev.lights.ptr,
            std::mem::size_of::<LightsUbo>(),
        );

        let shadow_cb = shadow_ubo(&fits, input, plan.as_ref().map(|p| p.filter), dev)?;
        ptr::copy_nonoverlapping(
            &shadow_cb as *const ShadowUbo as *const u8,
            dev.shadow_constants.ptr,
            std::mem::size_of::<ShadowUbo>(),
        );
    }

    let cmd = dev.command_buffer;
    unsafe {
        dev.device
            .wait_for_fences(&[dev.fence], true, u64::MAX)
            .map_err(|e| verr("wait for fence", e))?;
        dev.device
            .reset_fences(&[dev.fence])
            .map_err(|e| verr("reset fence", e))?;
        dev.device
            .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
            .map_err(|e| verr("reset command buffer", e))?;
        dev.device
            .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
            .map_err(|e| verr("begin command buffer", e))?;
    }

    if let Some((map_width, map_height, cascades, _)) = &dev.shadow_map {
        let width = *map_width;
        let height = *map_height;
        let cascade_list: Vec<_> = fits.iter().take(*cascades as usize).collect();
        let viewport = vk::Viewport::default()
            .width(width as f32)
            .height(height as f32)
            .min_depth(0.0)
            .max_depth(1.0);
        let area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        };
        let clear = [vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 },
        }];
        for (index, fit) in cascade_list.iter().enumerate() {
            let info = vk::RenderPassBeginInfo::default()
                .render_pass(dev.shadow_pass)
                .framebuffer(dev.shadow_framebuffers[index])
                .render_area(area)
                .clear_values(&clear);
            unsafe {
                dev.device
                    .cmd_begin_render_pass(cmd, &info, vk::SubpassContents::INLINE);
                dev.device.cmd_set_viewport(cmd, 0, &[viewport]);
                dev.device.cmd_set_scissor(cmd, 0, &[area]);
                dev.device
                    .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, dev.depth_pipeline);
                dev.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    dev.pipeline_layout,
                    0,
                    &[dev.descriptor_set],
                    &[],
                );
            }
            for draw in input.draws.iter().filter(|d| d.casts_shadow) {
                shadowed_triangles += draw_triangles(draw);
                let push = DrawPush {
                    transform: reconl_raster::math::mul(&fit.view_proj, &draw.model),
                    flags: [0.0; 4],
                    flags1: [0.0; 4],
                };
                record_draw(dev, draw, &push)?;
            }
            unsafe { dev.device.cmd_end_render_pass(cmd) };
            cascades_rendered += 1;
        }
    }

    let shadow_ns = shadow_start.elapsed().as_nanos() as u64;

    // ---- colour pass ------------------------------------------------------
    let raster_start = Instant::now();
    let viewport = vk::Viewport::default()
        .width(dev.width as f32)
        .height(dev.height as f32)
        .min_depth(0.0)
        .max_depth(1.0);
    let area = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D { width: dev.width, height: dev.height },
    };
    let clears = [
        vk::ClearValue {
            color: vk::ClearColorValue { float32: input.clear_color },
        },
        vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: input.clear_depth,
                stencil: 0,
            },
        },
    ];
    let info = vk::RenderPassBeginInfo::default()
        .render_pass(dev.main_pass)
        .framebuffer(dev.framebuffer)
        .render_area(area)
        .clear_values(&clears);
    unsafe {
        dev.device
            .cmd_begin_render_pass(cmd, &info, vk::SubpassContents::INLINE);
        dev.device.cmd_set_viewport(cmd, 0, &[viewport]);
        dev.device.cmd_set_scissor(cmd, 0, &[area]);
        dev.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            dev.pipeline_layout,
            0,
            &[dev.descriptor_set],
            &[],
        );
    }

    let mut bound = u32::MAX;
    let mut triangles_in = 0u64;
    for draw in input.draws {
        triangles_in += draw_triangles(draw);
        let blend = draw.pipeline.blend & 3;
        let cull = cull_index(draw.pipeline.cull);
        let key = (cull as u32) * 4 + blend;
        if key != bound {
            unsafe {
                dev.device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    dev.graphics_pipelines[cull][blend as usize],
                );
            }
            bound = key;
        }
        let surface = match draw.shader {
            ShaderRef::Surface(s) => Some(s),
            ShaderRef::DepthOnly => None,
        };
        let push = DrawPush {
            transform: draw.transform,
            flags: [
                surface.map(|s| s.textured as u32 as f32).unwrap_or(0.0),
                surface.map(|s| s.lit as u32 as f32).unwrap_or(0.0),
                surface.map(|s| s.receives_shadow as u32 as f32).unwrap_or(0.0),
                blend as f32,
            ],
            flags1: [draw.pipeline.two_sided as u32 as f32, 0.0, 0.0, 0.0],
        };
        record_draw(dev, draw, &push)?;
    }

    unsafe { dev.device.cmd_end_render_pass(cmd) };

    // ---- readback ---------------------------------------------------------
    if let Some(readback) = &dev.readback {
        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D {
                width: dev.width,
                height: dev.height,
                depth: 1,
            });
        unsafe {
            dev.device.cmd_copy_image_to_buffer(
                cmd,
                dev.color.as_ref().unwrap().image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                readback.buffer,
                &[region],
            );
        }
    }

    unsafe {
        dev.device
            .end_command_buffer(cmd)
            .map_err(|e| verr("end command buffer", e))?;
        let cmd_list = [cmd];
        let submits = [vk::SubmitInfo::default().command_buffers(&cmd_list)];
        dev.device
            .queue_submit(dev.queue, &submits, dev.fence)
            .map_err(|e| verr("queue submit", e))?;
        dev.device
            .wait_for_fences(&[dev.fence], true, u64::MAX)
            .map_err(|e| verr("wait for submit", e))?;
    }
    let raster_ns = raster_start.elapsed().as_nanos() as u64;

    // The checksum comes from the same bytes a present hands over, so the two
    // can never describe different frames.
    let mut checksum = 0u64;
    if input.checksum {
        if let Some(readback) = &dev.readback {
            let len = (dev.width as usize) * (dev.height as usize) * 4;
            let bytes = unsafe { std::slice::from_raw_parts(readback.ptr, len) };
            checksum = checksum_bytes(bytes);
        }
    }
    dev.color_checksum = checksum;

    dev.shadows.cascades_rendered = cascades_rendered;
    dev.shadows.shadow_pass_ns = shadow_ns;
    dev.shadows.fit_ns = fit_ns;

    let frame = FrameNumbers {
        frame_index: input.frame_index,
        total_ns: frame_start.elapsed().as_nanos() as u64,
        shadow_ns,
        raster_ns,
        bin_ns: 0,
        upload_ns: 0,
        spill_wait_ns: 0,
        // Binning, tiling and shading counters describe the tiled reference
        // rasteriser; on a GPU they stay honestly zero rather than invented.
        tiles_total: 0,
        tiles_rendered: 0,
        triangles_in: triangles_in.min(u32::MAX as u64) as u32,
        triangles_binned: (triangles_in + shadowed_triangles).min(u32::MAX as u64) as u32,
        triangles_culled: 0,
        pixels_shaded: 0,
        worker_threads: 0,
        resolution_scale: dev.config.resolution_scale,
        spill_io_bytes: 0,
        allocations_in_frame: 0,
    };
    dev.frame = frame;
    dev.counters.frames_since_tier_change += 1;
    Ok(frame)
}

fn shadow_ubo(
    fits: &shadow::CascadeSet,
    input: &FrameInput<'_>,
    filter: Option<ShadowFilter>,
    dev: &VulkanDevice,
) -> Result<ShadowUbo> {
    let (map_w, map_h) = dev
        .shadow_map
        .as_ref()
        .map(|(w, h, _, _)| (*w, *h))
        .unwrap_or((1, 1));
    let map_size = map_w.max(map_h);
    let cascades = fits.as_lookup_cascades();
    let bias = input.shadow.bias.unwrap_or_else(|| {
        shadow::bias_preset(
            dev.tier,
            map_size.max(1),
            filter.unwrap_or(ShadowFilter::Hard),
        )
    });
    let mut cb = ShadowUbo {
        // z is the penumbra in world units. The shader divides it by the
        // cascade's texel size to get a kernel radius, so shadow softness is a
        // property of the scene rather than of the map resolution.
        params_c: [map_w as f32, map_h as f32, SHADOW_PENUMBRA_WORLD, 0.0],
        // Row 2 of the camera view: the reference's camera-space depth row.
        view_z_row: [
            input.camera_view[2],
            input.camera_view[6],
            input.camera_view[10],
            input.camera_view[14],
        ],
        params_a: [
            filter.map(|f| f as u32 as f32).unwrap_or(0.0),
            fits.count as f32,
            bias.normal_bias,
            bias.depth_bias,
        ],
        params_b: [
            bias.slope_bias,
            input.shadow.max_distance,
            input.shadow.blend_band,
            map_size as f32,
        ],
        cascade_mvp: [[0.0; 16]; shadow::MAX_CASCADES],
        cascade_params: [[0.0; 4]; shadow::MAX_CASCADES],
    };
    for (index, cascade) in cascades.iter().enumerate() {
        if index >= shadow::MAX_CASCADES {
            break;
        }
        cb.cascade_mvp[index] = cascade.view_proj;
        cb.cascade_params[index] = [
            cascade.split_distance,
            cascade.texel_world,
            cascade.depth_span,
            cascade.map_index as f32,
        ];
    }
    Ok(cb)
}

/// The pipeline-set index for a `PipelineState::cull` value, mirroring the
/// d3d11 backend's `cull_index`.
fn cull_index(cull: u32) -> usize {
    match cull {
        1 => 1,
        2 => 2,
        _ => 0,
    }
}

/// The Vulkan cull flags for a `PipelineState::cull` value.
fn cull_flags(cull: u32) -> vk::CullModeFlags {
    match cull {
        1 => vk::CullModeFlags::BACK,
        2 => vk::CullModeFlags::FRONT,
        _ => vk::CullModeFlags::NONE,
    }
}

fn record_draw(dev: &VulkanDevice, draw: &DrawItem<'_>, push: &DrawPush) -> Result<()> {
    let cmd = dev.command_buffer;
    let vertex_offset = dev.upload_vertex.get();
    let index_offset = dev.upload_index.get();
    let vertex_bytes = (draw.vertices.len() * std::mem::size_of::<Vertex>()) as u64;
    let index_bytes = draw.indices.map_or(0, |i| i.len() as u64 * 4);
    if vertex_offset + vertex_bytes > dev.vertex.bytes {
        return Err(vfail(
            "vertex upload",
            format!(
                "draw needs {} bytes at offset {vertex_offset}, buffer holds {}",
                vertex_bytes, dev.vertex.bytes
            ),
        ));
    }
    if index_offset + index_bytes > dev.index.bytes {
        return Err(vfail(
            "index upload",
            format!(
                "draw needs {} bytes at offset {index_offset}, buffer holds {}",
                index_bytes, dev.index.bytes
            ),
        ));
    }
    unsafe {
        let bytes = std::slice::from_raw_parts(
            push as *const DrawPush as *const u8,
            std::mem::size_of::<DrawPush>(),
        );
        dev.device.cmd_push_constants(
            cmd,
            dev.pipeline_layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            bytes,
        );

        // The buffers are HOST_VISIBLE and coherent, so no staging copy is
        // involved: writing the mapped memory is visible to the device at submit.
        if vertex_bytes > 0 {
            ptr::copy_nonoverlapping(
                draw.vertices.as_ptr() as *const u8,
                dev.vertex.ptr.add(vertex_offset as usize),
                vertex_bytes as usize,
            );
        }
        dev.device
            .cmd_bind_vertex_buffers(cmd, 0, &[dev.vertex.buffer], &[vertex_offset]);

        match draw.indices {
            Some(indices) => {
                ptr::copy_nonoverlapping(
                    indices.as_ptr() as *const u8,
                    dev.index.ptr.add(index_offset as usize),
                    index_bytes as usize,
                );
                dev.device.cmd_bind_index_buffer(
                    cmd,
                    dev.index.buffer,
                    index_offset,
                    vk::IndexType::UINT32,
                );
                dev.device
                    .cmd_draw_indexed(cmd, indices.len() as u32, 1, 0, 0, 0);
            }
            None => {
                dev.device.cmd_draw(cmd, draw.vertices.len() as u32, 1, 0, 0);
            }
        }
    }
    dev.upload_vertex.set(vertex_offset + vertex_bytes);
    dev.upload_index.set(index_offset + index_bytes);
    Ok(())
}

// ---------------------------------------------------------------------------
// Readback
// ---------------------------------------------------------------------------

pub fn read_frame(dev: &mut VulkanDevice, out: &mut [u8], row_bytes: u32, _pad: u32) -> Result<()> {
    let readback = dev
        .readback
        .as_ref()
        .ok_or_else(|| vfail("read frame", "no frame has been rendered"))?;
    let src_row = (dev.width as usize) * 4;
    let dst_row = row_bytes as usize;
    let height = dev.height as usize;
    if dst_row < src_row || out.len() < dst_row.saturating_mul(height) {
        return Err(vfail(
            "read frame",
            format!(
                "output holds {} bytes at pitch {}; {}x{} needs {}",
                out.len(),
                row_bytes,
                dev.width,
                dev.height,
                dst_row.saturating_mul(height)
            ),
        ));
    }
    unsafe {
        let src = std::slice::from_raw_parts(readback.ptr, src_row * height);
        for y in 0..height {
            let at = y * dst_row;
            out[at..at + src_row].copy_from_slice(&src[y * src_row..(y + 1) * src_row]);
        }
    }
    Ok(())
}
