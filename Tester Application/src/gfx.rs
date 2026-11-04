//! The tester's own frame submission.
//!
//! `reconl_host::frame::Renderer` builds exactly one pipeline, which is right for
//! a benchmark - it times one configuration - and wrong here. A reflected scene
//! needs two: an opaque pipeline for the solid objects, and an alpha pipeline for
//! the mirrored copies and the floor drawn over them. A pipeline is created per
//! (shading, blend, cull, depth) combination and cannot be changed per draw, so
//! the only way to have both in one frame is to own the command list.
//!
//! The camera is owned here too. `reconl_host::scene::CameraSpec` describes one
//! fixed pose - an eye at a height and distance looking at the origin - because
//! the reference scene is fixed. A tester you can walk around needs an arbitrary
//! eye, so the view matrix is built from the camera's own basis.

use reconl::abi;
use reconl_host::device::Device;
use reconl_host::scene::{Chunk, Scene};
use reconl_host::{failed, hdr};
use std::ffi::c_void;
use std::time::Instant;

/// Which pipeline a draw is issued on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pass {
    Opaque,
    Alpha,
}

impl Pass {
    fn index(self) -> usize {
        match self {
            Pass::Opaque => 0,
            Pass::Alpha => 1,
        }
    }
}

/// One draw: a chunk, the model matrix it is drawn with, and the pipeline it
/// goes through. `tag` is the tester's own label for the object class, used to
/// decide what a reflection level includes; the renderer ignores it.
#[derive(Clone, Copy, Debug)]
pub struct Draw {
    pub chunk: usize,
    pub model: [f32; 16],
    pub pass: Pass,
    pub tag: u8,
}

/// A camera frame: the view and projection to draw it with.
pub struct View {
    pub view: [f32; 16],
    pub proj: [f32; 16],
    pub fov_y_deg: f32,
    pub near: f32,
    pub far: f32,
}

impl View {
    pub fn view_proj(&self) -> [f32; 16] {
        reconl_host::math::mul(&self.proj, &self.view)
    }
}

/// One command's slot in the command list, as the header publishes it.
const COMMAND_BYTES: u32 = 64;

pub struct Gfx {
    device: *mut reconl::DeviceHandle,
    swapchain: *mut reconl::SwapchainHandle,
    commands: *mut reconl::CommandListHandle,
    pipelines: [*mut reconl::PipelineHandle; 2],
    buffers: Vec<(*mut reconl::BufferHandle, *mut reconl::BufferHandle)>,
    pub width: u32,
    pub height: u32,
}

impl Gfx {
    pub fn new(
        device: &Device,
        chunks: &[Chunk],
        width: u32,
        height: u32,
        max_draws: u32,
    ) -> Result<Gfx, String> {
        let device_ptr = device.handle();
        let (width, height) = (width.max(1), height.max(1));
        // SAFETY: the device is live for this call and every descriptor is filled
        // from the Rust types, so the header and the struct cannot disagree.
        unsafe {
            let sd = abi::ReconLSwapchainDesc {
                base: hdr::<abi::ReconLSwapchainDesc>(),
                width,
                height,
                format: 1,
                image_count: 2,
                present_to_memory: 1,
                depth_format: 1,
                flags: 0,
                reserved: 0,
            };
            let mut swapchain: *mut reconl::SwapchainHandle = core::ptr::null_mut();
            let r = reconl::reconlCreateSwapchain(device_ptr, &sd, &mut swapchain);
            if r != abi::result::OK {
                return Err(failed("reconlCreateSwapchain", r));
            }

            let mut pipelines: [*mut reconl::PipelineHandle; 2] = [core::ptr::null_mut(); 2];
            for (index, blend) in [0u32, 1u32].iter().enumerate() {
                let pd = abi::ReconLPipelineDesc {
                    base: hdr::<abi::ReconLPipelineDesc>(),
                    shading: abi::shading::LAMBERT,
                    blend: *blend,
                    // No culling: a mirrored draw has its winding flipped, and
                    // culling it would leave the reflection empty.
                    cull: 0,
                    depth_compare: 1,
                    depth_write: 1,
                    texture_slots: 0,
                    texture_formats: [0; abi::RECONL_MAX_TEXTURE_SLOTS],
                    receives_shadow: 1,
                    casts_shadow: 1,
                    flags: 0,
                    reserved: 0,
                    debug_name: core::ptr::null(),
                };
                let mut pipeline: *mut reconl::PipelineHandle = core::ptr::null_mut();
                let r = reconl::reconlCreatePipeline(device_ptr, &pd, &mut pipeline);
                if r != abi::result::OK {
                    reconl::reconlRelease(swapchain as *mut c_void);
                    for p in pipelines {
                        if !p.is_null() {
                            reconl::reconlRelease(p as *mut c_void);
                        }
                    }
                    return Err(failed("reconlCreatePipeline", r));
                }
                pipelines[index] = pipeline;
            }

            let cd = abi::ReconLCommandListDesc {
                base: hdr::<abi::ReconLCommandListDesc>(),
                capacity_bytes: (max_draws.max(1) * 5 + 3) * COMMAND_BYTES,
                reserved: 0,
                debug_name: core::ptr::null(),
            };
            let mut commands: *mut reconl::CommandListHandle = core::ptr::null_mut();
            let r = reconl::reconlCreateCommandList(device_ptr, &cd, &mut commands);
            if r != abi::result::OK {
                reconl::reconlRelease(swapchain as *mut c_void);
                for p in pipelines {
                    reconl::reconlRelease(p as *mut c_void);
                }
                return Err(failed("reconlCreateCommandList", r));
            }

            let mut buffers = Vec::with_capacity(chunks.len());
            for chunk in chunks {
                let verts = chunk.verts.as_slice();
                let vbd = abi::ReconLBufferDesc {
                    base: hdr::<abi::ReconLBufferDesc>(),
                    size_bytes: core::mem::size_of_val(verts) as u64,
                    usage: abi::buffer_usage::VERTEX | abi::buffer_usage::STATIC,
                    reserved: 0,
                    data: verts.as_ptr() as *const c_void,
                    data_size: core::mem::size_of_val(verts) as u64,
                    debug_name: core::ptr::null(),
                };
                let mut vb: *mut reconl::BufferHandle = core::ptr::null_mut();
                let r = reconl::reconlCreateBuffer(device_ptr, &vbd, &mut vb);
                if r != abi::result::OK {
                    release_all(swapchain, pipelines, commands, &buffers);
                    return Err(failed("reconlCreateBuffer (vertices)", r));
                }
                let indices = chunk.indices.as_slice();
                let ibd = abi::ReconLBufferDesc {
                    base: hdr::<abi::ReconLBufferDesc>(),
                    size_bytes: core::mem::size_of_val(indices) as u64,
                    usage: abi::buffer_usage::INDEX | abi::buffer_usage::STATIC,
                    reserved: 0,
                    data: indices.as_ptr() as *const c_void,
                    data_size: core::mem::size_of_val(indices) as u64,
                    debug_name: core::ptr::null(),
                };
                let mut ib: *mut reconl::BufferHandle = core::ptr::null_mut();
                let r = reconl::reconlCreateBuffer(device_ptr, &ibd, &mut ib);
                if r != abi::result::OK {
                    reconl::reconlRelease(vb as *mut c_void);
                    release_all(swapchain, pipelines, commands, &buffers);
                    return Err(failed("reconlCreateBuffer (indices)", r));
                }
                buffers.push((vb, ib));
            }

            Ok(Gfx {
                device: device_ptr,
                swapchain,
                commands,
                pipelines,
                buffers,
                width,
                height,
            })
        }
    }

    pub fn frame_bytes(&self) -> usize {
        (self.width as usize) * (self.height as usize) * 4
    }

    /// Renders one frame and presents it into `pixels`, returning the host-side
    /// wall time in milliseconds at the three ABI boundaries.
    pub fn frame(
        &self,
        scene: &Scene,
        view: &View,
        rig: &[reconl_host::scene::LightSpec],
        draws: &[Draw],
        pixels: &mut [u8],
    ) -> Result<f64, String> {
        if pixels.len() < self.frame_bytes() {
            return Err(format!(
                "present buffer is {} bytes, a {}x{} frame needs {}",
                pixels.len(),
                self.width,
                self.height,
                self.frame_bytes()
            ));
        }
        // SAFETY: the device and every handle are live for the call; the
        // descriptors are filled from Rust types and outlive each ABI call.
        unsafe {
            // One light can only ever produce a scene lit by one thing; a second,
            // shadowless light is what keeps a shadow from being a black hole.
            let abi_lights: Vec<abi::ReconLLight> = rig.iter().map(|l| l.abi()).collect();
            let light_list = abi::ReconLLightList {
                base: hdr::<abi::ReconLLightList>(),
                count: abi_lights.len() as u32,
                reserved: 0,
                lights: abi_lights.as_ptr(),
            };
            let shadows = scene.shadows.abi();
            let mut camera: abi::ReconLCamera = core::mem::zeroed();
            camera.base = hdr::<abi::ReconLCamera>();
            camera.view = view.view;
            camera.fov_y_deg = view.fov_y_deg;
            camera.near = view.near;
            camera.far = view.far;
            let framegen = abi::ReconLFrameGenDesc {
                base: hdr::<abi::ReconLFrameGenDesc>(),
                enabled: 0,
                reserved: 0,
            };
            let fd = abi::ReconLFrameDesc {
                base: hdr::<abi::ReconLFrameDesc>(),
                width: self.width,
                height: self.height,
                seed: 1,
                reserved: 0,
                lights: &light_list,
                shadows: &shadows,
                camera: &camera,
                framegen: &framegen,
            };

            let started = Instant::now();
            let r = reconl::reconlBeginFrame(self.device, &fd as *const _ as *mut _);
            if r != abi::result::OK {
                return Err(failed("reconlBeginFrame", r));
            }

            let rp = abi::ReconLRenderPassDesc {
                base: hdr::<abi::ReconLRenderPassDesc>(),
                color_count: 0,
                reserved: 0,
                color: [core::mem::zeroed(); abi::RECONL_MAX_ATTACHMENTS],
                depth: core::ptr::null_mut(),
                viewport_width: self.width,
                viewport_height: self.height,
                load_color: 1,
                load_depth: 1,
                clear_color: [0.02, 0.03, 0.05, 1.0],
                clear_depth: 0.0,
                stencil_clear: 0,
                reserved2: 0,
            };
            reconl::reconlCmdReset(self.commands);
            check(
                reconl::reconlCmdBeginRenderPass(self.commands, &rp),
                "reconlCmdBeginRenderPass",
            )?;
            let view_proj = view.view_proj();
            check(
                reconl::reconlCmdPushConstants(
                    self.commands,
                    0,
                    view_proj.as_ptr() as *const c_void,
                    64,
                ),
                "reconlCmdPushConstants (view-proj)",
            )?;

            let mut bound: Option<usize> = None;
            for draw in draws {
                let (chunk, (vb, ib)) = match (
                    scene.chunks.get(draw.chunk),
                    self.buffers.get(draw.chunk),
                ) {
                    (Some(c), Some(b)) => (c, b),
                    _ => {
                        reconl::reconlCmdEndRenderPass(self.commands);
                        return Err(format!(
                            "a draw names chunk {} but the scene has {}",
                            draw.chunk,
                            scene.chunks.len()
                        ));
                    }
                };
                let want = draw.pass.index();
                if bound != Some(want) {
                    check(
                        reconl::reconlCmdSetPipeline(self.commands, self.pipelines[want]),
                        "reconlCmdSetPipeline",
                    )?;
                    bound = Some(want);
                }
                check(
                    reconl::reconlCmdSetVertexBuffer(self.commands, 0, *vb, 0),
                    "reconlCmdSetVertexBuffer",
                )?;
                check(
                    reconl::reconlCmdSetIndexBuffer(self.commands, *ib, 0, 1),
                    "reconlCmdSetIndexBuffer",
                )?;
                check(
                    reconl::reconlCmdPushConstants(
                        self.commands,
                        1,
                        draw.model.as_ptr() as *const c_void,
                        64,
                    ),
                    "reconlCmdPushConstants (model)",
                )?;
                check(
                    reconl::reconlCmdDrawIndexed(self.commands, chunk.indices.len() as u32, 0, 0),
                    "reconlCmdDrawIndexed",
                )?;
            }
            check(
                reconl::reconlCmdEndRenderPass(self.commands),
                "reconlCmdEndRenderPass",
            )?;
            let r = reconl::reconlSubmit(self.device, self.commands, core::ptr::null_mut());
            if r != abi::result::OK {
                return Err(failed("reconlSubmit", r));
            }

            let mut prd = abi::ReconLPresentDesc {
                base: hdr::<abi::ReconLPresentDesc>(),
                out_pixels: pixels.as_mut_ptr() as *mut c_void,
                out_pixels_size: pixels.len() as u64,
                out_row_pitch: self.width * 4,
                out_format: 1,
                flip: 0,
            };
            let r = reconl::reconlPresent(self.device, self.swapchain, &mut prd);
            if r != abi::result::OK {
                return Err(failed("reconlPresent", r));
            }
            Ok(started.elapsed().as_secs_f64() * 1000.0)
        }
    }
}

fn check(r: i32, what: &str) -> Result<(), String> {
    if r == abi::result::OK {
        Ok(())
    } else {
        Err(failed(what, r))
    }
}

/// # Safety
/// Every handle must be live and not released elsewhere.
unsafe fn release_all(
    swapchain: *mut reconl::SwapchainHandle,
    pipelines: [*mut reconl::PipelineHandle; 2],
    commands: *mut reconl::CommandListHandle,
    buffers: &[(*mut reconl::BufferHandle, *mut reconl::BufferHandle)],
) {
    for (vb, ib) in buffers {
        // SAFETY: live handles, as the caller's contract states.
        unsafe {
            reconl::reconlRelease(*vb as *mut c_void);
            reconl::reconlRelease(*ib as *mut c_void);
        }
    }
    // SAFETY: live handles, as the caller's contract states.
    unsafe {
        reconl::reconlRelease(commands as *mut c_void);
        for p in pipelines {
            reconl::reconlRelease(p as *mut c_void);
        }
        reconl::reconlRelease(swapchain as *mut c_void);
    }
}

impl Drop for Gfx {
    fn drop(&mut self) {
        // SAFETY: every handle was created by `Gfx::new` and is released once.
        unsafe { release_all(self.swapchain, self.pipelines, self.commands, &self.buffers) };
        self.swapchain = core::ptr::null_mut();
        self.commands = core::ptr::null_mut();
        self.pipelines = [core::ptr::null_mut(); 2];
        self.buffers.clear();
    }
}
