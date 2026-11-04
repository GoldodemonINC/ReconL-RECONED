//! Shared probe harness.

use reconl_ffi;
use reconl_ffi::abi;
use reconl_ffi::StructHeader;
use reconl_ffi::ABIStruct;
use std;

use reconl_core::alloc::HostAlloc;


pub struct ProbeCounters {
    pub checks: usize,
    pub findings: usize,
}

impl ProbeCounters {
    pub fn new() -> Self { Self { checks: 0, findings: 0 } }
    pub fn check(&mut self, ok: bool, what: &str) {
        self.checks += 1;
        if ok {
            eprintln!("  ok   {}", what);
        } else {
            eprintln!("  FIND {}", what);
            self.findings += 1;
        }
    }
}

impl Drop for ProbeCounters {
    fn drop(&mut self) {
        eprintln!("");
        eprintln!("probes: {} checks, {} findings", self.checks, self.findings);
    }
}

pub struct HostAllocator;

impl HostAllocator {
    pub fn new() -> Self { Self }
    pub fn allocator(&self) -> abi::ReconLAllocator {
        // Use the library's own system allocator, which correctly handles the
        // ABI contract: free is told the size but not the alignment.
        let a = HostAlloc::system();
        let abi_a = a.abi();
        abi::ReconLAllocator {
            alloc: abi_a.alloc,
            realloc: abi_a.realloc,
            free: abi_a.free,
            user: abi_a.user,
        }
    }
}

pub struct Rig {
    pub device: *mut reconl_ffi::DeviceHandle,
    pub swapchain: *mut reconl_ffi::SwapchainHandle,
    pub pipeline: *mut reconl_ffi::PipelineHandle,
    pub commands: *mut reconl_ffi::CommandListHandle,
    pub vb: *mut reconl_ffi::BufferHandle,
    pub ib: *mut reconl_ffi::BufferHandle,
}

impl Rig {
    pub fn new(backend_hint: u32, tier_hint: u32, width: u32, height: u32,
               allow_downgrade: u32, target_frame_ms: u32, downgrade_after_frames: u32)
        -> Result<Self, String> {
        let alloc = HostAllocator::new();
        let dd = abi::ReconLDeviceDesc {
            base: StructHeader::new(abi::ReconLDeviceDesc::MIN_SIZE, abi::ReconLDeviceDesc::STRUCT_TYPE),
            backend_hint, tier_hint, allow_downgrade,
            worker_threads: 1, target_frame_ms, downgrade_after_frames,
            seed: 7, flags: 0, budget: std::ptr::null(),
            allocator: alloc.allocator(), backend_desc: std::ptr::null(),
        };
        let mut device: *mut _ = std::ptr::null_mut();
        if unsafe { reconl_ffi::reconlCreateDevice(&dd, &mut device) } != abi::result::OK {
            return Err("device".into());
        }
        if device.is_null() {
            return Err("device returned null".into());
        }
        let sd = abi::ReconLSwapchainDesc {
            base: StructHeader::new(abi::ReconLSwapchainDesc::MIN_SIZE, abi::ReconLSwapchainDesc::STRUCT_TYPE),
            width, height, format: abi::format::R8G8B8A8_UNORM, image_count: 2,
            present_to_memory: 1, depth_format: abi::format::D32_FLOAT,
            flags: 0, reserved: 0,
        };
        let mut swapchain: *mut _ = std::ptr::null_mut();
        if unsafe { reconl_ffi::reconlCreateSwapchain(device, &sd, &mut swapchain) } != abi::result::OK {
            return Err("swapchain".into());
        }
        let pd = abi::ReconLPipelineDesc {
            base: StructHeader::new(abi::ReconLPipelineDesc::MIN_SIZE, abi::ReconLPipelineDesc::STRUCT_TYPE),
            shading: abi::shading::LAMBERT, blend: 0, cull: 1,
            depth_compare: 1, depth_write: 1,
            texture_slots: 0, texture_formats: [0; abi::RECONL_MAX_TEXTURE_SLOTS],
            receives_shadow: 0, casts_shadow: 0,
            flags: 0, reserved: 0, debug_name: std::ptr::null(),
        };
        let mut pipeline: *mut _ = std::ptr::null_mut();
        if unsafe { reconl_ffi::reconlCreatePipeline(device, &pd, &mut pipeline) } != abi::result::OK {
            return Err("pipeline".into());
        }
        let cd = abi::ReconLCommandListDesc {
            base: StructHeader::new(abi::ReconLCommandListDesc::MIN_SIZE, abi::ReconLCommandListDesc::STRUCT_TYPE),
            capacity_bytes: 8192, reserved: 0, debug_name: std::ptr::null(),
        };
        let mut commands: *mut _ = std::ptr::null_mut();
        if unsafe { reconl_ffi::reconlCreateCommandList(device, &cd, &mut commands) } != abi::result::OK {
            return Err("cmd list".into());
        }
        // Quad vertex buffer
        let verts: [abi::ReconLVertex; 6] = {
            let mut v = [abi::ReconLVertex::default(); 6];
            let c: [[f32;2];6] = [[-1.0,1.0],[1.0,1.0],[1.0,-1.0],[-1.0,1.0],[1.0,-1.0],[-1.0,-1.0]];
            for (i, corner) in c.iter().enumerate() {
                v[i].position = [corner[0], corner[1], 0.5];
                v[i].normal = [0.0; 3];
                v[i].color = [1.0; 4];
            }
            v
        };
        let vbd = abi::ReconLBufferDesc {
            base: StructHeader::new(abi::ReconLBufferDesc::MIN_SIZE, abi::ReconLBufferDesc::STRUCT_TYPE),
            size_bytes: (std::mem::size_of_val(&verts) as u64), usage: abi::buffer_usage::VERTEX | abi::buffer_usage::STATIC,
            reserved: 0, data: verts.as_ptr() as *const _, data_size: std::mem::size_of_val(&verts) as u64,
            debug_name: std::ptr::null(),
        };
        let mut vb: *mut _ = std::ptr::null_mut();
        if unsafe { reconl_ffi::reconlCreateBuffer(device, &vbd, &mut vb) } != abi::result::OK {
            return Err("vb".into());
        }
        let indices: [u32; 6] = [0,1,2,3,4,5];
        let ibd = abi::ReconLBufferDesc {
            base: StructHeader::new(abi::ReconLBufferDesc::MIN_SIZE, abi::ReconLBufferDesc::STRUCT_TYPE),
            size_bytes: (indices.len() * 4) as u64, usage: abi::buffer_usage::INDEX,
            reserved: 0, data: indices.as_ptr() as *const _, data_size: (indices.len() * 4) as u64,
            debug_name: std::ptr::null(),
        };
        let mut ib: *mut _ = std::ptr::null_mut();
        if unsafe { reconl_ffi::reconlCreateBuffer(device, &ibd, &mut ib) } != abi::result::OK {
            return Err("ib".into());
        }
        Ok(Rig { device, swapchain, pipeline, commands, vb, ib })
    }
}

impl Drop for Rig {
    fn drop(&mut self) {
        unsafe {
            if !self.ib.is_null() { reconl_ffi::reconlRelease(self.ib as *mut _); }
            if !self.vb.is_null() { reconl_ffi::reconlRelease(self.vb as *mut _); }
            if !self.commands.is_null() { reconl_ffi::reconlRelease(self.commands as *mut _); }
            if !self.pipeline.is_null() { reconl_ffi::reconlRelease(self.pipeline as *mut _); }
            if !self.swapchain.is_null() { reconl_ffi::reconlRelease(self.swapchain as *mut _); }
            if !self.device.is_null() { reconl_ffi::reconlRelease(self.device as *mut _); }
        }
    }
}

pub unsafe fn record_frame(rig: &Rig, width: u32, height: u32) -> i32 {
    let light = abi::ReconLLight {
        base: StructHeader::new(abi::ReconLLight::MIN_SIZE, abi::ReconLLight::STRUCT_TYPE),
        r#type: 0, cast_shadow: 0, shadow_priority: 0, shadow_quality: 0,
        position: [0.0; 3], direction: [0.0, 0.0, -1.0],
        color: [1.0; 3], intensity: 1.0, range: 0.0,
        cone_inner_deg: 0.0, cone_outer_deg: 0.0, reserved: 0,
    };
    let ll = abi::ReconLLightList {
        base: StructHeader::new(abi::ReconLLightList::MIN_SIZE, abi::ReconLLightList::STRUCT_TYPE),
        count: 1, reserved: 0, lights: &light,
    };
    let fd = abi::ReconLFrameDesc {
        base: StructHeader::new(abi::ReconLFrameDesc::MIN_SIZE, abi::ReconLFrameDesc::STRUCT_TYPE),
        width, height, seed: 1, reserved: 0,
        lights: &ll, shadows: std::ptr::null(), camera: std::ptr::null(), framegen: std::ptr::null(),
    };
    reconl_ffi::reconlBeginFrame(rig.device, (&fd as *const _) as *mut _);
    let rp = abi::ReconLRenderPassDesc {
        base: StructHeader::new(abi::ReconLRenderPassDesc::MIN_SIZE, abi::ReconLRenderPassDesc::STRUCT_TYPE),
        color_count: 0, reserved: 0,
        color: [unsafe { std::mem::zeroed() }; abi::RECONL_MAX_ATTACHMENTS],
        depth: std::ptr::null_mut(),
        viewport_width: width, viewport_height: height,
        load_color: 1, load_depth: 1,
        clear_color: [0.0, 0.0, 1.0, 1.0], clear_depth: 0.0,
        stencil_clear: 0, reserved2: 0,
    };
    let _ = reconl_ffi::reconlCmdReset(rig.commands);
    let _ = reconl_ffi::reconlCmdBeginRenderPass(rig.commands, &rp);
    let _ = reconl_ffi::reconlCmdEndRenderPass(rig.commands);
    let _ = reconl_ffi::reconlSubmit(rig.device, rig.commands, std::ptr::null_mut());
    let mut pixels = vec![0u8; (width * height * 4) as usize];
    let r = reconl_ffi::reconlPresent(rig.device, rig.swapchain,
        &mut abi::ReconLPresentDesc {
            base: StructHeader::new(abi::ReconLPresentDesc::MIN_SIZE, abi::ReconLPresentDesc::STRUCT_TYPE),
            out_pixels: pixels.as_mut_ptr() as *mut _,
            out_pixels_size: pixels.len() as u64, out_row_pitch: width * 4,
            out_format: abi::format::R8G8B8A8_UNORM, flip: 0,
        });
    r
}

pub fn view_at(x: f32, out: &mut [f32; 16]) {
    out[0] = 1.0; out[5] = 1.0; out[10] = 1.0; out[15] = 1.0;
    out[12] = -x;
    for i in 0..16 {
        if i != 0 && i != 5 && i != 10 && i != 12 && i != 15 { out[i] = 0.0; }
    }
}



pub fn backend_name(id: u32) -> String {
    unsafe { std::ffi::CStr::from_ptr(reconl_ffi::reconlBackendName(id)) }
        .to_string_lossy().into_owned()
}

pub fn tier_name(id: u32) -> String {
    unsafe { std::ffi::CStr::from_ptr(reconl_ffi::reconlTierName(id)) }
        .to_string_lossy().into_owned()
}

pub fn get_stats(rig: &Rig) -> abi::ReconLStats {
    unsafe {
        let mut stats: abi::ReconLStats = unsafe { std::mem::zeroed() };
    stats.base = StructHeader::new(abi::ReconLStats::MIN_SIZE, abi::ReconLStats::STRUCT_TYPE);
        let desc = abi::ReconLStatsDesc {
            base: StructHeader::new(abi::ReconLStatsDesc::MIN_SIZE, abi::ReconLStatsDesc::STRUCT_TYPE),
            include_downgrades: 1, reserved: 0,
        };
        let _ = reconl_ffi::reconlGetStats(rig.device, &mut stats);
        stats
    }
}

pub fn tier_reason_name(id: u32) -> &'static str {
    match id {
        0 => "startup-probe", 1 => "host-request", 2 => "over-budget",
        3 => "over-target", 4 => "device-removed", 5 => "device-lost",
        6 => "memory-pressure", 7 => "disk-cache-full", 8 => "build",
        9 => "no-gpu-api", 10 => "recovery", _ => "?",
    }
}
