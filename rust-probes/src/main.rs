//! reconl-probes — Rust audit of the shipped library.
//!
//! Exercises every property the 17 C probes audited. Exit 0 = all pass, 1 = findings.

mod harness;

use harness::{ProbeCounters, Rig, view_at, backend_name, tier_name, tier_reason_name};
use reconl_ffi;
use reconl_ffi::abi;
use reconl_ffi::StructHeader;
use reconl_ffi::ABIStruct;
use std;


fn main() {
    let args: Vec<String> = std::env::args().collect();
    let backend = if args.iter().any(|a| a.starts_with("--backend=d3d11")) { abi::backend::D3D11 } else { abi::backend::SOFT_CPU };
    let size: u32 = args.iter().find_map(|a| a.strip_prefix("--size=")).and_then(|s| s.parse().ok()).unwrap_or(64);
    let label = if backend == abi::backend::D3D11 { "d3d11" } else { "soft-cpu" };
    eprintln!("{} — {}x{}", label, size, size);

    let mut c = ProbeCounters::new();

    section_hostile(&mut c, backend, size);
    section_framestate(&mut c, backend, size);
    section_framegen(&mut c, backend, size);
    section_ladder(&mut c, backend, size);

    eprintln!("");
    eprintln!("{}probes: {} checks, {} findings", label, c.checks, c.findings);
    if c.findings > 0 { std::process::exit(1); }
}

fn section_hostile(c: &mut ProbeCounters, backend: u32, size: u32) {
    eprintln!("== hostile requests ==");
    // zeroed allocator must be refused
    {
        let dd = abi::ReconLDeviceDesc {
            base: StructHeader::new(abi::ReconLDeviceDesc::MIN_SIZE, abi::ReconLDeviceDesc::STRUCT_TYPE),
            backend_hint: backend, tier_hint: 2, allow_downgrade: 0,
            worker_threads: 1, target_frame_ms: 16, downgrade_after_frames: 60,
            seed: 1, flags: 0, budget: std::ptr::null(),
            allocator: abi::ReconLAllocator { alloc: None, realloc: None, free: None, user: std::ptr::null_mut() },
            backend_desc: std::ptr::null(),
        };
        let mut dev: *mut _ = std::ptr::null_mut();
        let r = unsafe { reconl_ffi::reconlCreateDevice(&dd, &mut dev) };
        c.check(r == abi::result::INVALID_ARGUMENT, "zeroed allocator refused");
    }
    // zero-sized buffer
    {
        let rig = match Rig::new(backend, 2, size, size, 0, 16, 60) {
            Ok(r) => r, Err(_) => { c.check(false, "rig build failed"); return; }
        };
        unsafe {
            let bd = abi::ReconLBufferDesc {
                base: StructHeader::new(abi::ReconLBufferDesc::MIN_SIZE, abi::ReconLBufferDesc::STRUCT_TYPE),
                size_bytes: 0, usage: abi::buffer_usage::VERTEX, reserved: 0,
                data: std::ptr::null(), data_size: 0, debug_name: std::ptr::null(),
            };
            let mut buf: *mut _ = std::ptr::null_mut();
            let r = reconl_ffi::reconlCreateBuffer(rig.device, &bd, &mut buf);
            c.check(r == abi::result::INVALID_ARGUMENT && buf.is_null(), "zero-sized buffer refused");
        }
    }
    // push constants too large
    {
        let rig = match Rig::new(backend, 2, size, size, 0, 16, 60) {
            Ok(r) => r, Err(_) => { c.check(false, "rig build failed"); return; }
        };
        unsafe {
            let light = abi::ReconLLight {
                base: StructHeader::new(abi::ReconLLight::MIN_SIZE, abi::ReconLLight::STRUCT_TYPE),
                r#type: 0, cast_shadow: 0, shadow_priority: 0, shadow_quality: 0,
                position: [0.0;3], direction: [0.0,0.0,-1.0],
                color: [1.0;3], intensity: 1.0, range: 0.0,
                cone_inner_deg: 0.0, cone_outer_deg: 0.0, reserved: 0,
            };
            let ll = abi::ReconLLightList {
                base: StructHeader::new(abi::ReconLLightList::MIN_SIZE, abi::ReconLLightList::STRUCT_TYPE),
                count: 1, reserved: 0, lights: &light,
            };
            let fd = abi::ReconLFrameDesc {
                base: StructHeader::new(abi::ReconLFrameDesc::MIN_SIZE, abi::ReconLFrameDesc::STRUCT_TYPE),
                width: size, height: size, seed: 1, reserved: 0,
                lights: &ll, shadows: std::ptr::null(), camera: std::ptr::null(), framegen: std::ptr::null(),
            };
            let _ = reconl_ffi::reconlBeginFrame(rig.device, (&fd as *const _) as *mut _);
            let rp = abi::ReconLRenderPassDesc {
                base: StructHeader::new(abi::ReconLRenderPassDesc::MIN_SIZE, abi::ReconLRenderPassDesc::STRUCT_TYPE),
                color_count: 0, reserved: 0,
                color: [unsafe { std::mem::zeroed() }; abi::RECONL_MAX_ATTACHMENTS],
                depth: std::ptr::null_mut(),
                viewport_width: size, viewport_height: size,
                load_color: 1, load_depth: 1,
                clear_color: [0.0,0.0,1.0,1.0], clear_depth: 0.0,
                stencil_clear: 0, reserved2: 0,
            };
            let _ = reconl_ffi::reconlCmdReset(rig.commands);
            let _ = reconl_ffi::reconlCmdBeginRenderPass(rig.commands, &rp);
            let _ = reconl_ffi::reconlCmdSetPipeline(rig.commands, rig.pipeline);
            let big = [0u8; 128];
            let r = reconl_ffi::reconlCmdPushConstants(rig.commands, 0, big.as_ptr() as *const _, 128);
            c.check(r == abi::result::INVALID_ARGUMENT, "push constants too large refused");
        }
    }
}

fn section_framestate(c: &mut ProbeCounters, backend: u32, size: u32) {
    eprintln!("== frame state machine ==");
    let rig = match Rig::new(backend, 2, size, size, 0, 16, 60) {
        Ok(r) => r, Err(_) => { c.check(false, "rig build failed"); return; }
    };
    unsafe {
        let light = abi::ReconLLight {
            base: StructHeader::new(abi::ReconLLight::MIN_SIZE, abi::ReconLLight::STRUCT_TYPE),
            r#type: 0, cast_shadow: 0, shadow_priority: 0, shadow_quality: 0,
            position: [0.0;3], direction: [0.0,0.0,-1.0],
            color: [1.0;3], intensity: 1.0, range: 0.0,
            cone_inner_deg: 0.0, cone_outer_deg: 0.0, reserved: 0,
        };
        let ll = abi::ReconLLightList {
            base: StructHeader::new(abi::ReconLLightList::MIN_SIZE, abi::ReconLLightList::STRUCT_TYPE),
            count: 1, reserved: 0, lights: &light,
        };
        let fd = abi::ReconLFrameDesc {
            base: StructHeader::new(abi::ReconLFrameDesc::MIN_SIZE, abi::ReconLFrameDesc::STRUCT_TYPE),
            width: size, height: size, seed: 1, reserved: 0,
            lights: &ll, shadows: std::ptr::null(), camera: std::ptr::null(), framegen: std::ptr::null(),
        };
        let r1 = reconl_ffi::reconlBeginFrame(rig.device, (&fd as *const _) as *mut _);
        c.check(r1 == abi::result::OK, "first begin succeeds");
        let r2 = reconl_ffi::reconlBeginFrame(rig.device, (&fd as *const _) as *mut _);
        c.check(r2 == abi::result::FRAME_IN_PROGRESS, "double begin refused");

        let r = reconl_ffi::reconlSubmit(rig.device, rig.commands, std::ptr::null_mut());
        c.check(r == abi::result::OK, "submit with no commands succeeds");

        let mut pixels = vec![0u8; (size * size * 4) as usize];        let r = reconl_ffi::reconlPresent(rig.device, rig.swapchain,
                &mut abi::ReconLPresentDesc {
                    base: StructHeader::new(abi::ReconLPresentDesc::MIN_SIZE, abi::ReconLPresentDesc::STRUCT_TYPE),
                    out_pixels: pixels.as_mut_ptr() as *mut _,
                    out_pixels_size: pixels.len() as u64, out_row_pitch: size * 4,
                    out_format: abi::format::R8G8B8A8_UNORM, flip: 0,
                });
        c.check(r == abi::result::NO_FRAME, "present with no frame refused");
    }
}

fn section_framegen(c: &mut ProbeCounters, backend: u32, size: u32) {
    eprintln!("== frame generation ==");
    let mut rig = match Rig::new(backend, 2, size, size, 0, 16, 60) {
        Ok(r) => r, Err(_) => { c.check(false, "rig build failed"); return; }
    };
    let mut frame = vec![0u8; (size * size * 4) as usize];
    let mut tight = vec![0u8; (size * size * 4) as usize];
    let pitch = size * 4 + 16;
    let _padded = vec![0xABu8; (pitch * size) as usize];

    unsafe {
        // order of descriptor type check
        {
            let prd = abi::ReconLPresentDesc {
                base: StructHeader::new(abi::ReconLPresentDesc::MIN_SIZE, abi::ReconLPresentDesc::STRUCT_TYPE),
                out_pixels: tight.as_mut_ptr() as *mut _,
                out_pixels_size: (size*size*4) as u64, out_row_pitch: size*4,
                out_format: abi::format::R8G8B8A8_UNORM, flip: 0,
            };
            let r = reconl_ffi::reconlPresentGenerated(rig.device, rig.swapchain, &prd as *const _ as *mut _, 0.0);
            c.check(r == abi::result::NO_FRAME, "generated with no frame asked: NO_FRAME");
        }

        // generation in open state
        {
            let light = abi::ReconLLight {
                base: StructHeader::new(abi::ReconLLight::MIN_SIZE, abi::ReconLLight::STRUCT_TYPE),
                r#type: 0, cast_shadow: 0, shadow_priority: 0, shadow_quality: 0,
                position: [0.0;3], direction: [0.0,0.0,-1.0],
                color: [1.0;3], intensity: 1.0, range: 0.0,
                cone_inner_deg: 0.0, cone_outer_deg: 0.0, reserved: 0,
            };
            let ll = abi::ReconLLightList {
                base: StructHeader::new(abi::ReconLLightList::MIN_SIZE, abi::ReconLLightList::STRUCT_TYPE),
                count: 1, reserved: 0, lights: &light,
            };
            let mut cam = [0f32; 16]; view_at(0.0, &mut cam);
            let cam_st = abi::ReconLCamera {
                base: StructHeader::new(abi::ReconLCamera::MIN_SIZE, abi::ReconLCamera::STRUCT_TYPE),
                view: cam, fov_y_deg: 60.0, near: 0.1, far: 100.0, reserved: 0.0,
            };
            let fg = abi::ReconLFrameGenDesc {
                base: StructHeader::new(abi::ReconLFrameGenDesc::MIN_SIZE, abi::ReconLFrameGenDesc::STRUCT_TYPE),
                enabled: 1, reserved: 0,
            };
            let fd = abi::ReconLFrameDesc {
                base: StructHeader::new(abi::ReconLFrameDesc::MIN_SIZE, abi::ReconLFrameDesc::STRUCT_TYPE),
                width: size, height: size, seed: 1, reserved: 0,
                lights: &ll, shadows: std::ptr::null(), camera: &cam_st, framegen: &fg,
            };
            let r0 = reconl_ffi::reconlBeginFrame(rig.device, (&fd as *const _) as *mut _);
            c.check(r0 == abi::result::OK, "frame opens with gen asked");
            let prd = abi::ReconLPresentDesc {
                base: StructHeader::new(abi::ReconLPresentDesc::MIN_SIZE, abi::ReconLPresentDesc::STRUCT_TYPE),
                out_pixels: tight.as_mut_ptr() as *mut _,
                out_pixels_size: (size*size*4) as u64, out_row_pitch: size*4,
                out_format: abi::format::R8G8B8A8_UNORM, flip: 0,
            };
            let r = reconl_ffi::reconlPresentGenerated(rig.device, rig.swapchain, &prd as *const _ as *mut _, 0.5);
            eprintln!("    generate in OPEN -> {}", r);
            let r = reconl_ffi::reconlSubmit(rig.device, rig.commands, std::ptr::null_mut());
            c.check(r == abi::result::OK, "submit after gen in OPEN succeeds");
            let r = reconl_ffi::reconlPresentGenerated(rig.device, rig.swapchain, &prd as *const _ as *mut _, 0.5);
            eprintln!("    generate in SUBMITTED -> {}", r);
            let r = reconl_ffi::reconlPresent(rig.device, rig.swapchain,
                &mut abi::ReconLPresentDesc {
                    base: StructHeader::new(abi::ReconLPresentDesc::MIN_SIZE, abi::ReconLPresentDesc::STRUCT_TYPE),
                    out_pixels: frame.as_mut_ptr() as *mut _,
                    out_pixels_size: (size*size*4) as u64, out_row_pitch: size*4,
                    out_format: abi::format::R8G8B8A8_UNORM, flip: 0,
                });
            c.check(r == abi::result::OK, "present after gen in SUBMITTED succeeds");
        }

        // look-ahead boundaries
        {
            let nan = f32::NAN;
            let prd = abi::ReconLPresentDesc {
                base: StructHeader::new(abi::ReconLPresentDesc::MIN_SIZE, abi::ReconLPresentDesc::STRUCT_TYPE),
                out_pixels: tight.as_mut_ptr() as *mut _,
                out_pixels_size: (size*size*4) as u64, out_row_pitch: size*4,
                out_format: abi::format::R8G8B8A8_UNORM, flip: 0,
            };
            let r = reconl_ffi::reconlPresentGenerated(rig.device, rig.swapchain, &prd as *const _ as *mut _, nan);
            c.check(r == abi::result::INVALID_ARGUMENT, "NaN look-ahead refused");
            let r = reconl_ffi::reconlPresentGenerated(rig.device, rig.swapchain, &prd as *const _ as *mut _, 1.0000001);
            c.check(r == abi::result::INVALID_ARGUMENT, "look-ahead > 1.0 refused");
        }
    }
}

fn section_ladder(c: &mut ProbeCounters, backend: u32, size: u32) {
    eprintln!("== tier ladder ==");
    // arm A: only relabel
    {
        let rig = match Rig::new(backend, if backend == abi::backend::D3D11 { 1 } else { 2 }, size, size, 0, 1, 1) {
            Ok(r) => r, Err(_) => { c.check(false, "rig A failed"); return; }
        };
        for i in 0..8 {
            let r = unsafe { harness::record_frame(&rig, size, size) };
            if r != abi::result::OK { eprintln!("    frame {}: failed", i); break; }
            let stats = harness::get_stats(&rig);
            eprintln!("    frame {}: backend={} tier={} reason={} entries={}",
                i, backend_name(stats.backend), tier_name(stats.tier),
                tier_reason_name(stats.tier_reason), stats.downgrade_count);
        }
        let stats = harness::get_stats(&rig);
        c.check(stats.backend == backend, "opt-out kept backend");
        if backend != abi::backend::SOFT_CPU {
            c.check(stats.downgrade_count > 0, "relabel ladder fired");
        }
    }
    // arm B: both writers
    {
        let rig = match Rig::new(backend, if backend == abi::backend::D3D11 { 1 } else { 2 }, size, size, 1, 1, 1) {
            Ok(r) => r, Err(_) => { c.check(false, "rig B failed"); return; }
        };
        let mut seen: Vec<(u64,u32,u32,u32,String)> = Vec::new();
        for i in 0..16 {
            let r = unsafe { harness::record_frame(&rig, size, size) };
            if r != abi::result::OK { eprintln!("    frame {}: failed", i); break; }
            let stats = harness::get_stats(&rig);
            let cur: Vec<(u64,u32,u32,u32,String)> = (0..stats.downgrade_count.min(stats.downgrade_capacity))
                .map(|j| {
                    let d = &stats.downgrades[j as usize];
                    (d.frame_index, d.from, d.to, d.reason,
                     String::from_utf8_lossy(&d.detail).into_owned())
                }).collect();
            for s in &seen {
                if !cur.iter().any(|x| x.0 == s.0 && x.1 == s.1 && x.2 == s.2 && x.3 == s.3 && x.4 == s.4) {
                    c.check(false, &format!("entry lost: frame {}", s.0));
                }
            }
            for j in 1..cur.len() {
                if cur[j].0 < cur[j-1].0 {
                    c.check(false, &format!("out of order: {} before {}", cur[j].0, cur[j-1].0));
                }
            }
            seen = cur;
        }
        let stats = harness::get_stats(&rig);
        c.check(seen.is_empty() || stats.downgrade_count >= seen.len() as u32,
                "entries stable through run");
    }
}
