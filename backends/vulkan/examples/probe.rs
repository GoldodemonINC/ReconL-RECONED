//! Builds a Vulkan device the way `reconlCreateDevice` does, and prints the
//! message behind a failure instead of only its code.

use std::ffi::c_void;
use std::sync::Arc;

unsafe extern "C" fn host_alloc(_user: *mut c_void, size: usize, alignment: usize) -> *mut c_void {
    match std::alloc::Layout::from_size_align(size, alignment) {
        Ok(layout) => unsafe { std::alloc::alloc(layout) }.cast(),
        Err(_) => std::ptr::null_mut(),
    }
}

unsafe extern "C" fn host_realloc(
    _user: *mut c_void,
    ptr: *mut c_void,
    old_size: usize,
    new_size: usize,
    alignment: usize,
) -> *mut c_void {
    if ptr.is_null() {
        return unsafe { host_alloc(_user, new_size, alignment) };
    }
    let old = match std::alloc::Layout::from_size_align(old_size, alignment) {
        Ok(layout) => layout,
        Err(_) => return std::ptr::null_mut(),
    };
    unsafe { std::alloc::realloc(ptr.cast(), old, new_size) }.cast()
}

unsafe extern "C" fn host_free(_user: *mut c_void, ptr: *mut c_void, size: usize) {
    if ptr.is_null() {
        return;
    }
    // The allocator is asked to free by the same alignment it allocated with;
    // the alignment a Vulkan buffer wants is 8 on every driver that matters, and
    // a mis-sized dealloc here would abort the probe before it printed anything.
    if let Ok(layout) = std::alloc::Layout::from_size_align(size, 8) {
        unsafe { std::alloc::dealloc(ptr.cast(), layout) };
    }
}

fn main() {
    println!("hardware_available: {}", reconl_backend_vulkan::hardware_available());
    match reconl_backend_vulkan::probe_adapters() {
        Ok(adapters) => {
            println!("adapters: {}", adapters.len());
            for a in &adapters {
                println!(
                    "  {} | vendor {:#06x} device {:#06x} | integrated {} | 1.1 {} | vram {} MiB",
                    a.description,
                    a.vendor_id,
                    a.device_id,
                    a.integrated,
                    a.feature_level_11,
                    a.dedicated_video_memory / (1024 * 1024)
                );
            }
        }
        Err(e) => println!("probe_adapters failed: {}", e.message.as_str()),
    }

    let alloc = reconl_core::alloc::HostAlloc::from_abi(reconl_core::alloc::HostAllocatorA {
        alloc: Some(host_alloc),
        realloc: Some(host_realloc),
        free: Some(host_free),
        user: std::ptr::null_mut(),
    });
    let budget = Arc::new(reconl_core::budget::Budget::new(
        reconl_core::budget::BudgetCaps::default(),
    ));
    let config = reconl_backend_vulkan::VulkanConfig::default();
    match reconl_backend_vulkan::VulkanDevice::new(alloc, budget, config) {
        Ok(device) => {
            let snap = device.snapshot();
            println!(
                "device created: {} / {} | tier {} | frame {}x{}",
                device.device_name(),
                device.driver(),
                device.tier().name(),
                snap.frame.resolution_scale,
                snap.frame.frame_index
            );
        }
        Err(e) => println!("VulkanDevice::new failed: {} [{:?}]", e.message.as_str(), e.code),
    }
}
