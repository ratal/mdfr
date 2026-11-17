//! Dedicated Integration Suite for Zero-Allocation Tracking Tests.
//!
//! Isolates the custom `#[global_allocator]` (`TrackingAllocator`) from the main
//! parallel suites so `SYS_gettid` allocation checks do not affect other binaries.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};

pub mod alloc;
mod common;

struct TrackingAllocator;
pub static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
pub static TRACKED_TID: AtomicI64 = AtomicI64::new(0);
pub static ALLOC_TRACK_LOCK: Mutex<()> = Mutex::new(());

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let tracked = TRACKED_TID.load(Ordering::Relaxed);
        if tracked != 0 {
            let tid = unsafe { libc::syscall(libc::SYS_gettid) as i64 };
            if tid == tracked {
                ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            }
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;
