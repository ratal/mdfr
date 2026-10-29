//! Dedicated static-arena allocator for embedded C / MCU targets (Option A).
//!
//! Isolates all `mdfr-core` heap allocations into a fixed static byte buffer (either the
//! built-in 64 KiB `.bss` arena or a caller-provided static buffer registered via [`init_arena`]),
//! completely avoiding C `malloc`/`free` fragmentation and reentrancy issues.
use core::alloc::{GlobalAlloc, Layout};
use core::cell::UnsafeCell;
use core::ptr::{self, NonNull};
use core::sync::atomic::{AtomicBool, Ordering};

const HEADER_SIZE: usize = 16; // size (usize) + next_free (usize), padded to 16 bytes for 8/16-byte alignment
const MIN_BLOCK_SIZE: usize = 32;
const DEFAULT_ARENA_BYTES: usize = 64 * 1024;

#[repr(C, align(16))]
struct AlignedDefaultArena([u8; DEFAULT_ARENA_BYTES]);

static mut DEFAULT_ARENA: AlignedDefaultArena = AlignedDefaultArena([0u8; DEFAULT_ARENA_BYTES]);

#[repr(C, align(16))]
struct FreeBlock {
    /// Total block size including this 16-byte `FreeBlock` header. Bit 0 is `1` if allocated, `0` if free.
    size_and_flags: usize,
    next_free: *mut FreeBlock,
}

struct ArenaInner {
    initialized: bool,
    free_head: *mut FreeBlock,
    base: *mut u8,
    total_size: usize,
}

/// Coalescing free-list arena allocator over a static memory region.
pub struct StaticArena {
    lock: AtomicBool,
    inner: UnsafeCell<ArenaInner>,
}

unsafe impl Sync for StaticArena {}

impl Default for StaticArena {
    fn default() -> Self {
        Self::new()
    }
}

impl StaticArena {
    pub const fn new() -> Self {
        Self {
            lock: AtomicBool::new(false),
            inner: UnsafeCell::new(ArenaInner {
                initialized: false,
                free_head: ptr::null_mut(),
                base: ptr::null_mut(),
                total_size: 0,
            }),
        }
    }

    #[inline]
    fn with_lock<R>(&self, f: impl FnOnce(&mut ArenaInner) -> R) -> R {
        while self
            .lock
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            core::hint::spin_loop();
        }
        let res = f(unsafe { &mut *self.inner.get() });
        self.lock.store(false, Ordering::Release);
        res
    }

    /// Initializes the arena with a caller-supplied static buffer `[buf..buf+len)`.
    /// Must be called before any allocation if overriding the default 64 KiB arena.
    ///
    /// # Safety
    /// `buf` must point to a valid, mutable static memory region of at least `len` bytes.
    pub unsafe fn init_custom(&self, buf: *mut u8, len: usize) -> bool {
        self.with_lock(|inner| unsafe { Self::init_locked(inner, buf, len) })
    }

    unsafe fn init_locked(inner: &mut ArenaInner, buf: *mut u8, len: usize) -> bool {
        if buf.is_null() || len < MIN_BLOCK_SIZE * 2 {
            return false;
        }
        let addr = buf as usize;
        let aligned_addr = (addr + 15) & !15;
        let offset = aligned_addr - addr;
        if len <= offset + MIN_BLOCK_SIZE {
            return false;
        }
        let usable_len = (len - offset) & !15;
        let head = aligned_addr as *mut FreeBlock;
        unsafe {
            (*head).size_and_flags = usable_len;
            (*head).next_free = ptr::null_mut();
        }
        inner.base = aligned_addr as *mut u8;
        inner.total_size = usable_len;
        inner.free_head = head;
        inner.initialized = true;
        true
    }

    unsafe fn ensure_initialized(inner: &mut ArenaInner) {
        if !inner.initialized {
            let ptr = unsafe { ptr::addr_of_mut!(DEFAULT_ARENA.0) as *mut u8 };
            unsafe {
                Self::init_locked(inner, ptr, DEFAULT_ARENA_BYTES);
            }
        }
    }

    /// Allocates memory matching `layout` from the arena.
    pub fn alloc_layout(&self, layout: Layout) -> *mut u8 {
        if layout.size() == 0 {
            return NonNull::<u8>::dangling().as_ptr();
        }
        let align = layout.align().max(16);
        // For alignments > 16, over-allocate slightly or align payload
        if align > 16 {
            return ptr::null_mut();
        }
        let payload_size = (layout.size() + 15) & !15;
        let needed = (HEADER_SIZE + payload_size).max(MIN_BLOCK_SIZE);

        self.with_lock(|inner| unsafe {
            Self::ensure_initialized(inner);

            let mut prev: *mut FreeBlock = ptr::null_mut();
            let mut curr = inner.free_head;

            while !curr.is_null() {
                let blk_size = (*curr).size_and_flags & !1usize;
                if blk_size >= needed {
                    let rem = blk_size - needed;
                    if rem >= MIN_BLOCK_SIZE {
                        let split_block = (curr as *mut u8).add(needed) as *mut FreeBlock;
                        (*split_block).size_and_flags = rem;
                        (*split_block).next_free = (*curr).next_free;
                        if prev.is_null() {
                            inner.free_head = split_block;
                        } else {
                            (*prev).next_free = split_block;
                        }
                        (*curr).size_and_flags = needed | 1;
                    } else {
                        if prev.is_null() {
                            inner.free_head = (*curr).next_free;
                        } else {
                            (*prev).next_free = (*curr).next_free;
                        }
                        (*curr).size_and_flags = blk_size | 1;
                    }
                    (*curr).next_free = ptr::null_mut();
                    return (curr as *mut u8).add(HEADER_SIZE);
                }
                prev = curr;
                curr = (*curr).next_free;
            }
            ptr::null_mut()
        })
    }

    /// Returns a previously allocated block to the address-ordered free list and coalesces adjacent blocks.
    ///
    /// # Safety
    /// `ptr` must have been returned by [`StaticArena::alloc_layout`].
    pub unsafe fn dealloc_layout(&self, ptr: *mut u8, _layout: Layout) {
        if ptr.is_null() {
            return;
        }
        self.with_lock(|inner| unsafe {
            let blk = ptr.sub(HEADER_SIZE) as *mut FreeBlock;
            let blk_size = (*blk).size_and_flags & !1usize;
            (*blk).size_and_flags = blk_size;

            // Insert into address-sorted free list for O(1) neighbor coalescing
            let mut prev: *mut FreeBlock = ptr::null_mut();
            let mut curr = inner.free_head;
            while !curr.is_null() && (curr as usize) < (blk as usize) {
                prev = curr;
                curr = (*curr).next_free;
            }

            (*blk).next_free = curr;
            if prev.is_null() {
                inner.free_head = blk;
            } else {
                (*prev).next_free = blk;
            }

            // Coalesce with next block if adjacent
            if !curr.is_null() && (blk as *mut u8).add(blk_size) == (curr as *mut u8) {
                let next_size = (*curr).size_and_flags & !1usize;
                (*blk).size_and_flags = blk_size + next_size;
                (*blk).next_free = (*curr).next_free;
            }

            // Coalesce with previous block if adjacent
            if !prev.is_null() {
                let prev_size = (*prev).size_and_flags & !1usize;
                if (prev as *mut u8).add(prev_size) == (blk as *mut u8) {
                    let cur_size = (*blk).size_and_flags & !1usize;
                    (*prev).size_and_flags = prev_size + cur_size;
                    (*prev).next_free = (*blk).next_free;
                }
            }
        });
    }
}

unsafe impl GlobalAlloc for StaticArena {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.alloc_layout(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { self.dealloc_layout(ptr, layout) }
    }
}

/// Global static arena instance used when the `capi-arena` feature is enabled on `no_std`.
#[cfg_attr(
    all(feature = "capi-arena", not(feature = "std"), not(test)),
    global_allocator
)]
pub static GLOBAL_ARENA: StaticArena = StaticArena::new();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_alloc_dealloc_and_coalesce() {
        let mut backing = [0u64; 1024]; // 8 KiB
        let arena = StaticArena::new();
        unsafe {
            assert!(arena.init_custom(backing.as_mut_ptr() as *mut u8, 8192));
            let l1 = Layout::from_size_align(256, 8).unwrap();
            let l2 = Layout::from_size_align(512, 8).unwrap();
            let p1 = arena.alloc_layout(l1);
            let p2 = arena.alloc_layout(l2);
            assert!(!p1.is_null());
            assert!(!p2.is_null());
            ptr::write_bytes(p1, 0xAA, 256);
            ptr::write_bytes(p2, 0x55, 512);
            arena.dealloc_layout(p1, l1);
            arena.dealloc_layout(p2, l2);

            // After coalescing, we can allocate almost the entire 8 KiB buffer
            let l_big = Layout::from_size_align(7000, 8).unwrap();
            let p_big = arena.alloc_layout(l_big);
            assert!(!p_big.is_null());
            arena.dealloc_layout(p_big, l_big);
        }
    }
}
