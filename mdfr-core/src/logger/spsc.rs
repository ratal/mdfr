//! Single-Producer Single-Consumer (SPSC) ring buffer for ISR / real-time task decoupling (`no_std + alloc`).
//!
//! Allows an interrupt service routine (ISR) or high-priority acquisition task to push raw records
//! (`(GroupHandle, ticks, &[u8])`) without blocking on flash/SD I/O or DZ compression, while a
//! background task drains the queue into [`Logger`](super::Logger).
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicUsize, Ordering};

use crate::error::Result;
use crate::io::Sink;
use crate::time::TimeSource;

use super::engine::Logger;
use super::schema::GroupHandle;

/// Lock-free single-producer single-consumer byte ring buffer for framed logger records.
///
/// Each frame in the ring is encoded as:
/// `[group: u16 LE][len: u16 LE][ticks: u64 LE][payload: len bytes]` (`12 + len` bytes).
/// If a frame does not fit in the contiguous tail of the ring buffer, a sentinel `group = 0xFFFF`
/// marker wraps the write cursor to index `0`.
pub struct SpscRing {
    buf: Box<[UnsafeCell<u8>]>,
    capacity: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
}

unsafe impl Sync for SpscRing {}
unsafe impl Send for SpscRing {}

impl SpscRing {
    /// Allocates a new SPSC ring with `capacity_bytes` of storage (rounded up to at least 256 bytes).
    pub fn with_capacity(capacity_bytes: usize) -> Self {
        let cap = capacity_bytes.max(256);
        let mut v = Vec::with_capacity(cap);
        for _ in 0..cap {
            v.push(UnsafeCell::new(0u8));
        }
        Self {
            buf: v.into_boxed_slice(),
            capacity: cap,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    #[inline]
    fn buffer_ptr(&self) -> *mut u8 {
        self.buf.as_ptr() as *mut u8
    }

    /// Pushes a record (`group`, `ticks`, `record_bytes` including the 8-byte timestamp slot)
    /// into the ring buffer. Returns `true` if enqueued, or `false` if the ring is full.
    ///
    /// Safe to call from a single producer while a single consumer calls [`SpscRing::drain_into`].
    pub fn try_push(&self, group: GroupHandle, ticks: u64, record_bytes: &[u8]) -> bool {
        let frame_len = 12 + record_bytes.len();
        if frame_len + 4 >= self.capacity {
            return false;
        }
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        let ptr = self.buffer_ptr();

        if head >= tail {
            let right_free = self.capacity - head;
            if right_free >= frame_len + 2 {
                unsafe {
                    write_frame(ptr.add(head), group.0, ticks, record_bytes);
                }
                self.head.store(head + frame_len, Ordering::Release);
                return true;
            }
            // Need to wrap around to 0: check if `tail > frame_len`
            if tail > frame_len {
                if right_free >= 2 {
                    unsafe {
                        core::ptr::copy_nonoverlapping(
                            0xFFFFu16.to_le_bytes().as_ptr(),
                            ptr.add(head),
                            2,
                        );
                    }
                }
                unsafe {
                    write_frame(ptr, group.0, ticks, record_bytes);
                }
                self.head.store(frame_len, Ordering::Release);
                return true;
            }
            false
        } else {
            let free = tail - head;
            if free > frame_len {
                unsafe {
                    write_frame(ptr.add(head), group.0, ticks, record_bytes);
                }
                self.head.store(head + frame_len, Ordering::Release);
                true
            } else {
                false
            }
        }
    }

    /// Drains all currently queued records from the ring into `logger`, returning the number of records drained.
    pub fn drain_into<S: Sink, T: TimeSource>(&self, logger: &mut Logger<S, T>) -> Result<usize> {
        let mut tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        let mut count = 0usize;
        let ptr = self.buffer_ptr();

        while tail != head {
            if self.capacity - tail < 2 {
                tail = 0;
                continue;
            }
            let mut g_raw_bytes = [0u8; 2];
            unsafe {
                core::ptr::copy_nonoverlapping(ptr.add(tail), g_raw_bytes.as_mut_ptr(), 2);
            }
            let g_raw = u16::from_le_bytes(g_raw_bytes);
            if g_raw == 0xFFFF {
                tail = 0;
                if tail == head {
                    break;
                }
            }
            if self.capacity - tail < 12 {
                tail = 0;
                continue;
            }
            let mut frame_hdr = [0u8; 12];
            unsafe {
                core::ptr::copy_nonoverlapping(ptr.add(tail), frame_hdr.as_mut_ptr(), 12);
            }
            let group = u16::from_le_bytes([frame_hdr[0], frame_hdr[1]]);
            let len = u16::from_le_bytes([frame_hdr[2], frame_hdr[3]]) as usize;
            let ticks = u64::from_le_bytes(frame_hdr[4..12].try_into().unwrap());
            let payload = unsafe { core::slice::from_raw_parts(ptr.add(tail + 12), len) };
            logger.push_raw_at(GroupHandle(group), ticks, payload)?;
            tail += 12 + len;
            count += 1;
        }

        self.tail.store(tail, Ordering::Release);
        Ok(count)
    }
}

unsafe fn write_frame(dst: *mut u8, group: u16, ticks: u64, payload: &[u8]) {
    let len = payload.len() as u16;
    unsafe {
        core::ptr::copy_nonoverlapping(group.to_le_bytes().as_ptr(), dst, 2);
        core::ptr::copy_nonoverlapping(len.to_le_bytes().as_ptr(), dst.add(2), 2);
        core::ptr::copy_nonoverlapping(ticks.to_le_bytes().as_ptr(), dst.add(4), 8);
        core::ptr::copy_nonoverlapping(payload.as_ptr(), dst.add(12), payload.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocks::cn::CnDataType;
    use crate::compress::CompressionAlgorithm;
    use crate::io::VecSink;
    use crate::logger::{DataLayout, LoggerBuilder};
    use crate::time::ManualClock;

    #[test]
    fn spsc_wrap_and_drain() {
        let mut b = LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::NoCompression);
        let g = b.add_group("FastISR");
        b.add_channel(g, "ADC", CnDataType::UIntLE, 16).unwrap();
        let schema = b.build().unwrap();
        let mut logger =
            Logger::start(schema, VecSink::default(), ManualClock::new(1e-6, 1)).unwrap();

        let ring = SpscRing::with_capacity(256);
        for i in 0..40u64 {
            let mut rec = [0u8; 10];
            rec[8..10].copy_from_slice(&(i as u16).to_le_bytes());
            if !ring.try_push(g, i * 100, &rec) {
                ring.drain_into(&mut logger).unwrap();
                assert!(ring.try_push(g, i * 100, &rec));
            }
        }
        ring.drain_into(&mut logger).unwrap();
        let (sink, _) = logger.close().unwrap();
        assert!(&sink.data[0..8] == b"MDF     ");
    }
}
