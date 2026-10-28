//! Embedded-ready `extern "C"` API for the MDF 4.3 streaming logger (`no_std + alloc`).
//!
//! Uses C function-pointer vtables for [`Sink`] (`mdfr_sink_t`) and [`TimeSource`] (`mdfr_clock_t`),
//! returns integer error codes (`0` = OK, negative = [`Error::code`]), and never panics across FFI.
use alloc::boxed::Box;
use alloc::string::String;
use core::ffi::c_char;
use core::slice;

use crate::blocks::cn::CnDataType;
use crate::compress::CompressionAlgorithm;
use crate::error::{Error, Result};
use crate::io::{SeekFrom, Sink};
use crate::logger::{
    BusPreset, ChannelHandle, ConversionDef, DataLayout, GroupHandle, Logger, LoggerBuilder,
};
use crate::time::TimeSource;

/// C-compatible sink vtable (`write`, `seek`, `flush`, optional `close` + user context pointer).
#[repr(C)]
pub struct MdfrSinkVtable {
    /// Opaque user context passed to each callback (e.g. `FIL*` for FatFs or `FILE*` for POSIX).
    pub ctx: *mut core::ffi::c_void,
    /// Writes `len` bytes from `buf`. Returns `0` on success, non-zero on error.
    pub write:
        Option<extern "C" fn(ctx: *mut core::ffi::c_void, buf: *const u8, len: usize) -> i32>,
    /// Seeks to absolute byte offset `pos`. Returns `0` on success, non-zero on error.
    pub seek: Option<extern "C" fn(ctx: *mut core::ffi::c_void, pos: u64) -> i32>,
    /// Flushes buffered data. Returns `0` on success, non-zero on error.
    pub flush: Option<extern "C" fn(ctx: *mut core::ffi::c_void) -> i32>,
    /// Optional cleanup callback invoked when the sink is closed or rotated out.
    pub close: Option<extern "C" fn(ctx: *mut core::ffi::c_void) -> i32>,
}

impl Drop for MdfrSinkVtable {
    fn drop(&mut self) {
        if let Some(cb) = self.close {
            let _ = cb(self.ctx);
        }
    }
}

impl Sink for MdfrSinkVtable {
    fn write_all(&mut self, buf: &[u8]) -> Result<()> {
        let cb = self.write.ok_or(Error::Io)?;
        if cb(self.ctx, buf.as_ptr(), buf.len()) == 0 {
            Ok(())
        } else {
            Err(Error::Io)
        }
    }

    fn seek(&mut self, pos: SeekFrom) -> Result<u64> {
        let SeekFrom::Start(offset) = pos else {
            return Err(Error::Io);
        };
        let cb = self.seek.ok_or(Error::Io)?;
        if cb(self.ctx, offset) == 0 {
            Ok(offset)
        } else {
            Err(Error::Io)
        }
    }

    fn flush(&mut self) -> Result<()> {
        if let Some(cb) = self.flush {
            if cb(self.ctx) != 0 {
                return Err(Error::Io);
            }
        }
        Ok(())
    }
}

/// C-compatible clock vtable (`now_ticks`, `tick_ns`, optional `wall_clock_ns` + user context pointer).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MdfrClockVtable {
    /// Opaque user context passed to callbacks.
    pub ctx: *mut core::ffi::c_void,
    /// Returns current monotonic tick counter.
    pub now_ticks: Option<extern "C" fn(ctx: *mut core::ffi::c_void) -> u64>,
    /// Nanoseconds per tick (e.g. `1000` for a 1 MHz microsecond timer).
    pub tick_ns: u64,
    /// Optional callback that writes UTC nanoseconds since 1970-01-01 to `*out_ns` and returns `0` if valid.
    pub wall_clock_ns: Option<extern "C" fn(ctx: *mut core::ffi::c_void, out_ns: *mut u64) -> i32>,
}

impl TimeSource for MdfrClockVtable {
    fn now_ticks(&mut self) -> u64 {
        self.now_ticks.map(|cb| cb(self.ctx)).unwrap_or(0)
    }

    fn seconds_per_tick(&self) -> f64 {
        if self.tick_ns == 0 {
            1e-6
        } else {
            (self.tick_ns as f64) * 1e-9
        }
    }

    fn wall_clock_ns(&mut self) -> Option<u64> {
        let cb = self.wall_clock_ns?;
        let mut out = 0u64;
        if cb(self.ctx, &mut out) == 0 && out != 0 {
            Some(out)
        } else {
            None
        }
    }
}

/// Opaque handle to a [`LoggerBuilder`].
pub struct MdfrBuilderHandle(pub LoggerBuilder);

/// Opaque handle to an active [`Logger`] over C vtables.
pub struct MdfrLoggerHandle(pub Logger<MdfrSinkVtable, MdfrClockVtable>);

unsafe fn cstr_to_str<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    let mut len = 0usize;
    unsafe {
        while *ptr.add(len) != 0 {
            len += 1;
        }
        let bytes = slice::from_raw_parts(ptr as *const u8, len);
        core::str::from_utf8(bytes).ok()
    }
}

/// Creates a new logger schema builder.
/// - `layout`: `0` = Sorted, `1` = Unsorted
/// - `compression`: `255` = NoCompression, `0` = Deflate, `1` = DeflateTranspose, `2` = Zstd, `3` = ZstdTranspose, `4` = Lz4, `5` = Lz4Transpose
#[unsafe(no_mangle)]
pub extern "C" fn mdfr_logger_builder_new(layout: u8, compression: u8) -> *mut MdfrBuilderHandle {
    let l = if layout == 1 {
        DataLayout::Unsorted
    } else {
        DataLayout::Sorted
    };
    let c = match compression {
        0 => CompressionAlgorithm::Deflate,
        1 => CompressionAlgorithm::DeflateTranspose,
        2 => CompressionAlgorithm::Zstd,
        3 => CompressionAlgorithm::ZstdTranspose,
        4 => CompressionAlgorithm::Lz4,
        5 => CompressionAlgorithm::Lz4Transpose,
        _ => CompressionAlgorithm::NoCompression,
    };
    Box::into_raw(Box::new(MdfrBuilderHandle(LoggerBuilder::new(l, c))))
}

/// Frees a [`MdfrBuilderHandle`] if it was not consumed by [`mdfr_logger_start`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdfr_logger_builder_free(builder: *mut MdfrBuilderHandle) {
    if !builder.is_null() {
        unsafe {
            drop(Box::from_raw(builder));
        }
    }
}

/// Adds a channel group with an automatic 64-bit tick master time channel. Writes the group index to `*out_group`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdfr_logger_add_group(
    builder: *mut MdfrBuilderHandle,
    name: *const c_char,
    out_group: *mut u16,
) -> i32 {
    let Some(b) = (unsafe { builder.as_mut() }) else {
        return -3;
    };
    let Some(name_str) = (unsafe { cstr_to_str(name) }) else {
        return -3;
    };
    let g = b.0.add_group(name_str);
    if !out_group.is_null() {
        unsafe {
            *out_group = g.0;
        }
    }
    0
}

/// Appends a byte-aligned scalar channel to `group`. Writes a packed 32-bit handle (`(group << 16) | channel`) to `*out_ch`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdfr_logger_add_channel(
    builder: *mut MdfrBuilderHandle,
    group: u16,
    name: *const c_char,
    data_type: u8,
    bit_count: u32,
    out_ch: *mut u32,
) -> i32 {
    let Some(b) = (unsafe { builder.as_mut() }) else {
        return -3;
    };
    let Some(name_str) = (unsafe { cstr_to_str(name) }) else {
        return -3;
    };
    let Ok(dt) = CnDataType::try_from(data_type) else {
        return -3;
    };
    match b.0.add_channel(GroupHandle(group), name_str, dt, bit_count) {
        Ok(ch) => {
            if !out_ch.is_null() {
                unsafe {
                    *out_ch = ((ch.group as u32) << 16) | (ch.channel as u32);
                }
            }
            0
        }
        Err(e) => e.code(),
    }
}

/// Appends a VLSD channel (`data_type`: `7` = UTF-8 string, `10` = ByteArray) to `group`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdfr_logger_add_vlsd_channel(
    builder: *mut MdfrBuilderHandle,
    group: u16,
    name: *const c_char,
    data_type: u8,
    out_ch: *mut u32,
) -> i32 {
    let Some(b) = (unsafe { builder.as_mut() }) else {
        return -3;
    };
    let Some(name_str) = (unsafe { cstr_to_str(name) }) else {
        return -3;
    };
    let Ok(dt) = CnDataType::try_from(data_type) else {
        return -3;
    };
    match b.0.add_vlsd_channel(GroupHandle(group), name_str, dt) {
        Ok(ch) => {
            if !out_ch.is_null() {
                unsafe {
                    *out_ch = ((ch.group as u32) << 16) | (ch.channel as u32);
                }
            }
            0
        }
        Err(e) => e.code(),
    }
}

/// Sets a linear conversion (`phys = p1 + p2 * raw`) on a packed channel handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdfr_logger_set_conversion_linear(
    builder: *mut MdfrBuilderHandle,
    ch_packed: u32,
    p1: f64,
    p2: f64,
) -> i32 {
    let Some(b) = (unsafe { builder.as_mut() }) else {
        return -3;
    };
    let ch = unpack_ch(ch_packed);
    match b
        .0
        .set_channel_conversion(ch, ConversionDef::Linear { p1, p2 })
    {
        Ok(_) => 0,
        Err(e) => e.code(),
    }
}

/// Sets the physical unit string on a packed channel handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdfr_logger_set_channel_unit(
    builder: *mut MdfrBuilderHandle,
    ch_packed: u32,
    unit: *const c_char,
) -> i32 {
    let Some(b) = (unsafe { builder.as_mut() }) else {
        return -3;
    };
    let Some(unit_str) = (unsafe { cstr_to_str(unit) }) else {
        return -3;
    };
    let ch = unpack_ch(ch_packed);
    match b.0.set_channel_unit(ch, unit_str) {
        Ok(_) => 0,
        Err(e) => e.code(),
    }
}

/// Adds an ASAM bus logging preset group (`bus_kind`: `0` = CAN, `1` = CAN-FD, `2` = LIN, `3` = Ethernet).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdfr_logger_add_bus_preset(
    builder: *mut MdfrBuilderHandle,
    bus_kind: u8,
    bus_name: *const c_char,
    out_group: *mut u16,
) -> i32 {
    let Some(b) = (unsafe { builder.as_mut() }) else {
        return -3;
    };
    let Some(name_str) = (unsafe { cstr_to_str(bus_name) }) else {
        return -3;
    };
    let res = match bus_kind {
        0 => BusPreset::can(&mut b.0, name_str).map(|h| h.group),
        1 => BusPreset::can_fd(&mut b.0, name_str).map(|h| h.group),
        2 => BusPreset::lin(&mut b.0, name_str),
        3 => BusPreset::ethernet(&mut b.0, name_str),
        _ => return -5,
    };
    match res {
        Ok(g) => {
            if !out_group.is_null() {
                unsafe {
                    *out_group = g.0;
                }
            }
            0
        }
        Err(e) => e.code(),
    }
}

/// Consumes `builder`, writes the MDF 4.3 header hierarchy to `sink`, and returns an active [`MdfrLoggerHandle`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdfr_logger_start(
    builder: *mut MdfrBuilderHandle,
    sink: MdfrSinkVtable,
    clock: MdfrClockVtable,
    out_err: *mut i32,
) -> *mut MdfrLoggerHandle {
    if builder.is_null() {
        if !out_err.is_null() {
            unsafe {
                *out_err = -3;
            }
        }
        return core::ptr::null_mut();
    }
    let b = unsafe { Box::from_raw(builder) };
    let schema = match b.0.build() {
        Ok(s) => s,
        Err(e) => {
            if !out_err.is_null() {
                unsafe {
                    *out_err = e.code();
                }
            }
            return core::ptr::null_mut();
        }
    };
    match Logger::start(schema, sink, clock) {
        Ok(logger) => {
            if !out_err.is_null() {
                unsafe {
                    *out_err = 0;
                }
            }
            Box::into_raw(Box::new(MdfrLoggerHandle(logger)))
        }
        Err(e) => {
            if !out_err.is_null() {
                unsafe {
                    *out_err = e.code();
                }
            }
            core::ptr::null_mut()
        }
    }
}

/// Pushes a complete raw record (`len` bytes, including the 8-byte master timestamp slot at `0..8`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdfr_logger_push_raw(
    logger: *mut MdfrLoggerHandle,
    group: u16,
    record_bytes: *const u8,
    len: usize,
) -> i32 {
    let Some(l) = (unsafe { logger.as_mut() }) else {
        return -3;
    };
    if record_bytes.is_null() {
        return -3;
    }
    let slice = unsafe { slice::from_raw_parts(record_bytes, len) };
    match l.0.push_raw(GroupHandle(group), slice) {
        Ok(()) => 0,
        Err(e) => e.code(),
    }
}

/// Begins assembling a record for `group` (stamping the master time channel from the logger's clock).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdfr_logger_record_begin(
    logger: *mut MdfrLoggerHandle,
    group: u16,
) -> i32 {
    let Some(l) = (unsafe { logger.as_mut() }) else {
        return -3;
    };
    match l.0.begin_record(GroupHandle(group)) {
        Ok(_) => 0,
        Err(e) => e.code(),
    }
}

/// Sets an unsigned integer value on `ch_packed` in its group's staging record.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdfr_logger_record_set_u64(
    logger: *mut MdfrLoggerHandle,
    ch_packed: u32,
    val: u64,
) -> i32 {
    let Some(l) = (unsafe { logger.as_mut() }) else {
        return -3;
    };
    match l.0.set_channel_u64(unpack_ch(ch_packed), val) {
        Ok(()) => 0,
        Err(e) => e.code(),
    }
}

/// Sets a signed integer value on `ch_packed` in its group's staging record.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdfr_logger_record_set_i64(
    logger: *mut MdfrLoggerHandle,
    ch_packed: u32,
    val: i64,
) -> i32 {
    let Some(l) = (unsafe { logger.as_mut() }) else {
        return -3;
    };
    match l.0.set_channel_i64(unpack_ch(ch_packed), val) {
        Ok(()) => 0,
        Err(e) => e.code(),
    }
}

/// Sets a floating-point value on `ch_packed` in its group's staging record.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdfr_logger_record_set_f64(
    logger: *mut MdfrLoggerHandle,
    ch_packed: u32,
    val: f64,
) -> i32 {
    let Some(l) = (unsafe { logger.as_mut() }) else {
        return -3;
    };
    match l.0.set_channel_f64(unpack_ch(ch_packed), val) {
        Ok(()) => 0,
        Err(e) => e.code(),
    }
}

/// Sets a byte array / string / VLSD value on `ch_packed` in its group's staging record.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdfr_logger_record_set_bytes(
    logger: *mut MdfrLoggerHandle,
    ch_packed: u32,
    data: *const u8,
    len: usize,
) -> i32 {
    let Some(l) = (unsafe { logger.as_mut() }) else {
        return -3;
    };
    if data.is_null() && len > 0 {
        return -3;
    }
    let s = if len == 0 {
        &[][..]
    } else {
        unsafe { slice::from_raw_parts(data, len) }
    };
    match l.0.set_channel_bytes(unpack_ch(ch_packed), s) {
        Ok(()) => 0,
        Err(e) => e.code(),
    }
}

/// Commits the staged record for `group`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdfr_logger_record_commit(
    logger: *mut MdfrLoggerHandle,
    group: u16,
) -> i32 {
    let Some(l) = (unsafe { logger.as_mut() }) else {
        return -3;
    };
    match l.0.commit_staged_record(GroupHandle(group)) {
        Ok(()) => 0,
        Err(e) => e.code(),
    }
}

/// Flushes buffered data and patches `DL`/`CG` links and cycle counts in place.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdfr_logger_checkpoint(logger: *mut MdfrLoggerHandle) -> i32 {
    let Some(l) = (unsafe { logger.as_mut() }) else {
        return -3;
    };
    match l.0.checkpoint() {
        Ok(()) => 0,
        Err(e) => e.code(),
    }
}

/// Rotates to `new_sink` (finalizing the old sink in place and incrementing the sequence index).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdfr_logger_rotate(
    logger: *mut MdfrLoggerHandle,
    new_sink: MdfrSinkVtable,
) -> i32 {
    let Some(l) = (unsafe { logger.as_mut() }) else {
        return -3;
    };
    match l.0.rotate(new_sink, None) {
        Ok(_) => 0,
        Err(e) => e.code(),
    }
}

/// Finalizes the MDF 4.3 file in place and frees the logger handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdfr_logger_close(logger: *mut MdfrLoggerHandle) -> i32 {
    if logger.is_null() {
        return -3;
    }
    let l = unsafe { Box::from_raw(logger) };
    match l.0.close() {
        Ok(_) => 0,
        Err(e) => e.code(),
    }
}

#[inline]
fn unpack_ch(packed: u32) -> ChannelHandle {
    ChannelHandle {
        group: (packed >> 16) as u16,
        channel: (packed & 0xFFFF) as u16,
    }
}

#[allow(dead_code)]
const _: fn() = || {
    let _ = String::new;
};
