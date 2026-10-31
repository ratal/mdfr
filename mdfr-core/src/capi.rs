//! Embedded-ready `extern "C"` API for the MDF 4.3 streaming logger (`no_std + alloc`).
//!
//! Uses C function-pointer vtables for [`Sink`] (`mdfr_sink_t`) and [`TimeSource`] (`mdfr_clock_t`),
//! returns integer error codes (`0` = OK, negative = [`Error::code`]), and never panics across FFI.
#![allow(clippy::missing_safety_doc)]

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
        if let Some(cb) = self.flush
            && cb(self.ctx) != 0
        {
            return Err(Error::Io);
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

/// Appends an MDF 4.3 VLSC channel (`data_type`: `7` = UTF-8 string, `10` = ByteArray) and its
/// companion 32-bit size channel (`"{name}_size"`) to `group`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdfr_logger_add_vlsc_channel(
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
    match b.0.add_vlsc_channel(GroupHandle(group), name_str, dt) {
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

/// Sets the custom HD block comment text.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdfr_logger_set_hd_comment(
    builder: *mut MdfrBuilderHandle,
    comment: *const c_char,
) -> i32 {
    let Some(b) = (unsafe { builder.as_mut() }) else {
        return -3;
    };
    let Some(comment_str) = (unsafe { cstr_to_str(comment) }) else {
        return -3;
    };
    b.0.set_hd_comment(comment_str);
    0
}

/// Sets custom tool metadata for the file history (FHBLOCK).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdfr_logger_set_tool_info(
    builder: *mut MdfrBuilderHandle,
    tool_id: *const c_char,
    tool_vendor: *const c_char,
    tool_version: *const c_char,
) -> i32 {
    let Some(b) = (unsafe { builder.as_mut() }) else {
        return -3;
    };
    let Some(id_str) = (unsafe { cstr_to_str(tool_id) }) else {
        return -3;
    };
    let Some(vendor_str) = (unsafe { cstr_to_str(tool_vendor) }) else {
        return -3;
    };
    let Some(ver_str) = (unsafe { cstr_to_str(tool_version) }) else {
        return -3;
    };
    b.0.set_tool_info(id_str, vendor_str, ver_str);
    0
}

/// Adds an ASAM bus logging preset group (`bus_kind`: `0` = CAN, `1` = CAN-FD, `2` = LIN, `3` = Ethernet, `4` = FlexRay).
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
        4 => BusPreset::flexray(&mut b.0, name_str),
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

/// Optional: initializes the dedicated `mdfr-core` static arena with a caller-provided static buffer `[buf..buf+len)`.
/// If not called, `capi-arena` automatically uses its built-in 64 KiB `.bss` arena.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdfr_heap_init(buf: *mut u8, len: usize) -> i32 {
    if unsafe { crate::arena::GLOBAL_ARENA.init_custom(buf, len) } {
        0
    } else {
        -3
    }
}

/// Returns a static null-terminated ASCII description for an `MDFR_ERR_*` code.
#[unsafe(no_mangle)]
pub extern "C" fn mdfr_error_str(err: i32) -> *const c_char {
    let bytes: &'static [u8] = match err {
        0 => b"OK\0",
        -1 => b"I/O error on sink\0",
        -2 => b"block serialization error\0",
        -3 => b"invalid schema or null pointer\0",
        -4 => b"invalid channel or group handle\0",
        -5 => b"unsupported feature\0",
        -6 => b"compression error\0",
        -7 => b"SPSC ring buffer is full\0",
        -8 => b"logger is already closed\0",
        -9 => b"record byte length mismatch\0",
        _ => b"unknown error\0",
    };
    bytes.as_ptr() as *const c_char
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

/// Pushes a complete raw record (`len` bytes) with an explicit hardware tick timestamp `ticks`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdfr_logger_push_raw_at(
    logger: *mut MdfrLoggerHandle,
    group: u16,
    ticks: u64,
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
    match l.0.push_raw_at(GroupHandle(group), ticks, slice) {
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

/// Begins assembling a record for `group` with an explicit hardware tick timestamp `ticks`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdfr_logger_record_begin_at(
    logger: *mut MdfrLoggerHandle,
    group: u16,
    ticks: u64,
) -> i32 {
    let Some(l) = (unsafe { logger.as_mut() }) else {
        return -3;
    };
    match l.0.begin_record_at(GroupHandle(group), ticks) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    struct MemSink {
        data: Vec<u8>,
        pos: u64,
        closed: bool,
    }

    extern "C" fn mem_write(ctx: *mut core::ffi::c_void, buf: *const u8, len: usize) -> i32 {
        let s = unsafe { &mut *(ctx as *mut MemSink) };
        let slice = unsafe { core::slice::from_raw_parts(buf, len) };
        let end = (s.pos as usize) + len;
        if end > s.data.len() {
            s.data.resize(end, 0);
        }
        s.data[s.pos as usize..end].copy_from_slice(slice);
        s.pos += len as u64;
        0
    }

    extern "C" fn mem_seek(ctx: *mut core::ffi::c_void, pos: u64) -> i32 {
        let s = unsafe { &mut *(ctx as *mut MemSink) };
        s.pos = pos;
        0
    }

    extern "C" fn mem_flush(_ctx: *mut core::ffi::c_void) -> i32 {
        0
    }

    extern "C" fn mem_close(ctx: *mut core::ffi::c_void) -> i32 {
        let s = unsafe { &mut *(ctx as *mut MemSink) };
        s.closed = true;
        0
    }

    extern "C" fn mock_now_ticks(_ctx: *mut core::ffi::c_void) -> u64 {
        42_000
    }

    extern "C" fn mock_wall_clock(_ctx: *mut core::ffi::c_void, out_ns: *mut u64) -> i32 {
        if !out_ns.is_null() {
            unsafe {
                *out_ns = 1_700_000_000_000_000_000;
            }
            0
        } else {
            -1
        }
    }

    #[test]
    fn test_capi_builder_and_logger_lifecycle() {
        unsafe {
            let builder = mdfr_logger_builder_new(0, 255); // Sorted, NoCompression
            assert!(!builder.is_null());

            let mut g0 = 0u16;
            assert_eq!(
                mdfr_logger_add_group(builder, c"Engine".as_ptr(), &mut g0),
                0
            );
            assert_eq!(g0, 0);

            let mut ch_u = 0u32;
            assert_eq!(
                mdfr_logger_add_channel(builder, g0, c"RPM".as_ptr(), 0, 16, &mut ch_u),
                0
            );

            let mut ch_i = 0u32;
            assert_eq!(
                mdfr_logger_add_channel(builder, g0, c"Torque".as_ptr(), 2, 16, &mut ch_i),
                0
            );

            let mut ch_f = 0u32;
            assert_eq!(
                mdfr_logger_add_channel(builder, g0, c"Temp".as_ptr(), 4, 32, &mut ch_f),
                0
            );

            let mut ch_vlsd = 0u32;
            assert_eq!(
                mdfr_logger_add_vlsd_channel(builder, g0, c"Msg".as_ptr(), 7, &mut ch_vlsd),
                0
            );

            assert_eq!(
                mdfr_logger_set_conversion_linear(builder, ch_u, 0.0, 0.5),
                0
            );
            assert_eq!(
                mdfr_logger_set_channel_unit(builder, ch_u, c"rpm".as_ptr()),
                0
            );

            let mut g_can = 0u16;
            assert_eq!(
                mdfr_logger_add_bus_preset(builder, 0, c"CAN1".as_ptr(), &mut g_can),
                0
            );
            assert_eq!(g_can, 1);

            let mut mem1 = Box::new(MemSink {
                data: Vec::new(),
                pos: 0,
                closed: false,
            });
            let sink1 = MdfrSinkVtable {
                ctx: &mut *mem1 as *mut MemSink as *mut core::ffi::c_void,
                write: Some(mem_write),
                seek: Some(mem_seek),
                flush: Some(mem_flush),
                close: Some(mem_close),
            };

            let clock = MdfrClockVtable {
                ctx: core::ptr::null_mut(),
                now_ticks: Some(mock_now_ticks),
                tick_ns: 1_000,
                wall_clock_ns: Some(mock_wall_clock),
            };

            let mut err = -999;
            let logger = mdfr_logger_start(builder, sink1, clock, &mut err);
            assert!(!logger.is_null());
            assert_eq!(err, 0);

            // Record with begin_record
            assert_eq!(mdfr_logger_record_begin(logger, g0), 0);
            assert_eq!(mdfr_logger_record_set_u64(logger, ch_u, 3000), 0);
            assert_eq!(mdfr_logger_record_set_i64(logger, ch_i, -150), 0);
            assert_eq!(mdfr_logger_record_set_f64(logger, ch_f, 92.5), 0);
            assert_eq!(
                mdfr_logger_record_set_bytes(logger, ch_vlsd, b"running".as_ptr(), 7),
                0
            );
            assert_eq!(mdfr_logger_record_commit(logger, g0), 0);

            // Record with begin_at
            assert_eq!(mdfr_logger_record_begin_at(logger, g0, 50_000), 0);
            assert_eq!(mdfr_logger_record_set_u64(logger, ch_u, 3100), 0);
            assert_eq!(mdfr_logger_record_commit(logger, g0), 0);

            // Checkpoint
            assert_eq!(mdfr_logger_checkpoint(logger), 0);

            // Rotate
            let mut mem2 = Box::new(MemSink {
                data: Vec::new(),
                pos: 0,
                closed: false,
            });
            let sink2 = MdfrSinkVtable {
                ctx: &mut *mem2 as *mut MemSink as *mut core::ffi::c_void,
                write: Some(mem_write),
                seek: Some(mem_seek),
                flush: Some(mem_flush),
                close: Some(mem_close),
            };
            assert_eq!(mdfr_logger_rotate(logger, sink2), 0);
            assert!(mem1.closed);

            // Close
            assert_eq!(mdfr_logger_close(logger), 0);
            assert!(mem2.closed);
        }
    }

    #[test]
    fn test_capi_null_safety_and_errors() {
        unsafe {
            assert_eq!(
                mdfr_logger_add_group(core::ptr::null_mut(), c"G".as_ptr(), core::ptr::null_mut()),
                -3
            );
            assert_eq!(
                mdfr_logger_add_channel(
                    core::ptr::null_mut(),
                    0,
                    c"C".as_ptr(),
                    0,
                    8,
                    core::ptr::null_mut()
                ),
                -3
            );
            assert_eq!(
                mdfr_logger_add_vlsd_channel(
                    core::ptr::null_mut(),
                    0,
                    c"V".as_ptr(),
                    7,
                    core::ptr::null_mut()
                ),
                -3
            );
            assert_eq!(
                mdfr_logger_set_conversion_linear(core::ptr::null_mut(), 0, 0.0, 1.0),
                -3
            );
            assert_eq!(
                mdfr_logger_set_channel_unit(core::ptr::null_mut(), 0, c"u".as_ptr()),
                -3
            );
            assert_eq!(
                mdfr_logger_add_bus_preset(
                    core::ptr::null_mut(),
                    0,
                    c"B".as_ptr(),
                    core::ptr::null_mut()
                ),
                -3
            );
            assert_eq!(mdfr_logger_record_begin(core::ptr::null_mut(), 0), -3);
            assert_eq!(
                mdfr_logger_record_begin_at(core::ptr::null_mut(), 0, 100),
                -3
            );
            assert_eq!(mdfr_logger_record_set_u64(core::ptr::null_mut(), 0, 1), -3);
            assert_eq!(mdfr_logger_record_set_i64(core::ptr::null_mut(), 0, 1), -3);
            assert_eq!(
                mdfr_logger_record_set_f64(core::ptr::null_mut(), 0, 1.0),
                -3
            );
            assert_eq!(
                mdfr_logger_record_set_bytes(core::ptr::null_mut(), 0, b"a".as_ptr(), 1),
                -3
            );
            assert_eq!(mdfr_logger_record_commit(core::ptr::null_mut(), 0), -3);
            assert_eq!(
                mdfr_logger_push_raw(core::ptr::null_mut(), 0, b"a".as_ptr(), 1),
                -3
            );
            assert_eq!(
                mdfr_logger_push_raw_at(core::ptr::null_mut(), 0, 100, b"a".as_ptr(), 1),
                -3
            );
            assert_eq!(mdfr_logger_checkpoint(core::ptr::null_mut()), -3);
            assert_eq!(mdfr_logger_close(core::ptr::null_mut()), -3);

            // Builder free with null
            mdfr_logger_builder_free(core::ptr::null_mut());

            // Builder free with valid pointer
            let b = mdfr_logger_builder_new(1, 0);
            mdfr_logger_builder_free(b);

            // Error strings
            assert_eq!(
                core::ffi::CStr::from_ptr(mdfr_error_str(0)).to_bytes(),
                b"OK"
            );
            assert_eq!(
                core::ffi::CStr::from_ptr(mdfr_error_str(-1)).to_bytes(),
                b"I/O error on sink"
            );
            assert_eq!(
                core::ffi::CStr::from_ptr(mdfr_error_str(-3)).to_bytes(),
                b"invalid schema or null pointer"
            );
            assert_eq!(
                core::ffi::CStr::from_ptr(mdfr_error_str(-999)).to_bytes(),
                b"unknown error"
            );

            // Bus preset invalid kind
            let builder = mdfr_logger_builder_new(0, 255);
            let mut g_out = 0u16;
            assert_eq!(
                mdfr_logger_add_bus_preset(builder, 99, c"UNKNOWN".as_ptr(), &mut g_out),
                -5 // unsupported
            );

            // Test bus preset kinds 1 (CAN-FD), 2 (LIN), 3 (Ethernet), 4 (FlexRay)
            for (kind, name) in [
                (1, c"CANFD".as_ptr()),
                (2, c"LIN1".as_ptr()),
                (3, c"ETH1".as_ptr()),
                (4, c"FR1".as_ptr()),
            ] {
                assert_eq!(
                    mdfr_logger_add_bus_preset(builder, kind, name, &mut g_out),
                    0
                );
            }

            // Test VLSC channel addition
            let mut ch_vlsc = 0u32;
            assert_eq!(
                mdfr_logger_add_vlsc_channel(
                    builder,
                    g_out,
                    c"VLSC_Payload".as_ptr(),
                    7,
                    &mut ch_vlsc
                ),
                0
            );

            // Test heap_init
            let mut heap_buf = [0u8; 1024];
            assert_eq!(mdfr_heap_init(core::ptr::null_mut(), 1024), -3);
            assert_eq!(mdfr_heap_init(heap_buf.as_mut_ptr(), 0), -3);
            assert_eq!(mdfr_heap_init(heap_buf.as_mut_ptr(), heap_buf.len()), 0);

            // Add a simple group with 1 u32 channel for push_raw testing (record size = 8 + 4 = 12 bytes)
            let mut g_raw = 0u16;
            assert_eq!(
                mdfr_logger_add_group(builder, c"RawGroup".as_ptr(), &mut g_raw),
                0
            );
            let mut ch_raw = 0u32;
            assert_eq!(
                mdfr_logger_add_channel(builder, g_raw, c"Val".as_ptr(), 0, 32, &mut ch_raw),
                0
            );

            // Start logger and test push_raw and push_raw_at
            let mut mem = Box::new(MemSink {
                data: Vec::new(),
                pos: 0,
                closed: false,
            });
            let sink = MdfrSinkVtable {
                ctx: &mut *mem as *mut MemSink as *mut core::ffi::c_void,
                write: Some(mem_write),
                seek: Some(mem_seek),
                flush: Some(mem_flush),
                close: Some(mem_close),
            };
            let clock = MdfrClockVtable {
                ctx: core::ptr::null_mut(),
                now_ticks: Some(mock_now_ticks),
                tick_ns: 1_000,
                wall_clock_ns: Some(mock_wall_clock),
            };
            let mut err = 0;
            let logger = mdfr_logger_start(builder, sink, clock, &mut err);
            assert!(!logger.is_null());

            // Push raw records (12 bytes)
            let raw_data = [0u8; 12];
            assert_eq!(
                mdfr_logger_push_raw(logger, g_raw, raw_data.as_ptr(), raw_data.len()),
                0
            );
            assert_eq!(
                mdfr_logger_push_raw_at(logger, g_raw, 60_000, raw_data.as_ptr(), raw_data.len()),
                0
            );

            // Mismatched record size
            let bad_data = [0u8; 5];
            assert_eq!(
                mdfr_logger_push_raw(logger, g_raw, bad_data.as_ptr(), bad_data.len()),
                -4
            );

            assert_eq!(mdfr_logger_close(logger), 0);
        }
    }
}
