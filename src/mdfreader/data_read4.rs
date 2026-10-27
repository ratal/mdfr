//! this module implements low level data reading for mdf4 files.
use crate::mdfinfo::mdfinfo4::{CN_F_DATA_STREAM_MODE, Cn4, CnType, Compo};
use anyhow::{Context, Error, Ok, Result, bail};
use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use encoding_rs::{GB18030, UTF_8, UTF_16BE, UTF_16LE, WINDOWS_1252};
use half::f16;
use rayon::prelude::*;
use std::io::Cursor;
use std::str;
use std::sync::{Arc, Mutex};
use unicode_bom::Bom;

use crate::data_holder::channel_data::ChannelData;

/// CANopen Time (6 bytes) → ms since Unix epoch.
/// Layout: bytes 0-3 = ms-within-day [27:0]; bytes 4-5 = days since 1984-01-01.
fn canopen_time_to_unix_ms(bytes: &[u8]) -> i64 {
    let ms = (u32::from_le_bytes(bytes[0..4].try_into().unwrap()) & 0x0FFF_FFFF) as i64;
    let days = u16::from_le_bytes(bytes[4..6].try_into().unwrap()) as i64;
    // CANopen epoch (1984-01-01) is 5113 days = 441_763_200_000 ms after Unix epoch
    441_763_200_000 + days * 86_400_000 + ms
}

/// CANopen Date (7 bytes) → ms since Unix epoch.
/// Layout: b[0-1] ms-in-minute [15:0]; b[2] min [5:0]; b[3] hour [4:0];
/// b[4] day [4:0] (1-based); b[5] month [3:0] (1-based); b[6] year-1984 [6:0].
fn canopen_date_to_unix_ms(bytes: &[u8]) -> i64 {
    let ms_in_min = u16::from_le_bytes(bytes[0..2].try_into().unwrap()) as i64;
    let min = (bytes[2] & 0x3F) as i64;
    let hour = (bytes[3] & 0x1F) as i64;
    let day = (bytes[4] & 0x1F) as i64;
    let mon = (bytes[5] & 0x0F) as i64;
    let year = (bytes[6] & 0x7F) as i64 + 1984;
    days_since_unix_epoch(year, mon, day) * 86_400_000 + hour * 3_600_000 + min * 60_000 + ms_in_min
}

/// Days from Unix epoch (1970-01-01) to the given calendar date.
/// Verified: days_since_unix_epoch(1984, 1, 1) == 5113.
fn days_since_unix_epoch(year: i64, month: i64, day: i64) -> i64 {
    const MONTH_OFFSET: [i64; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let y = year - 1970;
    let leap_days = (y + 1) / 4 - (y + 69) / 100 + (y + 369) / 400;
    let mut d = y * 365 + leap_days + MONTH_OFFSET[(month - 1) as usize] + (day - 1);
    if month > 2 && year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
        d += 1;
    }
    d
}

/// Helper to split a slice into exact compile-time-sized chunks `[u8; N]`,
/// returning an error if there are any trailing remainder bytes or fewer than
/// `expected_chunks` elements.
#[inline]
fn as_exact_chunks<'a, const N: usize>(
    bytes: &'a [u8],
    expected_chunks: usize,
    context_msg: &'static str,
) -> Result<&'a [[u8; N]], Error> {
    let (chunks, remainder) = bytes.as_chunks::<N>();
    if !remainder.is_empty() {
        bail!(
            "{context_msg}: {} remainder bytes left over (chunk size {N})",
            remainder.len()
        );
    }
    if chunks.len() < expected_chunks {
        bail!(
            "{context_msg}: expected at least {expected_chunks} chunks of {N} bytes, got {}",
            chunks.len()
        );
    }
    Ok(&chunks[..expected_chunks])
}

/// converts raw data block containing only one channel into a ndarray
pub fn read_one_channel_array(
    data_bytes: &[u8],
    cn: &mut Cn4,
    cycle_count: usize,
) -> Result<(), Error> {
    if (cn.block.cn_type == 0
        || cn.block.cn_type == 2
        || cn.block.cn_type == 5
        || cn.block.cn_type == 7)
        && !cn.data.is_empty()
    {
        // cn_type == 5 : Maximum length data channel, removing no valid bytes done by another size channel pointed by cn_data
        // cn_type == 0 : fixed length data channel
        // cn_type == 2 : master channel
        // cn_type == 7 : VLSC channel (stores offsets into VD block)
        let n_bytes = cn.n_bytes as usize;
        let list_size = cn.list_size;

        // CANopen Date/Time are stored as Int64 (ms since Unix epoch); parse binary here.
        if cn.block.cn_data_type == 13 || cn.block.cn_data_type == 14 {
            if let ChannelData::Int64(a) = &mut cn.data {
                if cn.block.cn_data_type == 13 {
                    let chunks = as_exact_chunks::<7>(data_bytes, cycle_count, "CANopen Date")?;
                    for record in chunks {
                        a.append_value(canopen_date_to_unix_ms(record));
                    }
                } else {
                    let chunks = as_exact_chunks::<6>(data_bytes, cycle_count, "CANopen Time")?;
                    for record in chunks {
                        a.append_value(canopen_time_to_unix_ms(record));
                    }
                }
            }
            return Ok(());
        }

        match &mut cn.data {
            ChannelData::Int8(a) => {
                let buf = &mut a.values_slice_mut()[..cycle_count];
                Cursor::new(&data_bytes[..cycle_count])
                    .read_i8_into(buf)
                    .context("Could not read i8 array")?;
            }
            ChannelData::UInt8(a) => {
                let buf = &mut a.values_slice_mut()[..cycle_count];
                buf.copy_from_slice(&data_bytes[..cycle_count]);
            }
            ChannelData::Int16(a) => {
                let buf = &mut a.values_slice_mut()[..cycle_count];
                let src = &data_bytes[..cycle_count * std::mem::size_of::<i16>()];
                if cn.endian.is_big() {
                    Cursor::new(src)
                        .read_i16_into::<BigEndian>(buf)
                        .context("Could not read be i16 array")?;
                } else {
                    Cursor::new(src)
                        .read_i16_into::<LittleEndian>(buf)
                        .context("Could not read le i16 array")?;
                }
            }
            ChannelData::UInt16(a) => {
                let buf = &mut a.values_slice_mut()[..cycle_count];
                let src = &data_bytes[..cycle_count * std::mem::size_of::<u16>()];
                if cn.endian.is_big() {
                    Cursor::new(src)
                        .read_u16_into::<BigEndian>(buf)
                        .context("Could not read be u16 array")?;
                } else {
                    Cursor::new(src)
                        .read_u16_into::<LittleEndian>(buf)
                        .context("Could not read le 16 array")?;
                }
            }
            ChannelData::Int32(a) => {
                let data = &mut a.values_slice_mut()[..cycle_count];
                if n_bytes == 4 {
                    let src = &data_bytes[..cycle_count * std::mem::size_of::<i32>()];
                    if cn.endian.is_big() {
                        Cursor::new(src)
                            .read_i32_into::<BigEndian>(data)
                            .context("Could not read be i32 array")?;
                    } else {
                        Cursor::new(src)
                            .read_i32_into::<LittleEndian>(data)
                            .context("Could not read le i32 array")?;
                    }
                } else if n_bytes == 3 {
                    let chunks =
                        as_exact_chunks::<3>(data_bytes, cycle_count, "Could not read i24")?;
                    if cn.endian.is_big() {
                        for (dst, value) in data.iter_mut().zip(chunks) {
                            *dst = value
                                .as_slice()
                                .read_i24::<BigEndian>()
                                .context("Could not read be i24")?;
                        }
                    } else {
                        for (dst, value) in data.iter_mut().zip(chunks) {
                            *dst = value
                                .as_slice()
                                .read_i24::<LittleEndian>()
                                .context("Could not read le i24")?;
                        }
                    }
                }
            }
            ChannelData::UInt32(a) => {
                let data = &mut a.values_slice_mut()[..cycle_count];
                if n_bytes == 4 {
                    let src = &data_bytes[..cycle_count * std::mem::size_of::<u32>()];
                    if cn.endian.is_big() {
                        Cursor::new(src)
                            .read_u32_into::<BigEndian>(data)
                            .context("Could not read be u32 array")?;
                    } else {
                        Cursor::new(src)
                            .read_u32_into::<LittleEndian>(data)
                            .context("Could not read le u32 array")?;
                    }
                } else if n_bytes == 3 {
                    let chunks =
                        as_exact_chunks::<3>(data_bytes, cycle_count, "Could not read u24")?;
                    if cn.endian.is_big() {
                        for (dst, value) in data.iter_mut().zip(chunks) {
                            *dst = value
                                .as_slice()
                                .read_u24::<BigEndian>()
                                .context("Could not read be u24")?;
                        }
                    } else {
                        for (dst, value) in data.iter_mut().zip(chunks) {
                            *dst = value
                                .as_slice()
                                .read_u24::<LittleEndian>()
                                .context("Could not read le u24")?;
                        }
                    }
                }
            }
            ChannelData::Float32(a) => {
                let data = &mut a.values_slice_mut()[..cycle_count];
                if n_bytes == 4 {
                    let chunks = as_exact_chunks::<{ std::mem::size_of::<f32>() }>(
                        data_bytes,
                        cycle_count,
                        "Could not read f32",
                    )?;
                    if cn.endian.is_big() {
                        for (dst, src) in data.iter_mut().zip(chunks) {
                            *dst = f32::from_be_bytes(*src);
                        }
                    } else {
                        for (dst, src) in data.iter_mut().zip(chunks) {
                            *dst = f32::from_le_bytes(*src);
                        }
                    }
                } else if n_bytes == 2 {
                    let chunks = as_exact_chunks::<{ std::mem::size_of::<f16>() }>(
                        data_bytes,
                        cycle_count,
                        "Could not read f16",
                    )?;
                    if cn.endian.is_big() {
                        for (dst, value) in data.iter_mut().zip(chunks) {
                            *dst = f16::from_be_bytes(*value).to_f32();
                        }
                    } else {
                        for (dst, value) in data.iter_mut().zip(chunks) {
                            *dst = f16::from_le_bytes(*value).to_f32();
                        }
                    }
                }
            }
            ChannelData::Int64(a) => {
                let data = &mut a.values_slice_mut()[..cycle_count];
                if n_bytes == 8 {
                    let src = &data_bytes[..cycle_count * std::mem::size_of::<i64>()];
                    if cn.endian.is_big() {
                        Cursor::new(src)
                            .read_i64_into::<BigEndian>(data)
                            .context("Could not read be i64 array")?;
                    } else {
                        Cursor::new(src)
                            .read_i64_into::<LittleEndian>(data)
                            .context("Could not read le i64 array")?;
                    }
                } else if n_bytes == 7 {
                    let chunks =
                        as_exact_chunks::<7>(data_bytes, cycle_count, "Could not read i56")?;
                    let mut temp = [0u8; std::mem::size_of::<u64>()];
                    if cn.endian.is_big() {
                        for (dst, value) in data.iter_mut().zip(chunks) {
                            temp[1..8].copy_from_slice(value);
                            *dst = ((u64::from_be_bytes(temp) as i64) << 8) >> 8;
                        }
                    } else {
                        for (dst, value) in data.iter_mut().zip(chunks) {
                            temp[0..7].copy_from_slice(value);
                            *dst = ((u64::from_le_bytes(temp) as i64) << 8) >> 8;
                        }
                    }
                } else if n_bytes == 6 {
                    let chunks =
                        as_exact_chunks::<6>(data_bytes, cycle_count, "Could not read i48")?;
                    if cn.endian.is_big() {
                        for (dst, value) in data.iter_mut().zip(chunks) {
                            *dst = value
                                .as_slice()
                                .read_i48::<BigEndian>()
                                .context("Could not read be i48")?;
                        }
                    } else {
                        for (dst, value) in data.iter_mut().zip(chunks) {
                            *dst = value
                                .as_slice()
                                .read_i48::<LittleEndian>()
                                .context("Could not read le i48")?;
                        }
                    }
                } else if n_bytes == 5 {
                    let chunks =
                        as_exact_chunks::<5>(data_bytes, cycle_count, "Could not read i40")?;
                    if cn.endian.is_big() {
                        for (dst, src) in data.iter_mut().zip(chunks) {
                            let raw = ((src[0] as u64) << 32)
                                | ((src[1] as u64) << 24)
                                | ((src[2] as u64) << 16)
                                | ((src[3] as u64) << 8)
                                | (src[4] as u64);
                            *dst = ((raw as i64) << 24) >> 24;
                        }
                    } else {
                        for (dst, src) in data.iter_mut().zip(chunks) {
                            let raw = (src[0] as u64)
                                | ((src[1] as u64) << 8)
                                | ((src[2] as u64) << 16)
                                | ((src[3] as u64) << 24)
                                | ((src[4] as u64) << 32);
                            *dst = ((raw as i64) << 24) >> 24;
                        }
                    }
                }
            }
            ChannelData::UInt64(a) => {
                let data = &mut a.values_slice_mut()[..cycle_count];
                if n_bytes == 8 {
                    let src = &data_bytes[..cycle_count * std::mem::size_of::<u64>()];
                    if cn.endian.is_big() {
                        Cursor::new(src)
                            .read_u64_into::<BigEndian>(data)
                            .context("Could not read be u64 array")?;
                    } else {
                        Cursor::new(src)
                            .read_u64_into::<LittleEndian>(data)
                            .context("Could not read le u64 array")?;
                    }
                } else if n_bytes == 7 {
                    let chunks =
                        as_exact_chunks::<7>(data_bytes, cycle_count, "Could not read u56")?;
                    let mut temp = [0u8; std::mem::size_of::<u64>()];
                    if cn.endian.is_big() {
                        for (dst, value) in data.iter_mut().zip(chunks) {
                            temp[1..8].copy_from_slice(value);
                            *dst = u64::from_be_bytes(temp);
                        }
                    } else {
                        for (dst, value) in data.iter_mut().zip(chunks) {
                            temp[0..7].copy_from_slice(value);
                            *dst = u64::from_le_bytes(temp);
                        }
                    }
                } else if n_bytes == 6 {
                    let chunks =
                        as_exact_chunks::<6>(data_bytes, cycle_count, "Could not read u48")?;
                    if cn.endian.is_big() {
                        for (dst, value) in data.iter_mut().zip(chunks) {
                            *dst = value
                                .as_slice()
                                .read_u48::<BigEndian>()
                                .context("Could not read be u48")?;
                        }
                    } else {
                        for (dst, value) in data.iter_mut().zip(chunks) {
                            *dst = value
                                .as_slice()
                                .read_u48::<LittleEndian>()
                                .context("Could not read le u48")?;
                        }
                    }
                } else {
                    // n_bytes = 5
                    let chunks =
                        as_exact_chunks::<5>(data_bytes, cycle_count, "Could not read u40")?;
                    if cn.endian.is_big() {
                        for (dst, src) in data.iter_mut().zip(chunks) {
                            *dst = ((src[0] as u64) << 32)
                                | ((src[1] as u64) << 24)
                                | ((src[2] as u64) << 16)
                                | ((src[3] as u64) << 8)
                                | (src[4] as u64);
                        }
                    } else {
                        for (dst, src) in data.iter_mut().zip(chunks) {
                            *dst = (src[0] as u64)
                                | ((src[1] as u64) << 8)
                                | ((src[2] as u64) << 16)
                                | ((src[3] as u64) << 24)
                                | ((src[4] as u64) << 32);
                        }
                    }
                }
            }
            ChannelData::Float64(a) => {
                let buf = &mut a.values_slice_mut()[..cycle_count];
                let chunks = as_exact_chunks::<{ std::mem::size_of::<f64>() }>(
                    data_bytes,
                    cycle_count,
                    "Could not read f64",
                )?;
                if cn.endian.is_big() {
                    for (dst, src) in buf.iter_mut().zip(chunks) {
                        *dst = f64::from_be_bytes(*src);
                    }
                } else {
                    for (dst, src) in buf.iter_mut().zip(chunks) {
                        *dst = f64::from_le_bytes(*src);
                    }
                }
            }
            ChannelData::Complex32(a) => {
                let total_elems = cycle_count * list_size;
                let data = &mut a.values().values_slice_mut()[..total_elems];
                if n_bytes <= 4 {
                    // complex 16
                    let chunks = as_exact_chunks::<{ std::mem::size_of::<f16>() }>(
                        data_bytes,
                        total_elems,
                        "Could not read complex f16",
                    )?;
                    if cn.endian.is_big() {
                        for (dst, value) in data.iter_mut().zip(chunks) {
                            *dst = f16::from_be_bytes(*value).to_f32();
                        }
                    } else {
                        for (dst, value) in data.iter_mut().zip(chunks) {
                            *dst = f16::from_le_bytes(*value).to_f32();
                        }
                    }
                } else if n_bytes <= 8 {
                    // complex 32
                    let chunks = as_exact_chunks::<{ std::mem::size_of::<f32>() }>(
                        data_bytes,
                        total_elems,
                        "Could not read complex f32",
                    )?;
                    if cn.endian.is_big() {
                        for (dst, src) in data.iter_mut().zip(chunks) {
                            *dst = f32::from_be_bytes(*src);
                        }
                    } else {
                        for (dst, src) in data.iter_mut().zip(chunks) {
                            *dst = f32::from_le_bytes(*src);
                        }
                    }
                }
            }
            ChannelData::Complex64(a) => {
                let total_elems = cycle_count * list_size;
                let data = &mut a.values().values_slice_mut()[..total_elems];
                let chunks = as_exact_chunks::<{ std::mem::size_of::<f64>() }>(
                    data_bytes,
                    total_elems,
                    "Could not read complex f64",
                )?;
                if cn.endian.is_big() {
                    for (dst, src) in data.iter_mut().zip(chunks) {
                        *dst = f64::from_be_bytes(*src);
                    }
                } else {
                    for (dst, src) in data.iter_mut().zip(chunks) {
                        *dst = f64::from_le_bytes(*src);
                    }
                }
            }
            ChannelData::Utf8(data) => {
                let chunks = data_bytes.chunks_exact(n_bytes);
                if !chunks.remainder().is_empty() {
                    bail!(
                        "String channel {} has {} remainder bytes (n_bytes={})",
                        cn.unique_name,
                        chunks.remainder().len(),
                        n_bytes
                    );
                }
                let mut dst = String::with_capacity(n_bytes);
                if cn.block.cn_data_type == 6 {
                    // SBC ISO-8859-1 to be converted into UTF8
                    let mut decoder = WINDOWS_1252.new_decoder();
                    for value in chunks.take(cycle_count) {
                        dst.clear();
                        let (_result, _size, _replacement) =
                            decoder.decode_to_string(value, &mut dst, false);
                        let s = dst.split('\0').next().unwrap_or("");
                        data.append_value(s);
                    }
                } else if cn.block.cn_data_type == 7 {
                    // 7: String UTF8
                    for value in chunks.take(cycle_count) {
                        let full = str::from_utf8(value).context("Found invalid UTF-8")?;
                        let s = full.split('\0').next().unwrap_or("");
                        data.append_value(s);
                    }
                } else if cn.block.cn_data_type == 8 || cn.block.cn_data_type == 9 {
                    // 8 | 9 :String UTF16 to be converted into UTF8
                    if cn.endian.is_big() {
                        let mut decoder = UTF_16BE.new_decoder();
                        for record in chunks.take(cycle_count) {
                            dst.clear();
                            let (_result, _size, _replacement) =
                                decoder.decode_to_string(record, &mut dst, false);
                            let s = dst.split('\0').next().unwrap_or("");
                            data.append_value(s);
                        }
                    } else {
                        let mut decoder = UTF_16LE.new_decoder();
                        for record in chunks.take(cycle_count) {
                            dst.clear();
                            let (_result, _size, _replacement) =
                                decoder.decode_to_string(record, &mut dst, false);
                            let s = dst.split('\0').next().unwrap_or("");
                            data.append_value(s);
                        }
                    }
                } else if cn.block.cn_data_type == 17 {
                    // 17: Unicode with BOM, assuming each sample has same BOM
                    // identifies BOM at beginning of stream
                    let bom = Bom::from(data_bytes);
                    let mut decoder = match bom {
                        Bom::Utf8 => UTF_8.new_decoder(),
                        Bom::Utf16Be => UTF_16BE.new_decoder(),
                        Bom::Utf16Le => UTF_16LE.new_decoder(),
                        Bom::Gb18030 => GB18030.new_decoder(),
                        _ => {
                            bail!("not implemented BOM type");
                        }
                    };
                    for record in chunks.take(cycle_count) {
                        dst.clear();
                        let (_result, _size, _replacement) =
                            decoder.decode_to_string(record, &mut dst, false);
                        let s = dst.split('\0').next().unwrap_or("");
                        data.append_value(s);
                    }
                }
            }
            ChannelData::VariableSizeByteArray(a) => {
                // no validity at this point
                let chunks = data_bytes.chunks_exact(n_bytes);
                if !chunks.remainder().is_empty() {
                    bail!(
                        "ByteArray channel {} has {} remainder bytes (n_bytes={})",
                        cn.unique_name,
                        chunks.remainder().len(),
                        n_bytes
                    );
                }
                for value in chunks.take(cycle_count) {
                    a.append_value(value);
                }
            }
            ChannelData::FixedSizeByteArray(a) => {
                let chunks = data_bytes.chunks_exact(n_bytes);
                if !chunks.remainder().is_empty() {
                    bail!(
                        "FixedSizeByteArray channel {} has {} remainder bytes (n_bytes={})",
                        cn.unique_name,
                        chunks.remainder().len(),
                        n_bytes
                    );
                }
                for value in chunks.take(cycle_count) {
                    a.append_value(value)
                        .context("failed appending new value")?;
                }
            }
            ChannelData::ArrayDInt8(a) => {
                let total_elems = cycle_count * list_size;
                let buf = &mut a.values_slice_mut()[..total_elems];
                Cursor::new(&data_bytes[..total_elems])
                    .read_i8_into(buf)
                    .context("Could not read i8 array")?;
            }
            ChannelData::ArrayDUInt8(a) => {
                let total_elems = cycle_count * list_size;
                let buf = &mut a.values_slice_mut()[..total_elems];
                buf.copy_from_slice(&data_bytes[..total_elems]);
            }
            ChannelData::ArrayDInt16(a) => {
                let total_elems = cycle_count * list_size;
                let buf = &mut a.values_slice_mut()[..total_elems];
                let src = &data_bytes[..total_elems * std::mem::size_of::<i16>()];
                if cn.endian.is_big() {
                    Cursor::new(src)
                        .read_i16_into::<BigEndian>(buf)
                        .context("Could not read be i16 array")?;
                } else {
                    Cursor::new(src)
                        .read_i16_into::<LittleEndian>(buf)
                        .context("Could not read le i16 array")?;
                }
            }
            ChannelData::ArrayDUInt16(a) => {
                let total_elems = cycle_count * list_size;
                let buf = &mut a.values_slice_mut()[..total_elems];
                let src = &data_bytes[..total_elems * std::mem::size_of::<u16>()];
                if cn.endian.is_big() {
                    Cursor::new(src)
                        .read_u16_into::<BigEndian>(buf)
                        .context("Could not read be u16 array")?;
                } else {
                    Cursor::new(src)
                        .read_u16_into::<LittleEndian>(buf)
                        .context("Could not read le 16 array")?;
                }
            }
            ChannelData::ArrayDInt32(a) => {
                let total_elems = cycle_count * list_size;
                let buf = &mut a.values_slice_mut()[..total_elems];
                if cn.endian.is_big() {
                    if n_bytes <= 3 {
                        let chunks = as_exact_chunks::<3>(
                            data_bytes,
                            total_elems,
                            "Could not read be i24 array",
                        )?;
                        for (dst, value) in buf.iter_mut().zip(chunks) {
                            *dst = value
                                .as_slice()
                                .read_i24::<BigEndian>()
                                .context("Could not read be i24")?;
                        }
                    } else {
                        let src = &data_bytes[..total_elems * std::mem::size_of::<i32>()];
                        Cursor::new(src)
                            .read_i32_into::<BigEndian>(buf)
                            .context("Could not read be i32 array")?;
                    }
                } else if n_bytes <= 3 {
                    let chunks = as_exact_chunks::<3>(
                        data_bytes,
                        total_elems,
                        "Could not read le i24 array",
                    )?;
                    for (dst, value) in buf.iter_mut().zip(chunks) {
                        *dst = value
                            .as_slice()
                            .read_i24::<LittleEndian>()
                            .context("Could not read le i24")?;
                    }
                } else {
                    let src = &data_bytes[..total_elems * std::mem::size_of::<i32>()];
                    Cursor::new(src)
                        .read_i32_into::<LittleEndian>(buf)
                        .context("Could not read le i32 array")?;
                }
            }
            ChannelData::ArrayDUInt32(a) => {
                let total_elems = cycle_count * list_size;
                let buf = &mut a.values_slice_mut()[..total_elems];
                if cn.endian.is_big() {
                    if n_bytes <= 3 {
                        let chunks = as_exact_chunks::<3>(
                            data_bytes,
                            total_elems,
                            "Could not read be u24 array",
                        )?;
                        for (dst, value) in buf.iter_mut().zip(chunks) {
                            *dst = value
                                .as_slice()
                                .read_u24::<BigEndian>()
                                .context("Could not read be u24")?;
                        }
                    } else {
                        let src = &data_bytes[..total_elems * std::mem::size_of::<u32>()];
                        Cursor::new(src)
                            .read_u32_into::<BigEndian>(buf)
                            .context("Could not read be u32 array")?;
                    }
                } else if n_bytes <= 3 {
                    let chunks = as_exact_chunks::<3>(
                        data_bytes,
                        total_elems,
                        "Could not read le u24 array",
                    )?;
                    for (dst, value) in buf.iter_mut().zip(chunks) {
                        *dst = value
                            .as_slice()
                            .read_u24::<LittleEndian>()
                            .context("Could not read le u24")?;
                    }
                } else {
                    let src = &data_bytes[..total_elems * std::mem::size_of::<u32>()];
                    Cursor::new(src)
                        .read_u32_into::<LittleEndian>(buf)
                        .context("Could not read le u32 array")?;
                }
            }
            ChannelData::ArrayDFloat32(a) => {
                let total_elems = cycle_count * list_size;
                let buf = &mut a.values_slice_mut()[..total_elems];
                if cn.endian.is_big() {
                    if n_bytes == 2 {
                        let chunks = as_exact_chunks::<{ std::mem::size_of::<f16>() }>(
                            data_bytes,
                            total_elems,
                            "Could not read be f16 array",
                        )?;
                        for (dst, value) in buf.iter_mut().zip(chunks) {
                            *dst = f16::from_be_bytes(*value).to_f32();
                        }
                    } else {
                        let chunks = as_exact_chunks::<{ std::mem::size_of::<f32>() }>(
                            data_bytes,
                            total_elems,
                            "Could not read be f32 array",
                        )?;
                        for (dst, src) in buf.iter_mut().zip(chunks) {
                            *dst = f32::from_be_bytes(*src);
                        }
                    }
                } else if n_bytes == 2 {
                    let chunks = as_exact_chunks::<{ std::mem::size_of::<f16>() }>(
                        data_bytes,
                        total_elems,
                        "Could not read le f16 array",
                    )?;
                    for (dst, value) in buf.iter_mut().zip(chunks) {
                        *dst = f16::from_le_bytes(*value).to_f32();
                    }
                } else {
                    let chunks = as_exact_chunks::<{ std::mem::size_of::<f32>() }>(
                        data_bytes,
                        total_elems,
                        "Could not read le f32 array",
                    )?;
                    for (dst, src) in buf.iter_mut().zip(chunks) {
                        *dst = f32::from_le_bytes(*src);
                    }
                }
            }
            ChannelData::ArrayDInt64(a) => {
                let total_elems = cycle_count * list_size;
                let buf = &mut a.values_slice_mut()[..total_elems];
                if cn.endian.is_big() {
                    if n_bytes == 8 {
                        let src = &data_bytes[..total_elems * std::mem::size_of::<i64>()];
                        Cursor::new(src)
                            .read_i64_into::<BigEndian>(buf)
                            .context("Could not read be i64 array")?;
                    } else if n_bytes == 7 {
                        let chunks = as_exact_chunks::<7>(
                            data_bytes,
                            total_elems,
                            "Could not read be i56 array",
                        )?;
                        let mut temp = [0u8; std::mem::size_of::<u64>()];
                        for (dst, value) in buf.iter_mut().zip(chunks) {
                            temp[1..8].copy_from_slice(value);
                            *dst = ((u64::from_be_bytes(temp) as i64) << 8) >> 8;
                        }
                    } else if n_bytes == 6 {
                        let chunks = as_exact_chunks::<6>(
                            data_bytes,
                            total_elems,
                            "Could not read be i48 array",
                        )?;
                        for (dst, value) in buf.iter_mut().zip(chunks) {
                            *dst = value
                                .as_slice()
                                .read_i48::<BigEndian>()
                                .context("Could not read be i48")?;
                        }
                    } else if n_bytes == 5 {
                        let chunks = as_exact_chunks::<5>(
                            data_bytes,
                            total_elems,
                            "Could not read be i40 array",
                        )?;
                        for (dst, src) in buf.iter_mut().zip(chunks) {
                            let raw = ((src[0] as u64) << 32)
                                | ((src[1] as u64) << 24)
                                | ((src[2] as u64) << 16)
                                | ((src[3] as u64) << 8)
                                | (src[4] as u64);
                            *dst = ((raw as i64) << 24) >> 24;
                        }
                    }
                } else if n_bytes == 8 {
                    let src = &data_bytes[..total_elems * std::mem::size_of::<i64>()];
                    Cursor::new(src)
                        .read_i64_into::<LittleEndian>(buf)
                        .context("Could not read le i64 array")?;
                } else if n_bytes == 7 {
                    let chunks = as_exact_chunks::<7>(
                        data_bytes,
                        total_elems,
                        "Could not read le i56 array",
                    )?;
                    let mut temp = [0u8; std::mem::size_of::<u64>()];
                    for (dst, value) in buf.iter_mut().zip(chunks) {
                        temp[0..7].copy_from_slice(value);
                        *dst = ((u64::from_le_bytes(temp) as i64) << 8) >> 8;
                    }
                } else if n_bytes == 6 {
                    let chunks = as_exact_chunks::<6>(
                        data_bytes,
                        total_elems,
                        "Could not read le i48 array",
                    )?;
                    for (dst, value) in buf.iter_mut().zip(chunks) {
                        *dst = value
                            .as_slice()
                            .read_i48::<LittleEndian>()
                            .context("Could not read le i48")?;
                    }
                } else if n_bytes == 5 {
                    let chunks = as_exact_chunks::<5>(
                        data_bytes,
                        total_elems,
                        "Could not read le i40 array",
                    )?;
                    for (dst, src) in buf.iter_mut().zip(chunks) {
                        let raw = (src[0] as u64)
                            | ((src[1] as u64) << 8)
                            | ((src[2] as u64) << 16)
                            | ((src[3] as u64) << 24)
                            | ((src[4] as u64) << 32);
                        *dst = ((raw as i64) << 24) >> 24;
                    }
                }
            }
            ChannelData::ArrayDUInt64(a) => {
                let total_elems = cycle_count * list_size;
                let buf = &mut a.values_slice_mut()[..total_elems];
                if n_bytes == 8 {
                    let src = &data_bytes[..total_elems * std::mem::size_of::<u64>()];
                    if cn.endian.is_big() {
                        Cursor::new(src)
                            .read_u64_into::<BigEndian>(buf)
                            .context("Could not read be u64 array")?;
                    } else {
                        Cursor::new(src)
                            .read_u64_into::<LittleEndian>(buf)
                            .context("Could not read le u64 array")?;
                    }
                } else if n_bytes == 7 {
                    let chunks =
                        as_exact_chunks::<7>(data_bytes, total_elems, "Could not read u56 array")?;
                    let mut temp = [0u8; std::mem::size_of::<u64>()];
                    if cn.endian.is_big() {
                        for (dst, value) in buf.iter_mut().zip(chunks) {
                            temp[1..8].copy_from_slice(value);
                            *dst = u64::from_be_bytes(temp);
                        }
                    } else {
                        for (dst, value) in buf.iter_mut().zip(chunks) {
                            temp[0..7].copy_from_slice(value);
                            *dst = u64::from_le_bytes(temp);
                        }
                    }
                } else if n_bytes == 6 {
                    let chunks =
                        as_exact_chunks::<6>(data_bytes, total_elems, "Could not read u48 array")?;
                    if cn.endian.is_big() {
                        for (dst, value) in buf.iter_mut().zip(chunks) {
                            *dst = value
                                .as_slice()
                                .read_u48::<BigEndian>()
                                .context("Could not read be u48")?;
                        }
                    } else {
                        for (dst, value) in buf.iter_mut().zip(chunks) {
                            *dst = value
                                .as_slice()
                                .read_u48::<LittleEndian>()
                                .context("Could not read le u48")?;
                        }
                    }
                } else if n_bytes == 5 {
                    let chunks =
                        as_exact_chunks::<5>(data_bytes, total_elems, "Could not read u40 array")?;
                    if cn.endian.is_big() {
                        for (dst, src) in buf.iter_mut().zip(chunks) {
                            *dst = ((src[0] as u64) << 32)
                                | ((src[1] as u64) << 24)
                                | ((src[2] as u64) << 16)
                                | ((src[3] as u64) << 8)
                                | (src[4] as u64);
                        }
                    } else {
                        for (dst, src) in buf.iter_mut().zip(chunks) {
                            *dst = (src[0] as u64)
                                | ((src[1] as u64) << 8)
                                | ((src[2] as u64) << 16)
                                | ((src[3] as u64) << 24)
                                | ((src[4] as u64) << 32);
                        }
                    }
                }
            }
            ChannelData::ArrayDFloat64(a) => {
                let total_elems = cycle_count * list_size;
                let buf = &mut a.values_slice_mut()[..total_elems];
                let chunks = as_exact_chunks::<{ std::mem::size_of::<f64>() }>(
                    data_bytes,
                    total_elems,
                    "Could not read f64 array",
                )?;
                if cn.endian.is_big() {
                    for (dst, src) in buf.iter_mut().zip(chunks) {
                        *dst = f64::from_be_bytes(*src);
                    }
                } else {
                    for (dst, src) in buf.iter_mut().zip(chunks) {
                        *dst = f64::from_le_bytes(*src);
                    }
                }
            }
            ChannelData::Union(_) => {} // Union channels are constructed post-read
        }
    }
    // Other channel types : virtual channels cn_type 3 & 6 are handled at initialisation
    // cn_type == 1 VLSD not possible for sorted data
    Ok(())
}

/// copies data from data_chunk into each channel array
pub fn read_channels_from_bytes(
    data_chunk: &[u8],
    channels: &mut CnType,
    record_length: usize,
    previous_index: usize,
    record_with_invalid_data: bool,
) -> Result<Vec<(u8, i32)>, Error> {
    if record_length == 0 || data_chunk.is_empty() {
        return Ok(Vec::new());
    }
    if !data_chunk.len().is_multiple_of(record_length) {
        bail!(
            "Data chunk length ({}) is not a multiple of record_length ({}): {} remainder bytes",
            data_chunk.len(),
            record_length,
            data_chunk.len() % record_length
        );
    }
    let vlsd_channels: Arc<Mutex<Vec<(u8, i32)>>> = Arc::new(Mutex::new(Vec::new()));
    channels
        .par_iter_mut()
        .filter(|(_cn_record_position, cn)| cn.should_read)
        .try_for_each(|(rec_pos, cn): (&i32, &mut Cn4)| -> Result<(), Error> {
            if let Some(compo) = &cn.composition
                && let Compo::CA(ca) = &compo.block
                && (ca.ca_storage == 1 || ca.ca_storage == 2)
            {
                return Ok(());
            }
            if cn.block.cn_type == 0
                || cn.block.cn_type == 2
                || cn.block.cn_type == 5
                || cn.block.cn_type == 7
            {
                let mut value: &[u8];
                let pos_byte_beg = cn.pos_byte_beg as usize;
                let n_bytes = cn.n_bytes as usize;
                let list_size = cn.list_size;
                match &mut cn.data {
                    ChannelData::Int8(a) => {
                        let data = a.values_slice_mut();
                        for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                            data[i + previous_index] = record[pos_byte_beg] as i8;
                        }
                    }
                    ChannelData::UInt8(a) => {
                        let data = a.values_slice_mut();
                        for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                            data[i + previous_index] = record[pos_byte_beg];
                        }
                    }
                    ChannelData::Int16(a) => {
                        let data = a.values_slice_mut();
                        if cn.endian.is_big() {
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                data[i + previous_index] = i16::from_be_bytes([
                                    record[pos_byte_beg],
                                    record[pos_byte_beg + 1],
                                ]);
                            }
                        } else {
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                data[i + previous_index] = i16::from_le_bytes([
                                    record[pos_byte_beg],
                                    record[pos_byte_beg + 1],
                                ]);
                            }
                        }
                    }
                    ChannelData::UInt16(a) => {
                        let data = a.values_slice_mut();
                        if cn.endian.is_big() {
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                data[i + previous_index] = u16::from_be_bytes([
                                    record[pos_byte_beg],
                                    record[pos_byte_beg + 1],
                                ]);
                            }
                        } else {
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                data[i + previous_index] = u16::from_le_bytes([
                                    record[pos_byte_beg],
                                    record[pos_byte_beg + 1],
                                ]);
                            }
                        }
                    }
                    ChannelData::Int32(a) => {
                        let data = a.values_slice_mut();
                        if n_bytes == 3 {
                            if cn.endian.is_big() {
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                    data[i + previous_index] = value
                                        .read_i24::<BigEndian>()
                                        .context("Could not read be i24")?;
                                }
                            } else {
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                    data[i + previous_index] = value
                                        .read_i24::<LittleEndian>()
                                        .context("Could not read le i24")?;
                                }
                            }
                        } else if cn.endian.is_big() {
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                data[i + previous_index] = i32::from_be_bytes([
                                    record[pos_byte_beg],
                                    record[pos_byte_beg + 1],
                                    record[pos_byte_beg + 2],
                                    record[pos_byte_beg + 3],
                                ]);
                            }
                        } else {
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                data[i + previous_index] = i32::from_le_bytes([
                                    record[pos_byte_beg],
                                    record[pos_byte_beg + 1],
                                    record[pos_byte_beg + 2],
                                    record[pos_byte_beg + 3],
                                ]);
                            }
                        }
                    }
                    ChannelData::UInt32(a) => {
                        let data = a.values_slice_mut();
                        if n_bytes == 3 {
                            if cn.endian.is_big() {
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                    data[i + previous_index] = value
                                        .read_u24::<BigEndian>()
                                        .context("Could not read be u24")?;
                                }
                            } else {
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                    data[i + previous_index] = value
                                        .read_u24::<LittleEndian>()
                                        .context("Could not read le u24")?;
                                }
                            }
                        } else if cn.endian.is_big() {
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                data[i + previous_index] = u32::from_be_bytes([
                                    record[pos_byte_beg],
                                    record[pos_byte_beg + 1],
                                    record[pos_byte_beg + 2],
                                    record[pos_byte_beg + 3],
                                ]);
                            }
                        } else {
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                data[i + previous_index] = u32::from_le_bytes([
                                    record[pos_byte_beg],
                                    record[pos_byte_beg + 1],
                                    record[pos_byte_beg + 2],
                                    record[pos_byte_beg + 3],
                                ]);
                            }
                        }
                    }
                    ChannelData::Float32(a) => {
                        let data = a.values_slice_mut();
                        if cn.endian.is_big() {
                            if n_bytes == 2 {
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    data[i + previous_index] = f16::from_be_bytes([
                                        record[pos_byte_beg],
                                        record[pos_byte_beg + 1],
                                    ])
                                    .to_f32();
                                }
                            } else {
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    data[i + previous_index] = f32::from_be_bytes([
                                        record[pos_byte_beg],
                                        record[pos_byte_beg + 1],
                                        record[pos_byte_beg + 2],
                                        record[pos_byte_beg + 3],
                                    ]);
                                }
                            }
                        } else if n_bytes == 2 {
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                data[i + previous_index] = f16::from_le_bytes([
                                    record[pos_byte_beg],
                                    record[pos_byte_beg + 1],
                                ])
                                .to_f32();
                            }
                        } else {
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                data[i + previous_index] = f32::from_le_bytes([
                                    record[pos_byte_beg],
                                    record[pos_byte_beg + 1],
                                    record[pos_byte_beg + 2],
                                    record[pos_byte_beg + 3],
                                ]);
                            }
                        }
                    }
                    ChannelData::Int64(a) => {
                        let data = a.values_slice_mut();
                        if cn.endian.is_big() {
                            if n_bytes == 8 {
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                    data[i + previous_index] =
                                        i64::from_be_bytes(value.try_into().unwrap());
                                }
                            } else if n_bytes == 7 {
                                let mut buf = [0u8; std::mem::size_of::<u64>()];
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    buf[1..8]
                                        .copy_from_slice(&record[pos_byte_beg..pos_byte_beg + 7]);
                                    data[i + previous_index] =
                                        ((u64::from_be_bytes(buf) as i64) << 8) >> 8;
                                }
                            } else if n_bytes == 6 {
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                    data[i + previous_index] = value
                                        .read_i48::<BigEndian>()
                                        .context("Could not read be i48")?;
                                }
                            } else if n_bytes == 5 {
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    let src = &record[pos_byte_beg..pos_byte_beg + 5];
                                    let raw = ((src[0] as u64) << 32)
                                        | ((src[1] as u64) << 24)
                                        | ((src[2] as u64) << 16)
                                        | ((src[3] as u64) << 8)
                                        | (src[4] as u64);
                                    data[i + previous_index] = ((raw as i64) << 24) >> 24;
                                }
                            }
                        } else if n_bytes == 8 {
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                data[i + previous_index] =
                                    i64::from_le_bytes(value.try_into().unwrap());
                            }
                        } else if n_bytes == 7 {
                            let mut buf = [0u8; std::mem::size_of::<u64>()];
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                buf[0..7].copy_from_slice(&record[pos_byte_beg..pos_byte_beg + 7]);
                                data[i + previous_index] =
                                    ((u64::from_le_bytes(buf) as i64) << 8) >> 8;
                            }
                        } else if n_bytes == 6 {
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                data[i + previous_index] = value
                                    .read_i48::<LittleEndian>()
                                    .context("Could not read le i48")?;
                            }
                        } else if n_bytes == 5 {
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                let src = &record[pos_byte_beg..pos_byte_beg + 5];
                                let raw = (src[0] as u64)
                                    | ((src[1] as u64) << 8)
                                    | ((src[2] as u64) << 16)
                                    | ((src[3] as u64) << 24)
                                    | ((src[4] as u64) << 32);
                                data[i + previous_index] = ((raw as i64) << 24) >> 24;
                            }
                        }
                    }
                    ChannelData::UInt64(a) => {
                        let data = a.values_slice_mut();
                        if cn.endian.is_big() {
                            if n_bytes == 8 {
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                    data[i + previous_index] =
                                        u64::from_be_bytes(value.try_into().unwrap());
                                }
                            } else if n_bytes == 7 {
                                let mut buf = [0u8; std::mem::size_of::<u64>()];
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    buf[1..8]
                                        .copy_from_slice(&record[pos_byte_beg..pos_byte_beg + 7]);
                                    data[i + previous_index] = u64::from_be_bytes(buf);
                                }
                            } else if n_bytes == 6 {
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                    data[i + previous_index] = value
                                        .read_u48::<BigEndian>()
                                        .context("Could not read be u48")?;
                                }
                            } else {
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    let src = &record[pos_byte_beg..pos_byte_beg + 5];
                                    data[i + previous_index] = ((src[0] as u64) << 32)
                                        | ((src[1] as u64) << 24)
                                        | ((src[2] as u64) << 16)
                                        | ((src[3] as u64) << 8)
                                        | (src[4] as u64);
                                }
                            }
                        } else if n_bytes == 8 {
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                data[i + previous_index] =
                                    u64::from_le_bytes(value.try_into().unwrap());
                            }
                        } else if n_bytes == 7 {
                            // little endian
                            let mut buf = [0u8; std::mem::size_of::<u64>()];
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                buf[0..7].copy_from_slice(&record[pos_byte_beg..pos_byte_beg + 7]);
                                data[i + previous_index] = u64::from_le_bytes(buf);
                            }
                        } else if n_bytes == 6 {
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                data[i + previous_index] = value
                                    .read_u48::<LittleEndian>()
                                    .context("Could not read le u48")?;
                            }
                        } else {
                            // n_bytes = 5
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                let src = &record[pos_byte_beg..pos_byte_beg + 5];
                                data[i + previous_index] = (src[0] as u64)
                                    | ((src[1] as u64) << 8)
                                    | ((src[2] as u64) << 16)
                                    | ((src[3] as u64) << 24)
                                    | ((src[4] as u64) << 32);
                            }
                        }
                    }
                    ChannelData::Float64(a) => {
                        let data = a.values_slice_mut();
                        if cn.endian.is_big() {
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                value = &record
                                    [pos_byte_beg..pos_byte_beg + std::mem::size_of::<f64>()];
                                data[i + previous_index] =
                                    f64::from_be_bytes(value.try_into().unwrap());
                            }
                        } else {
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                value = &record
                                    [pos_byte_beg..pos_byte_beg + std::mem::size_of::<f64>()];
                                data[i + previous_index] =
                                    f64::from_le_bytes(value.try_into().unwrap());
                            }
                        }
                    }
                    ChannelData::Complex32(a) => {
                        let data = a.values().values_slice_mut();
                        if n_bytes <= 4 {
                            // complex 16
                            let mut re_val: &[u8];
                            let mut im_val: &[u8];
                            if cn.endian.is_big() {
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    re_val = &record
                                        [pos_byte_beg..pos_byte_beg + std::mem::size_of::<f16>()];
                                    im_val = &record[pos_byte_beg + std::mem::size_of::<f16>()
                                        ..pos_byte_beg + 2 * std::mem::size_of::<f16>()];
                                    data[i * 2 + previous_index] = f16::from_be_bytes(
                                        re_val
                                            .try_into()
                                            .context("Could not read be real f16 complex")?,
                                    )
                                    .to_f32();
                                    data[i * 2 + 1 + previous_index] = f16::from_be_bytes(
                                        im_val
                                            .try_into()
                                            .context("Could not read be img f16 complex")?,
                                    )
                                    .to_f32();
                                }
                            } else {
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    re_val = &record
                                        [pos_byte_beg..pos_byte_beg + std::mem::size_of::<f16>()];
                                    im_val = &record[pos_byte_beg + std::mem::size_of::<f16>()
                                        ..pos_byte_beg + 2 * std::mem::size_of::<f16>()];
                                    data[i * 2 + previous_index] = f16::from_le_bytes(
                                        re_val
                                            .try_into()
                                            .context("Could not read le real f16 complex")?,
                                    )
                                    .to_f32();
                                    data[i * 2 + 1 + previous_index] = f16::from_le_bytes(
                                        im_val
                                            .try_into()
                                            .context("Could not read le img f16 complex")?,
                                    )
                                    .to_f32();
                                }
                            }
                        } else if n_bytes <= 8 {
                            // complex 32
                            let mut re_val: &[u8];
                            let mut im_val: &[u8];
                            if cn.endian.is_big() {
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    re_val = &record
                                        [pos_byte_beg..pos_byte_beg + std::mem::size_of::<f32>()];
                                    im_val = &record[pos_byte_beg + std::mem::size_of::<f32>()
                                        ..pos_byte_beg + 2 * std::mem::size_of::<f32>()];
                                    data[i * 2 + previous_index] = f32::from_be_bytes(
                                        re_val
                                            .try_into()
                                            .context("Could not read be real f32 complex")?,
                                    );
                                    data[i * 2 + 1 + previous_index] = f32::from_be_bytes(
                                        im_val
                                            .try_into()
                                            .context("Could not read be img f32 complex")?,
                                    );
                                }
                            } else {
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    re_val = &record
                                        [pos_byte_beg..pos_byte_beg + std::mem::size_of::<f32>()];
                                    im_val = &record[pos_byte_beg + std::mem::size_of::<f32>()
                                        ..pos_byte_beg + 2 * std::mem::size_of::<f32>()];
                                    data[i * 2 + previous_index] = f32::from_le_bytes(
                                        re_val
                                            .try_into()
                                            .context("Could not read le real f32 complex")?,
                                    );
                                    data[i * 2 + 1 + previous_index] = f32::from_le_bytes(
                                        im_val
                                            .try_into()
                                            .context("Could not read le img f32 complex")?,
                                    );
                                }
                            }
                        }
                    }
                    ChannelData::Complex64(a) => {
                        // complex 64
                        let mut re_val: &[u8];
                        let mut im_val: &[u8];
                        let data = a.values().values_slice_mut();
                        if cn.endian.is_big() {
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                re_val = &record
                                    [pos_byte_beg..pos_byte_beg + std::mem::size_of::<f64>()];
                                im_val = &record[pos_byte_beg + std::mem::size_of::<f64>()
                                    ..pos_byte_beg + 2 * std::mem::size_of::<f64>()];
                                data[i * 2 + previous_index] = f64::from_be_bytes(
                                    re_val
                                        .try_into()
                                        .context("Could not read be real f64 complex")?,
                                );
                                data[i * 2 + 1 + previous_index] = f64::from_be_bytes(
                                    im_val
                                        .try_into()
                                        .context("Could not read be img f64 complex")?,
                                );
                            }
                        } else {
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                re_val = &record
                                    [pos_byte_beg..pos_byte_beg + std::mem::size_of::<f64>()];
                                im_val = &record[pos_byte_beg + std::mem::size_of::<f64>()
                                    ..pos_byte_beg + 2 * std::mem::size_of::<f64>()];
                                data[i * 2 + previous_index] = f64::from_le_bytes(
                                    re_val
                                        .try_into()
                                        .context("Could not read le real f64 complex")?,
                                );
                                data[i * 2 + 1 + previous_index] = f64::from_le_bytes(
                                    im_val
                                        .try_into()
                                        .context("Could not read le img f64 complex")?,
                                );
                            }
                        }
                    }
                    ChannelData::Utf8(array) => {
                        if cn.block.cn_data_type == 6 {
                            // SBC ISO-8859-1 to be converted into UTF8
                            let mut decoder = WINDOWS_1252.new_decoder();
                            let mut dst = String::with_capacity(n_bytes);
                            for record in data_chunk.chunks_exact(record_length) {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                dst.clear();
                                let (_result, _size, _replacement) =
                                    decoder.decode_to_string(value, &mut dst, false);
                                let s = dst.split('\0').next().unwrap_or("");
                                array.append_value(s);
                            }
                        } else if cn.block.cn_data_type == 7 {
                            // 7: String UTF8
                            for record in data_chunk.chunks_exact(record_length) {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                let full = str::from_utf8(value).context("Found invalid UTF-8")?;
                                let s = full.split('\0').next().unwrap_or("");
                                array.append_value(s);
                            }
                        } else if cn.block.cn_data_type == 8 || cn.block.cn_data_type == 9 {
                            // 8 | 9 :String UTF16 to be converted into UTF8
                            let mut dst = String::with_capacity(n_bytes);
                            if cn.endian.is_big() {
                                let mut decoder = UTF_16BE.new_decoder();
                                for record in data_chunk.chunks_exact(record_length) {
                                    value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                    dst.clear();
                                    let (_result, _size, _replacement) =
                                        decoder.decode_to_string(value, &mut dst, false);
                                    let s = dst.split('\0').next().unwrap_or("");
                                    array.append_value(s);
                                }
                            } else {
                                let mut decoder = UTF_16LE.new_decoder();
                                for record in data_chunk.chunks_exact(record_length) {
                                    value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                    dst.clear();
                                    let (_result, _size, _replacement) =
                                        decoder.decode_to_string(value, &mut dst, false);
                                    let s = dst.split('\0').next().unwrap_or("");
                                    array.append_value(s);
                                }
                            }
                        } else if cn.block.cn_data_type == 17 {
                            // 17: Unicode with BOM
                            let mut dst = String::with_capacity(n_bytes);
                            for record in data_chunk.chunks_exact(record_length) {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                // identifies BOM
                                let bom = Bom::from(value);
                                let mut decoder = match bom {
                                    Bom::Utf8 => UTF_8.new_decoder(),
                                    Bom::Utf16Be => UTF_16BE.new_decoder(),
                                    Bom::Utf16Le => UTF_16LE.new_decoder(),
                                    Bom::Gb18030 => GB18030.new_decoder(),
                                    _ => {
                                        bail!("not implemented BOM type");
                                    }
                                };
                                dst.clear();
                                let (_result, _size, _replacement) =
                                    decoder.decode_to_string(value, &mut dst, false);
                                let s = dst.split('\0').next().unwrap_or("");
                                array.append_value(s);
                            }
                        }
                    }
                    ChannelData::VariableSizeByteArray(array) => {
                        for record in data_chunk.chunks_exact(record_length) {
                            if pos_byte_beg + n_bytes <= record.len() {
                                array.append_value(&record[pos_byte_beg..pos_byte_beg + n_bytes]);
                            } else {
                                array.append_null();
                            }
                        }
                    }
                    ChannelData::FixedSizeByteArray(a) => {
                        for record in data_chunk.chunks_exact(record_length) {
                            if pos_byte_beg + n_bytes <= record.len() {
                                a.append_value(&record[pos_byte_beg..pos_byte_beg + n_bytes])
                                    .context("failed appending new value")?;
                            } else {
                                a.append_null();
                            }
                        }
                    }
                    ChannelData::ArrayDInt8(a) => {
                        let data = a.values_slice_mut();
                        let row_bytes = list_size * std::mem::size_of::<i8>();
                        for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                            let dst_start = (i + previous_index) * list_size;
                            let src = &record[pos_byte_beg..pos_byte_beg + row_bytes];
                            for (d, &s) in
                                data[dst_start..dst_start + list_size].iter_mut().zip(src)
                            {
                                *d = s as i8;
                            }
                        }
                    }
                    ChannelData::ArrayDUInt8(a) => {
                        let data = a.values_slice_mut();
                        let row_bytes = list_size * std::mem::size_of::<u8>();
                        for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                            let dst_start = (i + previous_index) * list_size;
                            data[dst_start..dst_start + list_size]
                                .copy_from_slice(&record[pos_byte_beg..pos_byte_beg + row_bytes]);
                        }
                    }
                    ChannelData::ArrayDInt16(a) => {
                        let data = a.values_slice_mut();
                        let row_bytes = list_size * std::mem::size_of::<i16>();
                        if cn.endian.is_big() {
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                let dst_start = (i + previous_index) * list_size;
                                Cursor::new(&record[pos_byte_beg..pos_byte_beg + row_bytes])
                                    .read_i16_into::<BigEndian>(
                                        &mut data[dst_start..dst_start + list_size],
                                    )
                                    .context("Could not read be i16 array")?;
                            }
                        } else {
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                let dst_start = (i + previous_index) * list_size;
                                Cursor::new(&record[pos_byte_beg..pos_byte_beg + row_bytes])
                                    .read_i16_into::<LittleEndian>(
                                        &mut data[dst_start..dst_start + list_size],
                                    )
                                    .context("Could not read le i16 array")?;
                            }
                        }
                    }
                    ChannelData::ArrayDUInt16(a) => {
                        let data = a.values_slice_mut();
                        let row_bytes = list_size * std::mem::size_of::<u16>();
                        if cn.endian.is_big() {
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                let dst_start = (i + previous_index) * list_size;
                                Cursor::new(&record[pos_byte_beg..pos_byte_beg + row_bytes])
                                    .read_u16_into::<BigEndian>(
                                        &mut data[dst_start..dst_start + list_size],
                                    )
                                    .context("Could not read be u16 array")?;
                            }
                        } else {
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                let dst_start = (i + previous_index) * list_size;
                                Cursor::new(&record[pos_byte_beg..pos_byte_beg + row_bytes])
                                    .read_u16_into::<LittleEndian>(
                                        &mut data[dst_start..dst_start + list_size],
                                    )
                                    .context("Could not read le u16 array")?;
                            }
                        }
                    }
                    ChannelData::ArrayDInt32(a) => {
                        let data = a.values_slice_mut();
                        if cn.endian.is_big() {
                            if n_bytes <= 3 {
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    for j in 0..list_size {
                                        value = &record[pos_byte_beg + j * n_bytes
                                            ..pos_byte_beg + (j + 1) * n_bytes];
                                        data[(i + previous_index) * list_size + j] = value
                                            .read_i24::<BigEndian>()
                                            .context("Could not read be i24 array")?;
                                    }
                                }
                            } else {
                                let row_bytes = list_size * std::mem::size_of::<i32>();
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    let dst_start = (i + previous_index) * list_size;
                                    Cursor::new(&record[pos_byte_beg..pos_byte_beg + row_bytes])
                                        .read_i32_into::<BigEndian>(
                                            &mut data[dst_start..dst_start + list_size],
                                        )
                                        .context("Could not read be i32 array")?;
                                }
                            }
                        } else if n_bytes <= 3 {
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                for j in 0..list_size {
                                    value = &record[pos_byte_beg + j * n_bytes
                                        ..pos_byte_beg + (j + 1) * n_bytes];
                                    data[(i + previous_index) * list_size + j] = value
                                        .read_i24::<LittleEndian>()
                                        .context("Could not read le i24 array")?;
                                }
                            }
                        } else {
                            let row_bytes = list_size * std::mem::size_of::<i32>();
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                let dst_start = (i + previous_index) * list_size;
                                Cursor::new(&record[pos_byte_beg..pos_byte_beg + row_bytes])
                                    .read_i32_into::<LittleEndian>(
                                        &mut data[dst_start..dst_start + list_size],
                                    )
                                    .context("Could not read le i32 array")?;
                            }
                        }
                    }
                    ChannelData::ArrayDUInt32(a) => {
                        let data = a.values_slice_mut();
                        if cn.endian.is_big() {
                            if n_bytes <= 3 {
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    for j in 0..list_size {
                                        value = &record[pos_byte_beg + j * n_bytes
                                            ..pos_byte_beg + (j + 1) * n_bytes];
                                        data[(i + previous_index) * list_size + j] = value
                                            .read_u24::<BigEndian>()
                                            .context("Could not read be u24 array")?;
                                    }
                                }
                            } else {
                                let row_bytes = list_size * std::mem::size_of::<u32>();
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    let dst_start = (i + previous_index) * list_size;
                                    Cursor::new(&record[pos_byte_beg..pos_byte_beg + row_bytes])
                                        .read_u32_into::<BigEndian>(
                                            &mut data[dst_start..dst_start + list_size],
                                        )
                                        .context("Could not read be u32 array")?;
                                }
                            }
                        } else if n_bytes <= 3 {
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                for j in 0..list_size {
                                    value = &record[pos_byte_beg + j * n_bytes
                                        ..pos_byte_beg + (j + 1) * n_bytes];
                                    data[(i + previous_index) * list_size + j] = value
                                        .read_u24::<LittleEndian>()
                                        .context("Could not read le u24 array")?;
                                }
                            }
                        } else {
                            let row_bytes = list_size * std::mem::size_of::<u32>();
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                let dst_start = (i + previous_index) * list_size;
                                Cursor::new(&record[pos_byte_beg..pos_byte_beg + row_bytes])
                                    .read_u32_into::<LittleEndian>(
                                        &mut data[dst_start..dst_start + list_size],
                                    )
                                    .context("Could not read le u32 array")?;
                            }
                        }
                    }
                    ChannelData::ArrayDFloat32(a) => {
                        let data = a.values_slice_mut();
                        if cn.endian.is_big() {
                            if n_bytes <= 2 {
                                let row_bytes = list_size * std::mem::size_of::<f16>();
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    let dst_start = (i + previous_index) * list_size;
                                    let chunks = as_exact_chunks::<{ std::mem::size_of::<f16>() }>(
                                        &record[pos_byte_beg..pos_byte_beg + row_bytes],
                                        list_size,
                                        "Could not read be f16 array",
                                    )?;
                                    for (dst, src) in data[dst_start..dst_start + list_size]
                                        .iter_mut()
                                        .zip(chunks)
                                    {
                                        *dst = f16::from_be_bytes(*src).to_f32();
                                    }
                                }
                            } else {
                                let row_bytes = list_size * std::mem::size_of::<f32>();
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    let dst_start = (i + previous_index) * list_size;
                                    let chunks = as_exact_chunks::<{ std::mem::size_of::<f32>() }>(
                                        &record[pos_byte_beg..pos_byte_beg + row_bytes],
                                        list_size,
                                        "Could not read be f32 array",
                                    )?;
                                    for (dst, src) in data[dst_start..dst_start + list_size]
                                        .iter_mut()
                                        .zip(chunks)
                                    {
                                        *dst = f32::from_be_bytes(*src);
                                    }
                                }
                            }
                        } else if n_bytes <= 2 {
                            let row_bytes = list_size * std::mem::size_of::<f16>();
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                let dst_start = (i + previous_index) * list_size;
                                let chunks = as_exact_chunks::<{ std::mem::size_of::<f16>() }>(
                                    &record[pos_byte_beg..pos_byte_beg + row_bytes],
                                    list_size,
                                    "Could not read le f16 array",
                                )?;
                                for (dst, src) in data[dst_start..dst_start + list_size]
                                    .iter_mut()
                                    .zip(chunks)
                                {
                                    *dst = f16::from_le_bytes(*src).to_f32();
                                }
                            }
                        } else {
                            let row_bytes = list_size * std::mem::size_of::<f32>();
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                let dst_start = (i + previous_index) * list_size;
                                let chunks = as_exact_chunks::<{ std::mem::size_of::<f32>() }>(
                                    &record[pos_byte_beg..pos_byte_beg + row_bytes],
                                    list_size,
                                    "Could not read le f32 array",
                                )?;
                                for (dst, src) in data[dst_start..dst_start + list_size]
                                    .iter_mut()
                                    .zip(chunks)
                                {
                                    *dst = f32::from_le_bytes(*src);
                                }
                            }
                        }
                    }
                    ChannelData::ArrayDInt64(a) => {
                        let data = a.values_slice_mut();
                        if cn.endian.is_big() {
                            if n_bytes == 8 {
                                let row_bytes = list_size * std::mem::size_of::<i64>();
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    let dst_start = (i + previous_index) * list_size;
                                    Cursor::new(&record[pos_byte_beg..pos_byte_beg + row_bytes])
                                        .read_i64_into::<BigEndian>(
                                            &mut data[dst_start..dst_start + list_size],
                                        )
                                        .context("Could not read be i64 array")?;
                                }
                            } else if n_bytes == 7 {
                                let mut buf = [0u8; std::mem::size_of::<u64>()];
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    for j in 0..list_size {
                                        buf[1..8].copy_from_slice(
                                            &record[pos_byte_beg + j * n_bytes
                                                ..pos_byte_beg + (j + 1) * n_bytes],
                                        );
                                        data[(i + previous_index) * list_size + j] =
                                            ((u64::from_be_bytes(buf) as i64) << 8) >> 8;
                                    }
                                }
                            } else if n_bytes == 6 {
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    for j in 0..list_size {
                                        value = &record[pos_byte_beg + j * n_bytes
                                            ..pos_byte_beg + (j + 1) * n_bytes];
                                        data[(i + previous_index) * list_size + j] = value
                                            .read_i48::<BigEndian>()
                                            .context("Could not read be i48 array")?;
                                    }
                                }
                            } else if n_bytes == 5 {
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    for j in 0..list_size {
                                        let src = &record[pos_byte_beg + j * n_bytes
                                            ..pos_byte_beg + (j + 1) * n_bytes];
                                        let raw = ((src[0] as u64) << 32)
                                            | ((src[1] as u64) << 24)
                                            | ((src[2] as u64) << 16)
                                            | ((src[3] as u64) << 8)
                                            | (src[4] as u64);
                                        data[(i + previous_index) * list_size + j] =
                                            ((raw as i64) << 24) >> 24;
                                    }
                                }
                            }
                        } else if n_bytes == 8 {
                            let row_bytes = list_size * std::mem::size_of::<i64>();
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                let dst_start = (i + previous_index) * list_size;
                                Cursor::new(&record[pos_byte_beg..pos_byte_beg + row_bytes])
                                    .read_i64_into::<LittleEndian>(
                                        &mut data[dst_start..dst_start + list_size],
                                    )
                                    .context("Could not read le i64 array")?;
                            }
                        } else if n_bytes == 7 {
                            let mut buf = [0u8; std::mem::size_of::<u64>()];
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                for j in 0..list_size {
                                    buf[0..7].copy_from_slice(
                                        &record[pos_byte_beg + j * n_bytes
                                            ..pos_byte_beg + (j + 1) * n_bytes],
                                    );
                                    data[(i + previous_index) * list_size + j] =
                                        ((u64::from_le_bytes(buf) as i64) << 8) >> 8;
                                }
                            }
                        } else if n_bytes == 6 {
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                for j in 0..list_size {
                                    value = &record[pos_byte_beg + j * n_bytes
                                        ..pos_byte_beg + (j + 1) * n_bytes];
                                    data[(i + previous_index) * list_size + j] = value
                                        .read_i48::<LittleEndian>()
                                        .context("Could not read le i48 array")?;
                                }
                            }
                        } else if n_bytes == 5 {
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                for j in 0..list_size {
                                    let src = &record[pos_byte_beg + j * n_bytes
                                        ..pos_byte_beg + (j + 1) * n_bytes];
                                    let raw = (src[0] as u64)
                                        | ((src[1] as u64) << 8)
                                        | ((src[2] as u64) << 16)
                                        | ((src[3] as u64) << 24)
                                        | ((src[4] as u64) << 32);
                                    data[(i + previous_index) * list_size + j] =
                                        ((raw as i64) << 24) >> 24;
                                }
                            }
                        }
                    }
                    ChannelData::ArrayDUInt64(a) => {
                        let data = a.values_slice_mut();
                        if cn.endian.is_big() {
                            if n_bytes == 8 {
                                let row_bytes = list_size * std::mem::size_of::<u64>();
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    let dst_start = (i + previous_index) * list_size;
                                    Cursor::new(&record[pos_byte_beg..pos_byte_beg + row_bytes])
                                        .read_u64_into::<BigEndian>(
                                            &mut data[dst_start..dst_start + list_size],
                                        )
                                        .context("Could not read be u64 array")?;
                                }
                            } else if n_bytes == 7 {
                                let mut buf = [0u8; std::mem::size_of::<u64>()];
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    for j in 0..list_size {
                                        buf[1..8].copy_from_slice(
                                            &record[pos_byte_beg + j * n_bytes
                                                ..pos_byte_beg + (j + 1) * n_bytes],
                                        );
                                        data[(i + previous_index) * list_size + j] =
                                            u64::from_be_bytes(buf);
                                    }
                                }
                            } else if n_bytes == 6 {
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    for j in 0..list_size {
                                        value = &record[pos_byte_beg + j * n_bytes
                                            ..pos_byte_beg + (j + 1) * n_bytes];
                                        data[(i + previous_index) * list_size + j] = value
                                            .read_u48::<BigEndian>()
                                            .context("Could not read be u48 array")?;
                                    }
                                }
                            } else if n_bytes == 5 {
                                for (i, record) in
                                    data_chunk.chunks_exact(record_length).enumerate()
                                {
                                    for j in 0..list_size {
                                        let src = &record[pos_byte_beg + j * n_bytes
                                            ..pos_byte_beg + (j + 1) * n_bytes];
                                        data[(i + previous_index) * list_size + j] =
                                            ((src[0] as u64) << 32)
                                                | ((src[1] as u64) << 24)
                                                | ((src[2] as u64) << 16)
                                                | ((src[3] as u64) << 8)
                                                | (src[4] as u64);
                                    }
                                }
                            }
                        } else if n_bytes == 8 {
                            let row_bytes = list_size * std::mem::size_of::<u64>();
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                let dst_start = (i + previous_index) * list_size;
                                Cursor::new(&record[pos_byte_beg..pos_byte_beg + row_bytes])
                                    .read_u64_into::<LittleEndian>(
                                        &mut data[dst_start..dst_start + list_size],
                                    )
                                    .context("Could not read le u64 array")?;
                            }
                        } else if n_bytes == 7 {
                            let mut buf = [0u8; std::mem::size_of::<u64>()];
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                for j in 0..list_size {
                                    buf[0..7].copy_from_slice(
                                        &record[pos_byte_beg + j * n_bytes
                                            ..pos_byte_beg + (j + 1) * n_bytes],
                                    );
                                    data[(i + previous_index) * list_size + j] =
                                        u64::from_le_bytes(buf);
                                }
                            }
                        } else if n_bytes == 6 {
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                for j in 0..list_size {
                                    value = &record[pos_byte_beg + j * n_bytes
                                        ..pos_byte_beg + (j + 1) * n_bytes];
                                    data[(i + previous_index) * list_size + j] = value
                                        .read_u48::<LittleEndian>()
                                        .context("Could not read le u48 array")?;
                                }
                            }
                        } else {
                            // n_bytes = 5
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                for j in 0..list_size {
                                    let src = &record[pos_byte_beg + j * n_bytes
                                        ..pos_byte_beg + (j + 1) * n_bytes];
                                    data[(i + previous_index) * list_size + j] = (src[0] as u64)
                                        | ((src[1] as u64) << 8)
                                        | ((src[2] as u64) << 16)
                                        | ((src[3] as u64) << 24)
                                        | ((src[4] as u64) << 32);
                                }
                            }
                        }
                    }
                    ChannelData::ArrayDFloat64(a) => {
                        let data = a.values_slice_mut();
                        let row_bytes = list_size * std::mem::size_of::<f64>();
                        if cn.endian.is_big() {
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                let dst_start = (i + previous_index) * list_size;
                                let chunks = as_exact_chunks::<{ std::mem::size_of::<f64>() }>(
                                    &record[pos_byte_beg..pos_byte_beg + row_bytes],
                                    list_size,
                                    "Could not read be f64 array",
                                )?;
                                for (dst, src) in data[dst_start..dst_start + list_size]
                                    .iter_mut()
                                    .zip(chunks)
                                {
                                    *dst = f64::from_be_bytes(*src);
                                }
                            }
                        } else {
                            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                                let dst_start = (i + previous_index) * list_size;
                                let chunks = as_exact_chunks::<{ std::mem::size_of::<f64>() }>(
                                    &record[pos_byte_beg..pos_byte_beg + row_bytes],
                                    list_size,
                                    "Could not read le f64 array",
                                )?;
                                for (dst, src) in data[dst_start..dst_start + list_size]
                                    .iter_mut()
                                    .zip(chunks)
                                {
                                    *dst = f64::from_le_bytes(*src);
                                }
                            }
                        }
                    }
                    ChannelData::Union(_) => {} // Union channels are constructed post-read
                }
                // VLSC channels: offsets were read above, now mark for VD block processing
                if cn.block.cn_type == 7 {
                    let c_vlsd_channel = Arc::clone(&vlsd_channels);
                    let mut vlsd_channel = c_vlsd_channel
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    vlsd_channel.push((cn.block.cn_type, *rec_pos));
                }
            } else {
                let mut is_dynamic =
                    cn.block.cn_type == 1 || (cn.block.cn_flags & CN_F_DATA_STREAM_MODE != 0);
                if !is_dynamic && let Some(composition) = &cn.composition {
                    match &composition.block {
                        Compo::CA(ca) if ca.ca_storage == 5 => is_dynamic = true,
                        Compo::DS(_) => is_dynamic = true,
                        Compo::CL(_) => is_dynamic = true, // Channel list - variable length
                        Compo::CV(_) => {
                            // Channel Variant - option channels parsed separately
                            // discriminator determines which option is active
                        }
                        Compo::CU(_) => {
                            // Channel Union - all members share same bytes
                            // Members parsed as separate channels
                        }
                        _ => {}
                    }
                }

                if is_dynamic {
                    // cn_type == 1: VLSD - SD Block attached as data block
                    // cn_type == 7: VLSC - VD Block with (time, size, offset) triplet
                    // CN_F_DATA_STREAM_MODE: Data Stream - DS Block points to data
                    // ca_storage == 5: Dynamic array storage
                    let c_vlsd_channel = Arc::clone(&vlsd_channels);
                    let mut vlsd_channel = c_vlsd_channel
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    vlsd_channel.push((cn.block.cn_type, *rec_pos));
                }
            }
            // Other channel types : virtual channels cn_type 3 & 6 are handled at initialisation
            if record_with_invalid_data {
                // invalidation bits to store in bitmap.
                if let Some((Some(mask), invalid_byte_position, invalid_byte_mask)) =
                    &mut cn.invalid_mask
                {
                    for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                        mask.set_bit(
                            i + previous_index,
                            (*invalid_byte_mask & record[*invalid_byte_position]) == 0,
                        );
                    }
                };
            }
            Ok(())
        })
        .with_context(|| {
            format!("Parallel channels bytes reading failed for channel {channels:?}")
        })?;

    let lock = vlsd_channels
        .lock()
        .expect("Could not get lock from vlsd channel arc vec");
    Ok(lock.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_days_since_unix_epoch_canopen_epoch() {
        // CANopen epoch is 1984-01-01; must be exactly 5113 days after Unix epoch
        assert_eq!(days_since_unix_epoch(1984, 1, 1), 5113);
    }

    #[test]
    fn test_days_since_unix_epoch_unix_epoch() {
        assert_eq!(days_since_unix_epoch(1970, 1, 1), 0);
    }

    #[test]
    fn test_days_since_unix_epoch_leap_year() {
        // 1972 is a leap year; 1972-03-01 = 365 + 365 + 31 + 29 = 790 days after epoch
        assert_eq!(days_since_unix_epoch(1972, 3, 1), 790);
    }

    #[test]
    fn test_canopen_time_epoch() {
        // days=0, ms=0 → CANopen epoch 1984-01-01 = +441_763_200_000 ms in Unix time
        let bytes = [0u8; 6];
        assert_eq!(canopen_time_to_unix_ms(&bytes), 441_763_200_000);
    }

    #[test]
    fn test_canopen_time_1984_01_02_midnight() {
        // days=1 since 1984-01-01, ms=0 → one day after CANopen epoch
        let mut bytes = [0u8; 6];
        bytes[4..6].copy_from_slice(&1u16.to_le_bytes());
        let expected = 441_763_200_000 + 86_400_000;
        assert_eq!(canopen_time_to_unix_ms(&bytes), expected);
    }

    #[test]
    fn test_canopen_date_1984_01_01_midnight() {
        // year=0 (→1984), month=1, day=1, hour=0, min=0, ms=0
        // 1984-01-01 00:00:00 UTC = 441_763_200_000 ms after Unix epoch
        let mut bytes = [0u8; 7];
        bytes[4] = 1; // day
        bytes[5] = 1; // month
        bytes[6] = 0; // year - 1984 = 0
        assert_eq!(canopen_date_to_unix_ms(&bytes), 441_763_200_000);
    }

    #[test]
    fn test_canopen_date_known_value() {
        // 2000-01-01 00:00:00.000 — well-known Y2K date
        // year byte = 2000-1984 = 16, month=1, day=1
        let mut bytes = [0u8; 7];
        bytes[4] = 1; // day
        bytes[5] = 1; // month
        bytes[6] = 16; // year = 1984 + 16 = 2000
        // Days from Unix epoch to 2000-01-01 = 10957
        let expected = 10957i64 * 86_400_000;
        assert_eq!(canopen_date_to_unix_ms(&bytes), expected);
    }

    #[test]
    fn test_be_7byte_uint64_and_array_uint64() {
        use crate::data_holder::channel_data::data_type_init;
        use crate::data_holder::tensor_arrow::Order;
        use crate::mdfinfo::mdfinfo4::Endianness;

        // 1) Scalar 7-byte BE UInt64: 0x01_02_03_04_05_06_07
        let mut cn = Cn4 {
            n_bytes: 7,
            endian: Endianness::Big,
            list_size: 1,
            shape: (vec![1], Order::RowMajor),
            pos_byte_beg: 0,
            should_read: true,
            data: data_type_init(0, 1, 7, 1, 1).unwrap(),
            ..Default::default()
        };
        cn.block.cn_data_type = 1; // UInt BE
        cn.block.cn_bit_count = 56;
        cn.data = cn.data.zeros(0, 2, 7, (vec![1], Order::RowMajor)).unwrap();
        let raw_bytes = [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, // rec 0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, // rec 1 -> 256
        ];
        read_one_channel_array(&raw_bytes, &mut cn, 2).unwrap();
        if let ChannelData::UInt64(arr) = &cn.data {
            assert_eq!(arr.values_slice(), &[0x0001_0203_0405_0607u64, 256u64]);
        } else {
            panic!("expected UInt64");
        }

        // Also test via read_channels_from_bytes
        cn.data = cn.data.zeros(0, 2, 7, (vec![1], Order::RowMajor)).unwrap();
        let mut channels = CnType::default();
        channels.insert(0, cn);
        read_channels_from_bytes(&raw_bytes, &mut channels, 7, 0, false).unwrap();
        if let ChannelData::UInt64(arr) = &channels[&0].data {
            assert_eq!(arr.values_slice(), &[0x0001_0203_0405_0607u64, 256u64]);
        } else {
            panic!("expected UInt64");
        }

        // 2) ArrayDUInt64 8-byte BE
        let shape = (vec![2, 2], Order::RowMajor);
        let mut cn_arr = Cn4 {
            n_bytes: 8,
            endian: Endianness::Big,
            list_size: 2,
            shape: shape.clone(),
            pos_byte_beg: 0,
            should_read: true,
            data: data_type_init(0, 1, 8, 2, 1).unwrap(),
            ..Default::default()
        };
        cn_arr.block.cn_data_type = 1; // UInt BE
        cn_arr.block.cn_bit_count = 64;
        cn_arr.data = cn_arr.data.zeros(0, 2, 8, shape).unwrap();
        let mut raw_arr = Vec::new();
        for v in [10u64, 20, 30, 40] {
            raw_arr.extend_from_slice(&v.to_be_bytes());
        }
        let mut channels_arr = CnType::default();
        channels_arr.insert(0, cn_arr);
        read_channels_from_bytes(&raw_arr, &mut channels_arr, 16, 0, false).unwrap();
        if let ChannelData::ArrayDUInt64(arr) = &channels_arr[&0].data {
            assert_eq!(arr.values_slice(), &[10u64, 20, 30, 40]);
        } else {
            panic!("expected ArrayDUInt64");
        }
    }

    #[test]
    fn test_remainder_and_truncated_chunks_rejected() {
        use crate::data_holder::channel_data::data_type_init;
        use crate::data_holder::tensor_arrow::Order;
        use crate::mdfinfo::mdfinfo4::Endianness;

        let mut cn = Cn4 {
            n_bytes: 8,
            endian: Endianness::Little,
            list_size: 1,
            shape: (vec![1], Order::RowMajor),
            pos_byte_beg: 0,
            should_read: true,
            data: data_type_init(0, 4, 8, 1, 1).unwrap(),
            ..Default::default()
        };
        cn.block.cn_data_type = 4; // Float LE
        cn.block.cn_bit_count = 64;
        cn.data = cn.data.zeros(0, 2, 8, (vec![1], Order::RowMajor)).unwrap();

        // 1) Non-zero remainder (17 bytes for 2 f64s) must return an error
        let bytes_with_remainder = [0u8; 17];
        assert!(read_one_channel_array(&bytes_with_remainder, &mut cn, 2).is_err());

        // 2) Truncated buffer (8 bytes for 2 f64s) must return an error
        let truncated_bytes = [0u8; 8];
        assert!(read_one_channel_array(&truncated_bytes, &mut cn, 2).is_err());

        // 3) read_channels_from_bytes with remainder bytes (not a multiple of record_length) must return an error
        let mut channels = CnType::default();
        channels.insert(0, cn);
        assert!(
            read_channels_from_bytes(&bytes_with_remainder, &mut channels, 8, 0, false).is_err()
        );
    }
}
