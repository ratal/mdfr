//! this module implements low level data reading for mdf4 files.
use crate::mdfinfo::mdfinfo4::{CN_F_DATA_STREAM_MODE, Cn4, CnType, Compo};
use anyhow::{Context, Error, Ok, Result, bail};
use encoding_rs::{GB18030, UTF_8, UTF_16BE, UTF_16LE, WINDOWS_1252};
use half::f16;
use rayon::prelude::*;
use std::str;
use std::sync::{Arc, Mutex};
use unicode_bom::Bom;

use crate::data_holder::channel_builder::ChannelBuilder;

/// CANopen Time (6 bytes) → ms since Unix epoch.
/// Layout: bytes 0-3 = ms-within-day [27:0]; bytes 4-5 = days since 1984-01-01.
pub fn canopen_time_to_unix_ms(bytes: &[u8]) -> i64 {
    let ms = (u32::from_le_bytes(bytes[0..4].try_into().unwrap()) & 0x0FFF_FFFF) as i64;
    let days = u16::from_le_bytes(bytes[4..6].try_into().unwrap()) as i64;
    // CANopen epoch (1984-01-01) is 5113 days = 441_763_200_000 ms after Unix epoch
    441_763_200_000 + days * 86_400_000 + ms
}

/// CANopen Date (7 bytes) → ms since Unix epoch.
/// Layout: b[0-1] ms-in-minute [15:0]; b[2] min [5:0]; b[3] hour [4:0];
/// b[4] day [4:0] (1-based); b[5] month [3:0] (1-based); b[6] year-1984 [6:0].
pub fn canopen_date_to_unix_ms(bytes: &[u8]) -> i64 {
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
pub fn days_since_unix_epoch(year: i64, month: i64, day: i64) -> i64 {
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

// ─── Direct Scalar Record Readers ──────────────────────────────────────────

#[inline(always)]
pub fn read_i16_at(record: &[u8], pos: usize, is_big: bool) -> i16 {
    let bytes = [record[pos], record[pos + 1]];
    if is_big {
        i16::from_be_bytes(bytes)
    } else {
        i16::from_le_bytes(bytes)
    }
}

#[inline(always)]
pub fn read_u16_at(record: &[u8], pos: usize, is_big: bool) -> u16 {
    let bytes = [record[pos], record[pos + 1]];
    if is_big {
        u16::from_be_bytes(bytes)
    } else {
        u16::from_le_bytes(bytes)
    }
}

#[inline(always)]
pub fn read_i32_at(record: &[u8], pos: usize, n_bytes: usize, is_big: bool) -> i32 {
    if n_bytes == 4 {
        let bytes = [
            record[pos],
            record[pos + 1],
            record[pos + 2],
            record[pos + 3],
        ];
        if is_big {
            i32::from_be_bytes(bytes)
        } else {
            i32::from_le_bytes(bytes)
        }
    } else {
        // 3 bytes (i24)
        if is_big {
            ((record[pos] as i8 as i32) << 16)
                | ((record[pos + 1] as i32) << 8)
                | (record[pos + 2] as i32)
        } else {
            ((record[pos + 2] as i8 as i32) << 16)
                | ((record[pos + 1] as i32) << 8)
                | (record[pos] as i32)
        }
    }
}

#[inline(always)]
pub fn read_u32_at(record: &[u8], pos: usize, n_bytes: usize, is_big: bool) -> u32 {
    if n_bytes == 4 {
        let bytes = [
            record[pos],
            record[pos + 1],
            record[pos + 2],
            record[pos + 3],
        ];
        if is_big {
            u32::from_be_bytes(bytes)
        } else {
            u32::from_le_bytes(bytes)
        }
    } else {
        // 3 bytes (u24)
        if is_big {
            ((record[pos] as u32) << 16)
                | ((record[pos + 1] as u32) << 8)
                | (record[pos + 2] as u32)
        } else {
            ((record[pos + 2] as u32) << 16)
                | ((record[pos + 1] as u32) << 8)
                | (record[pos] as u32)
        }
    }
}

#[inline(always)]
pub fn read_f32_at(record: &[u8], pos: usize, n_bytes: usize, is_big: bool) -> f32 {
    if n_bytes == 2 {
        let bytes = [record[pos], record[pos + 1]];
        if is_big {
            f16::from_be_bytes(bytes).to_f32()
        } else {
            f16::from_le_bytes(bytes).to_f32()
        }
    } else {
        let bytes = [
            record[pos],
            record[pos + 1],
            record[pos + 2],
            record[pos + 3],
        ];
        if is_big {
            f32::from_be_bytes(bytes)
        } else {
            f32::from_le_bytes(bytes)
        }
    }
}

#[inline(always)]
pub fn read_f64_at(record: &[u8], pos: usize, is_big: bool) -> f64 {
    let bytes: [u8; 8] = record[pos..pos + 8].try_into().unwrap();
    if is_big {
        f64::from_be_bytes(bytes)
    } else {
        f64::from_le_bytes(bytes)
    }
}

#[inline(always)]
pub fn read_i64_at(record: &[u8], pos: usize, n_bytes: usize, is_big: bool) -> i64 {
    match n_bytes {
        8 => {
            let bytes: [u8; 8] = record[pos..pos + 8].try_into().unwrap();
            if is_big {
                i64::from_be_bytes(bytes)
            } else {
                i64::from_le_bytes(bytes)
            }
        }
        7 => {
            let mut buf = [0u8; 8];
            if is_big {
                buf[1..8].copy_from_slice(&record[pos..pos + 7]);
                ((u64::from_be_bytes(buf) as i64) << 8) >> 8
            } else {
                buf[0..7].copy_from_slice(&record[pos..pos + 7]);
                ((u64::from_le_bytes(buf) as i64) << 8) >> 8
            }
        }
        6 => {
            if is_big {
                let raw = ((record[pos] as u64) << 40)
                    | ((record[pos + 1] as u64) << 32)
                    | ((record[pos + 2] as u64) << 24)
                    | ((record[pos + 3] as u64) << 16)
                    | ((record[pos + 4] as u64) << 8)
                    | (record[pos + 5] as u64);
                ((raw as i64) << 16) >> 16
            } else {
                let raw = (record[pos] as u64)
                    | ((record[pos + 1] as u64) << 8)
                    | ((record[pos + 2] as u64) << 16)
                    | ((record[pos + 3] as u64) << 24)
                    | ((record[pos + 4] as u64) << 32)
                    | ((record[pos + 5] as u64) << 40);
                ((raw as i64) << 16) >> 16
            }
        }
        5 => {
            if is_big {
                let raw = ((record[pos] as u64) << 32)
                    | ((record[pos + 1] as u64) << 24)
                    | ((record[pos + 2] as u64) << 16)
                    | ((record[pos + 3] as u64) << 8)
                    | (record[pos + 4] as u64);
                ((raw as i64) << 24) >> 24
            } else {
                let raw = (record[pos] as u64)
                    | ((record[pos + 1] as u64) << 8)
                    | ((record[pos + 2] as u64) << 16)
                    | ((record[pos + 3] as u64) << 24)
                    | ((record[pos + 4] as u64) << 32);
                ((raw as i64) << 24) >> 24
            }
        }
        _ => unreachable!("Unsupported i64 width: {n_bytes}"),
    }
}

#[inline(always)]
pub fn read_u64_at(record: &[u8], pos: usize, n_bytes: usize, is_big: bool) -> u64 {
    match n_bytes {
        8 => {
            let bytes: [u8; 8] = record[pos..pos + 8].try_into().unwrap();
            if is_big {
                u64::from_be_bytes(bytes)
            } else {
                u64::from_le_bytes(bytes)
            }
        }
        7 => {
            let mut buf = [0u8; 8];
            if is_big {
                buf[1..8].copy_from_slice(&record[pos..pos + 7]);
                u64::from_be_bytes(buf)
            } else {
                buf[0..7].copy_from_slice(&record[pos..pos + 7]);
                u64::from_le_bytes(buf)
            }
        }
        6 => {
            if is_big {
                ((record[pos] as u64) << 40)
                    | ((record[pos + 1] as u64) << 32)
                    | ((record[pos + 2] as u64) << 24)
                    | ((record[pos + 3] as u64) << 16)
                    | ((record[pos + 4] as u64) << 8)
                    | (record[pos + 5] as u64)
            } else {
                (record[pos] as u64)
                    | ((record[pos + 1] as u64) << 8)
                    | ((record[pos + 2] as u64) << 16)
                    | ((record[pos + 3] as u64) << 24)
                    | ((record[pos + 4] as u64) << 32)
                    | ((record[pos + 5] as u64) << 40)
            }
        }
        5 => {
            if is_big {
                ((record[pos] as u64) << 32)
                    | ((record[pos + 1] as u64) << 24)
                    | ((record[pos + 2] as u64) << 16)
                    | ((record[pos + 3] as u64) << 8)
                    | (record[pos + 4] as u64)
            } else {
                (record[pos] as u64)
                    | ((record[pos + 1] as u64) << 8)
                    | ((record[pos + 2] as u64) << 16)
                    | ((record[pos + 3] as u64) << 24)
                    | ((record[pos + 4] as u64) << 32)
            }
        }
        _ => unreachable!("Unsupported u64 width: {n_bytes}"),
    }
}

// ─── Direct Contiguous Slice Decoders ─────────────────────────────────────

#[inline]
pub fn decode_i16_slice(src: &[u8], dst: &mut [i16], is_big: bool) -> Result<(), Error> {
    let chunks = as_exact_chunks::<2>(src, dst.len(), "Could not read i16 array")?;
    if is_big {
        for (d, s) in dst.iter_mut().zip(chunks) {
            *d = i16::from_be_bytes(*s);
        }
    } else {
        for (d, s) in dst.iter_mut().zip(chunks) {
            *d = i16::from_le_bytes(*s);
        }
    }
    Ok(())
}

#[inline]
pub fn decode_u16_slice(src: &[u8], dst: &mut [u16], is_big: bool) -> Result<(), Error> {
    let chunks = as_exact_chunks::<2>(src, dst.len(), "Could not read u16 array")?;
    if is_big {
        for (d, s) in dst.iter_mut().zip(chunks) {
            *d = u16::from_be_bytes(*s);
        }
    } else {
        for (d, s) in dst.iter_mut().zip(chunks) {
            *d = u16::from_le_bytes(*s);
        }
    }
    Ok(())
}

#[inline]
pub fn decode_i32_slice(
    src: &[u8],
    dst: &mut [i32],
    n_bytes: usize,
    is_big: bool,
) -> Result<(), Error> {
    if n_bytes == 4 {
        let chunks = as_exact_chunks::<4>(src, dst.len(), "Could not read i32 array")?;
        if is_big {
            for (d, s) in dst.iter_mut().zip(chunks) {
                *d = i32::from_be_bytes(*s);
            }
        } else {
            for (d, s) in dst.iter_mut().zip(chunks) {
                *d = i32::from_le_bytes(*s);
            }
        }
    } else if n_bytes == 3 {
        let chunks = as_exact_chunks::<3>(src, dst.len(), "Could not read i24 array")?;
        if is_big {
            for (d, s) in dst.iter_mut().zip(chunks) {
                *d = ((s[0] as i8 as i32) << 16) | ((s[1] as i32) << 8) | (s[2] as i32);
            }
        } else {
            for (d, s) in dst.iter_mut().zip(chunks) {
                *d = ((s[2] as i8 as i32) << 16) | ((s[1] as i32) << 8) | (s[0] as i32);
            }
        }
    } else {
        bail!("Unsupported i32 byte width: {n_bytes}");
    }
    Ok(())
}

#[inline]
pub fn decode_u32_slice(
    src: &[u8],
    dst: &mut [u32],
    n_bytes: usize,
    is_big: bool,
) -> Result<(), Error> {
    if n_bytes == 4 {
        let chunks = as_exact_chunks::<4>(src, dst.len(), "Could not read u32 array")?;
        if is_big {
            for (d, s) in dst.iter_mut().zip(chunks) {
                *d = u32::from_be_bytes(*s);
            }
        } else {
            for (d, s) in dst.iter_mut().zip(chunks) {
                *d = u32::from_le_bytes(*s);
            }
        }
    } else if n_bytes == 3 {
        let chunks = as_exact_chunks::<3>(src, dst.len(), "Could not read u24 array")?;
        if is_big {
            for (d, s) in dst.iter_mut().zip(chunks) {
                *d = ((s[0] as u32) << 16) | ((s[1] as u32) << 8) | (s[2] as u32);
            }
        } else {
            for (d, s) in dst.iter_mut().zip(chunks) {
                *d = ((s[2] as u32) << 16) | ((s[1] as u32) << 8) | (s[0] as u32);
            }
        }
    } else {
        bail!("Unsupported u32 byte width: {n_bytes}");
    }
    Ok(())
}

#[inline]
pub fn decode_f32_slice(
    src: &[u8],
    dst: &mut [f32],
    n_bytes: usize,
    is_big: bool,
) -> Result<(), Error> {
    if n_bytes == 4 {
        let chunks = as_exact_chunks::<4>(src, dst.len(), "Could not read f32 array")?;
        if is_big {
            for (d, s) in dst.iter_mut().zip(chunks) {
                *d = f32::from_be_bytes(*s);
            }
        } else {
            for (d, s) in dst.iter_mut().zip(chunks) {
                *d = f32::from_le_bytes(*s);
            }
        }
    } else if n_bytes == 2 {
        let chunks = as_exact_chunks::<2>(src, dst.len(), "Could not read f16 array")?;
        if is_big {
            for (d, s) in dst.iter_mut().zip(chunks) {
                *d = f16::from_be_bytes(*s).to_f32();
            }
        } else {
            for (d, s) in dst.iter_mut().zip(chunks) {
                *d = f16::from_le_bytes(*s).to_f32();
            }
        }
    } else {
        bail!("Unsupported f32 byte width: {n_bytes}");
    }
    Ok(())
}

#[inline]
pub fn decode_f64_slice(src: &[u8], dst: &mut [f64], is_big: bool) -> Result<(), Error> {
    let chunks = as_exact_chunks::<8>(src, dst.len(), "Could not read f64 array")?;
    if is_big {
        for (d, s) in dst.iter_mut().zip(chunks) {
            *d = f64::from_be_bytes(*s);
        }
    } else {
        for (d, s) in dst.iter_mut().zip(chunks) {
            *d = f64::from_le_bytes(*s);
        }
    }
    Ok(())
}

#[inline]
pub fn decode_i64_slice(
    src: &[u8],
    dst: &mut [i64],
    n_bytes: usize,
    is_big: bool,
) -> Result<(), Error> {
    match n_bytes {
        8 => {
            let chunks = as_exact_chunks::<8>(src, dst.len(), "Could not read i64 array")?;
            if is_big {
                for (d, s) in dst.iter_mut().zip(chunks) {
                    *d = i64::from_be_bytes(*s);
                }
            } else {
                for (d, s) in dst.iter_mut().zip(chunks) {
                    *d = i64::from_le_bytes(*s);
                }
            }
        }
        7 => {
            let chunks = as_exact_chunks::<7>(src, dst.len(), "Could not read i56 array")?;
            let mut temp = [0u8; 8];
            if is_big {
                for (d, s) in dst.iter_mut().zip(chunks) {
                    temp[1..8].copy_from_slice(s);
                    *d = ((u64::from_be_bytes(temp) as i64) << 8) >> 8;
                }
            } else {
                for (d, s) in dst.iter_mut().zip(chunks) {
                    temp[0..7].copy_from_slice(s);
                    *d = ((u64::from_le_bytes(temp) as i64) << 8) >> 8;
                }
            }
        }
        6 => {
            let chunks = as_exact_chunks::<6>(src, dst.len(), "Could not read i48 array")?;
            if is_big {
                for (d, s) in dst.iter_mut().zip(chunks) {
                    let raw = ((s[0] as u64) << 40)
                        | ((s[1] as u64) << 32)
                        | ((s[2] as u64) << 24)
                        | ((s[3] as u64) << 16)
                        | ((s[4] as u64) << 8)
                        | (s[5] as u64);
                    *d = ((raw as i64) << 16) >> 16;
                }
            } else {
                for (d, s) in dst.iter_mut().zip(chunks) {
                    let raw = (s[0] as u64)
                        | ((s[1] as u64) << 8)
                        | ((s[2] as u64) << 16)
                        | ((s[3] as u64) << 24)
                        | ((s[4] as u64) << 32)
                        | ((s[5] as u64) << 40);
                    *d = ((raw as i64) << 16) >> 16;
                }
            }
        }
        5 => {
            let chunks = as_exact_chunks::<5>(src, dst.len(), "Could not read i40 array")?;
            if is_big {
                for (d, s) in dst.iter_mut().zip(chunks) {
                    let raw = ((s[0] as u64) << 32)
                        | ((s[1] as u64) << 24)
                        | ((s[2] as u64) << 16)
                        | ((s[3] as u64) << 8)
                        | (s[4] as u64);
                    *d = ((raw as i64) << 24) >> 24;
                }
            } else {
                for (d, s) in dst.iter_mut().zip(chunks) {
                    let raw = (s[0] as u64)
                        | ((s[1] as u64) << 8)
                        | ((s[2] as u64) << 16)
                        | ((s[3] as u64) << 24)
                        | ((s[4] as u64) << 32);
                    *d = ((raw as i64) << 24) >> 24;
                }
            }
        }
        _ => bail!("Unsupported i64 byte width: {n_bytes}"),
    }
    Ok(())
}

#[inline]
pub fn decode_u64_slice(
    src: &[u8],
    dst: &mut [u64],
    n_bytes: usize,
    is_big: bool,
) -> Result<(), Error> {
    match n_bytes {
        8 => {
            let chunks = as_exact_chunks::<8>(src, dst.len(), "Could not read u64 array")?;
            if is_big {
                for (d, s) in dst.iter_mut().zip(chunks) {
                    *d = u64::from_be_bytes(*s);
                }
            } else {
                for (d, s) in dst.iter_mut().zip(chunks) {
                    *d = u64::from_le_bytes(*s);
                }
            }
        }
        7 => {
            let chunks = as_exact_chunks::<7>(src, dst.len(), "Could not read u56 array")?;
            let mut temp = [0u8; 8];
            if is_big {
                for (d, s) in dst.iter_mut().zip(chunks) {
                    temp[1..8].copy_from_slice(s);
                    *d = u64::from_be_bytes(temp);
                }
            } else {
                for (d, s) in dst.iter_mut().zip(chunks) {
                    temp[0..7].copy_from_slice(s);
                    *d = u64::from_le_bytes(temp);
                }
            }
        }
        6 => {
            let chunks = as_exact_chunks::<6>(src, dst.len(), "Could not read u48 array")?;
            if is_big {
                for (d, s) in dst.iter_mut().zip(chunks) {
                    *d = ((s[0] as u64) << 40)
                        | ((s[1] as u64) << 32)
                        | ((s[2] as u64) << 24)
                        | ((s[3] as u64) << 16)
                        | ((s[4] as u64) << 8)
                        | (s[5] as u64);
                }
            } else {
                for (d, s) in dst.iter_mut().zip(chunks) {
                    *d = (s[0] as u64)
                        | ((s[1] as u64) << 8)
                        | ((s[2] as u64) << 16)
                        | ((s[3] as u64) << 24)
                        | ((s[4] as u64) << 32)
                        | ((s[5] as u64) << 40);
                }
            }
        }
        5 => {
            let chunks = as_exact_chunks::<5>(src, dst.len(), "Could not read u40 array")?;
            if is_big {
                for (d, s) in dst.iter_mut().zip(chunks) {
                    *d = ((s[0] as u64) << 32)
                        | ((s[1] as u64) << 24)
                        | ((s[2] as u64) << 16)
                        | ((s[3] as u64) << 8)
                        | (s[4] as u64);
                }
            } else {
                for (d, s) in dst.iter_mut().zip(chunks) {
                    *d = (s[0] as u64)
                        | ((s[1] as u64) << 8)
                        | ((s[2] as u64) << 16)
                        | ((s[3] as u64) << 24)
                        | ((s[4] as u64) << 32);
                }
            }
        }
        _ => bail!("Unsupported u64 byte width: {n_bytes}"),
    }
    Ok(())
}

/// converts raw data block containing only one channel into a ndarray
pub fn read_one_channel_array(
    data_bytes: &[u8],
    cn: &mut Cn4,
    cycle_count: usize,
) -> Result<(), Error> {
    if cn.block.cn_type == 0
        || cn.block.cn_type == 2
        || cn.block.cn_type == 5
        || cn.block.cn_type == 7
    {
        if cn.builder.is_none() {
            cn.init_builder(cycle_count)?;
        }
        // cn_type == 5 : Maximum length data channel, removing no valid bytes done by another size channel pointed by cn_data
        // cn_type == 0 : fixed length data channel
        // cn_type == 2 : master channel
        // cn_type == 7 : VLSC channel (stores offsets into VD block)
        let n_bytes = cn.n_bytes as usize;
        let list_size = cn.list_size;
        let expected_len = cycle_count * n_bytes * list_size;
        if data_bytes.len() != expected_len {
            bail!(
                "read_one_channel_array: data buffer length {} does not match expected length {} (cycle_count={}, n_bytes={}, list_size={})",
                data_bytes.len(),
                expected_len,
                cycle_count,
                n_bytes,
                list_size
            );
        }

        // CANopen Date/Time are stored as Int64 (ms since Unix epoch); parse binary here.
        if cn.block.cn_data_type == 13 || cn.block.cn_data_type == 14 {
            if let Some(ChannelBuilder::Int64(a)) = &mut cn.builder {
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
            cn.data = cn.builder.take().unwrap().finish();
            return Ok(());
        }

        let is_big = cn.endian.is_big();
        match cn.builder.as_mut().unwrap() {
            ChannelBuilder::Int8(a) => {
                let buf = &mut a.values_slice_mut()[..cycle_count];
                for (dst, &src) in buf.iter_mut().zip(&data_bytes[..cycle_count]) {
                    *dst = src as i8;
                }
            }
            ChannelBuilder::UInt8(a) => {
                let buf = &mut a.values_slice_mut()[..cycle_count];
                buf.copy_from_slice(&data_bytes[..cycle_count]);
            }
            ChannelBuilder::Int16(a) => {
                decode_i16_slice(
                    &data_bytes[..cycle_count * 2],
                    &mut a.values_slice_mut()[..cycle_count],
                    is_big,
                )?;
            }
            ChannelBuilder::UInt16(a) => {
                decode_u16_slice(
                    &data_bytes[..cycle_count * 2],
                    &mut a.values_slice_mut()[..cycle_count],
                    is_big,
                )?;
            }
            ChannelBuilder::Int32(a) => {
                decode_i32_slice(
                    &data_bytes[..cycle_count * n_bytes],
                    &mut a.values_slice_mut()[..cycle_count],
                    n_bytes,
                    is_big,
                )?;
            }
            ChannelBuilder::UInt32(a) => {
                decode_u32_slice(
                    &data_bytes[..cycle_count * n_bytes],
                    &mut a.values_slice_mut()[..cycle_count],
                    n_bytes,
                    is_big,
                )?;
            }
            ChannelBuilder::Float32(a) => {
                decode_f32_slice(
                    &data_bytes[..cycle_count * n_bytes],
                    &mut a.values_slice_mut()[..cycle_count],
                    n_bytes,
                    is_big,
                )?;
            }
            ChannelBuilder::Int64(a) => {
                decode_i64_slice(
                    &data_bytes[..cycle_count * n_bytes],
                    &mut a.values_slice_mut()[..cycle_count],
                    n_bytes,
                    is_big,
                )?;
            }
            ChannelBuilder::UInt64(a) => {
                decode_u64_slice(
                    &data_bytes[..cycle_count * n_bytes],
                    &mut a.values_slice_mut()[..cycle_count],
                    n_bytes,
                    is_big,
                )?;
            }
            ChannelBuilder::Float64(a) => {
                decode_f64_slice(
                    &data_bytes[..cycle_count * 8],
                    &mut a.values_slice_mut()[..cycle_count],
                    is_big,
                )?;
            }
            ChannelBuilder::Complex32(a) => {
                let total_elems = cycle_count * list_size;
                let elem_bytes = if n_bytes <= 4 { 2 } else { 4 };
                decode_f32_slice(
                    &data_bytes[..total_elems * elem_bytes],
                    &mut a.values_slice_mut()[..total_elems],
                    elem_bytes,
                    is_big,
                )?;
            }
            ChannelBuilder::Complex64(a) => {
                let total_elems = cycle_count * list_size;
                decode_f64_slice(
                    &data_bytes[..total_elems * 8],
                    &mut a.values_slice_mut()[..total_elems],
                    is_big,
                )?;
            }
            ChannelBuilder::Utf8(data) => {
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
            ChannelBuilder::VariableSizeByteArray(a) => {
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
            ChannelBuilder::FixedSizeByteArray(a) => {
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
            ChannelBuilder::ArrayDInt8(a) => {
                let total_elems = cycle_count * list_size;
                let buf = &mut a.values_slice_mut()[..total_elems];
                for (dst, &src) in buf.iter_mut().zip(&data_bytes[..total_elems]) {
                    *dst = src as i8;
                }
            }
            ChannelBuilder::ArrayDUInt8(a) => {
                let total_elems = cycle_count * list_size;
                let buf = &mut a.values_slice_mut()[..total_elems];
                buf.copy_from_slice(&data_bytes[..total_elems]);
            }
            ChannelBuilder::ArrayDInt16(a) => {
                let total_elems = cycle_count * list_size;
                decode_i16_slice(
                    &data_bytes[..total_elems * 2],
                    &mut a.values_slice_mut()[..total_elems],
                    is_big,
                )?;
            }
            ChannelBuilder::ArrayDUInt16(a) => {
                let total_elems = cycle_count * list_size;
                decode_u16_slice(
                    &data_bytes[..total_elems * 2],
                    &mut a.values_slice_mut()[..total_elems],
                    is_big,
                )?;
            }
            ChannelBuilder::ArrayDInt32(a) => {
                let total_elems = cycle_count * list_size;
                decode_i32_slice(
                    &data_bytes[..total_elems * n_bytes],
                    &mut a.values_slice_mut()[..total_elems],
                    n_bytes,
                    is_big,
                )?;
            }
            ChannelBuilder::ArrayDUInt32(a) => {
                let total_elems = cycle_count * list_size;
                decode_u32_slice(
                    &data_bytes[..total_elems * n_bytes],
                    &mut a.values_slice_mut()[..total_elems],
                    n_bytes,
                    is_big,
                )?;
            }
            ChannelBuilder::ArrayDFloat32(a) => {
                let total_elems = cycle_count * list_size;
                decode_f32_slice(
                    &data_bytes[..total_elems * n_bytes],
                    &mut a.values_slice_mut()[..total_elems],
                    n_bytes,
                    is_big,
                )?;
            }
            ChannelBuilder::ArrayDInt64(a) => {
                let total_elems = cycle_count * list_size;
                decode_i64_slice(
                    &data_bytes[..total_elems * n_bytes],
                    &mut a.values_slice_mut()[..total_elems],
                    n_bytes,
                    is_big,
                )?;
            }
            ChannelBuilder::ArrayDUInt64(a) => {
                let total_elems = cycle_count * list_size;
                decode_u64_slice(
                    &data_bytes[..total_elems * n_bytes],
                    &mut a.values_slice_mut()[..total_elems],
                    n_bytes,
                    is_big,
                )?;
            }
            ChannelBuilder::ArrayDFloat64(a) => {
                let total_elems = cycle_count * list_size;
                decode_f64_slice(
                    &data_bytes[..total_elems * 8],
                    &mut a.values_slice_mut()[..total_elems],
                    is_big,
                )?;
            }
            ChannelBuilder::Union(_) => {} // Union channels are constructed post-read
        }
        cn.data = cn.builder.take().unwrap().finish();
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
                || (cn.block.cn_type == 7
                    && matches!(
                        &cn.builder,
                        Some(
                            ChannelBuilder::UInt8(_)
                                | ChannelBuilder::UInt16(_)
                                | ChannelBuilder::UInt32(_)
                                | ChannelBuilder::UInt64(_)
                        )
                    ))
            {
                let mut value: &[u8];
                let pos_byte_beg = cn.pos_byte_beg as usize;
                let n_bytes = cn.n_bytes as usize;
                let list_size = cn.list_size;
                let is_big = cn.endian.is_big();
                let builder = match &mut cn.builder {
                    Some(b) => b,
                    None => return Ok(()),
                };
                match builder {
                    ChannelBuilder::Int8(a) => {
                        let data = a.values_slice_mut();
                        for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                            data[i + previous_index] = record[pos_byte_beg] as i8;
                        }
                    }
                    ChannelBuilder::UInt8(a) => {
                        let data = a.values_slice_mut();
                        for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                            data[i + previous_index] = record[pos_byte_beg];
                        }
                    }
                    ChannelBuilder::Int16(a) => {
                        let data = a.values_slice_mut();
                        for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                            data[i + previous_index] = read_i16_at(record, pos_byte_beg, is_big);
                        }
                    }
                    ChannelBuilder::UInt16(a) => {
                        let data = a.values_slice_mut();
                        for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                            data[i + previous_index] = read_u16_at(record, pos_byte_beg, is_big);
                        }
                    }
                    ChannelBuilder::Int32(a) => {
                        let data = a.values_slice_mut();
                        for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                            data[i + previous_index] =
                                read_i32_at(record, pos_byte_beg, n_bytes, is_big);
                        }
                    }
                    ChannelBuilder::UInt32(a) => {
                        let data = a.values_slice_mut();
                        for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                            data[i + previous_index] =
                                read_u32_at(record, pos_byte_beg, n_bytes, is_big);
                        }
                    }
                    ChannelBuilder::Float32(a) => {
                        let data = a.values_slice_mut();
                        for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                            data[i + previous_index] =
                                read_f32_at(record, pos_byte_beg, n_bytes, is_big);
                        }
                    }
                    ChannelBuilder::Int64(a) => {
                        let data = a.values_slice_mut();
                        for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                            data[i + previous_index] =
                                read_i64_at(record, pos_byte_beg, n_bytes, is_big);
                        }
                    }
                    ChannelBuilder::UInt64(a) => {
                        let data = a.values_slice_mut();
                        for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                            data[i + previous_index] =
                                read_u64_at(record, pos_byte_beg, n_bytes, is_big);
                        }
                    }
                    ChannelBuilder::Float64(a) => {
                        let data = a.values_slice_mut();
                        for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                            data[i + previous_index] = read_f64_at(record, pos_byte_beg, is_big);
                        }
                    }
                    ChannelBuilder::Complex32(a) => {
                        let data = a.values_slice_mut();
                        let elem_bytes = if n_bytes <= 4 { 2 } else { 4 };
                        for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                            let dst_idx = (i + previous_index) * 2;
                            data[dst_idx] = read_f32_at(record, pos_byte_beg, elem_bytes, is_big);
                            data[dst_idx + 1] =
                                read_f32_at(record, pos_byte_beg + elem_bytes, elem_bytes, is_big);
                        }
                    }
                    ChannelBuilder::Complex64(a) => {
                        let data = a.values_slice_mut();
                        for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                            let dst_idx = (i + previous_index) * 2;
                            data[dst_idx] = read_f64_at(record, pos_byte_beg, is_big);
                            data[dst_idx + 1] = read_f64_at(record, pos_byte_beg + 8, is_big);
                        }
                    }
                    ChannelBuilder::Utf8(array) => {
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
                    ChannelBuilder::VariableSizeByteArray(array) => {
                        for record in data_chunk.chunks_exact(record_length) {
                            if pos_byte_beg + n_bytes <= record.len() {
                                array.append_value(&record[pos_byte_beg..pos_byte_beg + n_bytes]);
                            } else {
                                array.append_null();
                            }
                        }
                    }
                    ChannelBuilder::FixedSizeByteArray(a) => {
                        for record in data_chunk.chunks_exact(record_length) {
                            if pos_byte_beg + n_bytes <= record.len() {
                                a.append_value(&record[pos_byte_beg..pos_byte_beg + n_bytes])
                                    .context("failed appending new value")?;
                            } else {
                                a.append_null();
                            }
                        }
                    }
                    ChannelBuilder::ArrayDInt8(a) => {
                        let data = a.values_slice_mut();
                        let row_bytes = list_size;
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
                    ChannelBuilder::ArrayDUInt8(a) => {
                        let data = a.values_slice_mut();
                        let row_bytes = list_size;
                        for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                            let dst_start = (i + previous_index) * list_size;
                            data[dst_start..dst_start + list_size]
                                .copy_from_slice(&record[pos_byte_beg..pos_byte_beg + row_bytes]);
                        }
                    }
                    ChannelBuilder::ArrayDInt16(a) => {
                        let data = a.values_slice_mut();
                        let row_bytes = list_size * 2;
                        for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                            let dst_start = (i + previous_index) * list_size;
                            decode_i16_slice(
                                &record[pos_byte_beg..pos_byte_beg + row_bytes],
                                &mut data[dst_start..dst_start + list_size],
                                is_big,
                            )?;
                        }
                    }
                    ChannelBuilder::ArrayDUInt16(a) => {
                        let data = a.values_slice_mut();
                        let row_bytes = list_size * 2;
                        for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                            let dst_start = (i + previous_index) * list_size;
                            decode_u16_slice(
                                &record[pos_byte_beg..pos_byte_beg + row_bytes],
                                &mut data[dst_start..dst_start + list_size],
                                is_big,
                            )?;
                        }
                    }
                    ChannelBuilder::ArrayDInt32(a) => {
                        let data = a.values_slice_mut();
                        let row_bytes = list_size * n_bytes;
                        for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                            let dst_start = (i + previous_index) * list_size;
                            decode_i32_slice(
                                &record[pos_byte_beg..pos_byte_beg + row_bytes],
                                &mut data[dst_start..dst_start + list_size],
                                n_bytes,
                                is_big,
                            )?;
                        }
                    }
                    ChannelBuilder::ArrayDUInt32(a) => {
                        let data = a.values_slice_mut();
                        let row_bytes = list_size * n_bytes;
                        for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                            let dst_start = (i + previous_index) * list_size;
                            decode_u32_slice(
                                &record[pos_byte_beg..pos_byte_beg + row_bytes],
                                &mut data[dst_start..dst_start + list_size],
                                n_bytes,
                                is_big,
                            )?;
                        }
                    }
                    ChannelBuilder::ArrayDFloat32(a) => {
                        let data = a.values_slice_mut();
                        let row_bytes = list_size * n_bytes;
                        for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                            let dst_start = (i + previous_index) * list_size;
                            decode_f32_slice(
                                &record[pos_byte_beg..pos_byte_beg + row_bytes],
                                &mut data[dst_start..dst_start + list_size],
                                n_bytes,
                                is_big,
                            )?;
                        }
                    }
                    ChannelBuilder::ArrayDInt64(a) => {
                        let data = a.values_slice_mut();
                        let row_bytes = list_size * n_bytes;
                        for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                            let dst_start = (i + previous_index) * list_size;
                            decode_i64_slice(
                                &record[pos_byte_beg..pos_byte_beg + row_bytes],
                                &mut data[dst_start..dst_start + list_size],
                                n_bytes,
                                is_big,
                            )?;
                        }
                    }
                    ChannelBuilder::ArrayDUInt64(a) => {
                        let data = a.values_slice_mut();
                        let row_bytes = list_size * n_bytes;
                        for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                            let dst_start = (i + previous_index) * list_size;
                            decode_u64_slice(
                                &record[pos_byte_beg..pos_byte_beg + row_bytes],
                                &mut data[dst_start..dst_start + list_size],
                                n_bytes,
                                is_big,
                            )?;
                        }
                    }
                    ChannelBuilder::ArrayDFloat64(a) => {
                        let data = a.values_slice_mut();
                        let row_bytes = list_size * 8;
                        for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                            let dst_start = (i + previous_index) * list_size;
                            decode_f64_slice(
                                &record[pos_byte_beg..pos_byte_beg + row_bytes],
                                &mut data[dst_start..dst_start + list_size],
                                is_big,
                            )?;
                        }
                    }
                    ChannelBuilder::Union(_) => {} // Union channels are constructed post-read
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
