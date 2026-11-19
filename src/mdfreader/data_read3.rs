//! this module implements low level data reading for mdf3 files.
use crate::mdfinfo::mdfinfo3::Cn3;
use anyhow::{Context, Error, Result, bail};
use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use encoding_rs::WINDOWS_1252;
use half::f16;
use rayon::prelude::*;
use std::collections::HashMap;

use crate::data_holder::channel_builder::ChannelBuilder;

/// copies data from data_chunk into each channel array
pub fn read_channels_from_bytes(
    data_chunk: &[u8],
    channels: &mut HashMap<u32, Cn3>,
    record_length: usize,
    previous_index: usize,
) -> Result<(), Error> {
    if record_length == 0 || data_chunk.is_empty() {
        return Ok(());
    }
    if !data_chunk.len().is_multiple_of(record_length) {
        bail!(
            "Data chunk length ({}) is not a multiple of record_length ({}): {} remainder bytes",
            data_chunk.len(),
            record_length,
            data_chunk.len() % record_length
        );
    }
    let min_for_parallel = (32 * 1024usize).div_ceil(data_chunk.len()).max(2);
    let readable_sample = channels
        .values()
        .filter(|cn| cn.should_read)
        .take(min_for_parallel)
        .count();
    if readable_sample == 0 {
        return Ok(());
    }
    if readable_sample < min_for_parallel {
        for cn in channels.values_mut() {
            if cn.should_read {
                decode_cn3_chunk(cn, data_chunk, record_length, previous_index)?;
            }
        }
        return Ok(());
    }

    // iterates for each channel in parallel with rayon crate
    channels
        .par_iter_mut()
        .filter(|(_cn_record_position, cn)| cn.should_read)
        .try_for_each(|(_cn_pos, cn): (&u32, &mut Cn3)| -> Result<(), Error> {
            decode_cn3_chunk(cn, data_chunk, record_length, previous_index)
        })
        .context("failed reading channels from bytes")?;
    Ok(())
}

/// Decodes one MDF3 channel's records from `data_chunk` into `cn.builder`.
fn decode_cn3_chunk(
    cn: &mut Cn3,
    data_chunk: &[u8],
    record_length: usize,
    previous_index: usize,
) -> Result<(), Error> {
    let mut value: &[u8]; // value of channel at record
    let pos_byte_beg = cn.pos_byte_beg as usize;
    let n_bytes = cn.n_bytes as usize;
    if pos_byte_beg.saturating_add(n_bytes) > record_length {
        bail!(
            "Channel '{}' field ({}..{}) exceeds record length {}",
            cn.unique_name,
            pos_byte_beg,
            pos_byte_beg.saturating_add(n_bytes),
            record_length
        );
    }
    let builder = match cn.builder.as_deref_mut() {
        Some(b) => b,
        None => return Ok(()),
    };
    match builder {
        ChannelBuilder::Int8(a) => {
            let data = a.values_slice_mut();
            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i8>()];
                data[i + previous_index] =
                    i8::from_le_bytes(value.try_into().context("Could not read i8")?);
            }
        }
        ChannelBuilder::UInt8(a) => {
            let data = a.values_slice_mut();
            for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u8>()];
                data[i + previous_index] =
                    u8::from_le_bytes(value.try_into().context("Could not read u8")?);
            }
        }
        ChannelBuilder::Int16(a) => {
            let data = a.values_slice_mut();
            if cn.endian.is_big() {
                for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                    value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i16>()];
                    data[i + previous_index] =
                        i16::from_be_bytes(value.try_into().context("Could not read be i16")?);
                }
            } else {
                for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                    value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i16>()];
                    data[i + previous_index] =
                        i16::from_le_bytes(value.try_into().context("Could not read le i16")?);
                }
            }
        }
        ChannelBuilder::UInt16(a) => {
            let data = a.values_slice_mut();
            if cn.endian.is_big() {
                for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                    value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u16>()];
                    data[i + previous_index] =
                        u16::from_be_bytes(value.try_into().context("Could not read be u16")?);
                }
            } else {
                for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                    value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u16>()];
                    data[i + previous_index] =
                        u16::from_le_bytes(value.try_into().context("Could not read le u16")?);
                }
            }
        }
        ChannelBuilder::Int32(a) => {
            let data = a.values_slice_mut();
            if n_bytes == 3 {
                if cn.endian.is_big() {
                    for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                        data[i + previous_index] = value
                            .read_i24::<BigEndian>()
                            .context("Could not read be i24")?;
                    }
                } else {
                    for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                        data[i + previous_index] = value
                            .read_i24::<LittleEndian>()
                            .context("Could not read le i24")?;
                    }
                }
            } else if cn.endian.is_big() {
                for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                    value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i32>()];
                    data[i + previous_index] =
                        i32::from_be_bytes(value.try_into().context("Could not read be i32")?);
                }
            } else {
                for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                    value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i32>()];
                    data[i + previous_index] =
                        i32::from_le_bytes(value.try_into().context("Could not read le i32")?);
                }
            }
        }
        ChannelBuilder::UInt32(a) => {
            let data = a.values_slice_mut();
            if n_bytes == 3 {
                if cn.endian.is_big() {
                    for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                        data[i + previous_index] = value
                            .read_u24::<BigEndian>()
                            .context("Could not read be u24")?;
                    }
                } else {
                    for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                        data[i + previous_index] = value
                            .read_u24::<LittleEndian>()
                            .context("Could not read le u24")?;
                    }
                }
            } else if cn.endian.is_big() {
                for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                    value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u32>()];
                    data[i + previous_index] =
                        u32::from_be_bytes(value.try_into().context("Could not read be u32")?);
                }
            } else {
                for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                    value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u32>()];
                    data[i + previous_index] =
                        u32::from_le_bytes(value.try_into().context("Could not read le u32")?);
                }
            }
        }
        ChannelBuilder::Float32(a) => {
            let data = a.values_slice_mut();
            if cn.endian.is_big() {
                if n_bytes == 2 {
                    for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f16>()];
                        data[i + previous_index] =
                            f16::from_be_bytes(value.try_into().context("Could not read be f16")?)
                                .to_f32();
                    }
                } else {
                    for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f32>()];
                        data[i + previous_index] =
                            f32::from_be_bytes(value.try_into().context("Could not read be f32")?);
                    }
                }
            } else if n_bytes == 2 {
                for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                    value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f16>()];
                    data[i + previous_index] =
                        f16::from_le_bytes(value.try_into().context("Could not read le f16")?)
                            .to_f32();
                }
            } else {
                for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                    value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f32>()];
                    data[i + previous_index] =
                        f32::from_le_bytes(value.try_into().context("Could not read le f32")?);
                }
            }
        }
        ChannelBuilder::Int64(a) => {
            let data = a.values_slice_mut();
            if cn.endian.is_big() {
                if n_bytes == 8 {
                    for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + 8];
                        data[i + previous_index] =
                            i64::from_be_bytes(value.try_into().context("Could not read be i64")?);
                    }
                } else if n_bytes == 7 {
                    for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                        let slice = &record[pos_byte_beg..pos_byte_beg + 7];
                        let sign = if slice[0] & 0x80 != 0 { 0xFF } else { 0x00 };
                        let mut buf = [sign; 8];
                        buf[1..8].copy_from_slice(slice);
                        data[i + previous_index] = i64::from_be_bytes(buf);
                    }
                } else if n_bytes == 6 {
                    for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + 6];
                        data[i + previous_index] = value
                            .read_i48::<BigEndian>()
                            .context("Could not read be i48")?;
                    }
                } else if n_bytes == 5 {
                    for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                        let slice = &record[pos_byte_beg..pos_byte_beg + 5];
                        let sign = if slice[0] & 0x80 != 0 { 0xFF } else { 0x00 };
                        let mut buf = [sign; 8];
                        buf[3..8].copy_from_slice(slice);
                        data[i + previous_index] = i64::from_be_bytes(buf);
                    }
                }
            } else if n_bytes == 8 {
                for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                    value = &record[pos_byte_beg..pos_byte_beg + 8];
                    data[i + previous_index] =
                        i64::from_le_bytes(value.try_into().context("Could not read le i64")?);
                }
            } else if n_bytes == 7 {
                for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                    let slice = &record[pos_byte_beg..pos_byte_beg + 7];
                    let sign = if slice[6] & 0x80 != 0 { 0xFF } else { 0x00 };
                    let mut buf = [sign; 8];
                    buf[0..7].copy_from_slice(slice);
                    data[i + previous_index] = i64::from_le_bytes(buf);
                }
            } else if n_bytes == 6 {
                for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                    value = &record[pos_byte_beg..pos_byte_beg + 6];
                    data[i + previous_index] = value
                        .read_i48::<LittleEndian>()
                        .context("Could not read le i48")?;
                }
            } else if n_bytes == 5 {
                for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                    let slice = &record[pos_byte_beg..pos_byte_beg + 5];
                    let sign = if slice[4] & 0x80 != 0 { 0xFF } else { 0x00 };
                    let mut buf = [sign; 8];
                    buf[0..5].copy_from_slice(slice);
                    data[i + previous_index] = i64::from_le_bytes(buf);
                }
            }
        }
        ChannelBuilder::UInt64(a) => {
            let data = a.values_slice_mut();
            if cn.endian.is_big() {
                if n_bytes == 8 {
                    for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + 8];
                        data[i + previous_index] =
                            u64::from_be_bytes(value.try_into().context("Could not read be u64")?);
                    }
                } else if n_bytes == 7 {
                    let mut buf = [0u8; 8];
                    for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                        buf[1..8].copy_from_slice(&record[pos_byte_beg..pos_byte_beg + 7]);
                        data[i + previous_index] = u64::from_be_bytes(buf);
                    }
                } else if n_bytes == 6 {
                    for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + 6];
                        data[i + previous_index] = value
                            .read_u48::<BigEndian>()
                            .context("Could not read be u48")?;
                    }
                } else if n_bytes == 5 {
                    let mut buf = [0u8; 8];
                    for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                        buf[3..8].copy_from_slice(&record[pos_byte_beg..pos_byte_beg + 5]);
                        data[i + previous_index] = u64::from_be_bytes(buf);
                    }
                }
            } else if n_bytes == 8 {
                for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                    value = &record[pos_byte_beg..pos_byte_beg + 8];
                    data[i + previous_index] =
                        u64::from_le_bytes(value.try_into().context("Could not read le u64")?);
                }
            } else if n_bytes == 7 {
                let mut buf = [0u8; 8];
                for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                    buf[0..7].copy_from_slice(&record[pos_byte_beg..pos_byte_beg + 7]);
                    data[i + previous_index] = u64::from_le_bytes(buf);
                }
            } else if n_bytes == 6 {
                for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                    value = &record[pos_byte_beg..pos_byte_beg + 6];
                    data[i + previous_index] = value
                        .read_u48::<LittleEndian>()
                        .context("Could not read le u48")?;
                }
            } else if n_bytes == 5 {
                let mut buf = [0u8; 8];
                for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                    buf[0..5].copy_from_slice(&record[pos_byte_beg..pos_byte_beg + 5]);
                    data[i + previous_index] = u64::from_le_bytes(buf);
                }
            }
        }
        ChannelBuilder::Float64(a) => {
            let data = a.values_slice_mut();
            if cn.endian.is_big() {
                for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                    value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f64>()];
                    data[i + previous_index] =
                        f64::from_be_bytes(value.try_into().context("Could not read be f64")?);
                }
            } else {
                for (i, record) in data_chunk.chunks_exact(record_length).enumerate() {
                    value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f64>()];
                    data[i + previous_index] =
                        f64::from_le_bytes(value.try_into().context("Could not read le f64")?);
                }
            }
        }
        ChannelBuilder::Utf8(array) => {
            let n_bytes = cn.n_bytes as usize;
            // SBC ISO-8859-1 to be converted into UTF8
            let mut decoder = WINDOWS_1252.new_decoder();
            let mut dst = String::with_capacity(n_bytes);
            for record in data_chunk.chunks_exact(record_length) {
                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                dst.clear();
                let (_result, _size, _replacement) =
                    decoder.decode_to_string(value, &mut dst, false);
                array.append_value(dst.trim_end_matches('\0'));
            }
        }
        ChannelBuilder::VariableSizeByteArray(array) => {
            let n_bytes = cn.n_bytes as usize;
            for record in data_chunk.chunks_exact(record_length) {
                array.append_value(&record[pos_byte_beg..pos_byte_beg + n_bytes]);
            }
        }
        ChannelBuilder::FixedSizeByteArray(a) => {
            let n_bytes = cn.n_bytes as usize;
            for record in data_chunk.chunks_exact(record_length) {
                a.append_value(&record[pos_byte_beg..pos_byte_beg + n_bytes])
                    .context("failed appending new value")?;
            }
        }
        _ => bail!("mdf3 data type array not possible"),
    }
    // channel was properly read
    cn.channel_data_valid = true;

    // Other channel types : virtual channels cn_type 3 & 6 are handled at initialisation
    Ok(())
}
