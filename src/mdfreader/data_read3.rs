//! this module implements low level data reading for mdf3 files.
use crate::mdfinfo::mdfinfo3::Cn3;
use anyhow::{bail, Context, Error, Result};
use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use encoding_rs::WINDOWS_1252;
use half::f16;
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::Cursor;
use std::string::String;
use std::{collections::HashSet, convert::TryInto};

use super::channel_data::ChannelData;

/// copies data from data_chunk into each channel array
pub fn read_channels_from_bytes(
    data_chunk: &[u8],
    channels: &mut HashMap<u32, Cn3>,
    record_length: usize,
    previous_index: usize,
    channel_names_to_read_in_dg: &HashSet<String>,
) -> Result<(), Error> {
    // iterates for each channel in parallel with rayon crate
    channels
        .par_iter_mut()
        .filter(|(_cn_record_position, cn)| {
            channel_names_to_read_in_dg.contains(&cn.unique_name) && !cn.data.is_empty()
        })
        .try_for_each(|(_cn_pos, cn): (&u32, &mut Cn3)| -> Result<(), Error> {
            let mut value: &[u8]; // value of channel at record
            let pos_byte_beg = cn.pos_byte_beg as usize;
            let n_bytes = cn.n_bytes as usize;
            match &mut cn.data {
                ChannelData::Int8(a)  => {
                    let data = a.get_mut_values().context("Read channels from bytes function could not get mutable values from primitive array i8")?;
                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i8>()];
                        data[i + previous_index] = i8::from_le_bytes(value.try_into().context("Could not read i8")?);
                    }
                }
                ChannelData::UInt8(a)  => {
                    let data = a.get_mut_values().context("Read channels from bytes function could not get mutable values from primitive array u8")?;
                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u8>()];
                        data[i + previous_index] = u8::from_le_bytes(value.try_into().context("Could not read u8")?);
                    }
                }
                ChannelData::Int16(a)  => {
                    let data = a.get_mut_values().context("Read channels from bytes function could not get mutable values from primitive array i16")?;
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i16>()];
                            data[i + previous_index] = i16::from_be_bytes(
                                value.try_into().context("Could not read be i16")?,
                            );
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i16>()];
                            data[i + previous_index] = i16::from_le_bytes(
                                value.try_into().context("Could not read le i16")?,
                            );
                        }
                    }
                }
                ChannelData::UInt16(a)  => {
                    let data = a.get_mut_values().context("Read channels from bytes function could not get mutable values from primitive array u16")?;
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u16>()];
                                data[i + previous_index]  = u16::from_be_bytes(
                                value.try_into().context("Could not read be u16")?,
                            );
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u16>()];
                                data[i + previous_index]  = u16::from_le_bytes(
                                value.try_into().context("Could not read le u16")?,
                            );
                        }
                    }
                }
                ChannelData::Int32(a)  => {
                    let data = a.get_mut_values().with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array i32, channel {}", cn.unique_name))?;
                    if  n_bytes == 3 {
                        if cn.endian {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                data[i + previous_index] = value
                                    .read_i24::<BigEndian>()
                                    .context("Could not read be i24")?;
                            }
                        } else {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                data[i + previous_index] = value
                                    .read_i24::<LittleEndian>()
                                    .context("Could not read le i24")?;
                            }
                        }
                    } else if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i32>()];
                            data[i + previous_index] = i32::from_be_bytes(
                                value.try_into().context("Could not read be i32")?,
                            );
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i32>()];
                            data[i + previous_index] = i32::from_le_bytes(
                                value.try_into().context("Could not read le i32")?,
                            );
                        }
                    }
                }
                ChannelData::UInt32(a)  => {
                    let data = a.get_mut_values().with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array u32, channel {}", cn.unique_name))?;
                    if n_bytes == 3 {
                        if cn.endian {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                data[i + previous_index] = value
                                    .read_u24::<BigEndian>()
                                    .context("Could not read be u24")?;
                            }
                        } else {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                data[i + previous_index] = value
                                    .read_u24::<LittleEndian>()
                                    .context("Could not read le u24")?;
                            }
                        }
                    } else if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u32>()];
                            data[i + previous_index] = u32::from_be_bytes(
                                value.try_into().context("Could not read be u32")?,
                            );
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u32>()];
                            data[i + previous_index] = u32::from_le_bytes(
                                value.try_into().context("Could not read le u32")?,
                            );
                        }
                    }
                }
                ChannelData::Float32(a)  => {
                    let data = a.get_mut_values().with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array f32, channel {}", cn.unique_name))?;
                    if cn.endian {
                        if n_bytes == 2 {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value =
                                    &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f16>()];
                                    data[i + previous_index] = f16::from_be_bytes(
                                    value.try_into().context("Could not read be f16")?,
                                )
                                .to_f32();
                            }
                        } else {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value =
                                    &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f32>()];
                                    data[i + previous_index] = f32::from_be_bytes(
                                    value.try_into().context("Could not read be f32")?,
                                );
                            }
                        }
                    } else if n_bytes ==2 {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f16>()];
                                data[i + previous_index] = f16::from_le_bytes(
                                value.try_into().context("Could not read le f16")?,
                            )
                            .to_f32();
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f32>()];
                            data[i + previous_index] = f32::from_le_bytes(
                                value.try_into().context("Could not read le f32")?,
                            );
                        }
                    }
                }
                ChannelData::Int64(a)  => {
                    let data = a.get_mut_values().with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array i64, channel {}", cn.unique_name))?;
                    if cn.endian {
                        if n_bytes == 8 {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                data[i + previous_index] = i64::from_be_bytes(
                                    value.try_into().context("Could not read be i64")?,
                                );
                            }
                        } else if n_bytes == 6 {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                data[i + previous_index] = value
                                    .read_i48::<BigEndian>()
                                    .context("Could not read be i48")?;
                            }
                        }
                    } else if n_bytes == 8 {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            data[i + previous_index] = i64::from_le_bytes(
                                value.try_into().context("Could not read le i64")?,
                            );
                        }
                    } else if n_bytes == 6 {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            data[i + previous_index] = value
                                .read_i48::<LittleEndian>()
                                .context("Could not read le i48")?;
                        }
                    }
                }
                ChannelData::UInt64(a)  => {
                    let data = a.get_mut_values().with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array u64, channel {}", cn.unique_name))?;
                    if cn.endian {
                        if n_bytes == 8 {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                data[i + previous_index] = u64::from_be_bytes(
                                    value.try_into().context("Could not read be u64")?,
                                );
                            }
                        } else if n_bytes == 7 {
                            let mut buf = [0u8; std::mem::size_of::<u64>()];
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                buf[0..7].copy_from_slice(&record[pos_byte_beg..pos_byte_beg + 7]);
                                data[i + previous_index] = u64::from_be_bytes(buf);
                            }
                        } else if n_bytes == 6 {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                data[i + previous_index] = value
                                    .read_u48::<BigEndian>()
                                    .context("Could not read be u48")?;
                            }
                        } else {
                            let mut buf = [0u8; 6];
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                buf[0..5]
                                    .copy_from_slice(&record[pos_byte_beg..pos_byte_beg + n_bytes]);
                                data[i + previous_index] = Cursor::new(buf)
                                    .read_u48::<BigEndian>()
                                    .context("Could not read be u48 from 5 bytes")?;
                            }
                        }
                    } else if n_bytes == 8 {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            data[i + previous_index] = u64::from_le_bytes(
                                value.try_into().context("Could not read le u64")?,
                            );
                        }
                    } else  if n_bytes == 7 { // little endian
                        let mut buf = [0u8; std::mem::size_of::<u64>()];
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            buf[0..7].copy_from_slice(&record[pos_byte_beg..pos_byte_beg + 7]);
                            data[i + previous_index] = u64::from_le_bytes(buf);
                        }
                    } else if n_bytes == 6 {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            data[i + previous_index] = value
                                .read_u48::<LittleEndian>()
                                .context("Could not read le u48")?;
                        }
                    } else { // n_bytes = 5
                        let mut buf = [0u8; 6];
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            buf[0..5].copy_from_slice(&record[pos_byte_beg..pos_byte_beg + 5]);
                            data[i + previous_index] = Cursor::new(buf)
                                .read_u48::<LittleEndian>()
                                .context("Could not read le u48 from 5 bytes")?;
                        }
                    }
                }
                ChannelData::Float64(a)  => {
                    let data = a.get_mut_values().with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array f64, channel {}", cn.unique_name))?;
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f64>()];
                            data[i + previous_index] = f64::from_be_bytes(
                                    value.try_into().context("Could not read be f64")?,
                                );
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f64>()];
                            data[i + previous_index] = f64::from_le_bytes(
                                    value.try_into().context("Could not read le f64")?,
                                );
                        }
                    }
                }
                ChannelData::Utf8(mut array)  => {
                    let n_bytes = cn.n_bytes as usize;
                    // SBC ISO-8859-1 to be converted into UTF8
                    let mut decoder = WINDOWS_1252.new_decoder();
                    for record in data_chunk.chunks(record_length) {
                        value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                        let mut dst = String::with_capacity(value.len());
                        let (_result, _size, _replacement) =
                            decoder.decode_to_string(value, &mut dst, false);
                        dst = dst.trim_end_matches('\0').to_owned();
                        array.push(Some(dst));
                    }
                }
                ChannelData::VariableSizeByteArray(mut array)  => {
                    let n_bytes = cn.n_bytes as usize;
                    for record in data_chunk.chunks(record_length) {
                        array.push(Some(&record[pos_byte_beg..pos_byte_beg + n_bytes]));
                    }
                }
                ChannelData::FixedSizeByteArray(a)  => {
                    let n_bytes = cn.n_bytes as usize;
                    for record in data_chunk.chunks(record_length) {
                        a.push(Some(&record[pos_byte_beg..pos_byte_beg + n_bytes]));
                    }
                }
                _=> bail!("mdf3 data type array not possible")
            }
            // channel was properly read
            cn.channel_data_valid = true;

            // Other channel types : virtual channels cn_type 3 & 6 are handled at initialisation
            Ok(())
        }).context("failed reading channels from bytes")?;
    Ok(())
}
