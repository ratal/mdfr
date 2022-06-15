//! this module implements low level data reading for mdf3 files.
use super::channel_data::ChannelData;
use crate::mdfinfo::mdfinfo3::Cn3;
use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use encoding_rs::WINDOWS_1252;
use half::f16;
use rayon::prelude::*;
use std::collections::HashMap;
use std::str;
use std::string::String;
use std::{collections::HashSet, convert::TryInto};

/// copies data from data_chunk into each channel array
pub fn read_channels_from_bytes(
    data_chunk: &[u8],
    channels: &mut HashMap<u32, Cn3>,
    record_length: usize,
    previous_index: usize,
    channel_names_to_read_in_dg: &HashSet<String>,
) {
    // iterates for each channel in parallel with rayon crate
    channels
        .par_iter_mut()
        .filter(|(_cn_record_position, cn)| {
            channel_names_to_read_in_dg.contains(&cn.unique_name) && !cn.data.is_empty()
        })
        .for_each(|(_cn_pos, cn)| {
            let mut value: &[u8]; // value of channel at record
            let pos_byte_beg = cn.pos_byte_beg as usize;
            let n_bytes = cn.n_bytes as usize;
            match &mut cn.data {
                ChannelData::Boolean(data) => {
                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u8>()];
                        data[i + previous_index] =
                            u8::from_le_bytes(value.try_into().expect("Could not read u8"));
                    }
                }
                ChannelData::Int8(data) => {
                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i8>()];
                        data[i + previous_index] =
                            i8::from_le_bytes(value.try_into().expect("Could not read i8"));
                    }
                }
                ChannelData::UInt8(data) => {
                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u8>()];
                        data[i + previous_index] =
                            u8::from_le_bytes(value.try_into().expect("Could not read u8"));
                    }
                }
                ChannelData::Int16(data) => {
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i16>()];
                            data[i + previous_index] = i16::from_be_bytes(
                                value.try_into().expect("Could not read be i16"),
                            );
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i16>()];
                            data[i + previous_index] = i16::from_le_bytes(
                                value.try_into().expect("Could not read le i16"),
                            );
                        }
                    }
                }
                ChannelData::UInt16(data) => {
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u16>()];
                            data[i + previous_index] = u16::from_be_bytes(
                                value.try_into().expect("Could not read be u16"),
                            );
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u16>()];
                            data[i + previous_index] = u16::from_le_bytes(
                                value.try_into().expect("Could not read le u16"),
                            );
                        }
                    }
                }
                ChannelData::Float16(data) => {
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f16>()];
                            data[i + previous_index] = f16::from_be_bytes(
                                value.try_into().expect("Could not read be f16"),
                            )
                            .to_f32();
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f16>()];
                            data[i + previous_index] = f16::from_le_bytes(
                                value.try_into().expect("Could not read le f16"),
                            )
                            .to_f32();
                        }
                    }
                }
                ChannelData::Int24(data) => {
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            data[i + previous_index] = value
                                .read_i24::<BigEndian>()
                                .expect("Could not read be i24");
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            data[i + previous_index] = value
                                .read_i24::<LittleEndian>()
                                .expect("Could not read le i24");
                        }
                    }
                }
                ChannelData::UInt24(data) => {
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            data[i + previous_index] = value
                                .read_u24::<BigEndian>()
                                .expect("Could not read be u24");
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            data[i + previous_index] = value
                                .read_u24::<LittleEndian>()
                                .expect("Could not read le u24");
                        }
                    }
                }
                ChannelData::Int32(data) => {
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i32>()];
                            data[i + previous_index] = i32::from_be_bytes(
                                value.try_into().expect("Could not read be i32"),
                            );
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i32>()];
                            data[i + previous_index] = i32::from_le_bytes(
                                value.try_into().expect("Could not read le i32"),
                            );
                        }
                    }
                }
                ChannelData::UInt32(data) => {
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u32>()];
                            data[i + previous_index] = u32::from_be_bytes(
                                value.try_into().expect("Could not read be u32"),
                            );
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u32>()];
                            data[i + previous_index] = u32::from_le_bytes(
                                value.try_into().expect("Could not read le u32"),
                            );
                        }
                    }
                }
                ChannelData::Float32(data) => {
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f32>()];
                            data[i + previous_index] = f32::from_be_bytes(
                                value.try_into().expect("Could not read be f32"),
                            );
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f32>()];
                            data[i + previous_index] = f32::from_le_bytes(
                                value.try_into().expect("Could not read le f32"),
                            );
                        }
                    }
                }
                ChannelData::Int48(data) => {
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            data[i + previous_index] = value
                                .read_i48::<BigEndian>()
                                .expect("Could not read be i48");
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            data[i + previous_index] = value
                                .read_i48::<LittleEndian>()
                                .expect("Could not read le i48");
                        }
                    }
                }
                ChannelData::UInt48(data) => {
                    if cn.endian {
                        if n_bytes == 6 {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                data[i + previous_index] = value
                                    .read_u48::<BigEndian>()
                                    .expect("Could not read be u48");
                            }
                        } else {
                            // n_bytes = 5
                            let mut buf = [0u8; 6];
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                buf[0..5]
                                    .copy_from_slice(&record[pos_byte_beg..pos_byte_beg + n_bytes]);
                                data[i + previous_index] = Box::new(&buf[..])
                                    .read_u48::<BigEndian>()
                                    .expect("Could not read be u48 from 5 bytes");
                            }
                        }
                    } else if n_bytes == 6 {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            data[i + previous_index] = value
                                .read_u48::<LittleEndian>()
                                .expect("Could not read le u48");
                        }
                    } else {
                        // n_bytes = 5
                        let mut buf = [0u8; 6];
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            buf[0..5].copy_from_slice(&record[pos_byte_beg..pos_byte_beg + 5]);
                            data[i + previous_index] = Box::new(&buf[..])
                                .read_u48::<LittleEndian>()
                                .expect("Could not read le u48 from 5 bytes");
                        }
                    }
                }
                ChannelData::Int64(data) => {
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            data[i + previous_index] = i64::from_be_bytes(
                                value.try_into().expect("Could not read be i64"),
                            );
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            data[i + previous_index] = i64::from_le_bytes(
                                value.try_into().expect("Could not read le i64"),
                            );
                        }
                    }
                }
                ChannelData::UInt64(data) => {
                    if cn.endian {
                        if n_bytes == 8 {
                            for record in data_chunk.chunks(record_length) {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                data.push(Some(u64::from_be_bytes(
                                    value.try_into().expect("Could not read be u64"),
                                )));
                            }
                        } else {
                            // n_bytes = 7
                            let mut buf = [0u8; std::mem::size_of::<u64>()];
                            for record in data_chunk.chunks(record_length) {
                                buf[0..7].copy_from_slice(&record[pos_byte_beg..pos_byte_beg + 7]);
                                data.push(Some(u64::from_be_bytes(buf)));
                            }
                        }
                    } else if n_bytes == 8 {
                        for record in data_chunk.chunks(record_length) {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            data.push(Some(u64::from_le_bytes(
                                value.try_into().expect("Could not read le u64"),
                            )));
                        }
                    } else {
                        // n_bytes = 7, little endian
                        let mut buf = [0u8; std::mem::size_of::<u64>()];
                        for record in data_chunk.chunks(record_length) {
                            buf[0..7].copy_from_slice(&record[pos_byte_beg..pos_byte_beg + 7]);
                            data.push(Some(u64::from_le_bytes(buf)));
                        }
                    }
                }
                ChannelData::Float64(data) => {
                    if cn.endian {
                        for record in data_chunk.chunks(record_length) {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f64>()];
                            data.push(Some(f64::from_be_bytes(
                                value.try_into().expect("Could not read be f64"),
                            )));
                        }
                    } else {
                        for record in data_chunk.chunks(record_length) {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f64>()];
                            data.push(Some(f64::from_le_bytes(
                                value.try_into().expect("Could not read le f64"),
                            )));
                        }
                    }
                }
                ChannelData::Complex16(_) => {}
                ChannelData::Complex32(_) => {}
                ChannelData::Complex64(_) => {}
                ChannelData::StringSBC(data) => {
                    let n_bytes = cn.n_bytes as usize;
                    let mut decoder = WINDOWS_1252.new_decoder();
                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                        let (_result, _size, _replacement) =
                            decoder.decode_to_string(value, &mut data[i + previous_index], false);
                        data[i + previous_index] =
                            data[i + previous_index].trim_end_matches('\0').to_string();
                    }
                }
                ChannelData::StringUTF8(data) => {
                    let n_bytes = cn.n_bytes as usize;
                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                        data[i + previous_index] = str::from_utf8(value)
                            .expect("Found invalid UTF-8")
                            .trim_end_matches('\0')
                            .to_string();
                    }
                }
                ChannelData::StringUTF16(_) => {}
                ChannelData::VariableSizeByteArray(data) => {
                    let n_bytes = cn.n_bytes as usize;
                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                        data[i + previous_index] =
                            record[pos_byte_beg..pos_byte_beg + n_bytes].to_vec();
                    }
                }
                ChannelData::FixedSizeByteArray(data) => {
                    let n_bytes = cn.n_bytes as usize;
                    for record in data_chunk.chunks(record_length) {
                        data.0
                            .extend_from_slice(&record[pos_byte_beg..pos_byte_beg + n_bytes]);
                    }
                }
                ChannelData::ArrayDInt8(_) => {}
                ChannelData::ArrayDUInt8(_) => {}
                ChannelData::ArrayDInt16(_) => {}
                ChannelData::ArrayDUInt16(_) => {}
                ChannelData::ArrayDFloat16(_) => {}
                ChannelData::ArrayDInt24(_) => {}
                ChannelData::ArrayDUInt24(_) => {}
                ChannelData::ArrayDInt32(_) => {}
                ChannelData::ArrayDUInt32(_) => {}
                ChannelData::ArrayDFloat32(_) => {}
                ChannelData::ArrayDInt48(_) => {}
                ChannelData::ArrayDUInt48(_) => {}
                ChannelData::ArrayDInt64(_) => {}
                ChannelData::ArrayDUInt64(_) => {}
                ChannelData::ArrayDFloat64(_) => {}
                ChannelData::ArrayDComplex16(_) => {}
                ChannelData::ArrayDComplex32(_) => {}
                ChannelData::ArrayDComplex64(_) => {}
            }
            // channel was properly read
            cn.channel_data_valid = true;

            // Other channel types : virtual channels cn_type 3 & 6 are handled at initialisation
        });
}
