//! this module implements low level data reading for mdf4 files.
use super::channel_data::ChannelData;
use crate::mdfinfo::mdfinfo4::{Cn4, CnType, Compo};
use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use encoding_rs::{UTF_16BE, UTF_16LE, WINDOWS_1252};
use half::f16;
use log::warn;
use rayon::prelude::*;
use std::io::Cursor;
use std::string::String;
use std::{
    collections::HashSet,
    convert::TryInto,
    sync::{Arc, Mutex},
};
use arrow2::datatypes::DataType;
use arrow2::array::{PrimitiveArray, Utf8Array, BinaryArray, FixedSizeBinaryArray};
use std::{mem, str};

/// converts raw data block containing only one channel into a ndarray
pub fn read_one_channel_array(data_bytes: &mut Vec<u8>, cn: &mut Cn4, cycle_count: usize) {
    if (cn.block.cn_type == 0
        || cn.block.cn_type == 2
        || cn.block.cn_type == 4
        || cn.block.cn_type == 5)
        && !cn.data.is_empty()
    {
        // cn_type == 5 : Maximum length data channel, removing no valid bytes done by another size channel pointed by cn_data
        // cn_type == 0 : fixed length data channel
        // cn_type == 2 : master channel
        // cn_type == 4 : synchronisation channel
        let n_bytes = cn.n_bytes as usize;
        match &mut cn.data {
            ChannelData::Int8(data) => {
                let mut buf = vec![0; cycle_count];
                Cursor::new(data_bytes)
                    .read_i8_into(&mut buf)
                    .expect("Could not read i8 array");
                *data = buf;
            }
            ChannelData::UInt8(data) => {
                *data = mem::take(data_bytes);
            }
            ChannelData::Int16(data) => {
                let mut buf = vec![0; cycle_count];
                if cn.endian {
                    Cursor::new(data_bytes)
                        .read_i16_into::<BigEndian>(&mut buf)
                        .expect("Could not read be i16 array");
                } else {
                    Cursor::new(data_bytes)
                        .read_i16_into::<LittleEndian>(&mut buf)
                        .expect("Could not read le i16 array");
                }
                *data = buf;
            }
            ChannelData::UInt16(data) => {
                let mut buf = vec![0; cycle_count];
                if cn.endian {
                    Cursor::new(data_bytes)
                        .read_u16_into::<BigEndian>(&mut buf)
                        .expect("Could not read be u16 array");
                } else {
                    Cursor::new(data_bytes)
                        .read_u16_into::<LittleEndian>(&mut buf)
                        .expect("Could not read le 16 array");
                }
                *data = buf;
            }
            ChannelData::Float16(data) => {
                if cn.endian {
                    for (i, value) in data_bytes.chunks(std::mem::size_of::<f16>()).enumerate() {
                        data[i] =
                            f16::from_be_bytes(value.try_into().expect("Could not read be f16"))
                                .to_f32();
                    }
                } else {
                    for (i, value) in data_bytes.chunks(std::mem::size_of::<f16>()).enumerate() {
                        data[i] =
                            f16::from_le_bytes(value.try_into().expect("Could not read le f16"))
                                .to_f32();
                    }
                }
            }
            ChannelData::Int24(data) => {
                if cn.endian {
                    for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                        data[i] = value
                            .read_i24::<BigEndian>()
                            .expect("Could not read be i24");
                    }
                } else {
                    for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                        data[i] = value
                            .read_i24::<LittleEndian>()
                            .expect("Could not read le i24");
                    }
                }
            }
            ChannelData::UInt24(data) => {
                if cn.endian {
                    for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                        data[i] = value
                            .read_u24::<BigEndian>()
                            .expect("Could not read be u24");
                    }
                } else {
                    for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                        data[i] = value
                            .read_u24::<LittleEndian>()
                            .expect("Could not read le u24");
                    }
                }
            }
            ChannelData::Int32(data) => {
                let mut buf = vec![0; cycle_count];
                if cn.endian {
                    Cursor::new(data_bytes)
                        .read_i32_into::<BigEndian>(&mut buf)
                        .expect("Could not read be i32 array");
                } else {
                    Cursor::new(data_bytes)
                        .read_i32_into::<LittleEndian>(&mut buf)
                        .expect("Could not read le i32 array");
                }
                *data = buf;
            }
            ChannelData::UInt32(data) => {
                let mut buf = vec![0; cycle_count];
                if cn.endian {
                    Cursor::new(data_bytes)
                        .read_u32_into::<BigEndian>(&mut buf)
                        .expect("Could not read be u32 array");
                } else {
                    Cursor::new(data_bytes)
                        .read_u32_into::<LittleEndian>(&mut buf)
                        .expect("Could not read le u32 array");
                }
                *data = buf;
            }
            ChannelData::Float32(data) => {
                let mut buf = vec![0f32; cycle_count];
                if cn.endian {
                    Cursor::new(data_bytes)
                        .read_f32_into::<BigEndian>(&mut buf)
                        .expect("Could not read be f32 array");
                } else {
                    Cursor::new(data_bytes)
                        .read_f32_into::<LittleEndian>(&mut buf)
                        .expect("Could not read le f32 array");
                }
                *data = buf;
            }
            ChannelData::Int48(data) => {
                if cn.endian {
                    for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                        data[i] = value
                            .read_i48::<BigEndian>()
                            .expect("Could not read be i48");
                    }
                } else {
                    for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                        data[i] = value
                            .read_i48::<LittleEndian>()
                            .expect("Could not read le i48");
                    }
                }
            }
            ChannelData::UInt48(data) => {
                if cn.endian {
                    // big endian
                    if n_bytes == 6 {
                        for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                            data[i] = value
                                .read_u48::<BigEndian>()
                                .expect("Could not read be u48");
                        }
                    } else {
                        // n_bytes = 5
                        let mut temp = [0u8; 6];
                        for (i, value) in data_bytes.chunks(n_bytes).enumerate() {
                            temp[0..5].copy_from_slice(&value[0..n_bytes]);
                            data[i] = Cursor::new(temp)
                                .read_u48::<BigEndian>()
                                .expect("Could not read be u48 from 5 bytes");
                        }
                    }
                } else if n_bytes == 6 {
                    // little endian
                    for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                        data[i] = value
                            .read_u48::<LittleEndian>()
                            .expect("Could not read le u48");
                    }
                } else {
                    // n_bytes = 5
                    let mut temp = [0u8; 6];
                    for (i, value) in data_bytes.chunks(n_bytes).enumerate() {
                        temp[0..5].copy_from_slice(&value[0..n_bytes]);
                        data[i] = Cursor::new(temp)
                            .read_u48::<LittleEndian>()
                            .expect("Could not read le u48 from 5 bytes");
                    }
                }
            }
            ChannelData::Int64(data) => {
                let mut buf = vec![0; cycle_count];
                if cn.endian {
                    Cursor::new(data_bytes)
                        .read_i64_into::<BigEndian>(&mut buf)
                        .expect("Could not read be i64 array");
                } else {
                    Cursor::new(data_bytes)
                        .read_i64_into::<LittleEndian>(&mut buf)
                        .expect("Could not read le i64 array");
                }
                *data = buf;
            }
            ChannelData::UInt64(data) => {
                if n_bytes == 8 {
                    let mut buf = vec![0; cycle_count];
                    if cn.endian {
                        Cursor::new(data_bytes)
                            .read_u64_into::<BigEndian>(&mut buf)
                            .expect("Could not read be u64 array");
                    } else {
                        Cursor::new(data_bytes)
                            .read_u64_into::<LittleEndian>(&mut buf)
                            .expect("Could not read le u64 array");
                    }
                    *data = buf;
                } else {
                    // n_bytes = 7
                    let mut temp = [0u8; std::mem::size_of::<u64>()];
                    if cn.endian {
                        for (i, value) in data_bytes.chunks(n_bytes).enumerate() {
                            temp[0..7].copy_from_slice(&value[0..7]);
                            data[i] = u64::from_be_bytes(temp);
                        }
                    } else {
                        for (i, value) in data_bytes.chunks(n_bytes).enumerate() {
                            temp[0..7].copy_from_slice(&value[0..7]);
                            data[i] = u64::from_le_bytes(temp);
                        }
                    }
                }
            }
            ChannelData::Float64(data) => {
                let mut buf = vec![0f64; cycle_count];
                if cn.endian {
                    Cursor::new(data_bytes)
                        .read_f64_into::<BigEndian>(&mut buf)
                        .expect("Could not read be f64 array");
                } else {
                    Cursor::new(data_bytes)
                        .read_f64_into::<LittleEndian>(&mut buf)
                        .expect("Could not read le f64 array");
                }
                *data = buf;
            }
            ChannelData::Complex16(data) => {
                if cn.endian {
                    for (i, value) in data_bytes.chunks(std::mem::size_of::<f16>()).enumerate() {
                        data.0[i] = f16::from_be_bytes(
                            value.try_into().expect("Could not read be f16 complex"),
                        )
                        .to_f32();
                    }
                } else {
                    for (i, value) in data_bytes.chunks(std::mem::size_of::<f16>()).enumerate() {
                        data.0[i] = f16::from_le_bytes(
                            value.try_into().expect("Could not read le f16 complex"),
                        )
                        .to_f32();
                    }
                }
            }
            ChannelData::Complex32(data) => {
                if cn.endian {
                    for (i, value) in data_bytes.chunks(std::mem::size_of::<f32>()).enumerate() {
                        data.0[i] = f32::from_be_bytes(
                            value.try_into().expect("Could not read be f32 complex"),
                        );
                    }
                } else {
                    for (i, value) in data_bytes.chunks(std::mem::size_of::<f32>()).enumerate() {
                        data.0[i] = f32::from_le_bytes(
                            value.try_into().expect("Could not read le f32 complex"),
                        );
                    }
                }
            }
            ChannelData::Complex64(data) => {
                if cn.endian {
                    for (i, value) in data_bytes.chunks(std::mem::size_of::<f64>()).enumerate() {
                        data.0[i] = f64::from_be_bytes(
                            value.try_into().expect("Could not read be f64 complex"),
                        );
                    }
                } else {
                    for (i, value) in data_bytes.chunks(std::mem::size_of::<f64>()).enumerate() {
                        data.0[i] = f64::from_le_bytes(
                            value.try_into().expect("Could not read le f64 complex"),
                        );
                    }
                }
            }
            ChannelData::StringSBC(data) => {
                let mut decoder = WINDOWS_1252.new_decoder();
                for (i, value) in data_bytes.chunks(n_bytes).enumerate() {
                    let (_result, _size, _replacement) =
                        decoder.decode_to_string(value, &mut data[i], false);
                    data[i] = data[i].trim_end_matches('\0').to_string();
                }
            }
            ChannelData::StringUTF8(data) => {
                for (i, value) in data_bytes.chunks(n_bytes).enumerate() {
                    data[i] = str::from_utf8(value)
                        .expect("Found invalid UTF-8")
                        .trim_end_matches('\0')
                        .to_string();
                }
            }
            ChannelData::StringUTF16(data) => {
                if cn.endian {
                    let mut decoder = UTF_16BE.new_decoder();
                    for (i, value) in data_bytes.chunks(n_bytes).enumerate() {
                        let (_result, _size, _replacement) =
                            decoder.decode_to_string(value, &mut data[i], false);
                        data[i] = data[i].trim_end_matches('\0').to_string();
                    }
                } else {
                    let mut decoder = UTF_16LE.new_decoder();
                    for (i, value) in data_bytes.chunks(n_bytes).enumerate() {
                        let (_result, _size, _replacement) =
                            decoder.decode_to_string(value, &mut data[i], false);
                        data[i] = data[i].trim_end_matches('\0').to_string();
                    }
                }
            }
            ChannelData::VariableSizeByteArray(data) => {
                for (i, value) in data_bytes.chunks(n_bytes).enumerate() {
                    data[i] = value.to_vec();
                }
            }
            ChannelData::FixedSizeByteArray((data, _size)) => {
                *data = mem::take(data_bytes);
            }
            ChannelData::ArrayDInt8(data) => {
                if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(ca) => {
                            let mut buf = vec![0; cycle_count * ca.pnd];
                            Cursor::new(data_bytes)
                                .read_i8_into(&mut buf)
                                .expect("Could not read i8 array");
                            mem::swap(&mut buf, &mut data.0);
                        }
                        Compo::CN(_) => {}
                    }
                }
            }
            ChannelData::ArrayDUInt8(data) => {
                if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(_) => {
                            mem::swap(data_bytes, &mut data.0);
                        }
                        Compo::CN(_) => {}
                    }
                }
            }
            ChannelData::ArrayDInt16(data) => {
                if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(ca) => {
                            let mut buf = vec![0; cycle_count * ca.pnd];
                            if cn.endian {
                                Cursor::new(data_bytes)
                                    .read_i16_into::<BigEndian>(&mut buf)
                                    .expect("Could not read be i16 array");
                            } else {
                                Cursor::new(data_bytes)
                                    .read_i16_into::<LittleEndian>(&mut buf)
                                    .expect("Could not read le i16 array");
                            }
                            mem::swap(&mut buf, &mut data.0);
                        }
                        Compo::CN(_) => {}
                    }
                }
            }
            ChannelData::ArrayDUInt16(data) => {
                if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(ca) => {
                            let mut buf = vec![0; cycle_count * ca.pnd];
                            if cn.endian {
                                Cursor::new(data_bytes)
                                    .read_u16_into::<BigEndian>(&mut buf)
                                    .expect("Could not read be u16 array");
                            } else {
                                Cursor::new(data_bytes)
                                    .read_u16_into::<LittleEndian>(&mut buf)
                                    .expect("Could not read le 16 array");
                            }
                            mem::swap(&mut buf, &mut data.0);
                        }
                        Compo::CN(_) => {}
                    }
                }
            }
            ChannelData::ArrayDFloat16(data) => {
                if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(_) => {
                            if cn.endian {
                                for (i, value) in
                                    data_bytes.chunks(std::mem::size_of::<f16>()).enumerate()
                                {
                                    data.0[i] = f16::from_be_bytes(
                                        value.try_into().expect("Could not read be f16"),
                                    )
                                    .to_f32();
                                }
                            } else {
                                for (i, value) in
                                    data_bytes.chunks(std::mem::size_of::<f16>()).enumerate()
                                {
                                    data.0[i] = f16::from_le_bytes(
                                        value.try_into().expect("Could not read le f16"),
                                    )
                                    .to_f32();
                                }
                            }
                        }
                        Compo::CN(_) => {}
                    }
                }
            }
            ChannelData::ArrayDInt24(data) => {
                if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(_) => {
                            let mut temp = vec![0i32; data.0.len()]; // initialisation
                            if cn.endian {
                                for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                                    temp[i] = value
                                        .read_i24::<BigEndian>()
                                        .expect("Could not read be i24");
                                }
                            } else {
                                for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                                    temp[i] = value
                                        .read_i24::<LittleEndian>()
                                        .expect("Could not read le i24");
                                }
                            }
                            mem::swap(&mut temp, &mut data.0);
                        }
                        Compo::CN(_) => {}
                    }
                }
            }
            ChannelData::ArrayDUInt24(data) => {
                if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(_) => {
                            let mut temp = vec![0u32; data.0.len()]; // initialisation
                            if cn.endian {
                                for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                                    temp[i] = value
                                        .read_u24::<BigEndian>()
                                        .expect("Could not read be u24");
                                }
                            } else {
                                for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                                    temp[i] = value
                                        .read_u24::<LittleEndian>()
                                        .expect("Could not read le u24");
                                }
                            }
                            mem::swap(&mut temp, &mut data.0);
                        }
                        Compo::CN(_) => {}
                    }
                }
            }
            ChannelData::ArrayDInt32(data) => {
                if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(ca) => {
                            let mut buf = vec![0; cycle_count * ca.pnd];
                            if cn.endian {
                                Cursor::new(data_bytes)
                                    .read_i32_into::<BigEndian>(&mut buf)
                                    .expect("Could not read be i32 array");
                            } else {
                                Cursor::new(data_bytes)
                                    .read_i32_into::<LittleEndian>(&mut buf)
                                    .expect("Could not read le i32 array");
                            }
                            mem::swap(&mut buf, &mut data.0);
                        }
                        Compo::CN(_) => {}
                    }
                }
            }
            ChannelData::ArrayDUInt32(data) => {
                if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(ca) => {
                            let mut buf = vec![0; cycle_count * ca.pnd];
                            if cn.endian {
                                Cursor::new(data_bytes)
                                    .read_u32_into::<BigEndian>(&mut buf)
                                    .expect("Could not read be u32 array");
                            } else {
                                Cursor::new(data_bytes)
                                    .read_u32_into::<LittleEndian>(&mut buf)
                                    .expect("Could not read le u32 array");
                            }
                            mem::swap(&mut buf, &mut data.0);
                        }
                        Compo::CN(_) => {}
                    }
                }
            }
            ChannelData::ArrayDFloat32(data) => {
                if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(ca) => {
                            let mut buf = vec![0f32; cycle_count * ca.pnd];
                            if cn.endian {
                                Cursor::new(data_bytes)
                                    .read_f32_into::<BigEndian>(&mut buf)
                                    .expect("Could not read be f32 array");
                            } else {
                                Cursor::new(data_bytes)
                                    .read_f32_into::<LittleEndian>(&mut buf)
                                    .expect("Could not read le f32 array");
                            }
                            mem::swap(&mut buf, &mut data.0);
                        }
                        Compo::CN(_) => {}
                    }
                }
            }
            ChannelData::ArrayDInt48(data) => {
                if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(_) => {
                            if cn.endian {
                                for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                                    data.0[i] = value
                                        .read_i48::<BigEndian>()
                                        .expect("Could not read be i48");
                                }
                            } else {
                                for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                                    data.0[i] = value
                                        .read_i48::<LittleEndian>()
                                        .expect("Could not read le i48");
                                }
                            }
                        }
                        Compo::CN(_) => {}
                    }
                }
            }
            ChannelData::ArrayDUInt48(data) => {
                if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(_) => {
                            if cn.endian {
                                // big endian
                                if n_bytes == 6 {
                                    for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                                        data.0[i] = value
                                            .read_u48::<BigEndian>()
                                            .expect("Could not read be u48");
                                    }
                                } else {
                                    // n_bytes = 5
                                    let mut temp = [0u8; 6];
                                    for (i, value) in data_bytes.chunks(n_bytes).enumerate() {
                                        temp[0..5].copy_from_slice(&value[0..n_bytes]);
                                        data.0[i] = Cursor::new(temp)
                                            .read_u48::<BigEndian>()
                                            .expect("Could not read be u48 from 5 bytes");
                                    }
                                }
                            } else if n_bytes == 6 {
                                // little endian
                                for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                                    data.0[i] = value
                                        .read_u48::<LittleEndian>()
                                        .expect("Could not read le u48");
                                }
                            } else {
                                // n_bytes = 5
                                let mut temp = [0u8; 6];
                                for (i, value) in data_bytes.chunks(n_bytes).enumerate() {
                                    temp[0..5].copy_from_slice(&value[0..n_bytes]);
                                    data.0[i] = Cursor::new(temp)
                                        .read_u48::<LittleEndian>()
                                        .expect("Could not read le u48 from 5 bytes");
                                }
                            }
                        }
                        Compo::CN(_) => {}
                    }
                }
            }
            ChannelData::ArrayDInt64(data) => {
                if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(ca) => {
                            let mut buf = vec![0; cycle_count * ca.pnd];
                            if cn.endian {
                                Cursor::new(data_bytes)
                                    .read_i64_into::<BigEndian>(&mut buf)
                                    .expect("Could not read be i64 array");
                            } else {
                                Cursor::new(data_bytes)
                                    .read_i64_into::<LittleEndian>(&mut buf)
                                    .expect("Could not read le i64 array");
                            }
                            mem::swap(&mut buf, &mut data.0);
                        }
                        Compo::CN(_) => {}
                    }
                }
            }
            ChannelData::ArrayDUInt64(data) => {
                if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(ca) => {
                            if n_bytes == 8 {
                                let mut buf = vec![0; cycle_count * ca.pnd];
                                if cn.endian {
                                    Cursor::new(data_bytes)
                                        .read_u64_into::<BigEndian>(&mut buf)
                                        .expect("Could not read be u64 array");
                                } else {
                                    Cursor::new(data_bytes)
                                        .read_u64_into::<LittleEndian>(&mut buf)
                                        .expect("Could not read le u64 array");
                                }
                                mem::swap(&mut buf, &mut data.0);
                            } else {
                                // n_bytes = 7
                                let mut temp = [0u8; std::mem::size_of::<u64>()];
                                if cn.endian {
                                    for (i, value) in data_bytes.chunks(n_bytes).enumerate() {
                                        temp[0..7].copy_from_slice(&value[0..7]);
                                        data.0[i] = u64::from_be_bytes(temp);
                                    }
                                } else {
                                    for (i, value) in data_bytes.chunks(n_bytes).enumerate() {
                                        temp[0..7].copy_from_slice(&value[0..7]);
                                        data.0[i] = u64::from_le_bytes(temp);
                                    }
                                }
                            }
                        }
                        Compo::CN(_) => {}
                    }
                }
            }
            ChannelData::ArrayDFloat64(data) => {
                if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(ca) => {
                            let mut buf = vec![0f64; cycle_count * (ca.pnd)];
                            if cn.endian {
                                Cursor::new(data_bytes)
                                    .read_f64_into::<BigEndian>(&mut buf)
                                    .expect("Could not read be f64 array");
                            } else {
                                Cursor::new(data_bytes)
                                    .read_f64_into::<LittleEndian>(&mut buf)
                                    .expect("Could not read le f64 array");
                            }
                            mem::swap(&mut buf, &mut data.0);
                        }
                        Compo::CN(_) => {}
                    }
                }
            }
            ChannelData::ArrayDComplex16(data) => {
                if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(_) => {
                            if cn.endian {
                                for (i, value) in
                                    data_bytes.chunks(std::mem::size_of::<f16>()).enumerate()
                                {
                                    data.0 .0[i] = f16::from_be_bytes(
                                        value
                                            .try_into()
                                            .expect("Could not read be f16 complex array"),
                                    )
                                    .to_f32();
                                }
                            } else {
                                for (i, value) in
                                    data_bytes.chunks(std::mem::size_of::<f16>()).enumerate()
                                {
                                    data.0 .0[i] = f16::from_le_bytes(
                                        value
                                            .try_into()
                                            .expect("Could not read le f16 complex array"),
                                    )
                                    .to_f32();
                                }
                            }
                        }
                        Compo::CN(_) => {}
                    }
                }
            }
            ChannelData::ArrayDComplex32(data) => {
                if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(_) => {
                            if cn.endian {
                                for (i, value) in
                                    data_bytes.chunks(std::mem::size_of::<f32>()).enumerate()
                                {
                                    data.0 .0[i] = f32::from_be_bytes(
                                        value
                                            .try_into()
                                            .expect("Could not read be f32 complex array"),
                                    );
                                }
                            } else {
                                for (i, value) in
                                    data_bytes.chunks(std::mem::size_of::<f32>()).enumerate()
                                {
                                    data.0 .0[i] = f32::from_le_bytes(
                                        value
                                            .try_into()
                                            .expect("Could not read le f32 complex array"),
                                    );
                                }
                            }
                        }
                        Compo::CN(_) => {}
                    }
                }
            }
            ChannelData::ArrayDComplex64(data) => {
                if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(_) => {
                            if cn.endian {
                                for (i, value) in
                                    data_bytes.chunks(std::mem::size_of::<f64>()).enumerate()
                                {
                                    data.0 .0[i] = f64::from_be_bytes(
                                        value
                                            .try_into()
                                            .expect("Could not read be f64 complex array"),
                                    );
                                }
                            } else {
                                for (i, value) in
                                    data_bytes.chunks(std::mem::size_of::<f64>()).enumerate()
                                {
                                    data.0 .0[i] = f64::from_le_bytes(
                                        value
                                            .try_into()
                                            .expect("Could not read le f64 complex array"),
                                    );
                                }
                            }
                        }
                        Compo::CN(_) => {}
                    }
                }
            }
        }
        // channel was properly read
        cn.channel_data_valid = true;
    }
    // Other channel types : virtual channels cn_type 3 & 6 are handled at initialisation
    // cn_type == 1 VLSD not possible for sorted data
}

/// copies data from data_chunk into each channel array
pub fn read_channels_from_bytes(
    data_chunk: &[u8],
    channels: &mut CnType,
    record_length: usize,
    previous_index: usize,
    channel_names_to_read_in_dg: &HashSet<String>,
    record_with_invalid_data: bool,
) -> Vec<i32> {
    let vlsd_channels: Arc<Mutex<Vec<i32>>> = Arc::new(Mutex::new(Vec::new()));
    // iterates for each channel in parallel with rayon crate
    channels.par_iter_mut()
        .filter(|(_cn_record_position, cn)| {channel_names_to_read_in_dg.contains(&cn.unique_name) && !cn.data.is_empty() && !cn.channel_data_valid})
        .for_each(|(rec_pos, cn)| {
        if cn.block.cn_type == 0
            || cn.block.cn_type == 2
            || cn.block.cn_type == 4
            || cn.block.cn_type == 5
        {
            // cn_type == 5 : Maximum length data channel, removing no valid bytes done by another size channel pointed by cn_data
            // cn_type == 0 : fixed length data channel
            // cn_type == 2 : master channel
            // cn_type == 4 : synchronisation channel
            let mut value: &[u8]; // value of channel at record
            let pos_byte_beg = cn.pos_byte_beg as usize;
            let n_bytes = cn.n_bytes as usize;
            match cn.data.data_type() {
                DataType::Int8 => {
                    let mut data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<i8>>().unwrap().into_mut().unwrap_right();
                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i8>()];
                        data.set(i + previous_index, Some(i8::from_le_bytes(value.try_into().expect("Could not read i8"))));
                    }
                }
                DataType::UInt8 => {
                    let mut data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<u8>>().unwrap().into_mut().unwrap_right();
                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u8>()];
                        data.set(i + previous_index, Some(u8::from_le_bytes(value.try_into().expect("Could not read u8"))));
                    }
                }
                DataType::Int16 => {
                    let mut data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<i16>>().unwrap().into_mut().unwrap_right();
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i16>()];
                            data.set(i + previous_index, Some(i16::from_be_bytes(
                                value.try_into().expect("Could not read be i16"),
                            )));
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i16>()];
                            data.set(i + previous_index, Some(i16::from_le_bytes(
                                value.try_into().expect("Could not read le i16"),
                            )));
                        }
                    }
                }
                DataType::UInt16 => {
                    let mut data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<u16>>().unwrap().into_mut().unwrap_right();
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u16>()];
                                data.set(i + previous_index, Some(u16::from_be_bytes(
                                value.try_into().expect("Could not read be u16"),
                            )));
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u16>()];
                                data.set(i + previous_index, Some(u16::from_le_bytes(
                                value.try_into().expect("Could not read le u16"),
                            )));
                        }
                    }
                }
                DataType::Int32 => {
                    let mut data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<i32>>().unwrap().into_mut().unwrap_right();
                    if  n_bytes == 3 {
                        if cn.endian {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                data.set(i + previous_index, Some(value
                                    .read_i24::<BigEndian>()
                                    .expect("Could not read be i24")));
                            }
                        } else {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                data.set(i + previous_index, Some(value
                                    .read_i24::<LittleEndian>()
                                    .expect("Could not read le i24")));
                            }
                        }
                    } else {
                        if cn.endian {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value =
                                    &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i32>()];
                                data.set(i + previous_index, Some(i32::from_be_bytes(
                                    value.try_into().expect("Could not read be i32"),
                                )));
                            }
                        } else {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value =
                                    &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i32>()];
                                data.set(i + previous_index, Some(i32::from_le_bytes(
                                    value.try_into().expect("Could not read le i32"),
                                )));
                            }
                        }
                    }
                }
                DataType::UInt32 => {
                    let mut data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<u32>>().unwrap().into_mut().unwrap_right();
                    if n_bytes == 3 {
                        if cn.endian {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                data.set(i + previous_index, Some(value
                                    .read_u24::<BigEndian>()
                                    .expect("Could not read be u24")));
                            }
                        } else {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                data.set(i + previous_index, Some(value
                                    .read_u24::<LittleEndian>()
                                    .expect("Could not read le u24")));
                            }
                        }
                    } else {
                        if cn.endian {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value =
                                    &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u32>()];
                                data.set(i + previous_index, Some(u32::from_be_bytes(
                                    value.try_into().expect("Could not read be u32"),
                                )));
                            }
                        } else {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value =
                                    &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u32>()];
                                data.set(i + previous_index, Some(u32::from_le_bytes(
                                    value.try_into().expect("Could not read le u32"),
                                )));
                            }
                        }
                    }
                }
                DataType::Float32 => {
                    let mut data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<f32>>().unwrap().into_mut().unwrap_right();
                    if cn.endian {
                        if n_bytes == 2 {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value =
                                    &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f16>()];
                                    data.set(i + previous_index, Some(f16::from_be_bytes(
                                    value.try_into().expect("Could not read be f16"),
                                )
                                .to_f32()));
                            }
                        } else {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value =
                                    &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f32>()];
                                    data.set(i + previous_index, Some(f32::from_be_bytes(
                                    value.try_into().expect("Could not read be f32"),
                                )));
                            }
                        }
                    } else {
                        if n_bytes ==2 {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value =
                                    &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f16>()];
                                    data.set(i + previous_index, Some(f16::from_le_bytes(
                                    value.try_into().expect("Could not read le f16"),
                                )
                                .to_f32()));
                            }
                        } else {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value =
                                    &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f32>()];
                                data.set(i + previous_index, Some(f32::from_le_bytes(
                                    value.try_into().expect("Could not read le f32"),
                                )));
                            }
                        }
                    }
                }
                DataType::Int64 => {
                    let mut data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<i64>>().unwrap().into_mut().unwrap_right();
                    if cn.endian {
                        if n_bytes == 8 {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                data.set(i + previous_index, Some(i64::from_be_bytes(
                                    value.try_into().expect("Could not read be i64"),
                                )));
                            }
                        } else if n_bytes == 6 {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                data.set(i + previous_index, Some(value
                                    .read_i48::<BigEndian>()
                                    .expect("Could not read be i48")));
                            }
                        }
                    } else {
                        if n_bytes == 8 {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                data.set(i + previous_index, Some(i64::from_le_bytes(
                                    value.try_into().expect("Could not read le i64"),
                                )));
                            }
                        } else if n_bytes == 6 {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                data.set(i + previous_index, Some(value
                                    .read_i48::<LittleEndian>()
                                    .expect("Could not read le i48")));
                            }
                        }
                    }
                }
                DataType::UInt64 => {
                    let mut data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<u64>>().unwrap().into_mut().unwrap_right();
                    if cn.endian {
                        if n_bytes == 8 {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                data.set(i + previous_index, Some(u64::from_be_bytes(
                                    value.try_into().expect("Could not read be u64"),
                                )));
                            }
                        } else if n_bytes == 7 {
                            // n_bytes = 7
                            let mut buf = [0u8; std::mem::size_of::<u64>()];
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                buf[0..7].copy_from_slice(&record[pos_byte_beg..pos_byte_beg + 7]);
                                data.set(i + previous_index, Some(u64::from_be_bytes(buf)));
                            }
                        } else if n_bytes == 6 {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                data.set(i + previous_index, Some(value
                                    .read_u48::<BigEndian>()
                                    .expect("Could not read be u48")));
                            }
                        } else {
                            // n_bytes = 5
                            let mut buf = [0u8; 6];
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                buf[0..5]
                                    .copy_from_slice(&record[pos_byte_beg..pos_byte_beg + n_bytes]);
                                data.set(i + previous_index, Some(Cursor::new(buf)
                                    .read_u48::<BigEndian>()
                                    .expect("Could not read be u48 from 5 bytes")));
                            }
                        }
                    } else if n_bytes == 8 {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            data.set(i + previous_index, Some(u64::from_le_bytes(
                                value.try_into().expect("Could not read le u64"),
                            )));
                        }
                    } else  if n_bytes == 7 {
                        // n_bytes = 7, little endian
                        let mut buf = [0u8; std::mem::size_of::<u64>()];
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            buf[0..7].copy_from_slice(&record[pos_byte_beg..pos_byte_beg + 7]);
                            data.set(i + previous_index, Some(u64::from_le_bytes(buf)));
                        }
                    } else if n_bytes == 6 {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            data.set(i + previous_index, Some(value
                                .read_u48::<LittleEndian>()
                                .expect("Could not read le u48")));
                        }
                    } else {
                        // n_bytes = 5
                        let mut buf = [0u8; 6];
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            buf[0..5].copy_from_slice(&record[pos_byte_beg..pos_byte_beg + 5]);
                            data.set(i + previous_index, Some(Cursor::new(buf)
                                .read_u48::<LittleEndian>()
                                .expect("Could not read le u48 from 5 bytes")));
                        }
                    }
                }
                DataType::Float64 => {
                    let mut data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<f64>>().unwrap().into_mut().unwrap_right();
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f64>()];
                            data.set(i + previous_index, Some(f64::from_be_bytes(
                                    value.try_into().expect("Could not read be f64"),
                                )));
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f64>()];
                            data.set(i + previous_index, Some(f64::from_le_bytes(
                                    value.try_into().expect("Could not read le f64"),
                                )));
                        }
                    }
                }
                DataType::FixedSizeList(field, _size) => {
                    if field.name.eq(&"complex32".to_string()) {
                        let mut data = cn.data.as_any().downcast_ref::<PrimitiveArray<f32>>().unwrap().into_mut().unwrap_right();
                        if n_bytes <= 2 { 
                            // complex 16
                            let mut re_val: &[u8];
                            let mut im_val: &[u8];
                            if cn.endian {
                                for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                    re_val =
                                        &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f16>()];
                                    im_val = &record[pos_byte_beg + std::mem::size_of::<f16>()
                                        ..pos_byte_beg + 2 * std::mem::size_of::<f16>()];
                                    data.set(i*2 + previous_index, Some(f16::from_be_bytes(
                                        re_val
                                            .try_into()
                                            .expect("Could not read be real f16 complex"),
                                    )
                                    .to_f32()));
                                    data.set(i*2 + 1 + previous_index, Some(f16::from_be_bytes(
                                        im_val
                                            .try_into()
                                            .expect("Could not read be img f16 complex"),
                                    )
                                    .to_f32()));
                                }
                            } else {
                                for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                    re_val =
                                        &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f16>()];
                                    im_val = &record[pos_byte_beg + std::mem::size_of::<f16>()
                                        ..pos_byte_beg + 2 * std::mem::size_of::<f16>()];
                                    data.set(i*2 + previous_index, Some(f16::from_le_bytes(
                                        re_val
                                            .try_into()
                                            .expect("Could not read le real f16 complex"),
                                    )
                                    .to_f32()));
                                    data.set(i*2 + 1 + previous_index, Some(f16::from_le_bytes(
                                        im_val
                                            .try_into()
                                            .expect("Could not read le img f16 complex"),
                                    )
                                    .to_f32()));
                                }
                            }
                        } else if n_bytes <= 4 {
                            // complex 32
                            let mut re_val: &[u8];
                            let mut im_val: &[u8];
                            if cn.endian {
                                for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                    re_val =
                                        &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f32>()];
                                    im_val = &record[pos_byte_beg + std::mem::size_of::<f32>()
                                        ..pos_byte_beg + 2 * std::mem::size_of::<f32>()];
                                    data.set(i*2 + previous_index, Some(f32::from_be_bytes(
                                        re_val
                                            .try_into()
                                            .expect("Could not read be real f32 complex"),
                                    )));
                                    data.set(i*2 + 1 + previous_index, Some(f32::from_be_bytes(
                                        im_val
                                            .try_into()
                                            .expect("Could not read be img f32 complex"),
                                    )));
                                }
                            } else {
                                for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                    re_val =
                                        &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f32>()];
                                    im_val = &record[pos_byte_beg + std::mem::size_of::<f32>()
                                        ..pos_byte_beg + 2 * std::mem::size_of::<f32>()];
                                    data.set(i*2 + previous_index, Some(f32::from_le_bytes(
                                        re_val
                                            .try_into()
                                            .expect("Could not read le real f32 complex"),
                                    )));
                                    data.set(i*2 + 1 + previous_index, Some(f32::from_le_bytes(
                                        im_val
                                            .try_into()
                                            .expect("Could not read le img f32 complex"),
                                    )));
                                }
                            }
                        }
                    } else if field.name.eq(&"complex64".to_string())  { 
                        // complex 64
                        let mut re_val: &[u8];
                        let mut im_val: &[u8];
                        let mut data = cn.data.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap().into_mut().unwrap_right();
                        if cn.endian {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                re_val =
                                    &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f64>()];
                                im_val = &record[pos_byte_beg + std::mem::size_of::<f64>()
                                    ..pos_byte_beg + 2 * std::mem::size_of::<f64>()];
                                data.set(i*2 + previous_index, Some(f64::from_be_bytes(
                                    re_val
                                        .try_into()
                                        .expect("Could not read be real f64 complex"),
                                )));
                                data.set(i*2 + 1 + previous_index, Some(f64::from_be_bytes(
                                    im_val
                                        .try_into()
                                        .expect("Could not read be img f64 complex"),
                                )));
                            }
                        } else {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                re_val =
                                    &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f64>()];
                                im_val = &record[pos_byte_beg + std::mem::size_of::<f64>()
                                    ..pos_byte_beg + 2 * std::mem::size_of::<f64>()];
                                data.set(i*2 + previous_index, Some(f64::from_le_bytes(
                                    re_val
                                        .try_into()
                                        .expect("Could not read le real f64 complex"),
                                )));
                                data.set(i*2 + 1 + previous_index, Some(f64::from_le_bytes(
                                    im_val
                                        .try_into()
                                        .expect("Could not read le img f64 complex"),
                                )));
                            }
                        }
                    }
                }
                DataType::LargeUtf8 => {
                    let mut data = cn.data.as_any_mut().downcast_mut::<Utf8Array<i64>>().unwrap().into_mut().unwrap_right();
                    let n_bytes = cn.n_bytes as usize;
                    if cn.block.cn_data_type == 6 {
                        // SBC ISO-8859-1 to be converted into UTF8
                        let mut decoder = WINDOWS_1252.new_decoder();
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            let mut dst = String::new();
                            let (_result, _size, _replacement) =
                                decoder.decode_to_string(value, &mut dst, false);
                            data.push(Some(dst.trim_end_matches('\0').to_string()));
                        }
                    } else if cn.block.cn_data_type == 7 {
                        // 7: String UTF8
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            data.push(Some(str::from_utf8(value)
                                .expect("Found invalid UTF-8")
                                .trim_end_matches('\0')
                                .to_string()));
                        }
                    } else if cn.block.cn_data_type == 8 || cn.block.cn_data_type == 9 {
                        // 8 | 9 :String UTF16 to be converted into UTF8
                        if cn.endian {
                            let mut decoder = UTF_16BE.new_decoder();
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                let mut dst = String::new();
                                let (_result, _size, _replacement) = decoder.decode_to_string(
                                    value,
                                    &mut dst,
                                    false,
                                );
                                data.push(Some(dst.trim_end_matches('\0').to_string()));
                            }
                        } else {
                            let mut decoder = UTF_16LE.new_decoder();
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                let mut dst = String::new();
                                let (_result, _size, _replacement) = decoder.decode_to_string(
                                    value,
                                    &mut dst,
                                    false,
                                );
                                data.push(Some(dst.trim_end_matches('\0').to_string()));
                            }
                        }
                    }
                }
                DataType::LargeBinary => {
                    let n_bytes = cn.n_bytes as usize;
                    let mut data = cn.data.as_any_mut().downcast_mut::<BinaryArray<i64>>().unwrap().into_mut().unwrap_right();
                    for record in data_chunk.chunks(record_length) {
                        data.push(Some(record[pos_byte_beg..pos_byte_beg + n_bytes].to_vec()));
                    }
                }
                DataType::FixedSizeBinary(size) => {
                    let n_bytes = cn.n_bytes as usize;
                    let mut data = cn.data.as_any_mut().downcast_mut::<FixedSizeBinaryArray>().unwrap().values();
                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                        data[i*n_bytes+previous_index..(i+1)*n_bytes+previous_index].copy_from_slice(&record[pos_byte_beg..pos_byte_beg + n_bytes]);
                    }
                }
                DataType::Extension(extension_name, data_type, _) => {
                    if extension_name.eq(&"Tensor".to_string()) {
                        match **data_type {
                            DataType::Int8 => {
                                if let Some(compo) = &cn.composition {
                                    match &compo.block {
                                        Compo::CA(ca) => {
                                            let mut data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<i8>>().unwrap().get_mut_values().unwrap();
                                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                for j in 0..ca.pnd {
                                                    value = &record[pos_byte_beg + j * std::mem::size_of::<i8>()..pos_byte_beg + (j + 1) * std::mem::size_of::<i8>()];
                                                    data[(i + previous_index) * ca.pnd + j] =
                                                        i8::from_le_bytes(value.try_into().expect("Could not read i8 array"));
                                                }
                                            }
                                        }
                                        Compo::CN(_) => {},
                                    }
                                }
                            }
                            DataType::UInt8 => {
                                if let Some(compo) = &cn.composition {
                                    match &compo.block {
                                        Compo::CA(ca) => {
                                            let mut data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<u8>>().unwrap().get_mut_values().unwrap();
                                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                for j in 0..ca.pnd {
                                                    value = &record[pos_byte_beg + j * std::mem::size_of::<u8>()..pos_byte_beg + (j + 1) * std::mem::size_of::<u8>()];
                                                    data[(i + previous_index) * ca.pnd + j] =
                                                        u8::from_le_bytes(value.try_into().expect("Could not read u8 array"));
                                                }
                                            }
                                        }
                                        Compo::CN(_) => {},
                                    }
                                }
                            }
                            DataType::Int16 => {
                                if let Some(compo) = &cn.composition {
                                    match &compo.block {
                                        Compo::CA(ca) => {
                                            let mut data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<i16>>().unwrap().get_mut_values().unwrap();
                                            if cn.endian {
                                                for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                    for j in 0..ca.pnd {
                                                        value =
                                                            &record[pos_byte_beg + j * std::mem::size_of::<i16>()..pos_byte_beg + (j + 1) * std::mem::size_of::<i16>()];
                                                        data[(i + previous_index) * ca.pnd + j] = i16::from_be_bytes(
                                                            value.try_into().expect("Could not read be i16 array"),
                                                        );
                                                    }
                                                }
                                            } else {
                                                for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                    for j in 0..ca.pnd {
                                                        value =
                                                        &record[pos_byte_beg + j * std::mem::size_of::<i16>()..pos_byte_beg + (j + 1) * std::mem::size_of::<i16>()];
                                                        data[(i + previous_index) * ca.pnd + j] = i16::from_le_bytes(
                                                            value.try_into().expect("Could not read le i16 array"),
                                                        );
                                                    }
                                                }
                                            }
                                        }
                                        Compo::CN(_) => {},
                                    }
                                }
                            }
                            DataType::UInt16 => {
                                if let Some(compo) = &cn.composition {
                                    match &compo.block {
                                        Compo::CA(ca) => {
                                            let mut data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<u16>>().unwrap().get_mut_values().unwrap();
                                            if cn.endian {
                                                for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                    for j in 0..ca.pnd {
                                                        value =
                                                            &record[pos_byte_beg + j * std::mem::size_of::<u16>()..pos_byte_beg + (j + 1) * std::mem::size_of::<u16>()];
                                                        data[(i + previous_index) * ca.pnd + j] = u16::from_be_bytes(
                                                            value.try_into().expect("Could not read be u16 array"),
                                                        );
                                                    }
                                                }
                                            } else {
                                                for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                    for j in 0..ca.pnd {
                                                        value =
                                                        &record[pos_byte_beg + j * std::mem::size_of::<i16>()..pos_byte_beg + (j + 1) * std::mem::size_of::<u16>()];
                                                        data[(i + previous_index) * ca.pnd + j] = u16::from_le_bytes(
                                                            value.try_into().expect("Could not read le u16 array"),
                                                        );
                                                    }
                                                }
                                            }
                                        }
                                        Compo::CN(_) => {},
                                    }
                                }
                            }
                            DataType::Int32 => {
                                if let Some(compo) = &cn.composition {
                                    match &compo.block {
                                        Compo::CA(ca) => {
                                            let mut data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<i32>>().unwrap().get_mut_values().unwrap();
                                            if cn.endian {
                                                if n_bytes <=3 {
                                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                        for j in 0..ca.pnd {
                                                            value = &record[pos_byte_beg + j * n_bytes..pos_byte_beg + (j + 1) * n_bytes];
                                                            data[(i + previous_index) * ca.pnd + j] = value
                                                                .read_i24::<BigEndian>()
                                                                .expect("Could not read be i24 array");
                                                        }
                                                    }
                                                }
                                                else {
                                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                        for j in 0..ca.pnd {
                                                            value =
                                                            &record[pos_byte_beg + j * std::mem::size_of::<i32>()..pos_byte_beg + (j + 1) * std::mem::size_of::<i32>()];
                                                            data[(i + previous_index) * ca.pnd + j] = i32::from_be_bytes(
                                                                value.try_into().expect("Could not read be i32 array"),
                                                            );
                                                        }
                                                    }
                                                }
                                            } else {
                                                if n_bytes <=3 {
                                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                        for j in 0..ca.pnd {
                                                            value = &record[pos_byte_beg + j * n_bytes..pos_byte_beg + (j + 1) * n_bytes];
                                                            data[(i + previous_index) * ca.pnd + j] = value
                                                                .read_i24::<LittleEndian>()
                                                                .expect("Could not read le i24 array");
                                                        }
                                                    }
                                                }
                                                else {
                                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                        for j in 0..ca.pnd {
                                                            value =
                                                            &record[pos_byte_beg + j * std::mem::size_of::<i32>()..pos_byte_beg + (j + 1) * std::mem::size_of::<i32>()];
                                                            data[(i + previous_index) * ca.pnd + j] = i32::from_le_bytes(
                                                                value.try_into().expect("Could not read le i32 array"),
                                                            );
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        Compo::CN(_) => {},
                                    }
                                }
                            }
                            DataType::UInt32 => {
                                if let Some(compo) = &cn.composition {
                                    match &compo.block {
                                        Compo::CA(ca) => {
                                            let mut data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<u32>>().unwrap().get_mut_values().unwrap();
                                            if cn.endian {
                                                if n_bytes <=3 {
                                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                        for j in 0..ca.pnd {
                                                            value = &record[pos_byte_beg + j * n_bytes..pos_byte_beg + (j + 1) * n_bytes];
                                                            data[(i + previous_index) * ca.pnd + j] = value
                                                                .read_u24::<BigEndian>()
                                                                .expect("Could not read be u24 array");
                                                        }
                                                    }
                                                }
                                                else {
                                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                        for j in 0..ca.pnd {
                                                            value =
                                                            &record[pos_byte_beg + j * std::mem::size_of::<u32>()..pos_byte_beg + (j + 1) * std::mem::size_of::<u32>()];
                                                            data[(i + previous_index) * ca.pnd + j] = u32::from_be_bytes(
                                                                value.try_into().expect("Could not read be u32 array"),
                                                            );
                                                        }
                                                    }
                                                }
                                            } else {
                                                if n_bytes <=3 {
                                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                        for j in 0..ca.pnd {
                                                            value = &record[pos_byte_beg + j * n_bytes..pos_byte_beg + (j + 1) * n_bytes];
                                                            data[(i + previous_index) * ca.pnd + j] = value
                                                                .read_u24::<LittleEndian>()
                                                                .expect("Could not read le u24 array");
                                                        }
                                                    }
                                                }else {
                                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                        for j in 0..ca.pnd {
                                                            value =
                                                            &record[pos_byte_beg + j * std::mem::size_of::<u32>()..pos_byte_beg + (j + 1) * std::mem::size_of::<u32>()];
                                                            data[(i + previous_index) * ca.pnd + j] = u32::from_le_bytes(
                                                                value.try_into().expect("Could not read le u32 array"),
                                                            );
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        Compo::CN(_) => {},
                                    }
                                }
                            }
                            DataType::Float32 => {
                                if let Some(compo) = &cn.composition {
                                    match &compo.block {
                                        Compo::CA(ca) => {
                                            let mut data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<f32>>().unwrap().get_mut_values().unwrap();
                                            if cn.endian {
                                                if n_bytes <= 2 {
                                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                        for j in 0..ca.pnd {
                                                            value =
                                                            &record[pos_byte_beg + j * std::mem::size_of::<f16>()..pos_byte_beg + (j + 1) * std::mem::size_of::<f16>()];
                                                            data[(i + previous_index) * ca.pnd + j] = f16::from_be_bytes(
                                                                value.try_into().expect("Could not read be f16 array"),
                                                            )
                                                            .to_f32();
                                                        }
                                                    }
                                                }
                                                else {
                                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                        for j in 0..ca.pnd {
                                                            value =
                                                            &record[pos_byte_beg + j * std::mem::size_of::<f32>()..pos_byte_beg + (j + 1) * std::mem::size_of::<f32>()];
                                                            data[(i + previous_index) * ca.pnd + j] = f32::from_be_bytes(
                                                                value.try_into().expect("Could not read be f32 array"),
                                                            );
                                                        }
                                                    }
                                                }
                                            } else {
                                                if n_bytes <= 2 {
                                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                        for j in 0..ca.pnd {
                                                            value =
                                                            &record[pos_byte_beg + j * std::mem::size_of::<f16>()..pos_byte_beg + (j + 1) * std::mem::size_of::<f16>()];
                                                            data[(i + previous_index) * ca.pnd + j] = f16::from_le_bytes(
                                                                value.try_into().expect("Could not read le f16 array"),
                                                            )
                                                            .to_f32();
                                                        }
                                                    }
                                                }
                                                else {
                                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                        for j in 0..ca.pnd {
                                                            value =
                                                            &record[pos_byte_beg + j * std::mem::size_of::<f32>()..pos_byte_beg + (j + 1) * std::mem::size_of::<f32>()];
                                                            data[(i + previous_index) * ca.pnd + j] = f32::from_le_bytes(
                                                                value.try_into().expect("Could not read le f32 array"),
                                                            );
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        Compo::CN(_) => {},
                                    }
                                }
                            }
                            DataType::Int64 => {
                                if let Some(compo) = &cn.composition {
                                    match &compo.block {
                                        Compo::CA(ca) => {
                                            let mut data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<i64>>().unwrap().get_mut_values().unwrap();
                                            if cn.endian {
                                                if n_bytes == 8 {
                                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                        for j in 0..ca.pnd {
                                                            value = &record[pos_byte_beg + j * n_bytes..pos_byte_beg + (j + 1) * n_bytes];
                                                            data[(i + previous_index) * ca.pnd + j] = i64::from_be_bytes(
                                                                value.try_into().expect("Could not read be i64 array"),
                                                            );
                                                        }
                                                    }
                                                } else if n_bytes == 6 {
                                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                        for j in 0..ca.pnd {
                                                            value = &record[pos_byte_beg + j * n_bytes..pos_byte_beg + (j + 1) * n_bytes];
                                                            data[(i + previous_index) * ca.pnd + j] = value
                                                                .read_i48::<BigEndian>()
                                                                .expect("Could not read be i48 array");
                                                        }
                                                    }
                                                }
                                            } else {
                                                if n_bytes == 8 {
                                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                        for j in 0..ca.pnd {
                                                            value = &record[pos_byte_beg + j * n_bytes..pos_byte_beg + (j + 1) * n_bytes];
                                                            data[(i + previous_index) * ca.pnd + j] = i64::from_le_bytes(
                                                                value.try_into().expect("Could not read le i64 array"),
                                                            );
                                                        }
                                                    }
                                                } else if n_bytes == 6 {
                                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                        for j in 0..ca.pnd {
                                                            value = &record[pos_byte_beg + j * n_bytes..pos_byte_beg + (j + 1) * n_bytes];
                                                            data[(i + previous_index) * ca.pnd + j] = value
                                                                .read_i48::<LittleEndian>()
                                                                .expect("Could not read le i48 array");
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        Compo::CN(_) => {},
                                    }
                                }
                            }
                            DataType::UInt64 => {
                                if let Some(compo) = &cn.composition {
                                    match &compo.block {
                                        Compo::CA(ca) => {
                                            let mut data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<u64>>().unwrap().get_mut_values().unwrap();
                                            if cn.endian {
                                                if n_bytes == 8 {
                                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                        for j in 0..ca.pnd {
                                                            value = &record[pos_byte_beg + j * n_bytes..pos_byte_beg + (j + 1) * n_bytes];
                                                            data[(i + previous_index) * ca.pnd + j] = u64::from_le_bytes(
                                                                value.try_into().expect("Could not read be u64 array"),
                                                            );
                                                        }
                                                    }
                                                } else if n_bytes == 7 {
                                                    let mut buf = [0u8; std::mem::size_of::<u64>()];
                                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                        for j in 0..ca.pnd {
                                                            buf[0..7].copy_from_slice(&record[pos_byte_beg + j * n_bytes..pos_byte_beg + (j + 1) * n_bytes]);
                                                            data[(i + previous_index) * ca.pnd + j] = u64::from_le_bytes(buf);
                                                        }
                                                    }
                                                } else if n_bytes == 6 {
                                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                        for j in 0..ca.pnd {
                                                            value = &record[pos_byte_beg + j * n_bytes..pos_byte_beg + (j + 1) * n_bytes];
                                                            data[(i + previous_index) * ca.pnd + j] = value
                                                                .read_u48::<BigEndian>()
                                                                .expect("Could not read be u48 array");
                                                        }
                                                    }
                                                } else if n_bytes == 5 {
                                                    let mut buf = [0u8; 6];
                                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                        for j in 0..ca.pnd {
                                                            buf[0..5]
                                                                .copy_from_slice(&record[pos_byte_beg + j * n_bytes..pos_byte_beg + (j + 1) * n_bytes]);
                                                            data[(i + previous_index) * ca.pnd + j] = Cursor::new(buf)
                                                                .read_u48::<BigEndian>()
                                                                .expect("Could not read be u48 from 5 bytes in array");
                                                        }
                                                    }
                                                }
                                            } else if n_bytes == 8 {
                                                for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                    for j in 0..ca.pnd {
                                                        value = &record[pos_byte_beg + j * n_bytes..pos_byte_beg + (j + 1) * n_bytes];
                                                        data[(i + previous_index) * ca.pnd + j] = u64::from_le_bytes(
                                                            value.try_into().expect("Could not read le u64"),
                                                        );
                                                    }
                                                }
                                            } else if n_bytes == 7 {
                                                let mut buf = [0u8; std::mem::size_of::<u64>()];
                                                for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                    for j in 0..ca.pnd {
                                                        buf[0..7].copy_from_slice(&record[pos_byte_beg + j * n_bytes..pos_byte_beg + (j + 1) * n_bytes]);
                                                        data[(i + previous_index) * ca.pnd + j] = u64::from_le_bytes(buf);
                                                    }
                                                }
                                            } else if n_bytes == 6 {
                                                for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                    for j in 0..ca.pnd {
                                                        value = &record[pos_byte_beg + j * n_bytes..pos_byte_beg + (j + 1) * n_bytes];
                                                        data[(i + previous_index) * ca.pnd + j] = value
                                                            .read_u48::<LittleEndian>()
                                                            .expect("Could not read le u48 array");
                                                    }
                                                }
                                            } else {
                                                // n_bytes = 5
                                                let mut buf = [0u8; 6];
                                                for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                    for j in 0..ca.pnd {
                                                        buf[0..5].copy_from_slice(&record[pos_byte_beg + j * n_bytes..pos_byte_beg + (j + 1) * n_bytes]);
                                                        data[(i + previous_index) * ca.pnd + j] = Cursor::new(buf)
                                                            .read_u48::<LittleEndian>()
                                                            .expect("Could not read le u48 from 5 bytes in array");
                                                    }
                                                }
                                            }
                                        }
                                        Compo::CN(_) => {},
                                    }
                                }
                            }
                            DataType::Float64 => {
                                if let Some(compo) = &cn.composition {
                                    match &compo.block {
                                        Compo::CA(ca) => {
                                            let mut data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<f64>>().unwrap().get_mut_values().unwrap();
                                            if cn.endian {
                                                for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                    for j in 0..ca.pnd {
                                                        value = &record[pos_byte_beg + j * std::mem::size_of::<u64>()..pos_byte_beg + (j + 1) * std::mem::size_of::<u64>()];
                                                        data[(i + previous_index) * ca.pnd + j] = f64::from_be_bytes(
                                                            value.try_into().expect("Could not read be f64"),
                                                        );
                                                    }
                                                }
                                            } else {
                                                for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                    for j in 0..ca.pnd {
                                                        value = &record[pos_byte_beg + j * std::mem::size_of::<u64>()..pos_byte_beg + (j + 1) * std::mem::size_of::<u64>()];
                                                        data[(i + previous_index) * ca.pnd + j] = f64::from_le_bytes(
                                                            value.try_into().expect("Could not read le f64"),
                                                        );
                                                    }
                                                }
                                            }
                                        }
                                        Compo::CN(_) => {},
                                    }
                                }
                            }
                            DataType::FixedSizeList(field, _size) => {
                                if field.name.eq(&"complex32".to_string()) {
                                    let mut data = cn.data.as_any().downcast_ref::<PrimitiveArray<f32>>().unwrap().into_mut().unwrap_right().values();
                                    if n_bytes <= 2 { 
                                        // complex 16
                                        if let Some(compo) = &cn.composition {
                                            match &compo.block {
                                                Compo::CA(ca) => {
                                                    let mut re_val: &[u8];
                                                    let mut im_val: &[u8];
                                                    if cn.endian {
                                                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                            for j in 0..ca.pnd {
                                                                re_val =
                                                                &record[pos_byte_beg + 2 * j * std::mem::size_of::<f16>()
                                                                    ..pos_byte_beg + (j + 1) * 2 * std::mem::size_of::<f16>()];
                                                                im_val = &record[pos_byte_beg + (j + 1) * 2 * std::mem::size_of::<f16>()
                                                                    ..pos_byte_beg + (j + 2) * 2 * std::mem::size_of::<f16>()];
                                                                data[(i*2 + previous_index) * ca.pnd + j] = f16::from_be_bytes(
                                                                    re_val
                                                                        .try_into()
                                                                        .expect("Could not read be real f16 complex array"),
                                                                ).to_f32();
                                                                data[(i*2 + previous_index) * ca.pnd + j + 1] = f16::from_be_bytes(
                                                                    im_val
                                                                        .try_into()
                                                                        .expect("Could not read be img f16 complex array"),
                                                                ).to_f32();
                                                            }
                                                        }
                                                    } else {
                                                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                            for j in 0..ca.pnd {
                                                                re_val =
                                                                &record[pos_byte_beg + 2 * j * std::mem::size_of::<f16>()
                                                                    ..pos_byte_beg + (j + 1) * 2 * std::mem::size_of::<f16>()];
                                                                im_val = &record[pos_byte_beg + (j + 1) * 2 * std::mem::size_of::<f16>()
                                                                    ..pos_byte_beg + (j + 2) * 2 * std::mem::size_of::<f16>()];
                                                                data[(i*2 + previous_index) * ca.pnd + j] = f16::from_le_bytes(
                                                                    re_val
                                                                        .try_into()
                                                                        .expect("Could not read le real f16 complex array"),
                                                                ).to_f32();
                                                                data[(i*2 + previous_index) * ca.pnd + j + 1] = f16::from_le_bytes(
                                                                    im_val
                                                                        .try_into()
                                                                        .expect("Could not read le img f16 complex array"),
                                                                ).to_f32();
                                                            }
                                                        }
                                                    }
                                                }
                                                Compo::CN(_) => {},
                                            }
                                        }
                                    } else if n_bytes <= 4 {
                                        if let Some(compo) = &cn.composition {
                                            match &compo.block {
                                                Compo::CA(ca) => {
                                                    let mut re_val: &[u8];
                                                    let mut im_val: &[u8];
                                                    if cn.endian {
                                                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                            for j in 0..ca.pnd {
                                                                re_val =
                                                                    &record[pos_byte_beg + 2 * j * std::mem::size_of::<f32>()
                                                                        ..pos_byte_beg + (j + 1) * 2 * std::mem::size_of::<f32>()];
                                                                im_val = &record[pos_byte_beg + (j + 1) * 2 * std::mem::size_of::<f32>()
                                                                    ..pos_byte_beg + (j + 2) * 2 * std::mem::size_of::<f32>()];
                                                                data[(i*2 + previous_index) * ca.pnd + j] = f32::from_be_bytes(
                                                                    re_val
                                                                        .try_into()
                                                                        .expect("Could not read be real f32 complex array"),
                                                                );
                                                                data[(i*2 + previous_index) * ca.pnd + j + 1] = f32::from_be_bytes(
                                                                    im_val
                                                                        .try_into()
                                                                        .expect("Could not read be img f32 complex array"),
                                                                );
                                                            }
                                                        }
                                                    } else {
                                                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                            for j in 0..ca.pnd {
                                                                re_val = &record[pos_byte_beg + 2 * j * std::mem::size_of::<f32>()
                                                                    ..pos_byte_beg + (j + 1) * 2 * std::mem::size_of::<f32>()];
                                                                im_val = &record[pos_byte_beg + (j + 1) * 2 * std::mem::size_of::<f32>()
                                                                    ..pos_byte_beg + (j + 2) * 2 * std::mem::size_of::<f32>()];
                                                                data[(i*2 + previous_index) * ca.pnd + j] = f32::from_le_bytes(
                                                                    re_val
                                                                        .try_into()
                                                                        .expect("Could not read le real f32 complex array"),
                                                                );
                                                                data[(i*2 + previous_index) * ca.pnd + j + 1] = f32::from_le_bytes(
                                                                    im_val
                                                                        .try_into()
                                                                        .expect("Could not read le img f32 complex array"),
                                                                );
                                                            }
                                                        }
                                                    }
                                                }
                                                Compo::CN(_) => {},
                                            }
                                        }
                                    }
                                } else if field.name.eq(&"complex64".to_string())  { 
                                    if let Some(compo) = &cn.composition {
                                        match &compo.block {
                                            Compo::CA(ca) => {
                                                let mut data = cn.data.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap().into_mut().unwrap_right().values();
                                                let mut re_val: &[u8];
                                                let mut im_val: &[u8];
                                                if cn.endian {
                                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                        for j in 0..ca.pnd {
                                                            re_val = &record[pos_byte_beg + 2 * j * std::mem::size_of::<f64>()
                                                                ..pos_byte_beg + (j + 1) * 2 * std::mem::size_of::<f64>()];
                                                            im_val = &record[pos_byte_beg + (j + 1) * 2 * std::mem::size_of::<f64>()
                                                                ..pos_byte_beg + (j + 2) * 2 * std::mem::size_of::<f64>()];
                                                            data[(i*2 + previous_index) * ca.pnd + j] = f64::from_be_bytes(
                                                                re_val
                                                                    .try_into()
                                                                    .expect("Could not read be real f64 complex"),
                                                            );
                                                            data[(i*2 + previous_index) * ca.pnd + j + 1] = f64::from_be_bytes(
                                                                im_val
                                                                    .try_into()
                                                                    .expect("Could not read be img f64 complex"),
                                                            );
                                                        }
                                                    }
                                                } else {
                                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                                        for j in 0..ca.pnd {
                                                            re_val = &record[pos_byte_beg + 2 * j * std::mem::size_of::<f64>()
                                                                ..pos_byte_beg + (j + 1) * 2 * std::mem::size_of::<f64>()];
                                                            im_val = &record[pos_byte_beg + (j + 1) * 2 * std::mem::size_of::<f64>()
                                                                ..pos_byte_beg + (j + 2) * 2 * std::mem::size_of::<f64>()];
                                                            data[(i*2 + previous_index) * ca.pnd + j] = f64::from_le_bytes(
                                                                re_val
                                                                    .try_into()
                                                                    .expect("Could not read le real f64 complex array"),
                                                            );
                                                            data[(i*2 + previous_index) * ca.pnd + j + 1] = f64::from_le_bytes(
                                                                im_val
                                                                    .try_into()
                                                                    .expect("Could not read le img f64 complex array"),
                                                            );
                                                        }
                                                    }
                                                }
                                            }
                                            Compo::CN(_) => todo!(),
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                _ => warn!("unrecognised data type at data chunk reading from channel {}", cn.unique_name)
            }
        } else if cn.block.cn_type == 1 {
            // SD Block attached as data block is sorted
            if cn.block.cn_data != 0 {
                let c_vlsd_channel = Arc::clone(&vlsd_channels);
                let mut vlsd_channel = c_vlsd_channel
                    .lock()
                    .expect("Could not get lock from vlsd channel arc vec");
                vlsd_channel.push(*rec_pos);
            }
        }
        // Other channel types : virtual channels cn_type 3 & 6 are handled at initialisation
        if record_with_invalid_data {
            // invalidation bits to store in bitmap.
            if let Some((mask, invalid_byte_position, invalid_byte_mask)) = &mut cn.invalid_mask {
                for (i, record) in data_chunk.chunks(record_length).enumerate() {
                    mask.set(i + previous_index, (*invalid_byte_mask & record[*invalid_byte_position]) == 0);
                }
            };
        }
    });

    let lock = vlsd_channels
        .lock()
        .expect("Could not get lock from vlsd channel arc vec");
    lock.clone()
}
