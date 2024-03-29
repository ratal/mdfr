//! this module implements low level data reading for mdf4 files.
use crate::export::tensor::{Order, Tensor};
use crate::mdfinfo::mdfinfo4::{Cn4, CnType, Compo};
use anyhow::{Context, Error, Ok, Result};
use arrow2::array::{MutableBinaryValuesArray, MutableFixedSizeBinaryArray, PrimitiveArray};
use arrow2::datatypes::DataType;
use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use encoding_rs::{UTF_16BE, UTF_16LE, WINDOWS_1252};
use half::f16;
use rayon::prelude::*;
use std::io::Cursor;
use std::str;
use std::string::String;
use std::{
    collections::HashSet,
    convert::TryInto,
    sync::{Arc, Mutex},
};

use super::channel_data::ChannelData;

/// converts raw data block containing only one channel into a ndarray
pub fn read_one_channel_array(
    data_bytes: &Vec<u8>,
    cn: &mut Cn4,
    cycle_count: usize,
    shape: Option<(Vec<usize>, Order)>,
) -> Result<(), Error> {
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
            ChannelData::Int8(a) => {
                let mut buf = vec![0; cycle_count];
                Cursor::new(data_bytes)
                    .read_i8_into(&mut buf)
                    .context("Could not read i8 array")?;
                *a = PrimitiveArray::from_vec(buf);
            }
            ChannelData::UInt8(a) => {
                *a = PrimitiveArray::from_vec(data_bytes.clone());
            }
            ChannelData::Int16(a) => {
                let mut buf = vec![0; cycle_count];
                if cn.endian {
                    Cursor::new(data_bytes)
                        .read_i16_into::<BigEndian>(&mut buf)
                        .context("Could not read be i16 array")?;
                } else {
                    Cursor::new(data_bytes)
                        .read_i16_into::<LittleEndian>(&mut buf)
                        .context("Could not read le i16 array")?;
                }
                *a = PrimitiveArray::from_vec(buf);
            }
            ChannelData::UInt16(a) => {
                let mut buf = vec![0; cycle_count];
                if cn.endian {
                    Cursor::new(data_bytes)
                        .read_u16_into::<BigEndian>(&mut buf)
                        .context("Could not read be u16 array")?;
                } else {
                    Cursor::new(data_bytes)
                        .read_u16_into::<LittleEndian>(&mut buf)
                        .context("Could not read le 16 array")?;
                }
                *a = PrimitiveArray::from_vec(buf);
            }
            ChannelData::Int32(a) => {
                if n_bytes == 4 {
                    let mut buf = vec![0; cycle_count];
                    if cn.endian {
                        Cursor::new(data_bytes)
                            .read_i32_into::<BigEndian>(&mut buf)
                            .context("Could not read be i32 array")?;
                    } else {
                        Cursor::new(data_bytes)
                            .read_i32_into::<LittleEndian>(&mut buf)
                            .context("Could not read le i32 array")?;
                    }
                    *a = PrimitiveArray::from_vec(buf);
                } else if n_bytes == 3 {
                    let data = a.get_mut_values()
                    .context("One channel array function could not get mutable values from primitive array i32")?;
                    if cn.endian {
                        for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                            data[i] = value
                                .read_i24::<BigEndian>()
                                .context("Could not read be i24")?;
                        }
                    } else {
                        for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                            data[i] = value
                                .read_i24::<LittleEndian>()
                                .context("Could not read le i24")?;
                        }
                    }
                }
            }
            ChannelData::UInt32(a) => {
                if n_bytes == 4 {
                    let mut buf = vec![0; cycle_count];
                    if cn.endian {
                        Cursor::new(data_bytes)
                            .read_u32_into::<BigEndian>(&mut buf)
                            .context("Could not read be u32 array")?;
                    } else {
                        Cursor::new(data_bytes)
                            .read_u32_into::<LittleEndian>(&mut buf)
                            .context("Could not read le u32 array")?;
                    }
                    *a = PrimitiveArray::from_vec(buf);
                } else if n_bytes == 3 {
                    let data = a.get_mut_values()
                    .context("One channel array function could not get mutable values from primitive array u32")?;
                    if cn.endian {
                        for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                            data[i] = value
                                .read_u24::<BigEndian>()
                                .context("Could not read be u24")?;
                        }
                    } else {
                        for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                            data[i] = value
                                .read_u24::<LittleEndian>()
                                .context("Could not read le u24")?;
                        }
                    }
                }
            }
            ChannelData::Float32(a) => {
                if n_bytes == 4 {
                    let mut buf = vec![0f32; cycle_count];
                    if cn.endian {
                        Cursor::new(data_bytes)
                            .read_f32_into::<BigEndian>(&mut buf)
                            .context("Could not read be f32 array")?;
                    } else {
                        Cursor::new(data_bytes)
                            .read_f32_into::<LittleEndian>(&mut buf)
                            .context("Could not read le f32 array")?;
                    }
                    *a = PrimitiveArray::from_vec(buf);
                } else if n_bytes == 2 {
                    let data = a.get_mut_values().context("One channel array function could not get mutable values from primitive array float32")?;
                    if cn.endian {
                        for (i, value) in data_bytes.chunks(std::mem::size_of::<f16>()).enumerate()
                        {
                            data[i] = f16::from_be_bytes(
                                value.try_into().context("Could not read be f16")?,
                            )
                            .to_f32();
                        }
                    } else {
                        for (i, value) in data_bytes.chunks(std::mem::size_of::<f16>()).enumerate()
                        {
                            data[i] = f16::from_le_bytes(
                                value.try_into().context("Could not read le f16")?,
                            )
                            .to_f32();
                        }
                    }
                }
            }
            ChannelData::Int64(a) => {
                if n_bytes == 8 {
                    let mut buf = vec![0; cycle_count];
                    if cn.endian {
                        Cursor::new(data_bytes)
                            .read_i64_into::<BigEndian>(&mut buf)
                            .context("Could not read be i64 array")?;
                    } else {
                        Cursor::new(data_bytes)
                            .read_i64_into::<LittleEndian>(&mut buf)
                            .context("Could not read le i64 array")?;
                    }
                    *a = PrimitiveArray::from_vec(buf);
                } else if n_bytes == 6 {
                    let data = a.get_mut_values()
                    .context("One channel array function could not get mutable values from primitive array i64")?;
                    if cn.endian {
                        for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                            data[i] = value
                                .read_i48::<BigEndian>()
                                .context("Could not read be i48")?;
                        }
                    } else {
                        for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                            data[i] = value
                                .read_i48::<LittleEndian>()
                                .context("Could not read le i48")?;
                        }
                    }
                }
            }
            ChannelData::UInt64(a) => {
                if n_bytes == 8 {
                    let mut buf = vec![0; cycle_count];
                    if cn.endian {
                        Cursor::new(data_bytes)
                            .read_u64_into::<BigEndian>(&mut buf)
                            .context("Could not read be u64 array")?;
                    } else {
                        Cursor::new(data_bytes)
                            .read_u64_into::<LittleEndian>(&mut buf)
                            .context("Could not read le u64 array")?;
                    }
                    *a = PrimitiveArray::from_vec(buf);
                } else if n_bytes == 7 {
                    let mut temp = [0u8; std::mem::size_of::<u64>()];
                    let data = a.get_mut_values()
                    .context("One channel array function could not get mutable values from primitive array u64")?;
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
                } else if n_bytes == 6 {
                    let data = a.get_mut_values()
                    .context("One channel array function could not get mutable values from primitive array u64")?;
                    if cn.endian {
                        for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                            data[i] = value
                                .read_u48::<BigEndian>()
                                .context("Could not read be u48")?;
                        }
                    } else {
                        for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                            data[i] = value
                                .read_u48::<LittleEndian>()
                                .context("Could not read le u48")?;
                        }
                    }
                } else {
                    // n_bytes = 5
                    let mut temp = [0u8; 6];
                    let data = a.get_mut_values()
                    .context("One channel array function could not get mutable values from primitive array u64")?;
                    if cn.endian {
                        for (i, value) in data_bytes.chunks(n_bytes).enumerate() {
                            temp[0..5].copy_from_slice(&value[0..n_bytes]);
                            data[i] = Cursor::new(temp)
                                .read_u48::<BigEndian>()
                                .context("Could not read be u48 from 5 bytes")?;
                        }
                    } else {
                        for (i, value) in data_bytes.chunks(n_bytes).enumerate() {
                            temp[0..5].copy_from_slice(&value[0..n_bytes]);
                            data[i] = Cursor::new(temp)
                                .read_u48::<LittleEndian>()
                                .context("Could not read le u48 from 5 bytes")?;
                        }
                    }
                }
            }
            ChannelData::Float64(a) => {
                let mut buf = vec![0f64; cycle_count];
                if cn.endian {
                    Cursor::new(data_bytes)
                        .read_f64_into::<BigEndian>(&mut buf)
                        .context("Could not read be f64 array")?;
                } else {
                    Cursor::new(data_bytes)
                        .read_f64_into::<LittleEndian>(&mut buf)
                        .context("Could not read le f64 array")?;
                }
                *a = PrimitiveArray::from_vec(buf);
            }
            ChannelData::Complex32(a) => {
                let data = a
                .get_mut_values()
                .with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array complex f32, channel {}", cn.unique_name))?;
                if n_bytes <= 2 {
                    // complex 16
                    if cn.endian {
                        for (i, value) in data_bytes.chunks(std::mem::size_of::<f16>()).enumerate()
                        {
                            data[i] = f16::from_be_bytes(
                                value.try_into().context("Could not read be f16 complex")?,
                            )
                            .to_f32();
                        }
                    } else {
                        for (i, value) in data_bytes.chunks(std::mem::size_of::<f16>()).enumerate()
                        {
                            data[i] = f16::from_le_bytes(
                                value.try_into().context("Could not read le f16 complex")?,
                            )
                            .to_f32();
                        }
                    }
                } else if n_bytes <= 4 {
                    // complex 32
                    if cn.endian {
                        for (i, value) in data_bytes.chunks(std::mem::size_of::<f32>()).enumerate()
                        {
                            data[i] = f32::from_be_bytes(
                                value.try_into().context("Could not read be f32 complex")?,
                            );
                        }
                    } else {
                        for (i, value) in data_bytes.chunks(std::mem::size_of::<f32>()).enumerate()
                        {
                            data[i] = f32::from_le_bytes(
                                value.try_into().context("Could not read le f32 complex")?,
                            );
                        }
                    }
                }
            }
            ChannelData::Complex64(a) => {
                let data = a
                .get_mut_values()
                .with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array complex f64, channel {}", cn.unique_name))?;
                if cn.endian {
                    for (i, value) in data_bytes.chunks(std::mem::size_of::<f64>()).enumerate() {
                        data[i] = f64::from_be_bytes(
                            value.try_into().context("Could not read be f64 complex")?,
                        );
                    }
                } else {
                    for (i, value) in data_bytes.chunks(std::mem::size_of::<f64>()).enumerate() {
                        data[i] = f64::from_le_bytes(
                            value.try_into().context("Could not read le f64 complex")?,
                        );
                    }
                }
            }
            ChannelData::Utf8(data) => {
                if cn.block.cn_data_type == 6 {
                    // SBC ISO-8859-1 to be converted into UTF8
                    let mut decoder = WINDOWS_1252.new_decoder();
                    for value in data_bytes.chunks(n_bytes) {
                        let mut dst = String::new();
                        let (_result, _size, _replacement) =
                            decoder.decode_to_string(value, &mut dst, false);
                        data.push(Some(dst.trim_end_matches('\0')));
                    }
                } else if cn.block.cn_data_type == 7 {
                    // 7: String UTF8
                    for value in data_bytes.chunks(n_bytes) {
                        data.push(Some(
                            str::from_utf8(value)
                                .context("Found invalid UTF-8")?
                                .trim_end_matches('\0'),
                        ));
                    }
                } else if cn.block.cn_data_type == 8 || cn.block.cn_data_type == 9 {
                    // 8 | 9 :String UTF16 to be converted into UTF8
                    if cn.endian {
                        let mut decoder = UTF_16BE.new_decoder();
                        for record in data_bytes.chunks(n_bytes) {
                            let mut dst = String::new();
                            let (_result, _size, _replacement) =
                                decoder.decode_to_string(record, &mut dst, false);
                            data.push(Some(dst.trim_end_matches('\0')));
                        }
                    } else {
                        let mut decoder = UTF_16LE.new_decoder();
                        for record in data_bytes.chunks(n_bytes) {
                            let mut dst = String::new();
                            let (_result, _size, _replacement) =
                                decoder.decode_to_string(record, &mut dst, false);
                            data.push(Some(dst.trim_end_matches('\0')));
                        }
                    }
                }
            }
            ChannelData::VariableSizeByteArray(a) => {
                // no validity at this point
                let mut data =
                    MutableBinaryValuesArray::<i64>::with_capacities(cycle_count, n_bytes);
                for value in data_bytes.chunks(n_bytes) {
                    data.push(value);
                }
                *a = data.into();
            }
            ChannelData::FixedSizeByteArray(a) => {
                *a = MutableFixedSizeBinaryArray::try_new(
                    DataType::FixedSizeBinary(n_bytes),
                    data_bytes.to_vec(),
                    None,
                )
                .context("failed creating new MutableFixedSizeBinary")?;
            }
            ChannelData::ArrayDInt8(a) => {
                if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(ca) => {
                            let mut buf = vec![0; cycle_count * ca.pnd];
                            Cursor::new(data_bytes)
                                .read_i8_into(&mut buf)
                                .context("Could not read i8 array")?;
                            *a = Tensor::try_new(
                                DataType::Extension(
                                    "Tensor".to_owned(),
                                    Box::new(DataType::Int8),
                                    None,
                                ),
                                buf.into(),
                                shape.clone().map(|s| s.0.clone()),
                                shape.map(|o| o.1.clone()),
                                None,
                                None,
                            )
                            .context("failed creating tensor i8 from one channel array")?;
                        }
                        Compo::CN(_) => {}
                    }
                }
            }
            ChannelData::ArrayDUInt8(a) => {
                if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(_) => {
                            *a = Tensor::try_new(
                                DataType::Extension(
                                    "Tensor".to_owned(),
                                    Box::new(DataType::UInt8),
                                    None,
                                ),
                                data_bytes.clone().into(),
                                shape.clone().map(|s| s.0.clone()),
                                shape.map(|o| o.1.clone()),
                                None,
                                None,
                            )
                            .context("failed creating tensor u8 from one channel array")?;
                        }
                        Compo::CN(_) => {}
                    }
                }
            }
            ChannelData::ArrayDInt16(a) => {
                if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(ca) => {
                            let mut buf = vec![0; cycle_count * ca.pnd];
                            if cn.endian {
                                Cursor::new(data_bytes)
                                    .read_i16_into::<BigEndian>(&mut buf)
                                    .context("Could not read be i16 array")?;
                            } else {
                                Cursor::new(data_bytes)
                                    .read_i16_into::<LittleEndian>(&mut buf)
                                    .context("Could not read le i16 array")?;
                            }
                            *a = Tensor::try_new(
                                DataType::Extension(
                                    "Tensor".to_owned(),
                                    Box::new(DataType::Int16),
                                    None,
                                ),
                                buf.into(),
                                shape.clone().map(|s| s.0.clone()),
                                shape.map(|o| o.1.clone()),
                                None,
                                None,
                            )
                            .context("failed creating tensor i16 from one channel array")?;
                        }
                        Compo::CN(_) => {}
                    }
                }
            }
            ChannelData::ArrayDUInt16(a) => {
                if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(ca) => {
                            let mut buf = vec![0; cycle_count * ca.pnd];
                            if cn.endian {
                                Cursor::new(data_bytes)
                                    .read_u16_into::<BigEndian>(&mut buf)
                                    .context("Could not read be u16 array")?;
                            } else {
                                Cursor::new(data_bytes)
                                    .read_u16_into::<LittleEndian>(&mut buf)
                                    .context("Could not read le 16 array")?;
                            }
                            *a = Tensor::try_new(
                                DataType::Extension(
                                    "Tensor".to_owned(),
                                    Box::new(DataType::UInt16),
                                    None,
                                ),
                                buf.into(),
                                shape.clone().map(|s| s.0.clone()),
                                shape.map(|o| o.1.clone()),
                                None,
                                None,
                            )
                            .context("failed creating tensor u16 from one channel array")?;
                        }
                        Compo::CN(_) => {}
                    }
                }
            }
            ChannelData::ArrayDInt32(a) => {
                if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(ca) => {
                            let mut buf = vec![0i32; cycle_count * ca.pnd];
                            if cn.endian {
                                if n_bytes <= 3 {
                                    for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                                        buf[i] = value
                                            .read_i24::<BigEndian>()
                                            .context("Could not read be i24")?;
                                    }
                                } else {
                                    Cursor::new(data_bytes)
                                        .read_i32_into::<BigEndian>(&mut buf)
                                        .context("Could not read be i32 array")?;
                                }
                            } else if n_bytes <= 3 {
                                for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                                    buf[i] = value
                                        .read_i24::<LittleEndian>()
                                        .context("Could not read le i24")?;
                                }
                            } else {
                                Cursor::new(data_bytes)
                                    .read_i32_into::<LittleEndian>(&mut buf)
                                    .context("Could not read le i32 array")?;
                            }
                            *a = Tensor::try_new(
                                DataType::Extension(
                                    "Tensor".to_owned(),
                                    Box::new(DataType::Int32),
                                    None,
                                ),
                                buf.into(),
                                shape.clone().map(|s| s.0.clone()),
                                shape.map(|o| o.1.clone()),
                                None,
                                None,
                            )
                            .context("failed creating tensor i32 from one channel array")?;
                        }
                        Compo::CN(_) => {}
                    }
                }
            }
            ChannelData::ArrayDUInt32(a) => {
                if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(ca) => {
                            let mut buf = vec![0; cycle_count * ca.pnd];
                            if cn.endian {
                                if n_bytes <= 3 {
                                    for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                                        buf[i] = value
                                            .read_u24::<BigEndian>()
                                            .context("Could not read be u24")?;
                                    }
                                } else {
                                    Cursor::new(data_bytes)
                                        .read_u32_into::<BigEndian>(&mut buf)
                                        .context("Could not read be u32 array")?;
                                }
                            } else if n_bytes <= 3 {
                                for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                                    buf[i] = value
                                        .read_u24::<LittleEndian>()
                                        .context("Could not read le u24")?;
                                }
                            } else {
                                Cursor::new(data_bytes)
                                    .read_u32_into::<LittleEndian>(&mut buf)
                                    .context("Could not read le u32 array")?;
                            }
                            *a = Tensor::try_new(
                                DataType::Extension(
                                    "Tensor".to_owned(),
                                    Box::new(DataType::UInt32),
                                    None,
                                ),
                                buf.into(),
                                shape.clone().map(|s| s.0.clone()),
                                shape.map(|o| o.1.clone()),
                                None,
                                None,
                            )
                            .context("failed creating tensor u32 from one channel array")?;
                        }
                        Compo::CN(_) => {}
                    }
                }
            }
            ChannelData::ArrayDFloat32(a) => {
                if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(ca) => {
                            let mut buf = vec![0f32; cycle_count * ca.pnd];
                            if cn.endian {
                                if n_bytes == 2 {
                                    for (i, value) in
                                        data_bytes.chunks(std::mem::size_of::<f16>()).enumerate()
                                    {
                                        buf[i] = f16::from_be_bytes(
                                            value.try_into().context("Could not read be f16")?,
                                        )
                                        .to_f32();
                                    }
                                } else {
                                    Cursor::new(data_bytes)
                                        .read_f32_into::<BigEndian>(&mut buf)
                                        .context("Could not read be f32 array")?;
                                }
                            } else if n_bytes == 2 {
                                for (i, value) in
                                    data_bytes.chunks(std::mem::size_of::<f16>()).enumerate()
                                {
                                    buf[i] = f16::from_le_bytes(
                                        value.try_into().context("Could not read le f16")?,
                                    )
                                    .to_f32();
                                }
                            } else {
                                Cursor::new(data_bytes)
                                    .read_f32_into::<LittleEndian>(&mut buf)
                                    .context("Could not read le f32 array")?;
                            }
                            *a = Tensor::try_new(
                                DataType::Extension(
                                    "Tensor".to_owned(),
                                    Box::new(DataType::Float32),
                                    None,
                                ),
                                buf.into(),
                                shape.clone().map(|s| s.0.clone()),
                                shape.map(|o| o.1.clone()),
                                None,
                                None,
                            )
                            .context("failed creating tensor f32 from one channel array")?;
                        }
                        Compo::CN(_) => {}
                    }
                }
            }
            ChannelData::ArrayDInt64(a) => {
                if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(ca) => {
                            let mut buf = vec![0; cycle_count * ca.pnd];
                            if cn.endian {
                                if n_bytes == 8 {
                                    Cursor::new(data_bytes)
                                        .read_i64_into::<BigEndian>(&mut buf)
                                        .context("Could not read be i64 array")?;
                                } else if n_bytes == 6 {
                                    for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                                        buf[i] = value
                                            .read_i48::<BigEndian>()
                                            .context("Could not read be i48")?;
                                    }
                                }
                            } else if n_bytes == 8 {
                                Cursor::new(data_bytes)
                                    .read_i64_into::<LittleEndian>(&mut buf)
                                    .context("Could not read le i64 array")?;
                            } else if n_bytes == 6 {
                                for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                                    buf[i] = value
                                        .read_i48::<LittleEndian>()
                                        .context("Could not read le i48")?;
                                }
                            }
                            *a = Tensor::try_new(
                                DataType::Extension(
                                    "Tensor".to_owned(),
                                    Box::new(DataType::Int64),
                                    None,
                                ),
                                buf.into(),
                                shape.clone().map(|s| s.0.clone()),
                                shape.map(|o| o.1.clone()),
                                None,
                                None,
                            )
                            .context("failed creating tensor i64 from one channel array")?;
                        }
                        Compo::CN(_) => {}
                    }
                }
            }
            ChannelData::ArrayDUInt64(a) => {
                if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(ca) => {
                            let mut buf = vec![0; cycle_count * ca.pnd];
                            if n_bytes == 8 {
                                if cn.endian {
                                    Cursor::new(data_bytes)
                                        .read_u64_into::<BigEndian>(&mut buf)
                                        .context("Could not read be u64 array")?;
                                } else {
                                    Cursor::new(data_bytes)
                                        .read_u64_into::<LittleEndian>(&mut buf)
                                        .context("Could not read le u64 array")?;
                                }
                            } else if n_bytes == 7 {
                                let mut temp = [0u8; std::mem::size_of::<u64>()];
                                if cn.endian {
                                    for (i, value) in data_bytes.chunks(n_bytes).enumerate() {
                                        temp[0..7].copy_from_slice(&value[0..7]);
                                        buf[i] = u64::from_be_bytes(temp);
                                    }
                                } else {
                                    for (i, value) in data_bytes.chunks(n_bytes).enumerate() {
                                        temp[0..7].copy_from_slice(&value[0..7]);
                                        buf[i] = u64::from_le_bytes(temp);
                                    }
                                }
                            } else if n_bytes == 6 {
                                if cn.endian {
                                    for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                                        buf[i] = value
                                            .read_u48::<BigEndian>()
                                            .context("Could not read be u48")?;
                                    }
                                } else {
                                    for (i, mut value) in data_bytes.chunks(n_bytes).enumerate() {
                                        buf[i] = value
                                            .read_u48::<LittleEndian>()
                                            .context("Could not read le u48")?;
                                    }
                                }
                            } else if n_bytes == 5 {
                                let mut temp = [0u8; 6];
                                if cn.endian {
                                    for (i, value) in data_bytes.chunks(n_bytes).enumerate() {
                                        temp[0..5].copy_from_slice(&value[0..n_bytes]);
                                        buf[i] = Cursor::new(temp)
                                            .read_u48::<BigEndian>()
                                            .context("Could not read be u48 from 5 bytes")?;
                                    }
                                } else {
                                    for (i, value) in data_bytes.chunks(n_bytes).enumerate() {
                                        temp[0..5].copy_from_slice(&value[0..n_bytes]);
                                        buf[i] = Cursor::new(temp)
                                            .read_u48::<LittleEndian>()
                                            .context("Could not read le u48 from 5 bytes")?;
                                    }
                                }
                            }
                            *a = Tensor::try_new(
                                DataType::Extension(
                                    "Tensor".to_owned(),
                                    Box::new(DataType::UInt64),
                                    None,
                                ),
                                buf.into(),
                                shape.clone().map(|s| s.0.clone()),
                                shape.map(|o| o.1.clone()),
                                None,
                                None,
                            )
                            .context("failed creating tensor u64 from one channel array")?;
                        }
                        Compo::CN(_) => {}
                    }
                }
            }
            ChannelData::ArrayDFloat64(a) => {
                if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(ca) => {
                            let mut buf = vec![0f64; cycle_count * (ca.pnd)];
                            if cn.endian {
                                Cursor::new(data_bytes)
                                    .read_f64_into::<BigEndian>(&mut buf)
                                    .context("Could not read be f64 array")?;
                            } else {
                                Cursor::new(data_bytes)
                                    .read_f64_into::<LittleEndian>(&mut buf)
                                    .context("Could not read le f64 array")?;
                            }
                            *a = Tensor::try_new(
                                DataType::Extension(
                                    "Tensor".to_owned(),
                                    Box::new(DataType::Float64),
                                    None,
                                ),
                                buf.into(),
                                shape.clone().map(|s| s.0.clone()),
                                shape.map(|o| o.1.clone()),
                                None,
                                None,
                            )
                            .context("failed creating tensor f64 from one channel array")?;
                        }
                        Compo::CN(_) => {}
                    }
                }
            }
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
    channel_names_to_read_in_dg: &HashSet<String>,
    record_with_invalid_data: bool,
) -> Result<Vec<i32>, Error> {
    let vlsd_channels: Arc<Mutex<Vec<i32>>> = Arc::new(Mutex::new(Vec::new()));
    // iterates for each channel in parallel with rayon crate
    channels.par_iter_mut()
        .filter(|(_cn_record_position, cn)| {channel_names_to_read_in_dg.contains(&cn.unique_name)})
        .try_for_each(|(rec_pos, cn):(&i32, &mut Cn4)| -> Result<(), Error> {
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
            match &mut cn.data {
                ChannelData::Int8(a)  => {
                    let data = a.get_mut_values()
                    .with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array i8, channel {}", cn.unique_name))?;
                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i8>()];
                        data[i + previous_index] = i8::from_le_bytes(value.try_into().context("Could not read i8")?);
                    }
                }
                ChannelData::UInt8(a)  => {
                    let data = a.get_mut_values()
                    .context("Read channels from bytes function could not get mutable values from primitive array u8")?;
                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u8>()];
                        data[i + previous_index] = u8::from_le_bytes(value.try_into().context("Could not read u8")?);
                    }
                }
                ChannelData::Int16(a)  => {
                    let data = a.get_mut_values()
                    .context("Read channels from bytes function could not get mutable values from primitive array i16")?;
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
                    let data = a.get_mut_values()
                    .context("Read channels from bytes function could not get mutable values from primitive array u16")?;
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
                    let data = a
                    .get_mut_values().with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array i32, channel {}", cn.unique_name))?;
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
                    let data = a.get_mut_values()
                    .with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array u32, channel {}", cn.unique_name))?;
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
                    let data = a.get_mut_values()
                    .with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array f32, channel {}", cn.unique_name))?;
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
                    let data = a.get_mut_values()
                    .with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array i64, channel {}", cn.unique_name))?;
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
                    let data = a.get_mut_values()
                    .with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array u64, channel {}", cn.unique_name))?;
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
                    let data = a.get_mut_values()
                    .with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array f64, channel {}", cn.unique_name))?;
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
                ChannelData::Complex32(a)  => {
                    let data = a
                    .get_mut_values().context("Read channels from bytes function could not get mutable values from primitive array complex f32")?;
                    if n_bytes <= 2 {  // complex 16
                        let mut re_val: &[u8];
                        let mut im_val: &[u8];
                        if cn.endian {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                re_val =
                                    &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f16>()];
                                im_val = &record[pos_byte_beg + std::mem::size_of::<f16>()
                                    ..pos_byte_beg + 2 * std::mem::size_of::<f16>()];
                                data[i*2 + previous_index] = f16::from_be_bytes(
                                    re_val
                                        .try_into()
                                        .context("Could not read be real f16 complex")?,
                                )
                                .to_f32();
                                data[i*2 + 1 + previous_index] = f16::from_be_bytes(
                                    im_val
                                        .try_into()
                                        .context("Could not read be img f16 complex")?,
                                )
                                .to_f32();
                            }
                        } else {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                re_val =
                                    &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f16>()];
                                im_val = &record[pos_byte_beg + std::mem::size_of::<f16>()
                                    ..pos_byte_beg + 2 * std::mem::size_of::<f16>()];
                                data[i*2 + previous_index] = f16::from_le_bytes(
                                    re_val
                                        .try_into()
                                        .context("Could not read le real f16 complex")?,
                                )
                                .to_f32();
                                data[i*2 + 1 + previous_index] = f16::from_le_bytes(
                                    im_val
                                        .try_into()
                                        .context("Could not read le img f16 complex")?,
                                )
                                .to_f32();
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
                                data[i*2 + previous_index] = f32::from_be_bytes(
                                    re_val
                                        .try_into()
                                        .context("Could not read be real f32 complex")?,
                                );
                                data[i*2 + 1 + previous_index] = f32::from_be_bytes(
                                    im_val
                                        .try_into()
                                        .context("Could not read be img f32 complex")?,
                                );
                            }
                        } else {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                re_val =
                                    &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f32>()];
                                im_val = &record[pos_byte_beg + std::mem::size_of::<f32>()
                                    ..pos_byte_beg + 2 * std::mem::size_of::<f32>()];
                                data[i*2 + previous_index] = f32::from_le_bytes(
                                    re_val
                                        .try_into()
                                        .context("Could not read le real f32 complex")?,
                                );
                                data[i*2 + 1 + previous_index] = f32::from_le_bytes(
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
                    let data = a
                    .get_mut_values()
                    .with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array complex f64, channel {}", cn.unique_name))?;
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            re_val =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f64>()];
                            im_val = &record[pos_byte_beg + std::mem::size_of::<f64>()
                                ..pos_byte_beg + 2 * std::mem::size_of::<f64>()];
                            data[i*2 + previous_index]  = f64::from_be_bytes(
                                re_val
                                    .try_into()
                                    .context("Could not read be real f64 complex")?,
                            );
                            data[i*2 + 1 + previous_index] = f64::from_be_bytes(
                                im_val
                                    .try_into()
                                    .context("Could not read be img f64 complex")?,
                            );
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            re_val =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f64>()];
                            im_val = &record[pos_byte_beg + std::mem::size_of::<f64>()
                                ..pos_byte_beg + 2 * std::mem::size_of::<f64>()];
                            data[i*2 + previous_index] = f64::from_le_bytes(
                                re_val
                                    .try_into()
                                    .context("Could not read le real f64 complex")?,
                            );
                            data[i*2 + 1 + previous_index] = f64::from_le_bytes(
                                im_val
                                    .try_into()
                                    .context("Could not read le img f64 complex")?,
                            );
                        }
                    }
                }
                ChannelData::Utf8(array)  => {
                    if cn.block.cn_data_type == 6 {
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
                    } else if cn.block.cn_data_type == 7 {
                        // 7: String UTF8
                        for record in data_chunk.chunks(record_length) {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            let dst = str::from_utf8(value)
                                .context("Found invalid UTF-8")?.trim_end_matches('\0');
                            array.push(Some(dst));
                        }
                    } else if cn.block.cn_data_type == 8 || cn.block.cn_data_type == 9 {
                        // 8 | 9 :String UTF16 to be converted into UTF8
                        if cn.endian {
                            let mut decoder = UTF_16BE.new_decoder();
                            for record in data_chunk.chunks(record_length) {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                let mut dst = String::with_capacity(value.len());
                                let (_result, _size, _replacement) = decoder.decode_to_string(
                                    value,
                                    &mut dst,
                                    false,
                                );
                                dst = dst.trim_end_matches('\0').to_owned();
                                array.push(Some(dst));
                            }
                        } else {
                            let mut decoder = UTF_16LE.new_decoder();
                            for record in data_chunk.chunks(record_length) {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                let mut dst = String::with_capacity(value.len());
                                let (_result, _size, _replacement) = decoder.decode_to_string(
                                    value,
                                    &mut dst,
                                    false,
                                );
                                dst = dst.trim_end_matches('\0').to_owned();
                                array.push(Some(dst));
                            }
                        }
                    }
                }
                ChannelData::VariableSizeByteArray(array)  => {
                    for record in data_chunk.chunks(record_length) {
                        array.push(Some(&record[pos_byte_beg..pos_byte_beg + n_bytes]));
                    }
                }
                ChannelData::FixedSizeByteArray(a)  => {
                    for  record in data_chunk.chunks(record_length) {
                        a.push(Some(&record[pos_byte_beg..pos_byte_beg + n_bytes]));
                    }
                }
                ChannelData::ArrayDInt8(a) => {
                    if let Some(compo) = &cn.composition {
                        match &compo.block {
                            Compo::CA(ca) => {
                                let data = a
                                .get_mut_values().with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array tensor i8, channel {}", cn.unique_name))?;
                                for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                    for j in 0..ca.pnd {
                                        value = &record[pos_byte_beg + j * std::mem::size_of::<i8>()..pos_byte_beg + (j + 1) * std::mem::size_of::<i8>()];
                                        data[(i + previous_index) * ca.pnd + j] =
                                            i8::from_le_bytes(value.try_into().context("Could not read i8 array")?);
                                    }
                                }
                            }
                            Compo::CN(_) => {},
                        }
                    }
                },
                ChannelData::ArrayDUInt8(a) => {
                    if let Some(compo) = &cn.composition {
                        match &compo.block {
                            Compo::CA(ca) => {
                                let data = a
                                .get_mut_values().with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array tensor u8, channel {}", cn.unique_name))?;
                                for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                    for j in 0..ca.pnd {
                                        value = &record[pos_byte_beg + j * std::mem::size_of::<u8>()..pos_byte_beg + (j + 1) * std::mem::size_of::<u8>()];
                                        data[(i + previous_index) * ca.pnd + j] =
                                            u8::from_le_bytes(value.try_into().context("Could not read u8 array")?);
                                    }
                                }
                            }
                            Compo::CN(_) => {},
                        }
                    }
                },
                ChannelData::ArrayDInt16(a) => {
                    if let Some(compo) = &cn.composition {
                        match &compo.block {
                            Compo::CA(ca) => {
                                let data = a
                                .get_mut_values().with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array tensor i16, channel {}", cn.unique_name))?;
                                if cn.endian {
                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                        for j in 0..ca.pnd {
                                            value =
                                                &record[pos_byte_beg + j * std::mem::size_of::<i16>()..pos_byte_beg + (j + 1) * std::mem::size_of::<i16>()];
                                            data[(i + previous_index) * ca.pnd + j] = i16::from_be_bytes(
                                                value.try_into().context("Could not read be i16 array")?,
                                            );
                                        }
                                    }
                                } else {
                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                        for j in 0..ca.pnd {
                                            value =
                                            &record[pos_byte_beg + j * std::mem::size_of::<i16>()..pos_byte_beg + (j + 1) * std::mem::size_of::<i16>()];
                                            data[(i + previous_index) * ca.pnd + j] = i16::from_le_bytes(
                                                value.try_into().context("Could not read le i16 array")?,
                                            );
                                        }
                                    }
                                }
                            }
                            Compo::CN(_) => {},
                        }
                    }
                }
                ChannelData::ArrayDUInt16(a) => {
                    if let Some(compo) = &cn.composition {
                        match &compo.block {
                            Compo::CA(ca) => {
                                let data = a
                                .get_mut_values().with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array tensor u16, channel {}", cn.unique_name))?;
                                if cn.endian {
                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                        for j in 0..ca.pnd {
                                            value =
                                                &record[pos_byte_beg + j * std::mem::size_of::<u16>()..pos_byte_beg + (j + 1) * std::mem::size_of::<u16>()];
                                            data[(i + previous_index) * ca.pnd + j] = u16::from_be_bytes(
                                                value.try_into().context("Could not read be u16 array")?,
                                            );
                                        }
                                    }
                                } else {
                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                        for j in 0..ca.pnd {
                                            value =
                                            &record[pos_byte_beg + j * std::mem::size_of::<i16>()..pos_byte_beg + (j + 1) * std::mem::size_of::<u16>()];
                                            data[(i + previous_index) * ca.pnd + j] = u16::from_le_bytes(
                                                value.try_into().context("Could not read le u16 array")?,
                                            );
                                        }
                                    }
                                }
                            }
                            Compo::CN(_) => {},
                        }
                    }
                }
                ChannelData::ArrayDInt32(a) => {
                    if let Some(compo) = &cn.composition {
                        match &compo.block {
                            Compo::CA(ca) => {
                                let data = a
                                .get_mut_values().with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array tensor i32, channel {}", cn.unique_name))?;
                                if cn.endian {
                                    if n_bytes <=3 {
                                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                            for j in 0..ca.pnd {
                                                value = &record[pos_byte_beg + j * n_bytes..pos_byte_beg + (j + 1) * n_bytes];
                                                data[(i + previous_index) * ca.pnd + j] = value
                                                    .read_i24::<BigEndian>()
                                                    .context("Could not read be i24 array")?;
                                            }
                                        }
                                    }
                                    else {
                                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                            for j in 0..ca.pnd {
                                                value =
                                                &record[pos_byte_beg + j * std::mem::size_of::<i32>()..pos_byte_beg + (j + 1) * std::mem::size_of::<i32>()];
                                                data[(i + previous_index) * ca.pnd + j] = i32::from_be_bytes(
                                                    value.try_into().context("Could not read be i32 array")?,
                                                );
                                            }
                                        }
                                    }
                                } else if n_bytes <=3 {
                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                        for j in 0..ca.pnd {
                                            value = &record[pos_byte_beg + j * n_bytes..pos_byte_beg + (j + 1) * n_bytes];
                                            data[(i + previous_index) * ca.pnd + j] = value
                                                .read_i24::<LittleEndian>()
                                                .context("Could not read le i24 array")?;
                                        }
                                    }
                                }
                                else {
                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                        for j in 0..ca.pnd {
                                            value =
                                            &record[pos_byte_beg + j * std::mem::size_of::<i32>()..pos_byte_beg + (j + 1) * std::mem::size_of::<i32>()];
                                            data[(i + previous_index) * ca.pnd + j] = i32::from_le_bytes(
                                                value.try_into().context("Could not read le i32 array")?,
                                            );
                                        }
                                    }
                                }
                            }
                            Compo::CN(_) => {},
                        }
                    }
                }
                ChannelData::ArrayDUInt32(a) => {
                    if let Some(compo) = &cn.composition {
                        match &compo.block {
                            Compo::CA(ca) => {
                                let data = a
                                .get_mut_values().with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array tensor u32, channel {}", cn.unique_name))?;
                                if cn.endian {
                                    if n_bytes <=3 {
                                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                            for j in 0..ca.pnd {
                                                value = &record[pos_byte_beg + j * n_bytes..pos_byte_beg + (j + 1) * n_bytes];
                                                data[(i + previous_index) * ca.pnd + j] = value
                                                    .read_u24::<BigEndian>()
                                                    .context("Could not read be u24 array")?;
                                            }
                                        }
                                    }
                                    else {
                                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                            for j in 0..ca.pnd {
                                                value =
                                                &record[pos_byte_beg + j * std::mem::size_of::<u32>()..pos_byte_beg + (j + 1) * std::mem::size_of::<u32>()];
                                                data[(i + previous_index) * ca.pnd + j] = u32::from_be_bytes(
                                                    value.try_into().context("Could not read be u32 array")?,
                                                );
                                            }
                                        }
                                    }
                                } else if n_bytes <=3 {
                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                        for j in 0..ca.pnd {
                                            value = &record[pos_byte_beg + j * n_bytes..pos_byte_beg + (j + 1) * n_bytes];
                                            data[(i + previous_index) * ca.pnd + j] = value
                                                .read_u24::<LittleEndian>()
                                                .context("Could not read le u24 array")?;
                                        }
                                    }
                                } else {
                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                        for j in 0..ca.pnd {
                                            value =
                                            &record[pos_byte_beg + j * std::mem::size_of::<u32>()..pos_byte_beg + (j + 1) * std::mem::size_of::<u32>()];
                                            data[(i + previous_index) * ca.pnd + j] = u32::from_le_bytes(
                                                value.try_into().context("Could not read le u32 array")?,
                                            );
                                        }
                                    }
                                }
                            }
                            Compo::CN(_) => {},
                        }
                    }
                }
                ChannelData::ArrayDFloat32(a) => {
                    if let Some(compo) = &cn.composition {
                        match &compo.block {
                            Compo::CA(ca) => {
                                let data = a
                                .get_mut_values().with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array tensor f32, channel {}", cn.unique_name))?;
                                if cn.endian {
                                    if n_bytes <= 2 {
                                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                            for j in 0..ca.pnd {
                                                value =
                                                &record[pos_byte_beg + j * std::mem::size_of::<f16>()..pos_byte_beg + (j + 1) * std::mem::size_of::<f16>()];
                                                data[(i + previous_index) * ca.pnd + j] = f16::from_be_bytes(
                                                    value.try_into().context("Could not read be f16 array")?,
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
                                                    value.try_into().context("Could not read be f32 array")?,
                                                );
                                            }
                                        }
                                    }
                                } else if n_bytes <= 2 {
                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                        for j in 0..ca.pnd {
                                            value =
                                            &record[pos_byte_beg + j * std::mem::size_of::<f16>()..pos_byte_beg + (j + 1) * std::mem::size_of::<f16>()];
                                            data[(i + previous_index) * ca.pnd + j] = f16::from_le_bytes(
                                                value.try_into().context("Could not read le f16 array")?,
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
                                                value.try_into().context("Could not read le f32 array")?,
                                            );
                                        }
                                    }
                                }
                            }
                            Compo::CN(_) => {},
                        }
                    }
                }
                ChannelData::ArrayDInt64(a) => {
                    if let Some(compo) = &cn.composition {
                        match &compo.block {
                            Compo::CA(ca) => {
                                let data = a
                                .get_mut_values().with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array tensor i64, channel {}", cn.unique_name))?;
                                if cn.endian {
                                    if n_bytes == 8 {
                                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                            for j in 0..ca.pnd {
                                                value = &record[pos_byte_beg + j * n_bytes..pos_byte_beg + (j + 1) * n_bytes];
                                                data[(i + previous_index) * ca.pnd + j] = i64::from_be_bytes(
                                                    value.try_into().context("Could not read be i64 array")?,
                                                );
                                            }
                                        }
                                    } else if n_bytes == 6 {
                                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                            for j in 0..ca.pnd {
                                                value = &record[pos_byte_beg + j * n_bytes..pos_byte_beg + (j + 1) * n_bytes];
                                                data[(i + previous_index) * ca.pnd + j] = value
                                                    .read_i48::<BigEndian>()
                                                    .context("Could not read be i48 array")?;
                                            }
                                        }
                                    }
                                } else if n_bytes == 8 {
                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                        for j in 0..ca.pnd {
                                            value = &record[pos_byte_beg + j * n_bytes..pos_byte_beg + (j + 1) * n_bytes];
                                            data[(i + previous_index) * ca.pnd + j] = i64::from_le_bytes(
                                                value.try_into().context("Could not read le i64 array")?,
                                            );
                                        }
                                    }
                                } else if n_bytes == 6 {
                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                        for j in 0..ca.pnd {
                                            value = &record[pos_byte_beg + j * n_bytes..pos_byte_beg + (j + 1) * n_bytes];
                                            data[(i + previous_index) * ca.pnd + j] = value
                                                .read_i48::<LittleEndian>()
                                                .context("Could not read le i48 array")?;
                                        }
                                    }
                                }
                            }
                            Compo::CN(_) => {},
                        }
                    }
                }
                ChannelData::ArrayDUInt64(a) => {
                    if let Some(compo) = &cn.composition {
                        match &compo.block {
                            Compo::CA(ca) => {
                                let data = a
                                .get_mut_values().with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array tensor u64, channel {}", cn.unique_name))?;
                                if cn.endian {
                                    if n_bytes == 8 {
                                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                            for j in 0..ca.pnd {
                                                value = &record[pos_byte_beg + j * n_bytes..pos_byte_beg + (j + 1) * n_bytes];
                                                data[(i + previous_index) * ca.pnd + j] = u64::from_le_bytes(
                                                    value.try_into().context("Could not read be u64 array")?,
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
                                                    .context("Could not read be u48 array")?;
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
                                                    .context("Could not read be u48 from 5 bytes in array")?;
                                            }
                                        }
                                    }
                                } else if n_bytes == 8 {
                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                        for j in 0..ca.pnd {
                                            value = &record[pos_byte_beg + j * n_bytes..pos_byte_beg + (j + 1) * n_bytes];
                                            data[(i + previous_index) * ca.pnd + j] = u64::from_le_bytes(
                                                value.try_into().context("Could not read le u64")?,
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
                                                .context("Could not read le u48 array")?;
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
                                                .context("Could not read le u48 from 5 bytes in array")?;
                                        }
                                    }
                                }
                            }
                            Compo::CN(_) => {},
                        }
                    }
                }
                ChannelData::ArrayDFloat64(a) => {
                    if let Some(compo) = &cn.composition {
                        match &compo.block {
                            Compo::CA(ca) => {
                                let data = a
                                .get_mut_values().with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array tensor f64, channel {}", cn.unique_name))?;
                                if cn.endian {
                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                        for j in 0..ca.pnd {
                                            value = &record[pos_byte_beg + j * std::mem::size_of::<u64>()..pos_byte_beg + (j + 1) * std::mem::size_of::<u64>()];
                                            data[(i + previous_index) * ca.pnd + j] = f64::from_be_bytes(
                                                value.try_into().context("Could not read be f64")?,
                                            );
                                        }
                                    }
                                } else {
                                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                        for j in 0..ca.pnd {
                                            value = &record[pos_byte_beg + j * std::mem::size_of::<u64>()..pos_byte_beg + (j + 1) * std::mem::size_of::<u64>()];
                                            data[(i + previous_index) * ca.pnd + j] = f64::from_le_bytes(
                                                value.try_into().context("Could not read le f64")?,
                                            );
                                        }
                                    }
                                }
                            }
                            Compo::CN(_) => {},
                        }
                    }
                }
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
            if let Some((Some(mask), invalid_byte_position, invalid_byte_mask)) = &mut cn.invalid_mask {
                for (i, record) in data_chunk.chunks(record_length).enumerate() {
                    mask.set(i + previous_index, (*invalid_byte_mask & record[*invalid_byte_position]) == 0);
                }
            };
        }
        Ok(())
    }).with_context(|| format!("Parallel channels bytes reading failed for channel {:?}", channels))?;

    let lock = vlsd_channels
        .lock()
        .expect("Could not get lock from vlsd channel arc vec");
    Ok(lock.clone())
}
