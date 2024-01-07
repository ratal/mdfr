//! this module implements low level data reading for mdf4 files.
use crate::mdfinfo::mdfinfo4::{Cn4, CnType, Compo};
use anyhow::{bail, Context, Error, Ok, Result};
use arrow2::array::{
    BinaryArray, FixedSizeBinaryArray, MutableArray, MutableBinaryValuesArray,
    MutableUtf8ValuesArray, PrimitiveArray, Utf8Array,
};
use arrow2::buffer::Buffer;
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

/// converts raw data block containing only one channel into a ndarray
pub fn read_one_channel_array(
    data_bytes: &Vec<u8>,
    cn: &mut Cn4,
    cycle_count: usize,
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
        match cn.data.data_type() {
            DataType::Int8 => {
                let mut buf = vec![0; cycle_count];
                Cursor::new(data_bytes)
                    .read_i8_into(&mut buf)
                    .context("Could not read i8 array")?;
                cn.data = PrimitiveArray::from_vec(buf).boxed();
            }
            DataType::UInt8 => {
                cn.data = PrimitiveArray::from_vec(data_bytes.clone()).boxed();
            }
            DataType::Int16 => {
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
                cn.data = PrimitiveArray::from_vec(buf).boxed();
            }
            DataType::UInt16 => {
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
                cn.data = PrimitiveArray::from_vec(buf).boxed();
            }
            DataType::Int32 => {
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
                    cn.data = PrimitiveArray::from_vec(buf).boxed();
                } else if n_bytes == 3 {
                    let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<i32>>()
                        .context("One channel array function could not downcast to primitive array i32")?
                        .get_mut_values().context("One channel array function could not get mutable values from primitive array i32")?;
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
            DataType::UInt32 => {
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
                    cn.data = PrimitiveArray::from_vec(buf).boxed();
                } else if n_bytes == 3 {
                    let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<u32>>()
                        .context("One channel array function could not downcast to primitive array u32")?
                        .get_mut_values().context("One channel array function could not get mutable values from primitive array u32")?;
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
            DataType::Float32 => {
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
                    cn.data = PrimitiveArray::from_vec(buf).boxed();
                } else if n_bytes == 2 {
                    let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<f32>>()
                        .context("One channel array function could not downcast to primitive array float32")?
                        .get_mut_values().context("One channel array function could not get mutable values from primitive array float32")?;
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
            DataType::Int64 => {
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
                    cn.data = PrimitiveArray::from_vec(buf).boxed();
                } else if n_bytes == 6 {
                    let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<i64>>()
                        .context("One channel array function could not downcast to primitive array i64")?
                        .get_mut_values().context("One channel array function could not get mutable values from primitive array i64")?;
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
            DataType::UInt64 => {
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
                    cn.data = PrimitiveArray::from_vec(buf).boxed();
                } else if n_bytes == 7 {
                    let mut temp = [0u8; std::mem::size_of::<u64>()];
                    let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<u64>>()
                        .context("One channel array function could not downcast to primitive array u64")?
                        .get_mut_values().context("One channel array function could not get mutable values from primitive array u64")?;
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
                    let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<u64>>()
                        .context("One channel array function could not downcast to primitive array u64")?
                        .get_mut_values().context("One channel array function could not get mutable values from primitive array u64")?;
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
                    let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<u64>>()
                        .context("One channel array function could not downcast to primitive array u64")?
                        .get_mut_values().context("One channel array function could not get mutable values from primitive array u64")?;
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
            DataType::Float64 => {
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
                cn.data = PrimitiveArray::from_vec(buf).boxed();
            }
            DataType::FixedSizeList(field, _size) => {
                if field.name.eq(&"complex32".to_string()) {
                    let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<f32>>()
                        .context("One channel array function could not downcast to primitive array f32 for complex32")?
                        .get_mut_values().context("One channel array function could not get mutable values from primitive array f32 for complex32")?;
                    if n_bytes <= 2 {
                        // complex 16
                        if cn.endian {
                            for (i, value) in
                                data_bytes.chunks(std::mem::size_of::<f16>()).enumerate()
                            {
                                data[i] = f16::from_be_bytes(
                                    value.try_into().context("Could not read be f16 complex")?,
                                )
                                .to_f32();
                            }
                        } else {
                            for (i, value) in
                                data_bytes.chunks(std::mem::size_of::<f16>()).enumerate()
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
                            for (i, value) in
                                data_bytes.chunks(std::mem::size_of::<f32>()).enumerate()
                            {
                                data[i] = f32::from_be_bytes(
                                    value.try_into().context("Could not read be f32 complex")?,
                                );
                            }
                        } else {
                            for (i, value) in
                                data_bytes.chunks(std::mem::size_of::<f32>()).enumerate()
                            {
                                data[i] = f32::from_le_bytes(
                                    value.try_into().context("Could not read le f32 complex")?,
                                );
                            }
                        }
                    }
                } else if field.name.eq(&"complex64".to_string()) {
                    // complex 64
                    let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<f64>>()
                        .context("One channel array function could not downcast to primitive array f64 for complex64")?
                        .get_mut_values().context("One channel array function could not get mutable values from primitive array f64 for complex64")?;
                    if cn.endian {
                        for (i, value) in data_bytes.chunks(std::mem::size_of::<f64>()).enumerate()
                        {
                            data[i] = f64::from_be_bytes(
                                value.try_into().context("Could not read be f64 complex")?,
                            );
                        }
                    } else {
                        for (i, value) in data_bytes.chunks(std::mem::size_of::<f64>()).enumerate()
                        {
                            data[i] = f64::from_le_bytes(
                                value.try_into().context("Could not read le f64 complex")?,
                            );
                        }
                    }
                }
            }
            DataType::LargeUtf8 => {
                let mut data = MutableUtf8ValuesArray::<i64>::with_capacities(cycle_count, n_bytes);
                if cn.block.cn_data_type == 6 {
                    // SBC ISO-8859-1 to be converted into UTF8
                    let mut decoder = WINDOWS_1252.new_decoder();
                    for value in data_bytes.chunks(n_bytes) {
                        let mut dst = String::new();
                        let (_result, _size, _replacement) =
                            decoder.decode_to_string(value, &mut dst, false);
                        data.push(dst.trim_end_matches('\0'));
                    }
                } else if cn.block.cn_data_type == 7 {
                    // 7: String UTF8
                    for value in data_bytes.chunks(n_bytes) {
                        data.push(
                            str::from_utf8(value)
                                .context("Found invalid UTF-8")?
                                .trim_end_matches('\0'),
                        );
                    }
                } else if cn.block.cn_data_type == 8 || cn.block.cn_data_type == 9 {
                    // 8 | 9 :String UTF16 to be converted into UTF8
                    if cn.endian {
                        let mut decoder = UTF_16BE.new_decoder();
                        for record in data_bytes.chunks(n_bytes) {
                            let mut dst = String::new();
                            let (_result, _size, _replacement) =
                                decoder.decode_to_string(record, &mut dst, false);
                            data.push(dst.trim_end_matches('\0'));
                        }
                    } else {
                        let mut decoder = UTF_16LE.new_decoder();
                        for record in data_bytes.chunks(n_bytes) {
                            let mut dst = String::new();
                            let (_result, _size, _replacement) =
                                decoder.decode_to_string(record, &mut dst, false);
                            data.push(dst.trim_end_matches('\0'));
                        }
                    }
                }
                cn.data = data.as_box();
            }
            DataType::LargeBinary => {
                let mut data =
                    MutableBinaryValuesArray::<i64>::with_capacities(cycle_count, n_bytes);
                for value in data_bytes.chunks(n_bytes) {
                    data.push(value);
                }
                cn.data = data.as_box();
            }
            DataType::FixedSizeBinary(_size) => {
                cn.data = FixedSizeBinaryArray::new(
                    DataType::FixedSizeBinary(n_bytes),
                    Buffer::<u8>::from(data_bytes.to_vec()),
                    None,
                )
                .boxed();
            }
            DataType::Extension(extension_name, data_type, _) => {
                if extension_name.eq(&"Tensor".to_string()) {
                    match *data_type.clone() {
                        DataType::Int8 => {
                            if let Some(compo) = &cn.composition {
                                match &compo.block {
                                    Compo::CA(ca) => {
                                        let mut buf = vec![0; cycle_count * ca.pnd];
                                        Cursor::new(data_bytes)
                                            .read_i8_into(&mut buf)
                                            .context("Could not read i8 array")?;
                                        cn.data = PrimitiveArray::from_vec(buf).boxed();
                                    }
                                    Compo::CN(_) => {}
                                }
                            }
                        }
                        DataType::UInt8 => {
                            if let Some(compo) = &cn.composition {
                                match &compo.block {
                                    Compo::CA(_) => {
                                        cn.data = PrimitiveArray::from_vec(data_bytes.clone()).boxed();
                                    }
                                    Compo::CN(_) => {}
                                }
                            }
                        }
                        DataType::Int16 => {
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
                                        cn.data = PrimitiveArray::from_vec(buf).boxed();
                                    }
                                    Compo::CN(_) => {}
                                }
                            }
                        }
                        DataType::UInt16 => {
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
                                        cn.data = PrimitiveArray::from_vec(buf).boxed();
                                    }
                                    Compo::CN(_) => {}
                                }
                            }
                        }
                        DataType::Int32 => {
                            if let Some(compo) = &cn.composition {
                                match &compo.block {
                                    Compo::CA(ca) => {
                                        let mut buf = vec![0i32; cycle_count * ca.pnd];
                                        if cn.endian {
                                            if n_bytes <=3 {
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
                                        } else if n_bytes <=3 {
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
                                        cn.data = PrimitiveArray::from_vec(buf).boxed();
                                    }
                                    Compo::CN(_) => {}
                                }
                            }
                        }
                        DataType::UInt32 => {
                            if let Some(compo) = &cn.composition {
                                match &compo.block {
                                    Compo::CA(ca) => {
                                        let mut buf = vec![0; cycle_count * ca.pnd];
                                        if cn.endian {
                                            if n_bytes <=3 {
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
                                        } else  if n_bytes <=3 {
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
                                        cn.data = PrimitiveArray::from_vec(buf).boxed();
                                    }
                                    Compo::CN(_) => {}
                                }
                            }
                        }
                        DataType::Float32 => {
                            if let Some(compo) = &cn.composition {
                                match &compo.block {
                                    Compo::CA(ca) => {
                                        let mut buf = vec![0f32; cycle_count * ca.pnd];
                                        if cn.endian {
                                            if n_bytes ==2 {
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
                                        } else if n_bytes ==2 {
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
                                        cn.data = PrimitiveArray::from_vec(buf).boxed();
                                    }
                                    Compo::CN(_) => {}
                                }
                            }
                        }
                        DataType::Int64 => {
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
                                        cn.data = PrimitiveArray::from_vec(buf).boxed();
                                    }
                                    Compo::CN(_) => {}
                                }
                            }
                        }
                        DataType::UInt64 => {
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
                                            } else{
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
                                        cn.data = PrimitiveArray::from_vec(buf).boxed();
                                    }
                                    Compo::CN(_) => {}
                                }
                            }
                        }
                        DataType::Float64 => {
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
                                        cn.data = PrimitiveArray::from_vec(buf).boxed();
                                    }
                                    Compo::CN(_) => {}
                                }
                            }
                        }
                        DataType::FixedSizeList(field, _size) => {
                            if field.name.eq(&"complex32".to_string()) {
                                let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<f32>>()
                                    .context("One channel array function could not downcast to primitive array f32 for array complex32")?
                                    .get_mut_values().context("One channel array function could not get mutable values from primitive array f32 for array complex32")?;
                                if n_bytes <= 2 { // complex 16
                                    if let Some(compo) = &cn.composition {
                                        match &compo.block {
                                            Compo::CA(_) => {
                                                if cn.endian {
                                                    for (i, value) in
                                                        data_bytes.chunks(std::mem::size_of::<f16>()).enumerate()
                                                    {
                                                        data[i] = f16::from_be_bytes(
                                                            value
                                                                .try_into()
                                                                .context("Could not read be f16 complex array")?,
                                                        )
                                                        .to_f32();
                                                    }
                                                } else {
                                                    for (i, value) in
                                                        data_bytes.chunks(std::mem::size_of::<f16>()).enumerate()
                                                    {
                                                        data[i] = f16::from_le_bytes(
                                                            value
                                                                .try_into()
                                                                .context("Could not read le f16 complex array")?,
                                                        )
                                                        .to_f32();
                                                    }
                                                }
                                            }
                                            Compo::CN(_) => {}
                                        }
                                    }
                                }  else if n_bytes <= 4 {
                                    if let Some(compo) = &cn.composition {
                                        match &compo.block {
                                            Compo::CA(_) => {
                                                if cn.endian {
                                                    for (i, value) in
                                                        data_bytes.chunks(std::mem::size_of::<f32>()).enumerate()
                                                    {
                                                        data[i] = f32::from_be_bytes(
                                                            value
                                                                .try_into()
                                                                .context("Could not read be f32 complex array")?,
                                                        );
                                                    }
                                                } else {
                                                    for (i, value) in
                                                        data_bytes.chunks(std::mem::size_of::<f32>()).enumerate()
                                                    {
                                                        data[i] = f32::from_le_bytes(
                                                            value
                                                                .try_into()
                                                                .context("Could not read le f32 complex array")?,
                                                        );
                                                    }
                                                }
                                            }
                                            Compo::CN(_) => {}
                                        }
                                    }
                                }
                            } else if field.name.eq(&"complex64".to_string())  {
                                let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<f64>>()
                                    .context("One channel array function could not downcast to primitive array f64 for array complex64")?.get_mut_values()
                                    .context("One channel array function could not get mutable values from primitive array f64 for array complex64")?;
                                if let Some(compo) = &cn.composition {
                                    match &compo.block {
                                        Compo::CA(_) => {
                                            if cn.endian {
                                                for (i, value) in
                                                    data_bytes.chunks(std::mem::size_of::<f64>()).enumerate()
                                                {
                                                    data[i] = f64::from_be_bytes(
                                                        value
                                                            .try_into()
                                                            .context("Could not read be f64 complex array")?,
                                                    );
                                                }
                                            } else {
                                                for (i, value) in
                                                    data_bytes.chunks(std::mem::size_of::<f64>()).enumerate()
                                                {
                                                    data[i] = f64::from_le_bytes(
                                                        value
                                                            .try_into()
                                                            .context("Could not read le f64 complex array")?,
                                                    );
                                                }
                                            }
                                        }
                                        Compo::CN(_) => {}
                                    }
                                }
                            }
                        }
                        _ => bail!(
                            "channel {} type is not of valid type for a tensor to be read for a mdf",
                            cn.unique_name
                        ),
                    }
                }
            }
            _ => bail!(
                "channel {} type is not of valid type to be read for a mdf",
                cn.unique_name
            ),
        }
        // channel was properly read
        cn.channel_data_valid = true;
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
        .filter(|(_cn_record_position, cn)| {channel_names_to_read_in_dg.contains(&cn.unique_name) && !cn.channel_data_valid})
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
            match cn.data.data_type() {
                DataType::Int8 => {
                    let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<i8>>()
                        .with_context(|| format!("Read channels from bytes function could not downcast to primitive array i8, channel {}", cn.unique_name))?
                        .get_mut_values().with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array i8, channel {}", cn.unique_name))?;
                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i8>()];
                        data[i + previous_index] = i8::from_le_bytes(value.try_into().context("Could not read i8")?);
                    }
                }
                DataType::UInt8 => {
                    let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<u8>>()
                        .context("Read channels from bytes function could not downcast to primitive array u8")?
                        .get_mut_values().context("Read channels from bytes function could not get mutable values from primitive array u8")?;
                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u8>()];
                        data[i + previous_index] = u8::from_le_bytes(value.try_into().context("Could not read u8")?);
                    }
                }
                DataType::Int16 => {
                    let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<i16>>()
                        .context("Read channels from bytes function could not downcast to primitive array i16")?
                        .get_mut_values().context("Read channels from bytes function could not get mutable values from primitive array i16")?;
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
                DataType::UInt16 => {
                    let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<u16>>()
                    .context("Read channels from bytes function could not downcast to primitive array u16")?
                    .get_mut_values().context("Read channels from bytes function could not get mutable values from primitive array u16")?;
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
                DataType::Int32 => {
                    let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<i32>>()
                    .with_context(|| format!("Read channels from bytes function could not downcast to primitive array i32, channel {}", cn.unique_name))?.get_mut_values().with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array i32, channel {}", cn.unique_name))?;
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
                DataType::UInt32 => {
                    let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<u32>>().with_context(|| format!("Read channels from bytes function could not downcast to primitive array u32, channel {}", cn.unique_name))?.get_mut_values().with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array u32, channel {}", cn.unique_name))?;
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
                DataType::Float32 => {
                    let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<f32>>().with_context(|| format!("Read channels from bytes function could not downcast to primitive array f32, channel {}", cn.unique_name))?
                    .get_mut_values().with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array f32, channel {}", cn.unique_name))?;
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
                DataType::Int64 => {
                    let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<i64>>()
                    .with_context(|| format!("Read channels from bytes function could not downcast to primitive array i64, channel {}", cn.unique_name))?
                    .get_mut_values().with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array i64, channel {}", cn.unique_name))?;
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
                DataType::UInt64 => {
                    let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<u64>>()
                    .with_context(|| format!("Read channels from bytes function could not downcast to primitive array u64, channel {}", cn.unique_name))?
                    .get_mut_values().with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array u64, channel {}", cn.unique_name))?;
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
                DataType::Float64 => {
                    let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<f64>>()
                    .with_context(|| format!("Read channels from bytes function could not downcast to primitive array f64, channel {}", cn.unique_name))?
                    .get_mut_values().with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array f64, channel {}", cn.unique_name))?;
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
                DataType::FixedSizeList(field, _size) => {
                    if field.name.eq(&"complex32".to_string()) {
                        let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<f32>>()
                        .with_context(|| format!("Read channels from bytes function could not downcast to primitive array complex f32, channel {}", cn.unique_name))?
                        .get_mut_values().with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array complex f32, channel {}", cn.unique_name))?;
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
                    } else if field.name.eq(&"complex64".to_string())  { 
                        // complex 64
                        let mut re_val: &[u8];
                        let mut im_val: &[u8];
                        let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<f64>>()
                        .with_context(|| format!("Read channels from bytes function could not downcast to primitive array complex f64, channel {}", cn.unique_name))?
                        .get_mut_values().with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array complex f64, channel {}", cn.unique_name))?;
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
                }
                DataType::LargeUtf8 => {
                    let mut data = cn.data.as_any_mut().downcast_mut::<Utf8Array<i64>>()
                    .with_context(|| format!("Read channels from bytes function could not downcast to large Utf8, channel {}", cn.unique_name))?.clone()
                    .into_mut().expect_right("failed converting Utf8 channel Array in mutableArray");
                    let n_bytes = cn.n_bytes as usize;
                    if cn.block.cn_data_type == 6 {
                        // SBC ISO-8859-1 to be converted into UTF8
                        let mut decoder = WINDOWS_1252.new_decoder();
                        for record in data_chunk.chunks(record_length) {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            let mut dst = String::new();
                            let (_result, _size, _replacement) =
                                decoder.decode_to_string(value, &mut dst, false);
                            data.push(Some(dst.trim_end_matches('\0')));
                        }
                    } else if cn.block.cn_data_type == 7 {
                        // 7: String UTF8
                        for record in data_chunk.chunks(record_length) {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            data.push(Some(str::from_utf8(value)
                                .context("Found invalid UTF-8")?));
                        }
                    } else if cn.block.cn_data_type == 8 || cn.block.cn_data_type == 9 {
                        // 8 | 9 :String UTF16 to be converted into UTF8
                        if cn.endian {
                            let mut decoder = UTF_16BE.new_decoder();
                            for record in data_chunk.chunks(record_length) {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                let mut dst = String::new();
                                let (_result, _size, _replacement) = decoder.decode_to_string(
                                    value,
                                    &mut dst,
                                    false,
                                );
                                data.push(Some(dst.trim_end_matches('\0')));
                            }
                        } else {
                            let mut decoder = UTF_16LE.new_decoder();
                            for record in data_chunk.chunks(record_length) {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                let mut dst = String::new();
                                let (_result, _size, _replacement) = decoder.decode_to_string(
                                    value,
                                    &mut dst,
                                    false,
                                );
                                data.push(Some(dst.trim_end_matches('\0')));
                            }
                        }
                    }
                    cn.data = data.as_box();
                }
                DataType::LargeBinary => {
                    let n_bytes = cn.n_bytes as usize;
                    let mut data = cn.data.as_any().downcast_ref::<BinaryArray<i64>>()
                    .with_context(|| format!("Read channels from bytes function could not downcast to mutable binary values array, channel {}", cn.unique_name))?.clone()
                    .into_mut().expect_right("failed converting LargeBinary channel Array in mutableArray");
                    for record in data_chunk.chunks(record_length) {
                        data.push(Some(&record[pos_byte_beg..pos_byte_beg + n_bytes]));
                    }
                    cn.data = data.as_box();
                }
                DataType::FixedSizeBinary(_size) => {
                    let n_bytes = cn.n_bytes as usize;
                    let d = cn.data.as_any_mut()
                    .downcast_mut::<FixedSizeBinaryArray>()
                    .with_context(|| format!("Read channels from bytes function could not downcast to fixed size binary array, channel {}", cn.unique_name))?;
                    let data = d.get_mut_values().with_context(|| format!("failed creating MutableFixedSizeBinaryArray, channel {}", cn.unique_name))?;
                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                        data[i*n_bytes+previous_index..(i+1)*n_bytes+previous_index].copy_from_slice(&record[pos_byte_beg..pos_byte_beg + n_bytes]);
                    }
                }
                DataType::Extension(extension_name, data_type, _) => {
                    if extension_name.eq(&"Tensor".to_string()) {
                        match *data_type.clone() {
                            DataType::Int8 => {
                                if let Some(compo) = &cn.composition {
                                    match &compo.block {
                                        Compo::CA(ca) => {
                                            let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<i8>>()
                                            .with_context(|| format!("Read channels from bytes function could not downcast to primitive tensor array i8, channel {}", cn.unique_name))?
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
                            }
                            DataType::UInt8 => {
                                if let Some(compo) = &cn.composition {
                                    match &compo.block {
                                        Compo::CA(ca) => {
                                            let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<u8>>()
                                            .with_context(|| format!("Read channels from bytes function could not downcast to primitive tensor array u8, channel {}", cn.unique_name))?
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
                            }
                            DataType::Int16 => {
                                if let Some(compo) = &cn.composition {
                                    match &compo.block {
                                        Compo::CA(ca) => {
                                            let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<i16>>()
                                            .with_context(|| format!("Read channels from bytes function could not downcast to primitive tensor array i16, channel {}", cn.unique_name))?
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
                            DataType::UInt16 => {
                                if let Some(compo) = &cn.composition {
                                    match &compo.block {
                                        Compo::CA(ca) => {
                                            let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<u16>>()
                                            .with_context(|| format!("Read channels from bytes function could not downcast to primitive tensor array u16, channel {}", cn.unique_name))?
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
                            DataType::Int32 => {
                                if let Some(compo) = &cn.composition {
                                    match &compo.block {
                                        Compo::CA(ca) => {
                                            let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<i32>>()
                                            .with_context(|| format!("Read channels from bytes function could not downcast to primitive tensor array i32, channel {}", cn.unique_name))?
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
                            DataType::UInt32 => {
                                if let Some(compo) = &cn.composition {
                                    match &compo.block {
                                        Compo::CA(ca) => {
                                            let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<u32>>()
                                            .with_context(|| format!("Read channels from bytes function could not downcast to primitive tensor array u32, channel {}", cn.unique_name))?
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
                            DataType::Float32 => {
                                if let Some(compo) = &cn.composition {
                                    match &compo.block {
                                        Compo::CA(ca) => {
                                            let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<f32>>()
                                            .with_context(|| format!("Read channels from bytes function could not downcast to primitive tensor array f32, channel {}", cn.unique_name))?
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
                            DataType::Int64 => {
                                if let Some(compo) = &cn.composition {
                                    match &compo.block {
                                        Compo::CA(ca) => {
                                            let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<i64>>()
                                            .with_context(|| format!("Read channels from bytes function could not downcast to primitive tensor array i64, channel {}", cn.unique_name))?
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
                            DataType::UInt64 => {
                                if let Some(compo) = &cn.composition {
                                    match &compo.block {
                                        Compo::CA(ca) => {
                                            let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<u64>>()
                                            .with_context(|| format!("Read channels from bytes function could not downcast to primitive tensor array u64, channel {}", cn.unique_name))?
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
                            DataType::Float64 => {
                                if let Some(compo) = &cn.composition {
                                    match &compo.block {
                                        Compo::CA(ca) => {
                                            let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<f64>>()
                                            .with_context(|| format!("Read channels from bytes function could not downcast to primitive tensor array f64, channel {}", cn.unique_name))?
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
                            DataType::FixedSizeList(field, _size) => {
                                if field.name.eq(&"complex32".to_string()) {
                                    let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<f32>>()
                                    .with_context(|| format!("Read channels from bytes function could not downcast to primitive tensor array complex f32, channel {}", cn.unique_name))?
                                    .get_mut_values().with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array tensor complex f32, channel {}", cn.unique_name))?;
                                    if n_bytes <= 2 {  // complex 16
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
                                                                        .context("Could not read be real f16 complex array")?,
                                                                ).to_f32();
                                                                data[(i*2 + previous_index) * ca.pnd + j + 1] = f16::from_be_bytes(
                                                                    im_val
                                                                        .try_into()
                                                                        .context("Could not read be img f16 complex array")?,
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
                                                                        .context("Could not read le real f16 complex array")?,
                                                                ).to_f32();
                                                                data[(i*2 + previous_index) * ca.pnd + j + 1] = f16::from_le_bytes(
                                                                    im_val
                                                                        .try_into()
                                                                        .context("Could not read le img f16 complex array")?,
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
                                                                        .context("Could not read be real f32 complex array")?,
                                                                );
                                                                data[(i*2 + previous_index) * ca.pnd + j + 1] = f32::from_be_bytes(
                                                                    im_val
                                                                        .try_into()
                                                                        .context("Could not read be img f32 complex array")?,
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
                                                                        .context("Could not read le real f32 complex array")?,
                                                                );
                                                                data[(i*2 + previous_index) * ca.pnd + j + 1] = f32::from_le_bytes(
                                                                    im_val
                                                                        .try_into()
                                                                        .context("Could not read le img f32 complex array")?,
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
                                                let data = cn.data.as_any_mut().downcast_mut::<PrimitiveArray<f64>>()
                                                .with_context(|| format!("Read channels from bytes function could not downcast to primitive tensor array complex f64, channel {}", cn.unique_name))?
                                                .get_mut_values().with_context(|| format!("Read channels from bytes function could not get mutable values from primitive array tensor complex f64, channel {}", cn.unique_name))?;
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
                                                                    .context("Could not read be real f64 complex")?,
                                                            );
                                                            data[(i*2 + previous_index) * ca.pnd + j + 1] = f64::from_be_bytes(
                                                                im_val
                                                                    .try_into()
                                                                    .context("Could not read be img f64 complex")?,
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
                                                                    .context("Could not read le real f64 complex array")?,
                                                            );
                                                            data[(i*2 + previous_index) * ca.pnd + j + 1] = f64::from_le_bytes(
                                                                im_val
                                                                    .try_into()
                                                                    .context("Could not read le img f64 complex array")?,
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
                            _ => bail!("unrecognised tensor data type at data chunk reading from channel {}", cn.unique_name)
                        }
                    }
                }
                _ => bail!("unrecognised data type at data chunk reading from channel {}", cn.unique_name)
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
        Ok(())
    }).with_context(|| format!("Parallel channels bytes reading failed for channel {:?}", channels))?;

    let lock = vlsd_channels
        .lock()
        .expect("Could not get lock from vlsd channel arc vec");
    Ok(lock.clone())
}
