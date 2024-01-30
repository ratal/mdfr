//! this module holds the channel data enum and related implementations

use anyhow::{bail, Context, Error, Result};
use arrow2::array::{
    Array, BinaryArray, FixedSizeBinaryArray, FixedSizeListArray, MutableArray, MutableBinaryArray,
    MutableFixedSizeBinaryArray, MutableUtf8Array, PrimitiveArray, Utf8Array,
};
use arrow2::bitmap::MutableBitmap;
use arrow2::buffer::Buffer;
use arrow2::datatypes::{DataType, PhysicalType};
use arrow2::offset::Offsets;
use arrow2::types::PrimitiveType;
use std::fmt;

use crate::export::tensor::{Order, Tensor};

/// channel data type enum.
/// most common data type is 1D ndarray for timeseries with element types numeric.
/// vector of string or bytes also exists.
/// Dynamic dimension arrays ArrayD are also existing to cover CABlock arrays data.
#[derive(Debug, Clone)]
pub enum ChannelData {
    Int8(PrimitiveArray<i8>),
    UInt8(PrimitiveArray<u8>),
    Int16(PrimitiveArray<i16>),
    UInt16(PrimitiveArray<u16>),
    Int32(PrimitiveArray<i32>),
    UInt32(PrimitiveArray<u32>),
    Float32(PrimitiveArray<f32>),
    Int64(PrimitiveArray<i64>),
    UInt64(PrimitiveArray<u64>),
    Float64(PrimitiveArray<f64>),
    Complex32(PrimitiveArray<f32>),
    Complex64(PrimitiveArray<f64>),
    Utf8(MutableUtf8Array<i64>),
    VariableSizeByteArray(MutableBinaryArray<i64>),
    FixedSizeByteArray(MutableFixedSizeBinaryArray),
    ArrayDInt8(Tensor<i8>),
    ArrayDUInt8(Tensor<u8>),
    ArrayDInt16(Tensor<i16>),
    ArrayDUInt16(Tensor<u16>),
    ArrayDInt32(Tensor<i32>),
    ArrayDUInt32(Tensor<u32>),
    ArrayDFloat32(Tensor<f32>),
    ArrayDInt64(Tensor<i64>),
    ArrayDUInt64(Tensor<u64>),
    ArrayDFloat64(Tensor<f64>),
}

/// ChannelData implementation
#[allow(dead_code)]
impl ChannelData {
    /// based on already existing type, rewrite the array filled with zeros at needed size based on cycle_count
    pub fn zeros(
        &self,
        cn_type: u8,
        cycle_count: u64,
        n_bytes: u32,
        shape: (Vec<usize>, Order),
    ) -> Result<ChannelData, Error> {
        if cn_type == 3 || cn_type == 6 {
            // virtual channel
            let mut vect = vec![0u64; cycle_count as usize];
            let mut counter = 0u64;
            vect.iter_mut().for_each(|v| {
                *v = counter.clone();
                counter += 1;
            });
            Ok(ChannelData::UInt64(PrimitiveArray::from_vec(vect)))
        } else {
            match self {
                ChannelData::Int8(_) => Ok(ChannelData::Int8(PrimitiveArray::from_vec(
                    vec![0i8; cycle_count as usize],
                ))),
                ChannelData::UInt8(_) => Ok(ChannelData::UInt8(PrimitiveArray::from_vec(
                    vec![0u8; cycle_count as usize],
                ))),
                ChannelData::Int16(_) => Ok(ChannelData::Int16(PrimitiveArray::from_vec(
                    vec![0i16; cycle_count as usize],
                ))),
                ChannelData::UInt16(_) => Ok(ChannelData::UInt16(PrimitiveArray::from_vec(
                    vec![0u16; cycle_count as usize],
                ))),
                ChannelData::Int32(_) => Ok(ChannelData::Int32(PrimitiveArray::from_vec(
                    vec![0i32; cycle_count as usize],
                ))),
                ChannelData::UInt32(_) => Ok(ChannelData::UInt32(PrimitiveArray::from_vec(
                    vec![0u32; cycle_count as usize],
                ))),
                ChannelData::Float32(_) => Ok(ChannelData::Float32(PrimitiveArray::from_vec(
                    vec![0f32; cycle_count as usize],
                ))),
                ChannelData::Int64(_) => Ok(ChannelData::Int64(PrimitiveArray::from_vec(
                    vec![0i64; cycle_count as usize],
                ))),
                ChannelData::UInt64(_) => Ok(ChannelData::UInt64(PrimitiveArray::from_vec(
                    vec![0u64; cycle_count as usize],
                ))),
                ChannelData::Float64(_) => Ok(ChannelData::Float64(PrimitiveArray::from_vec(
                    vec![0f64; cycle_count as usize],
                ))),
                ChannelData::Complex32(_) => {
                    Ok(ChannelData::Float32(PrimitiveArray::from_vec(vec![
                        0f32;
                        cycle_count as usize
                            * 2
                    ])))
                }
                ChannelData::Complex64(_) => {
                    Ok(ChannelData::Float64(PrimitiveArray::from_vec(vec![
                        0f64;
                        cycle_count as usize
                            * 2
                    ])))
                }
                ChannelData::Utf8(_) => {
                    Ok(ChannelData::Utf8(MutableUtf8Array::<i64>::with_capacities(
                        cycle_count as usize,
                        n_bytes as usize,
                    )))
                }
                ChannelData::VariableSizeByteArray(_) => Ok(ChannelData::VariableSizeByteArray(
                    MutableBinaryArray::<i64>::with_capacities(
                        cycle_count as usize,
                        n_bytes as usize,
                    ),
                )),
                ChannelData::FixedSizeByteArray(_) => Ok(ChannelData::FixedSizeByteArray(
                    MutableFixedSizeBinaryArray::with_capacity(
                        n_bytes as usize,
                        cycle_count as usize,
                    ),
                )),
                ChannelData::ArrayDInt8(_) => Ok(ChannelData::ArrayDInt8(Tensor::try_new(
                    DataType::Extension("Tensor".to_owned(), Box::new(DataType::Int8), None),
                    Buffer::from(vec![0i8; shape.0.iter().product::<usize>()]),
                    Some(shape.0),
                    Some(shape.1),
                    None,
                    None,
                )?)),
                ChannelData::ArrayDUInt8(_) => Ok(ChannelData::ArrayDUInt8(Tensor::try_new(
                    DataType::Extension("Tensor".to_owned(), Box::new(DataType::UInt8), None),
                    Buffer::from(vec![0u8; shape.0.iter().product::<usize>()]),
                    Some(shape.0),
                    Some(shape.1),
                    None,
                    None,
                )?)),
                ChannelData::ArrayDInt16(_) => Ok(ChannelData::ArrayDInt16(Tensor::try_new(
                    DataType::Extension("Tensor".to_owned(), Box::new(DataType::Int16), None),
                    Buffer::from(vec![0i16; shape.0.iter().product::<usize>()]),
                    Some(shape.0),
                    Some(shape.1),
                    None,
                    None,
                )?)),
                ChannelData::ArrayDUInt16(_) => Ok(ChannelData::ArrayDUInt16(Tensor::try_new(
                    DataType::Extension("Tensor".to_owned(), Box::new(DataType::UInt16), None),
                    Buffer::from(vec![0u16; shape.0.iter().product::<usize>()]),
                    Some(shape.0),
                    Some(shape.1),
                    None,
                    None,
                )?)),
                ChannelData::ArrayDInt32(_) => Ok(ChannelData::ArrayDInt32(Tensor::try_new(
                    DataType::Extension("Tensor".to_owned(), Box::new(DataType::Int32), None),
                    Buffer::from(vec![0i32; shape.0.iter().product::<usize>()]),
                    Some(shape.0),
                    Some(shape.1),
                    None,
                    None,
                )?)),
                ChannelData::ArrayDUInt32(_) => Ok(ChannelData::ArrayDUInt32(Tensor::try_new(
                    DataType::Extension("Tensor".to_owned(), Box::new(DataType::UInt32), None),
                    Buffer::from(vec![0u32; shape.0.iter().product::<usize>()]),
                    Some(shape.0),
                    Some(shape.1),
                    None,
                    None,
                )?)),
                ChannelData::ArrayDFloat32(_) => Ok(ChannelData::ArrayDFloat32(Tensor::try_new(
                    DataType::Extension("Tensor".to_owned(), Box::new(DataType::Float32), None),
                    Buffer::from(vec![0f32; shape.0.iter().product::<usize>()]),
                    Some(shape.0),
                    Some(shape.1),
                    None,
                    None,
                )?)),
                ChannelData::ArrayDInt64(_) => Ok(ChannelData::ArrayDInt64(Tensor::try_new(
                    DataType::Extension("Tensor".to_owned(), Box::new(DataType::Int64), None),
                    Buffer::from(vec![0i64; shape.0.iter().product::<usize>()]),
                    Some(shape.0),
                    Some(shape.1),
                    None,
                    None,
                )?)),
                ChannelData::ArrayDUInt64(_) => Ok(ChannelData::ArrayDUInt64(Tensor::try_new(
                    DataType::Extension("Tensor".to_owned(), Box::new(DataType::UInt64), None),
                    Buffer::from(vec![0u64; shape.0.iter().product::<usize>()]),
                    Some(shape.0),
                    Some(shape.1),
                    None,
                    None,
                )?)),
                ChannelData::ArrayDFloat64(_) => Ok(ChannelData::ArrayDFloat64(Tensor::try_new(
                    DataType::Extension("Tensor".to_owned(), Box::new(DataType::Float64), None),
                    Buffer::from(vec![0f64; shape.0.iter().product::<usize>()]),
                    Some(shape.0),
                    Some(shape.1),
                    None,
                    None,
                )?)),
            }
        }
    }
    /// checks is if ndarray is empty
    pub fn is_empty(&self) -> bool {
        match self {
            ChannelData::Int8(data) => data.is_empty(),
            ChannelData::UInt8(data) => data.is_empty(),
            ChannelData::Int16(data) => data.is_empty(),
            ChannelData::UInt16(data) => data.is_empty(),
            ChannelData::Int32(data) => data.is_empty(),
            ChannelData::UInt32(data) => data.is_empty(),
            ChannelData::Float32(data) => data.is_empty(),
            ChannelData::Int64(data) => data.is_empty(),
            ChannelData::UInt64(data) => data.is_empty(),
            ChannelData::Float64(data) => data.is_empty(),
            ChannelData::Complex32(data) => data.is_empty(),
            ChannelData::Complex64(data) => data.is_empty(),
            ChannelData::Utf8(data) => data.is_empty(),
            ChannelData::VariableSizeByteArray(data) => data.is_empty(),
            ChannelData::FixedSizeByteArray(data) => data.is_empty(),
            ChannelData::ArrayDInt8(data) => data.is_empty(),
            ChannelData::ArrayDUInt8(data) => data.is_empty(),
            ChannelData::ArrayDInt16(data) => data.is_empty(),
            ChannelData::ArrayDUInt16(data) => data.is_empty(),
            ChannelData::ArrayDInt32(data) => data.is_empty(),
            ChannelData::ArrayDUInt32(data) => data.is_empty(),
            ChannelData::ArrayDFloat32(data) => data.is_empty(),
            ChannelData::ArrayDInt64(data) => data.is_empty(),
            ChannelData::ArrayDUInt64(data) => data.is_empty(),
            ChannelData::ArrayDFloat64(data) => data.is_empty(),
        }
    }
    /// checks is if ndarray is empty
    pub fn len(&self) -> usize {
        match self {
            ChannelData::Int8(data) => data.len(),
            ChannelData::UInt8(data) => data.len(),
            ChannelData::Int16(data) => data.len(),
            ChannelData::UInt16(data) => data.len(),
            ChannelData::Int32(data) => data.len(),
            ChannelData::UInt32(data) => data.len(),
            ChannelData::Float32(data) => data.len(),
            ChannelData::Int64(data) => data.len(),
            ChannelData::UInt64(data) => data.len(),
            ChannelData::Float64(data) => data.len(),
            ChannelData::Complex32(data) => data.len(),
            ChannelData::Complex64(data) => data.len(),
            ChannelData::Utf8(data) => data.len(),
            ChannelData::VariableSizeByteArray(data) => data.len(),
            ChannelData::FixedSizeByteArray(data) => data.values().len(),
            ChannelData::ArrayDInt8(data) => data.values().len(),
            ChannelData::ArrayDUInt8(data) => data.values().len(),
            ChannelData::ArrayDInt16(data) => data.values().len(),
            ChannelData::ArrayDUInt16(data) => data.values().len(),
            ChannelData::ArrayDInt32(data) => data.values().len(),
            ChannelData::ArrayDUInt32(data) => data.values().len(),
            ChannelData::ArrayDFloat32(data) => data.values().len(),
            ChannelData::ArrayDInt64(data) => data.values().len(),
            ChannelData::ArrayDUInt64(data) => data.values().len(),
            ChannelData::ArrayDFloat64(data) => data.values().len(),
        }
    }
    /// returns the max bit count of each values in array
    pub fn bit_count(&self) -> u32 {
        match self {
            ChannelData::Int8(_) => 8,
            ChannelData::UInt8(_) => 8,
            ChannelData::Int16(_) => 16,
            ChannelData::UInt16(_) => 16,
            ChannelData::Int32(_) => 32,
            ChannelData::UInt32(_) => 32,
            ChannelData::Float32(_) => 32,
            ChannelData::Int64(_) => 64,
            ChannelData::UInt64(_) => 64,
            ChannelData::Float64(_) => 64,
            ChannelData::Complex32(_) => 64,
            ChannelData::Complex64(_) => 128,
            ChannelData::Utf8(data) => {
                data.iter()
                    .map(|s| s.unwrap_or("").len() as u32)
                    .max()
                    .unwrap_or(0)
                    * 8
            }
            ChannelData::VariableSizeByteArray(data) => {
                data.iter()
                    .map(|s| s.unwrap_or_default().len() as u32)
                    .max()
                    .unwrap_or(0)
                    * 8
            }
            ChannelData::FixedSizeByteArray(data) => data.size() as u32 * 8,
            ChannelData::ArrayDInt8(_) => 8,
            ChannelData::ArrayDUInt8(_) => 8,
            ChannelData::ArrayDInt16(_) => 16,
            ChannelData::ArrayDUInt16(_) => 16,
            ChannelData::ArrayDInt32(_) => 32,
            ChannelData::ArrayDUInt32(_) => 32,
            ChannelData::ArrayDFloat32(_) => 32,
            ChannelData::ArrayDInt64(_) => 64,
            ChannelData::ArrayDUInt64(_) => 64,
            ChannelData::ArrayDFloat64(_) => 64,
        }
    }
    /// returns the max byte count of each values in array
    pub fn byte_count(&self) -> u32 {
        match self {
            ChannelData::Int8(_) => 1,
            ChannelData::UInt8(_) => 1,
            ChannelData::Int16(_) => 2,
            ChannelData::UInt16(_) => 2,
            ChannelData::Int32(_) => 4,
            ChannelData::UInt32(_) => 4,
            ChannelData::Float32(_) => 4,
            ChannelData::Int64(_) => 8,
            ChannelData::UInt64(_) => 8,
            ChannelData::Float64(_) => 8,
            ChannelData::Complex32(_) => 8,
            ChannelData::Complex64(_) => 16,
            ChannelData::Utf8(data) => data
                .iter()
                .map(|s| s.unwrap_or("").len() as u32)
                .max()
                .unwrap_or(0),
            ChannelData::VariableSizeByteArray(data) => data
                .iter()
                .map(|s| s.unwrap_or_default().len() as u32)
                .max()
                .unwrap_or(0),
            ChannelData::FixedSizeByteArray(data) => data.size() as u32,
            ChannelData::ArrayDInt8(_) => 1,
            ChannelData::ArrayDUInt8(_) => 1,
            ChannelData::ArrayDInt16(_) => 2,
            ChannelData::ArrayDUInt16(_) => 2,
            ChannelData::ArrayDInt32(_) => 4,
            ChannelData::ArrayDUInt32(_) => 4,
            ChannelData::ArrayDFloat32(_) => 4,
            ChannelData::ArrayDInt64(_) => 8,
            ChannelData::ArrayDUInt64(_) => 8,
            ChannelData::ArrayDFloat64(_) => 8,
        }
    }
    /// returns mdf4 data type
    pub fn data_type(&self, endian: bool) -> u8 {
        if endian {
            // BE
            match self {
                ChannelData::Int8(_) => 3,
                ChannelData::UInt8(_) => 1,
                ChannelData::Int16(_) => 3,
                ChannelData::UInt16(_) => 1,
                ChannelData::Int32(_) => 3,
                ChannelData::UInt32(_) => 1,
                ChannelData::Float32(_) => 5,
                ChannelData::Int64(_) => 3,
                ChannelData::UInt64(_) => 1,
                ChannelData::Float64(_) => 5,
                ChannelData::Complex32(_) => 16,
                ChannelData::Complex64(_) => 16,
                ChannelData::VariableSizeByteArray(_) => 10,
                ChannelData::FixedSizeByteArray(_) => 10,
                ChannelData::ArrayDInt8(_) => 3,
                ChannelData::ArrayDUInt8(_) => 1,
                ChannelData::ArrayDInt16(_) => 3,
                ChannelData::ArrayDUInt16(_) => 1,
                ChannelData::ArrayDInt32(_) => 3,
                ChannelData::ArrayDUInt32(_) => 1,
                ChannelData::ArrayDFloat32(_) => 5,
                ChannelData::ArrayDInt64(_) => 3,
                ChannelData::ArrayDUInt64(_) => 1,
                ChannelData::ArrayDFloat64(_) => 5,
                ChannelData::Utf8(_) => 7,
            }
        } else {
            // LE
            match self {
                ChannelData::Int8(_) => 2,
                ChannelData::UInt8(_) => 0,
                ChannelData::Int16(_) => 2,
                ChannelData::UInt16(_) => 0,
                ChannelData::Int32(_) => 2,
                ChannelData::UInt32(_) => 0,
                ChannelData::Float32(_) => 4,
                ChannelData::Int64(_) => 2,
                ChannelData::UInt64(_) => 0,
                ChannelData::Float64(_) => 4,
                ChannelData::Complex32(_) => 15,
                ChannelData::Complex64(_) => 15,
                ChannelData::VariableSizeByteArray(_) => 10,
                ChannelData::FixedSizeByteArray(_) => 10,
                ChannelData::ArrayDInt8(_) => 2,
                ChannelData::ArrayDUInt8(_) => 0,
                ChannelData::ArrayDInt16(_) => 2,
                ChannelData::ArrayDUInt16(_) => 0,
                ChannelData::ArrayDInt32(_) => 2,
                ChannelData::ArrayDUInt32(_) => 0,
                ChannelData::ArrayDFloat32(_) => 4,
                ChannelData::ArrayDInt64(_) => 2,
                ChannelData::ArrayDUInt64(_) => 0,
                ChannelData::ArrayDFloat64(_) => 4,
                ChannelData::Utf8(_) => 7,
            }
        }
    }
    pub fn arrow_data_type(&self) -> DataType {
        match self {
            ChannelData::Int8(_) => DataType::Int8,
            ChannelData::UInt8(_) => DataType::UInt8,
            ChannelData::Int16(_) => DataType::Int16,
            ChannelData::UInt16(_) => DataType::UInt16,
            ChannelData::Int32(_) => DataType::Int32,
            ChannelData::UInt32(_) => DataType::UInt32,
            ChannelData::Float32(_) => DataType::Float32,
            ChannelData::Int64(_) => DataType::Int64,
            ChannelData::UInt64(_) => DataType::UInt64,
            ChannelData::Float64(_) => DataType::Float64,
            ChannelData::Complex32(_) => DataType::Float32,
            ChannelData::Complex64(_) => DataType::Float64,
            ChannelData::VariableSizeByteArray(_) => DataType::LargeBinary,
            ChannelData::FixedSizeByteArray(a) => DataType::FixedSizeBinary(a.size()),
            ChannelData::ArrayDInt8(_) => {
                DataType::Extension("Tensor".to_owned(), Box::new(DataType::Int8), None)
            }
            ChannelData::ArrayDUInt8(_) => {
                DataType::Extension("Tensor".to_owned(), Box::new(DataType::UInt8), None)
            }
            ChannelData::ArrayDInt16(_) => {
                DataType::Extension("Tensor".to_owned(), Box::new(DataType::Int16), None)
            }
            ChannelData::ArrayDUInt16(_) => {
                DataType::Extension("Tensor".to_owned(), Box::new(DataType::UInt16), None)
            }
            ChannelData::ArrayDInt32(_) => {
                DataType::Extension("Tensor".to_owned(), Box::new(DataType::Int32), None)
            }
            ChannelData::ArrayDUInt32(_) => {
                DataType::Extension("Tensor".to_owned(), Box::new(DataType::UInt32), None)
            }
            ChannelData::ArrayDFloat32(_) => {
                DataType::Extension("Tensor".to_owned(), Box::new(DataType::Float32), None)
            }
            ChannelData::ArrayDInt64(_) => {
                DataType::Extension("Tensor".to_owned(), Box::new(DataType::Int64), None)
            }
            ChannelData::ArrayDUInt64(_) => {
                DataType::Extension("Tensor".to_owned(), Box::new(DataType::UInt64), None)
            }
            ChannelData::ArrayDFloat64(_) => {
                DataType::Extension("Tensor".to_owned(), Box::new(DataType::Float64), None)
            }
            ChannelData::Utf8(_) => DataType::LargeUtf8,
        }
    }
    /// returns raw bytes vectors from ndarray
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        match self {
            ChannelData::Int8(a) => Ok(a.values().iter().flat_map(|x| x.to_ne_bytes()).collect()),
            ChannelData::UInt8(a) => Ok(a.values().iter().flat_map(|x| x.to_ne_bytes()).collect()),
            ChannelData::Int16(a) => Ok(a.values().iter().flat_map(|x| x.to_ne_bytes()).collect()),
            ChannelData::UInt16(a) => Ok(a.values().iter().flat_map(|x| x.to_ne_bytes()).collect()),
            ChannelData::Int32(a) => Ok(a.values().iter().flat_map(|x| x.to_ne_bytes()).collect()),
            ChannelData::UInt32(a) => Ok(a.values().iter().flat_map(|x| x.to_ne_bytes()).collect()),
            ChannelData::Float32(a) => {
                Ok(a.values().iter().flat_map(|x| x.to_ne_bytes()).collect())
            }
            ChannelData::Int64(a) => Ok(a.values().iter().flat_map(|x| x.to_ne_bytes()).collect()),
            ChannelData::UInt64(a) => Ok(a.values().iter().flat_map(|x| x.to_ne_bytes()).collect()),
            ChannelData::Float64(a) => {
                Ok(a.values().iter().flat_map(|x| x.to_ne_bytes()).collect())
            }
            ChannelData::Complex32(a) => {
                Ok(a.values().iter().flat_map(|x| x.to_ne_bytes()).collect())
            }
            ChannelData::Complex64(a) => {
                Ok(a.values().iter().flat_map(|x| x.to_ne_bytes()).collect())
            }
            ChannelData::Utf8(a) => {
                let nbytes = self.byte_count() as usize;
                Ok(a.iter()
                    .flat_map(|x| {
                        let str_bytes = x.unwrap_or_default().to_string().into_bytes();
                        let n_str_bytes = str_bytes.len();
                        if nbytes > n_str_bytes {
                            [str_bytes, vec![0u8; nbytes - n_str_bytes]].concat()
                        } else {
                            str_bytes
                        }
                    })
                    .collect())
            }
            ChannelData::VariableSizeByteArray(a) => Ok(a.values().to_vec()),
            ChannelData::FixedSizeByteArray(a) => Ok(a.values().clone()),
            ChannelData::ArrayDInt8(a) => {
                Ok(a.values().iter().flat_map(|x| x.to_ne_bytes()).collect())
            }
            ChannelData::ArrayDUInt8(a) => {
                Ok(a.values().iter().flat_map(|x| x.to_ne_bytes()).collect())
            }
            ChannelData::ArrayDInt16(a) => {
                Ok(a.values().iter().flat_map(|x| x.to_ne_bytes()).collect())
            }
            ChannelData::ArrayDUInt16(a) => {
                Ok(a.values().iter().flat_map(|x| x.to_ne_bytes()).collect())
            }
            ChannelData::ArrayDInt32(a) => {
                Ok(a.values().iter().flat_map(|x| x.to_ne_bytes()).collect())
            }
            ChannelData::ArrayDUInt32(a) => {
                Ok(a.values().iter().flat_map(|x| x.to_ne_bytes()).collect())
            }
            ChannelData::ArrayDFloat32(a) => {
                Ok(a.values().iter().flat_map(|x| x.to_ne_bytes()).collect())
            }
            ChannelData::ArrayDInt64(a) => {
                Ok(a.values().iter().flat_map(|x| x.to_ne_bytes()).collect())
            }
            ChannelData::ArrayDUInt64(a) => {
                Ok(a.values().iter().flat_map(|x| x.to_ne_bytes()).collect())
            }
            ChannelData::ArrayDFloat64(a) => {
                Ok(a.values().iter().flat_map(|x| x.to_ne_bytes()).collect())
            }
        }
    }
    /// returns the number of dimensions of the channel
    pub fn ndim(&self) -> usize {
        match self {
            ChannelData::Int8(_) => 1,
            ChannelData::UInt8(_) => 1,
            ChannelData::Int16(_) => 1,
            ChannelData::UInt16(_) => 1,
            ChannelData::Int32(_) => 1,
            ChannelData::UInt32(_) => 1,
            ChannelData::Float32(_) => 1,
            ChannelData::Int64(_) => 1,
            ChannelData::UInt64(_) => 1,
            ChannelData::Float64(_) => 1,
            ChannelData::Complex32(_) => 1,
            ChannelData::Complex64(_) => 1,
            ChannelData::VariableSizeByteArray(_) => 1,
            ChannelData::FixedSizeByteArray(_) => 1,
            ChannelData::ArrayDInt8(a) => a.shape().len(),
            ChannelData::ArrayDUInt8(a) => a.shape().len(),
            ChannelData::ArrayDInt16(a) => a.shape().len(),
            ChannelData::ArrayDUInt16(a) => a.shape().len(),
            ChannelData::ArrayDInt32(a) => a.shape().len(),
            ChannelData::ArrayDUInt32(a) => a.shape().len(),
            ChannelData::ArrayDFloat32(a) => a.shape().len(),
            ChannelData::ArrayDInt64(a) => a.shape().len(),
            ChannelData::ArrayDUInt64(a) => a.shape().len(),
            ChannelData::ArrayDFloat64(a) => a.shape().len(),
            ChannelData::Utf8(_) => 1,
        }
    }
    /// returns the shape of channel
    pub fn shape(&self) -> (Vec<usize>, Order) {
        match self {
            ChannelData::Int8(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::UInt8(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::Int16(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::UInt16(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::Int32(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::UInt32(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::Float32(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::Int64(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::UInt64(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::Float64(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::Complex32(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::Complex64(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::Utf8(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::VariableSizeByteArray(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::FixedSizeByteArray(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::ArrayDInt8(a) => (a.shape().clone(), a.order().clone()),
            ChannelData::ArrayDUInt8(a) => (a.shape().clone(), a.order().clone()),
            ChannelData::ArrayDInt16(a) => (a.shape().clone(), a.order().clone()),
            ChannelData::ArrayDUInt16(a) => (a.shape().clone(), a.order().clone()),
            ChannelData::ArrayDInt32(a) => (a.shape().clone(), a.order().clone()),
            ChannelData::ArrayDUInt32(a) => (a.shape().clone(), a.order().clone()),
            ChannelData::ArrayDFloat32(a) => (a.shape().clone(), a.order().clone()),
            ChannelData::ArrayDInt64(a) => (a.shape().clone(), a.order().clone()),
            ChannelData::ArrayDUInt64(a) => (a.shape().clone(), a.order().clone()),
            ChannelData::ArrayDFloat64(a) => (a.shape().clone(), a.order().clone()),
        }
    }
    /// returns optional tuple of minimum and maximum values contained in the channel
    pub fn min_max(&self) -> (Option<f64>, Option<f64>) {
        match self {
            ChannelData::Int8(a) => {
                let min = a.iter().unwrap_required().min().map(|v| *v as f64);
                let max = a.iter().unwrap_required().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::UInt8(a) => {
                let min = a.iter().unwrap_required().min().map(|v| *v as f64);
                let max = a.iter().unwrap_required().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::Int16(a) => {
                let min = a.iter().unwrap_required().min().map(|v| *v as f64);
                let max = a.iter().unwrap_required().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::UInt16(a) => {
                let min = a.iter().unwrap_required().min().map(|v| *v as f64);
                let max = a.iter().unwrap_required().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::Int32(a) => {
                let min = a.iter().unwrap_required().min().map(|v| *v as f64);
                let max = a.iter().unwrap_required().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::UInt32(a) => {
                let min = a.iter().unwrap_required().min().map(|v| *v as f64);
                let max = a.iter().unwrap_required().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::Float32(a) => {
                let max = a
                    .iter()
                    .unwrap_required()
                    .reduce(|accum, item| if accum >= item { accum } else { item })
                    .map(|v| *v as f64);
                let min = a
                    .iter()
                    .unwrap_required()
                    .reduce(|accum, item| if accum <= item { accum } else { item })
                    .map(|v| *v as f64);
                (min, max)
            }
            ChannelData::Int64(a) => {
                let min = a.iter().unwrap_required().min().map(|v| *v as f64);
                let max = a.iter().unwrap_required().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::UInt64(a) => {
                let min = a.iter().unwrap_required().min().map(|v| *v as f64);
                let max = a.iter().unwrap_required().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::Float64(a) => {
                let max = a
                    .iter()
                    .unwrap_required()
                    .reduce(|accum, item| if accum >= item { accum } else { item })
                    .map(|v| *v as f64);
                let min = a
                    .iter()
                    .unwrap_required()
                    .reduce(|accum, item| if accum <= item { accum } else { item })
                    .map(|v| *v as f64);
                (min, max)
            }
            ChannelData::Complex32(_) => (None, None),
            ChannelData::Complex64(_) => (None, None),
            ChannelData::VariableSizeByteArray(_) => (None, None),
            ChannelData::FixedSizeByteArray(_) => (None, None),
            ChannelData::ArrayDInt8(a) => {
                let min = a.values().iter().min().map(|v| *v as f64);
                let max = a.values().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDUInt8(a) => {
                let min = a.values().iter().min().map(|v| *v as f64);
                let max = a.values().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDInt16(a) => {
                let min = a.values().iter().min().map(|v| *v as f64);
                let max = a.values().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDUInt16(a) => {
                let min = a.values().iter().min().map(|v| *v as f64);
                let max = a.values().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDInt32(a) => {
                let min = a.values().iter().min().map(|v| *v as f64);
                let max = a.values().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDUInt32(a) => {
                let min = a.values().iter().min().map(|v| *v as f64);
                let max = a.values().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDFloat32(a) => {
                let max = a
                    .values()
                    .iter()
                    .reduce(|accum, item| if accum >= item { accum } else { item })
                    .map(|v| *v as f64);
                let min = a
                    .values()
                    .iter()
                    .reduce(|accum, item| if accum <= item { accum } else { item })
                    .map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDInt64(a) => {
                let min = a.values().iter().min().map(|v| *v as f64);
                let max = a.values().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDUInt64(a) => {
                let min = a.values().iter().min().map(|v| *v as f64);
                let max = a.values().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDFloat64(a) => {
                let max = a
                    .values()
                    .iter()
                    .reduce(|accum, item| if accum >= item { accum } else { item })
                    .cloned();
                let min = a
                    .values()
                    .iter()
                    .reduce(|accum, item| if accum <= item { accum } else { item })
                    .cloned();
                (min, max)
            }
            ChannelData::Utf8(_) => (None, None),
        }
    }
    /// boxing channel arrow data
    pub fn boxed(&self) -> Box<dyn Array> {
        match &self {
            ChannelData::Int8(a) => a.clone().boxed(),
            ChannelData::UInt8(a) => a.clone().boxed(),
            ChannelData::Int16(a) => a.clone().boxed(),
            ChannelData::UInt16(a) => a.clone().boxed(),
            ChannelData::Int32(a) => a.clone().boxed(),
            ChannelData::UInt32(a) => a.clone().boxed(),
            ChannelData::Float32(a) => a.clone().boxed(),
            ChannelData::Int64(a) => a.clone().boxed(),
            ChannelData::UInt64(a) => a.clone().boxed(),
            ChannelData::Float64(a) => a.clone().boxed(),
            ChannelData::Complex32(a) => a.clone().boxed(),
            ChannelData::Complex64(a) => a.clone().boxed(),
            ChannelData::Utf8(a) => a.clone().as_box(),
            ChannelData::VariableSizeByteArray(a) => a.clone().as_box(),
            ChannelData::FixedSizeByteArray(a) => a.clone().as_box(),
            ChannelData::ArrayDInt8(a) => a.clone().boxed(),
            ChannelData::ArrayDUInt8(a) => a.clone().boxed(),
            ChannelData::ArrayDInt16(a) => a.clone().boxed(),
            ChannelData::ArrayDUInt16(a) => a.clone().boxed(),
            ChannelData::ArrayDInt32(a) => a.clone().boxed(),
            ChannelData::ArrayDUInt32(a) => a.clone().boxed(),
            ChannelData::ArrayDFloat32(a) => a.clone().boxed(),
            ChannelData::ArrayDInt64(a) => a.clone().boxed(),
            ChannelData::ArrayDUInt64(a) => a.clone().boxed(),
            ChannelData::ArrayDFloat64(a) => a.clone().boxed(),
        }
    }
    pub fn set_validity(&mut self, mask: MutableBitmap) -> Result<(), Error> {
        match self {
            ChannelData::Int8(a) => a.set_validity(mask.into()),
            ChannelData::UInt8(a) => a.set_validity(mask.into()),
            ChannelData::Int16(a) => a.set_validity(mask.into()),
            ChannelData::UInt16(a) => a.set_validity(mask.into()),
            ChannelData::Int32(a) => a.set_validity(mask.into()),
            ChannelData::UInt32(a) => a.set_validity(mask.into()),
            ChannelData::Float32(a) => a.set_validity(mask.into()),
            ChannelData::Int64(a) => a.set_validity(mask.into()),
            ChannelData::UInt64(a) => a.set_validity(mask.into()),
            ChannelData::Float64(a) => a.set_validity(mask.into()),
            ChannelData::Complex32(a) => a.set_validity(mask.into()),
            ChannelData::Complex64(a) => a.set_validity(mask.into()),
            ChannelData::Utf8(a) => a.set_validity(mask.into()),
            ChannelData::VariableSizeByteArray(a) => a.set_validity(mask.into()),
            ChannelData::FixedSizeByteArray(a) => {
                *a = MutableFixedSizeBinaryArray::try_new(
                    a.data_type().clone(),
                    a.values().to_vec(),
                    Some(mask),
                )
                .context("failed creating new fixed size binary array with new validity")?
            }
            ChannelData::ArrayDInt8(_) => {}
            ChannelData::ArrayDUInt8(_) => {}
            ChannelData::ArrayDInt16(_) => {}
            ChannelData::ArrayDUInt16(_) => {}
            ChannelData::ArrayDInt32(_) => {}
            ChannelData::ArrayDUInt32(_) => {}
            ChannelData::ArrayDFloat32(_) => {}
            ChannelData::ArrayDInt64(_) => {}
            ChannelData::ArrayDUInt64(_) => {}
            ChannelData::ArrayDFloat64(_) => {}
        }
        Ok(())
    }
}

impl Default for ChannelData {
    fn default() -> Self {
        ChannelData::UInt8(PrimitiveArray::new(
            DataType::UInt8,
            Buffer::<u8>::new(),
            None,
        ))
    }
}

/// Initialises a channel array type depending of cn_type, cn_data_type and if array
pub fn data_type_init(
    cn_type: u8,
    cn_data_type: u8,
    n_bytes: u32,
    is_array: bool,
) -> Result<ChannelData, Error> {
    if !is_array {
        // Not an array
        if cn_type != 3 || cn_type != 6 {
            // not virtual channel or vlsd
            match cn_data_type {
                0 | 1 => {
                    // unsigned int
                    if n_bytes <= 1 {
                        Ok(ChannelData::UInt8(PrimitiveArray::new(
                            DataType::UInt8,
                            Buffer::<u8>::new(),
                            None,
                        )))
                    } else if n_bytes == 2 {
                        Ok(ChannelData::UInt16(PrimitiveArray::new(
                            DataType::UInt16,
                            Buffer::<u16>::new(),
                            None,
                        )))
                    } else if n_bytes <= 4 {
                        Ok(ChannelData::UInt32(PrimitiveArray::new(
                            DataType::UInt32,
                            Buffer::<u32>::new(),
                            None,
                        )))
                    } else {
                        Ok(ChannelData::UInt64(PrimitiveArray::new(
                            DataType::UInt64,
                            Buffer::<u64>::new(),
                            None,
                        )))
                    }
                }
                2 | 3 => {
                    // signed int
                    if n_bytes <= 1 {
                        Ok(ChannelData::Int8(PrimitiveArray::new(
                            DataType::Int8,
                            Buffer::<i8>::new(),
                            None,
                        )))
                    } else if n_bytes == 2 {
                        Ok(ChannelData::Int16(PrimitiveArray::new(
                            DataType::Int16,
                            Buffer::<i16>::new(),
                            None,
                        )))
                    } else if n_bytes <= 4 {
                        Ok(ChannelData::Int32(PrimitiveArray::new(
                            DataType::Int32,
                            Buffer::<i32>::new(),
                            None,
                        )))
                    } else {
                        Ok(ChannelData::Int64(PrimitiveArray::new(
                            DataType::Int64,
                            Buffer::<i64>::new(),
                            None,
                        )))
                    }
                }
                4 | 5 => {
                    // float
                    if n_bytes <= 4 {
                        Ok(ChannelData::Float32(PrimitiveArray::new(
                            DataType::Float32,
                            Buffer::<f32>::new(),
                            None,
                        )))
                    } else {
                        Ok(ChannelData::Float64(PrimitiveArray::new(
                            DataType::Float64,
                            Buffer::<f64>::new(),
                            None,
                        )))
                    }
                }
                15 | 16 => {
                    // complex
                    if n_bytes <= 4 {
                        Ok(ChannelData::Float32(PrimitiveArray::new(
                            DataType::Float32,
                            Buffer::<f32>::new(),
                            None,
                        )))
                    } else {
                        Ok(ChannelData::Float64(PrimitiveArray::new(
                            DataType::Float64,
                            Buffer::<f64>::new(),
                            None,
                        )))
                    }
                }
                6..=9 => {
                    // String UTF8
                    Ok(ChannelData::Utf8(MutableUtf8Array::<i64>::try_new(
                        DataType::LargeUtf8,
                        Offsets::<i64>::new(),
                        Vec::<u8>::new(),
                        None,
                    )?))
                }
                _ => {
                    // bytearray
                    if cn_type == 1 {
                        // VLSD
                        Ok(ChannelData::VariableSizeByteArray(
                            MutableBinaryArray::<i64>::try_new(
                                DataType::LargeBinary,
                                Offsets::new(),
                                Vec::<u8>::new(),
                                None,
                            )?,
                        ))
                    } else {
                        Ok(ChannelData::FixedSizeByteArray(
                            MutableFixedSizeBinaryArray::new(n_bytes as usize),
                        ))
                    }
                }
            }
        } else {
            // virtual channels, cn_bit_count = 0 -> n_bytes = 0, must be LE unsigned int
            Ok(ChannelData::UInt64(PrimitiveArray::new(
                DataType::UInt64,
                Buffer::<u64>::new(),
                None,
            )))
        }
    } else if cn_type != 3 && cn_type != 6 {
        // Array not virtual
        match cn_data_type {
            0 | 1 => {
                // unsigned int
                if n_bytes <= 1 {
                    Ok(ChannelData::ArrayDUInt8(Tensor::new_empty(
                        DataType::Extension("Tensor".to_owned(), Box::new(DataType::UInt8), None),
                    )))
                } else if n_bytes == 2 {
                    Ok(ChannelData::ArrayDUInt16(Tensor::new_empty(
                        DataType::Extension("Tensor".to_owned(), Box::new(DataType::UInt16), None),
                    )))
                } else if n_bytes <= 4 {
                    Ok(ChannelData::ArrayDUInt32(Tensor::new_empty(
                        DataType::Extension("Tensor".to_owned(), Box::new(DataType::UInt32), None),
                    )))
                } else {
                    Ok(ChannelData::ArrayDUInt64(Tensor::new_empty(
                        DataType::Extension("Tensor".to_owned(), Box::new(DataType::UInt64), None),
                    )))
                }
            }
            2 | 3 => {
                // signed int
                if n_bytes <= 1 {
                    Ok(ChannelData::ArrayDInt8(Tensor::new_empty(
                        DataType::Extension("Tensor".to_owned(), Box::new(DataType::Int8), None),
                    )))
                } else if n_bytes == 2 {
                    Ok(ChannelData::ArrayDInt16(Tensor::new_empty(
                        DataType::Extension("Tensor".to_owned(), Box::new(DataType::Int16), None),
                    )))
                } else if n_bytes == 4 {
                    Ok(ChannelData::ArrayDInt32(Tensor::new_empty(
                        DataType::Extension("Tensor".to_owned(), Box::new(DataType::Int32), None),
                    )))
                } else {
                    Ok(ChannelData::ArrayDInt64(Tensor::new_empty(
                        DataType::Extension("Tensor".to_owned(), Box::new(DataType::Int64), None),
                    )))
                }
            }
            4 | 5 => {
                // float
                if n_bytes <= 4 {
                    Ok(ChannelData::ArrayDFloat32(Tensor::new_empty(
                        DataType::Extension("Tensor".to_owned(), Box::new(DataType::Float32), None),
                    )))
                } else {
                    Ok(ChannelData::ArrayDFloat64(Tensor::new_empty(
                        DataType::Extension("Tensor".to_owned(), Box::new(DataType::Float64), None),
                    )))
                }
            }
            15 | 16 => {
                // complex
                if n_bytes <= 4 {
                    bail!("f32 complex tensors not implemented")
                } else {
                    bail!("f32 complex tensors not implemented")
                }
            }
            _ => {
                // strings or bytes arrays not implemented for tensors, theoritically not possible from spec
                bail!("strings or bytes arrays not implemented for tensors, should it be ?");
            }
        }
    } else {
        // virtual channels arrays not implemented, can it even exists ?
        bail!("Virtual channel arrays not implemented, should it even exist ?");
    }
}

pub fn try_from(value: Box<dyn Array>) -> Result<ChannelData, Error> {
    match value.data_type().to_physical_type() {
        PhysicalType::Null => Ok(ChannelData::UInt8(
            value
                .as_any()
                .downcast_ref::<PrimitiveArray<u8>>()
                .context("could not downcast Null type to primitive array u8")?
                .clone(),
        )),
        PhysicalType::Boolean => Ok(ChannelData::UInt8(
            value
                .as_any()
                .downcast_ref::<PrimitiveArray<u8>>()
                .context("could not downcast Boolean type to primitive array u8")?
                .clone(),
        )),
        PhysicalType::Primitive(p) => match p {
            PrimitiveType::Int8 => Ok(ChannelData::Int8(
                value
                    .as_any()
                    .downcast_ref::<PrimitiveArray<i8>>()
                    .context("could not downcast to primitive array i8")?
                    .clone(),
            )),
            PrimitiveType::Int16 => Ok(ChannelData::Int16(
                value
                    .as_any()
                    .downcast_ref::<PrimitiveArray<i16>>()
                    .context("could not downcast to primitive array i16")?
                    .clone(),
            )),
            PrimitiveType::Int32 => Ok(ChannelData::Int32(
                value
                    .as_any()
                    .downcast_ref::<PrimitiveArray<i32>>()
                    .context("could not downcast to primitive array i32")?
                    .clone(),
            )),
            PrimitiveType::Int64 => Ok(ChannelData::Int64(
                value
                    .as_any()
                    .downcast_ref::<PrimitiveArray<i64>>()
                    .context("could not downcast to primitive array i64")?
                    .clone(),
            )),
            PrimitiveType::UInt8 => Ok(ChannelData::UInt8(
                value
                    .as_any()
                    .downcast_ref::<PrimitiveArray<u8>>()
                    .context("could not downcast to primitive array u8")?
                    .clone(),
            )),
            PrimitiveType::UInt16 => Ok(ChannelData::UInt16(
                value
                    .as_any()
                    .downcast_ref::<PrimitiveArray<u16>>()
                    .context("could not downcast to primitive array u16")?
                    .clone(),
            )),
            PrimitiveType::UInt32 => Ok(ChannelData::UInt32(
                value
                    .as_any()
                    .downcast_ref::<PrimitiveArray<u32>>()
                    .context("could not downcast to primitive array u32")?
                    .clone(),
            )),
            PrimitiveType::UInt64 => Ok(ChannelData::UInt32(
                value
                    .as_any()
                    .downcast_ref::<PrimitiveArray<u32>>()
                    .context("could not downcast to primitive array u32")?
                    .clone(),
            )),
            PrimitiveType::Float16 => todo!(),
            PrimitiveType::Float32 => Ok(ChannelData::Float32(
                value
                    .as_any()
                    .downcast_ref::<PrimitiveArray<f32>>()
                    .context("could not downcast to primitive array f32")?
                    .clone(),
            )),
            PrimitiveType::Float64 => Ok(ChannelData::Float64(
                value
                    .as_any()
                    .downcast_ref::<PrimitiveArray<f64>>()
                    .context("could not downcast to primitive array f64")?
                    .clone(),
            )),
            PrimitiveType::Int128 => todo!(),
            PrimitiveType::Int256 => todo!(),
            PrimitiveType::DaysMs => todo!(),
            PrimitiveType::MonthDayNano => todo!(),
        },
        PhysicalType::Binary => {
            let array = value
                .as_any()
                .downcast_ref::<BinaryArray<i32>>()
                .context("could not downcast to Binary array")?
                .clone();
            let array_i64 = BinaryArray::<i64>::try_new(
                array.data_type().clone(),
                array.offsets().into(),
                array.values().clone(),
                array.validity().cloned(),
            )
            .context("failed creating binary array with offsets i64")?;
            Ok(ChannelData::VariableSizeByteArray(
                array_i64
                    .into_mut()
                    .expect_right("could not convert binary i64 into mutable array"),
            ))
        }
        PhysicalType::FixedSizeBinary => {
            let array = value
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .context("could not downcast to fixed size binary array")?
                .clone();
            Ok(ChannelData::FixedSizeByteArray(
                MutableFixedSizeBinaryArray::try_new(
                    array.data_type().clone(),
                    array.values().to_vec(),
                    array.validity().map(|x| x.clone().make_mut()),
                )
                .context("failed creating new mutable fixed size binary array")?,
            ))
        }
        PhysicalType::LargeBinary => {
            let array = value
                .as_any()
                .downcast_ref::<BinaryArray<i64>>()
                .context("could not downcast to Large Binary array")?
                .clone();
            Ok(ChannelData::VariableSizeByteArray(
                array
                    .into_mut()
                    .expect_right("could not convert binary i64 into mutable array"),
            ))
        }
        PhysicalType::Utf8 => {
            let array = value
                .as_any()
                .downcast_ref::<Utf8Array<i32>>()
                .context("could not downcast to Utf8 array")?
                .clone();
            let array_i64 = Utf8Array::<i64>::try_new(
                array.data_type().clone(),
                array.offsets().into(),
                array.values().clone(),
                array.validity().cloned(),
            )
            .context("failed creating utf8 array with offsets i64")?;
            Ok(ChannelData::Utf8(array_i64.into_mut().expect_right(
                "could not convert utf8 i64 into mutable array",
            )))
        }
        PhysicalType::LargeUtf8 => Ok(ChannelData::Utf8(
            value
                .as_any()
                .downcast_ref::<MutableUtf8Array<i64>>()
                .context("could not downcast to large Utf8 array")?
                .clone(),
        )),
        PhysicalType::List => todo!(),
        PhysicalType::FixedSizeList => {
            // used for complex number, size of 2
            let array = value
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .context("could not downcast to fixed size list array, used for complex")?
                .clone();
            if array.size() == 2 {
                match array.values().data_type() {
                    DataType::Float32 => Ok(ChannelData::Complex32(
                        array
                            .as_any()
                            .downcast_ref::<PrimitiveArray<f32>>()
                            .context("could not downcast to primitive array f32")?
                            .clone(),
                    )),
                    DataType::Float64 => Ok(ChannelData::Complex64(
                        array
                            .as_any()
                            .downcast_ref::<PrimitiveArray<f64>>()
                            .context("could not downcast to primitive array f64")?
                            .clone(),
                    )),
                    _ => bail!("FixedSizeList shall be either f23 or f64 to be used for complex"),
                }
            } else {
                bail!("FixedSizeList is not of size 2, to be used for complex")
            }
        }
        PhysicalType::LargeList => todo!(),
        PhysicalType::Struct => todo!(),
        PhysicalType::Union => todo!(),
        PhysicalType::Map => todo!(),
        PhysicalType::Dictionary(_) => todo!(),
    }
}

impl fmt::Display for ChannelData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChannelData::Int8(array) => {
                writeln!(f, "{array:?}")
            }
            ChannelData::UInt8(array) => {
                writeln!(f, "{array:?}")
            }
            ChannelData::Int16(array) => {
                writeln!(f, "{array:?}")
            }
            ChannelData::UInt16(array) => {
                writeln!(f, "{array:?}")
            }
            ChannelData::Int32(array) => {
                writeln!(f, "{array:?}")
            }
            ChannelData::UInt32(array) => {
                writeln!(f, "{array:?}")
            }
            ChannelData::Float32(array) => {
                writeln!(f, "{array:?}")
            }
            ChannelData::Int64(array) => {
                writeln!(f, "{array:?}")
            }
            ChannelData::UInt64(array) => {
                writeln!(f, "{array:?}")
            }
            ChannelData::Float64(array) => {
                writeln!(f, "{array:?}")
            }
            ChannelData::Complex32(array) => {
                writeln!(f, "{array:?}")
            }
            ChannelData::Complex64(array) => {
                writeln!(f, "{array:?}")
            }
            ChannelData::Utf8(array) => {
                for text in array.iter() {
                    writeln!(f, " {text:?} ")?;
                }
                writeln!(f, " ")
            }
            ChannelData::VariableSizeByteArray(array) => {
                for text in array.iter() {
                    writeln!(f, " {text:?} ")?;
                }
                writeln!(f, " ")
            }
            ChannelData::FixedSizeByteArray(array) => {
                for text in array.iter() {
                    writeln!(f, " {text:?} ")?;
                }
                writeln!(f, " ")
            }
            ChannelData::ArrayDInt8(array) => {
                writeln!(f, "{:?}", array)
            }
            ChannelData::ArrayDUInt8(array) => {
                writeln!(f, "{:?}", array)
            }
            ChannelData::ArrayDInt16(array) => {
                writeln!(f, "{:?}", array)
            }
            ChannelData::ArrayDUInt16(array) => {
                writeln!(f, "{:?}", array)
            }
            ChannelData::ArrayDInt32(array) => {
                writeln!(f, "{:?}", array)
            }
            ChannelData::ArrayDUInt32(array) => {
                writeln!(f, "{:?}", array)
            }
            ChannelData::ArrayDFloat32(array) => {
                writeln!(f, "{:?}", array)
            }
            ChannelData::ArrayDInt64(array) => {
                writeln!(f, "{:?}", array)
            }
            ChannelData::ArrayDUInt64(array) => {
                writeln!(f, "{:?}", array)
            }
            ChannelData::ArrayDFloat64(array) => {
                writeln!(f, "{:?}", array)
            }
        }
    }
}
