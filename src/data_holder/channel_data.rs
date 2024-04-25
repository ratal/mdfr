//! this module holds the channel data enum and related implementations

use anyhow::{bail, Context, Error, Result};
use arrow::array::{
    as_primitive_array, Array, ArrayBuilder, ArrayData, ArrayRef, BinaryArray,
    BooleanBufferBuilder, FixedSizeBinaryArray, FixedSizeBinaryBuilder, FixedSizeListArray,
    Int8Builder, LargeBinaryArray, LargeBinaryBuilder, LargeStringArray, LargeStringBuilder,
    PrimitiveBuilder, StringArray,
};
use arrow::buffer::{MutableBuffer, NullBuffer};
use arrow::datatypes::{
    DataType, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type,
    UInt32Type, UInt64Type, UInt8Type,
};
use arrow::util::display::{ArrayFormatter, FormatOptions};
use itertools::Itertools;

use std::fmt;
use std::mem::size_of;
use std::sync::Arc;

use crate::data_holder::complex_arrow::ComplexArrow;
#[cfg(feature = "numpy")]
use crate::data_holder::dtype::NumpyDType;

use super::tensor_arrow::{Order, TensorArrow};

/// channel data type enum.
/// most common data type is 1D ndarray for timeseries with element types numeric.
/// vector of string or bytes also exists.
/// Dynamic dimension arrays ArrayD are also existing to cover CABlock arrays data.
#[derive(Debug)]
pub enum ChannelData {
    Int8(Int8Builder),
    UInt8(PrimitiveBuilder<UInt8Type>),
    Int16(PrimitiveBuilder<Int16Type>),
    UInt16(PrimitiveBuilder<UInt16Type>),
    Int32(PrimitiveBuilder<Int32Type>),
    UInt32(PrimitiveBuilder<UInt32Type>),
    Float32(PrimitiveBuilder<Float32Type>),
    Int64(PrimitiveBuilder<Int64Type>),
    UInt64(PrimitiveBuilder<UInt64Type>),
    Float64(PrimitiveBuilder<Float64Type>),
    Complex32(ComplexArrow<Float32Type>),
    Complex64(ComplexArrow<Float64Type>),
    Utf8(LargeStringBuilder),
    VariableSizeByteArray(LargeBinaryBuilder),
    FixedSizeByteArray(FixedSizeBinaryBuilder),
    ArrayDInt8(TensorArrow<Int8Type>),
    ArrayDUInt8(TensorArrow<UInt8Type>),
    ArrayDInt16(TensorArrow<Int16Type>),
    ArrayDUInt16(TensorArrow<UInt16Type>),
    ArrayDInt32(TensorArrow<Int32Type>),
    ArrayDUInt32(TensorArrow<UInt32Type>),
    ArrayDFloat32(TensorArrow<Float32Type>),
    ArrayDInt64(TensorArrow<Int64Type>),
    ArrayDUInt64(TensorArrow<UInt64Type>),
    ArrayDFloat64(TensorArrow<Float64Type>),
}

impl PartialEq for ChannelData {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Int8(l0), Self::Int8(r0)) => l0.finish_cloned() == r0.finish_cloned(),
            (Self::UInt8(l0), Self::UInt8(r0)) => l0.finish_cloned() == r0.finish_cloned(),
            (Self::Int16(l0), Self::Int16(r0)) => l0.finish_cloned() == r0.finish_cloned(),
            (Self::UInt16(l0), Self::UInt16(r0)) => l0.finish_cloned() == r0.finish_cloned(),
            (Self::Int32(l0), Self::Int32(r0)) => l0.finish_cloned() == r0.finish_cloned(),
            (Self::UInt32(l0), Self::UInt32(r0)) => l0.finish_cloned() == r0.finish_cloned(),
            (Self::Float32(l0), Self::Float32(r0)) => l0.finish_cloned() == r0.finish_cloned(),
            (Self::Int64(l0), Self::Int64(r0)) => l0.finish_cloned() == r0.finish_cloned(),
            (Self::UInt64(l0), Self::UInt64(r0)) => l0.finish_cloned() == r0.finish_cloned(),
            (Self::Float64(l0), Self::Float64(r0)) => l0.finish_cloned() == r0.finish_cloned(),
            (Self::Complex32(l0), Self::Complex32(r0)) => l0 == r0,
            (Self::Complex64(l0), Self::Complex64(r0)) => l0 == r0,
            (Self::Utf8(l0), Self::Utf8(r0)) => l0.finish_cloned() == r0.finish_cloned(),
            (Self::VariableSizeByteArray(l0), Self::VariableSizeByteArray(r0)) => {
                l0.finish_cloned() == r0.finish_cloned()
            }
            (Self::FixedSizeByteArray(l0), Self::FixedSizeByteArray(r0)) => {
                l0.finish_cloned() == r0.finish_cloned()
            }
            (Self::ArrayDInt8(l0), Self::ArrayDInt8(r0)) => l0 == r0,
            (Self::ArrayDUInt8(l0), Self::ArrayDUInt8(r0)) => l0 == r0,
            (Self::ArrayDInt16(l0), Self::ArrayDInt16(r0)) => l0 == r0,
            (Self::ArrayDUInt16(l0), Self::ArrayDUInt16(r0)) => l0 == r0,
            (Self::ArrayDInt32(l0), Self::ArrayDInt32(r0)) => l0 == r0,
            (Self::ArrayDUInt32(l0), Self::ArrayDUInt32(r0)) => l0 == r0,
            (Self::ArrayDFloat32(l0), Self::ArrayDFloat32(r0)) => l0 == r0,
            (Self::ArrayDInt64(l0), Self::ArrayDInt64(r0)) => l0 == r0,
            (Self::ArrayDUInt64(l0), Self::ArrayDUInt64(r0)) => l0 == r0,
            (Self::ArrayDFloat64(l0), Self::ArrayDFloat64(r0)) => l0 == r0,
            _ => false,
        }
    }
}

impl Clone for ChannelData {
    fn clone(&self) -> Self {
        match self {
            Self::Int8(arg0) => Self::Int8(
                arg0.finish_cloned()
                    .into_builder()
                    .expect("failed getting back mutable array"),
            ),
            Self::UInt8(arg0) => Self::UInt8(
                arg0.finish_cloned()
                    .into_builder()
                    .expect("failed getting back mutable array"),
            ),
            Self::Int16(arg0) => Self::Int16(
                arg0.finish_cloned()
                    .into_builder()
                    .expect("failed getting back mutable array"),
            ),
            Self::UInt16(arg0) => Self::UInt16(
                arg0.finish_cloned()
                    .into_builder()
                    .expect("failed getting back mutable array"),
            ),
            Self::Int32(arg0) => Self::Int32(
                arg0.finish_cloned()
                    .into_builder()
                    .expect("failed getting back mutable array"),
            ),
            Self::UInt32(arg0) => Self::UInt32(
                arg0.finish_cloned()
                    .into_builder()
                    .expect("failed getting back mutable array"),
            ),
            Self::Float32(arg0) => Self::Float32(
                arg0.finish_cloned()
                    .into_builder()
                    .expect("failed getting back mutable array"),
            ),
            Self::Int64(arg0) => Self::Int64(
                arg0.finish_cloned()
                    .into_builder()
                    .expect("failed getting back mutable array"),
            ),
            Self::UInt64(arg0) => Self::UInt64(
                arg0.finish_cloned()
                    .into_builder()
                    .expect("failed getting back mutable array"),
            ),
            Self::Float64(arg0) => Self::Float64(
                arg0.finish_cloned()
                    .into_builder()
                    .expect("failed getting back mutable array"),
            ),
            Self::Complex32(arg0) => Self::Complex32(arg0.clone()),
            Self::Complex64(arg0) => Self::Complex64(arg0.clone()),
            Self::Utf8(arg0) => Self::Utf8(
                arg0.finish_cloned()
                    .into_builder()
                    .expect("failed getting back mutable array"),
            ),
            Self::VariableSizeByteArray(array) => Self::VariableSizeByteArray(
                array
                    .finish_cloned()
                    .into_builder()
                    .expect("failed getting back mutable array"),
            ),
            Self::FixedSizeByteArray(array) => {
                let array: FixedSizeBinaryArray = array.finish_cloned();
                let mut new_array =
                    FixedSizeBinaryBuilder::with_capacity(array.len(), array.value_length());
                if let Some(validity) = array.logical_nulls() {
                    array
                        .values()
                        .chunks(array.value_length() as usize)
                        .zip(validity.iter())
                        .for_each(|(value, validity)| {
                            if validity {
                                new_array
                                    .append_value(value)
                                    .expect("failed appending new fixed binary value");
                            } else {
                                new_array.append_null();
                            }
                        });
                } else {
                    array
                        .values()
                        .chunks(array.value_length() as usize)
                        .for_each(|value| {
                            new_array
                                .append_value(value)
                                .expect("failed appending new fixed binary value");
                        });
                }
                Self::FixedSizeByteArray(new_array)
            }
            Self::ArrayDInt8(arg0) => Self::ArrayDInt8(arg0.clone()),
            Self::ArrayDUInt8(arg0) => Self::ArrayDUInt8(arg0.clone()),
            Self::ArrayDInt16(arg0) => Self::ArrayDInt16(arg0.clone()),
            Self::ArrayDUInt16(arg0) => Self::ArrayDUInt16(arg0.clone()),
            Self::ArrayDInt32(arg0) => Self::ArrayDInt32(arg0.clone()),
            Self::ArrayDUInt32(arg0) => Self::ArrayDUInt32(arg0.clone()),
            Self::ArrayDFloat32(arg0) => Self::ArrayDFloat32(arg0.clone()),
            Self::ArrayDInt64(arg0) => Self::ArrayDInt64(arg0.clone()),
            Self::ArrayDUInt64(arg0) => Self::ArrayDUInt64(arg0.clone()),
            Self::ArrayDFloat64(arg0) => Self::ArrayDFloat64(arg0.clone()),
        }
    }
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
                *v = counter;
                counter += 1;
            });
            Ok(ChannelData::UInt64(PrimitiveBuilder::new_from_buffer(
                vect.into(),
                None,
            )))
        } else {
            match self {
                ChannelData::Int8(_) => Ok(ChannelData::Int8(PrimitiveBuilder::new_from_buffer(
                    MutableBuffer::from_len_zeroed(cycle_count as usize * size_of::<i8>()),
                    None,
                ))),
                ChannelData::UInt8(_) => Ok(ChannelData::UInt8(PrimitiveBuilder::new_from_buffer(
                    MutableBuffer::from_len_zeroed(cycle_count as usize * size_of::<u8>()),
                    None,
                ))),
                ChannelData::Int16(_) => Ok(ChannelData::Int16(PrimitiveBuilder::new_from_buffer(
                    MutableBuffer::from_len_zeroed(cycle_count as usize * size_of::<i16>()),
                    None,
                ))),
                ChannelData::UInt16(_) => {
                    Ok(ChannelData::UInt16(PrimitiveBuilder::new_from_buffer(
                        MutableBuffer::from_len_zeroed(cycle_count as usize * size_of::<u16>()),
                        None,
                    )))
                }
                ChannelData::Int32(_) => Ok(ChannelData::Int32(PrimitiveBuilder::new_from_buffer(
                    MutableBuffer::from_len_zeroed(cycle_count as usize * size_of::<i32>()),
                    None,
                ))),
                ChannelData::UInt32(_) => {
                    Ok(ChannelData::UInt32(PrimitiveBuilder::new_from_buffer(
                        MutableBuffer::from_len_zeroed(cycle_count as usize * size_of::<u32>()),
                        None,
                    )))
                }
                ChannelData::Float32(_) => {
                    Ok(ChannelData::Float32(PrimitiveBuilder::new_from_buffer(
                        MutableBuffer::from_len_zeroed(cycle_count as usize * size_of::<f32>()),
                        None,
                    )))
                }
                ChannelData::Int64(_) => Ok(ChannelData::Int64(PrimitiveBuilder::new_from_buffer(
                    MutableBuffer::from_len_zeroed(cycle_count as usize * size_of::<i64>()),
                    None,
                ))),
                ChannelData::UInt64(_) => {
                    Ok(ChannelData::UInt64(PrimitiveBuilder::new_from_buffer(
                        MutableBuffer::from_len_zeroed(cycle_count as usize * size_of::<u64>()),
                        None,
                    )))
                }
                ChannelData::Float64(_) => {
                    Ok(ChannelData::Float64(PrimitiveBuilder::new_from_buffer(
                        MutableBuffer::from_len_zeroed(cycle_count as usize * size_of::<f64>()),
                        None,
                    )))
                }
                ChannelData::Complex32(_) => Ok(ChannelData::Complex32(
                    ComplexArrow::new_from_buffer(vec![0f32; cycle_count as usize * 2].into()),
                )),
                ChannelData::Complex64(_) => Ok(ChannelData::Complex64(
                    ComplexArrow::new_from_buffer(vec![0f64; cycle_count as usize * 2].into()),
                )),
                ChannelData::Utf8(_) => Ok(ChannelData::Utf8(LargeStringBuilder::with_capacity(
                    cycle_count as usize,
                    n_bytes as usize,
                ))),
                ChannelData::VariableSizeByteArray(_) => Ok(ChannelData::VariableSizeByteArray(
                    LargeBinaryBuilder::with_capacity(cycle_count as usize, n_bytes as usize),
                )),
                ChannelData::FixedSizeByteArray(_) => Ok(ChannelData::FixedSizeByteArray(
                    FixedSizeBinaryBuilder::with_capacity(cycle_count as usize, n_bytes as i32),
                )),
                ChannelData::ArrayDInt8(_) => {
                    Ok(ChannelData::ArrayDInt8(TensorArrow::new_from_buffer(
                        vec![0i8; shape.0.iter().product::<usize>()].into(),
                        shape.0,
                        shape.1,
                    )))
                }
                ChannelData::ArrayDUInt8(_) => {
                    Ok(ChannelData::ArrayDUInt8(TensorArrow::new_from_buffer(
                        vec![0u8; shape.0.iter().product::<usize>()].into(),
                        shape.0,
                        shape.1,
                    )))
                }
                ChannelData::ArrayDInt16(_) => {
                    Ok(ChannelData::ArrayDInt16(TensorArrow::new_from_buffer(
                        vec![0i16; shape.0.iter().product::<usize>()].into(),
                        shape.0,
                        shape.1,
                    )))
                }
                ChannelData::ArrayDUInt16(_) => {
                    Ok(ChannelData::ArrayDUInt16(TensorArrow::new_from_buffer(
                        vec![0u16; shape.0.iter().product::<usize>()].into(),
                        shape.0,
                        shape.1,
                    )))
                }
                ChannelData::ArrayDInt32(_) => {
                    Ok(ChannelData::ArrayDInt32(TensorArrow::new_from_buffer(
                        vec![0i32; shape.0.iter().product::<usize>()].into(),
                        shape.0,
                        shape.1,
                    )))
                }
                ChannelData::ArrayDUInt32(_) => {
                    Ok(ChannelData::ArrayDUInt32(TensorArrow::new_from_buffer(
                        vec![0u32; shape.0.iter().product::<usize>()].into(),
                        shape.0,
                        shape.1,
                    )))
                }
                ChannelData::ArrayDFloat32(_) => {
                    Ok(ChannelData::ArrayDFloat32(TensorArrow::new_from_buffer(
                        vec![0f32; shape.0.iter().product::<usize>()].into(),
                        shape.0,
                        shape.1,
                    )))
                }
                ChannelData::ArrayDInt64(_) => {
                    Ok(ChannelData::ArrayDInt64(TensorArrow::new_from_buffer(
                        vec![0i64; shape.0.iter().product::<usize>()].into(),
                        shape.0,
                        shape.1,
                    )))
                }
                ChannelData::ArrayDUInt64(_) => {
                    Ok(ChannelData::ArrayDUInt64(TensorArrow::new_from_buffer(
                        vec![0u64; shape.0.iter().product::<usize>()].into(),
                        shape.0,
                        shape.1,
                    )))
                }
                ChannelData::ArrayDFloat64(_) => {
                    Ok(ChannelData::ArrayDFloat64(TensorArrow::new_from_buffer(
                        vec![0f64; cycle_count as usize * shape.0.iter().product::<usize>()].into(),
                        shape.0,
                        shape.1,
                    )))
                }
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
    /// flatten length of tensor
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
            ChannelData::FixedSizeByteArray(data) => data.len(),
            ChannelData::ArrayDInt8(data) => data.len(),
            ChannelData::ArrayDUInt8(data) => data.len(),
            ChannelData::ArrayDInt16(data) => data.len(),
            ChannelData::ArrayDUInt16(data) => data.len(),
            ChannelData::ArrayDInt32(data) => data.len(),
            ChannelData::ArrayDUInt32(data) => data.len(),
            ChannelData::ArrayDFloat32(data) => data.len(),
            ChannelData::ArrayDInt64(data) => data.len(),
            ChannelData::ArrayDUInt64(data) => data.len(),
            ChannelData::ArrayDFloat64(data) => data.len(),
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
                (data
                    .offsets_slice()
                    .iter()
                    .tuple_windows::<(_, _)>()
                    .map(|w| w.1 - w.0)
                    .max()
                    .unwrap_or(0)
                    * 8) as u32
            }
            ChannelData::VariableSizeByteArray(data) => {
                (data
                    .offsets_slice()
                    .iter()
                    .tuple_windows::<(_, _)>()
                    .map(|w| w.1 - w.0)
                    .max()
                    .unwrap_or(0)
                    * 8) as u32
            }
            ChannelData::FixedSizeByteArray(data) => {
                (data.finish_cloned().value_length() * 8) as u32
            }
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
            ChannelData::Utf8(data) => {
                (data
                    .offsets_slice()
                    .iter()
                    .tuple_windows::<(_, _)>()
                    .map(|w| w.1 - w.0)
                    .max()
                    .unwrap_or(0)) as u32
            }
            ChannelData::VariableSizeByteArray(data) => {
                (data
                    .offsets_slice()
                    .iter()
                    .tuple_windows::<(_, _)>()
                    .map(|w| w.1 - w.0)
                    .max()
                    .unwrap_or(0)) as u32
            }
            ChannelData::FixedSizeByteArray(data) => data.finish_cloned().value_length() as u32,
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
            ChannelData::FixedSizeByteArray(a) => {
                DataType::FixedSizeBinary(a.finish_cloned().value_length())
            }
            ChannelData::ArrayDInt8(_a) => DataType::Int8,
            ChannelData::ArrayDUInt8(_a) => DataType::UInt8,
            ChannelData::ArrayDInt16(_a) => DataType::Int16,
            ChannelData::ArrayDUInt16(_a) => DataType::UInt16,
            ChannelData::ArrayDInt32(_a) => DataType::Int32,
            ChannelData::ArrayDUInt32(_a) => DataType::UInt32,
            ChannelData::ArrayDFloat32(_a) => DataType::Float32,
            ChannelData::ArrayDInt64(_a) => DataType::Int64,
            ChannelData::ArrayDUInt64(_a) => DataType::UInt64,
            ChannelData::ArrayDFloat64(_a) => DataType::Float64,
            ChannelData::Utf8(_) => DataType::LargeUtf8,
        }
    }
    /// returns raw bytes vectors from ndarray
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        match self {
            ChannelData::Int8(a) => Ok(a
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
            ChannelData::UInt8(a) => Ok(a
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
            ChannelData::Int16(a) => Ok(a
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
            ChannelData::UInt16(a) => Ok(a
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
            ChannelData::Int32(a) => Ok(a
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
            ChannelData::UInt32(a) => Ok(a
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
            ChannelData::Float32(a) => Ok(a
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
            ChannelData::Int64(a) => Ok(a
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
            ChannelData::UInt64(a) => Ok(a
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
            ChannelData::Float64(a) => Ok(a
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
            ChannelData::Complex32(a) => Ok(a
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
            ChannelData::Complex64(a) => Ok(a
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
            ChannelData::Utf8(a) => {
                let nbytes = self.byte_count() as usize;
                Ok(a.finish_cloned()
                    .iter()
                    .flat_map(|x| {
                        let str_bytes = x.unwrap_or("").as_bytes();
                        let n_str_bytes = str_bytes.len();
                        if nbytes > n_str_bytes {
                            [str_bytes, &vec![0u8; nbytes - n_str_bytes]].concat()
                        } else {
                            str_bytes.to_vec()
                        }
                    })
                    .collect())
            }
            ChannelData::VariableSizeByteArray(a) => Ok(a.values_slice().to_vec()),
            ChannelData::FixedSizeByteArray(a) => Ok(a.finish_cloned().value_data().to_vec()),
            ChannelData::ArrayDInt8(a) => Ok(a
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
            ChannelData::ArrayDUInt8(a) => Ok(a
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
            ChannelData::ArrayDInt16(a) => Ok(a
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
            ChannelData::ArrayDUInt16(a) => Ok(a
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
            ChannelData::ArrayDInt32(a) => Ok(a
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
            ChannelData::ArrayDUInt32(a) => Ok(a
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
            ChannelData::ArrayDFloat32(a) => Ok(a
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
            ChannelData::ArrayDInt64(a) => Ok(a
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
            ChannelData::ArrayDUInt64(a) => Ok(a
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
            ChannelData::ArrayDFloat64(a) => Ok(a
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
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
            ChannelData::ArrayDInt8(a) => a.ndim(),
            ChannelData::ArrayDUInt8(a) => a.ndim(),
            ChannelData::ArrayDInt16(a) => a.ndim(),
            ChannelData::ArrayDUInt16(a) => a.ndim(),
            ChannelData::ArrayDInt32(a) => a.ndim(),
            ChannelData::ArrayDUInt32(a) => a.ndim(),
            ChannelData::ArrayDFloat32(a) => a.ndim(),
            ChannelData::ArrayDInt64(a) => a.ndim(),
            ChannelData::ArrayDUInt64(a) => a.ndim(),
            ChannelData::ArrayDFloat64(a) => a.ndim(),
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
                let min = a.values_slice().iter().min().map(|v| *v as f64);
                let max = a.values_slice().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::UInt8(a) => {
                let min = a.values_slice().iter().min().map(|v| *v as f64);
                let max = a.values_slice().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::Int16(a) => {
                let min = a.values_slice().iter().min().map(|v| *v as f64);
                let max = a.values_slice().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::UInt16(a) => {
                let min = a.values_slice().iter().min().map(|v| *v as f64);
                let max = a.values_slice().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::Int32(a) => {
                let min = a.values_slice().iter().min().map(|v| *v as f64);
                let max = a.values_slice().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::UInt32(a) => {
                let min = a.values_slice().iter().min().map(|v| *v as f64);
                let max = a.values_slice().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::Float32(a) => {
                let max = a
                    .values_slice()
                    .iter()
                    .reduce(|accum, item| if accum >= item { accum } else { item })
                    .map(|v| *v as f64);
                let min = a
                    .values_slice()
                    .iter()
                    .reduce(|accum, item| if accum <= item { accum } else { item })
                    .map(|v| *v as f64);
                (min, max)
            }
            ChannelData::Int64(a) => {
                let min = a.values_slice().iter().min().map(|v| *v as f64);
                let max = a.values_slice().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::UInt64(a) => {
                let min = a.values_slice().iter().min().map(|v| *v as f64);
                let max = a.values_slice().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::Float64(a) => {
                let max = a
                    .values_slice()
                    .iter()
                    .reduce(|accum, item| if accum >= item { accum } else { item })
                    .copied();
                let min = a
                    .values_slice()
                    .iter()
                    .reduce(|accum, item| if accum <= item { accum } else { item })
                    .copied();
                (min, max)
            }
            ChannelData::Complex32(_) => (None, None),
            ChannelData::Complex64(_) => (None, None),
            ChannelData::VariableSizeByteArray(_) => (None, None),
            ChannelData::FixedSizeByteArray(_) => (None, None),
            ChannelData::ArrayDInt8(a) => {
                let min = a.values_slice().iter().min().map(|v| *v as f64);
                let max = a.values_slice().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDUInt8(a) => {
                let min = a.values_slice().iter().min().map(|v| *v as f64);
                let max = a.values_slice().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDInt16(a) => {
                let min = a.values_slice().iter().min().map(|v| *v as f64);
                let max = a.values_slice().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDUInt16(a) => {
                let min = a.values_slice().iter().min().map(|v| *v as f64);
                let max = a.values_slice().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDInt32(a) => {
                let min = a.values_slice().iter().min().map(|v| *v as f64);
                let max = a.values_slice().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDUInt32(a) => {
                let min = a.values_slice().iter().min().map(|v| *v as f64);
                let max = a.values_slice().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDFloat32(a) => {
                let max = a
                    .values_slice()
                    .iter()
                    .reduce(|accum, item| if accum >= item { accum } else { item })
                    .map(|v| *v as f64);
                let min = a
                    .values_slice()
                    .iter()
                    .reduce(|accum, item| if accum <= item { accum } else { item })
                    .map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDInt64(a) => {
                let min = a.values_slice().iter().min().map(|v| *v as f64);
                let max = a.values_slice().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDUInt64(a) => {
                let min = a.values_slice().iter().min().map(|v| *v as f64);
                let max = a.values_slice().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDFloat64(a) => {
                let max = a
                    .values_slice()
                    .iter()
                    .reduce(|accum, item| if accum >= item { accum } else { item })
                    .cloned();
                let min = a
                    .values_slice()
                    .iter()
                    .reduce(|accum, item| if accum <= item { accum } else { item })
                    .cloned();
                (min, max)
            }
            ChannelData::Utf8(_) => (None, None),
        }
    }
    /// convert channel arrow data into dyn Array
    pub fn finish_cloned(&self) -> Arc<dyn Array> {
        match &self {
            ChannelData::Int8(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::UInt8(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::Int16(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::UInt16(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::Int32(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::UInt32(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::Float32(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::Int64(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::UInt64(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::Float64(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::Complex32(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::Complex64(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::Utf8(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::VariableSizeByteArray(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::FixedSizeByteArray(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::ArrayDInt8(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::ArrayDUInt8(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::ArrayDInt16(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::ArrayDUInt16(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::ArrayDInt32(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::ArrayDUInt32(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::ArrayDFloat32(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::ArrayDInt64(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::ArrayDUInt64(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::ArrayDFloat64(a) => Arc::new(a.finish_cloned()) as ArrayRef,
        }
    }
    /// convert channel arrow data into dyn Array
    pub fn finish(&mut self) -> Arc<dyn Array> {
        match self {
            ChannelData::Int8(a) => Arc::new(a.finish()) as ArrayRef,
            ChannelData::UInt8(a) => Arc::new(a.finish()) as ArrayRef,
            ChannelData::Int16(a) => Arc::new(a.finish()) as ArrayRef,
            ChannelData::UInt16(a) => Arc::new(a.finish()) as ArrayRef,
            ChannelData::Int32(a) => Arc::new(a.finish()) as ArrayRef,
            ChannelData::UInt32(a) => Arc::new(a.finish()) as ArrayRef,
            ChannelData::Float32(a) => Arc::new(a.finish()) as ArrayRef,
            ChannelData::Int64(a) => Arc::new(a.finish()) as ArrayRef,
            ChannelData::UInt64(a) => Arc::new(a.finish()) as ArrayRef,
            ChannelData::Float64(a) => Arc::new(a.finish()) as ArrayRef,
            ChannelData::Complex32(a) => Arc::new(a.finish()) as ArrayRef,
            ChannelData::Complex64(a) => Arc::new(a.finish()) as ArrayRef,
            ChannelData::Utf8(a) => Arc::new(a.finish()) as ArrayRef,
            ChannelData::VariableSizeByteArray(a) => Arc::new(a.finish()) as ArrayRef,
            ChannelData::FixedSizeByteArray(a) => Arc::new(a.finish()) as ArrayRef,
            ChannelData::ArrayDInt8(a) => Arc::new(a.finish()) as ArrayRef,
            ChannelData::ArrayDUInt8(a) => Arc::new(a.finish()) as ArrayRef,
            ChannelData::ArrayDInt16(a) => Arc::new(a.finish()) as ArrayRef,
            ChannelData::ArrayDUInt16(a) => Arc::new(a.finish()) as ArrayRef,
            ChannelData::ArrayDInt32(a) => Arc::new(a.finish()) as ArrayRef,
            ChannelData::ArrayDUInt32(a) => Arc::new(a.finish()) as ArrayRef,
            ChannelData::ArrayDFloat32(a) => Arc::new(a.finish()) as ArrayRef,
            ChannelData::ArrayDInt64(a) => Arc::new(a.finish()) as ArrayRef,
            ChannelData::ArrayDUInt64(a) => Arc::new(a.finish()) as ArrayRef,
            ChannelData::ArrayDFloat64(a) => Arc::new(a.finish()) as ArrayRef,
        }
    }
    /// Convert ChannelData into ArrayData
    pub fn to_data(&self) -> ArrayData {
        match &self {
            ChannelData::Int8(a) => a.finish_cloned().to_data(),
            ChannelData::UInt8(a) => a.finish_cloned().to_data(),
            ChannelData::Int16(a) => a.finish_cloned().to_data(),
            ChannelData::UInt16(a) => a.finish_cloned().to_data(),
            ChannelData::Int32(a) => a.finish_cloned().to_data(),
            ChannelData::UInt32(a) => a.finish_cloned().to_data(),
            ChannelData::Float32(a) => a.finish_cloned().to_data(),
            ChannelData::Int64(a) => a.finish_cloned().to_data(),
            ChannelData::UInt64(a) => a.finish_cloned().to_data(),
            ChannelData::Float64(a) => a.finish_cloned().to_data(),
            ChannelData::Complex32(a) => a.finish_cloned().to_data(),
            ChannelData::Complex64(a) => a.finish_cloned().to_data(),
            ChannelData::Utf8(a) => a.finish_cloned().to_data(),
            ChannelData::VariableSizeByteArray(a) => a.finish_cloned().to_data(),
            ChannelData::FixedSizeByteArray(a) => a.finish_cloned().to_data(),
            ChannelData::ArrayDInt8(a) => a.finish_cloned().to_data(),
            ChannelData::ArrayDUInt8(a) => a.finish_cloned().to_data(),
            ChannelData::ArrayDInt16(a) => a.finish_cloned().to_data(),
            ChannelData::ArrayDUInt16(a) => a.finish_cloned().to_data(),
            ChannelData::ArrayDInt32(a) => a.finish_cloned().to_data(),
            ChannelData::ArrayDUInt32(a) => a.finish_cloned().to_data(),
            ChannelData::ArrayDFloat32(a) => a.finish_cloned().to_data(),
            ChannelData::ArrayDInt64(a) => a.finish_cloned().to_data(),
            ChannelData::ArrayDUInt64(a) => a.finish_cloned().to_data(),
            ChannelData::ArrayDFloat64(a) => a.finish_cloned().to_data(),
        }
    }
    /// Change the validity mask of the channel
    pub fn set_validity(&mut self, mask: &mut BooleanBufferBuilder) -> Result<(), Error> {
        match self {
            ChannelData::Int8(a) => {
                let _ = a.validity_slice_mut().insert(mask.as_slice_mut());
            }
            ChannelData::UInt8(a) => {
                let _ = a.validity_slice_mut().insert(mask.as_slice_mut());
            }
            ChannelData::Int16(a) => {
                let _ = a.validity_slice_mut().insert(mask.as_slice_mut());
            }
            ChannelData::UInt16(a) => {
                let _ = a.validity_slice_mut().insert(mask.as_slice_mut());
            }
            ChannelData::Int32(a) => {
                let _ = a.validity_slice_mut().insert(mask.as_slice_mut());
            }
            ChannelData::UInt32(a) => {
                let _ = a.validity_slice_mut().insert(mask.as_slice_mut());
            }
            ChannelData::Float32(a) => {
                let _ = a.validity_slice_mut().insert(mask.as_slice_mut());
            }
            ChannelData::Int64(a) => {
                let _ = a.validity_slice_mut().insert(mask.as_slice_mut());
            }
            ChannelData::UInt64(a) => {
                let _ = a.validity_slice_mut().insert(mask.as_slice_mut());
            }
            ChannelData::Float64(a) => {
                let _ = a.validity_slice_mut().insert(mask.as_slice_mut());
            }
            ChannelData::Complex32(a) => {
                a.set_validity(mask);
            }
            ChannelData::Complex64(a) => {
                a.set_validity(mask);
            }
            ChannelData::Utf8(a) => {
                let _ = a.validity_slice_mut().insert(mask.as_slice_mut());
            }
            ChannelData::VariableSizeByteArray(a) => {
                let _ = a.validity_slice_mut().insert(mask.as_slice_mut());
            }
            ChannelData::FixedSizeByteArray(a) => {
                let array = a.finish();
                let mut new_array =
                    FixedSizeBinaryBuilder::with_capacity(array.len(), array.value_length());
                array
                    .values()
                    .chunks(array.value_length() as usize)
                    .zip(mask.finish().iter())
                    .for_each(|(value, mask)| {
                        if mask {
                            new_array
                                .append_value(value)
                                .expect("failed appending new fixed binary value");
                        } else {
                            new_array.append_null();
                        }
                    });
                *a = new_array;
            }
            ChannelData::ArrayDInt8(a) => {
                a.set_validity(mask);
            }
            ChannelData::ArrayDUInt8(a) => {
                a.set_validity(mask);
            }
            ChannelData::ArrayDInt16(a) => {
                a.set_validity(mask);
            }
            ChannelData::ArrayDUInt16(a) => {
                a.set_validity(mask);
            }
            ChannelData::ArrayDInt32(a) => {
                a.set_validity(mask);
            }
            ChannelData::ArrayDUInt32(a) => {
                a.set_validity(mask);
            }
            ChannelData::ArrayDFloat32(a) => {
                a.set_validity(mask);
            }
            ChannelData::ArrayDInt64(a) => {
                a.set_validity(mask);
            }
            ChannelData::ArrayDUInt64(a) => {
                a.set_validity(mask);
            }
            ChannelData::ArrayDFloat64(a) => {
                a.set_validity(mask);
            }
        }
        Ok(())
    }
    /// Returns the channel's mask NullBuffer if existing
    pub fn validity(&self) -> Option<NullBuffer> {
        match self {
            ChannelData::Int8(a) => a.finish_cloned().nulls().cloned(),
            ChannelData::UInt8(a) => a.finish_cloned().nulls().cloned(),
            ChannelData::Int16(a) => a.finish_cloned().nulls().cloned(),
            ChannelData::UInt16(a) => a.finish_cloned().nulls().cloned(),
            ChannelData::Int32(a) => a.finish_cloned().nulls().cloned(),
            ChannelData::UInt32(a) => a.finish_cloned().nulls().cloned(),
            ChannelData::Float32(a) => a.finish_cloned().nulls().cloned(),
            ChannelData::Int64(a) => a.finish_cloned().nulls().cloned(),
            ChannelData::UInt64(a) => a.finish_cloned().nulls().cloned(),
            ChannelData::Float64(a) => a.finish_cloned().nulls().cloned(),
            ChannelData::Complex32(a) => a.finish_cloned().nulls().cloned(),
            ChannelData::Complex64(a) => a.finish_cloned().nulls().cloned(),
            ChannelData::Utf8(a) => a.finish_cloned().nulls().cloned(),
            ChannelData::VariableSizeByteArray(a) => a.finish_cloned().nulls().cloned(),
            ChannelData::FixedSizeByteArray(a) => a.finish_cloned().nulls().cloned(),
            ChannelData::ArrayDInt8(a) => a.finish_cloned().nulls().cloned(),
            ChannelData::ArrayDUInt8(a) => a.finish_cloned().nulls().cloned(),
            ChannelData::ArrayDInt16(a) => a.finish_cloned().nulls().cloned(),
            ChannelData::ArrayDUInt16(a) => a.finish_cloned().nulls().cloned(),
            ChannelData::ArrayDInt32(a) => a.finish_cloned().nulls().cloned(),
            ChannelData::ArrayDUInt32(a) => a.finish_cloned().nulls().cloned(),
            ChannelData::ArrayDFloat32(a) => a.finish_cloned().nulls().cloned(),
            ChannelData::ArrayDInt64(a) => a.finish_cloned().nulls().cloned(),
            ChannelData::ArrayDUInt64(a) => a.finish_cloned().nulls().cloned(),
            ChannelData::ArrayDFloat64(a) => a.finish_cloned().nulls().cloned(),
        }
    }
    /// Returns the channel's validity mask as a slice
    pub fn validity_slice(&self) -> Option<&[u8]> {
        match self {
            ChannelData::Int8(a) => a.validity_slice(),
            ChannelData::UInt8(a) => a.validity_slice(),
            ChannelData::Int16(a) => a.validity_slice(),
            ChannelData::UInt16(a) => a.validity_slice(),
            ChannelData::Int32(a) => a.validity_slice(),
            ChannelData::UInt32(a) => a.validity_slice(),
            ChannelData::Float32(a) => a.validity_slice(),
            ChannelData::Int64(a) => a.validity_slice(),
            ChannelData::UInt64(a) => a.validity_slice(),
            ChannelData::Float64(a) => a.validity_slice(),
            ChannelData::Complex32(a) => a.validity_slice(),
            ChannelData::Complex64(a) => a.validity_slice(),
            ChannelData::Utf8(a) => a.validity_slice(),
            ChannelData::VariableSizeByteArray(a) => a.validity_slice(),
            ChannelData::FixedSizeByteArray(_a) => None,
            ChannelData::ArrayDInt8(a) => a.validity_slice(),
            ChannelData::ArrayDUInt8(a) => a.validity_slice(),
            ChannelData::ArrayDInt16(a) => a.validity_slice(),
            ChannelData::ArrayDUInt16(a) => a.validity_slice(),
            ChannelData::ArrayDInt32(a) => a.validity_slice(),
            ChannelData::ArrayDUInt32(a) => a.validity_slice(),
            ChannelData::ArrayDFloat32(a) => a.validity_slice(),
            ChannelData::ArrayDInt64(a) => a.validity_slice(),
            ChannelData::ArrayDUInt64(a) => a.validity_slice(),
            ChannelData::ArrayDFloat64(a) => a.validity_slice(),
        }
    }
    /// returns True if a validity mask is existing for the channel
    pub fn nullable(&self) -> bool {
        match self {
            ChannelData::Int8(a) => a.validity_slice().is_some(),
            ChannelData::UInt8(a) => a.validity_slice().is_some(),
            ChannelData::Int16(a) => a.validity_slice().is_some(),
            ChannelData::UInt16(a) => a.validity_slice().is_some(),
            ChannelData::Int32(a) => a.validity_slice().is_some(),
            ChannelData::UInt32(a) => a.validity_slice().is_some(),
            ChannelData::Float32(a) => a.validity_slice().is_some(),
            ChannelData::Int64(a) => a.validity_slice().is_some(),
            ChannelData::UInt64(a) => a.validity_slice().is_some(),
            ChannelData::Float64(a) => a.validity_slice().is_some(),
            ChannelData::Complex32(a) => a.nulls().is_some(),
            ChannelData::Complex64(a) => a.nulls().is_some(),
            ChannelData::Utf8(a) => a.validity_slice().is_some(),
            ChannelData::VariableSizeByteArray(a) => a.validity_slice().is_some(),
            ChannelData::FixedSizeByteArray(a) => a.finish_cloned().nulls().is_some(),
            ChannelData::ArrayDInt8(a) => a.nulls().is_some(),
            ChannelData::ArrayDUInt8(a) => a.nulls().is_some(),
            ChannelData::ArrayDInt16(a) => a.nulls().is_some(),
            ChannelData::ArrayDUInt16(a) => a.nulls().is_some(),
            ChannelData::ArrayDInt32(a) => a.nulls().is_some(),
            ChannelData::ArrayDUInt32(a) => a.nulls().is_some(),
            ChannelData::ArrayDFloat32(a) => a.nulls().is_some(),
            ChannelData::ArrayDInt64(a) => a.nulls().is_some(),
            ChannelData::ArrayDUInt64(a) => a.nulls().is_some(),
            ChannelData::ArrayDFloat64(a) => a.nulls().is_some(),
        }
    }
    /// converts the ChannelData into a ArrayRef
    pub fn as_ref(&self) -> Arc<dyn Array> {
        match self {
            ChannelData::Int8(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::UInt8(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::Int16(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::UInt16(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::Int32(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::UInt32(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::Float32(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::Int64(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::UInt64(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::Float64(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::Complex32(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::Complex64(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::Utf8(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::VariableSizeByteArray(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::FixedSizeByteArray(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::ArrayDInt8(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::ArrayDUInt8(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::ArrayDInt16(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::ArrayDUInt16(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::ArrayDInt32(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::ArrayDUInt32(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::ArrayDFloat32(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::ArrayDInt64(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::ArrayDUInt64(a) => Arc::new(a.finish_cloned()) as ArrayRef,
            ChannelData::ArrayDFloat64(a) => Arc::new(a.finish_cloned()) as ArrayRef,
        }
    }
    #[cfg(feature = "numpy")]
    pub fn get_dtype(&self) -> NumpyDType {
        use super::dtype::NumpyDType;

        match self {
            ChannelData::Int8(a) => NumpyDType {
                shape: vec![a.len()],
                kind: "i1".to_string(),
            },
            ChannelData::UInt8(a) => NumpyDType {
                shape: vec![a.len()],
                kind: "u1".to_string(),
            },
            ChannelData::Int16(a) => NumpyDType {
                shape: vec![a.len()],
                kind: "i2".to_string(),
            },
            ChannelData::UInt16(a) => NumpyDType {
                shape: vec![a.len()],
                kind: "u2".to_string(),
            },
            ChannelData::Int32(a) => NumpyDType {
                shape: vec![a.len()],
                kind: "i4".to_string(),
            },
            ChannelData::UInt32(a) => NumpyDType {
                shape: vec![a.len()],
                kind: "u4".to_string(),
            },
            ChannelData::Float32(a) => NumpyDType {
                shape: vec![a.len()],
                kind: "f4".to_string(),
            },
            ChannelData::Int64(a) => NumpyDType {
                shape: vec![a.len()],
                kind: "i8".to_string(),
            },
            ChannelData::UInt64(a) => NumpyDType {
                shape: vec![a.len()],
                kind: "u8".to_string(),
            },
            ChannelData::Float64(a) => NumpyDType {
                shape: vec![a.len()],
                kind: "f8".to_string(),
            },
            ChannelData::Complex32(a) => NumpyDType {
                shape: vec![a.len() * 2],
                kind: "f4".to_string(),
            },
            ChannelData::Complex64(a) => NumpyDType {
                shape: vec![a.len() * 2],
                kind: "f8".to_string(),
            },
            ChannelData::Utf8(a) => NumpyDType {
                shape: vec![a.len()],
                kind: format!("U{}", self.byte_count()),
            },
            ChannelData::VariableSizeByteArray(a) => NumpyDType {
                shape: vec![a.len()],
                kind: format!("S{}", self.byte_count()),
            },
            ChannelData::FixedSizeByteArray(a) => NumpyDType {
                shape: vec![a.len()],
                kind: format!("S{}", self.byte_count()),
            },
            ChannelData::ArrayDInt8(a) => NumpyDType {
                shape: a.shape().to_vec(),
                kind: "i1".to_string(),
            },
            ChannelData::ArrayDUInt8(a) => NumpyDType {
                shape: a.shape().to_vec(),
                kind: "u1".to_string(),
            },
            ChannelData::ArrayDInt16(a) => NumpyDType {
                shape: a.shape().to_vec(),
                kind: "i2".to_string(),
            },
            ChannelData::ArrayDUInt16(a) => NumpyDType {
                shape: a.shape().to_vec(),
                kind: "u2".to_string(),
            },
            ChannelData::ArrayDInt32(a) => NumpyDType {
                shape: a.shape().to_vec(),
                kind: "i4".to_string(),
            },
            ChannelData::ArrayDUInt32(a) => NumpyDType {
                shape: a.shape().to_vec(),
                kind: "i4".to_string(),
            },
            ChannelData::ArrayDFloat32(a) => NumpyDType {
                shape: a.shape().to_vec(),
                kind: "f4".to_string(),
            },
            ChannelData::ArrayDInt64(a) => NumpyDType {
                shape: a.shape().to_vec(),
                kind: "i8".to_string(),
            },
            ChannelData::ArrayDUInt64(a) => NumpyDType {
                shape: a.shape().to_vec(),
                kind: "u8".to_string(),
            },
            ChannelData::ArrayDFloat64(a) => NumpyDType {
                shape: a.shape().to_vec(),
                kind: "f8".to_string(),
            },
        }
    }
}

impl Default for ChannelData {
    fn default() -> Self {
        ChannelData::UInt8(PrimitiveBuilder::new())
    }
}

/// Initialises a channel array type depending of cn_type, cn_data_type and if array
pub fn data_type_init(
    cn_type: u8,
    cn_data_type: u8,
    n_bytes: u32,
    list_size: usize,
) -> Result<ChannelData, Error> {
    if list_size == 1 {
        // Not an array
        if cn_type != 3 || cn_type != 6 {
            // not virtual channel or vlsd
            match cn_data_type {
                0 | 1 => {
                    // unsigned int
                    if n_bytes <= 1 {
                        Ok(ChannelData::UInt8(PrimitiveBuilder::new()))
                    } else if n_bytes == 2 {
                        Ok(ChannelData::UInt16(PrimitiveBuilder::new()))
                    } else if n_bytes <= 4 {
                        Ok(ChannelData::UInt32(PrimitiveBuilder::new()))
                    } else {
                        Ok(ChannelData::UInt64(PrimitiveBuilder::new()))
                    }
                }
                2 | 3 => {
                    // signed int
                    if n_bytes <= 1 {
                        Ok(ChannelData::Int8(PrimitiveBuilder::new()))
                    } else if n_bytes == 2 {
                        Ok(ChannelData::Int16(PrimitiveBuilder::new()))
                    } else if n_bytes <= 4 {
                        Ok(ChannelData::Int32(PrimitiveBuilder::new()))
                    } else {
                        Ok(ChannelData::Int64(PrimitiveBuilder::new()))
                    }
                }
                4 | 5 => {
                    // float
                    if n_bytes <= 4 {
                        Ok(ChannelData::Float32(PrimitiveBuilder::new()))
                    } else {
                        Ok(ChannelData::Float64(PrimitiveBuilder::new()))
                    }
                }
                15 | 16 => {
                    // complex, should not happen here
                    if n_bytes <= 4 {
                        Ok(ChannelData::Complex32(ComplexArrow::new()))
                    } else {
                        Ok(ChannelData::Complex64(ComplexArrow::new()))
                    }
                }
                6..=9 => {
                    // String UTF8
                    Ok(ChannelData::Utf8(LargeStringBuilder::new()))
                }
                _ => {
                    // bytearray
                    if cn_type == 1 {
                        // VLSD
                        Ok(ChannelData::VariableSizeByteArray(LargeBinaryBuilder::new()))
                    } else {
                        Ok(ChannelData::FixedSizeByteArray(
                            FixedSizeBinaryBuilder::new(n_bytes as i32),
                        ))
                    }
                }
            }
        } else {
            // virtual channels, cn_bit_count = 0 -> n_bytes = 0, must be LE unsigned int
            Ok(ChannelData::UInt64(PrimitiveBuilder::new()))
        }
    } else if cn_type != 3 && cn_type != 6 {
        // Array not virtual
        match cn_data_type {
            0 | 1 => {
                // unsigned int
                if n_bytes <= 1 {
                    Ok(ChannelData::ArrayDUInt8(TensorArrow::new()))
                } else if n_bytes == 2 {
                    Ok(ChannelData::ArrayDUInt16(TensorArrow::new()))
                } else if n_bytes <= 4 {
                    Ok(ChannelData::ArrayDUInt32(TensorArrow::new()))
                } else {
                    Ok(ChannelData::ArrayDUInt64(TensorArrow::new()))
                }
            }
            2 | 3 => {
                // signed int
                if n_bytes <= 1 {
                    Ok(ChannelData::ArrayDInt8(TensorArrow::new()))
                } else if n_bytes == 2 {
                    Ok(ChannelData::ArrayDInt16(TensorArrow::new()))
                } else if n_bytes == 4 {
                    Ok(ChannelData::ArrayDInt32(TensorArrow::new()))
                } else {
                    Ok(ChannelData::ArrayDInt64(TensorArrow::new()))
                }
            }
            4 | 5 => {
                // float
                if n_bytes <= 4 {
                    Ok(ChannelData::ArrayDFloat32(TensorArrow::new()))
                } else {
                    Ok(ChannelData::ArrayDFloat64(TensorArrow::new()))
                }
            }
            15 | 16 => {
                // complex
                if list_size == 2 {
                    if n_bytes <= 4 {
                        Ok(ChannelData::Complex32(ComplexArrow::new()))
                    } else {
                        Ok(ChannelData::Complex64(ComplexArrow::new()))
                    }
                } else {
                    // tensor of complex
                    if n_bytes <= 4 {
                        unimplemented!(
                            "Tensor of complex is not implemented, if needed, please inform"
                        );
                    } else {
                        unimplemented!(
                            "Tensor of complex is not implemented, if needed, please inform"
                        );
                    }
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

/// converts a dyn Array in a ChannelData
pub fn try_from(value: &dyn Array) -> Result<ChannelData, Error> {
    match value.data_type() {
        DataType::Null => {
            let data = as_primitive_array::<UInt8Type>(value);
            let mut new_data = PrimitiveBuilder::with_capacity(data.len());
            data.iter().for_each(|v| new_data.append_option(v));
            Ok(ChannelData::UInt8(new_data))
        }
        DataType::Boolean => {
            let data = as_primitive_array::<UInt8Type>(value);
            let mut new_data = PrimitiveBuilder::with_capacity(data.len());
            data.iter().for_each(|v| new_data.append_option(v));
            Ok(ChannelData::UInt8(new_data))
        }
        DataType::Int8 => {
            let data = as_primitive_array::<Int8Type>(value);
            let mut new_data = PrimitiveBuilder::with_capacity(data.len());
            data.iter().for_each(|v| new_data.append_option(v));
            Ok(ChannelData::Int8(new_data))
        }
        DataType::Int16 => {
            let data = as_primitive_array::<Int16Type>(value);
            let mut new_data = PrimitiveBuilder::with_capacity(data.len());
            data.iter().for_each(|v| new_data.append_option(v));
            Ok(ChannelData::Int16(new_data))
        }
        DataType::Int32 => {
            let data = as_primitive_array::<Int32Type>(value);
            let mut new_data = PrimitiveBuilder::with_capacity(data.len());
            data.iter().for_each(|v| new_data.append_option(v));
            Ok(ChannelData::Int32(new_data))
        }
        DataType::Int64 => {
            let data = as_primitive_array::<Int64Type>(value);
            let mut new_data = PrimitiveBuilder::with_capacity(data.len());
            data.iter().for_each(|v| new_data.append_option(v));
            Ok(ChannelData::Int64(new_data))
        }
        DataType::UInt8 => {
            let data = as_primitive_array::<UInt8Type>(value);
            let mut new_data = PrimitiveBuilder::with_capacity(data.len());
            data.iter().for_each(|v| new_data.append_option(v));
            Ok(ChannelData::UInt8(new_data))
        }
        DataType::UInt16 => {
            let data = as_primitive_array::<UInt16Type>(value);
            let mut new_data = PrimitiveBuilder::with_capacity(data.len());
            data.iter().for_each(|v| new_data.append_option(v));
            Ok(ChannelData::UInt16(new_data))
        }
        DataType::UInt32 => {
            let data = as_primitive_array::<UInt32Type>(value);
            let mut new_data = PrimitiveBuilder::with_capacity(data.len());
            data.iter().for_each(|v| new_data.append_option(v));
            Ok(ChannelData::UInt32(new_data))
        }
        DataType::UInt64 => {
            let data = as_primitive_array::<UInt64Type>(value);
            let mut new_data = PrimitiveBuilder::with_capacity(data.len());
            data.iter().for_each(|v| new_data.append_option(v));
            Ok(ChannelData::UInt64(new_data))
        }
        DataType::Float16 => todo!(),
        DataType::Float32 => {
            let data = as_primitive_array::<Float32Type>(value);
            let mut new_data = PrimitiveBuilder::with_capacity(data.len());
            data.iter().for_each(|v| new_data.append_option(v));
            Ok(ChannelData::Float32(new_data))
        }
        DataType::Float64 => {
            let data = as_primitive_array::<Float64Type>(value);
            let mut new_data = PrimitiveBuilder::with_capacity(data.len());
            data.iter().for_each(|v| new_data.append_option(v));
            Ok(ChannelData::Float64(new_data))
        }
        DataType::Binary => {
            let array = value
                .as_any()
                .downcast_ref::<BinaryArray>()
                .context("could not downcast to Binary array")?;
            let array_i64 = LargeBinaryArray::from_opt_vec(array.iter().collect());
            Ok(ChannelData::VariableSizeByteArray(
                array_i64
                    .into_builder()
                    .expect("could not convert binary i64 into mutable array"),
            ))
        }
        DataType::FixedSizeBinary(size) => {
            let array = value
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .context("could not downcast to fixed size binary array")?;
            let mut new_array = FixedSizeBinaryBuilder::with_capacity(array.len(), *size);
            if let Some(validity) = array.logical_nulls() {
                array
                    .values()
                    .chunks(array.value_length() as usize)
                    .zip(validity.iter())
                    .for_each(|(value, validity)| {
                        if validity {
                            new_array
                                .append_value(value)
                                .expect("failed appending new fixed binary value");
                        } else {
                            new_array.append_null();
                        }
                    });
            } else {
                array
                    .values()
                    .chunks(array.value_length() as usize)
                    .for_each(|value| {
                        new_array
                            .append_value(value)
                            .expect("failed appending new fixed binary value");
                    });
            }
            Ok(ChannelData::FixedSizeByteArray(new_array))
        }
        DataType::LargeBinary => {
            let array = value
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .context("could not downcast to Large Binary array")?;
            Ok(ChannelData::VariableSizeByteArray(
                LargeBinaryArray::from_opt_vec(array.iter().collect())
                    .into_builder()
                    .expect("could not convert binary i64 into mutable array"),
            ))
        }
        DataType::Utf8 => {
            let array = value
                .as_any()
                .downcast_ref::<StringArray>()
                .context("could not downcast to Utf8 array")?;
            let array_i64 = LargeStringArray::from(array.iter().collect::<Vec<_>>());
            Ok(ChannelData::Utf8(
                array_i64
                    .into_builder()
                    .expect("could not convert utf8 into mutable array"),
            ))
        }
        DataType::LargeUtf8 => {
            let array = value
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .context("could not downcast to Large Utf8 array")?;
            Ok(ChannelData::Utf8(
                LargeStringArray::from(array.iter().collect::<Vec<_>>())
                    .into_builder()
                    .expect("could not convert large utf8 into mutable array"),
            ))
        }
        DataType::FixedSizeList(_, size) => {
            // used for complex number, size of 2
            let array = value
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .context("could not downcast to fixed size list array, used for complex")?;
            if *size == 2 {
                match array.value_type() {
                    DataType::Float32 => {
                        let data = as_primitive_array::<Float32Type>(value);
                        let mut new_data = PrimitiveBuilder::with_capacity(data.len());
                        data.iter().for_each(|v| new_data.append_option(v));
                        Ok(ChannelData::Float32(new_data))
                    }
                    DataType::Float64 => {
                        let data = as_primitive_array::<Float64Type>(value);
                        let mut new_data = PrimitiveBuilder::with_capacity(data.len());
                        data.iter().for_each(|v| new_data.append_option(v));
                        Ok(ChannelData::Float64(new_data))
                    }
                    _ => bail!("FixedSizeList shall be either f23 or f64 to be used for complex"),
                }
            } else {
                bail!("FixedSizeList is not of size 2, to be used for complex")
            }
        }
        _ => todo!(),
    }
}

impl fmt::Display for ChannelData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let format_option = FormatOptions::new();
        let data = self.as_ref();
        let displayer =
            ArrayFormatter::try_new(&data, &format_option).map_err(|_| std::fmt::Error)?;
        for i in 0..self.len() {
            write!(f, " {}", displayer.value(i))?;
        }
        Ok(())
    }
}
