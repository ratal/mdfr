//! this module holds the channel data enum and related implementations

use anyhow::{bail, Context, Error, Result};
use arrow::array::{
    as_primitive_array, Array, ArrayBuilder, ArrayRef, BinaryArray, BooleanBufferBuilder,
    FixedSizeBinaryArray, FixedSizeBinaryBuilder, FixedSizeListArray, Float32Array, Float64Array,
    Int8Builder, LargeBinaryArray, LargeBinaryBuilder, LargeStringArray, LargeStringBuilder,
    PrimitiveBuilder, StringArray,
};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{
    DataType, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type,
    UInt32Type, UInt64Type, UInt8Type,
};
use itertools::Itertools;

use std::fmt;
use std::sync::Arc;

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
    Complex32(PrimitiveBuilder<Float32Type>),
    Complex64(PrimitiveBuilder<Float64Type>),
    Utf8(LargeStringBuilder),
    VariableSizeByteArray(LargeBinaryBuilder),
    FixedSizeByteArray(FixedSizeBinaryBuilder),
    ArrayDInt8((PrimitiveBuilder<Int8Type>, (Vec<usize>, Order))),
    ArrayDUInt8((PrimitiveBuilder<UInt8Type>, (Vec<usize>, Order))),
    ArrayDInt16((PrimitiveBuilder<Int16Type>, (Vec<usize>, Order))),
    ArrayDUInt16((PrimitiveBuilder<UInt16Type>, (Vec<usize>, Order))),
    ArrayDInt32((PrimitiveBuilder<Int32Type>, (Vec<usize>, Order))),
    ArrayDUInt32((PrimitiveBuilder<UInt32Type>, (Vec<usize>, Order))),
    ArrayDFloat32((PrimitiveBuilder<Float32Type>, (Vec<usize>, Order))),
    ArrayDInt64((PrimitiveBuilder<Int64Type>, (Vec<usize>, Order))),
    ArrayDUInt64((PrimitiveBuilder<UInt64Type>, (Vec<usize>, Order))),
    ArrayDFloat64((PrimitiveBuilder<Float64Type>, (Vec<usize>, Order))),
}

impl PartialEq for ChannelData {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Int8(l0), Self::Int8(r0)) => l0.finish() == r0.finish(),
            (Self::UInt8(l0), Self::UInt8(r0)) => l0.finish() == r0.finish(),
            (Self::Int16(l0), Self::Int16(r0)) => l0.finish() == r0.finish(),
            (Self::UInt16(l0), Self::UInt16(r0)) => l0.finish() == r0.finish(),
            (Self::Int32(l0), Self::Int32(r0)) => l0.finish() == r0.finish(),
            (Self::UInt32(l0), Self::UInt32(r0)) => l0.finish() == r0.finish(),
            (Self::Float32(l0), Self::Float32(r0)) => l0.finish() == r0.finish(),
            (Self::Int64(l0), Self::Int64(r0)) => l0.finish() == r0.finish(),
            (Self::UInt64(l0), Self::UInt64(r0)) => l0.finish() == r0.finish(),
            (Self::Float64(l0), Self::Float64(r0)) => l0.finish() == r0.finish(),
            (Self::Complex32(l0), Self::Complex32(r0)) => l0.finish() == r0.finish(),
            (Self::Complex64(l0), Self::Complex64(r0)) => l0.finish() == r0.finish(),
            (Self::Utf8(l0), Self::Utf8(r0)) => l0.finish() == r0.finish(),
            (Self::VariableSizeByteArray(l0), Self::VariableSizeByteArray(r0)) => {
                l0.finish() == r0.finish()
            }
            (Self::FixedSizeByteArray(l0), Self::FixedSizeByteArray(r0)) => {
                l0.finish() == r0.finish()
            }
            (Self::ArrayDInt8((l0, _)), Self::ArrayDInt8((r0, _))) => l0.finish() == r0.finish(),
            (Self::ArrayDUInt8((l0, _)), Self::ArrayDUInt8((r0, _))) => l0.finish() == r0.finish(),
            (Self::ArrayDInt16((l0, _)), Self::ArrayDInt16((r0, _))) => l0.finish() == r0.finish(),
            (Self::ArrayDUInt16((l0, _)), Self::ArrayDUInt16((r0, _))) => {
                l0.finish() == r0.finish()
            }
            (Self::ArrayDInt32((l0, _)), Self::ArrayDInt32((r0, _))) => l0.finish() == r0.finish(),
            (Self::ArrayDUInt32((l0, _)), Self::ArrayDUInt32((r0, _))) => {
                l0.finish() == r0.finish()
            }
            (Self::ArrayDFloat32((l0, _)), Self::ArrayDFloat32((r0, _))) => {
                l0.finish() == r0.finish()
            }
            (Self::ArrayDInt64((l0, _)), Self::ArrayDInt64((r0, _))) => l0.finish() == r0.finish(),
            (Self::ArrayDUInt64((l0, _)), Self::ArrayDUInt64((r0, _))) => {
                l0.finish() == r0.finish()
            }
            (Self::ArrayDFloat64((l0, _)), Self::ArrayDFloat64((r0, _))) => {
                l0.finish() == r0.finish()
            }
            _ => false,
        }
    }
}

impl Clone for ChannelData {
    fn clone(&self) -> Self {
        match self {
            Self::Int8(arg0) => Self::Int8(
                arg0.finish()
                    .clone()
                    .into_builder()
                    .expect("failed getting back mutable array"),
            ),
            Self::UInt8(arg0) => Self::UInt8(
                arg0.finish()
                    .clone()
                    .into_builder()
                    .expect("failed getting back mutable array"),
            ),
            Self::Int16(arg0) => Self::Int16(
                arg0.finish()
                    .clone()
                    .into_builder()
                    .expect("failed getting back mutable array"),
            ),
            Self::UInt16(arg0) => Self::UInt16(
                arg0.finish()
                    .clone()
                    .into_builder()
                    .expect("failed getting back mutable array"),
            ),
            Self::Int32(arg0) => Self::Int32(
                arg0.finish()
                    .clone()
                    .into_builder()
                    .expect("failed getting back mutable array"),
            ),
            Self::UInt32(arg0) => Self::UInt32(
                arg0.finish()
                    .clone()
                    .into_builder()
                    .expect("failed getting back mutable array"),
            ),
            Self::Float32(arg0) => Self::Float32(
                arg0.finish()
                    .clone()
                    .into_builder()
                    .expect("failed getting back mutable array"),
            ),
            Self::Int64(arg0) => Self::Int64(
                arg0.finish()
                    .clone()
                    .into_builder()
                    .expect("failed getting back mutable array"),
            ),
            Self::UInt64(arg0) => Self::UInt64(
                arg0.finish()
                    .clone()
                    .into_builder()
                    .expect("failed getting back mutable array"),
            ),
            Self::Float64(arg0) => Self::Float64(
                arg0.finish()
                    .clone()
                    .into_builder()
                    .expect("failed getting back mutable array"),
            ),
            Self::Complex32(arg0) => Self::Complex32(
                arg0.finish()
                    .clone()
                    .into_builder()
                    .expect("failed getting back mutable array"),
            ),
            Self::Complex64(arg0) => Self::Complex64(
                arg0.finish()
                    .clone()
                    .into_builder()
                    .expect("failed getting back mutable array"),
            ),
            Self::Utf8(arg0) => Self::Utf8(
                arg0.finish()
                    .clone()
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
            Self::ArrayDInt8((arg0, shape)) => Self::ArrayDInt8((
                arg0.finish()
                    .clone()
                    .into_builder()
                    .expect("failed getting back mutable array"),
                shape.clone(),
            )),
            Self::ArrayDUInt8((arg0, shape)) => Self::ArrayDUInt8((
                arg0.finish()
                    .clone()
                    .into_builder()
                    .expect("failed getting back mutable array"),
                shape.clone(),
            )),
            Self::ArrayDInt16((arg0, shape)) => Self::ArrayDInt16((
                arg0.finish()
                    .clone()
                    .into_builder()
                    .expect("failed getting back mutable array"),
                shape.clone(),
            )),
            Self::ArrayDUInt16((arg0, shape)) => Self::ArrayDUInt16((
                arg0.finish()
                    .clone()
                    .into_builder()
                    .expect("failed getting back mutable array"),
                shape.clone(),
            )),
            Self::ArrayDInt32((arg0, shape)) => Self::ArrayDInt32((
                arg0.finish()
                    .clone()
                    .into_builder()
                    .expect("failed getting back mutable array"),
                shape.clone(),
            )),
            Self::ArrayDUInt32((arg0, shape)) => Self::ArrayDUInt32((
                arg0.finish()
                    .clone()
                    .into_builder()
                    .expect("failed getting back mutable array"),
                shape.clone(),
            )),
            Self::ArrayDFloat32((arg0, shape)) => Self::ArrayDFloat32((
                arg0.finish()
                    .clone()
                    .into_builder()
                    .expect("failed getting back mutable array"),
                shape.clone(),
            )),
            Self::ArrayDInt64((arg0, shape)) => Self::ArrayDInt64((
                arg0.finish()
                    .clone()
                    .into_builder()
                    .expect("failed getting back mutable array"),
                shape.clone(),
            )),
            Self::ArrayDUInt64((arg0, shape)) => Self::ArrayDUInt64((
                arg0.finish()
                    .clone()
                    .into_builder()
                    .expect("failed getting back mutable array"),
                shape.clone(),
            )),
            Self::ArrayDFloat64((arg0, shape)) => Self::ArrayDFloat64((
                arg0.finish()
                    .clone()
                    .into_builder()
                    .expect("failed getting back mutable array"),
                shape.clone(),
            )),
        }
    }
}

/// Order of the array, Row or Column Major (first)
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Order {
    #[default]
    RowMajor,
    ColumnMajor,
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
                ChannelData::Int8(_) => Ok(ChannelData::Int8(PrimitiveBuilder::with_capacity(
                    cycle_count as usize,
                ))),
                ChannelData::UInt8(_) => Ok(ChannelData::UInt8(PrimitiveBuilder::with_capacity(
                    cycle_count as usize,
                ))),
                ChannelData::Int16(_) => Ok(ChannelData::Int16(PrimitiveBuilder::with_capacity(
                    cycle_count as usize,
                ))),
                ChannelData::UInt16(_) => Ok(ChannelData::UInt16(PrimitiveBuilder::with_capacity(
                    cycle_count as usize,
                ))),
                ChannelData::Int32(_) => Ok(ChannelData::Int32(PrimitiveBuilder::with_capacity(
                    cycle_count as usize,
                ))),
                ChannelData::UInt32(_) => Ok(ChannelData::UInt32(PrimitiveBuilder::with_capacity(
                    cycle_count as usize,
                ))),
                ChannelData::Float32(_) => Ok(ChannelData::Float32(
                    PrimitiveBuilder::with_capacity(cycle_count as usize),
                )),
                ChannelData::Int64(_) => Ok(ChannelData::Int64(PrimitiveBuilder::with_capacity(
                    cycle_count as usize,
                ))),
                ChannelData::UInt64(_) => Ok(ChannelData::UInt64(PrimitiveBuilder::with_capacity(
                    cycle_count as usize,
                ))),
                ChannelData::Float64(_) => Ok(ChannelData::Float64(
                    PrimitiveBuilder::with_capacity(cycle_count as usize),
                )),
                ChannelData::Complex32(_) => Ok(ChannelData::Float32(
                    PrimitiveBuilder::with_capacity(cycle_count as usize * 2),
                )),
                ChannelData::Complex64(_) => Ok(ChannelData::Float64(
                    PrimitiveBuilder::with_capacity(cycle_count as usize * 2),
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
                ChannelData::ArrayDInt8(_) => Ok(ChannelData::ArrayDInt8((
                    PrimitiveBuilder::with_capacity(
                        cycle_count as usize * shape.0.iter().product::<usize>(),
                    ),
                    shape,
                ))),
                ChannelData::ArrayDUInt8(_) => Ok(ChannelData::ArrayDUInt8((
                    PrimitiveBuilder::with_capacity(
                        cycle_count as usize * shape.0.iter().product::<usize>(),
                    ),
                    shape,
                ))),
                ChannelData::ArrayDInt16(_) => Ok(ChannelData::ArrayDInt16((
                    PrimitiveBuilder::with_capacity(
                        cycle_count as usize * shape.0.iter().product::<usize>(),
                    ),
                    shape,
                ))),
                ChannelData::ArrayDUInt16(_) => Ok(ChannelData::ArrayDUInt16((
                    PrimitiveBuilder::with_capacity(
                        cycle_count as usize * shape.0.iter().product::<usize>(),
                    ),
                    shape,
                ))),
                ChannelData::ArrayDInt32(_) => Ok(ChannelData::ArrayDInt32((
                    PrimitiveBuilder::with_capacity(
                        cycle_count as usize * shape.0.iter().product::<usize>(),
                    ),
                    shape,
                ))),
                ChannelData::ArrayDUInt32(_) => Ok(ChannelData::ArrayDUInt32((
                    PrimitiveBuilder::with_capacity(
                        cycle_count as usize * shape.0.iter().product::<usize>(),
                    ),
                    shape,
                ))),
                ChannelData::ArrayDFloat32(_) => Ok(ChannelData::ArrayDFloat32((
                    PrimitiveBuilder::with_capacity(
                        cycle_count as usize * shape.0.iter().product::<usize>(),
                    ),
                    shape,
                ))),
                ChannelData::ArrayDInt64(_) => Ok(ChannelData::ArrayDInt64((
                    PrimitiveBuilder::with_capacity(
                        cycle_count as usize * shape.0.iter().product::<usize>(),
                    ),
                    shape,
                ))),
                ChannelData::ArrayDUInt64(_) => Ok(ChannelData::ArrayDUInt64((
                    PrimitiveBuilder::with_capacity(
                        cycle_count as usize * shape.0.iter().product::<usize>(),
                    ),
                    shape,
                ))),
                ChannelData::ArrayDFloat64(_) => Ok(ChannelData::ArrayDFloat64((
                    PrimitiveBuilder::with_capacity(
                        cycle_count as usize * shape.0.iter().product::<usize>(),
                    ),
                    shape,
                ))),
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
            ChannelData::ArrayDInt8(data) => data.0.is_empty(),
            ChannelData::ArrayDUInt8(data) => data.0.is_empty(),
            ChannelData::ArrayDInt16(data) => data.0.is_empty(),
            ChannelData::ArrayDUInt16(data) => data.0.is_empty(),
            ChannelData::ArrayDInt32(data) => data.0.is_empty(),
            ChannelData::ArrayDUInt32(data) => data.0.is_empty(),
            ChannelData::ArrayDFloat32(data) => data.0.is_empty(),
            ChannelData::ArrayDInt64(data) => data.0.is_empty(),
            ChannelData::ArrayDUInt64(data) => data.0.is_empty(),
            ChannelData::ArrayDFloat64(data) => data.0.is_empty(),
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
            ChannelData::ArrayDInt8(data) => data.0.len(),
            ChannelData::ArrayDUInt8(data) => data.0.len(),
            ChannelData::ArrayDInt16(data) => data.0.len(),
            ChannelData::ArrayDUInt16(data) => data.0.len(),
            ChannelData::ArrayDInt32(data) => data.0.len(),
            ChannelData::ArrayDUInt32(data) => data.0.len(),
            ChannelData::ArrayDFloat32(data) => data.0.len(),
            ChannelData::ArrayDInt64(data) => data.0.len(),
            ChannelData::ArrayDUInt64(data) => data.0.len(),
            ChannelData::ArrayDFloat64(data) => data.0.len(),
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
            ChannelData::FixedSizeByteArray(data) => (data.finish().value_length() * 8) as u32,
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
            ChannelData::FixedSizeByteArray(data) => data.finish().value_length() as u32,
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
                DataType::FixedSizeBinary(a.finish().value_length())
            }
            ChannelData::ArrayDInt8(a) => DataType::Int8,
            ChannelData::ArrayDUInt8(a) => DataType::UInt8,
            ChannelData::ArrayDInt16(a) => DataType::Int16,
            ChannelData::ArrayDUInt16(a) => DataType::UInt16,
            ChannelData::ArrayDInt32(a) => DataType::Int32,
            ChannelData::ArrayDUInt32(a) => DataType::UInt32,
            ChannelData::ArrayDFloat32(a) => DataType::Float32,
            ChannelData::ArrayDInt64(a) => DataType::Int64,
            ChannelData::ArrayDUInt64(a) => DataType::UInt64,
            ChannelData::ArrayDFloat64(a) => DataType::Float64,
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
                Ok(a.values_slice()
                    .iter()
                    .flat_map(|x| {
                        let str_bytes = x.to_string().into_bytes();
                        let n_str_bytes = str_bytes.len();
                        if nbytes > n_str_bytes {
                            [str_bytes, vec![0u8; nbytes - n_str_bytes]].concat()
                        } else {
                            str_bytes
                        }
                    })
                    .collect())
            }
            ChannelData::VariableSizeByteArray(a) => Ok(a.values_slice().to_vec()),
            ChannelData::FixedSizeByteArray(a) => Ok(a.finish().value_data().to_vec()),
            ChannelData::ArrayDInt8(a) => Ok(a
                .0
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
            ChannelData::ArrayDUInt8(a) => Ok(a
                .0
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
            ChannelData::ArrayDInt16(a) => Ok(a
                .0
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
            ChannelData::ArrayDUInt16(a) => Ok(a
                .0
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
            ChannelData::ArrayDInt32(a) => Ok(a
                .0
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
            ChannelData::ArrayDUInt32(a) => Ok(a
                .0
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
            ChannelData::ArrayDFloat32(a) => Ok(a
                .0
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
            ChannelData::ArrayDInt64(a) => Ok(a
                .0
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
            ChannelData::ArrayDUInt64(a) => Ok(a
                .0
                .values_slice()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect()),
            ChannelData::ArrayDFloat64(a) => Ok(a
                .0
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
            ChannelData::ArrayDInt8(a) => a.1 .0.len(),
            ChannelData::ArrayDUInt8(a) => a.1 .0.len(),
            ChannelData::ArrayDInt16(a) => a.1 .0.len(),
            ChannelData::ArrayDUInt16(a) => a.1 .0.len(),
            ChannelData::ArrayDInt32(a) => a.1 .0.len(),
            ChannelData::ArrayDUInt32(a) => a.1 .0.len(),
            ChannelData::ArrayDFloat32(a) => a.1 .0.len(),
            ChannelData::ArrayDInt64(a) => a.1 .0.len(),
            ChannelData::ArrayDUInt64(a) => a.1 .0.len(),
            ChannelData::ArrayDFloat64(a) => a.1 .0.len(),
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
            ChannelData::ArrayDInt8(a) => a.1.clone(),
            ChannelData::ArrayDUInt8(a) => a.1.clone(),
            ChannelData::ArrayDInt16(a) => a.1.clone(),
            ChannelData::ArrayDUInt16(a) => a.1.clone(),
            ChannelData::ArrayDInt32(a) => a.1.clone(),
            ChannelData::ArrayDUInt32(a) => a.1.clone(),
            ChannelData::ArrayDFloat32(a) => a.1.clone(),
            ChannelData::ArrayDInt64(a) => a.1.clone(),
            ChannelData::ArrayDUInt64(a) => a.1.clone(),
            ChannelData::ArrayDFloat64(a) => a.1.clone(),
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
                let min = a.0.values_slice().iter().min().map(|v| *v as f64);
                let max = a.0.values_slice().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDUInt8(a) => {
                let min = a.0.values_slice().iter().min().map(|v| *v as f64);
                let max = a.0.values_slice().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDInt16(a) => {
                let min = a.0.values_slice().iter().min().map(|v| *v as f64);
                let max = a.0.values_slice().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDUInt16(a) => {
                let min = a.0.values_slice().iter().min().map(|v| *v as f64);
                let max = a.0.values_slice().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDInt32(a) => {
                let min = a.0.values_slice().iter().min().map(|v| *v as f64);
                let max = a.0.values_slice().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDUInt32(a) => {
                let min = a.0.values_slice().iter().min().map(|v| *v as f64);
                let max = a.0.values_slice().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDFloat32(a) => {
                let max =
                    a.0.values_slice()
                        .iter()
                        .reduce(|accum, item| if accum >= item { accum } else { item })
                        .map(|v| *v as f64);
                let min =
                    a.0.values_slice()
                        .iter()
                        .reduce(|accum, item| if accum <= item { accum } else { item })
                        .map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDInt64(a) => {
                let min = a.0.values_slice().iter().min().map(|v| *v as f64);
                let max = a.0.values_slice().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDUInt64(a) => {
                let min = a.0.values_slice().iter().min().map(|v| *v as f64);
                let max = a.0.values_slice().iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDFloat64(a) => {
                let max =
                    a.0.values_slice()
                        .iter()
                        .reduce(|accum, item| if accum >= item { accum } else { item })
                        .cloned();
                let min =
                    a.0.values_slice()
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
            ChannelData::ArrayDInt8(a) => Arc::new(a.0.finish_cloned()) as ArrayRef,
            ChannelData::ArrayDUInt8(a) => Arc::new(a.0.finish_cloned()) as ArrayRef,
            ChannelData::ArrayDInt16(a) => Arc::new(a.0.finish_cloned()) as ArrayRef,
            ChannelData::ArrayDUInt16(a) => Arc::new(a.0.finish_cloned()) as ArrayRef,
            ChannelData::ArrayDInt32(a) => Arc::new(a.0.finish_cloned()) as ArrayRef,
            ChannelData::ArrayDUInt32(a) => Arc::new(a.0.finish_cloned()) as ArrayRef,
            ChannelData::ArrayDFloat32(a) => Arc::new(a.0.finish_cloned()) as ArrayRef,
            ChannelData::ArrayDInt64(a) => Arc::new(a.0.finish_cloned()) as ArrayRef,
            ChannelData::ArrayDUInt64(a) => Arc::new(a.0.finish_cloned()) as ArrayRef,
            ChannelData::ArrayDFloat64(a) => Arc::new(a.0.finish_cloned()) as ArrayRef,
        }
    }
    /// convert channel arrow data into dyn Array
    pub fn finish(&mut self) -> Arc<dyn Array> {
        match &mut self {
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
            ChannelData::ArrayDInt8(a) => Arc::new(a.0.finish()) as ArrayRef,
            ChannelData::ArrayDUInt8(a) => Arc::new(a.0.finish()) as ArrayRef,
            ChannelData::ArrayDInt16(a) => Arc::new(a.0.finish()) as ArrayRef,
            ChannelData::ArrayDUInt16(a) => Arc::new(a.0.finish()) as ArrayRef,
            ChannelData::ArrayDInt32(a) => Arc::new(a.0.finish()) as ArrayRef,
            ChannelData::ArrayDUInt32(a) => Arc::new(a.0.finish()) as ArrayRef,
            ChannelData::ArrayDFloat32(a) => Arc::new(a.0.finish()) as ArrayRef,
            ChannelData::ArrayDInt64(a) => Arc::new(a.0.finish()) as ArrayRef,
            ChannelData::ArrayDUInt64(a) => Arc::new(a.0.finish()) as ArrayRef,
            ChannelData::ArrayDFloat64(a) => Arc::new(a.0.finish()) as ArrayRef,
        }
    }
    pub fn set_validity(&mut self, mask: &mut BooleanBufferBuilder) -> Result<(), Error> {
        match self {
            ChannelData::Int8(a) => {
                a.validity_slice_mut().insert(mask.as_slice_mut());
            }
            ChannelData::UInt8(a) => {
                a.validity_slice_mut().insert(mask.as_slice_mut());
            }
            ChannelData::Int16(a) => {
                a.validity_slice_mut().insert(mask.as_slice_mut());
            }
            ChannelData::UInt16(a) => {
                a.validity_slice_mut().insert(mask.as_slice_mut());
            }
            ChannelData::Int32(a) => {
                a.validity_slice_mut().insert(mask.as_slice_mut());
            }
            ChannelData::UInt32(a) => {
                a.validity_slice_mut().insert(mask.as_slice_mut());
            }
            ChannelData::Float32(a) => {
                a.validity_slice_mut().insert(mask.as_slice_mut());
            }
            ChannelData::Int64(a) => {
                a.validity_slice_mut().insert(mask.as_slice_mut());
            }
            ChannelData::UInt64(a) => {
                a.validity_slice_mut().insert(mask.as_slice_mut());
            }
            ChannelData::Float64(a) => {
                a.validity_slice_mut().insert(mask.as_slice_mut());
            }
            ChannelData::Complex32(a) => {
                a.validity_slice_mut().insert(mask.as_slice_mut());
            }
            ChannelData::Complex64(a) => {
                a.validity_slice_mut().insert(mask.as_slice_mut());
            }
            ChannelData::Utf8(a) => {
                a.validity_slice_mut().insert(mask.as_slice_mut());
            }
            ChannelData::VariableSizeByteArray(a) => {
                a.validity_slice_mut().insert(mask.as_slice_mut());
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
    pub fn validity(&self) -> Option<NullBuffer> {
        match self {
            ChannelData::Int8(a) => a.finish().logical_nulls(),
            ChannelData::UInt8(a) => a.finish().logical_nulls(),
            ChannelData::Int16(a) => a.finish().logical_nulls(),
            ChannelData::UInt16(a) => a.finish().logical_nulls(),
            ChannelData::Int32(a) => a.finish().logical_nulls(),
            ChannelData::UInt32(a) => a.finish().logical_nulls(),
            ChannelData::Float32(a) => a.finish().logical_nulls(),
            ChannelData::Int64(a) => a.finish().logical_nulls(),
            ChannelData::UInt64(a) => a.finish().logical_nulls(),
            ChannelData::Float64(a) => a.finish().logical_nulls(),
            ChannelData::Complex32(a) => a.finish().logical_nulls(),
            ChannelData::Complex64(a) => a.finish().logical_nulls(),
            ChannelData::Utf8(a) => a.finish().logical_nulls(),
            ChannelData::VariableSizeByteArray(a) => a.finish().logical_nulls(),
            ChannelData::FixedSizeByteArray(a) => a.finish().logical_nulls(),
            ChannelData::ArrayDInt8((a, _)) => a.finish().logical_nulls(),
            ChannelData::ArrayDUInt8((a, _)) => a.finish().logical_nulls(),
            ChannelData::ArrayDInt16((a, _)) => a.finish().logical_nulls(),
            ChannelData::ArrayDUInt16((a, _)) => a.finish().logical_nulls(),
            ChannelData::ArrayDInt32((a, _)) => a.finish().logical_nulls(),
            ChannelData::ArrayDUInt32((a, _)) => a.finish().logical_nulls(),
            ChannelData::ArrayDFloat32((a, _)) => a.finish().logical_nulls(),
            ChannelData::ArrayDInt64((a, _)) => a.finish().logical_nulls(),
            ChannelData::ArrayDUInt64((a, _)) => a.finish().logical_nulls(),
            ChannelData::ArrayDFloat64((a, _)) => a.finish().logical_nulls(),
        }
    }
    pub fn as_ref(&self) -> &dyn Array {
        match self {
            ChannelData::Int8(a) => &a.finish(),
            ChannelData::UInt8(a) => &a.finish(),
            ChannelData::Int16(a) => &a.finish(),
            ChannelData::UInt16(a) => &a.finish(),
            ChannelData::Int32(a) => &a.finish(),
            ChannelData::UInt32(a) => &a.finish(),
            ChannelData::Float32(a) => &a.finish(),
            ChannelData::Int64(a) => &a.finish(),
            ChannelData::UInt64(a) => &a.finish(),
            ChannelData::Float64(a) => &a.finish(),
            ChannelData::Complex32(a) => &a.finish(),
            ChannelData::Complex64(a) => &a.finish(),
            ChannelData::Utf8(a) => &a.finish(),
            ChannelData::VariableSizeByteArray(a) => &a.finish(),
            ChannelData::FixedSizeByteArray(a) => &a.finish(),
            ChannelData::ArrayDInt8((a, _)) => &a.finish(),
            ChannelData::ArrayDUInt8((a, _)) => &a.finish(),
            ChannelData::ArrayDInt16((a, _)) => &a.finish(),
            ChannelData::ArrayDUInt16((a, _)) => &a.finish(),
            ChannelData::ArrayDInt32((a, _)) => &a.finish(),
            ChannelData::ArrayDUInt32((a, _)) => &a.finish(),
            ChannelData::ArrayDFloat32((a, _)) => &a.finish(),
            ChannelData::ArrayDInt64((a, _)) => &a.finish(),
            ChannelData::ArrayDUInt64((a, _)) => &a.finish(),
            ChannelData::ArrayDFloat64((a, _)) => &a.finish(),
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
                    // complex
                    if n_bytes <= 4 {
                        Ok(ChannelData::Float32(PrimitiveBuilder::new()))
                    } else {
                        Ok(ChannelData::Float64(PrimitiveBuilder::new()))
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
                    Ok(ChannelData::ArrayDUInt8((
                        PrimitiveBuilder::new(),
                        (Vec::new(), Order::RowMajor),
                    )))
                } else if n_bytes == 2 {
                    Ok(ChannelData::ArrayDUInt16((
                        PrimitiveBuilder::new(),
                        (Vec::new(), Order::RowMajor),
                    )))
                } else if n_bytes <= 4 {
                    Ok(ChannelData::ArrayDUInt32((
                        PrimitiveBuilder::new(),
                        (Vec::new(), Order::RowMajor),
                    )))
                } else {
                    Ok(ChannelData::ArrayDUInt64((
                        PrimitiveBuilder::new(),
                        (Vec::new(), Order::RowMajor),
                    )))
                }
            }
            2 | 3 => {
                // signed int
                if n_bytes <= 1 {
                    Ok(ChannelData::ArrayDInt8((
                        PrimitiveBuilder::new(),
                        (Vec::new(), Order::RowMajor),
                    )))
                } else if n_bytes == 2 {
                    Ok(ChannelData::ArrayDInt16((
                        PrimitiveBuilder::new(),
                        (Vec::new(), Order::RowMajor),
                    )))
                } else if n_bytes == 4 {
                    Ok(ChannelData::ArrayDInt32((
                        PrimitiveBuilder::new(),
                        (Vec::new(), Order::RowMajor),
                    )))
                } else {
                    Ok(ChannelData::ArrayDInt64((
                        PrimitiveBuilder::new(),
                        (Vec::new(), Order::RowMajor),
                    )))
                }
            }
            4 | 5 => {
                // float
                if n_bytes <= 4 {
                    Ok(ChannelData::ArrayDFloat32((
                        PrimitiveBuilder::new(),
                        (Vec::new(), Order::RowMajor),
                    )))
                } else {
                    Ok(ChannelData::ArrayDFloat64((
                        PrimitiveBuilder::new(),
                        (Vec::new(), Order::RowMajor),
                    )))
                }
            }
            15 | 16 => {
                // complex
                if n_bytes <= 4 {
                    bail!("f32 complex tensors not implemented")
                } else {
                    bail!("f64 complex tensors not implemented")
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

pub fn try_from(value: &dyn Array) -> Result<ChannelData, Error> {
    match value.data_type() {
        DataType::Null => Ok(ChannelData::UInt8(
            as_primitive_array::<UInt8Type>(value)
                .into_builder()
                .expect("could not downcast Null type to primitive array u8"),
        )),
        DataType::Boolean => Ok(ChannelData::UInt8(
            as_primitive_array::<UInt8Type>(value)
                .into_builder()
                .expect("could not downcast Boolean type to primitive array u8"),
        )),
        DataType::Int8 => Ok(ChannelData::Int8(
            as_primitive_array::<Int8Type>(value)
                .into_builder()
                .expect("could not downcast to primitive array i8"),
        )),
        DataType::Int16 => Ok(ChannelData::Int16(
            as_primitive_array::<Int16Type>(value)
                .into_builder()
                .expect("could not downcast to primitive array i16"),
        )),
        DataType::Int32 => Ok(ChannelData::Int32(
            as_primitive_array::<Int32Type>(value)
                .into_builder()
                .expect("could not downcast to primitive array i32"),
        )),
        DataType::Int64 => Ok(ChannelData::Int64(
            as_primitive_array::<Int64Type>(value)
                .into_builder()
                .expect("could not downcast to primitive array i64"),
        )),
        DataType::UInt8 => Ok(ChannelData::UInt8(
            as_primitive_array::<UInt8Type>(value)
                .into_builder()
                .expect("could not downcast to primitive array u8"),
        )),
        DataType::UInt16 => Ok(ChannelData::UInt16(
            as_primitive_array::<UInt16Type>(value)
                .into_builder()
                .expect("could not downcast to primitive array u16"),
        )),
        DataType::UInt32 => Ok(ChannelData::UInt32(
            as_primitive_array::<UInt32Type>(value)
                .into_builder()
                .expect("could not downcast to primitive array u32"),
        )),
        DataType::UInt64 => Ok(ChannelData::UInt64(
            as_primitive_array::<UInt64Type>(value)
                .into_builder()
                .expect("could not downcast to primitive array u32"),
        )),
        DataType::Float16 => todo!(),
        DataType::Float32 => Ok(ChannelData::Float32(
            as_primitive_array::<Float32Type>(value)
                .into_builder()
                .expect("could not downcast to primitive array f32"),
        )),
        DataType::Float64 => Ok(ChannelData::Float64(
            as_primitive_array::<Float64Type>(value)
                .into_builder()
                .expect("could not downcast to primitive array f64"),
        )),
        DataType::Binary => {
            let array = value
                .as_any()
                .downcast_ref::<BinaryArray>()
                .context("could not downcast to Binary array")?
                .clone();
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
                .context("could not downcast to fixed size binary array")?
                .clone();
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
                .context("could not downcast to Large Binary array")?
                .clone();
            Ok(ChannelData::VariableSizeByteArray(
                array
                    .into_builder()
                    .expect("could not convert binary i64 into mutable array"),
            ))
        }
        DataType::Utf8 => {
            let array = value
                .as_any()
                .downcast_ref::<StringArray>()
                .context("could not downcast to Utf8 array")?
                .clone();
            let array_i64 = LargeStringArray::from(array.iter().collect::<Vec<_>>());
            Ok(ChannelData::Utf8(
                array_i64
                    .into_builder()
                    .expect("could not convert utf8 into mutable array"),
            ))
        }
        DataType::LargeUtf8 => Ok(ChannelData::Utf8(
            value
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .context("could not downcast to Large Utf8 array")?
                .clone()
                .into_builder()
                .expect("could not convert large utf8 into mutable array"),
        )),
        DataType::FixedSizeList(field, size) => {
            // used for complex number, size of 2
            let array = value
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .context("could not downcast to fixed size list array, used for complex")?
                .clone();
            if *size == 2 {
                match array.value_type() {
                    DataType::Float32 => Ok(ChannelData::Complex32(
                        array
                            .values()
                            .as_any()
                            .downcast_ref::<Float32Array>()
                            .context("could not downcast to primitive array f32")?
                            .into_builder()
                            .expect("could not downcast to primitive builder f32"),
                    )),
                    DataType::Float64 => Ok(ChannelData::Complex64(
                        array
                            .values()
                            .as_any()
                            .downcast_ref::<Float64Array>()
                            .context("could not downcast to primitive array f64")?
                            .into_builder()
                            .expect("could not downcast to primitive builder f64"),
                    )),
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
                for text in array.values_slice().iter() {
                    writeln!(f, " {text:?} ")?;
                }
                writeln!(f, " ")
            }
            ChannelData::VariableSizeByteArray(array) => {
                for text in array.values_slice().iter() {
                    writeln!(f, " {text:?} ")?;
                }
                writeln!(f, " ")
            }
            ChannelData::FixedSizeByteArray(array) => {
                for text in array.finish().iter() {
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
