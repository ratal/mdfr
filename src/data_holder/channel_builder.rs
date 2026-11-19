//! this module holds the channel builder enum for mutable data accumulation during reading

use anyhow::{Error, Result};
use arrow::array::{
    FixedSizeBinaryBuilder, LargeBinaryBuilder, LargeStringBuilder, PrimitiveArray, UnionArray,
};
use arrow::buffer::ScalarBuffer;
use arrow::datatypes::{
    ArrowPrimitiveType, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
    UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use std::marker::PhantomData;

use crate::data_holder::channel_data::ChannelData;
use crate::data_holder::complex_arrow::ComplexBuilder;
use crate::data_holder::tensor_arrow::{Order, TensorBuilder};

/// Zero-copy Vec-backed staging buffer for primitive Arrow arrays.
/// Avoids `PrimitiveBuilder::new_from_buffer`'s 64-byte alignment `to_vec()` copy.
#[derive(Debug)]
pub struct VecBuffer<T: ArrowPrimitiveType> {
    values: Vec<T::Native>,
    _marker: PhantomData<T>,
}

impl<T: ArrowPrimitiveType> VecBuffer<T> {
    #[inline]
    pub fn from_zeroed(len: usize) -> Self
    where
        T::Native: Default + Clone,
    {
        Self {
            values: vec![T::Native::default(); len],
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn from_vec(values: Vec<T::Native>) -> Self {
        Self {
            values,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn values_slice(&self) -> &[T::Native] {
        &self.values
    }

    #[inline]
    pub fn values_slice_mut(&mut self) -> &mut [T::Native] {
        &mut self.values
    }

    #[inline]
    pub fn finish(self) -> PrimitiveArray<T> {
        PrimitiveArray::new(ScalarBuffer::from(self.values), None)
    }
}

/// Mutable staging enum used while decoding MDF blocks
#[derive(Debug)]
pub enum ChannelBuilder {
    Int8(VecBuffer<Int8Type>),
    UInt8(VecBuffer<UInt8Type>),
    Int16(VecBuffer<Int16Type>),
    UInt16(VecBuffer<UInt16Type>),
    Int32(VecBuffer<Int32Type>),
    UInt32(VecBuffer<UInt32Type>),
    Float32(VecBuffer<Float32Type>),
    Int64(VecBuffer<Int64Type>),
    UInt64(VecBuffer<UInt64Type>),
    Float64(VecBuffer<Float64Type>),
    Complex32(ComplexBuilder<Float32Type>),
    Complex64(ComplexBuilder<Float64Type>),
    Utf8(LargeStringBuilder),
    VariableSizeByteArray(LargeBinaryBuilder),
    FixedSizeByteArray(FixedSizeBinaryBuilder),
    ArrayDInt8(TensorBuilder<Int8Type>),
    ArrayDUInt8(TensorBuilder<UInt8Type>),
    ArrayDInt16(TensorBuilder<Int16Type>),
    ArrayDUInt16(TensorBuilder<UInt16Type>),
    ArrayDInt32(TensorBuilder<Int32Type>),
    ArrayDUInt32(TensorBuilder<UInt32Type>),
    ArrayDFloat32(TensorBuilder<Float32Type>),
    ArrayDInt64(TensorBuilder<Int64Type>),
    ArrayDUInt64(TensorBuilder<UInt64Type>),
    ArrayDFloat64(TensorBuilder<Float64Type>),
    #[allow(dead_code)]
    Union(UnionArray),
}

impl ChannelBuilder {
    /// Finishes building and freezes into an immutable ChannelData in O(1) time
    pub fn finish(self) -> ChannelData {
        match self {
            Self::Int8(b) => ChannelData::Int8(b.finish()),
            Self::UInt8(b) => ChannelData::UInt8(b.finish()),
            Self::Int16(b) => ChannelData::Int16(b.finish()),
            Self::UInt16(b) => ChannelData::UInt16(b.finish()),
            Self::Int32(b) => ChannelData::Int32(b.finish()),
            Self::UInt32(b) => ChannelData::UInt32(b.finish()),
            Self::Float32(b) => ChannelData::Float32(b.finish()),
            Self::Int64(b) => ChannelData::Int64(b.finish()),
            Self::UInt64(b) => ChannelData::UInt64(b.finish()),
            Self::Float64(b) => ChannelData::Float64(b.finish()),
            Self::Complex32(b) => ChannelData::Complex32(b.finish()),
            Self::Complex64(b) => ChannelData::Complex64(b.finish()),
            Self::Utf8(mut b) => ChannelData::Utf8(b.finish()),
            Self::VariableSizeByteArray(mut b) => ChannelData::VariableSizeByteArray(b.finish()),
            Self::FixedSizeByteArray(mut b) => ChannelData::FixedSizeByteArray(b.finish()),
            Self::ArrayDInt8(b) => ChannelData::ArrayDInt8(b.finish()),
            Self::ArrayDUInt8(b) => ChannelData::ArrayDUInt8(b.finish()),
            Self::ArrayDInt16(b) => ChannelData::ArrayDInt16(b.finish()),
            Self::ArrayDUInt16(b) => ChannelData::ArrayDUInt16(b.finish()),
            Self::ArrayDInt32(b) => ChannelData::ArrayDInt32(b.finish()),
            Self::ArrayDUInt32(b) => ChannelData::ArrayDUInt32(b.finish()),
            Self::ArrayDFloat32(b) => ChannelData::ArrayDFloat32(b.finish()),
            Self::ArrayDInt64(b) => ChannelData::ArrayDInt64(b.finish()),
            Self::ArrayDUInt64(b) => ChannelData::ArrayDUInt64(b.finish()),
            Self::ArrayDFloat64(b) => ChannelData::ArrayDFloat64(b.finish()),
            Self::Union(u) => ChannelData::Union(u),
        }
    }

    /// Creates a zeroed ChannelBuilder matching the requested channel data type and dimensions
    pub fn new_zeroed(
        cn_type: u8,
        cn_data_type: u8,
        cycle_count: usize,
        n_bytes: usize,
        shape: (Vec<usize>, Order),
        list_size: usize,
    ) -> Result<Self, Error> {
        if cn_type == 3 || cn_type == 6 {
            // Virtual ramp channel: 0..cycle_count
            let mut vect = Vec::with_capacity(cycle_count);
            for i in 0..cycle_count {
                vect.push(i as u64);
            }
            return Ok(ChannelBuilder::UInt64(VecBuffer::from_vec(vect)));
        }

        if cn_type == 7 {
            // VLSC channel: raw record contains offset as unsigned integer
            return match n_bytes {
                1 => Ok(ChannelBuilder::UInt8(VecBuffer::from_zeroed(cycle_count))),
                2 => Ok(ChannelBuilder::UInt16(VecBuffer::from_zeroed(cycle_count))),
                3 | 4 => Ok(ChannelBuilder::UInt32(VecBuffer::from_zeroed(cycle_count))),
                _ => Ok(ChannelBuilder::UInt64(VecBuffer::from_zeroed(cycle_count))),
            };
        }

        if cn_data_type == 15 || cn_data_type == 16 {
            let elem_count = shape.0.iter().product::<usize>().max(1);
            if n_bytes <= 8 {
                return Ok(ChannelBuilder::Complex32(ComplexBuilder::new_zeroed(
                    cycle_count * elem_count * 2,
                    shape.0,
                    shape.1,
                )));
            } else {
                return Ok(ChannelBuilder::Complex64(ComplexBuilder::new_zeroed(
                    cycle_count * elem_count * 2,
                    shape.0,
                    shape.1,
                )));
            }
        }

        let is_tensor = list_size > 1 && !shape.0.is_empty() && shape.0 != [1];

        if is_tensor {
            let total_elems = cycle_count * list_size;
            return match cn_data_type {
                0 | 1 => match n_bytes {
                    1 => Ok(ChannelBuilder::ArrayDUInt8(TensorBuilder::new_zeroed(
                        total_elems,
                        shape.0,
                        shape.1,
                    ))),
                    2 => Ok(ChannelBuilder::ArrayDUInt16(TensorBuilder::new_zeroed(
                        total_elems,
                        shape.0,
                        shape.1,
                    ))),
                    3 | 4 => Ok(ChannelBuilder::ArrayDUInt32(TensorBuilder::new_zeroed(
                        total_elems,
                        shape.0,
                        shape.1,
                    ))),
                    _ => Ok(ChannelBuilder::ArrayDUInt64(TensorBuilder::new_zeroed(
                        total_elems,
                        shape.0,
                        shape.1,
                    ))),
                },
                2 | 3 => match n_bytes {
                    1 => Ok(ChannelBuilder::ArrayDInt8(TensorBuilder::new_zeroed(
                        total_elems,
                        shape.0,
                        shape.1,
                    ))),
                    2 => Ok(ChannelBuilder::ArrayDInt16(TensorBuilder::new_zeroed(
                        total_elems,
                        shape.0,
                        shape.1,
                    ))),
                    3 | 4 => Ok(ChannelBuilder::ArrayDInt32(TensorBuilder::new_zeroed(
                        total_elems,
                        shape.0,
                        shape.1,
                    ))),
                    _ => Ok(ChannelBuilder::ArrayDInt64(TensorBuilder::new_zeroed(
                        total_elems,
                        shape.0,
                        shape.1,
                    ))),
                },
                4 | 5 => match n_bytes {
                    1..=4 => Ok(ChannelBuilder::ArrayDFloat32(TensorBuilder::new_zeroed(
                        total_elems,
                        shape.0,
                        shape.1,
                    ))),
                    _ => Ok(ChannelBuilder::ArrayDFloat64(TensorBuilder::new_zeroed(
                        total_elems,
                        shape.0,
                        shape.1,
                    ))),
                },
                _ => Ok(ChannelBuilder::ArrayDFloat64(TensorBuilder::new_zeroed(
                    total_elems,
                    shape.0,
                    shape.1,
                ))),
            };
        }

        match cn_data_type {
            0 | 1 => match n_bytes {
                1 => Ok(ChannelBuilder::UInt8(VecBuffer::from_zeroed(cycle_count))),
                2 => Ok(ChannelBuilder::UInt16(VecBuffer::from_zeroed(cycle_count))),
                3 | 4 => Ok(ChannelBuilder::UInt32(VecBuffer::from_zeroed(cycle_count))),
                _ => Ok(ChannelBuilder::UInt64(VecBuffer::from_zeroed(cycle_count))),
            },
            2 | 3 => match n_bytes {
                1 => Ok(ChannelBuilder::Int8(VecBuffer::from_zeroed(cycle_count))),
                2 => Ok(ChannelBuilder::Int16(VecBuffer::from_zeroed(cycle_count))),
                3 | 4 => Ok(ChannelBuilder::Int32(VecBuffer::from_zeroed(cycle_count))),
                _ => Ok(ChannelBuilder::Int64(VecBuffer::from_zeroed(cycle_count))),
            },
            4 | 5 => match n_bytes {
                1..=4 => Ok(ChannelBuilder::Float32(VecBuffer::from_zeroed(cycle_count))),
                _ => Ok(ChannelBuilder::Float64(VecBuffer::from_zeroed(cycle_count))),
            },
            6..=9 | 17 => Ok(ChannelBuilder::Utf8(LargeStringBuilder::with_capacity(
                cycle_count,
                cycle_count * n_bytes.max(1),
            ))),
            10..=12 => {
                if cn_type == 1 {
                    Ok(ChannelBuilder::VariableSizeByteArray(
                        LargeBinaryBuilder::with_capacity(
                            cycle_count,
                            cycle_count * n_bytes.max(1),
                        ),
                    ))
                } else {
                    Ok(ChannelBuilder::FixedSizeByteArray(
                        FixedSizeBinaryBuilder::with_capacity(cycle_count, n_bytes as i32),
                    ))
                }
            }
            13 | 14 => Ok(ChannelBuilder::Int64(VecBuffer::from_zeroed(cycle_count))),
            _ => Ok(ChannelBuilder::Float64(VecBuffer::from_zeroed(cycle_count))),
        }
    }

    /// Extract unsigned integer values as Vec<u64>, if this builder holds an integer type.
    pub fn to_u64_vec(&self) -> Option<Vec<u64>> {
        match self {
            ChannelBuilder::UInt8(b) => Some(b.values_slice().iter().map(|&x| x as u64).collect()),
            ChannelBuilder::UInt16(b) => Some(b.values_slice().iter().map(|&x| x as u64).collect()),
            ChannelBuilder::UInt32(b) => Some(b.values_slice().iter().map(|&x| x as u64).collect()),
            ChannelBuilder::UInt64(b) => Some(b.values_slice().to_vec()),
            ChannelBuilder::Int8(b) => Some(b.values_slice().iter().map(|&x| x as u64).collect()),
            ChannelBuilder::Int16(b) => Some(b.values_slice().iter().map(|&x| x as u64).collect()),
            ChannelBuilder::Int32(b) => Some(b.values_slice().iter().map(|&x| x as u64).collect()),
            ChannelBuilder::Int64(b) => Some(b.values_slice().iter().map(|&x| x as u64).collect()),
            _ => None,
        }
    }
}
