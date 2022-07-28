//! this module holds the channel data enum and related implementations

use num::Complex;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::fmt;
use std::iter::IntoIterator;

use crate::export::tensor::Order as TensorOrder;

/// channel data type enum.
/// most common data type is 1D ndarray for timeseries with element types numeric.
/// vector of string or bytes also exists.
/// Dynamic dimension arrays ArrayD are also existing to cover CABlock arrays data.
#[derive(Debug, Clone, PartialEq)]
pub enum ChannelData {
    Int8(Vec<i8>),
    UInt8(Vec<u8>),
    Int16(Vec<i16>),
    UInt16(Vec<u16>),
    Float16(Vec<f32>),
    Int24(Vec<i32>),
    UInt24(Vec<u32>),
    Int32(Vec<i32>),
    UInt32(Vec<u32>),
    Float32(Vec<f32>),
    Int48(Vec<i64>),
    UInt48(Vec<u64>),
    Int64(Vec<i64>),
    UInt64(Vec<u64>),
    Float64(Vec<f64>),
    Complex16(ArrowComplex<f32>),
    Complex32(ArrowComplex<f32>),
    Complex64(ArrowComplex<f64>),
    StringSBC(Vec<String>),
    StringUTF8(Vec<String>),
    StringUTF16(Vec<String>),
    VariableSizeByteArray(Vec<Vec<u8>>),
    FixedSizeByteArray((Vec<u8>, usize)),
    ArrayDInt8((Vec<i8>, (Vec<usize>, Order))),
    ArrayDUInt8((Vec<u8>, (Vec<usize>, Order))),
    ArrayDInt16((Vec<i16>, (Vec<usize>, Order))),
    ArrayDUInt16((Vec<u16>, (Vec<usize>, Order))),
    ArrayDFloat16((Vec<f32>, (Vec<usize>, Order))),
    ArrayDInt24((Vec<i32>, (Vec<usize>, Order))),
    ArrayDUInt24((Vec<u32>, (Vec<usize>, Order))),
    ArrayDInt32((Vec<i32>, (Vec<usize>, Order))),
    ArrayDUInt32((Vec<u32>, (Vec<usize>, Order))),
    ArrayDFloat32((Vec<f32>, (Vec<usize>, Order))),
    ArrayDInt48((Vec<i64>, (Vec<usize>, Order))),
    ArrayDUInt48((Vec<u64>, (Vec<usize>, Order))),
    ArrayDInt64((Vec<i64>, (Vec<usize>, Order))),
    ArrayDUInt64((Vec<u64>, (Vec<usize>, Order))),
    ArrayDFloat64((Vec<f64>, (Vec<usize>, Order))),
    ArrayDComplex16((ArrowComplex<f32>, (Vec<usize>, Order))),
    ArrayDComplex32((ArrowComplex<f32>, (Vec<usize>, Order))),
    ArrayDComplex64((ArrowComplex<f64>, (Vec<usize>, Order))),
}

/// Order of the array, Row or Column Major (first)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Order {
    RowMajor,
    ColumnMajor,
}

impl From<Order> for TensorOrder {
    fn from(order: Order) -> Self {
        match order {
            Order::RowMajor => TensorOrder::RowMajor,
            Order::ColumnMajor => TensorOrder::ColumnMajor,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrowComplex<T>(pub Vec<T>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrowComplexIter<T> {
    values: ArrowComplex<T>,
    index: usize,
}

impl<T> ArrowComplex<T> {
    pub fn len(&self) -> usize {
        self.0.len() / 2
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Iterator for ArrowComplexIter<f64> {
    type Item = Complex<f64>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.values.len() {
            return None;
        }

        self.index += 1;
        Some(Complex::<f64>::new(
            self.values.0[2 * (self.index - 1)],
            self.values.0[2 * (self.index - 1) + 1],
        ))
    }
}

impl Iterator for ArrowComplexIter<f32> {
    type Item = Complex<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.values.len() {
            return None;
        }

        self.index += 1;
        Some(Complex::<f32>::new(
            self.values.0[2 * (self.index - 1)],
            self.values.0[2 * (self.index - 1) + 1],
        ))
    }
}

impl IntoIterator for ArrowComplex<f64> {
    type Item = Complex<f64>;
    type IntoIter = ArrowComplexIter<f64>;

    fn into_iter(self) -> Self::IntoIter {
        ArrowComplexIter {
            values: self,
            index: 0,
        }
    }
}

impl IntoIterator for ArrowComplex<f32> {
    type Item = Complex<f32>;
    type IntoIter = ArrowComplexIter<f32>;

    fn into_iter(self) -> Self::IntoIter {
        ArrowComplexIter {
            values: self,
            index: 0,
        }
    }
}

impl ArrowComplex<f64> {
    pub fn zeros(length: usize) -> Self {
        ArrowComplex(vec![0.0; 2 * length])
    }
}

impl ArrowComplex<f32> {
    pub fn zeros(length: usize) -> Self {
        ArrowComplex(vec![0.0; 2 * length])
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
        dim: (Vec<usize>, Order),
    ) -> ChannelData {
        if cn_type == 3 || cn_type == 6 {
            // virtual channel
            ChannelData::UInt64(Vec::<u64>::with_capacity(cycle_count as usize))
        } else {
            match self {
                ChannelData::Int8(_) => ChannelData::Int8(vec![0i8; cycle_count as usize]),
                ChannelData::UInt8(_) => ChannelData::UInt8(vec![0u8; cycle_count as usize]),
                ChannelData::Int16(_) => ChannelData::Int16(vec![0i16; cycle_count as usize]),
                ChannelData::UInt16(_) => ChannelData::UInt16(vec![0u16; cycle_count as usize]),
                ChannelData::Float16(_) => ChannelData::Float16(vec![0f32; cycle_count as usize]),
                ChannelData::Int24(_) => ChannelData::Int24(vec![0i32; cycle_count as usize]),
                ChannelData::UInt24(_) => ChannelData::UInt24(vec![0u32; cycle_count as usize]),
                ChannelData::Int32(_) => ChannelData::Int32(vec![0i32; cycle_count as usize]),
                ChannelData::UInt32(_) => ChannelData::UInt32(vec![0u32; cycle_count as usize]),
                ChannelData::Float32(_) => ChannelData::Float32(vec![0f32; cycle_count as usize]),
                ChannelData::Int48(_) => ChannelData::Int48(vec![0i64; cycle_count as usize]),
                ChannelData::UInt48(_) => ChannelData::UInt48(vec![0u64; cycle_count as usize]),
                ChannelData::Int64(_) => ChannelData::Int64(vec![0i64; cycle_count as usize]),
                ChannelData::UInt64(_) => {
                    ChannelData::UInt64(Vec::<u64>::with_capacity(cycle_count as usize))
                }
                ChannelData::Float64(_) => {
                    ChannelData::Float64(Vec::<f64>::with_capacity(cycle_count as usize))
                }
                ChannelData::Complex16(_) => {
                    ChannelData::Complex16(ArrowComplex(vec![0f32; cycle_count as usize * 2]))
                }
                ChannelData::Complex32(_) => {
                    ChannelData::Complex32(ArrowComplex(vec![0f32; cycle_count as usize * 2]))
                }
                ChannelData::Complex64(_) => {
                    ChannelData::Complex64(ArrowComplex(vec![0f64; cycle_count as usize * 2]))
                }
                ChannelData::StringSBC(_) => {
                    let mut target_vect: Vec<String> = vec![String::new(); cycle_count as usize];
                    for txt in target_vect.iter_mut() {
                        txt.reserve(n_bytes as usize);
                    }
                    ChannelData::StringSBC(target_vect)
                }
                ChannelData::StringUTF8(_) => {
                    ChannelData::StringUTF8(vec![
                        String::with_capacity(n_bytes as usize);
                        cycle_count as usize
                    ])
                }
                ChannelData::StringUTF16(_) => {
                    let mut target_vect: Vec<String> = vec![String::new(); cycle_count as usize];
                    for txt in target_vect.iter_mut() {
                        txt.reserve(n_bytes as usize);
                    }
                    ChannelData::StringUTF16(target_vect)
                }
                ChannelData::VariableSizeByteArray(_) => ChannelData::VariableSizeByteArray(vec![
                        vec![0u8; n_bytes as usize];
                        cycle_count as usize
                    ]),
                ChannelData::FixedSizeByteArray(_) => ChannelData::FixedSizeByteArray((
                    Vec::<u8>::with_capacity(n_bytes as usize * cycle_count as usize),
                    n_bytes as usize,
                )),
                ChannelData::ArrayDInt8(_) => ChannelData::ArrayDInt8((
                    Vec::<i8>::with_capacity(
                        (cycle_count as usize) * dim.0.iter().product::<usize>(),
                    ),
                    dim,
                )),
                ChannelData::ArrayDUInt8(_) => ChannelData::ArrayDUInt8((
                    Vec::<u8>::with_capacity(
                        (cycle_count as usize) * dim.0.iter().product::<usize>(),
                    ),
                    dim,
                )),
                ChannelData::ArrayDInt16(_) => ChannelData::ArrayDInt16((
                    Vec::<i16>::with_capacity(
                        (cycle_count as usize) * dim.0.iter().product::<usize>(),
                    ),
                    dim,
                )),
                ChannelData::ArrayDUInt16(_) => ChannelData::ArrayDUInt16((
                    Vec::<u16>::with_capacity(
                        (cycle_count as usize) * dim.0.iter().product::<usize>(),
                    ),
                    dim,
                )),
                ChannelData::ArrayDFloat16(_) => ChannelData::ArrayDFloat16((
                    Vec::<f32>::with_capacity(
                        (cycle_count as usize) * dim.0.iter().product::<usize>(),
                    ),
                    dim,
                )),
                ChannelData::ArrayDInt24(_) => ChannelData::ArrayDInt24((
                    Vec::<i32>::with_capacity(
                        (cycle_count as usize) * dim.0.iter().product::<usize>(),
                    ),
                    dim,
                )),
                ChannelData::ArrayDUInt24(_) => ChannelData::ArrayDUInt24((
                    Vec::<u32>::with_capacity(
                        (cycle_count as usize) * dim.0.iter().product::<usize>(),
                    ),
                    dim,
                )),
                ChannelData::ArrayDInt32(_) => ChannelData::ArrayDInt32((
                    Vec::<i32>::with_capacity(
                        (cycle_count as usize) * dim.0.iter().product::<usize>(),
                    ),
                    dim,
                )),
                ChannelData::ArrayDUInt32(_) => ChannelData::ArrayDUInt32((
                    Vec::<u32>::with_capacity(
                        (cycle_count as usize) * dim.0.iter().product::<usize>(),
                    ),
                    dim,
                )),
                ChannelData::ArrayDFloat32(_) => ChannelData::ArrayDFloat32((
                    Vec::<f32>::with_capacity(
                        (cycle_count as usize) * dim.0.iter().product::<usize>(),
                    ),
                    dim,
                )),
                ChannelData::ArrayDInt48(_) => ChannelData::ArrayDInt48((
                    Vec::<i64>::with_capacity(
                        (cycle_count as usize) * dim.0.iter().product::<usize>(),
                    ),
                    dim,
                )),
                ChannelData::ArrayDUInt48(_) => ChannelData::ArrayDUInt48((
                    Vec::<u64>::with_capacity(
                        (cycle_count as usize) * dim.0.iter().product::<usize>(),
                    ),
                    dim,
                )),
                ChannelData::ArrayDInt64(_) => ChannelData::ArrayDInt64((
                    Vec::<i64>::with_capacity(
                        (cycle_count as usize) * dim.0.iter().product::<usize>(),
                    ),
                    dim,
                )),
                ChannelData::ArrayDUInt64(_) => ChannelData::ArrayDUInt64((
                    Vec::<u64>::with_capacity(
                        (cycle_count as usize) * dim.0.iter().product::<usize>(),
                    ),
                    dim,
                )),
                ChannelData::ArrayDFloat64(_) => ChannelData::ArrayDFloat64((
                    Vec::<f64>::with_capacity(
                        (cycle_count as usize) * dim.0.iter().product::<usize>(),
                    ),
                    dim,
                )),
                ChannelData::ArrayDComplex16(_) => ChannelData::ArrayDComplex16((
                    ArrowComplex::<f32>(Vec::with_capacity(
                        (cycle_count as usize) * dim.0.iter().product::<usize>() * 2,
                    )),
                    dim,
                )),
                ChannelData::ArrayDComplex32(_) => ChannelData::ArrayDComplex32((
                    ArrowComplex::<f32>(Vec::<f32>::with_capacity(
                        (cycle_count as usize) * dim.0.iter().product::<usize>() * 2,
                    )),
                    dim,
                )),
                ChannelData::ArrayDComplex64(_) => ChannelData::ArrayDComplex64((
                    ArrowComplex::<f64>(Vec::with_capacity(
                        (cycle_count as usize) * dim.0.iter().product::<usize>() * 2,
                    )),
                    dim,
                )),
            }
        }
    }
    /// small helper to support ChannelData display, returning first and last element of ndarray
    pub fn first_last(&self) -> String {
        let output: String;
        match self {
            ChannelData::Int8(array) => {
                if array.len() > 1 {
                    output = format!("[{}, .., {}]", array[0], array[array.len() - 1]);
                } else {
                    output = String::new();
                }
            }
            ChannelData::UInt8(array) => {
                if array.len() > 1 {
                    output = format!("[{}, .., {}]", array[0], array[array.len() - 1]);
                } else {
                    output = String::new();
                }
            }
            ChannelData::Int16(array) => {
                if array.len() > 1 {
                    output = format!("[{}, .., {}]", array[0], array[array.len() - 1]);
                } else {
                    output = String::new();
                }
            }
            ChannelData::UInt16(array) => {
                if array.len() > 1 {
                    output = format!("[{}, .., {}]", array[0], array[array.len() - 1]);
                } else {
                    output = String::new();
                }
            }
            ChannelData::Float16(array) => {
                if array.len() > 1 {
                    output = format!("[{}, .., {}]", array[0], array[array.len() - 1]);
                } else {
                    output = String::new();
                }
            }
            ChannelData::Int24(array) => {
                if array.len() > 1 {
                    output = format!("[{}, .., {}]", array[0], array[array.len() - 1]);
                } else {
                    output = String::new();
                }
            }
            ChannelData::UInt24(array) => {
                if array.len() > 1 {
                    output = format!("[{}, .., {}]", array[0], array[array.len() - 1]);
                } else {
                    output = String::new();
                }
            }
            ChannelData::Int32(array) => {
                if array.len() > 1 {
                    output = format!("[{}, .., {}]", array[0], array[array.len() - 1]);
                } else {
                    output = String::new();
                }
            }
            ChannelData::UInt32(array) => {
                if array.len() > 1 {
                    output = format!("[{}, .., {}]", array[0], array[array.len() - 1]);
                } else {
                    output = String::new();
                }
            }
            ChannelData::Float32(array) => {
                if array.len() > 1 {
                    output = format!("[{}, .., {}]", array[0], array[array.len() - 1]);
                } else {
                    output = String::new();
                }
            }
            ChannelData::Int48(array) => {
                if array.len() > 1 {
                    output = format!("[{}, .., {}]", array[0], array[array.len() - 1]);
                } else {
                    output = String::new();
                }
            }
            ChannelData::UInt48(array) => {
                if array.len() > 1 {
                    output = format!("[{}, .., {}]", array[0], array[array.len() - 1]);
                } else {
                    output = String::new();
                }
            }
            ChannelData::Int64(array) => {
                if array.len() > 1 {
                    output = format!("[{}, .., {}]", array[0], array[array.len() - 1]);
                } else {
                    output = String::new();
                }
            }
            ChannelData::UInt64(array) => {
                if array.len() > 1 {
                    output = format!("[{}, .., {}]", array[0], array[array.len() - 1]);
                } else {
                    output = String::new();
                }
            }
            ChannelData::Float64(array) => {
                if array.len() > 1 {
                    output = format!("[{}, .., {}]", array[0], array[array.len() - 1]);
                } else {
                    output = String::new();
                }
            }
            ChannelData::Complex16(array) => {
                if array.len() > 1 {
                    output = format!(
                        "[{} + i{}, .., {} + i{}]",
                        array.0[0],
                        array.0[0],
                        array.0[array.len() - 2],
                        array.0[array.len() - 1]
                    );
                } else {
                    output = String::new();
                }
            }
            ChannelData::Complex32(array) => {
                if array.len() > 1 {
                    output = format!(
                        "[{} + i{}, .., {} + i{}]",
                        array.0[0],
                        array.0[0],
                        array.0[array.len() - 2],
                        array.0[array.len() - 1]
                    );
                } else {
                    output = String::new();
                }
            }
            ChannelData::Complex64(array) => {
                if array.len() > 1 {
                    output = format!(
                        "[{} + i{}, .., {} + i{}]",
                        array.0[0],
                        array.0[0],
                        array.0[array.len() - 2],
                        array.0[array.len() - 1]
                    );
                } else {
                    output = String::new();
                }
            }
            ChannelData::StringSBC(array) => {
                if array.len() > 1 {
                    output = format!("[{}, .., {}]", array[0], array[array.len() - 1]);
                } else {
                    output = String::new();
                }
            }
            ChannelData::StringUTF8(array) => {
                if array.len() > 1 {
                    output = format!("[{}, .., {}]", array[0], array[array.len() - 1]);
                } else {
                    output = String::new();
                }
            }
            ChannelData::StringUTF16(array) => {
                if array.len() > 1 {
                    output = format!("[{}, .., {}]", array[0], array[array.len() - 1]);
                } else {
                    output = String::new();
                }
            }
            ChannelData::VariableSizeByteArray(array) => {
                if array.len() > 1 {
                    output = format!("[{:?}, .., {:?}]", array[0], array[array.len() - 1]);
                } else {
                    output = String::new();
                }
            }
            ChannelData::FixedSizeByteArray(array) => {
                if array.0.len() > 1 {
                    output = format!("[{:?}, .., {:?}]", array.0[0], array.0[array.0.len() - 1]);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDInt8(array) => {
                if array.0.len() > 1 {
                    output = format!("dim {:?} {:?}", array.1, array.0);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDUInt8(array) => {
                if array.0.len() > 1 {
                    output = format!("dim {:?} {:?}", array.1, array.0);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDInt16(array) => {
                if array.0.len() > 1 {
                    output = format!("dim {:?} {:?}", array.1, array.0);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDUInt16(array) => {
                if array.0.len() > 1 {
                    output = format!("dim {:?} {:?}", array.1, array.0);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDFloat16(array) => {
                if array.0.len() > 1 {
                    output = format!("dim {:?} {:?}", array.1, array.0);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDInt24(array) => {
                if array.0.len() > 1 {
                    output = format!("dim {:?} {:?}", array.1, array.0);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDUInt24(array) => {
                if array.0.len() > 1 {
                    output = format!("dim {:?} {:?}", array.1, array.0);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDInt32(array) => {
                if array.0.len() > 1 {
                    output = format!("dim {:?} {:?}", array.1, array.0);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDUInt32(array) => {
                if array.0.len() > 1 {
                    output = format!("dim {:?} {:?}", array.1, array.0);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDFloat32(array) => {
                if array.0.len() > 1 {
                    output = format!("dim {:?} {:?}", array.1, array.0);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDInt48(array) => {
                if array.0.len() > 1 {
                    output = format!("dim {:?} {:?}", array.1, array.0);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDUInt48(array) => {
                if array.0.len() > 1 {
                    output = format!("dim {:?} {:?}", array.1, array.0);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDInt64(array) => {
                if array.0.len() > 1 {
                    output = format!("dim {:?} {:?}", array.1, array.0);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDUInt64(array) => {
                if array.0.len() > 1 {
                    output = format!("dim {:?} {:?}", array.1, array.0);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDFloat64(array) => {
                if array.0.len() > 1 {
                    output = format!("dim {:?} {:?}", array.1, array.0);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDComplex16(array) => {
                if array.0.len() > 1 {
                    output = format!("dim {:?} {:?}", array.1, array.0);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDComplex32(array) => {
                if array.0.len() > 1 {
                    output = format!("dim {:?} {:?}", array.1, array.0);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDComplex64(array) => {
                if array.0.len() > 1 {
                    output = format!("dim {:?} {:?}", array.1, array.0);
                } else {
                    output = String::new();
                }
            }
        }
        output
    }
    /// checks is if ndarray is empty
    pub fn is_empty(&self) -> bool {
        match self {
            ChannelData::Int8(data) => data.is_empty(),
            ChannelData::UInt8(data) => data.is_empty(),
            ChannelData::Int16(data) => data.is_empty(),
            ChannelData::UInt16(data) => data.is_empty(),
            ChannelData::Float16(data) => data.is_empty(),
            ChannelData::Int24(data) => data.is_empty(),
            ChannelData::UInt24(data) => data.is_empty(),
            ChannelData::Int32(data) => data.is_empty(),
            ChannelData::UInt32(data) => data.is_empty(),
            ChannelData::Float32(data) => data.is_empty(),
            ChannelData::Int48(data) => data.is_empty(),
            ChannelData::UInt48(data) => data.is_empty(),
            ChannelData::Int64(data) => data.is_empty(),
            ChannelData::UInt64(data) => data.is_empty(),
            ChannelData::Float64(data) => data.is_empty(),
            ChannelData::Complex16(data) => data.is_empty(),
            ChannelData::Complex32(data) => data.is_empty(),
            ChannelData::Complex64(data) => data.is_empty(),
            ChannelData::StringSBC(data) => data.is_empty(),
            ChannelData::StringUTF8(data) => data.is_empty(),
            ChannelData::StringUTF16(data) => data.is_empty(),
            ChannelData::VariableSizeByteArray(data) => data.is_empty(),
            ChannelData::FixedSizeByteArray(data) => data.0.is_empty(),
            ChannelData::ArrayDInt8(data) => data.0.is_empty(),
            ChannelData::ArrayDUInt8(data) => data.0.is_empty(),
            ChannelData::ArrayDInt16(data) => data.0.is_empty(),
            ChannelData::ArrayDUInt16(data) => data.0.is_empty(),
            ChannelData::ArrayDFloat16(data) => data.0.is_empty(),
            ChannelData::ArrayDInt24(data) => data.0.is_empty(),
            ChannelData::ArrayDUInt24(data) => data.0.is_empty(),
            ChannelData::ArrayDInt32(data) => data.0.is_empty(),
            ChannelData::ArrayDUInt32(data) => data.0.is_empty(),
            ChannelData::ArrayDFloat32(data) => data.0.is_empty(),
            ChannelData::ArrayDInt48(data) => data.0.is_empty(),
            ChannelData::ArrayDUInt48(data) => data.0.is_empty(),
            ChannelData::ArrayDInt64(data) => data.0.is_empty(),
            ChannelData::ArrayDUInt64(data) => data.0.is_empty(),
            ChannelData::ArrayDFloat64(data) => data.0.is_empty(),
            ChannelData::ArrayDComplex16(data) => data.0.is_empty(),
            ChannelData::ArrayDComplex32(data) => data.0.is_empty(),
            ChannelData::ArrayDComplex64(data) => data.0.is_empty(),
        }
    }
    /// checks is if ndarray is empty
    pub fn len(&self) -> usize {
        match self {
            ChannelData::Int8(data) => data.len(),
            ChannelData::UInt8(data) => data.len(),
            ChannelData::Int16(data) => data.len(),
            ChannelData::UInt16(data) => data.len(),
            ChannelData::Float16(data) => data.len(),
            ChannelData::Int24(data) => data.len(),
            ChannelData::UInt24(data) => data.len(),
            ChannelData::Int32(data) => data.len(),
            ChannelData::UInt32(data) => data.len(),
            ChannelData::Float32(data) => data.len(),
            ChannelData::Int48(data) => data.len(),
            ChannelData::UInt48(data) => data.len(),
            ChannelData::Int64(data) => data.len(),
            ChannelData::UInt64(data) => data.len(),
            ChannelData::Float64(data) => data.len(),
            ChannelData::Complex16(data) => data.len(),
            ChannelData::Complex32(data) => data.len(),
            ChannelData::Complex64(data) => data.len(),
            ChannelData::StringSBC(data) => data.len(),
            ChannelData::StringUTF8(data) => data.len(),
            ChannelData::StringUTF16(data) => data.len(),
            ChannelData::VariableSizeByteArray(data) => data.len(),
            ChannelData::FixedSizeByteArray(data) => data.0.len(),
            ChannelData::ArrayDInt8(data) => data.0.len(),
            ChannelData::ArrayDUInt8(data) => data.0.len(),
            ChannelData::ArrayDInt16(data) => data.0.len(),
            ChannelData::ArrayDUInt16(data) => data.0.len(),
            ChannelData::ArrayDFloat16(data) => data.0.len(),
            ChannelData::ArrayDInt24(data) => data.0.len(),
            ChannelData::ArrayDUInt24(data) => data.0.len(),
            ChannelData::ArrayDInt32(data) => data.0.len(),
            ChannelData::ArrayDUInt32(data) => data.0.len(),
            ChannelData::ArrayDFloat32(data) => data.0.len(),
            ChannelData::ArrayDInt48(data) => data.0.len(),
            ChannelData::ArrayDUInt48(data) => data.0.len(),
            ChannelData::ArrayDInt64(data) => data.0.len(),
            ChannelData::ArrayDUInt64(data) => data.0.len(),
            ChannelData::ArrayDFloat64(data) => data.0.len(),
            ChannelData::ArrayDComplex16(data) => data.0.len(),
            ChannelData::ArrayDComplex32(data) => data.0.len(),
            ChannelData::ArrayDComplex64(data) => data.0.len(),
        }
    }
    /// returns the max bit count of each values in array
    pub fn bit_count(&self) -> u32 {
        match self {
            ChannelData::Int8(_) => 8,
            ChannelData::UInt8(_) => 8,
            ChannelData::Int16(_) => 16,
            ChannelData::UInt16(_) => 16,
            ChannelData::Float16(_) => 16,
            ChannelData::Int24(_) => 24,
            ChannelData::UInt24(_) => 24,
            ChannelData::Int32(_) => 32,
            ChannelData::UInt32(_) => 32,
            ChannelData::Float32(_) => 32,
            ChannelData::Int48(_) => 48,
            ChannelData::UInt48(_) => 48,
            ChannelData::Int64(_) => 64,
            ChannelData::UInt64(_) => 64,
            ChannelData::Float64(_) => 64,
            ChannelData::Complex16(_) => 32,
            ChannelData::Complex32(_) => 64,
            ChannelData::Complex64(_) => 128,
            ChannelData::StringSBC(data) => {
                data.par_iter().map(|s| s.len() as u32).max().unwrap_or(0) * 8
            }
            ChannelData::StringUTF8(data) => {
                data.par_iter().map(|s| s.len() as u32).max().unwrap_or(0) * 8
            }
            ChannelData::StringUTF16(data) => {
                data.par_iter().map(|s| s.len() as u32).max().unwrap_or(0) * 8
            }
            ChannelData::VariableSizeByteArray(data) => {
                data.par_iter().map(|s| s.len() as u32).max().unwrap_or(0) * 8
            }
            ChannelData::FixedSizeByteArray(data) => data.1 as u32 * 8,
            ChannelData::ArrayDInt8(_) => 8,
            ChannelData::ArrayDUInt8(_) => 8,
            ChannelData::ArrayDInt16(_) => 16,
            ChannelData::ArrayDUInt16(_) => 16,
            ChannelData::ArrayDFloat16(_) => 16,
            ChannelData::ArrayDInt24(_) => 24,
            ChannelData::ArrayDUInt24(_) => 24,
            ChannelData::ArrayDInt32(_) => 32,
            ChannelData::ArrayDUInt32(_) => 32,
            ChannelData::ArrayDFloat32(_) => 32,
            ChannelData::ArrayDInt48(_) => 48,
            ChannelData::ArrayDUInt48(_) => 48,
            ChannelData::ArrayDInt64(_) => 64,
            ChannelData::ArrayDUInt64(_) => 64,
            ChannelData::ArrayDFloat64(_) => 64,
            ChannelData::ArrayDComplex16(_) => 32,
            ChannelData::ArrayDComplex32(_) => 64,
            ChannelData::ArrayDComplex64(_) => 128,
        }
    }
    /// returns the max byte count of each values in array
    pub fn byte_count(&self) -> u32 {
        match self {
            ChannelData::Int8(_) => 1,
            ChannelData::UInt8(_) => 1,
            ChannelData::Int16(_) => 2,
            ChannelData::UInt16(_) => 2,
            ChannelData::Float16(_) => 2,
            ChannelData::Int24(_) => 3,
            ChannelData::UInt24(_) => 3,
            ChannelData::Int32(_) => 4,
            ChannelData::UInt32(_) => 4,
            ChannelData::Float32(_) => 4,
            ChannelData::Int48(_) => 6,
            ChannelData::UInt48(_) => 6,
            ChannelData::Int64(_) => 8,
            ChannelData::UInt64(_) => 8,
            ChannelData::Float64(_) => 8,
            ChannelData::Complex16(_) => 4,
            ChannelData::Complex32(_) => 8,
            ChannelData::Complex64(_) => 16,
            ChannelData::StringSBC(data) => {
                data.par_iter().map(|s| s.len() as u32).max().unwrap_or(0)
            }
            ChannelData::StringUTF8(data) => {
                data.par_iter().map(|s| s.len() as u32).max().unwrap_or(0)
            }
            ChannelData::StringUTF16(data) => {
                data.par_iter().map(|s| s.len() as u32).max().unwrap_or(0)
            }
            ChannelData::VariableSizeByteArray(data) => {
                data.par_iter().map(|s| s.len() as u32).max().unwrap_or(0)
            }
            ChannelData::FixedSizeByteArray(data) => data.1 as u32,
            ChannelData::ArrayDInt8(_) => 1,
            ChannelData::ArrayDUInt8(_) => 1,
            ChannelData::ArrayDInt16(_) => 2,
            ChannelData::ArrayDUInt16(_) => 2,
            ChannelData::ArrayDFloat16(_) => 2,
            ChannelData::ArrayDInt24(_) => 3,
            ChannelData::ArrayDUInt24(_) => 3,
            ChannelData::ArrayDInt32(_) => 4,
            ChannelData::ArrayDUInt32(_) => 4,
            ChannelData::ArrayDFloat32(_) => 4,
            ChannelData::ArrayDInt48(_) => 6,
            ChannelData::ArrayDUInt48(_) => 6,
            ChannelData::ArrayDInt64(_) => 8,
            ChannelData::ArrayDUInt64(_) => 8,
            ChannelData::ArrayDFloat64(_) => 8,
            ChannelData::ArrayDComplex16(_) => 4,
            ChannelData::ArrayDComplex32(_) => 8,
            ChannelData::ArrayDComplex64(_) => 16,
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
                ChannelData::Float16(_) => 5,
                ChannelData::Int24(_) => 3,
                ChannelData::UInt24(_) => 1,
                ChannelData::Int32(_) => 3,
                ChannelData::UInt32(_) => 1,
                ChannelData::Float32(_) => 5,
                ChannelData::Int48(_) => 3,
                ChannelData::UInt48(_) => 1,
                ChannelData::Int64(_) => 3,
                ChannelData::UInt64(_) => 1,
                ChannelData::Float64(_) => 5,
                ChannelData::Complex16(_) => 16,
                ChannelData::Complex32(_) => 16,
                ChannelData::Complex64(_) => 16,
                ChannelData::StringSBC(_) => 7,
                ChannelData::StringUTF8(_) => 7,
                ChannelData::StringUTF16(_) => 7,
                ChannelData::VariableSizeByteArray(_) => 10,
                ChannelData::FixedSizeByteArray(_) => 10,
                ChannelData::ArrayDInt8(_) => 3,
                ChannelData::ArrayDUInt8(_) => 1,
                ChannelData::ArrayDInt16(_) => 3,
                ChannelData::ArrayDUInt16(_) => 1,
                ChannelData::ArrayDFloat16(_) => 5,
                ChannelData::ArrayDInt24(_) => 3,
                ChannelData::ArrayDUInt24(_) => 1,
                ChannelData::ArrayDInt32(_) => 3,
                ChannelData::ArrayDUInt32(_) => 1,
                ChannelData::ArrayDFloat32(_) => 5,
                ChannelData::ArrayDInt48(_) => 3,
                ChannelData::ArrayDUInt48(_) => 1,
                ChannelData::ArrayDInt64(_) => 3,
                ChannelData::ArrayDUInt64(_) => 1,
                ChannelData::ArrayDFloat64(_) => 5,
                ChannelData::ArrayDComplex16(_) => 16,
                ChannelData::ArrayDComplex32(_) => 16,
                ChannelData::ArrayDComplex64(_) => 16,
            }
        } else {
            // LE
            match self {
                ChannelData::Int8(_) => 2,
                ChannelData::UInt8(_) => 0,
                ChannelData::Int16(_) => 2,
                ChannelData::UInt16(_) => 0,
                ChannelData::Float16(_) => 4,
                ChannelData::Int24(_) => 2,
                ChannelData::UInt24(_) => 0,
                ChannelData::Int32(_) => 2,
                ChannelData::UInt32(_) => 0,
                ChannelData::Float32(_) => 4,
                ChannelData::Int48(_) => 2,
                ChannelData::UInt48(_) => 0,
                ChannelData::Int64(_) => 2,
                ChannelData::UInt64(_) => 0,
                ChannelData::Float64(_) => 4,
                ChannelData::Complex16(_) => 15,
                ChannelData::Complex32(_) => 15,
                ChannelData::Complex64(_) => 15,
                ChannelData::StringSBC(_) => 7,
                ChannelData::StringUTF8(_) => 7,
                ChannelData::StringUTF16(_) => 7,
                ChannelData::VariableSizeByteArray(_) => 10,
                ChannelData::FixedSizeByteArray(_) => 10,
                ChannelData::ArrayDInt8(_) => 2,
                ChannelData::ArrayDUInt8(_) => 0,
                ChannelData::ArrayDInt16(_) => 2,
                ChannelData::ArrayDUInt16(_) => 0,
                ChannelData::ArrayDFloat16(_) => 4,
                ChannelData::ArrayDInt24(_) => 2,
                ChannelData::ArrayDUInt24(_) => 0,
                ChannelData::ArrayDInt32(_) => 2,
                ChannelData::ArrayDUInt32(_) => 0,
                ChannelData::ArrayDFloat32(_) => 4,
                ChannelData::ArrayDInt48(_) => 2,
                ChannelData::ArrayDUInt48(_) => 0,
                ChannelData::ArrayDInt64(_) => 2,
                ChannelData::ArrayDUInt64(_) => 0,
                ChannelData::ArrayDFloat64(_) => 4,
                ChannelData::ArrayDComplex16(_) => 15,
                ChannelData::ArrayDComplex32(_) => 15,
                ChannelData::ArrayDComplex64(_) => 15,
            }
        }
    }
    /// Compares f32 floating point values
    pub fn compare_f32(&self, other: &ChannelData, epsilon: f32) -> bool {
        match self {
            ChannelData::Int8(_) => false,
            ChannelData::UInt8(_) => false,
            ChannelData::Int16(_) => false,
            ChannelData::UInt16(_) => false,
            ChannelData::Float16(a) => match other {
                ChannelData::Int8(_) => false,
                ChannelData::UInt8(_) => false,
                ChannelData::Int16(_) => false,
                ChannelData::UInt16(_) => false,
                ChannelData::Float16(b) => {
                    a.len() == b.len()
                        && a.iter().zip(b.iter()).all(|(a, b)| (a - b).abs() < epsilon)
                }
                ChannelData::Int24(_) => false,
                ChannelData::UInt24(_) => false,
                ChannelData::Int32(_) => false,
                ChannelData::UInt32(_) => false,
                ChannelData::Float32(_) => false,
                ChannelData::Int48(_) => false,
                ChannelData::UInt48(_) => false,
                ChannelData::Int64(_) => false,
                ChannelData::UInt64(_) => false,
                ChannelData::Float64(_) => false,
                ChannelData::Complex16(_) => false,
                ChannelData::Complex32(_) => false,
                ChannelData::Complex64(_) => false,
                ChannelData::StringSBC(_) => false,
                ChannelData::StringUTF8(_) => false,
                ChannelData::StringUTF16(_) => false,
                ChannelData::VariableSizeByteArray(_) => false,
                ChannelData::FixedSizeByteArray(_) => false,
                ChannelData::ArrayDInt8(_) => false,
                ChannelData::ArrayDUInt8(_) => false,
                ChannelData::ArrayDInt16(_) => false,
                ChannelData::ArrayDUInt16(_) => false,
                ChannelData::ArrayDFloat16(_) => false,
                ChannelData::ArrayDInt24(_) => false,
                ChannelData::ArrayDUInt24(_) => false,
                ChannelData::ArrayDInt32(_) => false,
                ChannelData::ArrayDUInt32(_) => false,
                ChannelData::ArrayDFloat32(_) => false,
                ChannelData::ArrayDInt48(_) => false,
                ChannelData::ArrayDUInt48(_) => false,
                ChannelData::ArrayDInt64(_) => false,
                ChannelData::ArrayDUInt64(_) => false,
                ChannelData::ArrayDFloat64(_) => false,
                ChannelData::ArrayDComplex16(_) => false,
                ChannelData::ArrayDComplex32(_) => false,
                ChannelData::ArrayDComplex64(_) => false,
            },
            ChannelData::Int24(_) => false,
            ChannelData::UInt24(_) => false,
            ChannelData::Int32(_) => false,
            ChannelData::UInt32(_) => false,
            ChannelData::Float32(a) => match other {
                ChannelData::Int8(_) => false,
                ChannelData::UInt8(_) => false,
                ChannelData::Int16(_) => false,
                ChannelData::UInt16(_) => false,
                ChannelData::Float16(_) => false,
                ChannelData::Int24(_) => false,
                ChannelData::UInt24(_) => false,
                ChannelData::Int32(_) => false,
                ChannelData::UInt32(_) => false,
                ChannelData::Float32(b) => {
                    a.len() == b.len()
                        && a.iter().zip(b.iter()).all(|(a, b)| (a - b).abs() < epsilon)
                }
                ChannelData::Int48(_) => false,
                ChannelData::UInt48(_) => false,
                ChannelData::Int64(_) => false,
                ChannelData::UInt64(_) => false,
                ChannelData::Float64(_) => false,
                ChannelData::Complex16(_) => false,
                ChannelData::Complex32(_) => false,
                ChannelData::Complex64(_) => false,
                ChannelData::StringSBC(_) => false,
                ChannelData::StringUTF8(_) => false,
                ChannelData::StringUTF16(_) => false,
                ChannelData::VariableSizeByteArray(_) => false,
                ChannelData::FixedSizeByteArray(_) => false,
                ChannelData::ArrayDInt8(_) => false,
                ChannelData::ArrayDUInt8(_) => false,
                ChannelData::ArrayDInt16(_) => false,
                ChannelData::ArrayDUInt16(_) => false,
                ChannelData::ArrayDFloat16(_) => false,
                ChannelData::ArrayDInt24(_) => false,
                ChannelData::ArrayDUInt24(_) => false,
                ChannelData::ArrayDInt32(_) => false,
                ChannelData::ArrayDUInt32(_) => false,
                ChannelData::ArrayDFloat32(_) => false,
                ChannelData::ArrayDInt48(_) => false,
                ChannelData::ArrayDUInt48(_) => false,
                ChannelData::ArrayDInt64(_) => false,
                ChannelData::ArrayDUInt64(_) => false,
                ChannelData::ArrayDFloat64(_) => false,
                ChannelData::ArrayDComplex16(_) => false,
                ChannelData::ArrayDComplex32(_) => false,
                ChannelData::ArrayDComplex64(_) => false,
            },
            ChannelData::Int48(_) => false,
            ChannelData::UInt48(_) => false,
            ChannelData::Int64(_) => false,
            ChannelData::UInt64(_) => false,
            ChannelData::Float64(_) => false,
            ChannelData::Complex16(a) => match other {
                ChannelData::Int8(_) => false,
                ChannelData::UInt8(_) => false,
                ChannelData::Int16(_) => false,
                ChannelData::UInt16(_) => false,
                ChannelData::Float16(_) => false,
                ChannelData::Int24(_) => false,
                ChannelData::UInt24(_) => false,
                ChannelData::Int32(_) => false,
                ChannelData::UInt32(_) => false,
                ChannelData::Float32(_) => false,
                ChannelData::Int48(_) => false,
                ChannelData::UInt48(_) => false,
                ChannelData::Int64(_) => false,
                ChannelData::UInt64(_) => false,
                ChannelData::Float64(_) => false,
                ChannelData::Complex16(b) => {
                    a.len() == b.len()
                        && a.clone()
                            .into_iter()
                            .zip(b.clone().into_iter())
                            .all(|(a, b)| {
                                ((a.re - b.re).abs() < epsilon) && ((a.im - b.im).abs() < epsilon)
                            })
                }
                ChannelData::Complex32(_) => false,
                ChannelData::Complex64(_) => false,
                ChannelData::StringSBC(_) => false,
                ChannelData::StringUTF8(_) => false,
                ChannelData::StringUTF16(_) => false,
                ChannelData::VariableSizeByteArray(_) => false,
                ChannelData::FixedSizeByteArray(_) => false,
                ChannelData::ArrayDInt8(_) => false,
                ChannelData::ArrayDUInt8(_) => false,
                ChannelData::ArrayDInt16(_) => false,
                ChannelData::ArrayDUInt16(_) => false,
                ChannelData::ArrayDFloat16(_) => false,
                ChannelData::ArrayDInt24(_) => false,
                ChannelData::ArrayDUInt24(_) => false,
                ChannelData::ArrayDInt32(_) => false,
                ChannelData::ArrayDUInt32(_) => false,
                ChannelData::ArrayDFloat32(_) => false,
                ChannelData::ArrayDInt48(_) => false,
                ChannelData::ArrayDUInt48(_) => false,
                ChannelData::ArrayDInt64(_) => false,
                ChannelData::ArrayDUInt64(_) => false,
                ChannelData::ArrayDFloat64(_) => false,
                ChannelData::ArrayDComplex16(_) => false,
                ChannelData::ArrayDComplex32(_) => false,
                ChannelData::ArrayDComplex64(_) => false,
            },
            ChannelData::Complex32(a) => match other {
                ChannelData::Int8(_) => false,
                ChannelData::UInt8(_) => false,
                ChannelData::Int16(_) => false,
                ChannelData::UInt16(_) => false,
                ChannelData::Float16(_) => false,
                ChannelData::Int24(_) => false,
                ChannelData::UInt24(_) => false,
                ChannelData::Int32(_) => false,
                ChannelData::UInt32(_) => false,
                ChannelData::Float32(_) => false,
                ChannelData::Int48(_) => false,
                ChannelData::UInt48(_) => false,
                ChannelData::Int64(_) => false,
                ChannelData::UInt64(_) => false,
                ChannelData::Float64(_) => false,
                ChannelData::Complex16(_) => false,
                ChannelData::Complex32(b) => {
                    a.len() == b.len()
                        && a.clone()
                            .into_iter()
                            .zip(b.clone().into_iter())
                            .all(|(a, b)| {
                                ((a.re - b.re).abs() < epsilon) && ((a.im - b.im).abs() < epsilon)
                            })
                }
                ChannelData::Complex64(_) => false,
                ChannelData::StringSBC(_) => false,
                ChannelData::StringUTF8(_) => false,
                ChannelData::StringUTF16(_) => false,
                ChannelData::VariableSizeByteArray(_) => false,
                ChannelData::FixedSizeByteArray(_) => false,
                ChannelData::ArrayDInt8(_) => false,
                ChannelData::ArrayDUInt8(_) => false,
                ChannelData::ArrayDInt16(_) => false,
                ChannelData::ArrayDUInt16(_) => false,
                ChannelData::ArrayDFloat16(_) => false,
                ChannelData::ArrayDInt24(_) => false,
                ChannelData::ArrayDUInt24(_) => false,
                ChannelData::ArrayDInt32(_) => false,
                ChannelData::ArrayDUInt32(_) => false,
                ChannelData::ArrayDFloat32(_) => false,
                ChannelData::ArrayDInt48(_) => false,
                ChannelData::ArrayDUInt48(_) => false,
                ChannelData::ArrayDInt64(_) => false,
                ChannelData::ArrayDUInt64(_) => false,
                ChannelData::ArrayDFloat64(_) => false,
                ChannelData::ArrayDComplex16(_) => false,
                ChannelData::ArrayDComplex32(_) => false,
                ChannelData::ArrayDComplex64(_) => false,
            },
            ChannelData::Complex64(_) => false,
            ChannelData::StringSBC(_) => false,
            ChannelData::StringUTF8(_) => false,
            ChannelData::StringUTF16(_) => false,
            ChannelData::VariableSizeByteArray(_) => false,
            ChannelData::FixedSizeByteArray(_) => false,
            ChannelData::ArrayDInt8(_) => false,
            ChannelData::ArrayDUInt8(_) => false,
            ChannelData::ArrayDInt16(_) => false,
            ChannelData::ArrayDUInt16(_) => false,
            ChannelData::ArrayDFloat16(_) => todo!(),
            ChannelData::ArrayDInt24(_) => false,
            ChannelData::ArrayDUInt24(_) => false,
            ChannelData::ArrayDInt32(_) => false,
            ChannelData::ArrayDUInt32(_) => false,
            ChannelData::ArrayDFloat32(_) => todo!(),
            ChannelData::ArrayDInt48(_) => false,
            ChannelData::ArrayDUInt48(_) => false,
            ChannelData::ArrayDInt64(_) => false,
            ChannelData::ArrayDUInt64(_) => false,
            ChannelData::ArrayDFloat64(_) => todo!(),
            ChannelData::ArrayDComplex16(_) => todo!(),
            ChannelData::ArrayDComplex32(_) => todo!(),
            ChannelData::ArrayDComplex64(_) => todo!(),
        }
    }
    /// Compares floating point f64 values
    pub fn compare_f64(&self, other: &ChannelData, epsilon: f64) -> bool {
        match self {
            ChannelData::Int8(_) => false,
            ChannelData::UInt8(_) => false,
            ChannelData::Int16(_) => false,
            ChannelData::UInt16(_) => false,
            ChannelData::Float16(_) => false,
            ChannelData::Int24(_) => false,
            ChannelData::UInt24(_) => false,
            ChannelData::Int32(_) => false,
            ChannelData::UInt32(_) => false,
            ChannelData::Float32(_) => false,
            ChannelData::Int48(_) => false,
            ChannelData::UInt48(_) => false,
            ChannelData::Int64(_) => false,
            ChannelData::UInt64(_) => false,
            ChannelData::Float64(a) => match other {
                ChannelData::Int8(_) => false,
                ChannelData::UInt8(_) => false,
                ChannelData::Int16(_) => false,
                ChannelData::UInt16(_) => false,
                ChannelData::Float16(_) => false,
                ChannelData::Int24(_) => false,
                ChannelData::UInt24(_) => false,
                ChannelData::Int32(_) => false,
                ChannelData::UInt32(_) => false,
                ChannelData::Float32(_) => false,
                ChannelData::Int48(_) => false,
                ChannelData::UInt48(_) => false,
                ChannelData::Int64(_) => false,
                ChannelData::UInt64(_) => false,
                ChannelData::Float64(b) => {
                    a.len() == b.len()
                        && a.iter().zip(b.iter()).all(|(a, b)| (a - b).abs() < epsilon)
                }
                ChannelData::Complex16(_) => false,
                ChannelData::Complex32(_) => false,
                ChannelData::Complex64(_) => false,
                ChannelData::StringSBC(_) => false,
                ChannelData::StringUTF8(_) => false,
                ChannelData::StringUTF16(_) => false,
                ChannelData::VariableSizeByteArray(_) => false,
                ChannelData::FixedSizeByteArray(_) => false,
                ChannelData::ArrayDInt8(_) => false,
                ChannelData::ArrayDUInt8(_) => false,
                ChannelData::ArrayDInt16(_) => false,
                ChannelData::ArrayDUInt16(_) => false,
                ChannelData::ArrayDFloat16(_) => false,
                ChannelData::ArrayDInt24(_) => false,
                ChannelData::ArrayDUInt24(_) => false,
                ChannelData::ArrayDInt32(_) => false,
                ChannelData::ArrayDUInt32(_) => false,
                ChannelData::ArrayDFloat32(_) => false,
                ChannelData::ArrayDInt48(_) => false,
                ChannelData::ArrayDUInt48(_) => false,
                ChannelData::ArrayDInt64(_) => false,
                ChannelData::ArrayDUInt64(_) => false,
                ChannelData::ArrayDFloat64(_) => false,
                ChannelData::ArrayDComplex16(_) => false,
                ChannelData::ArrayDComplex32(_) => false,
                ChannelData::ArrayDComplex64(_) => false,
            },
            ChannelData::Complex16(_) => false,
            ChannelData::Complex32(_) => false,
            ChannelData::Complex64(a) => match other {
                ChannelData::Int8(_) => false,
                ChannelData::UInt8(_) => false,
                ChannelData::Int16(_) => false,
                ChannelData::UInt16(_) => false,
                ChannelData::Float16(_) => false,
                ChannelData::Int24(_) => false,
                ChannelData::UInt24(_) => false,
                ChannelData::Int32(_) => false,
                ChannelData::UInt32(_) => false,
                ChannelData::Float32(_) => false,
                ChannelData::Int48(_) => false,
                ChannelData::UInt48(_) => false,
                ChannelData::Int64(_) => false,
                ChannelData::UInt64(_) => false,
                ChannelData::Float64(_) => false,
                ChannelData::Complex16(_) => false,
                ChannelData::Complex32(_) => false,
                ChannelData::Complex64(b) => {
                    a.len() == b.len()
                        && a.clone()
                            .into_iter()
                            .zip(b.clone().into_iter())
                            .all(|(a, b)| {
                                ((a.re - b.re).abs() < epsilon) && ((a.im - b.im).abs() < epsilon)
                            })
                }
                ChannelData::StringSBC(_) => false,
                ChannelData::StringUTF8(_) => false,
                ChannelData::StringUTF16(_) => false,
                ChannelData::VariableSizeByteArray(_) => false,
                ChannelData::FixedSizeByteArray(_) => false,
                ChannelData::ArrayDInt8(_) => false,
                ChannelData::ArrayDUInt8(_) => false,
                ChannelData::ArrayDInt16(_) => false,
                ChannelData::ArrayDUInt16(_) => false,
                ChannelData::ArrayDFloat16(_) => false,
                ChannelData::ArrayDInt24(_) => false,
                ChannelData::ArrayDUInt24(_) => false,
                ChannelData::ArrayDInt32(_) => false,
                ChannelData::ArrayDUInt32(_) => false,
                ChannelData::ArrayDFloat32(_) => false,
                ChannelData::ArrayDInt48(_) => false,
                ChannelData::ArrayDUInt48(_) => false,
                ChannelData::ArrayDInt64(_) => false,
                ChannelData::ArrayDUInt64(_) => false,
                ChannelData::ArrayDFloat64(_) => false,
                ChannelData::ArrayDComplex16(_) => false,
                ChannelData::ArrayDComplex32(_) => false,
                ChannelData::ArrayDComplex64(_) => false,
            },
            ChannelData::StringSBC(_) => false,
            ChannelData::StringUTF8(_) => false,
            ChannelData::StringUTF16(_) => false,
            ChannelData::VariableSizeByteArray(_) => false,
            ChannelData::FixedSizeByteArray(_) => false,
            ChannelData::ArrayDInt8(_) => false,
            ChannelData::ArrayDUInt8(_) => false,
            ChannelData::ArrayDInt16(_) => false,
            ChannelData::ArrayDUInt16(_) => false,
            ChannelData::ArrayDFloat16(_) => todo!(),
            ChannelData::ArrayDInt24(_) => false,
            ChannelData::ArrayDUInt24(_) => false,
            ChannelData::ArrayDInt32(_) => false,
            ChannelData::ArrayDUInt32(_) => false,
            ChannelData::ArrayDFloat32(_) => todo!(),
            ChannelData::ArrayDInt48(_) => false,
            ChannelData::ArrayDUInt48(_) => false,
            ChannelData::ArrayDInt64(_) => false,
            ChannelData::ArrayDUInt64(_) => false,
            ChannelData::ArrayDFloat64(_) => todo!(),
            ChannelData::ArrayDComplex16(_) => todo!(),
            ChannelData::ArrayDComplex32(_) => todo!(),
            ChannelData::ArrayDComplex64(_) => todo!(),
        }
    }
    /// returns raw bytes vectors from ndarray
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            ChannelData::Int8(a) => a.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::UInt8(a) => a.to_vec(),
            ChannelData::Int16(a) => a.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::UInt16(a) => a.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::Float16(a) => a.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::Int24(a) => a.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::UInt24(a) => a.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::Int32(a) => a.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::UInt32(a) => a.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::Float32(a) => a.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::Int48(a) => a.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::UInt48(a) => a.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::Int64(a) => a.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::UInt64(a) => a.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::Float64(a) => a.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::Complex16(a) => a.0.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::Complex32(a) => a.0.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::Complex64(a) => a.0.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::StringSBC(a) => {
                let nbytes = self.byte_count() as usize;
                a.iter()
                    .flat_map(|x| {
                        let str_bytes = x.to_string().into_bytes();
                        let n_str_bytes = str_bytes.len();
                        if nbytes > n_str_bytes {
                            [str_bytes, vec![0u8; nbytes - n_str_bytes]].concat()
                        } else {
                            str_bytes
                        }
                    })
                    .collect()
            }
            ChannelData::StringUTF8(a) => {
                let nbytes = self.byte_count() as usize;
                a.iter()
                    .flat_map(|x| {
                        let str_bytes = x.to_string().into_bytes();
                        let n_str_bytes = str_bytes.len();
                        if nbytes > n_str_bytes {
                            [str_bytes, vec![0u8; nbytes - n_str_bytes]].concat()
                        } else {
                            str_bytes
                        }
                    })
                    .collect()
            }
            ChannelData::StringUTF16(a) => {
                let nbytes = self.byte_count() as usize;
                a.iter()
                    .flat_map(|x| {
                        let str_bytes = x.to_string().into_bytes();
                        let n_str_bytes = str_bytes.len();
                        if nbytes > n_str_bytes {
                            [str_bytes, vec![0u8; nbytes - n_str_bytes]].concat()
                        } else {
                            str_bytes
                        }
                    })
                    .collect()
            }
            ChannelData::VariableSizeByteArray(a) => {
                a.iter().flatten().cloned().collect::<Vec<u8>>()
            }
            ChannelData::FixedSizeByteArray(a) => a.0.clone(),
            ChannelData::ArrayDInt8(a) => a.0.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::ArrayDUInt8(a) => a.0.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::ArrayDInt16(a) => a.0.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::ArrayDUInt16(a) => a.0.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::ArrayDFloat16(a) => a.0.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::ArrayDInt24(a) => a.0.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::ArrayDUInt24(a) => a.0.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::ArrayDInt32(a) => a.0.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::ArrayDUInt32(a) => a.0.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::ArrayDFloat32(a) => a.0.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::ArrayDInt48(a) => a.0.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::ArrayDUInt48(a) => a.0.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::ArrayDInt64(a) => a.0.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::ArrayDUInt64(a) => a.0.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::ArrayDFloat64(a) => a.0.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::ArrayDComplex16(a) => {
                a.0 .0.iter().flat_map(|x| x.to_ne_bytes()).collect()
            }
            ChannelData::ArrayDComplex32(a) => {
                a.0 .0.iter().flat_map(|x| x.to_ne_bytes()).collect()
            }
            ChannelData::ArrayDComplex64(a) => {
                a.0 .0.iter().flat_map(|x| x.to_ne_bytes()).collect()
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
            ChannelData::Float16(_) => 1,
            ChannelData::Int24(_) => 1,
            ChannelData::UInt24(_) => 1,
            ChannelData::Int32(_) => 1,
            ChannelData::UInt32(_) => 1,
            ChannelData::Float32(_) => 1,
            ChannelData::Int48(_) => 1,
            ChannelData::UInt48(_) => 1,
            ChannelData::Int64(_) => 1,
            ChannelData::UInt64(_) => 1,
            ChannelData::Float64(_) => 1,
            ChannelData::Complex16(_) => 1,
            ChannelData::Complex32(_) => 1,
            ChannelData::Complex64(_) => 1,
            ChannelData::StringSBC(_) => 1,
            ChannelData::StringUTF8(_) => 1,
            ChannelData::StringUTF16(_) => 1,
            ChannelData::VariableSizeByteArray(_) => 1,
            ChannelData::FixedSizeByteArray(_) => 1,
            ChannelData::ArrayDInt8(a) => a.1 .0.len(),
            ChannelData::ArrayDUInt8(a) => a.1 .0.len(),
            ChannelData::ArrayDInt16(a) => a.1 .0.len(),
            ChannelData::ArrayDUInt16(a) => a.1 .0.len(),
            ChannelData::ArrayDFloat16(a) => a.1 .0.len(),
            ChannelData::ArrayDInt24(a) => a.1 .0.len(),
            ChannelData::ArrayDUInt24(a) => a.1 .0.len(),
            ChannelData::ArrayDInt32(a) => a.1 .0.len(),
            ChannelData::ArrayDUInt32(a) => a.1 .0.len(),
            ChannelData::ArrayDFloat32(a) => a.1 .0.len(),
            ChannelData::ArrayDInt48(a) => a.1 .0.len(),
            ChannelData::ArrayDUInt48(a) => a.1 .0.len(),
            ChannelData::ArrayDInt64(a) => a.1 .0.len(),
            ChannelData::ArrayDUInt64(a) => a.1 .0.len(),
            ChannelData::ArrayDFloat64(a) => a.1 .0.len(),
            ChannelData::ArrayDComplex16(a) => a.1 .0.len(),
            ChannelData::ArrayDComplex32(a) => a.1 .0.len(),
            ChannelData::ArrayDComplex64(a) => a.1 .0.len(),
        }
    }
    /// returns the shape of channel
    pub fn shape(&self) -> (Vec<usize>, Order) {
        match self {
            ChannelData::Int8(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::UInt8(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::Int16(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::UInt16(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::Float16(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::Int24(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::UInt24(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::Int32(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::UInt32(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::Float32(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::Int48(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::UInt48(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::Int64(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::UInt64(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::Float64(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::Complex16(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::Complex32(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::Complex64(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::StringSBC(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::StringUTF8(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::StringUTF16(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::VariableSizeByteArray(a) => (vec![a.len(); 1], Order::RowMajor),
            ChannelData::FixedSizeByteArray(a) => (vec![a.0.len(); 1], Order::RowMajor),
            ChannelData::ArrayDInt8(a) => a.1.clone(),
            ChannelData::ArrayDUInt8(a) => a.1.clone(),
            ChannelData::ArrayDInt16(a) => a.1.clone(),
            ChannelData::ArrayDUInt16(a) => a.1.clone(),
            ChannelData::ArrayDFloat16(a) => a.1.clone(),
            ChannelData::ArrayDInt24(a) => a.1.clone(),
            ChannelData::ArrayDUInt24(a) => a.1.clone(),
            ChannelData::ArrayDInt32(a) => a.1.clone(),
            ChannelData::ArrayDUInt32(a) => a.1.clone(),
            ChannelData::ArrayDFloat32(a) => a.1.clone(),
            ChannelData::ArrayDInt48(a) => a.1.clone(),
            ChannelData::ArrayDUInt48(a) => a.1.clone(),
            ChannelData::ArrayDInt64(a) => a.1.clone(),
            ChannelData::ArrayDUInt64(a) => a.1.clone(),
            ChannelData::ArrayDFloat64(a) => a.1.clone(),
            ChannelData::ArrayDComplex16(a) => a.1.clone(),
            ChannelData::ArrayDComplex32(a) => a.1.clone(),
            ChannelData::ArrayDComplex64(a) => a.1.clone(),
        }
    }
    /// returns optional tuple of minimum and maximum values contained in the channel
    pub fn min_max(&self) -> (Option<f64>, Option<f64>) {
        match self {
            ChannelData::Int8(a) => {
                let min = a.iter().min().map(|v| *v as f64);
                let max = a.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::UInt8(a) => {
                let min = a.iter().min().map(|v| *v as f64);
                let max = a.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::Int16(a) => {
                let min = a.iter().min().map(|v| *v as f64);
                let max = a.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::UInt16(a) => {
                let min = a.iter().min().map(|v| *v as f64);
                let max = a.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::Float16(a) => {
                let max = a
                    .iter()
                    .reduce(|accum, item| if accum >= item { accum } else { item })
                    .map(|v| *v as f64);
                let min = a
                    .iter()
                    .reduce(|accum, item| if accum <= item { accum } else { item })
                    .map(|v| *v as f64);
                (min, max)
            }
            ChannelData::Int24(a) => {
                let min = a.iter().min().map(|v| *v as f64);
                let max = a.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::UInt24(a) => {
                let min = a.iter().min().map(|v| *v as f64);
                let max = a.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::Int32(a) => {
                let min = a.iter().min().map(|v| *v as f64);
                let max = a.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::UInt32(a) => {
                let min = a.iter().min().map(|v| *v as f64);
                let max = a.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::Float32(a) => {
                let max = a
                    .iter()
                    .reduce(|accum, item| if accum >= item { accum } else { item })
                    .map(|v| *v as f64);
                let min = a
                    .iter()
                    .reduce(|accum, item| if accum <= item { accum } else { item })
                    .map(|v| *v as f64);
                (min, max)
            }
            ChannelData::Int48(a) => {
                let min = a.iter().min().map(|v| *v as f64);
                let max = a.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::UInt48(a) => {
                let min = a.iter().min().map(|v| *v as f64);
                let max = a.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::Int64(a) => {
                let min = a.iter().min().map(|v| *v as f64);
                let max = a.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::UInt64(a) => {
                let min = a.iter().min().map(|v| *v as f64);
                let max = a.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::Float64(a) => {
                let max = a
                    .iter()
                    .reduce(|accum, item| if accum >= item { accum } else { item });
                let min = a
                    .iter()
                    .reduce(|accum, item| if accum <= item { accum } else { item });
                (min.cloned(), max.cloned())
            }
            ChannelData::Complex16(_) => (None, None),
            ChannelData::Complex32(_) => (None, None),
            ChannelData::Complex64(_) => (None, None),
            ChannelData::StringSBC(_) => (None, None),
            ChannelData::StringUTF8(_) => (None, None),
            ChannelData::StringUTF16(_) => (None, None),
            ChannelData::VariableSizeByteArray(_) => (None, None),
            ChannelData::FixedSizeByteArray(_) => (None, None),
            ChannelData::ArrayDInt8(a) => {
                let min = a.0.iter().min().map(|v| *v as f64);
                let max = a.0.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDUInt8(a) => {
                let min = a.0.iter().min().map(|v| *v as f64);
                let max = a.0.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDInt16(a) => {
                let min = a.0.iter().min().map(|v| *v as f64);
                let max = a.0.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDUInt16(a) => {
                let min = a.0.iter().min().map(|v| *v as f64);
                let max = a.0.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDFloat16(a) => {
                let max =
                    a.0.iter()
                        .reduce(|accum, item| if accum >= item { accum } else { item })
                        .map(|v| *v as f64);
                let min =
                    a.0.iter()
                        .reduce(|accum, item| if accum <= item { accum } else { item })
                        .map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDInt24(a) => {
                let min = a.0.iter().min().map(|v| *v as f64);
                let max = a.0.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDUInt24(a) => {
                let min = a.0.iter().min().map(|v| *v as f64);
                let max = a.0.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDInt32(a) => {
                let min = a.0.iter().min().map(|v| *v as f64);
                let max = a.0.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDUInt32(a) => {
                let min = a.0.iter().min().map(|v| *v as f64);
                let max = a.0.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDFloat32(a) => {
                let max =
                    a.0.iter()
                        .reduce(|accum, item| if accum >= item { accum } else { item })
                        .map(|v| *v as f64);
                let min =
                    a.0.iter()
                        .reduce(|accum, item| if accum <= item { accum } else { item })
                        .map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDInt48(a) => {
                let min = a.0.iter().min().map(|v| *v as f64);
                let max = a.0.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDUInt48(a) => {
                let min = a.0.iter().min().map(|v| *v as f64);
                let max = a.0.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDInt64(a) => {
                let min = a.0.iter().min().map(|v| *v as f64);
                let max = a.0.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDUInt64(a) => {
                let min = a.0.iter().min().map(|v| *v as f64);
                let max = a.0.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDFloat64(a) => {
                let max =
                    a.0.iter()
                        .reduce(|accum, item| if accum >= item { accum } else { item })
                        .cloned();
                let min =
                    a.0.iter()
                        .reduce(|accum, item| if accum <= item { accum } else { item })
                        .cloned();
                (min, max)
            }
            ChannelData::ArrayDComplex16(_) => (None, None),
            ChannelData::ArrayDComplex32(_) => (None, None),
            ChannelData::ArrayDComplex64(_) => (None, None),
        }
    }
}

impl Default for ChannelData {
    fn default() -> Self {
        ChannelData::UInt8(Vec::<u8>::new())
    }
}

/// Initialises a channel array type depending of cn_type, cn_data_type and if array
pub fn data_type_init(cn_type: u8, cn_data_type: u8, n_bytes: u32, is_array: bool) -> ChannelData {
    if !is_array {
        // Not an array
        if cn_type != 3 || cn_type != 6 {
            // not virtual channel or vlsd
            match cn_data_type {
                0 | 1 => {
                    // unsigned int
                    if n_bytes <= 1 {
                        ChannelData::UInt8(Vec::<u8>::new())
                    } else if n_bytes == 2 {
                        ChannelData::UInt16(Vec::<u16>::new())
                    } else if n_bytes == 3 {
                        ChannelData::UInt24(Vec::<u32>::new())
                    } else if n_bytes == 4 {
                        ChannelData::UInt32(Vec::<u32>::new())
                    } else if n_bytes <= 6 {
                        ChannelData::UInt48(Vec::<u64>::new())
                    } else {
                        ChannelData::UInt64(Vec::<u64>::new())
                    }
                }
                2 | 3 => {
                    // signed int
                    if n_bytes <= 1 {
                        ChannelData::Int8(Vec::<i8>::new())
                    } else if n_bytes == 2 {
                        ChannelData::Int16(Vec::<i16>::new())
                    } else if n_bytes == 3 {
                        ChannelData::Int24(Vec::<i32>::new())
                    } else if n_bytes == 4 {
                        ChannelData::Int32(Vec::<i32>::new())
                    } else if n_bytes <= 6 {
                        ChannelData::Int48(Vec::<i64>::new())
                    } else {
                        ChannelData::Int64(Vec::<i64>::new())
                    }
                }
                4 | 5 => {
                    // float
                    if n_bytes <= 2 {
                        ChannelData::Float16(Vec::<f32>::new())
                    } else if n_bytes <= 4 {
                        ChannelData::Float32(Vec::<f32>::new())
                    } else {
                        ChannelData::Float64(Vec::<f64>::new())
                    }
                }
                15 | 16 => {
                    // complex
                    if n_bytes <= 2 {
                        ChannelData::Complex16(ArrowComplex::<f32>::zeros(0))
                    } else if n_bytes <= 4 {
                        ChannelData::Complex32(ArrowComplex::<f32>::zeros(0))
                    } else {
                        ChannelData::Complex64(ArrowComplex::<f64>::zeros(0))
                    }
                }
                6 => {
                    // SBC ISO-8859-1 to be converted into UTF8
                    ChannelData::StringSBC(vec![String::new(); 0])
                }
                7 => {
                    // String UTF8
                    ChannelData::StringUTF8(vec![String::new(); 0])
                }
                8 | 9 => {
                    // String UTF16 to be converted into UTF8
                    ChannelData::StringUTF16(vec![String::new(); 0])
                }
                _ => {
                    // bytearray
                    if cn_type == 1 {
                        // VLSD
                        ChannelData::VariableSizeByteArray(vec![vec![0u8; 0]; 0])
                    } else {
                        ChannelData::FixedSizeByteArray((vec![0u8; 0], 0))
                    }
                }
            }
        } else {
            // virtual channels, cn_bit_count = 0 -> n_bytes = 0, must be LE unsigned int
            ChannelData::UInt64(Vec::<u64>::new())
        }
    } else if cn_type != 3 && cn_type != 6 {
        // Array not virtual
        match cn_data_type {
            0 | 1 => {
                // unsigned int
                if n_bytes <= 1 {
                    ChannelData::ArrayDUInt8((
                        Vec::<u8>::new(),
                        (Vec::<usize>::new(), Order::RowMajor),
                    ))
                } else if n_bytes == 2 {
                    ChannelData::ArrayDUInt16((
                        Vec::<u16>::new(),
                        (Vec::<usize>::new(), Order::RowMajor),
                    ))
                } else if n_bytes == 3 {
                    ChannelData::ArrayDUInt24((
                        Vec::<u32>::new(),
                        (Vec::<usize>::new(), Order::RowMajor),
                    ))
                } else if n_bytes == 4 {
                    ChannelData::ArrayDUInt32((
                        Vec::<u32>::new(),
                        (Vec::<usize>::new(), Order::RowMajor),
                    ))
                } else if n_bytes <= 6 {
                    ChannelData::ArrayDUInt48((
                        Vec::<u64>::new(),
                        (Vec::<usize>::new(), Order::RowMajor),
                    ))
                } else {
                    ChannelData::ArrayDUInt64((
                        Vec::<u64>::new(),
                        (Vec::<usize>::new(), Order::RowMajor),
                    ))
                }
            }
            2 | 3 => {
                // signed int
                if n_bytes <= 1 {
                    ChannelData::ArrayDInt8((
                        Vec::<i8>::new(),
                        (Vec::<usize>::new(), Order::RowMajor),
                    ))
                } else if n_bytes == 2 {
                    ChannelData::ArrayDInt16((
                        Vec::<i16>::new(),
                        (Vec::<usize>::new(), Order::RowMajor),
                    ))
                } else if n_bytes == 3 {
                    ChannelData::ArrayDInt24((
                        Vec::<i32>::new(),
                        (Vec::<usize>::new(), Order::RowMajor),
                    ))
                } else if n_bytes == 4 {
                    ChannelData::ArrayDInt32((
                        Vec::<i32>::new(),
                        (Vec::<usize>::new(), Order::RowMajor),
                    ))
                } else if n_bytes <= 6 {
                    ChannelData::ArrayDInt48((
                        Vec::<i64>::new(),
                        (Vec::<usize>::new(), Order::RowMajor),
                    ))
                } else {
                    ChannelData::ArrayDInt64((
                        Vec::<i64>::new(),
                        (Vec::<usize>::new(), Order::RowMajor),
                    ))
                }
            }
            4 | 5 => {
                // float
                if n_bytes <= 2 {
                    ChannelData::ArrayDFloat16((
                        Vec::<f32>::new(),
                        (Vec::<usize>::new(), Order::RowMajor),
                    ))
                } else if n_bytes <= 4 {
                    ChannelData::ArrayDFloat32((
                        Vec::<f32>::new(),
                        (Vec::<usize>::new(), Order::RowMajor),
                    ))
                } else {
                    ChannelData::ArrayDFloat64((
                        Vec::<f64>::new(),
                        (Vec::<usize>::new(), Order::RowMajor),
                    ))
                }
            }
            15 | 16 => {
                // complex
                if n_bytes <= 2 {
                    ChannelData::ArrayDComplex16((
                        ArrowComplex::<f32>::zeros(0),
                        (Vec::<usize>::new(), Order::RowMajor),
                    ))
                } else if n_bytes <= 4 {
                    ChannelData::ArrayDComplex32((
                        ArrowComplex::<f32>::zeros(0),
                        (Vec::<usize>::new(), Order::RowMajor),
                    ))
                } else {
                    ChannelData::ArrayDComplex64((
                        ArrowComplex::<f64>::zeros(0),
                        (Vec::<usize>::new(), Order::RowMajor),
                    ))
                }
            }
            _ => {
                // strings or bytes arrays not implemented
                todo!();
            }
        }
    } else {
        // virtual channels arrays not implemented, can it even exists ?
        todo!();
    }
}

impl fmt::Display for ChannelData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChannelData::Int8(array) => {
                writeln!(f, "{:?}", array)
            }
            ChannelData::UInt8(array) => {
                writeln!(f, "{:?}", array)
            }
            ChannelData::Int16(array) => {
                writeln!(f, "{:?}", array)
            }
            ChannelData::UInt16(array) => {
                writeln!(f, "{:?}", array)
            }
            ChannelData::Float16(array) => {
                writeln!(f, "{:?}", array)
            }
            ChannelData::Int24(array) => {
                writeln!(f, "{:?}", array)
            }
            ChannelData::UInt24(array) => {
                writeln!(f, "{:?}", array)
            }
            ChannelData::Int32(array) => {
                writeln!(f, "{:?}", array)
            }
            ChannelData::UInt32(array) => {
                writeln!(f, "{:?}", array)
            }
            ChannelData::Float32(array) => {
                writeln!(f, "{:?}", array)
            }
            ChannelData::Int48(array) => {
                writeln!(f, "{:?}", array)
            }
            ChannelData::UInt48(array) => {
                writeln!(f, "{:?}", array)
            }
            ChannelData::Int64(array) => {
                writeln!(f, "{:?}", array)
            }
            ChannelData::UInt64(array) => {
                writeln!(f, "{:?}", array)
            }
            ChannelData::Float64(array) => {
                writeln!(f, "{:?}", array)
            }
            ChannelData::Complex16(array) => {
                writeln!(f, "{:?}", array)
            }
            ChannelData::Complex32(array) => {
                writeln!(f, "{:?}", array)
            }
            ChannelData::Complex64(array) => {
                writeln!(f, "{:?}", array)
            }
            ChannelData::StringSBC(array) => {
                for text in array.iter() {
                    writeln!(f, " {} ", text)?;
                }
                writeln!(f, " ")
            }
            ChannelData::StringUTF8(array) => {
                for text in array.iter() {
                    writeln!(f, " {} ", text)?;
                }
                writeln!(f, " ")
            }
            ChannelData::StringUTF16(array) => {
                for text in array.iter() {
                    writeln!(f, " {} ", text)?;
                }
                writeln!(f, " ")
            }
            ChannelData::VariableSizeByteArray(array) => {
                for text in array.iter() {
                    writeln!(f, " {:?} ", text)?;
                }
                writeln!(f, " ")
            }
            ChannelData::FixedSizeByteArray(array) => {
                for text in array.0.iter() {
                    writeln!(f, " {:?} ", text)?;
                }
                writeln!(f, " ")
            }
            ChannelData::ArrayDInt8(array) => {
                writeln!(f, "{:?}", array.0)
            }
            ChannelData::ArrayDUInt8(array) => {
                writeln!(f, "{:?}", array.0)
            }
            ChannelData::ArrayDInt16(array) => {
                writeln!(f, "{:?}", array.0)
            }
            ChannelData::ArrayDUInt16(array) => {
                writeln!(f, "{:?}", array.0)
            }
            ChannelData::ArrayDFloat16(array) => {
                writeln!(f, "{:?}", array.0)
            }
            ChannelData::ArrayDInt24(array) => {
                writeln!(f, "{:?}", array.0)
            }
            ChannelData::ArrayDUInt24(array) => {
                writeln!(f, "{:?}", array.0)
            }
            ChannelData::ArrayDInt32(array) => {
                writeln!(f, "{:?}", array.0)
            }
            ChannelData::ArrayDUInt32(array) => {
                writeln!(f, "{:?}", array.0)
            }
            ChannelData::ArrayDFloat32(array) => {
                writeln!(f, "{:?}", array.0)
            }
            ChannelData::ArrayDInt48(array) => {
                writeln!(f, "{:?}", array.0)
            }
            ChannelData::ArrayDUInt48(array) => {
                writeln!(f, "{:?}", array.0)
            }
            ChannelData::ArrayDInt64(array) => {
                writeln!(f, "{:?}", array.0)
            }
            ChannelData::ArrayDUInt64(array) => {
                writeln!(f, "{:?}", array.0)
            }
            ChannelData::ArrayDFloat64(array) => {
                writeln!(f, "{:?}", array.0)
            }
            ChannelData::ArrayDComplex16(array) => {
                writeln!(f, "{:?}", array.0)
            }
            ChannelData::ArrayDComplex32(array) => {
                writeln!(f, "{:?}", array.0)
            }
            ChannelData::ArrayDComplex64(array) => {
                writeln!(f, "{:?}", array.0)
            }
        }
    }
}
