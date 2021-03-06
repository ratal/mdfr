//! this module holds the channel data enum and related implementations
use ndarray::{Array1, ArrayD, IxDyn};
use num::Complex;
use numpy::{IntoPyArray, PyArray1, PyArrayDyn, ToPyArray};
use pyo3::prelude::*;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::Serialize;
use std::fmt;

/// channel data type enum.
/// most common data type is 1D ndarray for timeseries with element types numeric.
/// vector of string or bytes also exists.
/// Dynamic dimension arrays ArrayD are also existing to cover CABlock arrays data.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ChannelData {
    Int8(Array1<i8>),
    UInt8(Array1<u8>),
    Int16(Array1<i16>),
    UInt16(Array1<u16>),
    Float16(Array1<f32>),
    Int24(Array1<i32>),
    UInt24(Array1<u32>),
    Int32(Array1<i32>),
    UInt32(Array1<u32>),
    Float32(Array1<f32>),
    Int48(Array1<i64>),
    UInt48(Array1<u64>),
    Int64(Array1<i64>),
    UInt64(Array1<u64>),
    Float64(Array1<f64>),
    Complex16(Array1<Complex<f32>>),
    Complex32(Array1<Complex<f32>>),
    Complex64(Array1<Complex<f64>>),
    StringSBC(Vec<String>),
    StringUTF8(Vec<String>),
    StringUTF16(Vec<String>),
    ByteArray(Vec<Vec<u8>>),
    ArrayDInt8(ArrayD<i8>),
    ArrayDUInt8(ArrayD<u8>),
    ArrayDInt16(ArrayD<i16>),
    ArrayDUInt16(ArrayD<u16>),
    ArrayDFloat16(ArrayD<f32>),
    ArrayDInt24(ArrayD<i32>),
    ArrayDUInt24(ArrayD<u32>),
    ArrayDInt32(ArrayD<i32>),
    ArrayDUInt32(ArrayD<u32>),
    ArrayDFloat32(ArrayD<f32>),
    ArrayDInt48(ArrayD<i64>),
    ArrayDUInt48(ArrayD<u64>),
    ArrayDInt64(ArrayD<i64>),
    ArrayDUInt64(ArrayD<u64>),
    ArrayDFloat64(ArrayD<f64>),
    ArrayDComplex16(ArrayD<Complex<f32>>),
    ArrayDComplex32(ArrayD<Complex<f32>>),
    ArrayDComplex64(ArrayD<Complex<f64>>),
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
        n_elements: usize,
    ) -> ChannelData {
        if cn_type == 3 || cn_type == 6 {
            // virtual channel
            ChannelData::UInt64(Array1::<u64>::from_iter(0..cycle_count))
        } else {
            match self {
                ChannelData::Int8(_) => {
                    ChannelData::Int8(Array1::<i8>::zeros(cycle_count as usize))
                }
                ChannelData::UInt8(_) => {
                    ChannelData::UInt8(Array1::<u8>::zeros(cycle_count as usize))
                }
                ChannelData::Int16(_) => {
                    ChannelData::Int16(Array1::<i16>::zeros(cycle_count as usize))
                }
                ChannelData::UInt16(_) => {
                    ChannelData::UInt16(Array1::<u16>::zeros(cycle_count as usize))
                }
                ChannelData::Float16(_) => {
                    ChannelData::Float16(Array1::<f32>::zeros(cycle_count as usize))
                }
                ChannelData::Int24(_) => {
                    ChannelData::Int24(Array1::<i32>::zeros(cycle_count as usize))
                }
                ChannelData::UInt24(_) => {
                    ChannelData::UInt24(Array1::<u32>::zeros(cycle_count as usize))
                }
                ChannelData::Int32(_) => {
                    ChannelData::Int32(Array1::<i32>::zeros(cycle_count as usize))
                }
                ChannelData::UInt32(_) => {
                    ChannelData::UInt32(Array1::<u32>::zeros(cycle_count as usize))
                }
                ChannelData::Float32(_) => {
                    ChannelData::Float32(Array1::<f32>::zeros(cycle_count as usize))
                }
                ChannelData::Int48(_) => {
                    ChannelData::Int48(Array1::<i64>::zeros(cycle_count as usize))
                }
                ChannelData::UInt48(_) => {
                    ChannelData::UInt48(Array1::<u64>::zeros(cycle_count as usize))
                }
                ChannelData::Int64(_) => {
                    ChannelData::Int64(Array1::<i64>::zeros(cycle_count as usize))
                }
                ChannelData::UInt64(_) => {
                    ChannelData::UInt64(Array1::<u64>::zeros(cycle_count as usize))
                }
                ChannelData::Float64(_) => {
                    ChannelData::Float64(Array1::<f64>::zeros(cycle_count as usize))
                }
                ChannelData::Complex16(_) => {
                    ChannelData::Complex16(Array1::<Complex<f32>>::zeros(cycle_count as usize))
                }
                ChannelData::Complex32(_) => {
                    ChannelData::Complex32(Array1::<Complex<f32>>::zeros(cycle_count as usize))
                }
                ChannelData::Complex64(_) => {
                    ChannelData::Complex64(Array1::<Complex<f64>>::zeros(cycle_count as usize))
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
                ChannelData::ByteArray(_) => {
                    ChannelData::ByteArray(vec![vec![0u8; n_bytes as usize]; cycle_count as usize])
                }
                ChannelData::ArrayDInt8(_) => {
                    ChannelData::ArrayDInt8(ArrayD::<i8>::zeros(IxDyn(&[
                        (cycle_count as usize) * n_elements
                    ])))
                }
                ChannelData::ArrayDUInt8(_) => {
                    ChannelData::ArrayDUInt8(ArrayD::<u8>::zeros(IxDyn(&[
                        (cycle_count as usize) * n_elements
                    ])))
                }
                ChannelData::ArrayDInt16(_) => {
                    ChannelData::ArrayDInt16(ArrayD::<i16>::zeros(IxDyn(&[
                        (cycle_count as usize) * n_elements
                    ])))
                }
                ChannelData::ArrayDUInt16(_) => {
                    ChannelData::ArrayDUInt16(ArrayD::<u16>::zeros(IxDyn(&[(cycle_count
                        as usize)
                        * n_elements])))
                }
                ChannelData::ArrayDFloat16(_) => {
                    ChannelData::ArrayDFloat16(ArrayD::<f32>::zeros(IxDyn(&[(cycle_count
                        as usize)
                        * n_elements])))
                }
                ChannelData::ArrayDInt24(_) => {
                    ChannelData::ArrayDInt24(ArrayD::<i32>::zeros(IxDyn(&[
                        (cycle_count as usize) * n_elements
                    ])))
                }
                ChannelData::ArrayDUInt24(_) => {
                    ChannelData::ArrayDUInt24(ArrayD::<u32>::zeros(IxDyn(&[(cycle_count
                        as usize)
                        * n_elements])))
                }
                ChannelData::ArrayDInt32(_) => {
                    ChannelData::ArrayDInt32(ArrayD::<i32>::zeros(IxDyn(&[
                        (cycle_count as usize) * n_elements
                    ])))
                }
                ChannelData::ArrayDUInt32(_) => {
                    ChannelData::ArrayDUInt32(ArrayD::<u32>::zeros(IxDyn(&[(cycle_count
                        as usize)
                        * n_elements])))
                }
                ChannelData::ArrayDFloat32(_) => {
                    ChannelData::ArrayDFloat32(ArrayD::<f32>::zeros(IxDyn(&[(cycle_count
                        as usize)
                        * n_elements])))
                }
                ChannelData::ArrayDInt48(_) => {
                    ChannelData::ArrayDInt48(ArrayD::<i64>::zeros(IxDyn(&[
                        (cycle_count as usize) * n_elements
                    ])))
                }
                ChannelData::ArrayDUInt48(_) => {
                    ChannelData::ArrayDUInt48(ArrayD::<u64>::zeros(IxDyn(&[(cycle_count
                        as usize)
                        * n_elements])))
                }
                ChannelData::ArrayDInt64(_) => {
                    ChannelData::ArrayDInt64(ArrayD::<i64>::zeros(IxDyn(&[
                        (cycle_count as usize) * n_elements
                    ])))
                }
                ChannelData::ArrayDUInt64(_) => {
                    ChannelData::ArrayDUInt64(ArrayD::<u64>::zeros(IxDyn(&[(cycle_count
                        as usize)
                        * n_elements])))
                }
                ChannelData::ArrayDFloat64(_) => {
                    ChannelData::ArrayDFloat64(ArrayD::<f64>::zeros(IxDyn(&[(cycle_count
                        as usize)
                        * n_elements])))
                }
                ChannelData::ArrayDComplex16(_) => {
                    ChannelData::ArrayDComplex16(ArrayD::<Complex<f32>>::zeros(IxDyn(&[
                        (cycle_count as usize) * n_elements,
                    ])))
                }
                ChannelData::ArrayDComplex32(_) => {
                    ChannelData::ArrayDComplex32(ArrayD::<Complex<f32>>::zeros(IxDyn(&[
                        (cycle_count as usize) * n_elements,
                    ])))
                }
                ChannelData::ArrayDComplex64(_) => {
                    ChannelData::ArrayDComplex64(ArrayD::<Complex<f64>>::zeros(IxDyn(&[
                        (cycle_count as usize) * n_elements,
                    ])))
                }
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
                    output = format!("[{}, .., {}]", array[0], array[array.len() - 1]);
                } else {
                    output = String::new();
                }
            }
            ChannelData::Complex32(array) => {
                if array.len() > 1 {
                    output = format!("[{}, .., {}]", array[0], array[array.len() - 1]);
                } else {
                    output = String::new();
                }
            }
            ChannelData::Complex64(array) => {
                if array.len() > 1 {
                    output = format!("[{}, .., {}]", array[0], array[array.len() - 1]);
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
            ChannelData::ByteArray(array) => {
                if array.len() > 1 {
                    output = format!("[{:?}, .., {:?}]", array[0], array[array.len() - 1]);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDInt8(array) => {
                if array.len() > 1 {
                    output = format!("dim {:?} {}", array.dim(), array);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDUInt8(array) => {
                if array.len() > 1 {
                    output = format!("dim {:?} {}", array.dim(), array);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDInt16(array) => {
                if array.len() > 1 {
                    output = format!("dim {:?} {}", array.dim(), array);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDUInt16(array) => {
                if array.len() > 1 {
                    output = format!("dim {:?} {}", array.dim(), array);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDFloat16(array) => {
                if array.len() > 1 {
                    output = format!("dim {:?} {}", array.dim(), array);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDInt24(array) => {
                if array.len() > 1 {
                    output = format!("dim {:?} {}", array.dim(), array);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDUInt24(array) => {
                if array.len() > 1 {
                    output = format!("dim {:?} {}", array.dim(), array);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDInt32(array) => {
                if array.len() > 1 {
                    output = format!("dim {:?} {}", array.dim(), array);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDUInt32(array) => {
                if array.len() > 1 {
                    output = format!("dim {:?} {}", array.dim(), array);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDFloat32(array) => {
                if array.len() > 1 {
                    output = format!("dim {:?} {}", array.dim(), array);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDInt48(array) => {
                if array.len() > 1 {
                    output = format!("dim {:?} {}", array.dim(), array);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDUInt48(array) => {
                if array.len() > 1 {
                    output = format!("dim {:?} {}", array.dim(), array);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDInt64(array) => {
                if array.len() > 1 {
                    output = format!("dim {:?} {}", array.dim(), array);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDUInt64(array) => {
                if array.len() > 1 {
                    output = format!("dim {:?} {}", array.dim(), array);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDFloat64(array) => {
                if array.len() > 1 {
                    output = format!("dim {:?} {}", array.dim(), array);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDComplex16(array) => {
                if array.len() > 1 {
                    output = format!("dim {:?} {}", array.dim(), array);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDComplex32(array) => {
                if array.len() > 1 {
                    output = format!("dim {:?} {}", array.dim(), array);
                } else {
                    output = String::new();
                }
            }
            ChannelData::ArrayDComplex64(array) => {
                if array.len() > 1 {
                    output = format!("dim {:?} {}", array.dim(), array);
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
            ChannelData::ByteArray(data) => data.is_empty(),
            ChannelData::ArrayDInt8(data) => data.is_empty(),
            ChannelData::ArrayDUInt8(data) => data.is_empty(),
            ChannelData::ArrayDInt16(data) => data.is_empty(),
            ChannelData::ArrayDUInt16(data) => data.is_empty(),
            ChannelData::ArrayDFloat16(data) => data.is_empty(),
            ChannelData::ArrayDInt24(data) => data.is_empty(),
            ChannelData::ArrayDUInt24(data) => data.is_empty(),
            ChannelData::ArrayDInt32(data) => data.is_empty(),
            ChannelData::ArrayDUInt32(data) => data.is_empty(),
            ChannelData::ArrayDFloat32(data) => data.is_empty(),
            ChannelData::ArrayDInt48(data) => data.is_empty(),
            ChannelData::ArrayDUInt48(data) => data.is_empty(),
            ChannelData::ArrayDInt64(data) => data.is_empty(),
            ChannelData::ArrayDUInt64(data) => data.is_empty(),
            ChannelData::ArrayDFloat64(data) => data.is_empty(),
            ChannelData::ArrayDComplex16(data) => data.is_empty(),
            ChannelData::ArrayDComplex32(data) => data.is_empty(),
            ChannelData::ArrayDComplex64(data) => data.is_empty(),
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
            ChannelData::ByteArray(data) => data.len(),
            ChannelData::ArrayDInt8(data) => data.len(),
            ChannelData::ArrayDUInt8(data) => data.len(),
            ChannelData::ArrayDInt16(data) => data.len(),
            ChannelData::ArrayDUInt16(data) => data.len(),
            ChannelData::ArrayDFloat16(data) => data.len(),
            ChannelData::ArrayDInt24(data) => data.len(),
            ChannelData::ArrayDUInt24(data) => data.len(),
            ChannelData::ArrayDInt32(data) => data.len(),
            ChannelData::ArrayDUInt32(data) => data.len(),
            ChannelData::ArrayDFloat32(data) => data.len(),
            ChannelData::ArrayDInt48(data) => data.len(),
            ChannelData::ArrayDUInt48(data) => data.len(),
            ChannelData::ArrayDInt64(data) => data.len(),
            ChannelData::ArrayDUInt64(data) => data.len(),
            ChannelData::ArrayDFloat64(data) => data.len(),
            ChannelData::ArrayDComplex16(data) => data.len(),
            ChannelData::ArrayDComplex32(data) => data.len(),
            ChannelData::ArrayDComplex64(data) => data.len(),
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
            ChannelData::ByteArray(data) => {
                data.par_iter().map(|s| s.len() as u32).max().unwrap_or(0) * 8
            }
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
            ChannelData::ByteArray(data) => {
                data.par_iter().map(|s| s.len() as u32).max().unwrap_or(0)
            }
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
                ChannelData::ByteArray(_) => 10,
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
                ChannelData::ByteArray(_) => 10,
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
                ChannelData::ByteArray(_) => false,
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
                ChannelData::ByteArray(_) => false,
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
                        && a.iter().zip(b.iter()).all(|(a, b)| {
                            ((a.re - b.re).abs() < epsilon) && ((a.im - b.im).abs() < epsilon)
                        })
                }
                ChannelData::Complex32(_) => false,
                ChannelData::Complex64(_) => false,
                ChannelData::StringSBC(_) => false,
                ChannelData::StringUTF8(_) => false,
                ChannelData::StringUTF16(_) => false,
                ChannelData::ByteArray(_) => false,
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
                        && a.iter().zip(b.iter()).all(|(a, b)| {
                            ((a.re - b.re).abs() < epsilon) && ((a.im - b.im).abs() < epsilon)
                        })
                }
                ChannelData::Complex64(_) => false,
                ChannelData::StringSBC(_) => false,
                ChannelData::StringUTF8(_) => false,
                ChannelData::StringUTF16(_) => false,
                ChannelData::ByteArray(_) => false,
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
            ChannelData::ByteArray(_) => false,
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
                ChannelData::ByteArray(_) => false,
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
                        && a.iter().zip(b.iter()).all(|(a, b)| {
                            ((a.re - b.re).abs() < epsilon) && ((a.im - b.im).abs() < epsilon)
                        })
                }
                ChannelData::StringSBC(_) => false,
                ChannelData::StringUTF8(_) => false,
                ChannelData::StringUTF16(_) => false,
                ChannelData::ByteArray(_) => false,
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
            ChannelData::ByteArray(_) => false,
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
            ChannelData::Complex16(a) => a
                .iter()
                .flat_map(|x| [x.re.to_ne_bytes(), x.im.to_ne_bytes()].concat())
                .collect(),
            ChannelData::Complex32(a) => a
                .iter()
                .flat_map(|x| [x.re.to_ne_bytes(), x.im.to_ne_bytes()].concat())
                .collect(),
            ChannelData::Complex64(a) => a
                .iter()
                .flat_map(|x| [x.re.to_ne_bytes(), x.im.to_ne_bytes()].concat())
                .collect(),
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
            ChannelData::ByteArray(a) => a.iter().flatten().cloned().collect::<Vec<u8>>(),
            ChannelData::ArrayDInt8(a) => a.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::ArrayDUInt8(a) => a.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::ArrayDInt16(a) => a.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::ArrayDUInt16(a) => a.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::ArrayDFloat16(a) => a.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::ArrayDInt24(a) => a.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::ArrayDUInt24(a) => a.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::ArrayDInt32(a) => a.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::ArrayDUInt32(a) => a.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::ArrayDFloat32(a) => a.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::ArrayDInt48(a) => a.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::ArrayDUInt48(a) => a.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::ArrayDInt64(a) => a.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::ArrayDUInt64(a) => a.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::ArrayDFloat64(a) => a.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            ChannelData::ArrayDComplex16(a) => a
                .iter()
                .flat_map(|x| [x.re.to_ne_bytes(), x.im.to_ne_bytes()].concat())
                .collect(),
            ChannelData::ArrayDComplex32(a) => a
                .iter()
                .flat_map(|x| [x.re.to_ne_bytes(), x.im.to_ne_bytes()].concat())
                .collect(),
            ChannelData::ArrayDComplex64(a) => a
                .iter()
                .flat_map(|x| [x.re.to_ne_bytes(), x.im.to_ne_bytes()].concat())
                .collect(),
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
            ChannelData::ByteArray(_) => 1,
            ChannelData::ArrayDInt8(a) => a.ndim(),
            ChannelData::ArrayDUInt8(a) => a.ndim(),
            ChannelData::ArrayDInt16(a) => a.ndim(),
            ChannelData::ArrayDUInt16(a) => a.ndim(),
            ChannelData::ArrayDFloat16(a) => a.ndim(),
            ChannelData::ArrayDInt24(a) => a.ndim(),
            ChannelData::ArrayDUInt24(a) => a.ndim(),
            ChannelData::ArrayDInt32(a) => a.ndim(),
            ChannelData::ArrayDUInt32(a) => a.ndim(),
            ChannelData::ArrayDFloat32(a) => a.ndim(),
            ChannelData::ArrayDInt48(a) => a.ndim(),
            ChannelData::ArrayDUInt48(a) => a.ndim(),
            ChannelData::ArrayDInt64(a) => a.ndim(),
            ChannelData::ArrayDUInt64(a) => a.ndim(),
            ChannelData::ArrayDFloat64(a) => a.ndim(),
            ChannelData::ArrayDComplex16(a) => a.ndim(),
            ChannelData::ArrayDComplex32(a) => a.ndim(),
            ChannelData::ArrayDComplex64(a) => a.ndim(),
        }
    }
    /// returns the shape of channel
    pub fn shape(&self) -> Vec<usize> {
        match self {
            ChannelData::Int8(a) => {
                vec![a.len(); 1]
            }
            ChannelData::UInt8(a) => vec![a.len(); 1],
            ChannelData::Int16(a) => vec![a.len(); 1],
            ChannelData::UInt16(a) => vec![a.len(); 1],
            ChannelData::Float16(a) => vec![a.len(); 1],
            ChannelData::Int24(a) => vec![a.len(); 1],
            ChannelData::UInt24(a) => vec![a.len(); 1],
            ChannelData::Int32(a) => vec![a.len(); 1],
            ChannelData::UInt32(a) => vec![a.len(); 1],
            ChannelData::Float32(a) => vec![a.len(); 1],
            ChannelData::Int48(a) => vec![a.len(); 1],
            ChannelData::UInt48(a) => vec![a.len(); 1],
            ChannelData::Int64(a) => vec![a.len(); 1],
            ChannelData::UInt64(a) => vec![a.len(); 1],
            ChannelData::Float64(a) => vec![a.len(); 1],
            ChannelData::Complex16(a) => vec![a.len(); 1],
            ChannelData::Complex32(a) => vec![a.len(); 1],
            ChannelData::Complex64(a) => vec![a.len(); 1],
            ChannelData::StringSBC(a) => vec![a.len(); 1],
            ChannelData::StringUTF8(a) => vec![a.len(); 1],
            ChannelData::StringUTF16(a) => vec![a.len(); 1],
            ChannelData::ByteArray(a) => vec![a.len(); 1],
            ChannelData::ArrayDInt8(a) => a.shape().to_owned(),
            ChannelData::ArrayDUInt8(a) => a.shape().to_owned(),
            ChannelData::ArrayDInt16(a) => a.shape().to_owned(),
            ChannelData::ArrayDUInt16(a) => a.shape().to_owned(),
            ChannelData::ArrayDFloat16(a) => a.shape().to_owned(),
            ChannelData::ArrayDInt24(a) => a.shape().to_owned(),
            ChannelData::ArrayDUInt24(a) => a.shape().to_owned(),
            ChannelData::ArrayDInt32(a) => a.shape().to_owned(),
            ChannelData::ArrayDUInt32(a) => a.shape().to_owned(),
            ChannelData::ArrayDFloat32(a) => a.shape().to_owned(),
            ChannelData::ArrayDInt48(a) => a.shape().to_owned(),
            ChannelData::ArrayDUInt48(a) => a.shape().to_owned(),
            ChannelData::ArrayDInt64(a) => a.shape().to_owned(),
            ChannelData::ArrayDUInt64(a) => a.shape().to_owned(),
            ChannelData::ArrayDFloat64(a) => a.shape().to_owned(),
            ChannelData::ArrayDComplex16(a) => a.shape().to_owned(),
            ChannelData::ArrayDComplex32(a) => a.shape().to_owned(),
            ChannelData::ArrayDComplex64(a) => a.shape().to_owned(),
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
                    .reduce(|accum, item| if accum >= item { accum } else { item })
                    .cloned();
                let min = a
                    .iter()
                    .reduce(|accum, item| if accum <= item { accum } else { item })
                    .cloned();
                (min, max)
            }
            ChannelData::Complex16(_) => (None, None),
            ChannelData::Complex32(_) => (None, None),
            ChannelData::Complex64(_) => (None, None),
            ChannelData::StringSBC(_) => (None, None),
            ChannelData::StringUTF8(_) => (None, None),
            ChannelData::StringUTF16(_) => (None, None),
            ChannelData::ByteArray(_) => (None, None),
            ChannelData::ArrayDInt8(a) => {
                let min = a.iter().min().map(|v| *v as f64);
                let max = a.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDUInt8(a) => {
                let min = a.iter().min().map(|v| *v as f64);
                let max = a.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDInt16(a) => {
                let min = a.iter().min().map(|v| *v as f64);
                let max = a.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDUInt16(a) => {
                let min = a.iter().min().map(|v| *v as f64);
                let max = a.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDFloat16(a) => {
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
            ChannelData::ArrayDInt24(a) => {
                let min = a.iter().min().map(|v| *v as f64);
                let max = a.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDUInt24(a) => {
                let min = a.iter().min().map(|v| *v as f64);
                let max = a.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDInt32(a) => {
                let min = a.iter().min().map(|v| *v as f64);
                let max = a.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDUInt32(a) => {
                let min = a.iter().min().map(|v| *v as f64);
                let max = a.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDFloat32(a) => {
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
            ChannelData::ArrayDInt48(a) => {
                let min = a.iter().min().map(|v| *v as f64);
                let max = a.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDUInt48(a) => {
                let min = a.iter().min().map(|v| *v as f64);
                let max = a.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDInt64(a) => {
                let min = a.iter().min().map(|v| *v as f64);
                let max = a.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDUInt64(a) => {
                let min = a.iter().min().map(|v| *v as f64);
                let max = a.iter().max().map(|v| *v as f64);
                (min, max)
            }
            ChannelData::ArrayDFloat64(a) => {
                let max = a
                    .iter()
                    .reduce(|accum, item| if accum >= item { accum } else { item })
                    .cloned();
                let min = a
                    .iter()
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

/// IntoPy implementation to convert a ChannelData into a PyObject
impl IntoPy<PyObject> for ChannelData {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            ChannelData::Int8(array) => array.into_pyarray(py).into_py(py),
            ChannelData::UInt8(array) => array.into_pyarray(py).into_py(py),
            ChannelData::Int16(array) => array.into_pyarray(py).into_py(py),
            ChannelData::UInt16(array) => array.into_pyarray(py).into_py(py),
            ChannelData::Float16(array) => array.into_pyarray(py).into_py(py),
            ChannelData::Int24(array) => array.into_pyarray(py).into_py(py),
            ChannelData::UInt24(array) => array.into_pyarray(py).into_py(py),
            ChannelData::Int32(array) => array.into_pyarray(py).into_py(py),
            ChannelData::UInt32(array) => array.into_pyarray(py).into_py(py),
            ChannelData::Float32(array) => array.into_pyarray(py).into_py(py),
            ChannelData::Int48(array) => array.into_pyarray(py).into_py(py),
            ChannelData::UInt48(array) => array.into_pyarray(py).into_py(py),
            ChannelData::Int64(array) => array.into_pyarray(py).into_py(py),
            ChannelData::UInt64(array) => array.into_pyarray(py).into_py(py),
            ChannelData::Float64(array) => array.into_pyarray(py).into_py(py),
            ChannelData::Complex16(array) => array.into_pyarray(py).into_py(py),
            ChannelData::Complex32(array) => array.into_pyarray(py).into_py(py),
            ChannelData::Complex64(array) => array.into_pyarray(py).into_py(py),
            ChannelData::StringSBC(array) => array.into_py(py),
            ChannelData::StringUTF8(array) => array.into_py(py),
            ChannelData::StringUTF16(array) => array.into_py(py),
            ChannelData::ByteArray(array) => array.into_py(py),
            ChannelData::ArrayDInt8(array) => array.into_pyarray(py).into_py(py),
            ChannelData::ArrayDUInt8(array) => array.into_pyarray(py).into_py(py),
            ChannelData::ArrayDInt16(array) => array.into_pyarray(py).into_py(py),
            ChannelData::ArrayDUInt16(array) => array.into_pyarray(py).into_py(py),
            ChannelData::ArrayDFloat16(array) => array.into_pyarray(py).into_py(py),
            ChannelData::ArrayDInt24(array) => array.into_pyarray(py).into_py(py),
            ChannelData::ArrayDUInt24(array) => array.into_pyarray(py).into_py(py),
            ChannelData::ArrayDInt32(array) => array.into_pyarray(py).into_py(py),
            ChannelData::ArrayDUInt32(array) => array.into_pyarray(py).into_py(py),
            ChannelData::ArrayDFloat32(array) => array.into_pyarray(py).into_py(py),
            ChannelData::ArrayDInt48(array) => array.into_pyarray(py).into_py(py),
            ChannelData::ArrayDUInt48(array) => array.into_pyarray(py).into_py(py),
            ChannelData::ArrayDInt64(array) => array.into_pyarray(py).into_py(py),
            ChannelData::ArrayDUInt64(array) => array.into_pyarray(py).into_py(py),
            ChannelData::ArrayDFloat64(array) => array.into_pyarray(py).into_py(py),
            ChannelData::ArrayDComplex16(array) => array.into_pyarray(py).into_py(py),
            ChannelData::ArrayDComplex32(array) => array.into_pyarray(py).into_py(py),
            ChannelData::ArrayDComplex64(array) => array.into_pyarray(py).into_py(py),
        }
    }
}

/// ToPyObject implementation to convert a ChannelData into a PyObject
impl ToPyObject for ChannelData {
    fn to_object(&self, py: Python) -> PyObject {
        match self {
            ChannelData::Int8(array) => array.to_pyarray(py).into_py(py),
            ChannelData::UInt8(array) => array.to_pyarray(py).into_py(py),
            ChannelData::Int16(array) => array.to_pyarray(py).into_py(py),
            ChannelData::UInt16(array) => array.to_pyarray(py).into_py(py),
            ChannelData::Float16(array) => array.to_pyarray(py).into_py(py),
            ChannelData::Int24(array) => array.to_pyarray(py).into_py(py),
            ChannelData::UInt24(array) => array.to_pyarray(py).into_py(py),
            ChannelData::Int32(array) => array.to_pyarray(py).into_py(py),
            ChannelData::UInt32(array) => array.to_pyarray(py).into_py(py),
            ChannelData::Float32(array) => array.to_pyarray(py).into_py(py),
            ChannelData::Int48(array) => array.to_pyarray(py).into_py(py),
            ChannelData::UInt48(array) => array.to_pyarray(py).into_py(py),
            ChannelData::Int64(array) => array.to_pyarray(py).into_py(py),
            ChannelData::UInt64(array) => array.to_pyarray(py).into_py(py),
            ChannelData::Float64(array) => array.to_pyarray(py).into_py(py),
            ChannelData::Complex16(array) => array.to_pyarray(py).into_py(py),
            ChannelData::Complex32(array) => array.to_pyarray(py).into_py(py),
            ChannelData::Complex64(array) => array.to_pyarray(py).into_py(py),
            ChannelData::StringSBC(array) => array.to_object(py),
            ChannelData::StringUTF8(array) => array.to_object(py),
            ChannelData::StringUTF16(array) => array.to_object(py),
            ChannelData::ByteArray(array) => array.to_object(py),
            ChannelData::ArrayDInt8(array) => array.to_pyarray(py).into_py(py),
            ChannelData::ArrayDUInt8(array) => array.to_pyarray(py).into_py(py),
            ChannelData::ArrayDInt16(array) => array.to_pyarray(py).into_py(py),
            ChannelData::ArrayDUInt16(array) => array.to_pyarray(py).into_py(py),
            ChannelData::ArrayDFloat16(array) => array.to_pyarray(py).into_py(py),
            ChannelData::ArrayDInt24(array) => array.to_pyarray(py).into_py(py),
            ChannelData::ArrayDUInt24(array) => array.to_pyarray(py).into_py(py),
            ChannelData::ArrayDInt32(array) => array.to_pyarray(py).into_py(py),
            ChannelData::ArrayDUInt32(array) => array.to_pyarray(py).into_py(py),
            ChannelData::ArrayDFloat32(array) => array.to_pyarray(py).into_py(py),
            ChannelData::ArrayDInt48(array) => array.to_pyarray(py).into_py(py),
            ChannelData::ArrayDUInt48(array) => array.to_pyarray(py).into_py(py),
            ChannelData::ArrayDInt64(array) => array.to_pyarray(py).into_py(py),
            ChannelData::ArrayDUInt64(array) => array.to_pyarray(py).into_py(py),
            ChannelData::ArrayDFloat64(array) => array.to_pyarray(py).into_py(py),
            ChannelData::ArrayDComplex16(array) => array.to_pyarray(py).into_py(py),
            ChannelData::ArrayDComplex32(array) => array.to_pyarray(py).into_py(py),
            ChannelData::ArrayDComplex64(array) => array.to_pyarray(py).into_py(py),
        }
    }
}

/// FromPyObject implementation to allow conversion from a Python object to a ChannelData
impl FromPyObject<'_> for ChannelData {
    fn extract(ob: &'_ PyAny) -> PyResult<Self> {
        let truc: NumpyArray = ob.extract()?;
        match truc {
            NumpyArray::Int8(array) => {
                Ok(ChannelData::Int8(array.readonly().as_array().to_owned()))
            }
            NumpyArray::UInt8(array) => {
                Ok(ChannelData::UInt8(array.readonly().as_array().to_owned()))
            }
            NumpyArray::Int16(array) => {
                Ok(ChannelData::Int16(array.readonly().as_array().to_owned()))
            }
            NumpyArray::UInt16(array) => {
                Ok(ChannelData::UInt16(array.readonly().as_array().to_owned()))
            }
            NumpyArray::Int32(array) => {
                Ok(ChannelData::Int32(array.readonly().as_array().to_owned()))
            }
            NumpyArray::UInt32(array) => {
                Ok(ChannelData::UInt32(array.readonly().as_array().to_owned()))
            }
            NumpyArray::Float32(array) => {
                Ok(ChannelData::Float32(array.readonly().as_array().to_owned()))
            }
            NumpyArray::Int64(array) => {
                Ok(ChannelData::Int64(array.readonly().as_array().to_owned()))
            }
            NumpyArray::UInt64(array) => {
                Ok(ChannelData::UInt64(array.readonly().as_array().to_owned()))
            }
            NumpyArray::Float64(array) => {
                Ok(ChannelData::Float64(array.readonly().as_array().to_owned()))
            }
            NumpyArray::Complex32(array) => Ok(ChannelData::Complex32(
                array.readonly().as_array().to_owned(),
            )),
            NumpyArray::Complex64(array) => Ok(ChannelData::Complex64(
                array.readonly().as_array().to_owned(),
            )),
            NumpyArray::ArrayDInt8(array) => Ok(ChannelData::ArrayDInt8(
                array.readonly().as_array().to_owned(),
            )),
            NumpyArray::ArrayDUInt8(array) => Ok(ChannelData::ArrayDUInt8(
                array.readonly().as_array().to_owned(),
            )),
            NumpyArray::ArrayDInt16(array) => Ok(ChannelData::ArrayDInt16(
                array.readonly().as_array().to_owned(),
            )),
            NumpyArray::ArrayDUInt16(array) => Ok(ChannelData::ArrayDUInt16(
                array.readonly().as_array().to_owned(),
            )),
            NumpyArray::ArrayDInt32(array) => Ok(ChannelData::ArrayDInt32(
                array.readonly().as_array().to_owned(),
            )),
            NumpyArray::ArrayDUInt32(array) => Ok(ChannelData::ArrayDUInt32(
                array.readonly().as_array().to_owned(),
            )),
            NumpyArray::ArrayDFloat32(array) => Ok(ChannelData::ArrayDFloat32(
                array.readonly().as_array().to_owned(),
            )),
            NumpyArray::ArrayDInt64(array) => Ok(ChannelData::ArrayDInt64(
                array.readonly().as_array().to_owned(),
            )),
            NumpyArray::ArrayDUInt64(array) => Ok(ChannelData::ArrayDUInt64(
                array.readonly().as_array().to_owned(),
            )),
            NumpyArray::ArrayDFloat64(array) => Ok(ChannelData::ArrayDFloat64(
                array.readonly().as_array().to_owned(),
            )),
            NumpyArray::ArrayDComplex32(array) => Ok(ChannelData::ArrayDComplex32(
                array.readonly().as_array().to_owned(),
            )),
            NumpyArray::ArrayDComplex64(array) => Ok(ChannelData::ArrayDComplex64(
                array.readonly().as_array().to_owned(),
            )),
        }
    }
}

/// Enum to identify the dtype of numpy array
#[derive(Clone, FromPyObject)]
enum NumpyArray<'a> {
    Int8(&'a PyArray1<i8>),
    UInt8(&'a PyArray1<u8>),
    Int16(&'a PyArray1<i16>),
    UInt16(&'a PyArray1<u16>),
    Int32(&'a PyArray1<i32>),
    UInt32(&'a PyArray1<u32>),
    Float32(&'a PyArray1<f32>),
    Int64(&'a PyArray1<i64>),
    UInt64(&'a PyArray1<u64>),
    Float64(&'a PyArray1<f64>),
    Complex32(&'a PyArray1<Complex<f32>>),
    Complex64(&'a PyArray1<Complex<f64>>),
    ArrayDInt8(&'a PyArrayDyn<i8>),
    ArrayDUInt8(&'a PyArrayDyn<u8>),
    ArrayDInt16(&'a PyArrayDyn<i16>),
    ArrayDUInt16(&'a PyArrayDyn<u16>),
    ArrayDInt32(&'a PyArrayDyn<i32>),
    ArrayDUInt32(&'a PyArrayDyn<u32>),
    ArrayDFloat32(&'a PyArrayDyn<f32>),
    ArrayDInt64(&'a PyArrayDyn<i64>),
    ArrayDUInt64(&'a PyArrayDyn<u64>),
    ArrayDFloat64(&'a PyArrayDyn<f64>),
    ArrayDComplex32(&'a PyArrayDyn<Complex<f32>>),
    ArrayDComplex64(&'a PyArrayDyn<Complex<f64>>),
}

impl Default for ChannelData {
    fn default() -> Self {
        ChannelData::UInt8(Array1::<u8>::zeros(0))
    }
}

/// Initialises a channel array type depending of cn_type, cn_data_type and if array
pub fn data_type_init(cn_type: u8, cn_data_type: u8, n_bytes: u32, is_array: bool) -> ChannelData {
    let data_type: ChannelData;
    if !is_array {
        if cn_type != 3 || cn_type != 6 {
            if cn_data_type == 0 || cn_data_type == 1 {
                // unsigned int
                if n_bytes <= 1 {
                    data_type = ChannelData::UInt8(Array1::<u8>::zeros(0));
                } else if n_bytes == 2 {
                    data_type = ChannelData::UInt16(Array1::<u16>::zeros(0));
                } else if n_bytes == 3 {
                    data_type = ChannelData::UInt24(Array1::<u32>::zeros(0));
                } else if n_bytes == 4 {
                    data_type = ChannelData::UInt32(Array1::<u32>::zeros(0));
                } else if n_bytes <= 6 {
                    data_type = ChannelData::UInt48(Array1::<u64>::zeros(0));
                } else {
                    data_type = ChannelData::UInt64(Array1::<u64>::zeros(0));
                }
            } else if cn_data_type == 2 || cn_data_type == 3 {
                // signed int
                if n_bytes <= 1 {
                    data_type = ChannelData::Int8(Array1::<i8>::zeros(0));
                } else if n_bytes == 2 {
                    data_type = ChannelData::Int16(Array1::<i16>::zeros(0));
                } else if n_bytes == 3 {
                    data_type = ChannelData::Int24(Array1::<i32>::zeros(0));
                } else if n_bytes == 4 {
                    data_type = ChannelData::Int32(Array1::<i32>::zeros(0));
                } else if n_bytes <= 6 {
                    data_type = ChannelData::Int48(Array1::<i64>::zeros(0));
                } else {
                    data_type = ChannelData::Int64(Array1::<i64>::zeros(0));
                }
            } else if cn_data_type == 4 || cn_data_type == 5 {
                // float
                if n_bytes <= 2 {
                    data_type = ChannelData::Float16(Array1::<f32>::zeros(0));
                } else if n_bytes <= 4 {
                    data_type = ChannelData::Float32(Array1::<f32>::zeros(0));
                } else {
                    data_type = ChannelData::Float64(Array1::<f64>::zeros(0));
                }
            } else if cn_data_type == 15 || cn_data_type == 16 {
                // complex
                if n_bytes <= 2 {
                    data_type = ChannelData::Complex16(Array1::<Complex<f32>>::zeros(0));
                } else if n_bytes <= 4 {
                    data_type = ChannelData::Complex32(Array1::<Complex<f32>>::zeros(0));
                } else {
                    data_type = ChannelData::Complex64(Array1::<Complex<f64>>::zeros(0));
                }
            } else if cn_data_type == 6 {
                // SBC ISO-8859-1 to be converted into UTF8
                data_type = ChannelData::StringSBC(vec![String::new(); 0]);
            } else if cn_data_type == 7 {
                // String UTF8
                data_type = ChannelData::StringUTF8(vec![String::new(); 0]);
            } else if cn_data_type == 8 || cn_data_type == 9 {
                // String UTF16 to be converted into UTF8
                data_type = ChannelData::StringUTF16(vec![String::new(); 0]);
            } else {
                // bytearray
                data_type = ChannelData::ByteArray(vec![vec![0u8; 0]; 0]);
            }
        } else {
            // virtual channels, cn_bit_count = 0 -> n_bytes = 0, must be LE unsigned int
            data_type = ChannelData::UInt64(Array1::<u64>::zeros(0));
        }
    } else if cn_type != 3 || cn_type != 6 {
        if cn_data_type == 0 || cn_data_type == 1 {
            // unsigned int
            if n_bytes <= 1 {
                data_type = ChannelData::ArrayDUInt8(ArrayD::<u8>::zeros(IxDyn(&[0])));
            } else if n_bytes == 2 {
                data_type = ChannelData::ArrayDUInt16(ArrayD::<u16>::zeros(IxDyn(&[0])));
            } else if n_bytes == 3 {
                data_type = ChannelData::ArrayDUInt24(ArrayD::<u32>::zeros(IxDyn(&[0])));
            } else if n_bytes == 4 {
                data_type = ChannelData::ArrayDUInt32(ArrayD::<u32>::zeros(IxDyn(&[0])));
            } else if n_bytes <= 6 {
                data_type = ChannelData::ArrayDUInt48(ArrayD::<u64>::zeros(IxDyn(&[0])));
            } else {
                data_type = ChannelData::ArrayDUInt64(ArrayD::<u64>::zeros(IxDyn(&[0])));
            }
        } else if cn_data_type == 2 || cn_data_type == 3 {
            // signed int
            if n_bytes <= 1 {
                data_type = ChannelData::ArrayDInt8(ArrayD::<i8>::zeros(IxDyn(&[0])));
            } else if n_bytes == 2 {
                data_type = ChannelData::ArrayDInt16(ArrayD::<i16>::zeros(IxDyn(&[0])));
            } else if n_bytes == 3 {
                data_type = ChannelData::ArrayDInt24(ArrayD::<i32>::zeros(IxDyn(&[0])));
            } else if n_bytes == 4 {
                data_type = ChannelData::ArrayDInt32(ArrayD::<i32>::zeros(IxDyn(&[0])));
            } else if n_bytes <= 6 {
                data_type = ChannelData::ArrayDInt48(ArrayD::<i64>::zeros(IxDyn(&[0])));
            } else {
                data_type = ChannelData::ArrayDInt64(ArrayD::<i64>::zeros(IxDyn(&[0])));
            }
        } else if cn_data_type == 4 || cn_data_type == 5 {
            // float
            if n_bytes <= 2 {
                data_type = ChannelData::ArrayDFloat16(ArrayD::<f32>::zeros(IxDyn(&[0])));
            } else if n_bytes <= 4 {
                data_type = ChannelData::ArrayDFloat32(ArrayD::<f32>::zeros(IxDyn(&[0])));
            } else {
                data_type = ChannelData::ArrayDFloat64(ArrayD::<f64>::zeros(IxDyn(&[0])));
            }
        } else if cn_data_type == 15 || cn_data_type == 16 {
            // complex
            if n_bytes <= 2 {
                data_type =
                    ChannelData::ArrayDComplex16(ArrayD::<Complex<f32>>::zeros(IxDyn(&[0])));
            } else if n_bytes <= 4 {
                data_type =
                    ChannelData::ArrayDComplex32(ArrayD::<Complex<f32>>::zeros(IxDyn(&[0])));
            } else {
                data_type =
                    ChannelData::ArrayDComplex64(ArrayD::<Complex<f64>>::zeros(IxDyn(&[0])));
            }
        } else {
            // strings or bytes arrays not implemented
            todo!();
        }
    } else {
        // virtual channels arrays not implemented, can it even exists ?
        todo!();
    }
    data_type
}

impl fmt::Display for ChannelData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChannelData::Int8(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::UInt8(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::Int16(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::UInt16(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::Float16(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::Int24(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::UInt24(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::Int32(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::UInt32(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::Float32(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::Int48(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::UInt48(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::Int64(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::UInt64(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::Float64(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::Complex16(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::Complex32(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::Complex64(array) => {
                writeln!(f, "{}", array)
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
            ChannelData::ByteArray(array) => {
                for text in array.iter() {
                    writeln!(f, " {:?} ", text)?;
                }
                writeln!(f, " ")
            }
            ChannelData::ArrayDInt8(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::ArrayDUInt8(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::ArrayDInt16(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::ArrayDUInt16(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::ArrayDFloat16(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::ArrayDInt24(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::ArrayDUInt24(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::ArrayDInt32(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::ArrayDUInt32(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::ArrayDFloat32(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::ArrayDInt48(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::ArrayDUInt48(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::ArrayDInt64(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::ArrayDUInt64(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::ArrayDFloat64(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::ArrayDComplex16(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::ArrayDComplex32(array) => {
                writeln!(f, "{}", array)
            }
            ChannelData::ArrayDComplex64(array) => {
                writeln!(f, "{}", array)
            }
        }
    }
}
