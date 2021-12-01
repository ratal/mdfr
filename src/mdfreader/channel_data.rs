//! this module holds the channel data enum and related implementations
use ndarray::{Array1, ArrayD, IxDyn};
use num::Complex;
use numpy::{IntoPyArray, ToPyArray};
use pyo3::prelude::*;
use std::fmt;

/// channel data type enum.
/// most common data type is 1D ndarray for timeseries with element types numeric.
/// vector of string or bytes also exists.
/// Dynamic dimension arrays ArrayD are also existing to cover CABlock arrays data.
#[derive(Debug, Clone, PartialEq)]
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
                    let mut target_vect: Vec<String> =
                        vec![String::with_capacity(n_bytes as usize); cycle_count as usize];
                    for i in 0..cycle_count as usize {
                        target_vect[i].reserve(n_bytes as usize);
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
                    let mut target_vect: Vec<String> =
                        vec![String::with_capacity(n_bytes as usize); cycle_count as usize];
                    for i in 0..cycle_count as usize {
                        target_vect[i].reserve(n_bytes as usize);
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
    // Compares floating point values
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
    pub fn compare_f64(&self, other: &ChannelData, epsilon: f64) -> bool {
        match self {
            ChannelData::Int8(_) => false,
            ChannelData::UInt8(_) => false,
            ChannelData::Int16(_) => false,
            ChannelData::UInt16(_) => false,
            ChannelData::Float16(a) => false,
            ChannelData::Int24(_) => false,
            ChannelData::UInt24(_) => false,
            ChannelData::Int32(_) => false,
            ChannelData::UInt32(_) => false,
            ChannelData::Float32(a) => false,
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
            ChannelData::Complex16(a) => false,
            ChannelData::Complex32(a) => false,
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
}

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
    } else {
        if cn_type != 3 || cn_type != 6 {
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
