//! this module provides methods to get directly channelData into python

use arrow::array::{
    Float32Builder, Float64Builder, Int16Builder, Int32Builder, Int64Builder, Int8Builder,
    UInt16Builder, UInt32Builder, UInt64Builder, UInt8Builder, 
};

use numpy::npyffi::types::NPY_ORDER;
use numpy::{PyArray1, PyArrayDyn, ToPyArray};
use pyo3::prelude::*;
use pyo3::{PyAny, PyObject, PyResult};

use crate::mdfreader::channel_data::{ChannelData, Order};

impl From<Order> for NPY_ORDER {
    fn from(order: Order) -> Self {
        match order {
            Order::RowMajor => NPY_ORDER::NPY_CORDER,
            Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
        }
    }
}

impl IntoPy<PyObject> for ChannelData {
    /// IntoPy implementation to convert a ChannelData into a PyObject
    fn into_py(self, py: Python) -> PyObject {
        match self {
            ChannelData::Int8(array) => array.values_slice().to_pyarray(py).into_py(py),
            ChannelData::UInt8(array) => array.values_slice().to_pyarray(py).into_py(py),
            ChannelData::Int16(array) => array.values_slice().to_pyarray(py).into_py(py),
            ChannelData::UInt16(array) => array.values_slice().to_pyarray(py).into_py(py),
            ChannelData::Int32(array) => array.values_slice().to_pyarray(py).into_py(py),
            ChannelData::UInt32(array) => array.values_slice().to_pyarray(py).into_py(py),
            ChannelData::Float32(array) => array.values_slice().to_pyarray(py).into_py(py),
            ChannelData::Int64(array) => array.values_slice().to_pyarray(py).into_py(py),
            ChannelData::UInt64(array) => array.values_slice().to_pyarray(py).into_py(py),
            ChannelData::Float64(array) => array.values_slice().to_pyarray(py).into_py(py),
            ChannelData::Complex32(array) => {
                let a = array.finish_cloned().values();
                a.as_any().downcast_ref::<Float32Builder>().expect("Failed downcasting to Float32 primitive").values_slice().to_pyarray(py).into_py(py)
            }
            ChannelData::Complex64(array) => {
                let a = array.finish_cloned().values();
                a.as_any().downcast_ref::<Float64Builder>().expect("Failed downcasting to Float64 primitive").values_slice().to_pyarray(py).into_py(py)
            }
            ChannelData::VariableSizeByteArray(array) => array.values_slice().into_py(py),
            ChannelData::FixedSizeByteArray(array) => {
                let binary_array = array.finish_cloned();
                let out: Vec<Vec<u8>> = binary_array
                    .values()
                    .chunks(binary_array.value_length() as usize)
                    .map(|x| x.to_vec())
                    .collect();
                out.into_py(py)
            }
            ChannelData::ArrayDInt8(array) => array
                .0
                .values_slice()
                .to_pyarray(py)
                .reshape_with_order(array.1 .0.clone(), array.1 .1.clone().into())
                .expect("could not reshape i8")
                .into_py(py),
            ChannelData::ArrayDUInt8(array) => array
                .0
                .values_slice()
                .to_pyarray(py)
                .reshape_with_order(array.1 .0.clone(), array.1 .1.clone().into())
                .expect("could not reshape u8")
                .into_py(py),
            ChannelData::ArrayDInt16(array) => array
                .0
                .values_slice()
                .to_pyarray(py)
                .reshape_with_order(array.1 .0.clone(), array.1 .1.clone().into())
                .expect("could not reshape u16")
                .into_py(py),
            ChannelData::ArrayDUInt16(array) => array
                .0
                .values_slice()
                .to_pyarray(py)
                .reshape_with_order(array.1 .0.clone(), array.1 .1.clone().into())
                .expect("could not reshape i16")
                .into_py(py),
            ChannelData::ArrayDInt32(array) => array
                .0
                .values_slice()
                .to_pyarray(py)
                .reshape_with_order(array.1 .0.clone(), array.1 .1.clone().into())
                .expect("could not reshape i32")
                .into_py(py),
            ChannelData::ArrayDUInt32(array) => array
                .0
                .values_slice()
                .to_pyarray(py)
                .reshape_with_order(array.1 .0.clone(), array.1 .1.clone().into())
                .expect("could not reshape u32")
                .into_py(py),
            ChannelData::ArrayDFloat32(array) => array
                .0
                .values_slice()
                .to_pyarray(py)
                .reshape_with_order(array.1 .0.clone(), array.1 .1.clone().into())
                .expect("could not reshape f32")
                .into_py(py),
            ChannelData::ArrayDInt64(array) => array
                .0
                .values_slice()
                .to_pyarray(py)
                .reshape_with_order(array.1 .0.clone(), array.1 .1.clone().into())
                .expect("could not reshape i64")
                .into_py(py),
            ChannelData::ArrayDUInt64(array) => array
                .0
                .values_slice()
                .to_pyarray(py)
                .reshape_with_order(array.1 .0.clone(), array.1 .1.clone().into())
                .expect("could not reshape u64")
                .into_py(py),
            ChannelData::ArrayDFloat64(array) => array
                .0
                .values_slice()
                .to_pyarray(py)
                .reshape_with_order(array.1 .0.clone(), array.1 .1.clone().into())
                .expect("could not reshape f64")
                .into_py(py),
            ChannelData::Utf8(array) => array
                .finish_cloned()
                .iter()
                .collect::<Option<String>>()
                .into_py(py),
        }
    }
}

impl ToPyObject for ChannelData {
    /// ToPyObject implementation to convert a ChannelData into a PyObject
    fn to_object(&self, py: Python) -> PyObject {
        match self {
            ChannelData::Int8(array) => array.values_slice().to_pyarray(py).into_py(py),
            ChannelData::UInt8(array) => array.values_slice().to_pyarray(py).into_py(py),
            ChannelData::Int16(array) => array.values_slice().to_pyarray(py).into_py(py),
            ChannelData::UInt16(array) => array.values_slice().to_pyarray(py).into_py(py),
            ChannelData::Int32(array) => array.values_slice().to_pyarray(py).into_py(py),
            ChannelData::UInt32(array) => array.values_slice().to_pyarray(py).into_py(py),
            ChannelData::Float32(array) => array.values_slice().to_pyarray(py).into_py(py),
            ChannelData::Int64(array) => array.values_slice().to_pyarray(py).into_py(py),
            ChannelData::UInt64(array) => array.values_slice().to_pyarray(py).into_py(py),
            ChannelData::Float64(array) => array.values_slice().to_pyarray(py).into_py(py),
            ChannelData::Complex32(array) => {
                let a = array.finish_cloned().values();
                a.as_any().downcast_ref::<Float32Builder>().expect("Failed downcasting to Float32 primitive").values_slice().to_pyarray(py).into_py(py)
            }
            ChannelData::Complex64(array) => {
                let a = array.finish_cloned().values();
                a.as_any().downcast_ref::<Float64Builder>().expect("Failed downcasting to Float64 primitive").values_slice().to_pyarray(py).into_py(py)
            },
            ChannelData::Utf8(array) => array
                .finish_cloned()
                .iter()
                .collect::<Option<String>>()
                .to_object(py),
            ChannelData::VariableSizeByteArray(array) => array
                .finish_cloned()
                .iter()
                .map(|x| x.unwrap_or_default().to_vec())
                .collect::<Vec<Vec<u8>>>()
                .to_object(py),
            ChannelData::FixedSizeByteArray(array) => {
                let binary_array = array.finish_cloned();
                let out: Vec<Vec<u8>> = binary_array
                    .values()
                    .chunks(binary_array.value_length() as usize)
                    .map(|x| x.to_vec())
                    .collect();
                out.to_object(py)
            }
            ChannelData::ArrayDInt8(array) => array.0.values_slice().to_pyarray(py).into_py(py),
            ChannelData::ArrayDUInt8(array) => array.0.values_slice().to_pyarray(py).into_py(py),
            ChannelData::ArrayDInt16(array) => array.0.values_slice().to_pyarray(py).into_py(py),
            ChannelData::ArrayDUInt16(array) => array.0.values_slice().to_pyarray(py).into_py(py),
            ChannelData::ArrayDInt32(array) => array.0.values_slice().to_pyarray(py).into_py(py),
            ChannelData::ArrayDUInt32(array) => array.0.values_slice().to_pyarray(py).into_py(py),
            ChannelData::ArrayDFloat32(array) => array.0.values_slice().to_pyarray(py).into_py(py),
            ChannelData::ArrayDInt64(array) => array.0.values_slice().to_pyarray(py).into_py(py),
            ChannelData::ArrayDUInt64(array) => array.0.values_slice().to_pyarray(py).into_py(py),
            ChannelData::ArrayDFloat64(array) => array.0.values_slice().to_pyarray(py).into_py(py),
        }
    }
}

impl FromPyObject<'_> for ChannelData {
    /// FromPyObject implementation to allow conversion from a Python object to a ChannelData
    fn extract(ob: &'_ PyAny) -> PyResult<Self> {
        let truc: NumpyArray = ob.extract()?;
        match truc {
            NumpyArray::Boolean(array) => Ok(ChannelData::UInt8(UInt8Builder::new_from_buffer(
                array.readonly().as_array().to_owned().to_vec().into(),
                None,
            ))),
            NumpyArray::Int8(array) => Ok(ChannelData::Int8(Int8Builder::new_from_buffer(
                array.readonly().as_array().to_owned().to_vec().into(),
                None,
            ))),
            NumpyArray::UInt8(array) => Ok(ChannelData::UInt8(UInt8Builder::new_from_buffer(
                array.readonly().as_array().to_owned().to_vec().into(),
                None,
            ))),
            NumpyArray::Int16(array) => Ok(ChannelData::Int16(Int16Builder::new_from_buffer(
                array.readonly().as_array().to_owned().to_vec().into(),
                None,
            ))),
            NumpyArray::UInt16(array) => Ok(ChannelData::UInt16(UInt16Builder::new_from_buffer(
                array.readonly().as_array().to_owned().to_vec().into(),
                None,
            ))),
            NumpyArray::Int32(array) => Ok(ChannelData::Int32(Int32Builder::new_from_buffer(
                array.readonly().as_array().to_owned().to_vec().into(),
                None,
            ))),
            NumpyArray::UInt32(array) => Ok(ChannelData::UInt32(UInt32Builder::new_from_buffer(
                array.readonly().as_array().to_owned().to_vec().into(),
                None,
            ))),
            NumpyArray::Float32(array) => {
                Ok(ChannelData::Float32(Float32Builder::new_from_buffer(
                    array.readonly().as_array().to_owned().to_vec().into(),
                    None,
                )))
            }
            NumpyArray::Int64(array) => Ok(ChannelData::Int64(Int64Builder::new_from_buffer(
                array.readonly().as_array().to_owned().to_vec().into(),
                None,
            ))),
            NumpyArray::UInt64(array) => Ok(ChannelData::UInt64(UInt64Builder::new_from_buffer(
                array.readonly().as_array().to_owned().to_vec().into(),
                None,
            ))),
            NumpyArray::Float64(array) => {
                Ok(ChannelData::Float64(Float64Builder::new_from_buffer(
                    array.readonly().as_array().to_owned().to_vec().into(),
                    None,
                )))
            }
            NumpyArray::Complex32(array) => {
                Ok(ChannelData::Float32(Float32Builder::new_from_buffer(
                    array.readonly().as_array().to_owned().to_vec().into(),
                    None,
                )))
            }
            NumpyArray::Complex64(array) => {
                Ok(ChannelData::Float64(Float64Builder::new_from_buffer(
                    array.readonly().as_array().to_owned().to_vec().into(),
                    None,
                )))
            }
            NumpyArray::ArrayDInt8(array) => {
                let order: Order = if array.is_c_contiguous() {
                    Order::RowMajor
                } else {
                    Order::ColumnMajor
                };
                Ok(ChannelData::ArrayDInt8((
                    Int8Builder::new_from_buffer(
                        array.readonly().as_array().to_owned().into_raw_vec().into(),
                        None,
                    ),
                    (array.shape().to_vec(), order),
                )))
            }
            NumpyArray::ArrayDUInt8(array) => {
                let order: Order = if array.is_c_contiguous() {
                    Order::RowMajor
                } else {
                    Order::ColumnMajor
                };
                Ok(ChannelData::ArrayDUInt8((
                    UInt8Builder::new_from_buffer(
                        array.readonly().as_array().to_owned().into_raw_vec().into(),
                        None,
                    ),
                    (array.shape().to_vec(), order),
                )))
            }
            NumpyArray::ArrayDInt16(array) => {
                let order: Order = if array.is_c_contiguous() {
                    Order::RowMajor
                } else {
                    Order::ColumnMajor
                };
                Ok(ChannelData::ArrayDInt16((
                    Int16Builder::new_from_buffer(
                        array.readonly().as_array().to_owned().into_raw_vec().into(),
                        None,
                    ),
                    (array.shape().to_vec(), order),
                )))
            }
            NumpyArray::ArrayDUInt16(array) => {
                let order: Order = if array.is_c_contiguous() {
                    Order::RowMajor
                } else {
                    Order::ColumnMajor
                };
                Ok(ChannelData::ArrayDUInt16((
                    UInt16Builder::new_from_buffer(
                        array.readonly().as_array().to_owned().into_raw_vec().into(),
                        None,
                    ),
                    (array.shape().to_vec(), order),
                )))
            }
            NumpyArray::ArrayDInt32(array) => {
                let order: Order = if array.is_c_contiguous() {
                    Order::RowMajor
                } else {
                    Order::ColumnMajor
                };
                Ok(ChannelData::ArrayDInt32((
                    Int32Builder::new_from_buffer(
                        array.readonly().as_array().to_owned().into_raw_vec().into(),
                        None,
                    ),
                    (array.shape().to_vec(), order),
                )))
            }
            NumpyArray::ArrayDUInt32(array) => {
                let order: Order = if array.is_c_contiguous() {
                    Order::RowMajor
                } else {
                    Order::ColumnMajor
                };
                Ok(ChannelData::ArrayDUInt32((
                    UInt32Builder::new_from_buffer(
                        array.readonly().as_array().to_owned().into_raw_vec().into(),
                        None,
                    ),
                    (array.shape().to_vec(), order),
                )))
            }
            NumpyArray::ArrayDFloat32(array) => {
                let order: Order = if array.is_c_contiguous() {
                    Order::RowMajor
                } else {
                    Order::ColumnMajor
                };
                Ok(ChannelData::ArrayDFloat32((
                    Float32Builder::new_from_buffer(
                        array.readonly().as_array().to_owned().into_raw_vec().into(),
                        None,
                    ),
                    (array.shape().to_vec(), order),
                )))
            }
            NumpyArray::ArrayDInt64(array) => {
                let order: Order = if array.is_c_contiguous() {
                    Order::RowMajor
                } else {
                    Order::ColumnMajor
                };
                Ok(ChannelData::ArrayDInt64((
                    Int64Builder::new_from_buffer(
                        array.readonly().as_array().to_owned().into_raw_vec().into(),
                        None,
                    ),
                    (array.shape().to_vec(), order),
                )))
            }
            NumpyArray::ArrayDUInt64(array) => {
                let order: Order = if array.is_c_contiguous() {
                    Order::RowMajor
                } else {
                    Order::ColumnMajor
                };
                Ok(ChannelData::ArrayDUInt64((
                    UInt64Builder::new_from_buffer(
                        array.readonly().as_array().to_owned().into_raw_vec().into(),
                        None,
                    ),
                    (array.shape().to_vec(), order),
                )))
            }
            NumpyArray::ArrayDFloat64(array) => {
                let order: Order = if array.is_c_contiguous() {
                    Order::RowMajor
                } else {
                    Order::ColumnMajor
                };
                Ok(ChannelData::ArrayDFloat64((
                    Float64Builder::new_from_buffer(
                        array.readonly().as_array().to_owned().into_raw_vec().into(),
                        None,
                    ),
                    (array.shape().to_vec(), order),
                )))
            }
        }
    }
}

/// Enum to identify the dtype of numpy array
#[derive(Clone, FromPyObject)]
enum NumpyArray<'a> {
    Boolean(&'a PyArray1<u8>),
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
    Complex32(&'a PyArray1<f32>),
    Complex64(&'a PyArray1<f64>),
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
}
