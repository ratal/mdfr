//! this module provides methods to get directly channelData into python

use arrow::array::{Array, ArrayData, UnionArray, make_array};
use arrow::pyarrow::PyArrowType;

use numpy::npyffi::types::NPY_ORDER;
use numpy::{PyArrayMethods, ToPyArray};
use pyo3::{Bound, prelude::*};

use crate::data_holder::channel_data::ChannelData;
use crate::data_holder::tensor_arrow::Order;

impl From<Order> for NPY_ORDER {
    fn from(order: Order) -> Self {
        match order {
            Order::RowMajor => NPY_ORDER::NPY_CORDER,
            Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
        }
    }
}
use std::sync::Arc;

/// Take an arrow array from python and convert it to a rust arrow array.
/// This operation does not copy data.
#[allow(dead_code)]
pub fn array_to_rust(arrow_array: PyArrowType<ArrayData>) -> PyResult<Arc<dyn Array>> {
    // prepare a pointer to receive the Array struct
    let array = arrow_array.0; // Extract from PyArrowType wrapper
    Ok(make_array(array))
}

/// Arrow array to Python.
pub(crate) fn to_py_array(_: Python, array: Arc<dyn Array>) -> PyResult<PyArrowType<ArrayData>> {
    Ok(PyArrowType(array.into_data()))
}

impl<'py> IntoPyObject<'py> for ChannelData {
    type Target = PyAny; // the Python type
    type Output = Bound<'py, Self::Target>; // in most cases this will be `Bound`
    type Error = PyErr;
    /// IntoPyObject implementation to convert a ChannelData into a PyObject
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            ChannelData::Int8(array) => Ok(array.values_slice().to_pyarray(py).into_any()),
            ChannelData::UInt8(array) => Ok(array.values_slice().to_pyarray(py).into_any()),
            ChannelData::Int16(array) => Ok(array.values_slice().to_pyarray(py).into_any()),
            ChannelData::UInt16(array) => Ok(array.values_slice().to_pyarray(py).into_any()),
            ChannelData::Int32(array) => Ok(array.values_slice().to_pyarray(py).into_any()),
            ChannelData::UInt32(array) => Ok(array.values_slice().to_pyarray(py).into_any()),
            ChannelData::Float32(array) => Ok(array.values_slice().to_pyarray(py).into_any()),
            ChannelData::Int64(array) => Ok(array.values_slice().to_pyarray(py).into_any()),
            ChannelData::UInt64(array) => Ok(array.values_slice().to_pyarray(py).into_any()),
            ChannelData::Float64(array) => Ok(array.values_slice().to_pyarray(py).into_any()),
            ChannelData::Complex32(array) => Ok(array.values_slice().to_pyarray(py).into_any()),
            ChannelData::Complex64(array) => Ok(array.values_slice().to_pyarray(py).into_any()),
            ChannelData::VariableSizeByteArray(array) => {
                Ok(array.values_slice().to_pyarray(py).into_any())
            }
            ChannelData::FixedSizeByteArray(array) => {
                let binary_array = array.finish_cloned();
                let out: Vec<Vec<u8>> = binary_array
                    .values()
                    .chunks(binary_array.value_length() as usize)
                    .map(<[u8]>::to_vec)
                    .collect();
                out.into_pyobject(py)
            }
            ChannelData::ArrayDInt8(array) => {
                let flat = array.values_slice().to_pyarray(py);
                Ok(flat
                    .reshape_with_order(array.shape().clone(), array.order().clone().into())?
                    .into_any())
            }
            ChannelData::ArrayDUInt8(array) => {
                let flat = array.values_slice().to_pyarray(py);
                Ok(flat
                    .reshape_with_order(array.shape().clone(), array.order().clone().into())?
                    .into_any())
            }
            ChannelData::ArrayDInt16(array) => {
                let flat = array.values_slice().to_pyarray(py);
                Ok(flat
                    .reshape_with_order(array.shape().clone(), array.order().clone().into())?
                    .into_any())
            }
            ChannelData::ArrayDUInt16(array) => {
                let flat = array.values_slice().to_pyarray(py);
                Ok(flat
                    .reshape_with_order(array.shape().clone(), array.order().clone().into())?
                    .into_any())
            }
            ChannelData::ArrayDInt32(array) => {
                let flat = array.values_slice().to_pyarray(py);
                Ok(flat
                    .reshape_with_order(array.shape().clone(), array.order().clone().into())?
                    .into_any())
            }
            ChannelData::ArrayDUInt32(array) => {
                let flat = array.values_slice().to_pyarray(py);
                Ok(flat
                    .reshape_with_order(array.shape().clone(), array.order().clone().into())?
                    .into_any())
            }
            ChannelData::ArrayDFloat32(array) => {
                let flat = array.values_slice().to_pyarray(py);
                Ok(flat
                    .reshape_with_order(array.shape().clone(), array.order().clone().into())?
                    .into_any())
            }
            ChannelData::ArrayDInt64(array) => {
                let flat = array.values_slice().to_pyarray(py);
                Ok(flat
                    .reshape_with_order(array.shape().clone(), array.order().clone().into())?
                    .into_any())
            }
            ChannelData::ArrayDUInt64(array) => {
                let flat = array.values_slice().to_pyarray(py);
                Ok(flat
                    .reshape_with_order(array.shape().clone(), array.order().clone().into())?
                    .into_any())
            }
            ChannelData::ArrayDFloat64(array) => {
                let flat = array.values_slice().to_pyarray(py);
                Ok(flat
                    .reshape_with_order(array.shape().clone(), array.order().clone().into())?
                    .into_any())
            }
            ChannelData::Utf8(array) => {
                let string_array = array.finish_cloned();
                let strings: Vec<Option<&str>> = string_array.iter().collect();
                strings.into_pyobject(py)
            }
            ChannelData::Union(array) => {
                let arrow_data = to_py_array(py, Arc::new(UnionArray::from(array.to_data())))?;
                arrow_data.into_pyobject(py)
            }
        }
    }
}
