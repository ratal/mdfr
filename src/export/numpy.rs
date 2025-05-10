//! this module provides methods to get directly channelData into python

use arrow::array::{make_array, Array, ArrayData};
use arrow::pyarrow::PyArrowType;

use numpy::npyffi::types::NPY_ORDER;
use numpy::{PyArrayMethods, ToPyArray};
use pyo3::{prelude::*, Bound};

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
    type Error = std::convert::Infallible;
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
                    .map(|x| x.to_vec())
                    .collect();
                Ok(out
                    .into_pyobject(py)
                    .expect("error converting fixed size binary array into python object"))
            }
            ChannelData::ArrayDInt8(array) => Ok(array
                .values_slice()
                .to_pyarray(py)
                .reshape_with_order(array.shape().clone(), array.order().clone().into())
                .expect("could not reshape i8")
                .into_any()),
            ChannelData::ArrayDUInt8(array) => Ok(array
                .values_slice()
                .to_pyarray(py)
                .reshape_with_order(array.shape().clone(), array.order().clone().into())
                .expect("could not reshape u8")
                .into_any()),
            ChannelData::ArrayDInt16(array) => Ok(array
                .values_slice()
                .to_pyarray(py)
                .reshape_with_order(array.shape().clone(), array.order().clone().into())
                .expect("could not reshape u16")
                .into_any()),
            ChannelData::ArrayDUInt16(array) => Ok(array
                .values_slice()
                .to_pyarray(py)
                .reshape_with_order(array.shape().clone(), array.order().clone().into())
                .expect("could not reshape i16")
                .into_any()),
            ChannelData::ArrayDInt32(array) => Ok(array
                .values_slice()
                .to_pyarray(py)
                .reshape_with_order(array.shape().clone(), array.order().clone().into())
                .expect("could not reshape i32")
                .into_any()),
            ChannelData::ArrayDUInt32(array) => Ok(array
                .values_slice()
                .to_pyarray(py)
                .reshape_with_order(array.shape().clone(), array.order().clone().into())
                .expect("could not reshape u32")
                .into_any()),
            ChannelData::ArrayDFloat32(array) => Ok(array
                .values_slice()
                .to_pyarray(py)
                .reshape_with_order(array.shape().clone(), array.order().clone().into())
                .expect("could not reshape f32")
                .into_any()),
            ChannelData::ArrayDInt64(array) => Ok(array
                .values_slice()
                .to_pyarray(py)
                .reshape_with_order(array.shape().clone(), array.order().clone().into())
                .expect("could not reshape i64")
                .into_any()),
            ChannelData::ArrayDUInt64(array) => Ok(array
                .values_slice()
                .to_pyarray(py)
                .reshape_with_order(array.shape().clone(), array.order().clone().into())
                .expect("could not reshape u64")
                .into_any()),
            ChannelData::ArrayDFloat64(array) => Ok(array
                .values_slice()
                .to_pyarray(py)
                .reshape_with_order(array.shape().clone(), array.order().clone().into())
                .expect("could not reshape f64")
                .into_any()),
            ChannelData::Utf8(array) => Ok(array
                .finish_cloned()
                .iter()
                .collect::<Option<String>>()
                .into_pyobject(py)
                .expect("error converting Utf8 array into python object")),
        }
    }
}
