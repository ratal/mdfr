//! this module provides methods to get directly channelData into python

use arrow::array::{Array, ArrayData, UnionArray, make_array};
use arrow::datatypes::ArrowPrimitiveType;
use arrow::pyarrow::PyArrowType;

use numpy::npyffi::types::NPY_ORDER;
use numpy::{Element, PyArray1, PyArrayMethods, ToPyArray};
use pyo3::types::{PyBytes, PyList};
use pyo3::{Bound, prelude::*};

use crate::data_holder::channel_data::ChannelData;
use crate::data_holder::tensor_arrow::{Order, TensorArrow};

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

/// Reorder each sample's elements from column-major (Fortran) to row-major (C) order.
///
/// MDF4 `CABLOCK` `ca_flags` bit 6 ("inverse layout") applies per record/sample, while the
/// sample axis itself always remains outermost.
fn reorder_samples_fortran_to_c<T: Copy>(values: &[T], dims: &[usize]) -> Vec<T> {
    let sample_elems: usize = dims.iter().product();
    if sample_elems == 0 || dims.len() <= 1 {
        return values.to_vec();
    }

    let mut f_strides = Vec::with_capacity(dims.len());
    let mut stride = 1usize;
    for &d in dims {
        f_strides.push(stride);
        stride *= d;
    }

    let mut reordered = Vec::with_capacity(values.len());
    let mut indices = vec![0usize; dims.len()];
    let mut f_idx = 0usize;

    for sample in values.chunks_exact(sample_elems) {
        for _ in 0..sample_elems {
            reordered.push(sample[f_idx]);
            for dim in (0..dims.len()).rev() {
                indices[dim] += 1;
                f_idx += f_strides[dim];
                if indices[dim] < dims[dim] {
                    break;
                }
                f_idx -= dims[dim] * f_strides[dim];
                indices[dim] = 0;
            }
        }
    }

    reordered
}

/// Convert a [`TensorArrow`] to a NumPy `ndarray` of shape `[N, *dims]` in C-order.
fn tensor_to_pyarray<'py, T>(py: Python<'py>, array: &TensorArrow<T>) -> PyResult<Bound<'py, PyAny>>
where
    T: ArrowPrimitiveType,
    T::Native: Element,
{
    let dims = array.shape();
    let mut shape = Vec::with_capacity(dims.len() + 1);
    shape.push(array.len());
    shape.extend_from_slice(dims);

    let flat = if *array.order() == Order::ColumnMajor && dims.len() > 1 && !array.is_empty() {
        let reordered = reorder_samples_fortran_to_c(array.values_slice(), dims);
        PyArray1::from_vec(py, reordered)
    } else {
        array.values_slice().to_pyarray(py)
    };

    Ok(flat
        .reshape_with_order(shape, NPY_ORDER::NPY_CORDER)?
        .into_any())
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
            ChannelData::Int8(array) => Ok(array.values().to_pyarray(py).into_any()),
            ChannelData::UInt8(array) => Ok(array.values().to_pyarray(py).into_any()),
            ChannelData::Int16(array) => Ok(array.values().to_pyarray(py).into_any()),
            ChannelData::UInt16(array) => Ok(array.values().to_pyarray(py).into_any()),
            ChannelData::Int32(array) => Ok(array.values().to_pyarray(py).into_any()),
            ChannelData::UInt32(array) => Ok(array.values().to_pyarray(py).into_any()),
            ChannelData::Float32(array) => Ok(array.values().to_pyarray(py).into_any()),
            ChannelData::Int64(array) => Ok(array.values().to_pyarray(py).into_any()),
            ChannelData::UInt64(array) => Ok(array.values().to_pyarray(py).into_any()),
            ChannelData::Float64(array) => Ok(array.values().to_pyarray(py).into_any()),
            ChannelData::Complex32(array) => Ok(array.values_slice().to_pyarray(py).into_any()),
            ChannelData::Complex64(array) => Ok(array.values_slice().to_pyarray(py).into_any()),
            ChannelData::VariableSizeByteArray(array) => {
                let list = PyList::new(
                    py,
                    array
                        .iter()
                        .map(|opt| opt.map(|bytes| PyBytes::new(py, bytes))),
                )?;
                Ok(list.into_any())
            }
            ChannelData::FixedSizeByteArray(array) => {
                let out: Vec<Vec<u8>> = if array.value_length() == 0 {
                    vec![vec![]; array.len()]
                } else {
                    array
                        .values()
                        .chunks(array.value_length() as usize)
                        .map(<[u8]>::to_vec)
                        .collect()
                };
                out.into_pyobject(py)
            }
            ChannelData::ArrayDInt8(array) => tensor_to_pyarray(py, &array),
            ChannelData::ArrayDUInt8(array) => tensor_to_pyarray(py, &array),
            ChannelData::ArrayDInt16(array) => tensor_to_pyarray(py, &array),
            ChannelData::ArrayDUInt16(array) => tensor_to_pyarray(py, &array),
            ChannelData::ArrayDInt32(array) => tensor_to_pyarray(py, &array),
            ChannelData::ArrayDUInt32(array) => tensor_to_pyarray(py, &array),
            ChannelData::ArrayDFloat32(array) => tensor_to_pyarray(py, &array),
            ChannelData::ArrayDInt64(array) => tensor_to_pyarray(py, &array),
            ChannelData::ArrayDUInt64(array) => tensor_to_pyarray(py, &array),
            ChannelData::ArrayDFloat64(array) => tensor_to_pyarray(py, &array),
            ChannelData::Utf8(array) => {
                let strings: Vec<Option<&str>> = array.iter().collect();
                strings.into_pyobject(py)
            }
            ChannelData::Union(array) => {
                let arrow_data = to_py_array(py, Arc::new(UnionArray::from(array.to_data())))?;
                arrow_data.into_pyobject(py)
            }
        }
    }
}
