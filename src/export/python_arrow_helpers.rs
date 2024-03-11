//! helpers for arrow use with python
use std::sync::Arc;

use arrow::array::{make_array, Array, ArrayData};
use arrow::pyarrow::PyArrowType;
use pyo3::prelude::*;

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
