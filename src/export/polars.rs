//! this module provides methods to get directly from arrow into polars (rust or python)
use arrow2::array::Array;
use pyo3::{PyObject, PyResult, ToPyObject};

use crate::mdfreader::arrow::to_py_array;

/// converts rust arrow array into python polars series
#[allow(dead_code)]
pub fn rust_arrow_to_py_series(array: Box<dyn Array>) -> PyResult<PyObject> {
    // ensure we have a single chunk

    // acquire the gil
    pyo3::Python::with_gil(|py| {
        // import pyarrow
        let pyarrow = py.import("pyarrow").expect("could not import pyarrow");

        // pyarrow array
        let pyarrow_array = to_py_array(py, pyarrow, array)
            .expect("failed to convert arrow array to pyarrow array");

        // import polars
        let polars = py.import("polars").expect("could not import polars");
        let out = polars
            .call_method1("from_arrow", (pyarrow_array,))
            .expect("method from_arrow not existing");
        Ok(out.to_object(py))
    })
}
