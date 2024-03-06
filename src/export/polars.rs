//! this module provides methods to get directly from arrow into polars (rust or python)
use std::sync::Arc;

use arrow::array::Array;
use pyo3::{types::PyList, PyObject, PyResult, ToPyObject};

use crate::export::python_arrow_helpers::to_py_array;

/// converts rust arrow array into python polars series
#[allow(dead_code)]
pub fn rust_arrow_to_py_series(array: Arc<dyn Array>, name: String) -> PyResult<PyObject> {
    // ensure we have a single chunk

    // acquire the gil
    pyo3::Python::with_gil(|py| {
        // pyarrow array
        let pyarrow_array =
            to_py_array(py, array).expect("failed to convert arrow array to pyarrow array");

        // import polars
        let polars = py.import("polars").expect("could not import polars");
        let vecname: Vec<String> = vec![name];
        let pyname = PyList::new(py, vecname);
        let out = polars
            .call_method1("from_arrow", (pyarrow_array, pyname))
            .expect("method from_arrow not existing");
        Ok(out.to_object(py))
    })
}
