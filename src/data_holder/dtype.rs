//! Very basic numpy dtype
use pyo3::prelude::*;

#[derive(Default)]
#[pyclass]
pub struct NumpyDType {
    pub shape: Vec<usize>,
    pub kind: String,
}

#[pymethods]
impl NumpyDType {
    #[new]
    fn new() -> Self {
        NumpyDType::default()
    }
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }
    fn kind(&self) -> String {
        self.kind.clone()
    }
    fn __repr__(&mut self) -> PyResult<String> {
        Ok(format!(
            "dtype kind {}, shape {:?}",
            self.kind(),
            self.shape()
        ))
    }
}
