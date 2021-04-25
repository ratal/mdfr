pub mod mdfinfo;
mod tests;

use pyo3::prelude::*;

#[pymodule]
fn mdfr(_py: Python, _module: &PyModule) -> PyResult<()> {
    Ok(())
}