#![forbid(unsafe_code)]
pub mod mdfr;
mod tests;
pub mod mdfinfo;
pub mod mdfreader;

use pyo3::prelude::*;

#[pymodule]
fn mdfr(py: Python, m: &PyModule) -> PyResult<()> {
    mdfr::register(py, m)?;
    Ok(())
}
