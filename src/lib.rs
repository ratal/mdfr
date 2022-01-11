#![forbid(unsafe_code)]
pub mod mdfinfo;
pub mod mdfr;
pub mod mdfreader;
pub mod mdfwriter;
mod tests;

use pyo3::prelude::*;

#[pymodule]
fn mdfr(py: Python, m: &PyModule) -> PyResult<()> {
    mdfr::register(py, m)?;
    Ok(())
}
