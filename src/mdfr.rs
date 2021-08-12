use crate::mdfinfo::MdfInfo;
use crate::mdfreader;
use pyo3::prelude::*;

#[pyclass]
struct Mdf(MdfInfo);

pub(crate) fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Mdf>()?;
    Ok(())
}

#[pymethods]
impl Mdf {
    #[new]
    fn new(file_name: &str) -> Self {
        Mdf(mdfreader::mdfreader(file_name))
    }
}
