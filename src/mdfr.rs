use crate::{mdfreader, mdfinfo::MdfInfo};
use pyo3::prelude::*;
use pyo3::PyObjectProtocol;

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
    fn get_channel_data(&self, channel_name: String) -> Py<PyAny> {
        let Mdf(mdf) = self;
        // default py_array value is python None
        pyo3::Python::with_gil(|py| {
            let py_array: Py<PyAny>;
            match mdf {
                MdfInfo::V3(_mdfinfo3) => {
                    py_array = py.None();
                },
                MdfInfo::V4(mdfinfo4) => {
                    if let Some(data) = mdfinfo4.get_channel_data(&channel_name) {
                        py_array = data.to_object(py);
                    } else {
                        py_array = py.None();
                    }
                }
            };
            py_array
        })
    }
}

#[pyproto]
impl PyObjectProtocol for Mdf {
    fn __repr__(&self) -> PyResult<String> {
        let mut output: String;
        match &self.0 {
            MdfInfo::V3(mdfinfo3) => {
                output = format!("Version : {}\n", mdfinfo3.ver);
                output.push_str(&format!("Version : {:?}\n", mdfinfo3.hdblock));
            }
            MdfInfo::V4(mdfinfo4) => {
                output = format!("Version : {}\n", mdfinfo4.ver);
                output.push_str(&format!("{}\n", mdfinfo4.hd_block));
                let comments = &mdfinfo4.hd_comment;
                for c in comments.iter() {
                    output.push_str(&format!("{} {}", c.0, c.1));
                }
                for (master, list) in mdfinfo4.db.master_channel_list.iter() {
                    output.push_str(&format!("\nMaster: {}\n", master));
                    for channel in list.iter() {
                        if let Some(data) = mdfinfo4.get_channel_data(channel) {
                            let data_min_max = data.min_max();
                            output.push_str(&format!(" {} {}\n", channel, data_min_max));
                        } else {
                            output.push_str(&format!(" {} \n", channel));
                        }
                    }
                }
                output.push_str(&format!("\n"));
            }
        }
        Ok(output)
    }
}
