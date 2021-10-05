use std::collections::HashSet;

use crate::mdfinfo::MdfInfo;
use pyo3::prelude::*;
use pyo3::PyObjectProtocol;

#[pyclass]
struct Mdf(MdfInfo);

pub(crate) fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Mdf>()?;
    Ok(())
}
//TODO export to hdf5 and parquet using arrow, xlswriter
#[pymethods]
impl Mdf {
    /// creates new object from file name
    #[new]
    fn new(file_name: &str) -> Self {
        Mdf(MdfInfo::new(file_name))
    }
    /// returns channel's data, numpy array or list, depending if data type is numeric or string|bytes
    fn get_channel_data(&mut self, channel_name: String) -> Py<PyAny> {
        let Mdf(mdf) = self;
        // default py_array value is python None
        pyo3::Python::with_gil(|py| {
            let py_array: Py<PyAny> = mdf.get_channel_data(&channel_name).to_object(py);
            py_array
        })
    }
    /// returns channel's unit string
    fn get_channel_unit(&self, channel_name: String) -> Py<PyAny> {
        let Mdf(mdf) = self;
        pyo3::Python::with_gil(|py| {
            let unit: Py<PyAny> = mdf.get_channel_unit(&channel_name).to_object(py);
            unit
        })
    }
    /// returns channel's description string
    fn get_channel_desc(&self, channel_name: String) -> Py<PyAny> {
        let Mdf(mdf) = self;
        pyo3::Python::with_gil(|py| {
            let desc: Py<PyAny> = mdf.get_channel_desc(&channel_name).to_object(py);
            desc
        })
    }
    /// returns channel's associated master channel name string
    pub fn get_channel_master(&self, channel_name: String) -> Py<PyAny> {
        let Mdf(mdf) = self;
        pyo3::Python::with_gil(|py| {
            let master: Py<PyAny> = mdf.get_channel_master(&channel_name).to_object(py);
            master
        })
    }
    /// returns channel's associated master channel type string
    /// 0 = None (normal data channels), 1 = Time (seconds), 2 = Angle (radians),
    /// 3 = Distance (meters), 4 = Index (zero-based index values)
    pub fn get_channel_master_type(&self, channel_name: String) -> Py<PyAny> {
        let Mdf(mdf) = self;
        pyo3::Python::with_gil(|py| {
            let master_type: Py<PyAny> = mdf.get_channel_master_type(&channel_name).to_object(py);
            master_type
        })
    }
    /// returns a set of all channel names contained in file
    pub fn get_channel_names_set(&self) -> Py<PyAny> {
        let Mdf(mdf) = self;
        pyo3::Python::with_gil(|py| {
            let channel_list: Py<PyAny> = mdf.get_channel_names_set().into_py(py);
            channel_list
        })
    }
    /// returns a dict of master names keys for which values are a set of associated channel names
    pub fn get_master_channel_names_set(&self) -> Py<PyAny> {
        let Mdf(mdf) = self;
        pyo3::Python::with_gil(|py| {
            let master_channel_list: Py<PyAny> = mdf.get_master_channel_names_set().into_py(py);
            master_channel_list
        })
    }
    /// load a set of channels in memory
    pub fn load_channels_data_in_memory(&mut self, channel_names: HashSet<String>) {
        let Mdf(mdf) = self;
        pyo3::Python::with_gil(|_py| {
            mdf.load_channels_data_in_memory(channel_names);
        })
    }
    /// clear channels from memory
    pub fn clear_channel_data_from_memory(&mut self, channel_names: HashSet<String>) {
        let Mdf(mdf) = self;
        pyo3::Python::with_gil(|_py| {
            mdf.clear_channel_data_from_memory(channel_names);
        })
    }
    /// load all channels in memory
    pub fn load_all_channels_data_in_memory(&mut self) {
        let Mdf(mdf) = self;
        pyo3::Python::with_gil(|_py| {
            mdf.load_all_channels_data_in_memory();
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
                    output.push_str(&format!("{} {}\n", c.0, c.1));
                }
                for (master, list) in mdfinfo4.get_master_channel_names_set().iter() {
                    output.push_str(&format!("\nMaster: {}\n", master));
                    for channel in list.iter() {
                        let unit = self.get_channel_unit(channel.to_string());
                        let desc = self.get_channel_desc(channel.to_string());
                        if let Some(data) = mdfinfo4.get_channel_data_from_memory(channel) {
                            let data_first_last = data.first_last();

                            output.push_str(&format!(
                                " {} {} {} {} \n",
                                channel, data_first_last, unit, desc
                            ));
                        } else {
                            output.push_str(&format!(" {} {} {} \n", channel, unit, desc));
                        }
                    }
                }
                output.push_str(&format!("\n"));
            }
        }
        Ok(output)
    }
}
