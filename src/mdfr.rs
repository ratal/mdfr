//! This module provides python interface using pyo3s
use std::collections::HashSet;

use crate::mdfinfo::MdfInfo;
use numpy::ToPyArray;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict};

/// This function is used to create a python dictionary from a MdfInfo object
#[pyclass]
struct Mdf(MdfInfo);

pub(crate) fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Mdf>()?;
    Ok(())
}
// TODO export to hdf5 and parquet using arrow, xlswriter
// TODO resample and export to csv ?

/// Implements Mdf class to provide API to python using pyo3
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
            let mut py_array: Py<PyAny>;
            let (data, mask) = mdf.get_channel_data(&channel_name);
            if let Some(m) = mask {
                py_array = data.to_object(py);
                let mask: Py<PyAny> = m.to_pyarray(py).into_py(py);
                let locals = [("numpy", py.import("numpy").expect("could not import numpy"))]
                    .into_py_dict(py);
                locals
                    .set_item("py_array", &py_array)
                    .expect("cannot set python data");
                locals
                    .set_item("mask", mask)
                    .expect("cannot set python mask");
                py_array = py
                    .eval(r#"numpy.ma.array(py_array, mask=mask)"#, None, Some(locals))
                    .expect("masked array creation failed")
                    .into_py(py);
            } else {
                py_array = data.to_object(py);
            }
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
    /// returns channel's master data, numpy array or list, depending if data type is numeric or string|bytes
    fn get_channel_master_data(&mut self, channel_name: String) -> Py<PyAny> {
        // default py_array value is python None
        let master = self.get_channel_master(channel_name);
        self.get_channel_data(master.to_string())
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
    /// writes file
    pub fn write(&mut self, file_name: &str, compression: bool) -> Mdf {
        let Mdf(mdf) = self;
        pyo3::Python::with_gil(|_py| Mdf(mdf.write(file_name, compression)))
    }
    /// plot one channel
    pub fn plot(&mut self, channel_name: String) {
        let Mdf(mdf) = self;
        pyo3::Python::with_gil(|py| {
            let locals = PyDict::new(py);
            locals
                .set_item("channel_name", &channel_name)
                .expect("cannot set python channel_name");
            locals
                .set_item("channel_unit", mdf.get_channel_unit(&channel_name))
                .expect("cannot set python channel_unit");
            if let Some(master_name) = mdf.get_channel_master(&channel_name) {
                locals
                    .set_item("master_channel_name", &master_name)
                    .expect("cannot set python master_channel_name");
                locals
                    .set_item("master_channel_unit", mdf.get_channel_unit(&master_name))
                    .expect("cannot set python master_channel_unit");
                let (data, _mask) = mdf.get_channel_data(&master_name);
                locals
                    .set_item("master_data", data)
                    .expect("cannot set python master_data");
            } else {
                locals
                    .set_item("master_channel_name", py.None())
                    .expect("cannot set python master_channel_name");
                locals
                    .set_item("master_channel_unit", py.None())
                    .expect("cannot set python master_channel_unit");
                locals
                    .set_item("master_data", py.None())
                    .expect("cannot set python master_data");
            }
            let (data, _mask) = mdf.get_channel_data(&channel_name);
            locals
                .set_item("channel_data", data)
                .expect("cannot set python channel_data");
            py.import("matplotlib")
                .expect("Could not plot channel with matplotlib");
            py.run(
                r#"
from matplotlib import pyplot
from numpy import arange
if master_data is None:
    master_data = arange(0, len(channel_data), 1)
pyplot.plot(master_data, channel_data, label='{0} [{1}]'.format(channel_name, channel_unit))
if master_channel_name is not None:
    if master_channel_unit is not None:
        pyplot.xlabel('{0} [{1}]'.format(master_channel_name, master_channel_unit))
    else:
        pyplot.xlabel('{0}'.format(master_channel_name))
pyplot.ylabel('{0} [{1}]'.format(channel_name, channel_unit))
pyplot.grid(True)
pyplot.show()
"#,
                None,
                Some(locals),
            )
            .expect("plot python script failed");
        })
    }
    fn __repr__(&self) -> PyResult<String> {
        let mut output: String;
        match &self.0 {
            MdfInfo::V3(mdfinfo3) => {
                output = format!("Version : {}\n", mdfinfo3.id_block.id_ver);
                output.push_str(&format!(
                    "Header :\n Author: {}  Organisation:{}\n",
                    mdfinfo3.hd_block.hd_author, mdfinfo3.hd_block.hd_organization
                ));
                output.push_str(&format!(
                    "Project: {}  Subject:{}\n",
                    mdfinfo3.hd_block.hd_project, mdfinfo3.hd_block.hd_subject
                ));
                output.push_str(&format!(
                    "Date: {:?}  Time:{:?}\n",
                    mdfinfo3.hd_block.hd_date, mdfinfo3.hd_block.hd_time
                ));
                output.push_str(&format!("Comments: {}", mdfinfo3.hd_comment));
                for (master, list) in mdfinfo3.get_master_channel_names_set().iter() {
                    if let Some(master_name) = master {
                        output.push_str(&format!("\nMaster: {}\n", master_name));
                    } else {
                        output.push_str("\nWithout Master channel\n");
                    }
                    for channel in list.iter() {
                        let unit = self.get_channel_unit(channel.to_string());
                        let desc = self.get_channel_desc(channel.to_string());
                        if let Some(data) = mdfinfo3.get_channel_data_from_memory(channel) {
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
                output.push_str("\n ");
            }
            MdfInfo::V4(mdfinfo4) => {
                output = format!("Version : {}\n", mdfinfo4.id_block.id_ver);
                output.push_str(&format!("{}\n", mdfinfo4.hd_block));
                let comments = &mdfinfo4
                    .sharable
                    .get_comments(mdfinfo4.hd_block.hd_md_comment);
                for c in comments.iter() {
                    output.push_str(&format!("{} {}\n", c.0, c.1));
                }
                for (master, list) in mdfinfo4.get_master_channel_names_set().iter() {
                    if let Some(master_name) = master {
                        output.push_str(&format!("\nMaster: {}\n", master_name));
                    } else {
                        output.push_str("\nWithout Master channel\n");
                    }
                    for channel in list.iter() {
                        let unit = self.get_channel_unit(channel.to_string());
                        let desc = self.get_channel_desc(channel.to_string());
                        let dtmsk = mdfinfo4.get_channel_data_from_memory(channel);
                        if let Some(data) = dtmsk.0 {
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
                output.push_str("\n ");
            }
        }
        Ok(output)
    }
}
