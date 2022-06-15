//! This module provides python interface using pyo3s
use std::collections::HashSet;
use std::fmt::Write;

use crate::export::arrow::{array_to_rust, to_py_array};
use crate::mdfinfo::MdfInfo;
use crate::mdfreader::Mdf;
use arrow2::array::get_display;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict};

/// This function is used to create a python dictionary from a MdfInfo object
#[pyclass]
struct Mdfr(Mdf);

pub(crate) fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Mdfr>()?;
    Ok(())
}
// TODO export to hdf5 and parquet using arrow, xlswriter
// TODO resample and export to csv ?

/// Imple&ments Mdf class to provide API to python using pyo3
#[pymethods]
impl Mdfr {
    /// creates new object from file name
    #[new]
    fn new(file_name: &str) -> Self {
        Mdfr(Mdf::new(file_name))
    }
    /// returns channel's data, numpy array or list, depending if data type is numeric or string|bytes
    fn get_channel_data(&mut self, channel_name: String) -> Py<PyAny> {
        let Mdfr(mdf) = self;
        // default py_array value is python None
        pyo3::Python::with_gil(|py| {
            let mut py_array: Py<PyAny>;
            let dt = mdf.get_channel_data(&channel_name);
            let field = mdf.get_channel_field(&channel_name);
            if let Some(data) = dt {
                if let Some(field) = field {
                    py_array = to_py_array(
                        py,
                        py.import("arrow2").expect("could not import arrow2"),
                        data,
                        &field,
                    )
                    .expect("failed to convert data to py array");
                    if let Some(m) = data.validity() {
                        let mask: Py<PyAny> = m.iter().collect::<Vec<bool>>().into_py(py);
                        let locals =
                            [("numpy", py.import("numpy").expect("could not import numpy"))]
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
                    }
                }
            }
            py_array
        })
    }
    /// returns channel's unit string
    fn get_channel_unit(&self, channel_name: String) -> Py<PyAny> {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|py| {
            let unit: Py<PyAny> = mdf.mdf_info.get_channel_unit(&channel_name).to_object(py);
            unit
        })
    }
    /// returns channel's description string
    fn get_channel_desc(&self, channel_name: String) -> Py<PyAny> {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|py| {
            let desc: Py<PyAny> = mdf.mdf_info.get_channel_desc(&channel_name).to_object(py);
            desc
        })
    }
    /// returns channel's associated master channel name string
    pub fn get_channel_master(&self, channel_name: String) -> Py<PyAny> {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|py| {
            let master: Py<PyAny> = mdf.mdf_info.get_channel_master(&channel_name).to_object(py);
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
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|py| {
            let master_type: Py<PyAny> = mdf
                .mdf_info
                .get_channel_master_type(&channel_name)
                .to_object(py);
            master_type
        })
    }
    /// returns a set of all channel names contained in file
    pub fn get_channel_names_set(&self) -> Py<PyAny> {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|py| {
            let channel_list: Py<PyAny> = mdf.mdf_info.get_channel_names_set().into_py(py);
            channel_list
        })
    }
    /// returns a dict of master names keys for which values are a set of associated channel names
    pub fn get_master_channel_names_set(&self) -> Py<PyAny> {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|py| {
            let master_channel_list: Py<PyAny> =
                mdf.mdf_info.get_master_channel_names_set().into_py(py);
            master_channel_list
        })
    }
    /// load a set of channels in memory
    pub fn load_channels_data_in_memory(&mut self, channel_names: HashSet<String>) {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|_py| {
            mdf.load_channels_data_in_memory(channel_names);
        })
    }
    /// clear channels from memory
    pub fn clear_channel_data_from_memory(&mut self, channel_names: HashSet<String>) {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|_py| {
            mdf.clear_channel_data_from_memory(channel_names);
        })
    }
    /// load all channels in memory
    pub fn load_all_channels_data_in_memory(&mut self) {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|_py| {
            mdf.load_all_channels_data_in_memory();
        })
    }
    /// writes file
    pub fn write(&mut self, file_name: &str, compression: bool) -> Mdfr {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|_py| Mdfr(mdf.write(file_name, compression)))
    }
    /// Adds a new channel in memory (no file modification)
    pub fn add_channel(
        &mut self,
        channel_name: String,
        data: Py<PyAny>,
        master_channel: Option<String>,
        master_type: Option<u8>,
        master_flag: bool,
        unit: Option<String>,
        description: Option<String>,
    ) {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|py| {
            let array = array_to_rust(py, &data)
                .expect("data modification failed, could not extract numpy array");
            mdf.add_channel(
                channel_name,
                array,
                master_channel,
                master_type,
                master_flag,
                unit,
                description,
            );
        })
    }
    /// defines channel's data in memory
    pub fn set_channel_data(&mut self, channel_name: &str, data: Py<PyAny>) {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|py| {
            let array = array_to_rust(py, &data)
                .expect("data modification failed, could not extract numpy array");
            mdf.set_channel_data(channel_name, array);
        })
    }
    /// Sets the channel's related master channel type in memory
    pub fn set_channel_master_type(&mut self, master_name: &str, master_type: u8) {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|_py| {
            mdf.mdf_info
                .set_channel_master_type(master_name, master_type);
        })
    }
    /// Removes a channel in memory (no file modification)
    pub fn remove_channel(&mut self, channel_name: &str) {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|_py| {
            mdf.mdf_info.remove_channel(channel_name);
        })
    }
    /// Renames a channel's name in memory
    pub fn rename_channel(&mut self, channel_name: &str, new_name: &str) {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|_py| {
            mdf.mdf_info.rename_channel(channel_name, new_name);
        })
    }
    /// Sets the channel unit in memory
    pub fn set_channel_unit(&mut self, channel_name: &str, unit: &str) {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|_py| {
            mdf.mdf_info.set_channel_unit(channel_name, unit);
        })
    }
    /// Sets the channel description in memory
    pub fn set_channel_desc(&mut self, channel_name: &str, desc: &str) {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|_py| {
            mdf.mdf_info.set_channel_desc(channel_name, desc);
        })
    }
    /// plot one channel
    pub fn plot(&mut self, channel_name: String) {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|py| {
            let locals = PyDict::new(py);
            locals
                .set_item("channel_name", &channel_name)
                .expect("cannot set python channel_name");
            locals
                .set_item("channel_unit", mdf.mdf_info.get_channel_unit(&channel_name))
                .expect("cannot set python channel_unit");
            if let Some(master_name) = mdf.mdf_info.get_channel_master(&channel_name) {
                locals
                    .set_item("master_channel_name", &master_name)
                    .expect("cannot set python master_channel_name");
                locals
                    .set_item(
                        "master_channel_unit",
                        mdf.mdf_info.get_channel_unit(&master_name),
                    )
                    .expect("cannot set python master_channel_unit");
                let data = self.get_channel_data(master_name);
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
            let data = self.get_channel_data(channel_name);
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
    /// export to Parquet file
    pub fn export_to_parquet(&mut self, file_name: &str, compression_option: Option<&str>) {
        let Mdfr(mdf) = self;
        mdf.export_to_parquet(file_name, compression_option)
            .expect("could not export to parquet")
    }
    fn __repr__(&self) -> PyResult<String> {
        let mut output: String;
        match &self.0.mdf_info {
            MdfInfo::V3(mdfinfo3) => {
                output = format!("Version : {}\n", mdfinfo3.id_block.id_ver);
                writeln!(
                    output,
                    "Header :\n Author: {}  Organisation:{}",
                    mdfinfo3.hd_block.hd_author, mdfinfo3.hd_block.hd_organization
                )
                .expect("cannot print author and organisation");
                writeln!(
                    output,
                    "Project: {}  Subject:{}",
                    mdfinfo3.hd_block.hd_project, mdfinfo3.hd_block.hd_subject
                )
                .expect("cannot print project and subject");
                writeln!(
                    output,
                    "Date: {:?}  Time:{:?}",
                    mdfinfo3.hd_block.hd_date, mdfinfo3.hd_block.hd_time
                )
                .expect("cannot print date and time");
                write!(output, "Comments: {}", mdfinfo3.hd_comment).expect("cannot print comments");
                for (master, list) in mdfinfo3.get_master_channel_names_set().iter() {
                    if let Some(master_name) = master {
                        writeln!(output, "\nMaster: {}", master_name)
                            .expect("cannot print master channel name");
                    } else {
                        writeln!(output, "\nWithout Master channel")
                            .expect("cannot print thre is no master channel");
                    }
                    for channel in list.iter() {
                        let unit = self.get_channel_unit(channel.to_string());
                        let desc = self.get_channel_desc(channel.to_string());
                        writeln!(output, " {} ", channel).expect("cannot print channel name");
                        if let Some(data) = self.0.get_channel_data(channel) {
                            if !data.is_empty() {
                                let displayer = get_display(data.as_ref(), "null");
                                displayer(&mut output, 0).expect("cannot channel data");
                                writeln!(output, " ");
                                displayer(&mut output, data.len() - 1)
                                    .expect("cannot channel data");
                            }
                            writeln!(
                                output,
                                "{} {} ",
                                unit, desc
                            ).expect("cannot print channel unit and description with first and last item");
                        }
                        writeln!(output, " {} {} ", unit, desc)
                            .expect("cannot print channel unit and description");
                    }
                }
                output.push_str("\n ");
            }
            MdfInfo::V4(mdfinfo4) => {
                output = format!("Version : {}\n", mdfinfo4.id_block.id_ver);
                writeln!(output, "{}", mdfinfo4.hd_block).expect("cannot print header block");
                let comments = &mdfinfo4
                    .sharable
                    .get_comments(mdfinfo4.hd_block.hd_md_comment);
                for c in comments.iter() {
                    writeln!(output, "{} {}", c.0, c.1).expect("cannot print header comments");
                }
                for (master, list) in mdfinfo4.get_master_channel_names_set().iter() {
                    if let Some(master_name) = master {
                        writeln!(output, "\nMaster: {}", master_name)
                            .expect("cannot print master channel name");
                    } else {
                        writeln!(output, "\nWithout Master channel")
                            .expect("cannot print thre is no master channel");
                    }
                    for channel in list.iter() {
                        let unit = self.get_channel_unit(channel.to_string());
                        let desc = self.get_channel_desc(channel.to_string());
                        writeln!(output, " {} ", channel).expect("cannot print channel name");
                        if let Some(data) = self.0.get_channel_data(channel) {
                            if !data.is_empty() {
                                let displayer = get_display(data.as_ref(), "null");
                                displayer(&mut output, 0).expect("cannot channel data");
                                writeln!(output, " ");
                                displayer(&mut output, data.len() - 1)
                                    .expect("cannot channel data");
                            }
                            writeln!(
                                output,
                                "{} {} ",
                                unit, desc
                            ).expect("cannot print channel unit and description with first and last item");
                        }
                        writeln!(output, " {} {} ", unit, desc)
                            .expect("cannot print channel unit and description");
                    }
                }
                output.push_str("\n ");
            }
        }
        Ok(output)
    }
}
