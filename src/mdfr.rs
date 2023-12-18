//! This module provides python interface using pyo3s
use std::collections::HashSet;
use std::fmt::Write;

use crate::export::numpy::arrow_to_numpy;
use crate::export::polars::rust_arrow_to_py_series;
use crate::mdfinfo::MdfInfo;
use crate::mdfreader::arrow::array_to_rust;
use crate::mdfreader::MasterSignature;
use crate::mdfreader::Mdf;
use arrow2::array::get_display;
use pyo3::exceptions::PyUnicodeDecodeError;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyBytes, PyDict, PyList};

#[pymodule]
#[pyo3(name = "mdfr")]
fn mdfr(py: Python, m: &PyModule) -> PyResult<()> {
    register(py, m)?;
    Ok(())
}

/// This function is used to create a python dictionary from a MdfInfo object
#[pyclass]
struct Mdfr(Mdf);

pub(crate) fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Mdfr>()?;
    Ok(())
}

/// Imple&ments Mdf class to provide API to python using pyo3
#[pymethods]
impl Mdfr {
    /// creates new object from file name
    #[new]
    fn new(file_name: &str) -> PyResult<Self> {
        Ok(Mdfr(Mdf::new(file_name)?))
    }
    /// gets the version of mdf file
    pub fn get_version(&mut self) -> u16 {
        let Mdfr(mdf) = self;
        mdf.get_version()
    }
    /// returns channel's data, numpy array or list, depending if data type is numeric or string|bytes
    fn get_channel_data(&self, channel_name: String) -> Py<PyAny> {
        let Mdfr(mdf) = self;
        // default py_array value is python None
        pyo3::Python::with_gil(|py| {
            let mut py_array: Py<PyAny>;
            let dt = mdf.get_channel_data(&channel_name);
            if let Some(data) = dt {
                py_array = arrow_to_numpy(py, data.clone());
                if let Some(m) = data.clone().validity() {
                    let mask: Py<PyAny> = m.iter().collect::<Vec<bool>>().into_py(py);
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
                }
            } else {
                py_array = Python::None(py);
            }
            py_array
        })
    }
    /// returns polars serie of channel
    fn get_polars_series(&self, channel_name: &str) -> PyResult<PyObject> {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|py| {
            let mut py_serie = Ok(Python::None(py));
            if let Some(array) = mdf.get_channel_data(channel_name) {
                py_serie = rust_arrow_to_py_series(array);
            };
            py_serie
        })
    }
    /// returns polar dataframe including channel
    fn get_polars_dataframe(&self, channel_name: String) -> Py<PyAny> {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|py| {
            let mut py_dataframe = Python::None(py);
            let channel_list = mdf.mdf_info.get_channel_names_cg_set(&channel_name);
            let series_dict = PyDict::new(py);
            for channel in channel_list {
                if let Some(channel_data) = mdf.get_channel_data(&channel) {
                    series_dict
                        .set_item(
                            channel.clone(),
                            rust_arrow_to_py_series(channel_data)
                                .expect("Could not convert to python series"),
                        )
                        .expect("could not store the serie in dict");
                }
            }
            if !series_dict.is_empty() {
                let locals = PyDict::new(py);
                locals
                    .set_item("series", series_dict)
                    .expect("cannot set python series_list");
                py.import("polars").expect("Could import polars");
                py.run(
                    r#"
import polars
df=polars.DataFrame(series)
"#,
                    None,
                    Some(locals),
                )
                .expect("dataframe creation failed");
                if let Ok(Some(df)) = locals.get_item("df") {
                    py_dataframe = df.into();
                }
            }
            py_dataframe
        })
    }
    /// returns channel's unit string
    fn get_channel_unit(&self, channel_name: String) -> PyResult<Py<PyAny>> {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|py| {
            let unit_or_error = mdf.mdf_info.get_channel_unit(&channel_name);
            match unit_or_error {
                Ok(unit) => Ok(unit.to_object(py)),
                Err(_) => Err(PyUnicodeDecodeError::new_err(
                    "Invalid UTF-8 sequence in metadata",
                )),
            }
        })
    }
    /// returns channel's description string
    fn get_channel_desc(&self, channel_name: String) -> PyResult<Py<PyAny>> {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|py| {
            let desc_or_error = mdf.mdf_info.get_channel_desc(&channel_name);
            match desc_or_error {
                Ok(desc) => Ok(desc.to_object(py)),
                Err(_) => Err(PyUnicodeDecodeError::new_err(
                    "Invalid UTF-8 sequence in metadata",
                )),
            }
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
    pub fn load_channels_data_in_memory(&mut self, channel_names: HashSet<String>) -> PyResult<()> {
        let Mdfr(mdf) = self;
        mdf.load_channels_data_in_memory(channel_names)?;
        Ok(())
    }
    /// clear channels from memory
    pub fn clear_channel_data_from_memory(&mut self, channel_names: HashSet<String>) {
        let Mdfr(mdf) = self;
        mdf.clear_channel_data_from_memory(channel_names);
    }
    /// load all channels in memory
    pub fn load_all_channels_data_in_memory(&mut self) -> PyResult<()> {
        let Mdfr(mdf) = self;
        mdf.load_all_channels_data_in_memory()?;
        Ok(())
    }
    /// writes file
    pub fn write(&mut self, file_name: &str, compression: bool) -> PyResult<Mdfr> {
        let Mdfr(mdf) = self;
        Ok(Mdfr(mdf.write(file_name, compression)?))
    }
    /// Adds a new channel in memory (no file modification)
    /// Master must be a dict with keys name, type and flag
    pub fn add_channel(
        &mut self,
        channel_name: String,
        data: Py<PyAny>,
        master: MasterSignature,
        unit: Option<String>,
        description: Option<String>,
    ) {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|py| {
            let array = array_to_rust(data.as_ref(py))
                .expect("data modification failed, could not extract numpy array");
            mdf.add_channel(
                channel_name,
                array,
                master.master_channel,
                master.master_type,
                master.master_flag,
                unit,
                description,
            );
        })
    }
    /// defines channel's data in memory
    pub fn set_channel_data(&mut self, channel_name: &str, data: Py<PyAny>) {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|py| {
            let array = array_to_rust(data.as_ref(py))
                .expect("data modification failed, could not extract numpy array");
            mdf.set_channel_data(channel_name, array);
        })
    }
    /// Sets the channel's related master channel type in memory
    pub fn set_channel_master_type(&mut self, master_name: &str, master_type: u8) {
        let Mdfr(mdf) = self;
        mdf.set_channel_master_type(master_name, master_type);
    }
    /// Removes a channel in memory (no file modification)
    pub fn remove_channel(&mut self, channel_name: &str) {
        let Mdfr(mdf) = self;
        mdf.remove_channel(channel_name);
    }
    /// Renames a channel's name in memory
    pub fn rename_channel(&mut self, channel_name: &str, new_name: &str) {
        let Mdfr(mdf) = self;
        mdf.rename_channel(channel_name, new_name);
    }
    /// Sets the channel unit in memory
    pub fn set_channel_unit(&mut self, channel_name: &str, unit: &str) {
        let Mdfr(mdf) = self;
        mdf.set_channel_unit(channel_name, unit);
    }
    /// Sets the channel description in memory
    pub fn set_channel_desc(&mut self, channel_name: &str, desc: &str) {
        let Mdfr(mdf) = self;
        mdf.set_channel_desc(channel_name, desc);
    }
    /// list attachments
    pub fn list_attachments(&mut self) -> PyResult<String> {
        let Mdfr(mdf) = self;
        Ok(mdf.mdf_info.list_attachments())
    }
    /// get attachment blocks
    pub fn get_attachment_blocks(&mut self) -> Py<PyAny> {
        let Mdfr(mdf) = self;
        let atbs = mdf.mdf_info.get_attachement_blocks();
        pyo3::Python::with_gil(|py| {
            if let Some(at) = atbs {
                let atl = PyList::empty(py);
                for (position, atb) in at {
                    let atdict = PyDict::new(py);
                    let _ = atdict.set_item("position", position);
                    if let Ok(res) = mdf.mdf_info.get_tx(atb.at_tx_filename) {
                        let _ = atdict.set_item("tx_name", res);
                    }
                    if let Ok(res) = mdf.mdf_info.get_tx(atb.at_tx_mimetype) {
                        let _ = atdict.set_item("tx_mimetype", res);
                    }
                    let _ =
                        atdict.set_item("md_comment", mdf.mdf_info.get_comments(atb.at_md_comment));
                    let _ = atdict.set_item("flags", atb.at_flags);
                    let _ = atdict.set_item("creator_index", atb.at_creator_index);
                    let _ = atl.append(atdict);
                }
                atl.into()
            } else {
                py.None()
            }
        })
    }
    /// get embedded data in attachment
    pub fn get_attachment_embedded_data(&self, position: i64) -> Py<PyAny> {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|py| {
            if let Some(data) = mdf.mdf_info.get_attachment_embedded_data(position) {
                PyBytes::new(py, &data).into()
            } else {
                py.None()
            }
        })
    }
    /// list events
    pub fn list_events(&mut self) -> PyResult<String> {
        let Mdfr(mdf) = self;
        Ok(mdf.mdf_info.list_events())
    }
    /// get event blocks
    pub fn get_event_blocks(&mut self) -> Py<PyAny> {
        let Mdfr(mdf) = self;
        let evbs = mdf.mdf_info.get_event_blocks();
        pyo3::Python::with_gil(|py| {
            if let Some(ev) = evbs {
                let evl = PyList::empty(py);
                for (_position, evb) in ev {
                    let evdict = PyDict::new(py);
                    if let Ok(res) = mdf.mdf_info.get_tx(evb.ev_tx_name) {
                        let _ = evdict.set_item("tx_name", res);
                    }
                    let _ =
                        evdict.set_item("md_comment", mdf.mdf_info.get_comments(evb.ev_md_comment));
                    let _ = evdict.set_item("type", evb.ev_type);
                    let _ = evdict.set_item("sync_type", evb.ev_sync_type);
                    let _ = evdict.set_item("range_type", evb.ev_range_type);
                    let _ = evl.append(evdict);
                }
                evl.into()
            } else {
                py.None()
            }
        })
    }
    /// get file history
    pub fn get_file_history_blocks(&mut self) -> Py<PyAny> {
        let Mdfr(mdf) = self;
        let fhbs = mdf.mdf_info.get_file_history_blocks();
        pyo3::Python::with_gil(|py| {
            if let Some(fh) = fhbs {
                let fhl = PyList::empty(py);
                for fhb in fh {
                    let fhdict: &PyDict = PyDict::new(py);
                    let _ =
                        fhdict.set_item("comment", mdf.mdf_info.get_comments(fhb.fh_md_comment));
                    let _ = fhdict.set_item("time_ns", fhb.fh_time_ns);
                    let _ = fhdict.set_item("tz_offset_min", fhb.fh_tz_offset_min);
                    let _ = fhdict.set_item("dst_offset_min", fhb.fh_dst_offset_min);
                    let _ = fhdict.set_item("time_flags", fhb.fh_time_flags);
                    let _ = fhl.append(fhdict);
                }
                fhl.into()
            } else {
                py.None()
            }
        })
    }
    /// plot one channel
    pub fn plot(&self, channel_name: String) {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|py| {
            let locals = PyDict::new(py);
            locals
                .set_item("channel_name", &channel_name)
                .expect("cannot set python channel_name");
            locals
                .set_item(
                    "channel_unit",
                    mdf.mdf_info.get_channel_unit(&channel_name).unwrap_or(None),
                )
                .expect("cannot set python channel_unit");
            if let Some(master_name) = mdf.mdf_info.get_channel_master(&channel_name) {
                locals
                    .set_item("master_channel_name", &master_name)
                    .expect("cannot set python master_channel_name");
                locals
                    .set_item(
                        "master_channel_unit",
                        mdf.mdf_info.get_channel_unit(&master_name).unwrap_or(None),
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
                .expect("Could not import matplotlib");
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
    fn __repr__(&mut self) -> PyResult<String> {
        let mut output: String;
        match &mut self.0.mdf_info {
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
                        writeln!(output, "\nMaster: {master_name}")
                            .expect("cannot print master channel name");
                    } else {
                        writeln!(output, "\nWithout Master channel")
                            .expect("cannot print thre is no master channel");
                    }
                    for channel in list.iter() {
                        let unit = self.get_channel_unit(channel.to_string())?;
                        let desc = self.get_channel_desc(channel.to_string())?;
                        write!(output, " {channel} ").expect("cannot print channel name");
                        if let Some(data) = self.0.get_channel_data(channel) {
                            if !data.is_empty() {
                                let displayer = get_display(data.as_ref(), "null");
                                displayer(&mut output, 0).expect("cannot channel data");
                                write!(output, " ").expect("cannot print simple space character");
                                displayer(&mut output, data.len() - 1)
                                    .expect("cannot channel data");
                            }
                            writeln!(
                                output,
                                " {unit} {desc} "
                            ).expect("cannot print channel unit and description with first and last item");
                        } else {
                            writeln!(output, " {unit} {desc} ")
                                .expect("cannot print channel unit and description");
                        }
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
                        writeln!(output, "\nMaster: {master_name}")
                            .expect("cannot print master channel name");
                    } else {
                        writeln!(output, "\nWithout Master channel")
                            .expect("cannot print thre is no master channel");
                    }
                    for channel in list.iter() {
                        let unit = self.get_channel_unit(channel.to_string())?;
                        let desc = self.get_channel_desc(channel.to_string())?;
                        write!(output, " {channel} ").expect("cannot print channel name");
                        if let Some(data) = self.0.get_channel_data(channel) {
                            if !data.is_empty() {
                                let displayer = get_display(data.as_ref(), "null");
                                displayer(&mut output, 0).expect("cannot print channel data");
                                write!(output, " .. ")
                                    .expect("cannot print simple space character");
                                displayer(&mut output, data.len() - 1)
                                    .expect("cannot channel data");
                            }
                            writeln!(
                                output,
                                " {unit} {desc} "
                            ).expect("cannot print channel unit and description with first and last item");
                        } else {
                            writeln!(output, " {unit} {desc} ")
                                .expect("cannot print channel unit and description");
                        }
                    }
                }
                output.push_str("\n ");
            }
        }
        Ok(output)
    }
}
