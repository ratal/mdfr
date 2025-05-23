//! This module provides python interface using pyo3s
use std::collections::HashSet;
use std::fmt::Write;

use crate::data_holder::channel_data::ChannelData;

use crate::mdfinfo::MdfInfo;
use crate::mdfreader::MasterSignature;
use crate::mdfreader::Mdf;
use anyhow::Context;
use arrow::array::ArrayData;
use arrow::pyarrow::PyArrowType;
use arrow::util::display::{ArrayFormatter, FormatOptions};

use crate::export::numpy::array_to_rust;
#[cfg(feature = "polars")]
use crate::export::polars::rust_arrow_to_py_series;
use pyo3::exceptions::PyUnicodeDecodeError;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyBytes, PyDict, PyList};

#[pymodule]
fn mdfr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register(m)?;
    Ok(())
}

/// This function is used to create a python dictionary from a MdfInfo object
#[pyclass]
struct Mdfr(Mdf);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    fn get_channel_data(&self, channel_name: String) -> PyResult<Py<PyAny>> {
        let Mdfr(mdf) = self;
        // default py_array value is python None
        let data = pyo3::Python::with_gil(|py| -> PyResult<Py<PyAny>> {
            let mut py_array: Py<PyAny>;
            let dt = mdf.get_channel_data(&channel_name);
            if let Some(data) = dt {
                py_array = data
                    .clone()
                    .into_pyobject(py)
                    .context("error converting ChannelData into python object")?
                    .into();
                if let Some(m) = data.clone().validity() {
                    let mask: Py<PyAny> = m
                        .iter()
                        .collect::<Vec<bool>>()
                        .into_pyobject(py)
                        .context("error converting validity into python object")?
                        .into();
                    let locals = [(
                        "numpy",
                        py.import("numpy").context("could not import numpy")?,
                    )]
                    .into_py_dict(py)
                    .context("error converting validity into dictionary")?;
                    locals
                        .set_item("py_array", &py_array)
                        .context("cannot set python data")?;
                    locals
                        .set_item("mask", mask)
                        .context("cannot set python mask")?;
                    py_array = py
                        .eval(
                            c_str!("numpy.ma.array(py_array, mask=mask)"),
                            None,
                            Some(&locals),
                        )
                        .context("masked array creation failed")?
                        .into_pyobject(py)
                        .context("error converting masked array into python object")?
                        .into();
                }
            } else {
                py_array = Python::None(py);
            }
            Ok(py_array)
        })?;
        Ok(data)
    }
    /// returns channel's numpy dtype
    fn get_channel_dtype(&self, channel_name: String) -> PyResult<Py<PyAny>> {
        let Mdfr(mdf) = self;
        let mut data: Option<&ChannelData> = None;
        // extract channelData, even empty but initialised
        match &mdf.mdf_info {
            MdfInfo::V3(mdfinfo3) => {
                if let Some((_master, dg_pos, (_cg_pos, rec_id), cn_pos)) =
                    mdfinfo3.get_channel_id(&channel_name)
                {
                    if let Some(dg) = mdfinfo3.dg.get(dg_pos) {
                        if let Some(cg) = dg.cg.get(rec_id) {
                            if let Some(cn) = cg.cn.get(cn_pos) {
                                data = Some(&cn.data);
                            }
                        }
                    }
                }
            }
            MdfInfo::V4(mdfinfo4) => {
                if let Some((_master, dg_pos, (_cg_pos, rec_id), (_cn_pos, rec_pos))) =
                    mdfinfo4.get_channel_id(&channel_name)
                {
                    if let Some(dg) = mdfinfo4.dg.get(dg_pos) {
                        if let Some(cg) = dg.cg.get(rec_id) {
                            if let Some(cn) = cg.cn.get(rec_pos) {
                                data = Some(&cn.data);
                            }
                        }
                    }
                }
            }
        };
        pyo3::Python::with_gil(|py| {
            Ok(data
                .map(|d| d.get_dtype())
                .into_pyobject(py)
                .context("error converting dtype into python object")?
                .into())
        })
    }
    /// returns polars serie of channel
    #[cfg(feature = "polars")]
    fn get_polars_series(&self, channel_name: &str) -> PyResult<PyObject> {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|py| {
            let mut py_serie = Ok(Python::None(py));
            if let Some(array) = mdf.get_channel_data(channel_name) {
                py_serie = rust_arrow_to_py_series(array.as_ref(), channel_name.to_string());
            };
            py_serie
        })
    }
    /// returns polar dataframe including channel
    #[cfg(feature = "polars")]
    fn get_polars_dataframe(&self, channel_name: String) -> PyResult<Py<PyAny>> {
        let Mdfr(mdf) = self;
        Python::with_gil(|py| {
            let mut py_dataframe = Python::None(py);
            let channel_list = mdf.mdf_info.get_channel_names_cg_set(&channel_name);
            let series_dict = PyDict::new(py);
            for channel in channel_list {
                if let Some(channel_data) = mdf.get_channel_data(&channel) {
                    series_dict
                        .set_item(
                            channel.clone(),
                            rust_arrow_to_py_series(channel_data.as_ref(), channel)
                                .context("Could not convert to python series")?,
                        )
                        .context("could not store the serie in dict")?;
                }
            }
            if !series_dict.is_empty() {
                let locals = PyDict::new(py);
                locals
                    .set_item("series", series_dict)
                    .context("cannot set python series_list")?;
                py.import("polars").context("Could import polars")?;
                py.run(
                    c_str!(r#"
import polars
df=polars.DataFrame(series)
"#),
                    None,
                    Some(&locals),
                )
                .context("dataframe creation failed")?;
                if let Ok(Some(df)) = locals.get_item("df") {
                    py_dataframe = df.into();
                }
            }
            Ok(py_dataframe)
        })
    }
    /// returns channel's unit string
    fn get_channel_unit(&self, channel_name: String) -> PyResult<Option<String>> {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|_py| {
            let unit_or_error = mdf.mdf_info.get_channel_unit(&channel_name);
            match unit_or_error {
                Ok(unit) => Ok(unit),
                Err(_) => Err(PyUnicodeDecodeError::new_err(
                    "Invalid UTF-8 sequence in metadata",
                )),
            }
        })
    }
    /// returns channel's description string
    fn get_channel_desc(&self, channel_name: String) -> PyResult<Option<String>> {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|_py| {
            let desc_or_error = mdf.mdf_info.get_channel_desc(&channel_name);
            match desc_or_error {
                Ok(desc) => Ok(desc),
                Err(_) => Err(PyUnicodeDecodeError::new_err(
                    "Invalid UTF-8 sequence in metadata",
                )),
            }
        })
    }
    /// returns channel's associated master channel name string
    pub fn get_channel_master(&self, channel_name: String) -> PyResult<Py<PyAny>> {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|py| {
            let master: Py<PyAny> = mdf
                .mdf_info
                .get_channel_master(&channel_name)
                .into_pyobject(py)
                .context("error converting channel master name into python object")?
                .into();
            Ok(master)
        })
    }
    /// returns channel's master data, numpy array or list, depending if data type is numeric or string|bytes
    fn get_channel_master_data(&mut self, channel_name: String) -> PyResult<Py<PyAny>> {
        // default py_array value is python None
        let master = self
            .get_channel_master(channel_name)
            .context("error getting master channel name")?;
        self.get_channel_data(master.to_string())
    }
    /// returns channel's associated master channel type string
    /// 0 = None (normal data channels), 1 = Time (seconds), 2 = Angle (radians),
    /// 3 = Distance (meters), 4 = Index (zero-based index values)
    pub fn get_channel_master_type(&self, channel_name: String) -> PyResult<Py<PyAny>> {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|py| {
            let master_type: Py<PyAny> = mdf
                .mdf_info
                .get_channel_master_type(&channel_name)
                .into_pyobject(py)
                .context("error converting channel master type into python object")?
                .into();
            Ok(master_type)
        })
    }
    /// returns a set of all channel names contained in file
    pub fn get_channel_names_set(&self) -> PyResult<Py<PyAny>> {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|py| {
            let channel_list: Py<PyAny> = mdf
                .mdf_info
                .get_channel_names_set()
                .into_pyobject(py)
                .context("error converting channel names set into python object")?
                .into();
            Ok(channel_list)
        })
    }
    /// returns a dict of master names keys for which values are a set of associated channel names
    pub fn get_master_channel_names_set(&self) -> PyResult<Py<PyAny>> {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|py| {
            let master_channel_list: Py<PyAny> = mdf
                .mdf_info
                .get_master_channel_names_set()
                .into_pyobject(py)
                .context("error converting master channel names set into python object")?
                .into();
            Ok(master_channel_list)
        })
    }
    /// load a set of channels in memory
    pub fn load_channels_data_in_memory(&mut self, channel_names: HashSet<String>) -> PyResult<()> {
        let Mdfr(mdf) = self;
        mdf.load_channels_data_in_memory(channel_names)?;
        Ok(())
    }
    /// clear channels from memory
    pub fn clear_channel_data_from_memory(
        &mut self,
        channel_names: HashSet<String>,
    ) -> PyResult<()> {
        let Mdfr(mdf) = self;
        mdf.clear_channel_data_from_memory(channel_names)?;
        Ok(())
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
    /// Data  has to be a PyArrow
    pub fn add_channel(
        &mut self,
        channel_name: String,
        data: PyArrowType<ArrayData>,
        master: MasterSignature,
        unit: Option<String>,
        description: Option<String>,
    ) -> PyResult<()> {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|_| -> Result<(), PyErr> {
            let array = array_to_rust(data)
                .context("data modification failed, could not extract numpy array")?;
            mdf.add_channel(
                channel_name,
                array,
                master.master_channel,
                master.master_type,
                master.master_flag,
                unit,
                description,
            )?;
            Ok(())
        })?;
        Ok(())
    }
    /// defines channel's data in memory from PyArrow
    pub fn set_channel_data(
        &mut self,
        channel_name: &str,
        data: PyArrowType<ArrayData>,
    ) -> PyResult<()> {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|_| {
            let array = array_to_rust(data)
                .expect("data modification failed, could not extract numpy array");
            mdf.set_channel_data(channel_name, array)?;
            Ok(())
        })
    }
    /// Sets the channel's related master channel type in memory
    pub fn set_channel_master_type(&mut self, master_name: &str, master_type: u8) -> PyResult<()> {
        let Mdfr(mdf) = self;
        mdf.set_channel_master_type(master_name, master_type)?;
        Ok(())
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
    /// export to Parquet files
    #[cfg(feature = "parquet")]
    pub fn export_to_parquet(&self, file_name: &str, compression: Option<&str>) -> PyResult<()> {
        let Mdfr(mdf) = self;
        mdf.export_to_parquet(file_name, compression)?;
        Ok(())
    }
    /// export dataframe to Parquet files
    #[cfg(feature = "parquet")]
    pub fn export_dataframe_to_parquet(
        &self,
        channel_name: String,
        file_name: &str,
        compression: Option<&str>,
    ) -> PyResult<()> {
        let Mdfr(mdf) = self;
        mdf.export_dataframe_to_parquet(channel_name, file_name, compression)?;
        Ok(())
    }
    /// export to hdf5 files
    #[cfg(feature = "hdf5")]
    pub fn export_to_hdf5(&self, file_name: &str, compression: Option<&str>) -> PyResult<()> {
        let Mdfr(mdf) = self;
        mdf.export_to_hdf5(file_name, compression)?;
        Ok(())
    }
    /// export dataframe to Parquet files
    #[cfg(feature = "hdf5")]
    pub fn export_dataframe_to_hdf5(
        &self,
        channel_name: String,
        file_name: &str,
        compression: Option<&str>,
    ) -> PyResult<()> {
        let Mdfr(mdf) = self;
        mdf.export_dataframe_to_hdf5(channel_name, file_name, compression)?;
        Ok(())
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
                    let fhdict = PyDict::new(py);
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
    pub fn plot(&self, channel_name: String) -> PyResult<()> {
        let Mdfr(mdf) = self;
        pyo3::Python::with_gil(|py| -> PyResult<()> {
            let locals = PyDict::new(py);
            locals
                .set_item("channel_name", &channel_name)
                .context("cannot set python channel_name")?;
            locals
                .set_item(
                    "channel_unit",
                    mdf.mdf_info.get_channel_unit(&channel_name).unwrap_or(None),
                )
                .context("cannot set python channel_unit")?;
            if let Some(master_name) = mdf.mdf_info.get_channel_master(&channel_name) {
                locals
                    .set_item("master_channel_name", &master_name)
                    .context("cannot set python master_channel_name")?;
                locals
                    .set_item(
                        "master_channel_unit",
                        mdf.mdf_info.get_channel_unit(&master_name).unwrap_or(None),
                    )
                    .context("cannot set python master_channel_unit")?;
                let data = self
                    .get_channel_data(master_name)
                    .context("failed getting master channel data")?;
                locals
                    .set_item("master_data", data)
                    .context("cannot set python master_data")?;
            } else {
                locals
                    .set_item("master_channel_name", py.None())
                    .context("cannot set python master_channel_name")?;
                locals
                    .set_item("master_channel_unit", py.None())
                    .context("cannot set python master_channel_unit")?;
                locals
                    .set_item("master_data", py.None())
                    .context("cannot set python master_data")?;
            }
            let data = self
                .get_channel_data(channel_name)
                .context("failed getting channel data")?;
            locals
                .set_item("channel_data", data)
                .context("cannot set python channel_data")?;
            py.import("matplotlib")
                .context("Could not import matplotlib")?;
            py.run(
                c_str!(r#"
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
"#),
                None,
                Some(&locals),
            )
            .context("plot python script failed")?;
            Ok(())
        })
    }
    /// display a representation of mdfinfo object content
    fn __repr__(&mut self) -> PyResult<String> {
        let mut output: String;
        let format_option = FormatOptions::new();
        match &mut self.0.mdf_info {
            MdfInfo::V3(mdfinfo3) => {
                output = format!("Version : {}\n", mdfinfo3.id_block.id_ver);
                writeln!(
                    output,
                    "Header :\n Author: {}  Organisation:{}",
                    mdfinfo3.hd_block.hd_author, mdfinfo3.hd_block.hd_organization
                )
                .context("cannot print author and organisation")?;
                writeln!(
                    output,
                    "Project: {}  Subject:{}",
                    mdfinfo3.hd_block.hd_project, mdfinfo3.hd_block.hd_subject
                )
                .context("cannot print project and subject")?;
                writeln!(
                    output,
                    "Date: {:?}  Time:{:?}",
                    mdfinfo3.hd_block.hd_date, mdfinfo3.hd_block.hd_time
                )
                .context("cannot print date and time")?;
                write!(output, "Comments: {}", mdfinfo3.hd_comment)
                    .context("cannot print comments")?;
                for (master, list) in mdfinfo3.get_master_channel_names_set().iter() {
                    if let Some(master_name) = master {
                        writeln!(output, "\nMaster: {master_name}")
                            .context("cannot print master channel name")?;
                    } else {
                        writeln!(output, "\nWithout Master channel")
                            .context("cannot print thre is no master channel")?;
                    }
                    for channel in list.iter() {
                        let unit = self
                            .get_channel_unit(channel.to_string())
                            .context("failed printing channel unit")?
                            .unwrap_or_default();
                        let desc = self
                            .get_channel_desc(channel.to_string())
                            .context("failed printing channel description")?
                            .unwrap_or_default();
                        write!(output, " {channel} ").context("cannot print channel name")?;
                        if let Some(data) = self.0.get_channel_data(channel) {
                            if !data.is_empty() {
                                let array = &data.as_ref();
                                let displayer = ArrayFormatter::try_new(array, &format_option)
                                    .context("failed creating formatter for arrow array")?;
                                write!(&mut output, "{}", displayer.value(0))
                                    .context("failed writing first value of array")?;
                                write!(output, " ")
                                    .context("cannot print simple space character")?;
                                write!(&mut output, "{}", displayer.value(data.len() - 1))
                                    .context("failed writing last value of array")?;
                            }
                            writeln!(
                                output,
                                " {unit:?} {desc:?} "
                            ).context("cannot print channel unit and description with first and last item")?;
                        } else {
                            writeln!(output, " {unit:?} {desc:?} ")
                                .context("cannot print channel unit and description")?;
                        }
                    }
                }
                output.push_str("\n ");
            }
            MdfInfo::V4(mdfinfo4) => {
                output = format!("Version : {}\n", mdfinfo4.id_block.id_ver);
                writeln!(output, "{}", mdfinfo4.hd_block).context("cannot print header block")?;
                let comments = &mdfinfo4
                    .sharable
                    .get_comments(mdfinfo4.hd_block.hd_md_comment);
                for c in comments.iter() {
                    writeln!(output, "{} {}", c.0, c.1).context("cannot print header comments")?;
                }
                for (master, list) in mdfinfo4.get_master_channel_names_set().iter() {
                    if let Some(master_name) = master {
                        writeln!(output, "\nMaster: {master_name}")
                            .context("cannot print master channel name")?;
                    } else {
                        writeln!(output, "\nWithout Master channel")
                            .context("cannot print thre is no master channel")?;
                    }
                    for channel in list.iter() {
                        let unit = self
                            .get_channel_unit(channel.to_string())
                            .context("failed printing channel unit")?
                            .unwrap_or_default();
                        let desc = self
                            .get_channel_desc(channel.to_string())
                            .context("failed printing channel description")?
                            .unwrap_or_default();
                        write!(output, " {channel} ").context("cannot print channel name")?;
                        if let Some(data) = self.0.get_channel_data(channel) {
                            if !data.is_empty() {
                                let array = &data.as_ref();
                                let displayer = ArrayFormatter::try_new(array, &format_option)
                                    .context("failed creating formatter for arrow array")?;
                                write!(&mut output, "{}", displayer.value(0))
                                    .context("cannot print channel data")?;
                                write!(output, " .. ")
                                    .context("cannot print simple space character")?;
                                write!(&mut output, "{}", displayer.value(data.len() - 1))
                                    .context("cannot channel data")?;
                            }
                            writeln!(
                                output,
                                " {unit:?} {desc:?} " 
                            ).context("cannot print channel unit and description with first and last item")?;
                        } else {
                            writeln!(output, " {unit:?} {desc:?} ")
                                .context("cannot print channel unit and description")?;
                        }
                    }
                }
                output.push_str("\n ");
            }
        }
        Ok(output)
    }
}
