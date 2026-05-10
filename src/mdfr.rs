//! This module provides python interface using pyo3s
use std::collections::HashSet;
use std::fmt::Write;

use crate::data_holder::channel_data::ChannelData;

use crate::mdfinfo::{ChannelsDb, MdfInfo};
use crate::mdfreader::MasterSignature;
use crate::mdfreader::Mdf;
use anyhow::Context;
use arrow::array::ArrayData;
use arrow::pyarrow::PyArrowType;
use arrow::util::display::{ArrayFormatter, FormatOptions};

use crate::export::numpy::array_to_rust;
#[cfg(feature = "polars")]
use crate::export::polars::rust_arrow_to_py_series;
use crate::mdfinfo::mdfinfo4::CompressionAlgorithm;
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
    m.add_class::<CompressionAlgorithm>()?;
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
    /// returns true if the file was marked as unfinalized
    pub fn is_unfinalized(&self) -> bool {
        let Mdfr(mdf) = self;
        mdf.is_unfinalized()
    }
    /// returns the standard and custom unfinalization flags (0, 0) if finalized or MDF3
    pub fn get_unfin_flags(&self) -> (u16, u16) {
        let Mdfr(mdf) = self;
        mdf.get_unfin_flags()
    }
    /// Index of this file within a recording sequence (MDF 4.3 common_properties).
    pub fn get_recorder_sequence_index(&self) -> Option<u64> {
        let Mdfr(mdf) = self;
        mdf.get_recorder_sequence_index()
    }
    /// Index of this file within the recorder's file set (MDF 4.3 common_properties).
    pub fn get_recorder_file_index(&self) -> Option<u64> {
        let Mdfr(mdf) = self;
        mdf.get_recorder_file_index()
    }
    /// True if this is the last file in the recorder sequence (MDF 4.3 common_properties).
    pub fn get_recorder_file_last(&self) -> Option<bool> {
        let Mdfr(mdf) = self;
        mdf.get_recorder_file_last()
    }
    /// UUID of the recorder device (MDF 4.3 common_properties).
    pub fn get_recorder_uuid(&self) -> Option<String> {
        let Mdfr(mdf) = self;
        mdf.get_recorder_uuid()
    }
    /// UUID identifying the measurement (MDF 4.3 common_properties).
    pub fn get_measurement_uuid(&self) -> Option<String> {
        let Mdfr(mdf) = self;
        mdf.get_measurement_uuid()
    }
    /// Author from HD common_properties (MDF 4.3).
    pub fn get_author(&self) -> Option<String> {
        let Mdfr(mdf) = self;
        mdf.get_author()
    }
    /// Department from HD common_properties (MDF 4.3).
    pub fn get_department(&self) -> Option<String> {
        let Mdfr(mdf) = self;
        mdf.get_department()
    }
    /// Project from HD common_properties (MDF 4.3).
    pub fn get_project(&self) -> Option<String> {
        let Mdfr(mdf) = self;
        mdf.get_project()
    }
    /// Subject from HD common_properties (MDF 4.3).
    pub fn get_subject(&self) -> Option<String> {
        let Mdfr(mdf) = self;
        mdf.get_subject()
    }
    /// returns channel's data, numpy array or list, depending if data type is numeric or string|bytes
    fn get_channel_data(&self, channel_name: String) -> PyResult<Py<PyAny>> {
        let Mdfr(mdf) = self;
        // default py_array value is python None
        let data = pyo3::Python::attach(|py| -> PyResult<Py<PyAny>> {
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
                    && let Some(dg) = mdfinfo3.dg.get(dg_pos)
                    && let Some(cg) = dg.cg.get(rec_id)
                    && let Some(cn) = cg.cn.get(cn_pos)
                {
                    data = Some(&cn.data);
                }
            }
            MdfInfo::V4(mdfinfo4) => {
                if let Some((_master, dg_pos, (_cg_pos, rec_id), (_cn_pos, rec_pos))) =
                    mdfinfo4.get_channel_id(&channel_name)
                    && let Some(dg) = mdfinfo4.dg.get(dg_pos)
                    && let Some(cg) = dg.cg.get(rec_id)
                    && let Some(cn) = cg.cn.get(rec_pos)
                {
                    data = Some(&cn.data);
                }
            }
        };
        pyo3::Python::attach(|py| {
            Ok(data
                .map(super::data_holder::channel_data::ChannelData::get_dtype)
                .into_pyobject(py)
                .context("error converting dtype into python object")?
                .into())
        })
    }
    /// returns polars serie of channel
    #[cfg(feature = "polars")]
    fn get_polars_series(&self, channel_name: &str) -> PyResult<Py<PyAny>> {
        let Mdfr(mdf) = self;
        pyo3::Python::attach(|py| {
            let mut py_serie = Ok(Python::None(py));
            if let Some(array) = mdf.get_channel_data(channel_name) {
                py_serie = rust_arrow_to_py_series(array.as_ref(), channel_name);
            };
            py_serie
        })
    }
    /// returns polar dataframe including channel
    #[cfg(feature = "polars")]
    fn get_polars_dataframe(&self, channel_name: String) -> PyResult<Py<PyAny>> {
        let Mdfr(mdf) = self;
        Python::attach(|py| {
            let mut py_dataframe = Python::None(py);
            let channel_list = mdf.mdf_info.get_channel_names_cg_set(&channel_name);
            let series_dict = PyDict::new(py);
            for channel in channel_list {
                if let Some(channel_data) = mdf.get_channel_data(&channel) {
                    series_dict
                        .set_item(
                            channel.clone(),
                            rust_arrow_to_py_series(channel_data.as_ref(), &channel)
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
                    c_str!(
                        r#"
import polars
df=polars.DataFrame(series)
"#
                    ),
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
        pyo3::Python::attach(|_py| {
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
        pyo3::Python::attach(|_py| {
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
        pyo3::Python::attach(|py| {
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
        pyo3::Python::attach(|py| {
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
        pyo3::Python::attach(|py| {
            let channel_list: Py<PyAny> = mdf
                .mdf_info
                .get_channel_names_set()
                .into_pyobject(py)
                .context("error converting channel names set into python object")?
                .into();
            Ok(channel_list)
        })
    }
    /// returns the set of channel names that are in the same channel group as input channel name
    pub fn get_channel_names_cg_set(&self, channel_name: String) -> PyResult<Py<PyAny>> {
        let Mdfr(mdf) = self;
        pyo3::Python::attach(|py| {
            let channel_list: Py<PyAny> = mdf
                .mdf_info
                .get_channel_names_cg_set(&channel_name)
                .into_pyobject(py)
                .context("error converting channel group names set into python object")?
                .into();
            Ok(channel_list)
        })
    }
    /// returns dict mapping channel name → (master_name, dg_pos, cg_pos, rec_id, cn_pos, rec_pos)
    pub fn get_channels_db(&self) -> ChannelsDb {
        let Mdfr(mdf) = self;
        mdf.get_channels_db()
    }
    /// returns a dict of master names keys for which values are a set of associated channel names
    pub fn get_master_channel_names_set(&self) -> PyResult<Py<PyAny>> {
        let Mdfr(mdf) = self;
        pyo3::Python::attach(|py| {
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
    pub fn write(&mut self, file_name: &str, compression: CompressionAlgorithm) -> PyResult<Mdfr> {
        let Mdfr(mdf) = self;
        Ok(Mdfr(mdf.write(file_name, compression)?))
    }
    /// converts MDF version 3.x to 4.2 in memory
    pub fn convert3to4(&mut self, file_name: &str) -> PyResult<()> {
        let Mdfr(mdf) = self;
        let converted = mdf.mdf_info.convert3to4(file_name)?;
        mdf.mdf_info = converted;
        Ok(())
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
        pyo3::Python::attach(|_| -> Result<(), PyErr> {
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
        pyo3::Python::attach(|_| {
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
        pyo3::Python::attach(|py| {
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
                    let _ = atdict.set_item(
                        "md_comment",
                        mdf.mdf_info
                            .get_md_comment(atb.at_md_comment)
                            .map(|c| format!("{c}")),
                    );
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
        pyo3::Python::attach(|py| {
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
        pyo3::Python::attach(|py| {
            if let Some(ev) = evbs {
                let evl = PyList::empty(py);
                for (_position, evb) in ev {
                    let evdict = PyDict::new(py);
                    if let Ok(res) = mdf.mdf_info.get_tx(evb.ev_tx_name) {
                        let _ = evdict.set_item("tx_name", res);
                    }
                    let _ = evdict.set_item(
                        "md_comment",
                        mdf.mdf_info
                            .get_md_comment(evb.ev_md_comment)
                            .map(|c| format!("{c}")),
                    );
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
    /// list file history entries (MDF 4.x only)
    pub fn list_file_history(&mut self) -> PyResult<String> {
        let Mdfr(mdf) = self;
        Ok(mdf.mdf_info.list_file_history())
    }
    /// get file history
    pub fn get_file_history_blocks(&mut self) -> Py<PyAny> {
        let Mdfr(mdf) = self;
        let fhbs = mdf.mdf_info.get_file_history_blocks();
        pyo3::Python::attach(|py| {
            if let Some(fh) = fhbs {
                let fhl = PyList::empty(py);
                for fhb in fh {
                    let fhdict = PyDict::new(py);
                    let _ = fhdict.set_item(
                        "comment",
                        mdf.mdf_info
                            .get_md_comment(fhb.fh_md_comment)
                            .map(|c| format!("{c}")),
                    );
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
    /// list channel hierarchy in a human-readable format (MDF 4.x only)
    pub fn list_channel_hierarchy(&mut self) -> PyResult<String> {
        let Mdfr(mdf) = self;
        Ok(mdf.mdf_info.list_channel_hierarchy())
    }
    /// list source information blocks (MDF 4.x only)
    pub fn list_source_information(&self) -> PyResult<String> {
        let Mdfr(mdf) = self;
        Ok(mdf.mdf_info.list_source_information())
    }
    /// list sample reduction blocks for all channel groups (MDF 4.x only)
    pub fn list_sample_reductions(&self) -> PyResult<String> {
        let Mdfr(mdf) = self;
        Ok(mdf.mdf_info.list_sample_reductions())
    }
    /// get source information blocks (MDF 4.x only)
    pub fn get_source_information_blocks(&mut self) -> Py<PyAny> {
        let Mdfr(mdf) = self;
        let sibs = mdf.mdf_info.get_source_information_blocks();
        pyo3::Python::attach(|py| {
            if let Some(si_map) = sibs {
                let sil = PyList::empty(py);
                for (position, sib) in si_map {
                    let sidict = PyDict::new(py);
                    let _ = sidict.set_item("position", position);
                    if let Ok(res) = mdf.mdf_info.get_tx(sib.si_tx_name) {
                        let _ = sidict.set_item("name", res);
                    }
                    if let Ok(res) = mdf.mdf_info.get_tx(sib.si_tx_path) {
                        let _ = sidict.set_item("path", res);
                    }
                    let _ = sidict.set_item(
                        "comment",
                        mdf.mdf_info
                            .get_md_comment(sib.si_md_comment)
                            .map(|c| format!("{c}")),
                    );
                    let _ = sidict.set_item("type", sib.get_type_str());
                    let _ = sidict.set_item("type_id", sib.si_type);
                    let _ = sidict.set_item("bus_type", sib.get_bus_type_str());
                    let _ = sidict.set_item("bus_type_id", sib.si_bus_type);
                    let _ = sidict.set_item("flags", sib.si_flags);
                    let _ = sil.append(sidict);
                }
                sil.into()
            } else {
                py.None()
            }
        })
    }
    /// get sample reduction blocks (MDF 4.x only)
    pub fn get_sample_reduction_blocks(&self) -> Py<PyAny> {
        let Mdfr(mdf) = self;
        let srbs = mdf.mdf_info.get_sample_reduction_blocks();
        pyo3::Python::attach(|py| {
            if let Some(sr_list) = srbs {
                let srl = PyList::empty(py);
                for (dg_pos, rec_id, sr_blocks) in sr_list {
                    for (i, sr) in sr_blocks.iter().enumerate() {
                        let srdict = PyDict::new(py);
                        let _ = srdict.set_item("dg_position", dg_pos);
                        let _ = srdict.set_item("rec_id", rec_id);
                        let _ = srdict.set_item("index", i);
                        let _ = srdict.set_item("cycle_count", sr.sr_cycle_count);
                        let _ = srdict.set_item("interval", sr.sr_interval);
                        let _ = srdict.set_item("sync_type", sr.get_sync_type_str());
                        let _ = srdict.set_item("sync_type_id", sr.sr_sync_type);
                        let _ = srdict.set_item("flags", sr.sr_flags);
                        let _ = srl.append(srdict);
                    }
                }
                srl.into()
            } else {
                py.None()
            }
        })
    }
    /// get channel hierarchy blocks (MDF 4.x only)
    pub fn get_channel_hierarchy_blocks(&mut self) -> Py<PyAny> {
        let Mdfr(mdf) = self;
        let chbs = mdf.mdf_info.get_channel_hierarchy_blocks();
        pyo3::Python::attach(|py| {
            if let Some(ch) = chbs {
                let chl = PyList::empty(py);
                for (position, chb) in ch {
                    let chdict = PyDict::new(py);
                    let _ = chdict.set_item("position", position);
                    if let Ok(res) = mdf.mdf_info.get_tx(chb.ch_tx_name) {
                        let _ = chdict.set_item("name", res);
                    }
                    let _ = chdict.set_item(
                        "comment",
                        mdf.mdf_info
                            .get_md_comment(chb.ch_md_comment)
                            .map(|c| format!("{c}")),
                    );
                    let _ = chdict.set_item("type", chb.get_type_str());
                    let _ = chdict.set_item("type_id", chb.ch_type);
                    let _ = chdict.set_item("element_count", chb.ch_element_count);
                    let _ = chdict.set_item("first_child", chb.ch_ch_first);
                    let _ = chdict.set_item("next_sibling", chb.ch_ch_next);
                    // Elements as list of (DG, CG, CN) triplets
                    let elements = PyList::empty(py);
                    for i in 0..chb.ch_element_count as usize {
                        let base_idx = i * 3;
                        if base_idx + 2 < chb.ch_element.len() {
                            let triplet = PyDict::new(py);
                            let _ = triplet.set_item("dg", chb.ch_element[base_idx]);
                            let _ = triplet.set_item("cg", chb.ch_element[base_idx + 1]);
                            let _ = triplet.set_item("cn", chb.ch_element[base_idx + 2]);
                            let _ = elements.append(triplet);
                        }
                    }
                    let _ = chdict.set_item("elements", elements);
                    let _ = chl.append(chdict);
                }
                chl.into()
            } else {
                py.None()
            }
        })
    }
    /// plot one channel
    pub fn plot(&self, channel_name: String) -> PyResult<()> {
        let Mdfr(mdf) = self;
        pyo3::Python::attach(|py| -> PyResult<()> {
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
                c_str!(
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
"#
                ),
                None,
                Some(&locals),
            )
            .context("plot python script failed")?;
            Ok(())
        })
    }
    /// display a representation of mdfinfo object content
    fn __repr__(&mut self) -> PyResult<String> {
        let mut output = String::new();
        let format_option = FormatOptions::new();

        match &mut self.0.mdf_info {
            MdfInfo::V3(mdfinfo3) => {
                // Use helper methods for header
                writeln!(output, "{}", mdfinfo3.summary()).context("cannot print summary")?;
                writeln!(output, "{}", mdfinfo3.format_header()).context("cannot print header")?;
            }
            MdfInfo::V4(mdfinfo4) => {
                // Use helper methods for header
                writeln!(output, "{}", mdfinfo4.summary()).context("cannot print summary")?;
                writeln!(output, "{}", mdfinfo4.hd_block).context("cannot print header block")?;
                let header_comments = mdfinfo4.format_header_comments();
                if !header_comments.is_empty() {
                    write!(output, "{header_comments}").context("cannot print header comments")?;
                }
                // MDF4-specific sections
                let si_info = mdfinfo4.list_source_information();
                if !si_info.is_empty() {
                    writeln!(output, "\n--- Source Information ---")
                        .context("cannot print source info header")?;
                    write!(output, "{si_info}").context("cannot print source information")?;
                }
                let at_info = mdfinfo4.list_attachments();
                if !at_info.is_empty() {
                    writeln!(output, "\n--- Attachments ---")
                        .context("cannot print attachments header")?;
                    write!(output, "{at_info}").context("cannot print attachments")?;
                }
                let ev_info = mdfinfo4.list_events();
                if !ev_info.is_empty() {
                    writeln!(output, "\n--- Events ---").context("cannot print events header")?;
                    write!(output, "{ev_info}").context("cannot print events")?;
                }
                let ch_info = mdfinfo4.list_channel_hierarchy();
                if !ch_info.is_empty() {
                    writeln!(output, "\n--- Channel Hierarchy ---")
                        .context("cannot print channel hierarchy header")?;
                    write!(output, "{ch_info}").context("cannot print channel hierarchy")?;
                }
            }
        }

        // Channels section (common for both versions, with data preview)
        writeln!(output, "\n--- Channels ---").context("cannot print channels header")?;
        for (master, list) in &self.0.mdf_info.get_master_channel_names_set() {
            if let Some(master_name) = master {
                writeln!(output, "\nMaster: {master_name}")
                    .context("cannot print master channel name")?;
            } else {
                writeln!(output, "\nWithout Master channel")
                    .context("cannot print no master channel")?;
            }
            for channel in list {
                let unit = self
                    .get_channel_unit(channel.to_string())
                    .context("failed getting channel unit")?
                    .unwrap_or_default();
                let desc = self
                    .get_channel_desc(channel.to_string())
                    .context("failed getting channel description")?
                    .unwrap_or_default();
                write!(output, "  {channel} ").context("cannot print channel name")?;
                // Data preview (first .. last values)
                if let Some(data) = self.0.get_channel_data(channel)
                    && !data.is_empty()
                {
                    write!(output, "[{}] ", data.len()).context("cannot print data length")?;
                    let array = &data.as_ref();
                    if let Ok(displayer) = ArrayFormatter::try_new(array, &format_option) {
                        write!(
                            output,
                            "{} .. {} ",
                            displayer.value(0),
                            displayer.value(data.len() - 1)
                        )
                        .context("cannot print data preview")?;
                    }
                }
                if !unit.is_empty() {
                    write!(output, "\"{unit}\" ").context("cannot print unit")?;
                }
                if !desc.is_empty() {
                    write!(output, "// {desc}").context("cannot print description")?;
                }
                writeln!(output).context("cannot print newline")?;
            }
        }

        Ok(output)
    }
}
