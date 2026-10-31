//! This module provides python interface using pyo3s
use std::collections::HashSet;
use std::fmt::Write;
use std::sync::Arc;

use crate::data_holder::channel_data::ChannelData;

use crate::mdfinfo::{ChannelsDb, MdfInfo};
use crate::mdfreader::MasterSignature;
use crate::mdfreader::Mdf;
use anyhow::Context;
use arrow::array::{Array, ArrayData, RecordBatchReader};
use arrow::pyarrow::PyArrowType;
use arrow::util::display::{ArrayFormatter, FormatOptions};

use crate::export::numpy::array_to_rust;
#[cfg(feature = "polars")]
use crate::export::polars::rust_arrow_to_py_series;
use crate::mdfinfo::mdfinfo4::CompressionAlgorithm;
use pyo3::exceptions::PyUnicodeDecodeError;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

#[pymodule]
fn mdfr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register(m)?;
    Ok(())
}

/// Represents an MDF channel in Python with metadata and direct Arrow/NumPy/Polars integration.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PyChannel {
    pub(crate) name: String,
    pub(crate) data: ChannelData,
    pub(crate) unit: Option<String>,
    pub(crate) desc: Option<String>,
    pub(crate) master_name: Option<String>,
}

#[pymethods]
impl PyChannel {
    /// Channel name
    #[getter]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Engineering unit (e.g. "km/h", "bar")
    #[getter]
    pub fn unit(&self) -> Option<&str> {
        self.unit.as_deref()
    }

    /// Channel description / comment
    #[getter]
    pub fn comment(&self) -> Option<&str> {
        self.desc.as_deref()
    }

    /// Channel description (alias for comment)
    #[getter]
    pub fn desc(&self) -> Option<&str> {
        self.desc.as_deref()
    }

    /// Associated master channel name, if any
    #[getter]
    pub fn master(&self) -> Option<&str> {
        self.master_name.as_deref()
    }

    /// Number of samples in the channel
    pub fn __len__(&self) -> usize {
        self.data.len()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "<PyChannel '{}' len={} unit={:?}>",
            self.name,
            self.data.len(),
            self.unit
        )
    }

    /// Converts channel data to a NumPy ndarray (or MaskedArray if invalid samples exist).
    pub fn to_numpy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let py_obj = self
            .data
            .clone()
            .into_pyobject(py)
            .context("error converting ChannelData into python numpy object")?;
        let mut py_array: Py<PyAny> = py_obj.into();
        if let Some(m) = self.data.validity() {
            let mask: Py<PyAny> = m
                .iter()
                .collect::<Vec<bool>>()
                .into_pyobject(py)
                .context("error converting validity into python object")?
                .into();
            let np_ma = py.import("numpy.ma").context("could not import numpy.ma")?;
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs
                .set_item(pyo3::intern!(py, "mask"), mask)
                .context("cannot set python mask")?;
            py_array = np_ma
                .getattr(pyo3::intern!(py, "array"))
                .context("could not get numpy.ma.array")?
                .call((&py_array,), Some(&kwargs))
                .context("masked array creation failed")?
                .unbind();
        }
        Ok(py_array)
    }

    /// NumPy array protocol: np.asarray(channel)
    #[pyo3(signature = (dtype=None, copy=None))]
    pub fn __array__(
        &self,
        py: Python<'_>,
        dtype: Option<&Bound<'_, PyAny>>,
        copy: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let np_arr = self.to_numpy(py)?;
        let bound = np_arr.bind(py);
        if dtype.is_some() || copy.is_some() {
            let np = py.import("numpy")?;
            let kwargs = pyo3::types::PyDict::new(py);
            if let Some(d) = dtype {
                kwargs.set_item("dtype", d)?;
            }
            if let Some(c) = copy {
                kwargs.set_item("copy", c)?;
            }
            Ok(np.call_method("asarray", (bound,), Some(&kwargs))?.unbind())
        } else {
            Ok(np_arr)
        }
    }

    /// Indexing and slicing: ch[0], ch[-1], ch[10:20]
    pub fn __getitem__(&self, py: Python<'_>, index: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let np_arr = self.to_numpy(py)?;
        np_arr.bind(py).get_item(index).map(|b| b.unbind())
    }

    /// Arrow PyCapsule zero-copy Array protocol: pa.array(channel), pl.Series("name", channel)
    #[pyo3(signature = (requested_schema=None))]
    #[allow(unused_variables)]
    pub fn __arrow_c_array__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let arr_data = self.data.to_data();
        crate::export::arrow_capsule::export_array_to_pycapsules(py, &arr_data)
    }

    /// Arrow PyCapsule Stream protocol: pa.table(channel), pl.DataFrame(channel)
    #[pyo3(signature = (requested_schema=None))]
    #[allow(unused_variables)]
    pub fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let arr = self.data.as_arrow();
        let field = arrow::datatypes::Field::new(&self.name, arr.data_type().clone(), true);
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![field]));
        let batch = arrow::array::RecordBatch::try_new(schema.clone(), vec![arr])
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let reader = Box::new(arrow::array::RecordBatchIterator::new(vec![Ok(batch)], schema));
        crate::export::arrow_capsule::export_reader_to_pycapsule(py, reader)
    }

    /// Converts channel to a Polars Series using the Arrow PyCapsule protocol zero-copy.
    #[cfg(feature = "polars")]
    pub fn to_polars(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let polars = py.import("polars")?;
        polars.call_method1("Series", (self.name.as_str(), self.clone())).map(|b| b.unbind())
    }
}

/// This function is used to create a python dictionary from a MdfInfo object
#[pyclass]
pub struct Mdfr(Mdf);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Mdfr>()?;
    m.add_class::<PyChannel>()?;
    m.add_class::<CompressionAlgorithm>()?;
    m.add_class::<MdfLogger>()?;
    m.add_function(wrap_pyfunction!(finalize_file_in_place, m)?)?;
    m.add_function(wrap_pyfunction!(write_polars, m)?)?;
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
            let dt = mdf.get_channel_converted_data(&channel_name);
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
                    let np_ma = py.import("numpy.ma").context("could not import numpy.ma")?;
                    let kwargs = pyo3::types::PyDict::new(py);
                    kwargs
                        .set_item(pyo3::intern!(py, "mask"), mask)
                        .context("cannot set python mask")?;
                    py_array = np_ma
                        .getattr(pyo3::intern!(py, "array"))
                        .context("could not get numpy.ma.array")?
                        .call((&py_array,), Some(&kwargs))
                        .context("masked array creation failed")?
                        .unbind();
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
    fn get_polars_series(&mut self, channel_name: &str) -> PyResult<Py<PyAny>> {
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
    fn get_polars_dataframe(&mut self, channel_name: String) -> PyResult<Py<PyAny>> {
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
                let polars = py.import("polars").context("Could not import polars")?;
                let df = polars
                    .getattr(pyo3::intern!(py, "DataFrame"))
                    .context("could not get polars.DataFrame")?
                    .call1((series_dict,))
                    .context("dataframe creation failed")?;
                py_dataframe = df.unbind();
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
        let master_opt = {
            let Mdfr(mdf) = self;
            mdf.mdf_info.get_channel_master(&channel_name).map(|s| s.to_string())
        };
        match master_opt {
            Some(master) => self.get_channel_data(master),
            None => pyo3::Python::attach(|py| Ok(py.None())),
        }
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
    /// Returns the path separator character for the channel group containing the given channel (MDF 4.x only).
    /// The path separator is a UTF-16 LE encoded character (2 bytes) used to separate parts
    /// in group name (gn), group source (gs), group path (gp), channel name (cn),
    /// channel source (cs), and channel path (cp).
    /// Common values: 0x002F (47) for '/', 0x005C (92) for '\\', 0 if not specified.
    /// Returns the raw u16 value as an integer, or None if not found/MDF3.
    pub fn get_channel_group_path_separator(&self, channel_name: String) -> PyResult<Option<u16>> {
        let Mdfr(mdf) = self;
        pyo3::Python::attach(|_py| Ok(mdf.mdf_info.get_channel_group_path_separator(&channel_name)))
    }
    /// Returns event signal information for an event signal channel as a dict.
    /// Returns None if the channel is not an event signal channel.
    pub fn get_event_signal_info(&self, channel_name: String) -> PyResult<Py<PyAny>> {
        let Mdfr(mdf) = self;
        pyo3::Python::attach(|py| {
            let info = mdf.mdf_info.get_event_signal_info(&channel_name);
            match info {
                Some(ev) => {
                    let dict = pyo3::types::PyDict::new(py);
                    dict.set_item("event_type", ev.ev_type)?;
                    dict.set_item("event_type_str", ev.get_event_type_str())?;
                    dict.set_item("sync_type", ev.ev_sync_type)?;
                    dict.set_item("sync_type_str", ev.get_sync_type_str())?;
                    dict.set_item("cause", ev.ev_cause)?;
                    dict.set_item("cause_str", ev.get_cause_str())?;
                    dict.set_item("sync_base_value", ev.ev_sync_base_value)?;
                    dict.set_item("sync_factor", ev.ev_sync_factor)?;
                    dict.set_item("scope_count", ev.ev_scope_count)?;
                    dict.set_item("attachment_count", ev.ev_attachment_count)?;
                    dict.set_item("sync_value", ev.get_sync_value())?;
                    Ok(dict.into())
                }
                None => Ok(py.None()),
            }
        })
    }
    /// Returns true if the channel is a synchronization channel.
    /// Sync channels reference an ATBLOCK (attachment) rather than containing raw data.
    pub fn is_sync_channel(&self, channel_name: String) -> PyResult<bool> {
        let Mdfr(mdf) = self;
        Ok(mdf.mdf_info.is_sync_channel(&channel_name))
    }
    /// Returns the precision of a channel if the precision flag is set.
    /// Returns None if the channel is not found or precision is not specified.
    pub fn get_channel_precision(&self, channel_name: String) -> PyResult<Option<u8>> {
        let Mdfr(mdf) = self;
        Ok(mdf.get_channel_precision(&channel_name))
    }
    /// Returns the sampling rate in seconds if set (non-zero).
    /// Returns None if the channel is not found or sampling rate is not specified.
    pub fn get_channel_sampling_rate(&self, channel_name: String) -> PyResult<Option<f64>> {
        let Mdfr(mdf) = self;
        Ok(mdf.get_channel_sampling_rate(&channel_name))
    }
    /// Returns the minimum value of the valid range for a channel.
    /// Returns None if the channel is not found or range is not specified.
    /// If not present in the block, computes from channel data.
    pub fn get_channel_range_min(&mut self, channel_name: String) -> PyResult<Option<f64>> {
        let Mdfr(mdf) = self;
        Ok(mdf.get_channel_range_min(&channel_name))
    }
    /// Returns the maximum value of the valid range for a channel.
    /// Returns None if the channel is not found or range is not specified.
    /// If not present in the block, computes from channel data.
    pub fn get_channel_range_max(&mut self, channel_name: String) -> PyResult<Option<f64>> {
        let Mdfr(mdf) = self;
        Ok(mdf.get_channel_range_max(&channel_name))
    }
    /// Returns the minimum limit for a channel.
    /// Returns None if the channel is not found or limit is not specified.
    /// If not present in the block, falls back to the data range minimum.
    pub fn get_channel_limit_min(&mut self, channel_name: String) -> PyResult<Option<f64>> {
        let Mdfr(mdf) = self;
        Ok(mdf.get_channel_limit_min(&channel_name))
    }
    /// Returns the maximum limit for a channel.
    /// Returns None if the channel is not found or limit is not specified.
    /// If not present in the block, falls back to the data range maximum.
    pub fn get_channel_limit_max(&mut self, channel_name: String) -> PyResult<Option<f64>> {
        let Mdfr(mdf) = self;
        Ok(mdf.get_channel_limit_max(&channel_name))
    }
    /// Returns the minimum extended limit for a channel.
    /// Returns None if the channel is not found or extended limit is not specified.
    /// If not present in the block, falls back to the data range minimum.
    pub fn get_channel_limit_ext_min(&mut self, channel_name: String) -> PyResult<Option<f64>> {
        let Mdfr(mdf) = self;
        Ok(mdf.get_channel_limit_ext_min(&channel_name))
    }
    /// Returns the maximum extended limit for a channel.
    /// Returns None if the channel is not found or extended limit is not specified.
    /// If not present in the block, falls back to the data range maximum.
    pub fn get_channel_limit_ext_max(&mut self, channel_name: String) -> PyResult<Option<f64>> {
        let Mdfr(mdf) = self;
        Ok(mdf.get_channel_limit_ext_max(&channel_name))
    }
    /// Decodes event signal channel data into a list of event records.
    /// Each record is a dict with 'sync_value' (float) and 'event_data' (bytes).
    /// Returns None if the channel is not an event signal channel or has no data.
    pub fn decode_event_records(&mut self, channel_name: String) -> PyResult<Py<PyAny>> {
        let Mdfr(mdf) = self;
        pyo3::Python::attach(|py| {
            let records = mdf.decode_event_records(&channel_name);
            match records {
                Some(recs) => {
                    let mut items = Vec::new();
                    for rec in &recs {
                        let dict = PyDict::new(py);
                        dict.set_item("sync_value", rec.sync_value)?;
                        dict.set_item("event_data", PyBytes::new(py, &rec.event_data))?;
                        items.push(dict.into_pyobject(py)?.into_any());
                    }
                    let list = PyList::new(py, items)?;
                    let result: Py<PyAny> = list.into_pyobject(py)?.into();
                    Ok(result)
                }
                None => Ok(py.None()),
            }
        })
    }
    /// returns measurement start time as a timezone-aware datetime
    #[getter]
    pub fn start_time(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let Mdfr(mdf) = self;
        let ns = mdf.get_start_time_ns();
        let offset_s = mdf.get_tz_offset_min() as i32 * 60;
        let dt_mod = py.import("datetime")?;
        let td_kw = pyo3::types::PyDict::new(py);
        td_kw.set_item(pyo3::intern!(py, "seconds"), offset_s)?;
        let tz_delta = dt_mod.getattr(pyo3::intern!(py, "timedelta"))?.call((), Some(&td_kw))?;
        let tz = dt_mod.getattr(pyo3::intern!(py, "timezone"))?.call1((tz_delta,))?;
        let dt = dt_mod
            .getattr(pyo3::intern!(py, "datetime"))?
            .getattr(pyo3::intern!(py, "fromtimestamp"))?
            .call1((ns as f64 / 1e9, tz))?;
        Ok(dt.unbind())
    }
    /// returns master channel data as an Arrow TimestampNanosecond array with timezone.
    /// Raises ValueError if the channel has no Time master.
    pub fn get_master_channel_datetimes(
        &mut self,
        channel_name: String,
    ) -> PyResult<PyArrowType<ArrayData>> {
        let Mdfr(mdf) = self;
        mdf.get_master_channel_datetimes(&channel_name)
            .map(|arr| PyArrowType(arr.into_data()))
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "no Time master for channel '{channel_name}'"
                ))
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
    /// slices all channels in the master's group to [start_s, stop_s] seconds
    pub fn cut(&mut self, master_name: String, start_s: f64, stop_s: f64) -> PyResult<()> {
        let Mdfr(mdf) = self;
        mdf.cut(&master_name, start_s, stop_s)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
    /// keeps only the given channels; all others have their data cleared
    pub fn keep_channels(&mut self, channel_names: Vec<String>) -> PyResult<()> {
        let Mdfr(mdf) = self;
        mdf.keep_channels(channel_names.into_iter().collect())?;
        Ok(())
    }
    /// resamples channels in one master group to a uniform raster (seconds)
    pub fn resample_group(&mut self, master_name: String, raster_s: f64) -> PyResult<()> {
        let Mdfr(mdf) = self;
        mdf.resample_group(&master_name, raster_s)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
    /// resamples all Time master groups to a uniform raster (seconds)
    pub fn resample(&mut self, raster_s: f64) -> PyResult<()> {
        let Mdfr(mdf) = self;
        mdf.resample(raster_s)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
    /// appends other's data after self on the time axis (time-sequential concat)
    pub fn concat_mdf(&mut self, other: &mut Mdfr) -> PyResult<()> {
        let Mdfr(mdf) = self;
        let Mdfr(other_mdf) = other;
        mdf.concat_mdf(other_mdf)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
    /// imports channels from other into self (channel-axis join on shared time axis)
    pub fn merge(&mut self, other: &mut Mdfr) -> PyResult<()> {
        let Mdfr(mdf) = self;
        let Mdfr(other_mdf) = other;
        mdf.merge(other_mdf)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
    /// load all channels in memory
    pub fn load_all_channels_data_in_memory(&mut self) -> PyResult<()> {
        let Mdfr(mdf) = self;
        mdf.load_all_channels_data_in_memory()?;
        Ok(())
    }
    /// Eagerly converts all channels to physical values
    pub fn convert_all_channels(&mut self) -> PyResult<()> {
        let Mdfr(mdf) = self;
        mdf.convert_all_channels()?;
        Ok(())
    }
    /// Converts one channel's data to physical values
    pub fn convert_channel(&mut self, channel_name: &str) -> PyResult<()> {
        let Mdfr(mdf) = self;
        mdf.convert_channel(channel_name)?;
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
        let array = array_to_rust(data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("data modification failed, could not extract array: {e}")))?;
        mdf.set_channel_data(channel_name, array)?;
        Ok(())
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
    /// export all channel groups to CSV (one file per channel group)
    pub fn export_to_csv(&self, file_name: &str) -> PyResult<()> {
        let Mdfr(mdf) = self;
        mdf.export_to_csv(file_name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
    /// export the channel group containing the given channel to a CSV file
    pub fn export_dataframe_to_csv(&self, channel_name: String, file_name: &str) -> PyResult<()> {
        let Mdfr(mdf) = self;
        mdf.export_dataframe_to_csv(&channel_name, file_name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
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
                let fh_info = mdfinfo4.list_file_history();
                if !fh_info.is_empty() {
                    writeln!(output, "\n--- File History ---")
                        .context("cannot print file history header")?;
                    write!(output, "{fh_info}").context("cannot print file history")?;
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

    /// Dictionary indexing: mdf["ChannelName"] -> PyChannel
    pub fn __getitem__(&mut self, channel_name: &str) -> PyResult<PyChannel> {
        let Mdfr(mdf) = self;
        if !mdf.get_channel_names_set().contains(channel_name) {
            return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "Channel '{channel_name}' not found in MDF file"
            )));
        }
        if mdf.get_channel_converted_data(channel_name).is_none() {
            let mut set = HashSet::new();
            set.insert(channel_name.to_string());
            let _ = mdf.load_channels_data_in_memory(set);
        }
        let data = mdf.get_channel_converted_data(channel_name).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!(
                "Failed to read data for channel '{channel_name}'"
            ))
        })?;
        let unit = mdf.mdf_info.get_channel_unit(channel_name).ok().flatten();
        let desc = mdf.mdf_info.get_channel_desc(channel_name).ok().flatten();
        let master_name = mdf.mdf_info.get_channel_master(channel_name).map(|s| s.to_string());
        Ok(PyChannel {
            name: channel_name.to_string(),
            data,
            unit,
            desc,
            master_name,
        })
    }

    /// Membership check: "ChannelName" in mdf
    pub fn __contains__(&self, channel_name: &str) -> bool {
        let Mdfr(mdf) = self;
        mdf.get_channel_names_set().contains(channel_name)
    }

    /// Total number of channels in the MDF file
    pub fn __len__(&self) -> usize {
        let Mdfr(mdf) = self;
        mdf.get_channel_names_set().len()
    }

    /// List of all channel names in the MDF file
    pub fn keys(&self) -> Vec<String> {
        let Mdfr(mdf) = self;
        let mut list: Vec<String> = mdf.get_channel_names_set().into_iter().collect();
        list.sort();
        list
    }

    /// Iterator over channel names
    pub fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let keys = self.keys();
        PyList::new(py, keys)?.call_method0("__iter__").map(|b| b.unbind())
    }

    /// List of all PyChannel objects
    pub fn values(&mut self) -> PyResult<Vec<PyChannel>> {
        let keys = self.keys();
        let mut vals = Vec::with_capacity(keys.len());
        for k in keys {
            vals.push(self.__getitem__(&k)?);
        }
        Ok(vals)
    }

    /// List of (name, PyChannel) tuples
    pub fn items(&mut self) -> PyResult<Vec<(String, PyChannel)>> {
        let keys = self.keys();
        let mut items = Vec::with_capacity(keys.len());
        for k in keys {
            let ch = self.__getitem__(&k)?;
            items.push((k, ch));
        }
        Ok(items)
    }

    /// Context manager entry
    pub fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Context manager exit
    #[pyo3(signature = (_exc_type=None, _exc_val=None, _exc_tb=None))]
    pub fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        Ok(false)
    }

    /// Exports the MDF file's default channel group as an Arrow StructArray PyCapsule.
    /// Compatible with PyArrow (`pa.table(mdf)`) and Polars (`pl.from_arrow(mdf)`).
    #[pyo3(signature = (requested_schema=None))]
    #[allow(unused_variables)]
    pub fn __arrow_c_array__<'py>(
        &mut self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let Mdfr(mdf) = self;
        let groups = mdf.get_master_channel_names_set();
        let selected_channels: HashSet<String> = if let Some(chs) = groups.iter().find(|(m, _)| {
            m.as_ref().map(|s| s.to_lowercase() == "time").unwrap_or(false)
        }).map(|(_, chs)| chs) {
            chs.clone()
        } else if let Some((_, chs)) = groups.iter().find(|(m, _)| m.is_some()) {
            chs.clone()
        } else if let Some((_, chs)) = groups.iter().next() {
            chs.clone()
        } else {
            HashSet::new()
        };

        if !selected_channels.is_empty() {
            mdf.load_channels_data_in_memory(selected_channels.clone())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        }

        let mut fields = Vec::new();
        let mut arrays = Vec::new();
        for ch_name in &selected_channels {
            if let Some(ch_data) = mdf.get_channel_converted_data(ch_name) {
                let arr = ch_data.as_arrow();
                let field = arrow::datatypes::Field::new(ch_name, arr.data_type().clone(), true);
                fields.push(field);
                arrays.push(arr);
            }
        }

        let struct_arr = arrow::array::StructArray::try_new(
            arrow::datatypes::Fields::from(fields),
            arrays,
            None,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        crate::export::arrow_capsule::export_array_to_pycapsules(py, &struct_arr.to_data())
    }

    /// Exports the MDF file's default channel group as an Arrow ArrayStream PyCapsule.
    /// Compatible with PyArrow (`pa.table(mdf)`) and Polars (`pl.DataFrame(mdf)`).
    #[pyo3(signature = (requested_schema=None))]
    #[allow(unused_variables)]
    pub fn __arrow_c_stream__<'py>(
        &mut self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let Mdfr(mdf) = self;
        let groups = mdf.get_master_channel_names_set();
        let selected_channels: HashSet<String> = if let Some(chs) = groups.iter().find(|(m, _)| {
            m.as_ref().map(|s| s.to_lowercase() == "time").unwrap_or(false)
        }).map(|(_, chs)| chs) {
            chs.clone()
        } else if let Some((_, chs)) = groups.iter().find(|(m, _)| m.is_some()) {
            chs.clone()
        } else if let Some((_, chs)) = groups.iter().next() {
            chs.clone()
        } else {
            HashSet::new()
        };

        if !selected_channels.is_empty() {
            mdf.load_channels_data_in_memory(selected_channels.clone())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        }

        let mut fields = Vec::new();
        let mut arrays = Vec::new();
        for ch_name in &selected_channels {
            if let Some(ch_data) = mdf.get_channel_converted_data(ch_name) {
                let arr = ch_data.as_arrow();
                let field = arrow::datatypes::Field::new(ch_name, arr.data_type().clone(), true);
                fields.push(field);
                arrays.push(arr);
            }
        }

        let schema = Arc::new(arrow::datatypes::Schema::new(fields));
        let batch = arrow::array::RecordBatch::try_new(schema.clone(), arrays)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let reader = Box::new(arrow::array::RecordBatchIterator::new(vec![Ok(batch)], schema));
        crate::export::arrow_capsule::export_reader_to_pycapsule(py, reader)
    }
}

/// Finalizes an unfinalized MDF 4.x file (`"UnFinMF "`) in place on disk without copying.
/// Returns the number of channel groups patched.
#[pyfunction]
fn finalize_file_in_place(file_name: &str) -> PyResult<usize> {
    crate::mdfinfo::mdfinfo4::finalize::finalize_file_in_place(file_name)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Directly writes a Polars DataFrame, PyArrow Table, or any object implementing
/// the Arrow PyCapsule stream interface to an MDF 4.3 file.
#[pyfunction]
#[pyo3(signature = (df, file_name, time_col=None, compression=CompressionAlgorithm::NoCompression))]
pub fn write_polars(
    _py: Python<'_>,
    df: &Bound<'_, PyAny>,
    file_name: &str,
    time_col: Option<&str>,
    compression: CompressionAlgorithm,
) -> PyResult<Mdfr> {
    let stream_capsule = if let Ok(stream) = df.call_method0("__arrow_c_stream__") {
        stream
    } else if let Ok(table) = df.call_method0("to_arrow") {
        table.call_method0("__arrow_c_stream__")?
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "Expected a Polars DataFrame, PyArrow Table, or object implementing __arrow_c_stream__",
        ));
    };

    let reader = crate::export::arrow_capsule::import_stream_from_pycapsule(&stream_capsule)?;
    let schema = reader.schema();

    let mut batches = Vec::new();
    for batch_res in reader {
        let batch = batch_res.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        batches.push(batch);
    }
    if batches.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "DataFrame / Arrow stream contains no batches",
        ));
    }

    let batch = if batches.len() == 1 {
        batches.pop().unwrap()
    } else {
        arrow::compute::concat_batches(&schema, &batches)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
    };

    let master_name = if let Some(tc) = time_col {
        tc.to_string()
    } else if schema.column_with_name("time").is_some() {
        "time".to_string()
    } else if schema.column_with_name("Time").is_some() {
        "Time".to_string()
    } else if let Some(first_field) = schema.fields().first() {
        first_field.name().to_string()
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "DataFrame has no columns",
        ));
    };

    let num_cols = schema.fields().len() + 1;
    let mut mdf = Mdf {
        mdf_info: crate::mdfinfo::MdfInfo::V4(Box::new(
            crate::mdfinfo::mdfinfo4::MdfInfo4::new(file_name, num_cols),
        )),
    };

    let has_master_in_df = schema.column_with_name(&master_name).is_some();

    if has_master_in_df {
        let (idx, _) = schema.column_with_name(&master_name).unwrap();
        let col_array = batch.column(idx).clone();
        mdf.add_channel(
            master_name.clone(),
            col_array,
            None,
            Some(1), // 1 = Time
            true,    // master_flag = true
            Some("s".to_string()),
            None,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    } else {
        let time_vals: Vec<f64> = (0..batch.num_rows()).map(|i| i as f64).collect();
        let time_arr = Arc::new(arrow::array::Float64Array::from(time_vals));
        mdf.add_channel(
            master_name.clone(),
            time_arr,
            None,
            Some(1), // 1 = Time
            true,    // master_flag = true
            Some("s".to_string()),
            Some("Synthetic master time channel".to_string()),
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    }

    for (idx, field) in schema.fields().iter().enumerate() {
        let col_name = field.name().as_str();
        if col_name == master_name {
            continue;
        }
        let col_array = batch.column(idx).clone();
        mdf.add_channel(
            col_name.to_string(),
            col_array,
            Some(master_name.clone()),
            None,
            false,
            None,
            None,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    }

    let written = mdf.write(file_name, compression)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(Mdfr(written))
}

use mdfr_core::StdClock;

enum PyLoggerState {
    Building {
        path: String,
        builder: mdfr_core::logger::LoggerBuilder,
    },
    Active(mdfr_core::logger::Logger<std::fs::File, StdClock>),
    Closed,
}

/// Streaming MDF 4.3 logger backed by `mdfr-core`.
#[pyclass]
pub struct MdfLogger {
    state: PyLoggerState,
}

#[pymethods]
impl MdfLogger {
    #[new]
    #[pyo3(signature = (file_name, layout="sorted", compression=CompressionAlgorithm::NoCompression))]
    fn new(file_name: &str, layout: &str, compression: CompressionAlgorithm) -> PyResult<Self> {
        let dl = if layout.eq_ignore_ascii_case("unsorted") {
            mdfr_core::logger::DataLayout::Unsorted
        } else {
            mdfr_core::logger::DataLayout::Sorted
        };
        let builder = mdfr_core::logger::LoggerBuilder::new(dl, compression.into());
        Ok(Self {
            state: PyLoggerState::Building {
                path: file_name.to_string(),
                builder,
            },
        })
    }

    /// Adds a channel group with a 64-bit microsecond master time channel. Returns the group index (`u16`).
    pub fn add_group(&mut self, name: &str) -> PyResult<u16> {
        match &mut self.state {
            PyLoggerState::Building { builder, .. } => Ok(builder.add_group(name).0),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "cannot add group after start()",
            )),
        }
    }

    /// Appends a scalar channel (`data_type`: 0=UintLE, 2=IntLE, 4=FloatLE) and returns a packed channel handle (`u32`).
    #[pyo3(signature = (group, name, data_type, bit_count, unit=None))]
    pub fn add_channel(
        &mut self,
        group: u16,
        name: &str,
        data_type: u8,
        bit_count: u32,
        unit: Option<&str>,
    ) -> PyResult<u32> {
        let dt = mdfr_core::blocks::cn::CnDataType::try_from(data_type).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid data_type: {e}"))
        })?;
        match &mut self.state {
            PyLoggerState::Building { builder, .. } => {
                let ch = builder
                    .add_channel(mdfr_core::logger::GroupHandle(group), name, dt, bit_count)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                if let Some(u) = unit {
                    builder
                        .set_channel_unit(ch, u)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }
                Ok(((ch.group as u32) << 16) | (ch.channel as u32))
            }
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "cannot add channel after start()",
            )),
        }
    }

    /// Appends a VLSD channel (`data_type`: 7=StringUtf8, 10=ByteArray) and returns a packed channel handle (`u32`).
    pub fn add_vlsd_channel(&mut self, group: u16, name: &str, data_type: u8) -> PyResult<u32> {
        let dt = mdfr_core::blocks::cn::CnDataType::try_from(data_type).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid data_type: {e}"))
        })?;
        match &mut self.state {
            PyLoggerState::Building { builder, .. } => {
                let ch = builder
                    .add_vlsd_channel(mdfr_core::logger::GroupHandle(group), name, dt)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(((ch.group as u32) << 16) | (ch.channel as u32))
            }
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "cannot add VLSD channel after start()",
            )),
        }
    }

    /// Appends an MDF 4.3 VLSC channel (`data_type`: 7=StringUtf8, 10=ByteArray) and its companion size channel, returning a packed channel handle (`u32`).
    pub fn add_vlsc_channel(&mut self, group: u16, name: &str, data_type: u8) -> PyResult<u32> {
        let dt = mdfr_core::blocks::cn::CnDataType::try_from(data_type).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid data_type: {e}"))
        })?;
        match &mut self.state {
            PyLoggerState::Building { builder, .. } => {
                let ch = builder
                    .add_vlsc_channel(mdfr_core::logger::GroupHandle(group), name, dt)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(((ch.group as u32) << 16) | (ch.channel as u32))
            }
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "cannot add VLSC channel after start()",
            )),
        }
    }

    /// Appends an N-D array channel (`CABLOCK`) and returns a packed channel handle (`u32`).
    #[pyo3(signature = (group, name, data_type, bit_count, dims, unit=None))]
    pub fn add_array_channel(
        &mut self,
        group: u16,
        name: &str,
        data_type: u8,
        bit_count: u32,
        dims: Vec<u64>,
        unit: Option<&str>,
    ) -> PyResult<u32> {
        let dt = mdfr_core::blocks::cn::CnDataType::try_from(data_type).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid data_type: {e}"))
        })?;
        match &mut self.state {
            PyLoggerState::Building { builder, .. } => {
                let ch = builder
                    .add_array_channel(
                        mdfr_core::logger::GroupHandle(group),
                        name,
                        dt,
                        bit_count,
                        dims,
                    )
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                if let Some(u) = unit {
                    builder
                        .set_channel_unit(ch, u)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                }
                Ok(((ch.group as u32) << 16) | (ch.channel as u32))
            }
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "cannot add array channel after start()",
            )),
        }
    }

    /// Sets a linear conversion rule (`phys = p1 + p2 * raw`) on a channel before `start()`.
    pub fn set_channel_conversion_linear(
        &mut self,
        ch_packed: u32,
        p1: f64,
        p2: f64,
    ) -> PyResult<()> {
        let ch = mdfr_core::logger::ChannelHandle {
            group: (ch_packed >> 16) as u16,
            channel: (ch_packed & 0xFFFF) as u16,
        };
        match &mut self.state {
            PyLoggerState::Building { builder, .. } => {
                builder
                    .set_channel_conversion(
                        ch,
                        mdfr_core::logger::ConversionDef::Linear { p1, p2 },
                    )
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(())
            }
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "cannot set conversion after start()",
            )),
        }
    }

    /// Adds an ASAM bus logging preset group (`bus_type`: "can", "can_fd", "can_fd_vlsd", "lin", "ethernet", "flexray").
    pub fn add_bus_preset(&mut self, bus_type: &str, name: &str) -> PyResult<u16> {
        match &mut self.state {
            PyLoggerState::Building { builder, .. } => {
                let g = match bus_type.to_ascii_lowercase().as_str() {
                    "can" => mdfr_core::logger::BusPreset::can(builder, name).map(|h| h.group),
                    "can_fd" | "can-fd" | "canfd" => {
                        mdfr_core::logger::BusPreset::can_fd(builder, name).map(|h| h.group)
                    }
                    "can_fd_vlsd" | "can-fd-vlsd" | "canfd_vlsd" => {
                        mdfr_core::logger::BusPreset::can_fd_vlsd(builder, name).map(|h| h.group)
                    }
                    "lin" => mdfr_core::logger::BusPreset::lin(builder, name),
                    "ethernet" | "eth" => mdfr_core::logger::BusPreset::ethernet(builder, name),
                    "flexray" | "flx" => mdfr_core::logger::BusPreset::flexray(builder, name),
                    other => {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "unsupported bus preset '{other}': expected 'can', 'can_fd', 'can_fd_vlsd', 'lin', 'ethernet', or 'flexray'"
                        )));
                    }
                }
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(g.0)
            }
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "cannot add bus preset after start()",
            )),
        }
    }

    /// Transitions from schema builder to active streaming logger, writing the MDF 4.3 header hierarchy.
    pub fn start(&mut self) -> PyResult<()> {
        let prev = std::mem::replace(&mut self.state, PyLoggerState::Closed);
        match prev {
            PyLoggerState::Building { path, builder } => {
                let schema = builder
                    .build()
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                let file = std::fs::OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(&path)
                    .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
                let logger = mdfr_core::logger::Logger::start(schema, file, StdClock::new())
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                self.state = PyLoggerState::Active(logger);
                Ok(())
            }
            other => {
                self.state = other;
                Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "logger already started or closed",
                ))
            }
        }
    }

    /// Begins assembling a record for `group`.
    pub fn record_begin(&mut self, group: u16) -> PyResult<()> {
        match &mut self.state {
            PyLoggerState::Active(l) => {
                l.begin_record(mdfr_core::logger::GroupHandle(group))
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                Ok(())
            }
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "logger is not active",
            )),
        }
    }

    /// Sets an unsigned integer channel value in the staged record.
    pub fn record_set_u64(&mut self, ch_packed: u32, val: u64) -> PyResult<()> {
        let ch = mdfr_core::logger::ChannelHandle {
            group: (ch_packed >> 16) as u16,
            channel: (ch_packed & 0xFFFF) as u16,
        };
        match &mut self.state {
            PyLoggerState::Active(l) => l
                .set_channel_u64(ch, val)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "logger is not active",
            )),
        }
    }

    /// Sets a signed integer channel value in the staged record.
    pub fn record_set_i64(&mut self, ch_packed: u32, val: i64) -> PyResult<()> {
        let ch = mdfr_core::logger::ChannelHandle {
            group: (ch_packed >> 16) as u16,
            channel: (ch_packed & 0xFFFF) as u16,
        };
        match &mut self.state {
            PyLoggerState::Active(l) => l
                .set_channel_i64(ch, val)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "logger is not active",
            )),
        }
    }

    /// Sets a float channel value in the staged record.
    pub fn record_set_f64(&mut self, ch_packed: u32, val: f64) -> PyResult<()> {
        let ch = mdfr_core::logger::ChannelHandle {
            group: (ch_packed >> 16) as u16,
            channel: (ch_packed & 0xFFFF) as u16,
        };
        match &mut self.state {
            PyLoggerState::Active(l) => l
                .set_channel_f64(ch, val)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "logger is not active",
            )),
        }
    }

    /// Sets a byte slice / UTF-8 string channel value in the staged record.
    pub fn record_set_bytes(&mut self, ch_packed: u32, data: &[u8]) -> PyResult<()> {
        let ch = mdfr_core::logger::ChannelHandle {
            group: (ch_packed >> 16) as u16,
            channel: (ch_packed & 0xFFFF) as u16,
        };
        match &mut self.state {
            PyLoggerState::Active(l) => l
                .set_channel_bytes(ch, data)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "logger is not active",
            )),
        }
    }

    /// Commits the staged record for `group`.
    pub fn record_commit(&mut self, group: u16) -> PyResult<()> {
        match &mut self.state {
            PyLoggerState::Active(l) => l
                .commit_staged_record(mdfr_core::logger::GroupHandle(group))
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "logger is not active",
            )),
        }
    }

    /// Flushes buffered data and patches `DL`/`CG` links and cycle counts in place.
    pub fn checkpoint(&mut self) -> PyResult<()> {
        match &mut self.state {
            PyLoggerState::Active(l) => l
                .checkpoint()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "logger is not active",
            )),
        }
    }

    /// Rotates to `new_file_name`, finalizing the current file in place and incrementing the sequence index.
    pub fn rotate(&mut self, new_file_name: &str) -> PyResult<()> {
        match &mut self.state {
            PyLoggerState::Active(l) => {
                let file = std::fs::OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(new_file_name)
                    .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
                l.rotate(file, None)
                    .map(|_| ())
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
            }
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "logger is not active",
            )),
        }
    }

    /// Finalizes the MDF 4.3 file in place and closes the logger.
    pub fn close(&mut self) -> PyResult<()> {
        let prev = std::mem::replace(&mut self.state, PyLoggerState::Closed);
        if let PyLoggerState::Active(l) = prev {
            l.close()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        }
        Ok(())
    }

    /// Context manager entry.
    pub fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Context manager exit: finalizes and closes the logger.
    #[pyo3(signature = (_exc_type=None, _exc_val=None, _exc_tb=None))]
    pub fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        self.close()?;
        Ok(false)
    }
}
