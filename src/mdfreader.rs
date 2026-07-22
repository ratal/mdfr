//! This module contains the data reading features
pub mod conversions3;
pub mod conversions4;
pub mod data_read3;
pub mod data_read4;
pub mod datastream_decoder;
pub mod mdfreader3;
pub mod mdfreader4;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::BufReader;
use std::sync::Arc;

use anyhow::{Context, Error, Result, bail};
use arrow::array::{Array, TimestampNanosecondArray};
use arrow::util::display::{ArrayFormatter, FormatOptions};
use log::info;
#[cfg(feature = "numpy")]
use pyo3::prelude::*;

//use crate::export::parquet::export_to_parquet;
use crate::data_holder::channel_data::{interp_channel, try_from};
use crate::mdfinfo::mdfinfo4::CompressionAlgorithm;
use crate::mdfinfo::{ChannelsDb, MdfInfo};
use crate::mdfreader::mdfreader3::mdfreader3;
use crate::mdfreader::mdfreader4::mdfreader4;
use crate::mdfwriter::mdfwriter4::mdfwriter4;

use crate::export::csv::{export_dataframe_to_csv, export_to_csv};
#[cfg(feature = "parquet")]
use crate::export::parquet::export_dataframe_to_parquet;
#[cfg(feature = "parquet")]
use crate::export::parquet::export_to_parquet;

#[cfg(feature = "hdf5")]
use crate::export::hdf5::export_dataframe_to_hdf5;
#[cfg(feature = "hdf5")]
use crate::export::hdf5::export_to_hdf5;

use crate::data_holder::arrow_helpers::{
    arrow_bit_count, arrow_byte_count, arrow_to_mdf_data_type,
};
use crate::data_holder::channel_data::ChannelData;
use crate::data_holder::tensor_arrow::Order;

/// Main Mdf struct holding mdfinfo, arrow data and schema
#[derive(Debug)]
#[repr(C)]
pub struct Mdf {
    /// MdfInfo enum
    pub mdf_info: MdfInfo,
}

/// data generic description
#[repr(C)]
#[derive(Clone)]
pub struct DataSignature {
    pub(crate) len: usize,
    pub(crate) data_type: u8,
    pub(crate) bit_count: u32,
    pub(crate) byte_count: u32,
    pub(crate) ndim: usize,
    pub(crate) shape: (Vec<usize>, Order),
}

/// master channel generic description
#[repr(C)]
#[cfg_attr(feature = "numpy", derive(FromPyObject))]
#[derive(Clone)]
pub struct MasterSignature {
    #[cfg_attr(feature = "numpy", pyo3(attribute("name")))]
    pub(crate) master_channel: Option<String>,
    #[cfg_attr(feature = "numpy", pyo3(attribute("type")))]
    pub(crate) master_type: Option<u8>,
    #[cfg_attr(feature = "numpy", pyo3(attribute("flag")))]
    pub(crate) master_flag: bool,
}

#[allow(dead_code)]
impl Mdf {
    /// returns Mdf with metadata but no data
    pub fn new(file_name: &str) -> Result<Mdf> {
        let mdf = Mdf {
            mdf_info: MdfInfo::new(file_name)?,
        };
        Ok(mdf)
    }
    pub fn get_file_name(&self) -> String {
        match &self.mdf_info {
            MdfInfo::V3(mdfinfo3) => mdfinfo3.file_name.clone(),
            MdfInfo::V4(mdfinfo4) => mdfinfo4.file_name.clone(),
        }
    }
    /// gets the version of mdf file
    pub fn get_version(&self) -> u16 {
        self.mdf_info.get_version()
    }
    /// returns true if the file was marked as unfinalized
    pub fn is_unfinalized(&self) -> bool {
        self.mdf_info.is_unfinalized()
    }
    /// returns the standard and custom unfinalization flags (0, 0) if finalized or MDF3
    pub fn get_unfin_flags(&self) -> (u16, u16) {
        self.mdf_info.get_unfin_flags()
    }
    /// Index of this file within a recording sequence (MDF 4.3 common_properties).
    pub fn get_recorder_sequence_index(&self) -> Option<u64> {
        self.mdf_info.get_recorder_sequence_index()
    }
    /// Index of this file within the recorder's file set (MDF 4.3 common_properties).
    pub fn get_recorder_file_index(&self) -> Option<u64> {
        self.mdf_info.get_recorder_file_index()
    }
    /// True if this is the last file in the recorder sequence (MDF 4.3 common_properties).
    pub fn get_recorder_file_last(&self) -> Option<bool> {
        self.mdf_info.get_recorder_file_last()
    }
    /// UUID of the recorder device (MDF 4.3 common_properties).
    pub fn get_recorder_uuid(&self) -> Option<String> {
        self.mdf_info.get_recorder_uuid()
    }
    /// UUID identifying the measurement (MDF 4.3 common_properties).
    pub fn get_measurement_uuid(&self) -> Option<String> {
        self.mdf_info.get_measurement_uuid()
    }
    /// Author from HD common_properties (MDF 4.3).
    pub fn get_author(&self) -> Option<String> {
        self.mdf_info.get_author()
    }
    /// Department from HD common_properties (MDF 4.3).
    pub fn get_department(&self) -> Option<String> {
        self.mdf_info.get_department()
    }
    /// Project from HD common_properties (MDF 4.3).
    pub fn get_project(&self) -> Option<String> {
        self.mdf_info.get_project()
    }
    /// Subject from HD common_properties (MDF 4.3).
    pub fn get_subject(&self) -> Option<String> {
        self.mdf_info.get_subject()
    }
    /// List sample reduction blocks for all channel groups (MDF 4.x only)
    pub fn list_sample_reductions(&self) -> String {
        self.mdf_info.list_sample_reductions()
    }
    /// returns channel's unit string
    pub fn get_channel_unit(&self, channel_name: &str) -> Result<Option<String>> {
        self.mdf_info.get_channel_unit(channel_name)
    }
    /// Sets the channel unit in memory
    pub fn set_channel_unit(&mut self, channel_name: &str, unit: &str) {
        self.mdf_info.set_channel_unit(channel_name, unit)
    }
    /// returns channel's description string
    pub fn get_channel_desc(&self, channel_name: &str) -> Result<Option<String>> {
        self.mdf_info.get_channel_desc(channel_name)
    }
    /// Sets the channel description in memory
    pub fn set_channel_desc(&mut self, channel_name: &str, desc: &str) {
        self.mdf_info.set_channel_desc(channel_name, desc)
    }
    /// returns channel's associated master channel name string
    pub fn get_channel_master(&self, channel_name: &str) -> Option<String> {
        self.mdf_info.get_channel_master(channel_name)
    }
    /// returns channel's associated master channel type string
    /// 0 = None (normal data channels), 1 = Time (seconds), 2 = Angle (radians),
    /// 3 = Distance (meters), 4 = Index (zero-based index values)
    pub fn get_channel_master_type(&self, channel_name: &str) -> u8 {
        self.mdf_info.get_channel_master_type(channel_name)
    }
    /// Sets the channel's related master channel type in memory
    pub fn set_channel_master_type(&mut self, master_name: &str, master_type: u8) -> Result<()> {
        self.mdf_info
            .set_channel_master_type(master_name, master_type)?;
        Ok(())
    }
    /// returns a set of all channel names contained in file
    pub fn get_channel_names_set(&self) -> HashSet<String> {
        self.mdf_info.get_channel_names_set()
    }
    /// returns measurement start timestamp in nanoseconds since Unix epoch
    pub fn get_start_time_ns(&self) -> u64 {
        self.mdf_info.get_start_time_ns()
    }
    /// returns timezone + DST offset in minutes
    pub fn get_tz_offset_min(&self) -> i16 {
        self.mdf_info.get_tz_offset_min()
    }
    /// returns master channel data as absolute nanosecond timestamps.
    /// Returns None if the channel has no Time master (sync type 1).
    pub fn get_master_channel_datetimes(
        &mut self,
        channel_name: &str,
    ) -> Option<TimestampNanosecondArray> {
        let master = self.mdf_info.get_channel_master(channel_name)?;
        if self.mdf_info.get_channel_master_type(&master) != 1 {
            return None;
        }
        let start_ns = self.mdf_info.get_start_time_ns() as i64;
        let offset_min = self.mdf_info.get_tz_offset_min();
        let tz = format!(
            "{:+03}:{:02}",
            offset_min / 60,
            offset_min.unsigned_abs() % 60
        );
        match self.get_channel_data(&master)? {
            ChannelData::Float64(b) => {
                let ns: Vec<i64> = b
                    .values_slice()
                    .iter()
                    .map(|&s| start_ns + (s * 1e9) as i64)
                    .collect();
                Some(TimestampNanosecondArray::from(ns).with_timezone(tz))
            }
            _ => None,
        }
    }
    /// returns the full channel-name → position-tuple map
    pub fn get_channels_db(&self) -> ChannelsDb {
        self.mdf_info.get_channels_db()
    }
    /// returns a dict of master names keys for which values are a set of associated channel names
    pub fn get_master_channel_names_set(&self) -> HashMap<Option<String>, HashSet<String>> {
        self.mdf_info.get_master_channel_names_set()
    }
    /// Eagerly converts all channels to physical values.
    pub fn convert_all_channels(&mut self) -> Result<(), Error> {
        self.mdf_info.convert_all_channels()
    }
    /// Converts one channel's data to physical values.
    pub fn convert_channel(&mut self, channel_name: &str) -> Result<(), Error> {
        self.mdf_info.convert_channel(channel_name)
    }
    /// returns channel's arrow Array. If data were not yet converted, converts it and keep in memory
    pub fn get_channel_data(&mut self, channel_name: &str) -> Option<&ChannelData> {
        match &mut self.mdf_info {
            MdfInfo::V3(mdfinfo3) => mdfinfo3.get_channel_data(channel_name),
            MdfInfo::V4(mdfinfo4) => mdfinfo4.get_channel_data(channel_name),
        }
    }
    /// returns channel's arrow Array, conversion not kept in memory
    pub fn get_channel_converted_data(&self, channel_name: &str) -> Option<ChannelData> {
        match &self.mdf_info {
            MdfInfo::V3(mdfinfo3) => mdfinfo3.get_channel_converted_data(channel_name),
            MdfInfo::V4(mdfinfo4) => mdfinfo4.get_channel_converted_data(channel_name),
        }
    }
    /// defines channel's data in memory
    pub fn set_channel_data(&mut self, channel_name: &str, data: Arc<dyn Array>) -> Result<()> {
        self.mdf_info.set_channel_data(channel_name, data)
    }
    /// Renames a channel's name in memory
    pub fn rename_channel(&mut self, channel_name: &str, new_name: &str) {
        self.mdf_info.rename_channel(channel_name, new_name)
    }
    /// Adds a new channel in memory (no file modification)
    #[allow(clippy::too_many_arguments)]
    pub fn add_channel(
        &mut self,
        channel_name: String,
        data: Arc<dyn Array>,
        master_channel: Option<String>,
        master_type: Option<u8>,
        master_flag: bool,
        unit: Option<String>,
        description: Option<String>,
    ) -> Result<()> {
        // mdfinfo metadata but no data
        let machine_endian: bool = cfg!(target_endian = "big");
        let data_signature = DataSignature {
            len: data.len(),
            data_type: arrow_to_mdf_data_type(&data, machine_endian),
            bit_count: arrow_bit_count(&data),
            byte_count: arrow_byte_count(&data),
            ndim: 1,
            shape: (vec![data.len()], Order::RowMajor),
        };
        let master_signature = MasterSignature {
            master_channel: master_channel.clone(),
            master_type,
            master_flag,
        };
        self.mdf_info.add_channel(
            channel_name.clone(),
            try_from(&data).context("failed converting ")?,
            data_signature,
            master_signature,
            unit,
            description,
        )?;
        Ok(())
    }
    /// Removes a channel in memory (no file modification)
    pub fn remove_channel(&mut self, channel_name: &str) {
        self.mdf_info.remove_channel(channel_name);
    }
    /// load all channels data in memory
    pub fn load_all_channels_data_in_memory(&mut self) -> Result<(), Error> {
        let channel_names = self.get_channel_names_set();
        self.load_channels_data_in_memory(channel_names)
            .context("failed loading channels data from file to memory")?;
        Ok(())
    }
    /// load a set of channels data in memory
    pub fn load_channels_data_in_memory(
        &mut self,
        channel_names: HashSet<String>,
    ) -> Result<(), Error> {
        let f: File = OpenOptions::new()
            .read(true)
            .write(false)
            .open(self.get_file_name())
            .with_context(|| format!("Cannot find the file {}", self.get_file_name()))?;
        let mut rdr = BufReader::new(&f);
        info!("Opened file {}", self.get_file_name());

        match &mut self.mdf_info {
            MdfInfo::V3(_mdfinfo3) => {
                mdfreader3(&mut rdr, self, &channel_names).with_context(|| {
                    format!(
                        "failed reading data from mdf3 file {}",
                        self.get_file_name()
                    )
                })?;
            }
            MdfInfo::V4(_mdfinfo4) => {
                mdfreader4(&mut rdr, self, &channel_names).with_context(|| {
                    format!(
                        "failed reading data from mdf4 file {}",
                        self.get_file_name()
                    )
                })?;
            }
        };
        info!("Loaded all channels data into memory");

        Ok(())
    }
    /// Clears all data arrays
    pub fn clear_all_channel_data_from_memory(&mut self) -> Result<()> {
        let channel_names = self.get_channel_names_set();
        self.mdf_info
            .clear_channel_data_from_memory(channel_names)?;
        Ok(())
    }

    /// Clears data arrays
    pub fn clear_channel_data_from_memory(&mut self, channel_names: HashSet<String>) -> Result<()> {
        self.mdf_info
            .clear_channel_data_from_memory(channel_names)?;
        Ok(())
    }
    /// keeps only the given channels in memory; all others have their data cleared.
    /// Master channels of any kept channel are automatically retained.
    pub fn keep_channels(&mut self, names: HashSet<String>) -> Result<()> {
        let mut masters_to_keep = HashSet::new();
        for n in &names {
            if let Some(m) = self.mdf_info.get_channel_master(n) {
                masters_to_keep.insert(m);
            }
        }
        let to_drop: HashSet<String> = self
            .get_channel_names_set()
            .into_iter()
            .filter(|n| !names.contains(n) && !masters_to_keep.contains(n))
            .collect();
        self.mdf_info.clear_channel_data_from_memory(to_drop)
    }

    /// slices all channels in the same group as `master_name` to [start_s, stop_s].
    /// The master channel must be a Time master (sync_type == 1) with Float64 data.
    pub fn cut(&mut self, master_name: &str, start_s: f64, stop_s: f64) -> Result<()> {
        if self.mdf_info.get_channel_master_type(master_name) != 1 {
            bail!("cut: '{master_name}' is not a Time master channel");
        }
        // Collect values before taking a mutable borrow
        let values: Vec<f64> = match self.get_channel_data(master_name) {
            Some(ChannelData::Float64(b)) => b.values_slice().to_vec(),
            Some(_) => bail!("cut: master channel '{master_name}' is not Float64"),
            None => bail!("cut: master channel '{master_name}' has no data loaded"),
        };
        let start_idx = values.partition_point(|&v| v < start_s);
        let stop_idx = values.partition_point(|&v| v <= stop_s);
        // Collect channel names in the group (owned, no borrow conflict)
        let mut channel_set = self.mdf_info.get_channel_names_cg_set(master_name);
        channel_set.insert(master_name.to_string());
        self.mdf_info
            .slice_channels(&channel_set, start_idx, stop_idx)
    }
    /// resamples all channels in the master's group to a uniform `raster_s` second grid.
    /// `master_name` must be a Time master (sync_type == 1) with Float64 data.
    /// Float32/64 channels are linearly interpolated; all others use previous-value hold.
    pub fn resample_group(&mut self, master_name: &str, raster_s: f64) -> Result<()> {
        if self.mdf_info.get_channel_master_type(master_name) != 1 {
            bail!("resample_group: '{master_name}' is not a Time master channel");
        }
        let old_master: Vec<f64> = match self.get_channel_data(master_name) {
            Some(ChannelData::Float64(b)) => b.values_slice().to_vec(),
            Some(_) => bail!("resample_group: master '{master_name}' is not Float64"),
            None => bail!("resample_group: master '{master_name}' has no data loaded"),
        };
        if old_master.len() < 2 {
            return Ok(());
        }
        let first = old_master[0];
        let last = *old_master.last().unwrap();
        // Build uniformly-spaced new master
        let new_master: Vec<f64> = {
            let n = ((last - first) / raster_s).floor() as usize + 1;
            (0..n).map(|i| first + i as f64 * raster_s).collect()
        };
        // Collect channel data clones before taking mutable borrow
        let mut channel_set = self.mdf_info.get_channel_names_cg_set(master_name);
        channel_set.insert(master_name.to_string());
        let cloned: Vec<(String, ChannelData)> = channel_set
            .iter()
            .filter_map(|name| {
                let data = self.get_channel_data(name)?;
                data.slice_range(0, data.len())
                    .ok()
                    .map(|d| (name.clone(), d))
            })
            .collect();
        // Apply interpolated data
        for (name, data) in cloned {
            let new_data = if name == master_name {
                let mut b = arrow::array::Float64Builder::with_capacity(new_master.len());
                new_master.iter().for_each(|&v| b.append_value(v));
                ChannelData::Float64(b)
            } else {
                interp_channel(&old_master, &data, &new_master)?
            };
            self.mdf_info.replace_channel_data(&name, new_data)?;
        }
        Ok(())
    }
    /// resamples every Time master group in the file to a uniform `raster_s` second grid.
    pub fn resample(&mut self, raster_s: f64) -> Result<()> {
        let masters: Vec<String> = self
            .mdf_info
            .get_master_channel_names_set()
            .into_keys()
            .flatten()
            .filter(|name| self.mdf_info.get_channel_master_type(name) == 1)
            .collect();
        for master in masters {
            self.resample_group(&master, raster_s)?;
        }
        Ok(())
    }
    /// appends `other` after `self` on the time axis.
    /// For each shared Time master group, other's timestamps are offset so they follow self's last
    /// timestamp. Channels present in only one file are null-padded for the missing span.
    /// Both files must have their data loaded in memory before calling this.
    pub fn concat_mdf(&mut self, other: &mut Mdf) -> Result<()> {
        use arrow::array::new_null_array;
        use arrow::compute::concat;
        let self_masters: Vec<String> = self
            .mdf_info
            .get_master_channel_names_set()
            .into_keys()
            .flatten()
            .filter(|name| self.mdf_info.get_channel_master_type(name) == 1)
            .collect();
        for master_name in self_masters {
            // Collect self's channel set (owned)
            let mut self_channels = self.mdf_info.get_channel_names_cg_set(&master_name);
            self_channels.insert(master_name.clone());
            // Check other has same master
            if other.mdf_info.get_channel_master_type(&master_name) != 1 {
                continue;
            }
            let other_channels: HashSet<String> = {
                let mut s = other.mdf_info.get_channel_names_cg_set(&master_name);
                s.insert(master_name.clone());
                s
            };
            // Compute time offset: self_last + sample_interval - other_first
            let self_master_vals: Vec<f64> = match self.get_channel_data(&master_name) {
                Some(ChannelData::Float64(b)) => b.values_slice().to_vec(),
                _ => continue,
            };
            let other_master_vals: Vec<f64> = match other.get_channel_data(&master_name) {
                Some(ChannelData::Float64(b)) => b.values_slice().to_vec(),
                _ => continue,
            };
            if self_master_vals.is_empty() || other_master_vals.is_empty() {
                continue;
            }
            let self_len = self_master_vals.len();
            let other_len = other_master_vals.len();
            let delta = if self_master_vals.len() >= 2 {
                self_master_vals[self_len - 1] - self_master_vals[self_len - 2]
            } else {
                0.0
            };
            let time_offset = self_master_vals[self_len - 1] + delta - other_master_vals[0];
            // Collect clones before mutable borrow
            let self_clones: HashMap<String, ChannelData> = self_channels
                .iter()
                .filter_map(|n| {
                    let d = self.get_channel_data(n)?;
                    d.slice_range(0, d.len()).ok().map(|c| (n.clone(), c))
                })
                .collect();
            let other_clones: HashMap<String, ChannelData> = other_channels
                .iter()
                .filter_map(|n| {
                    let d = other.get_channel_data(n)?;
                    d.slice_range(0, d.len()).ok().map(|c| (n.clone(), c))
                })
                .collect();
            // Channels in both: concatenate
            for name in self_channels.iter() {
                if let (Some(sd), Some(od)) = (self_clones.get(name), other_clones.get(name)) {
                    if name == &master_name {
                        // Build shifted other master
                        let shifted: Vec<f64> =
                            other_master_vals.iter().map(|&v| v + time_offset).collect();
                        let mut b =
                            arrow::array::Float64Builder::with_capacity(self_len + other_len);
                        sd.finish_cloned()
                            .as_any()
                            .downcast_ref::<arrow::array::Float64Array>()
                            .into_iter()
                            .flat_map(|a| a.iter())
                            .for_each(|v| b.append_option(v));
                        shifted.iter().for_each(|&v| b.append_value(v));
                        self.mdf_info
                            .replace_channel_data(name, ChannelData::Float64(b))?;
                    } else {
                        let self_arr = sd.finish_cloned();
                        let other_arr = od.finish_cloned();
                        let combined = concat(&[self_arr.as_ref(), other_arr.as_ref()])?;
                        self.mdf_info
                            .replace_channel_data(name, try_from(&*combined)?)?;
                    }
                } else if let Some(sd) = self_clones.get(name) {
                    // Only in self — append nulls for other's length
                    let self_arr = sd.finish_cloned();
                    let null_arr = new_null_array(self_arr.data_type(), other_len);
                    let combined = concat(&[self_arr.as_ref(), null_arr.as_ref()])?;
                    self.mdf_info
                        .replace_channel_data(name, try_from(&*combined)?)?;
                }
            }
            // Channels only in other — prepend nulls and add
            for name in other_channels.iter() {
                if self_channels.contains(name) || name == &master_name {
                    continue;
                }
                if let Some(od) = other_clones.get(name) {
                    let other_arr = od.finish_cloned();
                    let null_arr = new_null_array(other_arr.data_type(), self_len);
                    let combined = concat(&[null_arr.as_ref(), other_arr.as_ref()])?;
                    self.add_channel(
                        name.clone(),
                        combined,
                        Some(master_name.clone()),
                        None,
                        false,
                        other.get_channel_unit(name).unwrap_or(None),
                        None,
                    )?;
                }
            }
        }
        Ok(())
    }
    /// imports channels from `other` into `self` (horizontal join on shared time axis).
    /// Channels already present in `self` are skipped. Master channels from `other` are
    /// added only when at least one of their data channels is being imported.
    /// Both files must have their data loaded in memory before calling this.
    pub fn merge(&mut self, other: &mut Mdf) -> Result<()> {
        let self_channels = self.get_channel_names_set();
        let other_masters = other.mdf_info.get_master_channel_names_set();
        for (master_opt, channel_set) in other_masters {
            let master_name = match &master_opt {
                Some(m) => m.clone(),
                None => continue,
            };
            // Determine which channels from this group need to be imported
            let to_import: Vec<String> = channel_set
                .into_iter()
                .filter(|n| !self_channels.contains(n) && n != &master_name)
                .collect();
            if to_import.is_empty() {
                continue;
            }
            // Add master first if not already in self
            if !self_channels.contains(&master_name)
                && let Some(master_data) = other.get_channel_data(&master_name)
            {
                let arr = master_data.finish_cloned();
                let master_type = other.mdf_info.get_channel_master_type(&master_name);
                self.add_channel(
                    master_name.clone(),
                    arr,
                    None,
                    Some(master_type),
                    true,
                    other.get_channel_unit(&master_name).unwrap_or(None),
                    None,
                )?;
            }
            // Add each data channel
            for name in to_import {
                if let Some(data) = other.get_channel_data(&name) {
                    let arr = data.finish_cloned();
                    self.add_channel(
                        name.clone(),
                        arr,
                        Some(master_name.clone()),
                        None,
                        false,
                        other.get_channel_unit(&name).unwrap_or(None),
                        None,
                    )?;
                }
            }
        }
        Ok(())
    }
    /// export to Parquet files, one for each channel group (or dataframe)
    #[cfg(feature = "parquet")]
    pub fn export_to_parquet(&self, file_name: &str, compression: Option<&str>) -> Result<()> {
        export_to_parquet(self, file_name, compression)
    }
    /// export a dataframe including a given channel to a Parquet file
    #[cfg(feature = "parquet")]
    pub fn export_dataframe_to_parquet(
        &self,
        channel_name: String,
        file_name: &str,
        compression: Option<&str>,
    ) -> Result<()> {
        export_dataframe_to_parquet(self, &channel_name, file_name, compression)
    }
    /// export a dataframe including a given channel to a hdf5 file
    #[cfg(feature = "hdf5")]
    pub fn export_dataframe_to_hdf5(
        &self,
        channel_name: String,
        file_name: &str,
        compression: Option<&str>,
    ) -> Result<()> {
        export_dataframe_to_hdf5(self, &channel_name, file_name, compression)
    }
    /// export all data to hdf5 file
    #[cfg(feature = "hdf5")]
    pub fn export_to_hdf5(&self, file_name: &str, compression: Option<&str>) -> Result<()> {
        export_to_hdf5(self, file_name, compression)
    }
    /// Exports all loaded channel groups to CSV (one file per channel group).
    pub fn export_to_csv(&self, file_name: &str) -> Result<()> {
        export_to_csv(self, file_name)
    }
    /// Exports the channel group containing `channel_name` to a CSV file.
    pub fn export_dataframe_to_csv(&self, channel_name: &str, file_name: &str) -> Result<()> {
        export_dataframe_to_csv(self, channel_name, file_name)
    }
    /// Writes mdf4 file
    pub fn write(&mut self, file_name: &str, compression: CompressionAlgorithm) -> Result<Mdf> {
        mdfwriter4(self, file_name, compression)
    }
    /// Returns a Polars [`Series`] for the named channel.
    ///
    /// The channel must already be loaded in memory. Returns an error if the
    /// channel is not found or has an unsupported type.
    #[cfg(feature = "polars")]
    pub fn get_channel_polars_series(
        &mut self,
        channel_name: &str,
    ) -> polars::prelude::PolarsResult<polars::prelude::Series> {
        let data = self.get_channel_data(channel_name).ok_or_else(|| {
            polars::prelude::PolarsError::ColumnNotFound(
                format!("channel '{channel_name}' not found").into(),
            )
        })?;
        crate::export::polars::channel_data_to_series(channel_name, data)
    }
    /// Returns a Polars [`DataFrame`] for all channels sharing `master_channel_name`.
    ///
    /// Pass `None` for channels that have no master. All channels in the group
    /// (including the master itself) become columns. Channels with unsupported
    /// types are silently skipped. All channels must already be loaded in memory.
    #[cfg(feature = "polars")]
    pub fn get_channel_polars_dataframe(
        &mut self,
        master_channel_name: Option<&str>,
    ) -> polars::prelude::PolarsResult<polars::prelude::DataFrame> {
        crate::export::polars::mdf_master_to_dataframe(self, master_channel_name)
    }
    /// Returns one Polars [`DataFrame`] per channel group (one per master channel).
    ///
    /// The map key is the master channel name (`None` = channels without a master).
    /// All channels must already be loaded in memory.
    #[cfg(feature = "polars")]
    pub fn get_polars_dataframes(
        &mut self,
    ) -> polars::prelude::PolarsResult<
        std::collections::HashMap<Option<String>, polars::prelude::DataFrame>,
    > {
        crate::export::polars::mdf_to_dataframes(self)
    }
}

impl fmt::Display for Mdf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let format_option = FormatOptions::new();
        match &self.mdf_info {
            MdfInfo::V3(mdfinfo3) => {
                writeln!(f, "Version : {}\n", mdfinfo3.id_block.id_ver)?;
                writeln!(
                    f,
                    "Header :\n Author: {}  Organisation:{}\n",
                    mdfinfo3.hd_block.hd_author, mdfinfo3.hd_block.hd_organization
                )?;
                writeln!(
                    f,
                    "Project: {}  Subject:{}\n",
                    mdfinfo3.hd_block.hd_project, mdfinfo3.hd_block.hd_subject
                )?;
                writeln!(
                    f,
                    "Date: {:?}  Time:{:?}\n",
                    mdfinfo3.hd_block.hd_date, mdfinfo3.hd_block.hd_time
                )?;
                writeln!(f, "Comments: {}", mdfinfo3.hd_comment)?;
                for (master, list) in &self.get_master_channel_names_set() {
                    if let Some(master_name) = master {
                        writeln!(f, "\nMaster: {master_name}")?;
                    } else {
                        writeln!(f, "\nWithout Master channel")?;
                    }
                    for channel in list {
                        writeln!(f, " {channel} ")?;
                        if let Some(data) = self.get_channel_converted_data(channel)
                            && !data.is_empty()
                        {
                            let array = &data.as_ref();
                            let displayer = ArrayFormatter::try_new(array, &format_option)
                                .map_err(|e| {
                                    log::warn!("Mdf Display: ArrayFormatter failed: {e}");
                                    std::fmt::Error
                                })?;
                            write!(f, "{}", displayer.value(0))?;
                            write!(f, " ")?;
                            write!(f, "{}", displayer.value(data.len() - 1))?;
                        }
                        if let Ok(Some(unit)) = self.get_channel_unit(channel) {
                            writeln!(f, " {unit} ")?;
                        }
                        if let Ok(Some(desc)) = self.get_channel_desc(channel) {
                            writeln!(f, " {desc} ")?;
                        }
                    }
                }
                writeln!(f, "\n")
            }
            MdfInfo::V4(mdfinfo4) => {
                writeln!(f, "Version : {}", mdfinfo4.id_block.id_ver)?;
                writeln!(f, "{}\n", mdfinfo4.hd_block)?;
                if let Some(hd) = mdfinfo4
                    .sharable
                    .get_hd_comments(mdfinfo4.hd_block.hd_md_comment)
                {
                    writeln!(f, "{hd}")?;
                }
                for (master, list) in &self.get_master_channel_names_set() {
                    if let Some(master_name) = master {
                        writeln!(f, "\nMaster: {master_name}")?;
                    } else {
                        writeln!(f, "\nWithout Master channel")?;
                    }
                    for channel in list {
                        writeln!(f, " {channel} ")?;
                        if let Some(data) = self.get_channel_converted_data(channel)
                            && !data.is_empty()
                        {
                            let array = &data.as_ref();
                            let displayer = ArrayFormatter::try_new(array, &format_option)
                                .map_err(|e| {
                                    log::warn!("Mdf Display: ArrayFormatter failed: {e}");
                                    std::fmt::Error
                                })?;
                            write!(f, "{}", displayer.value(0))?;
                            write!(f, " ")?;
                            write!(f, "{}", displayer.value(data.len() - 1))?;
                        }
                        if let Ok(Some(unit)) = self.get_channel_unit(channel) {
                            writeln!(f, " {unit} ")?;
                        }
                        if let Ok(Some(desc)) = self.get_channel_desc(channel) {
                            writeln!(f, " {desc} ")?;
                        }
                    }
                }
                writeln!(f, "\n")
            }
        }
    }
}
