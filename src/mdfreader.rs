//! This module contains the data reading features
pub mod conversions3;
pub mod conversions4;
pub mod data_read3;
pub mod data_read4;
pub mod mdfreader3;
pub mod mdfreader4;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::BufReader;
use std::sync::Arc;

use anyhow::{Context, Error, Result};
use arrow::array::Array;
use arrow::util::display::{ArrayFormatter, FormatOptions};
use log::info;
#[cfg(feature = "numpy")]
use pyo3::prelude::*;

//use crate::export::parquet::export_to_parquet;
use crate::data_holder::channel_data::try_from;
use crate::mdfinfo::MdfInfo;
use crate::mdfreader::mdfreader3::mdfreader3;
use crate::mdfreader::mdfreader4::mdfreader4;
use crate::mdfwriter::mdfwriter4::mdfwriter4;

#[cfg(feature = "parquet")]
use crate::export::parquet::export_dataframe_to_parquet;
#[cfg(feature = "parquet")]
use crate::export::parquet::export_to_parquet;

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
    /// returns a dict of master names keys for which values are a set of associated channel names
    pub fn get_master_channel_names_set(&self) -> HashMap<Option<String>, HashSet<String>> {
        self.mdf_info.get_master_channel_names_set()
    }
    /// returns channel's arrow Array.
    pub fn get_channel_data(&self, channel_name: &str) -> Option<&ChannelData> {
        match &self.mdf_info {
            MdfInfo::V3(mdfinfo3) => mdfinfo3.get_channel_data(channel_name),
            MdfInfo::V4(mdfinfo4) => mdfinfo4.get_channel_data(channel_name),
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

    /// export to Parquet files
    #[cfg(feature = "parquet")]
    pub fn export_to_parquet(&self, file_name: &str, compression: Option<&str>) -> Result<()> {
        export_to_parquet(self, file_name, compression)
    }
    /// export to Parquet files
    #[cfg(feature = "parquet")]
    pub fn export_dataframe_to_parquet(
        &self,
        channel_name: String,
        file_name: &str,
        compression: Option<&str>,
    ) -> Result<()> {
        export_dataframe_to_parquet(self, &channel_name, file_name, compression)
    }
    /// Writes mdf4 file
    pub fn write(&mut self, file_name: &str, compression: bool) -> Result<Mdf> {
        mdfwriter4(self, file_name, compression)
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
                for (master, list) in self.get_master_channel_names_set().iter() {
                    if let Some(master_name) = master {
                        writeln!(f, "\nMaster: {master_name}")
                            .expect("cannot print master channel name");
                    } else {
                        writeln!(f, "\nWithout Master channel")
                            .expect("cannot print thre is no master channel");
                    }
                    for channel in list.iter() {
                        writeln!(f, " {channel} ").expect("cannot print channel name");
                        if let Some(data) = self.get_channel_data(channel) {
                            if !data.is_empty() {
                                let array = &data.as_ref();
                                let displayer = ArrayFormatter::try_new(array, &format_option)
                                    .map_err(|_| std::fmt::Error)?;
                                write!(f, "{}", displayer.value(0)).expect("cannot channel data");
                                write!(f, " ").expect("cannot print simple space character");
                                write!(f, "{}", displayer.value(data.len() - 1))
                                    .expect("cannot channel data");
                            }
                        }
                        if let Ok(Some(unit)) = self.get_channel_unit(channel) {
                            writeln!(f, " {unit} ").expect("cannot print channel unit");
                        }
                        if let Ok(Some(desc)) = self.get_channel_desc(channel) {
                            writeln!(f, " {desc} ").expect("cannot print channel desc");
                        }
                    }
                }
                writeln!(f, "\n")
            }
            MdfInfo::V4(mdfinfo4) => {
                writeln!(f, "Version : {}", mdfinfo4.id_block.id_ver)?;
                writeln!(f, "{}\n", mdfinfo4.hd_block)?;
                let comments = &mdfinfo4
                    .sharable
                    .get_hd_comments(mdfinfo4.hd_block.hd_md_comment);
                for c in comments.iter() {
                    writeln!(f, "{} {}", c.0, c.1)?;
                }
                for (master, list) in self.get_master_channel_names_set().iter() {
                    if let Some(master_name) = master {
                        writeln!(f, "\nMaster: {master_name}")
                            .expect("cannot print master channel name");
                    } else {
                        writeln!(f, "\nWithout Master channel")
                            .expect("cannot print thre is no master channel");
                    }
                    for channel in list.iter() {
                        writeln!(f, " {channel} ").expect("cannot print channel name");
                        if let Some(data) = self.get_channel_data(channel) {
                            if !data.is_empty() {
                                let array = &data.as_ref();
                                let displayer = ArrayFormatter::try_new(array, &format_option)
                                    .map_err(|_| std::fmt::Error)?;
                                write!(f, "{}", displayer.value(0)).expect("cannot channel data");
                                write!(f, " ").expect("cannot print simple space character");
                                write!(f, "{}", displayer.value(data.len() - 1))
                                    .expect("cannot channel data");
                            }
                        }
                        if let Ok(Some(unit)) = self.get_channel_unit(channel) {
                            writeln!(f, " {unit} ").expect("cannot print channel unit");
                        }
                        if let Ok(Some(desc)) = self.get_channel_desc(channel) {
                            writeln!(f, " {desc} ").expect("cannot print channel desc");
                        }
                    }
                }
                writeln!(f, "\n")
            }
        }
    }
}
