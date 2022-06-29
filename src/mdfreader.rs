//! This module contains the data reading features
pub mod channel_data;
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

use arrow2::array::{get_display, Array};
use arrow2::chunk::Chunk;
use arrow2::datatypes::{Field, Schema};
use arrow2::error::Result;

use crate::export::arrow::mdf_data_to_arrow;
use crate::export::parquet::export_to_parquet;
use crate::mdfinfo::MdfInfo;
use crate::mdfreader::mdfreader3::mdfreader3;
use crate::mdfreader::mdfreader4::mdfreader4;
use crate::mdfwriter::mdfwriter4::mdfwriter4;

#[derive(Debug)]
pub struct Mdf {
    /// MdfInfo enum
    pub mdf_info: MdfInfo,
    /// contains the file data according to Arrow memory layout
    pub arrow_data: Vec<Chunk<Arc<dyn Array>>>,
    /// arrow schema and metadata for the data
    pub arrow_schema: Schema,
    /// tuple of chunk index, array index and field index
    pub channel_indexes: HashMap<String, ChannelIndexes>,
}

#[derive(Debug, Default)]
pub struct ChannelIndexes {
    /// index of the chunk in the arrow_data vector
    pub chunk_index: usize,
    /// index of the array in the chunk
    pub array_index: usize,
    /// index of the field in the schema
    pub field_index: usize,
}

/// reads files metadata and loads all channels data into memory
pub fn mdfreader(file_name: &str) -> Mdf {
    // read file blocks
    let mut mdf = Mdf::new(file_name);
    mdf.load_all_channels_data_in_memory();
    mdf
}

impl Mdf {
    /// returns new empty Mdf
    pub fn new(file_name: &str) -> Mdf {
        let mut mdf = Mdf {
            mdf_info: MdfInfo::new(file_name),
            arrow_data: Vec::new(),
            arrow_schema: Schema::default(),
            channel_indexes: HashMap::new(),
        };
        mdf.arrow_schema
            .metadata
            .insert("file_name".to_string(), file_name.to_string());
        mdf
    }
    pub fn get_file_name(&self) -> String {
        match &self.mdf_info {
            MdfInfo::V3(mdfinfo3) => mdfinfo3.file_name.clone(),
            MdfInfo::V4(mdfinfo4) => mdfinfo4.file_name.clone(),
        }
    }
    /// gets the version of mdf file
    pub fn get_version(&mut self) -> u16 {
        self.mdf_info.get_version()
    }
    /// returns channel's unit string
    pub fn get_channel_unit(&self, channel_name: &str) -> Option<String> {
        self.mdf_info.get_channel_unit(channel_name)
    }
    /// Sets the channel unit in memory
    pub fn set_channel_unit(&mut self, channel_name: &str, unit: &str) {
        self.mdf_info.set_channel_unit(channel_name, unit)
    }
    /// returns channel's description string
    pub fn get_channel_desc(&self, channel_name: &str) -> Option<String> {
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
    pub fn set_channel_master_type(&mut self, master_name: &str, master_type: u8) {
        self.mdf_info
            .set_channel_master_type(master_name, master_type)
    }
    /// returns a set of all channel names contained in file
    pub fn get_channel_names_set(&self) -> HashSet<String> {
        self.mdf_info.get_channel_names_set()
    }
    /// returns a dict of master names keys for which values are a set of associated channel names
    pub fn get_master_channel_names_set(&self) -> HashMap<Option<String>, HashSet<String>> {
        self.mdf_info.get_master_channel_names_set()
    }
    /// return the tuple of chunk index and array index corresponding to the channel name
    fn get_channel_index(&self, channel_name: &str) -> Option<&ChannelIndexes> {
        self.channel_indexes.get(channel_name)
    }
    /// returns channel's arrow2 Array.
    pub fn get_channel_data(&self, channel_name: &str) -> Option<Arc<dyn Array>> {
        if let Some(index) = self.get_channel_index(channel_name) {
            Some(self.arrow_data[index.chunk_index][index.array_index].clone())
        } else {
            None
        }
    }
    /// returns channel's arrow2 Field.
    pub fn get_channel_field(&self, channel_name: &str) -> Option<&Field> {
        if let Some(index) = self.get_channel_index(channel_name) {
            Some(&self.arrow_schema.fields[index.field_index])
        } else {
            None
        }
    }
    /// defines channel's data in memory
    pub fn set_channel_data(&mut self, channel_name: &str, data: Arc<dyn Array>) {
        todo!()
    }
    /// Renames a channel's name in memory
    pub fn rename_channel(&mut self, channel_name: &str, new_name: &str) {
        self.mdf_info.rename_channel(channel_name, new_name);
        todo!()
    }
    /// Adds a new channel in memory (no file modification)
    pub fn add_channel(
        &mut self,
        channel_name: String,
        data: Arc<dyn Array>,
        master_channel: Option<String>,
        master_type: Option<u8>,
        master_flag: bool,
        unit: Option<String>,
        description: Option<String>,
    ) {
        todo!()
    }
    /// Removes a channel in memory (no file modification)
    pub fn remove_channel(&mut self, channel_name: &str) {
        self.mdf_info.remove_channel(channel_name);
        todo!() // adapt to new structure
    }
    /// load all channels data in memory
    pub fn load_all_channels_data_in_memory(&mut self) {
        let channel_names = self.get_channel_names_set();
        self.load_channels_data_in_memory(channel_names);
    }
    /// load a set of channels data in memory
    pub fn load_channels_data_in_memory(&mut self, channel_names: HashSet<String>) {
        let f: File = OpenOptions::new()
            .read(true)
            .write(false)
            .open(self.get_file_name())
            .expect("Cannot find the file");
        let mut rdr = BufReader::new(&f);
        match &mut self.mdf_info {
            MdfInfo::V3(_mdfinfo3) => {
                mdfreader3(&mut rdr, self, &channel_names);
            }
            MdfInfo::V4(_mdfinfo4) => {
                mdfreader4(&mut rdr, self, &channel_names);
            }
        };
        // move the data from the MdfInfo3 structure into vect of chunks
        mdf_data_to_arrow(self, &channel_names);

        // bit masking of the data
        //self.apply_bit_mask_offset(&channel_names);

        // conversion of all channels to physical values
        // convert_all_channels(dg, &info.sharable);
    }
    /// Clears all data arrays
    pub fn clear_all_channel_data_from_memory(&mut self) {
        self.arrow_data = Vec::new();
        self.arrow_schema = Schema::default();
        self.channel_indexes = HashMap::new();
    }

    /// Clears data arrays
    pub fn clear_channel_data_from_memory(&mut self, channel_names: HashSet<String>) {
        self.arrow_data = Vec::new();
        self.arrow_schema = Schema::default();
        self.channel_indexes = HashMap::new();
    }

    /// export to Parquet file
    pub fn export_to_parquet(&mut self, file_name: &str, compression: Option<&str>) -> Result<()> {
        export_to_parquet(self, file_name, compression)
    }
    /// Writes mdf4 file
    pub fn write(&mut self, file_name: &str, compression: bool) -> Mdf {
        mdfwriter4(self, file_name, compression)
    }
    // TODO cut data
    // TODO resample data
}

impl fmt::Display for Mdf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
                        writeln!(f, "\nMaster: {}", master_name)
                            .expect("cannot print master channel name");
                    } else {
                        writeln!(f, "\nWithout Master channel")
                            .expect("cannot print thre is no master channel");
                    }
                    for channel in list.iter() {
                        writeln!(f, " {} ", channel).expect("cannot print channel name");
                        if let Some(data) = self.get_channel_data(channel) {
                            if !data.is_empty() {
                                let displayer = get_display(data.as_ref(), "null");
                                displayer(f, 0).expect("cannot channel data");
                                writeln!(f, " ").expect("cannot print simple space character");
                                displayer(f, data.len() - 1).expect("cannot channel data");
                            }
                        }
                        if let Some(unit) = self.get_channel_unit(channel) {
                            writeln!(f, " {} ", unit).expect("cannot print channel unit");
                        }
                        if let Some(desc) = self.get_channel_desc(channel) {
                            writeln!(f, " {} ", desc).expect("cannot print channel desc");
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
                    .get_comments(mdfinfo4.hd_block.hd_md_comment);
                for c in comments.iter() {
                    writeln!(f, "{} {}", c.0, c.1)?;
                }
                for (master, list) in self.get_master_channel_names_set().iter() {
                    if let Some(master_name) = master {
                        writeln!(f, "\nMaster: {}", master_name)
                            .expect("cannot print master channel name");
                    } else {
                        writeln!(f, "\nWithout Master channel")
                            .expect("cannot print thre is no master channel");
                    }
                    for channel in list.iter() {
                        writeln!(f, " {} ", channel).expect("cannot print channel name");
                        if let Some(data) = self.get_channel_data(channel) {
                            if !data.is_empty() {
                                let displayer = get_display(data.as_ref(), "null");
                                displayer(f, 0).expect("cannot channel data");
                                writeln!(f, " ").expect("cannot print simple space character");
                                displayer(f, data.len() - 1).expect("cannot channel data");
                            }
                        }
                        if let Some(unit) = self.get_channel_unit(channel) {
                            writeln!(f, " {} ", unit).expect("cannot print channel unit");
                        }
                        if let Some(desc) = self.get_channel_desc(channel) {
                            writeln!(f, " {} ", desc).expect("cannot print channel desc");
                        }
                    }
                }
                writeln!(f, "\n")
            }
        }
    }
}
