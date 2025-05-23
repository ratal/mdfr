//! Parsing of file metadata into MdfInfo3 struct
use anyhow::{Context, Error, Result};
use arrow::array::{UInt16Builder, UInt32Builder, UInt8Builder};
use binrw::{BinRead, BinReaderExt};
use byteorder::{LittleEndian, ReadBytesExt};
use chrono::NaiveDate;
use encoding_rs::Encoding;
use log::info;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::default::Default;
use std::fmt;
use std::fs::File;
use std::io::{prelude::*, Cursor};

use crate::data_holder::channel_data::{data_type_init, ChannelData};
use crate::data_holder::tensor_arrow::Order;
use crate::mdfinfo::IdBlock;

use super::sym_buf_reader::SymBufReader;

/// Specific to version 3.x mdf metadata structure
#[derive(Debug)]
#[repr(C)]
pub struct MdfInfo3 {
    /// file name string
    pub file_name: String,
    /// Identification block
    pub id_block: IdBlock,
    /// code page
    pub encoding: &'static Encoding,
    /// Header block
    pub hd_block: Hd3,
    /// Header comments
    pub hd_comment: String,
    /// data group block linking channel group/channel/conversion/..etc. and data block
    pub dg: BTreeMap<u32, Dg3>,
    /// Conversion and CE blocks
    pub sharable: SharableBlocks3,
    /// set of all channel names
    pub channel_names_set: ChannelNamesSet3,
}

pub(crate) type ChannelId3 = (Option<String>, u32, (u32, u16), u32);
pub(crate) type ChannelNamesSet3 = HashMap<String, ChannelId3>;

/// MdfInfo3's implementation
#[allow(dead_code)]
impl MdfInfo3 {
    pub fn get_channel_id(&self, channel_name: &str) -> Option<&ChannelId3> {
        self.channel_names_set.get(channel_name)
    }
    /// Returns the channel's unit string. If it does not exist, it is an empty string.
    pub fn get_channel_unit(&self, channel_name: &str) -> Option<String> {
        let mut unit: Option<String> = None;
        if let Some((_master, dg_pos, (_cg_pos, rec_id), cn_pos)) =
            self.get_channel_id(channel_name)
        {
            if let Some(dg) = self.dg.get(dg_pos) {
                if let Some(cg) = dg.cg.get(rec_id) {
                    if let Some(cn) = cg.cn.get(cn_pos) {
                        unit = self._get_unit(&cn.block1.cn_cc_conversion);
                    }
                }
            }
        }
        unit
    }
    pub fn _get_unit(&self, cc_position: &u32) -> Option<String> {
        if let Some(array) = self.sharable.cc.get(cc_position) {
            let txt = array.0.cc_unit;
            let u: String = self.encoding.decode(&txt).0.to_string();
            Some(u.trim_end_matches(char::from(0)).to_string())
        } else {
            None
        }
    }
    /// Returns the channel's description. If it does not exist, it is an empty string
    pub fn get_channel_desc(&self, channel_name: &str) -> Option<String> {
        let mut desc: Option<String> = None;
        if let Some((_master, dg_pos, (_cg_pos, rec_id), cn_pos)) =
            self.get_channel_id(channel_name)
        {
            if let Some(dg) = self.dg.get(dg_pos) {
                if let Some(cg) = dg.cg.get(rec_id) {
                    if let Some(cn) = cg.cn.get(cn_pos) {
                        desc = Some(cn.description.clone());
                    }
                }
            }
        }
        desc
    }
    /// returns the master channel associated to the input channel name
    pub fn get_channel_master(&self, channel_name: &str) -> Option<String> {
        let mut master = None;
        if let Some((m, _dg_pos, (_cg_pos, _rec_idd), _cn_pos)) = self.get_channel_id(channel_name)
        {
            master.clone_from(m);
        }
        master
    }
    /// returns type of master channel link to channel input in parameter:
    /// 0 = None (normal data channels), 1 = Time (seconds),
    pub fn get_channel_master_type(&self, channel_name: &str) -> u8 {
        let mut master_type: u16 = 0; // default to normal data channel
        if let Some((_master, dg_pos, (_cg_pos, rec_id), cn_pos)) =
            self.get_channel_id(channel_name)
        {
            if let Some(dg) = self.dg.get(dg_pos) {
                if let Some(cg) = dg.cg.get(rec_id) {
                    if let Some(cn) = cg.cn.get(cn_pos) {
                        master_type = cn.block1.cn_type;
                    }
                }
            }
        }
        master_type as u8
    }
    /// returns the set of channel names
    pub fn get_channel_names_set(&self) -> HashSet<String> {
        self.channel_names_set.keys().cloned().collect()
    }
    /// returns the set of channel names that are in same channel group as input channel name
    pub fn get_channel_names_cg_set(&self, channel_name: &str) -> HashSet<String> {
        if let Some((_master, dg_pos, (_cg_pos, rec_id), _cn_pos)) =
            self.get_channel_id(channel_name)
        {
            let mut channel_list = HashSet::new();
            if let Some(dg) = self.dg.get(dg_pos) {
                if let Some(cg) = dg.cg.get(rec_id) {
                    channel_list.clone_from(&cg.channel_names);
                }
            }
            channel_list
        } else {
            HashSet::new()
        }
    }
    /// returns a hashmap for which master channel names are keys and values its corresponding set of channel names
    pub fn get_master_channel_names_set(&self) -> HashMap<Option<String>, HashSet<String>> {
        let mut channel_master_list: HashMap<Option<String>, HashSet<String>> = HashMap::new();
        for (_dg_position, dg) in self.dg.iter() {
            for (_record_id, cg) in dg.cg.iter() {
                if let Some(list) = channel_master_list.get_mut(&None) {
                    list.extend(list.clone());
                } else {
                    channel_master_list
                        .insert(cg.master_channel_name.clone(), cg.channel_names.clone());
                }
            }
        }
        channel_master_list
    }
    // empty the channels' ndarray
    pub fn clear_channel_data_from_memory(
        &mut self,
        channel_names: HashSet<String>,
    ) -> Result<(), Error> {
        for channel_name in channel_names {
            if let Some((_master, dg_pos, (_cg_pos, rec_id), cn_pos)) =
                self.channel_names_set.get_mut(&channel_name)
            {
                if let Some(dg) = self.dg.get_mut(dg_pos) {
                    if let Some(cg) = dg.cg.get_mut(rec_id) {
                        if let Some(cn) = cg.cn.get_mut(cn_pos) {
                            if !cn.data.is_empty() {
                                cn.data = cn.data.zeros(0, 0, 0, (Vec::new(), Order::RowMajor))?;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
    /// Returns the channel's data ndarray if present in memory, otherwise None.
    pub fn get_channel_data_from_memory(&self, channel_name: &str) -> Option<&ChannelData> {
        let mut data: Option<&ChannelData> = None;
        if let Some((_master, dg_pos, (_cg_pos, rec_id), cn_pos)) =
            self.get_channel_id(channel_name)
        {
            if let Some(dg) = self.dg.get(dg_pos) {
                if let Some(cg) = dg.cg.get(rec_id) {
                    if let Some(cn) = cg.cn.get(cn_pos) {
                        if !cn.data.is_empty() {
                            data = Some(&cn.data);
                        }
                    }
                }
            }
        }
        data
    }
    /// Removes a channel in memory (no file modification)
    pub fn remove_channel(&mut self, channel_name: &str) {
        if let Some((_master, dg_pos, (_cg_pos, rec_id), cn_pos)) =
            self.channel_names_set.get(channel_name)
        {
            if let Some(dg) = self.dg.get_mut(dg_pos) {
                if let Some(cg) = dg.cg.get_mut(rec_id) {
                    cg.cn.remove(cn_pos);
                    cg.channel_names.remove(channel_name);
                    self.channel_names_set.remove(channel_name);
                }
            }
        }
    }
    /// True if channel contains data
    pub fn get_channel_data_validity(&self, channel_name: &str) -> bool {
        let mut state: bool = false;
        if let Some((_master, dg_pos, (_cg_pos, rec_id), cn_pos)) =
            self.get_channel_id(channel_name)
        {
            if let Some(dg) = self.dg.get(dg_pos) {
                if let Some(cg) = dg.cg.get(rec_id) {
                    if let Some(cn) = cg.cn.get(cn_pos) {
                        state = cn.channel_data_valid
                    }
                }
            }
        }
        state
    }
    /// returns channel's data ndarray.
    pub fn get_channel_data(&self, channel_name: &str) -> Option<&ChannelData> {
        self.get_channel_data_from_memory(channel_name)
    }
    /// Renames a channel's name in memory
    pub fn rename_channel(&mut self, channel_name: &str, new_name: &str) {
        if let Some((master, dg_pos, (cg_pos, rec_id), cn_pos)) =
            self.channel_names_set.remove(channel_name)
        {
            if let Some(dg) = self.dg.get_mut(&dg_pos) {
                if let Some(cg) = dg.cg.get_mut(&rec_id) {
                    if let Some(cn) = cg.cn.get_mut(&cn_pos) {
                        cn.unique_name = new_name.to_string();
                        cg.channel_names.remove(channel_name);
                        cg.channel_names.insert(new_name.to_string());
                        if let Some(master_name) = &master {
                            if master_name == channel_name {
                                cg.master_channel_name = Some(new_name.to_string());
                                cg.channel_names.iter().for_each(|channel| {
                                    if let Some(val) = self.channel_names_set.get_mut(channel) {
                                        val.0 = Some(new_name.to_string());
                                        val.1 = dg_pos;
                                        val.2 = (cg_pos, rec_id);
                                        val.3 = cn_pos;
                                    }
                                });
                            }
                        }

                        self.channel_names_set.insert(
                            new_name.to_string(),
                            (master, dg_pos, (cg_pos, rec_id), cn_pos),
                        );
                    }
                }
            }
        }
    }
    /// Sets the channel unit in memory
    pub fn set_channel_unit(&mut self, channel_name: &str, unit: &str) {
        if let Some((_master, dg_pos, (_cg_pos, rec_id), cn_pos)) =
            self.channel_names_set.get(channel_name)
        {
            if let Some(dg) = self.dg.get_mut(dg_pos) {
                if let Some(cg) = dg.cg.get_mut(rec_id) {
                    if let Some(cn) = cg.cn.get_mut(cn_pos) {
                        let unit_vector = self.encoding.encode(unit).0;
                        let mut unit_array = [0; 20];
                        unit_array = match unit_vector.len().cmp(&20) {
                            Ordering::Less => {
                                unit_array[0..unit_vector.len()].copy_from_slice(&unit_vector);
                                unit_array
                            }
                            _ => {
                                unit_array.copy_from_slice(&unit_vector[0..20]);
                                unit_array
                            }
                        };
                        if cn.block1.cn_cc_conversion != 0 {
                            if let Some(array) =
                                self.sharable.cc.get_mut(&cn.block1.cn_cc_conversion)
                            {
                                array.0.cc_unit = unit_array;
                            }
                        } else {
                            let cc_position = rand::random::<u32>();
                            cn.block1.cn_cc_conversion = cc_position;
                            let cc_block = Cc3Block {
                                cc_unit: unit_array,
                                ..Default::default()
                            };
                            self.sharable
                                .cc
                                .insert(cc_position, (cc_block, Conversion::default()));
                        }
                    }
                }
            }
        }
    }
    /// Sets the channel description in memory
    pub fn set_channel_desc(&mut self, channel_name: &str, desc: &str) {
        if let Some((_master, dg_pos, (_cg_pos, rec_id), cn_pos)) =
            self.channel_names_set.get(channel_name)
        {
            if let Some(dg) = self.dg.get_mut(dg_pos) {
                if let Some(cg) = dg.cg.get_mut(rec_id) {
                    if let Some(cn) = cg.cn.get_mut(cn_pos) {
                        cn.description = desc.to_string();
                    }
                }
            }
        }
    }
}

/// MdfInfo3 display implementation
impl fmt::Display for MdfInfo3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MdfInfo3: {}", self.file_name)?;
        writeln!(f, "Version : {}\n", self.id_block.id_ver)?;
        writeln!(f, "{}\n", self.hd_block)?;
        for (master, list) in self.get_master_channel_names_set().iter() {
            if let Some(master_name) = master {
                writeln!(f, "\nMaster: {master_name}\n")?;
            } else {
                writeln!(f, "\nWithout Master channel\n")?;
            }
            for channel in list.iter() {
                let unit = self.get_channel_unit(channel);
                let desc = self.get_channel_desc(channel);
                writeln!(f, " {channel} {unit:?} {desc:?} \n")?;
            }
        }
        writeln!(f, "\n")
    }
}

/// MDF3 - common Header
#[derive(Debug, BinRead, Default)]
#[br(little)]
#[allow(dead_code)]
pub struct Blockheader3 {
    hdr_id: [u8; 2], // 'XX' Block type identifier
    hdr_len: u16,    // block size
}

/// Generic block header parser
pub fn parse_block_header(rdr: &mut SymBufReader<&File>) -> Result<Blockheader3> {
    let header: Blockheader3 = rdr
        .read_le()
        .context("Could not read Blockheader3 struct")?;
    Ok(header)
}

/// HD3 strucutre
#[derive(Debug, PartialEq, Eq, Default)]
#[allow(dead_code)]
#[repr(C)]
pub struct Hd3 {
    /// HD
    hd_id: [u8; 2],
    /// Length of block in bytes
    hd_len: u16,
    /// Pointer to the first data group block (DGBLOCK) (can be NIL)
    pub hd_dg_first: u32,
    /// Pointer to the measurement file comment (TXBLOCK) (can be NIL)
    hd_md_comment: u32,
    /// Program block
    hd_pr: u32,

    // Data members
    /// Time stamp in nanoseconds elapsed since 00:00:00 01.01.1970 (UTC time or local time, depending on "local time" flag).
    hd_n_datagroups: u16,
    /// Date at which the recording was started in "DD:MM:YYYY" format
    pub hd_date: (u32, u32, i32),
    /// Time at which the recording was started in "HH:MM:SS" format
    pub hd_time: (u32, u32, u32),
    /// Author's name
    pub hd_author: String,
    /// name of the organization or department
    pub hd_organization: String,
    /// project name
    pub hd_project: String,
    /// subject or measurement object
    pub hd_subject: String,
    /// time stamp at which recording was started in nanosecond
    pub hd_start_time_ns: Option<u64>,
    /// time stamp at which recording was started in nanosecond
    hd_time_offset: Option<i16>,
    /// time quality class
    hd_time_quality: Option<u16>,
    /// timer identification or time source
    hd_time_identifier: Option<String>,
}

/// HD3 block structure
#[derive(Debug, PartialEq, Eq, Default, BinRead)]
#[repr(C)]
pub struct Hd3Block {
    hd_id: [u8; 2],            // HD
    hd_len: u16,               // Length of block in bytes
    pub hd_dg_first: u32,      // Pointer to the first data group block (DGBLOCK) (can be NIL)
    hd_md_comment: u32,        // Pointer to the measurement file comment (TXBLOCK) (can be NIL)
    hd_pr: u32,                // Program block
    hd_n_datagroups: u16,      // number of datagroup
    hd_date: [u8; 10],         // Date at which the recording was started in "DD:MM:YYYY" format
    hd_time: [u8; 8],          // Time at which the recording was started in "HH:MM:SS" format
    hd_author: [u8; 32],       // Author's name
    hd_organization: [u8; 32], // name of the organization or department
    hd_project: [u8; 32],      // project name
    hd_subject: [u8; 32],      // subject or measurement object
}

/// Specific from version 3.2 HD block extension
#[derive(Debug, PartialEq, Eq, Default, BinRead)]
#[repr(C)]
pub struct Hd3Block32 {
    hd_start_time_ns: u64, // time stamp at which recording was started in nanosecond
    hd_time_offset: i16,   // UTC time offset
    hd_time_quality: u16,  // time quality class
    hd_time_identifier: [u8; 32], // timer identification or time source
}

/// Header block parser
pub fn hd3_parser(
    rdr: &mut SymBufReader<&File>,
    ver: u16,
    encoding: &'static Encoding,
) -> Result<(Hd3, i64)> {
    let mut buf = [0u8; 164];
    rdr.read_exact(&mut buf)
        .context("Could not read hd3 buffer")?;
    let mut block = Cursor::new(buf);
    let block: Hd3Block = block
        .read_le()
        .context("Could not read buffer into Hd3Block struct")?;
    let datestr: String = encoding.decode(&block.hd_date).0.into();
    let mut dateiter = datestr.split(':');
    let day: u32 = dateiter
        .next()
        .context("date iteration ended")?
        .parse::<u32>()
        .context("Could not parse day")?;
    let month: u32 = dateiter
        .next()
        .context("date iteration ended")?
        .parse::<u32>()
        .context("Could not parse month")?;
    let year: i32 = dateiter
        .next()
        .context("date iteration ended")?
        .parse::<i32>()
        .context("Could not parse year")?;
    let hd_date = (day, month, year);
    let timestr: String = encoding.decode(&block.hd_time).0.into();
    let mut timeiter = timestr.split(':');
    let hour: u32 = timeiter
        .next()
        .context("time iteration ended")?
        .parse::<u32>()
        .context("Could not parse hour")?;
    let minute: u32 = timeiter
        .next()
        .context("time iteration ended")?
        .parse::<u32>()
        .context("Could not parse minute")?;
    let sec: u32 = timeiter
        .next()
        .context("time iteration ended")?
        .parse::<u32>()
        .context("Could not parse sec")?;
    let hd_time = (hour, minute, sec);
    let mut hd_author: String = encoding.decode(&block.hd_author).0.into();
    hd_author = hd_author.trim_end_matches(char::from(0)).to_string();
    let mut hd_organization: String = encoding.decode(&block.hd_organization).0.into();
    hd_organization = hd_organization.trim_end_matches(char::from(0)).to_string();
    let mut hd_project: String = encoding.decode(&block.hd_project).0.into();
    hd_project = hd_project.trim_end_matches(char::from(0)).to_string();
    let mut hd_subject: String = encoding.decode(&block.hd_subject).0.into();
    hd_subject = hd_subject.trim_end_matches(char::from(0)).to_string();
    let hd_start_time_ns: Option<u64>;
    let hd_time_offset: Option<i16>;
    let hd_time_quality: Option<u16>;
    let hd_time_identifier: Option<String>;
    let position: i64;
    if ver >= 320 {
        let mut buf = [0u8; 44];
        rdr.read_exact(&mut buf)
            .context("Could not read buffer for Hd3Block32")?;
        let mut block = Cursor::new(buf);
        let block: Hd3Block32 = block
            .read_le()
            .context("Could not read buffer into Hd3Block32 struct")?;
        let ti = encoding.decode(&block.hd_time_identifier).0;
        hd_start_time_ns = Some(block.hd_start_time_ns);
        hd_time_offset = Some(block.hd_time_offset);
        hd_time_quality = Some(block.hd_time_quality);
        hd_time_identifier = Some(ti.to_string());
        position = 208 + 64; // position after reading ID and HD
    } else {
        // calculate hd_start_time_ns
        hd_start_time_ns = NaiveDate::from_ymd_opt(hd_date.2, hd_date.1, hd_date.0)
            .unwrap_or_default()
            .and_hms_opt(hd_time.0, hd_time.1, hd_time.2)
            .unwrap_or_default()
            .and_utc()
            .timestamp_nanos_opt()
            .map(|t| t as u64);
        hd_time_offset = None;
        hd_time_quality = None;
        hd_time_identifier = None;
        position = 164 + 64; // position after reading ID and HD
    }
    Ok((
        Hd3 {
            hd_id: block.hd_id,
            hd_len: block.hd_len,
            hd_dg_first: block.hd_dg_first,
            hd_md_comment: block.hd_md_comment,
            hd_pr: block.hd_pr,
            hd_n_datagroups: block.hd_n_datagroups,
            hd_date,
            hd_time,
            hd_author,
            hd_organization,
            hd_project,
            hd_subject,
            hd_start_time_ns,
            hd_time_offset,
            hd_time_quality,
            hd_time_identifier,
        },
        position,
    ))
}

/// Hd3 display implementation
impl fmt::Display for Hd3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Date : {}:{}:{}, Time: {}:{}:{} ",
            self.hd_date.0,
            self.hd_date.1,
            self.hd_date.2,
            self.hd_time.0,
            self.hd_time.1,
            self.hd_time.2,
        )?;
        writeln!(f, "Author: {}", self.hd_author)?;
        writeln!(f, "Organization: {}", self.hd_organization)?;
        writeln!(f, "Project: {}", self.hd_project)?;
        writeln!(f, "Subject: {}", self.hd_subject)
    }
}

/// Header comment parser
pub fn hd3_comment_parser(
    rdr: &mut SymBufReader<&File>,
    hd3_block: &Hd3,
    mut position: i64,
    encoding: &'static Encoding,
) -> Result<(String, i64)> {
    let (_, comment, pos) = parse_tx(rdr, hd3_block.hd_md_comment, position, encoding)?;
    position = pos;
    Ok((comment, position))
}

/// TX text block parser, contexting ISO_8859_1 encoded text
pub fn parse_tx(
    rdr: &mut SymBufReader<&File>,
    target: u32,
    position: i64,
    encoding: &'static Encoding,
) -> Result<(Blockheader3, String, i64)> {
    rdr.seek_relative(target as i64 - position)
        .context("Could not reach position of TX block")?;
    let block_header: Blockheader3 = parse_block_header(rdr)?; // reads header

    // reads comment
    let mut comment_raw = vec![0; (block_header.hdr_len - 4) as usize];
    rdr.read_exact(&mut comment_raw)
        .context("Could not read comment raw data")?;
    let (comment, _encoding, error_flag) = encoding.decode(&comment_raw);
    let comment: String = comment.to_string().trim_end_matches(char::from(0)).into();
    if error_flag {
        info!("errors reading {comment}");
    }
    let position = target as i64 + block_header.hdr_len as i64;
    Ok((block_header, comment, position))
}

/// Data Group Block structure
#[derive(Debug, BinRead, Clone)]
#[br(little)]
#[allow(dead_code)]
#[repr(C)]
pub struct Dg3Block {
    /// DG
    dg_id: [u8; 2],
    /// Length of block in bytes
    dg_len: u16,
    /// Pointer to next data group block (DGBLOCK) (can be NIL)  
    dg_dg_next: u32,
    /// Pointer to first channel group block (CGBLOCK) (can be NIL)
    dg_cg_first: u32,
    /// Pointer to trigger block
    dg_tr: u32,
    /// Pointer to data block (DTBLOCK or DZBLOCK for this block type) or data list block (DLBLOCK of data blocks or its HLBLOCK)  (can be NIL)
    pub dg_data: u32,
    /// number of channel groups
    dg_n_cg: u16,
    /// number of record ids
    dg_n_record_ids: u16,
    // reserved: u32, // reserved
}

/// Data Group block parser
pub fn parse_dg3_block(
    rdr: &mut SymBufReader<&File>,
    target: u32,
    position: i64,
) -> Result<(Dg3Block, i64)> {
    rdr.seek_relative(target as i64 - position)
        .context("Could not reach position of Dg3 block")?;
    let mut buf = [0u8; 24];
    rdr.read_exact(&mut buf)
        .context("Could not read Dg3 Block buffer")?;
    let mut block = Cursor::new(buf);
    let block: Dg3Block = block
        .read_le()
        .context("Could not read buffer into Dg3Block structure")?;
    Ok((block, (target + 24).into()))
}

/// Dg3 struct wrapping block, comments and linked CG
#[derive(Debug)]
#[repr(C)]
pub struct Dg3 {
    /// DG Block
    pub block: Dg3Block,
    /// position of block in file
    pub block_position: u32,
    /// CG Block
    pub cg: HashMap<u16, Cg3>,
}

/// Parser for Dg3 and all linked blocks (cg, cn, cc)
pub fn parse_dg3(
    rdr: &mut SymBufReader<&File>,
    target: u32,
    mut position: i64,
    sharable: &mut SharableBlocks3,
    default_byte_order: u16,
    encoding: &'static Encoding,
) -> Result<(BTreeMap<u32, Dg3>, i64, u16, u16)> {
    let mut dg: BTreeMap<u32, Dg3> = BTreeMap::new();
    let mut n_cn: u16 = 0;
    let mut n_cg: u16 = 0;
    if target > 0 {
        let (block, pos) = parse_dg3_block(rdr, target, position)?;
        position = pos;
        let mut next_pointer = block.dg_dg_next;
        let (cg, pos, num_cn) = parse_cg3(
            rdr,
            block.dg_cg_first,
            position,
            sharable,
            block.dg_n_record_ids,
            default_byte_order,
            encoding,
        )?;
        n_cg += block.dg_n_cg;
        n_cn += num_cn;
        let dg_struct = Dg3 {
            block,
            block_position: target,
            cg,
        };
        dg.insert(dg_struct.block.dg_data, dg_struct);
        position = pos;
        while next_pointer > 0 {
            let block_start = next_pointer;
            let (block, pos) = parse_dg3_block(rdr, next_pointer, position)?;
            next_pointer = block.dg_dg_next;
            position = pos;
            let (cg, pos, num_cn) = parse_cg3(
                rdr,
                block.dg_cg_first,
                position,
                sharable,
                block.dg_n_record_ids,
                default_byte_order,
                encoding,
            )?;
            n_cg += block.dg_n_cg;
            n_cn += num_cn;
            let dg_struct = Dg3 {
                block,
                block_position: block_start,
                cg,
            };
            dg.insert(dg_struct.block.dg_data, dg_struct);
            position = pos;
        }
    }
    Ok((dg, position, n_cg, n_cn))
}

/// Cg3 Channel Group block struct
#[derive(Debug, Copy, Clone, Default, BinRead)]
#[br(little)]
#[allow(dead_code)]
#[repr(C)]
pub struct Cg3Block {
    /// CG
    cg_id: [u8; 2],
    // Length of block in bytes
    cg_len: u16,
    /// Pointer to next channel group block (CGBLOCK) (can be NIL)
    pub cg_cg_next: u32,
    /// Pointer to first channel block (CNBLOCK) (NIL allowed)
    cg_cn_first: u32,
    /// CG comment (TXBLOCK) (can be NIL)
    cg_comment: u32,
    /// Record ID, value of the identifier for a record if the DGBLOCK defines a number of record IDs > 0  
    pub cg_record_id: u16,
    /// number of channels
    cg_n_channels: u16,
    /// Size of data record in Bytes (without record ID)
    pub cg_data_bytes: u16,
    /// Number of records of this type in the data block
    pub cg_cycle_count: u32,
    /// Pointer to first sample reduction block (SRBLOCK) (NIL allowed)
    cg_sr: u32,
}

/// Cg3 (Channel Group) block struct parser with linked comments Source Information in sharable blocks
fn parse_cg3_block(
    rdr: &mut SymBufReader<&File>,
    target: u32,
    mut position: i64,
    sharable: &mut SharableBlocks3,
    record_id_size: u16,
    default_byte_order: u16,
    encoding: &'static Encoding,
) -> Result<(Cg3, i64, u16)> {
    rdr.seek_relative(target as i64 - position)
        .context("Could not reach position of Cg3Block")?; // change buffer position
    let mut buf = vec![0u8; 30];
    rdr.read_exact(&mut buf)
        .context("Could not read Cg3Block buffer")?;
    let mut block = Cursor::new(buf);
    let cg: Cg3Block = block
        .read_le()
        .context("Could not read buffer into Cg3Block structure")?;
    position = target as i64 + 30;

    // reads CN (and other linked block behind like CC, SI, CA, etc.)
    let (cn, pos, n_cn) = parse_cn3(
        rdr,
        cg.cg_cn_first,
        position,
        sharable,
        record_id_size,
        default_byte_order,
        encoding,
    )?;
    position = pos;

    let record_length = cg.cg_data_bytes;
    let block_position = target;

    let cg_struct = Cg3 {
        block: cg,
        cn,
        block_position,
        master_channel_name: None,
        channel_names: HashSet::new(),
        record_length,
    };

    Ok((cg_struct, position, n_cn))
}

/// Channel Group struct
/// it contains the related channels structure, a set of channel names, the dedicated master channel name and other helper data.
#[derive(Debug)]
#[repr(C)]
pub struct Cg3 {
    pub block: Cg3Block,
    pub cn: HashMap<u32, Cn3>, // hashmap of channels
    pub block_position: u32,
    pub master_channel_name: Option<String>,
    pub channel_names: HashSet<String>,
    pub record_length: u16, // record length including recordId
}

/// Cg3 blocks and linked blocks parsing
pub fn parse_cg3(
    rdr: &mut SymBufReader<&File>,
    target: u32,
    mut position: i64,
    sharable: &mut SharableBlocks3,
    record_id_size: u16,
    default_byte_order: u16,
    encoding: &'static Encoding,
) -> Result<(HashMap<u16, Cg3>, i64, u16)> {
    let mut cg: HashMap<u16, Cg3> = HashMap::new();
    let mut n_cn: u16 = 0;
    if target != 0 {
        let (mut cg_struct, pos, num_cn) = parse_cg3_block(
            rdr,
            target,
            position,
            sharable,
            record_id_size,
            default_byte_order,
            encoding,
        )?;
        position = pos;
        let mut next_pointer = cg_struct.block.cg_cg_next;
        cg_struct.record_length += record_id_size;
        cg.insert(cg_struct.block.cg_record_id, cg_struct);
        n_cn += num_cn;

        while next_pointer != 0 {
            let (mut cg_struct, pos, num_cn) = parse_cg3_block(
                rdr,
                next_pointer,
                position,
                sharable,
                record_id_size,
                default_byte_order,
                encoding,
            )?;
            position = pos;
            cg_struct.record_length += record_id_size;
            next_pointer = cg_struct.block.cg_cg_next;
            cg.insert(cg_struct.block.cg_record_id, cg_struct);
            n_cn += num_cn;
        }
    }
    Ok((cg, position, n_cn))
}

/// Cn3 structure containing block but also unique_name, ndarray data
/// and other attributes frequently needed and computed
#[derive(Debug, Default)]
#[repr(C)]
pub struct Cn3 {
    pub block1: Cn3Block1,
    pub block2: Cn3Block2,
    /// unique channel name string
    pub unique_name: String,
    /// channel comment
    pub comment: String,
    // channel description
    pub description: String,
    /// beginning position of channel in record
    pub pos_byte_beg: u16,
    /// number of bytes taken by channel in record
    pub n_bytes: u16,
    /// channel data
    pub data: ChannelData,
    /// false = little endian
    pub endian: bool,
    /// True if channel is valid = contains data converted
    pub channel_data_valid: bool,
}

/// creates recursively in the channel group the CN blocks and all its other linked blocks (CC, TX, CE, CD)
pub fn parse_cn3(
    rdr: &mut SymBufReader<&File>,
    mut target: u32,
    mut position: i64,
    sharable: &mut SharableBlocks3,
    record_id_size: u16,
    default_byte_order: u16,
    encoding: &'static Encoding,
) -> Result<(HashMap<u32, Cn3>, i64, u16)> {
    let mut cn: HashMap<u32, Cn3> = HashMap::new();
    let mut n_cn: u16 = 0;
    if target != 0 {
        let (cn_struct, pos) = parse_cn3_block(
            rdr,
            target,
            position,
            sharable,
            &mut cn,
            record_id_size,
            default_byte_order,
            encoding,
        )?;
        position = pos;
        n_cn += 1;
        let mut next_pointer = cn_struct.block1.cn_cn_next;
        cn.insert(target, cn_struct);

        while next_pointer != 0 {
            let (cn_struct, pos) = parse_cn3_block(
                rdr,
                next_pointer,
                position,
                sharable,
                &mut cn,
                record_id_size,
                default_byte_order,
                encoding,
            )?;
            position = pos;
            n_cn += 1;
            target = next_pointer;
            next_pointer = cn_struct.block1.cn_cn_next;
            cn.insert(target, cn_struct);
        }
    }
    Ok((cn, position, n_cn))
}

/// Cn3 Channel block struct, first sub block
#[derive(Debug, PartialEq, Eq, Default, Clone, BinRead)]
#[br(little)]
#[repr(C)]
pub struct Cn3Block1 {
    /// CN
    cn_id: [u8; 2],
    /// Length of block in bytes
    cn_len: u16,
    /// Pointer to next channel block (CNBLOCK) (can be NIL)
    cn_cn_next: u32,
    /// Pointer to the conversion formula (CCBLOCK) (can be NIL)
    pub cn_cc_conversion: u32,
    /// Pointer to the source-depending extensions (CEBLOCK) of this signal (can be NIL)
    cn_ce_source: u32,
    /// Pointer to the dependency block (CDBLOCK) of this signal (NIL allowed)
    cn_cd_source: u32,
    /// Pointer to the channel comment (TXBLOCK) of this signal (NIL allowed)
    cn_tx_comment: u32,
    /// Channel type, 0 normal data, 1 time channel
    pub cn_type: u16,
    /// Short signal name
    cn_short_name: [u8; 32],
}

/// Cn3 Channel block struct, second sub block
#[derive(Debug, PartialEq, Default, Clone, BinRead)]
#[br(little)]
#[repr(C)]
pub struct Cn3Block2 {
    /// Start offset in bits to determine the first bit of the signal in the data record.
    pub cn_bit_offset: u16,
    /// Number of bits used to encode the value of this signal in a data record
    pub cn_bit_count: u16,
    /// Channel data type of raw signal value
    pub cn_data_type: u16,
    cn_valid_range_flags: u16, // Value range valid flag:
    cn_val_range_min: f64, // Minimum signal value that occurred for this signal (raw value) Only valid if "value range valid" flag (bit 3) is set.
    cn_val_range_max: f64, // Maximum signal value that occurred for this signal (raw value) Only valid if "value range valid" flag (bit 3) is set.
    cn_sampling_rate: f64, // Sampling rate of the signal in sec
    cn_tx_long_name: u32,  // Long signal name
    cn_tx_display_name: u32, // Display signal name
    cn_byte_offset: u16,   //
}

/// CN3 Block parsing
#[allow(clippy::too_many_arguments)]
fn parse_cn3_block(
    rdr: &mut SymBufReader<&File>,
    target: u32,
    mut position: i64,
    sharable: &mut SharableBlocks3,
    cn: &mut HashMap<u32, Cn3>,
    record_id_size: u16,
    default_byte_order: u16,
    encoding: &'static Encoding,
) -> Result<(Cn3, i64)> {
    rdr.seek_relative(target as i64 - position)
        .context("Could not reach position of CN Block")?; // change buffer position
    let mut buf = vec![0u8; 228];
    rdr.read_exact(&mut buf)
        .context("Could not read Cn3 block buffer")?;
    position = target as i64 + 228;
    let mut block = Cursor::new(buf);
    let block1: Cn3Block1 = block
        .read_le()
        .context("Could not read buffer into Cn3Block1 structure")?;
    let mut desc = vec![0u8; 128];
    block
        .read_exact(&mut desc)
        .context("Could not read channel description")?;
    let block2: Cn3Block2 = block
        .read_le()
        .context("Could not read buffer into Cn3Block2 struct")?;
    let pos_byte_beg = block2.cn_bit_offset / 8 + record_id_size;
    let mut n_bytes = block2.cn_bit_count / 8u16;
    if (block2.cn_bit_count % 8) != 0 {
        n_bytes += 1;
    }

    let (name, _encoding, error_flag) = encoding.decode(&block1.cn_short_name);
    let mut unique_name = name.to_string().trim_end_matches(char::from(0)).to_string();
    if block2.cn_tx_long_name != 0 {
        // Reads TX long name
        let (_, name, pos) = parse_tx(rdr, block2.cn_tx_long_name, position, encoding)?;
        unique_name = name;
        position = pos;
    }
    if error_flag {
        info!("errors reading channel name {unique_name}");
    }

    let (desc, _encoding, error_flag) = encoding.decode(&desc);
    let description = desc.to_string().trim_end_matches(char::from(0)).to_string();
    if error_flag {
        info!("errors reading channel description {description}");
    }

    let mut comment = String::new();
    if block1.cn_tx_comment != 0 {
        // Reads TX comment
        let (_, cm, pos) = parse_tx(rdr, block1.cn_tx_comment, position, encoding)?;
        comment = cm;
        position = pos;
    }

    // Reads CC block
    if !sharable.cc.contains_key(&block1.cn_cc_conversion) {
        let (pos, cc_block) =
            parse_cc3_block(rdr, block1.cn_cc_conversion, position, sharable, encoding)?;
        position = pos;
        if cc_block.cc_type == 132 {
            // CANopen date
            let (date_ms, min, hour, day, month, year) =
                can_open_date(pos_byte_beg, block2.cn_bit_offset);
            cn.insert(block1.cn_cc_conversion + 1, date_ms);
            cn.insert(block1.cn_cc_conversion + 2, min);
            cn.insert(block1.cn_cc_conversion + 3, hour);
            cn.insert(block1.cn_cc_conversion + 4, day);
            cn.insert(block1.cn_cc_conversion + 5, month);
            cn.insert(block1.cn_cc_conversion + 6, year);
        } else if cc_block.cc_type == 133 {
            // CANopen time
            let (ms, days) = can_open_time(pos_byte_beg, block2.cn_bit_offset);
            cn.insert(block1.cn_cc_conversion + 1, ms);
            cn.insert(block1.cn_cc_conversion + 2 + 32, days);
        }
    }

    // Reads CE block
    if !sharable.ce.contains_key(&block1.cn_ce_source) {
        position = parse_ce(rdr, block1.cn_ce_source, position, sharable, encoding)?;
    }

    let mut endian: bool = false; // Little endian by default
    if block2.cn_data_type >= 13 {
        endian = false; // little endian
    } else if block2.cn_data_type >= 9 {
        endian = true; // big endian
    } else if block2.cn_data_type <= 3 {
        if default_byte_order == 0 {
            endian = false; // little endian
        } else {
            endian = true; // big endian
        }
    }
    let data_type = convert_data_type_3to4(block2.cn_data_type);

    let cn_struct = Cn3 {
        block1,
        block2,
        description,
        comment,
        unique_name,
        pos_byte_beg,
        n_bytes,
        data: data_type_init(0, data_type, n_bytes as u32, 1)?,
        endian,
        channel_data_valid: false,
    };

    Ok((cn_struct, position))
}

/// Converter of data type from 3.x to 4.x
pub fn convert_data_type_3to4(mdf3_datatype: u16) -> u8 {
    match mdf3_datatype {
        0 => 0,
        1 => 2,
        2 => 4,
        3 => 4,
        7 => 6,
        8 => 10,
        9 => 1,
        10 => 3,
        11 => 5,
        12 => 5,
        13 => 0,
        14 => 2,
        15 => 4,
        16 => 4,
        _ => 13,
    }
}

/// returns created CANopenDate channels
fn can_open_date(pos_byte_beg: u16, cn_bit_offset: u16) -> (Cn3, Cn3, Cn3, Cn3, Cn3, Cn3) {
    let block1 = Cn3Block1 {
        cn_type: 0,
        cn_cc_conversion: 0,
        ..Default::default()
    };
    let block2 = Cn3Block2 {
        cn_data_type: 13,
        cn_byte_offset: pos_byte_beg,
        cn_bit_count: 16,
        cn_bit_offset,
        ..Default::default()
    };
    let date_ms = Cn3 {
        block1: block1.clone(),
        block2,
        unique_name: String::from("ms"),
        comment: String::new(),
        description: String::from("Milliseconds"),
        pos_byte_beg,
        n_bytes: 2,
        data: ChannelData::UInt16(UInt16Builder::new()),
        endian: false,
        channel_data_valid: false,
    };
    let block2 = Cn3Block2 {
        cn_data_type: 13,
        cn_byte_offset: pos_byte_beg + 2,
        cn_bit_count: 6,
        cn_bit_offset: cn_bit_offset + 16,
        ..Default::default()
    };
    let min = Cn3 {
        block1: block1.clone(),
        block2,
        unique_name: String::from("min"),
        comment: String::new(),
        description: String::from("Minutes"),
        pos_byte_beg: pos_byte_beg + 2,
        n_bytes: 1,
        data: ChannelData::UInt8(UInt8Builder::new()),
        endian: false,
        channel_data_valid: false,
    };
    let block2 = Cn3Block2 {
        cn_data_type: 13,
        cn_byte_offset: pos_byte_beg + 3,
        cn_bit_count: 5,
        cn_bit_offset: cn_bit_offset + 24,
        ..Default::default()
    };
    let hour = Cn3 {
        block1: block1.clone(),
        block2,
        unique_name: String::from("hour"),
        comment: String::new(),
        description: String::from("Hours"),
        pos_byte_beg: pos_byte_beg + 3,
        n_bytes: 1,
        data: ChannelData::UInt8(UInt8Builder::new()),
        endian: false,
        channel_data_valid: false,
    };
    let block2 = Cn3Block2 {
        cn_data_type: 13,
        cn_byte_offset: pos_byte_beg + 4,
        cn_bit_count: 5,
        cn_bit_offset: cn_bit_offset + 32,
        ..Default::default()
    };
    let day = Cn3 {
        block1: block1.clone(),
        block2,
        unique_name: String::from("day"),
        comment: String::new(),
        description: String::from("Days"),
        pos_byte_beg: pos_byte_beg + 4,
        n_bytes: 1,
        data: ChannelData::UInt8(UInt8Builder::new()),
        endian: false,
        channel_data_valid: false,
    };
    let block2 = Cn3Block2 {
        cn_data_type: 13,
        cn_byte_offset: pos_byte_beg + 5,
        cn_bit_count: 6,
        cn_bit_offset: cn_bit_offset + 40,
        ..Default::default()
    };
    let month = Cn3 {
        block1: block1.clone(),
        block2,
        unique_name: String::from("month"),
        comment: String::new(),
        description: String::from("Month"),
        pos_byte_beg: pos_byte_beg + 5,
        n_bytes: 1,
        data: ChannelData::UInt8(UInt8Builder::new()),
        endian: false,
        channel_data_valid: false,
    };
    let block2 = Cn3Block2 {
        cn_data_type: 13,
        cn_byte_offset: pos_byte_beg + 6,
        cn_bit_count: 7,
        cn_bit_offset: cn_bit_offset + 48,
        ..Default::default()
    };
    let year = Cn3 {
        block1,
        block2,
        unique_name: String::from("year"),
        comment: String::new(),
        description: String::from("Years"),
        pos_byte_beg: pos_byte_beg + 7,
        n_bytes: 1,
        data: ChannelData::UInt8(UInt8Builder::new()),
        endian: false,
        channel_data_valid: false,
    };
    (date_ms, min, hour, day, month, year)
}

/// returns created CANopenTime channels
fn can_open_time(pos_byte_beg: u16, cn_bit_offset: u16) -> (Cn3, Cn3) {
    let block1 = Cn3Block1 {
        cn_type: 0,
        cn_cc_conversion: 0,
        ..Default::default()
    };
    let block2 = Cn3Block2 {
        cn_data_type: 13,
        cn_byte_offset: pos_byte_beg,
        cn_bit_count: 32,
        cn_bit_offset,
        ..Default::default()
    };
    let ms: Cn3 = Cn3 {
        block1: block1.clone(),
        block2,
        unique_name: String::from("ms"),
        comment: String::new(),
        description: String::from("Milliseconds"),
        pos_byte_beg,
        n_bytes: 4,
        data: ChannelData::UInt32(UInt32Builder::new()),
        endian: false,
        channel_data_valid: false,
    };
    let block2 = Cn3Block2 {
        cn_data_type: 13,
        cn_byte_offset: pos_byte_beg + 4,
        cn_bit_count: 16,
        cn_bit_offset: cn_bit_offset + 32,
        ..Default::default()
    };
    let days: Cn3 = Cn3 {
        block1,
        block2,
        unique_name: String::from("day"),
        comment: String::new(),
        description: String::from("Days"),
        pos_byte_beg: pos_byte_beg + 4,
        n_bytes: 2,
        data: ChannelData::UInt16(UInt16Builder::new()),
        endian: false,
        channel_data_valid: false,
    };
    (ms, days)
}

/// sharable blocks (most likely referenced multiple times and shared by several blocks)
/// that are in sharable fields and holds CC, CE, TX blocks
#[derive(Debug, Default, Clone)]
#[repr(C)]
pub struct SharableBlocks3 {
    pub(crate) cc: HashMap<u32, (Cc3Block, Conversion)>,
    pub(crate) ce: HashMap<u32, CeBlock>,
}
/// Cc3 Channel conversion block struct, second sub block
#[derive(Debug, Clone, BinRead)]
#[br(little)]
#[allow(dead_code)]
#[repr(C)]
pub struct Cc3Block {
    /// CC
    cc_id: [u8; 2],
    /// Length of block in bytes
    cc_len: u16,
    /// Physical value range valid flag
    cc_valid_range_flags: u16,
    /// Minimum physical signal value that occurred for this signal. Only valid if "physical value range valid" flag is set.
    cc_val_range_min: f64,
    /// Maximum physical signal value that occurred for this signal. Only valid if "physical value range valid" flag is set.
    cc_val_range_max: f64,
    /// physical unit of the signal
    pub cc_unit: [u8; 20],
    /// Conversion type
    cc_type: u16,
    /// Size information, meaning depends of conversion type
    cc_size: u16,
}

impl Default for Cc3Block {
    fn default() -> Self {
        Self {
            cc_id: [0; 2],
            cc_len: 48,
            cc_valid_range_flags: 0,
            cc_val_range_min: 0.0,
            cc_val_range_max: 0.0,
            cc_unit: [0; 20],
            cc_type: 65535, // 1:1 conversion
            cc_size: 0,
        }
    }
}

/// All kind of channel data conversions
#[derive(Debug, Clone)]
#[repr(C)]
pub enum Conversion {
    Linear(Vec<f64>),
    TabularInterpolation(Vec<f64>),
    Tabular(Vec<f64>),
    Polynomial(Vec<f64>),
    Exponential(Vec<f64>),
    Logarithmic(Vec<f64>),
    Rational(Vec<f64>),
    Formula(String),
    TextTable(Vec<(f64, String)>),
    TextRangeTable((Vec<(f64, f64, String)>, String)),
    Identity,
}

impl Default for Conversion {
    fn default() -> Self {
        Self::Identity
    }
}

/// Parser for channel conversion blocks
pub fn parse_cc3_block(
    rdr: &mut SymBufReader<&File>,
    target: u32,
    mut position: i64,
    sharable: &mut SharableBlocks3,
    encoding: &'static Encoding,
) -> Result<(i64, Cc3Block)> {
    rdr.seek_relative(target as i64 - position)
        .context("Could not reach CC Block position")?; // change buffer position
    let mut buf = vec![0u8; 46];
    rdr.read_exact(&mut buf)
        .context("Could not read Cc3Block buffer")?;
    position = target as i64 + 46;
    let mut block = Cursor::new(buf);
    let cc_block: Cc3Block = block
        .read_le()
        .context("Could not read buffer into Cc3Block structure")?;
    let conversion: Conversion;
    match cc_block.cc_type {
        0 => {
            let mut buf = vec![0.0f64; 2];
            rdr.read_f64_into::<LittleEndian>(&mut buf)
                .context("Could not read linear conversion parameters")?;
            conversion = Conversion::Linear(buf);
            position += 16;
        }
        1 => {
            let mut buf = vec![0.0f64; cc_block.cc_size as usize * 2];
            rdr.read_f64_into::<LittleEndian>(&mut buf)
                .context("Could not read tabular interpolation conversion parameters")?;
            conversion = Conversion::TabularInterpolation(buf);
            position += cc_block.cc_size as i64 * 2 * 8;
        }
        2 => {
            let mut buf = vec![0.0f64; cc_block.cc_size as usize * 2];
            rdr.read_f64_into::<LittleEndian>(&mut buf)
                .context("Could not read tabular conversion parameters")?;
            conversion = Conversion::Tabular(buf);
            position += cc_block.cc_size as i64 * 2 * 8;
        }
        6 => {
            let mut buf = vec![0.0f64; 6];
            rdr.read_f64_into::<LittleEndian>(&mut buf)
                .context("Could not read polynomial conversion parameters")?;
            conversion = Conversion::Polynomial(buf);
            position += 48;
        }
        7 => {
            let mut buf = vec![0.0f64; 7];
            rdr.read_f64_into::<LittleEndian>(&mut buf)
                .context("Could not read exponential conversion parameters")?;
            conversion = Conversion::Exponential(buf);
            position += 56;
        }
        8 => {
            let mut buf = vec![0.0f64; 7];
            rdr.read_f64_into::<LittleEndian>(&mut buf)
                .context("Could not read logarithmic conversion parameters")?;
            conversion = Conversion::Logarithmic(buf);
            position += 56;
        }
        9 => {
            let mut buf = vec![0.0f64; 6];
            rdr.read_f64_into::<LittleEndian>(&mut buf)
                .context("Could not read rational conversion parameters")?;
            conversion = Conversion::Rational(buf);
            position += 48;
        }
        10 => {
            let mut buf = vec![0u8; 256];
            rdr.read_exact(&mut buf)
                .context("Could not read formulae conversion parameters")?;
            position += 256;
            let mut formula: String = encoding.decode(&buf).0.into();
            let index = formula.find(char::from(0));
            if let Some(ind) = index {
                formula = formula[0..ind].to_string();
            }
            conversion = Conversion::Formula(formula);
        }
        11 => {
            let mut pairs: Vec<(f64, String)> =
                vec![(0.0f64, String::with_capacity(32)); cc_block.cc_size as usize];
            let mut buf = vec![0u8; 32];
            for pair in pairs.iter_mut() {
                pair.0 = rdr
                    .read_f64::<LittleEndian>()
                    .context("Could not read text table conversion value parameters")?;
                rdr.read_exact(&mut buf)
                    .context("Could not read text-table conversion text parameters")?;
                position += 40;
                let text: String = encoding.decode(&buf).0.into();
                pair.1 = text.trim_end_matches(char::from(0)).to_string();
            }
            conversion = Conversion::TextTable(pairs);
        }
        12 => {
            let mut pairs_pointer: Vec<(f64, f64, u32)> =
                Vec::with_capacity(cc_block.cc_size as usize - 1);
            let mut pairs_string: Vec<(f64, f64, String)> =
                Vec::with_capacity(cc_block.cc_size as usize - 1);
            let mut low_range;
            let mut high_range;
            let mut text_pointer;
            let mut buf_ignored = [0u8; 16];
            rdr.read_exact(&mut buf_ignored)
                .context("Could not read text range table conversion default value parameters")?;
            let default_text_pointer = rdr
                .read_u32::<LittleEndian>()
                .context("Could not read text range table conversion default text parameters")?;
            position += 20;
            for _index in 0..(cc_block.cc_size as usize - 1) {
                low_range = rdr
                    .read_f64::<LittleEndian>()
                    .context("Could not read text range table conversion low value parameters")?;
                high_range = rdr
                    .read_f64::<LittleEndian>()
                    .context("Could not read text range table conversion high value parameters")?;
                text_pointer = rdr
                    .read_u32::<LittleEndian>()
                    .context("Could not read text range table conversion value parameters")?;
                position += 20;
                pairs_pointer.push((low_range, high_range, text_pointer));
            }
            let (_block_header, default_string, pos) =
                parse_tx(rdr, default_text_pointer, position, encoding)?;
            position = pos;
            for (low_range, high_range, text_pointer) in pairs_pointer.iter() {
                let (_block_header, text, pos) = parse_tx(rdr, *text_pointer, position, encoding)?;
                position = pos;
                pairs_string.push((*low_range, *high_range, text));
            }
            conversion = Conversion::TextRangeTable((pairs_string, default_string));
        }
        _ => conversion = Conversion::Identity,
    }

    sharable.cc.insert(target, (cc_block.clone(), conversion));
    Ok((position, cc_block))
}

/// CE channel extension block struct, second sub block
#[derive(Debug, Clone)]
#[allow(dead_code)]
#[repr(C)]
pub struct CeBlock {
    /// CE
    ce_id: [u8; 2],
    /// Length of block in bytes
    ce_len: u16,
    /// extension type
    ce_extension_type: u16,
    ce_extension: CeSupplement,
}

/// Either a DIM or CAN Supplemental block
#[derive(Debug, Clone)]
#[repr(C)]
pub enum CeSupplement {
    Dim(DimBlock),
    Can(CanBlock),
    None,
}

/// DIM Block supplement
#[derive(Debug, Clone)]
#[allow(dead_code)]
#[repr(C)]
pub struct DimBlock {
    /// Module number
    ce_module_number: u16,
    /// address
    ce_address: u32,
    // description
    ce_desc: String,
    /// ECU identifier
    ce_ecu_id: String,
}

/// Vector CAN Block supplement
#[derive(Debug, Clone)]
#[allow(dead_code)]
#[repr(C)]
pub struct CanBlock {
    /// CAN identifier
    ce_can_id: u32,
    /// CAN channel index
    ce_can_index: u32,
    /// message name
    ce_message_name: String,
    /// sender name
    ce_sender_name: String,
}

/// parses Channel Extension block
fn parse_ce(
    rdr: &mut SymBufReader<&File>,
    target: u32,
    mut position: i64,
    sharable: &mut SharableBlocks3,
    encoding: &'static Encoding,
) -> Result<i64> {
    rdr.seek_relative(target as i64 - position)
        .context("Could not reach CE block position")?; // change buffer position
    let mut buf = vec![0u8; 6];
    rdr.read_exact(&mut buf)
        .context("Could not read buffer for CE Block")?;
    position = target as i64 + 6;
    let mut block = Cursor::new(buf);
    let ce_id: [u8; 2] = block.read_le().context("could not read ce_id")?;
    let ce_len: u16 = block.read_le().context("could not read ce_len")?;
    let ce_extension_type: u16 = block
        .read_le()
        .context("could not read ce_extension_type")?;

    let ce_extension: CeSupplement;
    if ce_extension_type == 0x02 {
        // Reads DIM
        let mut buf = vec![0u8; 118];
        rdr.read_exact(&mut buf)
            .context("Could not DIM Supplement buffer")?;
        position += 118;
        let mut block = Cursor::new(buf);
        let ce_module_number: u16 = block.read_le().context("could not read ce_module_number")?;
        let ce_address: u32 = block.read_le().context("could not read ce_address")?;
        let mut desc = vec![0u8; 80];
        block
            .read_exact(&mut desc)
            .context("Could not read DIM description")?;
        let mut ce_desc: String = encoding.decode(&desc).0.into();
        ce_desc = ce_desc.trim_end_matches(char::from(0)).to_string();
        let mut ecu_id = vec![0u8; 32];
        block
            .read_exact(&mut ecu_id)
            .context("Could not read DIM ecu_id")?;
        let mut ce_ecu_id: String = encoding.decode(&ecu_id).0.into();
        ce_ecu_id = ce_ecu_id.trim_end_matches(char::from(0)).to_string();
        ce_extension = CeSupplement::Dim(DimBlock {
            ce_module_number,
            ce_address,
            ce_desc,
            ce_ecu_id,
        });
    } else if ce_extension_type == 19 {
        // Reads CAN
        let mut buf = vec![0u8; 80];
        rdr.read_exact(&mut buf)
            .context("Could not CAN Supplement buffer")?;
        position += 80;
        let mut block = Cursor::new(buf);
        let ce_can_id: u32 = block.read_le().context("Could not read CAN ce_can_id")?;
        let ce_can_index: u32 = block.read_le().context("Could not read CAN ce_can_index")?;
        let mut message = vec![0u8; 36];
        block
            .read_exact(&mut message)
            .context("Could not read CAN Supplement message")?;
        let mut ce_message_name: String = encoding.decode(&message).0.into();
        ce_message_name = ce_message_name.trim_end_matches(char::from(0)).to_string();
        let mut sender = vec![0u8; 32];
        block
            .read_exact(&mut sender)
            .context("Could not read CAN Supplement sender")?;
        let mut ce_sender_name: String = encoding.decode(&sender).0.into();
        ce_sender_name = ce_sender_name.trim_end_matches(char::from(0)).to_string();
        ce_extension = CeSupplement::Can(CanBlock {
            ce_can_id,
            ce_can_index,
            ce_message_name,
            ce_sender_name,
        });
    } else {
        ce_extension = CeSupplement::None;
    }
    sharable.ce.insert(
        target,
        CeBlock {
            ce_id,
            ce_len,
            ce_extension_type,
            ce_extension,
        },
    );
    Ok(position)
}

/// parses mdfinfo structure to make channel names unique
/// creates channel names set and links master channels to set of channels
pub fn build_channel_db3(
    dg: &mut BTreeMap<u32, Dg3>,
    sharable: &SharableBlocks3,
    n_cg: u16,
    n_cn: u16,
) -> ChannelNamesSet3 {
    let mut channel_list: ChannelNamesSet3 = HashMap::with_capacity(n_cn as usize);
    let mut master_channel_list: HashMap<u32, String> = HashMap::with_capacity(n_cg as usize);
    // creating channel list for whole file and making channel names unique
    for (dg_position, dg) in dg.iter_mut() {
        for (record_id, cg) in dg.cg.iter_mut() {
            for (cn_position, cn) in cg.cn.iter_mut() {
                if channel_list.contains_key(&cn.unique_name) {
                    let mut changed: bool = false;
                    let space_char = String::from(" ");
                    // create unique channel name
                    if let Some(ce) = sharable.ce.get(&cn.block1.cn_ce_source) {
                        match &ce.ce_extension {
                            CeSupplement::Dim(dim) => {
                                cn.unique_name.push_str(&space_char);
                                cn.unique_name.push_str(&dim.ce_ecu_id);
                                changed = true;
                            }
                            CeSupplement::Can(can) => {
                                cn.unique_name.push_str(&space_char);
                                cn.unique_name.push_str(&can.ce_message_name);
                                changed = true;
                            }
                            _ => {}
                        }
                    }
                    // No souce name to make channel unique
                    if !changed {
                        // extend name with channel block position, unique
                        cn.unique_name.push_str(&space_char);
                        cn.unique_name.push_str(&cn_position.to_string());
                    }
                };
                channel_list.insert(
                    cn.unique_name.clone(),
                    (
                        None, // computes at second step master channel name
                        *dg_position,
                        (cg.block_position, *record_id),
                        *cn_position,
                    ),
                );
                if cn.block1.cn_type != 0 {
                    // Master channel
                    master_channel_list.insert(cg.block_position, cn.unique_name.clone());
                }
            }
        }
    }
    // identifying master channels
    for (_dg_position, dg) in dg.iter_mut() {
        for (_record_id, cg) in dg.cg.iter_mut() {
            let mut cg_channel_list: HashSet<String> =
                HashSet::with_capacity(cg.block.cg_n_channels as usize);
            let master_channel_name: Option<String> = master_channel_list
                .get(&cg.block_position)
                .map(|name| name.to_string());
            for (_cn_record_position, cn) in cg.cn.iter_mut() {
                cg_channel_list.insert(cn.unique_name.clone());
                // assigns master in channel_list
                if let Some(id) = channel_list.get_mut(&cn.unique_name) {
                    id.0.clone_from(&master_channel_name);
                }
            }
            cg.channel_names = cg_channel_list;
            cg.master_channel_name = master_channel_name;
        }
    }
    channel_list
}
