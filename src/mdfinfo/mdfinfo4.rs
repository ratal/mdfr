//! Parsing of file metadata into MdfInfo4 struct
use binrw::{binrw, BinReaderExt, BinWriterExt};
use byteorder::{LittleEndian, ReadBytesExt};
use chrono::Local;
use chrono::{naive::NaiveDateTime, DateTime, Utc};
use ndarray::{Array1, Order};
use rayon::prelude::*;
use roxmltree;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::default::Default;
use std::fmt::Debug;
use std::fs::{File, OpenOptions};
use std::io::{prelude::*, BufWriter};
use std::{fmt, str};
use std::{
    io::{BufReader, Cursor},
};
use transpose;
use yazi::{decompress, Adler32, Format};

use crate::mdfinfo::IdBlock;
use crate::mdfreader::channel_data::{data_type_init, ChannelData};
use crate::mdfreader::mdfreader4::mdfreader4;
use crate::mdfwriter::mdfwriter4::mdfwriter4;

pub(crate) type ChannelId = (Option<String>, i64, (i64, u64), (i64, i32));
pub(crate) type ChannelNamesSet = HashMap<String, ChannelId>;

/// MdfInfo4 is the struct holding whole metadata of mdf4.x files
/// * blocks with unique links are at top level like attachment, events and file history
/// * sharable blocks (most likely referenced multiple times and shared by several blocks)
/// that are in sharable fields and holds CC, SI, TX and MD blocks
/// * the dg fields nests cg itself nesting cn blocks and eventually compositions
/// (other cn or ca blocks) and conversion
/// * channel_names_set is the complete set of channel names contained in file
/// * in general the blocks are contained in HashMaps with key corresponding
/// to their position in the file
#[derive(Debug, Clone, Default)]
pub struct MdfInfo4 {
    /// file name string
    pub file_name: String,
    /// Identifier block
    pub id_block: IdBlock,
    /// header block
    pub hd_block: Hd4,
    /// file history blocks
    pub fh: Fh,
    /// attachment blocks
    pub at: At, // attachments
    /// event blocks
    pub ev: HashMap<i64, Ev4Block>, // events
    /// data group block linking channel group/channel/conversion/compostion/..etc. and data block
    pub dg: BTreeMap<i64, Dg4>, // contains most of the file structure
    /// cc, md, tx and si blocks that can be referenced by several blocks
    pub sharable: SharableBlocks,
    /// set of all channel names
    pub channel_names_set: ChannelNamesSet, // set of channel names
    /// flag for all data loaded in memory
    pub all_data_in_memory: bool,
}

/// MdfInfo4's implementation
impl MdfInfo4 {
    /// returns the hashmap with :
    /// key = channel_name,
    /// value = (master_name,
    ///          dg_position,
    ///            (cg.block_position, record_id),
    ///            (cn.block_position, cn_record_position))
    pub fn get_channel_id(&self, channel_name: &str) -> Option<&ChannelId> {
        self.channel_names_set.get(channel_name)
    }
    /// returns channel's data ndarray.
    pub fn get_channel_data<'a>(
        &'a mut self,
        channel_name: &'a str,
    ) -> (Option<&'a ChannelData>, &Option<Array1<u8>>) {
        let mut channel_names: HashSet<String> = HashSet::new();
        channel_names.insert(channel_name.to_string());
        if !self.all_data_in_memory || !self.get_channel_data_validity(channel_name) {
            self.load_channels_data_in_memory(channel_names); // will read data only if array is empty
        }
        self.get_channel_data_from_memory(channel_name)
    }
    /// Returns the channel's data ndarray if present in memory, otherwise None.
    pub fn get_channel_data_from_memory(
        &self,
        channel_name: &str,
    ) -> (Option<&ChannelData>, &Option<Array1<u8>>) {
        let mut data: Option<&ChannelData> = None;
        let mut mask: &Option<Array1<u8>> = &None;
        if let Some((_master, dg_pos, (_cg_pos, rec_id), (_cn_pos, rec_pos))) =
            self.get_channel_id(channel_name)
        {
            if let Some(dg) = self.dg.get(dg_pos) {
                if let Some(cg) = dg.cg.get(rec_id) {
                    if let Some(cn) = cg.cn.get(rec_pos) {
                        if !cn.data.is_empty() {
                            data = Some(&cn.data);
                        }
                        mask = &cn.invalid_mask;
                    }
                }
            }
        }
        (data, mask)
    }
    /// True if channel contains data
    pub fn get_channel_data_validity(&self, channel_name: &str) -> bool {
        let mut state: bool = false;
        if let Some((_master, dg_pos, (_cg_pos, rec_id), (_cn_pos, rec_pos))) =
            self.get_channel_id(channel_name)
        {
            if let Some(dg) = self.dg.get(dg_pos) {
                if let Some(cg) = dg.cg.get(rec_id) {
                    if let Some(cn) = cg.cn.get(rec_pos) {
                        state = cn.channel_data_valid
                    }
                }
            }
        }
        state
    }
    /// Returns the channel's unit string. If it does not exist, it is an empty string.
    pub fn get_channel_unit(&self, channel_name: &str) -> Option<String> {
        let mut unit: Option<String> = None;
        if let Some((_master, dg_pos, (_cg_pos, rec_id), (_cn_pos, rec_pos))) =
            self.get_channel_id(channel_name)
        {
            if let Some(dg) = self.dg.get(dg_pos) {
                if let Some(cg) = dg.cg.get(rec_id) {
                    if let Some(cn) = cg.cn.get(rec_pos) {
                        unit = self.sharable.get_tx(cn.block.cn_md_unit);
                    }
                }
            }
        }
        unit
    }
    /// Returns the channel's description. If it does not exist, it is an empty string
    pub fn get_channel_desc(&self, channel_name: &str) -> Option<String> {
        let mut desc: Option<String> = None;
        if let Some((_master, dg_pos, (_cg_pos, rec_id), (_cn_pos, rec_pos))) =
            self.get_channel_id(channel_name)
        {
            if let Some(dg) = self.dg.get(dg_pos) {
                if let Some(cg) = dg.cg.get(rec_id) {
                    if let Some(cn) = cg.cn.get(rec_pos) {
                        desc = self.sharable.get_tx(cn.block.cn_md_comment);
                    }
                }
            }
        }
        desc
    }
    /// returns the master channel associated to the input channel name
    pub fn get_channel_master(&self, channel_name: &str) -> Option<String> {
        let mut master: Option<String> = None;
        if let Some((m, _dg_pos, (_cg_pos, _rec_idd), (_cn_pos, _rec_pos))) =
            self.get_channel_id(channel_name)
        {
            master = m.clone();
        }
        master
    }
    /// returns type of master channel link to channel input in parameter:
    /// 0 = None (normal data channels), 1 = Time (seconds), 2 = Angle (radians),
    /// 3 = Distance (meters), 4 = Index (zero-based index values)
    pub fn get_channel_master_type(&self, channel_name: &str) -> u8 {
        let mut master_type: u8 = 0; // default to normal data channel
        if let Some((_master, dg_pos, (_cg_pos, rec_id), (_cn_pos, rec_pos))) =
            self.get_channel_id(channel_name)
        {
            if let Some(dg) = self.dg.get(dg_pos) {
                if let Some(cg) = dg.cg.get(rec_id) {
                    if let Some(cn) = cg.cn.get(rec_pos) {
                        master_type = cn.block.cn_sync_type;
                    }
                }
            }
        }
        master_type
    }
    /// returns the set of channel names
    pub fn get_channel_names_set(&self) -> HashSet<String> {
        let channel_list = self.channel_names_set.keys().cloned().collect();
        channel_list
    }
    /// returns a hashmap for which master channel names are keys and values its corresponding set of channel names
    pub fn get_master_channel_names_set(&self) -> HashMap<Option<String>, HashSet<String>> {
        let mut channel_master_list: HashMap<Option<String>, HashSet<String>> = HashMap::new();
        for (_dg_position, dg) in self.dg.iter() {
            for (_record_id, cg) in dg.cg.iter() {
                if let Some(list) = channel_master_list.get_mut(&cg.master_channel_name) {
                    list.extend(cg.channel_names.clone().into_iter());
                } else {
                    channel_master_list
                        .insert(cg.master_channel_name.clone(), cg.channel_names.clone());
                }
            }
        }
        channel_master_list
    }
    /// load in memory the ndarray data of a set of channels
    pub fn load_channels_data_in_memory(&mut self, channel_names: HashSet<String>) {
        let f: File = OpenOptions::new()
            .read(true)
            .write(false)
            .open(self.file_name.clone())
            .expect("Cannot find the file");
        let mut rdr = BufReader::new(&f);
        mdfreader4(&mut rdr, self, channel_names);
    }
    /// load all channels data in memory
    pub fn load_all_channels_data_in_memory(&mut self) {
        let channel_set = self.get_channel_names_set();
        self.load_channels_data_in_memory(channel_set);
        self.all_data_in_memory = true;
    }
    // empty the channels' ndarray
    pub fn clear_channel_data_from_memory(&mut self, channel_names: HashSet<String>) {
        for channel_name in channel_names {
            if let Some((_master, dg_pos, (_cg_pos, rec_id), (_cn_pos, rec_pos))) =
                self.channel_names_set.get_mut(&channel_name)
            {
                if let Some(dg) = self.dg.get_mut(dg_pos) {
                    if let Some(cg) = dg.cg.get_mut(rec_id) {
                        if let Some(cn) = cg.cn.get_mut(rec_pos) {
                            if !cn.data.is_empty() {
                                cn.data = cn.data.zeros(cn.block.cn_data_type, 0, 0, 0);
                            }
                        }
                    }
                }
            }
        }
        self.all_data_in_memory = false;
    }
    pub fn write(&mut self, file_name: &str, compression: bool) -> MdfInfo4 {
        let f: File = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(file_name)
            .expect("Cannot create the file");
        let mut rdr = BufWriter::new(&f);
        mdfwriter4(&mut rdr, self, file_name, compression)
    }
    pub fn new(file_name: &str, n_channels: usize) -> MdfInfo4 {
        MdfInfo4 {
            file_name: file_name.to_string(),
            dg: BTreeMap::new(),
            sharable: SharableBlocks::new(n_channels),
            channel_names_set: HashMap::with_capacity(n_channels),
            id_block: IdBlock::default(),
            fh: Vec::new(),
            at: HashMap::new(),
            ev: HashMap::new(),
            hd_block: Hd4::default(),
            all_data_in_memory: false,
        }
    }
    // TODO cut data
    // TODO resample data
    // TODO Write to mdf4 column
    // TODO Extract attachments
}

/// MdfInfo4 display implementation
impl fmt::Display for MdfInfo4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MdfInfo4: {}", self.file_name)?;
        writeln!(f, "Version : {}\n", self.id_block.id_ver)?;
        writeln!(f, "{}\n", self.hd_block)?;
        let comments = &self.sharable.get_comments(self.hd_block.hd_md_comment);
        for c in comments.iter() {
            writeln!(f, "{} {}\n", c.0, c.1)?;
        }
        for (master, list) in self.get_master_channel_names_set().iter() {
            if let Some(master_name) = master {
                writeln!(f, "\nMaster: {}\n", master_name)?;
            } else {
                writeln!(f, "\nWithout Master channel\n")?;
            }
            for channel in list.iter() {
                let unit = self.get_channel_unit(channel);
                let desc = self.get_channel_desc(channel);
                let dtmsk = self.get_channel_data_from_memory(channel);
                if let Some(data) = dtmsk.0 {
                    let data_first_last = data.first_last();
                    writeln!(
                        f,
                        " {} {} {:?} {:?} \n",
                        channel, data_first_last, unit, desc
                    )?;
                } else {
                    writeln!(f, " {} {:?} {:?} \n", channel, unit, desc)?;
                }
            }
        }
        writeln!(f, "\n")
    }
}

/// MDF4 - common block Header
#[derive(Debug, Copy, Clone)]
#[binrw]
#[br(little)]
pub struct Blockheader4 {
    /// '##XX'
    pub hdr_id: [u8; 4],
    /// reserved, must be 0
    hdr_gap: [u8; 4],
    /// Length of block in bytes
    pub hdr_len: u64,
    /// # of links
    pub hdr_links: u64,
}

impl Default for Blockheader4 {
    fn default() -> Self {
        Blockheader4 {
            hdr_id: [35, 35, 84, 88], // ##TX
            hdr_gap: [0x00, 0x00, 0x00, 0x00],
            hdr_len: 24,
            hdr_links: 0,
        }
    }
}

/// parse the block header and its fields id, (reserved), length and number of links
#[inline]
pub fn parse_block_header(rdr: &mut BufReader<&File>) -> Blockheader4 {
    let mut buf = [0u8; 24];
    rdr.read_exact(&mut buf)
        .expect("could not read blockheader4 Id");
    let mut block = Cursor::new(buf);
    let header: Blockheader4 = block.read_le().expect("could not parse blockheader4");
    header
}

/// MDF4 - common block Header without the number of links
#[derive(Debug, Copy, Clone)]
#[binrw]
#[br(little)]
#[allow(dead_code)]
pub struct Blockheader4Short {
    /// '##XX'
    hdr_id: [u8; 4],
    /// reserved, must be 0
    hdr_gap: [u8; 4],
    /// Length of block in bytes
    hdr_len: u64,
}

/// parse the block header and its fields id, (reserved), length except the number of links
#[inline]
fn parse_block_header_short(rdr: &mut BufReader<&File>) -> Blockheader4Short {
    let mut buf = [0u8; 16];
    rdr.read_exact(&mut buf)
        .expect("could not read short blockheader4 Id");
    let mut block = Cursor::new(buf);
    let header: Blockheader4Short = block.read_le().expect("could not parse short blockheader4");
    header
}

/// reads generically a block header and return links and members section part into a Seek buffer for further processing
#[inline]
fn parse_block(
    rdr: &mut BufReader<&File>,
    target: i64,
    mut position: i64,
) -> (Cursor<Vec<u8>>, Blockheader4, i64) {
    // Reads block header
    rdr.seek_relative(target - position)
        .expect("Could not reach block header position"); // change buffer position
    let block_header: Blockheader4 = parse_block_header(rdr); // reads header

    // Reads in buffer rest of block
    let mut buf = vec![0u8; (block_header.hdr_len - 24) as usize];
    rdr.read_exact(&mut buf)
        .expect("Could not read rest of block after header");
    position = target + block_header.hdr_len as i64;
    let block = Cursor::new(buf);
    (block, block_header, position)
}

/// reads generically a block header wihtout the number of links and returns links and members section part into a Seek buffer for further processing
#[inline]
fn parse_block_short(
    rdr: &mut BufReader<&File>,
    target: i64,
    mut position: i64,
) -> (Cursor<Vec<u8>>, Blockheader4Short, i64) {
    // Reads block header
    rdr.seek_relative(target - position)
        .expect("Could not reach block short header position"); // change buffer position
    let block_header: Blockheader4Short = parse_block_header_short(rdr); // reads header

    // Reads in buffer rest of block
    let mut buf = vec![0u8; (block_header.hdr_len - 16) as usize];
    rdr.read_exact(&mut buf)
        .expect("Could not read rest of block after short header");
    position = target + block_header.hdr_len as i64;
    let block = Cursor::new(buf);
    (block, block_header, position)
}

/// metadata are either stored in TX (text) or MD (xml) blocks for mdf version 4
#[derive(Debug, Clone, PartialEq)]
pub enum MetaDataBlockType {
    MdBlock,
    MdParsed,
    TX,
}

impl Default for MetaDataBlockType {
    fn default() -> Self {
        MetaDataBlockType::TX
    }
}

#[derive(Debug, Clone)]
pub enum BlockType {
    HD,
    FH,
    AT,
    EV,
    DG,
    CG,
    CN,
    CC,
    SI,
}

impl Default for BlockType {
    fn default() -> Self {
        BlockType::CN
    }
}

/// struct linking MD or TX block with
#[derive(Debug, Default, Clone)]
pub struct MetaData {
    pub block: Blockheader4,
    pub raw_data: Vec<u8>,
    pub block_type: MetaDataBlockType,
    pub comments: HashMap<String, String>,
    pub parent_block_type: BlockType,
}

fn read_meta_data(
    rdr: &mut BufReader<&File>,
    sharable: &mut SharableBlocks,
    target: i64,
    mut position: i64,
    parent_block_type: BlockType,
) -> i64 {
    if target != 0 && !sharable.md_tx.contains_key(&target) {
        rdr.seek_relative(target - position)
            .expect("Could not reach block short header position"); // change buffer position
        let block: Blockheader4 = rdr.read_le().expect("Could not read MD or TX Block");
        let mut raw_data = vec![0u8; (block.hdr_len - 24) as usize];
        rdr.read_exact(&mut raw_data)
            .expect("Could not read rest of block after header");
        position = target + block.hdr_len as i64;
        let block_type = match block.hdr_id {
            [35, 35, 77, 68] => MetaDataBlockType::MdBlock,
            [35, 35, 84, 88] => MetaDataBlockType::TX,
            _ => MetaDataBlockType::TX,
        };
        let md = MetaData {
            block,
            raw_data,
            block_type,
            comments: HashMap::new(),
            parent_block_type,
        };
        sharable.md_tx.insert(target, md);
        position
    } else {
        position
    }
}

impl MetaData {
    pub fn new(block_type: MetaDataBlockType, parent_block_type: BlockType) -> Self {
        let header = match block_type {
            MetaDataBlockType::MdBlock => Blockheader4 {
                hdr_id: [35, 35, 77, 68], // '##MD'
                hdr_gap: [0u8; 4],
                hdr_len: 24,
                hdr_links: 0,
            },
            MetaDataBlockType::TX => Blockheader4 {
                hdr_id: [35, 35, 84, 88], // '##TX'
                hdr_gap: [0u8; 4],
                hdr_len: 0,
                hdr_links: 0,
            },
            MetaDataBlockType::MdParsed => panic!("MdParsed not implemented for read function"),
        };
        MetaData {
            block: header,
            raw_data: Vec::new(),
            block_type,
            comments: HashMap::new(),
            parent_block_type,
        }
    }
    pub fn parse_xml(&mut self) {
        match self.block_type {
            MetaDataBlockType::MdBlock =>  {
                match self.parent_block_type {
                    BlockType::HD => self.parse_hd_xml(),
                    BlockType::FH => self.parse_fh_xml(),
                    _ => self.parse_generic_xml(),
                }
            },
            _ => {},
        }
    }
    pub fn get_data_string(&self) -> String {
        match self.block_type {
            MetaDataBlockType::MdParsed => String::new(),
            _ => {
                let comment = match str::from_utf8(&self.raw_data) {
                    Ok(v) => v,
                    Err(e) => panic!("Invalid UTF-8 sequence in metadata: {}", e),
                };
                let comment: String = comment.trim_end_matches(char::from(0)).into();
                comment
            }
        }
    }
    pub fn set_data_buffer(&mut self, data: String) {
        self.raw_data = format!("{:\0<width$}", data, width = (data.len() / 8 + 1) * 8).into();
        self.block.hdr_len = self.raw_data.len() as u64 + 24;
    }
    fn parse_hd_xml(&mut self) {
        let mut comments: HashMap<String, String> = HashMap::new();
        // MD Block from HD Block, reading xml
        let comment: String = self
            .get_data_string()
            .trim_end_matches(|c| c == '\n' || c == '\r' || c == ' ')
            .into(); // removes ending spaces
        match roxmltree::Document::parse(&comment) {
            Ok(md) => {
                for node in md.root().descendants().filter(|p| p.has_tag_name("e")) {
                    if let (Some(value), Some(text)) = (node.attribute("name"), node.text()) {
                        comments.insert(value.to_string(), text.to_string());
                    }
                }
            }
            Err(e) => {
                println!("Error parsing HD MD comment : \n{}\n{}", comment, e);
            }
        };
        self.comments = comments;
        self.block_type = MetaDataBlockType::MdParsed;
        self.raw_data = vec![]; // empty the data from block as already parsed
    }
    pub fn create_fh(&mut self) {
        let user_name = whoami::username();
        let comments = format!(
            "<FHcomment>
<TX>created</TX>
<tool_id>mdfr</tool_id>
<tool_vendor>ratalco</tool_vendor>
<tool_version>0.1</tool_version>
<user_name>{}</user_name>
</FHcomment>",
            user_name
        );
        let raw_comments = format!(
            "{:\0<width$}",
            comments,
            width = (comments.len() / 8 + 1) * 8
        );
        let fh_comments = raw_comments.as_bytes();
        self.block.hdr_len = fh_comments.len() as u64 + 24;
        self.raw_data = fh_comments.to_vec();
    }
    fn parse_fh_xml(&mut self) {
        let mut comments: HashMap<String, String> = HashMap::new();
        // MD Block from FH Block, reading xml
        let comment: String = self
            .get_data_string()
            .trim_end_matches(|c| c == '\n' || c == '\r' || c == ' ')
            .into(); // removes ending spaces
        match roxmltree::Document::parse(&comment) {
            Ok(md) => {
                for node in md.root().descendants() {
                    let text = match node.text() {
                        Some(text) => text.to_string(),
                        None => String::new(),
                    };
                    comments.insert(node.tag_name().name().to_string(), text);
                }
            }
            Err(e) => {
                println!("Error parsing FH comment : \n{}\n{}", comment, e);
            }
        };
        self.comments = comments;
        self.block_type = MetaDataBlockType::MdParsed;
        self.raw_data = vec![]; // empty the data from block as already parsed
    }
    fn parse_generic_xml(&mut self) {
        let mut comments: HashMap<String, String> = HashMap::new();
        let comment: String = self
            .get_data_string()
            .trim_end_matches(|c| c == '\n' || c == '\r' || c == ' ')
            .into(); // removes ending spaces
        match roxmltree::Document::parse(&comment) {
            Ok(md) => {
                for node in md.root().descendants() {
                    let text = match node.text() {
                        Some(text) => text.to_string(),
                        None => String::new(),
                    };
                    if node.is_element()
                        && !text.is_empty()
                        && !node.tag_name().name().to_string().is_empty()
                    {
                        comments.insert(node.tag_name().name().to_string(), text);
                    }
                }
            }
            Err(e) => {
                println!("Error parsing comment : \n{}\n{}", comment, e);
            }
        };
        self.comments = comments;
        self.block_type = MetaDataBlockType::MdParsed;
        self.raw_data = vec![]; // empty the data from block as already parsed
    }
    pub fn write(&self, writer: &mut BufWriter<&File>) {
        writer
            .write_le(&self.block)
            .expect("Could not write comment block header");
        writer
            .write_all(&self.raw_data)
            .expect("Could not write comment block data");
    }
}

/// Hd4 (Header) block structure
#[derive(Debug, Copy, Clone)]
#[binrw]
#[br(little)]
#[allow(dead_code)]
pub struct Hd4 {
    /// ##HD
    hd_id: [u8; 4],
    /// reserved  
    hd_reserved: [u8; 4],
    /// Length of block in bytes
    hd_len: u64,
    /// # of links
    hd_link_counts: u64,
    /// Pointer to the first data group block (DGBLOCK) (can be NIL)
    pub hd_dg_first: i64,
    /// Pointer to first file history block (FHBLOCK)
    /// There must be at least one FHBLOCK with information about the application which created the MDF file.
    pub hd_fh_first: i64,
    /// Pointer to first channel hierarchy block (CHBLOCK) (can be NIL).
    hd_ch_first: i64,
    /// Pointer to first attachment block (ATBLOCK) (can be NIL)
    pub hd_at_first: i64,
    /// Pointer to first event block (EVBLOCK) (can be NIL)
    pub hd_ev_first: i64,
    /// Pointer to the measurement file comment (TXBLOCK or MDBLOCK) (can be NIL) For MDBLOCK contents, see Table 14.
    pub hd_md_comment: i64,
    /// Data members
    /// Time stamp in nanoseconds elapsed since 00:00:00 01.01.1970 (UTC time or local time, depending on "local time" flag, see [UTC]).
    pub hd_start_time_ns: u64,
    /// Time zone offset in minutes. The value must be in range [-720,720], i.e. it can be negative! For example a value of 60 (min) means UTC+1 time zone = Central European Time (CET). Only valid if "time offsets valid" flag is set in time flags.
    pub hd_tz_offset_min: i16,
    /// Daylight saving time (DST) offset in minutes for start time stamp. During the summer months, most regions observe a DST offset of 60 min (1 hour). Only valid if "time offsets valid" flag is set in time flags.
    pub hd_dst_offset_min: i16,
    /// Time flags The value contains the following bit flags (see HD_TF_xxx)
    pub hd_time_flags: u8,
    /// Time quality class (see HD_TC[35, 35, 72, 68]_xxx)
    pub hd_time_class: u8,
    /// Flags The value contains the following bit flags (see HD_FL_xxx):
    pub hd_flags: u8,
    /// reserved
    pub hd_reserved2: u8,
    /// Start angle in radians at start of measurement (only for angle synchronous measurements) Only valid if "start angle valid" flag is set. All angle values for angle synchronized master channels or events are relative to this start angle.
    pub hd_start_angle_rad: f64,
    /// Start distance in meters at start of measurement (only for distance synchronous measurements) Only valid if "start distance valid" flag is set. All distance values for distance synchronized master channels or events are relative to this start distance.
    pub hd_start_distance_m: f64,
}

impl Default for Hd4 {
    fn default() -> Self {
        Hd4 {
            hd_id: [35, 35, 72, 68], // ##HD
            hd_len: 104,
            hd_link_counts: 6,
            hd_reserved: [0u8; 4],
            hd_dg_first: 0,
            hd_fh_first: 0,
            hd_ch_first: 0,
            hd_at_first: 0,
            hd_ev_first: 0,
            hd_md_comment: 0,
            hd_start_time_ns: Local::now().timestamp_nanos() as u64,
            hd_tz_offset_min: 0,
            hd_dst_offset_min: 0,
            hd_time_flags: 0,
            hd_time_class: 0,
            hd_flags: 0,
            hd_reserved2: 0,
            hd_start_angle_rad: 0.0,
            hd_start_distance_m: 0.0,
        }
    }
}

/// Hd4 display implementation
impl fmt::Display for Hd4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sec = self.hd_start_time_ns / 1000000000;
        let nsec = (self.hd_start_time_ns - sec * 1000000000) as u32;
        let naive = NaiveDateTime::from_timestamp(sec as i64, nsec);
        writeln!(
            f,
            "Time : {} ",
            DateTime::<Utc>::from_utc(naive, Utc).to_rfc3339()
        )
    }
}

/// Hd4 block struct parser
pub fn hd4_parser(rdr: &mut BufReader<&File>, sharable: &mut SharableBlocks) -> (Hd4, i64) {
    let mut buf = [0u8; 104];
    rdr.read_exact(&mut buf)
        .expect("could not read HB block buffer");
    let mut block = Cursor::new(buf);
    let hd: Hd4 = block
        .read_le()
        .expect("Could not read HD block buffer into Hd4 struct");
    let position = read_meta_data(rdr, sharable, hd.hd_md_comment, 168, BlockType::HD);
    (hd, position)
}

/// Fh4 (File History) block struct, including the header
#[derive(Debug, Copy, Clone)]
#[binrw]
#[br(little)]
#[allow(dead_code)]
pub struct FhBlock {
    /// '##FH'
    fh_id: [u8; 4],
    /// reserved, must be 0
    fh_gap: [u8; 4],
    /// Length of block in bytes
    fh_len: u64,
    /// # of links
    fh_links: u64,
    /// Link to next FHBLOCK (can be NIL if list finished)
    pub fh_fh_next: i64,
    /// Link to MDBLOCK containing comment about the creation or modification of the MDF file.    
    pub fh_md_comment: i64,
    /// time stamp in nanosecs
    pub fh_time_ns: u64,
    /// time zone offset on minutes
    pub fh_tz_offset_min: i16,
    /// daylight saving time offset in minutes for start time stamp
    pub fh_dst_offset_min: i16,
    /// time flags, but 1 local, bit 2 time offsets
    pub fh_time_flags: u8,
    /// reserved
    fh_reserved: [u8; 3],
}

impl Default for FhBlock {
    fn default() -> Self {
        FhBlock {
            fh_id: [35, 35, 70, 72], // '##FH'
            fh_gap: [0u8; 4],
            fh_len: 56,
            fh_links: 2,
            fh_fh_next: 0,
            fh_md_comment: 0,
            fh_time_ns: Local::now().timestamp_nanos() as u64,
            fh_tz_offset_min: 0,
            fh_dst_offset_min: 0,
            fh_time_flags: 0,
            fh_reserved: [0u8; 3],
        }
    }
}

/// Fh4 (File History) block struct parser
fn parse_fh_block(rdr: &mut BufReader<&File>, target: i64, position: i64) -> (FhBlock, i64) {
    rdr.seek_relative(target - position)
        .expect("Could not reach FH Block position"); // change buffer position
    let mut buf = [0u8; 56];
    rdr.read_exact(&mut buf)
        .expect("Could not read FH block buffer");
    let mut block = Cursor::new(buf);
    let fh: FhBlock = match block.read_le() {
        Ok(v) => v,
        Err(e) => panic!("Error reading fh block into FhBlock struct \n{}", e),
    }; // reads the fh block
    (fh, target + 56)
}

type Fh = Vec<FhBlock>;

/// parses File History blocks along with its linked comments returns a vect of Fh4 block with comments
pub fn parse_fh(
    rdr: &mut BufReader<&File>,
    sharable: &mut SharableBlocks,
    target: i64,
    mut position: i64,
) -> (Fh, i64) {
    let mut fh: Fh = Vec::new();
    let (block, pos) = parse_fh_block(rdr, target, position);
    position = pos;
    position = read_meta_data(rdr, sharable, block.fh_md_comment, position, BlockType::FH);
    let mut next_pointer = block.fh_fh_next;
    fh.push(block);
    while next_pointer != 0 {
        let (block, pos) = parse_fh_block(rdr, next_pointer, position);
        position = pos;
        next_pointer = block.fh_fh_next;
        position = read_meta_data(rdr, sharable, block.fh_md_comment, position, BlockType::FH);
        fh.push(block);
    }
    (fh, position)
}
/// At4 Attachment block struct
#[derive(Debug, Copy, Clone)]
#[binrw]
#[br(little)]
#[allow(dead_code)]
pub struct At4Block {
    /// ##DG
    at_id: [u8; 4],
    /// reserved
    reserved: [u8; 4],
    /// Length of block in bytes
    at_len: u64,
    /// # of links
    at_links: u64,
    /// Link to next ATBLOCK (linked list) (can be NIL)
    at_at_next: i64,
    /// Link to TXBLOCK with the path and file name of the embedded or referenced file (can only be NIL if data is embedded). The path of the file can be relative or absolute. If relative, it is relative to the directory of the MDF file. If no path is given, the file must be in the same directory as the MDF file.      
    at_tx_filename: i64,
    /// Link to TXBLOCK with MIME content-type text that gives information about the attached data. Can be NIL if the content-type is unknown, but should be specified whenever possible. The MIME content-type string must be written in lowercase.
    at_tx_mimetype: i64,
    /// Link to MDBLOCK with comment and additional information about the attachment (can be NIL).
    at_md_comment: i64,
    /// Flags The value contains the following bit flags (see AT_FL_xxx):
    at_flags: u16,
    /// Creator index, i.e. zero-based index of FHBLOCK in global list of FHBLOCKs that specifies which application has created this attachment, or changed it most recently.
    at_creator_index: u16,
    /// Reserved
    at_reserved: [u8; 4],
    /// 128-bit value for MD5 check sum (of the uncompressed data if data is embedded and compressed). Only valid if "MD5 check sum valid" flag (bit 2) is set.
    at_md5_checksum: [u8; 16],
    /// Original data size in Bytes, i.e. either for external file or for uncompressed data.
    at_original_size: u64,
    /// Embedded data size N, i.e. number of Bytes for binary embedded data following this element. Must be 0 if external file is referenced.
    at_embedded_size: u64,
    // followed by embedded data depending of flag
}

/// At4 (Attachment) block struct parser
fn parser_at4_block(
    rdr: &mut BufReader<&File>,
    target: i64,
    mut position: i64,
) -> (At4Block, Option<Vec<u8>>, i64) {
    rdr.seek_relative(target - position)
        .expect("Could not reach At4 Block position");
    let mut buf = [0u8; 96];
    rdr.read_exact(&mut buf)
        .expect("Could not read At4 Block buffer");
    let mut block = Cursor::new(buf);
    let block: At4Block = block
        .read_le()
        .expect("Could not read At4 Block buffer into At4Block struct");
    position = target + 96;

    // reads embedded if exists
    let data: Option<Vec<u8>> = if (block.at_flags & 0b1) > 0 {
        let mut embedded_data = vec![0u8; block.at_embedded_size as usize];
        rdr.read_exact(&mut embedded_data)
            .expect("Could not read At4Block embedded attachement");
        position += block.at_embedded_size as i64;
        Some(embedded_data)
    } else {
        None
    };
    (block, data, position)
}

type At = HashMap<i64, (At4Block, Option<Vec<u8>>)>;

/// parses Attachment blocks along with its linked comments, returns a hashmap of At4 block and attached data in a vect
pub fn parse_at4(
    rdr: &mut BufReader<&File>,
    sharable: &mut SharableBlocks,
    target: i64,
    mut position: i64,
) -> (At, i64) {
    let mut at: At = HashMap::new();
    if target > 0 {
        let (block, data, pos) = parser_at4_block(rdr, target, position);
        position = pos;
        // Reads MD
        position = read_meta_data(rdr, sharable, block.at_md_comment, position, BlockType::AT);
        // reads TX file_name
        position = read_meta_data(rdr, sharable, block.at_tx_filename, position, BlockType::AT);
        // Reads tx mime type
        position = read_meta_data(rdr, sharable, block.at_tx_mimetype, position, BlockType::AT);
        let mut next_pointer = block.at_at_next;
        at.insert(target, (block, data));

        while next_pointer > 0 {
            let block_start = next_pointer;
            let (block, data, pos) = parser_at4_block(rdr, next_pointer, position);
            position = pos;
            // Reads MD
            position = read_meta_data(rdr, sharable, block.at_md_comment, position, BlockType::AT);
            // reads TX file_name
            position = read_meta_data(rdr, sharable, block.at_tx_filename, position, BlockType::AT);
            // Reads tx mime type
            position = read_meta_data(rdr, sharable, block.at_tx_mimetype, position, BlockType::AT);
            next_pointer = block.at_at_next;
            at.insert(block_start, (block, data));
        }
    }
    (at, position)
}

/// Ev4 Event block struct
#[derive(Debug, Clone)]
#[binrw]
#[br(little)]
#[allow(dead_code)]
pub struct Ev4Block {
    //ev_id: [u8; 4],  // DG
    //reserved: [u8; 4],  // reserved
    //ev_len: u64,      // Length of block in bytes
    /// # of links
    ev_links: u64,
    /// Link to next EVBLOCK (linked list) (can be NIL)
    ev_ev_next: i64,
    /// Referencing link to EVBLOCK with parent event (can be NIL).
    ev_ev_parent: i64,
    /// Referencing link to EVBLOCK with event that defines the beginning of a range (can be NIL, must be NIL if ev_range_type ≠ 2).  
    ev_ev_range: i64,
    /// Pointer to TXBLOCK with event name (can be NIL) Name must be according to naming rules stated in 4.4.2 Naming Rules. If available, the name of a named trigger condition should be used as event name. Other event types may have individual names or no names.
    ev_tx_name: i64,
    /// Pointer to TX/MDBLOCK with event comment and additional information, e.g. trigger condition or formatted user comment text (can be NIL)
    ev_md_comment: i64,
    #[br(if(ev_links > 5), little, count = ev_links - 5)]
    /// links
    links: Vec<i64>,

    /// Event type (see EV_T_xxx)
    ev_type: u8,
    /// Sync type (see EV_S_xxx)
    ev_sync_type: u8,
    /// Range Type (see EV_R_xxx)
    ev_range_type: u8,
    /// Cause of event (see EV_C_xxx)
    ev_cause: u8,
    /// flags (see EV_F_xxx)
    ev_flags: u8,
    /// Reserved
    ev_reserved: [u8; 3],
    /// Length M of ev_scope list. Can be zero.
    ev_scope_count: u32,
    /// Length N of ev_at_reference list, i.e. number of attachments for this event. Can be zero.
    ev_attachment_count: u16,
    /// Creator index, i.e. zero-based index of FHBLOCK in global list of FHBLOCKs that specifies which application has created or changed this event (e.g. when generating event offline).
    ev_creator_index: u16,
    /// Base value for synchronization value.
    ev_sync_base_value: i64,
    /// Factor for event synchronization value.
    ev_sync_factor: f64,
}

/// Ev4 (Event) block struct parser
fn parse_ev4_block(rdr: &mut BufReader<&File>, target: i64, mut position: i64) -> (Ev4Block, i64) {
    let (mut block, _header, pos) = parse_block_short(rdr, target, position);
    position = pos;
    let block: Ev4Block = match block.read_le() {
        Ok(v) => v,
        Err(e) => panic!("Error reading ev block \n{}", e),
    }; // reads the fh block

    (block, position)
}

/// parses Event blocks along with its linked comments, returns a hashmap of Ev4 block with position as key
pub fn parse_ev4(
    rdr: &mut BufReader<&File>,
    sharable: &mut SharableBlocks,
    target: i64,
    mut position: i64,
) -> (HashMap<i64, Ev4Block>, i64) {
    let mut ev: HashMap<i64, Ev4Block> = HashMap::new();
    if target > 0 {
        let (block, pos) = parse_ev4_block(rdr, target, position);
        position = pos;
        // Reads MD
        position = read_meta_data(rdr, sharable, block.ev_md_comment, position, BlockType::EV);
        // reads TX event name
        position = read_meta_data(rdr, sharable, block.ev_tx_name, position, BlockType::EV);
        let mut next_pointer = block.ev_ev_next;
        ev.insert(target, block);

        while next_pointer > 0 {
            let block_start = next_pointer;
            let (block, pos) = parse_ev4_block(rdr, next_pointer, position);
            position = pos;
            // Reads MD
            position = read_meta_data(rdr, sharable, block.ev_md_comment, position, BlockType::EV);
            // reads TX event name
            position = read_meta_data(rdr, sharable, block.ev_tx_name, position, BlockType::EV);
            next_pointer = block.ev_ev_next;
            ev.insert(block_start, block);
        }
    }
    (ev, position)
}

/// Dg4 Data Group block struct
#[derive(Debug, Copy, Clone)]
#[binrw]
#[br(little)]
#[allow(dead_code)]
pub struct Dg4Block {
    /// ##DG
    dg_id: [u8; 4],
    /// reserved
    reserved: [u8; 4],
    /// Length of block in bytes
    pub dg_len: u64,
    /// # of links
    dg_links: u64,
    /// Pointer to next data group block (DGBLOCK) (can be NIL)
    pub dg_dg_next: i64,
    /// Pointer to first channel group block (CGBLOCK) (can be NIL)
    pub dg_cg_first: i64,
    // Pointer to data block (DTBLOCK or DZBLOCK for this block type) or data list block (DLBLOCK of data blocks or its HLBLOCK)  (can be NIL)
    pub dg_data: i64,
    /// comment
    dg_md_comment: i64,
    /// number of bytes used for record IDs. 0 no recordID
    pub dg_rec_id_size: u8,
    // reserved
    reserved_2: [u8; 7],
}

impl Default for Dg4Block {
    fn default() -> Self {
        Dg4Block {
            dg_id: [35, 35, 68, 71], // ##DG
            reserved: [0; 4],
            dg_len: 64,
            dg_links: 4,
            dg_dg_next: 0,
            dg_cg_first: 0,
            dg_data: 0,
            dg_md_comment: 0,
            dg_rec_id_size: 0,
            reserved_2: [0; 7],
        }
    }
}

/// Dg4 (Data Group) block struct parser with comments
fn parse_dg4_block(
    rdr: &mut BufReader<&File>,
    sharable: &mut SharableBlocks,
    target: i64,
    mut position: i64,
) -> (Dg4Block, i64) {
    rdr.seek_relative(target - position)
        .expect("Could not reach position of Dg4 block");
    let mut buf = [0u8; 64];
    rdr.read_exact(&mut buf)
        .expect("Could not read Dg4Blcok buffer");
    let mut block = Cursor::new(buf);
    let dg: Dg4Block = block
        .read_le()
        .expect("Could not read Dg4Block buffer into Dg4Block struct");
    position = target + 64;

    // Reads MD
    position = read_meta_data(rdr, sharable, dg.dg_md_comment, position, BlockType::DG);

    (dg, position)
}

/// Dg4 struct wrapping block, comments and linked CG
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Dg4 {
    /// DG Block
    pub block: Dg4Block,
    /// CG Block
    pub cg: HashMap<u64, Cg4>,
}

/// Parser for Dg4 and all linked blocks (cg, cn, cc, ca, si)
pub fn parse_dg4(
    rdr: &mut BufReader<&File>,
    target: i64,
    mut position: i64,
    sharable: &mut SharableBlocks,
) -> (BTreeMap<i64, Dg4>, i64, usize, usize) {
    let mut dg: BTreeMap<i64, Dg4> = BTreeMap::new();
    // TODO Hash to BTree map performance investigation for only data block positions (DG but also DL/LD ?)
    let mut n_cn: usize = 0;
    let mut n_cg: usize = 0;
    if target > 0 {
        let (block, pos) = parse_dg4_block(rdr, sharable, target, position);
        position = pos;
        let mut next_pointer = block.dg_dg_next;
        let (mut cg, pos, num_cg, num_cn) = parse_cg4(
            rdr,
            block.dg_cg_first,
            position,
            sharable,
            block.dg_rec_id_size,
        );
        n_cg += num_cg;
        n_cn += num_cn;
        identify_vlsd_cg(&mut cg);
        let dg_struct = Dg4 { block, cg };
        dg.insert(target, dg_struct);
        position = pos;
        while next_pointer > 0 {
            let block_start = next_pointer;
            let (block, pos) = parse_dg4_block(rdr, sharable, next_pointer, position);
            next_pointer = block.dg_dg_next;
            position = pos;
            let (mut cg, pos, num_cg, num_cn) = parse_cg4(
                rdr,
                block.dg_cg_first,
                position,
                sharable,
                block.dg_rec_id_size,
            );
            n_cg += num_cg;
            n_cn += num_cn;
            identify_vlsd_cg(&mut cg);
            let dg_struct = Dg4 { block, cg };
            dg.insert(block_start, dg_struct);
            position = pos;
        }
    }
    (dg, position, n_cg, n_cn)
}

fn identify_vlsd_cg(cg: &mut HashMap<u64, Cg4>) {
    // First find all VLSD Channel Groups
    let mut vlsd: HashMap<i64, u64> = HashMap::new();
    for (rec_id, channel_group) in cg.iter() {
        if (channel_group.block.cg_flags & 0b1) != 0 {
            // VLSD channel group found
            vlsd.insert(channel_group.block_position, *rec_id);
        }
    }
    if !vlsd.is_empty() {
        // try to find corresponding channel in other channel group
        let mut vlsd_matching: HashMap<u64, (u64, i32)> = HashMap::new();
        for (target_rec_id, channel_group) in cg.iter() {
            for (target_rec_pos, cn) in channel_group.cn.iter() {
                if let Some(vlsd_rec_id) = vlsd.get(&cn.block.cn_data) {
                    // Found matching channel with VLSD_CG
                    vlsd_matching.insert(*vlsd_rec_id, (*target_rec_id, *target_rec_pos));
                }
            }
        }
        for (vlsd_rec_id, (target_rec_id, target_rec_pos)) in vlsd_matching {
            if let Some(vlsd_cg) = cg.get_mut(&vlsd_rec_id) {
                vlsd_cg.vlsd_cg = Some((target_rec_id, target_rec_pos));
            }
        }
    }
}

/// sharable blocks (most likely referenced multiple times and shared by several blocks)
/// that are in sharable fields and holds CC, SI, TX and MD blocks
#[derive(Debug, Default, Clone)]
pub struct SharableBlocks {
    pub(crate) md_tx: HashMap<i64, MetaData>,
    pub(crate) cc: HashMap<i64, Cc4Block>,
    pub(crate) si: HashMap<i64, Si4Block>,
}

/// SharableBlocks display implementation to facilitate debugging
impl fmt::Display for SharableBlocks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MD TX comments : \n")?;
        for (k, c) in self.md_tx.iter() {
            match c.block_type {
                MetaDataBlockType::MdParsed => {
                    for (tag, text) in c.comments.iter() {
                        writeln!(f, "Tag: {}  Text: {}", tag, text)?;
                    }
                }
                MetaDataBlockType::TX => {
                    writeln!(f, "Text: {}", c.get_data_string())?;
                }
                _ => (),
            }
        }
        writeln!(f, "CC : \n")?;
        for (position, cc) in self.cc.iter() {
            writeln!(f, "Position: {}  Text: {:?}", position, cc)?;
        }
        writeln!(f, "SI : ")?;
        for (position, si) in self.si.iter() {
            writeln!(f, "Position: {}  Text: {:?}", position, si)?;
        }
        writeln!(f, "finished")
    }
}

impl SharableBlocks {
    pub fn get_tx(&self, position: i64) -> Option<String> {
        let mut txt: Option<String> = None;
        if let Some(md) = self.md_tx.get(&position) {
            match md.block_type {
                MetaDataBlockType::MdParsed => {
                    if let Some(t) = md.comments.get("TX") {
                        txt = Some(t.to_string());
                    }
                }
                MetaDataBlockType::TX => {
                    txt = Some(md.get_data_string());
                }
                MetaDataBlockType::MdBlock => {
                    txt = Some(md.get_data_string());
                }
            }
        };
        txt
    }
    pub fn get_comments(&self, position: i64) -> HashMap<String, String> {
        let mut comments: HashMap<String, String> = HashMap::new();
        if let Some(md) = self.md_tx.get(&position) {
            if let MetaDataBlockType::MdParsed = md.block_type {
                comments = md.comments.clone();
            }
        };
        comments
    }
    pub fn extract_xml(&mut self) {
        self.md_tx
            .par_iter_mut()
            .filter(|(_k, v)| v.block_type == MetaDataBlockType::MdBlock)
            .for_each(|(_k, val)| val.parse_xml());
    }
    pub fn new(n_channels: usize) -> SharableBlocks {
        let md_tx: HashMap<i64, MetaData> = HashMap::with_capacity(n_channels);
        let cc: HashMap<i64, Cc4Block> = HashMap::new();
        let si: HashMap<i64, Si4Block> = HashMap::new();
        SharableBlocks {
            md_tx,
            cc,
            si,
        }
    }
}
/// Cg4 Channel Group block struct
#[derive(Debug, Copy, Clone)]
#[binrw]
#[br(little)]
#[allow(dead_code)]
pub struct Cg4Block {
    /// ##CG
    cg_id: [u8; 4],
    /// reserved
    reserved: [u8; 4],
    /// Length of block in bytes
    pub cg_len: u64,
    /// # of links
    pub cg_links: u64,
    /// Pointer to next channel group block (CGBLOCK) (can be NIL)
    pub cg_cg_next: i64,
    /// Pointer to first channel block (CNBLOCK) (can be NIL, must be NIL for VLSD CGBLOCK, i.e. if "VLSD channel group" flag (bit 0) is set)
    pub cg_cn_first: i64,
    /// Pointer to acquisition name (TXBLOCK) (can be NIL, must be NIL for VLSD CGBLOCK)
    cg_tx_acq_name: i64,
    /// Pointer to acquisition source (SIBLOCK) (can be NIL, must be NIL for VLSD CGBLOCK) See also rules for uniqueness explained in 4.4.3 Identification of Channels.
    cg_si_acq_source: i64,
    /// Pointer to first sample reduction block (SRBLOCK) (can be NIL, must be NIL for VLSD CGBLOCK)
    cg_sr_first: i64,
    ///Pointer to comment and additional information (TXBLOCK or MDBLOCK) (can be NIL, must be NIL for VLSD CGBLOCK)
    cg_md_comment: i64,
    #[br(if(cg_links > 6))]
    pub cg_cg_master: Option<i64>,
    // Data Members
    /// Record ID, value must be less than maximum unsigned integer value allowed by dg_rec_id_size in parent DGBLOCK. Record ID must be unique within linked list of CGBLOCKs.
    pub cg_record_id: u64,
    /// Number of cycles, i.e. number of samples for this channel group. This specifies the number of records of this type in the data block.
    pub cg_cycle_count: u64,
    /// Flags The value contains the following bit flags (see CG_F_xx):
    pub cg_flags: u16,
    cg_path_separator: u16,
    /// Reserved.
    cg_reserved: [u8; 4],
    /// Normal CGBLOCK: Number of data Bytes (after record ID) used for signal values in record, i.e. size of plain data for each recorded sample of this channel group. VLSD CGBLOCK: Low part of a UINT64 value that specifies the total size in Bytes of all variable length signal values for the recorded samples of this channel group. See explanation for cg_inval_bytes.
    pub cg_data_bytes: u32,
    /// Normal CGBLOCK: Number of additional Bytes for record used for invalidation bits. Can be zero if no invalidation bits are used at all. Invalidation bits may only occur in the specified number of Bytes after the data Bytes, not within the data Bytes that contain the signal values. VLSD CGBLOCK: High part of UINT64 value that specifies the total size in Bytes of all variable length signal values for the recorded samples of this channel group, i.e. the total size in Bytes can be calculated by cg_data_bytes + (cg_inval_bytes << 32) Note: this value does not include the Bytes used to specify the length of each VLSD value!
    pub cg_inval_bytes: u32,
}

impl Default for Cg4Block {
    fn default() -> Self {
        Cg4Block {
            cg_id: [35, 35, 67, 71], // ##CG
            reserved: [0u8; 4],
            cg_len: 104, // 112 with cg_cg_master, 104 without
            cg_links: 6, // 7 with cg_cg_master, 6 without
            cg_cg_next: 0,
            cg_cn_first: 0,
            cg_tx_acq_name: 0,
            cg_si_acq_source: 0,
            cg_sr_first: 0,
            cg_md_comment: 0,
            cg_cg_master: None,
            cg_record_id: 0,
            cg_cycle_count: 0,
            cg_flags: 0, // bit 3 set for remote master
            cg_path_separator: 0,
            cg_reserved: [0; 4],
            cg_data_bytes: 0,
            cg_inval_bytes: 0,
        }
    }
}

/// Cg4 (Channel Group) block struct parser with linked comments Source Information in sharable blocks
fn parse_cg4_block(
    rdr: &mut BufReader<&File>,
    target: i64,
    mut position: i64,
    sharable: &mut SharableBlocks,
    record_id_size: u8,
) -> (Cg4, i64, usize) {
    rdr.seek_relative(target - position)
        .expect("Could not reach CG block position"); // change buffer position
    let cg: Cg4Block = rdr
        .read_le()
        .expect("Could not read buffer into Cg4Block struct");
    position = target + cg.cg_len as i64;

    // Reads MD
    position = read_meta_data(rdr, sharable, cg.cg_md_comment, position, BlockType::CG);

    // reads CN (and other linked block behind like CC, SI, CA, etc.)
    let (cn, pos, n_cn, _first_rec_pos) = parse_cn4(
        rdr,
        cg.cg_cn_first,
        position,
        sharable,
        record_id_size,
        cg.cg_cycle_count,
    );
    position = pos;

    // Reads Acq Name
    position = read_meta_data(rdr, sharable, cg.cg_tx_acq_name, position, BlockType::CG);

    // Reads SI Acq name
    let si_pointer = cg.cg_si_acq_source;
    if (si_pointer != 0) && !sharable.si.contains_key(&si_pointer) {
        let (mut si_block, _header, pos) = parse_block_short(rdr, si_pointer, position);
        position = pos;
        let si_block: Si4Block = si_block
            .read_le()
            .expect("Could not read buffer into Si4block struct");
        position = read_meta_data(rdr, sharable, si_block.si_tx_name, position, BlockType::SI);
        position = read_meta_data(rdr, sharable, si_block.si_tx_path, position, BlockType::SI);
        sharable.si.insert(si_pointer, si_block);
    }

    let record_length = cg.cg_data_bytes;

    // Invalid bytes
    let invalid_bytes: Option<Vec<u8>> = if cg.cg_inval_bytes > 0 {
        // invalid bytes exist, adding byte array channel
        Some(vec![
            0u8;
            (cg.cg_inval_bytes as u64 * cg.cg_cycle_count) as usize
        ])
    } else {
        None
    };

    let cg_struct = Cg4 {
        block: cg,
        cn,
        master_channel_name: None,
        channel_names: HashSet::new(),
        record_length,
        block_position: target,
        vlsd_cg: None,
        invalid_bytes,
    };

    (cg_struct, position, n_cn)
}

/// Channel Group struct
/// it contains the related channels structure, a set of channel names, the dedicated master channel name and other helper data.
#[derive(Debug, Clone)]
pub struct Cg4 {
    pub block: Cg4Block,
    /// hashmap of channels
    pub cn: CnType,
    pub master_channel_name: Option<String>,
    pub channel_names: HashSet<String>,
    /// as not stored in .block but can still be referenced by other blocks
    pub block_position: i64,
    /// record length including recordId and invalid bytes
    pub record_length: u32,
    /// pointing to another cg,cn
    pub vlsd_cg: Option<(u64, i32)>,
    /// invalid byte array, optional
    pub invalid_bytes: Option<Vec<u8>>,
}

/// Cg4 implementations for extracting acquisition and source name and path
impl Cg4 {
    fn get_cg_name(&self, sharable: &SharableBlocks) -> Option<String> {
        sharable.get_tx(self.block.cg_tx_acq_name)
    }
    fn get_cg_source_name(&self, sharable: &SharableBlocks) -> Option<String> {
        let si = sharable.si.get(&self.block.cg_si_acq_source);
        match si {
            Some(block) => block.get_si_source_name(sharable),
            None => None,
        }
    }
    fn get_cg_source_path(&self, sharable: &SharableBlocks) -> Option<String> {
        let si = sharable.si.get(&self.block.cg_si_acq_source);
        match si {
            Some(block) => block.get_si_path_name(sharable),
            None => None,
        }
    }
    pub fn new_channel(&mut self, bit_position: i32, cn: Cn4) {
        self.cn.insert(bit_position, cn);
    }
    pub fn process_channel_invalid_bits(&mut self, rec_pos: i32) -> Option<Array1<u8>> {
        // get invalid bytes
        if let Some(invalid_bytes) = &self.invalid_bytes {
            if let Some(cn) = self.cn.get_mut(&rec_pos) {
                let mut mask = Array1::<u8>::zeros((self.block.cg_cycle_count as usize,));
                let byte_offest = (cn.block.cn_inval_bit_pos >> 3) as usize;
                let mut bit = 1;
                bit <<= (cn.block.cn_inval_bit_pos & 0x07) as usize;
                for (index, record) in invalid_bytes
                    .chunks(self.block.cg_inval_bytes as usize)
                    .enumerate()
                {
                    let byte = record[byte_offest];
                    mask[index] = byte & bit;
                }
                Some(mask)
            } else {
                None
            }
        } else {
            None
        }
    }
    pub fn process_all_channel_invalid_bits(&mut self) {
        // get invalid bytes
        let cycle_count = self.block.cg_cycle_count as usize;
        let cg_inval_bytes = self.block.cg_inval_bytes as usize;
        if let Some(invalid_bytes) = &self.invalid_bytes {
            self.cn
                .par_iter_mut()
                .filter(|(_rec_pos, cn)| !cn.data.is_empty())
                .for_each(|(_rec_pos, cn)| {
                    let mut mask = Array1::<u8>::zeros((cycle_count,));
                    let byte_offest = (cn.block.cn_inval_bit_pos >> 3) as usize;
                    let mut bit = 1;
                    bit <<= (cn.block.cn_inval_bit_pos & 0x07) as usize;
                    for (index, record) in invalid_bytes.chunks(cg_inval_bytes).enumerate() {
                        let byte = record[byte_offest];
                        mask[index] = byte & bit;
                    }
                    cn.invalid_mask = Some(mask);
                });
            self.invalid_bytes = None; // Clears out invalid bytes channel
        }
    }
}

/// Cg4 blocks and linked blocks parsing
pub fn parse_cg4(
    rdr: &mut BufReader<&File>,
    target: i64,
    mut position: i64,
    sharable: &mut SharableBlocks,
    record_id_size: u8,
) -> (HashMap<u64, Cg4>, i64, usize, usize) {
    let mut cg: HashMap<u64, Cg4> = HashMap::new();
    let mut n_cg: usize = 0;
    let mut n_cn: usize = 0;
    if target != 0 {
        let (mut cg_struct, pos, num_cn) =
            parse_cg4_block(rdr, target, position, sharable, record_id_size);
        position = pos;
        let mut next_pointer = cg_struct.block.cg_cg_next;
        cg_struct.record_length += record_id_size as u32 + cg_struct.block.cg_inval_bytes;
        cg.insert(cg_struct.block.cg_record_id, cg_struct);
        n_cg += 1;
        n_cn += num_cn;

        while next_pointer != 0 {
            let (mut cg_struct, pos, num_cn) =
                parse_cg4_block(rdr, next_pointer, position, sharable, record_id_size);
            position = pos;
            cg_struct.record_length += record_id_size as u32 + cg_struct.block.cg_inval_bytes;
            next_pointer = cg_struct.block.cg_cg_next;
            cg.insert(cg_struct.block.cg_record_id, cg_struct);
            n_cg += 1;
            n_cn += num_cn;
        }
    }
    (cg, position, n_cg, n_cn)
}

/// Cn4 Channel block struct
#[derive(Debug, PartialEq, Clone)]
#[binrw]
#[br(little)]
pub struct Cn4Block {
    /// ##CN
    cn_id: [u8; 4],
    /// reserved
    reserved: [u8; 4],
    /// Length of block in bytes
    pub cn_len: u64,
    /// # of links
    cn_links: u64,
    /// Pointer to next channel block (CNBLOCK) (can be NIL)
    cn_cn_next: i64,
    /// Composition of channels: Pointer to channel array block (CABLOCK) or channel block (CNBLOCK) (can be NIL). Details see 4.18 Composition of Channels      
    pub cn_composition: i64,
    /// Pointer to TXBLOCK with name (identification) of channel. Name must be according to naming rules stated in 4.4.2 Naming Rules.
    pub cn_tx_name: i64,
    /// Pointer to channel source (SIBLOCK) (can be NIL) Must be NIL for component channels (members of a structure or array elements) because they all must have the same source and thus simply use the SIBLOCK of their parent CNBLOCK (direct child of CGBLOCK).
    cn_si_source: i64,
    /// Pointer to the conversion formula (CCBLOCK) (can be NIL, must be NIL for complex channel data types, i.e. for cn_data_type ≥ 10). If the pointer is NIL, this means that a 1:1 conversion is used (phys = int).  };
    pub cn_cc_conversion: i64,
    /// Pointer to channel type specific signal data For variable length data channel (cn_type = 1): unique link to signal data block (SDBLOCK) or data list block (DLBLOCK) or, only for unsorted data groups, referencing link to a VLSD channel group block (CGBLOCK). Can only be NIL if SDBLOCK would be empty. For synchronization channel (cn_type = 4): referencing link to attachment block (ATBLOCK) in global linked list of ATBLOCKs starting at hd_at_first. Cannot be NIL.
    pub cn_data: i64,
    /// Pointer to TXBLOCK/MDBLOCK with designation for physical unit of signal data (after conversion) or (only for channel data types "MIME sample" and "MIME stream") to MIME context-type text. (can be NIL). The unit can be used if no conversion rule is specified or to overwrite the unit specified for the conversion rule (e.g. if a conversion rule is shared between channels). If the link is NIL, then the unit from the conversion rule must be used. If the content is an empty string, no unit should be displayed. If an MDBLOCK is used, in addition the A-HDO unit definition can be stored, see Table 38. Note: for (virtual) master and synchronization channels the A-HDO definition should be omitted to avoid redundancy. Here the unit is already specified by cn_sync_type of the channel. In case of channel data types "MIME sample" and "MIME stream", the text of the unit must be the content-type text of a MIME type which specifies the content of the values of the channel (either fixed length in record or variable length in SDBLOCK). The MIME content-type string must be written in lowercase, and it must apply to the same rules as defined for at_tx_mimetype in 4.11 The Attachment Block ATBLOCK.
    pub cn_md_unit: i64,
    /// Pointer to TXBLOCK/MDBLOCK with designation for physical unit of signal data (after conversion) or (only for channel data types "MIME sample" and "MIME stream") to MIME context-type text. (can be NIL). The unit can be used if no conversion rule is specified or to overwrite the unit specified for the conversion rule (e.g. if a conversion rule is shared between channels). If the link is NIL, then the unit from the conversion rule must be used. If the content is an empty string, no unit should be displayed. If an MDBLOCK is used, in addition the A-HDO unit definition can be stored, see Table 38. Note: for (virtual) master and synchronization channels the A-HDO definition should be omitted to avoid redundancy. Here the unit is already specified by cn_sync_type of the channel. In case of channel data types "MIME sample" and "MIME stream", the text of the unit must be the content-type text of a MIME type which specifies the content of the values of the channel (either fixed length in record or variable length in SDBLOCK). The MIME content-type string must be written in lowercase, and it must apply to the same rules as defined for at_tx_mimetype in 4.11 The Attachment Block ATBLOCK.
    pub cn_md_comment: i64,
    #[br(if(cn_links > 8), little, count = cn_links - 8)]
    links: Vec<i64>,

    // Data Members
    /// Channel type (see CN_T_xxx)
    pub cn_type: u8,
    /// Sync type: (see CN_S_xxx)
    pub cn_sync_type: u8,
    /// Channel data type of raw signal value (see CN_DT_xxx)
    pub cn_data_type: u8,
    /// Bit offset (0-7): first bit (=LSB) of signal value after Byte offset has been applied (see 4.21.4.2 Reading the Signal Value). If zero, the signal value is 1-Byte aligned. A value different to zero is only allowed for Integer data types (cn_data_type ≤ 3) and if the Integer signal value fits into 8 contiguous Bytes (cn_bit_count + cn_bit_offset ≤ 64). For all other cases, cn_bit_offset must be zero.
    pub cn_bit_offset: u8,
    /// Offset to first Byte in the data record that contains bits of the signal value. The offset is applied to the plain record data, i.e. skipping the record ID.
    cn_byte_offset: u32,
    /// Number of bits for signal value in record
    pub cn_bit_count: u32,
    /// Flags (see CN_F_xxx)
    cn_flags: u32,
    /// Position of invalidation bit.
    cn_inval_bit_pos: u32,
    /// Precision for display of floating point values. 0xFF means unrestricted precision (infinite). Any other value specifies the number of decimal places to use for display of floating point values. Only valid if "precision valid" flag (bit 2) is set
    cn_precision: u8,
    /// Reserved
    cn_reserved: [u8; 3],
    /// Minimum signal value that occurred for this signal (raw value) Only valid if "value range valid" flag (bit 3) is set.
    cn_val_range_min: f64,
    /// Maximum signal value that occurred for this signal (raw value) Only valid if "value range valid" flag (bit 3) is set.
    cn_val_range_max: f64,
    /// Lower limit for this signal (physical value for numeric conversion rule, otherwise raw value) Only valid if "limit range valid" flag (bit 4) is set.
    cn_limit_min: f64,
    /// Upper limit for this signal (physical value for numeric conversion rule, otherwise raw value) Only valid if "limit range valid" flag (bit 4) is set.
    cn_limit_max: f64,
    /// Lower extended limit for this signal (physical value for numeric conversion rule, otherwise raw value) Only valid if "extended limit range valid" flag (bit 5) is set.
    cn_limit_ext_min: f64,
    /// Upper extended limit for this signal (physical value for numeric conversion rule, otherwise raw value) Only valid if "extended limit range valid" flag (bit 5) is set.
    cn_limit_ext_max: f64,
}

impl Default for Cn4Block {
    fn default() -> Self {
        Cn4Block {
            cn_id: [35, 35, 67, 78], // ##CN
            reserved: [0; 4],
            cn_len: 160,
            cn_links: 8,
            cn_cn_next: 0,
            cn_composition: 0,
            cn_tx_name: 0,
            cn_si_source: 0,
            cn_cc_conversion: 0,
            cn_data: 0,
            cn_md_unit: 0,
            cn_md_comment: 0,
            links: vec![],
            cn_type: 0,
            cn_sync_type: 0,
            cn_data_type: 0,
            cn_bit_offset: 0,
            cn_byte_offset: 0,
            cn_bit_count: 0,
            cn_flags: 0,
            cn_inval_bit_pos: 0,
            cn_precision: 0,
            cn_reserved: [0; 3],
            cn_val_range_min: 0.0,
            cn_val_range_max: 0.0,
            cn_limit_min: 0.0,
            cn_limit_max: 0.0,
            cn_limit_ext_min: 0.0,
            cn_limit_ext_max: 0.0,
        }
    }
}

/// Cn4 structure containing block but also unique_name, ndarray data, composition
/// and other attributes frequently needed and computed
#[derive(Debug, Default, Clone)]
pub struct Cn4 {
    pub block: Cn4Block,
    /// unique channel name string
    pub unique_name: String,
    pub block_position: i64,
    /// beginning position of channel in record
    pub pos_byte_beg: u32,
    /// number of bytes taken by channel in record
    pub n_bytes: u32,
    pub composition: Option<Composition>,
    /// channel data
    pub data: ChannelData,
    /// false = little endian
    pub endian: bool,
    /// optional invalid mask array
    pub invalid_mask: Option<Array1<u8>>,
    /// True if channel is valid = contains data converted
    pub channel_data_valid: bool,
}

/// hashmap's key is bit position in record, value Cn4
pub(crate) type CnType = HashMap<i32, Cn4>;

/// creates recursively in the channel group the CN blocks and all its other linked blocks (CC, MD, TX, CA, etc.)
pub fn parse_cn4(
    rdr: &mut BufReader<&File>,
    target: i64,
    mut position: i64,
    sharable: &mut SharableBlocks,
    record_id_size: u8,
    cg_cycle_count: u64,
) -> (CnType, i64, usize, i32) {
    let mut cn: CnType = HashMap::new();
    let mut n_cn: usize = 0;
    let mut first_rec_pos: i32 = 0;
    if target != 0 {
        let (cn_struct, pos, n_cns, cns) = parse_cn4_block(
            rdr,
            target,
            position,
            sharable,
            record_id_size,
            cg_cycle_count,
        );
        position = pos;
        n_cn += n_cns;
        cn.extend(cns);
        first_rec_pos = (cn_struct.block.cn_byte_offset as i32 + record_id_size as i32) * 8
            + cn_struct.block.cn_bit_offset as i32;
        let mut next_pointer = cn_struct.block.cn_cn_next;
        if cn_struct.block.cn_data_type == 13 {
            // CANopen date
            let (date_ms, min, hour, day, month, year) = can_open_date(
                cn_struct.block_position,
                cn_struct.pos_byte_beg,
                cn_struct.block.cn_byte_offset,
            );
            cn.insert(first_rec_pos, date_ms);
            cn.insert(first_rec_pos + 16, min);
            cn.insert(first_rec_pos + 24, hour);
            cn.insert(first_rec_pos + 32, day);
            cn.insert(first_rec_pos + 40, month);
            cn.insert(first_rec_pos + 48, year);
        } else if cn_struct.block.cn_data_type == 14 {
            // CANopen time
            let (ms, days) = can_open_time(
                cn_struct.block_position,
                cn_struct.pos_byte_beg,
                cn_struct.block.cn_byte_offset,
            );
            cn.insert(first_rec_pos, ms);
            cn.insert(first_rec_pos + 32, days);
        } else {
            if cn_struct.block.cn_type == 3 || cn_struct.block.cn_type == 6 {
                // virtual channel, position in record negative
                first_rec_pos = -1;
                while cn.contains_key(&first_rec_pos) {
                    first_rec_pos -= 1;
                }
            }
            cn.insert(first_rec_pos, cn_struct);
        }

        while next_pointer != 0 {
            let (cn_struct, pos, n_cns, cns) = parse_cn4_block(
                rdr,
                next_pointer,
                position,
                sharable,
                record_id_size,
                cg_cycle_count,
            );
            position = pos;
            n_cn += n_cns;
            cn.extend(cns);
            let mut rec_pos = (cn_struct.block.cn_byte_offset as i32 + record_id_size as i32) * 8
                + cn_struct.block.cn_bit_offset as i32;
            next_pointer = cn_struct.block.cn_cn_next;
            if cn_struct.block.cn_data_type == 13 {
                // CANopen date
                let (date_ms, min, hour, day, month, year) = can_open_date(
                    cn_struct.block_position,
                    cn_struct.pos_byte_beg,
                    cn_struct.block.cn_byte_offset,
                );
                cn.insert(rec_pos, date_ms);
                cn.insert(rec_pos + 16, min);
                cn.insert(rec_pos + 24, hour);
                cn.insert(rec_pos + 32, day);
                cn.insert(rec_pos + 40, month);
                cn.insert(rec_pos + 48, year);
            } else if cn_struct.block.cn_data_type == 14 {
                // CANopen time
                let (ms, days) = can_open_time(
                    cn_struct.block_position,
                    cn_struct.pos_byte_beg,
                    cn_struct.block.cn_byte_offset,
                );
                cn.insert(rec_pos, ms);
                cn.insert(rec_pos + 32, days);
            } else {
                if cn_struct.block.cn_type == 3 || cn_struct.block.cn_type == 6 {
                    // virtual channel, position in record negative
                    rec_pos = -1;
                    while cn.contains_key(&rec_pos) {
                        rec_pos -= 1;
                    }
                }
                cn.insert(rec_pos, cn_struct);
            }
        }
    }
    (cn, position, n_cn, first_rec_pos)
}

/// returns created CANopenDate channels
fn can_open_date(
    block_position: i64,
    pos_byte_beg: u32,
    cn_byte_offset: u32,
) -> (Cn4, Cn4, Cn4, Cn4, Cn4, Cn4) {
    let block = Cn4Block {
        cn_links: 8,
        cn_byte_offset,
        cn_bit_count: 16,
        ..Default::default()
    };
    let date_ms = Cn4 {
        block,
        unique_name: String::from("ms"),
        block_position,
        pos_byte_beg,
        n_bytes: 2,
        composition: None,
        data: ChannelData::UInt16(Array1::<u16>::zeros((0,))),
        endian: false,
        invalid_mask: None,
        channel_data_valid: false,
    };
    let block = Cn4Block {
        cn_links: 8,
        cn_byte_offset: cn_byte_offset + 2,
        cn_bit_count: 6,
        ..Default::default()
    };
    let min = Cn4 {
        block,
        unique_name: String::from("min"),
        block_position,
        pos_byte_beg,
        n_bytes: 1,
        composition: None,
        data: ChannelData::UInt8(Array1::<u8>::zeros((0,))),
        endian: false,
        invalid_mask: None,
        channel_data_valid: false,
    };
    let block = Cn4Block {
        cn_links: 8,
        cn_byte_offset: cn_byte_offset + 3,
        cn_bit_count: 5,
        ..Default::default()
    };
    let hour = Cn4 {
        block,
        unique_name: String::from("hour"),
        block_position,
        pos_byte_beg,
        n_bytes: 1,
        composition: None,
        data: ChannelData::UInt8(Array1::<u8>::zeros((0,))),
        endian: false,
        invalid_mask: None,
        channel_data_valid: false,
    };
    let block = Cn4Block {
        cn_links: 8,
        cn_byte_offset: cn_byte_offset + 4,
        cn_bit_count: 5,
        ..Default::default()
    };
    let day = Cn4 {
        block,
        unique_name: String::from("day"),
        block_position,
        pos_byte_beg,
        n_bytes: 1,
        composition: None,
        data: ChannelData::UInt8(Array1::<u8>::zeros((0,))),
        endian: false,
        invalid_mask: None,
        channel_data_valid: false,
    };
    let block = Cn4Block {
        cn_links: 8,
        cn_byte_offset: cn_byte_offset + 5,
        cn_bit_count: 6,
        ..Default::default()
    };
    let month = Cn4 {
        block,
        unique_name: String::from("month"),
        block_position,
        pos_byte_beg,
        n_bytes: 1,
        composition: None,
        data: ChannelData::UInt8(Array1::<u8>::zeros((0,))),
        endian: false,
        invalid_mask: None,
        channel_data_valid: false,
    };
    let block = Cn4Block {
        cn_links: 8,
        cn_byte_offset: cn_byte_offset + 6,
        cn_bit_count: 7,
        ..Default::default()
    };
    let year = Cn4 {
        block,
        unique_name: String::from("year"),
        block_position,
        pos_byte_beg,
        n_bytes: 1,
        composition: None,
        data: ChannelData::UInt8(Array1::<u8>::zeros((0,))),
        endian: false,
        invalid_mask: None,
        channel_data_valid: false,
    };
    (date_ms, min, hour, day, month, year)
}

/// returns created CANopenTime channels
fn can_open_time(block_position: i64, pos_byte_beg: u32, cn_byte_offset: u32) -> (Cn4, Cn4) {
    let block = Cn4Block {
        cn_links: 8,
        cn_byte_offset,
        cn_bit_count: 28,
        ..Default::default()
    };
    let ms: Cn4 = Cn4 {
        block,
        unique_name: String::from("ms"),
        block_position,
        pos_byte_beg,
        n_bytes: 4,
        composition: None,
        data: ChannelData::UInt32(Array1::<u32>::zeros((0,))),
        endian: false,
        invalid_mask: None,
        channel_data_valid: false,
    };
    let block = Cn4Block {
        cn_links: 8,
        cn_byte_offset: cn_byte_offset + 4,
        cn_bit_count: 16,
        ..Default::default()
    };
    let days: Cn4 = Cn4 {
        block,
        unique_name: String::from("day"),
        block_position,
        pos_byte_beg,
        n_bytes: 2,
        composition: None,
        data: ChannelData::UInt16(Array1::<u16>::zeros((0,))),
        endian: false,
        invalid_mask: None,
        channel_data_valid: false,
    };
    (ms, days)
}

/// Simple calculation to convert bit count into equivalent bytes count
fn calc_n_bytes_not_aligned(bitcount: u32) -> u32 {
    let mut n_bytes = bitcount / 8u32;
    if (bitcount % 8) != 0 {
        n_bytes += 1;
    }
    n_bytes
}

impl Cn4 {
    fn get_cn_source_name(&self, sharable: &SharableBlocks) -> Option<String> {
        let si = sharable.si.get(&self.block.cn_si_source);
        match si {
            Some(block) => block.get_si_source_name(sharable),
            None => None,
        }
    }
    fn get_cn_source_path(&self, sharable: &SharableBlocks) -> Option<String> {
        let si = sharable.si.get(&self.block.cn_si_source);
        match si {
            Some(block) => block.get_si_path_name(sharable),
            None => None,
        }
    }
}

fn parse_cn4_block(
    rdr: &mut BufReader<&File>,
    target: i64,
    mut position: i64,
    sharable: &mut SharableBlocks,
    record_id_size: u8,
    cg_cycle_count: u64,
) -> (Cn4, i64, usize, CnType) {
    let mut n_cn: usize = 1;
    let mut cns: HashMap<i32, Cn4> = HashMap::new();
    rdr.seek_relative(target - position)
        .expect("Could not reach CN block position"); // change buffer position
    let block: Cn4Block = rdr
        .read_le()
        .expect("Could not read buffer into Cn4Block struct");
    position = target + block.cn_len as i64;

    let pos_byte_beg = block.cn_byte_offset + record_id_size as u32;
    let n_bytes = calc_n_bytes_not_aligned(block.cn_bit_count + (block.cn_bit_offset as u32));

    // Reads TX name
    position = read_meta_data(rdr, sharable, block.cn_tx_name, position, BlockType::CN);
    let name: String = if let Some(n) = sharable.get_tx(block.cn_tx_name) {
        n
    } else {
        String::new()
    };

    // Reads unit
    position = read_meta_data(rdr, sharable, block.cn_md_unit, position, BlockType::CN);

    // Reads CC
    let cc_pointer = block.cn_cc_conversion;
    if (cc_pointer != 0) && !sharable.cc.contains_key(&cc_pointer) {
        let (cc_block, _header, pos) = parse_block_short(rdr, cc_pointer, position);
        position = pos;
        position = read_cc(rdr, &cc_pointer, position, cc_block, sharable);
    }

    // Reads MD
    position = read_meta_data(rdr, sharable, block.cn_md_comment, position, BlockType::CN);

    //Reads SI
    let si_pointer = block.cn_si_source;
    if (si_pointer != 0) && !sharable.si.contains_key(&si_pointer) {
        let (mut si_block, _header, pos) = parse_block_short(rdr, si_pointer, position);
        position = pos;
        let si_block: Si4Block = si_block
            .read_le()
            .expect("Could into read buffer into Si4Block struct");
        position = read_meta_data(rdr, sharable, si_block.si_tx_name, position, BlockType::SI);
        position = read_meta_data(rdr, sharable, si_block.si_tx_path, position, BlockType::SI);
        sharable.si.insert(si_pointer, si_block);
    }

    //Reads CA or composition
    let compo: Option<Composition>;
    let is_array: bool;
    if block.cn_composition != 0 {
        let (co, pos, array_flag, n_cns, cnss) = parse_composition(
            rdr,
            block.cn_composition,
            position,
            sharable,
            record_id_size,
            cg_cycle_count,
        );
        is_array = array_flag;
        compo = Some(co);
        position = pos;
        n_cn += n_cns;
        cns = cnss;
    } else {
        compo = None;
        is_array = false;
    }

    let mut endian: bool = false; // Little endian by default
    if block.cn_data_type == 0
        || block.cn_data_type == 2
        || block.cn_data_type == 4
        || block.cn_data_type == 8
        || block.cn_data_type == 15
    {
        endian = false; // little endian
    } else if block.cn_data_type == 1
        || block.cn_data_type == 3
        || block.cn_data_type == 5
        || block.cn_data_type == 9
        || block.cn_data_type == 16
    {
        endian = true; // big endian
    }
    let data_type = block.cn_data_type;
    let cn_type = block.cn_type;

    let cn_struct = Cn4 {
        block,
        unique_name: name,
        block_position: target,
        pos_byte_beg,
        n_bytes,
        composition: compo,
        data: data_type_init(cn_type, data_type, n_bytes, is_array),
        endian,
        invalid_mask: None,
        channel_data_valid: false,
    };

    (cn_struct, position, n_cn, cns)
}

/// reads pointed TX or CC Block(s) pointed by cc_ref in CCBlock
fn read_cc(
    rdr: &mut BufReader<&File>,
    target: &i64,
    mut position: i64,
    mut block: Cursor<Vec<u8>>,
    sharable: &mut SharableBlocks,
) -> i64 {
    let cc_block: Cc4Block = block
        .read_le()
        .expect("Could nto read buffer into Cc4Block struct");
    position = read_meta_data(rdr, sharable, cc_block.cc_md_unit, position, BlockType::CC);
    position = read_meta_data(rdr, sharable, cc_block.cc_tx_name, position, BlockType::CC);

    for pointer in &cc_block.cc_ref {
        if !sharable.cc.contains_key(pointer)
            && !sharable.md_tx.contains_key(pointer)
            && *pointer != 0
        {
            let (ref_block, header, _pos) = parse_block_short(rdr, *pointer, position);
            position = pointer + header.hdr_len as i64;
            if "##TX".as_bytes() == header.hdr_id {
                // TX Block
                position = read_meta_data(rdr, sharable, *pointer, position, BlockType::CC)
            } else {
                // CC Block
                position = read_cc(rdr, pointer, position, ref_block, sharable);
            }
        }
    }
    sharable.cc.insert(*target, cc_block);
    position
}

/// Cc4 Channel Conversion block struct
#[derive(Debug, Clone)]
#[binrw]
#[br(little)]
#[allow(dead_code)]
pub struct Cc4Block {
    // cc_id: [u8; 4],  // ##CC
    // reserved: [u8; 4],  // reserved
    // cc_len: u64,      // Length of block in bytes
    /// # of links
    cc_links: u64,
    /// Link to TXBLOCK with name (identifier) of conversion (can be NIL). Name must be according to naming rules stated in 4.4.2 Naming Rules.
    pub cc_tx_name: i64,
    /// Link to TXBLOCK/MDBLOCK with physical unit of signal data (after conversion). (can be NIL) Unit only applies if no unit defined in CNBLOCK. Otherwise the unit of the channel overwrites the conversion unit.
    cc_md_unit: i64,
    // An MDBLOCK can be used to additionally reference the A-HDO unit definition. Note: for channels with cn_sync_type > 0, the unit is already defined, thus a reference to an A-HDO definition should be omitted to avoid redundancy.
    /// Link to TXBLOCK/MDBLOCK with comment of conversion and additional information. (can be NIL)
    pub cc_md_comment: i64,
    /// Link to CCBLOCK for inverse formula (can be NIL, must be NIL for CCBLOCK of the inverse formula (no cyclic reference allowed).
    cc_cc_inverse: i64,
    #[br(if(cc_links > 4), little, count = cc_links - 4)]
    /// List of additional links to TXBLOCKs with strings or to CCBLOCKs with partial conversion rules. Length of list is given by cc_ref_count. The list can be empty. Details are explained in formula-specific block supplement.
    pub cc_ref: Vec<i64>,

    // Data Members
    /// Conversion type (formula identifier) (see CC_T_xxx)
    pub cc_type: u8,
    /// Precision for display of floating point values. 0xFF means unrestricted precision (infinite) Any other value specifies the number of decimal places to use for display of floating point values. Note: only valid if "precision valid" flag (bit 0) is set and if cn_precision of the parent CNBLOCK is invalid, otherwise cn_precision must be used.     
    cc_precision: u8,
    /// Flags  (see CC_F_xxx)
    cc_flags: u16,
    /// Length M of cc_ref list with additional links. See formula-specific block supplement for meaning of the links.
    cc_ref_count: u16,
    /// Length N of cc_val list with additional parameters. See formula-specific block supplement for meaning of the parameters.
    cc_val_count: u16,
    /// Minimum physical signal value that occurred for this signal. Only valid if "physical value range valid" flag (bit 1) is set.
    cc_phy_range_min: f64,
    /// Maximum physical signal value that occurred for this signal. Only valid if "physical value range valid" flag (bit 1) is set.
    cc_phy_range_max: f64,
    #[br(args(cc_val_count, cc_type))]
    pub cc_val: CcVal,
}

#[derive(Debug, Clone)]
#[binrw]
#[br(little, import(count: u16, cc_type: u8))]
pub enum CcVal {
    #[br(pre_assert(cc_type < 11))]
    Real(#[br(count = count)] Vec<f64>),

    #[br(pre_assert(cc_type == 11))]
    Uint(#[br(count = count)] Vec<u64>),
}

/// Si4 Source Information block struct
#[derive(Debug, PartialEq, Default, Copy, Clone)]
#[binrw]
#[br(little)]
pub struct Si4Block {
    // si_id: [u8; 4],  // ##SI
    // reserved: [u8; 4],  // reserved
    // si_len: u64,      // Length of block in bytes
    /// # of links
    si_links: u64,
    /// Pointer to TXBLOCK with name (identification) of source (must not be NIL). The source name must be according to naming rules stated in 4.4.2 Naming Rules.
    si_tx_name: i64,
    /// Pointer to TXBLOCK with (tool-specific) path of source (can be NIL). The path string must be according to naming rules stated in 4.4.2 Naming Rules.
    si_tx_path: i64,
    // Each tool may generate a different path string. The only purpose is to ensure uniqueness as explained in section 4.4.3 Identification of Channels. As a recommendation, the path should be a human readable string containing additional information about the source. However, the path string should not be used to store this information in order to retrieve it later by parsing the string. Instead, additional source information should be stored in generic or custom XML fields in the comment MDBLOCK si_md_comment.
    /// Pointer to source comment and additional information (TXBLOCK or MDBLOCK) (can be NIL)
    si_md_comment: i64,

    // Data Members
    /// Source type additional classification of source (see SI_T_xxx)
    si_type: u8,
    /// Bus type additional classification of used bus (should be 0 for si_type ≥ 3) (see SI_BUS_xxx)
    si_bus_type: u8,
    /// Flags The value contains the following bit flags (see SI_F_xxx)):
    si_flags: u8,
    /// reserved
    si_reserved: [u8; 5],
}

impl Si4Block {
    fn get_si_source_name(&self, sharable: &SharableBlocks) -> Option<String> {
        sharable.get_tx(self.si_tx_name)
    }
    fn get_si_path_name(&self, sharable: &SharableBlocks) -> Option<String> {
        sharable.get_tx(self.si_tx_path)
    }
}

/// Ca4 Channel Array block struct
#[derive(Debug, PartialEq, Clone)]
pub struct Ca4Block {
    // header
    /// ##CA
    pub ca_id: [u8; 4],
    /// reserved
    reserved: [u8; 4],
    /// Length of block in bytes
    pub ca_len: u64,
    /// # of links
    ca_links: u64,
    // links
    /// [] Array of composed elements: Pointer to a CNBLOCK for array of structures, or to a CABLOCK for array of arrays (can be NIL). If a CABLOCK is referenced, it must use the "CN template" storage type (ca_storage = 0).
    pub ca_composition: i64,
    /// [Π N(d) or empty] Only present for storage type "DG template". List of links to data blocks (DTBLOCK/DLBLOCK) for each element in case of "DG template" storage (ca_storage = 2). A link in this list may only be NIL if the cycle count of the respective element is 0: ca_data[k] = NIL => ca_cycle_count[k] = 0 The links are stored line-oriented, i.e. element k uses ca_data[k] (see explanation below). The size of the list must be equal to Π N(d), i.e. to the product of the number of elements per dimension N(d) over all dimensions D. Note: link ca_data[0] must be equal to dg_data link of the parent DGBLOCK.
    pub ca_data: Option<Vec<i64>>,
    /// [Dx3 or empty] Only present if "dynamic size" flag (bit 0) is set. References to channels for size signal of each dimension (can be NIL). Each reference is a link triple with pointer to parent DGBLOCK, parent CGBLOCK and CNBLOCK for the channel (either all three links are assigned or NIL). Thus the links have the following order: DGBLOCK for size signal of dimension 1 CGBLOCK for size signal of dimension 1 CNBLOCK for size signal of dimension 1 … DGBLOCK for size signal of dimension D CGBLOCK for size signal of dimension D CNBLOCK for size signal of dimension D The size signal can be used to model arrays whose number of elements per dimension can vary over time. If a size signal is specified for a dimension, the number of elements for this dimension at some point in time is equal to the value of the size signal at this time (i.e. for time-synchronized signals, the size signal value with highest time stamp less or equal to current time stamp). If the size signal has no recorded signal value for this time (yet), assume 0 as size.
    ca_dynamic_size: Option<Vec<i64>>,
    /// [Dx3 or empty] Only present if "input quantity" flag (bit 1) is set. Reference to channels for input quantity signal for each dimension (can be NIL). Each reference is a link triple with pointer to parent DGBLOCK, parent CGBLOCK and CNBLOCK for the channel (either all three links are assigned or NIL). Thus the links have the following order: DGBLOCK for input quantity of dimension 1 CGBLOCK for input quantity of dimension 1 CNBLOCK for input quantity of dimension 1 … DGBLOCK for input quantity of dimension D CGBLOCK for input quantity of dimension D CNBLOCK for input quantity of dimension D Since the input quantity signal and the array signal must be synchronized, their channel groups must contain at least one common master channel type.
    ca_input_quantity: Option<Vec<i64>>,
    /// [3 or empty] Only present if "output quantity" flag (bit 2) is set. Reference to channel for output quantity (can be NIL). The reference is a link triple with pointer to parent DGBLOCK, parent CGBLOCK and CNBLOCK for the channel (either all three links are assigned or NIL). Since the output quantity signal and the array signal must be synchronized, their channel groups must contain at least one common master channel type. For array type "look-up", the output quantity is the result of the complete look-up (see [MCD-2 MC] keyword RIP_ADDR_W). The output quantity should have the same physical unit as the array elements of the array that references it.
    ca_output_quantity: Option<Vec<i64>>,
    /// [3 or empty] Only present if "comparison quantity" flag (bit 3) is set. Reference to channel for comparison quantity (can be NIL). The reference is a link triple with pointer to parent DGBLOCK, parent CGBLOCK and CNBLOCK for the channel (either all three links are assigned or NIL). Since the comparison quantity signal and the array signal must be synchronized, their channel groups must contain at least one common master channel type. The comparison quantity should have the same physical unit as the array elements.
    ca_comparison_quantity: Option<Vec<i64>>,
    /// [D or empty] Only present if "axis" flag (bit 4) is set. Pointer to a conversion rule (CCBLOCK) for the scaling axis of each dimension. If a link NIL a 1:1 conversion must be used for this axis. If the "fixed axis" flag (Bit 5) is set, the conversion must be applied to the fixed axis values of the respective axis/dimension (ca_axis_value list stores the raw values as REAL). If the link to the CCBLOCK is NIL already the physical values are stored in the ca_axis_value list. If the "fixed axes" flag (Bit 5) is not set, the conversion must be applied to the raw values of the respective axis channel, i.e. it overrules the conversion specified for the axis channel, even if the ca_axis_conversion link is NIL! Note: ca_axis_conversion may reference the same CCBLOCK as referenced by the respective axis channel ("sharing" of CCBLOCK).
    ca_cc_axis_conversion: Option<Vec<i64>>,
    /// [Dx3 or empty] Only present if "axis" flag (bit 4) is set and "fixed axes flag" (bit 5) is not set. References to channels for scaling axis of respective dimension (can be NIL). Each reference is a link triple with pointer to parent DGBLOCK, parent CGBLOCK and CNBLOCK for the channel (either all three links are assigned or NIL). Thus the links have the following order: DGBLOCK for axis of dimension 1 CGBLOCK for axis of dimension 1 CNBLOCK for axis of dimension 1 … DGBLOCK for axis of dimension D CGBLOCK for axis of dimension D CNBLOCK for axis of dimension D Each referenced channel must be an array of type "axis". The maximum number of elements of each axis (ca_dim_size[0] in axis) must be equal to the maximum number of elements of respective dimension d in "look-up" array (ca_dim_size[d-1]).
    ca_axis: Option<Vec<i64>>,
    //members
    /// Array type (defines semantic of the array) see CA_T_xxx
    pub ca_type: u8,
    /// Storage type (defines how the element values are stored) see CA_S_xxx
    pub ca_storage: u8,
    /// Number of dimensions D > 0 For array type "axis", D must be 1.
    pub ca_ndim: u16,
    /// Flags The value contains the following bit flags (Bit 0 = LSB): see CA_F_xxx
    pub ca_flags: u32,
    /// Base factor for calculation of Byte offsets for "CN template" storage type. ca_byte_offset_base should be larger than or equal to the size of Bytes required to store a component channel value in the record (all must have the same size). If it is equal to this value, then the component values are stored next to each other without gaps. Exact formula for calculation of Byte offset for each component channel see below.
    pub ca_byte_offset_base: i32,
    /// Base factor for calculation of invalidation bit positions for CN template storage type.
    pub ca_inval_bit_pos_base: u32,
    pub ca_dim_size: Vec<u64>,
    pub ca_axis_value: Option<Vec<f64>>,
    pub ca_cycle_count: Option<Vec<u64>>,
    pub snd: usize,
    pub pnd: usize,
    pub shape: (Vec<usize>, Order),
}

impl Default for Ca4Block {
    fn default() -> Self {
        Self {
            ca_id: [35, 35, 67, 65], // ##CA
            reserved: [0u8; 4],
            ca_len: 48,
            ca_links: 1,
            ca_composition: 0,
            ca_data: None,
            ca_dynamic_size: None,
            ca_input_quantity: None,
            ca_output_quantity: None,
            ca_comparison_quantity: None,
            ca_cc_axis_conversion: None,
            ca_axis: None,
            ca_type: 0,    // Array
            ca_storage: 0, // CN template
            ca_ndim: 1,
            ca_flags: 0,
            ca_byte_offset_base: 0,   // first
            ca_inval_bit_pos_base: 0, // present in DIBlock
            ca_dim_size: vec![],
            ca_axis_value: None,
            ca_cycle_count: None,
            snd: 0,
            pnd: 1,
            shape: (vec![], Order::RowMajor),
        }
    }
}

#[derive(Debug, Clone)]
#[binrw]
#[br(little)]
pub struct Ca4BlockMembers {
    /// Array type (defines semantic of the array) see CA_T_xxx
    ca_type: u8,
    /// Storage type (defines how the element values are stored) see CA_S_xxx            
    ca_storage: u8,
    /// Number of dimensions D > 0 For array type "axis", D must be 1.
    pub ca_ndim: u16,
    /// Flags The value contains the following bit flags (Bit 0 = LSB): see CA_F_xxx
    ca_flags: u32,
    /// Base factor for calculation of Byte offsets for "CN template" storage type. ca_byte_offset_base should be larger than or equal to the size of Bytes required to store a component channel value in the record (all must have the same size). If it is equal to this value, then the component values are stored next to each other without gaps. Exact formula for calculation of Byte offset for each component channel see below.
    ca_byte_offset_base: i32,
    /// Base factor for calculation of invalidation bit positions for CN template storage type.
    ca_inval_bit_pos_base: u32,
    #[br(if(ca_ndim > 0), little, count = ca_ndim)]
    pub ca_dim_size: Vec<u64>,
}

impl Default for Ca4BlockMembers {
    fn default() -> Self {
        Self {
            ca_type: 0,
            ca_storage: 0,
            ca_ndim: 1,
            ca_flags: 0,
            ca_byte_offset_base: 0,
            ca_inval_bit_pos_base: 0,
            ca_dim_size: vec![],
        }
    }
}

fn parse_ca_block(
    ca_block: &mut Cursor<Vec<u8>>,
    block_header: Blockheader4,
    cg_cycle_count: u64,
) -> Ca4Block {
    //Reads members first
    ca_block.set_position(block_header.hdr_links * 8); // change buffer position after links section
    let ca_members: Ca4BlockMembers = ca_block
        .read_le()
        .expect("Coudl tno read buffer into CaBlockMembers struct");
    let mut snd: usize;
    let mut pnd: usize;
    // converts  ca_dim_size from u64 to usize
    let shape_dim_usize: Vec<usize> = ca_members.ca_dim_size.iter().map(|&d| d as usize).collect();
    if shape_dim_usize.len() == 1 {
        snd = shape_dim_usize[0];
        pnd = shape_dim_usize[0];
    } else {
        snd = 0;
        pnd = 1;
        let sizes = shape_dim_usize.clone();
        for x in sizes.into_iter() {
            snd += x;
            pnd *= x;
        }
    }
    let mut shape_dim: VecDeque<usize> = VecDeque::from(shape_dim_usize);
    shape_dim.push_front(cg_cycle_count as usize);

    let shape: (Vec<usize>, Order) = if (ca_members.ca_flags >> 6 & 1) != 0 {
        (shape_dim.into(), Order::ColumnMajor)
    } else {
        (shape_dim.into(), Order::RowMajor)
    };

    let mut val = vec![0.0f64; snd as usize];
    let ca_axis_value: Option<Vec<f64>> = if (ca_members.ca_flags & 0b100000) > 0 {
        ca_block
            .read_f64_into::<LittleEndian>(&mut val)
            .expect("Could not read ca_axis_value");
        Some(val)
    } else {
        None
    };

    let mut val = vec![0u64; pnd];
    let ca_cycle_count: Option<Vec<u64>> = if ca_members.ca_storage >= 1 {
        ca_block
            .read_u64_into::<LittleEndian>(&mut val)
            .expect("Could not read ca_cycle_count");
        Some(val)
    } else {
        None
    };

    // Reads links
    ca_block.set_position(0); // change buffer position to beginning of links section

    let ca_composition: i64 = ca_block
        .read_i64::<LittleEndian>()
        .expect("Could not read ca_composition");

    let mut val = vec![0i64; pnd];
    let ca_data: Option<Vec<i64>> = if ca_members.ca_storage == 2 {
        ca_block
            .read_i64_into::<LittleEndian>(&mut val)
            .expect("Could not read ca_storage");
        Some(val)
    } else {
        None
    };

    let mut val = vec![0i64; (ca_members.ca_ndim * 3) as usize];
    let ca_dynamic_size: Option<Vec<i64>> = if (ca_members.ca_flags & 0b1) > 0 {
        ca_block
            .read_i64_into::<LittleEndian>(&mut val)
            .expect("Could not read ca_dynamic_size");
        Some(val)
    } else {
        None
    };

    let mut val = vec![0i64; (ca_members.ca_ndim * 3) as usize];
    let ca_input_quantity: Option<Vec<i64>> = if (ca_members.ca_flags & 0b10) > 0 {
        ca_block
            .read_i64_into::<LittleEndian>(&mut val)
            .expect("Could not read ca_input_quantity");
        Some(val)
    } else {
        None
    };

    let mut val = vec![0i64; 3];
    let ca_output_quantity: Option<Vec<i64>> = if (ca_members.ca_flags & 0b100) > 0 {
        ca_block
            .read_i64_into::<LittleEndian>(&mut val)
            .expect("Could not read ca_output_quantity");
        Some(val)
    } else {
        None
    };

    let mut val = vec![0i64; 3];
    let ca_comparison_quantity: Option<Vec<i64>> = if (ca_members.ca_flags & 0b1000) > 0 {
        ca_block
            .read_i64_into::<LittleEndian>(&mut val)
            .expect("Could not read ca_comparison_quantity");
        Some(val)
    } else {
        None
    };

    let mut val = vec![0i64; ca_members.ca_ndim as usize];
    let ca_cc_axis_conversion: Option<Vec<i64>> = if (ca_members.ca_flags & 0b10000) > 0 {
        ca_block
            .read_i64_into::<LittleEndian>(&mut val)
            .expect("Could not read ca_cc_axis_conversion");
        Some(val)
    } else {
        None
    };

    let mut val = vec![0i64; (ca_members.ca_ndim * 3) as usize];
    let ca_axis: Option<Vec<i64>> =
        if ((ca_members.ca_flags & 0b10000) > 0) & ((ca_members.ca_flags & 0b100000) > 0) {
            ca_block
                .read_i64_into::<LittleEndian>(&mut val)
                .expect("Could not read ca_axis");
            Some(val)
        } else {
            None
        };

    Ca4Block {
        ca_id: block_header.hdr_id,
        reserved: block_header.hdr_gap,
        ca_len: block_header.hdr_len,
        ca_links: block_header.hdr_links,
        ca_composition,
        ca_data,
        ca_dynamic_size,
        ca_input_quantity,
        ca_output_quantity,
        ca_comparison_quantity,
        ca_cc_axis_conversion,
        ca_axis,
        ca_type: ca_members.ca_type,
        ca_storage: ca_members.ca_storage,
        ca_ndim: ca_members.ca_ndim,
        ca_flags: ca_members.ca_flags,
        ca_byte_offset_base: ca_members.ca_byte_offset_base,
        ca_inval_bit_pos_base: ca_members.ca_inval_bit_pos_base,
        ca_dim_size: ca_members.ca_dim_size,
        ca_axis_value,
        ca_cycle_count,
        snd,
        pnd,
        shape,
    }
}

/// contains composition blocks (CN or CA)
/// can optionaly point to another composition
#[derive(Debug, Clone)]
pub struct Composition {
    pub block: Compo,
    pub compo: Option<Box<Composition>>,
}

/// enum allowing to nest CA or CN blocks for a compostion
#[derive(Debug, Clone)]
pub enum Compo {
    CA(Box<Ca4Block>),
    CN(Box<Cn4>),
}

/// parses CN (structure) of CA (Array) blocks
/// CN (structures of composed channels )and CA (array of arrays) blocks can be nested or vene CA and CN nested and mixed: this is not supported, very complicated
fn parse_composition(
    rdr: &mut BufReader<&File>,
    target: i64,
    mut position: i64,
    sharable: &mut SharableBlocks,
    record_id_size: u8,
    cg_cycle_count: u64,
) -> (Composition, i64, bool, usize, CnType) {
    let (mut block, block_header, pos) = parse_block(rdr, target, position);
    position = pos;
    let is_array: bool;
    let mut cns: CnType;
    let mut n_cn: usize = 0;

    if block_header.hdr_id == "##CA".as_bytes() {
        // Channel Array
        is_array = true;
        let block = parse_ca_block(&mut block, block_header, cg_cycle_count);
        position = pos;
        let ca_compositon: Option<Box<Composition>>;
        if block.ca_composition != 0 {
            let (ca, pos, _is_array, n_cns, cnss) = parse_composition(
                rdr,
                block.ca_composition,
                position,
                sharable,
                record_id_size,
                cg_cycle_count,
            );
            position = pos;
            cns = cnss;
            n_cn += n_cns;
            ca_compositon = Some(Box::new(ca));
        } else {
            ca_compositon = None;
            cns = HashMap::new();
        }
        (
            Composition {
                block: Compo::CA(Box::new(block)),
                compo: ca_compositon,
            },
            position,
            is_array,
            n_cn,
            cns,
        )
    } else {
        // Channel structure
        is_array = false;
        let (cnss, pos, n_cns, first_rec_pos) = parse_cn4(
            rdr,
            target,
            position,
            sharable,
            record_id_size,
            cg_cycle_count,
        );
        position = pos;
        n_cn += n_cns;
        cns = cnss;
        let cn_composition: Option<Box<Composition>>;
        let cn_struct: Cn4 = if let Some(cn) = cns.get(&first_rec_pos) {
            cn.clone()
        } else {
            Cn4::default()
        };
        if cn_struct.block.cn_composition != 0 {
            let (cn, pos, _is_array, n_cns, cnss) = parse_composition(
                rdr,
                cn_struct.block.cn_composition,
                position,
                sharable,
                record_id_size,
                cg_cycle_count,
            );
            position = pos;
            n_cn += n_cns;
            cns.extend(cnss);
            cn_composition = Some(Box::new(cn));
        } else {
            cn_composition = None
        }
        (
            Composition {
                block: Compo::CN(Box::new(cn_struct)),
                compo: cn_composition,
            },
            position,
            is_array,
            n_cn,
            cns,
        )
    }
}

/// parses mdfinfo structure to make channel names unique
/// creates channel names set and links master channels to set of channels
pub fn build_channel_db(
    dg: &mut BTreeMap<i64, Dg4>,
    sharable: &SharableBlocks,
    n_cg: usize,
    n_cn: usize,
) -> ChannelNamesSet {
    let mut channel_list: ChannelNamesSet = HashMap::with_capacity(n_cn);
    let mut master_channel_list: HashMap<i64, String> = HashMap::with_capacity(n_cg);
    // creating channel list for whole file and making channel names unique
    for (dg_position, dg) in dg.iter_mut() {
        for (record_id, cg) in dg.cg.iter_mut() {
            let gn = cg.get_cg_name(sharable);
            let gs = cg.get_cg_source_name(sharable);
            let gp = cg.get_cg_source_path(sharable);
            for (cn_record_position, cn) in cg.cn.iter_mut() {
                if channel_list.contains_key(&cn.unique_name) {
                    let mut changed: bool = false;
                    let space_char = String::from(" ");
                    // create unique channel name
                    if let Some(cs) = cn.get_cn_source_name(sharable) {
                        cn.unique_name.push_str(&space_char);
                        cn.unique_name.push_str(&cs);
                        changed = true;
                    }
                    if let Some(cp) = cn.get_cn_source_path(sharable) {
                        cn.unique_name.push_str(&space_char);
                        cn.unique_name.push_str(&cp);
                        changed = true;
                    }
                    if let Some(name) = &gn {
                        cn.unique_name.push_str(&space_char);
                        cn.unique_name.push_str(name);
                        changed = true;
                    }
                    if let Some(source) = &gs {
                        cn.unique_name.push_str(&space_char);
                        cn.unique_name.push_str(source);
                        changed = true;
                    }
                    if let Some(path) = &gp {
                        cn.unique_name.push_str(&space_char);
                        cn.unique_name.push_str(path);
                        changed = true;
                    }
                    // No souce or path name to make channel unique
                    if !changed {
                        // extend name with channel block position, unique
                        cn.unique_name.push_str(&space_char);
                        cn.unique_name.push_str(&cn.block_position.to_string());
                    }
                };
                channel_list.insert(
                    cn.unique_name.clone(),
                    (
                        None, // computes at second step master channel because of cg_cg_master
                        *dg_position,
                        (cg.block_position, *record_id),
                        (cn.block_position, *cn_record_position),
                    ),
                );
                if cn.block.cn_type == 2 || cn.block.cn_type == 3 {
                    // Master channel
                    master_channel_list.insert(cg.block_position, cn.unique_name.clone());
                }
            }
        }
    }
    // identifying master channels
    let avg_ncn_per_cg = n_cn / n_cg;
    for (_dg_position, dg) in dg.iter_mut() {
        for (_record_id, cg) in dg.cg.iter_mut() {
            let mut cg_channel_list: HashSet<String> = HashSet::with_capacity(avg_ncn_per_cg);
            let mut master_channel_name: Option<String> = None;
            if let Some(name) = master_channel_list.get(&cg.block_position) {
                master_channel_name = Some(name.to_string());
            } else if let Some(cg_cg_master) = cg.block.cg_cg_master {
                // master is in another cg block, possible from 4.2
                if let Some(name) = master_channel_list.get(&cg_cg_master) {
                    master_channel_name = Some(name.to_string());
                }
            }
            for (_cn_record_position, cn) in cg.cn.iter_mut() {
                cg_channel_list.insert(cn.unique_name.clone());
                // assigns master in channel_list
                if let Some(id) = channel_list.get_mut(&cn.unique_name) {
                    id.0 = master_channel_name.clone();
                }
            }
            cg.channel_names = cg_channel_list;
            cg.master_channel_name = master_channel_name;
        }
    }
    channel_list
}

/// DT4 Data List block struct
#[derive(Debug, PartialEq, Default, Clone)]
#[binrw]
#[br(little)]
pub struct Dt4Block {
    //header
    // dl_id: [u8; 4],  // ##DL
    /// reserved
    reserved: [u8; 4],
    /// Length of block in bytes
    pub len: u64,
    /// # of links
    links: u64,
}

/// DL4 Data List block struct
#[derive(Debug, PartialEq, Default, Clone)]
#[binrw]
#[br(little)]
pub struct Dl4Block {
    //header
    // dl_id: [u8; 4],  // ##DL
    /// reserved
    reserved: [u8; 4],
    /// Length of block in bytes
    dl_len: u64,
    /// # of links
    dl_links: u64,
    // links
    /// next DL
    pub dl_dl_next: i64,
    #[br(if(dl_links > 1), little, count = dl_links - 1)]
    pub dl_data: Vec<i64>,
    // members
    dl_flags: u8,
    dl_reserved: [u8; 3],
    /// Number of data blocks
    dl_count: u32,
    #[br(if((dl_flags & 0b1)>0), little)]
    dl_equal_length: Option<u64>,
    #[br(if((dl_flags & 0b1)==0), little, count = dl_count)]
    dl_offset: Vec<u64>,
    #[br(if((dl_flags & 0b10)>0), little, count = dl_count)]
    dl_time_values: Vec<i64>,
    #[br(if((dl_flags & 0b100)>0), little, count = dl_count)]
    dl_angle_values: Vec<i64>,
    #[br(if((dl_flags & 0b1000)>0), little, count = dl_count)]
    dl_distance_values: Vec<i64>,
}

/// parses Data List block
/// pointing to DT, SD, RD or DZ blocks
pub fn parser_dl4_block(
    rdr: &mut BufReader<&File>,
    target: i64,
    mut position: i64,
) -> (Dl4Block, i64) {
    rdr.seek_relative(target - position)
        .expect("Could not reach position to read Dl4Block");
    let block: Dl4Block = rdr.read_le().expect("Could not read into Dl4Block struct");
    position = target + block.dl_len as i64;
    (block, position)
}

/// parses DZBlock
pub fn parse_dz(rdr: &mut BufReader<&File>) -> (Vec<u8>, Dz4Block) {
    let block: Dz4Block = rdr.read_le().expect("Could not read into Dz4Block struct");
    let mut buf = vec![0u8; block.dz_data_length as usize];
    rdr.read_exact(&mut buf).expect("Could not read Dz data");
    let (mut data, checksum) = decompress(&buf, Format::Zlib).expect("Could not decompress data");
    if Adler32::from_buf(&data).finish() != checksum.expect("dz block checksum") {
        panic!("Checksum not ok");
    }
    if block.dz_zip_type == 1 {
        let m = block.dz_org_data_length / block.dz_zip_parameter as u64;
        let tail: Vec<u8> = data.split_off((m * block.dz_zip_parameter as u64) as usize);
        let mut output = vec![0u8; (m * block.dz_zip_parameter as u64) as usize];
        transpose::transpose(
            &data,
            &mut output,
            m as usize,
            block.dz_zip_parameter as usize,
        );
        data = output;
        if !tail.is_empty() {
            data.extend(tail);
        }
    }
    (data, block)
}

/// DZ4 Data List block struct
#[derive(Debug, PartialEq, Clone)]
#[binrw]
#[br(little)]
pub struct Dz4Block {
    //header
    // dz_id: [u8; 4],  // ##DZ
    reserved: [u8; 4], // reserved
    pub len: u64,      // Length of block in bytes
    dz_links: u64,     // # of links
    // links
    // members
    /// "DT", "SD", "RD" or "DV", "DI", "RV", "RI"
    pub dz_org_block_type: [u8; 2],
    /// Zip algorithm, 0 deflate, 1 transpose + deflate
    dz_zip_type: u8,
    /// reserved
    dz_reserved: u8,
    dz_zip_parameter: u32, //
    /// length of uncompressed data
    pub dz_org_data_length: u64,
    /// length of compressed data
    pub dz_data_length: u64,
}

impl Default for Dz4Block {
    fn default() -> Self {
        Dz4Block {
            reserved: [0; 4],
            len: 0,
            dz_links: 0,
            dz_org_block_type: [68, 86], // DV
            dz_zip_type: 0,              // No transposition for a single channel
            dz_reserved: 0,
            dz_zip_parameter: 0,
            dz_org_data_length: 0,
            dz_data_length: 0,
        }
    }
}

/// DL4 Data List block struct
#[derive(Debug, PartialEq, Clone)]
#[binrw]
#[br(little)]
pub struct Ld4Block {
    // header
    // ld_id: [u8; 4],  // ##LD
    reserved: [u8; 4],   // reserved
    pub ld_len: u64,     // Length of block in bytes
    pub ld_n_links: u64, // # of links
    // links
    #[br(little, count = ld_n_links)]
    pub ld_links: Vec<i64>,
    //members
    pub ld_flags: u32,
    /// Number of data blocks
    pub ld_count: u32,
    #[br(if((ld_flags & 0b1)!=0), little)]
    pub ld_equal_sample_count: Option<u64>,
    #[br(if((ld_flags & 0b1)==0), little, count = ld_count)]
    pub ld_sample_offset: Vec<u64>,
    #[br(if((ld_flags & 0b10)>0), little, count = ld_count)]
    dl_time_values: Vec<i64>,
    #[br(if((ld_flags & 0b100)>0), little, count = ld_count)]
    dl_angle_values: Vec<i64>,
    #[br(if((ld_flags & 0b1000)>0), little, count = ld_count)]
    dl_distance_values: Vec<i64>,
}

impl Default for Ld4Block {
    fn default() -> Self {
        Ld4Block {
            reserved: [0; 4],
            ld_len: 56,
            ld_n_links: 2,
            ld_links: vec![0],
            ld_flags: 0,
            ld_count: 0,
            ld_equal_sample_count: None,
            ld_sample_offset: vec![],
            dl_time_values: vec![],
            dl_angle_values: vec![],
            dl_distance_values: vec![],
        }
    }
}

impl Ld4Block {
    pub fn ld_ld_next(&self) -> i64 {
        self.ld_links[0]
    }
    pub fn ld_data(&self) -> Vec<i64> {
        if (1u32 << 31) & self.ld_flags > 0 {
            self.ld_links.iter().skip(1).step_by(2).copied().collect()
        } else {
            self.ld_links[1..].to_vec()
        }
    }
    pub fn ld_invalid_data(&self) -> Vec<i64> {
        if (1u32 << 31) & self.ld_flags > 0 {
            self.ld_links.iter().skip(2).step_by(2).copied().collect()
        } else {
            Vec::<i64>::new()
        }
    }
}

/// parse List Data block
/// equivalent ot DLBlock but unsorted data is not allowed
/// pointing to DV/DI and RV/RI blocks
pub fn parser_ld4_block(
    rdr: &mut BufReader<&File>,
    target: i64,
    mut position: i64,
) -> (Ld4Block, i64) {
    rdr.seek_relative(target - position)
        .expect("Could not reach Ld4Block position");
    let block: Ld4Block = rdr
        .read_le()
        .expect("Could not read buffer into Ld4Block struct");
    position = target + block.ld_len as i64;
    (block, position)
}

/// HL4 Data List block struct
#[derive(Debug, PartialEq, Default, Clone)]
#[binrw]
#[br(little)]
pub struct Hl4Block {
    //header
    // ##HL
    // hl_id: [u8; 4],
    /// reserved
    reserved: [u8; 4],
    /// Length of block in bytes
    pub hl_len: u64,
    /// # of links
    hl_links: u64,
    /// links
    pub hl_dl_first: i64, // first LD block
    // members
    /// flags
    hl_flags: u16,
    /// Zip algorithn
    hl_zip_type: u8,
    /// reserved
    hl_reserved: [u8; 5],
}
