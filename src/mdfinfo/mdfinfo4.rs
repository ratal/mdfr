//! Parsing of file metadata into MdfInfo4 struct
use crate::mdfreader::{DataSignature, MasterSignature};
use anyhow::{anyhow, Context, Error, Result};
use arrow::array::{Array, BooleanBufferBuilder, UInt16Builder, UInt32Builder, UInt8Builder};
use binrw::{binrw, BinReaderExt, BinWriterExt};
use byteorder::{LittleEndian, ReadBytesExt};
use chrono::{DateTime, Local};
use log::warn;
use md5::{Digest, Md5};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::default::Default;
use std::fmt::Debug;
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, Write};
use std::sync::Arc;
use std::{fmt, str};
use yazi::{decompress, Adler32, Format};

use crate::data_holder::channel_data::{data_type_init, try_from, ChannelData};
use crate::data_holder::tensor_arrow::Order;
use crate::mdfinfo::IdBlock;

use super::sym_buf_reader::SymBufReader;

/// ChannelId : (Option<master_channelname>, dg_pos, (cg_pos, rec_id), (cn_pos, rec_pos))
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
#[derive(Debug, Default, Clone)]
#[repr(C)]
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
    /// Returns the channel's vector data if present in memory, otherwise None.
    pub fn get_channel_data(&self, channel_name: &str) -> Option<&ChannelData> {
        let mut data: Option<&ChannelData> = None;
        if let Some((_master, dg_pos, (_cg_pos, rec_id), (_cn_pos, rec_pos))) =
            self.get_channel_id(channel_name)
        {
            if let Some(dg) = self.dg.get(dg_pos) {
                if let Some(cg) = dg.cg.get(rec_id) {
                    if let Some(cn) = cg.cn.get(rec_pos) {
                        if !cn.data.is_empty() {
                            data = Some(&cn.data);
                        }
                    }
                }
            }
        }
        data
    }
    /// Returns the channel's unit string. If it does not exist, it is an empty string.
    pub fn get_channel_unit(&self, channel_name: &str) -> Result<Option<String>> {
        let mut unit: Option<String> = None;
        if let Some((_master, dg_pos, (_cg_pos, rec_id), (_cn_pos, rec_pos))) =
            self.get_channel_id(channel_name)
        {
            if let Some(dg) = self.dg.get(dg_pos) {
                if let Some(cg) = dg.cg.get(rec_id) {
                    if let Some(cn) = cg.cn.get(rec_pos) {
                        unit = self.sharable.get_tx(cn.block.cn_md_unit)?;
                    }
                }
            }
        }
        Ok(unit)
    }
    /// Returns the channel's description. If it does not exist, it is an empty string
    pub fn get_channel_desc(&self, channel_name: &str) -> Result<Option<String>> {
        let mut desc: Option<String> = None;
        if let Some((_master, dg_pos, (_cg_pos, rec_id), (_cn_pos, rec_pos))) =
            self.get_channel_id(channel_name)
        {
            if let Some(dg) = self.dg.get(dg_pos) {
                if let Some(cg) = dg.cg.get(rec_id) {
                    if let Some(cn) = cg.cn.get(rec_pos) {
                        desc = self.sharable.get_tx(cn.block.cn_md_comment)?;
                    }
                }
            }
        }
        Ok(desc)
    }
    /// returns the master channel associated to the input channel name
    pub fn get_channel_master(&self, channel_name: &str) -> Option<String> {
        let mut master: Option<String> = None;
        if let Some((m, _dg_pos, (_cg_pos, _rec_idd), (_cn_pos, _rec_pos))) =
            self.get_channel_id(channel_name)
        {
            master.clone_from(m);
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
    /// returns the set of channel names that are in same channel group as input channel name
    pub fn get_channel_names_cg_set(&self, channel_name: &str) -> HashSet<String> {
        if let Some((_master, dg_pos, (_cg_pos, rec_id), (_cn_pos, _rec_pos))) =
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
                if let Some(list) = channel_master_list.get_mut(&cg.master_channel_name) {
                    list.extend(cg.channel_names.clone());
                } else {
                    channel_master_list
                        .insert(cg.master_channel_name.clone(), cg.channel_names.clone());
                }
            }
        }
        channel_master_list
    }
    /// empty the channels' ndarray
    pub fn clear_channel_data_from_memory(&mut self, channel_names: HashSet<String>) -> Result<()> {
        for channel_name in channel_names {
            if let Some((_master, dg_pos, (_cg_pos, rec_id), (_cn_pos, rec_pos))) =
                self.channel_names_set.get_mut(&channel_name)
            {
                if let Some(dg) = self.dg.get_mut(dg_pos) {
                    if let Some(cg) = dg.cg.get_mut(rec_id) {
                        if let Some(cn) = cg.cn.get_mut(rec_pos) {
                            if !cn.data.is_empty() {
                                cn.data = cn.data.zeros(
                                    cn.block.cn_data_type,
                                    0,
                                    0,
                                    (Vec::new(), Order::RowMajor),
                                )?;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
    /// returns a new empty MdfInfo4 struct
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
        }
    }
    /// Adds a new channel in memory (no file modification)
    pub fn add_channel(
        &mut self,
        channel_name: String,
        data: ChannelData,
        data_signature: DataSignature,
        mut master: MasterSignature,
        unit: Option<String>,
        description: Option<String>,
    ) -> Result<(), Error> {
        let mut cg_block = Cg4Block {
            cg_cycle_count: data_signature.len as u64,
            ..Default::default()
        };
        // Basic channel block
        let mut cn_block = Cn4Block::default();
        let machine_endian: bool = cfg!(target_endian = "big");
        cn_block.cn_data_type = data_signature.data_type;
        cn_block.cn_bit_count = data_signature.bit_count;
        let cn_pos = position_generator();
        cn_block.cn_sync_type = master.master_type.unwrap_or(0);

        // channel name
        let channel_name_position = position_generator();
        cn_block.cn_tx_name = channel_name_position;
        self.sharable
            .create_tx(channel_name_position, channel_name.to_string());

        // Channel array
        let mut list_size = data_signature.shape.0.iter().product(); // primitive list size is 1
        if data_signature.data_type == 15 | 16 {
            //complex
            list_size *= 2;
        }
        let data_ndim = data_signature.ndim - 1;
        let mut composition: Option<Composition> = None;
        if data_ndim > 0 {
            let data_dim_size = data
                .shape()
                .0
                .iter()
                .skip(1)
                .map(|x| *x as u64)
                .collect::<Vec<_>>();
            // data_dim_size.remove(0);
            let mut ca_block = Ca4Block::default();
            cg_block.cg_data_bytes = list_size as u32 * data_signature.byte_count;

            let composition_position = position_generator();
            cn_block.cn_composition = composition_position;
            ca_block.ca_ndim = data_ndim as u16;
            ca_block.ca_dim_size.clone_from(&data_dim_size);
            ca_block.ca_len = 48 + 8 * data_ndim as u64;
            composition = Some(Composition {
                block: Compo::CA(Box::new(ca_block)),
                compo: None,
            });
        }

        // master channel
        if master.master_flag {
            cn_block.cn_type = 2; // master channel
        } else {
            cn_block.cn_type = 0; // data channel
            if let Some(master_channel_name) = master.master_channel.clone() {
                // looking for the master channel's cg position
                if let Some((m, _dg_pos, (cg_pos, _rec_id), (_cn_pos, _rec_pos))) =
                    self.channel_names_set.get(&master_channel_name)
                {
                    cg_block.cg_cg_master = Some(*cg_pos);
                    cg_block.cg_flags = 0b1000;
                    cg_block.cg_links = 7; // with cg_cg_master
                                           // cg_block.cg_len = 112;
                    master.master_channel.clone_from(m);
                }
            }
        }
        if let Some(sync_type) = master.master_type {
            cn_block.cn_sync_type = sync_type;
        }

        // unit
        if let Some(u) = unit {
            let unit_position = position_generator();
            cn_block.cn_md_unit = unit_position;
            self.sharable.create_tx(unit_position, u);
        }

        // description
        if let Some(d) = description {
            let md_comment = position_generator();
            cn_block.cn_md_comment = md_comment;
            self.sharable.create_tx(md_comment, d);
        }

        // CN
        let n_bytes = data_signature.byte_count;
        let cn = Cn4 {
            header: default_short_header(BlockType::CN),
            unique_name: channel_name.to_string(),
            data,
            block: cn_block,
            endian: machine_endian,
            block_position: cn_pos,
            pos_byte_beg: 0,
            n_bytes,
            composition,
            list_size,
            shape: data_signature.shape,
            invalid_mask: None,
        };

        // CG
        let cg_pos = position_generator();
        cg_block.cg_data_bytes = n_bytes;
        let mut cg = Cg4 {
            header: default_short_header(BlockType::CG),
            block: cg_block,
            master_channel_name: master.master_channel.clone(),
            cn: HashMap::new(),
            block_position: cg_pos,
            channel_names: HashSet::new(),
            record_length: n_bytes,
            vlsd_cg: None,
            invalid_bytes: None,
        };
        cg.cn.insert(0, cn);
        cg.channel_names.insert(channel_name.to_string());

        // DG
        let dg_pos = position_generator();
        let dg_block = Dg4Block::default();
        let mut dg = Dg4 {
            block: dg_block,
            cg: HashMap::new(),
        };
        dg.cg.insert(0, cg);
        self.dg.insert(dg_pos, dg);

        self.channel_names_set.insert(
            channel_name,
            (master.master_channel, dg_pos, (cg_pos, 0), (cn_pos, 0)),
        );
        Ok(())
    }
    /// Removes a channel in memory (no file modification)
    pub fn remove_channel(&mut self, channel_name: &str) {
        if let Some((_master, dg_pos, (_cg_pos, rec_id), (_cn_pos, rec_pos))) =
            self.channel_names_set.get(channel_name)
        {
            if let Some(dg) = self.dg.get_mut(dg_pos) {
                if let Some(cg) = dg.cg.get_mut(rec_id) {
                    cg.cn.remove(rec_pos);
                    cg.channel_names.remove(channel_name);
                    self.channel_names_set.remove(channel_name);
                }
            }
        }
    }
    /// Renames a channel's name in memory
    pub fn rename_channel(&mut self, channel_name: &str, new_name: &str) {
        if let Some((master, dg_pos, (cg_pos, rec_id), (cn_pos, rec_pos))) =
            self.channel_names_set.remove(channel_name)
        {
            if let Some(dg) = self.dg.get_mut(&dg_pos) {
                if let Some(cg) = dg.cg.get_mut(&rec_id) {
                    if let Some(cn) = cg.cn.get_mut(&rec_pos) {
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
                                        val.3 = (cn_pos, rec_pos);
                                    }
                                });
                            }
                        }

                        self.channel_names_set.insert(
                            new_name.to_string(),
                            (master, dg_pos, (cg_pos, rec_id), (cn_pos, rec_pos)),
                        );
                    }
                }
            }
        }
    }
    /// defines channel's data in memory
    pub fn set_channel_data(
        &mut self,
        channel_name: &str,
        data: Arc<dyn Array>,
    ) -> Result<(), Error> {
        if let Some((_master, dg_pos, (_cg_pos, rec_id), (_cn_pos, rec_pos))) =
            self.channel_names_set.get(channel_name)
        {
            if let Some(dg) = self.dg.get_mut(dg_pos) {
                if let Some(cg) = dg.cg.get_mut(rec_id) {
                    if let Some(cn) = cg.cn.get_mut(rec_pos) {
                        cn.data = try_from(&data)
                            .context("failed converting dyn array to ChannelData")?;
                    }
                }
            }
        }
        Ok(())
    }
    /// Sets the channel unit in memory
    pub fn set_channel_unit(&mut self, channel_name: &str, unit: &str) {
        if let Some((_master, dg_pos, (_cg_pos, rec_id), (_cn_pos, rec_pos))) =
            self.channel_names_set.get(channel_name)
        {
            if let Some(dg) = self.dg.get_mut(dg_pos) {
                if let Some(cg) = dg.cg.get_mut(rec_id) {
                    if let Some(cn) = cg.cn.get_mut(rec_pos) {
                        // hopefully never 2 times the same position
                        let position = position_generator();
                        self.sharable.create_tx(position, unit.to_string());
                        cn.block.cn_md_unit = position;
                    }
                }
            }
        }
    }
    /// Sets the channel description in memory
    pub fn set_channel_desc(&mut self, channel_name: &str, desc: &str) {
        if let Some((_master, dg_pos, (_cg_pos, rec_id), (_cn_pos, rec_pos))) =
            self.channel_names_set.get(channel_name)
        {
            if let Some(dg) = self.dg.get_mut(dg_pos) {
                if let Some(cg) = dg.cg.get_mut(rec_id) {
                    if let Some(cn) = cg.cn.get_mut(rec_pos) {
                        let position = position_generator();
                        self.sharable.create_tx(position, desc.to_string());
                        cn.block.cn_md_comment = position;
                    }
                }
            }
        }
    }
    /// Sets the channel's related master channel type in memory
    pub fn set_channel_master_type(&mut self, master_name: &str, master_type: u8) {
        if let Some((_master, dg_pos, (_cg_pos, rec_id), (_cn_pos, rec_pos))) =
            self.channel_names_set.get(master_name)
        {
            if let Some(dg) = self.dg.get_mut(dg_pos) {
                if let Some(cg) = dg.cg.get_mut(rec_id) {
                    if let Some(cn) = cg.cn.get_mut(rec_pos) {
                        cn.block.cn_sync_type = master_type;
                    }
                }
            }
        }
    }
    /// list attachments
    pub fn list_attachments(&mut self) -> String {
        let mut output = String::new();
        for (key, (block, _embedded_data)) in self.at.iter() {
            output.push_str(&format!(
                "position: {}, filename: {:?}, mimetype: {:?}, comment: {:?}\n ",
                key,
                self.sharable.get_tx(block.at_tx_filename),
                self.sharable.get_tx(block.at_tx_mimetype),
                self.sharable.get_comments(block.at_md_comment)
            ))
        }
        output
    }
    /// get embedded data in attachment for a block at position
    pub fn get_attachment_embedded_data(&self, position: i64) -> Option<Vec<u8>> {
        if let Some(at) = self.at.get(&position) {
            match &at.1 {
                None => None,
                Some(embedded_data) => {
                    // are data compressed
                    let data: Vec<u8>;
                    if (at.0.at_flags & 0b10) > 0 {
                        // Compressed data
                        let checksum: Option<u32>;
                        (data, checksum) = decompress(embedded_data, Format::Zlib)
                            .expect("Could not decompress attached embedded data");
                        // is checksum valid
                        if (at.0.at_flags & 0b100) > 0 {
                            // verify data integrity
                            let mut hasher = Md5::new();
                            hasher.update(data.clone());
                            let result = hasher.finalize();
                            if result == at.0.at_md5_checksum.into() {
                                Some(data)
                            } else {
                                warn!("Embedded data checksum not ok");
                                None
                            }
                        } else if Some(Adler32::from_buf(&data).finish()) != checksum {
                            warn!("Embedded data checksum not ok");
                            None
                        } else {
                            Some(data)
                        }
                    } else {
                        // not compressed data
                        if (at.0.at_flags & 0b100) > 0 {
                            // verify data integrity
                            let mut hasher = Md5::new();
                            hasher.update(embedded_data.clone());
                            let result = hasher.finalize();
                            if result == at.0.at_md5_checksum.into() {
                                Some(embedded_data.to_vec())
                            } else {
                                warn!("Embedded data checksum not ok");
                                None
                            }
                        } else {
                            Some(embedded_data.to_vec())
                        }
                    }
                }
            }
        } else {
            None
        }
    }
    /// get list attachment block
    pub fn get_attachment_block(&self, position: i64) -> Option<At4Block> {
        if let Some((block, _)) = self.at.get(&position) {
            Some(*block)
        } else {
            None
        }
    }
    /// get all attachment blocks
    pub fn get_attachment_blocks(&self) -> HashMap<i64, At4Block> {
        let mut output: HashMap<i64, At4Block> = HashMap::new();
        for (key, (block, _data)) in self.at.iter() {
            output.insert(*key, *block);
        }
        output
    }
    /// list events
    pub fn list_events(&mut self) -> String {
        let mut output = String::new();
        for (key, block) in self.ev.iter() {
            output.push_str(&format!(
                "position: {}, name: {:?}, comment: {:?}, scope: {:?}, attachment references: {:?}, event type: {}\n",
                key,
                self.sharable.get_tx(block.ev_tx_name),
                self.sharable.get_comments(block.ev_md_comment),
                block.links[0..block.ev_scope_count as usize].to_vec(),
                block.links[block.ev_scope_count as usize.. block.ev_attachment_count as usize].to_vec(),
                block.ev_type,
            ))
        }
        output
    }
    /// get event block from its position
    pub fn get_event_block(&self, position: i64) -> Option<Ev4Block> {
        self.ev.get(&position).cloned()
    }
    /// get all event blocks
    pub fn get_event_blocks(&self) -> HashMap<i64, Ev4Block> {
        self.ev.clone()
    }
    // TODO Extract CH
}

/// creates random negative position
pub fn position_generator() -> i64 {
    // hopefully never 2 times the same position
    let mut position = rand::random::<i64>();
    if position > 0 {
        // make sure position is negative to avoid interference with existing positions in file
        position = -position;
    }
    position
}

/// MdfInfo4 display implementation
impl fmt::Display for MdfInfo4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MdfInfo4: {}", self.file_name)?;
        writeln!(f, "Version : {}\n", self.id_block.id_ver)?;
        writeln!(f, "{}\n", self.hd_block)?;
        let comments = &self.sharable.get_hd_comments(self.hd_block.hd_md_comment);
        for c in comments.iter() {
            writeln!(f, "{} {}\n", c.0, c.1)?;
        }
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

/// MDF4 - common block Header
#[derive(Debug, Copy, Clone)]
#[binrw]
#[br(little)]
#[repr(C)]
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
pub fn parse_block_header(rdr: &mut SymBufReader<&File>) -> Result<Blockheader4> {
    let mut buf = [0u8; 24];
    rdr.read_exact(&mut buf)
        .context("could not read blockheader4 Id")?;
    let mut block = Cursor::new(buf);
    let header: Blockheader4 = block
        .read_le()
        .context("binread could not parse blockheader4")?;
    Ok(header)
}

/// MDF4 - common block Header without the number of links
#[derive(Debug, Copy, Clone)]
#[binrw]
#[br(little)]
#[allow(dead_code)]
#[repr(C)]
pub struct Blockheader4Short {
    /// '##XX'
    hdr_id: [u8; 4],
    /// reserved, must be 0
    hdr_gap: [u8; 4],
    /// Length of block in bytes
    pub hdr_len: u64,
}

impl Default for Blockheader4Short {
    fn default() -> Self {
        Blockheader4Short {
            hdr_id: [35, 35, 67, 78], // ##CN
            hdr_gap: [0u8; 4],
            hdr_len: 160,
        }
    }
}

pub fn default_short_header(variant: BlockType) -> Blockheader4Short {
    match variant {
        BlockType::CG => Blockheader4Short {
            hdr_id: [35, 35, 67, 71], // ##CG
            hdr_gap: [0u8; 4],
            hdr_len: 104, // 112 with cg_cg_master, 104 without,
        },
        BlockType::CN => Blockheader4Short {
            hdr_id: [35, 35, 67, 78], // ##CN
            hdr_gap: [0u8; 4],
            hdr_len: 160,
        },
        _ => Blockheader4Short {
            hdr_id: [35, 35, 67, 78], // ##CN
            hdr_gap: [0u8; 4],
            hdr_len: 160,
        },
    }
}

/// parse the block header and its fields id, (reserved), length except the number of links
#[inline]
fn parse_block_header_short(rdr: &mut SymBufReader<&File>) -> Result<Blockheader4Short> {
    let mut buf = [0u8; 16];
    rdr.read_exact(&mut buf)
        .context("could not read short blockheader4 Id")?;
    let mut block = Cursor::new(buf);
    let header: Blockheader4Short = block
        .read_le()
        .context("could not parse short blockheader4")?;
    Ok(header)
}

/// reads generically a block header and return links and members section part into a Seek buffer for further processing
#[inline]
fn parse_block(
    rdr: &mut SymBufReader<&File>,
    target: i64,
    mut position: i64,
) -> Result<(Cursor<Vec<u8>>, Blockheader4, i64)> {
    // Reads block header
    rdr.seek_relative(target - position)
        .context("Could not reach block header position")?; // change buffer position
    let block_header = parse_block_header(rdr).context(" could not read header block")?; // reads header

    // Reads in buffer rest of block
    let mut buf = vec![0u8; (block_header.hdr_len - 24) as usize];
    rdr.read_exact(&mut buf)
        .context("Could not read rest of block after header")?;
    position = target + block_header.hdr_len as i64;
    let block = Cursor::new(buf);
    Ok((block, block_header, position))
}

/// reads generically a block header wihtout the number of links and returns links and members section part into a Seek buffer for further processing
#[inline]
fn parse_block_short(
    rdr: &mut SymBufReader<&File>,
    target: i64,
    mut position: i64,
) -> Result<(Cursor<Vec<u8>>, Blockheader4Short, i64)> {
    // Reads block header
    rdr.seek_relative(target - position)
        .context("Could not reach block short header position")?; // change buffer position
    let block_header: Blockheader4Short =
        parse_block_header_short(rdr).context(" could not read short header block")?; // reads header

    // Reads in buffer rest of block
    let mut buf = vec![0u8; (block_header.hdr_len - 16) as usize];
    rdr.read_exact(&mut buf)
        .context("Could not read rest of block after short header")?;
    position = target + block_header.hdr_len as i64;
    let block = Cursor::new(buf);
    Ok((block, block_header, position))
}

/// metadata are either stored in TX (text) or MD (xml) blocks for mdf version 4
#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(C)]
#[derive(Default)]
pub enum MetaDataBlockType {
    MdBlock,
    MdParsed,
    #[default]
    TX,
}

/// Blocks types that could link to MDBlock
#[derive(Debug, Clone)]
#[repr(C)]
#[derive(Default)]
pub enum BlockType {
    HD,
    FH,
    AT,
    EV,
    DG,
    CG,
    #[default]
    CN,
    CC,
    SI,
}

/// struct linking MD or TX block with
#[derive(Debug, Default, Clone)]
#[repr(C)]
pub struct MetaData {
    /// Header of the block
    pub block: Blockheader4,
    /// Raw bytes for the block's data
    pub raw_data: Vec<u8>,
    /// Block type, TX, MD or MD not yet parsed
    pub block_type: MetaDataBlockType,
    /// Metadata after parsing
    pub comments: HashMap<String, String>,
    /// Parent block type
    pub parent_block_type: BlockType,
}

/// Parses the MD or TX block
fn read_meta_data(
    rdr: &mut SymBufReader<&File>,
    sharable: &mut SharableBlocks,
    target: i64,
    mut position: i64,
    parent_block_type: BlockType,
) -> Result<i64> {
    if target != 0 && !sharable.md_tx.contains_key(&target) {
        let (raw_data, block, pos) =
            parse_block(rdr, target, position).context("could not read metadata block")?;
        position = pos;
        let block_type = match block.hdr_id {
            [35, 35, 77, 68] => MetaDataBlockType::MdBlock,
            [35, 35, 84, 88] => MetaDataBlockType::TX,
            _ => MetaDataBlockType::TX,
        };
        let md = MetaData {
            block,
            raw_data: raw_data.into_inner(),
            block_type,
            comments: HashMap::new(),
            parent_block_type,
        };
        sharable.md_tx.insert(target, md);
        Ok(position)
    } else {
        Ok(position)
    }
}

impl MetaData {
    /// Returns a new MetaData struct
    pub fn new(block_type: MetaDataBlockType, parent_block_type: BlockType) -> Self {
        let header = match block_type {
            MetaDataBlockType::MdBlock => Blockheader4 {
                hdr_id: [35, 35, 77, 68], // '##MD'
                hdr_gap: [0u8; 4],
                hdr_len: 24,
                hdr_links: 0,
            },
            MetaDataBlockType::TX | MetaDataBlockType::MdParsed => Blockheader4 {
                hdr_id: [35, 35, 84, 88], // '##TX'
                hdr_gap: [0u8; 4],
                hdr_len: 24,
                hdr_links: 0,
            },
        };
        MetaData {
            block: header,
            raw_data: Vec::new(),
            block_type,
            comments: HashMap::new(),
            parent_block_type,
        }
    }
    /// Converts the metadata handling the parent block type's specificities
    pub fn parse_xml(&mut self) -> Result<()> {
        if self.block_type == MetaDataBlockType::MdBlock {
            match self.parent_block_type {
                BlockType::HD => self.parse_hd_xml()?,
                BlockType::FH => self.parse_fh_xml()?,
                _ => self.parse_generic_xml()?,
            };
        }
        Ok(())
    }
    /// Returns the text from TX Block or TX's tag text from MD Block
    pub fn get_tx(&self) -> Result<Option<String>, Error> {
        match self.block_type {
            MetaDataBlockType::MdParsed => Ok(self.comments.get("TX").cloned()),
            MetaDataBlockType::MdBlock => {
                // extract TX tag from xml
                let comment: String = self
                    .get_data_string()
                    .context("failed getting data string to extract TX tag")?
                    .trim_end_matches(|c| c == '\n' || c == '\r' || c == ' ')
                    .into(); // removes ending spaces
                match roxmltree::Document::parse(&comment) {
                    Ok(md) => {
                        let mut tx: Option<String> = None;
                        for node in md.root().descendants() {
                            let text = match node.text() {
                                Some(text) => text.to_string(),
                                None => String::new(),
                            };
                            if node.is_element()
                                && !text.is_empty()
                                && node.tag_name().name() == r"TX"
                            {
                                tx = Some(text);
                                break;
                            }
                        }
                        Ok(tx)
                    }
                    Err(e) => {
                        warn!("Error parsing comment : \n{}\n{}", comment, e);
                        Ok(None)
                    }
                }
            }
            MetaDataBlockType::TX => {
                let comment = str::from_utf8(&self.raw_data).with_context(|| {
                    format!("Invalid UTF-8 sequence in metadata: {:?}", self.raw_data)
                })?;
                let c: String = comment.trim_end_matches(char::from(0)).into();
                Ok(Some(c))
            }
        }
    }
    /// Returns the bytes of the text from TX Block or TX's tag text from MD Block
    pub fn get_tx_bytes(&self) -> Option<&[u8]> {
        match self.block_type {
            MetaDataBlockType::MdParsed => self.comments.get("TX").map(|s| s.as_bytes()),
            _ => Some(&self.raw_data),
        }
    }
    /// Decode string from raw_data field
    pub fn get_data_string(&self) -> Result<String> {
        match self.block_type {
            MetaDataBlockType::MdParsed => Ok(String::new()),
            _ => {
                let comment = str::from_utf8(&self.raw_data).with_context(|| {
                    format!("Invalid UTF-8 sequence in metadata: {:?}", self.raw_data)
                })?;
                let comment: String = comment.trim_end_matches(char::from(0)).into();
                Ok(comment)
            }
        }
    }
    /// allocate bytes to raw_data field, adjusting header length
    pub fn set_data_buffer(&mut self, data: &[u8]) {
        self.raw_data = [data, vec![0u8; 8 - data.len() % 8].as_slice()].concat();
        self.block.hdr_len = self.raw_data.len() as u64 + 24;
    }
    /// parses the xml bytes specifically for HD block contexted schema
    fn parse_hd_xml(&mut self) -> Result<()> {
        let mut comments: HashMap<String, String> = HashMap::new();
        // MD Block from HD Block, reading xml
        let comment: String = self
            .get_data_string()?
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
                warn!("Could not parse HD MD comment : \n{}\n{}", comment, e);
            }
        };
        self.comments = comments;
        self.block_type = MetaDataBlockType::MdParsed;
        self.raw_data = vec![]; // empty the data from block as already parsed
        Ok(())
    }
    /// Creates File History MetaData
    pub fn create_fh(&mut self) {
        let user_name = whoami::username();
        let comments = format!(
            "<FHcomment>
<TX>created</TX>
<tool_id>mdfr</tool_id>
<tool_vendor>ratalco</tool_vendor>
<tool_version>0.1</tool_version>
<user_name>{user_name}</user_name>
</FHcomment>"
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
    /// parses the xml bytes specifically for File History block contexted schema
    fn parse_fh_xml(&mut self) -> Result<()> {
        let mut comments: HashMap<String, String> = HashMap::new();
        // MD Block from FH Block, reading xml
        let comment: String = self
            .get_data_string()?
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
                warn!("Could not parse FH comment : \n{}\n{}", comment, e);
            }
        };
        self.comments = comments;
        self.block_type = MetaDataBlockType::MdParsed;
        self.raw_data = vec![]; // empty the data from block as already parsed
        Ok(())
    }
    /// Generic xml parser without schema consideration
    fn parse_generic_xml(&mut self) -> Result<()> {
        let mut comments: HashMap<String, String> = HashMap::new();
        let comment: String = self
            .get_data_string()?
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
                warn!("Error parsing comment : \n{}\n{}", comment, e);
            }
        };
        self.comments = comments;
        self.block_type = MetaDataBlockType::MdParsed;
        self.raw_data = vec![]; // empty the data from block as already parsed
        Ok(())
    }
    /// Writes the metadata to file
    pub fn write<W>(&self, writer: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        writer
            .write_le(&self.block)
            .context("Could not write comment block header")?;
        writer
            .write_all(&self.raw_data)
            .context("Could not write comment block data")?;
        Ok(())
    }
}

/// Hd4 (Header) block structure
#[derive(Debug, Copy, Clone)]
#[binrw]
#[br(little)]
#[allow(dead_code)]
#[repr(C)]
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
    /// Time stamp in nanoseconds elapsed since 00:00:00 01.01.1970 (UTC time or local time, depending on "local time" flag)
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
            hd_start_time_ns: Local::now()
                .timestamp_nanos_opt()
                .map(|t| t as u64)
                .unwrap_or(0),
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
        let naive = DateTime::from_timestamp(sec as i64, nsec).unwrap_or_default();
        writeln!(f, "Time : {} ", naive.to_rfc3339())
    }
}

/// Hd4 block struct parser
pub fn hd4_parser(
    rdr: &mut SymBufReader<&File>,
    sharable: &mut SharableBlocks,
) -> Result<(Hd4, i64)> {
    let mut buf = [0u8; 104];
    rdr.read_exact(&mut buf)
        .context("could not read HD block buffer")?;
    let mut block = Cursor::new(buf);
    let hd: Hd4 = block
        .read_le()
        .context("Could not parse HD block buffer into Hd4 struct")?;
    let position = read_meta_data(rdr, sharable, hd.hd_md_comment, 168, BlockType::HD)?;
    Ok((hd, position))
}

/// Fh4 (File History) block struct, including the header
#[derive(Debug, Copy, Clone)]
#[binrw]
#[br(little)]
#[allow(dead_code)]
#[repr(C)]
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
            fh_time_ns: Local::now()
                .timestamp_nanos_opt()
                .map(|t| t as u64)
                .unwrap_or(0),
            fh_tz_offset_min: 0,
            fh_dst_offset_min: 0,
            fh_time_flags: 0,
            fh_reserved: [0u8; 3],
        }
    }
}

/// Fh4 (File History) block struct parser
fn parse_fh_block(
    rdr: &mut SymBufReader<&File>,
    target: i64,
    position: i64,
) -> Result<(FhBlock, i64)> {
    rdr.seek_relative(target - position)
        .context("Could not reach FH Block position")?; // change buffer position
    let mut buf = [0u8; 56];
    rdr.read_exact(&mut buf)
        .context("Could not read FH block buffer")?;
    let mut block = Cursor::new(buf);
    let fh: FhBlock = block
        .read_le()
        .with_context(|| format!("Error parsing fh block into FhBlock struct \n{block:?}"))?; // reads the fh block
    Ok((fh, target + 56))
}

type Fh = Vec<FhBlock>;

/// parses File History blocks along with its linked comments returns a vect of Fh4 block with comments
pub fn parse_fh(
    rdr: &mut SymBufReader<&File>,
    sharable: &mut SharableBlocks,
    target: i64,
    mut position: i64,
) -> Result<(Fh, i64)> {
    let mut fh: Fh = Vec::new();
    let (block, pos) = parse_fh_block(rdr, target, position)?;
    position = pos;
    position = read_meta_data(rdr, sharable, block.fh_md_comment, position, BlockType::FH)?;
    let mut next_pointer = block.fh_fh_next;
    fh.push(block);
    while next_pointer != 0 {
        let (block, pos) = parse_fh_block(rdr, next_pointer, position)?;
        position = pos;
        next_pointer = block.fh_fh_next;
        position = read_meta_data(rdr, sharable, block.fh_md_comment, position, BlockType::FH)?;
        fh.push(block);
    }
    Ok((fh, position))
}
/// At4 Attachment block struct
#[derive(Debug, Copy, Clone)]
#[binrw]
#[br(little)]
#[allow(dead_code)]
#[repr(C)]
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
    pub at_tx_filename: i64,
    /// Link to TXBLOCK with MIME content-type text that gives information about the attached data. Can be NIL if the content-type is unknown, but should be specified whenever possible. The MIME content-type string must be written in lowercase.
    pub at_tx_mimetype: i64,
    /// Link to MDBLOCK with comment and additional information about the attachment (can be NIL).
    pub at_md_comment: i64,
    /// Flags The value contains the following bit flags (see AT_FL_xxx):
    pub at_flags: u16,
    /// Creator index, i.e. zero-based index of FHBLOCK in global list of FHBLOCKs that specifies which application has created this attachment, or changed it most recently.
    pub at_creator_index: u16,
    /// Reserved
    at_reserved: [u8; 4],
    /// 128-bit value for MD5 check sum (of the uncompressed data if data is embedded and compressed). Only valid if "MD5 check sum valid" flag (bit 2) is set.
    pub at_md5_checksum: [u8; 16],
    /// Original data size in Bytes, i.e. either for external file or for uncompressed data.
    pub at_original_size: u64,
    /// Embedded data size N, i.e. number of Bytes for binary embedded data following this element. Must be 0 if external file is referenced.
    pub at_embedded_size: u64,
    // followed by embedded data depending of flag
}

/// At4 (Attachment) block struct parser
fn parser_at4_block(
    rdr: &mut SymBufReader<&File>,
    target: i64,
    mut position: i64,
) -> Result<(At4Block, Option<Vec<u8>>, i64)> {
    rdr.seek_relative(target - position)
        .context("Could not reach At4 Block position")?;
    let mut buf = [0u8; 96];
    rdr.read_exact(&mut buf)
        .context("Could not read At4 Block buffer")?;
    let mut block = Cursor::new(buf);
    let block: At4Block = block
        .read_le()
        .context("Could not parse At4 Block buffer into At4Block struct")?;
    position = target + 96;

    // reads embedded if exists
    let data: Option<Vec<u8>> = if (block.at_flags & 0b1) > 0 {
        let mut embedded_data = vec![0u8; block.at_embedded_size as usize];
        rdr.read_exact(&mut embedded_data)
            .context("Could not parse At4Block embedded attachement")?;
        position += block.at_embedded_size as i64;
        Some(embedded_data)
    } else {
        None
    };
    Ok((block, data, position))
}

type At = HashMap<i64, (At4Block, Option<Vec<u8>>)>;

/// parses Attachment blocks along with its linked comments, returns a hashmap of At4 block and attached data in a vect
pub fn parse_at4(
    rdr: &mut SymBufReader<&File>,
    sharable: &mut SharableBlocks,
    target: i64,
    mut position: i64,
) -> Result<(At, i64)> {
    let mut at: At = HashMap::new();
    if target > 0 {
        let (block, data, pos) = parser_at4_block(rdr, target, position)?;
        position = pos;
        // Reads MD
        position = read_meta_data(rdr, sharable, block.at_md_comment, position, BlockType::AT)?;
        // reads TX file_name
        position = read_meta_data(rdr, sharable, block.at_tx_filename, position, BlockType::AT)?;
        // Reads tx mime type
        position = read_meta_data(rdr, sharable, block.at_tx_mimetype, position, BlockType::AT)?;
        let mut next_pointer = block.at_at_next;
        at.insert(target, (block, data));

        while next_pointer > 0 {
            let block_start = next_pointer;
            let (block, data, pos) = parser_at4_block(rdr, next_pointer, position)?;
            position = pos;
            // Reads MD
            position = read_meta_data(rdr, sharable, block.at_md_comment, position, BlockType::AT)?;
            // reads TX file_name
            position =
                read_meta_data(rdr, sharable, block.at_tx_filename, position, BlockType::AT)?;
            // Reads tx mime type
            position =
                read_meta_data(rdr, sharable, block.at_tx_mimetype, position, BlockType::AT)?;
            next_pointer = block.at_at_next;
            at.insert(block_start, (block, data));
        }
    }
    Ok((at, position))
}

/// Ev4 Event block struct
#[derive(Debug, Clone)]
#[binrw]
#[br(little)]
#[allow(dead_code)]
#[repr(C)]
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
    pub ev_tx_name: i64,
    /// Pointer to TX/MDBLOCK with event comment and additional information, e.g. trigger condition or formatted user comment text (can be NIL)
    pub ev_md_comment: i64,
    #[br(if(ev_links > 5), little, count = ev_links - 5)]
    /// links
    links: Vec<i64>,

    /// Event type (see EV_T_xxx)
    pub ev_type: u8,
    /// Sync type (see EV_S_xxx)
    pub ev_sync_type: u8,
    /// Range Type (see EV_R_xxx)
    pub ev_range_type: u8,
    /// Cause of event (see EV_C_xxx)
    pub ev_cause: u8,
    /// flags (see EV_F_xxx)
    pub ev_flags: u8,
    /// Reserved
    ev_reserved: [u8; 3],
    /// Length M of ev_scope list. Can be zero.
    pub ev_scope_count: u32,
    /// Length N of ev_at_reference list, i.e. number of attachments for this event. Can be zero.
    pub ev_attachment_count: u16,
    /// Creator index, i.e. zero-based index of FHBLOCK in global list of FHBLOCKs that specifies which application has created or changed this event (e.g. when generating event offline).
    pub ev_creator_index: u16,
    /// Base value for synchronization value.
    pub ev_sync_base_value: i64,
    /// Factor for event synchronization value.
    pub ev_sync_factor: f64,
}

/// Ev4 (Event) block struct parser
fn parse_ev4_block(
    rdr: &mut SymBufReader<&File>,
    target: i64,
    mut position: i64,
) -> Result<(Ev4Block, i64)> {
    let (mut block, _header, pos) = parse_block_short(rdr, target, position)?;
    position = pos;
    let block: Ev4Block = block.read_le().context("Error parsing ev block")?; // reads the fh block

    Ok((block, position))
}

/// parses Event blocks along with its linked comments, returns a hashmap of Ev4 block with position as key
pub fn parse_ev4(
    rdr: &mut SymBufReader<&File>,
    sharable: &mut SharableBlocks,
    target: i64,
    mut position: i64,
) -> Result<(HashMap<i64, Ev4Block>, i64)> {
    let mut ev: HashMap<i64, Ev4Block> = HashMap::new();
    if target > 0 {
        let (block, pos) = parse_ev4_block(rdr, target, position)?;
        position = pos;
        // Reads MD
        position = read_meta_data(rdr, sharable, block.ev_md_comment, position, BlockType::EV)?;
        // reads TX event name
        position = read_meta_data(rdr, sharable, block.ev_tx_name, position, BlockType::EV)?;
        let mut next_pointer = block.ev_ev_next;
        ev.insert(target, block);

        while next_pointer > 0 {
            let block_start = next_pointer;
            let (block, pos) = parse_ev4_block(rdr, next_pointer, position)?;
            position = pos;
            // Reads MD
            position = read_meta_data(rdr, sharable, block.ev_md_comment, position, BlockType::EV)?;
            // reads TX event name
            position = read_meta_data(rdr, sharable, block.ev_tx_name, position, BlockType::EV)?;
            next_pointer = block.ev_ev_next;
            ev.insert(block_start, block);
        }
    }
    Ok((ev, position))
}

/// Dg4 Data Group block struct
#[derive(Debug, Copy, Clone)]
#[binrw]
#[br(little)]
#[allow(dead_code)]
#[repr(C)]
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
    rdr: &mut SymBufReader<&File>,
    sharable: &mut SharableBlocks,
    target: i64,
    mut position: i64,
) -> Result<(Dg4Block, i64)> {
    rdr.seek_relative(target - position)
        .context("Could not reach position of Dg4 block")?;
    let mut buf = [0u8; 64];
    rdr.read_exact(&mut buf)
        .context("Could not read Dg4Blcok buffer")?;
    let mut block = Cursor::new(buf);
    let dg: Dg4Block = block
        .read_le()
        .context("Could not parse Dg4Block buffer into Dg4Block struct")?;
    position = target + 64;

    // Reads MD
    position = read_meta_data(rdr, sharable, dg.dg_md_comment, position, BlockType::DG)?;

    Ok((dg, position))
}

/// Dg4 struct wrapping block, comments and linked CG
#[derive(Debug, Clone)]
#[allow(dead_code)]
#[repr(C)]
pub struct Dg4 {
    /// DG Block
    pub block: Dg4Block,
    /// CG Block
    pub cg: HashMap<u64, Cg4>,
}

/// Parser for Dg4 and all linked blocks (cg, cn, cc, ca, si)
pub fn parse_dg4(
    rdr: &mut SymBufReader<&File>,
    target: i64,
    mut position: i64,
    sharable: &mut SharableBlocks,
) -> Result<(BTreeMap<i64, Dg4>, i64, usize, usize)> {
    let mut dg: BTreeMap<i64, Dg4> = BTreeMap::new();
    let mut n_cn: usize = 0;
    let mut n_cg: usize = 0;
    if target > 0 {
        let (block, pos) = parse_dg4_block(rdr, sharable, target, position)?;
        position = pos;
        let mut next_pointer = block.dg_dg_next;
        let (mut cg, pos, num_cg, num_cn) = parse_cg4(
            rdr,
            block.dg_cg_first,
            position,
            sharable,
            block.dg_rec_id_size,
        )?;
        n_cg += num_cg;
        n_cn += num_cn;
        identify_vlsd_cg(&mut cg);
        let dg_struct = Dg4 { block, cg };
        dg.insert(target, dg_struct);
        position = pos;
        while next_pointer > 0 {
            let block_start = next_pointer;
            let (block, pos) = parse_dg4_block(rdr, sharable, next_pointer, position)?;
            next_pointer = block.dg_dg_next;
            position = pos;
            let (mut cg, pos, num_cg, num_cn) = parse_cg4(
                rdr,
                block.dg_cg_first,
                position,
                sharable,
                block.dg_rec_id_size,
            )?;
            n_cg += num_cg;
            n_cn += num_cn;
            identify_vlsd_cg(&mut cg);
            let dg_struct = Dg4 { block, cg };
            dg.insert(block_start, dg_struct);
            position = pos;
        }
    }
    Ok((dg, position, n_cg, n_cn))
}

/// Try to link VLSD Channel Groups with matching channel in other groups
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
#[repr(C)]
pub struct SharableBlocks {
    pub(crate) md_tx: HashMap<i64, MetaData>,
    pub(crate) cc: HashMap<i64, Cc4Block>,
    pub(crate) si: HashMap<i64, Si4Block>,
}

/// SharableBlocks display implementation to facilitate debugging
impl fmt::Display for SharableBlocks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MD TX comments : \n")?;
        for (_k, c) in self.md_tx.iter() {
            match c.block_type {
                MetaDataBlockType::MdParsed => {
                    for (tag, text) in c.comments.iter() {
                        writeln!(f, "Tag: {tag}  Text: {text}")?;
                    }
                }
                MetaDataBlockType::TX => match c.get_data_string() {
                    Ok(s) => writeln!(f, "Text: {s}")?,
                    Err(e) => writeln!(f, "Text: {e:?}")?,
                },
                _ => (),
            }
        }
        writeln!(f, "CC : \n")?;
        for (position, cc) in self.cc.iter() {
            writeln!(f, "Position: {position}  Text: {cc:?}")?;
        }
        writeln!(f, "SI : ")?;
        for (position, si) in self.si.iter() {
            writeln!(f, "Position: {position}  Text: {si:?}")?;
        }
        writeln!(f, "finished")
    }
}

impl SharableBlocks {
    /// Returns the text from TX Block or TX tag's text from MD block
    pub fn get_tx(&self, position: i64) -> Result<Option<String>> {
        let mut txt: Option<String> = None;
        if let Some(md) = self.md_tx.get(&position) {
            txt = md.get_tx()?;
        };
        Ok(txt)
    }
    /// Creates a new SharableBlocks of type TX (not MD)
    pub fn create_tx(&mut self, position: i64, text: String) {
        let md = self
            .md_tx
            .entry(position)
            .or_insert_with(|| MetaData::new(MetaDataBlockType::TX, BlockType::CN));
        md.set_data_buffer(text.as_bytes());
    }
    /// Returns metadata from MD Block
    /// keys are tag and related value text of tag
    pub fn get_comments(&mut self, position: i64) -> HashMap<String, String> {
        let mut comments: HashMap<String, String> = HashMap::new();
        if let Some(md) = self.md_tx.get_mut(&position) {
            match md.block_type {
                MetaDataBlockType::MdParsed => {
                    comments.clone_from(&md.comments);
                }
                MetaDataBlockType::MdBlock => {
                    // not yet parsed, so let's parse it
                    let _ = md.parse_xml();
                    comments.clone_from(&md.comments);
                }
                MetaDataBlockType::TX => {
                    // should not happen
                }
            }
        };
        comments
    }
    /// Returns metadata from MD Block linked by HD Block
    /// keys are tag and related value text of tag
    pub fn get_hd_comments(&self, position: i64) -> HashMap<String, String> {
        // this method assumes the xml was already parsed
        let mut comments: HashMap<String, String> = HashMap::new();
        if let Some(md) = self.md_tx.get(&position) {
            if md.block_type == MetaDataBlockType::MdParsed {
                comments.clone_from(&md.comments);
            }
        };
        comments
    }
    /// parses the HD Block metadata comments
    /// done right after reading HD block
    pub fn parse_hd_comments(&mut self, position: i64) {
        if let Some(md) = self.md_tx.get_mut(&position) {
            let _ = md.parse_hd_xml();
        };
    }
    /// Create new Shared Block
    pub fn new(n_channels: usize) -> SharableBlocks {
        let md_tx: HashMap<i64, MetaData> = HashMap::with_capacity(n_channels);
        let cc: HashMap<i64, Cc4Block> = HashMap::new();
        let si: HashMap<i64, Si4Block> = HashMap::new();
        SharableBlocks { md_tx, cc, si }
    }
}
/// Cg4 Channel Group block struct
#[derive(Debug, Copy, Clone)]
#[binrw]
#[br(little)]
#[allow(dead_code)]
#[repr(C)]
pub struct Cg4Block {
    /// ##CG
    // cg_id: [u8; 4],
    /// reserved
    // reserved: [u8; 4],
    /// Length of block in bytes
    // pub cg_len: u64,
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
            // cg_id: [35, 35, 67, 71], // ##CG
            // reserved: [0u8; 4],
            // cg_len: 104, // 112 with cg_cg_master, 104 without
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
    rdr: &mut SymBufReader<&File>,
    target: i64,
    mut position: i64,
    sharable: &mut SharableBlocks,
    record_id_size: u8,
) -> Result<(Cg4, i64, usize)> {
    let (mut block, header, pos) = parse_block_short(rdr, target, position)?;
    position = pos;
    let cg: Cg4Block = block
        .read_le()
        .context("Could not read buffer into Cg4Block struct")?;

    // Reads MD
    position = read_meta_data(rdr, sharable, cg.cg_md_comment, position, BlockType::CG)?;
    let record_layout = (record_id_size, cg.cg_data_bytes, cg.cg_inval_bytes);

    // reads CN (and other linked block behind like CC, SI, CA, etc.)
    let (cn, pos, n_cn, _first_rec_pos) = parse_cn4(
        rdr,
        cg.cg_cn_first,
        position,
        sharable,
        record_layout,
        cg.cg_cycle_count,
    )?;
    position = pos;

    // Reads Acq Name
    position = read_meta_data(rdr, sharable, cg.cg_tx_acq_name, position, BlockType::CG)?;

    // Reads SI Acq name
    let si_pointer = cg.cg_si_acq_source;
    if (si_pointer != 0) && !sharable.si.contains_key(&si_pointer) {
        let (mut si_block, _header, pos) = parse_block_short(rdr, si_pointer, position)?;
        position = pos;
        let si_block: Si4Block = si_block
            .read_le()
            .context("Could not read buffer into Si4block struct")?;
        position = read_meta_data(rdr, sharable, si_block.si_tx_name, position, BlockType::SI)?;
        position = read_meta_data(rdr, sharable, si_block.si_tx_path, position, BlockType::SI)?;
        sharable.si.insert(si_pointer, si_block);
    }

    let record_length = cg.cg_data_bytes;

    let cg_struct = Cg4 {
        header,
        block: cg,
        cn,
        master_channel_name: None,
        channel_names: HashSet::new(),
        record_length,
        block_position: target,
        vlsd_cg: None,
        invalid_bytes: None,
    };

    Ok((cg_struct, position, n_cn))
}

/// Channel Group struct
/// it contains the related channels structure, a set of channel names, the dedicated master channel name and other helper data.
#[derive(Debug, Clone)]
#[repr(C)]
pub struct Cg4 {
    /// short header
    pub header: Blockheader4Short,
    /// CG block without header
    pub block: Cg4Block,
    /// hashmap of channels
    pub cn: CnType,
    /// Master channel name
    pub master_channel_name: Option<String>,
    /// Set of channel names belonging to this channel group
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
    /// Channel group acquisition name
    fn get_cg_name(&self, sharable: &SharableBlocks) -> Result<Option<String>> {
        sharable.get_tx(self.block.cg_tx_acq_name)
    }
    /// Channel group source name
    fn get_cg_source_name(&self, sharable: &SharableBlocks) -> Result<Option<String>> {
        let si = sharable.si.get(&self.block.cg_si_acq_source);
        match si {
            Some(block) => Ok(block.get_si_source_name(sharable)?),
            None => Ok(None),
        }
    }
    /// Channel group source path
    fn get_cg_source_path(&self, sharable: &SharableBlocks) -> Result<Option<String>> {
        let si = sharable.si.get(&self.block.cg_si_acq_source);
        match si {
            Some(block) => Ok(block.get_si_path_name(sharable)?),
            None => Ok(None),
        }
    }
    /// Computes the validity mask for each channel in the group
    /// clears out the common invalid bytes vector for the group at the end
    pub fn process_all_channel_invalid_bits(&mut self) -> Result<(), Error> {
        // get invalid bytes
        let cg_inval_bytes = self.block.cg_inval_bytes as usize;
        if let Some(invalid_bytes) = &self.invalid_bytes {
            // To extract invalidity for each channel from invalid_bytes
            self.cn
                .par_iter_mut()
                .filter(|(_rec_pos, cn)| !cn.data.is_empty())
                .try_for_each(|(_rec_pos, cn): (&i32, &mut Cn4)| -> Result<(), Error> {
                    if let Some((Some(mask), invalid_byte_position, invalid_byte_mask)) =
                        &mut cn.invalid_mask
                    {
                        // mask is already initialised to all valid values.
                        invalid_bytes.chunks(cg_inval_bytes).enumerate().for_each(
                            |(index, record)| {
                                // arrow considers bit set as valid while mdf spec considers bit set as invalid
                                mask.set_bit(
                                    index,
                                    (record[*invalid_byte_position] & *invalid_byte_mask) == 0,
                                );
                            },
                        );
                        cn.data.set_validity(mask).with_context(|| {
                            format!(
                                "failed applying invalid bits for channel {}",
                                cn.unique_name
                            )
                        })?;
                    }
                    Ok(())
                })?;
            self.invalid_bytes = None; // Clears out invalid bytes channel
        } else if cg_inval_bytes > 0 {
            // invalidity already stored in mask for each channel by read_channels_from_bytes()
            // to set validity in arrow array
            self.cn
                .par_iter_mut()
                .filter(|(_rec_pos, cn)| !cn.data.is_empty())
                .try_for_each(|(_rec_pos, cn): (&i32, &mut Cn4)| -> Result<(), Error> {
                    if let Some((validity, _invalid_byte_position, _invalid_byte_mask)) =
                        &mut cn.invalid_mask
                    {
                        if let Some(mask) = validity {
                            cn.data.set_validity(mask).with_context(|| {
                                format!(
                                    "failed applying invalid bits for channel {} from mask",
                                    cn.unique_name
                                )
                            })?;
                        }
                        *validity = None; // clean bitmask from Cn4 as present in arrow array
                    }
                    Ok(())
                })?;
        }
        Ok(())
    }
}

/// Cg4 blocks and linked blocks parsing
pub fn parse_cg4(
    rdr: &mut SymBufReader<&File>,
    target: i64,
    mut position: i64,
    sharable: &mut SharableBlocks,
    record_id_size: u8,
) -> Result<(HashMap<u64, Cg4>, i64, usize, usize)> {
    let mut cg: HashMap<u64, Cg4> = HashMap::new();
    let mut n_cg: usize = 0;
    let mut n_cn: usize = 0;
    if target != 0 {
        let (mut cg_struct, pos, num_cn) =
            parse_cg4_block(rdr, target, position, sharable, record_id_size)?;
        position = pos;
        let mut next_pointer = cg_struct.block.cg_cg_next;
        cg_struct.record_length += record_id_size as u32 + cg_struct.block.cg_inval_bytes;
        cg.insert(cg_struct.block.cg_record_id, cg_struct);
        n_cg += 1;
        n_cn += num_cn;

        while next_pointer != 0 {
            let (mut cg_struct, pos, num_cn) =
                parse_cg4_block(rdr, next_pointer, position, sharable, record_id_size)?;
            position = pos;
            cg_struct.record_length += record_id_size as u32 + cg_struct.block.cg_inval_bytes;
            next_pointer = cg_struct.block.cg_cg_next;
            cg.insert(cg_struct.block.cg_record_id, cg_struct);
            n_cg += 1;
            n_cn += num_cn;
        }
    }
    Ok((cg, position, n_cg, n_cn))
}

/// Cn4 Channel block struct
#[derive(Debug, PartialEq, Clone)]
#[binrw]
#[br(little)]
#[repr(C)]
pub struct Cn4Block {
    /// ##CN
    // cn_id: [u8; 4],
    /// reserved
    // reserved: [u8; 4],
    /// Length of block in bytes
    // pub cn_len: u64,
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
            // cn_id: [35, 35, 67, 78], // ##CN
            // reserved: [0; 4],
            // cn_len: 160,
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
#[derive(Debug, Default)]
#[repr(C)]
pub struct Cn4 {
    /// short header
    pub header: Blockheader4Short,
    /// CN Block without short header
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
    /// List size: 1 for normal primitive, 2 for complex, pnd for arrays
    pub list_size: usize,
    // Shape of array
    pub shape: (Vec<usize>, Order),
    /// optional invalid mask array, invalid byte position in record, invalid byte mask
    pub invalid_mask: Option<(Option<BooleanBufferBuilder>, usize, u8)>,
}

impl Clone for Cn4 {
    fn clone(&self) -> Self {
        let mut invalid_mask: Option<(Option<BooleanBufferBuilder>, usize, u8)> = None;
        if let Some((boolean_buffer, byte_position, byte_mask)) = &self.invalid_mask {
            let mut boolean_buffer_builder: Option<BooleanBufferBuilder> = None;
            if let Some(buffer) = boolean_buffer {
                let mut new_boolean_buffer_builder = BooleanBufferBuilder::new(buffer.len());
                new_boolean_buffer_builder.append_buffer(&buffer.finish_cloned());
                boolean_buffer_builder = Some(new_boolean_buffer_builder);
            }
            invalid_mask = Some((boolean_buffer_builder, *byte_position, *byte_mask));
        }
        Self {
            header: self.header,
            block: self.block.clone(),
            unique_name: self.unique_name.clone(),
            block_position: self.block_position,
            pos_byte_beg: self.pos_byte_beg,
            n_bytes: self.n_bytes,
            composition: self.composition.clone(),
            data: ChannelData::default(),
            endian: self.endian,
            list_size: self.list_size,
            shape: self.shape.clone(),
            invalid_mask,
        }
    }
}

/// hashmap's key is bit position in record, value Cn4
pub(crate) type CnType = HashMap<i32, Cn4>;

/// record layout type : record_id_size: u8, cg_data_bytes: u32, cg_inval_bytes: u32
type RecordLayout = (u8, u32, u32);

/// creates recursively in the channel group the CN blocks and all its other linked blocks (CC, MD, TX, CA, etc.)
pub fn parse_cn4(
    rdr: &mut SymBufReader<&File>,
    target: i64,
    mut position: i64,
    sharable: &mut SharableBlocks,
    record_layout: RecordLayout,
    cg_cycle_count: u64,
) -> Result<(CnType, i64, usize, i32)> {
    let mut cn: CnType = HashMap::new();
    let mut n_cn: usize = 0;
    let mut first_rec_pos: i32 = 0;
    let (record_id_size, _cg_data_bytes, _cg_inval_bytes) = record_layout;
    if target != 0 {
        let (cn_struct, pos, n_cns, cns) = parse_cn4_block(
            rdr,
            target,
            position,
            sharable,
            record_layout,
            cg_cycle_count,
        )?;
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
                record_layout,
                cg_cycle_count,
            )?;
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
    Ok((cn, position, n_cn, first_rec_pos))
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
        header: default_short_header(BlockType::CN),
        block,
        unique_name: String::from("ms"),
        block_position,
        pos_byte_beg,
        n_bytes: 2,
        composition: None,
        data: ChannelData::UInt16(UInt16Builder::new()),
        endian: false,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        invalid_mask: None,
    };
    let block = Cn4Block {
        cn_links: 8,
        cn_byte_offset: cn_byte_offset + 2,
        cn_bit_count: 6,
        ..Default::default()
    };
    let min = Cn4 {
        header: default_short_header(BlockType::CN),
        block,
        unique_name: String::from("min"),
        block_position,
        pos_byte_beg,
        n_bytes: 1,
        composition: None,
        data: ChannelData::UInt8(UInt8Builder::new()),
        endian: false,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        invalid_mask: None,
    };
    let block = Cn4Block {
        cn_links: 8,
        cn_byte_offset: cn_byte_offset + 3,
        cn_bit_count: 5,
        ..Default::default()
    };
    let hour = Cn4 {
        header: default_short_header(BlockType::CN),
        block,
        unique_name: String::from("hour"),
        block_position,
        pos_byte_beg,
        n_bytes: 1,
        composition: None,
        data: ChannelData::UInt8(UInt8Builder::new()),
        endian: false,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        invalid_mask: None,
    };
    let block = Cn4Block {
        cn_links: 8,
        cn_byte_offset: cn_byte_offset + 4,
        cn_bit_count: 5,
        ..Default::default()
    };
    let day = Cn4 {
        header: default_short_header(BlockType::CN),
        block,
        unique_name: String::from("day"),
        block_position,
        pos_byte_beg,
        n_bytes: 1,
        composition: None,
        data: ChannelData::UInt8(UInt8Builder::new()),
        endian: false,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        invalid_mask: None,
    };
    let block = Cn4Block {
        cn_links: 8,
        cn_byte_offset: cn_byte_offset + 5,
        cn_bit_count: 6,
        ..Default::default()
    };
    let month = Cn4 {
        header: default_short_header(BlockType::CN),
        block,
        unique_name: String::from("month"),
        block_position,
        pos_byte_beg,
        n_bytes: 1,
        composition: None,
        data: ChannelData::UInt8(UInt8Builder::new()),
        endian: false,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        invalid_mask: None,
    };
    let block = Cn4Block {
        cn_links: 8,
        cn_byte_offset: cn_byte_offset + 6,
        cn_bit_count: 7,
        ..Default::default()
    };
    let year = Cn4 {
        header: default_short_header(BlockType::CN),
        block,
        unique_name: String::from("year"),
        block_position,
        pos_byte_beg,
        n_bytes: 1,
        composition: None,
        data: ChannelData::UInt8(UInt8Builder::new()),
        endian: false,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        invalid_mask: None,
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
        header: default_short_header(BlockType::CN),
        block,
        unique_name: String::from("ms"),
        block_position,
        pos_byte_beg,
        n_bytes: 4,
        composition: None,
        data: ChannelData::UInt32(UInt32Builder::new()),
        endian: false,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        invalid_mask: None,
    };
    let block = Cn4Block {
        cn_links: 8,
        cn_byte_offset: cn_byte_offset + 4,
        cn_bit_count: 16,
        ..Default::default()
    };
    let days: Cn4 = Cn4 {
        header: default_short_header(BlockType::CN),
        block,
        unique_name: String::from("day"),
        block_position,
        pos_byte_beg,
        n_bytes: 2,
        composition: None,
        data: ChannelData::UInt16(UInt16Builder::new()),
        endian: false,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        invalid_mask: None,
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
    /// Returns the channel source name
    fn get_cn_source_name(&self, sharable: &SharableBlocks) -> Result<Option<String>> {
        let si = sharable.si.get(&self.block.cn_si_source);
        match si {
            Some(block) => Ok(block.get_si_source_name(sharable)?),
            None => Ok(None),
        }
    }
    /// Returns the channel source path
    fn get_cn_source_path(&self, sharable: &SharableBlocks) -> Result<Option<String>> {
        let si = sharable.si.get(&self.block.cn_si_source);
        match si {
            Some(block) => Ok(block.get_si_path_name(sharable)?),
            None => Ok(None),
        }
    }
}

/// Channel block parser
fn parse_cn4_block(
    rdr: &mut SymBufReader<&File>,
    target: i64,
    mut position: i64,
    sharable: &mut SharableBlocks,
    record_layout: RecordLayout,
    cg_cycle_count: u64,
) -> Result<(Cn4, i64, usize, CnType)> {
    let (record_id_size, _cg_data_bytes, cg_inval_bytes) = record_layout;
    let mut n_cn: usize = 1;
    let mut cns: HashMap<i32, Cn4> = HashMap::new();
    let (mut block, cnheader, pos) = parse_block_short(rdr, target, position)?;
    position = pos;
    let block: Cn4Block = block
        .read_le()
        .context("Could not read buffer into Cn4Block struct")?;

    let pos_byte_beg = block.cn_byte_offset + record_id_size as u32;
    let n_bytes = calc_n_bytes_not_aligned(block.cn_bit_count + (block.cn_bit_offset as u32));
    let invalid_mask: Option<(Option<BooleanBufferBuilder>, usize, u8)> = if cg_inval_bytes != 0 {
        let invalid_byte_position = (block.cn_inval_bit_pos >> 3) as usize;
        let invalid_byte_mask = 1 << (block.cn_inval_bit_pos & 0x07);
        let mut buffer = BooleanBufferBuilder::new(cg_cycle_count as usize);
        buffer.advance(cg_cycle_count as usize);
        Some((Some(buffer), invalid_byte_position, invalid_byte_mask))
    } else {
        None
    };

    // Reads TX name
    position = read_meta_data(rdr, sharable, block.cn_tx_name, position, BlockType::CN)?;
    let name: String = if let Some(n) = sharable.get_tx(block.cn_tx_name)? {
        n
    } else {
        String::new()
    };

    // Reads unit
    position = read_meta_data(rdr, sharable, block.cn_md_unit, position, BlockType::CN)?;

    // Reads CC
    let cc_pointer = block.cn_cc_conversion;
    if (cc_pointer != 0) && !sharable.cc.contains_key(&cc_pointer) {
        let (cc_block, _header, pos) = parse_block_short(rdr, cc_pointer, position)?;
        position = pos;
        position = read_cc(rdr, &cc_pointer, position, cc_block, sharable)?;
    }

    // Reads MD
    position = read_meta_data(rdr, sharable, block.cn_md_comment, position, BlockType::CN)?;

    //Reads SI
    let si_pointer = block.cn_si_source;
    if (si_pointer != 0) && !sharable.si.contains_key(&si_pointer) {
        let (mut si_block, _header, pos) = parse_block_short(rdr, si_pointer, position)?;
        position = pos;
        let si_block: Si4Block = si_block
            .read_le()
            .context("Could into read buffer into Si4Block struct")?;
        position = read_meta_data(rdr, sharable, si_block.si_tx_name, position, BlockType::SI)?;
        position = read_meta_data(rdr, sharable, si_block.si_tx_path, position, BlockType::SI)?;
        sharable.si.insert(si_pointer, si_block);
    }

    //Reads CA or composition
    let compo: Option<Composition>;
    let list_size: usize;
    let shape: (Vec<usize>, Order);
    if block.cn_composition != 0 {
        let (co, pos, array_size, s, n_cns, cnss) = parse_composition(
            rdr,
            block.cn_composition,
            position,
            sharable,
            record_layout,
            cg_cycle_count,
        )
        .context("Failed reading composition")?;
        shape = s;
        // list size calculation
        if block.cn_data_type == 15 | 16 {
            //complex
            list_size = 2 * array_size;
        } else {
            list_size = array_size;
        }
        compo = Some(co);
        position = pos;
        n_cn += n_cns;
        cns = cnss;
    } else {
        compo = None;
        shape = (vec![1], Order::RowMajor);
        // list size calculation
        if block.cn_data_type == 15 | 16 {
            //complex
            list_size = 2;
        } else {
            list_size = 1;
        }
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
        header: cnheader,
        block,
        unique_name: name,
        block_position: target,
        pos_byte_beg,
        n_bytes,
        composition: compo,
        data: data_type_init(cn_type, data_type, n_bytes, list_size)?,
        endian,
        list_size,
        shape,
        invalid_mask,
    };

    Ok((cn_struct, position, n_cn, cns))
}

/// reads pointed TX or CC Block(s) pointed by cc_ref in CCBlock
fn read_cc(
    rdr: &mut SymBufReader<&File>,
    target: &i64,
    mut position: i64,
    mut block: Cursor<Vec<u8>>,
    sharable: &mut SharableBlocks,
) -> Result<i64> {
    let cc_block: Cc4Block = block
        .read_le()
        .context("Could nto read buffer into Cc4Block struct")?;
    position = read_meta_data(rdr, sharable, cc_block.cc_md_unit, position, BlockType::CC)?;
    position = read_meta_data(rdr, sharable, cc_block.cc_tx_name, position, BlockType::CC)?;

    for pointer in &cc_block.cc_ref {
        if !sharable.cc.contains_key(pointer)
            && !sharable.md_tx.contains_key(pointer)
            && *pointer != 0
        {
            let (ref_block, header, _pos) = parse_block_short(rdr, *pointer, position)?;
            position = pointer + header.hdr_len as i64;
            if "##TX".as_bytes() == header.hdr_id {
                // TX Block
                position = read_meta_data(rdr, sharable, *pointer, position, BlockType::CC)?
            } else {
                // CC Block
                position = read_cc(rdr, pointer, position, ref_block, sharable)?;
            }
        }
    }
    sharable.cc.insert(*target, cc_block);
    Ok(position)
}

/// Cc4 Channel Conversion block struct
#[derive(Debug, Clone)]
#[binrw]
#[br(little)]
#[allow(dead_code)]
#[repr(C)]
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

/// Cc Values can be either a float or Uint64
#[derive(Debug, Clone)]
#[binrw]
#[br(little, import(count: u16, cc_type: u8))]
#[repr(C)]
pub enum CcVal {
    #[br(pre_assert(cc_type < 11))]
    Real(#[br(count = count)] Vec<f64>),

    #[br(pre_assert(cc_type == 11))]
    Uint(#[br(count = count)] Vec<u64>),
}

/// Si4 Source Information block struct
#[derive(Debug, PartialEq, Eq, Default, Copy, Clone)]
#[binrw]
#[br(little)]
#[repr(C)]
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
    /// returns the source name
    fn get_si_source_name(&self, sharable: &SharableBlocks) -> Result<Option<String>> {
        sharable.get_tx(self.si_tx_name)
    }
    /// returns the source path
    fn get_si_path_name(&self, sharable: &SharableBlocks) -> Result<Option<String>> {
        sharable.get_tx(self.si_tx_path)
    }
}

/// Ca4 Channel Array block struct
#[derive(Debug, PartialEq, Clone)]
#[repr(C)]
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
    /// [Π N(d) or empty] Only present for storage type "DG template". List of links to data blocks (DTBLOCK/DLBLOCK) for each element in case of "DG template" storage (ca_storage = 2). A link in this list may only be NIL if the cycle count of the respective element is 0: ca_data\[k\] = NIL => ca_cycle_count\[k\] = 0 The links are stored line-oriented, i.e. element k uses ca_data\[k\] (see explanation below). The size of the list must be equal to Π N(d), i.e. to the product of the number of elements per dimension N(d) over all dimensions D. Note: link ca_data\[0\] must be equal to dg_data link of the parent DGBLOCK.
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
    /// [Dx3 or empty] Only present if "axis" flag (bit 4) is set and "fixed axes flag" (bit 5) is not set. References to channels for scaling axis of respective dimension (can be NIL). Each reference is a link triple with pointer to parent DGBLOCK, parent CGBLOCK and CNBLOCK for the channel (either all three links are assigned or NIL). Thus the links have the following order: DGBLOCK for axis of dimension 1 CGBLOCK for axis of dimension 1 CNBLOCK for axis of dimension 1 … DGBLOCK for axis of dimension D CGBLOCK for axis of dimension D CNBLOCK for axis of dimension D Each referenced channel must be an array of type "axis". The maximum number of elements of each axis (ca_dim_size\[0\] in axis) must be equal to the maximum number of elements of respective dimension d in "look-up" array (ca_dim_size[d-1]).
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
        }
    }
}

/// Channel Array block structure, only members section, links section structure complex
#[derive(Debug, Clone)]
#[binrw]
#[br(little)]
#[repr(C)]
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

/// Channel Array block parser
fn parse_ca_block(
    ca_block: &mut Cursor<Vec<u8>>,
    block_header: Blockheader4,
    cg_cycle_count: u64,
) -> Result<(Ca4Block, (Vec<usize>, Order), usize, usize), Error> {
    //Reads members first
    ca_block.set_position(block_header.hdr_links * 8); // change buffer position after links section
    let ca_members: Ca4BlockMembers = ca_block
        .read_le()
        .context("Coudl tno read buffer into CaBlockMembers struct")?;
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

    let mut val = vec![0.0f64; snd];
    let ca_axis_value: Option<Vec<f64>> = if (ca_members.ca_flags & 0b100000) > 0 {
        ca_block
            .read_f64_into::<LittleEndian>(&mut val)
            .context("Could not read ca_axis_value")?;
        Some(val)
    } else {
        None
    };

    let mut val = vec![0u64; pnd];
    let ca_cycle_count: Option<Vec<u64>> = if ca_members.ca_storage >= 1 {
        ca_block
            .read_u64_into::<LittleEndian>(&mut val)
            .context("Could not read ca_cycle_count")?;
        Some(val)
    } else {
        None
    };

    // Reads links
    ca_block.set_position(0); // change buffer position to beginning of links section

    let ca_composition: i64 = ca_block
        .read_i64::<LittleEndian>()
        .context("Could not read ca_composition")?;

    let mut val = vec![0i64; pnd];
    let ca_data: Option<Vec<i64>> = if ca_members.ca_storage == 2 {
        ca_block
            .read_i64_into::<LittleEndian>(&mut val)
            .context("Could not read ca_storage")?;
        Some(val)
    } else {
        None
    };

    let mut val = vec![0i64; (ca_members.ca_ndim * 3) as usize];
    let ca_dynamic_size: Option<Vec<i64>> = if (ca_members.ca_flags & 0b1) > 0 {
        ca_block
            .read_i64_into::<LittleEndian>(&mut val)
            .context("Could not read ca_dynamic_size")?;
        Some(val)
    } else {
        None
    };

    let mut val = vec![0i64; (ca_members.ca_ndim * 3) as usize];
    let ca_input_quantity: Option<Vec<i64>> = if (ca_members.ca_flags & 0b10) > 0 {
        ca_block
            .read_i64_into::<LittleEndian>(&mut val)
            .context("Could not read ca_input_quantity")?;
        Some(val)
    } else {
        None
    };

    let mut val = vec![0i64; 3];
    let ca_output_quantity: Option<Vec<i64>> = if (ca_members.ca_flags & 0b100) > 0 {
        ca_block
            .read_i64_into::<LittleEndian>(&mut val)
            .context("Could not read ca_output_quantity")?;
        Some(val)
    } else {
        None
    };

    let mut val = vec![0i64; 3];
    let ca_comparison_quantity: Option<Vec<i64>> = if (ca_members.ca_flags & 0b1000) > 0 {
        ca_block
            .read_i64_into::<LittleEndian>(&mut val)
            .context("Could not read ca_comparison_quantity")?;
        Some(val)
    } else {
        None
    };

    let mut val = vec![0i64; ca_members.ca_ndim as usize];
    let ca_cc_axis_conversion: Option<Vec<i64>> = if (ca_members.ca_flags & 0b10000) > 0 {
        ca_block
            .read_i64_into::<LittleEndian>(&mut val)
            .context("Could not read ca_cc_axis_conversion")?;
        Some(val)
    } else {
        None
    };

    let mut val = vec![0i64; (ca_members.ca_ndim * 3) as usize];
    let ca_axis: Option<Vec<i64>> =
        if ((ca_members.ca_flags & 0b10000) > 0) & ((ca_members.ca_flags & 0b100000) > 0) {
            ca_block
                .read_i64_into::<LittleEndian>(&mut val)
                .context("Could not read ca_axis")?;
            Some(val)
        } else {
            None
        };

    Ok((
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
        },
        shape,
        snd,
        pnd,
    ))
}

/// contains composition blocks (CN or CA)
/// can optionaly point to another composition
#[derive(Debug, Clone)]
#[repr(C)]
pub struct Composition {
    pub block: Compo,
    pub compo: Option<Box<Composition>>,
}

/// enum allowing to nest CA or CN blocks for a compostion
#[derive(Debug, Clone)]
#[repr(C)]
pub enum Compo {
    CA(Box<Ca4Block>),
    CN(Box<Cn4>),
}

/// parses CN (structure) of CA (Array) blocks
/// CN (structures of composed channels )and CA (array of arrays) blocks can be nested or vene CA and CN nested and mixed: this is not supported, very complicated
fn parse_composition(
    rdr: &mut SymBufReader<&File>,
    target: i64,
    mut position: i64,
    sharable: &mut SharableBlocks,
    record_layout: RecordLayout,
    cg_cycle_count: u64,
) -> Result<(Composition, i64, usize, (Vec<usize>, Order), usize, CnType)> {
    let (mut block, block_header, pos) =
        parse_block(rdr, target, position).context("Failed parsing composition header block")?;
    position = pos;
    let array_size: usize;
    let mut cns: CnType;
    let mut n_cn: usize = 0;

    if block_header.hdr_id == "##CA".as_bytes() {
        // Channel Array
        let (block, mut shape, _snd, array_size) =
            parse_ca_block(&mut block, block_header, cg_cycle_count)
                .context("Failed parsing CA block")?;
        position = pos;
        let ca_compositon: Option<Box<Composition>>;
        if block.ca_composition != 0 {
            let (ca, pos, _array_size, s, n_cns, cnss) = parse_composition(
                rdr,
                block.ca_composition,
                position,
                sharable,
                record_layout,
                cg_cycle_count,
            )
            .context("Failed parsing composition block")?;
            shape = s;
            position = pos;
            cns = cnss;
            n_cn += n_cns;
            ca_compositon = Some(Box::new(ca));
        } else {
            ca_compositon = None;
            cns = HashMap::new();
        }
        Ok((
            Composition {
                block: Compo::CA(Box::new(block)),
                compo: ca_compositon,
            },
            position,
            array_size,
            shape,
            n_cn,
            cns,
        ))
    } else {
        // Channel structure
        array_size = 1;
        let (cnss, pos, n_cns, first_rec_pos) = parse_cn4(
            rdr,
            target,
            position,
            sharable,
            record_layout,
            cg_cycle_count,
        )?;
        position = pos;
        n_cn += n_cns;
        cns = cnss;
        let cn_composition: Option<Box<Composition>>;
        let cn_struct: Cn4 = if let Some(cn) = cns.get(&first_rec_pos) {
            cn.clone()
        } else {
            Cn4::default()
        };
        let shape: (Vec<usize>, Order);
        if cn_struct.block.cn_composition != 0 {
            let (cn, pos, _array_size, s, n_cns, cnss) = parse_composition(
                rdr,
                cn_struct.block.cn_composition,
                position,
                sharable,
                record_layout,
                cg_cycle_count,
            )?;
            shape = s;
            position = pos;
            n_cn += n_cns;
            cns.extend(cnss);
            cn_composition = Some(Box::new(cn));
        } else {
            cn_composition = None;
            shape = (vec![1], Order::RowMajor);
        }
        Ok((
            Composition {
                block: Compo::CN(Box::new(cn_struct)),
                compo: cn_composition,
            },
            position,
            array_size,
            shape,
            n_cn,
            cns,
        ))
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
    dg.iter_mut().for_each(|(dg_position, dg)| {
        dg.cg.iter_mut().for_each(|(record_id, cg)| {
            let gn = cg.get_cg_name(sharable);
            let gs = cg.get_cg_source_name(sharable);
            let gp = cg.get_cg_source_path(sharable);
            cg.cn.iter_mut().for_each(|(cn_record_position, cn)| {
                if channel_list.contains_key(&cn.unique_name) {
                    let mut changed: bool = false;
                    let space_char = String::from(" ");
                    // create unique channel name
                    if let Ok(Some(cs)) = cn.get_cn_source_name(sharable) {
                        cn.unique_name.push_str(&space_char);
                        cn.unique_name.push_str(&cs);
                        changed = true;
                    }
                    if let Ok(Some(cp)) = cn.get_cn_source_path(sharable) {
                        cn.unique_name.push_str(&space_char);
                        cn.unique_name.push_str(&cp);
                        changed = true;
                    }
                    if let Ok(Some(name)) = &gn {
                        cn.unique_name.push_str(&space_char);
                        cn.unique_name.push_str(name);
                        changed = true;
                    }
                    if let Ok(Some(source)) = &gs {
                        cn.unique_name.push_str(&space_char);
                        cn.unique_name.push_str(source);
                        changed = true;
                    }
                    if let Ok(Some(path)) = &gp {
                        cn.unique_name.push_str(&space_char);
                        cn.unique_name.push_str(path);
                        changed = true;
                    }
                    // No souce or path name to make channel unique
                    if !changed || channel_list.contains_key(&cn.unique_name) {
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
            });
        });
    });
    // identifying master channels
    let avg_ncn_per_cg = n_cn / n_cg;
    dg.iter_mut().for_each(|(_dg_position, dg)| {
        dg.cg.iter_mut().for_each(|(_record_id, cg)| {
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
            cg.cn.iter_mut().for_each(|(_cn_record_position, cn)| {
                cg_channel_list.insert(cn.unique_name.clone());
                // assigns master in channel_list
                if let Some(id) = channel_list.get_mut(&cn.unique_name) {
                    id.0.clone_from(&master_channel_name);
                }
            });
            cg.channel_names = cg_channel_list;
            cg.master_channel_name = master_channel_name;
        });
    });
    channel_list
}

/// DT4 Data List block struct, without the Id
#[derive(Debug, PartialEq, Eq, Default, Clone)]
#[binrw]
#[br(little)]
#[repr(C)]
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
#[derive(Debug, PartialEq, Eq, Default, Clone)]
#[binrw]
#[br(little)]
#[repr(C)]
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
    /// Flags
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
) -> Result<(Dl4Block, i64)> {
    rdr.seek_relative(target - position)
        .context("Could not reach position to read Dl4Block")?;
    let block: Dl4Block = rdr
        .read_le()
        .context("Could not read into Dl4Block struct")?;
    position = target + block.dl_len as i64;
    Ok((block, position))
}

/// parses DZBlock
pub fn parse_dz(rdr: &mut BufReader<&File>) -> Result<(Vec<u8>, Dz4Block)> {
    let block: Dz4Block = rdr
        .read_le()
        .context("Could not read into Dz4Block struct")?;
    let mut buf = vec![0u8; block.dz_data_length as usize];
    rdr.read_exact(&mut buf).context("Could not read Dz data")?;
    let mut data: Vec<u8>;
    let checksum: Option<u32>;
    (data, checksum) = decompress(&buf, Format::Zlib).expect("Could not decompress data");
    if Some(Adler32::from_buf(&data).finish()) != checksum {
        return Err(anyhow!("Checksum not ok"));
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
    Ok((data, block))
}

/// DZ4 Data List block struct
#[derive(Debug, PartialEq, Eq, Clone)]
#[binrw]
#[br(little)]
#[repr(C)]
pub struct Dz4Block {
    //header
    // dz_id: [u8; 4],  // ##DZ
    reserved: [u8; 4], // reserved
    /// Length of block in bytes
    pub len: u64,
    dz_links: u64, // # of links
    // links
    // members
    /// "DT", "SD", "RD" or "DV", "DI", "RV", "RI"
    pub dz_org_block_type: [u8; 2],
    /// Zip algorithm, 0 deflate, 1 transpose + deflate
    dz_zip_type: u8,
    /// reserved
    dz_reserved: u8,
    /// Zip algorithm parameter
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
#[derive(Debug, PartialEq, Eq, Clone)]
#[binrw]
#[br(little)]
#[repr(C)]
pub struct Ld4Block {
    // header
    // ld_id: [u8; 4],  // ##LD
    reserved: [u8; 4], // reserved
    /// Length of block in bytes
    pub ld_len: u64,
    /// # of links
    pub ld_n_links: u64,
    // links
    /// next ld block
    pub ld_next: i64,
    /// links
    #[br(if(ld_n_links > 1), little, count = ld_n_links - 1)]
    pub ld_links: Vec<i64>,
    // members
    /// Flags
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
            ld_next: 0,
            ld_links: vec![],
            ld_flags: 0,
            ld_count: 1,
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
        self.ld_next
    }
    /// Data block positions
    pub fn ld_data(&self) -> Vec<i64> {
        if (1u32 << 31) & self.ld_flags > 0 {
            self.ld_links.iter().step_by(2).copied().collect()
        } else {
            self.ld_links.clone()
        }
    }
    /// Invalid data block positions
    pub fn ld_invalid_data(&self) -> Vec<i64> {
        if (1u32 << 31) & self.ld_flags > 0 {
            self.ld_links.iter().skip(1).step_by(2).copied().collect()
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
) -> Result<(Ld4Block, i64)> {
    rdr.seek_relative(target - position)
        .context("Could not reach Ld4Block position")?;
    let block: Ld4Block = rdr
        .read_le()
        .context("Could not read buffer into Ld4Block struct")?;
    position = target + block.ld_len as i64;
    Ok((block, position))
}

/// HL4 Data List block struct
#[derive(Debug, PartialEq, Eq, Default, Clone)]
#[binrw]
#[br(little)]
#[repr(C)]
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
