//! Parsing of file metadata into MdfInfo4 struct
//!
//! This module contains MDF4 block types and parsing logic, split into
//! sub-modules per block type for maintainability.

// Sub-modules
pub mod at_block;
mod block_header;
mod ca_block;
mod cc_block;
mod cg_block;
mod ch_block;
mod cn_block;
mod composition;
mod data_block;
mod dg_block;
mod ev_block;
mod fh_block;
pub mod finalize;
mod hd_block;
mod metadata;
pub mod scanner;
mod si_block;
mod sr_block;

// Re-exports for backward compatibility
pub use at_block::*;
pub use block_header::*;
pub use ca_block::*;
pub use cc_block::*;
pub use cg_block::*;
pub use ch_block::*;
pub use cn_block::*;
pub use composition::*;
pub use data_block::*;
pub use dg_block::*;
pub use ev_block::*;
pub use fh_block::*;
pub use hd_block::*;
pub use metadata::*;
pub use si_block::*;
pub use sr_block::*;

use anyhow::{Context, Error, Result};
use std::collections::{BTreeMap, HashMap, HashSet};

/// ChannelId : (Option<master_channelname>, dg_pos, (cg_pos, rec_id), (cn_pos, rec_pos))
pub(crate) type ChannelId = (Option<String>, i64, (i64, u64), (i64, i32));
pub(crate) type ChannelNamesSet = HashMap<String, ChannelId>;
use std::fmt;
use std::sync::Arc;

use arrow::array::Array;

use crate::data_holder::channel_data::{ChannelData, try_from};
use crate::data_holder::tensor_arrow::Order;
use crate::mdfinfo::IdBlock;
use crate::mdfreader::{DataSignature, MasterSignature};

/// MdfInfo4 is the struct holding whole metadata of mdf4.x files
/// * blocks with unique links are at top level like attachment, events and file history
/// * sharable blocks (most likely referenced multiple times and shared by several blocks)
///   that are in sharable fields and holds CC, SI, TX and MD blocks
/// * the dg fields nests cg itself nesting cn blocks and eventually compositions
///   (other cn or ca blocks) and conversion
/// * channel_names_set is the complete set of channel names contained in file
/// * in general the blocks are contained in HashMaps with key corresponding
///   to their position in the file
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
    /// channel hierarchy blocks
    pub ch: HashMap<i64, Ch4Block>,
    /// whether the file was marked as unfinalized
    pub is_unfinalized: bool,
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
            && let Some(dg) = self.dg.get(dg_pos)
            && let Some(cg) = dg.cg.get(rec_id)
            && let Some(cn) = cg.cn.get(rec_pos)
            && !cn.data.is_empty()
        {
            data = Some(&cn.data);
        }
        data
    }
    /// Returns the channel's unit string. If it does not exist, it is an empty string.
    pub fn get_channel_unit(&self, channel_name: &str) -> Result<Option<String>> {
        let mut unit: Option<String> = None;
        if let Some((_master, dg_pos, (_cg_pos, rec_id), (_cn_pos, rec_pos))) =
            self.get_channel_id(channel_name)
            && let Some(dg) = self.dg.get(dg_pos)
            && let Some(cg) = dg.cg.get(rec_id)
            && let Some(cn) = cg.cn.get(rec_pos)
        {
            unit = self.sharable.get_tx(cn.block.cn_md_unit)?;
        }
        Ok(unit)
    }
    /// Returns the channel's description. If it does not exist, it is an empty string
    pub fn get_channel_desc(&self, channel_name: &str) -> Result<Option<String>> {
        let mut desc: Option<String> = None;
        if let Some((_master, dg_pos, (_cg_pos, rec_id), (_cn_pos, rec_pos))) =
            self.get_channel_id(channel_name)
            && let Some(dg) = self.dg.get(dg_pos)
            && let Some(cg) = dg.cg.get(rec_id)
            && let Some(cn) = cg.cn.get(rec_pos)
        {
            desc = self.sharable.get_tx(cn.block.cn_md_comment)?;
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
            && let Some(dg) = self.dg.get(dg_pos)
            && let Some(cg) = dg.cg.get(rec_id)
            && let Some(cn) = cg.cn.get(rec_pos)
        {
            master_type = cn.block.cn_sync_type;
        }
        master_type
    }
    /// returns the set of channel names
    pub fn get_channel_names_set(&self) -> HashSet<String> {
        self.channel_names_set.keys().cloned().collect()
    }
    /// returns the set of channel names that are in same channel group as input channel name
    pub fn get_channel_names_cg_set(&self, channel_name: &str) -> HashSet<String> {
        if let Some((_master, dg_pos, (_cg_pos, rec_id), (_cn_pos, _rec_pos))) =
            self.get_channel_id(channel_name)
        {
            let mut channel_list = HashSet::new();
            if let Some(dg) = self.dg.get(dg_pos)
                && let Some(cg) = dg.cg.get(rec_id)
            {
                channel_list.clone_from(&cg.channel_names);
            }
            channel_list
        } else {
            HashSet::new()
        }
    }
    /// returns a hashmap for which master channel names are keys and values its corresponding set of channel names
    pub fn get_master_channel_names_set(&self) -> HashMap<Option<String>, HashSet<String>> {
        let mut map: HashMap<Option<String>, HashSet<String>> = HashMap::new();
        for dg in self.dg.values() {
            for cg in dg.cg.values() {
                map.entry(cg.master_channel_name.clone())
                    .or_default()
                    .extend(cg.channel_names.iter().cloned());
            }
        }
        map
    }
    /// empty the channels' ndarray
    pub fn clear_channel_data_from_memory(&mut self, channel_names: HashSet<String>) -> Result<()> {
        for channel_name in channel_names {
            if let Some((_master, dg_pos, (_cg_pos, rec_id), (_cn_pos, rec_pos))) =
                self.channel_names_set.get_mut(&channel_name)
                && let Some(dg) = self.dg.get_mut(dg_pos)
                && let Some(cg) = dg.cg.get_mut(rec_id)
                && let Some(cn) = cg.cn.get_mut(rec_pos)
                && !cn.data.is_empty()
            {
                cn.data =
                    cn.data
                        .zeros(cn.block.cn_data_type, 0, 0, (Vec::new(), Order::RowMajor))?;
            }
        }
        Ok(())
    }
    /// replaces each channel's data with the slice [start_idx, end_idx) for named channels
    pub fn slice_channels(
        &mut self,
        channel_names: &HashSet<String>,
        start_idx: usize,
        end_idx: usize,
    ) -> Result<()> {
        for channel_name in channel_names {
            if let Some((_master, dg_pos, (_cg_pos, rec_id), (_cn_pos, rec_pos))) =
                self.channel_names_set.get(channel_name)
                && let Some(dg) = self.dg.get_mut(dg_pos)
                && let Some(cg) = dg.cg.get_mut(rec_id)
                && let Some(cn) = cg.cn.get_mut(rec_pos)
                && !cn.data.is_empty()
            {
                cn.data = cn.data.slice_range(start_idx, end_idx)?;
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
            ch: HashMap::new(),
            is_unfinalized: false,
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
        let mut cg_block = Cg4Block::default();
        cg_block.cg_cycle_count = data_signature.len as u64;
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
            endian: Endianness::from(machine_endian),
            block_position: cn_pos,
            pos_byte_beg: 0,
            n_bytes,
            composition,
            list_size,
            shape: data_signature.shape,
            invalid_mask: None,
            event_template: None,
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
            sr: Vec::new(),
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
            && let Some(dg) = self.dg.get_mut(dg_pos)
            && let Some(cg) = dg.cg.get_mut(rec_id)
        {
            cg.cn.remove(rec_pos);
            cg.channel_names.remove(channel_name);
            self.channel_names_set.remove(channel_name);
        }
    }
    /// Renames a channel's name in memory
    pub fn rename_channel(&mut self, channel_name: &str, new_name: &str) {
        if let Some((master, dg_pos, (cg_pos, rec_id), (cn_pos, rec_pos))) =
            self.channel_names_set.remove(channel_name)
            && let Some(dg) = self.dg.get_mut(&dg_pos)
            && let Some(cg) = dg.cg.get_mut(&rec_id)
            && let Some(cn) = cg.cn.get_mut(&rec_pos)
        {
            cn.unique_name = new_name.to_string();
            cg.channel_names.remove(channel_name);
            cg.channel_names.insert(new_name.to_string());
            if let Some(master_name) = &master
                && master_name == channel_name
            {
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

            self.channel_names_set.insert(
                new_name.to_string(),
                (master, dg_pos, (cg_pos, rec_id), (cn_pos, rec_pos)),
            );
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
            && let Some(dg) = self.dg.get_mut(dg_pos)
            && let Some(cg) = dg.cg.get_mut(rec_id)
            && let Some(cn) = cg.cn.get_mut(rec_pos)
        {
            cn.data = try_from(&data).context("failed converting dyn array to ChannelData")?;
        }

        Ok(())
    }
    /// Sets the channel unit in memory
    pub fn set_channel_unit(&mut self, channel_name: &str, unit: &str) {
        if let Some((_master, dg_pos, (_cg_pos, rec_id), (_cn_pos, rec_pos))) =
            self.channel_names_set.get(channel_name)
            && let Some(dg) = self.dg.get_mut(dg_pos)
            && let Some(cg) = dg.cg.get_mut(rec_id)
            && let Some(cn) = cg.cn.get_mut(rec_pos)
        {
            // hopefully never 2 times the same position
            let position = position_generator();
            self.sharable.create_tx(position, unit.to_string());
            cn.block.cn_md_unit = position;
        }
    }
    /// Sets the channel description in memory
    pub fn set_channel_desc(&mut self, channel_name: &str, desc: &str) {
        if let Some((_master, dg_pos, (_cg_pos, rec_id), (_cn_pos, rec_pos))) =
            self.channel_names_set.get(channel_name)
            && let Some(dg) = self.dg.get_mut(dg_pos)
            && let Some(cg) = dg.cg.get_mut(rec_id)
            && let Some(cn) = cg.cn.get_mut(rec_pos)
        {
            let position = position_generator();
            self.sharable.create_tx(position, desc.to_string());
            cn.block.cn_md_comment = position;
        }
    }
    /// Sets the channel's related master channel type in memory
    pub fn set_channel_master_type(&mut self, master_name: &str, master_type: u8) {
        if let Some((_master, dg_pos, (_cg_pos, rec_id), (_cn_pos, rec_pos))) =
            self.channel_names_set.get(master_name)
            && let Some(dg) = self.dg.get_mut(dg_pos)
            && let Some(cg) = dg.cg.get_mut(rec_id)
            && let Some(cn) = cg.cn.get_mut(rec_pos)
        {
            cn.block.cn_sync_type = master_type;
        }
    }
    /// list attachments
    pub fn list_attachments(&mut self) -> String {
        let mut output = String::new();
        for (key, (block, _embedded_data)) in &self.at {
            output.push_str(&format!(
                "position: {}, filename: {:?}, mimetype: {:?}, comment: {:?}\n ",
                key,
                self.sharable.get_tx(block.at_tx_filename),
                self.sharable.get_tx(block.at_tx_mimetype),
                self.sharable.get_md_comment(block.at_md_comment)
            ))
        }
        output
    }
    /// get embedded data in attachment for a block at position
    pub fn get_attachment_embedded_data(&self, position: i64) -> Option<Vec<u8>> {
        if let Some(at) = self.at.get(&position) {
            at.1.clone()
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
        for (key, (block, _data)) in &self.at {
            output.insert(*key, *block);
        }
        output
    }
    /// list file history entries
    pub fn list_file_history(&mut self) -> String {
        let mut output = String::new();
        for (i, fh) in self.fh.iter().enumerate() {
            output.push_str(&format!(
                "FH[{}]: {}, comment: {:?}\n",
                i,
                fh,
                self.sharable.get_md_comment(fh.fh_md_comment),
            ));
        }
        output
    }
    /// list events
    pub fn list_events(&mut self) -> String {
        let mut output = String::new();
        for (key, block) in &self.ev {
            output.push_str(&format!(
                "position: {}, name: {:?}, comment: {:?}, scope: {:?}, attachment references: {:?}, event type: {}\n",
                key,
                self.sharable.get_tx(block.ev_tx_name),
                self.sharable.get_md_comment(block.ev_md_comment),
                block.get_scope_links(),
                block.get_attachment_links(),
                block.ev_type,
            ))
        }
        output
    }
    /// list sample reduction blocks for all channel groups
    pub fn list_sample_reductions(&self) -> String {
        let mut output = String::new();
        for dg in self.dg.values() {
            for (rec_id, cg) in &dg.cg {
                if !cg.sr.is_empty() {
                    output.push_str(&format!(
                        "Channel group (rec_id={}): {} sample reduction(s)\n",
                        rec_id,
                        cg.sr.len()
                    ));
                    for (i, sr) in cg.sr.iter().enumerate() {
                        output.push_str(&format!(
                            "  SR[{}]: cycle_count={}, interval={}, sync_type={}, flags=0x{:02X}\n",
                            i,
                            sr.sr_cycle_count,
                            sr.sr_interval,
                            sr.get_sync_type_str(),
                            sr.sr_flags,
                        ));
                    }
                }
            }
        }
        output
    }
    /// get all sample reduction blocks across all channel groups
    /// Returns a vector of (dg_position, rec_id, sr_blocks) tuples
    pub fn get_sample_reduction_blocks(&self) -> Vec<(i64, u64, Vec<Sr4Block>)> {
        let mut result = Vec::new();
        for (&dg_pos, dg) in &self.dg {
            for (&rec_id, cg) in &dg.cg {
                if !cg.sr.is_empty() {
                    result.push((dg_pos, rec_id, cg.sr.clone()));
                }
            }
        }
        result
    }
    /// list source information blocks
    pub fn list_source_information(&self) -> String {
        let mut output = String::new();
        for (key, block) in &self.sharable.si {
            output.push_str(&format!(
                "position: {}, name: {:?}, path: {:?}, type: {}, bus: {}\n",
                key,
                self.sharable.get_tx(block.si_tx_name),
                self.sharable.get_tx(block.si_tx_path),
                block.get_type_str(),
                block.get_bus_type_str(),
            ))
        }
        output
    }
    /// get all source information blocks
    pub fn get_source_information_blocks(&self) -> HashMap<i64, Si4Block> {
        self.sharable.si.clone()
    }
    /// get event block from its position
    pub fn get_event_block(&self, position: i64) -> Option<Ev4Block> {
        self.ev.get(&position).cloned()
    }
    /// get all event blocks
    pub fn get_event_blocks(&self) -> HashMap<i64, Ev4Block> {
        self.ev.clone()
    }
    /// Get a channel hierarchy block from its position
    pub fn get_channel_hierarchy_block(&self, position: i64) -> Option<Ch4Block> {
        self.ch.get(&position).cloned()
    }
    /// Get all channel hierarchy blocks
    pub fn get_channel_hierarchy_blocks(&self) -> HashMap<i64, Ch4Block> {
        self.ch.clone()
    }
    /// List channel hierarchy in a human-readable format
    pub fn list_channel_hierarchy(&self) -> String {
        let mut output = String::new();
        // Find root blocks (blocks not referenced as children or siblings by any other block)
        let mut non_root_positions: HashSet<i64> = HashSet::new();
        for block in self.ch.values() {
            if block.ch_ch_first > 0 {
                non_root_positions.insert(block.ch_ch_first);
            }
            if block.ch_ch_next > 0 {
                non_root_positions.insert(block.ch_ch_next);
            }
        }

        let mut roots: Vec<i64> = self
            .ch
            .keys()
            .filter(|pos| !non_root_positions.contains(pos))
            .copied()
            .collect();
        roots.sort();

        for root_pos in roots {
            self.format_hierarchy_level(&mut output, root_pos, 0);
        }
        output
    }
    /// Helper to format a hierarchy level recursively
    fn format_hierarchy_level(&self, output: &mut String, position: i64, depth: usize) {
        if let Some(block) = self.ch.get(&position) {
            let indent = "  ".repeat(depth);
            let name = self
                .sharable
                .get_tx(block.ch_tx_name)
                .ok()
                .flatten()
                .unwrap_or_else(|| "<unnamed>".to_string());

            output.push_str(&format!(
                "{}[{}] {} (elements={})\n",
                indent,
                block.get_type_str(),
                name,
                block.ch_element_count
            ));

            // List elements (each element is a DG/CG/CN triplet)
            for i in 0..block.ch_element_count as usize {
                let base_idx = i * 3;
                if base_idx + 2 < block.ch_element.len() {
                    let dg_pos = block.ch_element[base_idx];
                    let cg_pos = block.ch_element[base_idx + 1];
                    let cn_pos = block.ch_element[base_idx + 2];
                    output.push_str(&format!(
                        "{indent}  -> DG:{dg_pos} CG:{cg_pos} CN:{cn_pos}\n"
                    ));
                }
            }

            // Traverse children first
            if block.ch_ch_first > 0 {
                self.format_hierarchy_level(output, block.ch_ch_first, depth + 1);
            }

            // Then traverse siblings at same level
            if block.ch_ch_next > 0 {
                self.format_hierarchy_level(output, block.ch_ch_next, depth);
            }
        }
    }
    /// Returns a concise one-line summary of the MDF4 file
    pub fn summary(&self) -> String {
        let total_channels = self.channel_names_set.len();
        let total_dgs = self.dg.len();
        let total_events = self.ev.len();
        let total_attachments = self.at.len();
        format!(
            "MDF4 v{}: {} DGs, {} channels, {} events, {} attachments",
            self.id_block.id_ver, total_dgs, total_channels, total_events, total_attachments
        )
    }
    /// Formats the channel list with optional data preview (first/last values)
    /// If `show_data` is true, shows first and last values for channels with data
    pub fn format_channels(&self, show_data: bool) -> String {
        let mut output = String::new();
        for (master, list) in &self.get_master_channel_names_set() {
            if let Some(master_name) = master {
                output.push_str(&format!("\nMaster: {master_name}\n"));
            } else {
                output.push_str("\nWithout Master channel\n");
            }
            for channel in list {
                let unit = self.get_channel_unit(channel).ok().flatten();
                let desc = self.get_channel_desc(channel).ok().flatten();
                output.push_str(&format!("  {channel} "));
                if show_data
                    && let Some(data) = self.get_channel_data(channel)
                    && !data.is_empty()
                {
                    output.push_str(&format!("[{}] ", data.len()));
                }
                if let Some(u) = unit {
                    output.push_str(&format!("\"{u}\" "));
                }
                if let Some(d) = desc
                    && !d.is_empty()
                {
                    output.push_str(&format!("// {d}"));
                }
                output.push('\n');
            }
        }
        output
    }
    /// Formats header comments
    pub fn format_header_comments(&self) -> String {
        let mut output = String::new();
        if let Some(hd) = self.sharable.get_hd_comments(self.hd_block.hd_md_comment) {
            if let Some(tx) = &hd.tx {
                output.push_str(&format!("TX: {tx}\n"));
            }
            if let Some(ts) = &hd.time_source {
                output.push_str(&format!("time_source: {ts}\n"));
            }
            for (name, value) in &hd.constants {
                output.push_str(&format!("const {name}: {value}\n"));
            }
            for (name, value) in &hd.common_properties {
                output.push_str(&format!("{name}: {value}\n"));
            }
        }
        output
    }
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
        writeln!(f, "{}", self.summary())?;
        writeln!(f, "File: {}", self.file_name)?;
        writeln!(f, "{}", self.hd_block)?;
        let header_comments = self.format_header_comments();
        if !header_comments.is_empty() {
            writeln!(f, "{header_comments}")?;
        }
        write!(f, "{}", self.format_channels(false))
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
    let avg_ncn_per_cg = n_cn.checked_div(n_cg).unwrap_or(0);
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
