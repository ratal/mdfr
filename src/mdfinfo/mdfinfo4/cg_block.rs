//! Channel Group block (CGBLOCK) for MDF4 — spec section 6.5, Tables 20-21
//!
//! A CGBLOCK describes a group of channels with the same record layout (same record
//! length and record ID). Flags in Table 21 define special CG types: VLSD, VLSC, bus events.
use anyhow::{Context, Error, Result};
use arrow::array::{Array, ArrayRef, UInt32Array, UnionArray};
use arrow::buffer::ScalarBuffer;
use arrow::compute::take;
use arrow::datatypes::{Field, UnionFields};
use binrw::{BinReaderExt, binrw};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fmt::{self, Display};
use std::fs::File;
use std::io::{Cursor, Read};
use std::sync::Arc;

use crate::data_holder::channel_data::ChannelData;
use crate::mdfinfo::sym_buf_reader::SymBufReader;

use super::block_header::{
    parse_block_header_short, parse_block_short, read_meta_data, Blockheader4Short, SharableBlocks,
};
use super::metadata::BlockType;
use super::cn_block::{parse_cn4, Cn4, CnType};
use super::composition::Compo;

// Channel Group (CG) flags - cg_flags field (u16)
/// Bit 0: VLSD channel group (Variable Length Signal Data)
pub const CG_F_VLSD: u16 = 1 << 0;
/// Bit 4: Event signal group - channel group contains event signals
pub const CG_F_EVENT_SIGNAL_GROUP: u16 = 1 << 4;
/// Bit 5: VLSC channel group (contains VLSC channels, MDF 4.3)
pub const CG_F_VLSC: u16 = 1 << 5;
/// Bit 6: Raw sensor event channel group
pub const CG_F_RAW_SENSOR_EVENT: u16 = 1 << 6;
/// Bit 7: Protocol event channel group
pub const CG_F_PROTOCOL_EVENT: u16 = 1 << 7;
use super::si_block::Si4Block;
use super::sr_block::Sr4Block;

/// Cg4 Channel Group block struct
#[derive(Debug, Copy, Clone)]
#[binrw]
#[br(little)]
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

impl Cg4Block {
    /// Returns a string representation of the channel group flags
    pub fn get_flags_str(&self) -> String {
        let mut flags = Vec::new();
        if (self.cg_flags & CG_F_VLSD) != 0 {
            flags.push("VLSD");
        }
        if (self.cg_flags & CG_F_VLSC) != 0 {
            flags.push("VLSC");
        }
        if (self.cg_flags & CG_F_EVENT_SIGNAL_GROUP) != 0 {
            flags.push("EventSignal");
        }
        if (self.cg_flags & CG_F_RAW_SENSOR_EVENT) != 0 {
            flags.push("RawSensor");
        }
        if (self.cg_flags & CG_F_PROTOCOL_EVENT) != 0 {
            flags.push("ProtocolEvent");
        }
        if (self.cg_flags & 0b1000) != 0 {
            // Bit 3: Remote master
            flags.push("RemoteMaster");
        }
        if flags.is_empty() {
            "None".to_string()
        } else {
            flags.join("|")
        }
    }
}

impl Display for Cg4Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CG: rec_id={} cycles={} data_bytes={} inval_bytes={} flags={}",
            self.cg_record_id,
            self.cg_cycle_count,
            self.cg_data_bytes,
            self.cg_inval_bytes,
            self.get_flags_str()
        )
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
    // For VLSD/VLSC, cg_inval_bytes is the high part of VL data size, not invalidation bytes
    let inval_bytes_for_record = if (cg.cg_flags & (CG_F_VLSD | CG_F_VLSC)) != 0 {
        0
    } else {
        cg.cg_inval_bytes
    };
    let record_layout = (record_id_size, cg.cg_data_bytes, inval_bytes_for_record);

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

    // Parse Sample Reduction blocks if present
    let (sr_blocks, pos) = parse_sr4(rdr, cg.cg_sr_first, position)?;
    position = pos;

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
        sr: sr_blocks,
    };

    Ok((cg_struct, position, n_cn))
}

/// Parses the linked list of Sample Reduction blocks (SRBLOCK) starting from target
fn parse_sr4(
    rdr: &mut SymBufReader<&File>,
    target: i64,
    mut position: i64,
) -> Result<(Vec<Sr4Block>, i64)> {
    let mut sr_blocks: Vec<Sr4Block> = Vec::new();
    if target <= 0 {
        return Ok((sr_blocks, position));
    }

    let mut next = target;
    while next > 0 {
        // Read just the 16-byte header first to validate before allocating
        rdr.seek_relative(next - position)
            .context("Could not reach SR block header position")?;
        let header: Blockheader4Short =
            parse_block_header_short(rdr).context("Could not read SR block header")?;
        // Validate block ID is ##SR
        if &header.hdr_id != b"##SR" {
            position = next + 16;
            break;
        }
        // Now read the rest of the block
        let mut buf = vec![0u8; (header.hdr_len - 16) as usize];
        rdr.read_exact(&mut buf)
            .context("Could not read SR block body")?;
        position = next + header.hdr_len as i64;
        let mut block = Cursor::new(buf);
        let sr: Sr4Block = block
            .read_le()
            .context("Could not read buffer into Sr4Block struct")?;
        next = sr.sr_sr_next;
        sr_blocks.push(sr);
    }

    Ok((sr_blocks, position))
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
    /// Sample reduction blocks linked from cg_sr_first
    pub sr: Vec<Sr4Block>,
}

/// Cg4 implementations for extracting acquisition and source name and path
#[allow(dead_code)]
impl Cg4 {
    /// Returns true if this channel group is an event signal group (cg_flags bit 4 set).
    /// Event signal groups contain channels that store event data instead of regular signal data.
    /// The event structure is described by a template EVBLOCK in each event signal channel's cn_data link.
    pub fn is_event_signal_group(&self) -> bool {
        (self.block.cg_flags & CG_F_EVENT_SIGNAL_GROUP) != 0
    }

    /// Returns true if this channel group has sample reduction data available.
    /// Sample reduction data provides mean/min/max values for fast preview/graphical display.
    pub fn has_sample_reduction(&self) -> bool {
        !self.sr.is_empty()
    }

    /// Returns the number of sample reduction blocks available for this channel group.
    pub fn sample_reduction_count(&self) -> usize {
        self.sr.len()
    }

    /// Returns a reference to the sample reduction blocks.
    /// Each Sr4Block contains metadata about the reduction (interval, sync type, etc.)
    /// and a pointer (sr_data) to the actual reduction data.
    pub fn get_sample_reduction_blocks(&self) -> &[Sr4Block] {
        &self.sr
    }

    /// Returns the data bytes per record for this channel group.
    /// This is needed to decode sample reduction records.
    pub fn get_data_bytes(&self) -> u32 {
        self.block.cg_data_bytes
    }

    /// Returns the invalidation bytes per record for this channel group.
    /// Returns 0 for VLSD/VLSC channel groups where this field has a different meaning.
    pub fn get_inval_bytes(&self) -> u32 {
        if (self.block.cg_flags & (CG_F_VLSD | CG_F_VLSC)) != 0 {
            0
        } else {
            self.block.cg_inval_bytes
        }
    }

    /// Channel group acquisition name
    pub fn get_cg_name(&self, sharable: &SharableBlocks) -> Result<Option<String>> {
        sharable.get_tx(self.block.cg_tx_acq_name)
    }
    /// Channel group source name
    pub fn get_cg_source_name(&self, sharable: &SharableBlocks) -> Result<Option<String>> {
        let si = sharable.si.get(&self.block.cg_si_acq_source);
        match si {
            Some(block) => Ok(block.get_si_source_name(sharable)?),
            None => Ok(None),
        }
    }
    /// Channel group source path
    pub fn get_cg_source_path(&self, sharable: &SharableBlocks) -> Result<Option<String>> {
        let si = sharable.si.get(&self.block.cg_si_acq_source);
        match si {
            Some(block) => Ok(block.get_si_path_name(sharable)?),
            None => Ok(None),
        }
    }
    /// Computes the validity mask for each channel in the group
    /// clears out the common invalid bytes vector for the group at the end
    pub fn process_all_channel_invalid_bits(&mut self) -> Result<(), Error> {
        // For VLSD/VLSC, cg_inval_bytes is the high part of VL data size, not invalidation bytes
        if (self.block.cg_flags & (CG_F_VLSD | CG_F_VLSC)) != 0 {
            return Ok(());
        }
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

    /// Process Channel Variant (CV) compositions after data is loaded.
    /// For each channel with a CV composition, this method:
    /// 1. Reads the discriminator channel values
    /// 2. Maps discriminator values to option indices using cv_option_val
    /// 3. Merges option channel data based on the discriminator
    ///
    /// After processing, the parent channel (with CV composition) contains the merged data.
    pub fn process_channel_variants(&mut self) -> Result<(), Error> {
        // Find channels with CV composition and collect info
        let cv_channels: Vec<(i32, i64, Vec<i64>, Vec<u64>)> = self
            .cn
            .iter()
            .filter_map(|(rec_pos, cn)| {
                if let Some(composition) = &cn.composition
                    && let Compo::CV(cv_block) = &composition.block
                {
                    return Some((
                        *rec_pos,
                        cv_block.cv_cn_discriminator,
                        cv_block.cv_cn_option.clone(),
                        cv_block.cv_option_val.clone(),
                    ));
                }
                None
            })
            .collect();

        for (parent_rec_pos, discriminator_ptr, option_ptrs, option_vals) in cv_channels {
            // First pass: collect all needed data (immutable borrows complete before mutable)
            let discriminator_values: Vec<u64>;
            let option_data: Vec<Option<ChannelData>>;
            let option_names: Vec<String>;

            {
                // Find the discriminator channel by block_position
                let discriminator_cn = self
                    .cn
                    .values()
                    .find(|cn| cn.block_position == discriminator_ptr);

                let Some(disc_cn) = discriminator_cn else {
                    log::warn!(
                        "CV discriminator channel not found for block_position {}",
                        discriminator_ptr
                    );
                    continue;
                };

                // Get discriminator values as u64
                discriminator_values = match disc_cn.data.to_u64_vec() {
                    Some(v) => v,
                    None => {
                        log::warn!("CV discriminator channel has unsupported data type");
                        continue;
                    }
                };

                if discriminator_values.is_empty() {
                    continue;
                }

                // Collect option channel data and names in a single pass
                let (data_vec, names_vec): (Vec<Option<ChannelData>>, Vec<String>) = option_ptrs
                    .iter()
                    .map(
                        |ptr| match self.cn.values().find(|cn| cn.block_position == *ptr) {
                            Some(cn) => (Some(cn.data.clone()), cn.unique_name.clone()),
                            None => (None, String::new()),
                        },
                    )
                    .unzip();
                option_data = data_vec;
                option_names = names_vec;
            }
            // Immutable borrows end here

            // Build index mapping: discriminator value -> option index
            let val_to_option: std::collections::HashMap<u64, usize> = option_vals
                .iter()
                .enumerate()
                .map(|(idx, val)| (*val, idx))
                .collect();

            // Check if all option channels have the same data type
            let all_same_type = {
                let mut discriminants: Vec<std::mem::Discriminant<ChannelData>> = Vec::new();
                for data in option_data.iter().flatten() {
                    discriminants.push(std::mem::discriminant(data));
                }
                discriminants.windows(2).all(|w| w[0] == w[1])
            };

            if all_same_type {
                // All options have the same type: use existing merge path
                let template = option_data.iter().find_map(|o| o.clone());

                // Second pass: update parent channel (mutable borrow)
                if let Some(parent_cn) = self.cn.get_mut(&parent_rec_pos)
                    && let Some(tmpl) = template
                {
                    let merged_data = merge_variant_data_owned(
                        &discriminator_values,
                        &option_data,
                        &val_to_option,
                        &tmpl,
                    );

                    if let Some(data) = merged_data {
                        parent_cn.data = data;
                    }
                }
            } else {
                // Mixed types: build a dense UnionArray
                // Effective sample count is the minimum of discriminator and all option lengths
                let n_samples = {
                    let mut min_len = discriminator_values.len();
                    for data in option_data.iter().flatten() {
                        min_len = min_len.min(data.len());
                    }
                    min_len
                };

                // Single pass: build type_ids, offsets, and per-child indices together
                let mut type_ids = Vec::with_capacity(n_samples);
                let mut offsets = Vec::with_capacity(n_samples);
                let mut child_indices: Vec<Vec<u32>> = vec![Vec::new(); option_data.len()];

                for (i, disc_val) in discriminator_values[..n_samples].iter().enumerate() {
                    let opt_idx = val_to_option.get(disc_val).copied().unwrap_or(0);
                    type_ids.push(opt_idx as i8);
                    offsets.push(child_indices[opt_idx].len() as i32);
                    child_indices[opt_idx].push(i as u32);
                }

                // Build child arrays using pre-collected indices
                let children: Vec<ArrayRef> = option_data
                    .iter()
                    .enumerate()
                    .map(|(opt_idx, opt)| {
                        if let Some(data) = opt {
                            let full_array = data.finish_cloned();
                            let indices_array = UInt32Array::from(child_indices[opt_idx].clone());
                            take(&*full_array, &indices_array, None).unwrap_or(full_array)
                        } else {
                            Arc::new(arrow::array::NullArray::new(0)) as ArrayRef
                        }
                    })
                    .collect();

                let union_fields = build_union_fields(&option_names, &children);
                let type_ids_buffer = ScalarBuffer::from(type_ids);
                let offsets_buffer = ScalarBuffer::from(offsets);

                match UnionArray::try_new(
                    union_fields,
                    type_ids_buffer,
                    Some(offsets_buffer),
                    children,
                ) {
                    Ok(union_array) => {
                        if let Some(parent_cn) = self.cn.get_mut(&parent_rec_pos) {
                            parent_cn.data = ChannelData::Union(union_array);
                        }
                    }
                    Err(e) => {
                        log::warn!("Failed to create dense UnionArray for CV variant: {}", e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Process Channel Union (CU) compositions after data is loaded.
    /// For each channel with a CU composition, this method:
    /// 1. Collects member channel data (already read by pipeline)
    /// 2. Builds UnionFields from member names and data types
    /// 3. Creates a sparse UnionArray where all members are valid at every row
    /// 4. Replaces parent channel data with ChannelData::Union
    ///
    /// CU blocks represent C-style unions: all members share the same bytes and are
    /// simultaneously valid, just interpreted differently.
    pub fn process_channel_unions(&mut self) -> Result<(), Error> {
        // Find channels with CU composition and collect info
        let cu_channels: Vec<(i32, Vec<i64>)> = self
            .cn
            .iter()
            .filter_map(|(rec_pos, cn)| {
                if let Some(composition) = &cn.composition
                    && let Compo::CU(cu_block) = &composition.block
                {
                    return Some((*rec_pos, cu_block.cu_cn_member.clone()));
                }
                None
            })
            .collect();

        for (parent_rec_pos, member_ptrs) in cu_channels {
            if member_ptrs.is_empty() {
                continue;
            }

            // Collect member channel info: (name, data as ArrayRef)
            let member_info: Vec<(String, ArrayRef)> = member_ptrs
                .iter()
                .filter_map(|ptr| {
                    self.cn
                        .values()
                        .find(|cn| cn.block_position == *ptr)
                        .map(|cn| {
                            let name = cn.unique_name.clone();
                            let array = cn.data.finish_cloned();
                            (name, array)
                        })
                })
                .collect();

            if member_info.is_empty() {
                log::warn!(
                    "CU member channels not found for parent at rec_pos {}",
                    parent_rec_pos
                );
                continue;
            }

            // All members should have the same length (same number of samples)
            let n_samples = member_info.first().map(|(_, arr)| arr.len()).unwrap_or(0);
            if n_samples == 0 {
                continue;
            }

            // Split member_info into names and children, then build UnionFields
            let (member_names, children): (Vec<String>, Vec<ArrayRef>) =
                member_info.into_iter().unzip();
            let union_fields = build_union_fields(&member_names, &children);

            // For sparse union: type_ids all set to 0 (first member as primary interpretation)
            // In reality for CU blocks, all members are equally valid - we just pick the first
            let type_ids: ScalarBuffer<i8> = ScalarBuffer::from(vec![0i8; n_samples]);

            // Create sparse UnionArray (offsets = None)
            let union_array = match UnionArray::try_new(
                union_fields,
                type_ids,
                None, // sparse union: no offsets
                children,
            ) {
                Ok(arr) => arr,
                Err(e) => {
                    log::warn!("Failed to create UnionArray for CU channel: {}", e);
                    continue;
                }
            };

            // Update parent channel data
            if let Some(parent_cn) = self.cn.get_mut(&parent_rec_pos) {
                parent_cn.data = ChannelData::Union(union_array);
            }
        }

        Ok(())
    }
}

impl Display for Cg4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let master = self.master_channel_name.as_deref().unwrap_or("None");
        write!(
            f,
            "CG: master={} channels={} record_len={} cycles={}",
            master,
            self.cn.len(),
            self.record_length,
            self.block.cg_cycle_count
        )
    }
}

/// Build UnionFields from parallel name and child arrays slices.
fn build_union_fields(names: &[String], children: &[ArrayRef]) -> UnionFields {
    let fields: Vec<(i8, Arc<Field>)> = children
        .iter()
        .enumerate()
        .map(|(idx, array)| {
            let name = names.get(idx).cloned().unwrap_or_default();
            (
                idx as i8,
                Arc::new(Field::new(name, array.data_type().clone(), true)),
            )
        })
        .collect();
    UnionFields::from_iter(fields)
}

/// Merge variant option data based on discriminator values (using owned ChannelData)
fn merge_variant_data_owned(
    discriminator_values: &[u64],
    option_data: &[Option<ChannelData>],
    val_to_option: &std::collections::HashMap<u64, usize>,
    template: &ChannelData,
) -> Option<ChannelData> {
    use crate::data_holder::channel_data::ChannelData;

    let n_samples = discriminator_values.len();

    macro_rules! merge_typed {
        ($builder_type:ty, $variant:ident) => {{
            let mut builder = <$builder_type>::with_capacity(n_samples);
            for (i, disc_val) in discriminator_values.iter().enumerate() {
                if let Some(&opt_idx) = val_to_option.get(disc_val)
                    && let Some(Some(ChannelData::$variant(b))) = option_data.get(opt_idx)
                    && i < b.values_slice().len()
                {
                    builder.append_value(b.values_slice()[i]);
                    continue;
                }
                // Default value if option not found
                builder.append_value(Default::default());
            }
            Some(ChannelData::$variant(builder))
        }};
    }

    match template {
        ChannelData::UInt8(_) => merge_typed!(arrow::array::UInt8Builder, UInt8),
        ChannelData::UInt16(_) => merge_typed!(arrow::array::UInt16Builder, UInt16),
        ChannelData::UInt32(_) => merge_typed!(arrow::array::UInt32Builder, UInt32),
        ChannelData::UInt64(_) => merge_typed!(arrow::array::UInt64Builder, UInt64),
        ChannelData::Int8(_) => merge_typed!(arrow::array::Int8Builder, Int8),
        ChannelData::Int16(_) => merge_typed!(arrow::array::Int16Builder, Int16),
        ChannelData::Int32(_) => merge_typed!(arrow::array::Int32Builder, Int32),
        ChannelData::Int64(_) => merge_typed!(arrow::array::Int64Builder, Int64),
        ChannelData::Float32(_) => merge_typed!(arrow::array::Float32Builder, Float32),
        ChannelData::Float64(_) => merge_typed!(arrow::array::Float64Builder, Float64),
        _ => {
            log::warn!("CV variant merge not implemented for this data type");
            None
        }
    }
}

/// Cg4 blocks and linked blocks parsing
pub(super) fn parse_cg4(
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
        // For VLSD/VLSC, cg_inval_bytes is the high part of total VL data size, not invalidation bytes
        let inval_bytes_size = if (cg_struct.block.cg_flags & (CG_F_VLSD | CG_F_VLSC)) != 0 {
            0
        } else {
            cg_struct.block.cg_inval_bytes
        };
        cg_struct.record_length += record_id_size as u32 + inval_bytes_size;
        cg.insert(cg_struct.block.cg_record_id, cg_struct);
        n_cg += 1;
        n_cn += num_cn;

        while next_pointer != 0 {
            let (mut cg_struct, pos, num_cn) =
                parse_cg4_block(rdr, next_pointer, position, sharable, record_id_size)?;
            position = pos;
            // For VLSD/VLSC, cg_inval_bytes is the high part of total VL data size, not invalidation bytes
            let inval_bytes_size = if (cg_struct.block.cg_flags & (CG_F_VLSD | CG_F_VLSC)) != 0 {
                0
            } else {
                cg_struct.block.cg_inval_bytes
            };
            cg_struct.record_length += record_id_size as u32 + inval_bytes_size;
            next_pointer = cg_struct.block.cg_cg_next;
            cg.insert(cg_struct.block.cg_record_id, cg_struct);
            n_cg += 1;
            n_cn += num_cn;
        }
    }
    Ok((cg, position, n_cg, n_cn))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cg_get_flags_str() {
        let mut cg = Cg4Block::default();
        assert_eq!(cg.get_flags_str(), "None");

        // Individual flags
        cg.cg_flags = CG_F_VLSD;
        assert_eq!(cg.get_flags_str(), "VLSD");

        cg.cg_flags = CG_F_VLSC;
        assert_eq!(cg.get_flags_str(), "VLSC");

        cg.cg_flags = CG_F_EVENT_SIGNAL_GROUP;
        assert_eq!(cg.get_flags_str(), "EventSignal");

        cg.cg_flags = CG_F_RAW_SENSOR_EVENT;
        assert_eq!(cg.get_flags_str(), "RawSensor");

        cg.cg_flags = CG_F_PROTOCOL_EVENT;
        assert_eq!(cg.get_flags_str(), "ProtocolEvent");

        cg.cg_flags = 0b1000; // RemoteMaster (bit 3)
        assert_eq!(cg.get_flags_str(), "RemoteMaster");

        // Combination
        cg.cg_flags = CG_F_VLSD | CG_F_EVENT_SIGNAL_GROUP;
        assert!(cg.get_flags_str().contains("VLSD"));
        assert!(cg.get_flags_str().contains("EventSignal"));
    }

    #[test]
    fn test_cg_display() {
        let cg = Cg4Block {
            cg_record_id: 1,
            cg_cycle_count: 1000,
            cg_data_bytes: 64,
            cg_inval_bytes: 2,
            cg_flags: CG_F_VLSD,
            ..Default::default()
        };
        let display = format!("{cg}");
        assert!(display.contains("CG:"));
        assert!(display.contains("rec_id=1"));
        assert!(display.contains("cycles=1000"));
        assert!(display.contains("data_bytes=64"));
        assert!(display.contains("inval_bytes=2"));
        assert!(display.contains("VLSD"));
    }
}
