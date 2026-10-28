//! Channel block (CNBLOCK) for MDF4 — spec section 6.6, Tables 22-29
//!
//! The CNBLOCK is the central block describing a single channel: its data type,
//! bit position/length in the record, sync type, composition, source, and conversion.
use crate::data_holder::channel_data::{ChannelData, data_type_init};
use crate::data_holder::tensor_arrow::Order;
use crate::mdfinfo::block_chain::BlockChain;
use crate::mdfinfo::sym_buf_reader::SymBufReader;
use anyhow::{Context, Result};
use arrow::array::{BooleanBufferBuilder, UInt8Builder, UInt16Builder, UInt32Builder};
use rustc_hash::FxHashMap;
use std::fmt::{self, Display};
use std::fs::File;

use super::block_header_impl::{
    Blockheader4Short, SharableBlocks, default_short_header, parse_block_short, read_meta_data,
};
use super::cc_impl::read_cc;
use super::composition_impl::{Composition, parse_composition};
use super::ev_impl::{Ev4Block, parse_ev4_block};
use super::metadata::BlockType;
use super::si_impl::{Si4Block, Si4BlockExt};
pub use mdfr_core::blocks::cn::*;

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
    /// absolute file position of this CN block
    pub block_position: i64,
    /// beginning position of channel in record
    pub pos_byte_beg: u32,
    /// number of bytes taken by channel in record
    pub n_bytes: u32,
    /// optional composition (CA array, nested CN structure, DS/CL/CV/CU VLSD layout)
    pub composition: Option<Composition>,
    /// channel data
    pub data: ChannelData,
    /// flag to track if channel's data is already converted or to be converted
    pub is_converted: bool,
    /// byte order of the channel's raw data
    pub endian: Endianness,
    /// number of elements per sample: 1 for scalars, 2 for complex, N for arrays
    pub list_size: usize,
    /// shape of array data: (dimension sizes, storage order); scalar channels use an empty vec
    pub shape: (Vec<usize>, Order),
    /// optional invalid mask array, invalid byte position in record, invalid byte mask
    pub invalid_mask: Option<(Option<BooleanBufferBuilder>, usize, u8)>,
    /// Template EVBLOCK for event signal channels (cn_flags bit 13 set)
    /// This describes the structure of event data stored in the channel
    pub event_template: Option<Ev4Block>,
    /// flag set during data read to indicate this channel is in the read set
    pub should_read: bool,
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
            data: self.data.clone(),
            is_converted: self.is_converted,
            endian: self.endian,
            list_size: self.list_size,
            shape: self.shape.clone(),
            invalid_mask,
            event_template: self.event_template.clone(),
            should_read: self.should_read,
        }
    }
}

impl Display for Cn4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let info = format!(
            "CN: {} @ byte {} ({} bytes) type={} sync={}",
            self.unique_name,
            self.pos_byte_beg,
            self.n_bytes,
            self.block.get_cn_type_str(),
            self.block.get_sync_type_str()
        );
        if self.is_event_signal()
            && let Some(ev_info) = self.event_signal_info()
        {
            write!(f, "{} event_signal={}", info, ev_info)?;
            return Ok(());
        }
        if self.is_mlsd() {
            write!(f, "{} mlsd", info)?;
            return Ok(());
        }
        if self.is_sync() {
            write!(f, "{} sync_channel", info)?;
            return Ok(());
        }
        if self.is_virtual_data() {
            write!(f, "{} virtual_data", info)?;
            return Ok(());
        }
        write!(f, "{info}")
    }
}

/// hashmap's key is bit position in record, value Cn4
pub type CnType = FxHashMap<i32, Cn4>;

/// record layout type : record_id_size: u8, cg_data_bytes: u32, cg_inval_bytes: u32
pub(crate) type RecordLayout = (u8, u32, u32);

/// creates recursively in the channel group the CN blocks and all its other linked blocks (CC, MD, TX, CA, etc.)
pub(super) fn parse_cn4(
    rdr: &mut SymBufReader<&File>,
    target: i64,
    mut position: i64,
    sharable: &mut SharableBlocks,
    record_layout: RecordLayout,
    cg_cycle_count: u64,
) -> Result<(CnType, i64, usize, i32)> {
    let mut cn: CnType = FxHashMap::default();
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
            } else if (cn_struct.block.cn_flags & CN_F_DATA_STREAM_MODE) != 0 {
                // data stream mode channel: use negative block_position as key to avoid collisions
                first_rec_pos = -(cn_struct.block_position as i32);
            }
            cn.insert(first_rec_pos, cn_struct);
        }

        let mut chain = BlockChain::new("CN");
        chain.visit(target);
        while next_pointer != 0 {
            if !chain.visit(next_pointer) {
                break;
            }
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
                } else if (cn_struct.block.cn_flags & CN_F_DATA_STREAM_MODE) != 0 {
                    // data stream mode channel: use negative block_position as key to avoid collisions
                    rec_pos = -(cn_struct.block_position as i32);
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
        is_converted: false,
        endian: Endianness::Little,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        invalid_mask: None,
        event_template: None,
        should_read: false,
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
        pos_byte_beg: pos_byte_beg + 2,
        n_bytes: 1,
        composition: None,
        data: ChannelData::UInt8(UInt8Builder::new()),
        is_converted: false,
        endian: Endianness::Little,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        invalid_mask: None,
        event_template: None,
        should_read: false,
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
        pos_byte_beg: pos_byte_beg + 3,
        n_bytes: 1,
        composition: None,
        data: ChannelData::UInt8(UInt8Builder::new()),
        is_converted: false,
        endian: Endianness::Little,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        invalid_mask: None,
        event_template: None,
        should_read: false,
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
        pos_byte_beg: pos_byte_beg + 4,
        n_bytes: 1,
        composition: None,
        data: ChannelData::UInt8(UInt8Builder::new()),
        is_converted: false,
        endian: Endianness::Little,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        invalid_mask: None,
        event_template: None,
        should_read: false,
    };
    let block = Cn4Block {
        cn_links: 8,
        cn_byte_offset: cn_byte_offset + 5,
        cn_bit_count: 4,
        ..Default::default()
    };
    let month = Cn4 {
        header: default_short_header(BlockType::CN),
        block,
        unique_name: String::from("month"),
        block_position,
        pos_byte_beg: pos_byte_beg + 5,
        n_bytes: 1,
        composition: None,
        data: ChannelData::UInt8(UInt8Builder::new()),
        is_converted: false,
        endian: Endianness::Little,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        invalid_mask: None,
        event_template: None,
        should_read: false,
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
        pos_byte_beg: pos_byte_beg + 6,
        n_bytes: 1,
        composition: None,
        data: ChannelData::UInt8(UInt8Builder::new()),
        is_converted: false,
        endian: Endianness::Little,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        invalid_mask: None,
        event_template: None,
        should_read: false,
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
        is_converted: false,
        endian: Endianness::Little,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        invalid_mask: None,
        event_template: None,
        should_read: false,
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
        pos_byte_beg: pos_byte_beg + 4,
        n_bytes: 2,
        composition: None,
        data: ChannelData::UInt16(UInt16Builder::new()),
        is_converted: false,
        endian: Endianness::Little,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        invalid_mask: None,
        event_template: None,
        should_read: false,
    };
    (ms, days)
}

/// Simple calculation to convert bit count into equivalent bytes count
fn calc_n_bytes_not_aligned(bitcount: u32) -> u32 {
    let mut n_bytes = bitcount / 8u32;
    if !bitcount.is_multiple_of(8) {
        n_bytes += 1;
    }
    n_bytes
}

/// A single decoded event record from an event signal channel.
#[derive(Debug, Clone)]
pub struct EventRecord {
    /// Sync value (time, angle, distance, index, or frequency) for this event
    pub sync_value: f64,
    /// Remaining event-specific data as raw bytes
    pub event_data: Vec<u8>,
}

impl EventRecord {
    /// Creates a new event record
    pub fn new(sync_value: f64, event_data: Vec<u8>) -> Self {
        Self {
            sync_value,
            event_data,
        }
    }
}

impl Display for EventRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EventRecord(sync={}, data_len={})",
            self.sync_value,
            self.event_data.len()
        )
    }
}

#[allow(dead_code)]
impl Cn4 {
    /// Decodes raw event signal data bytes into structured event records using the template EVBLOCK.
    /// Each record is parsed to extract the sync value (based on template sync type) and remaining event data.
    /// Returns `None` if the channel is not an event signal channel or has no data.
    pub fn decode_event_records(&self, data_bytes: &[u8]) -> Option<Vec<EventRecord>> {
        if !self.is_event_signal() {
            return None;
        }
        let template = self.event_template.as_ref()?;
        let record_size = self.n_bytes as usize;
        if record_size == 0 || data_bytes.is_empty() {
            return None;
        }
        let sync_type = template.ev_sync_type;
        let sync_base = template.ev_sync_base_value as f64;
        let sync_factor = template.ev_sync_factor;
        let mut records = Vec::new();
        for record in data_bytes.chunks(record_size) {
            let (sync_value, event_data) = match sync_type {
                0 => {
                    // No sync - entire record is event data
                    (0.0, record.to_vec())
                }
                1..=4 => {
                    // Time, Angle, Distance, Index - try 8-byte then 4-byte sync
                    if record.len() >= 8 {
                        let val = match self.endian {
                            Endianness::Little => {
                                i64::from_le_bytes(record[0..8].try_into().unwrap_or([0; 8]))
                            }
                            Endianness::Big => {
                                i64::from_be_bytes(record[0..8].try_into().unwrap_or([0; 8]))
                            }
                        };
                        (sync_base + val as f64 * sync_factor, record[8..].to_vec())
                    } else if record.len() >= 4 {
                        let val = match self.endian {
                            Endianness::Little => {
                                i32::from_le_bytes(record[0..4].try_into().unwrap_or([0; 4]))
                            }
                            Endianness::Big => {
                                i32::from_be_bytes(record[0..4].try_into().unwrap_or([0; 4]))
                            }
                        };
                        (sync_base + val as f64 * sync_factor, record[4..].to_vec())
                    } else {
                        (sync_base, record.to_vec())
                    }
                }
                5 => {
                    // Frequency - 8-byte float
                    if record.len() >= 8 {
                        let val = match self.endian {
                            Endianness::Little => {
                                f64::from_le_bytes(record[0..8].try_into().unwrap_or([0; 8]))
                            }
                            Endianness::Big => {
                                f64::from_be_bytes(record[0..8].try_into().unwrap_or([0; 8]))
                            }
                        };
                        (sync_base + val * sync_factor, record[8..].to_vec())
                    } else {
                        (sync_base, record.to_vec())
                    }
                }
                _ => (sync_base, record.to_vec()),
            };
            records.push(EventRecord::new(sync_value, event_data));
        }
        Some(records)
    }
}

#[allow(dead_code)]
impl Cn4 {
    /// Returns true if this channel is an event signal channel (cn_flags bit 13 set).
    /// Event signal channels store event data, with a template EVBLOCK describing the structure.
    pub fn is_event_signal(&self) -> bool {
        (self.block.cn_flags & CN_F_EVENT_SIGNAL) != 0
    }

    /// Returns a reference to the template EVBLOCK if this is an event signal channel.
    /// The template EVBLOCK describes the structure of event data stored in this channel.
    pub fn get_event_template(&self) -> Option<&Ev4Block> {
        self.event_template.as_ref()
    }

    /// Returns a reference to the template EVBLOCK if this is an event signal channel.
    /// The template EVBLOCK describes the structure of event data stored in this channel.
    pub fn event_signal_info(&self) -> Option<&Ev4Block> {
        self.get_event_template()
    }

    /// Returns true if this channel is an MLSD (Maximum Length Signal Data) channel.
    /// MLSD channels have per-record lengths determined by a separate size channel.
    pub fn is_mlsd(&self) -> bool {
        self.block.cn_type == 5
    }

    /// Returns true if this channel is a synchronization channel.
    /// Sync channels reference an ATBLOCK (attachment) rather than containing raw data.
    pub fn is_sync(&self) -> bool {
        self.block.cn_type == 4
    }

    /// Returns true if this channel is a virtual data channel.
    /// Virtual data channels have no raw data; values are generated from a conversion rule.
    pub fn is_virtual_data(&self) -> bool {
        self.block.cn_type == 6
    }

    /// Returns the channel source name
    pub fn get_cn_source_name(&self, sharable: &SharableBlocks) -> Result<Option<String>> {
        let si = sharable.si.get(&self.block.cn_si_source);
        match si {
            Some(block) => Ok(block.get_si_source_name(sharable)?),
            None => Ok(None),
        }
    }
    /// Returns the channel source path
    pub fn get_cn_source_path(&self, sharable: &SharableBlocks) -> Result<Option<String>> {
        let si = sharable.si.get(&self.block.cn_si_source);
        match si {
            Some(block) => Ok(block.get_si_path_name(sharable)?),
            None => Ok(None),
        }
    }
}

/// Channel block parser
pub(super) fn parse_cn4_block(
    rdr: &mut SymBufReader<&File>,
    target: i64,
    mut position: i64,
    sharable: &mut SharableBlocks,
    record_layout: RecordLayout,
    cg_cycle_count: u64,
) -> Result<(Cn4, i64, usize, CnType)> {
    let (record_id_size, _cg_data_bytes, cg_inval_bytes) = record_layout;
    let mut n_cn: usize = 1;
    let mut cns: CnType = FxHashMap::default();
    let (mut block, cnheader, pos) = parse_block_short(rdr, target, position)?;
    position = pos;
    let block: Cn4Block = binrw::BinReaderExt::read_le(&mut block)
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
    let name: String = sharable.get_tx(block.cn_tx_name)?.unwrap_or_default();

    if block.cn_data_type <= 3 {
        let total_bits = block.cn_bit_count + (block.cn_bit_offset as u32);
        if total_bits > 64 && !(block.cn_bit_count == 64 && block.cn_bit_offset == 0) {
            log::warn!(
                "Channel {} integer violates MDF4 spec §6.6.3: cn_bit_count ({}) + cn_bit_offset ({}) = {} > 64",
                name,
                block.cn_bit_count,
                block.cn_bit_offset,
                total_bits
            );
        }
    }

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
        let si_block: Si4Block = binrw::BinReaderExt::read_le(&mut si_block)
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
        if block.cn_data_type == 15 || block.cn_data_type == 16 {
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
        if block.cn_data_type == 15 || block.cn_data_type == 16 {
            //complex
            list_size = 2;
        } else {
            list_size = 1;
        }
    }

    let mut endian = Endianness::Little; // Little endian by default
    if block.cn_data_type == 0
        || block.cn_data_type == 2
        || block.cn_data_type == 4
        || block.cn_data_type == 8
        || block.cn_data_type == 15
    {
        endian = Endianness::Little;
    } else if block.cn_data_type == 1
        || block.cn_data_type == 3
        || block.cn_data_type == 5
        || block.cn_data_type == 9
        || block.cn_data_type == 16
    {
        endian = Endianness::Big;
    }
    // For VLSC/VLSD channels, cn_data_type describes the signal data block encoding
    // (e.g. UTF-16 BE), not the byte order of the integer offsets stored in the DT block.
    if block.cn_type == 1 || block.cn_type == 7 {
        endian = Endianness::Little;
    }
    let data_type = block.cn_data_type;
    let cn_type = block.cn_type;

    // Read template EVBLOCK for event signal channels (cn_flags bit 13 set)
    // For event signal channels, cn_data points to a template EVBLOCK that describes
    // the structure of event data stored in this channel
    let event_template: Option<Ev4Block> =
        if (block.cn_flags & CN_F_EVENT_SIGNAL) != 0 && block.cn_data != 0 {
            let (ev_block, pos) = parse_ev4_block(rdr, block.cn_data, position)?;
            position = pos;
            Some(ev_block)
        } else {
            None
        };

    let cn_struct = Cn4 {
        header: cnheader,
        unique_name: name,
        block_position: target,
        pos_byte_beg,
        n_bytes,
        composition: compo,
        data: data_type_init(cn_type, data_type, n_bytes, list_size, block.cn_flags)?,
        is_converted: false,
        block,
        endian,
        list_size,
        shape,
        invalid_mask,
        event_template,
        should_read: false,
    };

    Ok((cn_struct, position, n_cn, cns))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cn_get_cn_type_str() {
        let mut cn = Cn4Block::default();
        let expected = [
            (0, "Fixed"),
            (1, "VLSD"),
            (2, "Master"),
            (3, "Virtual Master"),
            (4, "Sync"),
            (5, "MLSD"),
            (6, "Virtual Data"),
            (7, "VLSC"),
            (255, "Unknown"),
        ];
        for (val, name) in expected {
            cn.cn_type = val;
            assert_eq!(cn.get_cn_type_str(), name, "cn_type={val}");
        }
    }

    #[test]
    fn test_cn_get_sync_type_str() {
        let mut cn = Cn4Block::default();
        let expected = [
            (0, "None"),
            (1, "Time"),
            (2, "Angle"),
            (3, "Distance"),
            (4, "Index"),
            (5, "Frequency"),
            (255, "Unknown"),
        ];
        for (val, name) in expected {
            cn.cn_sync_type = val;
            assert_eq!(cn.get_sync_type_str(), name, "cn_sync_type={val}");
        }
    }

    #[test]
    fn test_cn_get_data_type_str() {
        let mut cn = Cn4Block::default();
        let expected = [
            (0, "UInt LE"),
            (1, "UInt BE"),
            (2, "Int LE"),
            (3, "Int BE"),
            (4, "Float LE"),
            (5, "Float BE"),
            (6, "String ISO-8859-1"),
            (7, "String UTF-8"),
            (8, "String UTF-16 LE"),
            (9, "String UTF-16 BE"),
            (10, "Byte Array"),
            (11, "MIME Sample"),
            (12, "MIME Stream"),
            (13, "CANopen Date"),
            (14, "CANopen Time"),
            (15, "Complex LE"),
            (16, "Complex BE"),
            (255, "Unknown"),
        ];
        for (val, name) in expected {
            cn.cn_data_type = val;
            assert_eq!(cn.get_data_type_str(), name, "cn_data_type={val}");
        }
    }

    #[test]
    fn test_cn_cn_size() {
        // Non-VLSC type returns None
        let cn = Cn4Block::default(); // cn_type = 0
        assert!(cn.cn_cn_size().is_none());

        // VLSC with no extra links returns None
        let cn = Cn4Block {
            cn_type: 7,
            ..Default::default()
        };
        assert!(cn.cn_cn_size().is_none()); // links is empty

        // VLSC with extra links returns first link
        let cn = Cn4Block {
            cn_type: 7,
            cn_links: 9,
            links: vec![12345],
            ..Default::default()
        };
        assert_eq!(cn.cn_cn_size(), Some(12345));
    }

    #[test]
    fn test_cn_get_set_si_source() {
        let mut cn = Cn4Block::default();
        assert_eq!(cn.get_si_source(), 0);

        cn.set_si_source(42);
        assert_eq!(cn.get_si_source(), 42);
    }

    #[test]
    fn test_cn_block_display() {
        let cn = Cn4Block {
            cn_type: 2,
            cn_data_type: 4,
            cn_bit_count: 64,
            cn_byte_offset: 8,
            ..Default::default()
        };
        let display = format!("{cn}");
        assert!(display.contains("CN:"));
        assert!(display.contains("Master"));
        assert!(display.contains("Float LE"));
        assert!(display.contains("bits=64"));
        assert!(display.contains("byte_offset=8"));
    }

    #[test]
    fn test_cn4_display() {
        let cn4 = Cn4 {
            unique_name: "speed".to_string(),
            pos_byte_beg: 16,
            n_bytes: 8,
            block: Cn4Block {
                cn_type: 0,
                cn_sync_type: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let display = format!("{cn4}");
        assert!(display.contains("speed"));
        assert!(display.contains("byte 16"));
        assert!(display.contains("8 bytes"));
        assert!(display.contains("Fixed"));
        assert!(display.contains("Time"));
    }
}
