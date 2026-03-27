//! Channel block (CNBLOCK) for MDF4 — spec section 6.6, Tables 22-29
//!
//! The CNBLOCK is the central block describing a single channel: its data type,
//! bit position/length in the record, sync type, composition, source, and conversion.
use anyhow::{Context, Result};
use arrow::array::{BooleanBufferBuilder, UInt8Builder, UInt16Builder, UInt32Builder};
use binrw::{BinReaderExt, binrw};
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::fs::File;
use crate::data_holder::channel_data::{ChannelData, data_type_init};
use crate::data_holder::tensor_arrow::Order;
use crate::mdfinfo::sym_buf_reader::SymBufReader;

use super::block_header::{parse_block_short, read_meta_data, Blockheader4Short, SharableBlocks, default_short_header};
use super::metadata::BlockType;
use super::cc_block::read_cc;
use super::composition::{parse_composition, Composition};
use super::ev_block::{Ev4Block, parse_ev4_block};

/// Byte order for channel data.
///
/// `false`/`Little` = little-endian (default for most modern platforms and MDF files).
/// `true`/`Big` = big-endian.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Endianness {
    /// Little-endian (LSB first)
    #[default]
    Little,
    /// Big-endian (MSB first)
    Big,
}

impl Endianness {
    /// Returns `true` if big-endian.
    #[inline]
    pub fn is_big(self) -> bool {
        self == Endianness::Big
    }
}

impl From<bool> for Endianness {
    /// Converts from a raw bool: `true` → `Big`, `false` → `Little`.
    fn from(big: bool) -> Self {
        if big { Endianness::Big } else { Endianness::Little }
    }
}

// Channel (CN) flags - cn_flags field (u32)
/// Bit 13: Event signal - channel contains event data, cn_data points to template EVBLOCK
pub const CN_F_EVENT_SIGNAL: u32 = 1 << 13;
/// Bit 15: Raw sensor event channel
#[allow(dead_code)]
pub const CN_F_RAW_SENSOR_EVENT: u32 = 1 << 15;
/// Bit 16: Auxiliary channel
#[allow(dead_code)]
pub const CN_F_AUXILIARY: u32 = 1 << 16;
/// Bit 17: Data stream mode - channel uses data stream alignment
pub const CN_F_DATA_STREAM_MODE: u32 = 1 << 17;
/// Bit 18: Alignment reset - reset alignment to start of data stream
pub const CN_F_ALIGNMENT_RESET: u32 = 1 << 18;
/// Bit 19: Protocol event channel
#[allow(dead_code)]
pub const CN_F_PROTOCOL_EVENT: u32 = 1 << 19;
/// Bit 20: Data description mode - channel describes data structure
#[allow(dead_code)]
pub const CN_F_DATA_DESCRIPTION_MODE: u32 = 1 << 20;
use super::si_block::Si4Block;

/// Channel type (cn_type field) — spec section 6.6, Table 25
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CnChannelType {
    /// 0: Fixed-length data channel (normal signal)
    FixedLength = 0,
    /// 1: Variable-length signal data (VLSD)
    Vlsd = 1,
    /// 2: Master channel (fixed length)
    Master = 2,
    /// 3: Virtual master channel (generated time/angle axis, no raw data)
    VirtualMaster = 3,
    /// 4: Synchronisation channel (references an attachment)
    Synchronisation = 4,
    /// 5: Maximum-length data channel
    MaxLength = 5,
    /// 6: Virtual data channel (generated data, no raw data)
    VirtualData = 6,
    /// 7: VLSC channel (stores offsets into VD block, MDF 4.3)
    Vlsc = 7,
}

impl TryFrom<u8> for CnChannelType {
    type Error = u8;
    fn try_from(v: u8) -> Result<Self, u8> {
        match v {
            0 => Ok(Self::FixedLength),
            1 => Ok(Self::Vlsd),
            2 => Ok(Self::Master),
            3 => Ok(Self::VirtualMaster),
            4 => Ok(Self::Synchronisation),
            5 => Ok(Self::MaxLength),
            6 => Ok(Self::VirtualData),
            7 => Ok(Self::Vlsc),
            other => Err(other),
        }
    }
}

/// Channel sync type (cn_sync_type field) — spec section 6.6, Table 26
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CnSyncType {
    /// 0: No synchronisation (normal data channel)
    #[default]
    None = 0,
    /// 1: Time synchronisation (seconds)
    Time = 1,
    /// 2: Angle synchronisation (radians)
    Angle = 2,
    /// 3: Distance synchronisation (meters)
    Distance = 3,
    /// 4: Index synchronisation (zero-based sample index)
    Index = 4,
}

impl TryFrom<u8> for CnSyncType {
    type Error = u8;
    fn try_from(v: u8) -> Result<Self, u8> {
        match v {
            0 => Ok(Self::None),
            1 => Ok(Self::Time),
            2 => Ok(Self::Angle),
            3 => Ok(Self::Distance),
            4 => Ok(Self::Index),
            other => Err(other),
        }
    }
}

/// Channel data type (cn_data_type field) — spec section 6.6, Table 27
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CnDataType {
    /// 0: Unsigned integer, little-endian
    UIntLE = 0,
    /// 1: Unsigned integer, big-endian
    UIntBE = 1,
    /// 2: Signed integer, little-endian
    IntLE = 2,
    /// 3: Signed integer, big-endian
    IntBE = 3,
    /// 4: IEEE 754 float, little-endian
    FloatLE = 4,
    /// 5: IEEE 754 float, big-endian
    FloatBE = 5,
    /// 6: String (SBC/ISO-8859-1)
    StringSbc = 6,
    /// 7: String (UTF-8)
    StringUtf8 = 7,
    /// 8: String (UTF-16 LE)
    StringUtf16LE = 8,
    /// 9: String (UTF-16 BE)
    StringUtf16BE = 9,
    /// 10: Byte array
    ByteArray = 10,
    /// 11: MIME sample
    MimeSample = 11,
    /// 12: MIME stream
    MimeStream = 12,
    /// 13: CANopen date
    CanopenDate = 13,
    /// 14: CANopen time
    CanopenTime = 14,
    /// 15: Complex number, little-endian
    ComplexLE = 15,
    /// 16: Complex number, big-endian
    ComplexBE = 16,
    /// 17: String with BOM (Unicode with byte-order mark)
    StringBom = 17,
}

impl TryFrom<u8> for CnDataType {
    type Error = u8;
    fn try_from(v: u8) -> Result<Self, u8> {
        match v {
            0 => Ok(Self::UIntLE), 1 => Ok(Self::UIntBE),
            2 => Ok(Self::IntLE),  3 => Ok(Self::IntBE),
            4 => Ok(Self::FloatLE), 5 => Ok(Self::FloatBE),
            6 => Ok(Self::StringSbc), 7 => Ok(Self::StringUtf8),
            8 => Ok(Self::StringUtf16LE), 9 => Ok(Self::StringUtf16BE),
            10 => Ok(Self::ByteArray),
            11 => Ok(Self::MimeSample), 12 => Ok(Self::MimeStream),
            13 => Ok(Self::CanopenDate), 14 => Ok(Self::CanopenTime),
            15 => Ok(Self::ComplexLE), 16 => Ok(Self::ComplexBE),
            17 => Ok(Self::StringBom),
            other => Err(other),
        }
    }
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
    pub cn_byte_offset: u32,
    /// Number of bits for signal value in record
    pub cn_bit_count: u32,
    /// Flags (see CN_F_xxx)
    pub cn_flags: u32,
    /// Position of invalidation bit.
    pub cn_inval_bit_pos: u32,
    /// Precision for display of floating point values. 0xFF means unrestricted precision (infinite). Any other value specifies the number of decimal places to use for display of floating point values. Only valid if "precision valid" flag (bit 2) is set
    cn_precision: u8,
    /// Byte alignment with previous channel in data stream
    pub cn_alignment: u8,
    /// Number of attachment for this channel
    cn_attachment_count: u16,
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
            cn_alignment: 0,
            cn_attachment_count: 0,
            cn_val_range_min: 0.0,
            cn_val_range_max: 0.0,
            cn_limit_min: 0.0,
            cn_limit_max: 0.0,
            cn_limit_ext_min: 0.0,
            cn_limit_ext_max: 0.0,
        }
    }
}

impl Cn4Block {
    /// Returns the cn_cn_size link for VLSC channels (cn_type = 7).
    /// This link points to a channel containing the size information for variable length signal data.
    /// Only valid for MDF 4.3+ VLSC channels.
    pub fn cn_cn_size(&self) -> Option<i64> {
        if self.cn_type == 7 && !self.links.is_empty() {
            Some(self.links[0]) // First additional link (9th link) is cn_cn_size
        } else {
            None
        }
    }
    /// Returns the cn_si_source link
    pub fn get_si_source(&self) -> i64 {
        self.cn_si_source
    }
    /// Sets the cn_si_source link
    pub fn set_si_source(&mut self, si_source: i64) {
        self.cn_si_source = si_source;
    }
    /// Returns the typed channel type. Returns `Err(raw_value)` if the value is out of spec.
    #[allow(dead_code)]
    pub fn cn_channel_type(&self) -> Result<CnChannelType, u8> {
        CnChannelType::try_from(self.cn_type)
    }
    /// Returns the typed sync type. Returns `Err(raw_value)` if the value is out of spec.
    #[allow(dead_code)]
    pub fn cn_sync_type_enum(&self) -> Result<CnSyncType, u8> {
        CnSyncType::try_from(self.cn_sync_type)
    }
    /// Returns the typed data type. Returns `Err(raw_value)` if the value is out of spec.
    #[allow(dead_code)]
    pub fn cn_data_type_enum(&self) -> Result<CnDataType, u8> {
        CnDataType::try_from(self.cn_data_type)
    }
    /// Returns a string representation of the channel type (cn_type)
    pub fn get_cn_type_str(&self) -> &'static str {
        match self.cn_type {
            0 => "Fixed",
            1 => "VLSD",
            2 => "Master",
            3 => "Virtual Master",
            4 => "Sync",
            5 => "MLSD",
            6 => "Virtual Data",
            7 => "VLSC",
            _ => "Unknown",
        }
    }
    /// Returns a string representation of the sync type (cn_sync_type)
    pub fn get_sync_type_str(&self) -> &'static str {
        match self.cn_sync_type {
            0 => "None",
            1 => "Time",
            2 => "Angle",
            3 => "Distance",
            4 => "Index",
            5 => "Frequency",
            _ => "Unknown",
        }
    }
    /// Returns a string representation of the data type (cn_data_type)
    pub fn get_data_type_str(&self) -> &'static str {
        match self.cn_data_type {
            0 => "UInt LE",
            1 => "UInt BE",
            2 => "Int LE",
            3 => "Int BE",
            4 => "Float LE",
            5 => "Float BE",
            6 => "String ISO-8859-1",
            7 => "String UTF-8",
            8 => "String UTF-16 LE",
            9 => "String UTF-16 BE",
            10 => "Byte Array",
            11 => "MIME Sample",
            12 => "MIME Stream",
            13 => "CANopen Date",
            14 => "CANopen Time",
            15 => "Complex LE",
            16 => "Complex BE",
            _ => "Unknown",
        }
    }
}

impl Display for Cn4Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CN: type={} ({}) data_type={} ({}) bits={} byte_offset={}",
            self.get_cn_type_str(),
            self.cn_type,
            self.get_data_type_str(),
            self.cn_data_type,
            self.cn_bit_count,
            self.cn_byte_offset
        )
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
            event_template: self.event_template.clone(),
        }
    }
}

impl Display for Cn4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CN: {} @ byte {} ({} bytes) type={} sync={}",
            self.unique_name,
            self.pos_byte_beg,
            self.n_bytes,
            self.block.get_cn_type_str(),
            self.block.get_sync_type_str()
        )
    }
}

/// hashmap's key is bit position in record, value Cn4
pub(crate) type CnType = HashMap<i32, Cn4>;

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
            } else if (cn_struct.block.cn_flags & CN_F_DATA_STREAM_MODE) != 0 {
                // data stream mode channel: use negative block_position as key to avoid collisions
                first_rec_pos = -(cn_struct.block_position as i32);
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
        endian: Endianness::Little,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        invalid_mask: None,
        event_template: None,
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
        endian: Endianness::Little,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        invalid_mask: None,
        event_template: None,
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
        endian: Endianness::Little,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        invalid_mask: None,
        event_template: None,
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
        endian: Endianness::Little,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        invalid_mask: None,
        event_template: None,
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
        endian: Endianness::Little,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        invalid_mask: None,
        event_template: None,
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
        endian: Endianness::Little,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        invalid_mask: None,
        event_template: None,
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
        endian: Endianness::Little,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        invalid_mask: None,
        event_template: None,
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
        endian: Endianness::Little,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        invalid_mask: None,
        event_template: None,
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
    let name: String = sharable.get_tx(block.cn_tx_name)?.unwrap_or_default();

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
        if block.cn_data_type == 15 | 16 {
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
        block,
        endian,
        list_size,
        shape,
        invalid_mask,
        event_template,
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
