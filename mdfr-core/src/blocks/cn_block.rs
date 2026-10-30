//! Channel block (CNBLOCK) — spec section 6.6, Tables 22-29.
//!
//! The CNBLOCK is the central block describing a single channel: its data type,
//! bit position/length in the record, sync type, composition, source, and conversion.
use alloc::vec;
use alloc::vec::Vec;
use binrw::binrw;
use core::fmt::{self, Display};

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
        if big {
            Endianness::Big
        } else {
            Endianness::Little
        }
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
            0 => Ok(Self::UIntLE),
            1 => Ok(Self::UIntBE),
            2 => Ok(Self::IntLE),
            3 => Ok(Self::IntBE),
            4 => Ok(Self::FloatLE),
            5 => Ok(Self::FloatBE),
            6 => Ok(Self::StringSbc),
            7 => Ok(Self::StringUtf8),
            8 => Ok(Self::StringUtf16LE),
            9 => Ok(Self::StringUtf16BE),
            10 => Ok(Self::ByteArray),
            11 => Ok(Self::MimeSample),
            12 => Ok(Self::MimeStream),
            13 => Ok(Self::CanopenDate),
            14 => Ok(Self::CanopenTime),
            15 => Ok(Self::ComplexLE),
            16 => Ok(Self::ComplexBE),
            17 => Ok(Self::StringBom),
            other => Err(other),
        }
    }
}

/// Cn4 Channel block struct
#[derive(Debug, PartialEq, Clone)]
#[binrw]
#[br(little)]
#[bw(little)]
#[repr(C)]
pub struct Cn4Block {
    /// ##CN
    // cn_id: [u8; 4],
    /// reserved
    // reserved: [u8; 4],
    /// Length of block in bytes
    // pub cn_len: u64,
    /// # of links
    pub cn_links: u64,
    /// Pointer to next channel block (CNBLOCK) (can be NIL)
    pub cn_cn_next: i64,
    /// Composition of channels: Pointer to channel array block (CABLOCK) or channel block (CNBLOCK) (can be NIL). Details see 4.18 Composition of Channels
    pub cn_composition: i64,
    /// Pointer to TXBLOCK with name (identification) of channel. Name must be according to naming rules stated in 4.4.2 Naming Rules.
    pub cn_tx_name: i64,
    /// Pointer to channel source (SIBLOCK) (can be NIL) Must be NIL for component channels (members of a structure or array elements) because they all must have the same source and thus simply use the SIBLOCK of their parent CNBLOCK (direct child of CGBLOCK).
    pub cn_si_source: i64,
    /// Pointer to the conversion formula (CCBLOCK) (can be NIL, must be NIL for complex channel data types, i.e. for cn_data_type ≥ 10). If the pointer is NIL, this means that a 1:1 conversion is used (phys = int).  };
    pub cn_cc_conversion: i64,
    /// Pointer to channel type specific signal data For variable length data channel (cn_type = 1): unique link to signal data block (SDBLOCK) or data list block (DLBLOCK) or, only for unsorted data groups, referencing link to a VLSD channel group block (CGBLOCK). Can only be NIL if SDBLOCK would be empty. For synchronization channel (cn_type = 4): referencing link to attachment block (ATBLOCK) in global linked list of ATBLOCKs starting at hd_at_first. Cannot be NIL.
    pub cn_data: i64,
    /// Pointer to TXBLOCK/MDBLOCK with designation for physical unit of signal data (after conversion) or (only for channel data types "MIME sample" and "MIME stream") to MIME context-type text. (can be NIL). The unit can be used if no conversion rule is specified or to overwrite the unit specified for the conversion rule (e.g. if a conversion rule is shared between channels). If the link is NIL, then the unit from the conversion rule must be used. If the content is an empty string, no unit should be displayed. If an MDBLOCK is used, in addition the A-HDO unit definition can be stored, see Table 38. Note: for (virtual) master and synchronization channels the A-HDO definition should be omitted to avoid redundancy. Here the unit is already specified by cn_sync_type of the channel. In case of channel data types "MIME sample" and "MIME stream", the text of the unit must be the content-type text of a MIME type which specifies the content of the values of the channel (either fixed length in record or variable length in SDBLOCK). The MIME content-type string must be written in lowercase, and it must apply to the same rules as defined for at_tx_mimetype in 4.11 The Attachment Block ATBLOCK.
    pub cn_md_unit: i64,
    /// Pointer to TXBLOCK/MDBLOCK with designation for physical unit of signal data (after conversion) or (only for channel data types "MIME sample" and "MIME stream") to MIME context-type text. (can be NIL). The unit can be used if no conversion rule is specified or to overwrite the unit specified for the conversion rule (e.g. if a conversion rule is shared between channels). If the link is NIL, then the unit from the conversion rule must be used. If the content is an empty string, no unit should be displayed. If an MDBLOCK is used, in addition the A-HDO unit definition can be stored, see Table 38. Note: for (virtual) master and synchronization channels the A-HDO definition should be omitted to avoid redundancy. Here the unit is already specified by cn_sync_type of the channel. In case of channel data types "MIME sample" and "MIME stream", the text of the unit must be the content-type text of a MIME type which specifies the content of the values of the channel (either fixed length in record or variable length in SDBLOCK). The MIME content-type string must be written in lowercase, and it must apply to the same rules as defined for at_tx_mimetype in 4.11 The Attachment Block ATBLOCK.
    pub cn_md_comment: i64,
    #[br(if(cn_links > 8), little, count = cn_links - 8)]
    pub links: Vec<i64>,

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
    pub cn_precision: u8,
    /// Byte alignment with previous channel in data stream
    pub cn_alignment: u8,
    /// Number of attachment for this channel
    pub cn_attachment_count: u16,
    /// Minimum signal value that occurred for this signal (raw value) Only valid if "value range valid" flag (bit 3) is set.
    pub cn_val_range_min: f64,
    /// Maximum signal value that occurred for this signal (raw value) Only valid if "value range valid" flag (bit 3) is set.
    pub cn_val_range_max: f64,
    /// Lower limit for this signal (physical value for numeric conversion rule, otherwise raw value) Only valid if "limit range valid" flag (bit 4) is set.
    pub cn_limit_min: f64,
    /// Upper limit for this signal (physical value for numeric conversion rule, otherwise raw value) Only valid if "limit range valid" flag (bit 4) is set.
    pub cn_limit_max: f64,
    /// Lower extended limit for this signal (physical value for numeric conversion rule, otherwise raw value) Only valid if "extended limit range valid" flag (bit 5) is set.
    pub cn_limit_ext_min: f64,
    /// Upper extended limit for this signal (physical value for numeric conversion rule, otherwise raw value) Only valid if "extended limit range valid" flag (bit 5) is set.
    pub cn_limit_ext_max: f64,
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
    /// Returns the precision value if the "precision valid" flag (bit 2) is set.
    pub fn precision(&self) -> Option<u8> {
        if (self.cn_flags & (1 << 2)) != 0 {
            Some(self.cn_precision)
        } else {
            None
        }
    }
    /// Returns the minimum signal value (raw) if the "value range valid" flag (bit 3) is set.
    pub fn val_range_min(&self) -> Option<f64> {
        if (self.cn_flags & (1 << 3)) != 0 {
            Some(self.cn_val_range_min)
        } else {
            None
        }
    }
    /// Returns the maximum signal value (raw) if the "value range valid" flag (bit 3) is set.
    pub fn val_range_max(&self) -> Option<f64> {
        if (self.cn_flags & (1 << 3)) != 0 {
            Some(self.cn_val_range_max)
        } else {
            None
        }
    }
    /// Returns the lower limit (physical or raw) if the "limit range valid" flag (bit 4) is set.
    pub fn limit_min(&self) -> Option<f64> {
        if (self.cn_flags & (1 << 4)) != 0 {
            Some(self.cn_limit_min)
        } else {
            None
        }
    }
    /// Returns the upper limit (physical or raw) if the "limit range valid" flag (bit 4) is set.
    pub fn limit_max(&self) -> Option<f64> {
        if (self.cn_flags & (1 << 4)) != 0 {
            Some(self.cn_limit_max)
        } else {
            None
        }
    }
    /// Returns the lower extended limit if the "extended limit range valid" flag (bit 5) is set.
    pub fn limit_ext_min(&self) -> Option<f64> {
        if (self.cn_flags & (1 << 5)) != 0 {
            Some(self.cn_limit_ext_min)
        } else {
            None
        }
    }
    /// Returns the upper extended limit if the "extended limit range valid" flag (bit 5) is set.
    pub fn limit_ext_max(&self) -> Option<f64> {
        if (self.cn_flags & (1 << 5)) != 0 {
            Some(self.cn_limit_ext_max)
        } else {
            None
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

#[cfg(test)]
mod tests {
    use super::*;
    use binrw::{BinReaderExt, BinWrite, io::Cursor};

    #[test]
    fn body_size_160() {
        // CNBLOCK = 16 (short header) + body = 160 bytes
        let cn = Cn4Block {
            cn_bit_count: 16,
            cn_data_type: 2,
            cn_byte_offset: 4,
            ..Default::default()
        };
        let mut c = Cursor::new(Vec::new());
        cn.write_le(&mut c).unwrap();
        let buf = c.into_inner();
        assert_eq!(16 + buf.len(), 160);
        let back: Cn4Block = Cursor::new(buf).read_le().unwrap();
        assert_eq!(back, cn);
    }

    #[test]
    fn extra_links_roundtrip() {
        let cn = Cn4Block {
            cn_links: 9,
            links: vec![0x99],
            cn_type: 7,
            ..Default::default()
        };
        let mut c = Cursor::new(Vec::new());
        cn.write_le(&mut c).unwrap();
        let buf = c.into_inner();
        assert_eq!(16 + buf.len(), 168);
        let back: Cn4Block = Cursor::new(buf).read_le().unwrap();
        assert_eq!(back.cn_cn_size(), Some(0x99));
    }

    #[test]
    fn enum_conversions() {
        assert_eq!(CnChannelType::try_from(7), Ok(CnChannelType::Vlsc));
        assert_eq!(CnDataType::try_from(17), Ok(CnDataType::StringBom));
        assert_eq!(CnSyncType::try_from(9), Err(9));
        assert_eq!(Endianness::from(true), Endianness::Big);
        assert!(Endianness::Big.is_big());
        assert!(!Endianness::Little.is_big());
    }

    #[test]
    fn test_cn_helpers_and_strings() {
        let mut cn = Cn4Block {
            cn_type: 0,
            cn_sync_type: 1,
            cn_data_type: 4,
            cn_bit_count: 32,
            cn_byte_offset: 8,
            cn_flags: (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5),
            cn_precision: 3,
            cn_val_range_min: -100.0,
            cn_val_range_max: 100.0,
            cn_limit_min: -50.0,
            cn_limit_max: 50.0,
            cn_limit_ext_min: -80.0,
            cn_limit_ext_max: 80.0,
            ..Default::default()
        };

        assert_eq!(cn.get_cn_type_str(), "Fixed");
        assert_eq!(cn.get_sync_type_str(), "Time");
        assert_eq!(cn.get_data_type_str(), "Float LE");

        assert_eq!(cn.precision(), Some(3));
        assert_eq!(cn.val_range_min(), Some(-100.0));
        assert_eq!(cn.val_range_max(), Some(100.0));
        assert_eq!(cn.limit_min(), Some(-50.0));
        assert_eq!(cn.limit_max(), Some(50.0));
        assert_eq!(cn.limit_ext_min(), Some(-80.0));
        assert_eq!(cn.limit_ext_max(), Some(80.0));

        let s = alloc::format!("{cn}");
        assert!(s.contains("CN: type=Fixed"));
        assert!(s.contains("data_type=Float LE"));

        // Test flag unset returns None
        cn.cn_flags = 0;
        assert_eq!(cn.precision(), None);
        assert_eq!(cn.val_range_min(), None);
        assert_eq!(cn.val_range_max(), None);
        assert_eq!(cn.limit_min(), None);
        assert_eq!(cn.limit_max(), None);
        assert_eq!(cn.limit_ext_min(), None);
        assert_eq!(cn.limit_ext_max(), None);

        // Test other type strings
        cn.cn_type = 1;
        assert_eq!(cn.get_cn_type_str(), "VLSD");
        cn.cn_type = 2;
        assert_eq!(cn.get_cn_type_str(), "Master");
        cn.cn_type = 3;
        assert_eq!(cn.get_cn_type_str(), "Virtual Master");
        cn.cn_type = 4;
        assert_eq!(cn.get_cn_type_str(), "Sync");
        cn.cn_type = 5;
        assert_eq!(cn.get_cn_type_str(), "MLSD");
        cn.cn_type = 6;
        assert_eq!(cn.get_cn_type_str(), "Virtual Data");
        cn.cn_type = 7;
        assert_eq!(cn.get_cn_type_str(), "VLSC");
        cn.cn_type = 99;
        assert_eq!(cn.get_cn_type_str(), "Unknown");

        cn.cn_sync_type = 0;
        assert_eq!(cn.get_sync_type_str(), "None");
        cn.cn_sync_type = 2;
        assert_eq!(cn.get_sync_type_str(), "Angle");
        cn.cn_sync_type = 3;
        assert_eq!(cn.get_sync_type_str(), "Distance");
        cn.cn_sync_type = 4;
        assert_eq!(cn.get_sync_type_str(), "Index");
        cn.cn_sync_type = 5;
        assert_eq!(cn.get_sync_type_str(), "Frequency");
        cn.cn_sync_type = 99;
        assert_eq!(cn.get_sync_type_str(), "Unknown");
    }
}
