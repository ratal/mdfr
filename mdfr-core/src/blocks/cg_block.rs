//! Channel Group block (CGBLOCK) — spec section 6.5, Tables 20-21.
//!
//! A CGBLOCK describes a group of channels with the same record layout (same record
//! length and record ID). Flags in Table 21 define special CG types: VLSD, VLSC, bus events.
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use binrw::binrw;
use core::fmt::{self, Display};

// Channel Group (CG) flags - cg_flags field (u16)
/// Bit 0: VLSD channel group (Variable Length Signal Data)
pub const CG_F_VLSD: u16 = 1 << 0;
/// Bit 4: Event signal group - channel group contains event signals
pub const CG_F_EVENT_SIGNAL_GROUP: u16 = 1 << 4;
/// Bit 5: VLSC channel group (contains VLSC channels, MDF 4.3)
pub const CG_F_VLSC: u16 = 1 << 5;
/// Bit 6: Raw sensor event channel group
pub const CG_F_RAW_SENSOR_EVENT: u16 = 1 << 6;
/// Bit 3: Remote master — the master channel resides in a different channel group
pub const CG_F_REMOTE_MASTER: u16 = 1 << 3;
/// Bit 7: Protocol event channel group
pub const CG_F_PROTOCOL_EVENT: u16 = 1 << 7;
/// Cg4 Channel Group block struct
#[derive(Debug, Copy, Clone)]
#[binrw]
#[br(little)]
#[bw(little)]
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
    pub cg_tx_acq_name: i64,
    /// Pointer to acquisition source (SIBLOCK) (can be NIL, must be NIL for VLSD CGBLOCK) See also rules for uniqueness explained in 4.4.3 Identification of Channels.
    pub cg_si_acq_source: i64,
    /// Pointer to first sample reduction block (SRBLOCK) (can be NIL, must be NIL for VLSD CGBLOCK)
    pub cg_sr_first: i64,
    ///Pointer to comment and additional information (TXBLOCK or MDBLOCK) (can be NIL, must be NIL for VLSD CGBLOCK)
    pub cg_md_comment: i64,
    #[br(if(cg_links > 6))]
    pub cg_cg_master: Option<i64>,
    // Data Members
    /// Record ID, value must be less than maximum unsigned integer value allowed by dg_rec_id_size in parent DGBLOCK. Record ID must be unique within linked list of CGBLOCKs.
    pub cg_record_id: u64,
    /// Number of cycles, i.e. number of samples for this channel group. This specifies the number of records of this type in the data block.
    pub cg_cycle_count: u64,
    /// Flags The value contains the following bit flags (see CG_F_xx):
    pub cg_flags: u16,
    pub cg_path_separator: u16,
    /// Reserved.
    pub cg_reserved: [u8; 4],
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
        if (self.cg_flags & CG_F_REMOTE_MASTER) != 0 {
            // Bit 3: Remote master
            flags.push("RemoteMaster");
        }
        if flags.is_empty() {
            "None".to_string()
        } else {
            flags.join("|")
        }
    }
    /// Returns the path separator character used in source paths.
    pub fn path_separator(&self) -> u16 {
        self.cg_path_separator
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

#[cfg(test)]
mod tests {
    use super::*;
    use binrw::{BinReaderExt, BinWrite, io::Cursor};

    #[test]
    fn body_size_without_master() {
        // Total CGBLOCK = 16 (short header) + body = 104 bytes
        let cg = Cg4Block {
            cg_record_id: 3,
            cg_cycle_count: 10,
            cg_data_bytes: 8,
            ..Default::default()
        };
        let mut c = Cursor::new(Vec::new());
        cg.write_le(&mut c).unwrap();
        let buf = c.into_inner();
        assert_eq!(16 + buf.len(), 104);
        let back: Cg4Block = Cursor::new(buf).read_le().unwrap();
        assert_eq!(back.cg_record_id, 3);
        assert_eq!(back.cg_cycle_count, 10);
        assert_eq!(back.cg_cg_master, None);
    }

    #[test]
    fn body_size_with_master() {
        let cg = Cg4Block {
            cg_links: 7,
            cg_cg_master: Some(0x42),
            cg_flags: CG_F_REMOTE_MASTER,
            ..Default::default()
        };
        let mut c = Cursor::new(Vec::new());
        cg.write_le(&mut c).unwrap();
        let buf = c.into_inner();
        assert_eq!(16 + buf.len(), 112);
        let back: Cg4Block = Cursor::new(buf).read_le().unwrap();
        assert_eq!(back.cg_cg_master, Some(0x42));
        assert_eq!(back.get_flags_str(), "RemoteMaster");
    }
}
