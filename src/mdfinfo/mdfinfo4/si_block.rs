//! Source Information block (SIBLOCK) for MDF4 — spec section 6.9, Tables 44-46
//!
//! Describes the source of a channel or channel group (ECU, bus, tool, etc.).
//! SIBLOCKs are sharable and stored in the SharableBlocks hashmap.
use anyhow::Result;
use binrw::binrw;
use std::fmt::{self, Display};

use super::block_header::SharableBlocks;

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
    pub si_tx_name: i64,
    /// Pointer to TXBLOCK with (tool-specific) path of source (can be NIL). The path string must be according to naming rules stated in 4.4.2 Naming Rules.
    pub si_tx_path: i64,
    // Each tool may generate a different path string. The only purpose is to ensure uniqueness as explained in section 4.4.3 Identification of Channels. As a recommendation, the path should be a human readable string containing additional information about the source. However, the path string should not be used to store this information in order to retrieve it later by parsing the string. Instead, additional source information should be stored in generic or custom XML fields in the comment MDBLOCK si_md_comment.
    /// Pointer to source comment and additional information (TXBLOCK or MDBLOCK) (can be NIL)
    pub si_md_comment: i64,

    // Data Members
    /// Source type additional classification of source (see SI_T_xxx)
    pub si_type: u8,
    /// Bus type additional classification of used bus (should be 0 for si_type >= 3) (see SI_BUS_xxx)
    pub si_bus_type: u8,
    /// Flags The value contains the following bit flags (see SI_F_xxx)):
    pub si_flags: u8,
    /// reserved
    si_reserved: [u8; 5],
}

impl Si4Block {
    /// returns the source name
    pub fn get_si_source_name(&self, sharable: &SharableBlocks) -> Result<Option<String>> {
        sharable.get_tx(self.si_tx_name)
    }
    /// returns the source path
    pub fn get_si_path_name(&self, sharable: &SharableBlocks) -> Result<Option<String>> {
        sharable.get_tx(self.si_tx_path)
    }
    /// Calculate the total block size (header + links + data)
    pub fn calculate_block_size(&self) -> i64 {
        // 16 (short header) + 8 (link count) + 8*3 (3 links) + 8 (data members)
        16 + 8 + 24 + 8
    }

    /// Returns the source type as a string description
    pub fn get_type_str(&self) -> &'static str {
        match self.si_type {
            0 => "Other",
            1 => "ECU",
            2 => "Bus",
            3 => "I/O",
            4 => "Tool",
            5 => "User",
            _ => "Unknown",
        }
    }

    /// Returns the bus type as a string description
    pub fn get_bus_type_str(&self) -> &'static str {
        match self.si_bus_type {
            0 => "None",
            1 => "Other",
            2 => "CAN",
            3 => "LIN",
            4 => "MOST",
            5 => "FlexRay",
            6 => "K-Line",
            7 => "Ethernet",
            8 => "USB",
            _ => "Unknown",
        }
    }
}

impl Display for Si4Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SI: type={} ({}) bus={} ({}) flags=0x{:02X}",
            self.get_type_str(),
            self.si_type,
            self.get_bus_type_str(),
            self.si_bus_type,
            self.si_flags
        )
    }
}
