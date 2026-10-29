//! Source Information block (SIBLOCK) for MDF4 — spec section 6.9, Tables 44-46.
//!
//! Describes the source of a channel or channel group (ECU, bus, tool, etc.).
use binrw::binrw;
use core::fmt::{self, Display};

/// Si4 Source Information block struct
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
#[binrw]
#[br(little)]
#[bw(little)]
#[repr(C)]
pub struct Si4Block {
    /// # of links
    pub si_links: u64,
    /// Pointer to TXBLOCK with name (identification) of source (must not be NIL).
    pub si_tx_name: i64,
    /// Pointer to TXBLOCK with (tool-specific) path of source (can be NIL).
    pub si_tx_path: i64,
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
    pub si_reserved: [u8; 5],
}

impl Default for Si4Block {
    fn default() -> Self {
        Self {
            si_links: 3,
            si_tx_name: 0,
            si_tx_path: 0,
            si_md_comment: 0,
            si_type: 0,
            si_bus_type: 0,
            si_flags: 0,
            si_reserved: [0u8; 5],
        }
    }
}

impl Si4Block {
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
            "SI  |  type={} | bus={} | flags=0x{:02X}",
            self.get_type_str(),
            self.get_bus_type_str(),
            self.si_flags
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
    use alloc::vec::Vec;
    use binrw::{BinReaderExt, BinWrite, io::Cursor};

    #[test]
    fn test_si_get_type_str() {
        let mut si = Si4Block::default();
        let expected = [
            (0, "Other"),
            (1, "ECU"),
            (2, "Bus"),
            (3, "I/O"),
            (4, "Tool"),
            (5, "User"),
            (255, "Unknown"),
        ];
        for (val, name) in expected {
            si.si_type = val;
            assert_eq!(si.get_type_str(), name, "si_type={val}");
        }
    }

    #[test]
    fn test_si_get_bus_type_str() {
        let mut si = Si4Block::default();
        let expected = [
            (0, "None"),
            (1, "Other"),
            (2, "CAN"),
            (3, "LIN"),
            (4, "MOST"),
            (5, "FlexRay"),
            (6, "K-Line"),
            (7, "Ethernet"),
            (8, "USB"),
            (255, "Unknown"),
        ];
        for (val, name) in expected {
            si.si_bus_type = val;
            assert_eq!(si.get_bus_type_str(), name, "si_bus_type={val}");
        }
    }

    #[test]
    fn test_si_display() {
        let si = Si4Block {
            si_type: 1,
            si_bus_type: 2,
            si_flags: 0x03,
            ..Default::default()
        };

        let display = format!("{si}");
        assert!(display.contains("SI"));
        assert!(display.contains("ECU"));
        assert!(display.contains("CAN"));
        assert!(display.contains("flags=0x03"));
    }

    #[test]
    fn roundtrip_si() {
        let si = Si4Block {
            si_type: 2,
            si_bus_type: 2,
            ..Default::default()
        };
        let mut c = Cursor::new(Vec::new());
        si.write_le(&mut c).unwrap();
        let buf = c.into_inner();
        assert_eq!(16 + buf.len() as i64, si.calculate_block_size());
        let back: Si4Block = Cursor::new(buf).read_le().unwrap();
        assert_eq!(back, si);
    }
}
