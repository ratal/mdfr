//! Source Information block (SIBLOCK) for MDF4 — spec section 6.9, Tables 44-46
//!
//! Describes the source of a channel or channel group (ECU, bus, tool, etc.).
//! SIBLOCKs are sharable and stored in the SharableBlocks hashmap.
use anyhow::Result;

use super::block_header::SharableBlocks;

pub use mdfr_core::blocks::si::Si4Block;

/// Extension methods on [`Si4Block`] that resolve links through [`SharableBlocks`].
pub trait Si4BlockExt {
    /// returns the source name
    fn get_si_source_name(&self, sharable: &SharableBlocks) -> Result<Option<String>>;
    /// returns the source path
    fn get_si_path_name(&self, sharable: &SharableBlocks) -> Result<Option<String>>;
}

impl Si4BlockExt for Si4Block {
    fn get_si_source_name(&self, sharable: &SharableBlocks) -> Result<Option<String>> {
        sharable.get_tx(self.si_tx_name)
    }
    fn get_si_path_name(&self, sharable: &SharableBlocks) -> Result<Option<String>> {
        sharable.get_tx(self.si_tx_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_si_calculate_block_size() {
        let si = Si4Block::default();
        assert_eq!(si.calculate_block_size(), 56);
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
}
