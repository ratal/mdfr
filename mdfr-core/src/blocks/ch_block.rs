//! Channel Hierarchy block (CHBLOCK) for MDF4 — spec section 6.2, Tables 15-17.
//!
//! CHBLOCKs organize channels into a display hierarchy tree (groups, functions,
//! input/output variables). The hierarchy is separate from the DG→CG→CN data structure.
use alloc::vec::Vec;
use binrw::binrw;
use core::fmt::{self, Display};

/// CHBLOCK structure (MDF 4.2 spec, Table 15; body following the 16-byte short header).
#[derive(Debug, PartialEq, Eq, Default, Clone)]
#[binrw]
#[br(little)]
#[bw(little)]
#[repr(C)]
pub struct Ch4Block {
    /// # of links
    pub ch_links: u64,
    /// link to next CHBLOCK at this hierarchy level
    pub ch_ch_next: i64,
    /// link to first CHBLOCK at the next hierarchy level (child)
    pub ch_ch_first: i64,
    /// link to TXBLOCK with the name of the hierarchy level
    pub ch_tx_name: i64,
    /// link to MDBLOCK with a comment/description
    pub ch_md_comment: i64,
    /// list of elements in this hierarchy level
    #[br(count = ch_links.saturating_sub(4))]
    pub ch_element: Vec<i64>,
    /// number of elements in this hierarchy level (Nx3)
    pub ch_element_count: u32,
    /// hierarchy level type
    pub ch_type: u8,
    /// reserved
    pub ch_reserved: [u8; 3],
}

impl Ch4Block {
    /// Calculate the total block size (header + links + data)
    pub fn calculate_block_size(&self) -> i64 {
        16 + 8 + (self.ch_links * 8) as i64 + 8
    }

    /// Returns the hierarchy type as a string description
    pub fn get_type_str(&self) -> &'static str {
        match self.ch_type {
            0 => "Group",
            1 => "Function",
            2 => "Structure",
            3 => "Map list",
            4 => "Input variables",
            5 => "Output variables",
            6 => "Local variables",
            7 => "Defined calibration objects",
            8 => "Referenced calibration objects",
            _ => "Unknown",
        }
    }
}

impl Display for Ch4Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CH: type={} ({}) elements={} children={}",
            self.get_type_str(),
            self.ch_type,
            self.ch_element_count,
            if self.ch_ch_first > 0 { "yes" } else { "no" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn test_ch_get_type_str() {
        let mut ch = Ch4Block::default();
        let expected = [
            (0, "Group"),
            (1, "Function"),
            (2, "Structure"),
            (3, "Map list"),
            (4, "Input variables"),
            (5, "Output variables"),
            (6, "Local variables"),
            (7, "Defined calibration objects"),
            (8, "Referenced calibration objects"),
            (255, "Unknown"),
        ];
        for (val, name) in expected {
            ch.ch_type = val;
            assert_eq!(ch.get_type_str(), name, "ch_type={val}");
        }
    }

    #[test]
    fn test_ch_calculate_block_size() {
        let ch = Ch4Block {
            ch_links: 4,
            ..Default::default()
        };
        assert_eq!(ch.calculate_block_size(), 64);
    }

    #[test]
    fn test_ch_display() {
        let ch = Ch4Block {
            ch_type: 1,
            ch_element_count: 5,
            ch_ch_first: 0,
            ch_links: 4,
            ..Default::default()
        };
        let display = format!("{ch}");
        assert!(display.contains("Function"));
        assert!(display.contains("elements=5"));
    }
}
