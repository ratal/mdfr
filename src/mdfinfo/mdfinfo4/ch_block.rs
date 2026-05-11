//! Channel Hierarchy block (CHBLOCK) for MDF4 — spec section 6.2, Tables 15-17
//!
//! CHBLOCKs organize channels into a display hierarchy tree (groups, functions,
//! input/output variables). The hierarchy is separate from the DG→CG→CN data structure.
use anyhow::{Context, Result};
use binrw::{BinReaderExt, binrw};
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::fs::File;

use super::block_header::{SharableBlocks, parse_block_short, read_meta_data};
use super::metadata::BlockType;
use crate::mdfinfo::sym_buf_reader::SymBufReader;

/// CHBLOCK structure (MDF 4.2 spec, Table 15)
#[derive(Debug, PartialEq, Eq, Default, Clone)]
#[binrw]
#[br(little)]
#[repr(C)]
pub struct Ch4Block {
    // header
    // ##CH
    // ch_id [u8;4]
    /// reserved
    // reserved: [u8; 4],
    /// Length of block in bytes
    // pub ch_len: u64,
    /// # of links
    pub ch_links: u64,

    // link section
    /// link to next CHBLOCK at this hierarchy level
    pub ch_ch_next: i64,
    /// link to first CHBLOCK at the next hierarchy level (child)
    pub ch_ch_first: i64,
    /// link to TXBLOCK with the name of the hierarchy level
    pub ch_tx_name: i64,
    /// link to MDBLOCK with a comment/description
    pub ch_md_comment: i64,
    /// list of elements in this hierarchy level
    #[br(count = ch_links - 4)]
    pub ch_element: Vec<i64>,

    // data section
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
        // 16 (short header) + 8 (link count) + 8*ch_links (links) + 8 (data members)
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

/// parser Ch4Block
fn parser_ch4_block(
    rdr: &mut SymBufReader<&File>,
    target: i64,
    mut position: i64,
) -> Result<(Ch4Block, i64)> {
    let (mut block, _header, pos) = parse_block_short(rdr, target, position)?;
    position = pos;
    let block: Ch4Block = block.read_le().context("Error parsing ch block")?;

    Ok((block, position))
}

/// parses all CH blocks starting from target
pub fn parse_ch4(
    rdr: &mut SymBufReader<&File>,
    sharable: &mut SharableBlocks,
    target: i64,
    mut position: i64,
) -> Result<(HashMap<i64, Ch4Block>, i64)> {
    let mut ch = HashMap::new();
    let mut next_pointer = target;
    while next_pointer > 0 {
        let block_start = next_pointer;
        let (block, pos) = parser_ch4_block(rdr, next_pointer, position)?;
        position = pos;

        // Parse comments/names if exist
        position = read_meta_data(rdr, sharable, block.ch_tx_name, position, BlockType::CH)?;
        position = read_meta_data(rdr, sharable, block.ch_md_comment, position, BlockType::CH)?;

        // Traverse children
        if block.ch_ch_first > 0 {
            let (children, pos) = parse_ch4(rdr, sharable, block.ch_ch_first, position)?;
            position = pos;
            ch.extend(children);
        }

        next_pointer = block.ch_ch_next;
        ch.insert(block_start, block);
    }
    Ok((ch, position))
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(ch.calculate_block_size(), 16 + 8 + 4 * 8 + 8); // 64

        let ch = Ch4Block {
            ch_links: 7,
            ..Default::default()
        };
        assert_eq!(ch.calculate_block_size(), 16 + 8 + 7 * 8 + 8); // 88

        let ch = Ch4Block {
            ch_links: 10,
            ..Default::default()
        };
        assert_eq!(ch.calculate_block_size(), 16 + 8 + 10 * 8 + 8); // 112
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
        assert!(display.contains("CH:"));
        assert!(display.contains("Function"));
        assert!(display.contains("elements=5"));
        assert!(display.contains("children=no"));

        let ch = Ch4Block {
            ch_ch_first: 100,
            ..ch
        };
        let display = format!("{ch}");
        assert!(display.contains("children=yes"));
    }
}
