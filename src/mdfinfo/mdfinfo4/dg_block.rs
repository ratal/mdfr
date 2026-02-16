//! Data Group block (DGBLOCK) for MDF4 — spec section 6.4, Table 19
//!
//! Each DGBLOCK groups one or more CGBLOCKs that share the same data block.
//! DGBLOCKs form a linked list from `hd_dg_first`.
use anyhow::{Context, Result};
use binrw::{BinReaderExt, binrw};
use std::collections::{BTreeMap, HashMap};
use std::fmt::{self, Display};
use std::fs::File;
use std::io::{Cursor, Read};

use super::block_header::{read_meta_data, SharableBlocks};
use super::cg_block::{Cg4, parse_cg4, CG_F_VLSC, CG_F_VLSD};
use super::metadata::BlockType;
use crate::mdfinfo::sym_buf_reader::SymBufReader;

/// DGBLOCK structure (MDF 4.2 spec, Table 19)
#[derive(Debug, Copy, Clone)]
#[binrw]
#[br(little)]
#[repr(C)]
pub struct Dg4Block {
    /// ##DG
    dg_id: [u8; 4],
    /// reserved
    reserved: [u8; 4],
    /// Length of block in bytes
    pub dg_len: u64,
    /// # of links
    dg_links: u64,
    /// Pointer to next data group block (DGBLOCK) (can be NIL)
    pub dg_dg_next: i64,
    /// Pointer to first channel group block (CGBLOCK) (can be NIL)
    pub dg_cg_first: i64,
    // Pointer to data block (DTBLOCK or DZBLOCK for this block type) or data list block (DLBLOCK of data blocks or its HLBLOCK)  (can be NIL)
    pub dg_data: i64,
    /// comment
    dg_md_comment: i64,
    /// number of bytes used for record IDs. 0 no recordID
    pub dg_rec_id_size: u8,
    // reserved
    reserved_2: [u8; 7],
}

impl Default for Dg4Block {
    fn default() -> Self {
        Dg4Block {
            dg_id: [35, 35, 68, 71], // ##DG
            reserved: [0; 4],
            dg_len: 64,
            dg_links: 4,
            dg_dg_next: 0,
            dg_cg_first: 0,
            dg_data: 0,
            dg_md_comment: 0,
            dg_rec_id_size: 0,
            reserved_2: [0; 7],
        }
    }
}

impl Display for Dg4Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DG: rec_id_size={}", self.dg_rec_id_size)
    }
}

/// Parses a single DGBLOCK and its MD comment from the file.
fn parse_dg4_block(
    rdr: &mut SymBufReader<&File>,
    sharable: &mut SharableBlocks,
    target: i64,
    mut position: i64,
) -> Result<(Dg4Block, i64)> {
    rdr.seek_relative(target - position)
        .context("Could not reach position of Dg4 block")?;
    let mut buf = [0u8; 64];
    rdr.read_exact(&mut buf)
        .context("Could not read Dg4Blcok buffer")?;
    let mut block = Cursor::new(buf);
    let dg: Dg4Block = block
        .read_le()
        .context("Could not parse Dg4Block buffer into Dg4Block struct")?;
    position = target + 64;

    // Reads MD
    position = read_meta_data(rdr, sharable, dg.dg_md_comment, position, BlockType::DG)?;

    Ok((dg, position))
}

/// Parsed DG wrapper: the DGBLOCK and its linked CGBLOCKs (keyed by record ID)
#[derive(Debug, Clone)]
#[repr(C)]
pub struct Dg4 {
    /// DG Block
    pub block: Dg4Block,
    /// CG Block
    pub cg: HashMap<u64, Cg4>,
}

impl Display for Dg4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total_channels: usize = self.cg.values().map(|cg| cg.cn.len()).sum();
        write!(
            f,
            "DG: {} channel groups, {} channels",
            self.cg.len(),
            total_channels
        )
    }
}

/// Parses the DG→CG→CN block hierarchy. Traverses the DGBLOCK linked list
/// starting from `target` (`hd_dg_first`). Returns BTreeMap keyed by file position,
/// plus total CG and CN counts.
pub fn parse_dg4(
    rdr: &mut SymBufReader<&File>,
    target: i64,
    mut position: i64,
    sharable: &mut SharableBlocks,
) -> Result<(BTreeMap<i64, Dg4>, i64, usize, usize)> {
    let mut dg: BTreeMap<i64, Dg4> = BTreeMap::new();
    let mut n_cn: usize = 0;
    let mut n_cg: usize = 0;
    if target > 0 {
        let (block, pos) = parse_dg4_block(rdr, sharable, target, position)?;
        position = pos;
        let mut next_pointer = block.dg_dg_next;
        let (mut cg, pos, num_cg, num_cn) = parse_cg4(
            rdr,
            block.dg_cg_first,
            position,
            sharable,
            block.dg_rec_id_size,
        )?;
        n_cg += num_cg;
        n_cn += num_cn;
        identify_vlsd_cg(&mut cg);
        let dg_struct = Dg4 { block, cg };
        dg.insert(target, dg_struct);
        position = pos;
        while next_pointer > 0 {
            let block_start = next_pointer;
            let (block, pos) = parse_dg4_block(rdr, sharable, next_pointer, position)?;
            next_pointer = block.dg_dg_next;
            position = pos;
            let (mut cg, pos, num_cg, num_cn) = parse_cg4(
                rdr,
                block.dg_cg_first,
                position,
                sharable,
                block.dg_rec_id_size,
            )?;
            n_cg += num_cg;
            n_cn += num_cn;
            identify_vlsd_cg(&mut cg);
            let dg_struct = Dg4 { block, cg };
            dg.insert(block_start, dg_struct);
            position = pos;
        }
    }
    Ok((dg, position, n_cg, n_cn))
}

/// Links VLSD/VLSC channel groups (CG flag bit 0/1) to their matching CN in other groups.
/// A channel references a VLSD CG via `cn_data` pointing to the CG's block position.
fn identify_vlsd_cg(cg: &mut HashMap<u64, Cg4>) {
    // First find all VLSD/VLSC Channel Groups
    let mut vlsd: HashMap<i64, u64> = HashMap::new();
    for (rec_id, channel_group) in cg.iter() {
        if (channel_group.block.cg_flags & (CG_F_VLSD | CG_F_VLSC)) != 0 {
            // VLSD or VLSC channel group found
            vlsd.insert(channel_group.block_position, *rec_id);
        }
    }
    if !vlsd.is_empty() {
        // try to find corresponding channel in other channel group
        let mut vlsd_matching: HashMap<u64, (u64, i32)> = HashMap::new();
        for (target_rec_id, channel_group) in cg.iter() {
            for (target_rec_pos, cn) in channel_group.cn.iter() {
                if let Some(vlsd_rec_id) = vlsd.get(&cn.block.cn_data) {
                    // Found matching channel with VLSD_CG
                    vlsd_matching.insert(*vlsd_rec_id, (*target_rec_id, *target_rec_pos));
                }
            }
        }
        for (vlsd_rec_id, (target_rec_id, target_rec_pos)) in vlsd_matching {
            if let Some(vlsd_cg) = cg.get_mut(&vlsd_rec_id) {
                vlsd_cg.vlsd_cg = Some((target_rec_id, target_rec_pos));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::block_header::default_short_header;
    use super::super::cg_block::Cg4Block;
    use super::super::cn_block::Cn4;
    use super::super::metadata::BlockType;
    use std::collections::HashSet;

    #[test]
    fn test_dg_block_display() {
        let dg = Dg4Block::default();
        let display = format!("{dg}");
        assert!(display.contains("DG:"));
        assert!(display.contains("rec_id_size=0"));

        let dg = Dg4Block {
            dg_rec_id_size: 2,
            ..Default::default()
        };
        let display = format!("{dg}");
        assert!(display.contains("rec_id_size=2"));
    }

    #[test]
    fn test_dg_display() {
        // Empty DG
        let dg = Dg4 {
            block: Dg4Block::default(),
            cg: HashMap::new(),
        };
        let display = format!("{dg}");
        assert!(display.contains("0 channel groups"));
        assert!(display.contains("0 channels"));

        // DG with 2 CGs, one with 3 channels and one with 2
        let mut cg_map = HashMap::new();
        let mut cn1: HashMap<i32, Cn4> = HashMap::new();
        cn1.insert(0, Cn4::default());
        cn1.insert(8, Cn4::default());
        cn1.insert(16, Cn4::default());
        let cg1 = Cg4 {
            header: default_short_header(BlockType::CG),
            block: Cg4Block::default(),
            cn: cn1,
            master_channel_name: None,
            channel_names: HashSet::new(),
            record_length: 24,
            block_position: 100,
            vlsd_cg: None,
            invalid_bytes: None,
            sr: vec![],
        };
        cg_map.insert(0, cg1);

        let mut cn2: HashMap<i32, Cn4> = HashMap::new();
        cn2.insert(0, Cn4::default());
        cn2.insert(8, Cn4::default());
        let cg2 = Cg4 {
            header: default_short_header(BlockType::CG),
            block: Cg4Block::default(),
            cn: cn2,
            master_channel_name: None,
            channel_names: HashSet::new(),
            record_length: 16,
            block_position: 200,
            vlsd_cg: None,
            invalid_bytes: None,
            sr: vec![],
        };
        cg_map.insert(1, cg2);

        let dg = Dg4 {
            block: Dg4Block::default(),
            cg: cg_map,
        };
        let display = format!("{dg}");
        assert!(display.contains("2 channel groups"));
        assert!(display.contains("5 channels"));
    }
}
