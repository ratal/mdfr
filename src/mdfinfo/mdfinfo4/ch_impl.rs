//! Channel Hierarchy block (CHBLOCK) parsing for MDF4 — spec section 6.2, Tables 15-17
//!
//! CHBLOCKs organize channels into a display hierarchy tree (groups, functions,
//! input/output variables). The hierarchy is separate from the DG→CG→CN data structure.
use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::fs::File;

use super::block_header_impl::{SharableBlocks, parse_block_short, read_meta_data};
use super::metadata::BlockType;
use crate::mdfinfo::block_chain::BlockChain;
use crate::mdfinfo::sym_buf_reader::SymBufReader;

pub use mdfr_core::blocks::ch_block::Ch4Block;

/// parser Ch4Block
fn parser_ch4_block(
    rdr: &mut SymBufReader<&File>,
    target: i64,
    mut position: i64,
) -> Result<(Ch4Block, i64)> {
    let (mut block, header, pos) = parse_block_short(rdr, target, position)?;
    position = pos;
    // ch_links comes from the file and sizes ch_element, so check it against the
    // block length before binrw allocates the vector
    let ch_links: u64 =
        binrw::BinReaderExt::read_le(&mut block).context("Could not read ch_links")?;
    block.set_position(0);
    let max_links = header.hdr_len.saturating_sub(32) / 8;
    if !(4..=max_links).contains(&ch_links) {
        bail!(
            "CH block at 0x{target:x} declares {ch_links} links, expected 4 to {max_links} for a {} byte block",
            header.hdr_len
        );
    }
    let block: Ch4Block =
        binrw::BinReaderExt::read_le(&mut block).context("Error parsing ch block")?;

    Ok((block, position))
}

/// parses all CH blocks starting from target
pub fn parse_ch4(
    rdr: &mut SymBufReader<&File>,
    sharable: &mut SharableBlocks,
    target: i64,
    position: i64,
) -> Result<(HashMap<i64, Ch4Block>, i64)> {
    let mut ch = HashMap::new();
    // one tracker for the whole tree: it also stops a child link pointing back
    // at an ancestor, which would otherwise recurse until the stack overflows
    let mut chain = BlockChain::new("CH");
    let position = parse_ch4_level(rdr, sharable, target, position, &mut ch, &mut chain)?;
    Ok((ch, position))
}

/// parses one hierarchy level and, depth first, the levels below it
fn parse_ch4_level(
    rdr: &mut SymBufReader<&File>,
    sharable: &mut SharableBlocks,
    target: i64,
    mut position: i64,
    ch: &mut HashMap<i64, Ch4Block>,
    chain: &mut BlockChain,
) -> Result<i64> {
    let mut next_pointer = target;
    while next_pointer > 0 {
        if !chain.visit(next_pointer) {
            break;
        }
        let block_start = next_pointer;
        let (block, pos) = parser_ch4_block(rdr, next_pointer, position)?;
        position = pos;

        // Parse comments/names if exist
        position = read_meta_data(rdr, sharable, block.ch_tx_name, position, BlockType::CH)?;
        position = read_meta_data(rdr, sharable, block.ch_md_comment, position, BlockType::CH)?;

        // Traverse children
        if block.ch_ch_first > 0 {
            position = parse_ch4_level(rdr, sharable, block.ch_ch_first, position, ch, chain)?;
        }

        next_pointer = block.ch_ch_next;
        ch.insert(block_start, block);
    }
    Ok(position)
}
