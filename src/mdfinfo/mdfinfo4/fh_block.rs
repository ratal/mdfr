//! File History block (FHBLOCK) for MDF4 — spec section 6.3, Table 18
//!
//! Each FHBLOCK records a creation or modification event with a timestamp and
//! an optional MD comment. FHBLOCKs form a linked list from `hd_fh_first`.
use anyhow::{Context, Result};
use chrono::Local;
use std::fs::File;
use std::io::{Cursor, Read};

use super::block_header::{SharableBlocks, read_meta_data};
use super::metadata::BlockType;
use crate::mdfinfo::block_chain::BlockChain;
use crate::mdfinfo::sym_buf_reader::SymBufReader;

pub use mdfr_core::blocks::fh::FhBlock;

/// Creates an FH block with the current local time.
pub fn new_fh_block() -> FhBlock {
    FhBlock::new_at(
        Local::now()
            .timestamp_nanos_opt()
            .map(|t| t as u64)
            .unwrap_or(0),
    )
}

/// Parses a single FHBLOCK from the file at the given target position.
fn parse_fh_block(
    rdr: &mut SymBufReader<&File>,
    target: i64,
    position: i64,
) -> Result<(FhBlock, i64)> {
    rdr.seek_relative(target - position)
        .context("Could not reach FH Block position")?; // change buffer position
    let mut buf = [0u8; 56];
    rdr.read_exact(&mut buf)
        .context("Could not read FH block buffer")?;
    let mut block = Cursor::new(buf);
    let fh: FhBlock = binrw::BinReaderExt::read_le(&mut block)
        .with_context(|| format!("Error parsing fh block into FhBlock struct \n{block:?}"))?; // reads the fh block
    Ok((fh, target + 56))
}

pub type Fh = Vec<FhBlock>;

/// Parses the FHBLOCK linked list starting from `target`, including MD comments.
/// Returns a Vec of all FHBLOCKs in the chain.
pub fn parse_fh(
    rdr: &mut SymBufReader<&File>,
    sharable: &mut SharableBlocks,
    target: i64,
    mut position: i64,
) -> Result<(Fh, i64)> {
    let mut fh: Fh = Vec::new();
    if target > 0 {
        let (block, pos) = parse_fh_block(rdr, target, position)?;
        position = pos;
        position = read_meta_data(rdr, sharable, block.fh_md_comment, position, BlockType::FH)?;
        let mut next_pointer = block.fh_fh_next;
        fh.push(block);
        let mut chain = BlockChain::new("FH");
        chain.visit(target);
        while next_pointer != 0 {
            if !chain.visit(next_pointer) {
                break;
            }
            let (block, pos) = parse_fh_block(rdr, next_pointer, position)?;
            position = pos;
            next_pointer = block.fh_fh_next;
            position = read_meta_data(rdr, sharable, block.fh_md_comment, position, BlockType::FH)?;
            fh.push(block);
        }
    }
    Ok((fh, position))
}
