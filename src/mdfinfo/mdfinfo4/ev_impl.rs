//! Event block (EVBLOCK) parsing for MDF4 — spec section 6.11, Tables 48-54
//!
//! Describes events that occurred during measurement (triggers, markers, etc.).
//! Each event has a type, sync domain, cause, and can reference parent/range events.
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs::File;

use super::block_header_impl::{SharableBlocks, parse_block_short, read_meta_data};
use super::metadata::BlockType;
use crate::mdfinfo::block_chain::BlockChain;
use crate::mdfinfo::sym_buf_reader::SymBufReader;

pub use mdfr_core::blocks::ev_block::*;

/// Extension helper on [`Ev4Block`] using `mdfr`'s [`SharableBlocks`].
#[allow(dead_code)]
pub trait Ev4BlockExt {
    fn get_event_name(&self, sharable: &SharableBlocks) -> Result<Option<String>>;
}

impl Ev4BlockExt for Ev4Block {
    fn get_event_name(&self, sharable: &SharableBlocks) -> Result<Option<String>> {
        sharable.get_tx(self.ev_tx_name)
    }
}

/// Ev4 (Event) block struct parser
pub(super) fn parse_ev4_block(
    rdr: &mut SymBufReader<&File>,
    target: i64,
    mut position: i64,
) -> Result<(Ev4Block, i64)> {
    let (mut block, _header, pos) = parse_block_short(rdr, target, position)?;
    position = pos;
    let block: Ev4Block =
        binrw::BinReaderExt::read_le(&mut block).context("Error parsing ev block")?; // reads the fh block

    Ok((block, position))
}

/// parses Event blocks along with its linked comments, returns a hashmap of Ev4 block with position as key
pub fn parse_ev4(
    rdr: &mut SymBufReader<&File>,
    sharable: &mut SharableBlocks,
    target: i64,
    mut position: i64,
) -> Result<(HashMap<i64, Ev4Block>, i64)> {
    let mut ev: HashMap<i64, Ev4Block> = HashMap::new();
    if target > 0 {
        let (block, pos) = parse_ev4_block(rdr, target, position)?;
        position = pos;
        // Reads MD
        position = read_meta_data(rdr, sharable, block.ev_md_comment, position, BlockType::EV)?;
        // reads TX event name
        position = read_meta_data(rdr, sharable, block.ev_tx_name, position, BlockType::EV)?;
        let mut next_pointer = block.ev_ev_next;
        ev.insert(target, block);

        let mut chain = BlockChain::new("EV");
        chain.visit(target);
        while next_pointer > 0 {
            if !chain.visit(next_pointer) {
                break;
            }
            let block_start = next_pointer;
            let (block, pos) = parse_ev4_block(rdr, next_pointer, position)?;
            position = pos;
            // Reads MD
            position = read_meta_data(rdr, sharable, block.ev_md_comment, position, BlockType::EV)?;
            // reads TX event name
            position = read_meta_data(rdr, sharable, block.ev_tx_name, position, BlockType::EV)?;
            next_pointer = block.ev_ev_next;
            ev.insert(block_start, block);
        }
    }
    Ok((ev, position))
}
