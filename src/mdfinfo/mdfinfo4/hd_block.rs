//! Header block (HDBLOCK) for MDF4 — spec section 6.1, Tables 13-14
//!
//! The HDBLOCK is the root of the block hierarchy. It contains the start timestamp,
//! time zone information, and links to DG, FH, CH, AT, and EV blocks.
use anyhow::{Context, Result};
use chrono::Local;
use std::fs::File;
use std::io::{Cursor, Read};

use super::block_header::{SharableBlocks, read_meta_data};
use super::metadata::BlockType;
use crate::mdfinfo::sym_buf_reader::SymBufReader;

pub use mdfr_core::blocks::hd::Hd4;

/// Creates an HD block with the current local time as start time.
pub fn new_hd4() -> Hd4 {
    Hd4::new_at(
        Local::now()
            .timestamp_nanos_opt()
            .map(|t| t as u64)
            .unwrap_or(0),
    )
}

/// Parses the HDBLOCK at file offset 168 (after the 64-byte ID block + 104-byte HD header).
/// Also reads the HD metadata comment block (Table 14).
pub fn hd4_parser(
    rdr: &mut SymBufReader<&File>,
    sharable: &mut SharableBlocks,
) -> Result<(Hd4, i64)> {
    let mut buf = [0u8; 104];
    rdr.read_exact(&mut buf)
        .context("could not read HD block buffer")?;
    let mut block = Cursor::new(buf);
    let hd: Hd4 = binrw::BinReaderExt::read_le(&mut block)
        .context("Could not parse HD block buffer into Hd4 struct")?;
    let position = read_meta_data(rdr, sharable, hd.hd_md_comment, 168, BlockType::HD)?;
    Ok((hd, position))
}
