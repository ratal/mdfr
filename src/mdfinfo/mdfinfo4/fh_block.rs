//! File History block (FHBLOCK) for MDF4 — spec section 6.3, Table 18
//!
//! Each FHBLOCK records a creation or modification event with a timestamp and
//! an optional MD comment. FHBLOCKs form a linked list from `hd_fh_first`.
use anyhow::{Context, Result};
use binrw::{BinReaderExt, binrw};
use chrono::{DateTime, Local};
use std::fmt::{self, Display};
use std::fs::File;
use std::io::{Cursor, Read};

use super::block_header::{SharableBlocks, read_meta_data};
use super::metadata::BlockType;
use crate::mdfinfo::sym_buf_reader::SymBufReader;

/// FHBLOCK structure (MDF 4.2 spec, Table 18)
#[derive(Debug, Copy, Clone)]
#[binrw]
#[br(little)]
#[repr(C)]
pub struct FhBlock {
    /// '##FH'
    fh_id: [u8; 4],
    /// reserved, must be 0
    fh_gap: [u8; 4],
    /// Length of block in bytes
    fh_len: u64,
    /// # of links
    fh_links: u64,
    /// Link to next FHBLOCK (can be NIL if list finished)
    pub fh_fh_next: i64,
    /// Link to MDBLOCK containing comment about the creation or modification of the MDF file.
    pub fh_md_comment: i64,
    /// time stamp in nanosecs
    pub fh_time_ns: u64,
    /// time zone offset on minutes
    pub fh_tz_offset_min: i16,
    /// daylight saving time offset in minutes for start time stamp
    pub fh_dst_offset_min: i16,
    /// time flags, but 1 local, bit 2 time offsets
    pub fh_time_flags: u8,
    /// reserved
    fh_reserved: [u8; 3],
}

impl Default for FhBlock {
    fn default() -> Self {
        FhBlock {
            fh_id: [35, 35, 70, 72], // '##FH'
            fh_gap: [0u8; 4],
            fh_len: 56,
            fh_links: 2,
            fh_fh_next: 0,
            fh_md_comment: 0,
            fh_time_ns: Local::now()
                .timestamp_nanos_opt()
                .map(|t| t as u64)
                .unwrap_or(0),
            fh_tz_offset_min: 0,
            fh_dst_offset_min: 0,
            fh_time_flags: 0,
            fh_reserved: [0u8; 3],
        }
    }
}

impl Display for FhBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Convert nanoseconds to datetime
        let secs = (self.fh_time_ns / 1_000_000_000) as i64;
        let nsecs = (self.fh_time_ns % 1_000_000_000) as u32;
        let datetime = DateTime::from_timestamp(secs, nsecs)
            .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
            .unwrap_or_else(|| "Invalid timestamp".to_string());
        let local_time = if self.fh_time_flags & 0b1 != 0 {
            "local"
        } else {
            "UTC"
        };
        write!(
            f,
            "FH: {} ({}) tz_offset={}min dst_offset={}min",
            datetime, local_time, self.fh_tz_offset_min, self.fh_dst_offset_min
        )
    }
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
    let fh: FhBlock = block
        .read_le()
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
    let (block, pos) = parse_fh_block(rdr, target, position)?;
    position = pos;
    position = read_meta_data(rdr, sharable, block.fh_md_comment, position, BlockType::FH)?;
    let mut next_pointer = block.fh_fh_next;
    fh.push(block);
    while next_pointer != 0 {
        let (block, pos) = parse_fh_block(rdr, next_pointer, position)?;
        position = pos;
        next_pointer = block.fh_fh_next;
        position = read_meta_data(rdr, sharable, block.fh_md_comment, position, BlockType::FH)?;
        fh.push(block);
    }
    Ok((fh, position))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let fh = FhBlock::default();
        assert_eq!(fh.fh_fh_next, 0);
        assert_eq!(fh.fh_md_comment, 0);
        assert!(fh.fh_time_ns > 0); // current time
    }

    #[test]
    fn test_display_utc() {
        let fh = FhBlock {
            fh_time_ns: 1_700_000_000_000_000_000, // 2023-11-14T22:13:20 UTC
            fh_time_flags: 0,                      // UTC
            fh_tz_offset_min: 0,
            fh_dst_offset_min: 0,
            ..Default::default()
        };
        let display = format!("{fh}");
        assert!(display.contains("FH:"));
        assert!(display.contains("2023-11-14"));
        assert!(display.contains("UTC"));
        assert!(display.contains("tz_offset=0min"));
    }

    #[test]
    fn test_display_local() {
        let fh = FhBlock {
            fh_time_ns: 1_700_000_000_000_000_000,
            fh_time_flags: 0b1, // local time
            fh_tz_offset_min: 60,
            fh_dst_offset_min: 0,
            ..Default::default()
        };
        let display = format!("{fh}");
        assert!(display.contains("local"));
        assert!(display.contains("tz_offset=60min"));
    }
}
