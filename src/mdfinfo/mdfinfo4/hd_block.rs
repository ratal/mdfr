//! Header block (HDBLOCK) for MDF4 — spec section 6.1, Tables 13-14
//!
//! The HDBLOCK is the root of the block hierarchy. It contains the start timestamp,
//! time zone information, and links to DG, FH, CH, AT, and EV blocks.
use anyhow::{Context, Result};
use binrw::{BinReaderExt, binrw};
use chrono::{DateTime, Local};
use std::fmt;
use std::fs::File;
use std::io::{Cursor, Read};

use super::block_header::{read_meta_data, SharableBlocks};
use super::metadata::BlockType;
use crate::mdfinfo::sym_buf_reader::SymBufReader;

/// HDBLOCK structure (MDF 4.2 spec, Table 13)
#[derive(Debug, Copy, Clone)]
#[binrw]
#[br(little)]
#[repr(C)]
pub struct Hd4 {
    /// ##HD
    hd_id: [u8; 4],
    /// reserved
    hd_reserved: [u8; 4],
    /// Length of block in bytes
    hd_len: u64,
    /// # of links
    hd_link_counts: u64,
    /// Pointer to the first data group block (DGBLOCK) (can be NIL)
    pub hd_dg_first: i64,
    /// Pointer to first file history block (FHBLOCK)
    /// There must be at least one FHBLOCK with information about the application which created the MDF file.
    pub hd_fh_first: i64,
    /// Pointer to first channel hierarchy block (CHBLOCK) (can be NIL).
    pub hd_ch_first: i64,
    /// Pointer to first attachment block (ATBLOCK) (can be NIL)
    pub hd_at_first: i64,
    /// Pointer to first event block (EVBLOCK) (can be NIL)
    pub hd_ev_first: i64,
    /// Pointer to the measurement file comment (TXBLOCK or MDBLOCK) (can be NIL) For MDBLOCK contents, see Table 14.
    pub hd_md_comment: i64,
    /// Data members
    /// Time stamp in nanoseconds elapsed since 00:00:00 01.01.1970 (UTC time or local time, depending on "local time" flag)
    pub hd_start_time_ns: u64,
    /// Time zone offset in minutes. The value must be in range [-720,720], i.e. it can be negative! For example a value of 60 (min) means UTC+1 time zone = Central European Time (CET). Only valid if "time offsets valid" flag is set in time flags.
    pub hd_tz_offset_min: i16,
    /// Daylight saving time (DST) offset in minutes for start time stamp. During the summer months, most regions observe a DST offset of 60 min (1 hour). Only valid if "time offsets valid" flag is set in time flags.
    pub hd_dst_offset_min: i16,
    /// Time flags The value contains the following bit flags (see HD_TF_xxx)
    pub hd_time_flags: u8,
    /// Time quality class (see HD_TC[35, 35, 72, 68]_xxx)
    pub hd_time_class: u8,
    /// Flags The value contains the following bit flags (see HD_FL_xxx):
    pub hd_flags: u8,
    /// reserved
    pub hd_reserved2: u8,
    /// Start angle in radians at start of measurement (only for angle synchronous measurements) Only valid if "start angle valid" flag is set. All angle values for angle synchronized master channels or events are relative to this start angle.
    pub hd_start_angle_rad: f64,
    /// Start distance in meters at start of measurement (only for distance synchronous measurements) Only valid if "start distance valid" flag is set. All distance values for distance synchronized master channels or events are relative to this start distance.
    pub hd_start_distance_m: f64,
}

impl Default for Hd4 {
    fn default() -> Self {
        Hd4 {
            hd_id: [35, 35, 72, 68], // ##HD
            hd_len: 104,
            hd_link_counts: 6,
            hd_reserved: [0u8; 4],
            hd_dg_first: 0,
            hd_fh_first: 0,
            hd_ch_first: 0,
            hd_at_first: 0,
            hd_ev_first: 0,
            hd_md_comment: 0,
            hd_start_time_ns: Local::now()
                .timestamp_nanos_opt()
                .map(|t| t as u64)
                .unwrap_or(0),
            hd_tz_offset_min: 0,
            hd_dst_offset_min: 0,
            hd_time_flags: 0,
            hd_time_class: 0,
            hd_flags: 0,
            hd_reserved2: 0,
            hd_start_angle_rad: 0.0,
            hd_start_distance_m: 0.0,
        }
    }
}

/// Hd4 display implementation
impl fmt::Display for Hd4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sec = self.hd_start_time_ns / 1000000000;
        let nsec = (self.hd_start_time_ns - sec * 1000000000) as u32;
        let naive = DateTime::from_timestamp(sec as i64, nsec).unwrap_or_default();
        writeln!(f, "Time : {} ", naive.to_rfc3339())
    }
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
    let hd: Hd4 = block
        .read_le()
        .context("Could not parse HD block buffer into Hd4 struct")?;
    let position = read_meta_data(rdr, sharable, hd.hd_md_comment, 168, BlockType::HD)?;
    Ok((hd, position))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let hd = Hd4::default();
        assert_eq!(hd.hd_len, 104);
        assert_eq!(hd.hd_link_counts, 6);
        assert_eq!(hd.hd_dg_first, 0);
        assert_eq!(hd.hd_fh_first, 0);
    }

    #[test]
    fn test_display() {
        let hd = Hd4 {
            hd_start_time_ns: 1_700_000_000_000_000_000, // 2023-11-14T22:13:20 UTC
            ..Default::default()
        };
        let display = format!("{hd}");
        assert!(display.contains("Time :"));
        assert!(display.contains("2023-11-14"));
    }
}
