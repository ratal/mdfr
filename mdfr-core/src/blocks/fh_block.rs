//! File History block (FHBLOCK) — spec section 6.3, Table 18.
//!
//! Each FHBLOCK records a creation or modification event with a timestamp and
//! an optional MD comment. FHBLOCKs form a linked list from `hd_fh_first`.
use alloc::format;
use alloc::string::{String, ToString};
use binrw::binrw;
use chrono::DateTime;
use core::fmt::{self, Display};

/// Size of the FH block in bytes.
pub const FH_LEN: u64 = 56;

/// FHBLOCK structure (MDF 4.2 spec, Table 18)
#[derive(Debug, Copy, Clone)]
#[binrw]
#[br(little)]
#[bw(little)]
#[repr(C)]
pub struct FhBlock {
    /// '##FH'
    pub fh_id: [u8; 4],
    /// reserved, must be 0
    pub fh_gap: [u8; 4],
    /// Length of block in bytes
    pub fh_len: u64,
    /// # of links
    pub fh_links: u64,
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
    pub fh_reserved: [u8; 3],
}

/// Default has a time of 0: `no_std` has no clock. Use [`FhBlock::new_at`] to set it.
impl Default for FhBlock {
    fn default() -> Self {
        FhBlock {
            fh_id: [35, 35, 70, 72], // '##FH'
            fh_gap: [0u8; 4],
            fh_len: FH_LEN,
            fh_links: 2,
            fh_fh_next: 0,
            fh_md_comment: 0,
            fh_time_ns: 0,
            fh_tz_offset_min: 0,
            fh_dst_offset_min: 0,
            fh_time_flags: 0,
            fh_reserved: [0u8; 3],
        }
    }
}

impl FhBlock {
    /// New FH with the given time (ns since 1970-01-01).
    pub fn new_at(time_ns: u64) -> Self {
        FhBlock {
            fh_time_ns: time_ns,
            ..Default::default()
        }
    }
}

impl Display for FhBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let secs = (self.fh_time_ns / 1_000_000_000) as i64;
        let nsecs = (self.fh_time_ns % 1_000_000_000) as u32;
        let datetime = DateTime::from_timestamp(secs, nsecs)
            .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
            .unwrap_or_else(|| String::from("Invalid timestamp"));
        let local_time = if self.fh_time_flags & 0b1 != 0 {
            "local"
        } else {
            "UTC"
        };
        write!(
            f,
            "FH  |  {} {}  |  tz={}, dst={}",
            datetime,
            local_time,
            format_minutes_as_offset(self.fh_tz_offset_min),
            format_minutes_as_offset(self.fh_dst_offset_min)
        )
    }
}

fn format_minutes_as_offset(minutes: i16) -> String {
    let sign = if minutes >= 0 { '+' } else { '-' };
    let abs_minutes = minutes.unsigned_abs();
    let hours = abs_minutes / 60;
    let mins = abs_minutes % 60;
    format!("{}{:02}:{:02}", sign, hours, mins)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use binrw::{BinReaderExt, BinWrite, io::Cursor};

    #[test]
    fn test_default() {
        let fh = FhBlock::default();
        assert_eq!(fh.fh_fh_next, 0);
        assert_eq!(fh.fh_md_comment, 0);
    }

    #[test]
    fn test_display_utc() {
        let fh = FhBlock::new_at(1_700_000_000_000_000_000);
        let display = format!("{fh}");
        assert!(display.contains("FH"));
        assert!(display.contains("2023-11-14"));
        assert!(display.contains("UTC"));
        assert!(display.contains("tz=+00:00"));
        assert!(display.contains("dst=+00:00"));
    }

    #[test]
    fn test_display_local() {
        let fh = FhBlock {
            fh_time_flags: 0b1,
            fh_tz_offset_min: 60,
            ..FhBlock::new_at(1_700_000_000_000_000_000)
        };
        let display = format!("{fh}");
        assert!(display.contains("local"));
        assert!(display.contains("tz=+01:00"));
    }

    #[test]
    fn roundtrip_is_56_bytes() {
        let fh = FhBlock::new_at(7);
        let mut c = Cursor::new(Vec::new());
        fh.write_le(&mut c).unwrap();
        let buf = c.into_inner();
        assert_eq!(buf.len() as u64, FH_LEN);
        let back: FhBlock = Cursor::new(buf).read_le().unwrap();
        assert_eq!(back.fh_time_ns, 7);
    }
}
