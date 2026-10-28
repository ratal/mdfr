//! Header block (HDBLOCK) — spec section 6.1, Tables 13-14.
//!
//! The HDBLOCK is the root of the block hierarchy. It contains the start timestamp,
//! time zone information, and links to DG, FH, CH, AT, and EV blocks.
use alloc::format;
use alloc::string::ToString;
use binrw::binrw;
use chrono::{DateTime, FixedOffset, TimeZone};
use core::fmt;

/// Size of the HD block in bytes, located at file offset 64 (after the ID block).
pub const HD_LEN: u64 = 104;
/// Position of the first block after ID (64 bytes) and HD (104 bytes).
pub const HD_END_POSITION: i64 = 168;

/// HDBLOCK structure (MDF 4.2 spec, Table 13)
#[derive(Debug, Copy, Clone)]
#[binrw]
#[br(little)]
#[bw(little)]
#[repr(C)]
pub struct Hd4 {
    /// ##HD
    pub hd_id: [u8; 4],
    /// reserved
    pub hd_reserved: [u8; 4],
    /// Length of block in bytes
    pub hd_len: u64,
    /// # of links
    pub hd_link_counts: u64,
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

/// Default has a start time of 0: `no_std` has no clock. Use [`Hd4::new_at`] to set it.
impl Default for Hd4 {
    fn default() -> Self {
        Hd4 {
            hd_id: [35, 35, 72, 68], // ##HD
            hd_len: HD_LEN,
            hd_link_counts: 6,
            hd_reserved: [0u8; 4],
            hd_dg_first: 0,
            hd_fh_first: 0,
            hd_ch_first: 0,
            hd_at_first: 0,
            hd_ev_first: 0,
            hd_md_comment: 0,
            hd_start_time_ns: 0,
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

impl Hd4 {
    /// New HD with the given start time (ns since 1970-01-01).
    pub fn new_at(start_time_ns: u64) -> Self {
        Hd4 {
            hd_start_time_ns: start_time_ns,
            ..Default::default()
        }
    }
}

/// Hd4 display implementation
impl fmt::Display for Hd4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sec = (self.hd_start_time_ns / 1_000_000_000) as i64;
        let nsec = (self.hd_start_time_ns % 1_000_000_000) as u32;

        // Bit 1 of hd_time_flags: time offsets (tz + dst) are valid
        let offsets_valid = self.hd_time_flags & 0b10 != 0;
        // Bit 0 of hd_time_flags: timestamp is local time (not UTC)
        let is_local = self.hd_time_flags & 0b01 != 0;

        let datetime_str = if offsets_valid {
            // Total offset = timezone offset + DST offset, in seconds
            let total_offset_secs =
                (self.hd_tz_offset_min as i32 + self.hd_dst_offset_min as i32) * 60;
            match FixedOffset::east_opt(total_offset_secs) {
                Some(offset) => offset
                    .timestamp_opt(sec, nsec)
                    .single()
                    .map(|dt| dt.to_rfc3339())
                    .unwrap_or_else(|| "Invalid timestamp".to_string()),
                None => "Invalid offset".to_string(),
            }
        } else if is_local {
            // Local time stored, no valid offset info — display as-is with +00:00 annotation
            DateTime::from_timestamp(sec, nsec)
                .map(|dt| format!("{} (local, no offset)", dt.format("%Y-%m-%dT%H:%M:%S%.f")))
                .unwrap_or_else(|| "Invalid timestamp".to_string())
        } else {
            // Pure UTC
            DateTime::from_timestamp(sec, nsec)
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_else(|| "Invalid timestamp".to_string())
        };

        writeln!(f, "Time : {datetime_str}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use binrw::{BinReaderExt, BinWrite, io::Cursor};

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
        let hd = Hd4::new_at(1_700_000_000_000_000_000); // 2023-11-14T22:13:20 UTC
        let display = format!("{hd}");
        assert!(display.contains("Time :"));
        assert!(display.contains("2023-11-14"));
    }

    #[test]
    fn roundtrip_is_104_bytes() {
        let mut hd = Hd4::new_at(42);
        hd.hd_dg_first = 0x1234;
        let mut c = Cursor::new(Vec::new());
        hd.write_le(&mut c).unwrap();
        let buf = c.into_inner();
        assert_eq!(buf.len() as u64, HD_LEN);
        let back: Hd4 = Cursor::new(buf).read_le().unwrap();
        assert_eq!(back.hd_dg_first, 0x1234);
        assert_eq!(back.hd_start_time_ns, 42);
    }
}
