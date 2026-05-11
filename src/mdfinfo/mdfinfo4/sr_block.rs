//! Sample Reduction block (SRBLOCK) for MDF4 — spec section 6.29, Table 72
//!
//! Provides pre-calculated statistics (mean, min, max) for efficient display of large
//! datasets. Linked from a CGBLOCK. Sync types: time, angle, distance, index, frequency.
use binrw::binrw;
use std::fmt::{self, Display};

/// SR4 Sample Reduction block struct (Section 6.29 of MDF 4.3 spec)
#[derive(Debug, PartialEq, Clone)]
#[binrw]
#[br(little)]
#[repr(C)]
pub struct Sr4Block {
    /// Pointer to next sample reduction block (SRBLOCK) (can be NIL)
    pub sr_sr_next: i64,
    /// Pointer to reduction data block (RD-/RV-/DZBLOCK or DL-/LD-/HLBLOCK)
    pub sr_data: i64,
    /// Number of cycles, i.e. number of sample reduction records
    pub sr_cycle_count: u64,
    /// Length of sample interval used to calculate the reduction records (unit depends on sr_sync_type)
    pub sr_interval: f64,
    /// Sync type: 1=time(s), 2=angle(rad), 3=distance(m), 4=index, 5=frequency(Hz)
    pub sr_sync_type: u8,
    /// Flags: bit 0 = invalidation bytes present, bit 1 = dominant invalidation bit
    pub sr_flags: u8,
    /// Reserved
    sr_reserved: [u8; 6],
}

impl Default for Sr4Block {
    fn default() -> Self {
        Sr4Block {
            sr_sr_next: 0,
            sr_data: 0,
            sr_cycle_count: 0,
            sr_interval: 0.0,
            sr_sync_type: 1,
            sr_flags: 0,
            sr_reserved: [0; 6],
        }
    }
}

// =============================================================================
// Sample Reduction (SR) Constants and Types
// =============================================================================

/// SR flag bit 0: Invalidation bytes present in reduction records
#[allow(dead_code)]
pub const SR_F_INVALIDATION_BYTES: u8 = 1 << 0;
/// SR flag bit 1: Dominant invalidation bit (if set, invalid bit indicates "at least one invalid sample")
#[allow(dead_code)]
pub const SR_F_DOMINANT_INVALIDATION: u8 = 1 << 1;

/// SR sync type: Time based (seconds)
pub const SR_SYNC_TIME: u8 = 1;
/// SR sync type: Angle based (radians)
pub const SR_SYNC_ANGLE: u8 = 2;
/// SR sync type: Distance based (meters)
pub const SR_SYNC_DISTANCE: u8 = 3;
/// SR sync type: Index based (sample count)
pub const SR_SYNC_INDEX: u8 = 4;
/// SR sync type: Frequency based (Hz)
pub const SR_SYNC_FREQUENCY: u8 = 5;

impl Sr4Block {
    /// Returns true if invalidation bytes are present in reduction records
    #[allow(dead_code)]
    pub fn has_invalidation_bytes(&self) -> bool {
        (self.sr_flags & SR_F_INVALIDATION_BYTES) != 0
    }

    /// Returns the sync type as a human-readable string
    pub fn get_sync_type_str(&self) -> &'static str {
        match self.sr_sync_type {
            SR_SYNC_TIME => "Time (seconds)",
            SR_SYNC_ANGLE => "Angle (radians)",
            SR_SYNC_DISTANCE => "Distance (meters)",
            SR_SYNC_INDEX => "Index (samples)",
            SR_SYNC_FREQUENCY => "Frequency (Hz)",
            _ => "Unknown",
        }
    }
}

impl Display for Sr4Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SR: cycle_count={} interval={} sync_type={} flags=0x{:02X}",
            self.sr_cycle_count,
            self.sr_interval,
            self.get_sync_type_str(),
            self.sr_flags
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sr_default() {
        let sr = Sr4Block::default();
        assert_eq!(sr.sr_sr_next, 0);
        assert_eq!(sr.sr_data, 0);
        assert_eq!(sr.sr_cycle_count, 0);
        assert_eq!(sr.sr_interval, 0.0);
        assert_eq!(sr.sr_sync_type, 1); // default is time
        assert_eq!(sr.sr_flags, 0);
    }

    #[test]
    fn test_sr_has_invalidation_bytes() {
        let mut sr = Sr4Block::default();
        assert!(!sr.has_invalidation_bytes());

        sr.sr_flags = SR_F_INVALIDATION_BYTES;
        assert!(sr.has_invalidation_bytes());

        sr.sr_flags = SR_F_DOMINANT_INVALIDATION; // bit 1 only
        assert!(!sr.has_invalidation_bytes());

        sr.sr_flags = SR_F_INVALIDATION_BYTES | SR_F_DOMINANT_INVALIDATION;
        assert!(sr.has_invalidation_bytes());
    }

    #[test]
    fn test_sr_get_sync_type_str() {
        let mut sr = Sr4Block {
            sr_sync_type: SR_SYNC_TIME,
            ..Default::default()
        };
        assert_eq!(sr.get_sync_type_str(), "Time (seconds)");

        sr.sr_sync_type = SR_SYNC_ANGLE;
        assert_eq!(sr.get_sync_type_str(), "Angle (radians)");

        sr.sr_sync_type = SR_SYNC_DISTANCE;
        assert_eq!(sr.get_sync_type_str(), "Distance (meters)");

        sr.sr_sync_type = SR_SYNC_INDEX;
        assert_eq!(sr.get_sync_type_str(), "Index (samples)");

        sr.sr_sync_type = SR_SYNC_FREQUENCY;
        assert_eq!(sr.get_sync_type_str(), "Frequency (Hz)");

        sr.sr_sync_type = 0;
        assert_eq!(sr.get_sync_type_str(), "Unknown");

        sr.sr_sync_type = 255;
        assert_eq!(sr.get_sync_type_str(), "Unknown");
    }

    #[test]
    fn test_sr_display() {
        let sr = Sr4Block {
            sr_cycle_count: 1000,
            sr_interval: 0.01,
            sr_sync_type: SR_SYNC_TIME,
            sr_flags: SR_F_INVALIDATION_BYTES,
            ..Default::default()
        };

        let display = format!("{sr}");
        assert!(display.contains("SR:"));
        assert!(display.contains("cycle_count=1000"));
        assert!(display.contains("interval=0.01"));
        assert!(display.contains("Time (seconds)"));
        assert!(display.contains("flags=0x01"));
    }
}
