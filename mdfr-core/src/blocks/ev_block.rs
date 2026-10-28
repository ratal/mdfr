//! Event block (EVBLOCK) for MDF4 — spec section 6.11, Tables 48-54.
//!
//! Describes events that occurred during measurement (triggers, markers, etc.).
//! Each event has a type, sync domain, cause, and can reference parent/range events.
use alloc::vec::Vec;
use binrw::binrw;
use core::fmt::{self, Display};

/// Ev4 Event block struct (body following the 16-byte short header).
#[derive(Debug, Clone, PartialEq)]
#[binrw]
#[br(little)]
#[bw(little)]
#[repr(C)]
pub struct Ev4Block {
    /// # of links
    pub ev_links: u64,
    /// Link to next EVBLOCK (linked list) (can be NIL)
    pub ev_ev_next: i64,
    /// Referencing link to EVBLOCK with parent event (can be NIL)
    pub ev_ev_parent: i64,
    /// Referencing link to EVBLOCK with event that defines the beginning of a range
    pub ev_ev_range: i64,
    /// Pointer to TXBLOCK with event name (can be NIL)
    pub ev_tx_name: i64,
    /// Pointer to TX/MDBLOCK with event comment and additional information (can be NIL)
    pub ev_md_comment: i64,
    /// Additional scope and attachment links (`ev_links - 5`)
    #[br(if(ev_links > 5), little, count = ev_links.saturating_sub(5))]
    pub links: Vec<i64>,

    /// Event type (see `EV_T_xxx`)
    pub ev_type: u8,
    /// Sync type (see `EV_S_xxx`)
    pub ev_sync_type: u8,
    /// Range Type (see `EV_R_xxx`)
    pub ev_range_type: u8,
    /// Cause of event (see `EV_C_xxx`)
    pub ev_cause: u8,
    /// flags (see `EV_F_xxx`)
    pub ev_flags: u8,
    /// Reserved
    pub ev_reserved: [u8; 3],
    /// Length M of `ev_scope` list. Can be zero.
    pub ev_scope_count: u32,
    /// Length N of `ev_at_reference` list, i.e. number of attachments for this event. Can be zero.
    pub ev_attachment_count: u16,
    /// Creator index (zero-based index of FHBLOCK)
    pub ev_creator_index: u16,
    /// Base value for synchronization value.
    pub ev_sync_base_value: i64,
    /// Factor for event synchronization value.
    pub ev_sync_factor: f64,
}

/// Recording event
pub const EV_T_RECORDING: u8 = 0;
/// Recording interrupt event
pub const EV_T_RECORDING_INTERRUPT: u8 = 1;
/// Acquisition interrupt event
pub const EV_T_ACQUISITION_INTERRUPT: u8 = 2;
/// Trigger (start/stop) event
pub const EV_T_TRIGGER: u8 = 3;
/// Marker event (user-defined)
pub const EV_T_MARKER: u8 = 4;

/// No sync
pub const EV_S_NONE: u8 = 0;
/// Time sync
pub const EV_S_TIME: u8 = 1;
/// Angle sync
pub const EV_S_ANGLE: u8 = 2;
/// Distance sync
pub const EV_S_DISTANCE: u8 = 3;
/// Index sync
pub const EV_S_INDEX: u8 = 4;
/// Frequency sync
pub const EV_S_FREQUENCY: u8 = 5;

/// Unknown/other cause
pub const EV_C_OTHER: u8 = 0;
/// Error cause
pub const EV_C_ERROR: u8 = 1;
/// Tool-internal cause
pub const EV_C_TOOL: u8 = 2;
/// Script cause
pub const EV_C_SCRIPT: u8 = 3;
/// User cause
pub const EV_C_USER: u8 = 4;

impl Ev4Block {
    /// Returns the event type as a string description
    pub fn get_event_type_str(&self) -> &'static str {
        match self.ev_type {
            EV_T_RECORDING => "Recording",
            EV_T_RECORDING_INTERRUPT => "Recording Interrupt",
            EV_T_ACQUISITION_INTERRUPT => "Acquisition Interrupt",
            EV_T_TRIGGER => "Trigger",
            EV_T_MARKER => "Marker",
            _ => "Unknown",
        }
    }

    /// Returns the synchronization type as a string description
    pub fn get_sync_type_str(&self) -> &'static str {
        match self.ev_sync_type {
            EV_S_NONE => "None",
            EV_S_TIME => "Time",
            EV_S_ANGLE => "Angle",
            EV_S_DISTANCE => "Distance",
            EV_S_INDEX => "Index",
            EV_S_FREQUENCY => "Frequency",
            _ => "Unknown",
        }
    }

    /// Returns the event cause as a string description
    pub fn get_cause_str(&self) -> &'static str {
        match self.ev_cause {
            EV_C_OTHER => "Other/Unknown",
            EV_C_ERROR => "Error",
            EV_C_TOOL => "Tool Internal",
            EV_C_SCRIPT => "Script",
            EV_C_USER => "User",
            _ => "Unknown",
        }
    }

    /// Returns the synchronization value computed from base value and factor
    pub fn get_sync_value(&self) -> f64 {
        self.ev_sync_base_value as f64 * self.ev_sync_factor
    }

    /// Returns the list of scope link positions (first `ev_scope_count` items from `links`)
    pub fn get_scope_links(&self) -> &[i64] {
        if self.links.is_empty() || self.ev_scope_count == 0 {
            &[]
        } else {
            let end = (self.ev_scope_count as usize).min(self.links.len());
            &self.links[0..end]
        }
    }

    /// Returns the list of attachment link positions (items after scope links)
    pub fn get_attachment_links(&self) -> &[i64] {
        if self.links.is_empty() || self.ev_attachment_count == 0 {
            &[]
        } else {
            let start = self.ev_scope_count as usize;
            if start >= self.links.len() {
                &[]
            } else {
                let end = (start + self.ev_attachment_count as usize).min(self.links.len());
                &self.links[start..end]
            }
        }
    }

    /// Calculates the block size for writing
    pub fn calculate_block_size(&self) -> i64 {
        16 + 8 + (self.ev_links * 8) as i64 + 32
    }

    /// Returns the range type as a string description
    pub fn get_range_type_str(&self) -> &'static str {
        match self.ev_range_type {
            0 => "Point",
            1 => "Beginning",
            2 => "End",
            _ => "Unknown",
        }
    }
}

impl Default for Ev4Block {
    fn default() -> Self {
        Ev4Block {
            ev_links: 5,
            ev_ev_next: 0,
            ev_ev_parent: 0,
            ev_ev_range: 0,
            ev_tx_name: 0,
            ev_md_comment: 0,
            links: Vec::new(),
            ev_type: 0,
            ev_sync_type: 0,
            ev_range_type: 0,
            ev_cause: 0,
            ev_flags: 0,
            ev_reserved: [0; 3],
            ev_scope_count: 0,
            ev_attachment_count: 0,
            ev_creator_index: 0,
            ev_sync_base_value: 0,
            ev_sync_factor: 0.0,
        }
    }
}

impl Display for Ev4Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EV  |  {} | sync={} | cause={} | range={} | scopes={}, attachments={}",
            self.get_event_type_str(),
            self.get_sync_type_str(),
            self.get_cause_str(),
            self.get_range_type_str(),
            self.ev_scope_count,
            self.ev_attachment_count
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{format, vec};

    #[test]
    fn test_ev_strings_and_links() {
        let mut ev = Ev4Block {
            ev_type: EV_T_TRIGGER,
            ev_sync_type: EV_S_TIME,
            ev_cause: EV_C_USER,
            ev_scope_count: 2,
            ev_attachment_count: 1,
            links: vec![100, 200, 300],
            ..Default::default()
        };
        assert_eq!(ev.get_event_type_str(), "Trigger");
        assert_eq!(ev.get_sync_type_str(), "Time");
        assert_eq!(ev.get_cause_str(), "User");
        assert_eq!(ev.get_scope_links(), &[100, 200]);
        assert_eq!(ev.get_attachment_links(), &[300]);
        assert!(format!("{ev}").contains("Trigger"));
        ev.ev_links = 8;
        assert_eq!(ev.calculate_block_size(), 120);
    }
}
