//! Event block (EVBLOCK) for MDF4 — spec section 6.11, Tables 48-54
//!
//! Describes events that occurred during measurement (triggers, markers, etc.).
//! Each event has a type, sync domain, cause, and can reference parent/range events.
use anyhow::{Context, Result};
use binrw::{BinReaderExt, binrw};
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::fs::File;

use super::block_header::{SharableBlocks, parse_block_short, read_meta_data};
use super::metadata::BlockType;
use crate::mdfinfo::sym_buf_reader::SymBufReader;

/// Ev4 Event block struct
#[derive(Debug, Clone)]
#[binrw]
#[br(little)]
#[repr(C)]
pub struct Ev4Block {
    //ev_id: [u8; 4],  // DG
    //reserved: [u8; 4],  // reserved
    //ev_len: u64,      // Length of block in bytes
    /// # of links
    pub ev_links: u64,
    /// Link to next EVBLOCK (linked list) (can be NIL)
    pub ev_ev_next: i64,
    /// Referencing link to EVBLOCK with parent event (can be NIL).
    pub ev_ev_parent: i64,
    /// Referencing link to EVBLOCK with event that defines the beginning of a range (can be NIL, must be NIL if ev_range_type != 2).
    pub ev_ev_range: i64,
    /// Pointer to TXBLOCK with event name (can be NIL) Name must be according to naming rules stated in 4.4.2 Naming Rules. If available, the name of a named trigger condition should be used as event name. Other event types may have individual names or no names.
    pub ev_tx_name: i64,
    /// Pointer to TX/MDBLOCK with event comment and additional information, e.g. trigger condition or formatted user comment text (can be NIL)
    pub ev_md_comment: i64,
    #[br(if(ev_links > 5), little, count = ev_links.saturating_sub(5))]
    /// links
    links: Vec<i64>,

    /// Event type (see EV_T_xxx)
    pub ev_type: u8,
    /// Sync type (see EV_S_xxx)
    pub ev_sync_type: u8,
    /// Range Type (see EV_R_xxx)
    pub ev_range_type: u8,
    /// Cause of event (see EV_C_xxx)
    pub ev_cause: u8,
    /// flags (see EV_F_xxx)
    pub ev_flags: u8,
    /// Reserved
    ev_reserved: [u8; 3],
    /// Length M of ev_scope list. Can be zero.
    pub ev_scope_count: u32,
    /// Length N of ev_at_reference list, i.e. number of attachments for this event. Can be zero.
    pub ev_attachment_count: u16,
    /// Creator index, i.e. zero-based index of FHBLOCK in global list of FHBLOCKs that specifies which application has created or changed this event (e.g. when generating event offline).
    pub ev_creator_index: u16,
    /// Base value for synchronization value.
    pub ev_sync_base_value: i64,
    /// Factor for event synchronization value.
    pub ev_sync_factor: f64,
}

// Event type constants (ev_type)
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

// Event sync type constants (ev_sync_type)
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

// Event cause constants (ev_cause)
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

#[allow(dead_code)]
impl Ev4Block {
    /// Returns the event name from sharable blocks
    pub fn get_event_name(&self, sharable: &SharableBlocks) -> Result<Option<String>> {
        sharable.get_tx(self.ev_tx_name)
    }

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

    /// Returns the list of scope link positions (first ev_scope_count items from links)
    /// These point to CGBLOCKs or CNBLOCKs that this event applies to
    pub fn get_scope_links(&self) -> &[i64] {
        if self.links.is_empty() || self.ev_scope_count == 0 {
            &[]
        } else {
            let end = (self.ev_scope_count as usize).min(self.links.len());
            &self.links[0..end]
        }
    }

    /// Returns the list of attachment link positions (items after scope links)
    /// These point to ATBLOCKs associated with this event
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
        // 16 (short header) + 8 (link count) + 8*links + 32 (data members)
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

/// Ev4 (Event) block struct parser
pub(super) fn parse_ev4_block(
    rdr: &mut SymBufReader<&File>,
    target: i64,
    mut position: i64,
) -> Result<(Ev4Block, i64)> {
    let (mut block, _header, pos) = parse_block_short(rdr, target, position)?;
    position = pos;
    let block: Ev4Block = block.read_le().context("Error parsing ev block")?; // reads the fh block

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

        while next_pointer > 0 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ev_get_event_type_str() {
        let mut ev = Ev4Block::default();
        let expected = [
            (EV_T_RECORDING, "Recording"),
            (EV_T_RECORDING_INTERRUPT, "Recording Interrupt"),
            (EV_T_ACQUISITION_INTERRUPT, "Acquisition Interrupt"),
            (EV_T_TRIGGER, "Trigger"),
            (EV_T_MARKER, "Marker"),
            (255, "Unknown"),
        ];
        for (val, name) in expected {
            ev.ev_type = val;
            assert_eq!(ev.get_event_type_str(), name, "ev_type={val}");
        }
    }

    #[test]
    fn test_ev_get_sync_type_str() {
        let mut ev = Ev4Block::default();
        let expected = [
            (EV_S_NONE, "None"),
            (EV_S_TIME, "Time"),
            (EV_S_ANGLE, "Angle"),
            (EV_S_DISTANCE, "Distance"),
            (EV_S_INDEX, "Index"),
            (EV_S_FREQUENCY, "Frequency"),
            (255, "Unknown"),
        ];
        for (val, name) in expected {
            ev.ev_sync_type = val;
            assert_eq!(ev.get_sync_type_str(), name, "ev_sync_type={val}");
        }
    }

    #[test]
    fn test_ev_get_cause_str() {
        let mut ev = Ev4Block::default();
        let expected = [
            (EV_C_OTHER, "Other/Unknown"),
            (EV_C_ERROR, "Error"),
            (EV_C_TOOL, "Tool Internal"),
            (EV_C_SCRIPT, "Script"),
            (EV_C_USER, "User"),
            (255, "Unknown"),
        ];
        for (val, name) in expected {
            ev.ev_cause = val;
            assert_eq!(ev.get_cause_str(), name, "ev_cause={val}");
        }
    }

    #[test]
    fn test_ev_get_range_type_str() {
        let mut ev = Ev4Block::default();
        let expected = [(0, "Point"), (1, "Beginning"), (2, "End"), (255, "Unknown")];
        for (val, name) in expected {
            ev.ev_range_type = val;
            assert_eq!(ev.get_range_type_str(), name, "ev_range_type={val}");
        }
    }

    #[test]
    fn test_ev_get_sync_value() {
        let mut ev = Ev4Block {
            ev_sync_base_value: 1000,
            ev_sync_factor: 1e-9,
            ..Default::default()
        };
        assert!((ev.get_sync_value() - 1e-6).abs() < 1e-15);

        ev.ev_sync_base_value = 0;
        ev.ev_sync_factor = 1.0;
        assert_eq!(ev.get_sync_value(), 0.0);

        ev.ev_sync_base_value = -100;
        ev.ev_sync_factor = 0.5;
        assert_eq!(ev.get_sync_value(), -50.0);
    }

    #[test]
    fn test_ev_get_scope_links() {
        let mut ev = Ev4Block::default();

        // Empty links, scope_count=0
        assert!(ev.get_scope_links().is_empty());

        // scope_count > 0 but links empty
        ev.ev_scope_count = 2;
        assert!(ev.get_scope_links().is_empty());

        // Populated links with scope_count=2
        ev.links = vec![100, 200, 300, 400];
        ev.ev_scope_count = 2;
        assert_eq!(ev.get_scope_links(), &[100, 200]);

        // scope_count > links.len()
        ev.ev_scope_count = 10;
        assert_eq!(ev.get_scope_links(), &[100, 200, 300, 400]);
    }

    #[test]
    fn test_ev_get_attachment_links() {
        let mut ev = Ev4Block::default();

        // Empty links
        assert!(ev.get_attachment_links().is_empty());

        // attachment_count > 0 but links empty
        ev.ev_attachment_count = 1;
        assert!(ev.get_attachment_links().is_empty());

        // scope_count=2, attachment_count=1, links has enough
        ev.links = vec![100, 200, 300, 400];
        ev.ev_scope_count = 2;
        ev.ev_attachment_count = 1;
        assert_eq!(ev.get_attachment_links(), &[300]);

        // scope_count=0, attachment_count=2
        ev.ev_scope_count = 0;
        ev.ev_attachment_count = 2;
        assert_eq!(ev.get_attachment_links(), &[100, 200]);

        // start >= links.len()
        ev.ev_scope_count = 10;
        ev.ev_attachment_count = 1;
        assert!(ev.get_attachment_links().is_empty());
    }

    #[test]
    fn test_ev_calculate_block_size() {
        let mut ev = Ev4Block {
            ev_links: 5,
            ..Default::default()
        };
        assert_eq!(ev.calculate_block_size(), 16 + 8 + 5 * 8 + 32); // 96

        ev.ev_links = 8;
        assert_eq!(ev.calculate_block_size(), 16 + 8 + 8 * 8 + 32); // 120
    }

    #[test]
    fn test_ev_display() {
        let ev = Ev4Block {
            ev_type: EV_T_TRIGGER,
            ev_sync_type: EV_S_TIME,
            ev_cause: EV_C_USER,
            ev_range_type: 0,
            ev_scope_count: 3,
            ev_attachment_count: 1,
            ..Default::default()
        };

        let display = format!("{ev}");
        assert!(display.contains("EV"));
        assert!(display.contains("Trigger"));
        assert!(display.contains("Time"));
        assert!(display.contains("User"));
        assert!(display.contains("Point"));
        assert!(display.contains("scopes=3"));
        assert!(display.contains("attachments=1"));
    }
}
