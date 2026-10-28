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
