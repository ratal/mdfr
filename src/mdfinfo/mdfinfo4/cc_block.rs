//! Channel Conversion block (CCBLOCK) for MDF4 — spec section 6.7, Tables 30-39
//!
//! Defines the conversion rule to transform raw channel values to physical values.
//! Supports 12 conversion types (Table 31): identity, linear, rational, polynomial,
//! tabular (value-to-value, value-to-text), algebraic formula, and more.
use anyhow::{Context, Result};
use std::fs::File;
use std::io::Cursor;

use super::block_header::{SharableBlocks, parse_block_short, read_meta_data};
use super::metadata::BlockType;
use crate::mdfinfo::sym_buf_reader::SymBufReader;

pub use mdfr_core::blocks::cc::{Cc4Block, CcVal};

/// reads and parses CC block and its linked metadata
pub(super) fn read_cc(
    rdr: &mut SymBufReader<&File>,
    target: &i64,
    mut position: i64,
    mut block: Cursor<Vec<u8>>,
    sharable: &mut SharableBlocks,
) -> Result<i64> {
    let cc_block: Cc4Block = binrw::BinReaderExt::read_le(&mut block)
        .context("Could nto read buffer into Cc4Block struct")?;
    position = read_meta_data(rdr, sharable, cc_block.cc_md_unit, position, BlockType::CC)?;
    position = read_meta_data(rdr, sharable, cc_block.cc_tx_name, position, BlockType::CC)?;

    for pointer in &cc_block.cc_ref {
        if !sharable.cc.contains_key(pointer)
            && !sharable.md_tx.contains_key(pointer)
            && *pointer != 0
        {
            let (ref_block, header, _pos) = parse_block_short(rdr, *pointer, position)?;
            position = pointer + header.hdr_len as i64;
            if "##TX".as_bytes() == header.hdr_id {
                // TX Block
                position = read_meta_data(rdr, sharable, *pointer, position, BlockType::CC)?
            } else {
                // CC Block
                position = read_cc(rdr, pointer, position, ref_block, sharable)?;
            }
        }
    }
    sharable.cc.insert(*target, cc_block);
    Ok(position)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cc(cc_type: u8, ref_count: u16, val_count: u16) -> Cc4Block {
        Cc4Block {
            cc_links: 4,
            cc_tx_name: 0,
            cc_md_unit: 0,
            cc_md_comment: 0,
            cc_cc_inverse: 0,
            cc_ref: vec![],
            cc_type,
            cc_precision: 0,
            cc_flags: 0,
            cc_ref_count: ref_count,
            cc_val_count: val_count,
            cc_phy_range_min: 0.0,
            cc_phy_range_max: 0.0,
            cc_val: CcVal::Real(vec![]),
        }
    }

    #[test]
    fn test_cc_get_cc_type_str() {
        let expected = [
            (0, "Identity"),
            (1, "Linear"),
            (2, "Rational"),
            (3, "Algebraic"),
            (4, "ValueToValueInterpolation"),
            (5, "ValueToValue"),
            (6, "ValueRangeToValue"),
            (7, "ValueToText"),
            (8, "ValueRangeToText"),
            (9, "TextToValue"),
            (10, "TextToText"),
            (11, "BitfieldToText"),
            (255, "Unknown"),
        ];
        for (val, name) in expected {
            let cc = make_cc(val, 0, 0);
            assert_eq!(cc.get_cc_type_str(), name, "cc_type={val}");
        }
    }

    #[test]
    fn test_cc_display() {
        let cc = make_cc(1, 2, 3);
        let display = format!("{cc}");
        assert!(display.contains("CC:"));
        assert!(display.contains("Linear"));
        assert!(display.contains("refs=2"));
        assert!(display.contains("vals=3"));
    }
}
