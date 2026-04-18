//! Channel Conversion block (CCBLOCK) for MDF4 — spec section 6.7, Tables 30-39
//!
//! Defines the conversion rule to transform raw channel values to physical values.
//! Supports 12 conversion types (Table 31): identity, linear, rational, polynomial,
//! tabular (value-to-value, value-to-text), algebraic formula, and more.
use anyhow::{Context, Result};
use binrw::{BinReaderExt, binrw};
use std::fmt::{self, Display};
use std::fs::File;
use std::io::Cursor;

use super::block_header::{SharableBlocks, parse_block_short, read_meta_data};
use super::metadata::BlockType;
use crate::mdfinfo::sym_buf_reader::SymBufReader;

/// CCBLOCK structure (MDF 4.2 spec, Table 30)
#[derive(Debug, Clone)]
#[binrw]
#[br(little)]
#[repr(C)]
pub struct Cc4Block {
    // cc_id: [u8; 4],  // ##CC
    // reserved: [u8; 4],  // reserved
    // cc_len: u64,      // Length of block in bytes
    /// # of links
    cc_links: u64,
    /// Link to TXBLOCK with name (identifier) of conversion (can be NIL). Name must be according to naming rules stated in 4.4.2 Naming Rules.
    pub cc_tx_name: i64,
    /// Link to TXBLOCK/MDBLOCK with physical unit of signal data (after conversion). (can be NIL) Unit only applies if no unit defined in CNBLOCK. Otherwise the unit of the channel overwrites the conversion unit.
    cc_md_unit: i64,
    // An MDBLOCK can be used to additionally reference the A-HDO unit definition. Note: for channels with cn_sync_type > 0, the unit is already defined, thus a reference to an A-HDO definition should be omitted to avoid redundancy.
    /// Link to TXBLOCK/MDBLOCK with comment of conversion and additional information. (can be NIL)
    pub cc_md_comment: i64,
    /// Link to CCBLOCK for inverse formula (can be NIL, must be NIL for CCBLOCK of the inverse formula (no cyclic reference allowed).
    cc_cc_inverse: i64,
    #[br(if(cc_links > 4), little, count = cc_links - 4)]
    /// List of additional links to TXBLOCKs with strings or to CCBLOCKs with partial conversion rules. Length of list is given by cc_ref_count. The list can be empty. Details are explained in formula-specific block supplement.
    pub cc_ref: Vec<i64>,

    // Data Members
    /// Conversion type (formula identifier) (see CC_T_xxx)
    pub cc_type: u8,
    /// Precision for display of floating point values. 0xFF means unrestricted precision (infinite) Any other value specifies the number of decimal places to use for display of floating point values. Note: only valid if "precision valid" flag (bit 0) is set and if cn_precision of the parent CNBLOCK is invalid, otherwise cn_precision must be used.
    cc_precision: u8,
    /// Flags  (see CC_F_xxx)
    cc_flags: u16,
    /// Length M of cc_ref list with additional links. See formula-specific block supplement for meaning of the links.
    cc_ref_count: u16,
    /// Length N of cc_val list with additional parameters. See formula-specific block supplement for meaning of the parameters.
    cc_val_count: u16,
    /// Minimum physical signal value that occurred for this signal. Only valid if "physical value range valid" flag (bit 1) is set.
    cc_phy_range_min: f64,
    /// Maximum physical signal value that occurred for this signal. Only valid if "physical value range valid" flag (bit 1) is set.
    cc_phy_range_max: f64,
    #[br(args(cc_val_count, cc_type))]
    pub cc_val: CcVal,
}

/// Cc Values can be either a float or Uint64
#[derive(Debug, Clone)]
#[binrw]
#[br(little, import(count: u16, cc_type: u8))]
#[repr(C)]
pub enum CcVal {
    #[br(pre_assert(cc_type < 11))]
    Real(#[br(count = count)] Vec<f64>),

    #[br(pre_assert(cc_type == 11))]
    Uint(#[br(count = count)] Vec<u64>),
}

impl Cc4Block {
    /// Returns a string representation of the conversion type (cc_type)
    pub fn get_cc_type_str(&self) -> &'static str {
        match self.cc_type {
            0 => "Identity",
            1 => "Linear",
            2 => "Rational",
            3 => "Algebraic",
            4 => "ValueToValueInterpolation",
            5 => "ValueToValue",
            6 => "ValueRangeToValue",
            7 => "ValueToText",
            8 => "ValueRangeToText",
            9 => "TextToValue",
            10 => "TextToText",
            11 => "BitfieldToText",
            _ => "Unknown",
        }
    }
}

impl Display for Cc4Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CC: type={} ({}) refs={} vals={}",
            self.get_cc_type_str(),
            self.cc_type,
            self.cc_ref_count,
            self.cc_val_count
        )
    }
}

/// reads and parses CC block and its linked metadata
pub(super) fn read_cc(
    rdr: &mut SymBufReader<&File>,
    target: &i64,
    mut position: i64,
    mut block: Cursor<Vec<u8>>,
    sharable: &mut SharableBlocks,
) -> Result<i64> {
    let cc_block: Cc4Block = block
        .read_le()
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
