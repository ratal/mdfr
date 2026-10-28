//! Channel Conversion block (CCBLOCK) for MDF4 — spec section 6.7, Tables 30-39.
//!
//! Defines the conversion rule to transform raw channel values to physical values.
//! Supports 12 conversion types (Table 31): identity, linear, rational, polynomial,
//! tabular (value-to-value, value-to-text), algebraic formula, and more.
use alloc::vec::Vec;
use binrw::binrw;
use core::fmt::{self, Display};

/// CCBLOCK structure (MDF 4.2 spec, Table 30)
#[derive(Debug, Clone, PartialEq)]
#[binrw]
#[br(little)]
#[bw(little)]
#[derive(Default)]
#[repr(C)]
pub struct Cc4Block {
    /// # of links
    pub cc_links: u64,
    /// Link to TXBLOCK with name (identifier) of conversion (can be NIL). Name must be according to naming rules stated in 4.4.2 Naming Rules.
    pub cc_tx_name: i64,
    /// Link to TXBLOCK/MDBLOCK with physical unit of signal data (after conversion). (can be NIL) Unit only applies if no unit defined in CNBLOCK. Otherwise the unit of the channel overwrites the conversion unit.
    pub cc_md_unit: i64,
    /// Link to TXBLOCK/MDBLOCK with comment of conversion and additional information. (can be NIL)
    pub cc_md_comment: i64,
    /// Link to CCBLOCK for inverse formula (can be NIL, must be NIL for CCBLOCK of the inverse formula (no cyclic reference allowed).
    pub cc_cc_inverse: i64,
    #[br(if(cc_links > 4), little, count = cc_links - 4)]
    /// List of additional links to TXBLOCKs with strings or to CCBLOCKs with partial conversion rules. Length of list is given by cc_ref_count. The list can be empty. Details are explained in formula-specific block supplement.
    pub cc_ref: Vec<i64>,

    // Data Members
    /// Conversion type (formula identifier) (see CC_T_xxx)
    pub cc_type: u8,
    /// Precision for display of floating point values. 0xFF means unrestricted precision (infinite) Any other value specifies the number of decimal places to use for display of floating point values. Note: only valid if "precision valid" flag (bit 0) is set and if cn_precision of the parent CNBLOCK is invalid, otherwise cn_precision must be used.
    pub cc_precision: u8,
    /// Flags  (see CC_F_xxx)
    pub cc_flags: u16,
    /// Length M of cc_ref list with additional links. See formula-specific block supplement for meaning of the links.
    pub cc_ref_count: u16,
    /// Length N of cc_val list with additional parameters. See formula-specific block supplement for meaning of the parameters.
    pub cc_val_count: u16,
    /// Minimum physical signal value that occurred for this signal. Only valid if "physical value range valid" flag (bit 1) is set.
    pub cc_phy_range_min: f64,
    /// Maximum physical signal value that occurred for this signal. Only valid if "physical value range valid" flag (bit 1) is set.
    pub cc_phy_range_max: f64,
    #[br(args(cc_val_count, cc_type))]
    pub cc_val: CcVal,
}

/// Cc Values can be either a float or Uint64
#[derive(Debug, Clone, PartialEq)]
#[binrw]
#[br(little, import(count: u16, cc_type: u8))]
#[bw(little)]
#[repr(C)]
pub enum CcVal {
    #[br(pre_assert(cc_type < 11))]
    Real(#[br(count = count)] Vec<f64>),

    #[br(pre_assert(cc_type == 11))]
    Uint(#[br(count = count)] Vec<u64>),
}

impl Default for CcVal {
    fn default() -> Self {
        CcVal::Real(Vec::new())
    }
}

impl Cc4Block {
    /// Helper constructor for tests and synthetic conversions
    pub fn new_with_val(cc_type: u8, cc_val: CcVal, cc_ref: Vec<i64>) -> Self {
        let val_count = match &cc_val {
            CcVal::Real(v) => v.len() as u16,
            CcVal::Uint(v) => v.len() as u16,
        };
        Self {
            cc_links: cc_ref.len() as u64 + 4,
            cc_tx_name: 0,
            cc_md_unit: 0,
            cc_md_comment: 0,
            cc_cc_inverse: 0,
            cc_ref_count: cc_ref.len() as u16,
            cc_ref,
            cc_type,
            cc_precision: 0,
            cc_flags: 0,
            cc_val_count: val_count,
            cc_phy_range_min: 0.0,
            cc_phy_range_max: 0.0,
            cc_val,
        }
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{format, vec};
    use binrw::{BinReaderExt, BinWrite, io::Cursor};

    #[test]
    fn roundtrip_linear() {
        let cc = Cc4Block::new_with_val(1, CcVal::Real(vec![0.0, 0.5]), vec![]);
        let mut c = Cursor::new(Vec::new());
        cc.write_le(&mut c).unwrap();
        let buf = c.into_inner();
        // 8 (links) + 4*8 (links) + 24 (members) + 2*8 (vals) = 80 bytes body (+16 short header = 96)
        assert_eq!(buf.len(), 80);
        let back: Cc4Block = Cursor::new(buf).read_le().unwrap();
        assert_eq!(back, cc);
        assert!(format!("{back}").contains("Linear"));
    }
}
