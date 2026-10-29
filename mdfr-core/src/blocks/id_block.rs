//! File Identification block (IDBLOCK) — spec section 5.1, Table 1.
use binrw::binrw;

/// "UnFinMF " identifier bytes for unfinalized MDF files.
pub const UNFINALIZED_ID: [u8; 8] = *b"UnFinMF ";
/// "MDF     " identifier bytes for finalized MDF files.
pub const FINALIZED_ID: [u8; 8] = *b"MDF     ";

/// Bit 0: Update of cycle counters for CG-/CABLOCK required
pub const UNFIN_CG_CYCLE_COUNTERS: u16 = 1 << 0;
/// Bit 1: Update of cycle counters for SRBLOCKs required
pub const UNFIN_SR_CYCLE_COUNTERS: u16 = 1 << 1;
/// Bit 2: Update of length for last DTBLOCK required
pub const UNFIN_LAST_DT_LENGTH: u16 = 1 << 2;
/// Bit 3: Update of length for last RDBLOCK required
pub const UNFIN_LAST_RD_LENGTH: u16 = 1 << 3;
/// Bit 4: Update of last DLBLOCK in each chained list required
pub const UNFIN_LAST_DL_BLOCK: u16 = 1 << 4;
/// Bit 5: Update of cg_data_bytes and cg_inval_bytes in VLSD CGBLOCK required
pub const UNFIN_VLSD_CG_BYTES: u16 = 1 << 5;
/// Bit 6: Update of offset values for VLSD channel required
pub const UNFIN_VLSD_OFFSET: u16 = 1 << 6;
/// Bit 7: Update of cg_data_bytes and cg_inval_bytes in VLSC CGBLOCK required
pub const UNFIN_VLSC_CG_BYTES: u16 = 1 << 7;
/// Bit 8: Update of offset values for VLSC channel required
pub const UNFIN_VLSC_OFFSET: u16 = 1 << 8;

/// Common Id block structure for both versions 3 and 4 (64 bytes at offset 0).
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
#[binrw]
#[br(little)]
#[bw(little)]
#[repr(C)]
pub struct IdBlock {
    /// "MDF     " or "UnFinMF "
    pub id_file_id: [u8; 8],
    /// version in char, e.g. "4.30    "
    pub id_vers: [u8; 8],
    /// logger id, e.g. "mdfr    "
    pub id_prog: [u8; 8],
    /// 0 Little endian, >= 1 Big endian, only valid for 3.x
    pub id_default_byteorder: u16,
    /// default floating point number. 0: IEEE754, 1: G_Float, 2: D_Float, only valid for 3.x
    pub id_floatingpointformat: u16,
    /// version number, valid for both 3.x and 4.x (e.g. 430)
    pub id_ver: u16,
    /// code page (only for version 3.3)
    pub id_codepage: u16,
    /// check
    pub id_check: [u8; 2],
    /// reserved fill
    pub id_fill: [u8; 26],
    /// standard unfinalization flags (4.x)
    pub id_unfin_flags: u16,
    /// custom unfinalization flags (4.x)
    pub id_custom_unfin_flags: u16,
}

impl Default for IdBlock {
    fn default() -> Self {
        IdBlock {
            id_file_id: FINALIZED_ID,
            id_vers: *b"4.30    ",
            id_prog: *b"mdfr    ",
            id_ver: 430,
            id_default_byteorder: 0,
            id_floatingpointformat: 0,
            id_codepage: 0,
            id_check: [0, 0],
            id_fill: [0u8; 26],
            id_unfin_flags: 0,
            id_custom_unfin_flags: 0,
        }
    }
}

impl IdBlock {
    /// Creates an unfinalized MDF 4.30 ID block with the given standard flags.
    pub fn unfinalized(unfin_flags: u16) -> Self {
        Self {
            id_file_id: UNFINALIZED_ID,
            id_unfin_flags: unfin_flags,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use binrw::{BinReaderExt, BinWrite, io::Cursor};

    #[test]
    fn id_is_64_bytes() {
        let id = IdBlock::unfinalized(UNFIN_CG_CYCLE_COUNTERS | UNFIN_LAST_DT_LENGTH);
        let mut c = Cursor::new(Vec::new());
        id.write_le(&mut c).unwrap();
        let buf = c.into_inner();
        assert_eq!(buf.len(), 64);
        let back: IdBlock = Cursor::new(buf).read_le().unwrap();
        assert_eq!(back, id);
    }
}
