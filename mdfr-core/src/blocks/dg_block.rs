//! Data Group block (DGBLOCK) — spec section 6.4, Table 19.
//!
//! A DGBLOCK groups channel groups that share one data block.
use binrw::binrw;
use core::fmt::{self, Display};

/// DGBLOCK structure (MDF 4.2 spec, Table 19)
#[derive(Debug, Copy, Clone)]
#[binrw]
#[br(little)]
#[bw(little)]
#[repr(C)]
pub struct Dg4Block {
    /// ##DG
    pub dg_id: [u8; 4],
    /// reserved
    pub reserved: [u8; 4],
    /// Length of block in bytes
    pub dg_len: u64,
    /// # of links
    pub dg_links: u64,
    /// Pointer to next data group block (DGBLOCK) (can be NIL)
    pub dg_dg_next: i64,
    /// Pointer to first channel group block (CGBLOCK) (can be NIL)
    pub dg_cg_first: i64,
    // Pointer to data block (DTBLOCK or DZBLOCK for this block type) or data list block (DLBLOCK of data blocks or its HLBLOCK)  (can be NIL)
    pub dg_data: i64,
    /// comment
    pub dg_md_comment: i64,
    /// number of bytes used for record IDs. 0 no recordID
    pub dg_rec_id_size: u8,
    // reserved
    pub reserved_2: [u8; 7],
}

impl Default for Dg4Block {
    fn default() -> Self {
        Dg4Block {
            dg_id: [35, 35, 68, 71], // ##DG
            reserved: [0; 4],
            dg_len: 64,
            dg_links: 4,
            dg_dg_next: 0,
            dg_cg_first: 0,
            dg_data: 0,
            dg_md_comment: 0,
            dg_rec_id_size: 0,
            reserved_2: [0; 7],
        }
    }
}

impl Display for Dg4Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DG: rec_id_size={}", self.dg_rec_id_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use binrw::{BinReaderExt, BinWrite, io::Cursor};

    #[test]
    fn roundtrip_is_64_bytes() {
        let dg = Dg4Block {
            dg_cg_first: 0x100,
            dg_data: 0x200,
            ..Default::default()
        };
        let mut c = Cursor::new(Vec::new());
        dg.write_le(&mut c).unwrap();
        let buf = c.into_inner();
        assert_eq!(buf.len() as u64, dg.dg_len);
        let back: Dg4Block = Cursor::new(buf).read_le().unwrap();
        assert_eq!(back.dg_cg_first, 0x100);
        assert_eq!(back.dg_data, 0x200);
    }
}
