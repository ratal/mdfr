//! Block header structures — MDF 4 spec section 5.2, Table 2.
//!
//! Every MDF4 block starts with a header: 4-byte ID (`##XX`), 4 reserved bytes,
//! 8-byte length and 8-byte link count.
use binrw::binrw;
use core::fmt::{self, Display};

use crate::error::{Error, Result};
use alloc::format;

/// MDF4 block header (spec Table 2) — common 24-byte prefix for all blocks
#[derive(Debug, Copy, Clone)]
#[binrw]
#[br(little)]
#[bw(little)]
#[repr(C)]
pub struct Blockheader4 {
    /// '##XX'
    pub hdr_id: [u8; 4],
    /// reserved, must be 0
    pub hdr_gap: [u8; 4],
    /// Length of block in bytes
    pub hdr_len: u64,
    /// # of links
    pub hdr_links: u64,
}

impl Default for Blockheader4 {
    fn default() -> Self {
        Blockheader4 {
            hdr_id: [35, 35, 84, 88], // ##TX
            hdr_gap: [0x00, 0x00, 0x00, 0x00],
            hdr_len: 24,
            hdr_links: 0,
        }
    }
}

impl Display for Blockheader4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let id = core::str::from_utf8(&self.hdr_id).unwrap_or("????");
        write!(
            f,
            "Block: {} len={} links={}",
            id, self.hdr_len, self.hdr_links
        )
    }
}

impl Blockheader4 {
    /// Validates that hdr_len is plausible before allocating the block body buffer.
    /// Guards against corrupted length fields causing OOM or arithmetic overflow.
    pub fn validate_len(&self, file_size: u64) -> Result<()> {
        if self.hdr_len < 24 {
            return Err(Error::InvalidBlock(format!(
                "block {:?} has hdr_len={} < 24 (minimum header size)",
                self.hdr_id, self.hdr_len
            )));
        }
        if self.hdr_len > file_size {
            return Err(Error::InvalidBlock(format!(
                "block {:?} has hdr_len={} > file size {} — corrupted length",
                self.hdr_id, self.hdr_len, file_size
            )));
        }
        Ok(())
    }
}

/// MDF4 - common block Header without the number of links
#[derive(Debug, Copy, Clone)]
#[binrw]
#[br(little)]
#[bw(little)]
#[repr(C)]
pub struct Blockheader4Short {
    /// '##XX'
    pub hdr_id: [u8; 4],
    /// reserved, must be 0
    pub hdr_gap: [u8; 4],
    /// Length of block in bytes
    pub hdr_len: u64,
}

impl Default for Blockheader4Short {
    fn default() -> Self {
        Blockheader4Short {
            hdr_id: [35, 35, 67, 78], // ##CN
            hdr_gap: [0u8; 4],
            hdr_len: 160,
        }
    }
}

impl Display for Blockheader4Short {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let id = core::str::from_utf8(&self.hdr_id).unwrap_or("????");
        write!(f, "Block: {} len={}", id, self.hdr_len)
    }
}

impl Blockheader4Short {
    /// Validates that hdr_len is plausible before allocating the block body buffer.
    pub fn validate_len(&self, file_size: u64) -> Result<()> {
        if self.hdr_len < 16 {
            return Err(Error::InvalidBlock(format!(
                "block {:?} has hdr_len={} < 16 (minimum short header size)",
                self.hdr_id, self.hdr_len
            )));
        }
        if self.hdr_len > file_size {
            return Err(Error::InvalidBlock(format!(
                "block {:?} has hdr_len={} > file size {} — corrupted length",
                self.hdr_id, self.hdr_len, file_size
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use binrw::{BinReaderExt, BinWrite, io::Cursor};

    #[test]
    fn header_roundtrip_is_24_bytes() {
        let h = Blockheader4 {
            hdr_id: *b"##DT",
            hdr_len: 100,
            hdr_links: 0,
            ..Default::default()
        };
        let mut c = Cursor::new(alloc::vec::Vec::new());
        h.write_le(&mut c).unwrap();
        let buf = c.into_inner();
        assert_eq!(buf.len(), 24);
        let back: Blockheader4 = Cursor::new(buf).read_le().unwrap();
        assert_eq!(back.hdr_id, *b"##DT");
        assert_eq!(back.hdr_len, 100);
    }

    #[test]
    fn validate_len_rejects_corrupt() {
        let h = Blockheader4 {
            hdr_len: 10,
            ..Default::default()
        };
        assert!(h.validate_len(1000).is_err());
        let h = Blockheader4 {
            hdr_len: 5000,
            ..Default::default()
        };
        assert!(h.validate_len(1000).is_err());
    }
}
