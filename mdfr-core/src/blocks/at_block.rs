//! Attachment block (ATBLOCK) for MDF4 — spec section 6.10, Table 47.
//!
//! Stores or references external files (e.g. calibration data, images) attached to the MDF file.
//! Attachments can be embedded or external, with optional MD5 checksum verification.
use binrw::binrw;
use core::fmt::{self, Display};

/// At4 Attachment block struct (96-byte fixed header + links + members).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[binrw]
#[br(little)]
#[bw(little)]
#[repr(C)]
pub struct At4Block {
    /// `##AT`
    pub at_id: [u8; 4],
    /// reserved
    pub reserved: [u8; 4],
    /// Length of block in bytes
    pub at_len: u64,
    /// # of links
    pub at_links: u64,
    /// Link to next ATBLOCK (linked list) (can be NIL)
    pub at_at_next: i64,
    /// Link to TXBLOCK with the path and file name of the embedded or referenced file
    pub at_tx_filename: i64,
    /// Link to TXBLOCK with MIME content-type text
    pub at_tx_mimetype: i64,
    /// Link to MDBLOCK with comment and additional information about the attachment (can be NIL)
    pub at_md_comment: i64,
    /// Flags (see `AT_FL_xxx`)
    pub at_flags: u16,
    /// Creator index (zero-based index of FHBLOCK)
    pub at_creator_index: u16,
    /// Compression algorithm used for embedded data
    pub at_zip_type: u8,
    /// File path format
    pub at_path_syntax: u8,
    /// Reserved
    pub at_reserved: [u8; 2],
    /// 128-bit value for MD5 check sum
    pub at_md5_checksum: [u8; 16],
    /// Original data size in Bytes
    pub at_original_size: u64,
    /// Embedded data size N in Bytes
    pub at_embedded_size: u64,
}

impl Default for At4Block {
    fn default() -> Self {
        At4Block {
            at_id: [35, 35, 65, 84], // ##AT
            reserved: [0; 4],
            at_len: 96,
            at_links: 4,
            at_at_next: 0,
            at_tx_filename: 0,
            at_tx_mimetype: 0,
            at_md_comment: 0,
            at_flags: 0,
            at_creator_index: 0,
            at_zip_type: 0,
            at_path_syntax: 0,
            at_reserved: [0; 2],
            at_md5_checksum: [0; 16],
            at_original_size: 0,
            at_embedded_size: 0,
        }
    }
}

impl At4Block {
    /// Returns true if this attachment has embedded data
    pub fn is_embedded(&self) -> bool {
        (self.at_flags & 0b1) > 0
    }

    /// Returns true if the data is compressed
    pub fn is_compressed(&self) -> bool {
        (self.at_flags & 0b10) > 0
    }

    /// Returns true if the MD5 checksum is valid
    pub fn has_md5_checksum(&self) -> bool {
        (self.at_flags & 0b100) > 0
    }

    /// Returns the compression type as a string description
    pub fn get_compression_str(&self) -> &'static str {
        if !self.is_compressed() {
            return "None";
        }
        match self.at_zip_type {
            0 | 1 => "Deflate",
            2 | 3 => "Zstd",
            4 | 5 => "LZ4",
            _ => "Unknown",
        }
    }
}

impl Display for At4Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let storage = if self.is_embedded() {
            "embedded"
        } else {
            "external"
        };
        let compression = self.get_compression_str();
        write!(
            f,
            "AT: {} size={} original={} compression={} creator_index={}",
            storage,
            self.at_embedded_size,
            self.at_original_size,
            compression,
            self.at_creator_index
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn test_at_is_embedded() {
        let mut at = At4Block::default();
        assert!(!at.is_embedded());
        at.at_flags = 0b1;
        assert!(at.is_embedded());
        at.at_flags = 0b11;
        assert!(at.is_embedded());
    }

    #[test]
    fn test_at_is_compressed() {
        let mut at = At4Block::default();
        assert!(!at.is_compressed());
        at.at_flags = 0b10;
        assert!(at.is_compressed());
    }

    #[test]
    fn test_at_has_md5_checksum() {
        let mut at = At4Block::default();
        assert!(!at.has_md5_checksum());
        at.at_flags = 0b100;
        assert!(at.has_md5_checksum());
    }

    #[test]
    fn test_at_get_compression_str() {
        let mut at = At4Block::default();
        assert_eq!(at.get_compression_str(), "None");
        at.at_flags = 0b10;
        at.at_zip_type = 0;
        assert_eq!(at.get_compression_str(), "Deflate");
        at.at_zip_type = 2;
        assert_eq!(at.get_compression_str(), "Zstd");
        at.at_zip_type = 4;
        assert_eq!(at.get_compression_str(), "LZ4");
        at.at_zip_type = 255;
        assert_eq!(at.get_compression_str(), "Unknown");
    }

    #[test]
    fn test_at_display() {
        let mut at = At4Block {
            at_original_size: 1024,
            at_embedded_size: 512,
            at_flags: 0b1,
            at_creator_index: 1,
            ..Default::default()
        };
        let display = format!("{at}");
        assert!(display.contains("AT:"));
        assert!(display.contains("embedded"));
        assert!(display.contains("size=512"));
        at.at_flags = 0;
        assert!(format!("{at}").contains("external"));
    }
}

