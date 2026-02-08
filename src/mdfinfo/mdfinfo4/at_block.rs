//! Attachment block (ATBLOCK) for MDF4
use anyhow::{Context, Result};
use binrw::{BinReaderExt, binrw};
use log::warn;
use md5::{Digest, Md5};
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::fs::File;
use std::io::{Cursor, Read};

use super::block_header::{read_meta_data, SharableBlocks};
use super::data_block::decompress_data;
use super::metadata::BlockType;
use crate::mdfinfo::sym_buf_reader::SymBufReader;

/// At4 Attachment block struct
#[derive(Debug, Copy, Clone)]
#[binrw]
#[br(little)]
#[repr(C)]
pub struct At4Block {
    /// ##DG
    at_id: [u8; 4],
    /// reserved
    reserved: [u8; 4],
    /// Length of block in bytes
    pub at_len: u64,
    /// # of links
    at_links: u64,
    /// Link to next ATBLOCK (linked list) (can be NIL)
    pub at_at_next: i64,
    /// Link to TXBLOCK with the path and file name of the embedded or referenced file (can only be NIL if data is embedded). The path of the file can be relative or absolute. If relative, it is relative to the directory of the MDF file. If no path is given, the file must be in the same directory as the MDF file.
    pub at_tx_filename: i64,
    /// Link to TXBLOCK with MIME content-type text that gives information about the attached data. Can be NIL if the content-type is unknown, but should be specified whenever possible. The MIME content-type string must be written in lowercase.
    pub at_tx_mimetype: i64,
    /// Link to MDBLOCK with comment and additional information about the attachment (can be NIL).
    pub at_md_comment: i64,
    /// Flags The value contains the following bit flags (see AT_FL_xxx):
    pub at_flags: u16,
    /// Creator index, i.e. zero-based index of FHBLOCK in global list of FHBLOCKs that specifies which application has created this attachment, or changed it most recently.
    pub at_creator_index: u16,
    /// Compression algorithm used for embedded data
    pub at_zip_type: u8,
    /// File path format
    pub at_path_syntax: u8,
    /// Reserved
    at_reserved: [u8; 2],
    /// 128-bit value for MD5 check sum (of the uncompressed data if data is embedded and compressed). Only valid if "MD5 check sum valid" flag (bit 2) is set.
    pub at_md5_checksum: [u8; 16],
    /// Original data size in Bytes, i.e. either for external file or for uncompressed data.
    pub at_original_size: u64,
    /// Embedded data size N, i.e. number of Bytes for binary embedded data following this element. Must be 0 if external file is referenced.
    pub at_embedded_size: u64,
    // followed by embedded data depending of flag
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
    #[allow(dead_code)]
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
        let storage = if self.is_embedded() { "embedded" } else { "external" };
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

/// At4 (Attachment) block struct parser
fn parser_at4_block(
    rdr: &mut SymBufReader<&File>,
    target: i64,
    mut position: i64,
) -> Result<(At4Block, Option<Vec<u8>>, i64)> {
    rdr.seek_relative(target - position)
        .context("Could not reach At4 Block position")?;
    let mut buf = [0u8; 96];
    rdr.read_exact(&mut buf)
        .context("Could not read At4 Block buffer")?;
    let mut block = Cursor::new(buf);
    let block: At4Block = block
        .read_le()
        .context("Could not parse At4 Block buffer into At4Block struct")?;
    position = target + 96;

    // reads embedded if exists
    let data: Option<Vec<u8>> = if (block.at_flags & 0b1) > 0 {
        let mut embedded_data = vec![0u8; block.at_embedded_size as usize];
        rdr.read_exact(&mut embedded_data)
            .context("Could not parse At4Block embedded attachement")?;

        let zip_type = block.at_zip_type;
        if (block.at_flags & 0b10) > 0 {
            embedded_data = decompress_data(zip_type, 0, embedded_data, block.at_original_size)?;
        }

        // MD5 Checksum verification
        if (block.at_flags & 0b100) > 0 {
            let mut hasher = Md5::new();
            hasher.update(&embedded_data);
            let result = hasher.finalize();
            if result.as_slice() != block.at_md5_checksum {
                warn!(
                    "MD5 checksum mismatch for attachment: expected {:?}, got {:?}",
                    block.at_md5_checksum, result
                );
            }
        }

        position += block.at_embedded_size as i64;
        Some(embedded_data)
    } else {
        None
    };
    Ok((block, data, position))
}

pub type At = HashMap<i64, (At4Block, Option<Vec<u8>>)>;

/// parses Attachment blocks along with its linked comments, returns a hashmap of At4 block and attached data in a vect
pub fn parse_at4(
    rdr: &mut SymBufReader<&File>,
    sharable: &mut SharableBlocks,
    target: i64,
    mut position: i64,
) -> Result<(At, i64)> {
    let mut at: At = HashMap::new();
    if target > 0 {
        let (block, data, pos) = parser_at4_block(rdr, target, position)?;
        position = pos;
        // Reads MD
        position = read_meta_data(rdr, sharable, block.at_md_comment, position, BlockType::AT)?;
        // reads TX file_name
        position = read_meta_data(rdr, sharable, block.at_tx_filename, position, BlockType::AT)?;
        // Reads tx mime type
        position = read_meta_data(rdr, sharable, block.at_tx_mimetype, position, BlockType::AT)?;
        let mut next_pointer = block.at_at_next;
        at.insert(target, (block, data));

        while next_pointer > 0 {
            let block_start = next_pointer;
            let (block, data, pos) = parser_at4_block(rdr, next_pointer, position)?;
            position = pos;
            // Reads MD
            position = read_meta_data(rdr, sharable, block.at_md_comment, position, BlockType::AT)?;
            // reads TX file_name
            position =
                read_meta_data(rdr, sharable, block.at_tx_filename, position, BlockType::AT)?;
            // Reads tx mime type
            position =
                read_meta_data(rdr, sharable, block.at_tx_mimetype, position, BlockType::AT)?;
            next_pointer = block.at_at_next;
            at.insert(block_start, (block, data));
        }
    }
    Ok((at, position))
}
