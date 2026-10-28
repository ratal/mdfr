//! Attachment block (ATBLOCK) parsing for MDF4 — spec section 6.10, Table 47
//!
//! Stores or references external files (e.g. calibration data, images) attached to the MDF file.
//! Attachments can be embedded or external, with optional MD5 checksum verification.
use anyhow::{Context, Result, bail};
use log::warn;
use md5::{Digest, Md5};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read};

use super::block_header_impl::{SharableBlocks, read_meta_data};
use super::data_impl::decompress_data;
use super::metadata::BlockType;
use crate::mdfinfo::block_chain::BlockChain;
use crate::mdfinfo::sym_buf_reader::SymBufReader;

pub use mdfr_core::blocks::at_block::At4Block;

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
    let block: At4Block = binrw::BinReaderExt::read_le(&mut block)
        .context("Could not parse At4 Block buffer into At4Block struct")?;
    position = target + 96;

    // reads embedded if exists
    let data: Option<Vec<u8>> = if (block.at_flags & 0b1) > 0 {
        let file_size = rdr.file_size().unwrap_or(u64::MAX);
        let available = file_size.saturating_sub(position as u64);
        if block.at_embedded_size > available {
            bail!(
                "attachment at 0x{target:x} declares {} embedded bytes but only {available} are left in the file",
                block.at_embedded_size
            );
        }
        let mut embedded_data = vec![0u8; block.at_embedded_size as usize];
        rdr.read_exact(&mut embedded_data)
            .context("Could not parse At4Block embedded attachement")?;

        let zip_type = block.at_zip_type;
        if (block.at_flags & 0b10) > 0 {
            let compressed = embedded_data.clone();
            embedded_data = decompress_data(zip_type, 0, embedded_data, block.at_original_size)
                .unwrap_or_else(|_| {
                    warn!("Failed to decompress attachment data, storing raw data");
                    compressed
                });
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

        let mut chain = BlockChain::new("AT");
        chain.visit(target);
        while next_pointer > 0 {
            if !chain.visit(next_pointer) {
                break;
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_at_is_embedded() {
        let mut at = At4Block::default();
        assert!(!at.is_embedded());

        at.at_flags = 0b1;
        assert!(at.is_embedded());

        at.at_flags = 0b11; // both bits set
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
        // Not compressed → None
        assert_eq!(at.get_compression_str(), "None");

        // Compressed with various zip types
        at.at_flags = 0b10;
        at.at_zip_type = 0;
        assert_eq!(at.get_compression_str(), "Deflate");
        at.at_zip_type = 1;
        assert_eq!(at.get_compression_str(), "Deflate");
        at.at_zip_type = 2;
        assert_eq!(at.get_compression_str(), "Zstd");
        at.at_zip_type = 3;
        assert_eq!(at.get_compression_str(), "Zstd");
        at.at_zip_type = 4;
        assert_eq!(at.get_compression_str(), "LZ4");
        at.at_zip_type = 5;
        assert_eq!(at.get_compression_str(), "LZ4");
        at.at_zip_type = 255;
        assert_eq!(at.get_compression_str(), "Unknown");
    }

    #[test]
    fn test_at_display() {
        let mut at = At4Block {
            at_original_size: 1024,
            at_embedded_size: 512,
            at_flags: 0b1, // embedded
            at_creator_index: 1,
            ..Default::default()
        };

        let display = format!("{at}");
        assert!(display.contains("AT:"));
        assert!(display.contains("embedded"));
        assert!(display.contains("size=512"));
        assert!(display.contains("original=1024"));
        assert!(display.contains("compression=None"));
        assert!(display.contains("creator_index=1"));

        // External
        at.at_flags = 0;
        let display = format!("{at}");
        assert!(display.contains("external"));
    }
}
