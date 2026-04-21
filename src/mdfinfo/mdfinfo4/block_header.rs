//! Block header structures, metadata parsing, and sharable blocks for MDF4 — spec section 5.2, Table 2
//!
//! Every MDF4 block starts with a header: 4-byte ID ("##XX"), 4 reserved bytes,
//! 8-byte length, and 8-byte link count. This module also manages SharableBlocks
//! (CC, SI, TX, MD blocks) stored in a global hashmap for deduplication.
use anyhow::{Context, Result};
use binrw::{BinReaderExt, binrw};
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::fs::File;
use std::io::{Cursor, Read};
use std::str;

use super::cc_block::Cc4Block;
use super::metadata::{BlockType, HdComment, MdComment, MetaData, MetaDataBlockType};
use super::si_block::Si4Block;
use crate::mdfinfo::sym_buf_reader::SymBufReader;

/// MDF4 block header (spec Table 2) — common 24-byte prefix for all blocks
#[derive(Debug, Copy, Clone)]
#[binrw]
#[br(little)]
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
        let id = str::from_utf8(&self.hdr_id).unwrap_or("????");
        write!(
            f,
            "Block: {} len={} links={}",
            id, self.hdr_len, self.hdr_links
        )
    }
}

/// parse the block header and its fields id, (reserved), length and number of links
#[inline]
pub fn parse_block_header(rdr: &mut SymBufReader<&File>) -> Result<Blockheader4> {
    let mut buf = [0u8; 24];
    rdr.read_exact(&mut buf)
        .context("could not read blockheader4 Id")?;
    let mut block = Cursor::new(buf);
    let header: Blockheader4 = block
        .read_le()
        .context("binread could not parse blockheader4")?;
    Ok(header)
}

/// MDF4 - common block Header without the number of links
#[derive(Debug, Copy, Clone)]
#[binrw]
#[br(little)]
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
        let id = str::from_utf8(&self.hdr_id).unwrap_or("????");
        write!(f, "Block: {} len={}", id, self.hdr_len)
    }
}

pub fn default_short_header(variant: BlockType) -> Blockheader4Short {
    match variant {
        BlockType::CG => Blockheader4Short {
            hdr_id: [35, 35, 67, 71], // ##CG
            hdr_gap: [0u8; 4],
            hdr_len: 104, // 112 with cg_cg_master, 104 without,
        },
        BlockType::CN => Blockheader4Short {
            hdr_id: [35, 35, 67, 78], // ##CN
            hdr_gap: [0u8; 4],
            hdr_len: 160,
        },
        _ => Blockheader4Short {
            hdr_id: [35, 35, 67, 78], // ##CN
            hdr_gap: [0u8; 4],
            hdr_len: 160,
        },
    }
}

/// parse the block header and its fields id, (reserved), length except the number of links
#[inline]
pub(super) fn parse_block_header_short(rdr: &mut SymBufReader<&File>) -> Result<Blockheader4Short> {
    let mut buf = [0u8; 16];
    rdr.read_exact(&mut buf)
        .context("could not read short blockheader4 Id")?;
    let mut block = Cursor::new(buf);
    let header: Blockheader4Short = block
        .read_le()
        .context("could not parse short blockheader4")?;
    Ok(header)
}

/// reads generically a block header and return links and members section part into a Seek buffer for further processing
#[inline]
pub(super) fn parse_block(
    rdr: &mut SymBufReader<&File>,
    target: i64,
    mut position: i64,
) -> Result<(Cursor<Vec<u8>>, Blockheader4, i64)> {
    // Reads block header
    rdr.seek_relative(target - position)
        .context("Could not reach block header position")?; // change buffer position
    let block_header = parse_block_header(rdr).context(" could not read header block")?; // reads header

    // Reads in buffer rest of block
    let mut buf = vec![0u8; (block_header.hdr_len - 24) as usize];
    rdr.read_exact(&mut buf)
        .context("Could not read rest of block after header")?;
    position = target + block_header.hdr_len as i64;
    let block = Cursor::new(buf);
    Ok((block, block_header, position))
}

/// reads generically a block header wihtout the number of links and returns links and members section part into a Seek buffer for further processing
#[inline]
pub(super) fn parse_block_short(
    rdr: &mut SymBufReader<&File>,
    target: i64,
    mut position: i64,
) -> Result<(Cursor<Vec<u8>>, Blockheader4Short, i64)> {
    // Reads block header
    rdr.seek_relative(target - position)
        .context("Could not reach block short header position")?; // change buffer position
    let block_header: Blockheader4Short =
        parse_block_header_short(rdr).context(" could not read short header block")?; // reads header

    // Reads in buffer rest of block
    let mut buf = vec![0u8; (block_header.hdr_len - 16) as usize];
    rdr.read_exact(&mut buf)
        .context("Could not read rest of block after short header")?;
    position = target + block_header.hdr_len as i64;
    let block = Cursor::new(buf);
    Ok((block, block_header, position))
}

/// Parses the MD or TX block
pub(super) fn read_meta_data(
    rdr: &mut SymBufReader<&File>,
    sharable: &mut SharableBlocks,
    target: i64,
    mut position: i64,
    parent_block_type: BlockType,
) -> Result<i64> {
    if target != 0 && !sharable.md_tx.contains_key(&target) {
        let (raw_data, block, pos) =
            parse_block(rdr, target, position).context("could not read metadata block")?;
        position = pos;
        let block_type = match block.hdr_id {
            [35, 35, 77, 68] => MetaDataBlockType::MdBlock,
            [35, 35, 84, 88] => MetaDataBlockType::TX,
            _ => MetaDataBlockType::TX,
        };
        let md = MetaData {
            block,
            raw_data: raw_data.into_inner(),
            block_type,
            parent_block_type,
            md_comment: None,
        };
        sharable.md_tx.insert(target, md);
        Ok(position)
    } else {
        Ok(position)
    }
}

/// sharable blocks (most likely referenced multiple times and shared by several blocks)
/// that are in sharable fields and holds CC, SI, TX and MD blocks
#[derive(Debug, Default, Clone)]
#[repr(C)]
pub struct SharableBlocks {
    pub(crate) md_tx: HashMap<i64, MetaData>,
    pub(crate) cc: HashMap<i64, Cc4Block>,
    pub(crate) si: HashMap<i64, Si4Block>,
}

/// SharableBlocks display implementation to facilitate debugging
impl fmt::Display for SharableBlocks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MD TX comments : \n")?;
        for (_k, c) in &self.md_tx {
            writeln!(f, "{c}")?;
        }
        writeln!(f, "CC : \n")?;
        for (position, cc) in &self.cc {
            writeln!(f, "Position: {position}  Text: {cc:?}")?;
        }
        writeln!(f, "SI : ")?;
        for (position, si) in &self.si {
            writeln!(f, "Position: {position}  Text: {si:?}")?;
        }
        writeln!(f, "finished")
    }
}

impl SharableBlocks {
    /// Returns the text from TX Block or TX tag's text from MD block
    pub fn get_tx(&self, position: i64) -> anyhow::Result<Option<String>> {
        let mut txt: Option<String> = None;
        if let Some(md) = self.md_tx.get(&position) {
            txt = md.get_tx()?;
        };
        Ok(txt)
    }
    /// Creates a new SharableBlocks of type TX (not MD)
    pub fn create_tx(&mut self, position: i64, text: String) {
        let md = self
            .md_tx
            .entry(position)
            .or_insert_with(|| MetaData::new(MetaDataBlockType::TX, BlockType::CN));
        md.set_data_buffer(text.as_bytes());
    }
    /// Returns typed metadata comment, parsing lazily if needed
    pub fn get_md_comment(&mut self, position: i64) -> Option<&MdComment> {
        if let Some(md) = self.md_tx.get_mut(&position) {
            match md.block_type {
                MetaDataBlockType::MdParsed => {}
                MetaDataBlockType::MdBlock => {
                    let _ = md.parse_xml();
                }
                MetaDataBlockType::TX => return None,
            }
        }
        self.md_tx
            .get(&position)
            .and_then(|md| md.md_comment.as_ref())
    }
    /// Returns HD comment, assumes already parsed
    pub fn get_hd_comments(&self, position: i64) -> Option<&HdComment> {
        if let Some(md) = self.md_tx.get(&position)
            && md.block_type == MetaDataBlockType::MdParsed
            && let Some(MdComment::Hd(hd)) = &md.md_comment
        {
            return Some(hd);
        }
        None
    }
    /// parses the HD Block metadata comments, done right after reading HD block
    pub fn parse_hd_comments(&mut self, position: i64) {
        if let Some(md) = self.md_tx.get_mut(&position) {
            let _ = md.parse_hd_comment();
        };
    }
    /// Create new Shared Block
    pub fn new(n_channels: usize) -> SharableBlocks {
        let md_tx: HashMap<i64, MetaData> = HashMap::with_capacity(n_channels);
        let cc: HashMap<i64, Cc4Block> = HashMap::new();
        let si: HashMap<i64, Si4Block> = HashMap::new();
        SharableBlocks { md_tx, cc, si }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blockheader4_default() {
        let h = Blockheader4::default();
        assert_eq!(h.hdr_id, [35, 35, 84, 88]); // ##TX
        assert_eq!(h.hdr_len, 24);
        assert_eq!(h.hdr_links, 0);
    }

    #[test]
    fn test_blockheader4_display() {
        let h = Blockheader4::default();
        let display = format!("{h}");
        assert!(display.contains("##TX"));
        assert!(display.contains("len=24"));
        assert!(display.contains("links=0"));
    }

    #[test]
    fn test_blockheader4short_default() {
        let h = Blockheader4Short::default();
        assert_eq!(h.hdr_id, [35, 35, 67, 78]); // ##CN
        assert_eq!(h.hdr_len, 160);
    }

    #[test]
    fn test_blockheader4short_display() {
        let h = Blockheader4Short::default();
        let display = format!("{h}");
        assert!(display.contains("##CN"));
        assert!(display.contains("len=160"));
    }

    #[test]
    fn test_default_short_header() {
        let cg = default_short_header(BlockType::CG);
        assert_eq!(cg.hdr_id, [35, 35, 67, 71]); // ##CG
        assert_eq!(cg.hdr_len, 104);

        let cn = default_short_header(BlockType::CN);
        assert_eq!(cn.hdr_id, [35, 35, 67, 78]); // ##CN
        assert_eq!(cn.hdr_len, 160);

        // Wildcard falls through to CN defaults
        let other = default_short_header(BlockType::HD);
        assert_eq!(other.hdr_id, [35, 35, 67, 78]);
        assert_eq!(other.hdr_len, 160);
    }
}
