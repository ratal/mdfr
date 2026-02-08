//! MetaData struct and related types for MDF4 TX/MD blocks
use anyhow::{Context, Result};
use binrw::BinWriterExt;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::io::{Seek, Write};
use std::str;

use super::block_header::Blockheader4;

/// metadata are either stored in TX (text) or MD (xml) blocks for mdf version 4
#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(C)]
#[derive(Default)]
pub enum MetaDataBlockType {
    MdBlock,
    MdParsed,
    #[default]
    TX,
}

/// Blocks types that could link to MDBlock
#[derive(Debug, Clone)]
#[repr(C)]
#[derive(Default)]
pub enum BlockType {
    HD,
    FH,
    AT,
    EV,
    DG,
    CG,
    #[default]
    CN,
    CC,
    SI,
    CH,
}

/// struct linking MD or TX block with
#[derive(Debug, Default, Clone)]
#[repr(C)]
pub struct MetaData {
    /// Header of the block
    pub block: Blockheader4,
    /// Raw bytes for the block's data
    pub raw_data: Vec<u8>,
    /// Block type, TX, MD or MD not yet parsed
    pub block_type: MetaDataBlockType,
    /// Metadata after parsing
    pub comments: HashMap<String, String>,
    /// Parent block type
    pub parent_block_type: BlockType,
}

impl Display for MetaData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let type_str = match self.block_type {
            MetaDataBlockType::MdBlock => "MD (unparsed)",
            MetaDataBlockType::MdParsed => "MD (parsed)",
            MetaDataBlockType::TX => "TX",
        };
        write!(
            f,
            "MetaData: type={} {} comments raw_bytes={}",
            type_str,
            self.comments.len(),
            self.raw_data.len()
        )
    }
}

impl MetaData {
    /// Returns a new MetaData struct
    pub fn new(block_type: MetaDataBlockType, parent_block_type: BlockType) -> Self {
        let header = match block_type {
            MetaDataBlockType::MdBlock => Blockheader4 {
                hdr_id: [35, 35, 77, 68], // '##MD'
                hdr_gap: [0u8; 4],
                hdr_len: 24,
                hdr_links: 0,
            },
            MetaDataBlockType::TX | MetaDataBlockType::MdParsed => Blockheader4 {
                hdr_id: [35, 35, 84, 88], // '##TX'
                hdr_gap: [0u8; 4],
                hdr_len: 24,
                hdr_links: 0,
            },
        };
        MetaData {
            block: header,
            raw_data: Vec::new(),
            block_type,
            comments: HashMap::new(),
            parent_block_type,
        }
    }
    /// Converts the metadata handling the parent block type's specificities
    pub fn parse_xml(&mut self) -> Result<()> {
        if self.block_type == MetaDataBlockType::MdBlock {
            match self.parent_block_type {
                BlockType::HD => self.parse_hd_xml()?,
                BlockType::FH => self.parse_fh_xml()?,
                _ => self.parse_generic_xml()?,
            };
        }
        Ok(())
    }
    /// Returns the text from TX Block or TX's tag text from MD Block
    pub fn get_tx(&self) -> Result<Option<String>, anyhow::Error> {
        match self.block_type {
            MetaDataBlockType::MdParsed => Ok(self.comments.get("TX").cloned()),
            MetaDataBlockType::MdBlock => {
                // extract TX tag from xml
                let comment: String = self
                    .get_data_string()
                    .context("failed getting data string to extract TX tag")?
                    .trim_end_matches(['\n', '\r', ' '])
                    .into(); // removes ending spaces
                match roxmltree::Document::parse(&comment) {
                    Ok(md) => {
                        let mut tx: Option<String> = None;
                        for node in md.root().descendants() {
                            let text = match node.text() {
                                Some(text) => text.to_string(),
                                None => String::new(),
                            };
                            if node.is_element()
                                && !text.is_empty()
                                && node.tag_name().name() == r"TX"
                            {
                                tx = Some(text);
                                break;
                            }
                        }
                        Ok(tx)
                    }
                    Err(e) => {
                        log::warn!("Error parsing comment : \n{comment}\n{e}");
                        Ok(None)
                    }
                }
            }
            MetaDataBlockType::TX => {
                let comment = str::from_utf8(&self.raw_data).with_context(|| {
                    format!("Invalid UTF-8 sequence in metadata: {:?}", self.raw_data)
                })?;
                let c: String = comment.trim_end_matches(char::from(0)).into();
                Ok(Some(c))
            }
        }
    }
    /// Returns the bytes of the text from TX Block or TX's tag text from MD Block
    pub fn get_tx_bytes(&self) -> Option<&[u8]> {
        match self.block_type {
            MetaDataBlockType::MdParsed => self.comments.get("TX").map(|s| s.as_bytes()),
            _ => Some(&self.raw_data),
        }
    }
    /// Decode string from raw_data field
    pub fn get_data_string(&self) -> Result<String> {
        match self.block_type {
            MetaDataBlockType::MdParsed => Ok(String::new()),
            _ => {
                let comment = str::from_utf8(&self.raw_data).with_context(|| {
                    format!("Invalid UTF-8 sequence in metadata: {:?}", self.raw_data)
                })?;
                let comment: String = comment.trim_end_matches(char::from(0)).into();
                Ok(comment)
            }
        }
    }
    /// allocate bytes to raw_data field, adjusting header length
    pub fn set_data_buffer(&mut self, data: &[u8]) {
        self.raw_data = [data, vec![0u8; 8 - data.len() % 8].as_slice()].concat();
        self.block.hdr_len = self.raw_data.len() as u64 + 24;
    }
    /// parses the xml bytes specifically for HD block contexted schema
    pub fn parse_hd_xml(&mut self) -> Result<()> {
        let mut comments: HashMap<String, String> = HashMap::new();
        // MD Block from HD Block, reading xml
        let comment: String = self
            .get_data_string()?
            .trim_end_matches(['\n', '\r', ' '])
            .into(); // removes ending spaces
        match roxmltree::Document::parse(&comment) {
            Ok(md) => {
                for node in md.root().descendants().filter(|p| p.has_tag_name("e")) {
                    if let (Some(value), Some(text)) = (node.attribute("name"), node.text()) {
                        comments.insert(value.to_string(), text.to_string());
                    }
                }
            }
            Err(e) => {
                log::warn!("Could not parse HD MD comment : \n{comment}\n{e}");
            }
        };
        self.comments = comments;
        self.block_type = MetaDataBlockType::MdParsed;
        self.raw_data = vec![]; // empty the data from block as already parsed
        Ok(())
    }
    /// Creates File History MetaData
    pub fn create_fh(&mut self) {
        let user_name = whoami::username().unwrap_or_else(|_| "unknown".to_string());
        let comments = format!(
            "<FHcomment>
<TX>created</TX>
<tool_id>mdfr</tool_id>
<tool_vendor>ratalco</tool_vendor>
<tool_version>0.1</tool_version>
<user_name>{user_name}</user_name>
</FHcomment>"
        );
        let raw_comments = format!(
            "{:\0<width$}",
            comments,
            width = (comments.len() / 8 + 1) * 8
        );
        let fh_comments = raw_comments.as_bytes();
        self.block.hdr_len = fh_comments.len() as u64 + 24;
        self.raw_data = fh_comments.to_vec();
    }
    /// parses the xml bytes specifically for File History block contexted schema
    fn parse_fh_xml(&mut self) -> Result<()> {
        let mut comments: HashMap<String, String> = HashMap::new();
        // MD Block from FH Block, reading xml
        let comment: String = self
            .get_data_string()?
            .trim_end_matches(['\n', '\r', ' '])
            .into(); // removes ending spaces
        match roxmltree::Document::parse(&comment) {
            Ok(md) => {
                for node in md.root().descendants() {
                    let text = match node.text() {
                        Some(text) => text.to_string(),
                        None => String::new(),
                    };
                    comments.insert(node.tag_name().name().to_string(), text);
                }
            }
            Err(e) => {
                log::warn!("Could not parse FH comment : \n{comment}\n{e}");
            }
        };
        self.comments = comments;
        self.block_type = MetaDataBlockType::MdParsed;
        self.raw_data = vec![]; // empty the data from block as already parsed
        Ok(())
    }
    /// Generic xml parser without schema consideration
    fn parse_generic_xml(&mut self) -> Result<()> {
        let mut comments: HashMap<String, String> = HashMap::new();
        let comment: String = self
            .get_data_string()?
            .trim_end_matches(['\n', '\r', ' '])
            .into(); // removes ending spaces
        match roxmltree::Document::parse(&comment) {
            Ok(md) => {
                for node in md.root().descendants() {
                    let text = match node.text() {
                        Some(text) => text.to_string(),
                        None => String::new(),
                    };
                    if node.is_element()
                        && !text.is_empty()
                        && !node.tag_name().name().to_string().is_empty()
                    {
                        comments.insert(node.tag_name().name().to_string(), text);
                    }
                }
            }
            Err(e) => {
                log::warn!("Error parsing comment : \n{comment}\n{e}");
            }
        };
        self.comments = comments;
        self.block_type = MetaDataBlockType::MdParsed;
        self.raw_data = vec![]; // empty the data from block as already parsed
        Ok(())
    }
    /// Writes the metadata to file
    pub fn write<W>(&self, writer: &mut W) -> Result<()>
    where
        W: Write + Seek,
    {
        writer
            .write_le(&self.block)
            .context("Could not write comment block header")?;
        writer
            .write_all(&self.raw_data)
            .context("Could not write comment block data")?;
        Ok(())
    }
}
