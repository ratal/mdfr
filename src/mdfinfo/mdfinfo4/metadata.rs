//! MetaData struct and related types for MDF4 TX/MD blocks — spec section 4.5
//!
//! TX blocks hold plain text strings. MD blocks hold XML metadata conforming to
//! schema-specific XSD definitions (Tables 13-54). The `MetaData` struct wraps
//! both forms and supports lazy XML parsing via `parse_xml()`.
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

/// Recursive representation of common_properties values per mdf_base.xsd
#[derive(Debug, Clone)]
pub enum PropertyValue {
    /// Simple value from `<e name="key">value</e>`
    Value(String),
    /// Nested properties from `<tree name="...">`
    Tree(HashMap<String, PropertyValue>),
    /// List of property maps from `<list name="..."><li>...</li></list>`
    List(Vec<HashMap<String, PropertyValue>>),
    /// Simple value list from `<elist name="..."><eli>v</eli></elist>`
    EList(Vec<String>),
}

impl Display for PropertyValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PropertyValue::Value(v) => write!(f, "{v}"),
            PropertyValue::Tree(map) => write!(f, "tree({} items)", map.len()),
            PropertyValue::List(items) => write!(f, "list({} items)", items.len()),
            PropertyValue::EList(items) => write!(f, "elist({} items)", items.len()),
        }
    }
}

pub type CommonProperties = HashMap<String, PropertyValue>;

/// Alternative names (from `<names>`, `<path>`, `<bus>` elements)
/// Stores only the default (first/no-lang) value for each field
#[derive(Debug, Clone, Default)]
pub struct MdNames {
    pub name: Option<String>,
    pub display: Option<String>,
    pub vendor: Option<String>,
    pub description: Option<String>,
}

impl Display for MdNames {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        if let Some(name) = &self.name {
            write!(f, "name={name}")?;
            first = false;
        }
        if let Some(display) = &self.display {
            if !first {
                write!(f, " ")?;
            }
            write!(f, "display={display}")?;
            first = false;
        }
        if let Some(vendor) = &self.vendor {
            if !first {
                write!(f, " ")?;
            }
            write!(f, "vendor={vendor}")?;
            first = false;
        }
        if let Some(desc) = &self.description {
            if !first {
                write!(f, " ")?;
            }
            write!(f, "desc={desc}")?;
        }
        Ok(())
    }
}

/// HD block comment per hd_comment.xsd
#[derive(Debug, Clone, Default)]
pub struct HdComment {
    pub tx: Option<String>,
    pub time_source: Option<String>,
    pub constants: HashMap<String, String>,
    pub common_properties: CommonProperties,
}

impl Display for HdComment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(tx) = &self.tx {
            write!(f, "{tx}")?;
        }
        if let Some(ts) = &self.time_source {
            write!(f, " time_source={ts}")?;
        }
        if !self.constants.is_empty() {
            write!(f, " constants={}", self.constants.len())?;
        }
        if !self.common_properties.is_empty() {
            write!(f, " props={}", self.common_properties.len())?;
        }
        Ok(())
    }
}

impl HdComment {
    /// Extracts a string value from `common_properties` by dotted path.
    ///
    /// Handles two equivalent XML encodings:
    /// - flat:   `map["Recorder.SequenceIndex"] = Value("1")`
    /// - nested: `map["Recorder"] = Tree { "SequenceIndex" => Value("1") }`
    fn extract_prop<'a>(props: &'a CommonProperties, path: &str) -> Option<&'a str> {
        let Some((head, tail)) = path.split_once('.') else {
            return match props.get(path)? {
                PropertyValue::Value(v) => Some(v.as_str()),
                _ => None,
            };
        };
        // flat key takes priority ("Recorder.SequenceIndex" stored as-is)
        if let Some(PropertyValue::Value(v)) = props.get(path) {
            return Some(v.as_str());
        }
        // nested tree: "Recorder" -> Tree -> tail
        if let Some(PropertyValue::Tree(sub)) = props.get(head) {
            return Self::extract_prop(sub, tail);
        }
        None
    }

    pub fn recorder_sequence_index(&self) -> Option<u64> {
        Self::extract_prop(&self.common_properties, "Recorder.SequenceIndex")
            .and_then(|s| s.parse().ok())
    }
    pub fn recorder_file_index(&self) -> Option<u64> {
        Self::extract_prop(&self.common_properties, "Recorder.FileIndex")
            .and_then(|s| s.parse().ok())
    }
    pub fn recorder_file_last(&self) -> Option<bool> {
        Self::extract_prop(&self.common_properties, "Recorder.FileLast")
            .map(|s| matches!(s.to_ascii_lowercase().as_str(), "true" | "1" | "yes"))
    }
    pub fn recorder_uuid(&self) -> Option<&str> {
        Self::extract_prop(&self.common_properties, "Recorder.UUID")
    }
    pub fn measurement_uuid(&self) -> Option<&str> {
        Self::extract_prop(&self.common_properties, "MeasurementUUID")
    }
    pub fn author(&self) -> Option<&str> {
        Self::extract_prop(&self.common_properties, "author")
    }
    pub fn department(&self) -> Option<&str> {
        Self::extract_prop(&self.common_properties, "department")
    }
    pub fn project(&self) -> Option<&str> {
        Self::extract_prop(&self.common_properties, "project")
    }
    pub fn subject(&self) -> Option<&str> {
        Self::extract_prop(&self.common_properties, "subject")
    }
}

/// FH block comment per fh_comment.xsd
#[derive(Debug, Clone, Default)]
pub struct FhComment {
    pub tx: Option<String>,
    pub tool_id: Option<String>,
    pub tool_vendor: Option<String>,
    pub tool_version: Option<String>,
    pub user_name: Option<String>,
    pub common_properties: CommonProperties,
}

impl Display for FhComment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(tx) = &self.tx {
            write!(f, "{tx}")?;
        }
        if let Some(tool) = &self.tool_id {
            write!(f, " tool={tool}")?;
        }
        if let Some(vendor) = &self.tool_vendor {
            write!(f, " vendor={vendor}")?;
        }
        if let Some(ver) = &self.tool_version {
            write!(f, " v{ver}")?;
        }
        if let Some(user) = &self.user_name {
            write!(f, " user={user}")?;
        }
        Ok(())
    }
}

/// CN block comment per cn_comment.xsd
#[derive(Debug, Clone, Default)]
pub struct CnComment {
    pub tx: Option<String>,
    pub names: MdNames,
    pub linker_name: Option<String>,
    pub linker_address: Option<String>,
    pub axis_monotony: Option<String>,
    /// Raster: (min, max, avg)
    pub raster: Option<(Option<f64>, Option<f64>, Option<f64>)>,
    pub formula: Option<String>,
    pub address: Option<String>,
    pub common_properties: CommonProperties,
}

impl Display for CnComment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(tx) = &self.tx {
            write!(f, "{tx}")?;
        }
        let names = format!("{}", self.names);
        if !names.is_empty() {
            write!(f, " names({names})")?;
        }
        if let Some(formula) = &self.formula {
            write!(f, " formula={formula}")?;
        }
        if let Some(addr) = &self.address {
            write!(f, " addr={addr}")?;
        }
        if let Some(mono) = &self.axis_monotony {
            write!(f, " monotony={mono}")?;
        }
        Ok(())
    }
}

/// CG block comment per cg_comment.xsd
#[derive(Debug, Clone, Default)]
pub struct CgComment {
    pub tx: Option<String>,
    pub names: MdNames,
    pub common_properties: CommonProperties,
}

impl Display for CgComment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(tx) = &self.tx {
            write!(f, "{tx}")?;
        }
        let names = format!("{}", self.names);
        if !names.is_empty() {
            write!(f, " names({names})")?;
        }
        Ok(())
    }
}

/// CC block comment per cc_comment.xsd
#[derive(Debug, Clone, Default)]
pub struct CcComment {
    pub tx: Option<String>,
    pub names: MdNames,
    pub formula: Option<String>,
    pub common_properties: CommonProperties,
}

impl Display for CcComment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(tx) = &self.tx {
            write!(f, "{tx}")?;
        }
        let names = format!("{}", self.names);
        if !names.is_empty() {
            write!(f, " names({names})")?;
        }
        if let Some(formula) = &self.formula {
            write!(f, " formula={formula}")?;
        }
        Ok(())
    }
}

/// SI block comment per si_comment.xsd
#[derive(Debug, Clone, Default)]
pub struct SiComment {
    pub tx: Option<String>,
    pub names: MdNames,
    pub path: MdNames,
    pub bus: MdNames,
    pub protocol: Option<String>,
    pub common_properties: CommonProperties,
}

impl Display for SiComment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(tx) = &self.tx {
            write!(f, "{tx}")?;
        }
        if let Some(proto) = &self.protocol {
            write!(f, " protocol={proto}")?;
        }
        let names = format!("{}", self.names);
        if !names.is_empty() {
            write!(f, " names({names})")?;
        }
        Ok(())
    }
}

/// EV block comment per ev_comment.xsd
#[derive(Debug, Clone, Default)]
pub struct EvComment {
    pub tx: Option<String>,
    pub pre_trigger_interval: Option<f64>,
    pub post_trigger_interval: Option<f64>,
    pub formula: Option<String>,
    pub timeout: Option<f64>,
    pub common_properties: CommonProperties,
}

impl Display for EvComment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(tx) = &self.tx {
            write!(f, "{tx}")?;
        }
        if let Some(pre) = &self.pre_trigger_interval {
            write!(f, " pre_trigger={pre}")?;
        }
        if let Some(post) = &self.post_trigger_interval {
            write!(f, " post_trigger={post}")?;
        }
        if let Some(formula) = &self.formula {
            write!(f, " formula={formula}")?;
        }
        if let Some(timeout) = &self.timeout {
            write!(f, " timeout={timeout}")?;
        }
        Ok(())
    }
}

/// AT block comment per at_comment.xsd
#[derive(Debug, Clone, Default)]
pub struct AtComment {
    pub tx: Option<String>,
    pub common_properties: CommonProperties,
}

impl Display for AtComment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(tx) = &self.tx {
            write!(f, "{tx}")?;
        }
        if !self.common_properties.is_empty() {
            write!(f, " props={}", self.common_properties.len())?;
        }
        Ok(())
    }
}

/// CH block comment per ch_comment.xsd
#[derive(Debug, Clone, Default)]
pub struct ChComment {
    pub tx: Option<String>,
    pub names: MdNames,
    pub common_properties: CommonProperties,
}

impl Display for ChComment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(tx) = &self.tx {
            write!(f, "{tx}")?;
        }
        let names = format!("{}", self.names);
        if !names.is_empty() {
            write!(f, " names({names})")?;
        }
        Ok(())
    }
}

/// DG block comment per dg_comment.xsd
#[derive(Debug, Clone, Default)]
pub struct DgComment {
    pub tx: Option<String>,
    pub common_properties: CommonProperties,
}

impl Display for DgComment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(tx) = &self.tx {
            write!(f, "{tx}")?;
        }
        if !self.common_properties.is_empty() {
            write!(f, " props={}", self.common_properties.len())?;
        }
        Ok(())
    }
}

/// Parsed metadata, typed per parent block schema
#[derive(Debug, Clone)]
pub enum MdComment {
    Hd(HdComment),
    Fh(FhComment),
    Cn(CnComment),
    Cg(CgComment),
    Cc(CcComment),
    Si(SiComment),
    Ev(EvComment),
    At(AtComment),
    Ch(ChComment),
    Dg(DgComment),
}

impl Display for MdComment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MdComment::Hd(c) => write!(f, "{c}"),
            MdComment::Fh(c) => write!(f, "{c}"),
            MdComment::Cn(c) => write!(f, "{c}"),
            MdComment::Cg(c) => write!(f, "{c}"),
            MdComment::Cc(c) => write!(f, "{c}"),
            MdComment::Si(c) => write!(f, "{c}"),
            MdComment::Ev(c) => write!(f, "{c}"),
            MdComment::At(c) => write!(f, "{c}"),
            MdComment::Ch(c) => write!(f, "{c}"),
            MdComment::Dg(c) => write!(f, "{c}"),
        }
    }
}

impl MdComment {
    /// Returns the TX text from any comment variant
    pub fn get_tx(&self) -> Option<&str> {
        match self {
            MdComment::Hd(c) => c.tx.as_deref(),
            MdComment::Fh(c) => c.tx.as_deref(),
            MdComment::Cn(c) => c.tx.as_deref(),
            MdComment::Cg(c) => c.tx.as_deref(),
            MdComment::Cc(c) => c.tx.as_deref(),
            MdComment::Si(c) => c.tx.as_deref(),
            MdComment::Ev(c) => c.tx.as_deref(),
            MdComment::At(c) => c.tx.as_deref(),
            MdComment::Ch(c) => c.tx.as_deref(),
            MdComment::Dg(c) => c.tx.as_deref(),
        }
    }
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
    /// Parent block type
    pub parent_block_type: BlockType,
    /// Typed parsed metadata (replaces flat comments HashMap)
    pub md_comment: Option<MdComment>,
}

impl Display for MetaData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.block_type {
            MetaDataBlockType::MdParsed => {
                if let Some(mc) = &self.md_comment {
                    write!(f, "{mc}")
                } else {
                    write!(f, "MD (parsed, empty)")
                }
            }
            MetaDataBlockType::MdBlock => {
                write!(f, "MD (unparsed) raw_bytes={}", self.raw_data.len())
            }
            MetaDataBlockType::TX => {
                write!(f, "TX raw_bytes={}", self.raw_data.len())
            }
        }
    }
}

/// Parse `<names>`, `<path>`, or `<bus>` element node into MdNames (single pass)
fn parse_names_node(node: roxmltree::Node) -> MdNames {
    let mut names = MdNames::default();
    for child in node.children().filter(roxmltree::Node::is_element) {
        match child.tag_name().name() {
            "name" => names.name = child.text().map(str::to_string),
            "display" => names.display = child.text().map(str::to_string),
            "vendor" => names.vendor = child.text().map(str::to_string),
            "description" => names.description = child.text().map(str::to_string),
            _ => {}
        }
    }
    names
}

/// Parse children of a common_properties or tree node
fn parse_properties_children(node: roxmltree::Node, props: &mut HashMap<String, PropertyValue>) {
    for child in node.children().filter(roxmltree::Node::is_element) {
        let tag = child.tag_name().name();
        match tag {
            "e" => {
                if let Some(name) = child.attribute("name") {
                    let value = child.text().unwrap_or("").to_string();
                    props.insert(name.to_string(), PropertyValue::Value(value));
                }
            }
            "tree" => {
                if let Some(name) = child.attribute("name") {
                    let mut sub = HashMap::new();
                    parse_properties_children(child, &mut sub);
                    props.insert(name.to_string(), PropertyValue::Tree(sub));
                }
            }
            "list" => {
                if let Some(name) = child.attribute("name") {
                    let mut items = Vec::new();
                    for li in child
                        .children()
                        .filter(|n| n.is_element() && n.has_tag_name("li"))
                    {
                        let mut item = HashMap::new();
                        parse_properties_children(li, &mut item);
                        items.push(item);
                    }
                    props.insert(name.to_string(), PropertyValue::List(items));
                }
            }
            "elist" => {
                if let Some(name) = child.attribute("name") {
                    let items: Vec<String> = child
                        .children()
                        .filter(|n| n.is_element() && n.has_tag_name("eli"))
                        .filter_map(|n| n.text().map(std::string::ToString::to_string))
                        .collect();
                    props.insert(name.to_string(), PropertyValue::EList(items));
                }
            }
            _ => {}
        }
    }
}

/// Parse `<raster>` element node into (min, max, avg) tuple (single pass)
fn parse_raster_node(node: roxmltree::Node) -> (Option<f64>, Option<f64>, Option<f64>) {
    let (mut min, mut max, mut avg) = (None, None, None);
    for child in node.children().filter(roxmltree::Node::is_element) {
        match child.tag_name().name() {
            "min" => min = child.text().and_then(|s| s.parse().ok()),
            "max" => max = child.text().and_then(|s| s.parse().ok()),
            "avg" => avg = child.text().and_then(|s| s.parse().ok()),
            _ => {}
        }
    }
    (min, max, avg)
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
            parent_block_type,
            md_comment: None,
        }
    }
    /// Converts the metadata handling the parent block type's specificities
    pub fn parse_xml(&mut self) -> Result<()> {
        if self.block_type == MetaDataBlockType::MdBlock {
            match self.parent_block_type {
                BlockType::HD => self.parse_hd_comment()?,
                BlockType::FH => self.parse_fh_comment()?,
                BlockType::CN => self.parse_cn_comment()?,
                BlockType::CG => self.parse_cg_comment()?,
                BlockType::CC => self.parse_cc_comment()?,
                BlockType::SI => self.parse_si_comment()?,
                BlockType::EV => self.parse_ev_comment()?,
                BlockType::AT => self.parse_at_comment()?,
                BlockType::CH => self.parse_ch_comment()?,
                BlockType::DG => self.parse_dg_comment()?,
            };
        }
        Ok(())
    }
    /// Returns the text from TX Block or TX's tag text from MD Block
    pub fn get_tx(&self) -> Result<Option<String>, anyhow::Error> {
        match self.block_type {
            MetaDataBlockType::MdParsed => Ok(self
                .md_comment
                .as_ref()
                .and_then(|mc| mc.get_tx())
                .map(std::string::ToString::to_string)),
            MetaDataBlockType::MdBlock => {
                let xml_str = self
                    .get_xml_str()
                    .context("failed getting data string to extract TX tag")?;
                match roxmltree::Document::parse(xml_str) {
                    Ok(md) => {
                        for node in md.root().descendants() {
                            if node.is_element()
                                && node.tag_name().name() == "TX"
                                && let Some(text) = node.text()
                                && !text.is_empty()
                            {
                                return Ok(Some(text.to_string()));
                            }
                        }
                        Ok(None)
                    }
                    Err(e) => {
                        log::warn!("Error parsing comment : \n{xml_str}\n{e}");
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
        if self.raw_data.is_empty() {
            None
        } else {
            Some(&self.raw_data)
        }
    }
    /// allocate bytes to raw_data field, adjusting header length
    pub fn set_data_buffer(&mut self, data: &[u8]) {
        self.raw_data = [data, vec![0u8; 8 - data.len() % 8].as_slice()].concat();
        self.block.hdr_len = self.raw_data.len() as u64 + 24;
    }
    /// Zero-allocation helper: borrow trimmed XML str from raw_data
    fn get_xml_str(&self) -> Result<&str> {
        let s = str::from_utf8(&self.raw_data).context("Invalid UTF-8 in metadata")?;
        Ok(s.trim_end_matches(['\0', '\n', '\r', ' ']))
    }
}

impl MetaData {
    /// Parse HD block MD comment (hd_comment.xsd)
    pub fn parse_hd_comment(&mut self) -> Result<()> {
        let mut hd = HdComment::default();
        {
            let xml_str = self.get_xml_str()?;
            match roxmltree::Document::parse(xml_str) {
                Ok(doc) => {
                    let root = doc.root_element();
                    for child in root.children().filter(roxmltree::Node::is_element) {
                        match child.tag_name().name() {
                            "TX" => hd.tx = child.text().map(str::to_string),
                            "time_source" => hd.time_source = child.text().map(str::to_string),
                            "constants" => {
                                for c in child
                                    .children()
                                    .filter(|n| n.is_element() && n.has_tag_name("const"))
                                {
                                    if let (Some(name), Some(text)) =
                                        (c.attribute("name"), c.text())
                                    {
                                        hd.constants.insert(name.to_string(), text.to_string());
                                    }
                                }
                            }
                            "common_properties" => {
                                parse_properties_children(child, &mut hd.common_properties)
                            }
                            _ => {}
                        }
                    }
                }
                Err(e) => {
                    log::warn!("Could not parse HD MD comment : \n{xml_str}\n{e}");
                }
            }
        }
        self.md_comment = Some(MdComment::Hd(hd));
        self.block_type = MetaDataBlockType::MdParsed;
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
    /// Parse FH block MD comment (fh_comment.xsd)
    fn parse_fh_comment(&mut self) -> Result<()> {
        let mut fh = FhComment::default();
        {
            let xml_str = self.get_xml_str()?;
            match roxmltree::Document::parse(xml_str) {
                Ok(doc) => {
                    let root = doc.root_element();
                    for child in root.children().filter(roxmltree::Node::is_element) {
                        match child.tag_name().name() {
                            "TX" => fh.tx = child.text().map(str::to_string),
                            "tool_id" => fh.tool_id = child.text().map(str::to_string),
                            "tool_vendor" => fh.tool_vendor = child.text().map(str::to_string),
                            "tool_version" => fh.tool_version = child.text().map(str::to_string),
                            "user_name" => fh.user_name = child.text().map(str::to_string),
                            "common_properties" => {
                                parse_properties_children(child, &mut fh.common_properties)
                            }
                            _ => {}
                        }
                    }
                }
                Err(e) => log::warn!("Could not parse FH comment : \n{xml_str}\n{e}"),
            }
        }
        self.md_comment = Some(MdComment::Fh(fh));
        self.block_type = MetaDataBlockType::MdParsed;
        Ok(())
    }
    /// Parse CN block MD comment (cn_comment.xsd)
    fn parse_cn_comment(&mut self) -> Result<()> {
        let mut cn = CnComment::default();
        {
            let xml_str = self.get_xml_str()?;
            match roxmltree::Document::parse(xml_str) {
                Ok(doc) => {
                    let root = doc.root_element();
                    for child in root.children().filter(roxmltree::Node::is_element) {
                        match child.tag_name().name() {
                            "TX" => cn.tx = child.text().map(str::to_string),
                            "names" => cn.names = parse_names_node(child),
                            "linker_name" => cn.linker_name = child.text().map(str::to_string),
                            "linker_address" => {
                                cn.linker_address = child.text().map(str::to_string)
                            }
                            "axis_monotony" => cn.axis_monotony = child.text().map(str::to_string),
                            "raster" => cn.raster = Some(parse_raster_node(child)),
                            "formula" => cn.formula = child.text().map(str::to_string),
                            "address" => cn.address = child.text().map(str::to_string),
                            "common_properties" => {
                                parse_properties_children(child, &mut cn.common_properties)
                            }
                            _ => {}
                        }
                    }
                }
                Err(e) => log::warn!("Could not parse CN comment : \n{xml_str}\n{e}"),
            }
        }
        self.md_comment = Some(MdComment::Cn(cn));
        self.block_type = MetaDataBlockType::MdParsed;
        Ok(())
    }
    /// Parse CG block MD comment (cg_comment.xsd)
    fn parse_cg_comment(&mut self) -> Result<()> {
        let mut cg = CgComment::default();
        {
            let xml_str = self.get_xml_str()?;
            match roxmltree::Document::parse(xml_str) {
                Ok(doc) => {
                    let root = doc.root_element();
                    for child in root.children().filter(roxmltree::Node::is_element) {
                        match child.tag_name().name() {
                            "TX" => cg.tx = child.text().map(str::to_string),
                            "names" => cg.names = parse_names_node(child),
                            "common_properties" => {
                                parse_properties_children(child, &mut cg.common_properties)
                            }
                            _ => {}
                        }
                    }
                }
                Err(e) => log::warn!("Could not parse CG comment : \n{xml_str}\n{e}"),
            }
        }
        self.md_comment = Some(MdComment::Cg(cg));
        self.block_type = MetaDataBlockType::MdParsed;
        Ok(())
    }
    /// Parse CC block MD comment (cc_comment.xsd)
    fn parse_cc_comment(&mut self) -> Result<()> {
        let mut cc = CcComment::default();
        {
            let xml_str = self.get_xml_str()?;
            match roxmltree::Document::parse(xml_str) {
                Ok(doc) => {
                    let root = doc.root_element();
                    for child in root.children().filter(roxmltree::Node::is_element) {
                        match child.tag_name().name() {
                            "TX" => cc.tx = child.text().map(str::to_string),
                            "names" => cc.names = parse_names_node(child),
                            "formula" => cc.formula = child.text().map(str::to_string),
                            "common_properties" => {
                                parse_properties_children(child, &mut cc.common_properties)
                            }
                            _ => {}
                        }
                    }
                }
                Err(e) => log::warn!("Could not parse CC comment : \n{xml_str}\n{e}"),
            }
        }
        self.md_comment = Some(MdComment::Cc(cc));
        self.block_type = MetaDataBlockType::MdParsed;
        Ok(())
    }
    /// Parse SI block MD comment (si_comment.xsd)
    fn parse_si_comment(&mut self) -> Result<()> {
        let mut si = SiComment::default();
        {
            let xml_str = self.get_xml_str()?;
            match roxmltree::Document::parse(xml_str) {
                Ok(doc) => {
                    let root = doc.root_element();
                    for child in root.children().filter(roxmltree::Node::is_element) {
                        match child.tag_name().name() {
                            "TX" => si.tx = child.text().map(str::to_string),
                            "names" => si.names = parse_names_node(child),
                            "path" => si.path = parse_names_node(child),
                            "bus" => si.bus = parse_names_node(child),
                            "protocol" => si.protocol = child.text().map(str::to_string),
                            "common_properties" => {
                                parse_properties_children(child, &mut si.common_properties)
                            }
                            _ => {}
                        }
                    }
                }
                Err(e) => log::warn!("Could not parse SI comment : \n{xml_str}\n{e}"),
            }
        }
        self.md_comment = Some(MdComment::Si(si));
        self.block_type = MetaDataBlockType::MdParsed;
        Ok(())
    }
    /// Parse EV block MD comment (ev_comment.xsd)
    fn parse_ev_comment(&mut self) -> Result<()> {
        let mut ev = EvComment::default();
        {
            let xml_str = self.get_xml_str()?;
            match roxmltree::Document::parse(xml_str) {
                Ok(doc) => {
                    let root = doc.root_element();
                    for child in root.children().filter(roxmltree::Node::is_element) {
                        match child.tag_name().name() {
                            "TX" => ev.tx = child.text().map(str::to_string),
                            "pre_trigger_interval" => {
                                ev.pre_trigger_interval = child.text().and_then(|s| s.parse().ok())
                            }
                            "post_trigger_interval" => {
                                ev.post_trigger_interval = child.text().and_then(|s| s.parse().ok())
                            }
                            "formula" => ev.formula = child.text().map(str::to_string),
                            "timeout" => ev.timeout = child.text().and_then(|s| s.parse().ok()),
                            "common_properties" => {
                                parse_properties_children(child, &mut ev.common_properties)
                            }
                            _ => {}
                        }
                    }
                }
                Err(e) => log::warn!("Could not parse EV comment : \n{xml_str}\n{e}"),
            }
        }
        self.md_comment = Some(MdComment::Ev(ev));
        self.block_type = MetaDataBlockType::MdParsed;
        Ok(())
    }
    /// Parse AT block MD comment (at_comment.xsd)
    fn parse_at_comment(&mut self) -> Result<()> {
        let mut at = AtComment::default();
        {
            let xml_str = self.get_xml_str()?;
            match roxmltree::Document::parse(xml_str) {
                Ok(doc) => {
                    let root = doc.root_element();
                    for child in root.children().filter(roxmltree::Node::is_element) {
                        match child.tag_name().name() {
                            "TX" => at.tx = child.text().map(str::to_string),
                            "common_properties" => {
                                parse_properties_children(child, &mut at.common_properties)
                            }
                            _ => {}
                        }
                    }
                }
                Err(e) => log::warn!("Could not parse AT comment : \n{xml_str}\n{e}"),
            }
        }
        self.md_comment = Some(MdComment::At(at));
        self.block_type = MetaDataBlockType::MdParsed;
        Ok(())
    }
    /// Parse CH block MD comment (ch_comment.xsd)
    fn parse_ch_comment(&mut self) -> Result<()> {
        let mut ch = ChComment::default();
        {
            let xml_str = self.get_xml_str()?;
            match roxmltree::Document::parse(xml_str) {
                Ok(doc) => {
                    let root = doc.root_element();
                    for child in root.children().filter(roxmltree::Node::is_element) {
                        match child.tag_name().name() {
                            "TX" => ch.tx = child.text().map(str::to_string),
                            "names" => ch.names = parse_names_node(child),
                            "common_properties" => {
                                parse_properties_children(child, &mut ch.common_properties)
                            }
                            _ => {}
                        }
                    }
                }
                Err(e) => log::warn!("Could not parse CH comment : \n{xml_str}\n{e}"),
            }
        }
        self.md_comment = Some(MdComment::Ch(ch));
        self.block_type = MetaDataBlockType::MdParsed;
        Ok(())
    }
    /// Parse DG block MD comment (dg_comment.xsd)
    fn parse_dg_comment(&mut self) -> Result<()> {
        let mut dg = DgComment::default();
        {
            let xml_str = self.get_xml_str()?;
            match roxmltree::Document::parse(xml_str) {
                Ok(doc) => {
                    let root = doc.root_element();
                    for child in root.children().filter(roxmltree::Node::is_element) {
                        match child.tag_name().name() {
                            "TX" => dg.tx = child.text().map(str::to_string),
                            "common_properties" => {
                                parse_properties_children(child, &mut dg.common_properties)
                            }
                            _ => {}
                        }
                    }
                }
                Err(e) => log::warn!("Could not parse DG comment : \n{xml_str}\n{e}"),
            }
        }
        self.md_comment = Some(MdComment::Dg(dg));
        self.block_type = MetaDataBlockType::MdParsed;
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── Constructor & data access tests ──

    #[test]
    fn test_metadata_new_tx() {
        let md = MetaData::new(MetaDataBlockType::TX, BlockType::CN);
        assert_eq!(md.block.hdr_id, [35, 35, 84, 88]); // ##TX
        assert_eq!(md.block_type, MetaDataBlockType::TX);
        assert!(md.raw_data.is_empty());
        assert!(md.md_comment.is_none());
    }

    #[test]
    fn test_metadata_new_md() {
        let md = MetaData::new(MetaDataBlockType::MdBlock, BlockType::HD);
        assert_eq!(md.block.hdr_id, [35, 35, 77, 68]); // ##MD
        assert_eq!(md.block_type, MetaDataBlockType::MdBlock);
    }

    #[test]
    fn test_metadata_set_data_buffer() {
        let mut md = MetaData::new(MetaDataBlockType::TX, BlockType::CN);

        // 1 byte → padded to 8
        md.set_data_buffer(&[65]);
        assert_eq!(md.raw_data.len(), 8);
        assert_eq!(md.raw_data[0], 65);
        assert_eq!(md.block.hdr_len, 8 + 24);

        // 7 bytes → padded to 8
        md.set_data_buffer(&[1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(md.raw_data.len(), 8);

        // 8 bytes → padded to 16 (always adds padding)
        md.set_data_buffer(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(md.raw_data.len(), 16);

        // 9 bytes → padded to 16
        md.set_data_buffer(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(md.raw_data.len(), 16);
        assert_eq!(md.block.hdr_len, 16 + 24);
    }

    #[test]
    fn test_metadata_get_tx_bytes() {
        let mut md = MetaData::new(MetaDataBlockType::TX, BlockType::CN);
        assert!(md.get_tx_bytes().is_none());

        md.set_data_buffer(b"hello");
        assert!(md.get_tx_bytes().is_some());
        assert!(md.get_tx_bytes().unwrap().starts_with(b"hello"));
    }

    #[test]
    fn test_metadata_get_tx_for_tx_block() {
        let mut md = MetaData::new(MetaDataBlockType::TX, BlockType::CN);
        md.set_data_buffer(b"test_channel\0");

        let tx = md.get_tx().unwrap();
        assert_eq!(tx, Some("test_channel".to_string()));
    }

    // ── XML parsing tests ──

    fn make_md(block_type: BlockType, xml: &str) -> MetaData {
        let mut md = MetaData::new(MetaDataBlockType::MdBlock, block_type);
        md.set_data_buffer(xml.as_bytes());
        md
    }

    #[test]
    fn test_parse_hd_comment() {
        let xml = r#"<HDcomment>
<TX>Test measurement</TX>
<time_source>PC timer</time_source>
<constants>
<const name="pi">3.14159</const>
<const name="g">9.81</const>
</constants>
<common_properties>
<e name="author">tester</e>
</common_properties>
</HDcomment>"#;
        let mut md = make_md(BlockType::HD, xml);
        md.parse_xml().unwrap();

        assert_eq!(md.block_type, MetaDataBlockType::MdParsed);
        let comment = md.md_comment.as_ref().unwrap();
        if let MdComment::Hd(hd) = comment {
            assert_eq!(hd.tx.as_deref(), Some("Test measurement"));
            assert_eq!(hd.time_source.as_deref(), Some("PC timer"));
            assert_eq!(hd.constants.get("pi").map(|s| s.as_str()), Some("3.14159"));
            assert_eq!(hd.constants.get("g").map(|s| s.as_str()), Some("9.81"));
            assert_eq!(hd.constants.len(), 2);
            if let Some(PropertyValue::Value(v)) = hd.common_properties.get("author") {
                assert_eq!(v, "tester");
            } else {
                panic!("Expected Value for 'author'");
            }
        } else {
            panic!("Expected MdComment::Hd");
        }
    }

    #[test]
    fn test_parse_fh_comment() {
        let xml = r#"<FHcomment>
<TX>created</TX>
<tool_id>mdfr</tool_id>
<tool_vendor>ratalco</tool_vendor>
<tool_version>0.1</tool_version>
<user_name>testuser</user_name>
</FHcomment>"#;
        let mut md = make_md(BlockType::FH, xml);
        md.parse_xml().unwrap();

        if let Some(MdComment::Fh(fh)) = &md.md_comment {
            assert_eq!(fh.tx.as_deref(), Some("created"));
            assert_eq!(fh.tool_id.as_deref(), Some("mdfr"));
            assert_eq!(fh.tool_vendor.as_deref(), Some("ratalco"));
            assert_eq!(fh.tool_version.as_deref(), Some("0.1"));
            assert_eq!(fh.user_name.as_deref(), Some("testuser"));
        } else {
            panic!("Expected MdComment::Fh");
        }
    }

    #[test]
    fn test_parse_cn_comment() {
        let xml = r#"<CNcomment>
<TX>Engine speed</TX>
<names><name>RPM</name><display>Engine RPM</display></names>
<formula>x * 0.1</formula>
<address>0x1234</address>
<raster><min>0.01</min><max>0.1</max><avg>0.05</avg></raster>
</CNcomment>"#;
        let mut md = make_md(BlockType::CN, xml);
        md.parse_xml().unwrap();

        if let Some(MdComment::Cn(cn)) = &md.md_comment {
            assert_eq!(cn.tx.as_deref(), Some("Engine speed"));
            assert_eq!(cn.names.name.as_deref(), Some("RPM"));
            assert_eq!(cn.names.display.as_deref(), Some("Engine RPM"));
            assert_eq!(cn.formula.as_deref(), Some("x * 0.1"));
            assert_eq!(cn.address.as_deref(), Some("0x1234"));
            let (min, max, avg) = cn.raster.unwrap();
            assert_eq!(min, Some(0.01));
            assert_eq!(max, Some(0.1));
            assert_eq!(avg, Some(0.05));
        } else {
            panic!("Expected MdComment::Cn");
        }
    }

    #[test]
    fn test_parse_ev_comment() {
        let xml = r#"<EVcomment>
<TX>trigger event</TX>
<pre_trigger_interval>1.5</pre_trigger_interval>
<post_trigger_interval>3.0</post_trigger_interval>
<formula>x &gt; 100</formula>
<timeout>10.0</timeout>
</EVcomment>"#;
        let mut md = make_md(BlockType::EV, xml);
        md.parse_xml().unwrap();

        if let Some(MdComment::Ev(ev)) = &md.md_comment {
            assert_eq!(ev.tx.as_deref(), Some("trigger event"));
            assert_eq!(ev.pre_trigger_interval, Some(1.5));
            assert_eq!(ev.post_trigger_interval, Some(3.0));
            assert_eq!(ev.formula.as_deref(), Some("x > 100"));
            assert_eq!(ev.timeout, Some(10.0));
        } else {
            panic!("Expected MdComment::Ev");
        }
    }

    #[test]
    fn test_parse_si_comment() {
        let xml = r#"<SIcomment>
<TX>ECU source</TX>
<names><name>ECU_1</name><vendor>Bosch</vendor></names>
<path><name>/CAN/ECU_1</name></path>
<bus><name>CAN1</name></bus>
<protocol>CAN</protocol>
</SIcomment>"#;
        let mut md = make_md(BlockType::SI, xml);
        md.parse_xml().unwrap();

        if let Some(MdComment::Si(si)) = &md.md_comment {
            assert_eq!(si.tx.as_deref(), Some("ECU source"));
            assert_eq!(si.names.name.as_deref(), Some("ECU_1"));
            assert_eq!(si.names.vendor.as_deref(), Some("Bosch"));
            assert_eq!(si.path.name.as_deref(), Some("/CAN/ECU_1"));
            assert_eq!(si.bus.name.as_deref(), Some("CAN1"));
            assert_eq!(si.protocol.as_deref(), Some("CAN"));
        } else {
            panic!("Expected MdComment::Si");
        }
    }

    #[test]
    fn test_parse_cg_comment() {
        let xml = r#"<CGcomment>
<TX>Group 1</TX>
<names><name>CG_1</name><description>First group</description></names>
</CGcomment>"#;
        let mut md = make_md(BlockType::CG, xml);
        md.parse_xml().unwrap();

        if let Some(MdComment::Cg(cg)) = &md.md_comment {
            assert_eq!(cg.tx.as_deref(), Some("Group 1"));
            assert_eq!(cg.names.name.as_deref(), Some("CG_1"));
            assert_eq!(cg.names.description.as_deref(), Some("First group"));
        } else {
            panic!("Expected MdComment::Cg");
        }
    }

    #[test]
    fn test_parse_cc_comment() {
        let xml = r#"<CCcomment>
<TX>Conversion rule</TX>
<names><name>linear_conv</name></names>
<formula>x * 2 + 1</formula>
</CCcomment>"#;
        let mut md = make_md(BlockType::CC, xml);
        md.parse_xml().unwrap();

        if let Some(MdComment::Cc(cc)) = &md.md_comment {
            assert_eq!(cc.tx.as_deref(), Some("Conversion rule"));
            assert_eq!(cc.names.name.as_deref(), Some("linear_conv"));
            assert_eq!(cc.formula.as_deref(), Some("x * 2 + 1"));
        } else {
            panic!("Expected MdComment::Cc");
        }
    }

    #[test]
    fn test_parse_common_properties() {
        let xml = r#"<HDcomment>
<TX>props test</TX>
<common_properties>
<e name="simple">value1</e>
<tree name="nested">
  <e name="inner_key">inner_val</e>
</tree>
<list name="items">
  <li><e name="a">1</e></li>
  <li><e name="b">2</e></li>
</list>
<elist name="tags">
  <eli>alpha</eli>
  <eli>beta</eli>
</elist>
</common_properties>
</HDcomment>"#;
        let mut md = make_md(BlockType::HD, xml);
        md.parse_xml().unwrap();

        if let Some(MdComment::Hd(hd)) = &md.md_comment {
            // Value
            if let Some(PropertyValue::Value(v)) = hd.common_properties.get("simple") {
                assert_eq!(v, "value1");
            } else {
                panic!("Expected Value for 'simple'");
            }
            // Tree
            if let Some(PropertyValue::Tree(sub)) = hd.common_properties.get("nested") {
                if let Some(PropertyValue::Value(v)) = sub.get("inner_key") {
                    assert_eq!(v, "inner_val");
                } else {
                    panic!("Expected inner_key in tree");
                }
            } else {
                panic!("Expected Tree for 'nested'");
            }
            // List
            if let Some(PropertyValue::List(items)) = hd.common_properties.get("items") {
                assert_eq!(items.len(), 2);
                if let Some(PropertyValue::Value(v)) = items[0].get("a") {
                    assert_eq!(v, "1");
                } else {
                    panic!("Expected 'a' in first list item");
                }
            } else {
                panic!("Expected List for 'items'");
            }
            // EList
            if let Some(PropertyValue::EList(items)) = hd.common_properties.get("tags") {
                assert_eq!(items, &["alpha", "beta"]);
            } else {
                panic!("Expected EList for 'tags'");
            }
        } else {
            panic!("Expected MdComment::Hd");
        }
    }

    // ── Display tests ──

    #[test]
    fn test_md_names_display() {
        let names = MdNames {
            name: Some("ch1".into()),
            display: Some("Channel 1".into()),
            vendor: None,
            description: Some("First channel".into()),
        };
        let s = format!("{names}");
        assert!(s.contains("name=ch1"));
        assert!(s.contains("display=Channel 1"));
        assert!(!s.contains("vendor="));
        assert!(s.contains("desc=First channel"));

        // All None
        let empty = MdNames::default();
        assert_eq!(format!("{empty}"), "");
    }

    #[test]
    fn test_property_value_display() {
        assert_eq!(format!("{}", PropertyValue::Value("hello".into())), "hello");
        assert!(format!("{}", PropertyValue::Tree(HashMap::new())).contains("tree(0 items)"));
        assert!(format!("{}", PropertyValue::List(vec![])).contains("list(0 items)"));
        assert!(
            format!("{}", PropertyValue::EList(vec!["a".into(), "b".into()]))
                .contains("elist(2 items)")
        );
    }

    #[test]
    fn test_md_comment_display() {
        let hd = MdComment::Hd(HdComment {
            tx: Some("measurement".into()),
            time_source: Some("GPS".into()),
            constants: HashMap::new(),
            common_properties: HashMap::new(),
        });
        let s = format!("{hd}");
        assert!(s.contains("measurement"));
        assert!(s.contains("time_source=GPS"));
    }

    #[test]
    fn test_md_comment_get_tx() {
        let hd = MdComment::Hd(HdComment {
            tx: Some("hd_text".into()),
            ..Default::default()
        });
        assert_eq!(hd.get_tx(), Some("hd_text"));

        let fh = MdComment::Fh(FhComment {
            tx: None,
            ..Default::default()
        });
        assert_eq!(fh.get_tx(), None);

        let ev = MdComment::Ev(EvComment {
            tx: Some("event".into()),
            ..Default::default()
        });
        assert_eq!(ev.get_tx(), Some("event"));
    }

    #[test]
    fn test_metadata_display() {
        // TX type
        let mut md = MetaData::new(MetaDataBlockType::TX, BlockType::CN);
        md.set_data_buffer(b"hello");
        let s = format!("{md}");
        assert!(s.contains("TX"));

        // MdBlock (unparsed)
        let mut md2 = MetaData::new(MetaDataBlockType::MdBlock, BlockType::HD);
        md2.set_data_buffer(b"<HDcomment></HDcomment>");
        let s2 = format!("{md2}");
        assert!(s2.contains("MD (unparsed)"));

        // MdParsed with comment
        md2.parse_xml().unwrap();
        let s3 = format!("{md2}");
        assert!(!s3.contains("unparsed"));

        // MdParsed without comment
        let md3 = MetaData {
            block_type: MetaDataBlockType::MdParsed,
            md_comment: None,
            ..MetaData::default()
        };
        let s4 = format!("{md3}");
        assert!(s4.contains("MD (parsed, empty)"));
    }

    #[test]
    fn test_metadata_get_tx_md_block() {
        let xml = r#"<CNcomment><TX>channel desc</TX></CNcomment>"#;
        let mut md = MetaData::new(MetaDataBlockType::MdBlock, BlockType::CN);
        md.set_data_buffer(xml.as_bytes());

        // get_tx on unparsed MD should extract TX tag from XML
        let tx = md.get_tx().unwrap();
        assert_eq!(tx, Some("channel desc".to_string()));
    }

    #[test]
    fn test_metadata_get_tx_md_parsed() {
        let xml = r#"<HDcomment><TX>parsed text</TX></HDcomment>"#;
        let mut md = make_md(BlockType::HD, xml);
        md.parse_xml().unwrap();

        let tx = md.get_tx().unwrap();
        assert_eq!(tx, Some("parsed text".to_string()));
    }

    #[test]
    fn test_parse_dg_comment() {
        let xml = r#"<DGcomment>
<TX>data group info</TX>
<common_properties><e name="key">val</e></common_properties>
</DGcomment>"#;
        let mut md = make_md(BlockType::DG, xml);
        md.parse_xml().unwrap();

        if let Some(MdComment::Dg(dg)) = &md.md_comment {
            assert_eq!(dg.tx.as_deref(), Some("data group info"));
            assert!(dg.common_properties.contains_key("key"));
        } else {
            panic!("Expected MdComment::Dg");
        }
    }

    #[test]
    fn test_parse_ch_comment() {
        let xml = r#"<CHcomment>
<TX>hierarchy</TX>
<names><name>Group1</name><display>First Group</display></names>
</CHcomment>"#;
        let mut md = make_md(BlockType::CH, xml);
        md.parse_xml().unwrap();

        if let Some(MdComment::Ch(ch)) = &md.md_comment {
            assert_eq!(ch.tx.as_deref(), Some("hierarchy"));
            assert_eq!(ch.names.name.as_deref(), Some("Group1"));
            assert_eq!(ch.names.display.as_deref(), Some("First Group"));
        } else {
            panic!("Expected MdComment::Ch");
        }
    }

    #[test]
    fn test_parse_at_comment() {
        let xml = r#"<ATcomment>
<TX>attachment info</TX>
<common_properties><e name="mime">application/octet-stream</e></common_properties>
</ATcomment>"#;
        let mut md = make_md(BlockType::AT, xml);
        md.parse_xml().unwrap();

        if let Some(MdComment::At(at)) = &md.md_comment {
            assert_eq!(at.tx.as_deref(), Some("attachment info"));
            assert!(at.common_properties.contains_key("mime"));
        } else {
            panic!("Expected MdComment::At");
        }
    }

    #[test]
    fn test_parse_xml_tx_block_noop() {
        // parse_xml should be a no-op for TX blocks
        let mut md = MetaData::new(MetaDataBlockType::TX, BlockType::CN);
        md.set_data_buffer(b"plain text");
        md.parse_xml().unwrap();
        assert!(md.md_comment.is_none());
    }

    // ── Comprehensive Display tests for all comment types ──

    #[test]
    fn test_md_names_display_all_fields() {
        let names = MdNames {
            name: Some("ch1".into()),
            display: Some("Channel 1".into()),
            vendor: Some("Acme Corp".into()),
            description: Some("Test channel".into()),
        };
        let s = format!("{names}");
        assert!(s.contains("name=ch1"));
        assert!(s.contains("display=Channel 1"));
        assert!(s.contains("vendor=Acme Corp"));
        assert!(s.contains("desc=Test channel"));
    }

    #[test]
    fn test_hd_comment_display_full() {
        let mut constants = HashMap::new();
        constants.insert("pi".into(), "3.14".into());
        let mut props = HashMap::new();
        props.insert("author".into(), PropertyValue::Value("tester".into()));
        let hd = HdComment {
            tx: Some("measurement".into()),
            time_source: Some("GPS".into()),
            constants,
            common_properties: props,
        };
        let s = format!("{hd}");
        assert!(s.contains("measurement"));
        assert!(s.contains("time_source=GPS"));
        assert!(s.contains("constants=1"));
        assert!(s.contains("props=1"));
    }

    #[test]
    fn test_fh_comment_display_full() {
        let fh = FhComment {
            tx: Some("created".into()),
            tool_id: Some("mdfr".into()),
            tool_vendor: Some("ratalco".into()),
            tool_version: Some("0.6".into()),
            user_name: Some("tester".into()),
            common_properties: HashMap::new(),
        };
        let s = format!("{fh}");
        assert!(s.contains("created"));
        assert!(s.contains("tool=mdfr"));
        assert!(s.contains("vendor=ratalco"));
        assert!(s.contains("v0.6"));
        assert!(s.contains("user=tester"));
    }

    #[test]
    fn test_cn_comment_display() {
        let cn = CnComment {
            tx: Some("Engine speed".into()),
            names: MdNames {
                name: Some("RPM".into()),
                display: Some("Engine RPM".into()),
                ..Default::default()
            },
            formula: Some("x * 0.1".into()),
            address: Some("0x1234".into()),
            axis_monotony: Some("increasing".into()),
            ..Default::default()
        };
        let s = format!("{cn}");
        assert!(s.contains("Engine speed"));
        assert!(s.contains("names("));
        assert!(s.contains("formula=x * 0.1"));
        assert!(s.contains("addr=0x1234"));
        assert!(s.contains("monotony=increasing"));
    }

    #[test]
    fn test_cg_comment_display() {
        let cg = CgComment {
            tx: Some("Group 1".into()),
            names: MdNames {
                name: Some("CG_1".into()),
                ..Default::default()
            },
            common_properties: HashMap::new(),
        };
        let s = format!("{cg}");
        assert!(s.contains("Group 1"));
        assert!(s.contains("names("));
    }

    #[test]
    fn test_cc_comment_display() {
        let cc = CcComment {
            tx: Some("Linear".into()),
            names: MdNames {
                name: Some("conv1".into()),
                ..Default::default()
            },
            formula: Some("2*x+1".into()),
            common_properties: HashMap::new(),
        };
        let s = format!("{cc}");
        assert!(s.contains("Linear"));
        assert!(s.contains("names("));
        assert!(s.contains("formula=2*x+1"));
    }

    #[test]
    fn test_si_comment_display() {
        let si = SiComment {
            tx: Some("ECU source".into()),
            names: MdNames {
                name: Some("ECU_1".into()),
                ..Default::default()
            },
            protocol: Some("CAN".into()),
            ..Default::default()
        };
        let s = format!("{si}");
        assert!(s.contains("ECU source"));
        assert!(s.contains("protocol=CAN"));
        assert!(s.contains("names("));
    }

    #[test]
    fn test_ev_comment_display() {
        let ev = EvComment {
            tx: Some("trigger".into()),
            pre_trigger_interval: Some(1.5),
            post_trigger_interval: Some(3.0),
            formula: Some("x > 100".into()),
            timeout: Some(10.0),
            common_properties: HashMap::new(),
        };
        let s = format!("{ev}");
        assert!(s.contains("trigger"));
        assert!(s.contains("pre_trigger=1.5"));
        assert!(s.contains("post_trigger=3"));
        assert!(s.contains("formula=x > 100"));
        assert!(s.contains("timeout=10"));
    }

    #[test]
    fn test_at_comment_display() {
        let mut props = HashMap::new();
        props.insert("mime".into(), PropertyValue::Value("image/png".into()));
        let at = AtComment {
            tx: Some("attachment".into()),
            common_properties: props,
        };
        let s = format!("{at}");
        assert!(s.contains("attachment"));
        assert!(s.contains("props=1"));
    }

    #[test]
    fn test_ch_comment_display() {
        let ch = ChComment {
            tx: Some("hierarchy".into()),
            names: MdNames {
                name: Some("Group1".into()),
                ..Default::default()
            },
            common_properties: HashMap::new(),
        };
        let s = format!("{ch}");
        assert!(s.contains("hierarchy"));
        assert!(s.contains("names("));
    }

    #[test]
    fn test_dg_comment_display() {
        let mut props = HashMap::new();
        props.insert("key".into(), PropertyValue::Value("val".into()));
        let dg = DgComment {
            tx: Some("data group".into()),
            common_properties: props,
        };
        let s = format!("{dg}");
        assert!(s.contains("data group"));
        assert!(s.contains("props=1"));
    }

    #[test]
    fn test_md_comment_display_all_variants() {
        // Fh
        let fh = MdComment::Fh(FhComment {
            tx: Some("fh_text".into()),
            ..Default::default()
        });
        assert!(format!("{fh}").contains("fh_text"));

        // Cn
        let cn = MdComment::Cn(CnComment {
            tx: Some("cn_text".into()),
            ..Default::default()
        });
        assert!(format!("{cn}").contains("cn_text"));

        // Cg
        let cg = MdComment::Cg(CgComment {
            tx: Some("cg_text".into()),
            ..Default::default()
        });
        assert!(format!("{cg}").contains("cg_text"));

        // Cc
        let cc = MdComment::Cc(CcComment {
            tx: Some("cc_text".into()),
            ..Default::default()
        });
        assert!(format!("{cc}").contains("cc_text"));

        // Si
        let si = MdComment::Si(SiComment {
            tx: Some("si_text".into()),
            ..Default::default()
        });
        assert!(format!("{si}").contains("si_text"));

        // Ev
        let ev = MdComment::Ev(EvComment {
            tx: Some("ev_text".into()),
            ..Default::default()
        });
        assert!(format!("{ev}").contains("ev_text"));

        // At
        let at = MdComment::At(AtComment {
            tx: Some("at_text".into()),
            ..Default::default()
        });
        assert!(format!("{at}").contains("at_text"));

        // Ch
        let ch = MdComment::Ch(ChComment {
            tx: Some("ch_text".into()),
            ..Default::default()
        });
        assert!(format!("{ch}").contains("ch_text"));

        // Dg
        let dg = MdComment::Dg(DgComment {
            tx: Some("dg_text".into()),
            ..Default::default()
        });
        assert!(format!("{dg}").contains("dg_text"));
    }

    #[test]
    fn test_md_comment_get_tx_all_variants() {
        // Cn
        let cn = MdComment::Cn(CnComment {
            tx: Some("cn_tx".into()),
            ..Default::default()
        });
        assert_eq!(cn.get_tx(), Some("cn_tx"));

        // Cg
        let cg = MdComment::Cg(CgComment {
            tx: Some("cg_tx".into()),
            ..Default::default()
        });
        assert_eq!(cg.get_tx(), Some("cg_tx"));

        // Cc
        let cc = MdComment::Cc(CcComment {
            tx: Some("cc_tx".into()),
            ..Default::default()
        });
        assert_eq!(cc.get_tx(), Some("cc_tx"));

        // Si
        let si = MdComment::Si(SiComment {
            tx: Some("si_tx".into()),
            ..Default::default()
        });
        assert_eq!(si.get_tx(), Some("si_tx"));

        // At
        let at = MdComment::At(AtComment {
            tx: Some("at_tx".into()),
            ..Default::default()
        });
        assert_eq!(at.get_tx(), Some("at_tx"));

        // Ch
        let ch = MdComment::Ch(ChComment {
            tx: Some("ch_tx".into()),
            ..Default::default()
        });
        assert_eq!(ch.get_tx(), Some("ch_tx"));

        // Dg
        let dg = MdComment::Dg(DgComment {
            tx: Some("dg_tx".into()),
            ..Default::default()
        });
        assert_eq!(dg.get_tx(), Some("dg_tx"));
    }

    #[test]
    fn test_extract_property_nested() {
        let mut props = CommonProperties::new();
        let mut tree = HashMap::new();
        tree.insert("SequenceIndex".into(), PropertyValue::Value("3".into()));
        props.insert("Recorder".into(), PropertyValue::Tree(tree));
        assert_eq!(
            HdComment::extract_prop(&props, "Recorder.SequenceIndex"),
            Some("3")
        );
        assert_eq!(HdComment::extract_prop(&props, "Recorder.Missing"), None);
        assert_eq!(HdComment::extract_prop(&props, "Other"), None);
    }

    #[test]
    fn test_extract_property_flat() {
        let mut props = CommonProperties::new();
        props.insert(
            "Recorder.SequenceIndex".into(),
            PropertyValue::Value("7".into()),
        );
        assert_eq!(
            HdComment::extract_prop(&props, "Recorder.SequenceIndex"),
            Some("7")
        );
    }

    #[test]
    fn test_hd_comment_accessor_methods() {
        let mut hd = HdComment::default();
        let mut tree = HashMap::new();
        tree.insert("SequenceIndex".into(), PropertyValue::Value("2".into()));
        tree.insert("FileLast".into(), PropertyValue::Value("true".into()));
        tree.insert("UUID".into(), PropertyValue::Value("abc-123".into()));
        tree.insert("FileIndex".into(), PropertyValue::Value("5".into()));
        hd.common_properties
            .insert("Recorder".into(), PropertyValue::Tree(tree));
        hd.common_properties.insert(
            "MeasurementUUID".into(),
            PropertyValue::Value("meas-xyz".into()),
        );

        assert_eq!(hd.recorder_sequence_index(), Some(2));
        assert_eq!(hd.recorder_file_last(), Some(true));
        assert_eq!(hd.recorder_uuid(), Some("abc-123"));
        assert_eq!(hd.recorder_file_index(), Some(5));
        assert_eq!(hd.measurement_uuid(), Some("meas-xyz"));
    }

    #[test]
    fn test_hd_comment_missing_properties() {
        let hd = HdComment::default();
        assert_eq!(hd.recorder_sequence_index(), None);
        assert_eq!(hd.recorder_file_last(), None);
        assert_eq!(hd.recorder_uuid(), None);
        assert_eq!(hd.recorder_file_index(), None);
        assert_eq!(hd.measurement_uuid(), None);
    }
}
