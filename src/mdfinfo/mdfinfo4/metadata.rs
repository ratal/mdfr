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

/// Extract text content of a child element by tag name
fn extract_text<'a>(node: roxmltree::Node<'a, 'a>, tag_name: &str) -> Option<String> {
    node.children()
        .find(|n| n.is_element() && n.has_tag_name(tag_name))
        .and_then(|n| n.text())
        .map(|s| s.to_string())
}

/// Parse `<names>`, `<path>`, or `<bus>` block into MdNames (default language only)
fn parse_names(node: roxmltree::Node, tag_name: &str) -> MdNames {
    let mut names = MdNames::default();
    if let Some(names_node) = node.children().find(|n| n.is_element() && n.has_tag_name(tag_name))
    {
        names.name = extract_text(names_node, "name");
        names.display = extract_text(names_node, "display");
        names.vendor = extract_text(names_node, "vendor");
        names.description = extract_text(names_node, "description");
    }
    names
}

/// Recursively parse `<common_properties>` into CommonProperties
fn parse_common_properties(node: roxmltree::Node) -> CommonProperties {
    let mut props = CommonProperties::new();
    if let Some(cp_node) = node
        .children()
        .find(|n| n.is_element() && n.has_tag_name("common_properties"))
    {
        parse_properties_children(cp_node, &mut props);
    }
    props
}

/// Parse children of a common_properties or tree node
fn parse_properties_children(node: roxmltree::Node, props: &mut HashMap<String, PropertyValue>) {
    for child in node.children().filter(|n| n.is_element()) {
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
                    for li in child.children().filter(|n| n.is_element() && n.has_tag_name("li")) {
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
                        .filter_map(|n| n.text().map(|s| s.to_string()))
                        .collect();
                    props.insert(name.to_string(), PropertyValue::EList(items));
                }
            }
            _ => {}
        }
    }
}

/// Parse raster element: `<raster><min>v</min><max>v</max><avg>v</avg></raster>`
fn parse_raster(node: roxmltree::Node) -> Option<(Option<f64>, Option<f64>, Option<f64>)> {
    node.children()
        .find(|n| n.is_element() && n.has_tag_name("raster"))
        .map(|raster_node| {
            let min = extract_text(raster_node, "min").and_then(|s| s.parse().ok());
            let max = extract_text(raster_node, "max").and_then(|s| s.parse().ok());
            let avg = extract_text(raster_node, "avg").and_then(|s| s.parse().ok());
            (min, max, avg)
        })
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
            MetaDataBlockType::MdParsed => {
                Ok(self.md_comment.as_ref().and_then(|mc| mc.get_tx()).map(|s| s.to_string()))
            }
            MetaDataBlockType::MdBlock => {
                // extract TX tag from xml
                let comment: String = self
                    .get_data_string()
                    .context("failed getting data string to extract TX tag")?
                    .trim_end_matches(['\n', '\r', ' '])
                    .into();
                match roxmltree::Document::parse(&comment) {
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
        if self.raw_data.is_empty() {
            None
        } else {
            Some(&self.raw_data)
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
    /// Helper: get trimmed XML string from raw_data
    fn get_xml_string(&self) -> Result<String> {
        let s = self
            .get_data_string()?
            .trim_end_matches(['\n', '\r', ' '])
            .to_string();
        Ok(s)
    }
    /// Parse HD block MD comment (hd_comment.xsd)
    pub fn parse_hd_comment(&mut self) -> Result<()> {
        let mut hd = HdComment::default();
        let xml = self.get_xml_string()?;
        match roxmltree::Document::parse(&xml) {
            Ok(doc) => {
                let root = doc.root_element();
                hd.tx = extract_text(root, "TX");
                hd.time_source = extract_text(root, "time_source");
                if let Some(constants_node) = root
                    .children()
                    .find(|n| n.is_element() && n.has_tag_name("constants"))
                {
                    for c in constants_node
                        .children()
                        .filter(|n| n.is_element() && n.has_tag_name("const"))
                    {
                        if let (Some(name), Some(text)) = (c.attribute("name"), c.text()) {
                            hd.constants.insert(name.to_string(), text.to_string());
                        }
                    }
                }
                hd.common_properties = parse_common_properties(root);
            }
            Err(e) => {
                log::warn!("Could not parse HD MD comment : \n{xml}\n{e}");
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
        let xml = self.get_xml_string()?;
        match roxmltree::Document::parse(&xml) {
            Ok(doc) => {
                let root = doc.root_element();
                fh.tx = extract_text(root, "TX");
                fh.tool_id = extract_text(root, "tool_id");
                fh.tool_vendor = extract_text(root, "tool_vendor");
                fh.tool_version = extract_text(root, "tool_version");
                fh.user_name = extract_text(root, "user_name");
                fh.common_properties = parse_common_properties(root);
            }
            Err(e) => log::warn!("Could not parse FH comment : \n{xml}\n{e}"),
        }
        self.md_comment = Some(MdComment::Fh(fh));
        self.block_type = MetaDataBlockType::MdParsed;
        Ok(())
    }
    /// Parse CN block MD comment (cn_comment.xsd)
    fn parse_cn_comment(&mut self) -> Result<()> {
        let mut cn = CnComment::default();
        let xml = self.get_xml_string()?;
        match roxmltree::Document::parse(&xml) {
            Ok(doc) => {
                let root = doc.root_element();
                cn.tx = extract_text(root, "TX");
                cn.names = parse_names(root, "names");
                cn.linker_name = extract_text(root, "linker_name");
                cn.linker_address = extract_text(root, "linker_address");
                cn.axis_monotony = extract_text(root, "axis_monotony");
                cn.raster = parse_raster(root);
                cn.formula = extract_text(root, "formula");
                cn.address = extract_text(root, "address");
                cn.common_properties = parse_common_properties(root);
            }
            Err(e) => log::warn!("Could not parse CN comment : \n{xml}\n{e}"),
        }
        self.md_comment = Some(MdComment::Cn(cn));
        self.block_type = MetaDataBlockType::MdParsed;
        Ok(())
    }
    /// Parse CG block MD comment (cg_comment.xsd)
    fn parse_cg_comment(&mut self) -> Result<()> {
        let mut cg = CgComment::default();
        let xml = self.get_xml_string()?;
        match roxmltree::Document::parse(&xml) {
            Ok(doc) => {
                let root = doc.root_element();
                cg.tx = extract_text(root, "TX");
                cg.names = parse_names(root, "names");
                cg.common_properties = parse_common_properties(root);
            }
            Err(e) => log::warn!("Could not parse CG comment : \n{xml}\n{e}"),
        }
        self.md_comment = Some(MdComment::Cg(cg));
        self.block_type = MetaDataBlockType::MdParsed;
        Ok(())
    }
    /// Parse CC block MD comment (cc_comment.xsd)
    fn parse_cc_comment(&mut self) -> Result<()> {
        let mut cc = CcComment::default();
        let xml = self.get_xml_string()?;
        match roxmltree::Document::parse(&xml) {
            Ok(doc) => {
                let root = doc.root_element();
                cc.tx = extract_text(root, "TX");
                cc.names = parse_names(root, "names");
                cc.formula = extract_text(root, "formula");
                cc.common_properties = parse_common_properties(root);
            }
            Err(e) => log::warn!("Could not parse CC comment : \n{xml}\n{e}"),
        }
        self.md_comment = Some(MdComment::Cc(cc));
        self.block_type = MetaDataBlockType::MdParsed;
        Ok(())
    }
    /// Parse SI block MD comment (si_comment.xsd)
    fn parse_si_comment(&mut self) -> Result<()> {
        let mut si = SiComment::default();
        let xml = self.get_xml_string()?;
        match roxmltree::Document::parse(&xml) {
            Ok(doc) => {
                let root = doc.root_element();
                si.tx = extract_text(root, "TX");
                si.names = parse_names(root, "names");
                si.path = parse_names(root, "path");
                si.bus = parse_names(root, "bus");
                si.protocol = extract_text(root, "protocol");
                si.common_properties = parse_common_properties(root);
            }
            Err(e) => log::warn!("Could not parse SI comment : \n{xml}\n{e}"),
        }
        self.md_comment = Some(MdComment::Si(si));
        self.block_type = MetaDataBlockType::MdParsed;
        Ok(())
    }
    /// Parse EV block MD comment (ev_comment.xsd)
    fn parse_ev_comment(&mut self) -> Result<()> {
        let mut ev = EvComment::default();
        let xml = self.get_xml_string()?;
        match roxmltree::Document::parse(&xml) {
            Ok(doc) => {
                let root = doc.root_element();
                ev.tx = extract_text(root, "TX");
                ev.pre_trigger_interval = extract_text(root, "pre_trigger_interval")
                    .and_then(|s| s.parse().ok());
                ev.post_trigger_interval = extract_text(root, "post_trigger_interval")
                    .and_then(|s| s.parse().ok());
                ev.formula = extract_text(root, "formula");
                ev.timeout = extract_text(root, "timeout").and_then(|s| s.parse().ok());
                ev.common_properties = parse_common_properties(root);
            }
            Err(e) => log::warn!("Could not parse EV comment : \n{xml}\n{e}"),
        }
        self.md_comment = Some(MdComment::Ev(ev));
        self.block_type = MetaDataBlockType::MdParsed;
        Ok(())
    }
    /// Parse AT block MD comment (at_comment.xsd)
    fn parse_at_comment(&mut self) -> Result<()> {
        let mut at = AtComment::default();
        let xml = self.get_xml_string()?;
        match roxmltree::Document::parse(&xml) {
            Ok(doc) => {
                let root = doc.root_element();
                at.tx = extract_text(root, "TX");
                at.common_properties = parse_common_properties(root);
            }
            Err(e) => log::warn!("Could not parse AT comment : \n{xml}\n{e}"),
        }
        self.md_comment = Some(MdComment::At(at));
        self.block_type = MetaDataBlockType::MdParsed;
        Ok(())
    }
    /// Parse CH block MD comment (ch_comment.xsd)
    fn parse_ch_comment(&mut self) -> Result<()> {
        let mut ch = ChComment::default();
        let xml = self.get_xml_string()?;
        match roxmltree::Document::parse(&xml) {
            Ok(doc) => {
                let root = doc.root_element();
                ch.tx = extract_text(root, "TX");
                ch.names = parse_names(root, "names");
                ch.common_properties = parse_common_properties(root);
            }
            Err(e) => log::warn!("Could not parse CH comment : \n{xml}\n{e}"),
        }
        self.md_comment = Some(MdComment::Ch(ch));
        self.block_type = MetaDataBlockType::MdParsed;
        Ok(())
    }
    /// Parse DG block MD comment (dg_comment.xsd)
    fn parse_dg_comment(&mut self) -> Result<()> {
        let mut dg = DgComment::default();
        let xml = self.get_xml_string()?;
        match roxmltree::Document::parse(&xml) {
            Ok(doc) => {
                let root = doc.root_element();
                dg.tx = extract_text(root, "TX");
                dg.common_properties = parse_common_properties(root);
            }
            Err(e) => log::warn!("Could not parse DG comment : \n{xml}\n{e}"),
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
