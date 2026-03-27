//! Composition blocks (DS, CL, CV, CU) and composition parsing for MDF4 — spec sections 6.24-6.27, Tables 67-70
//!
//! Compositions describe how a channel's data is structured internally:
//! - DS (Dynamic Size, Table 67): variable-length fields in a VLSD blob
//! - CL (Channel List, Table 68): named fields in a VLSD blob
//! - CV (Column Variable-length, Table 69): variable-length column in fixed records
//! - CU (Column Unordered, Table 70): unordered variable-length column
//! - CA (Channel Array): N-dimensional arrays (handled in ca_block.rs)
//! - CN (nested channel): structure composition via CN→CN chain
use anyhow::{Context, Result, bail};
use binrw::{BinReaderExt, binrw};
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::fs::File;

use super::block_header::parse_block_short;
use super::ca_block::{Ca4Block, parse_ca_block};
use super::cn_block::{Cn4, CnType, RecordLayout, parse_cn4};
use super::block_header::SharableBlocks;
use crate::data_holder::tensor_arrow::Order;
use crate::mdfinfo::sym_buf_reader::SymBufReader;

/// type alias for composition parse result
pub type CompositionParseResult = (Composition, i64, usize, (Vec<usize>, Order), usize, CnType);

/// Recursive composition tree. Each node holds a `Compo` block and optionally chains to another composition.
#[derive(Debug, Clone)]
#[repr(C)]
pub struct Composition {
    /// The composition block at this level (CA, CN, CL, CV, CU, or DS).
    pub block: Compo,
    /// Optional next composition in the chain (e.g. nested CA inside another CA).
    pub compo: Option<Box<Composition>>,
}

/// Composition block variant — one of CA, CN, CL, CV, CU, or DS
#[derive(Debug, Clone)]
#[repr(C)]
pub enum Compo {
    /// Channel Array block: N-dimensional array layout (spec section 6.18)
    CA(Box<Ca4Block>),
    /// Nested Channel block: structure composition via CN→CN chain
    CN(Box<Cn4>),
    /// Channel List: named fields packed into a VLSD blob (spec section 6.25)
    CL(Box<Cl4Block>),
    /// Column Variable-length: variable-length column in fixed-length records (spec section 6.26)
    CV(Box<Cv4Block>),
    /// Column Unordered: unordered variable-length column (spec section 6.27)
    CU(Box<Cu4Block>),
    /// Dynamic Size: variable-length fields in a VLSD blob (spec section 6.24)
    DS(Box<Ds4Block>),
}

impl Display for Compo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Compo::CA(ca) => write!(f, "CA({})", ca),
            Compo::CN(cn) => write!(f, "CN({})", cn),
            Compo::CL(cl) => write!(f, "CL({})", cl),
            Compo::CV(cv) => write!(f, "CV({})", cv),
            Compo::CU(cu) => write!(f, "CU({})", cu),
            Compo::DS(ds) => write!(f, "DS({})", ds),
        }
    }
}

impl Display for Composition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Composition: {}", self.block)?;
        if self.compo.is_some() {
            write!(f, " (nested)")?;
        }
        Ok(())
    }
}

/// DS4 Data Stream block struct
#[derive(Debug, PartialEq, Eq, Default, Clone)]
#[binrw]
#[br(little)]
#[repr(C)]
pub struct Ds4Block {
    //header
    // ##DS
    // ds_id: [u8; 4],
    /// reserved
    // reserved: [u8; 4],
    /// Length of block in bytes
    // pub ds_len: u64,
    /// # of links
    pub ds_links: u64,
    /// links
    #[br(count = ds_links)]
    pub links: Vec<i64>,
    /// data
    /// Minimum version of the reader to read the data
    pub ds_version: u16,
    /// DSBlock mode, 0 data stream, 1 data description
    pub ds_mode: u8,
    /// Reserved
    pub ds_reserved: [u8; 5],
}

/// Data stream mode constant - data layout described by CNBLOCK composition
pub const DS_MODE_DATA_STREAM: u8 = 0;
/// Data description mode constant - data layout described by external attachment (FIBEX, DBC, ARXML)
pub const DS_MODE_DATA_DESCRIPTION: u8 = 1;

#[allow(dead_code)]
impl Ds4Block {
    pub fn ds_cn_composition(&self) -> i64 {
        self.links.first().copied().unwrap_or(0)
    }
    pub fn ds_cn_alignment_start(&self) -> i64 {
        self.links.get(1).copied().unwrap_or(0)
    }
    pub fn ds_data(&self) -> i64 {
        self.links.get(2).copied().unwrap_or(0)
    }
    pub fn ds_md_comment(&self) -> i64 {
        self.links.get(3).copied().unwrap_or(0)
    }

    /// Returns true if this is data stream mode (ds_mode = 0).
    /// In data stream mode, the data layout is described by CNBLOCK composition structures.
    pub fn is_data_stream_mode(&self) -> bool {
        self.ds_mode == DS_MODE_DATA_STREAM
    }

    /// Returns true if this is data description mode (ds_mode = 1).
    /// In data description mode, the data layout is described by an external attachment
    /// file (e.g., FIBEX, DBC, ARXML) pointed to by ds_cn_composition.
    pub fn is_data_description_mode(&self) -> bool {
        self.ds_mode == DS_MODE_DATA_DESCRIPTION
    }

    /// Returns a string description of the data stream mode.
    pub fn get_mode_str(&self) -> &'static str {
        match self.ds_mode {
            DS_MODE_DATA_STREAM => "Data Stream Mode",
            DS_MODE_DATA_DESCRIPTION => "Data Description Mode",
            _ => "Unknown Mode",
        }
    }
}

impl Display for Ds4Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DS: mode={} ({}) version={} links={}",
            self.get_mode_str(),
            self.ds_mode,
            self.ds_version,
            self.ds_links
        )
    }
}

/// CL4 Channel List block struct
#[derive(Debug, PartialEq, Eq, Default, Clone)]
#[binrw]
#[br(little)]
#[repr(C)]
pub struct Cl4Block {
    //header
    // ##CL
    // cl_id: [u8; 4],
    /// reserved
    // reserved: [u8; 4],
    /// Length of block in bytes
    // pub cl_len: u64,
    /// # of links
    pub cl_links: u64,
    /// links
    /// link to CNBlock describing dynamic data
    pub cl_composition: i64,
    /// link to CNBlock for the alignment start with data stream mode
    pub cl_cn_size: i64,
    /// data
    /// Flags
    pub cl_flags: u16,
    /// Bytes alignment
    pub cl_alignment: u8,
    /// Bit Offset
    pub cl_bit_offset: u8,
    /// Byte Offset
    pub cl_byte_offset: u32,
}

impl Display for Cl4Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CL: flags=0x{:04X} alignment={} bit_offset={} byte_offset={}",
            self.cl_flags, self.cl_alignment, self.cl_bit_offset, self.cl_byte_offset
        )
    }
}

/// CV4 Channel Variant block struct
#[derive(Debug, PartialEq, Eq, Default, Clone)]
#[binrw]
#[br(little)]
#[repr(C)]
pub struct Cv4Block {
    //header
    // ##CV
    // cv_id: [u8; 4],
    /// reserved
    // reserved: [u8; 4],
    /// Length of block in bytes
    // pub cv_len: u64,
    /// # of links
    pub cv_n_links: u64,
    /// links
    /// link to CNBlock for discriminator channel
    pub cv_cn_discriminator: i64,
    /// list of option channel
    #[br(if(cv_n_links > 1), little, count = cv_n_links - 1)]
    pub cv_cn_option: Vec<i64>,
    /// data
    /// number of option channels
    pub cv_option_count: u32,
    /// reserved
    pub cv_reserved: [u8; 4],
    /// list of discriminator values for the options
    #[br(if(cv_option_count > 1), little, count = cv_option_count )]
    pub cv_option_val: Vec<u64>,
}

impl Display for Cv4Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CV: {} options", self.cv_option_count)
    }
}

/// CU4 Channel Union block struct
#[derive(Debug, PartialEq, Eq, Default, Clone)]
#[binrw]
#[br(little)]
#[repr(C)]
pub struct Cu4Block {
    //header
    // ##CU
    // cu_id: [u8; 4],
    /// reserved
    // reserved: [u8; 4],
    /// Length of block in bytes
    // pub cu_len: u64,
    /// # of links
    pub cu_n_links: u64,
    /// links
    /// list of member channel
    #[br(if(cu_n_links > 1), little, count = cu_n_links)]
    pub cu_cn_member: Vec<i64>,
    /// data
    /// number of member channels
    pub cu_member_count: u32,
    /// reserved
    pub cu_reserved: [u8; 4],
}

impl Display for Cu4Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CU: {} members", self.cu_member_count)
    }
}

/// Parse and re-key channel blocks for CU/CV compositions.
/// All member/option channels share the same byte offset in the record,
/// so they are re-keyed using negative block_position to avoid HashMap collisions.
fn parse_and_rekey_channels(
    rdr: &mut SymBufReader<&File>,
    targets: &[i64],
    position: &mut i64,
    sharable: &mut SharableBlocks,
    record_layout: RecordLayout,
    cg_cycle_count: u64,
) -> Result<(CnType, usize)> {
    let mut cns: CnType = HashMap::new();
    let mut n_cn: usize = 0;
    for target in targets {
        let (cnss, pos, n_cns, _first_rec_pos) = parse_cn4(
            rdr,
            *target,
            *position,
            sharable,
            record_layout,
            cg_cycle_count,
        )?;
        *position = pos;
        n_cn += n_cns;
        for (_rec_pos, cn_struct) in cnss {
            let unique_key = -(cn_struct.block_position as i32);
            cns.insert(unique_key, cn_struct);
        }
    }
    Ok((cns, n_cn))
}

/// parses composition linked blocks
/// CN (structures of composed channels )and CA (array of arrays) blocks can be nested or even CA and CN nested and mixed: this is not supported, very complicated
pub(super) fn parse_composition(
    rdr: &mut SymBufReader<&File>,
    target: i64,
    mut position: i64,
    sharable: &mut SharableBlocks,
    record_layout: RecordLayout,
    cg_cycle_count: u64,
) -> Result<CompositionParseResult> {
    let (mut block, block_header_short, pos) = parse_block_short(rdr, target, position)
        .context("Failed parsing composition header block")?;
    position = pos;
    let array_size: usize;
    let mut cns: CnType = HashMap::new();
    let mut n_cn: usize = 0;

    if block_header_short.hdr_id == "##CA".as_bytes() {
        // Channel Array
        let (block, mut shape, _snd, array_size) =
            parse_ca_block(&mut block, block_header_short, cg_cycle_count)
                .context("Failed parsing CA block")?;
        position = pos;
        let ca_composition: Option<Box<Composition>>;
        if block.ca_composition != 0 {
            let (ca, pos, _array_size, s, n_cns, cnss) = parse_composition(
                rdr,
                block.ca_composition,
                position,
                sharable,
                record_layout,
                cg_cycle_count,
            )
            .context("Failed parsing composition block from CA block")?;
            shape = s;
            position = pos;
            cns = cnss;
            n_cn += n_cns;
            ca_composition = Some(Box::new(ca));
        } else {
            ca_composition = None;
            cns = HashMap::new();
        }
        Ok((
            Composition {
                block: Compo::CA(Box::new(block)),
                compo: ca_composition,
            },
            position,
            array_size,
            shape,
            n_cn,
            cns,
        ))
    } else if block_header_short.hdr_id == "##DS".as_bytes() {
        // Data Stream
        let ds_block: Ds4Block = block.read_le().context("Failed parsing DS block")?;
        array_size = 1;
        let ds_pointer = ds_block.ds_cn_composition();
        let ds_composition: Option<Box<Composition>>;
        let mut shape = (Vec::<usize>::new(), Order::RowMajor);
        if ds_pointer != 0 {
            let (ds, pos, _array_size, s, n_cns, cnss) = parse_composition(
                rdr,
                ds_pointer,
                position,
                sharable,
                record_layout,
                cg_cycle_count,
            )
            .context("Failed parsing composition block from DS Block")?;
            shape = s;
            position = pos;
            cns = cnss;
            n_cn += n_cns;
            ds_composition = Some(Box::new(ds));
        } else {
            ds_composition = None;
            cns = HashMap::new();
        }
        Ok((
            Composition {
                block: Compo::DS(Box::new(ds_block)),
                compo: ds_composition,
            },
            position,
            array_size,
            shape,
            n_cn,
            cns,
        ))
    } else if block_header_short.hdr_id == "##CL".as_bytes() {
        // Channel List
        let cl_block: Cl4Block = block.read_le().context("Failed parsing CL block")?;
        let cl_composition: Option<Box<Composition>>;
        let mut shape = (Vec::<usize>::new(), Order::RowMajor);
        array_size = 0;
        // Note: cl_cn_size points to the size channel (parsed elsewhere in the CG)
        // Parse the composition (element type)
        if cl_block.cl_composition != 0 {
            let (ds, pos, _array_size, s, n_cns, cnss) = parse_composition(
                rdr,
                cl_block.cl_composition,
                position,
                sharable,
                record_layout,
                cg_cycle_count,
            )
            .context("Failed parsing composition block from CL Block")?;
            shape = s;
            position = pos;
            cns = cnss;
            n_cn += n_cns;
            cl_composition = Some(Box::new(ds));
        } else {
            cl_composition = None;
            cns = HashMap::new();
        }
        Ok((
            Composition {
                block: Compo::CL(Box::new(cl_block)),
                compo: cl_composition,
            },
            position,
            array_size,
            shape,
            n_cn,
            cns,
        ))
    } else if block_header_short.hdr_id == "##CV".as_bytes() {
        // Channel Variant
        let cv_block: Cv4Block = block.read_le().context("Failed parsing CV block")?;
        let cv_composition: Option<Box<Composition>> = None;
        let shape = (Vec::<usize>::new(), Order::RowMajor);
        array_size = 0;
        let (rekeyed_cns, rekeyed_n_cn) = parse_and_rekey_channels(
            rdr,
            &cv_block.cv_cn_option,
            &mut position,
            sharable,
            record_layout,
            cg_cycle_count,
        )?;
        n_cn += rekeyed_n_cn;
        cns.extend(rekeyed_cns);
        Ok((
            Composition {
                block: Compo::CV(Box::new(cv_block)),
                compo: cv_composition,
            },
            position,
            array_size,
            shape,
            n_cn,
            cns,
        ))
    } else if block_header_short.hdr_id == "##CU".as_bytes() {
        // Channel Union
        let cu_block: Cu4Block = block.read_le().context("Failed parsing CU block")?;
        let cu_composition: Option<Box<Composition>> = None;
        let shape = (Vec::<usize>::new(), Order::RowMajor);
        array_size = 0;
        let (rekeyed_cns, rekeyed_n_cn) = parse_and_rekey_channels(
            rdr,
            &cu_block.cu_cn_member,
            &mut position,
            sharable,
            record_layout,
            cg_cycle_count,
        )?;
        n_cn += rekeyed_n_cn;
        cns.extend(rekeyed_cns);
        Ok((
            Composition {
                block: Compo::CU(Box::new(cu_block)),
                compo: cu_composition,
            },
            position,
            array_size,
            shape,
            n_cn,
            cns,
        ))
    } else if block_header_short.hdr_id == "##CN".as_bytes() {
        // Channel structure
        array_size = 1;
        let (cnss, pos, n_cns, first_rec_pos) = parse_cn4(
            rdr,
            target,
            position,
            sharable,
            record_layout,
            cg_cycle_count,
        )?;
        position = pos;
        n_cn += n_cns;
        cns = cnss;
        let cn_composition: Option<Box<Composition>>;
        let cn_struct: Cn4 = if let Some(cn) = cns.get(&first_rec_pos) {
            cn.clone()
        } else {
            Cn4::default()
        };
        let shape: (Vec<usize>, Order);
        if cn_struct.block.cn_composition != 0 {
            let (cn, pos, _array_size, s, n_cns, cnss) = parse_composition(
                rdr,
                cn_struct.block.cn_composition,
                position,
                sharable,
                record_layout,
                cg_cycle_count,
            )?;
            shape = s;
            position = pos;
            n_cn += n_cns;
            cns.extend(cnss);
            cn_composition = Some(Box::new(cn));
        } else {
            cn_composition = None;
            shape = (vec![1], Order::RowMajor);
        }
        Ok((
            Composition {
                block: Compo::CN(Box::new(cn_struct)),
                compo: cn_composition,
            },
            position,
            array_size,
            shape,
            n_cn,
            cns,
        ))
    } else {
        bail!("Unknown composition block type")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Ds4Block tests ──

    #[test]
    fn test_ds_getters() {
        // Empty links
        let ds = Ds4Block::default();
        assert_eq!(ds.ds_cn_composition(), 0);
        assert_eq!(ds.ds_cn_alignment_start(), 0);
        assert_eq!(ds.ds_data(), 0);
        assert_eq!(ds.ds_md_comment(), 0);

        // Populated links
        let ds = Ds4Block {
            ds_links: 4,
            links: vec![100, 200, 300, 400],
            ds_version: 430,
            ds_mode: 0,
            ds_reserved: [0; 5],
        };
        assert_eq!(ds.ds_cn_composition(), 100);
        assert_eq!(ds.ds_cn_alignment_start(), 200);
        assert_eq!(ds.ds_data(), 300);
        assert_eq!(ds.ds_md_comment(), 400);
    }

    #[test]
    fn test_ds_mode_helpers() {
        let ds = Ds4Block {
            ds_mode: DS_MODE_DATA_STREAM,
            ..Default::default()
        };
        assert!(ds.is_data_stream_mode());
        assert!(!ds.is_data_description_mode());
        assert_eq!(ds.get_mode_str(), "Data Stream Mode");

        let ds = Ds4Block {
            ds_mode: DS_MODE_DATA_DESCRIPTION,
            ..Default::default()
        };
        assert!(!ds.is_data_stream_mode());
        assert!(ds.is_data_description_mode());
        assert_eq!(ds.get_mode_str(), "Data Description Mode");

        let ds = Ds4Block {
            ds_mode: 99,
            ..Default::default()
        };
        assert_eq!(ds.get_mode_str(), "Unknown Mode");
    }

    #[test]
    fn test_ds_display() {
        let ds = Ds4Block {
            ds_links: 4,
            links: vec![100, 200, 300, 400],
            ds_version: 430,
            ds_mode: 0,
            ds_reserved: [0; 5],
        };
        let display = format!("{ds}");
        assert!(display.contains("DS:"));
        assert!(display.contains("Data Stream Mode"));
        assert!(display.contains("version=430"));
        assert!(display.contains("links=4"));
    }

    // ── Cl4Block tests ──

    #[test]
    fn test_cl_display() {
        let cl = Cl4Block {
            cl_links: 2,
            cl_composition: 0,
            cl_cn_size: 0,
            cl_flags: 0x0003,
            cl_alignment: 4,
            cl_bit_offset: 2,
            cl_byte_offset: 16,
        };
        let display = format!("{cl}");
        assert!(display.contains("CL:"));
        assert!(display.contains("flags=0x0003"));
        assert!(display.contains("alignment=4"));
        assert!(display.contains("bit_offset=2"));
        assert!(display.contains("byte_offset=16"));
    }

    // ── Cv4Block tests ──

    #[test]
    fn test_cv_display() {
        let cv = Cv4Block {
            cv_option_count: 3,
            ..Default::default()
        };
        let display = format!("{cv}");
        assert!(display.contains("CV:"));
        assert!(display.contains("3 options"));
    }

    // ── Cu4Block tests ──

    #[test]
    fn test_cu_display() {
        let cu = Cu4Block {
            cu_member_count: 5,
            ..Default::default()
        };
        let display = format!("{cu}");
        assert!(display.contains("CU:"));
        assert!(display.contains("5 members"));
    }

    // ── Compo & Composition Display ──

    #[test]
    fn test_compo_display() {
        let ds = Compo::DS(Box::<Ds4Block>::default());
        let display = format!("{ds}");
        assert!(display.contains("DS("));

        let cl = Compo::CL(Box::<Cl4Block>::default());
        assert!(format!("{cl}").contains("CL("));

        let cv = Compo::CV(Box::<Cv4Block>::default());
        assert!(format!("{cv}").contains("CV("));

        let cu = Compo::CU(Box::<Cu4Block>::default());
        assert!(format!("{cu}").contains("CU("));
    }

    #[test]
    fn test_composition_display() {
        let comp = Composition {
            block: Compo::DS(Box::<Ds4Block>::default()),
            compo: None,
        };
        let display = format!("{comp}");
        assert!(display.contains("Composition:"));
        assert!(!display.contains("nested"));

        let nested = Composition {
            block: Compo::DS(Box::<Ds4Block>::default()),
            compo: Some(Box::new(Composition {
                block: Compo::CL(Box::<Cl4Block>::default()),
                compo: None,
            })),
        };
        let display = format!("{nested}");
        assert!(display.contains("(nested)"));
    }
}
