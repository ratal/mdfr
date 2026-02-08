//! Composition blocks (DS, CL, CV, CU) and composition parsing for MDF4
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

/// contains composition blocks (CN or CA)
/// can optionally point to another composition
#[derive(Debug, Clone)]
#[repr(C)]
pub struct Composition {
    pub block: Compo,
    pub compo: Option<Box<Composition>>,
}

/// enum allowing to nest CA or CN blocks for a composition
#[derive(Debug, Clone)]
#[repr(C)]
pub enum Compo {
    CA(Box<Ca4Block>),
    CN(Box<Cn4>),
    CL(Box<Cl4Block>),
    CV(Box<Cv4Block>),
    CU(Box<Cu4Block>),
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
