//! Channel Array block (CABLOCK) for MDF4
use anyhow::{Context, Error, Result};
use binrw::{BinReaderExt, binrw};
use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::VecDeque;
use std::fmt::{self, Display};
use std::io::Cursor;

use super::block_header::Blockheader4Short;
use crate::data_holder::tensor_arrow::Order;

/// type alias for Ca4Block parse result
pub type CaBlockParseResult = (Ca4Block, (Vec<usize>, Order), usize, usize);

/// Ca4 Channel Array block struct
#[derive(Debug, PartialEq, Clone)]
#[repr(C)]
pub struct Ca4Block {
    // header
    /// ##CA
    pub ca_id: [u8; 4],
    /// reserved
    reserved: [u8; 4],
    /// Length of block in bytes
    pub ca_len: u64,
    /// # of links
    ca_links: u64,
    // links
    /// [] Array of composed elements: Pointer to a CNBLOCK for array of structures, or to a CABLOCK for array of arrays (can be NIL). If a CABLOCK is referenced, it must use the "CN template" storage type (ca_storage = 0).
    pub ca_composition: i64,
    /// [Π N(d) or empty] Only present for storage type "DG template". List of links to data blocks (DTBLOCK/DLBLOCK) for each element in case of "DG template" storage (ca_storage = 2). A link in this list may only be NIL if the cycle count of the respective element is 0: ca_data\[k\] = NIL => ca_cycle_count\[k\] = 0 The links are stored line-oriented, i.e. element k uses ca_data\[k\] (see explanation below). The size of the list must be equal to Π N(d), i.e. to the product of the number of elements per dimension N(d) over all dimensions D. Note: link ca_data\[0\] must be equal to dg_data link of the parent DGBLOCK.
    pub ca_data: Option<Vec<i64>>,
    /// [Dx3 or empty] Only present if "dynamic size" flag (bit 0) is set. References to channels for size signal of each dimension (can be NIL). Each reference is a link triple with pointer to parent DGBLOCK, parent CGBLOCK and CNBLOCK for the channel (either all three links are assigned or NIL).
    ca_dynamic_size: Option<Vec<i64>>,
    /// [Dx3 or empty] Only present if "input quantity" flag (bit 1) is set. Reference to channels for input quantity signal for each dimension (can be NIL). Each reference is a link triple with pointer to parent DGBLOCK, parent CGBLOCK and CNBLOCK for the channel (either all three links are assigned or NIL).
    ca_input_quantity: Option<Vec<i64>>,
    /// [3 or empty] Only present if "output quantity" flag (bit 2) is set. Reference to channel for output quantity (can be NIL). The reference is a link triple with pointer to parent DGBLOCK, parent CGBLOCK and CNBLOCK for the channel (either all three links are assigned or NIL).
    ca_output_quantity: Option<Vec<i64>>,
    /// [3 or empty] Only present if "comparison quantity" flag (bit 3) is set. Reference to channel for comparison quantity (can be NIL). The reference is a link triple with pointer to parent DGBLOCK, parent CGBLOCK and CNBLOCK for the channel (either all three links are assigned or NIL).
    ca_comparison_quantity: Option<Vec<i64>>,
    /// [D or empty] Only present if "axis" flag (bit 4) is set. Pointer to a conversion rule (CCBLOCK) for the scaling axis of each dimension. If a link NIL a 1:1 conversion must be used for this axis.
    ca_cc_axis_conversion: Option<Vec<i64>>,
    /// [Dx3 or empty] Only present if "axis" flag (bit 4) is set and "fixed axes flag" (bit 5) is not set. References to channels for scaling axis of respective dimension (can be NIL). Each reference is a link triple with pointer to parent DGBLOCK, parent CGBLOCK and CNBLOCK for the channel (either all three links are assigned or NIL).
    ca_axis: Option<Vec<i64>>,
    //members
    /// Array type (defines semantic of the array) see CA_T_xxx
    pub ca_type: u8,
    /// Storage type (defines how the element values are stored) see CA_S_xxx
    pub ca_storage: u8,
    /// Number of dimensions D > 0 For array type "axis", D must be 1.
    pub ca_ndim: u16,
    /// Flags The value contains the following bit flags (Bit 0 = LSB): see CA_F_xxx
    pub ca_flags: u32,
    /// Base factor for calculation of Byte offsets for "CN template" storage type.
    pub ca_byte_offset_base: i32,
    /// Base factor for calculation of invalidation bit positions for CN template storage type.
    pub ca_inval_bit_pos_base: u32,
    pub ca_dim_size: Vec<u64>,
    pub ca_axis_value: Option<Vec<f64>>,
    pub ca_cycle_count: Option<Vec<u64>>,
}

impl Default for Ca4Block {
    fn default() -> Self {
        Self {
            ca_id: [35, 35, 67, 65], // ##CA
            reserved: [0u8; 4],
            ca_len: 48,
            ca_links: 1,
            ca_composition: 0,
            ca_data: None,
            ca_dynamic_size: None,
            ca_input_quantity: None,
            ca_output_quantity: None,
            ca_comparison_quantity: None,
            ca_cc_axis_conversion: None,
            ca_axis: None,
            ca_type: 0,    // Array
            ca_storage: 0, // CN template
            ca_ndim: 1,
            ca_flags: 0,
            ca_byte_offset_base: 0,   // first
            ca_inval_bit_pos_base: 0, // present in DIBlock
            ca_dim_size: vec![],
            ca_axis_value: None,
            ca_cycle_count: None,
        }
    }
}

impl Ca4Block {
    /// Returns a string representation of the array type (ca_type)
    pub fn get_ca_type_str(&self) -> &'static str {
        match self.ca_type {
            0 => "Array",
            1 => "ScalingAxis",
            2 => "LookUp",
            3 => "IntervalAxis",
            4 => "ClassificationResult",
            _ => "Unknown",
        }
    }
    /// Returns a string representation of the storage type (ca_storage)
    pub fn get_storage_str(&self) -> &'static str {
        match self.ca_storage {
            0 => "CN Template",
            1 => "CG Template",
            2 => "DG Template",
            _ => "Unknown",
        }
    }
}

impl Display for Ca4Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CA: type={} ({}) storage={} ({}) dims={:?}",
            self.get_ca_type_str(),
            self.ca_type,
            self.get_storage_str(),
            self.ca_storage,
            self.ca_dim_size
        )
    }
}

/// Channel Array block structure, only members section, links section structure complex
#[derive(Debug, Clone)]
#[binrw]
#[br(little)]
#[repr(C)]
pub struct Ca4BlockMembers {
    /// Array type (defines semantic of the array) see CA_T_xxx
    ca_type: u8,
    /// Storage type (defines how the element values are stored) see CA_S_xxx
    ca_storage: u8,
    /// Number of dimensions D > 0 For array type "axis", D must be 1.
    pub ca_ndim: u16,
    /// Flags The value contains the following bit flags (Bit 0 = LSB): see CA_F_xxx
    ca_flags: u32,
    /// Base factor for calculation of Byte offsets for "CN template" storage type.
    ca_byte_offset_base: i32,
    /// Base factor for calculation of invalidation bit positions for CN template storage type.
    ca_inval_bit_pos_base: u32,
    #[br(if(ca_ndim > 0), little, count = ca_ndim)]
    pub ca_dim_size: Vec<u64>,
}

impl Default for Ca4BlockMembers {
    fn default() -> Self {
        Self {
            ca_type: 0,
            ca_storage: 0,
            ca_ndim: 1,
            ca_flags: 0,
            ca_byte_offset_base: 0,
            ca_inval_bit_pos_base: 0,
            ca_dim_size: vec![],
        }
    }
}

impl Display for Ca4BlockMembers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CA: {}D array dims={:?}", self.ca_ndim, self.ca_dim_size)
    }
}

/// Channel Array block parser
pub(super) fn parse_ca_block(
    ca_block: &mut Cursor<Vec<u8>>,
    block_header: Blockheader4Short,
    cg_cycle_count: u64,
) -> Result<CaBlockParseResult, Error> {
    // reads links count
    let ca_links: u64 = ca_block
        .read_le()
        .context("Could not read links count in ca block")?;
    //Reads members first
    ca_block.set_position(8 + ca_links * 8); // change buffer position after links section
    let ca_members: Ca4BlockMembers = ca_block
        .read_le()
        .context("Could not read buffer into CaBlockMembers struct")?;
    let mut snd: usize;
    let mut pnd: usize;
    // converts  ca_dim_size from u64 to usize
    let shape_dim_usize: Vec<usize> = ca_members.ca_dim_size.iter().map(|&d| d as usize).collect();
    if shape_dim_usize.len() == 1 {
        snd = shape_dim_usize[0];
        pnd = shape_dim_usize[0];
    } else {
        snd = 0;
        pnd = 1;
        let sizes = shape_dim_usize.clone();
        for x in sizes.into_iter() {
            snd += x;
            pnd *= x;
        }
    }
    let mut shape_dim: VecDeque<usize> = VecDeque::from(shape_dim_usize);
    shape_dim.push_front(cg_cycle_count as usize);

    let shape: (Vec<usize>, Order) = if (ca_members.ca_flags >> 6 & 1) != 0 {
        (shape_dim.into(), Order::ColumnMajor)
    } else {
        (shape_dim.into(), Order::RowMajor)
    };

    let mut val = vec![0.0f64; snd];
    let ca_axis_value: Option<Vec<f64>> = if (ca_members.ca_flags & 0b100000) > 0 {
        ca_block
            .read_f64_into::<LittleEndian>(&mut val)
            .context("Could not read ca_axis_value")?;
        Some(val)
    } else {
        None
    };

    let mut val = vec![0u64; pnd];
    let ca_cycle_count: Option<Vec<u64>> = if ca_members.ca_storage >= 1 {
        ca_block
            .read_u64_into::<LittleEndian>(&mut val)
            .context("Could not read ca_cycle_count")?;
        Some(val)
    } else {
        None
    };

    // Reads links
    ca_block.set_position(8); // change buffer position to beginning of links section

    let ca_composition: i64 = ca_block
        .read_i64::<LittleEndian>()
        .context("Could not read ca_composition")?;

    let mut val = vec![0i64; pnd];
    let ca_data: Option<Vec<i64>> = if ca_members.ca_storage == 2 {
        ca_block
            .read_i64_into::<LittleEndian>(&mut val)
            .context("Could not read ca_data")?;
        Some(val)
    } else {
        None
    };

    let mut val = vec![0i64; (ca_members.ca_ndim * 3) as usize];
    let ca_dynamic_size: Option<Vec<i64>> = if (ca_members.ca_flags & 0b1) > 0 {
        ca_block
            .read_i64_into::<LittleEndian>(&mut val)
            .context("Could not read ca_dynamic_size")?;
        Some(val)
    } else {
        None
    };

    let mut val = vec![0i64; (ca_members.ca_ndim * 3) as usize];
    let ca_input_quantity: Option<Vec<i64>> = if (ca_members.ca_flags & 0b10) > 0 {
        ca_block
            .read_i64_into::<LittleEndian>(&mut val)
            .context("Could not read ca_input_quantity")?;
        Some(val)
    } else {
        None
    };

    let mut val = vec![0i64; 3];
    let ca_output_quantity: Option<Vec<i64>> = if (ca_members.ca_flags & 0b100) > 0 {
        ca_block
            .read_i64_into::<LittleEndian>(&mut val)
            .context("Could not read ca_output_quantity")?;
        Some(val)
    } else {
        None
    };

    let mut val = vec![0i64; 3];
    let ca_comparison_quantity: Option<Vec<i64>> = if (ca_members.ca_flags & 0b1000) > 0 {
        ca_block
            .read_i64_into::<LittleEndian>(&mut val)
            .context("Could not read ca_comparison_quantity")?;
        Some(val)
    } else {
        None
    };

    let mut val = vec![0i64; ca_members.ca_ndim as usize];
    let ca_cc_axis_conversion: Option<Vec<i64>> = if (ca_members.ca_flags & 0b10000) > 0 {
        ca_block
            .read_i64_into::<LittleEndian>(&mut val)
            .context("Could not read ca_cc_axis_conversion")?;
        Some(val)
    } else {
        None
    };

    let mut val = vec![0i64; (ca_members.ca_ndim * 3) as usize];
    let ca_axis: Option<Vec<i64>> =
        if ((ca_members.ca_flags & 0b10000) > 0) & ((ca_members.ca_flags & 0b100000) > 0) {
            ca_block
                .read_i64_into::<LittleEndian>(&mut val)
                .context("Could not read ca_axis")?;
            Some(val)
        } else {
            None
        };

    Ok((
        Ca4Block {
            ca_id: block_header.hdr_id,
            reserved: block_header.hdr_gap,
            ca_len: block_header.hdr_len,
            ca_links,
            ca_composition,
            ca_data,
            ca_dynamic_size,
            ca_input_quantity,
            ca_output_quantity,
            ca_comparison_quantity,
            ca_cc_axis_conversion,
            ca_axis,
            ca_type: ca_members.ca_type,
            ca_storage: ca_members.ca_storage,
            ca_ndim: ca_members.ca_ndim,
            ca_flags: ca_members.ca_flags,
            ca_byte_offset_base: ca_members.ca_byte_offset_base,
            ca_inval_bit_pos_base: ca_members.ca_inval_bit_pos_base,
            ca_dim_size: ca_members.ca_dim_size,
            ca_axis_value,
            ca_cycle_count,
        },
        shape,
        snd,
        pnd,
    ))
}
