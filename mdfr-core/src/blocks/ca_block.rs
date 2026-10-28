//! Channel Array block (CABLOCK) for MDF4 — spec section 6.8, Tables 40-43.
//!
//! Describes N-dimensional arrays of channel values. Supports multiple storage types
//! (CN template, CG template, DG template) and array types (plain, lookup, axis).
use alloc::vec;
use alloc::vec::Vec;
use binrw::{
    BinWriterExt, binrw,
    io::{Seek, Write},
};
use core::fmt::{self, Display};

use super::header::Blockheader4Short;
use crate::error::{Error, Result};

/// CABLOCK structure (MDF 4.2 spec, Table 40)
#[derive(Debug, PartialEq, Clone)]
#[repr(C)]
pub struct Ca4Block {
    // header
    /// ##CA
    pub ca_id: [u8; 4],
    /// reserved
    pub reserved: [u8; 4],
    /// Length of block in bytes
    pub ca_len: u64,
    /// # of links
    pub ca_links: u64,
    // links
    /// [] Array of composed elements: Pointer to a CNBLOCK for array of structures, or to a CABLOCK for array of arrays (can be NIL).
    pub ca_composition: i64,
    /// [Π N(d) or empty] Only present for storage type "DG template".
    pub ca_data: Option<Vec<i64>>,
    /// [Dx3 or empty] Only present if "dynamic size" flag (bit 0) is set.
    pub ca_dynamic_size: Option<Vec<i64>>,
    /// [Dx3 or empty] Only present if "input quantity" flag (bit 1) is set.
    pub ca_input_quantity: Option<Vec<i64>>,
    /// [3 or empty] Only present if "output quantity" flag (bit 2) is set.
    pub ca_output_quantity: Option<Vec<i64>>,
    /// [3 or empty] Only present if "comparison quantity" flag (bit 3) is set.
    pub ca_comparison_quantity: Option<Vec<i64>>,
    /// [D or empty] Only present if "axis" flag (bit 4) is set.
    pub ca_cc_axis_conversion: Option<Vec<i64>>,
    /// [Dx3 or empty] Only present if "axis" flag (bit 4) is set and "fixed axes flag" (bit 5) is not set.
    pub ca_axis: Option<Vec<i64>>,
    // members
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
            ca_byte_offset_base: 0,
            ca_inval_bit_pos_base: 0,
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
    /// Calculate the number of links based on storage type and flags
    pub fn calculate_link_count(&self) -> u64 {
        let d = self.ca_ndim as u64;
        let pnd = self.ca_dim_size.iter().product::<u64>().max(1);
        let mut links: u64 = 1; // ca_composition (always present)
        if self.ca_storage == 2 {
            links += pnd; // ca_data
        }
        if self.ca_flags & 0b1 != 0 {
            links += d * 3; // ca_dynamic_size
        }
        if self.ca_flags & 0b10 != 0 {
            links += d * 3; // ca_input_quantity
        }
        if self.ca_flags & 0b100 != 0 {
            links += 3; // ca_output_quantity
        }
        if self.ca_flags & 0b1000 != 0 {
            links += 3; // ca_comparison_quantity
        }
        if self.ca_flags & 0b10000 != 0 {
            links += d; // ca_cc_axis_conversion
        }
        if (self.ca_flags & 0b10000 != 0) && (self.ca_flags & 0b100000 == 0) {
            links += d * 3; // ca_axis (axis flag set, fixed axes NOT set)
        }
        links
    }
    /// Calculate the total block length in bytes
    pub fn calculate_block_len(&self) -> u64 {
        let d = self.ca_ndim as u64;
        let n_links = self.calculate_link_count();
        let mut len: u64 = 16 + 8 + n_links * 8 + 16 + d * 8;
        if let Some(ref vals) = self.ca_axis_value {
            len += vals.len() as u64 * 8;
        }
        if let Some(ref vals) = self.ca_cycle_count {
            len += vals.len() as u64 * 8;
        }
        len
    }
    /// Prepare CA block for writing: zero unmappable link values, recalculate sizes
    pub fn prepare_for_write(&mut self) {
        self.ca_composition = 0;
        if let Some(ref mut data) = self.ca_data {
            data.iter_mut().for_each(|v| *v = 0);
        }
        if let Some(ref mut ds) = self.ca_dynamic_size {
            ds.iter_mut().for_each(|v| *v = 0);
        }
        if let Some(ref mut iq) = self.ca_input_quantity {
            iq.iter_mut().for_each(|v| *v = 0);
        }
        if let Some(ref mut oq) = self.ca_output_quantity {
            oq.iter_mut().for_each(|v| *v = 0);
        }
        if let Some(ref mut cq) = self.ca_comparison_quantity {
            cq.iter_mut().for_each(|v| *v = 0);
        }
        if let Some(ref mut cc) = self.ca_cc_axis_conversion {
            cc.iter_mut().for_each(|v| *v = 0);
        }
        if let Some(ref mut ax) = self.ca_axis {
            ax.iter_mut().for_each(|v| *v = 0);
        }
        self.ca_links = self.calculate_link_count();
        self.ca_len = self.calculate_block_len();
    }
    /// Write the full CA block to a binrw writer
    pub fn write_to<W: Write + Seek>(&self, buffer: &mut W) -> Result<()> {
        let header = Blockheader4Short {
            hdr_id: self.ca_id,
            hdr_gap: self.reserved,
            hdr_len: self.ca_len,
        };
        buffer.write_le(&header).map_err(|_| Error::Io)?;
        buffer.write_le(&self.ca_links).map_err(|_| Error::Io)?;
        buffer
            .write_le(&self.ca_composition)
            .map_err(|_| Error::Io)?;
        for vals in [
            &self.ca_data,
            &self.ca_dynamic_size,
            &self.ca_input_quantity,
            &self.ca_output_quantity,
            &self.ca_comparison_quantity,
            &self.ca_cc_axis_conversion,
            &self.ca_axis,
        ]
        .into_iter()
        .flatten()
        {
            for val in vals {
                buffer.write_le(val).map_err(|_| Error::Io)?;
            }
        }
        buffer.write_le(&self.ca_type).map_err(|_| Error::Io)?;
        buffer.write_le(&self.ca_storage).map_err(|_| Error::Io)?;
        buffer.write_le(&self.ca_ndim).map_err(|_| Error::Io)?;
        buffer.write_le(&self.ca_flags).map_err(|_| Error::Io)?;
        buffer
            .write_le(&self.ca_byte_offset_base)
            .map_err(|_| Error::Io)?;
        buffer
            .write_le(&self.ca_inval_bit_pos_base)
            .map_err(|_| Error::Io)?;
        for dim in &self.ca_dim_size {
            buffer.write_le(dim).map_err(|_| Error::Io)?;
        }
        if let Some(ref vals) = self.ca_axis_value {
            for v in vals {
                buffer.write_le(v).map_err(|_| Error::Io)?;
            }
        }
        if let Some(ref vals) = self.ca_cycle_count {
            for v in vals {
                buffer.write_le(v).map_err(|_| Error::Io)?;
            }
        }
        Ok(())
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
#[bw(little)]
#[repr(C)]
pub struct Ca4BlockMembers {
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
