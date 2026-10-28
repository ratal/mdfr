//! Channel Array block (CABLOCK) for MDF4 — spec section 6.8, Tables 40-43
//!
//! Describes N-dimensional arrays of channel values. Supports multiple storage types
//! (CN template, CG template, DG template) and array types (plain, lookup, axis).
use anyhow::{Context, Error, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::VecDeque;
use std::io::Cursor;

use super::block_header::Blockheader4Short;
use crate::data_holder::tensor_arrow::Order;

/// type alias for Ca4Block parse result
pub type CaBlockParseResult = (Ca4Block, (Vec<usize>, Order), usize, usize);

pub use mdfr_core::blocks::ca::{Ca4Block, Ca4BlockMembers};

/// Channel Array block parser
pub(super) fn parse_ca_block(
    ca_block: &mut Cursor<Vec<u8>>,
    block_header: Blockheader4Short,
    cg_cycle_count: u64,
) -> Result<CaBlockParseResult, Error> {
    // reads links count
    let ca_links: u64 =
        binrw::BinReaderExt::read_le(ca_block).context("Could not read links count in ca block")?;
    //Reads members first
    ca_block.set_position(8 + ca_links * 8); // change buffer position after links section
    let ca_members: Ca4BlockMembers = binrw::BinReaderExt::read_le(ca_block)
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
        if ((ca_members.ca_flags & 0b10000) > 0) && ((ca_members.ca_flags & 0b100000) == 0) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let ca = Ca4Block::default();
        assert_eq!(ca.ca_id, [35, 35, 67, 65]); // ##CA
        assert_eq!(ca.ca_type, 0);
        assert_eq!(ca.ca_storage, 0);
        assert_eq!(ca.ca_ndim, 1);
        assert_eq!(ca.ca_flags, 0);
        assert!(ca.ca_dim_size.is_empty());
        assert!(ca.ca_data.is_none());
        assert!(ca.ca_axis_value.is_none());
        assert!(ca.ca_cycle_count.is_none());
    }

    #[test]
    fn test_get_ca_type_str() {
        let mut ca = Ca4Block::default();
        let expected = [
            (0, "Array"),
            (1, "ScalingAxis"),
            (2, "LookUp"),
            (3, "IntervalAxis"),
            (4, "ClassificationResult"),
            (255, "Unknown"),
        ];
        for (val, name) in expected {
            ca.ca_type = val;
            assert_eq!(ca.get_ca_type_str(), name, "ca_type={val}");
        }
    }

    #[test]
    fn test_get_storage_str() {
        let mut ca = Ca4Block::default();
        let expected = [
            (0, "CN Template"),
            (1, "CG Template"),
            (2, "DG Template"),
            (255, "Unknown"),
        ];
        for (val, name) in expected {
            ca.ca_storage = val;
            assert_eq!(ca.get_storage_str(), name, "ca_storage={val}");
        }
    }

    #[test]
    fn test_calculate_block_len() {
        // Default: 1 link (ca_composition), no optional fields, ndim=1
        let ca = Ca4Block {
            ca_ndim: 1,
            ca_dim_size: vec![5],
            ..Default::default()
        };
        // 16 (header) + 8 (link count) + 1*8 (links) + 16 (members) + 1*8 (dim_size) = 56
        assert_eq!(ca.calculate_block_len(), 56);

        // With DG template storage (adds pnd=5 data links)
        let ca = Ca4Block {
            ca_ndim: 1,
            ca_storage: 2,
            ca_dim_size: vec![5],
            ..Default::default()
        };
        // links = 1 + 5 = 6
        // 16 + 8 + 6*8 + 16 + 1*8 = 96
        assert_eq!(ca.calculate_block_len(), 96);

        // With axis_value
        let ca = Ca4Block {
            ca_ndim: 1,
            ca_dim_size: vec![3],
            ca_axis_value: Some(vec![1.0, 2.0, 3.0]),
            ..Default::default()
        };
        // 56 + 3*8 = 80
        assert_eq!(ca.calculate_block_len(), 80);
    }

    #[test]
    fn test_calculate_link_count_with_flags() {
        // No flags: just ca_composition = 1
        let ca = Ca4Block {
            ca_ndim: 2,
            ca_dim_size: vec![3, 4],
            ..Default::default()
        };
        assert_eq!(ca.calculate_block_len(), 16 + 8 + 8 + 16 + 2 * 8); // 64

        // Dynamic size flag (bit 0): adds D*3 = 6 links
        let ca = Ca4Block {
            ca_ndim: 2,
            ca_dim_size: vec![3, 4],
            ca_flags: 0b1,
            ..Default::default()
        };
        // links = 1 + 6 = 7
        assert_eq!(ca.calculate_block_len(), 16 + 8 + 7 * 8 + 16 + 2 * 8);

        // Axis flag without fixed axes (bit 4 set, bit 5 clear): adds D + D*3
        let ca = Ca4Block {
            ca_ndim: 2,
            ca_dim_size: vec![3, 4],
            ca_flags: 0b10000,
            ..Default::default()
        };
        // links = 1 + 2 + 6 = 9 (ca_cc_axis_conversion + ca_axis)
        assert_eq!(ca.calculate_block_len(), 16 + 8 + 9 * 8 + 16 + 2 * 8);

        // Axis flag with fixed axes (bits 4+5 set): adds D only (no ca_axis)
        let ca = Ca4Block {
            ca_ndim: 2,
            ca_dim_size: vec![3, 4],
            ca_flags: 0b110000,
            ..Default::default()
        };
        // links = 1 + 2 = 3
        assert_eq!(ca.calculate_block_len(), 16 + 8 + 3 * 8 + 16 + 2 * 8);
    }

    #[test]
    fn test_prepare_for_write() {
        let mut ca = Ca4Block {
            ca_ndim: 1,
            ca_dim_size: vec![3],
            ca_composition: 1000,
            ca_data: Some(vec![100, 200, 300]),
            ca_storage: 2,
            ca_dynamic_size: Some(vec![10, 20, 30]),
            ca_flags: 0b1,
            ..Default::default()
        };
        ca.prepare_for_write();

        assert_eq!(ca.ca_composition, 0);
        assert_eq!(ca.ca_data, Some(vec![0, 0, 0]));
        assert_eq!(ca.ca_dynamic_size, Some(vec![0, 0, 0]));
        // links recalculated: 1 + 3 (data) + 3 (dynamic_size) = 7
        assert_eq!(ca.ca_links, 7);
    }

    #[test]
    fn test_display() {
        let ca = Ca4Block::default();
        let display = format!("{ca}");
        assert!(display.contains("CA:"));
        assert!(display.contains("Array"));
        assert!(display.contains("CN Template"));

        let ca = Ca4Block {
            ca_type: 2,
            ca_storage: 1,
            ca_dim_size: vec![3, 4],
            ..Default::default()
        };
        let display = format!("{ca}");
        assert!(display.contains("LookUp"));
        assert!(display.contains("CG Template"));
        assert!(display.contains("[3, 4]"));
    }
}
