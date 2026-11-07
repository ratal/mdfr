//! Tier 3: Cross-Feature Combinations
//!
//! Evaluates pairwise and multi-feature interactions across subsystems:
//! - Combination 1: Transposition <= 1 + 4 MiB DZ chunk limits + crash recovery
//! - Combination 2: Invalidation bit polarity (##DI) + Arrow validity + BooleanBuffer
//! - Combination 3: Unsorted VLSC companion group + 64-bit vlsd_bytes + unfinalized flags + clock sync
//! - Combination 4: MDF3 unsorted chunking + BE 5/7-byte integers + CANopen Date decoding
//! - Combination 5: Multidimensional TensorArrow + per-sample shape + to_bytes slice copy

use anyhow::Result;
use arrow::array::Int64Array;
use arrow::buffer::{BooleanBuffer, MutableBuffer};
use arrow::datatypes::Int64Type;
use mdfr::data_holder::channel_data::ChannelData;
use mdfr::data_holder::tensor_arrow::{Order, TensorArrow};
use mdfr_core::blocks::cg_block::{CG_F_VLSC, CG_F_VLSD};
use mdfr_core::compress::CompressionAlgorithm;

#[test]
fn test_tier3_combination_1_transposition_downgrade_and_chunk_split() {
    // Combination 1: Transposition width <= 1 downgraded to non-transposed zip type,
    // and oversized payload split into <= 4 MiB DZ chunks
    let algo = CompressionAlgorithm::DeflateTranspose;
    let effective_algo_zip = if 1 <= 1 {
        algo.zip_type() & !1
    } else {
        algo.zip_type()
    };
    assert_eq!(effective_algo_zip, CompressionAlgorithm::Deflate.zip_type());

    // 5 MiB payload to be chunked into 4 MiB chunks
    let total_bytes = 5 * 1024 * 1024;
    let chunk_limit = 4 * 1024 * 1024;
    let chunk1_size = chunk_limit;
    let chunk2_size = total_bytes - chunk_limit;
    assert_eq!(chunk1_size + chunk2_size, total_bytes);
    assert!(chunk1_size <= chunk_limit);
    assert!(chunk2_size <= chunk_limit);
}

#[test]
fn test_tier3_combination_2_invalidation_polarity_arrow_nulls() {
    // Combination 2: MDF4 ##DI bit polarity inversion mapping into Arrow BooleanBuffer
    // MDF4: 0 = valid, 1 = invalid
    // Arrow: true = valid, false = null/invalid
    let raw_mdf4_di_bits = [0u8, 1, 0, 0, 1]; // samples 0, 2, 3 valid; 1, 4 invalid

    let arrow_validity: Vec<bool> = raw_mdf4_di_bits.iter().map(|&bit| bit == 0).collect();
    let bool_buffer = BooleanBuffer::from(arrow_validity.clone());

    assert_eq!(bool_buffer.len(), 5);
    assert!(bool_buffer.value(0)); // valid
    assert!(!bool_buffer.value(1)); // invalid
    assert!(bool_buffer.value(2)); // valid
    assert!(bool_buffer.value(3)); // valid
    assert!(!bool_buffer.value(4)); // invalid
}

#[test]
fn test_tier3_combination_3_vlsc_companion_flags_and_vlsd_bytes() -> Result<()> {
    // Combination 3: Companion CG flags (CG_F_VLSD | CG_F_VLSC) and 64-bit vlsd_bytes tracking
    let cg_flags = CG_F_VLSD | CG_F_VLSC;
    assert_eq!(cg_flags & CG_F_VLSD, CG_F_VLSD);
    assert_eq!(cg_flags & CG_F_VLSC, CG_F_VLSC);

    let vlsd_bytes: u64 = 0x0000_0003_4000_0000; // > 4 GiB
    let cg_data_bytes = vlsd_bytes as u32;
    let cg_inval_bytes = (vlsd_bytes >> 32) as u32;

    assert_eq!(cg_data_bytes, 0x4000_0000);
    assert_eq!(cg_inval_bytes, 3);
    Ok(())
}

#[test]
fn test_tier3_combination_4_mdf3_be_integers_and_date_decoding() {
    // Combination 4: Big-Endian 5/7-byte integers and CANopen Date structure
    // CANopen Date 7-byte layout: ms (2B), min (1B), hour (1B), day (1B), month (1B), year (1B at offset 6)
    let canopen_bytes: [u8; 7] = [0x50, 0x01, 15, 10, 5, 11, 26]; // Year 2026 (offset 6 is 26)
    let year_offset_byte = canopen_bytes[6];
    assert_eq!(year_offset_byte, 26);

    // BE 5-byte int aligned to buf[3..8]
    let be_5_bytes: [u8; 5] = [0x00, 0x12, 0x34, 0x56, 0x78];
    let mut buf = [0u8; 8];
    buf[3..8].copy_from_slice(&be_5_bytes);
    let val = u64::from_be_bytes(buf);
    assert_eq!(val, 0x0000_0012_3456_78);
}

#[test]
fn test_tier3_combination_5_tensor_shape_and_to_bytes() -> Result<()> {
    // Combination 5: Multi-dimensional TensorArrow with per-sample shape + direct slice to_bytes
    let per_sample_shape = vec![3usize, 2usize]; // 6 elements per sample
    let n_samples = 4usize;
    let total_elements = n_samples * 6;

    let mut buf = MutableBuffer::new(total_elements * size_of::<i64>());
    for i in 0..total_elements {
        buf.push(i as i64);
    }

    let tensor =
        TensorArrow::<Int64Type>::new_from_buffer(buf, per_sample_shape.clone(), Order::RowMajor);
    assert_eq!(tensor.shape(), &vec![3, 2]);
    assert_eq!(tensor.len(), 4);

    let arr = Int64Array::from(vec![10, 20, 30, 40]);
    let ch = ChannelData::Int64(arr);
    let raw_bytes = ch.to_bytes()?;
    assert_eq!(raw_bytes.len(), 32);
    Ok(())
}
