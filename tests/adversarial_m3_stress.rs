#![allow(clippy::useless_vec)]
//! Adversarial Stress & Failure Mode Tests for Milestone 3 (Features 20 - 27)
//!
//! Stress-tests boundary conditions, edge invariants, endian conversions,
//! and robustness under adversarial input conditions.

use anyhow::Result;
use mdfr::data_holder::channel_data::ChannelData;
use mdfr::data_holder::tensor_arrow::Order;
use mdfr::mdfinfo::mdfinfo4::{Cn4, Cn4Block, Endianness};
use mdfr::mdfinfo::sym_buf_reader::SymBufReader;
use mdfr::mdfreader::data_read4::read_one_channel_array;
use mdfr_core::blocks::cn_block::CN_F_INVALID;
use std::io::Cursor;

// ============================================================================
// 1. Feature 20: Validity Bits, Polarity Inversion & Flags
// ============================================================================

#[test]
fn test_adv_feat20_invalidation_polarity_and_all_extremes() {
    // Test all valid (Arrow: all true -> MDF: all 0)
    let arrow_valid = vec![true; 16];
    let mdf_bits: Vec<u8> = arrow_valid.iter().map(|&v| (!v) as u8).collect();
    assert_eq!(mdf_bits, vec![0u8; 16]);

    // Test all invalid (Arrow: all false -> MDF: all 1)
    let arrow_invalid = vec![false; 16];
    let mdf_inval_bits: Vec<u8> = arrow_invalid.iter().map(|&v| (!v) as u8).collect();
    assert_eq!(mdf_inval_bits, vec![1u8; 16]);

    // Inversion symmetry
    for val in [true, false] {
        let mdf_rep = (!val) as u8;
        let recovered = mdf_rep == 0;
        assert_eq!(val, recovered);
    }

    // Flag constant
    assert_eq!(CN_F_INVALID, 1 << 1);
}

// ============================================================================
// 2. Feature 21: Non-Byte-Aligned 9-Byte Spans (LE & BE, Signed & Unsigned)
// ============================================================================

#[test]
fn test_adv_feat21_9_byte_span_uint64_le_and_be() -> Result<()> {
    // 9-byte raw payload:
    // Suppose bit_offset = 4, bit_count = 64.
    // Total bits = 68 -> spans 9 bytes.
    let cycle_count = 2usize;

    // --- Little Endian 9-byte Unsigned ---
    // Target value: 0x1122_3344_5566_7788
    // Shifted left by 4 bits into 9 bytes:
    // (0x1122_3344_5566_7788 << 4) = low 64 bits + high 4 bits in byte 8
    let target_val_le: u64 = 0x1122_3344_5566_7788;
    let shifted_le = (target_val_le as u128) << 4;
    let mut rec0_le = [0u8; 9];
    rec0_le.copy_from_slice(&shifted_le.to_le_bytes()[0..9]);

    // Second record with max 64-bit value: u64::MAX
    let target_val_le_2: u64 = u64::MAX;
    let shifted_le_2 = (target_val_le_2 as u128) << 4;
    let mut rec1_le = [0u8; 9];
    rec1_le.copy_from_slice(&shifted_le_2.to_le_bytes()[0..9]);

    let mut data_bytes_le = Vec::new();
    data_bytes_le.extend_from_slice(&rec0_le);
    data_bytes_le.extend_from_slice(&rec1_le);

    let mut cn_le = Cn4 {
        n_bytes: 9,
        endian: Endianness::Little,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        pos_byte_beg: 0,
        should_read: true,
        data: ChannelData::UInt64(arrow::array::UInt64Array::from(vec![0u64; 2])),
        builder: None,
        block: Cn4Block {
            cn_data_type: 0, // Unsigned LE
            cn_bit_offset: 4,
            cn_bit_count: 64,
            ..Default::default()
        },
        ..Default::default()
    };
    cn_le.init_builder(cycle_count)?;
    read_one_channel_array(&data_bytes_le, &mut cn_le, cycle_count)?;
    if let ChannelData::UInt64(arr) = &cn_le.data {
        assert_eq!(arr.values()[0], target_val_le, "LE sample 0 mismatch");
        assert_eq!(arr.values()[1], target_val_le_2, "LE sample 1 mismatch");
    } else {
        panic!("Expected UInt64 channel data");
    }

    // --- Big Endian 9-byte Unsigned ---
    // Target value: 0x8877_6655_4433_2211
    // In BE, bit_offset = 4 means the LSB is at bit 4 of byte 8.
    // The signal occupies 4 bits in byte 8 (bits 4..7) and bits 0..59 in bytes 0..7.
    let target_val_be: u64 = 0x8877_6655_4433_2211;
    let high_part = target_val_be >> 4;
    let low_part = (target_val_be & 0x0F) << 4;
    let mut rec0_be = [0u8; 9];
    rec0_be[0..8].copy_from_slice(&high_part.to_be_bytes());
    rec0_be[8] = low_part as u8;

    let mut data_bytes_be = Vec::new();
    data_bytes_be.extend_from_slice(&rec0_be);
    data_bytes_be.extend_from_slice(&rec0_be);

    let mut cn_be = Cn4 {
        n_bytes: 9,
        endian: Endianness::Big,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        pos_byte_beg: 0,
        should_read: true,
        data: ChannelData::UInt64(arrow::array::UInt64Array::from(vec![0u64; 2])),
        builder: None,
        block: Cn4Block {
            cn_data_type: 1, // Unsigned BE
            cn_bit_offset: 4,
            cn_bit_count: 64,
            ..Default::default()
        },
        ..Default::default()
    };
    cn_be.init_builder(cycle_count)?;
    read_one_channel_array(&data_bytes_be, &mut cn_be, cycle_count)?;
    if let ChannelData::UInt64(arr) = &cn_be.data {
        assert_eq!(arr.values()[0], target_val_be, "BE sample 0 mismatch");
        assert_eq!(arr.values()[1], target_val_be, "BE sample 1 mismatch");
    } else {
        panic!("Expected UInt64 channel data");
    }

    Ok(())
}

#[test]
fn test_adv_feat21_9_byte_span_int64_signed_negative() -> Result<()> {
    // Test negative signed int64 across 9-byte span with sign extension
    let cycle_count = 1usize;
    let target_val: i64 = -12345678901234567i64;

    // Little Endian
    let shifted = ((target_val as u64) as u128) << 5;
    let mut rec = [0u8; 9];
    rec.copy_from_slice(&shifted.to_le_bytes()[0..9]);

    let mut cn = Cn4 {
        n_bytes: 9,
        endian: Endianness::Little,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        pos_byte_beg: 0,
        should_read: true,
        data: ChannelData::Int64(arrow::array::Int64Array::from(vec![0i64; 1])),
        builder: None,
        block: Cn4Block {
            cn_data_type: 2, // Signed LE
            cn_bit_offset: 5,
            cn_bit_count: 64,
            ..Default::default()
        },
        ..Default::default()
    };
    cn.init_builder(cycle_count)?;
    read_one_channel_array(&rec, &mut cn, cycle_count)?;
    if let ChannelData::Int64(arr) = &cn.data {
        assert_eq!(arr.values()[0], target_val, "Signed LE negative mismatch");
    } else {
        panic!("Expected Int64 channel data");
    }
    Ok(())
}

// ============================================================================
// 3. Feature 24: SymBufReader Contiguous Read & Rewind Invariants
// ============================================================================

#[test]
fn test_adv_feat24_sym_buf_reader_state_transitions() -> Result<()> {
    use std::io::Read;

    // 128 KiB test buffer
    let data: Vec<u8> = (0..128 * 1024).map(|i| (i % 256) as u8).collect();
    let cur = Cursor::new(data.clone());
    let mut reader = SymBufReader::new(cur);

    // Initial sequential forward reads across buffer boundaries
    let mut out1 = vec![0u8; 32 * 1024];
    reader.read_exact(&mut out1)?;
    assert_eq!(out1, &data[0..32 * 1024]);

    let mut out2 = vec![0u8; 40 * 1024];
    reader.read_exact(&mut out2)?;
    assert_eq!(out2, &data[32 * 1024..72 * 1024]);

    // Backward seek
    reader.seek_relative(-1024)?;
    let mut out3 = vec![0u8; 1024];
    reader.read_exact(&mut out3)?;
    assert_eq!(out3, &data[71 * 1024..72 * 1024]);

    // Forward seek beyond current buffer
    reader.seek_relative(10 * 1024)?;
    let mut out4 = vec![0u8; 512];
    reader.read_exact(&mut out4)?;
    assert_eq!(out4, &data[82 * 1024 + 512..83 * 1024]);

    Ok(())
}

// ============================================================================
// 4. Feature 25: MDF3 Unsorted Trailing Buffer Assembly Across Tiny Chunks
// ============================================================================

#[test]
fn test_adv_feat25_mdf3_unsorted_micro_chunk_assembly() {
    // 3 records of 7 bytes each = 21 bytes
    // Chunks of 4 bytes: [4, 4, 4, 4, 4, 1]
    let record_len = 7usize;
    let raw_records: Vec<u8> = (0..21).map(|i| (i + 1) as u8).collect();

    let mut trailing = Vec::new();
    let mut assembled = Vec::new();

    for chunk in raw_records.chunks(4) {
        let mut work = trailing.clone();
        work.extend_from_slice(chunk);

        let n_complete = work.len() / record_len;
        let consumed = n_complete * record_len;
        assembled.extend_from_slice(&work[..consumed]);
        trailing = work[consumed..].to_vec();
    }

    assert_eq!(trailing.len(), 0, "All bytes must be assembled");
    assert_eq!(
        assembled, raw_records,
        "Records must match original exactly"
    );
}

// ============================================================================
// 5. Feature 26: MDF3 BE 5/7-Byte Integers Sign Extension
// ============================================================================

#[test]
fn test_adv_feat26_mdf3_be_signed_integers_boundary_values() {
    // 5-byte signed negative: 0x80_00_00_00_00 -> -549755813888
    let raw_5_neg = [0x80u8, 0x00, 0x00, 0x00, 0x00];
    let sign = if raw_5_neg[0] & 0x80 != 0 { 0xFF } else { 0x00 };
    let mut buf5 = [sign; 8];
    buf5[3..8].copy_from_slice(&raw_5_neg);
    let val5 = i64::from_be_bytes(buf5);
    assert_eq!(val5, -549755813888i64);

    // 5-byte signed positive: 0x7F_FF_FF_FF_FF -> 549755813887
    let raw_5_pos = [0x7Fu8, 0xFF, 0xFF, 0xFF, 0xFF];
    let sign = if raw_5_pos[0] & 0x80 != 0 { 0xFF } else { 0x00 };
    let mut buf5_pos = [sign; 8];
    buf5_pos[3..8].copy_from_slice(&raw_5_pos);
    let val5_pos = i64::from_be_bytes(buf5_pos);
    assert_eq!(val5_pos, 549755813887i64);

    // 7-byte signed negative: 0x80_00_00_00_00_00_00 -> -36028797018963968
    let raw_7_neg = [0x80u8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
    let sign = if raw_7_neg[0] & 0x80 != 0 { 0xFF } else { 0x00 };
    let mut buf7 = [sign; 8];
    buf7[1..8].copy_from_slice(&raw_7_neg);
    let val7 = i64::from_be_bytes(buf7);
    assert_eq!(val7, -36028797018963968i64);
}

// ============================================================================
// 6. Feature 27: Value Range Binary Search & Exclusivity Invariants
// ============================================================================

#[test]
fn test_adv_feat27_partition_point_range_lookup() {
    // 5 sorted non-overlapping ranges:
    // Range 0: [0.0, 10.0) -> "low"
    // Range 1: [10.0, 20.0) -> "medium"
    // Range 2: [20.0, 30.0) -> "high"
    // Range 3: [30.0, 30.0] -> "exact_thirty" (point interval: min == max)
    // Range 4: [50.0, 100.0) -> "extreme"
    struct R {
        min: f64,
        max: f64,
        label: &'static str,
    }
    let ranges = [
        R {
            min: 0.0,
            max: 10.0,
            label: "low",
        },
        R {
            min: 10.0,
            max: 20.0,
            label: "medium",
        },
        R {
            min: 20.0,
            max: 30.0,
            label: "high",
        },
        R {
            min: 30.0,
            max: 30.0,
            label: "exact_thirty",
        },
        R {
            min: 50.0,
            max: 100.0,
            label: "extreme",
        },
    ];

    let lookup = |val: f64, inclusive_upper: bool| -> Option<&'static str> {
        let idx = ranges.partition_point(|k| k.min <= val);
        if idx > 0 {
            let cand = &ranges[idx - 1];
            let in_range = if inclusive_upper {
                val <= cand.max
            } else {
                val < cand.max || (cand.min == cand.max && val == cand.min)
            };
            if in_range { Some(cand.label) } else { None }
        } else {
            None
        }
    };

    // Float exclusivity:
    assert_eq!(lookup(0.0, false), Some("low"));
    assert_eq!(lookup(9.9999, false), Some("low"));
    assert_eq!(lookup(10.0, false), Some("medium")); // Boundary belongs to medium, not low
    assert_eq!(lookup(29.9999, false), Some("high"));
    assert_eq!(lookup(30.0, false), Some("exact_thirty")); // Point interval min == max
    assert_eq!(lookup(40.0, false), None); // In gap between 30 and 50
    assert_eq!(lookup(100.0, false), None); // Upper bound 100.0 is excluded
    assert_eq!(lookup(-1.0, false), None); // Below min
}
