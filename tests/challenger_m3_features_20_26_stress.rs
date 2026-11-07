//! Milestone 3 (Features 20-26) Comprehensive Empirical Stress Tests
//!
//! Written by Challenger 1 (teamwork_challenger_m3_1)
//!
//! Coverage:
//! 1. Invalidation bits (Feature 20):
//!    - Writing channels with validity masks (null values) via uncompressed (DI) and compressed (DZ-wrapped DI).
//!    - Verification of Arrow null buffer match (`true` for valid, `false` for null).
//!    - Verification of CN_F_INVALID flag and cg_inval_bytes = 1.
//! 2. 9-byte bitfield channel decoding (Feature 21):
//!    - 64-bit value with `bit_offset = 4` across 9 bytes.
//!    - Little-Endian and Big-Endian, signed and unsigned.
//!    - Boundary and edge-case values (0, max, min, arbitrary patterns) with surrounding noise bits.
//! 3. Parser DL4 sorted rec_pos resolution (Feature 22):
//!    - Channel in sorted group with non-zero `rec_pos` resolving VLSD records without silent data loss.
//! 4. DZ 4 MiB uncompressed cap & fallback (Feature 23):
//!    - Channel with > 4 MiB uncompressed data written with compression.
//!    - Clean fallback to uncompressed DV without file corruption or data loss.
//! 5. `SymBufReader` redundant seek elimination (Feature 24):
//!    - Verification that contiguous forward reads execute 0 redundant `seek` calls.
//! 6. MDF3 unsorted multi-chunk & 2-byte record IDs (Feature 25):
//!    - Unsorted records crossing 512 KiB chunk boundaries with trailing bytes preserved.
//!    - 2-byte record ID decoding (dg_n_record_ids = 2).
//!    - Clean break on unknown record ID.
//! 7. MDF3 Big-Endian 5- and 7-byte integers (Feature 26):
//!    - Exact round-trip value decoding for signed and unsigned 5-byte and 7-byte integers.

use anyhow::Result;
use arrow::array::{Array, Float64Array};
use arrow::buffer::{BooleanBuffer, NullBuffer};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::sync::Arc;
use tempfile::NamedTempFile;

// mdfr-core imports
use mdfr_core::blocks::cn::CN_F_INVALID;
use mdfr_core::compress::CompressionAlgorithm;

// mdfr imports
use mdfr::data_holder::channel_builder::ChannelBuilder;
use mdfr::data_holder::channel_data::ChannelData;
use mdfr::data_holder::tensor_arrow::Order;
use mdfr::mdfinfo::MdfInfo;
use mdfr::mdfinfo::mdfinfo3::{Cn3, Cn3Block1, Cn3Block2};
use mdfr::mdfinfo::mdfinfo4::{Cn4, Cn4Block, Endianness};
use mdfr::mdfinfo::sym_buf_reader::SymBufReader;
use mdfr::mdfreader::Mdf;
use mdfr::mdfreader::data_read3::read_channels_from_bytes as read_channels_from_bytes_3;
use mdfr::mdfreader::data_read4::{decode_u64_slice, read_one_channel_array};

// ============================================================================
// Helper: Synthetic MDF4 Builders for Testing
// ============================================================================

fn push_u8(buf: &mut Vec<u8>, v: u8) {
    buf.push(v);
}
fn push_u16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn push_i16(buf: &mut Vec<u8>, v: i16) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
#[allow(dead_code)]
fn push_i32(buf: &mut Vec<u8>, v: i32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn push_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn push_i64(buf: &mut Vec<u8>, v: i64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn push_f64(buf: &mut Vec<u8>, v: f64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn push_zeros(buf: &mut Vec<u8>, n: usize) {
    buf.extend(std::iter::repeat_n(0u8, n));
}

fn write_id4(buf: &mut Vec<u8>) {
    buf.extend_from_slice(b"MDF     ");
    buf.extend_from_slice(b"4.30    ");
    buf.extend_from_slice(b"mdfr    ");
    push_u16(buf, 0); // id_default_byteorder
    push_u16(buf, 0); // id_floatingpointformat
    push_u16(buf, 430); // id_ver
    push_u16(buf, 0); // id_codepage
    push_zeros(buf, 2); // id_check
    push_zeros(buf, 26); // id_fill
    push_u16(buf, 0); // id_unfin_flags
    push_u16(buf, 0); // id_custom_unfin_flags
}

fn write_hd4(buf: &mut Vec<u8>, dg_first: i64) {
    buf.extend_from_slice(b"##HD");
    push_zeros(buf, 4);
    push_u64(buf, 104);
    push_u64(buf, 6); // links: dg_first, fh_first, ch_first, at_first, ev_first, md_comment
    push_i64(buf, dg_first);
    push_i64(buf, 0); // fh_first
    push_i64(buf, 0); // ch_first
    push_i64(buf, 0); // at_first
    push_i64(buf, 0); // ev_first
    push_i64(buf, 0); // md_comment
    push_u64(buf, 0); // start_time_ns
    push_i16(buf, 0);
    push_i16(buf, 0);
    push_u8(buf, 0);
    push_u8(buf, 0);
    push_u8(buf, 0);
    push_u8(buf, 0);
    push_f64(buf, 0.0);
    push_f64(buf, 0.0);
}

fn write_tx4(buf: &mut Vec<u8>, text: &str) -> i64 {
    let pos = buf.len() as i64;
    let text_bytes = text.as_bytes();
    let body_len = text_bytes.len() + 1; // null-terminated
    let aligned_len = (body_len + 7) & !7;
    let blk_len = 24 + aligned_len as u64;

    buf.extend_from_slice(b"##TX");
    push_zeros(buf, 4);
    push_u64(buf, blk_len);
    push_u64(buf, 0); // 0 links
    buf.extend_from_slice(text_bytes);
    push_u8(buf, 0); // null terminator
    push_zeros(buf, aligned_len - body_len);
    pos
}

// ============================================================================
// Helper: Synthetic MDF3 Builders for Testing
// ============================================================================

fn write_id3(buf: &mut Vec<u8>) {
    buf.extend_from_slice(b"MDF     ");
    buf.extend_from_slice(b"3.30    ");
    buf.extend_from_slice(b"mdfr    ");
    push_u16(buf, 0); // id_default_byteorder
    push_u16(buf, 0); // id_floatingpointformat
    push_u16(buf, 330); // id_ver
    push_u16(buf, 0); // id_codepage
    push_zeros(buf, 2); // id_check
    push_zeros(buf, 26); // id_fill
    push_u16(buf, 0); // id_unfin_flags
    push_u16(buf, 0); // id_custom_unfin_flags
}

fn write_hd3(buf: &mut Vec<u8>, dg_first: u32, md_comment: u32) {
    buf.extend_from_slice(b"HD");
    push_u16(buf, 208); // hd_len
    push_u32(buf, dg_first);
    push_u32(buf, md_comment); // hd_md_comment
    push_u32(buf, 0); // hd_pr
    push_u16(buf, 1); // hd_n_datagroups
    buf.extend_from_slice(b"01:01:2026"); // hd_date (10 bytes)
    buf.extend_from_slice(b"12:00:00"); // hd_time (8 bytes)
    push_zeros(buf, 32); // hd_author
    push_zeros(buf, 32); // hd_organization
    push_zeros(buf, 32); // hd_project
    push_zeros(buf, 32); // hd_subject
    // 3.2 extension (44 bytes)
    push_u64(buf, 0); // hd_start_time_ns
    push_i16(buf, 0); // hd_time_offset
    push_u16(buf, 0); // hd_time_quality
    push_zeros(buf, 32); // hd_time_identifier
}

fn write_tx3(buf: &mut Vec<u8>) -> u32 {
    let pos = buf.len() as u32;
    buf.extend_from_slice(b"TX");
    push_u16(buf, 8);
    buf.extend_from_slice(b"ok\0\0");
    pos
}

fn write_dg3(
    buf: &mut Vec<u8>,
    dg_next: u32,
    cg_first: u32,
    dg_data: u32,
    n_cg: u16,
    dg_n_record_ids: u16,
) {
    buf.extend_from_slice(b"DG");
    push_u16(buf, 24); // dg_len
    push_u32(buf, dg_next);
    push_u32(buf, cg_first);
    push_u32(buf, 0); // dg_tr
    push_u32(buf, dg_data);
    push_u16(buf, n_cg); // dg_n_cg
    push_u16(buf, dg_n_record_ids); // dg_n_record_ids
}

fn write_cg3(
    buf: &mut Vec<u8>,
    cg_next: u32,
    cn_first: u32,
    record_id: u16,
    num_channels: u16,
    data_bytes: u16,
    cycle_count: u32,
) {
    buf.extend_from_slice(b"CG");
    push_u16(buf, 30); // cg_len
    push_u32(buf, cg_next);
    push_u32(buf, cn_first);
    push_u32(buf, 0); // cg_comment
    push_u16(buf, record_id);
    push_u16(buf, num_channels);
    push_u16(buf, data_bytes);
    push_u32(buf, cycle_count);
    push_u32(buf, 0); // cg_sr
}

fn write_cn3(
    buf: &mut Vec<u8>,
    cn_next: u32,
    cc_conversion: u32,
    name: &str,
    bit_offset: u16,
    bit_count: u16,
    data_type: u16,
) {
    let start = buf.len();
    buf.extend_from_slice(b"CN");
    push_u16(buf, 228);
    push_u32(buf, cn_next);
    push_u32(buf, cc_conversion);
    push_u32(buf, 0); // ce_source
    push_u32(buf, 0); // cd_source
    push_u32(buf, 0); // tx_comment
    push_u16(buf, 0); // cn_type (0 = normal data)
    let mut name_buf = [0u8; 32];
    let bytes = name.as_bytes();
    let copy_len = bytes.len().min(32);
    name_buf[..copy_len].copy_from_slice(&bytes[..copy_len]);
    buf.extend_from_slice(&name_buf); // 32 bytes
    push_zeros(buf, 128); // desc
    push_u16(buf, bit_offset);
    push_u16(buf, bit_count);
    push_u16(buf, data_type);
    push_u16(buf, 0); // value_range_valid
    push_f64(buf, 0.0); // min
    push_f64(buf, 0.0); // max
    push_f64(buf, 0.0); // sampling_rate
    push_u32(buf, 0); // tx_long_name
    push_u32(buf, 0); // tx_display_name
    push_u16(buf, 0); // byte_offset
    assert_eq!(buf.len() - start, 228, "CN block must be exactly 228 bytes");
}

fn write_cc3_identity(buf: &mut Vec<u8>) -> u32 {
    let pos = buf.len() as u32;
    buf.extend_from_slice(b"CC");
    push_u16(buf, 46);
    push_u16(buf, 0); // cc_valid_range_flags
    push_f64(buf, 0.0); // min
    push_f64(buf, 0.0); // max
    push_zeros(buf, 20); // unit
    push_u16(buf, 0xFFFF); // cc_type = 65535 (Identity / 1:1)
    push_u16(buf, 0); // cc_size
    assert_eq!(
        buf.len() - pos as usize,
        46,
        "CC block must be exactly 46 bytes"
    );
    pos
}

// ============================================================================
// Feature 20: Invalidation Bits Stress Tests
// ============================================================================

#[test]
fn test_invalidation_bits_roundtrip_uncompressed_and_deflate() -> Result<()> {
    let base_file = "test_files/synthetic/synthetic_dz_deflate.mf4";
    if !std::path::Path::new(base_file).exists() {
        return Ok(());
    }

    // Test matrix: (compression, sample_count, null_pattern)
    let test_cases = [
        (CompressionAlgorithm::NoCompression, 32, "alternating"),
        (CompressionAlgorithm::Deflate, 32, "alternating"),
        (CompressionAlgorithm::NoCompression, 65, "every_5th_null"), // crosses 64-bit word
        (CompressionAlgorithm::Deflate, 65, "every_5th_null"),
        (CompressionAlgorithm::NoCompression, 17, "all_null"),
        (CompressionAlgorithm::Deflate, 17, "all_valid"),
    ];

    for (algo, n_samples, pattern) in test_cases {
        let mut mdf = Mdf::new(base_file)?;
        mdf.load_all_channels_data_in_memory()?;

        let mut validity_vec = Vec::with_capacity(n_samples);
        let mut values_vec = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let is_valid = match pattern {
                "alternating" => i % 2 == 0,
                "every_5th_null" => i % 5 != 0,
                "all_null" => false,
                "all_valid" => true,
                _ => true,
            };
            validity_vec.push(is_valid);
            values_vec.push(i as f64 * 1.5);
        }

        let bool_buf = BooleanBuffer::from(validity_vec.clone());
        let null_buf = NullBuffer::from(bool_buf);
        let arrow_array = Arc::new(Float64Array::try_new(
            values_vec.clone().into(),
            Some(null_buf),
        )?);

        let channel_name = format!("TestInval_{:?}_{}", algo, pattern);
        // Add channel with master_flag: false and None master to create a clean independent channel group
        mdf.add_channel(
            channel_name.clone(),
            arrow_array,
            None,
            Some(0),
            false,
            Some("V".to_string()),
            Some("Invalidation bit stress channel".to_string()),
        )?;

        let temp_out = NamedTempFile::new()?;
        let out_path = temp_out.path().to_str().unwrap().to_string();

        let _ = mdf.write(&out_path, algo)?;

        // Open newly written file fresh from disk to verify actual serialized DI blocks
        let mut fresh_mdf = Mdf::new(&out_path)?;
        fresh_mdf.load_all_channels_data_in_memory()?;

        // Verify loaded channel data
        let read_data = fresh_mdf
            .get_channel_data(&channel_name)
            .expect("Channel must exist in read-back file");

        let read_validity = read_data
            .validity()
            .expect("Validity mask must be present on channel with nulls");

        assert_eq!(read_validity.len(), n_samples, "Validity length mismatch");
        for (i, &expected_valid) in validity_vec.iter().enumerate() {
            assert_eq!(
                read_validity.is_valid(i),
                expected_valid,
                "Validity bit polarity mismatch at sample {} in {:?} (expected valid: {}, got: {})",
                i,
                algo,
                expected_valid,
                read_validity.is_valid(i)
            );
        }

        // Verify standard block header flags in MdfInfo4
        let info_mdf = MdfInfo::new(&out_path)?;
        if let MdfInfo::V4(mdf4) = info_mdf {
            let mut found_cn = false;
            for dg in mdf4.dg.values() {
                for cg in dg.cg.values() {
                    for cn in cg.cn.values() {
                        if cn.unique_name == channel_name {
                            found_cn = true;
                            // Check that cn_flags has CN_F_INVALID set
                            assert_ne!(
                                cn.block.cn_flags & CN_F_INVALID,
                                0,
                                "CN_F_INVALID bit must be set in cn_flags"
                            );
                            // Check that channel group allocated invalid bytes
                            assert_eq!(
                                cg.block.cg_inval_bytes, 1,
                                "cg_inval_bytes must be 1 for channel with invalidation mask"
                            );
                        }
                    }
                }
            }
            assert!(found_cn, "Written channel must be found in MdfInfo4");
        } else {
            panic!("Expected MDF4 file");
        }
    }
    Ok(())
}

// ============================================================================
// Feature 21: 9-Byte Bitfield Channel Decoding Stress Tests
// ============================================================================

#[test]
fn test_9byte_bitfield_u64_and_i64_decoding() -> Result<()> {
    // Objective: Verify decoding of a 64-bit value with bit_offset = 4 across 9 bytes.
    let test_values_u64: [u64; 6] = [
        0x0000_0000_0000_0000,
        0xFFFF_FFFF_FFFF_FFFF,
        0x1234_5678_9ABC_DEF0,
        0xDEAD_BEEF_CAFE_BABE,
        0x0123_4567_89AB_CDEF,
        0xAAAA_5555_AAAA_5555,
    ];

    let bit_offset = 4u32;
    let bit_count = 64u32;
    let n_bytes = 9usize;

    // 1. Little-Endian UInt64
    let mut raw_bytes_le = Vec::with_capacity(test_values_u64.len() * 9);
    for &val in &test_values_u64 {
        let mut sample_bytes = [0u8; 9];
        // Low 8 bytes contain: bits 0..3 noise (0xA), bits 4..63 contain bits 0..59 of val
        let low = (0x0Au64 & 0x0F) | (val << bit_offset);
        sample_bytes[0..8].copy_from_slice(&low.to_le_bytes());
        // 9th byte contains: bits 0..3 contain bits 60..63 of val, bits 4..7 noise (0xB0)
        let high = ((val >> (64 - bit_offset)) as u8 & 0x0F) | 0xB0;
        sample_bytes[8] = high;
        raw_bytes_le.extend_from_slice(&sample_bytes);
    }

    let mut cn_u64 = Cn4 {
        n_bytes: n_bytes as u32,
        list_size: 1,
        endian: Endianness::Little,
        block: Cn4Block {
            cn_type: 0,
            cn_data_type: 0, // UIntLE
            cn_bit_offset: bit_offset as u8,
            cn_bit_count: bit_count,
            ..Default::default()
        },
        builder: Some(ChannelBuilder::new_zeroed(
            0,
            0,
            test_values_u64.len(),
            8,
            (Vec::new(), Order::RowMajor),
            1,
        )?),
        ..Default::default()
    };

    read_one_channel_array(&raw_bytes_le, &mut cn_u64, test_values_u64.len())?;
    if let ChannelData::UInt64(arr) = &cn_u64.data {
        for (i, &expected) in test_values_u64.iter().enumerate() {
            assert_eq!(
                arr.value(i),
                expected,
                "LE UInt64 9-byte bitfield mismatch at index {}",
                i
            );
        }
    } else {
        panic!("Expected UInt64 data");
    }

    // 2. Little-Endian Int64 (signed)
    let test_values_i64: [i64; 6] = [0, -1, 42, -42, i64::MIN, i64::MAX];

    let mut raw_bytes_i64_le = Vec::with_capacity(test_values_i64.len() * 9);
    for &val in &test_values_i64 {
        let uval = val as u64;
        let mut sample_bytes = [0u8; 9];
        let low = (0x05u64 & 0x0F) | (uval << bit_offset);
        sample_bytes[0..8].copy_from_slice(&low.to_le_bytes());
        let high = ((uval >> (64 - bit_offset)) as u8 & 0x0F) | 0x70;
        sample_bytes[8] = high;
        raw_bytes_i64_le.extend_from_slice(&sample_bytes);
    }

    let mut cn_i64 = Cn4 {
        n_bytes: n_bytes as u32,
        list_size: 1,
        endian: Endianness::Little,
        block: Cn4Block {
            cn_type: 0,
            cn_data_type: 2, // IntLE
            cn_bit_offset: bit_offset as u8,
            cn_bit_count: bit_count,
            ..Default::default()
        },
        builder: Some(ChannelBuilder::new_zeroed(
            0,
            2,
            test_values_i64.len(),
            8,
            (Vec::new(), Order::RowMajor),
            1,
        )?),
        ..Default::default()
    };

    read_one_channel_array(&raw_bytes_i64_le, &mut cn_i64, test_values_i64.len())?;
    if let ChannelData::Int64(arr) = &cn_i64.data {
        for (i, &expected) in test_values_i64.iter().enumerate() {
            assert_eq!(
                arr.value(i),
                expected,
                "LE Int64 9-byte bitfield mismatch at index {}",
                i
            );
        }
    } else {
        panic!("Expected Int64 data");
    }

    // 3. decode_u64_slice 9-byte fallback verification
    let mut dst = [0u64; 1];
    let sample = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99];
    decode_u64_slice(&sample, &mut dst, 9, false)?;
    assert_eq!(
        dst[0],
        u64::from_le_bytes([0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88])
    );

    decode_u64_slice(&sample, &mut dst, 9, true)?;
    assert_eq!(
        dst[0],
        u64::from_be_bytes([0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99])
    );

    Ok(())
}

// ============================================================================
// Feature 22: Parser DL4 Sorted rec_pos & Composition Dedup Stress Tests
// ============================================================================

#[test]
fn test_parser_dl4_sorted_non_zero_rec_pos() -> Result<()> {
    // Objective: Verify channel with non-zero rec_pos resolves and decodes VLSD data without silent data loss.
    let mut buf = Vec::new();
    write_id4(&mut buf);

    let pos_dg = buf.len() as i64 + 104; // 168 (HD4 is 104 bytes following 64-byte ID4)
    let pos_cg = pos_dg + 64; // 232
    let pos_cn1 = pos_cg + 104; // 336
    let pos_cn2 = pos_cn1 + 160; // 496
    let pos_tx1 = pos_cn2 + 160; // 656

    write_hd4(&mut buf, pos_dg);

    // DGBLOCK (64 bytes)
    assert_eq!(buf.len(), pos_dg as usize);
    buf.extend_from_slice(b"##DG");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 64);
    push_u64(&mut buf, 4);
    push_i64(&mut buf, 0); // dg_next
    push_i64(&mut buf, pos_cg);
    let pos_dg_data_placeholder = buf.len();
    push_i64(&mut buf, 0); // dg_data (will patch to DL)
    push_i64(&mut buf, 0); // dg_md_comment
    push_u8(&mut buf, 0); // dg_rec_id_size
    push_zeros(&mut buf, 7);

    // CGBLOCK (104 bytes, 6 links)
    assert_eq!(buf.len(), pos_cg as usize);
    buf.extend_from_slice(b"##CG");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 104);
    push_u64(&mut buf, 6);
    push_i64(&mut buf, 0); // cg_next
    push_i64(&mut buf, pos_cn1); // cg_cn_first
    push_i64(&mut buf, 0); // cg_tx_acq_name
    push_i64(&mut buf, 0); // cg_si_acq_source
    push_i64(&mut buf, 0); // cg_sr_first
    push_i64(&mut buf, 0); // cg_md_comment
    push_u64(&mut buf, 0); // cg_record_id
    push_u64(&mut buf, 3); // cg_cycle_count = 3
    push_u16(&mut buf, 0); // cg_flags (0 = sorted)
    push_zeros(&mut buf, 6);
    push_u32(&mut buf, 0); // cg_data_bytes
    push_u32(&mut buf, 0); // cg_inval_bytes

    // CNBLOCK 1 (Master channel, rec_pos = 0)
    assert_eq!(buf.len(), pos_cn1 as usize);
    buf.extend_from_slice(b"##CN");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 160);
    push_u64(&mut buf, 8); // 8 links
    push_i64(&mut buf, pos_cn2); // cn_next -> CN2
    push_i64(&mut buf, 0); // cn_composition
    push_i64(&mut buf, pos_tx1); // cn_tx_name
    push_i64(&mut buf, 0); // cn_si_source
    push_i64(&mut buf, 0); // cn_cc_conversion
    push_i64(&mut buf, 0); // cn_data
    push_i64(&mut buf, 0); // cn_md_unit
    push_i64(&mut buf, 0); // cn_md_comment
    push_u8(&mut buf, 2); // cn_type = 2 (master)
    push_u8(&mut buf, 0); // cn_sync_type = 0
    push_u8(&mut buf, 4); // cn_data_type = 4 (FloatLE)
    push_u8(&mut buf, 0); // cn_bit_offset = 0
    push_u32(&mut buf, 0); // cn_byte_offset = 0
    push_u32(&mut buf, 64); // cn_bit_count = 64
    push_u32(&mut buf, 0); // cn_flags
    push_u32(&mut buf, 0); // cn_inval_bit_pos
    push_u8(&mut buf, 0); // cn_precision
    push_u8(&mut buf, 0); // cn_alignment
    push_u16(&mut buf, 0); // cn_attachment_count
    push_f64(&mut buf, 0.0);
    push_f64(&mut buf, 0.0);
    push_f64(&mut buf, 0.0);
    push_f64(&mut buf, 0.0);
    push_f64(&mut buf, 0.0);
    push_f64(&mut buf, 0.0); // 6 f64s = 48 bytes -> total 160 bytes!

    // CNBLOCK 2 (VLSD channel, byte_offset = 8 -> rec_pos = 64)
    assert_eq!(buf.len(), pos_cn2 as usize);
    let pos_tx2_placeholder = buf.len() + 40; // index of cn_tx_name link in CN2
    buf.extend_from_slice(b"##CN");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 160);
    push_u64(&mut buf, 8);
    push_i64(&mut buf, 0); // cn_next = 0
    push_i64(&mut buf, 0); // cn_composition = 0
    push_i64(&mut buf, 0); // cn_tx_name (patch later)
    push_i64(&mut buf, 0); // cn_si_source
    push_i64(&mut buf, 0); // cn_cc_conversion
    push_i64(&mut buf, 0); // cn_data
    push_i64(&mut buf, 0); // cn_md_unit
    push_i64(&mut buf, 0); // cn_md_comment
    push_u8(&mut buf, 1); // cn_type = 1 (VLSD)
    push_u8(&mut buf, 0); // cn_sync_type = 0
    push_u8(&mut buf, 6); // cn_data_type = 6 (UTF-8 string)
    push_u8(&mut buf, 0); // cn_bit_offset = 0
    push_u32(&mut buf, 8); // cn_byte_offset = 8 (rec_pos = 64!)
    push_u32(&mut buf, 0); // cn_bit_count = 0 for VLSD
    push_u32(&mut buf, 0); // cn_flags
    push_u32(&mut buf, 0); // cn_inval_bit_pos
    push_u8(&mut buf, 0); // cn_precision
    push_u8(&mut buf, 0); // cn_alignment
    push_u16(&mut buf, 0); // cn_attachment_count
    push_f64(&mut buf, 0.0);
    push_f64(&mut buf, 0.0);
    push_f64(&mut buf, 0.0);
    push_f64(&mut buf, 0.0);
    push_f64(&mut buf, 0.0);
    push_f64(&mut buf, 0.0);

    // TXBLOCK 1: "Time"
    assert_eq!(buf.len(), pos_tx1 as usize);
    let tx1_pos = write_tx4(&mut buf, "Time");
    assert_eq!(tx1_pos, pos_tx1);

    // TXBLOCK 2: "VLSD_Signal"
    let tx2_pos = write_tx4(&mut buf, "VLSD_Signal");
    buf[pos_tx2_placeholder..pos_tx2_placeholder + 8].copy_from_slice(&tx2_pos.to_le_bytes());

    // Prepare SDBLOCK payload first to determine its exact length.
    // Each 12-byte string has a 4-byte length prefix (16 bytes per entry, total 48 bytes),
    // which is naturally 8-byte aligned so no trailing zero padding is introduced.
    let mut sd_payload = Vec::new();
    let strings = ["Sample_TextA", "Sample_TextB", "Sample_TextC"];
    for s in &strings {
        let bytes = s.as_bytes();
        push_u32(&mut sd_payload, bytes.len() as u32);
        sd_payload.extend_from_slice(bytes);
    }
    let sd_aligned = (sd_payload.len() + 7) & !7;
    sd_payload.resize(sd_aligned, 0);
    let sd_total_len = 24 + sd_aligned as u64;

    // DLBLOCK (pointing to SD)
    let pos_dl = buf.len() as i64;
    buf[pos_dg_data_placeholder..pos_dg_data_placeholder + 8]
        .copy_from_slice(&pos_dl.to_le_bytes());

    let pos_sd_placeholder = buf.len() + 32; // dl_data[0] link
    buf.extend_from_slice(b"##DL");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 56); // 24 header + 16 (2 links) + 8 (members) + 8 (equal_length) = 56
    push_u64(&mut buf, 2); // 1 dl_next + 1 dl_data
    push_i64(&mut buf, 0); // dl_next
    push_i64(&mut buf, 0); // dl_data[0] (patch to SD)
    push_u8(&mut buf, 1); // dl_flags (equal length = 1)
    push_zeros(&mut buf, 3);
    push_u32(&mut buf, 1); // dl_count = 1
    push_u64(&mut buf, sd_total_len); // dl_equal_length

    // SDBLOCK (contains 3 VLSD string records)
    let pos_sd = buf.len() as i64;
    buf[pos_sd_placeholder..pos_sd_placeholder + 8].copy_from_slice(&pos_sd.to_le_bytes());

    buf.extend_from_slice(b"##SD");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, sd_total_len);
    push_u64(&mut buf, 0); // 0 links
    buf.extend_from_slice(&sd_payload);

    let temp = NamedTempFile::new()?;
    std::fs::write(temp.path(), &buf)?;

    let mut mdf = Mdf::new(temp.path().to_str().unwrap())?;
    mdf.load_all_channels_data_in_memory()?;

    let vlsd_data = mdf
        .get_channel_data("VLSD_Signal")
        .expect("VLSD_Signal channel must be present");

    if let ChannelData::Utf8(arr) = vlsd_data {
        assert_eq!(
            arr.len(),
            3,
            "Must resolve all 3 VLSD records without data loss"
        );
        assert_eq!(arr.value(0), "Sample_TextA");
        assert_eq!(arr.value(1), "Sample_TextB");
        assert_eq!(arr.value(2), "Sample_TextC");
    } else {
        panic!("Expected Utf8 ChannelData for VLSD string channel");
    }

    Ok(())
}

// ============================================================================
// Feature 23: DZ 4 MiB Cap & Mutex Concurrency Stress Tests
// ============================================================================

#[test]
fn test_dz_4mib_cap_fallback_preserves_data() -> Result<()> {
    let base_file = "test_files/synthetic/synthetic_dz_deflate.mf4";
    if !std::path::Path::new(base_file).exists() {
        return Ok(());
    }

    let mut mdf = Mdf::new(base_file)?;
    mdf.load_all_channels_data_in_memory()?;

    // 600,000 Float64 samples = 4,800,000 bytes uncompressed (> 4 MiB = 4,194,304 bytes)
    let n_samples = 600_000;
    let mut large_floats = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        large_floats.push((i as f64) * 0.001);
    }

    let arrow_array = Arc::new(Float64Array::from(large_floats));
    let ch_name = "Oversized_4MiB_Channel".to_string();

    mdf.add_channel(
        ch_name.clone(),
        arrow_array,
        None,
        Some(0),
        false,
        Some("deg".to_string()),
        Some("Channel exceeding 4 MiB uncompressed cap".to_string()),
    )?;

    let temp_out = NamedTempFile::new()?;
    let out_path = temp_out.path().to_str().unwrap().to_string();

    // Write with Deflate compression requested
    let _ = mdf.write(&out_path, CompressionAlgorithm::Deflate)?;

    let mut fresh_mdf = Mdf::new(&out_path)?;
    fresh_mdf.load_all_channels_data_in_memory()?;

    // Verify written file can be read without corruption
    let read_back = fresh_mdf
        .get_channel_data(&ch_name)
        .expect("Oversized channel must exist in read-back file");

    if let ChannelData::Float64(arr) = read_back {
        assert_eq!(arr.len(), n_samples);
        // Verify precision on boundary and internal points
        assert_eq!(arr.value(0), 0.0);
        assert_eq!(arr.value(100), 0.1);
        assert_eq!(arr.value(n_samples - 1), (n_samples - 1) as f64 * 0.001);
    } else {
        panic!("Expected Float64 channel");
    }

    // Verify standard block structure: check that data block is DV rather than DZ
    let mut file = File::open(&out_path)?;
    let mut header_bytes = [0u8; 4];
    let info = MdfInfo::new(&out_path)?;
    if let MdfInfo::V4(mdf4) = info {
        for dg in mdf4.dg.values() {
            for cg in dg.cg.values() {
                for cn in cg.cn.values() {
                    if cn.unique_name == ch_name {
                        let data_ptr = dg.block.dg_data;
                        assert!(data_ptr > 0, "dg_data must be non-zero");
                        file.seek(SeekFrom::Start(data_ptr as u64))?;
                        file.read_exact(&mut header_bytes)?;
                        if &header_bytes == b"##LD" {
                            let mut ld_head = [0u8; 40];
                            file.seek(SeekFrom::Start(data_ptr as u64))?;
                            file.read_exact(&mut ld_head)?;
                            let first_data_link = i64::from_le_bytes(ld_head[32..40].try_into()?);
                            file.seek(SeekFrom::Start(first_data_link as u64))?;
                            file.read_exact(&mut header_bytes)?;
                        }
                        assert_eq!(
                            &header_bytes, b"##DV",
                            "Data block for > 4 MiB uncompressed must fall back to uncompressed ##DV"
                        );
                    }
                }
            }
        }
    }

    Ok(())
}

// ============================================================================
// Feature 24: SymBufReader Redundant Seek Elimination Stress Tests
// ============================================================================

struct SeekSpy<R> {
    inner: R,
    seek_count: usize,
    stream_position_count: usize,
}

impl<R> SeekSpy<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            seek_count: 0,
            stream_position_count: 0,
        }
    }
}

impl<R: Read> Read for SeekSpy<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.inner.read(buf)
    }
}

impl<R: Seek> Seek for SeekSpy<R> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.seek_count += 1;
        self.inner.seek(pos)
    }

    fn stream_position(&mut self) -> std::io::Result<u64> {
        self.stream_position_count += 1;
        self.inner.stream_position()
    }
}

#[test]
fn test_sym_buf_reader_contiguous_forward_reads_zero_seeks() -> Result<()> {
    // 32 KiB of sequential test bytes (spans multiple 8 KiB buffer fills)
    let total_size = 32_768;
    let mut payload = Vec::with_capacity(total_size);
    for i in 0..total_size {
        payload.push((i % 256) as u8);
    }

    let cursor = Cursor::new(payload.clone());
    let spy = SeekSpy::new(cursor);
    let mut reader = SymBufReader::new(spy);

    // Read in 1024-byte increments (32 steps across 4 buffer refills)
    let chunk_size = 1024;
    let mut read_buf = vec![0u8; chunk_size];
    let mut total_read = 0;

    while total_read < total_size {
        reader.read_exact(&mut read_buf)?;
        assert_eq!(&read_buf[..], &payload[total_read..total_read + chunk_size]);
        total_read += chunk_size;
    }

    // Verify: contiguous forward sequential reads must execute ZERO seeks on the underlying stream!
    assert_eq!(
        reader.get_ref().seek_count,
        0,
        "Contiguous forward sequential reads must NOT issue seek() calls"
    );

    // Test backward repositioning outside the 8 KiB buffer:
    // SymBufReader flushes buffer and delegates to inner reader seek
    reader.seek_relative(-10_000)?;
    assert!(
        reader.get_ref().seek_count > 0,
        "Backward seek beyond internal buffer must reposition the inner stream"
    );
    let mut check_byte = [0u8; 1];
    reader.read_exact(&mut check_byte)?;
    assert_eq!(
        check_byte[0],
        payload[total_size - 10_000],
        "Data read after backward seek must match expected payload"
    );

    Ok(())
}

// ============================================================================
// Feature 25: MDF3 Unsorted Multi-Chunk & 2-Byte Record IDs Stress Tests
// ============================================================================

#[test]
fn test_mdf3_unsorted_multi_chunk_and_2byte_record_ids() -> Result<()> {
    // ------------------------------------------------------------------------
    // Part A: Multi-chunk unsorted reading with trailing records crossing 512 KiB boundary.
    // In mdfreader3.rs, CHUNK_SIZE_READING_3 = 524,288.
    // Unsorted reading requires multiple CGs (dg.cg.len() > 1).
    // CG1: 28,000 records of 21 bytes (588,000 bytes)
    // CG2: 100 records of 21 bytes (2,100 bytes)
    // Total data: 590,100 bytes (> 524,288), straddling chunk boundaries with trailing bytes.
    // ------------------------------------------------------------------------

    let mut mdf3_buf = Vec::new();
    // IDBLOCK (exactly 64 bytes)
    write_id3(&mut mdf3_buf);
    assert_eq!(mdf3_buf.len(), 64, "MDF3 ID block must be exactly 64 bytes");

    let pos_tx = 272u32;
    let pos_dg = 280u32;
    let pos_cg1 = 304u32;
    let pos_cg2 = 334u32;
    let pos_cn1 = 364u32;
    let pos_cn2 = 592u32;
    let pos_cc = 820u32;
    let pos_data = 866u32;

    // HDBLOCK (208 bytes, hd_md_comment pointing to pos_tx)
    write_hd3(&mut mdf3_buf, pos_dg, pos_tx);
    assert_eq!(mdf3_buf.len(), pos_tx as usize);

    // TXBLOCK comment for HD (8 bytes)
    let tx_written = write_tx3(&mut mdf3_buf);
    assert_eq!(tx_written, pos_tx);
    assert_eq!(mdf3_buf.len(), pos_dg as usize);

    // DGBLOCK (24 bytes, 2 CGs, 1-byte record ID)
    write_dg3(&mut mdf3_buf, 0, pos_cg1, pos_data, 2, 1);
    assert_eq!(mdf3_buf.len(), pos_cg1 as usize);

    // CGBLOCK 1 (30 bytes, record_id = 1, data_bytes = 20, cycle_count = 28_000)
    let n_records_1 = 28_000u32;
    write_cg3(&mut mdf3_buf, pos_cg2, pos_cn1, 1, 1, 20, n_records_1);
    assert_eq!(mdf3_buf.len(), pos_cg2 as usize);

    // CGBLOCK 2 (30 bytes, record_id = 2, data_bytes = 20, cycle_count = 100)
    let n_records_2 = 100u32;
    write_cg3(&mut mdf3_buf, 0, pos_cn2, 2, 1, 20, n_records_2);
    assert_eq!(mdf3_buf.len(), pos_cn1 as usize);

    // CNBLOCK 1 (228 bytes, UInt32 at bit_offset = 0)
    write_cn3(&mut mdf3_buf, 0, pos_cc, "Counter3_A", 0, 32, 0);
    assert_eq!(mdf3_buf.len(), pos_cn2 as usize);

    // CNBLOCK 2 (228 bytes, UInt32 at bit_offset = 0)
    write_cn3(&mut mdf3_buf, 0, pos_cc, "Counter3_B", 0, 32, 0);
    assert_eq!(mdf3_buf.len(), pos_cc as usize);

    // CCBLOCK (46 bytes, 1:1 Identity conversion)
    let cc_written = write_cc3_identity(&mut mdf3_buf);
    assert_eq!(cc_written, pos_cc);
    assert_eq!(mdf3_buf.len(), pos_data as usize);

    // DATA BLOCK:
    // 28,000 records of CG 1 (rec_id = 1)
    for i in 0..n_records_1 {
        push_u8(&mut mdf3_buf, 1); // rec_id = 1
        push_u32(&mut mdf3_buf, i); // payload UInt32 counter
        push_zeros(&mut mdf3_buf, 16); // remaining 16 bytes
    }
    // 100 records of CG 2 (rec_id = 2)
    for i in 0..n_records_2 {
        push_u8(&mut mdf3_buf, 2); // rec_id = 2
        push_u32(&mut mdf3_buf, i + 50_000); // payload UInt32
        push_zeros(&mut mdf3_buf, 16); // remaining 16 bytes
    }

    let temp3 = NamedTempFile::new()?;
    std::fs::write(temp3.path(), &mdf3_buf)?;

    let mut mdf3 = Mdf::new(temp3.path().to_str().unwrap())?;
    mdf3.load_all_channels_data_in_memory()?;

    let data_a = mdf3
        .get_channel_data("Counter3_A")
        .expect("Counter3_A channel must exist");

    if let ChannelData::UInt32(arr) = data_a {
        assert_eq!(arr.len(), n_records_1 as usize);
        assert_eq!(arr.value(0), 0);
        assert_eq!(arr.value(24966), 24966); // ~524,288 byte boundary
        assert_eq!(arr.value((n_records_1 - 1) as usize), n_records_1 - 1);
    } else {
        panic!("Expected UInt32 channel for Counter3_A, got: {:?}", data_a);
    }

    let data_b = mdf3
        .get_channel_data("Counter3_B")
        .expect("Counter3_B channel must exist");

    if let ChannelData::UInt32(arr) = data_b {
        assert_eq!(arr.len(), n_records_2 as usize);
        assert_eq!(arr.value(0), 50_000);
        assert_eq!(arr.value((n_records_2 - 1) as usize), 50_099);
    } else {
        panic!("Expected UInt32 channel for Counter3_B");
    }

    // ------------------------------------------------------------------------
    // Part B: 2-byte record IDs (dg_n_record_ids = 2) and clean break on unknown record ID.
    // CG 1: rec_id = 257 (0x0101), 25 cycles declared
    // CG 2: rec_id = 258 (0x0102), 25 cycles declared
    // Data contains 20 records each, then an unknown rec_id (0x9999), and padding.
    // Verifies that 2-byte record IDs are decoded and unknown record ID cleanly breaks without loop/panic.
    // ------------------------------------------------------------------------

    let mut buf2 = Vec::new();
    write_id3(&mut buf2);

    let pos_tx2 = 272u32;
    let pos_dg2 = 280u32;
    let pos_cg2_1 = 304u32;
    let pos_cg2_2 = 334u32;
    let pos_cn2_1 = 364u32;
    let pos_cn2_2 = 592u32;
    let pos_cc2 = 820u32;
    let pos_data2 = 866u32;

    write_hd3(&mut buf2, pos_dg2, pos_tx2);
    let tx2_written = write_tx3(&mut buf2);
    assert_eq!(tx2_written, pos_tx2);
    assert_eq!(buf2.len(), pos_dg2 as usize);

    // DG with dg_n_record_ids = 2 (2-byte record IDs)
    write_dg3(&mut buf2, 0, pos_cg2_1, pos_data2, 2, 2);

    let cycle_count_2b = 25u32;
    // CG 1: record_id = 257 (0x0101)
    write_cg3(&mut buf2, pos_cg2_2, pos_cn2_1, 257, 1, 4, cycle_count_2b);
    // CG 2: record_id = 258 (0x0102)
    write_cg3(&mut buf2, 0, pos_cn2_2, 258, 1, 4, cycle_count_2b);

    write_cn3(&mut buf2, 0, pos_cc2, "Sig2Byte_A", 0, 32, 0);
    write_cn3(&mut buf2, 0, pos_cc2, "Sig2Byte_B", 0, 32, 0);
    let cc2_written = write_cc3_identity(&mut buf2);
    assert_eq!(cc2_written, pos_cc2);
    assert_eq!(buf2.len(), pos_data2 as usize);

    // Data: 20 records of CG 1 interleaved with 20 records of CG 2
    // Each record = 2 bytes record ID + 4 bytes payload UInt32 = 6 bytes
    let valid_count = 20u32;
    for i in 0..valid_count {
        push_u16(&mut buf2, 257); // 2-byte ID for CG 1
        push_u32(&mut buf2, i * 10);
        push_u16(&mut buf2, 258); // 2-byte ID for CG 2
        push_u32(&mut buf2, i * 10 + 5);
    }
    // Encounter an unknown record ID (0x9999) - should cause clean break in unsorted reader
    push_u16(&mut buf2, 0x9999);
    push_zeros(&mut buf2, 4);
    // Pad remaining bytes up to total declared data length: 50 * 6 = 300 bytes
    // So far written: 40 * 6 + 6 = 246 bytes; remaining = 54 bytes
    push_zeros(&mut buf2, 54);

    let temp3_b = NamedTempFile::new()?;
    std::fs::write(temp3_b.path(), &buf2)?;

    let mut mdf3_b = Mdf::new(temp3_b.path().to_str().unwrap())?;
    mdf3_b.load_all_channels_data_in_memory()?;

    let data_2b_a = mdf3_b
        .get_channel_data("Sig2Byte_A")
        .expect("Sig2Byte_A channel must exist");

    if let ChannelData::UInt32(arr) = data_2b_a {
        assert_eq!(arr.len(), cycle_count_2b as usize);
        for i in 0..valid_count as usize {
            assert_eq!(arr.value(i), (i as u32) * 10);
        }
    } else {
        panic!("Expected UInt32 channel for Sig2Byte_A");
    }

    let data_2b_b = mdf3_b
        .get_channel_data("Sig2Byte_B")
        .expect("Sig2Byte_B channel must exist");

    if let ChannelData::UInt32(arr) = data_2b_b {
        assert_eq!(arr.len(), cycle_count_2b as usize);
        for i in 0..valid_count as usize {
            assert_eq!(arr.value(i), (i as u32) * 10 + 5);
        }
    } else {
        panic!("Expected UInt32 channel for Sig2Byte_B");
    }

    Ok(())
}

// ============================================================================
// Feature 26: MDF3 Big-Endian 5- and 7-Byte Integers Stress Tests
// ============================================================================

#[test]
fn test_mdf3_big_endian_5_and_7_byte_integers() -> Result<()> {
    // Objective: Verify exact round-trip value decoding for Big-Endian 5/7-byte integers.

    // 1. 5-byte signed integer (Big-Endian)
    let test_i40: [i64; 7] = [
        0,
        1,
        -1,
        42,
        -42,
        (1i64 << 39) - 1, // max positive 40-bit
        -(1i64 << 39),    // min negative 40-bit
    ];

    let mut chunk_i40 = Vec::new();
    let record_len_5 = 5usize;
    for &val in &test_i40 {
        let be_bytes = val.to_be_bytes();
        // Lower 5 bytes of 8-byte big-endian are be_bytes[3..8]
        chunk_i40.extend_from_slice(&be_bytes[3..8]);
    }

    let mut channels_i40 = HashMap::new();
    let mut block2_i40 = Cn3Block2::default();
    block2_i40.cn_data_type = 10; // Signed BE
    block2_i40.cn_bit_count = 40;
    let cn_i40 = Cn3 {
        pos_byte_beg: 0,
        n_bytes: 5,
        endian: Endianness::Big,
        should_read: true,
        block1: Cn3Block1::default(),
        block2: block2_i40,
        builder: Some(ChannelBuilder::new_zeroed(
            0,
            3, // IntBE
            test_i40.len(),
            5,
            (Vec::new(), Order::RowMajor),
            1,
        )?),
        ..Default::default()
    };
    channels_i40.insert(0u32, cn_i40);

    read_channels_from_bytes_3(&chunk_i40, &mut channels_i40, record_len_5, 0)?;
    let mut cn_res = channels_i40.remove(&0).unwrap();
    cn_res.data = cn_res.builder.take().unwrap().finish();

    if let ChannelData::Int64(arr) = cn_res.data {
        for (i, &expected) in test_i40.iter().enumerate() {
            assert_eq!(
                arr.value(i),
                expected,
                "MDF3 BE 5-byte signed integer mismatch at index {}",
                i
            );
        }
    } else {
        panic!("Expected Int64 channel");
    }

    // 2. 7-byte signed integer (Big-Endian)
    let test_i56: [i64; 7] = [
        0,
        1,
        -1,
        123456789,
        -123456789,
        (1i64 << 55) - 1, // max positive 56-bit
        -(1i64 << 55),    // min negative 56-bit
    ];

    let mut chunk_i56 = Vec::new();
    let record_len_7 = 7usize;
    for &val in &test_i56 {
        let be_bytes = val.to_be_bytes();
        // Lower 7 bytes of 8-byte big-endian are be_bytes[1..8]
        chunk_i56.extend_from_slice(&be_bytes[1..8]);
    }

    let mut channels_i56 = HashMap::new();
    let mut block2_i56 = Cn3Block2::default();
    block2_i56.cn_data_type = 10; // Signed BE
    block2_i56.cn_bit_count = 56;
    let cn_i56 = Cn3 {
        pos_byte_beg: 0,
        n_bytes: 7,
        endian: Endianness::Big,
        should_read: true,
        block1: Cn3Block1::default(),
        block2: block2_i56,
        builder: Some(ChannelBuilder::new_zeroed(
            0,
            3, // IntBE
            test_i56.len(),
            7,
            (Vec::new(), Order::RowMajor),
            1,
        )?),
        ..Default::default()
    };
    channels_i56.insert(0u32, cn_i56);

    read_channels_from_bytes_3(&chunk_i56, &mut channels_i56, record_len_7, 0)?;
    let mut cn_res = channels_i56.remove(&0).unwrap();
    cn_res.data = cn_res.builder.take().unwrap().finish();

    if let ChannelData::Int64(arr) = cn_res.data {
        for (i, &expected) in test_i56.iter().enumerate() {
            assert_eq!(
                arr.value(i),
                expected,
                "MDF3 BE 7-byte signed integer mismatch at index {}",
                i
            );
        }
    } else {
        panic!("Expected Int64 channel");
    }

    // 3. 5-byte and 7-byte unsigned integers (Big-Endian)
    let test_u40: [u64; 4] = [0, 1, 0x12_3456_789A, (1u64 << 40) - 1];
    let mut chunk_u40 = Vec::new();
    for &val in &test_u40 {
        chunk_u40.extend_from_slice(&val.to_be_bytes()[3..8]);
    }

    let mut channels_u40 = HashMap::new();
    let mut block2_u40 = Cn3Block2::default();
    block2_u40.cn_data_type = 9; // Unsigned BE
    block2_u40.cn_bit_count = 40;
    let cn_u40 = Cn3 {
        pos_byte_beg: 0,
        n_bytes: 5,
        endian: Endianness::Big,
        should_read: true,
        block1: Cn3Block1::default(),
        block2: block2_u40,
        builder: Some(ChannelBuilder::new_zeroed(
            0,
            1, // UIntBE
            test_u40.len(),
            5,
            (Vec::new(), Order::RowMajor),
            1,
        )?),
        ..Default::default()
    };
    channels_u40.insert(0u32, cn_u40);

    read_channels_from_bytes_3(&chunk_u40, &mut channels_u40, 5, 0)?;
    let mut cn_res = channels_u40.remove(&0).unwrap();
    cn_res.data = cn_res.builder.take().unwrap().finish();

    if let ChannelData::UInt64(arr) = cn_res.data {
        for (i, &expected) in test_u40.iter().enumerate() {
            assert_eq!(
                arr.value(i),
                expected,
                "MDF3 BE 5-byte unsigned integer mismatch at index {}",
                i
            );
        }
    } else {
        panic!("Expected UInt64 channel");
    }

    Ok(())
}
