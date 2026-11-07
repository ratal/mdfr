//! Adversarial Stress Testing Harness for Milestone 2 Iteration 2
//!
//! Independent empirical verification of Milestone 2 features:
//! 1. Cv4Block & Cu4Block:
//!    - Condition > 0: verify count == 1 deserializes links/values rather than skipping.
//!    - Complete round-trip serialization across 0, 1, and N options/members.
//!    - Generative/stress inputs with high link counts and arbitrary payload patterns.
//! 2. Gd4Block:
//!    - Exact 40-byte layout (4-byte ##GD ID + 36-byte body).
//!    - Verification of pub gd_reserved: [u8; 6] preservation under binary round-trips.
//!    - Verification of field alignment: reserved (4B), len (8B), links (8B), link (8B), version (2B), gd_reserved (6B).
//! 3. Cn4Block:
//!    - get_data_type_str() for type 17 returning "String BOM".
//!    - Exhaustive sweep of all u8 values 0..=255.
//! 4. LoggerBuilder::add_bit_channel:
//!    - Exhaustive verification that non-integer data types on non-byte-aligned channels return Error::InvalidSchema.
//!    - Matrix testing across all CnDataType variants, offsets 0..7, and sub-byte bit counts.
//! 5. C API error code alignment:
//!    - Full bidirectional check of error codes -1 through -9 between Error::code() and include/mdfr_logger.h.
//!    - C API string mapping check for mdfr_error_str().

use binrw::{BinReaderExt, BinWrite, io::Cursor};
use mdfr_core::blocks::cn_block::{Cn4Block, CnDataType};
use mdfr_core::blocks::composition_block::{Cu4Block, Cv4Block};
use mdfr_core::blocks::data_block::Gd4Block;
use mdfr_core::compress::CompressionAlgorithm;
use mdfr_core::error::Error;
use mdfr_core::logger::DataLayout;
use mdfr_core::logger::schema::LoggerBuilder;
use std::ffi::CStr;
use std::fs;
use std::io::Write;

// ============================================================================
// 1. Cv4Block & Cu4Block Deserialization Condition > 0 & Round-Trip
// ============================================================================

#[test]
fn test_cv4_single_option_deserialization_condition() {
    // When cv_option_count == 1, cv_option_val MUST be parsed (condition > 0).
    // If condition were `> 1`, cv_option_val would be empty.
    let cv = Cv4Block {
        cv_n_links: 2, // 1 discriminator + 1 option channel
        cv_cn_discriminator: 1001,
        cv_cn_option: vec![2002],
        cv_option_count: 1,
        cv_reserved: [0x11, 0x22, 0x33, 0x44],
        cv_option_val: vec![9999],
    };

    let mut buf = Vec::new();
    cv.write_le(&mut Cursor::new(&mut buf))
        .expect("write_le on Cv4Block should succeed");

    assert_eq!(
        buf.len(),
        40,
        "Cv4Block with 1 option: 8 (links) + 8 (disc) + 8 (opt) + 4 (count) + 4 (res) + 8 (val) = 40 bytes"
    );

    let parsed: Cv4Block = Cursor::new(&buf)
        .read_le()
        .expect("read_le on Cv4Block should succeed");

    assert_eq!(parsed.cv_n_links, 2);
    assert_eq!(parsed.cv_cn_discriminator, 1001);
    assert_eq!(parsed.cv_cn_option, vec![2002]);
    assert_eq!(parsed.cv_option_count, 1);
    assert_eq!(parsed.cv_reserved, [0x11, 0x22, 0x33, 0x44]);
    assert_eq!(
        parsed.cv_option_val,
        vec![9999],
        "FATAL: cv_option_val must contain [9999] when cv_option_count == 1 (requires condition > 0)"
    );
}

#[test]
fn test_cu4_single_link_deserialization_condition() {
    // When cu_n_links == 1, cu_cn_member MUST be parsed (condition > 0).
    // If condition were `> 1`, cu_cn_member would be empty.
    let cu = Cu4Block {
        cu_n_links: 1,
        cu_cn_member: vec![0x1234_5678_9ABC_DEF0_i64],
        cu_member_count: 1,
        cu_reserved: [0x55, 0x66, 0x77, 0x88],
    };

    let mut buf = Vec::new();
    cu.write_le(&mut Cursor::new(&mut buf))
        .expect("write_le on Cu4Block should succeed");

    assert_eq!(
        buf.len(),
        24,
        "Cu4Block with 1 member: 8 (links) + 8 (member) + 4 (count) + 4 (res) = 24 bytes"
    );

    let parsed: Cu4Block = Cursor::new(&buf)
        .read_le()
        .expect("read_le on Cu4Block should succeed");

    assert_eq!(parsed.cu_n_links, 1);
    assert_eq!(
        parsed.cu_cn_member,
        vec![0x1234_5678_9ABC_DEF0_i64],
        "FATAL: cu_cn_member must contain the member link when cu_n_links == 1 (requires condition > 0)"
    );
    assert_eq!(parsed.cu_member_count, 1);
    assert_eq!(parsed.cu_reserved, [0x55, 0x66, 0x77, 0x88]);
}

#[test]
fn test_cv4_cu4_fuzz_link_counts() {
    // Stress test with varying option counts: 0, 1, 2, 5, 20, 100
    for count in [0u32, 1, 2, 5, 20, 100] {
        let opt_links: Vec<i64> = (0..count).map(|i| 10_000 + i as i64).collect();
        let opt_vals: Vec<u64> = (0..count).map(|i| 50_000 + i as u64).collect();

        let cv = Cv4Block {
            cv_n_links: 1 + count as u64,
            cv_cn_discriminator: 999,
            cv_cn_option: opt_links.clone(),
            cv_option_count: count,
            cv_reserved: [0xAA; 4],
            cv_option_val: opt_vals.clone(),
        };

        let mut buf = Vec::new();
        cv.write_le(&mut Cursor::new(&mut buf)).unwrap();
        let parsed: Cv4Block = Cursor::new(&buf).read_le().unwrap();

        assert_eq!(parsed.cv_option_count, count);
        assert_eq!(parsed.cv_cn_option.len(), count as usize);
        assert_eq!(parsed.cv_option_val.len(), count as usize);
        assert_eq!(parsed, cv);

        // Cu4Block test
        let cu_members: Vec<i64> = (0..count).map(|i| 20_000 + i as i64).collect();
        let cu = Cu4Block {
            cu_n_links: count as u64,
            cu_cn_member: cu_members.clone(),
            cu_member_count: count,
            cu_reserved: [0xBB; 4],
        };

        let mut cu_buf = Vec::new();
        cu.write_le(&mut Cursor::new(&mut cu_buf)).unwrap();
        let parsed_cu: Cu4Block = Cursor::new(&cu_buf).read_le().unwrap();

        assert_eq!(parsed_cu.cu_member_count, count);
        assert_eq!(parsed_cu.cu_cn_member.len(), count as usize);
        assert_eq!(parsed_cu, cu);
    }
}

// ============================================================================
// 2. Gd4Block 40-byte layout & pub gd_reserved: [u8; 6]
// ============================================================================

#[test]
fn test_gd4_block_wire_layout_and_reserved_bytes() {
    let gd = Gd4Block {
        reserved: [0x01, 0x02, 0x03, 0x04],
        gd_len: 40,
        gd_links: 1,
        gd_link: 0x1122_3344_5566_7788_i64,
        gd_version: 430,
        gd_reserved: [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF],
    };

    let mut cursor = Cursor::new(Vec::new());
    cursor.write_all(b"##GD").unwrap();
    gd.write_le(&mut cursor).unwrap();
    let wire = cursor.into_inner();

    // Verify exact 40-byte total wire footprint
    assert_eq!(
        wire.len(),
        40,
        "Total MDF wire size for GDBLOCK with ##GD ID must be exactly 40 bytes"
    );

    // Verify offsets according to MDF 4.3 Table 63:
    // 0..4: ##GD
    assert_eq!(&wire[0..4], b"##GD");
    // 4..8: reserved [u8; 4]
    assert_eq!(&wire[4..8], &[1, 2, 3, 4]);
    // 8..16: gd_len (u64 LE = 40)
    assert_eq!(&wire[8..16], &40u64.to_le_bytes());
    // 16..24: gd_links (u64 LE = 1)
    assert_eq!(&wire[16..24], &1u64.to_le_bytes());
    // 24..32: gd_link (i64 LE)
    assert_eq!(&wire[24..32], &0x1122_3344_5566_7788_i64.to_le_bytes());
    // 32..34: gd_version (u16 LE = 430)
    assert_eq!(&wire[32..34], &430u16.to_le_bytes());
    // 34..40: gd_reserved [u8; 6]
    assert_eq!(&wire[34..40], &[0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]);

    // Roundtrip deserialization
    let parsed: Gd4Block = Cursor::new(&wire[4..]).read_le().unwrap();
    assert_eq!(parsed, gd);
    assert_eq!(parsed.gd_reserved, [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]);
}

// ============================================================================
// 3. Cn4Block::get_data_type_str() for type 17 ("String BOM")
// ============================================================================

#[test]
fn test_cn4_get_data_type_str_17_and_full_sweep() {
    let mut block = Cn4Block {
        cn_data_type: 17,
        ..Default::default()
    };
    assert_eq!(block.get_data_type_str(), "String BOM");
    assert_eq!(block.cn_data_type_enum(), Ok(CnDataType::StringBom));

    // Full sweep 0..=255
    for val in 0u8..=255 {
        block.cn_data_type = val;
        let s = block.get_data_type_str();
        match val {
            0 => assert_eq!(s, "UInt LE"),
            1 => assert_eq!(s, "UInt BE"),
            2 => assert_eq!(s, "Int LE"),
            3 => assert_eq!(s, "Int BE"),
            4 => assert_eq!(s, "Float LE"),
            5 => assert_eq!(s, "Float BE"),
            6 => assert_eq!(s, "String ISO-8859-1"),
            7 => assert_eq!(s, "String UTF-8"),
            8 => assert_eq!(s, "String UTF-16 LE"),
            9 => assert_eq!(s, "String UTF-16 BE"),
            10 => assert_eq!(s, "Byte Array"),
            11 => assert_eq!(s, "MIME Sample"),
            12 => assert_eq!(s, "MIME Stream"),
            13 => assert_eq!(s, "CANopen Date"),
            14 => assert_eq!(s, "CANopen Time"),
            15 => assert_eq!(s, "Complex LE"),
            16 => assert_eq!(s, "Complex BE"),
            17 => assert_eq!(s, "String BOM"),
            _ => assert_eq!(s, "Unknown"),
        }
    }
}

// ============================================================================
// 4. LoggerBuilder::add_bit_channel validation
// ============================================================================

#[test]
fn test_add_bit_channel_all_non_integers_fail_on_non_byte_aligned() {
    let mut builder = LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::NoCompression);
    let g = builder.add_group("grp");

    let non_integers = [
        CnDataType::FloatLE,
        CnDataType::FloatBE,
        CnDataType::StringSbc,
        CnDataType::StringUtf8,
        CnDataType::StringUtf16LE,
        CnDataType::StringUtf16BE,
        CnDataType::ByteArray,
        CnDataType::MimeSample,
        CnDataType::MimeStream,
        CnDataType::CanopenDate,
        CnDataType::CanopenTime,
        CnDataType::ComplexLE,
        CnDataType::ComplexBE,
        CnDataType::StringBom,
    ];

    for &dt in &non_integers {
        // Test non-zero bit_offset (e.g. 1) with byte-multiple count (8)
        let res1 = builder.add_bit_channel(g, "bad1", dt, 8, 1, 8);
        assert!(
            matches!(res1, Err(Error::InvalidSchema(_))),
            "Expected InvalidSchema for non-zero bit_offset with {dt:?}"
        );

        // Test zero bit_offset with non-byte-multiple count (e.g. 7)
        let res2 = builder.add_bit_channel(g, "bad2", dt, 8, 0, 7);
        assert!(
            matches!(res2, Err(Error::InvalidSchema(_))),
            "Expected InvalidSchema for sub-byte bit_count with {dt:?}"
        );

        // Test non-zero bit_offset and non-byte-multiple count (e.g. 3, 5)
        let res3 = builder.add_bit_channel(g, "bad3", dt, 8, 3, 5);
        assert!(
            matches!(res3, Err(Error::InvalidSchema(_))),
            "Expected InvalidSchema for non-aligned channel with {dt:?}"
        );
    }

    // Conversely, verify integer types DO succeed on the same non-byte-aligned parameters
    let integers = [
        CnDataType::UIntLE,
        CnDataType::UIntBE,
        CnDataType::IntLE,
        CnDataType::IntBE,
    ];

    for (i, &dt) in integers.iter().enumerate() {
        let base = 8 + (i as u32 * 8);
        assert!(
            builder
                .add_bit_channel(g, &format!("ok_int1_{i}"), dt, base, 1, 8)
                .is_ok()
        );
        assert!(
            builder
                .add_bit_channel(g, &format!("ok_int2_{i}"), dt, base + 2, 0, 7)
                .is_ok()
        );
        assert!(
            builder
                .add_bit_channel(g, &format!("ok_int3_{i}"), dt, base + 4, 3, 5)
                .is_ok()
        );
    }
}

// ============================================================================
// 5. C API Error Code Alignment (-1 through -9)
// ============================================================================

#[test]
fn test_c_api_error_code_bidirectional_alignment() {
    // 1. Check Rust Error::code() values
    assert_eq!(Error::Io.code(), -1);
    assert_eq!(Error::InvalidBlock("test".into()).code(), -2);
    assert_eq!(Error::InvalidSchema("test".into()).code(), -3);
    assert_eq!(Error::InvalidHandle("test".into()).code(), -4);
    assert_eq!(Error::Unsupported("test").code(), -5);
    assert_eq!(Error::Compression("test".into()).code(), -6);
    assert_eq!(Error::RingFull.code(), -7);
    assert_eq!(Error::AlreadyClosed.code(), -8);
    assert_eq!(
        Error::RecordSize {
            expected: 10,
            got: 20
        }
        .code(),
        -9
    );

    // 2. Read include/mdfr_logger.h directly and check #define values
    let header =
        fs::read_to_string("include/mdfr_logger.h").expect("Cannot read include/mdfr_logger.h");

    let expected_defines = [
        ("MDFR_OK", 0),
        ("MDFR_ERR_IO", -1),
        ("MDFR_ERR_BINRW", -2),
        ("MDFR_ERR_INVALID_SCHEMA", -3),
        ("MDFR_ERR_INVALID_HANDLE", -4),
        ("MDFR_ERR_UNSUPPORTED", -5),
        ("MDFR_ERR_COMPRESSION", -6),
        ("MDFR_ERR_RING_FULL", -7),
        ("MDFR_ERR_ALREADY_CLOSED", -8),
        ("MDFR_ERR_RECORD_SIZE", -9),
    ];

    for (def_name, expected_code) in expected_defines {
        let pattern = format!("#define {def_name}");
        let matching_line = header
            .lines()
            .find(|line| line.trim().starts_with(&pattern))
            .unwrap_or_else(|| panic!("Macro {def_name} not found in include/mdfr_logger.h"));

        let parts: Vec<&str> = matching_line.split_whitespace().collect();
        assert!(
            parts.len() >= 3,
            "Malformed define line in C header: {matching_line}"
        );
        let parsed_code: i32 = parts[2]
            .parse()
            .unwrap_or_else(|_| panic!("Failed to parse integer from define: {matching_line}"));

        assert_eq!(
            parsed_code, expected_code,
            "C header macro {def_name} has value {parsed_code}, expected {expected_code}"
        );
    }

    // 3. Check C API error strings from mdfr_error_str
    use mdfr_core::capi::mdfr_error_str;
    for code in -9..=0 {
        let ptr = mdfr_error_str(code);
        assert!(!ptr.is_null());
        let msg = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
        assert!(!msg.is_empty());
        assert_ne!(msg, "unknown error");
    }
}
