//! Adversarial Test Suite: Milestone 2 Blocks, Wire Format & Schema Validation
//!
//! Stress-tests for Milestone 2 (Embedded Reliability & Blocks: Features 17-19):
//! 1. Cv4Block & Cu4Block link parsing (Feature 17):
//!    - Cv4Block with 0 options, 1 option, and multiple options roundtrip serialization and deserialization.
//!    - Cu4Block with 0 members, 1 member, and multiple members roundtrip serialization and deserialization.
//!    - Verification that count=1 does NOT skip link fields.
//! 2. Gd4Block wire size & alignment (Feature 18):
//!    - Write Gd4Block to a binary cursor and assert exact byte length: 4-byte ##GD + 36-byte body = 40 bytes total.
//!    - Verify all 6 reserved bytes are preserved across serialization.
//!    - Check Cn4Block::get_data_type_str() on type 17 returns "String BOM".
//! 3. LoggerBuilder::add_bit_channel validation (Feature 19):
//!    - Non-byte-aligned channels (bit_offset != 0 or !bit_count.is_multiple_of(8)):
//!      - FloatLE, FloatBE, StringUtf8, ByteArray, etc. MUST return Err(Error::InvalidSchema(_)).
//!      - UIntLE, UIntBE, IntLE, IntBE MUST succeed.
//!    - Byte-aligned bit channels (bit_offset == 0 && bit_count % 8 == 0) with non-integer types (e.g. 32-bit FloatLE) MUST succeed.
//! 4. Panic hardening in conversions & decoders (Feature 19):
//!    - Pass truncated/corrupted cc_val slices to conversion functions (linear_conversion, rational_conversion, value_range_to_value_table).
//!    - Verify graceful error return, zero panics.
#![allow(clippy::field_reassign_with_default)]

use arrow::array::{Float64Builder, Int32Builder, LargeStringBuilder, UInt8Builder};
use binrw::{BinReaderExt, BinWrite, io::Cursor};
use mdfr::data_holder::channel_data::ChannelData;
use mdfr::mdfinfo::mdfinfo4::Cn4;
use mdfr::mdfreader::conversions4::{
    linear_conversion, rational_conversion, value_range_to_value_table,
};
use mdfr_core::blocks::cn_block::{Cn4Block, CnDataType};
use mdfr_core::blocks::composition_block::{Cu4Block, Cv4Block};
use mdfr_core::blocks::data_block::Gd4Block;
use mdfr_core::compress::CompressionAlgorithm;
use mdfr_core::error::Error;
use mdfr_core::logger::DataLayout;
use mdfr_core::logger::schema::{GroupHandle, LoggerBuilder};

// ============================================================================
// Objective 1: Cv4Block & Cu4Block link parsing (Feature 17)
// ============================================================================

#[test]
fn test_cv4_block_0_options_roundtrip() {
    // 0 options: discriminator link present, but 0 option channels and 0 discriminator values
    let cv = Cv4Block {
        cv_n_links: 1, // Only discriminator link
        cv_cn_discriminator: 0x1234_5678,
        cv_cn_option: vec![],
        cv_option_count: 0,
        cv_reserved: [0xAA, 0xBB, 0xCC, 0xDD],
        cv_option_val: vec![],
    };

    let mut buf = Vec::new();
    cv.write_le(&mut Cursor::new(&mut buf))
        .expect("serialization of 0-option Cv4Block should succeed");

    // Expected wire size:
    // cv_n_links: 8 bytes
    // cv_cn_discriminator: 8 bytes
    // cv_cn_option: 0 bytes
    // cv_option_count: 4 bytes
    // cv_reserved: 4 bytes
    // cv_option_val: 0 bytes
    // Total = 24 bytes
    assert_eq!(buf.len(), 24, "Cv4Block with 0 options must be 24 bytes");

    let mut cursor = Cursor::new(&buf);
    let parsed: Cv4Block = cursor
        .read_le()
        .expect("deserialization of 0-option Cv4Block should succeed");

    assert_eq!(parsed.cv_n_links, 1);
    assert_eq!(parsed.cv_cn_discriminator, 0x1234_5678);
    assert!(
        parsed.cv_cn_option.is_empty(),
        "cv_cn_option must be empty for cv_n_links=1"
    );
    assert_eq!(parsed.cv_option_count, 0);
    assert_eq!(parsed.cv_reserved, [0xAA, 0xBB, 0xCC, 0xDD]);
    assert!(
        parsed.cv_option_val.is_empty(),
        "cv_option_val must be empty for cv_option_count=0"
    );
    assert_eq!(cursor.position(), 24);
}

#[test]
fn test_cv4_block_1_option_roundtrip_not_skipped() {
    // Crucial Feature 17 test: verify count=1 does NOT skip option links or option values.
    // In buggy implementations, `cv_option_count > 1` caused cv_option_val to be skipped when count==1.
    let cv = Cv4Block {
        cv_n_links: 2, // 1 discriminator + 1 option link
        cv_cn_discriminator: 0x1000,
        cv_cn_option: vec![0x2000],
        cv_option_count: 1,
        cv_reserved: [1, 2, 3, 4],
        cv_option_val: vec![42],
    };

    let mut buf = Vec::new();
    cv.write_le(&mut Cursor::new(&mut buf))
        .expect("serialization of 1-option Cv4Block should succeed");

    // Wire size: 8 + 8 + 8 (option link) + 4 (count) + 4 (reserved) + 8 (val) = 40 bytes
    assert_eq!(buf.len(), 40, "Cv4Block with 1 option must be 40 bytes");

    let mut cursor = Cursor::new(&buf);
    let parsed: Cv4Block = cursor
        .read_le()
        .expect("deserialization of 1-option Cv4Block should succeed");

    assert_eq!(parsed.cv_n_links, 2);
    assert_eq!(parsed.cv_cn_discriminator, 0x1000);
    assert_eq!(
        parsed.cv_cn_option,
        vec![0x2000],
        "cv_cn_option must contain the single option link"
    );
    assert_eq!(parsed.cv_option_count, 1);
    assert_eq!(parsed.cv_reserved, [1, 2, 3, 4]);
    assert_eq!(
        parsed.cv_option_val,
        vec![42],
        "cv_option_val MUST NOT be skipped when cv_option_count == 1"
    );
    assert_eq!(cursor.position(), 40);
}

#[test]
fn test_cv4_block_multiple_options_roundtrip() {
    let option_links = vec![0x2001, 0x2002, 0x2003, 0x2004, 0x2005];
    let option_vals = vec![100, 200, 300, 400, 500];

    let cv = Cv4Block {
        cv_n_links: 6, // 1 discriminator + 5 options
        cv_cn_discriminator: 0x1000,
        cv_cn_option: option_links.clone(),
        cv_option_count: 5,
        cv_reserved: [0x55, 0x66, 0x77, 0x88],
        cv_option_val: option_vals.clone(),
    };

    let mut buf = Vec::new();
    cv.write_le(&mut Cursor::new(&mut buf))
        .expect("serialization of 5-option Cv4Block should succeed");

    // Wire size: 8 (links count) + 8 (discriminator) + 5*8 (options) + 4 (count) + 4 (reserved) + 5*8 (values) = 104 bytes
    assert_eq!(buf.len(), 104);

    let mut cursor = Cursor::new(&buf);
    let parsed: Cv4Block = cursor
        .read_le()
        .expect("deserialization of 5-option Cv4Block should succeed");

    assert_eq!(parsed.cv_n_links, 6);
    assert_eq!(parsed.cv_cn_discriminator, 0x1000);
    assert_eq!(parsed.cv_cn_option, option_links);
    assert_eq!(parsed.cv_option_count, 5);
    assert_eq!(parsed.cv_reserved, [0x55, 0x66, 0x77, 0x88]);
    assert_eq!(parsed.cv_option_val, option_vals);
    assert_eq!(cursor.position(), 104);
}

#[test]
fn test_cu4_block_0_members_roundtrip() {
    let cu = Cu4Block {
        cu_n_links: 0,
        cu_cn_member: vec![],
        cu_member_count: 0,
        cu_reserved: [0; 4],
    };

    let mut buf = Vec::new();
    cu.write_le(&mut Cursor::new(&mut buf))
        .expect("serialization of 0-member Cu4Block should succeed");

    // Wire size: 8 (links) + 0 (members) + 4 (count) + 4 (reserved) = 16 bytes
    assert_eq!(buf.len(), 16);

    let mut cursor = Cursor::new(&buf);
    let parsed: Cu4Block = cursor
        .read_le()
        .expect("deserialization of 0-member Cu4Block should succeed");

    assert_eq!(parsed.cu_n_links, 0);
    assert!(parsed.cu_cn_member.is_empty());
    assert_eq!(parsed.cu_member_count, 0);
    assert_eq!(parsed.cu_reserved, [0; 4]);
    assert_eq!(cursor.position(), 16);
}

#[test]
fn test_cu4_block_1_member_roundtrip_not_skipped() {
    // Crucial Feature 17 test: verify count=1 does NOT skip member links.
    // In buggy implementations, `cu_n_links > 1` caused cu_cn_member to be skipped when cu_n_links==1.
    let cu = Cu4Block {
        cu_n_links: 1,
        cu_cn_member: vec![0x7AFE_BABE_DEAD_BEEF_i64],
        cu_member_count: 1,
        cu_reserved: [9, 8, 7, 6],
    };

    let mut buf = Vec::new();
    cu.write_le(&mut Cursor::new(&mut buf))
        .expect("serialization of 1-member Cu4Block should succeed");

    // Wire size: 8 (links) + 8 (1 member) + 4 (count) + 4 (reserved) = 24 bytes
    assert_eq!(buf.len(), 24);

    let mut cursor = Cursor::new(&buf);
    let parsed: Cu4Block = cursor
        .read_le()
        .expect("deserialization of 1-member Cu4Block should succeed");

    assert_eq!(parsed.cu_n_links, 1);
    assert_eq!(
        parsed.cu_cn_member,
        vec![0x7AFE_BABE_DEAD_BEEF_i64],
        "cu_cn_member MUST NOT be skipped when cu_n_links == 1"
    );
    assert_eq!(parsed.cu_member_count, 1);
    assert_eq!(parsed.cu_reserved, [9, 8, 7, 6]);
    assert_eq!(cursor.position(), 24);
}

#[test]
fn test_cu4_block_multiple_members_roundtrip() {
    let members = vec![
        0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000, 0x8000,
    ];
    let cu = Cu4Block {
        cu_n_links: 8,
        cu_cn_member: members.clone(),
        cu_member_count: 8,
        cu_reserved: [0xFF, 0xEE, 0xDD, 0xCC],
    };

    let mut buf = Vec::new();
    cu.write_le(&mut Cursor::new(&mut buf))
        .expect("serialization of 8-member Cu4Block should succeed");

    // Wire size: 8 + 8*8 + 4 + 4 = 80 bytes
    assert_eq!(buf.len(), 80);

    let mut cursor = Cursor::new(&buf);
    let parsed: Cu4Block = cursor
        .read_le()
        .expect("deserialization of 8-member Cu4Block should succeed");

    assert_eq!(parsed.cu_n_links, 8);
    assert_eq!(parsed.cu_cn_member, members);
    assert_eq!(parsed.cu_member_count, 8);
    assert_eq!(parsed.cu_reserved, [0xFF, 0xEE, 0xDD, 0xCC]);
    assert_eq!(cursor.position(), 80);
}

#[test]
fn test_cv_cu_truncated_stream_error_handling() {
    // Malformed/truncated streams must return Err without panicking
    let incomplete_cv = [0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]; // claims 2 links, but truncated
    let res: Result<Cv4Block, _> = Cursor::new(&incomplete_cv).read_le();
    assert!(res.is_err());

    let incomplete_cu = [0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]; // claims 5 links, but truncated
    let res: Result<Cu4Block, _> = Cursor::new(&incomplete_cu).read_le();
    assert!(res.is_err());
}

// ============================================================================
// Objective 2: Gd4Block wire size & alignment (Feature 18)
// ============================================================================

#[test]
fn test_gd4_block_wire_size_and_alignment() {
    let gd = Gd4Block {
        reserved: [0x11, 0x22, 0x33, 0x44],
        gd_len: 40,
        gd_links: 1,
        gd_link: 0x0123_4567_89AB_CDEF,
        gd_version: 430,
        gd_reserved: [0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x23],
    };

    use std::io::Write;
    let mut cursor = Cursor::new(Vec::new());
    cursor
        .write_all(b"##GD")
        .expect("writing ##GD magic must succeed");
    gd.write_le(&mut cursor)
        .expect("Gd4Block write_le should succeed");
    let buf = cursor.into_inner();

    // MDF 4.3 Table 63 exact wire size check:
    // 4-byte ##GD ID + 36-byte body = 40 bytes total
    assert_eq!(
        buf.len(),
        40,
        "Total serialized length of Gd4Block with ##GD magic must be exactly 40 bytes"
    );

    // Verify wire byte breakdown:
    // Offset 0..4: "##GD"
    assert_eq!(&buf[0..4], b"##GD");
    // Offset 4..8: reserved [u8; 4]
    assert_eq!(&buf[4..8], &[0x11, 0x22, 0x33, 0x44]);
    // Offset 8..16: gd_len (40u64 LE)
    assert_eq!(&buf[8..16], &40u64.to_le_bytes());
    // Offset 16..24: gd_links (1u64 LE)
    assert_eq!(&buf[16..24], &1u64.to_le_bytes());
    // Offset 24..32: gd_link (i64 LE)
    assert_eq!(&buf[24..32], &0x0123_4567_89AB_CDEF_i64.to_le_bytes());
    // Offset 32..34: gd_version (430u16 LE)
    assert_eq!(&buf[32..34], &430u16.to_le_bytes());
    // Offset 34..40: gd_reserved [u8; 6]
    assert_eq!(&buf[34..40], &[0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x23]);

    // Deserialization roundtrip
    let mut cursor = Cursor::new(&buf[4..]); // Skip ##GD
    let parsed: Gd4Block = cursor
        .read_le()
        .expect("Gd4Block deserialization should succeed");

    assert_eq!(parsed.reserved, [0x11, 0x22, 0x33, 0x44]);
    assert_eq!(parsed.gd_len, 40);
    assert_eq!(parsed.gd_links, 1);
    assert_eq!(parsed.gd_link, 0x0123_4567_89AB_CDEF);
    assert_eq!(parsed.gd_version, 430);
    // Verify all 6 reserved bytes are strictly preserved
    assert_eq!(
        parsed.gd_reserved,
        [0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x23],
        "All 6 reserved bytes must be preserved across serialization"
    );
    assert_eq!(cursor.position(), 36);
}

#[test]
fn test_cn4_get_data_type_str_string_bom_and_all_variants() {
    let mut cn = Cn4Block::default();

    // Specific check: type 17 returns "String BOM"
    cn.cn_data_type = 17;
    assert_eq!(
        cn.get_data_type_str(),
        "String BOM",
        "Cn4Block::get_data_type_str() for type 17 must return 'String BOM'"
    );
    assert_eq!(cn.cn_data_type_enum(), Ok(CnDataType::StringBom));

    // Exhaustive mapping verification for all ASAM MDF 4.3 data types (0..=17)
    let expected_mappings = [
        (0, "UInt LE", Some(CnDataType::UIntLE)),
        (1, "UInt BE", Some(CnDataType::UIntBE)),
        (2, "Int LE", Some(CnDataType::IntLE)),
        (3, "Int BE", Some(CnDataType::IntBE)),
        (4, "Float LE", Some(CnDataType::FloatLE)),
        (5, "Float BE", Some(CnDataType::FloatBE)),
        (6, "String ISO-8859-1", Some(CnDataType::StringSbc)),
        (7, "String UTF-8", Some(CnDataType::StringUtf8)),
        (8, "String UTF-16 LE", Some(CnDataType::StringUtf16LE)),
        (9, "String UTF-16 BE", Some(CnDataType::StringUtf16BE)),
        (10, "Byte Array", Some(CnDataType::ByteArray)),
        (11, "MIME Sample", Some(CnDataType::MimeSample)),
        (12, "MIME Stream", Some(CnDataType::MimeStream)),
        (13, "CANopen Date", Some(CnDataType::CanopenDate)),
        (14, "CANopen Time", Some(CnDataType::CanopenTime)),
        (15, "Complex LE", Some(CnDataType::ComplexLE)),
        (16, "Complex BE", Some(CnDataType::ComplexBE)),
        (17, "String BOM", Some(CnDataType::StringBom)),
    ];

    for (raw, expected_str, expected_enum) in expected_mappings {
        cn.cn_data_type = raw;
        assert_eq!(cn.get_data_type_str(), expected_str);
        assert_eq!(cn.cn_data_type_enum().ok(), expected_enum);
    }

    // Out-of-spec data types return "Unknown" and Err
    for raw in [18, 99, 255] {
        cn.cn_data_type = raw;
        assert_eq!(cn.get_data_type_str(), "Unknown");
        assert!(cn.cn_data_type_enum().is_err());
    }
}

// ============================================================================
// Objective 3: LoggerBuilder::add_bit_channel validation (Feature 19)
// ============================================================================

#[test]
fn test_add_bit_channel_non_byte_aligned_rejects_non_integer_types() {
    let mut builder = LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::NoCompression);
    let g = builder.add_group("test_group");

    let non_integer_types = [
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

    for &dt in &non_integer_types {
        // Case A: non-zero bit_offset (e.g., bit_offset = 1, bit_count = 8)
        let res_offset = builder.add_bit_channel(g, "bad_offset", dt, 8, 1, 8);
        assert!(
            matches!(res_offset, Err(Error::InvalidSchema(_))),
            "Expected Err(InvalidSchema) for non-zero bit_offset with {dt:?}, got {res_offset:?}"
        );

        // Case B: zero bit_offset but bit_count not a multiple of 8 (e.g., bit_offset = 0, bit_count = 7)
        let res_count = builder.add_bit_channel(g, "bad_count", dt, 8, 0, 7);
        assert!(
            matches!(res_count, Err(Error::InvalidSchema(_))),
            "Expected Err(InvalidSchema) for non-multiple-of-8 bit_count with {dt:?}, got {res_count:?}"
        );

        // Case C: both non-zero bit_offset and non-multiple bit_count (e.g., bit_offset = 3, bit_count = 5)
        let res_both = builder.add_bit_channel(g, "bad_both", dt, 8, 3, 5);
        assert!(
            matches!(res_both, Err(Error::InvalidSchema(_))),
            "Expected Err(InvalidSchema) for non-aligned bit channel with {dt:?}, got {res_both:?}"
        );
    }
}

#[test]
fn test_add_bit_channel_non_byte_aligned_accepts_integer_types() {
    let mut builder = LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::NoCompression);
    let g = builder.add_group("test_group");

    let integer_types = [
        CnDataType::UIntLE,
        CnDataType::UIntBE,
        CnDataType::IntLE,
        CnDataType::IntBE,
    ];

    for (idx, &dt) in integer_types.iter().enumerate() {
        // Case A: non-zero bit_offset, byte-sized count
        let ch1 = builder.add_bit_channel(
            g,
            &format!("int_offset_{idx}"),
            dt,
            8 + (idx as u32 * 8),
            1,
            8,
        );
        assert!(
            ch1.is_ok(),
            "Non-zero bit_offset with integer type {dt:?} must succeed"
        );

        // Case B: zero bit_offset, non-multiple bit_count
        let ch2 = builder.add_bit_channel(
            g,
            &format!("int_count_{idx}"),
            dt,
            10 + (idx as u32 * 8),
            0,
            5,
        );
        assert!(
            ch2.is_ok(),
            "Sub-byte bit_count with integer type {dt:?} must succeed"
        );

        // Case C: both non-zero offset and non-multiple count
        let ch3 = builder.add_bit_channel(
            g,
            &format!("int_both_{idx}"),
            dt,
            12 + (idx as u32 * 8),
            3,
            13,
        );
        assert!(
            ch3.is_ok(),
            "Non-aligned bit channel with integer type {dt:?} must succeed"
        );

        // Case D: single-bit flag
        let ch4 = builder.add_bit_channel(
            g,
            &format!("int_flag_{idx}"),
            dt,
            14 + (idx as u32 * 8),
            7,
            1,
        );
        assert!(
            ch4.is_ok(),
            "Single-bit flag with integer type {dt:?} must succeed"
        );
    }
}

#[test]
fn test_add_bit_channel_byte_aligned_accepts_non_integer_types() {
    let mut builder = LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::NoCompression);
    let g = builder.add_group("test_group");

    // When bit_offset == 0 and bit_count % 8 == 0, non-integer types ARE byte-aligned
    // and should be accepted by add_bit_channel.

    // 32-bit FloatLE (4 bytes)
    let ch_f32 = builder.add_bit_channel(g, "float32_aligned", CnDataType::FloatLE, 8, 0, 32);
    assert!(
        ch_f32.is_ok(),
        "32-bit byte-aligned FloatLE must be accepted"
    );

    // 64-bit FloatBE (8 bytes)
    let ch_f64 = builder.add_bit_channel(g, "float64_aligned", CnDataType::FloatBE, 12, 0, 64);
    assert!(
        ch_f64.is_ok(),
        "64-bit byte-aligned FloatBE must be accepted"
    );

    // 16-bit ByteArray (2 bytes)
    let ch_bytes = builder.add_bit_channel(g, "bytes_aligned", CnDataType::ByteArray, 20, 0, 16);
    assert!(
        ch_bytes.is_ok(),
        "16-bit byte-aligned ByteArray must be accepted"
    );

    // 24-bit StringUtf8 (3 bytes)
    let ch_str = builder.add_bit_channel(g, "string_aligned", CnDataType::StringUtf8, 22, 0, 24);
    assert!(
        ch_str.is_ok(),
        "24-bit byte-aligned StringUtf8 must be accepted"
    );
}

#[test]
fn test_add_bit_channel_boundary_conditions() {
    let mut builder = LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::NoCompression);
    let g = builder.add_group("test_group");

    // bit_offset > 7 must fail
    let err_offset = builder.add_bit_channel(g, "offset_8", CnDataType::UIntLE, 8, 8, 8);
    assert!(
        matches!(err_offset, Err(Error::InvalidSchema(_))),
        "bit_offset = 8 must return InvalidSchema"
    );

    // bit_count == 0 must fail
    let err_zero_len = builder.add_bit_channel(g, "zero_bits", CnDataType::UIntLE, 8, 0, 0);
    assert!(
        matches!(err_zero_len, Err(Error::InvalidSchema(_))),
        "bit_count = 0 must return InvalidSchema"
    );

    // bit_offset + bit_count > 64 must fail (exceeds 8-byte maximum word span)
    let err_overflow = builder.add_bit_channel(g, "span_65", CnDataType::UIntLE, 8, 1, 64);
    assert!(
        matches!(err_overflow, Err(Error::InvalidSchema(_))),
        "bit_offset + bit_count = 65 must return InvalidSchema"
    );

    // byte_offset < 8 (overlaps master time channel bytes 0..8) must fail
    for bo in 0..8 {
        let err_master =
            builder.add_bit_channel(g, &format!("overlap_{bo}"), CnDataType::UIntLE, bo, 0, 8);
        assert!(
            matches!(err_master, Err(Error::InvalidSchema(_))),
            "byte_offset {bo} overlapping master channel must return InvalidSchema"
        );
    }

    // Invalid group handle must fail
    let err_group = builder.add_bit_channel(
        GroupHandle(0xDEAD),
        "ghost_group",
        CnDataType::UIntLE,
        8,
        0,
        8,
    );
    assert!(
        matches!(err_group, Err(Error::InvalidSchema(_))),
        "Unknown GroupHandle must return InvalidSchema"
    );
}

// ============================================================================
// Objective 4: Panic hardening in conversions & decoders (Feature 19)
// ============================================================================

fn make_cn4_with_data(data: ChannelData) -> Cn4 {
    Cn4 {
        data,
        ..Default::default()
    }
}

#[test]
fn test_linear_conversion_truncated_cc_val_error() {
    let mut builder = Float64Builder::new();
    builder.append_value(10.0);
    builder.append_value(20.0);
    let mut cn = make_cn4_with_data(ChannelData::Float64(builder.finish()));

    // Truncated: empty slice
    let err0 = linear_conversion(&mut cn, &[]);
    assert!(
        err0.is_err(),
        "linear_conversion with empty cc_val must return Err"
    );
    let msg0 = format!("{:#}", err0.unwrap_err());
    assert!(
        msg0.contains("malformed linear conversion"),
        "Error message should indicate malformed conversion: {msg0}"
    );

    // Truncated: 1 element instead of required 2 (p1, p2)
    let err1 = linear_conversion(&mut cn, &[42.0]);
    assert!(
        err1.is_err(),
        "linear_conversion with 1-element cc_val must return Err"
    );
    let msg1 = format!("{:#}", err1.unwrap_err());
    assert!(
        msg1.contains("expected at least 2 parameters"),
        "Error message should indicate required parameter count: {msg1}"
    );

    // Valid: 2 elements
    let ok2 = linear_conversion(&mut cn, &[5.0, 2.0]);
    assert!(
        ok2.is_ok(),
        "linear_conversion with 2 elements must succeed"
    );
    if let ChannelData::Float64(ref arr) = cn.data {
        // x * 2 + 5 -> 10 * 2 + 5 = 25, 20 * 2 + 5 = 45
        assert_eq!(arr.value(0), 25.0);
        assert_eq!(arr.value(1), 45.0);
    } else {
        panic!("Expected Float64 channel data");
    }

    // Extra elements: valid, ignores tail without panicking
    let ok3 = linear_conversion(&mut cn, &[1.0, 3.0, 999.0, -100.0]);
    assert!(
        ok3.is_ok(),
        "linear_conversion with surplus elements must succeed"
    );
}

#[test]
fn test_rational_conversion_truncated_cc_val_error() {
    let mut builder = Float64Builder::new();
    builder.append_value(2.0);
    let mut cn = make_cn4_with_data(ChannelData::Float64(builder.finish()));

    // Truncated slices: lengths 0 through 5 must all return Err gracefully
    for len in 0..6 {
        let truncated: Vec<f64> = (0..len).map(|i| i as f64).collect();
        let res = rational_conversion(&mut cn, &truncated);
        assert!(
            res.is_err(),
            "rational_conversion with length {len} must return Err"
        );
        let msg = format!("{:#}", res.unwrap_err());
        assert!(
            msg.contains("malformed rational conversion")
                && msg.contains("expected at least 6 parameters"),
            "Error message should describe malformed rational conversion: {msg}"
        );
    }

    // Valid: length 6 (identity rational: (0*x^2 + 1*x + 0) / (0*x^2 + 0*x + 1) = x)
    let ok6 = rational_conversion(&mut cn, &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    assert!(
        ok6.is_ok(),
        "rational_conversion with 6 elements must succeed"
    );
    if let ChannelData::Float64(ref arr) = cn.data {
        assert_eq!(arr.value(0), 2.0);
    } else {
        panic!("Expected Float64 channel data");
    }

    // Division by zero in rational formula (denominator evaluates to 0.0)
    // Must produce Inf/NaN gracefully without panicking
    let div_zero = rational_conversion(&mut cn, &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    assert!(
        div_zero.is_ok(),
        "rational conversion with zero denominator must not panic"
    );
    if let ChannelData::Float64(ref arr) = cn.data {
        assert!(
            arr.value(0).is_infinite() || arr.value(0).is_nan(),
            "Value must be Inf or NaN"
        );
    }
}

#[test]
fn test_value_range_to_value_table_panic_hardening() {
    let mut builder = Int32Builder::new();
    builder.append_value(5);
    builder.append_value(15);
    builder.append_value(25);
    let mut cn = make_cn4_with_data(ChannelData::Int32(builder.finish()));

    // 1. Completely empty cc_val must return Err gracefully, no panic
    let res_empty = value_range_to_value_table(&mut cn, vec![]);
    assert!(
        res_empty.is_err(),
        "value_range_to_value_table with empty vec must return Err"
    );
    let msg_empty = format!("{:#}", res_empty.unwrap_err());
    assert!(msg_empty.contains("empty cc_val"));

    // 2. Only 1 element (default value, 0 ranges) must NOT panic and return default value for all
    let res_one = value_range_to_value_table(&mut cn, vec![999.0]);
    assert!(
        res_one.is_ok(),
        "Single-element cc_val (default only) must succeed without panic"
    );
    if let ChannelData::Float64(ref arr) = cn.data {
        assert_eq!(arr.value(0), 999.0);
        assert_eq!(arr.value(1), 999.0);
        assert_eq!(arr.value(2), 999.0);
    } else {
        panic!("Expected Float64 output");
    }

    // 3. Incomplete triple (e.g., 2 elements: not a full triple, but has default value)
    let mut cn2 = make_cn4_with_data(ChannelData::Int32(
        Int32Builder::new().finish(), // 0 elements
    ));
    let res_two = value_range_to_value_table(&mut cn2, vec![1.0, 42.0]);
    assert!(
        res_two.is_ok(),
        "2-element cc_val must succeed without panic"
    );

    // 4. Standard 1-range + default: range [0.0, 10.0] -> 100.0, default -> -1.0
    let mut builder3 = Int32Builder::new();
    builder3.append_value(5); // inside [0, 10] -> 100.0
    builder3.append_value(10); // boundary [0, 10] -> 100.0 (integer is inclusive)
    builder3.append_value(15); // outside -> -1.0
    let mut cn3 = make_cn4_with_data(ChannelData::Int32(builder3.finish()));
    let res_valid =
        value_range_to_value_table(&mut cn3, vec![0.0, 10.0, 100.0, -1.0 /* default */]);
    assert!(res_valid.is_ok());
    if let ChannelData::Float64(ref arr) = cn3.data {
        assert_eq!(arr.value(0), 100.0);
        assert_eq!(arr.value(1), 100.0);
        assert_eq!(arr.value(2), -1.0);
    }

    // 5. Pathological floats: NaN, Infinity, inverted ranges must not panic
    let pathological_cases: Vec<Vec<f64>> = vec![
        vec![f64::NAN, 10.0, 100.0, 0.0],
        vec![0.0, f64::NAN, 100.0, 0.0],
        vec![f64::NEG_INFINITY, f64::INFINITY, 100.0, 0.0],
        vec![10.0, 0.0, 100.0, 0.0], // inverted range min > max
        vec![f64::MIN, f64::MAX, 1.0, -1.0],
    ];

    for (case_idx, cc_val) in pathological_cases.into_iter().enumerate() {
        let mut b = Int32Builder::new();
        b.append_value(0);
        b.append_value(10);
        let mut cn_path = make_cn4_with_data(ChannelData::Int32(b.finish()));
        let res = value_range_to_value_table(&mut cn_path, cc_val);
        assert!(
            res.is_ok(),
            "Pathological float case {case_idx} must not panic"
        );
    }
}

#[test]
fn test_conversions_across_channel_data_types_graceful_handling() {
    // Conversion functions called on non-numeric channel types (e.g. Utf8)
    // should log a warning or no-op without panicking.
    let mut str_builder = LargeStringBuilder::new();
    str_builder.append_value("test_string");
    let mut cn = make_cn4_with_data(ChannelData::Utf8(str_builder.finish()));

    let res_lin = linear_conversion(&mut cn, &[1.0, 2.0]);
    assert!(
        res_lin.is_ok(),
        "linear_conversion on Utf8 channel must not panic"
    );
    assert!(matches!(cn.data, ChannelData::Utf8(_)));

    let res_rat = rational_conversion(&mut cn, &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    assert!(
        res_rat.is_ok(),
        "rational_conversion on Utf8 channel must not panic"
    );
    assert!(matches!(cn.data, ChannelData::Utf8(_)));

    // Test with primitive UInt8 channel
    let mut u8_builder = UInt8Builder::new();
    u8_builder.append_value(12);
    let mut cn_u8 = make_cn4_with_data(ChannelData::UInt8(u8_builder.finish()));
    let res_lin_u8 = linear_conversion(&mut cn_u8, &[10.0, 2.0]);
    assert!(res_lin_u8.is_ok());
    if let ChannelData::Float64(ref arr) = cn_u8.data {
        // 12 * 2 + 10 = 34
        assert_eq!(arr.value(0), 34.0);
    } else {
        panic!("Expected Float64 converted data");
    }
}
