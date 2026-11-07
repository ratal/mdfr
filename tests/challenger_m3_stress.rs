//! Challenger 2 Empirical Stress Test Harness for Milestone 3 Feature 27
//!
//! Written by Challenger 2 (teamwork_challenger_m3_2)
//!
//! Objectives:
//! 1. Float upper-bound exclusivity vs Integer upper-bound inclusivity:
//!    - Float range matching upper bound ($a == key\_max$): must fall into next range or default.
//!    - Integer range matching upper bound ($a == key\_max$): must stay inside the range.
//!    - Single-point range ($min == max$): float exact match vs slight offsets.
//!    - Tests across MDF4 (`conversions4`) and MDF3 (`conversions3`).
//! 2. Bitfield text tables:
//!    - Multi-mask values where a later mask does not match:
//!      verifying earlier matched bitfield descriptions are NOT overwritten.
//!    - Combinations of matching, default, and no-default (nothing) masks.
//!    - Mixed CC types (Type 7 value-to-text and Type 8 value-range-to-text).
//!    - Arrow null validity preservation.
//! 3. Conversions performance & binary search:
//!    - 10,000 non-overlapping ranges with partition_point lookups over 200,000 queries.
//!    - Verification against an exact oracle.
//!    - Sub-second performance benchmark verification.
//! 4. XML metadata preservation:
//!    - Read/write roundtrip of an MDF4 file with ##MD XML comments and units.
//!    - Verification that block header is '##MD' ([35, 35, 77, 68]), NOT '##TX'.
//!    - Verification that raw XML byte payload is preserved intact.
//!    - Verification that pre-parsing XML before write does not destroy raw XML bytes.

use anyhow::Result;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

// Arrow imports
use arrow::array::{
    Float32Builder, Float64Builder, Int32Builder, Int64Builder, UInt8Builder, UInt16Builder,
    UInt32Builder,
};

// mdfr-core imports
use mdfr_core::compress::CompressionAlgorithm;

// mdfr imports
use mdfr::data_holder::channel_data::{AsValuesSlice, ChannelData};
use mdfr::mdfinfo::MdfInfo;
use mdfr::mdfinfo::mdfinfo4::{
    BlockType, Cc4Block, CcVal, Cn4, MetaData, MetaDataBlockType, SharableBlocks,
};
use mdfr::mdfreader::Mdf;
use mdfr::mdfreader::conversions3::value_range_to_text_calculation as value_range_to_text_calc_3;
use mdfr::mdfreader::conversions4::{
    bitfield_text_table, convert_channel, value_range_to_value_table,
};

fn make_cn4(data: ChannelData) -> Cn4 {
    Cn4 {
        data,
        ..Default::default()
    }
}

// ============================================================================
// 1. Float Upper-Bound Exclusivity vs Integer Upper-Bound Inclusivity
// ============================================================================

#[test]
fn test_float_upper_bound_exclusivity_mdf4_value_table() {
    // Range table layout for value_range_to_value_table:
    // Triplets (min, max, value) followed by default_value:
    // Range 0: [0.0, 10.0] -> 100.0
    // Range 1: [10.0, 20.0] -> 200.0
    // Range 2: [30.0, 40.0] -> 300.0  (gap between 20.0 and 30.0)
    // Default: 999.0
    let cc_val = vec![
        0.0, 10.0, 100.0, // Range 0
        10.0, 20.0, 200.0, // Range 1
        30.0, 40.0, 300.0, // Range 2
        999.0, // Default
    ];

    // --- Floating-point test (Float64) ---
    // MDF4 spec 6.17.8: Floating-point ranges are half-open [min, max).
    // Therefore:
    // - 0.0 is in Range 0 -> 100.0
    // - 5.0 is in Range 0 -> 100.0
    // - 10.0 matches upper of Range 0 and lower of Range 1 -> MUST be Range 1 (200.0)
    // - 15.0 is in Range 1 -> 200.0
    // - 20.0 matches upper of Range 1 (next is 30.0) -> MUST be Default (999.0)
    // - 25.0 is in gap (20.0, 30.0) -> MUST be Default (999.0)
    // - 30.0 is lower of Range 2 -> 300.0
    // - 40.0 matches upper of Range 2 (last range) -> MUST be Default (999.0)
    // - -5.0 is below all ranges -> Default (999.0)
    // - 50.0 is above all ranges -> Default (999.0)
    let float_inputs = vec![0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, -5.0, 50.0];
    let expected_float = vec![
        100.0, 100.0, 200.0, 200.0, 999.0, 999.0, 300.0, 999.0, 999.0, 999.0,
    ];

    let mut f64_builder = Float64Builder::new();
    for v in &float_inputs {
        f64_builder.append_value(*v);
    }
    let mut cn_f64 = make_cn4(ChannelData::Float64(f64_builder.finish()));
    value_range_to_value_table(&mut cn_f64, cc_val.clone()).unwrap();

    if let ChannelData::Float64(ref out) = cn_f64.data {
        let actual = out.values_slice();
        assert_eq!(actual.len(), expected_float.len());
        for (i, (&act, &exp)) in actual.iter().zip(expected_float.iter()).enumerate() {
            assert_eq!(
                act, exp,
                "Float64 input {} at index {}: expected {}, got {}",
                float_inputs[i], i, exp, act
            );
        }
    } else {
        panic!("Expected Float64 channel data");
    }

    // Also verify Float32
    let mut f32_builder = Float32Builder::new();
    for v in &float_inputs {
        f32_builder.append_value(*v as f32);
    }
    let mut cn_f32 = make_cn4(ChannelData::Float32(f32_builder.finish()));
    value_range_to_value_table(&mut cn_f32, cc_val.clone()).unwrap();

    if let ChannelData::Float64(ref out) = cn_f32.data {
        let actual = out.values_slice();
        for (i, (&act, &exp)) in actual.iter().zip(expected_float.iter()).enumerate() {
            assert_eq!(
                act, exp,
                "Float32 input {} at index {}: expected {}, got {}",
                float_inputs[i], i, exp, act
            );
        }
    } else {
        panic!("Expected Float64 channel data after conversion");
    }

    // --- Integer test with non-overlapping ranges ---
    // When ranges have gaps, e.g. [0, 10] -> 100, [15, 25] -> 200, def: 999:
    // Value 10:
    // - Float: 10.0 < 10.0 is false -> falls to Default (999.0)
    // - Integer: 10 <= 10.0 is true -> stays inside Range 0 (100.0)
    let cc_val_gap = vec![
        0.0, 10.0, 100.0, // Range 0
        15.0, 25.0, 200.0, // Range 1
        999.0, // Default
    ];

    // Test Integer UInt32 at boundary 10
    let mut u32_builder = UInt32Builder::new();
    u32_builder.append_value(0); // inside Range 0 -> 100
    u32_builder.append_value(10); // EXACT UPPER of Range 0 -> MUST stay in Range 0 (100)!
    u32_builder.append_value(11); // in gap -> 999
    u32_builder.append_value(15); // inside Range 1 -> 200
    u32_builder.append_value(25); // EXACT UPPER of Range 1 -> MUST stay in Range 1 (200)!
    u32_builder.append_value(26); // above -> 999
    let mut cn_u32 = make_cn4(ChannelData::UInt32(u32_builder.finish()));
    value_range_to_value_table(&mut cn_u32, cc_val_gap.clone()).unwrap();

    if let ChannelData::Float64(ref out) = cn_u32.data {
        let actual = out.values_slice();
        assert_eq!(actual, &[100.0, 100.0, 999.0, 200.0, 200.0, 999.0]);
    } else {
        panic!("Expected Float64 channel data");
    }

    // For contrast, test Float64 on the EXACT same cc_val_gap at boundary 10 and 25
    let mut f64_gap = Float64Builder::new();
    f64_gap.append_value(0.0);
    f64_gap.append_value(10.0); // UPPER of Range 0 -> MUST be exclusive -> 999.0!
    f64_gap.append_value(11.0);
    f64_gap.append_value(15.0);
    f64_gap.append_value(25.0); // UPPER of Range 1 -> MUST be exclusive -> 999.0!
    f64_gap.append_value(26.0);
    let mut cn_f64_gap = make_cn4(ChannelData::Float64(f64_gap.finish()));
    value_range_to_value_table(&mut cn_f64_gap, cc_val_gap).unwrap();

    if let ChannelData::Float64(ref out) = cn_f64_gap.data {
        let actual = out.values_slice();
        assert_eq!(actual, &[100.0, 999.0, 999.0, 200.0, 999.0, 999.0]);
    } else {
        panic!("Expected Float64 channel data");
    }
}

#[test]
fn test_float_upper_bound_exclusivity_mdf4_value_range_to_text() {
    // CC_TYPE = 8 (value range to text)
    // Setup SharableBlocks with Cc4Block and text strings
    let mut sharable = SharableBlocks::default();
    let cc_ptr = 5000i64;

    // cc_val: pairs of (min, max)
    // Range 0: [0.0, 10.0] -> "RANGE_0"
    // Range 1: [10.0, 20.0] -> "RANGE_1"
    // Range 2: [25.0, 25.0] -> "EXACT_POINT_25"  (single point min == max)
    // Range 3: [30.0, 40.0] -> "RANGE_3"
    let cc_val = vec![
        0.0, 10.0, // Range 0
        10.0, 20.0, // Range 1
        25.0, 25.0, // Range 2 (single-point)
        30.0, 40.0, // Range 3
    ];
    let cc_ref = vec![6001, 6002, 6003, 6004, 6005]; // refs for 4 ranges + default
    sharable.create_tx(6001, "RANGE_0".to_string());
    sharable.create_tx(6002, "RANGE_1".to_string());
    sharable.create_tx(6003, "EXACT_POINT_25".to_string());
    sharable.create_tx(6004, "RANGE_3".to_string());
    sharable.create_tx(6005, "DEFAULT_TXT".to_string());

    let cc = Cc4Block::new_with_val(8, CcVal::Real(cc_val), cc_ref);
    sharable.cc.insert(cc_ptr, cc);

    // --- 1. Test Float64 channel ---
    let mut fb = Float64Builder::new();
    fb.append_value(0.0); // -> RANGE_0
    fb.append_value(5.0); // -> RANGE_0
    fb.append_value(10.0); // upper of 0, lower of 1 -> MUST be RANGE_1
    fb.append_value(15.0); // -> RANGE_1
    fb.append_value(20.0); // upper of 1 -> gap -> MUST be DEFAULT_TXT
    fb.append_value(24.99); // gap -> DEFAULT_TXT
    fb.append_value(25.0); // exact single point (min == max) -> MUST be EXACT_POINT_25
    fb.append_value(25.01); // gap -> DEFAULT_TXT
    fb.append_value(40.0); // upper of last range -> MUST be DEFAULT_TXT

    let mut cn = make_cn4(ChannelData::Float64(fb.finish()));
    cn.block.cn_cc_conversion = cc_ptr;
    convert_channel(&mut cn, &sharable).unwrap();

    if let ChannelData::Utf8(ref s) = cn.data {
        assert_eq!(s.value(0), "RANGE_0");
        assert_eq!(s.value(1), "RANGE_0");
        assert_eq!(s.value(2), "RANGE_1", "Float 10.0 must fall into RANGE_1");
        assert_eq!(s.value(3), "RANGE_1");
        assert_eq!(
            s.value(4),
            "DEFAULT_TXT",
            "Float 20.0 at upper bound must fall into default"
        );
        assert_eq!(s.value(5), "DEFAULT_TXT");
        assert_eq!(
            s.value(6),
            "EXACT_POINT_25",
            "Float 25.0 matching single-point range must match"
        );
        assert_eq!(s.value(7), "DEFAULT_TXT");
        assert_eq!(
            s.value(8),
            "DEFAULT_TXT",
            "Float 40.0 at upper bound of last range must fall into default"
        );
    } else {
        panic!("Expected Utf8 channel data");
    }

    // --- 2. Test Integer channel (Int32) with disjoint range ---
    let mut sharable_int = SharableBlocks::default();
    let cc_ptr_int = 7000i64;
    let cc_val_int = vec![
        0.0, 10.0, // Range 0
        15.0, 25.0, // Range 1
    ];
    let cc_ref_int = vec![7001, 7002, 7003];
    sharable_int.create_tx(7001, "INT_RANGE_0".to_string());
    sharable_int.create_tx(7002, "INT_RANGE_1".to_string());
    sharable_int.create_tx(7003, "INT_DEFAULT".to_string());

    let cc_int = Cc4Block::new_with_val(8, CcVal::Real(cc_val_int), cc_ref_int);
    sharable_int.cc.insert(cc_ptr_int, cc_int);

    let mut ib = Int32Builder::new();
    ib.append_value(0); // -> INT_RANGE_0
    ib.append_value(10); // upper bound of Range 0 -> MUST stay in INT_RANGE_0
    ib.append_value(11); // gap -> INT_DEFAULT
    ib.append_value(15); // -> INT_RANGE_1
    ib.append_value(25); // upper bound of Range 1 -> MUST stay in INT_RANGE_1
    ib.append_value(26); // above -> INT_DEFAULT

    let mut cn_int = make_cn4(ChannelData::Int32(ib.finish()));
    cn_int.block.cn_cc_conversion = cc_ptr_int;
    convert_channel(&mut cn_int, &sharable_int).unwrap();

    if let ChannelData::Utf8(ref s) = cn_int.data {
        assert_eq!(s.value(0), "INT_RANGE_0");
        assert_eq!(
            s.value(1),
            "INT_RANGE_0",
            "Integer 10 must stay inside INT_RANGE_0"
        );
        assert_eq!(s.value(2), "INT_DEFAULT");
        assert_eq!(s.value(3), "INT_RANGE_1");
        assert_eq!(
            s.value(4),
            "INT_RANGE_1",
            "Integer 25 must stay inside INT_RANGE_1"
        );
        assert_eq!(s.value(5), "INT_DEFAULT");
    } else {
        panic!("Expected Utf8 channel data");
    }
}

#[test]
fn test_float_upper_bound_exclusivity_mdf3() {
    // Test MDF3 conversions3::value_range_to_text_calculation
    let ranges = vec![
        (0.0, 10.0, "MDF3_RANGE_0".to_string()),
        (15.0, 25.0, "MDF3_RANGE_1".to_string()),
        (30.0, 30.0, "MDF3_POINT_30".to_string()),
    ];
    let cc_val_ref = (ranges, "MDF3_DEFAULT".to_string());

    // Float32 input: inclusive_upper = false
    let mut fb = Float32Builder::new();
    fb.append_value(0.0);
    fb.append_value(10.0); // upper of range 0 -> exclusive -> default
    fb.append_value(15.0); // range 1
    fb.append_value(25.0); // upper of range 1 -> exclusive -> default
    fb.append_value(30.0); // single point min==max -> match
    let f_array = fb.finish();

    let res_float = value_range_to_text_calc_3(&f_array, &cc_val_ref, false).unwrap();
    assert_eq!(res_float.value(0), "MDF3_RANGE_0");
    assert_eq!(
        res_float.value(1),
        "MDF3_DEFAULT",
        "MDF3 float 10.0 must be exclusive (default)"
    );
    assert_eq!(res_float.value(2), "MDF3_RANGE_1");
    assert_eq!(
        res_float.value(3),
        "MDF3_DEFAULT",
        "MDF3 float 25.0 must be exclusive (default)"
    );
    assert_eq!(
        res_float.value(4),
        "MDF3_POINT_30",
        "MDF3 float 30.0 single point must match"
    );

    // Integer Int64 input: inclusive_upper = true
    let mut ib = Int64Builder::new();
    ib.append_value(0);
    ib.append_value(10); // upper of range 0 -> inclusive -> RANGE_0
    ib.append_value(15); // range 1
    ib.append_value(25); // upper of range 1 -> inclusive -> RANGE_1
    ib.append_value(30); // single point -> match
    let i_array = ib.finish();

    let res_int = value_range_to_text_calc_3(&i_array, &cc_val_ref, true).unwrap();
    assert_eq!(res_int.value(0), "MDF3_RANGE_0");
    assert_eq!(
        res_int.value(1),
        "MDF3_RANGE_0",
        "MDF3 integer 10 must be inclusive (RANGE_0)"
    );
    assert_eq!(res_int.value(2), "MDF3_RANGE_1");
    assert_eq!(
        res_int.value(3),
        "MDF3_RANGE_1",
        "MDF3 integer 25 must be inclusive (RANGE_1)"
    );
    assert_eq!(res_int.value(4), "MDF3_POINT_30");
}

// ============================================================================
// 2. Bitfield Text Tables Stress Testing
// ============================================================================

#[test]
fn test_bitfield_multi_mask_non_matching_later_masks() {
    // Test that multi-mask bitfield tables do NOT overwrite earlier matched descriptions
    // when later masks do not match (or match their default/fallback).
    let mut sharable = SharableBlocks::default();

    // Mask 0: 0x01 (Bit 0) -> Value 1: "POWER_ON", Default: "POWER_STANDBY"
    let cc_ptr0 = 100i64;
    let cc0 = Cc4Block::new_with_val(7, CcVal::Real(vec![1.0]), vec![1001, 1002]);
    sharable.cc.insert(cc_ptr0, cc0);
    sharable.create_tx(1001, "POWER_ON".to_string());
    sharable.create_tx(1002, "POWER_STANDBY".to_string());

    // Mask 1: 0x02 (Bit 1) -> Value 2: "VALVE_OPEN", Default: "VALVE_CLOSED"
    let cc_ptr1 = 200i64;
    let cc1 = Cc4Block::new_with_val(7, CcVal::Real(vec![2.0]), vec![2001, 2002]);
    sharable.cc.insert(cc_ptr1, cc1);
    sharable.create_tx(2001, "VALVE_OPEN".to_string());
    sharable.create_tx(2002, "VALVE_CLOSED".to_string());

    // Mask 2: 0x04 (Bit 2) -> Value 4: "PUMP_RUNNING", Default: "PUMP_IDLE"
    let cc_ptr2 = 300i64;
    let cc2 = Cc4Block::new_with_val(7, CcVal::Real(vec![4.0]), vec![3001, 3002]);
    sharable.cc.insert(cc_ptr2, cc2);
    sharable.create_tx(3001, "PUMP_RUNNING".to_string());
    sharable.create_tx(3002, "PUMP_IDLE".to_string());

    // Mask 3: 0x08 (Bit 3) -> Value 8: "ALARM_ACTIVE", Default: "ALARM_CLEAR"
    let cc_ptr3 = 400i64;
    let cc3 = Cc4Block::new_with_val(7, CcVal::Real(vec![8.0]), vec![4001, 4002]);
    sharable.cc.insert(cc_ptr3, cc3);
    sharable.create_tx(4001, "ALARM_ACTIVE".to_string());
    sharable.create_tx(4002, "ALARM_CLEAR".to_string());

    let cc_val = vec![1u64, 2u64, 4u64, 8u64];
    let cc_ref = vec![cc_ptr0, cc_ptr1, cc_ptr2, cc_ptr3];

    // Inputs:
    // Sample 0: 0b0001 (1) -> Mask 0 matches, Masks 1,2,3 default
    // Sample 1: 0b0101 (5) -> Masks 0 and 2 match, Masks 1 and 3 default
    // Sample 2: 0b1010 (10) -> Masks 1 and 3 match, Masks 0 and 2 default
    // Sample 3: 0b1111 (15) -> All 4 match
    // Sample 4: 0b0000 (0) -> All 4 default
    let mut builder = UInt8Builder::new();
    builder.append_value(1);
    builder.append_value(5);
    builder.append_value(10);
    builder.append_value(15);
    builder.append_value(0);

    let mut cn = make_cn4(ChannelData::UInt8(builder.finish()));
    bitfield_text_table(&mut cn, &cc_val, &cc_ref, &sharable).unwrap();

    if let ChannelData::Utf8(ref s) = cn.data {
        assert_eq!(
            s.value(0),
            " | POWER_ON | VALVE_CLOSED | PUMP_IDLE | ALARM_CLEAR",
            "Earlier match POWER_ON must not be overwritten by later non-matching masks"
        );
        assert_eq!(
            s.value(1),
            " | POWER_ON | VALVE_CLOSED | PUMP_RUNNING | ALARM_CLEAR",
            "Matches on masks 0 and 2 must both be present and not overwritten"
        );
        assert_eq!(
            s.value(2),
            " | POWER_STANDBY | VALVE_OPEN | PUMP_IDLE | ALARM_ACTIVE",
            "Matches on masks 1 and 3 must both be present"
        );
        assert_eq!(
            s.value(3),
            " | POWER_ON | VALVE_OPEN | PUMP_RUNNING | ALARM_ACTIVE",
            "All masks matching"
        );
        assert_eq!(
            s.value(4),
            " | POWER_STANDBY | VALVE_CLOSED | PUMP_IDLE | ALARM_CLEAR",
            "All masks defaulting"
        );
    } else {
        panic!("Expected Utf8 channel data");
    }
}

#[test]
fn test_bitfield_named_masks_and_mixed_types() {
    let mut sharable = SharableBlocks::default();

    // Mask 0: 0x01 with name "Engine"
    let cc_ptr0 = 100i64;
    let mut cc0 = Cc4Block::new_with_val(7, CcVal::Real(vec![1.0]), vec![1001, 1002]);
    cc0.cc_tx_name = 1010;
    sharable.create_tx(1010, "Engine".to_string());
    sharable.create_tx(1001, "ON".to_string());
    sharable.create_tx(1002, "OFF".to_string());
    sharable.cc.insert(cc_ptr0, cc0);

    // Mask 1: 0x06 (Bits 1..2, values 0..6) using CC_TYPE 8 (ValueRangeToText)
    // Range 0: [0.0, 2.0] -> "SLOW"
    // Range 1: [4.0, 6.0] -> "FAST"
    // Default: "UNKNOWN_SPEED"
    let cc_ptr1 = 200i64;
    let mut cc1 = Cc4Block::new_with_val(
        8,
        CcVal::Real(vec![0.0, 2.0, 4.0, 6.0]),
        vec![2001, 2002, 2003],
    );
    cc1.cc_tx_name = 2010;
    sharable.create_tx(2010, "SpeedMode".to_string());
    sharable.create_tx(2001, "SLOW".to_string());
    sharable.create_tx(2002, "FAST".to_string());
    sharable.create_tx(2003, "UNKNOWN_SPEED".to_string());
    sharable.cc.insert(cc_ptr1, cc1);

    // Mask 2: 0x08 without name, and without default (only matching text)
    let cc_ptr2 = 300i64;
    let cc2 = Cc4Block::new_with_val(7, CcVal::Real(vec![8.0]), vec![3001]);
    sharable.create_tx(3001, "WARN_FLAG".to_string());
    sharable.cc.insert(cc_ptr2, cc2);

    let cc_val = vec![1u64, 6u64, 8u64];
    let cc_ref = vec![cc_ptr0, cc_ptr1, cc_ptr2];

    // Sample:
    // Engine = 1 (ON)
    // Speed = 4 (FAST)
    // Warn = 0 (no match, no default -> "nothing")
    let mut builder = UInt16Builder::new();
    builder.append_value(1 | 4); // 5: Engine ON, Speed FAST, Warn nothing
    builder.append_value(0); // 0: Engine OFF, Speed SLOW, Warn nothing
    builder.append_value(1 | 8); // 9: Engine ON, Speed SLOW (0 & 6 = 0 <= 2), Warn WARN_FLAG

    let mut cn = make_cn4(ChannelData::UInt16(builder.finish()));
    bitfield_text_table(&mut cn, &cc_val, &cc_ref, &sharable).unwrap();

    if let ChannelData::Utf8(ref s) = cn.data {
        assert_eq!(
            s.value(0),
            " | Engine = ON | SpeedMode = FAST | nothing",
            "Named bitfield formatting with type 8 range and nothing fallback"
        );
        assert_eq!(s.value(1), " | Engine = OFF | SpeedMode = SLOW | nothing");
        assert_eq!(s.value(2), " | Engine = ON | SpeedMode = SLOW | WARN_FLAG");
    } else {
        panic!("Expected Utf8 channel data");
    }
}

// ============================================================================
// 3. Conversions Performance & Binary Search Stress Testing
// ============================================================================

#[test]
fn test_large_range_table_partition_point_stress_and_performance() {
    // Generate 10,000 non-overlapping ranges:
    // Range k: [k * 10, k * 10 + 6] -> k * 100 + 42
    // There is a 4-unit gap (k * 10 + 6, (k + 1) * 10) between consecutive ranges.
    // Default = -999.0
    const NUM_RANGES: usize = 10_000;
    const NUM_QUERIES: usize = 200_000;

    let mut cc_val = Vec::with_capacity(NUM_RANGES * 3 + 1);
    for k in 0..NUM_RANGES {
        let min = (k * 10) as f64;
        let max = min + 6.0;
        let val = (k * 100 + 42) as f64;
        cc_val.push(min);
        cc_val.push(max);
        cc_val.push(val);
    }
    const DEFAULT_VAL: f64 = -999.0;
    cc_val.push(DEFAULT_VAL);

    // Oracle evaluation function for Float (exclusive upper bound)
    let oracle_float = |x: f64| -> f64 {
        if x < 0.0 {
            return DEFAULT_VAL;
        }
        let k = (x / 10.0).floor() as usize;
        if k < NUM_RANGES {
            let min = (k * 10) as f64;
            let max = min + 6.0;
            if x >= min && x < max {
                return (k * 100 + 42) as f64;
            }
        }
        DEFAULT_VAL
    };

    // Oracle evaluation function for Integer (inclusive upper bound)
    let oracle_int = |x: i64| -> f64 {
        if x < 0 {
            return DEFAULT_VAL;
        }
        let k = (x / 10) as usize;
        if k < NUM_RANGES {
            let min = (k * 10) as i64;
            let max = min + 6;
            if x >= min && x <= max {
                return (k * 100 + 42) as f64;
            }
        }
        DEFAULT_VAL
    };

    // Build 200,000 queries testing various scenarios:
    // - exact min
    // - exact max
    // - interior point
    // - gap point
    // - extreme out-of-bounds
    let mut f64_queries = Vec::with_capacity(NUM_QUERIES);
    let mut expected_f64 = Vec::with_capacity(NUM_QUERIES);

    for i in 0..NUM_QUERIES {
        let k = i % (NUM_RANGES + 100);
        let q = match i % 6 {
            0 => (k * 10) as f64,                // exact min
            1 => (k * 10) as f64 + 3.0,          // interior point
            2 => (k * 10) as f64 + 6.0,          // exact max (should be DEFAULT for float)
            3 => (k * 10) as f64 + 7.5,          // gap point
            4 => -100.0,                         // negative out-of-bounds
            _ => (NUM_RANGES * 10 + 500) as f64, // high out-of-bounds
        };
        f64_queries.push(q);
        expected_f64.push(oracle_float(q));
    }

    let mut builder = Float64Builder::with_capacity(NUM_QUERIES);
    for &q in &f64_queries {
        builder.append_value(q);
    }
    let mut cn = make_cn4(ChannelData::Float64(builder.finish()));

    let start = std::time::Instant::now();
    value_range_to_value_table(&mut cn, cc_val.clone()).unwrap();
    let elapsed = start.elapsed();

    // Verify sub-second speed: 200,000 queries over 10,000 ranges
    assert!(
        elapsed.as_millis() < 1000,
        "partition_point binary search too slow: took {:?}",
        elapsed
    );

    // Verify 100% correctness against oracle
    if let ChannelData::Float64(ref out) = cn.data {
        let vals = out.values_slice();
        assert_eq!(vals.len(), NUM_QUERIES);
        for i in 0..NUM_QUERIES {
            assert_eq!(
                vals[i], expected_f64[i],
                "Mismatch at query index {}: input was {}, expected {}, got {}",
                i, f64_queries[i], expected_f64[i], vals[i]
            );
        }
    } else {
        panic!("Expected Float64 channel data");
    }

    // Now test Integer queries (Int64) on the exact same large range table
    let mut i64_queries = Vec::with_capacity(NUM_QUERIES);
    let mut expected_i64 = Vec::with_capacity(NUM_QUERIES);

    for i in 0..NUM_QUERIES {
        let k = (i % (NUM_RANGES + 50)) as i64;
        let q = match i % 5 {
            0 => k * 10,     // exact min -> in range
            1 => k * 10 + 3, // interior -> in range
            2 => k * 10 + 6, // exact max -> MUST STAY IN RANGE for integer!
            3 => k * 10 + 8, // gap -> default
            _ => -50,        // out-of-bounds -> default
        };
        i64_queries.push(q);
        expected_i64.push(oracle_int(q));
    }

    let mut ibuilder = Int64Builder::with_capacity(NUM_QUERIES);
    for &q in &i64_queries {
        ibuilder.append_value(q);
    }
    let mut cn_int = make_cn4(ChannelData::Int64(ibuilder.finish()));

    let start_int = std::time::Instant::now();
    value_range_to_value_table(&mut cn_int, cc_val).unwrap();
    let elapsed_int = start_int.elapsed();

    assert!(
        elapsed_int.as_millis() < 1000,
        "integer partition_point too slow: took {:?}",
        elapsed_int
    );

    if let ChannelData::Float64(ref out) = cn_int.data {
        let vals = out.values_slice();
        assert_eq!(vals.len(), NUM_QUERIES);
        for i in 0..NUM_QUERIES {
            assert_eq!(
                vals[i], expected_i64[i],
                "Integer mismatch at query index {}: input was {}, expected {}, got {}",
                i, i64_queries[i], expected_i64[i], vals[i]
            );
        }
    } else {
        panic!("Expected Float64 channel data");
    }
}

// ============================================================================
// 4. XML Metadata Preservation (##MD) Roundtrip Stress Testing
// ============================================================================

#[test]
fn test_xml_metadata_preservation_and_roundtrip() -> Result<()> {
    // Test that ##MD XML comment and unit blocks:
    // 1. Maintain their '##MD' block header (hdr_id = [35, 35, 77, 68])
    // 2. Preserve raw XML byte payload intact through write and re-read
    // 3. Are fully parseable by parse_xml() after roundtrip
    // 4. Do not have their raw_data cleared by parse_xml() before write

    let base_file = "test_files/test_basic.mf4";
    if !Path::new(base_file).exists() {
        eprintln!("Skipping test_xml_metadata_preservation: base file missing");
        return Ok(());
    }

    let mut mdf = Mdf::new(base_file)?;
    mdf.load_all_channels_data_in_memory()?;

    let xml_hd_comment = "<HDcomment>\n<TX>Empirical Challenger Test Run #42</TX>\n<time_source>GPS_PPS</time_source>\n<common_properties>\n<e name=\"Challenger\">teamwork_challenger_m3_2</e>\n<e name=\"Recorder.SequenceIndex\">42</e>\n</common_properties>\n</HDcomment>";
    let xml_cn_unit = "<CNcomment>\n<TX>degC</TX>\n<common_properties>\n<e name=\"PhysicalDimension\">Temperature</e>\n<e name=\"SI_Unit\">K</e>\n</common_properties>\n</CNcomment>";
    let xml_cn_comment = "<CNcomment>\n<TX>Engine Coolant Temperature</TX>\n<names>\n<name>EngineCoolantTemp</name>\n<display>Coolant Temp</display>\n</names>\n<common_properties>\n<e name=\"Calibrated\">true</e>\n<e name=\"SamplingRateHz\">100</e>\n</common_properties>\n</CNcomment>";

    let target_channel_name: String = match &mut mdf.mdf_info {
        MdfInfo::V4(info4) => {
            // 1. Add HD MD comment
            let mut md_hd = MetaData::new(MetaDataBlockType::MdBlock, BlockType::HD);
            md_hd.set_data_buffer(xml_hd_comment.as_bytes());
            assert_eq!(md_hd.block.hdr_id, [35, 35, 77, 68]); // ##MD
            info4.hd_block.hd_md_comment = 888001;
            info4.sharable.md_tx.insert(888001, md_hd);

            // 2. Find a channel and attach CN MD unit and CN MD comment
            let mut found_cn = false;
            let mut found_name = String::new();
            for dg in info4.dg.values_mut() {
                for cg in dg.cg.values_mut() {
                    if let Some(cn) = cg.cn.values_mut().next() {
                        let mut md_unit = MetaData::new(MetaDataBlockType::MdBlock, BlockType::CN);
                        md_unit.set_data_buffer(xml_cn_unit.as_bytes());
                        assert_eq!(md_unit.block.hdr_id, [35, 35, 77, 68]);

                        let mut md_comment =
                            MetaData::new(MetaDataBlockType::MdBlock, BlockType::CN);
                        md_comment.set_data_buffer(xml_cn_comment.as_bytes());
                        assert_eq!(md_comment.block.hdr_id, [35, 35, 77, 68]);

                        // CRITICAL: Call parse_xml() on the comment BEFORE writing!
                        // This tests that parse_xml() does NOT destroy raw_data or change hdr_id!
                        md_comment.parse_xml().expect("parse_xml should succeed");
                        assert!(
                            !md_comment.raw_data.is_empty(),
                            "parse_xml() must not wipe raw_data!"
                        );
                        assert_eq!(
                            md_comment.block.hdr_id,
                            [35, 35, 77, 68],
                            "hdr_id must remain ##MD after parse_xml"
                        );

                        cn.block.cn_md_unit = 888002;
                        cn.block.cn_md_comment = 888003;
                        info4.sharable.md_tx.insert(888002, md_unit);
                        info4.sharable.md_tx.insert(888003, md_comment);

                        found_name = cn.unique_name.clone();
                        found_cn = true;
                    }
                    if found_cn {
                        break;
                    }
                }
                if found_cn {
                    break;
                }
            }
            assert!(found_cn, "Must find at least one channel in test file");
            found_name
        }
        _ => panic!("Expected MDF4 file"),
    };

    // Write file to temporary output
    let tmp = tempfile::Builder::new().suffix(".mf4").tempfile()?;
    let tmp_path = tmp.path().to_str().unwrap().to_string();

    mdf.write(&tmp_path, CompressionAlgorithm::NoCompression)?;

    // Re-read file from disk
    let mut reread = Mdf::new(&tmp_path)?;
    reread.load_all_channels_data_in_memory()?;

    match &mut reread.mdf_info {
        MdfInfo::V4(info4) => {
            // 1. Verify HD MD Comment
            assert!(
                info4.hd_block.hd_md_comment > 0,
                "hd_md_comment pointer must be non-zero in written file"
            );
            let hd_md = info4
                .sharable
                .md_tx
                .get_mut(&info4.hd_block.hd_md_comment)
                .expect("HD MD comment must be present in sharable");

            // Verify header ID is ##MD
            assert_eq!(
                hd_md.block.hdr_id,
                [35, 35, 77, 68],
                "hd_md_comment header ID must be ##MD ([35, 35, 77, 68]), not ##TX"
            );

            // Verify raw payload preserves original XML
            let hd_xml_str = std::str::from_utf8(&hd_md.raw_data)?;
            assert!(
                hd_xml_str.contains("Empirical Challenger Test Run #42"),
                "HD comment XML text must be preserved"
            );
            assert!(
                hd_xml_str.contains("<time_source>GPS_PPS</time_source>"),
                "HD time_source must be preserved"
            );
            assert!(
                hd_xml_str.contains("teamwork_challenger_m3_2"),
                "HD custom common property must be preserved"
            );

            // Verify parse_xml works on re-read block
            hd_md.parse_xml()?;
            assert_eq!(
                hd_md.get_tx()?,
                Some("Empirical Challenger Test Run #42".to_string())
            );

            // 2. Verify Channel Unit and Comment
            let mut found_target = false;
            for dg in info4.dg.values_mut() {
                for cg in dg.cg.values_mut() {
                    for cn in cg.cn.values_mut() {
                        if cn.unique_name == target_channel_name {
                            found_target = true;

                            // Channel Unit
                            assert!(
                                cn.block.cn_md_unit > 0,
                                "cn_md_unit pointer must be non-zero"
                            );
                            let unit_md = info4
                                .sharable
                                .md_tx
                                .get_mut(&cn.block.cn_md_unit)
                                .expect("Unit MD block must be in sharable");
                            assert_eq!(
                                unit_md.block.hdr_id,
                                [35, 35, 77, 68],
                                "cn_md_unit header ID must be ##MD"
                            );
                            let unit_xml_str = std::str::from_utf8(&unit_md.raw_data)?;
                            assert!(
                                unit_xml_str.contains("<TX>degC</TX>"),
                                "Unit XML text must contain <TX>degC</TX>"
                            );
                            assert!(
                                unit_xml_str.contains("PhysicalDimension"),
                                "Unit XML common property must be preserved"
                            );
                            unit_md.parse_xml()?;
                            assert_eq!(unit_md.get_tx()?, Some("degC".to_string()));

                            // Channel Comment
                            assert!(
                                cn.block.cn_md_comment > 0,
                                "cn_md_comment pointer must be non-zero"
                            );
                            let comment_md = info4
                                .sharable
                                .md_tx
                                .get_mut(&cn.block.cn_md_comment)
                                .expect("Comment MD block must be in sharable");
                            assert_eq!(
                                comment_md.block.hdr_id,
                                [35, 35, 77, 68],
                                "cn_md_comment header ID must be ##MD"
                            );
                            let comment_xml_str = std::str::from_utf8(&comment_md.raw_data)?;
                            assert!(
                                comment_xml_str.contains("Engine Coolant Temperature"),
                                "Comment XML text must contain 'Engine Coolant Temperature'"
                            );
                            assert!(
                                comment_xml_str.contains("EngineCoolantTemp"),
                                "Comment XML names must be preserved"
                            );
                            assert!(
                                comment_xml_str.contains("Calibrated"),
                                "Comment XML common property must be preserved"
                            );
                            comment_md.parse_xml()?;
                            assert_eq!(
                                comment_md.get_tx()?,
                                Some("Engine Coolant Temperature".to_string())
                            );

                            break;
                        }
                    }
                    if found_target {
                        break;
                    }
                }
                if found_target {
                    break;
                }
            }
            assert!(found_target, "Target channel must be found in re-read file");
        }
        _ => panic!("Expected MDF4 file"),
    }

    // 3. Raw Disk File Inspection
    // Independently read raw bytes from the file on disk to guarantee the binary on disk
    // actually contains '##MD' block headers and XML payloads at the referenced offsets.
    let mut disk_file = File::open(&tmp_path)?;
    let mut header_magic = [0u8; 4];

    // Read the offset of hd_md_comment directly from HD block:
    // ID block is 64 bytes. HD header is 24 bytes.
    // HD links: dg_first (0), fh_first (1), ch_first (2), at_first (3), ev_first (4), md_comment (5).
    // File position = 64 + 24 + 5 * 8 = 128.
    disk_file.seek(SeekFrom::Start(128))?;
    let mut hd_comment_offset_bytes = [0u8; 8];
    disk_file.read_exact(&mut hd_comment_offset_bytes)?;
    let hd_comment_pos = i64::from_le_bytes(hd_comment_offset_bytes);
    assert!(hd_comment_pos > 0, "hd_comment_pos must be > 0 on disk");

    disk_file.seek(SeekFrom::Start(hd_comment_pos as u64))?;
    disk_file.read_exact(&mut header_magic)?;
    assert_eq!(
        &header_magic, b"##MD",
        "Raw file on disk at HD comment offset MUST be '##MD'"
    );

    // Read payload length (u64 at offset + 8)
    let mut len_bytes = [0u8; 8];
    disk_file.seek(SeekFrom::Start((hd_comment_pos + 8) as u64))?;
    disk_file.read_exact(&mut len_bytes)?;
    let blk_len = u64::from_le_bytes(len_bytes);
    assert!(blk_len > 24, "Block length must exceed header");

    let mut payload = vec![0u8; (blk_len - 24) as usize];
    disk_file.seek(SeekFrom::Start((hd_comment_pos + 24) as u64))?;
    disk_file.read_exact(&mut payload)?;
    let disk_xml = std::str::from_utf8(&payload)?;
    assert!(
        disk_xml.contains("Empirical Challenger Test Run #42"),
        "Raw disk bytes must contain XML payload"
    );

    Ok(())
}
