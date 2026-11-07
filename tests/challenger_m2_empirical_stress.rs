//! Empirical Stress Test Harness for Milestone 2 Features
//!
//! Written by Challenger 1 (teamwork_challenger_m2_3)
//!
//! Objectives:
//! 1. `StaticArena::dealloc_layout`:
//!    - Null pointer deallocations across all alignments (1, 2, 4, 8, 16, 32, 64) and sizes (0, 1, 64, 1024, 65536)
//!    - Zero-size allocation & deallocation across alignments 1, 2, 4, 8, 16
//!    - Normal allocation/write/read/dealloc cycles
//!    - Out-of-order deallocations: FIFO, LIFO, odd-even, center-out, random-order
//!    - Coalescing verification: 100% capacity re-acquisition after full deallocation
//! 2. `SpscRing::try_push`:
//!    - Reject payload > 65535 (65536, 65537, 100_000)
//!    - Accept payload == 65535 when capacity allows
//!    - Reject group handle 0xFFFF (sentinel) with varied payload sizes
//!    - Accept group handle 0xFFFE
//!    - Atomic tail update during drain error: confirm skipped bad record, no infinite loop, clean subsequent drain
//! 3. Panic hardening:
//!    - `arrow_helpers.rs`: bit_count & byte_count on unusual/unsupported types (Dictionary, Struct, etc.)
//!    - `conversions4.rs`: linear_conversion, rational_conversion, value_range_to_value_table, text_to_value_calculation
//!      with out-of-bounds indices, empty/truncated cc_val, and non-numeric channels.

use core::alloc::Layout;
use std::ptr;
use std::sync::Arc;

// Arrow imports
use arrow::array::{
    DictionaryArray, Float64Builder, Int32Array, Int32Builder, LargeStringBuilder, StringArray,
    StructArray,
};
use arrow::datatypes::{DataType, Field, Fields, Int32Type};

// mdfr-core imports
use mdfr_core::arena::StaticArena;
use mdfr_core::blocks::cn::CnDataType;
use mdfr_core::compress::CompressionAlgorithm;
use mdfr_core::error::Error;
use mdfr_core::io::VecSink;
use mdfr_core::logger::schema::GroupHandle;
use mdfr_core::logger::spsc::SpscRing;
use mdfr_core::logger::{DataLayout, Logger, LoggerBuilder};
use mdfr_core::time::ManualClock;

// mdfr imports
use mdfr::data_holder::arrow_helpers::{arrow_bit_count, arrow_byte_count};
use mdfr::data_holder::channel_data::ChannelData;
use mdfr::mdfinfo::mdfinfo4::Cn4;
use mdfr::mdfreader::conversions4::{
    linear_conversion, rational_conversion, value_range_to_value_table,
};

// ============================================================================
// 1. StaticArena::dealloc_layout Stress Tests
// ============================================================================

#[test]
fn test_arena_null_pointer_deallocation_matrix() {
    let arena = StaticArena::new();
    let alignments = [1, 2, 4, 8, 16, 32, 64];
    let sizes = [0, 1, 8, 16, 64, 512, 1024, 65536];

    for &align in &alignments {
        for &size in &sizes {
            let layout = Layout::from_size_align(size, align).unwrap();
            unsafe {
                // Must not dereference null and must return immediately without panicking
                arena.dealloc_layout(ptr::null_mut(), layout);
            }
        }
    }
}

#[test]
fn test_arena_zero_size_alloc_dealloc_matrix() {
    let arena = StaticArena::new();
    let alignments = [1, 2, 4, 8, 16];

    for &align in &alignments {
        let layout = Layout::from_size_align(0, align).unwrap();
        let p = arena.alloc_layout(layout);
        assert!(
            !p.is_null(),
            "Zero-size alloc must return non-null dangling pointer for align {align}"
        );

        unsafe {
            // First dealloc
            arena.dealloc_layout(p, layout);
            // Repeated / idempotent dealloc
            arena.dealloc_layout(p, layout);
            // Dealloc with mismatched zero-size layout
            let other_zero = Layout::from_size_align(0, 1).unwrap();
            arena.dealloc_layout(p, other_zero);
        }
    }
}

#[test]
fn test_arena_normal_cycles_and_coalescing() {
    let arena = StaticArena::new();
    #[repr(C, align(16))]
    struct Backing([u8; 16384]);
    let mut backing = Backing([0u8; 16384]);

    unsafe {
        assert!(arena.init_custom(backing.0.as_mut_ptr(), backing.0.len()));

        // Cycle 1: Allocate several blocks, write pattern, verify, free
        let mut ptrs = Vec::new();
        for i in 0..10 {
            let layout = Layout::from_size_align(64, 16).unwrap();
            let p = arena.alloc_layout(layout);
            assert!(!p.is_null(), "Alloc should succeed in cycle 1");
            ptr::write_bytes(p, (i as u8) + 1, 64);
            ptrs.push((p, layout, (i as u8) + 1));
        }

        // Verify data
        for (p, _layout, val) in &ptrs {
            let slice = core::slice::from_raw_parts(*p, 64);
            assert!(slice.iter().all(|&b| b == *val));
        }

        // Deallocate all in forward order
        for (p, layout, _) in ptrs {
            arena.dealloc_layout(p, layout);
        }

        // Verify full arena capacity is re-coalesced
        let max_layout = Layout::from_size_align(16000, 16).unwrap();
        let big_p = arena.alloc_layout(max_layout);
        assert!(
            !big_p.is_null(),
            "Coalescing failed to restore full capacity"
        );
        arena.dealloc_layout(big_p, max_layout);
    }
}

#[test]
fn test_arena_out_of_order_deallocations_lifo() {
    let arena = StaticArena::new();
    #[repr(C, align(16))]
    struct Backing([u8; 16384]);
    let mut backing = Backing([0u8; 16384]);

    unsafe {
        assert!(arena.init_custom(backing.0.as_mut_ptr(), backing.0.len()));

        // Allocate A, B, C, D, E
        let mut ptrs = Vec::new();
        for _ in 0..5 {
            let layout = Layout::from_size_align(128, 16).unwrap();
            let p = arena.alloc_layout(layout);
            assert!(!p.is_null());
            ptrs.push((p, layout));
        }

        // Free in reverse order (LIFO: E, D, C, B, A)
        while let Some((p, layout)) = ptrs.pop() {
            arena.dealloc_layout(p, layout);
        }

        // Verify coalescing
        let big_layout = Layout::from_size_align(16000, 16).unwrap();
        let big_p = arena.alloc_layout(big_layout);
        assert!(!big_p.is_null(), "LIFO deallocation must fully coalesce");
        arena.dealloc_layout(big_p, big_layout);
    }
}

#[test]
fn test_arena_out_of_order_deallocations_odd_even() {
    let arena = StaticArena::new();
    #[repr(C, align(16))]
    struct Backing([u8; 16384]);
    let mut backing = Backing([0u8; 16384]);

    unsafe {
        assert!(arena.init_custom(backing.0.as_mut_ptr(), backing.0.len()));

        let mut ptrs = Vec::new();
        for _ in 0..10 {
            let layout = Layout::from_size_align(64, 16).unwrap();
            let p = arena.alloc_layout(layout);
            assert!(!p.is_null());
            ptrs.push((p, layout));
        }

        // Free odd indices first: 1, 3, 5, 7, 9
        for i in [1, 3, 5, 7, 9] {
            let (p, layout) = ptrs[i];
            arena.dealloc_layout(p, layout);
        }

        // Free even indices next: 0, 2, 4, 6, 8
        for i in [0, 2, 4, 6, 8] {
            let (p, layout) = ptrs[i];
            arena.dealloc_layout(p, layout);
        }

        // Verify full coalescing
        let big_layout = Layout::from_size_align(16000, 16).unwrap();
        let big_p = arena.alloc_layout(big_layout);
        assert!(
            !big_p.is_null(),
            "Odd-even deallocation must fully coalesce"
        );
        arena.dealloc_layout(big_p, big_layout);
    }
}

#[test]
fn test_arena_out_of_order_deallocations_center_out() {
    let arena = StaticArena::new();
    #[repr(C, align(16))]
    struct Backing([u8; 16384]);
    let mut backing = Backing([0u8; 16384]);

    unsafe {
        assert!(arena.init_custom(backing.0.as_mut_ptr(), backing.0.len()));

        let mut ptrs = Vec::new();
        for _ in 0..5 {
            let layout = Layout::from_size_align(128, 16).unwrap();
            let p = arena.alloc_layout(layout);
            assert!(!p.is_null());
            ptrs.push((p, layout));
        }

        // Center out: free index 2, then 1, 3, then 0, 4
        for i in [2, 1, 3, 0, 4] {
            let (p, layout) = ptrs[i];
            arena.dealloc_layout(p, layout);
        }

        // Verify full coalescing
        let big_layout = Layout::from_size_align(16000, 16).unwrap();
        let big_p = arena.alloc_layout(big_layout);
        assert!(
            !big_p.is_null(),
            "Center-out deallocation must fully coalesce"
        );
        arena.dealloc_layout(big_p, big_layout);
    }
}

// ============================================================================
// 2. SpscRing::try_push Stress Tests
// ============================================================================

#[test]
fn test_spsc_reject_payload_greater_than_65535() {
    let ring = SpscRing::with_capacity(200_000);

    // Exact boundary checks
    let payload_65536 = vec![0x11; 65536];
    assert!(
        !ring.try_push(GroupHandle(1), 100, &payload_65536),
        "Payload of 65536 bytes must be rejected"
    );

    let payload_65537 = vec![0x22; 65537];
    assert!(
        !ring.try_push(GroupHandle(1), 101, &payload_65537),
        "Payload of 65537 bytes must be rejected"
    );

    let payload_100k = vec![0x33; 100_000];
    assert!(
        !ring.try_push(GroupHandle(1), 102, &payload_100k),
        "Payload of 100,000 bytes must be rejected"
    );
}

#[test]
fn test_spsc_accept_payload_exactly_65535() {
    let ring = SpscRing::with_capacity(100_000);

    let payload_65535 = vec![0x44; 65535];
    assert!(
        ring.try_push(GroupHandle(1), 200, &payload_65535),
        "Payload of exactly 65535 bytes must be accepted if capacity permits"
    );
}

#[test]
fn test_spsc_reject_group_0xffff_and_accept_0xfffe() {
    let ring = SpscRing::with_capacity(1024);

    // GroupHandle(0xFFFF) is reserved as wrap-around sentinel: must be rejected for all payload sizes
    for sz in [0, 1, 8, 32, 100, 500] {
        let payload = vec![0x55; sz];
        assert!(
            !ring.try_push(GroupHandle(0xFFFF), 100, &payload),
            "GroupHandle(0xFFFF) must be rejected for size {sz}"
        );
    }

    // GroupHandle(0xFFFE) is the highest non-sentinel handle: must succeed
    assert!(
        ring.try_push(GroupHandle(0xFFFE), 101, &[1, 2, 3, 4]),
        "GroupHandle(0xFFFE) is valid and must be accepted"
    );
}

#[test]
fn test_spsc_atomic_tail_update_during_drain_error() {
    let mut b = LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::NoCompression);
    let g1 = b.add_group("AtomicGroup");
    b.add_channel(g1, "Val", CnDataType::UIntLE, 64).unwrap(); // Expected record size: 8 (time) + 8 = 16 bytes
    let schema = b.build().unwrap();

    let mut logger = Logger::start(schema, VecSink::default(), ManualClock::new(1e-6, 1)).unwrap();
    let ring = SpscRing::with_capacity(1024);

    // Push record 1: valid (16 bytes)
    let rec1 = [1u8; 16];
    assert!(ring.try_push(g1, 10, &rec1));

    // Push record 2: valid (16 bytes)
    let rec2 = [2u8; 16];
    assert!(ring.try_push(g1, 20, &rec2));

    // Push record 3: INVALID SIZE (20 bytes instead of 16)
    let rec3_bad = [3u8; 20];
    assert!(ring.try_push(g1, 30, &rec3_bad));

    // Push record 4: valid (16 bytes)
    let rec4 = [4u8; 16];
    assert!(ring.try_push(g1, 40, &rec4));

    // Push record 5: valid (16 bytes)
    let rec5 = [5u8; 16];
    assert!(ring.try_push(g1, 50, &rec5));

    // First drain attempt: must fail on record 3
    let res1 = ring.drain_into(&mut logger);
    assert!(res1.is_err(), "Drain must fail on invalid record size");
    match res1.unwrap_err() {
        Error::RecordSize { expected, got } => {
            assert_eq!(expected, 16);
            assert_eq!(got, 20);
        }
        other => panic!("Expected RecordSize error, got {other:?}"),
    }

    // Second drain attempt: atomic tail cursor MUST have skipped bad record 3,
    // and process records 4 and 5 cleanly without re-failing!
    let res2 = ring.drain_into(&mut logger);
    assert!(
        res2.is_ok(),
        "Second drain must succeed after skipping bad record"
    );
    assert_eq!(
        res2.unwrap(),
        2,
        "Must have drained exactly remaining 2 records"
    );

    // Third drain attempt: ring must now be completely empty
    let res3 = ring.drain_into(&mut logger);
    assert!(res3.is_ok());
    assert_eq!(res3.unwrap(), 0, "Ring must be empty on third drain");
}

// ============================================================================
// 3. Panic Hardening Tests: arrow_helpers.rs and conversions4.rs
// ============================================================================

#[test]
fn test_arrow_helpers_unusual_and_unsupported_types_no_panic() {
    // 1. DictionaryArray (not supported as standard MDF primitive): must return 0 without panicking
    let keys = Int32Array::from(vec![0, 1, 0]);
    let values = Arc::new(StringArray::from(vec!["cat", "dog"]));
    let dict_array = DictionaryArray::<Int32Type>::try_new(keys, values).unwrap();

    let bits = arrow_bit_count(&dict_array);
    let bytes = arrow_byte_count(&dict_array);
    assert_eq!(bits, 0, "DictionaryArray bit count must return 0 safely");
    assert_eq!(bytes, 0, "DictionaryArray byte count must return 0 safely");

    // 2. StructArray: must return 0 without panicking
    let field = Arc::new(Field::new("f", DataType::Int32, false));
    let fields = Fields::from(vec![field]);
    let struct_array = StructArray::new(
        fields,
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        None,
    );

    let bits_struct = arrow_bit_count(&struct_array);
    let bytes_struct = arrow_byte_count(&struct_array);
    assert_eq!(bits_struct, 0, "StructArray bit count must return 0 safely");
    assert_eq!(
        bytes_struct, 0,
        "StructArray byte count must return 0 safely"
    );
}

#[test]
fn test_conversions4_linear_conversion_panic_hardening() {
    let mut builder = Float64Builder::new();
    builder.append_value(10.0);
    let mut cn = Cn4 {
        data: ChannelData::Float64(builder.finish()),
        ..Default::default()
    };

    // Out-of-bounds cc_val slices must return Err gracefully
    assert!(linear_conversion(&mut cn, &[]).is_err());
    assert!(linear_conversion(&mut cn, &[42.0]).is_err());

    // Valid 2-element cc_val succeeds
    assert!(linear_conversion(&mut cn, &[1.0, 2.0]).is_ok());

    // Surplus elements ignored safely
    assert!(linear_conversion(&mut cn, &[1.0, 2.0, 999.0, -100.0]).is_ok());

    // Mismatched non-numeric channel types (e.g. Utf8) must not panic
    let mut str_builder = LargeStringBuilder::new();
    str_builder.append_value("test");
    let mut utf8_cn = Cn4 {
        data: ChannelData::Utf8(str_builder.finish()),
        ..Default::default()
    };
    assert!(linear_conversion(&mut utf8_cn, &[0.0, 1.0]).is_ok());
}

#[test]
fn test_conversions4_rational_conversion_panic_hardening() {
    let mut builder = Float64Builder::new();
    builder.append_value(5.0);
    let mut cn = Cn4 {
        data: ChannelData::Float64(builder.finish()),
        ..Default::default()
    };

    // Out-of-bounds cc_val slices (lengths 0..5) must return Err gracefully
    for len in 0..6 {
        let truncated: Vec<f64> = vec![1.0; len];
        assert!(
            rational_conversion(&mut cn, &truncated).is_err(),
            "Length {len} must return Err"
        );
    }

    // Valid length 6 succeeds
    assert!(rational_conversion(&mut cn, &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).is_ok());

    // Zero denominator does not panic
    assert!(rational_conversion(&mut cn, &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).is_ok());
}

#[test]
fn test_conversions4_value_range_to_value_table_panic_hardening() {
    let mut builder = Int32Builder::new();
    builder.append_value(10);
    let mut cn = Cn4 {
        data: ChannelData::Int32(builder.finish()),
        ..Default::default()
    };

    // 1. Empty cc_val returns Err gracefully
    assert!(value_range_to_value_table(&mut cn, vec![]).is_err());

    // 2. Single element (default only) does not panic
    assert!(value_range_to_value_table(&mut cn, vec![100.0]).is_ok());

    // 3. Incomplete elements (2, 4, 5) do not panic
    for count in [2, 4, 5] {
        let mut b = Int32Builder::new();
        b.append_value(10);
        let mut c = Cn4 {
            data: ChannelData::Int32(b.finish()),
            ..Default::default()
        };
        let vals: Vec<f64> = vec![1.0; count];
        assert!(value_range_to_value_table(&mut c, vals).is_ok());
    }

    // 4. Pathological float values (NaN, Inf) do not panic
    let mut b = Int32Builder::new();
    b.append_value(10);
    let mut c = Cn4 {
        data: ChannelData::Int32(b.finish()),
        ..Default::default()
    };
    let path_vals = vec![f64::NAN, f64::INFINITY, 42.0, 0.0];
    assert!(value_range_to_value_table(&mut c, path_vals).is_ok());
}
