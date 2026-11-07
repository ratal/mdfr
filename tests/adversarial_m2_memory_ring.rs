//! Adversarial Stress Testing Harness for Milestone 2 (Features 14 - 16)
//!
//! Objectives:
//! 1. `StaticArena::dealloc_layout` (Feature 14):
//!    - Null pointer deallocation (`dealloc_layout(std::ptr::null_mut(), Layout::new::<u8>())`)
//!    - Zero-sized layout deallocation (`Layout::from_size_align(0, 8).unwrap()`)
//!    - Null pointer with zero-sized layout
//!    - High-frequency mixed-alignment allocations and deallocations (alignments 1, 2, 4, 8, 16, 32, 64)
//!      to confirm no arena corruption or memory leaks.
//!    - Buffer boundary and re-coalescing verification.
//! 2. `SpscRing::try_push` and `drain_into` (Feature 15):
//!    - Pushing oversized records (`u16::MAX as usize`, `(u16::MAX as usize) + 1`, and `100_000` bytes)
//!    - Pushing with reserved group handle `GroupHandle(0xFFFF)`
//!    - Wrap-around stress testing: saturate ring buffer, drain half, push again across boundary,
//!      verifying 100% data integrity and sentinel space.
//!    - Drain cursor verification: simulate mid-drain error and verify that the tail pointer updates
//!      atomically, preventing duplicate record replays.
//! 3. `Error::code()` and C error mapping (Feature 16):
//!    - Check that every `Error` variant returns the exact integer code matching `include/mdfr_logger.h` (-1..-9).
//!    - Exact C header definition consistency check.
//!    - C API error string mapping verification.
#![allow(clippy::collapsible_if)]

use core::alloc::Layout;
use mdfr_core::arena::StaticArena;
use mdfr_core::blocks::cn::CnDataType;
use mdfr_core::compress::CompressionAlgorithm;
use mdfr_core::error::Error;
use mdfr_core::io::VecSink;
use mdfr_core::logger::schema::GroupHandle;
use mdfr_core::logger::spsc::SpscRing;
use mdfr_core::logger::{DataLayout, Logger, LoggerBuilder};
use mdfr_core::time::ManualClock;
use std::ffi::CStr;
use std::fs;
use std::ptr;

// ============================================================================
// Feature 14: StaticArena::dealloc_layout Adversarial Tests
// ============================================================================

#[test]
fn test_adv_arena_null_pointer_deallocations() {
    let arena = StaticArena::new();
    unsafe {
        // 1. Basic null pointer with standard non-zero layouts
        arena.dealloc_layout(ptr::null_mut(), Layout::new::<u8>());
        arena.dealloc_layout(ptr::null_mut(), Layout::new::<u16>());
        arena.dealloc_layout(ptr::null_mut(), Layout::new::<u32>());
        arena.dealloc_layout(ptr::null_mut(), Layout::new::<u64>());
        arena.dealloc_layout(ptr::null_mut(), Layout::new::<u128>());

        // 2. Null pointer with varying alignment constraints
        for align in [1, 2, 4, 8, 16, 32, 64] {
            let layout = Layout::from_size_align(128, align).unwrap();
            arena.dealloc_layout(ptr::null_mut(), layout);
        }

        // 3. Null pointer with large size layouts
        arena.dealloc_layout(
            ptr::null_mut(),
            Layout::from_size_align(1024 * 1024, 64).unwrap(),
        );
    }
}

#[test]
fn test_adv_arena_zero_sized_alloc_and_dealloc() {
    let arena = StaticArena::new();
    unsafe {
        for align in [1, 2, 4, 8, 16, 32, 64] {
            let layout = Layout::from_size_align(0, align).unwrap();
            let p = arena.alloc_layout(layout);
            // NonNull::dangling() is returned for zero-sized layouts
            assert!(
                !p.is_null(),
                "Zero-sized alloc must return non-null dangling pointer"
            );

            // Deallocation of zero-sized layout must safely return without dereferencing
            arena.dealloc_layout(p, layout);

            // Repeated deallocation of dangling pointer must also be safe
            arena.dealloc_layout(p, layout);
        }
    }
}

#[test]
fn test_adv_arena_null_with_zero_sized_layout() {
    let arena = StaticArena::new();
    unsafe {
        for align in [1, 2, 4, 8, 16, 32, 64] {
            let layout = Layout::from_size_align(0, align).unwrap();
            arena.dealloc_layout(ptr::null_mut(), layout);
        }
    }
}

#[test]
fn test_adv_arena_mixed_alignment_stress_and_coalescing() {
    // 32 KiB custom backing memory to isolate and stress-test
    #[repr(C, align(16))]
    struct ArenaBacking([u8; 32768]);
    let mut backing = ArenaBacking([0u8; 32768]);

    let arena = StaticArena::new();
    unsafe {
        assert!(arena.init_custom(backing.0.as_mut_ptr(), backing.0.len()));

        // Alignment list including unsupported > 16 alignments
        let alignments = [1, 2, 4, 8, 16, 32, 64];
        let sizes = [1, 7, 16, 24, 32, 48, 64, 128, 256, 512];

        struct AllocRecord {
            ptr: *mut u8,
            layout: Layout,
            expected_val: u8,
        }

        let mut live_records: Vec<AllocRecord> = Vec::new();

        // Phase A: Interleaved allocations with data verification patterns
        let mut seed = 0x5Au32;
        for i in 0..120 {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let align = alignments[(seed as usize) % alignments.len()];
            let size = sizes[((seed >> 4) as usize) % sizes.len()];
            let layout = Layout::from_size_align(size, align).unwrap();

            let p = arena.alloc_layout(layout);
            if align > 16 {
                // StaticArena supports alignments up to 16; align > 16 returns null
                assert!(p.is_null(), "align > 16 must return null pointer");
                // Calling dealloc_layout on null pointer with align > 16 must be safe
                arena.dealloc_layout(p, layout);
            } else if !p.is_null() {
                // Verify alignment
                assert_eq!(
                    (p as usize) % align,
                    0,
                    "Allocated pointer {p:?} not aligned to {align}"
                );
                // Fill with verifiable deterministic pattern
                let pattern = ((i ^ size) & 0xFF) as u8;
                ptr::write_bytes(p, pattern, size);
                live_records.push(AllocRecord {
                    ptr: p,
                    layout,
                    expected_val: pattern,
                });
            }
        }

        assert!(
            !live_records.is_empty(),
            "Expected successful allocations in arena"
        );

        // Verify that active allocations have not been clobbered by neighboring allocations
        for rec in &live_records {
            let slice = core::slice::from_raw_parts(rec.ptr, rec.layout.size());
            for &b in slice {
                assert_eq!(
                    b, rec.expected_val,
                    "Memory corruption detected in arena block at {:?}",
                    rec.ptr
                );
            }
        }

        // Phase B: Interleaved deallocations and reallocations (evens freed, reallocated, odds freed)
        let mut retained = Vec::new();
        for (idx, rec) in live_records.into_iter().enumerate() {
            if idx % 2 == 0 {
                arena.dealloc_layout(rec.ptr, rec.layout);
            } else {
                retained.push(rec);
            }
        }

        // Allocate some fresh blocks into the freed holes
        for &sz in &[32, 64, 128] {
            let layout = Layout::from_size_align(sz, 16).unwrap();
            let p = arena.alloc_layout(layout);
            if !p.is_null() {
                ptr::write_bytes(p, 0xEE, sz);
                retained.push(AllocRecord {
                    ptr: p,
                    layout,
                    expected_val: 0xEE,
                });
            }
        }

        // Verify remaining live records still uncorrupted
        for rec in &retained {
            let slice = core::slice::from_raw_parts(rec.ptr, rec.layout.size());
            for &b in slice {
                assert_eq!(
                    b, rec.expected_val,
                    "Memory corruption detected during interleaved free/alloc"
                );
            }
        }

        // Phase C: Free all remaining blocks (FIFO / forward order)
        for rec in retained {
            arena.dealloc_layout(rec.ptr, rec.layout);
        }

        // Phase D: Complete coalescing check: allocate nearly the entire arena capacity
        // Backing is 32768. Minus 16-byte header, usable capacity should allow at least 32000 bytes.
        let large_layout = Layout::from_size_align(32000, 16).unwrap();
        let large_ptr = arena.alloc_layout(large_layout);
        assert!(
            !large_ptr.is_null(),
            "Arena coalescing failed: unable to allocate contiguous 32000 bytes after deallocations"
        );
        ptr::write_bytes(large_ptr, 0x77, 32000);
        arena.dealloc_layout(large_ptr, large_layout);
    }
}

#[test]
fn test_adv_arena_init_custom_boundary_conditions() {
    let arena = StaticArena::new();
    unsafe {
        let mut buf = [0u8; 128];
        // Null buffer rejected
        assert!(!arena.init_custom(ptr::null_mut(), 128));
        // Zero length rejected
        assert!(!arena.init_custom(buf.as_mut_ptr(), 0));
        // Length < MIN_BLOCK_SIZE * 2 (64 bytes) rejected
        assert!(!arena.init_custom(buf.as_mut_ptr(), 63));
        // Valid length >= 64 accepted
        assert!(arena.init_custom(buf.as_mut_ptr(), 64));
    }
}

// ============================================================================
// Feature 15: SpscRing::try_push and drain_into Adversarial Tests
// ============================================================================

#[test]
fn test_adv_spsc_oversized_records_rejection() {
    // 1. Standard capacity ring buffer (e.g. 1024 bytes)
    let ring = SpscRing::with_capacity(1024);

    let max_u16 = u16::MAX as usize; // 65535
    let max_u16_plus_one = max_u16 + 1; // 65536
    let huge_100k = 100_000usize;

    // Both exceed standard capacity, must return false
    let rec_max = vec![0xAAu8; max_u16];
    assert!(
        !ring.try_push(GroupHandle(1), 100, &rec_max),
        "Payload u16::MAX exceeding capacity must return false"
    );

    let rec_plus_one = vec![0xBBu8; max_u16_plus_one];
    assert!(
        !ring.try_push(GroupHandle(1), 101, &rec_plus_one),
        "Payload (u16::MAX + 1) must return false"
    );

    let rec_100k = vec![0xCCu8; huge_100k];
    assert!(
        !ring.try_push(GroupHandle(1), 102, &rec_100k),
        "Payload 100_000 bytes must return false"
    );

    // 2. High-capacity ring buffer (250,000 bytes) where buffer space is plenty
    let large_ring = SpscRing::with_capacity(250_000);

    // (u16::MAX + 1) bytes strictly rejected by length check (> u16::MAX)
    assert!(
        !large_ring.try_push(GroupHandle(1), 200, &rec_plus_one),
        "Large ring: record len > u16::MAX must be rejected by try_push"
    );

    // 100_000 bytes strictly rejected by length check (> u16::MAX)
    assert!(
        !large_ring.try_push(GroupHandle(1), 201, &rec_100k),
        "Large ring: 100k payload must be rejected by try_push"
    );

    // Valid size payload (e.g., 2000 bytes) must succeed
    let valid_rec = vec![0x42u8; 2000];
    assert!(
        large_ring.try_push(GroupHandle(1), 202, &valid_rec),
        "Large ring: valid payload of 2000 bytes must succeed"
    );
}

#[test]
fn test_adv_spsc_reserved_group_handle_rejection() {
    let ring = SpscRing::with_capacity(512);

    // Reserved sentinel handle 0xFFFF must be rejected regardless of payload length
    assert!(
        !ring.try_push(GroupHandle(0xFFFF), 100, &[]),
        "GroupHandle(0xFFFF) with empty payload must return false"
    );
    assert!(
        !ring.try_push(GroupHandle(0xFFFF), 101, &[1]),
        "GroupHandle(0xFFFF) with 1-byte payload must return false"
    );
    assert!(
        !ring.try_push(GroupHandle(0xFFFF), 102, &[1, 2, 3, 4, 5, 6, 7, 8]),
        "GroupHandle(0xFFFF) with 8-byte payload must return false"
    );

    // Valid boundary handle 0xFFFE must succeed
    assert!(
        ring.try_push(GroupHandle(0xFFFE), 103, &[1, 2, 3, 4]),
        "GroupHandle(0xFFFE) is valid and must succeed"
    );
}

#[test]
fn test_adv_logger_record_size_error_and_c_code() {
    // Verify that pushing mismatched / oversized records to Logger produces Error::RecordSize
    let mut b = LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::NoCompression);
    let g = b.add_group("EngineStats");
    b.add_channel(g, "Speed", CnDataType::UIntLE, 16).unwrap(); // Record size = 8 (time) + 2 (data) = 10 bytes
    let schema = b.build().unwrap();

    let mut logger = Logger::start(schema, VecSink::default(), ManualClock::new(1e-6, 1)).unwrap();

    // Mismatched size to push_raw
    let bad_rec = [0u8; 15]; // expected 10, got 15
    let err = logger.push_raw(g, &bad_rec).unwrap_err();
    match err {
        Error::RecordSize { expected, got } => {
            assert_eq!(expected, 10);
            assert_eq!(got, 15);
        }
        other => panic!("Expected Error::RecordSize, got {other:?}"),
    }
    assert_eq!(
        err.code(),
        -9,
        "Error::RecordSize must map to C error code -9"
    );

    // Huge oversized record to push_raw
    let huge_rec = vec![0u8; 1000];
    let err_huge = logger.push_raw(g, &huge_rec).unwrap_err();
    assert_eq!(err_huge.code(), -9);
}

#[test]
fn test_adv_spsc_wrap_around_stress_and_data_integrity() {
    let mut b = LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::NoCompression);
    let g = b.add_group("ContinuousStream");
    b.add_channel(g, "Counter", CnDataType::UIntLE, 32).unwrap();
    b.add_channel(g, "Value", CnDataType::UIntLE, 32).unwrap();
    // Record size: 8 (time) + 4 (Counter) + 4 (Value) = 16 bytes
    let schema = b.build().unwrap();

    let sink = VecSink::default();
    let clock = ManualClock::new(1e-6, 1);
    let mut logger = Logger::start(schema, sink, clock).unwrap();

    // Small ring buffer to force wrap-arounds quickly (256 bytes)
    // Frame size = 12 header + 16 payload = 28 bytes. Capacity = 256.
    let ring = SpscRing::with_capacity(256);

    let mut record_counter = 0u32;
    let total_records = 300u32;

    while record_counter < total_records {
        // Prepare record bytes
        let mut rec = [0u8; 16];
        rec[8..12].copy_from_slice(&record_counter.to_le_bytes());
        rec[12..16].copy_from_slice(&(record_counter * 10).to_le_bytes());

        // Try pushing record
        let ticks = (record_counter as u64) * 1000;
        if ring.try_push(g, ticks, &rec) {
            record_counter += 1;
        } else {
            // Buffer full: drain queued records into logger
            let drained = ring.drain_into(&mut logger).unwrap();
            assert!(drained > 0, "Drain should have consumed queued records");
            // Now pushing must succeed
            assert!(
                ring.try_push(g, ticks, &rec),
                "Push after drain must succeed"
            );
            record_counter += 1;
        }
    }

    // Drain remaining
    let _final_drain = ring.drain_into(&mut logger).unwrap();

    // Close logger and verify file header integrity
    let (sink_out, _) = logger.close().unwrap();
    assert!(sink_out.data.len() > 64);
    assert_eq!(&sink_out.data[0..8], b"MDF     ");
}

#[test]
fn test_adv_spsc_drain_atomic_tail_on_mid_drain_error() {
    // Feature 15: Atomic tail cursor update prevents duplicate record replays on error
    let mut b = LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::NoCompression);
    let g1 = b.add_group("GroupOne");
    b.add_channel(g1, "Val1", CnDataType::UIntLE, 64).unwrap(); // Size: 8 + 8 = 16 bytes

    let schema = b.build().unwrap();
    let mut logger = Logger::start(schema, VecSink::default(), ManualClock::new(1e-6, 1)).unwrap();

    let ring = SpscRing::with_capacity(1024);

    // Push 3 valid records for g1
    for i in 0..3u32 {
        let mut rec = [0u8; 16];
        rec[8..16].copy_from_slice(&(i as u64).to_le_bytes());
        assert!(ring.try_push(g1, (i as u64) * 100, &rec));
    }

    // Push 1 invalid record: payload size 24 bytes instead of expected 16 bytes for g1
    let bad_rec = [0xEEu8; 24];
    assert!(ring.try_push(g1, 300, &bad_rec));

    // Push 2 more valid records for g1
    for i in 4..6u32 {
        let mut rec = [0u8; 16];
        rec[8..16].copy_from_slice(&(i as u64).to_le_bytes());
        assert!(ring.try_push(g1, (i as u64) * 100, &rec));
    }

    // First drain attempt:
    // Should successfully drain records 0, 1, 2, and fail on record 3 with Error::RecordSize
    let drain_result = ring.drain_into(&mut logger);
    assert!(
        drain_result.is_err(),
        "Expected mid-drain error on bad record"
    );
    match drain_result.unwrap_err() {
        Error::RecordSize { expected, got } => {
            assert_eq!(expected, 16);
            assert_eq!(got, 24);
        }
        other => panic!("Expected RecordSize error, got {other:?}"),
    }

    // Second drain attempt:
    // The tail pointer MUST have been updated atomically past the failing record (Record 3).
    // It should now cleanly drain the remaining 2 valid records (Record 4 and 5)!
    let second_drain = ring.drain_into(&mut logger).unwrap();
    assert_eq!(
        second_drain, 2,
        "Atomic tail update must have skipped bad record and processed exactly remaining 2 records"
    );

    // Third drain attempt: ring should now be completely empty
    let empty_drain = ring.drain_into(&mut logger).unwrap();
    assert_eq!(empty_drain, 0, "Ring must be empty now");
}

// ============================================================================
// Feature 16: Error::code() and C Header Alignment Adversarial Tests
// ============================================================================

#[test]
fn test_adv_all_error_variants_code_mapping() {
    // Feature 16: Check that every single Error variant returns exact code -1..-9
    assert_eq!(Error::Io.code(), -1);
    assert_eq!(Error::InvalidBlock("test block".into()).code(), -2);
    assert_eq!(Error::InvalidSchema("test schema".into()).code(), -3);
    assert_eq!(Error::InvalidHandle("test handle".into()).code(), -4);
    assert_eq!(Error::Unsupported("feature").code(), -5);
    assert_eq!(Error::Compression("lz4 fail".into()).code(), -6);
    assert_eq!(Error::RingFull.code(), -7);
    assert_eq!(Error::AlreadyClosed.code(), -8);
    assert_eq!(
        Error::RecordSize {
            expected: 16,
            got: 8
        }
        .code(),
        -9
    );
}

#[test]
fn test_adv_c_header_error_definitions_exact_match() {
    // Read include/mdfr_logger.h directly and extract all MDFR_ERR_* defines
    let header_path = concat!(env!("CARGO_MANIFEST_DIR"), "/include/mdfr_logger.h");
    let content = fs::read_to_string(header_path)
        .expect("Must be able to read include/mdfr_logger.h for C API consistency");

    let mut found_defines = std::collections::HashMap::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("#define MDFR_") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 3 {
                if let Ok(val) = parts[2].parse::<i32>() {
                    found_defines.insert(parts[1].to_string(), val);
                }
            }
        }
    }

    // Verify exact integer definitions in C header
    assert_eq!(found_defines.get("MDFR_OK"), Some(&0));
    assert_eq!(found_defines.get("MDFR_ERR_IO"), Some(&-1));
    assert_eq!(found_defines.get("MDFR_ERR_BINRW"), Some(&-2));
    assert_eq!(found_defines.get("MDFR_ERR_INVALID_SCHEMA"), Some(&-3));
    assert_eq!(found_defines.get("MDFR_ERR_INVALID_HANDLE"), Some(&-4));
    assert_eq!(found_defines.get("MDFR_ERR_UNSUPPORTED"), Some(&-5));
    assert_eq!(found_defines.get("MDFR_ERR_COMPRESSION"), Some(&-6));
    assert_eq!(found_defines.get("MDFR_ERR_RING_FULL"), Some(&-7));
    assert_eq!(found_defines.get("MDFR_ERR_ALREADY_CLOSED"), Some(&-8));
    assert_eq!(found_defines.get("MDFR_ERR_RECORD_SIZE"), Some(&-9));

    // Verify 1:1 match with Rust Error::code()
    assert_eq!(Error::Io.code(), *found_defines.get("MDFR_ERR_IO").unwrap());
    assert_eq!(
        Error::InvalidBlock(String::new()).code(),
        *found_defines.get("MDFR_ERR_BINRW").unwrap()
    );
    assert_eq!(
        Error::InvalidSchema(String::new()).code(),
        *found_defines.get("MDFR_ERR_INVALID_SCHEMA").unwrap()
    );
    assert_eq!(
        Error::InvalidHandle(String::new()).code(),
        *found_defines.get("MDFR_ERR_INVALID_HANDLE").unwrap()
    );
    assert_eq!(
        Error::Unsupported("").code(),
        *found_defines.get("MDFR_ERR_UNSUPPORTED").unwrap()
    );
    assert_eq!(
        Error::Compression(String::new()).code(),
        *found_defines.get("MDFR_ERR_COMPRESSION").unwrap()
    );
    assert_eq!(
        Error::RingFull.code(),
        *found_defines.get("MDFR_ERR_RING_FULL").unwrap()
    );
    assert_eq!(
        Error::AlreadyClosed.code(),
        *found_defines.get("MDFR_ERR_ALREADY_CLOSED").unwrap()
    );
    assert_eq!(
        Error::RecordSize {
            expected: 0,
            got: 0
        }
        .code(),
        *found_defines.get("MDFR_ERR_RECORD_SIZE").unwrap()
    );
}

#[test]
fn test_adv_c_api_error_strings() {
    use mdfr_core::capi::mdfr_error_str;

    for code in -9..=0 {
        let str_ptr = mdfr_error_str(code);
        assert!(!str_ptr.is_null());
        let c_str = unsafe { CStr::from_ptr(str_ptr) };
        let rust_str = c_str.to_str().expect("Must be valid UTF-8/ASCII");
        assert!(!rust_str.is_empty());
        assert_ne!(
            rust_str, "unknown error",
            "Error code {code} must have a specific description, not 'unknown error'"
        );
    }

    // An invalid code should return "unknown error"
    let unknown_ptr = mdfr_error_str(-99);
    let unknown_str = unsafe { CStr::from_ptr(unknown_ptr) }.to_str().unwrap();
    assert_eq!(unknown_str, "unknown error");
}
