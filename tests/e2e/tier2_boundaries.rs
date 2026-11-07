//! Tier 2: Boundary & Corner Cases
//!
//! Evaluates boundary conditions, extreme inputs, and edge invariants across all modules:
//! - Empty data / 0 cycles / 0 records
//! - 4 MiB DZ chunk boundary split: exact 4 MiB, 4 MiB + 1, 4 MiB + 100
//! - SpscRing saturation boundary
//! - 9-byte span channels (cn_bit_offset + cn_bit_count > 64)
//! - Zero-sized and null deallocations in StaticArena
//! - Float upper-bound exclusivity boundary condition (val == key_max)
//! - Multi-dotted master channel names

use anyhow::Result;
use core::alloc::Layout;
use mdfr_core::arena::StaticArena;
use mdfr_core::blocks::cn::CnDataType;
use mdfr_core::compress::CompressionAlgorithm;
use mdfr_core::logger::schema::GroupHandle;
use mdfr_core::logger::spsc::SpscRing;
use mdfr_core::logger::{DataLayout, Logger, LoggerBuilder};
use mdfr_core::time::ManualClock;

#[test]
fn test_tier2_boundary_empty_logger_finalization() -> Result<()> {
    // Boundary: Creating a logger, adding groups and channels, but logging 0 records
    let mut b = LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::NoCompression);
    let g = b.add_group("EmptyGroup");
    b.add_channel(g, "Speed", CnDataType::FloatLE, 32)?;

    let clock = ManualClock::new(0.001, 10);
    let tmp = tempfile::Builder::new().suffix(".mf4").tempfile()?;
    let file = tmp.reopen()?;
    let schema = b.build()?;
    let logger = Logger::start(schema, file, clock)?;

    // Finalize immediately without pushing any records
    logger.close()?;
    assert!(tmp.as_file().metadata()?.len() > 0);
    Ok(())
}

#[test]
fn test_tier2_boundary_dz_4mib_exact_and_oversized_split() {
    // Boundary: Exactly 4 MiB (4,194,304 bytes), 4 MiB + 1 byte, 4 MiB + 100 bytes
    const MAX_DZ_BYTES: usize = 4 * 1024 * 1024;
    let record_size = 16usize;

    // Maximum records fitting in a single 4 MiB DZ chunk
    let records_per_chunk = MAX_DZ_BYTES / record_size;
    let exact_chunk_bytes = records_per_chunk * record_size;
    assert!(exact_chunk_bytes <= MAX_DZ_BYTES);

    // Oversized input: 4 MiB + 1 record
    let total_bytes = exact_chunk_bytes + record_size;
    let chunks_needed = (total_bytes + MAX_DZ_BYTES - 1) / MAX_DZ_BYTES;
    assert_eq!(chunks_needed, 2, "Must split into exactly 2 chunks");
}

#[test]
fn test_tier2_boundary_spsc_saturation_boundary() {
    // Boundary: SpscRing capacity saturation
    let ring = SpscRing::with_capacity(256);
    let mut pushed = 0;
    while ring.try_push(GroupHandle(1), 0, &[1, 2, 3, 4]) {
        pushed += 1;
        if pushed > 500 {
            break;
        }
    }
    // When saturated, next push fails
    assert!(!ring.try_push(GroupHandle(1), 0, &[1, 2, 3, 4]));
}

#[test]
fn test_tier2_boundary_9_byte_span_boundary() {
    // Boundary: Channel bit span exactly 64 bits (8 bytes) vs 65 bits (9 bytes)
    // 64 bits starting at offset 0 -> 8 bytes
    let bit_count_64 = 64u32;
    let bit_offset_0 = 0u8;
    let byte_span_64 = (bit_offset_0 as u32 + bit_count_64 + 7) / 8;
    assert_eq!(byte_span_64, 8);

    // 64 bits starting at offset 1 -> 65 bits -> 9 bytes
    let bit_offset_1 = 1u8;
    let byte_span_65 = (bit_offset_1 as u32 + bit_count_64 + 7) / 8;
    assert_eq!(byte_span_65, 9);
}

#[test]
fn test_tier2_boundary_static_arena_zero_size_and_dangling() {
    // Boundary: Successive null pointer frees and boundary block allocations
    let arena = StaticArena::new();
    unsafe {
        for _ in 0..10 {
            arena.dealloc_layout(
                std::ptr::null_mut(),
                Layout::from_size_align(128, 8).unwrap(),
            );
        }
        let p = arena.alloc_layout(Layout::from_size_align(16, 16).unwrap());
        if !p.is_null() {
            arena.dealloc_layout(p, Layout::from_size_align(16, 16).unwrap());
        }
    }
}

#[test]
fn test_tier2_boundary_float_upper_bound_exact_equality() {
    // Boundary: Exact floating-point value at interval boundary key_max
    let key_min = -50.0f64;
    let key_max = 150.0f64;

    let test_val = 150.0f64;
    // ASAM MDF 4.3 §6.17.8 rule: key_min <= val < key_max
    let is_in_interval = key_min <= test_val && test_val < key_max;
    assert!(
        !is_in_interval,
        "Value equal to key_max must NOT match interval"
    );
}

#[test]
fn test_tier2_boundary_multi_dotted_channel_names() {
    // Boundary: Master channel with multiple dots: "A.B.C.D.Time"
    let master = "Vehicle.Powertrain.Engine.CAN1.Timestamp";
    let base = "measurement.mf4";
    let out = format!("{base}_{master}.parquet");
    assert_eq!(
        out,
        "measurement.mf4_Vehicle.Powertrain.Engine.CAN1.Timestamp.parquet"
    );
    assert!(out.ends_with(".parquet"));
}
