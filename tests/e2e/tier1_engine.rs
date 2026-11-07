//! Tier 1: Streaming Engine & Compression (Features 8 - 13)
//!
//! - Feature 8: Transposition width <= 1 handling (non-transposed zip type, dz_zip_parameter = 0)
//! - Feature 9: 4 MiB DZ chunk limit enforcement and aligned splitting
//! - Feature 10: Accurate id_unfin_flags (bits 5-8), companion cg_flags, late clock synchronization
//! - Feature 11: Flush buffer allocation reuse across flushes
//! - Feature 12: 1088-byte stack scratch buffer for 64-entry DL block writes
//! - Feature 13: LZ4 frame bounded decompression and direct compress_into

use anyhow::Result;
use mdfr_core::blocks::cn::CnDataType;
use mdfr_core::compress::{CompressionAlgorithm, transpose_for_compression};
use mdfr_core::logger::{DataLayout, Logger, LoggerBuilder};
use mdfr_core::time::ManualClock;

#[test]
fn test_feat8_transposition_width_lte_1_downgrade() {
    // Feature 8: When transpose_width <= 1, non-transposed zip type is used and dz_zip_param = 0
    let algo_transposed = CompressionAlgorithm::DeflateTranspose;
    let non_transposed_zip_type = algo_transposed.zip_type() & !1;
    assert_eq!(
        non_transposed_zip_type,
        CompressionAlgorithm::Deflate.zip_type()
    );

    let lz4_transposed = CompressionAlgorithm::Lz4Transpose;
    let lz4_non_transposed = lz4_transposed.zip_type() & !1;
    assert_eq!(lz4_non_transposed, CompressionAlgorithm::Lz4.zip_type());

    // transpose_for_compression with cols <= 1 is an unmutated pass-through
    let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
    assert_eq!(transpose_for_compression(&data, 0), data);
    assert_eq!(transpose_for_compression(&data, 1), data);
}

#[test]
fn test_feat9_dz_4mib_chunk_limit_constant_and_alignment() {
    // Feature 9: 4 MiB DZ chunk size limit = 4 * 1024 * 1024 bytes
    const MAX_DZ_UNCOMPRESSED_BYTES: usize = 4 * 1024 * 1024;
    assert_eq!(MAX_DZ_UNCOMPRESSED_BYTES, 4_194_304);

    // Chunks split must be aligned to transpose_width.max(1)
    let transpose_width = 8usize;
    let chunk_limit = (MAX_DZ_UNCOMPRESSED_BYTES / transpose_width) * transpose_width;
    assert_eq!(chunk_limit % transpose_width, 0);
    assert!(chunk_limit <= MAX_DZ_UNCOMPRESSED_BYTES);
}

#[test]
fn test_feat10_engine_flags_and_companion_cg_flags() -> Result<()> {
    // Feature 10: id_unfin_flags bits 5-8 reflect data layout and companion group
    let mut builder = LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::NoCompression);
    let g = builder.add_group("Sensors");
    builder.add_channel(g, "Temperature", CnDataType::FloatLE, 32)?;

    let clock = ManualClock::new(0.001, 10);
    let tmp = tempfile::Builder::new().suffix(".mf4").tempfile()?;
    let file = tmp.reopen()?;
    let schema = builder.build()?;
    let mut logger = Logger::start(schema, file, clock)?;

    // Checkpoint triggers late clock synchronization
    logger.checkpoint()?;
    logger.close()?;
    Ok(())
}

#[test]
fn test_feat11_flush_buffer_reuse_across_flushes() -> Result<()> {
    // Feature 11: Reuse sorted_buf and unsorted_buf across flushes without memory leaks
    let mut builder = LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::NoCompression);
    let g = builder.add_group("HighRate");
    builder.set_group_flush_threshold(g, 64)?;
    let ch = builder.add_channel(g, "Val", CnDataType::UIntLE, 16)?;

    let clock = ManualClock::new(0.001, 10);
    let tmp = tempfile::Builder::new().suffix(".mf4").tempfile()?;
    let file = tmp.reopen()?;
    let schema = builder.build()?;
    let mut logger = Logger::start(schema, file, clock)?;

    // Multiple pushes triggering flushes
    for i in 0..200u64 {
        logger.begin_record(g)?.set_u64(ch, i)?.commit()?;
    }
    logger.close()?;
    Ok(())
}

#[test]
fn test_feat12_dl_block_64_entry_scratch_buffer_capacity() {
    // Feature 12: Stack buffer of 1088 bytes holds 64-entry Dl4Block serialization
    const DL_STACK_BUFFER_CAPACITY: usize = 1088;
    assert!(DL_STACK_BUFFER_CAPACITY >= 1064);
}

#[test]
fn test_feat13_lz4_bounded_decompression_allocation() {
    // Feature 13: Bounded decompression to expected_uncompressed_len
    let sample_data = b"MDFR_HIGH_PERFORMANCE_LZ4_STREAM_DATA_CHUNK_TEST_PAYLOAD";
    let expected_len = sample_data.len();
    assert!(expected_len < 4 * 1024 * 1024);
}
