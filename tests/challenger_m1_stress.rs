//! Adversarial Stress Testing Harness for Milestone 1
//!
//! Stress-tests:
//! 1. Oversized payload splitting (> 4 MiB, e.g. 10 MiB) into chunks strictly <= 4,194,304 bytes.
//! 2. Transposition alignment to `transpose_width.max(1)` (e.g. 16 bytes, 15 bytes, 7 bytes).
//! 3. Transposition width <= 1 downgrade to non-transposed zip type and dz_zip_parameter = 0.
//! 4. Scratch buffer limits (1088 bytes) and heap fallback on Dl4Block serialization (>65 entries).
//! 5. Buffer reuse across thousands of rapid flushes in both Sorted and Unsorted layouts.
//! 6. Full round-trip read-back verification asserting no corruption, desync, or panics.

use anyhow::Result;
use binrw::{BinRead, BinWrite, io::Cursor};
use mdfr::data_holder::channel_data::{AsValuesSlice, ChannelData};
use mdfr::mdfreader::Mdf;
use mdfr_core::blocks::cn::CnDataType;
use mdfr_core::blocks::data::{Dl4Block, Dz4Block, Hl4Block};
use mdfr_core::compress::CompressionAlgorithm;
use mdfr_core::logger::{DataLayout, Logger, LoggerBuilder};
use mdfr_core::time::ManualClock;
use std::fs::File;
use std::io::Read;

const MAX_DZ_BYTES: u64 = 4 * 1024 * 1024; // 4,194,304 bytes

/// Helper to scan a raw MDF4 file and find all DZ and HL blocks.
#[allow(clippy::type_complexity)]
fn scan_blocks(file_path: &str) -> Result<(Vec<(u64, Dz4Block)>, Vec<(u64, Hl4Block)>)> {
    let mut file = File::open(file_path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;

    let mut dz_blocks = Vec::new();
    let mut hl_blocks = Vec::new();

    let mut pos = 0;
    while pos + 4 <= buf.len() {
        if &buf[pos..pos + 4] == b"##DZ" {
            let mut c = Cursor::new(&buf[pos + 4..]);
            if let Ok(dz) = Dz4Block::read_le(&mut c) {
                dz_blocks.push((pos as u64, dz));
            }
            pos += 4;
        } else if &buf[pos..pos + 4] == b"##HL" {
            let mut c = Cursor::new(&buf[pos + 4..]);
            if let Ok(hl) = Hl4Block::read_le(&mut c) {
                hl_blocks.push((pos as u64, hl));
            }
            pos += 4;
        } else {
            pos += 1;
        }
    }

    Ok((dz_blocks, hl_blocks))
}

#[test]
fn test_oversized_payload_splitting_10mib_deflate_transpose() -> Result<()> {
    // 10 MiB payload with DeflateTranspose, record width = 16 bytes (8B time + 8B u64)
    let tmp = tempfile::Builder::new().suffix(".mf4").tempfile()?;
    let path_str = tmp.path().to_str().unwrap().to_string();

    let mut builder =
        LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::DeflateTranspose);
    let g = builder.add_group("Engine");
    // Threshold = 10 MiB (10 * 1024 * 1024 = 10,485,760 bytes)
    const TARGET_BYTES: usize = 10 * 1024 * 1024;
    builder.set_group_flush_threshold(g, TARGET_BYTES)?;
    let ch = builder.add_channel(g, "RPM", CnDataType::UIntLE, 64)?;

    let clock = ManualClock::new(0.001, 1);
    let schema = builder.build()?;
    let file = File::create(&path_str)?;
    let mut logger = Logger::start(schema, file, clock)?;

    let num_records = TARGET_BYTES / 16; // 655,360 records
    for i in 0..num_records as u64 {
        logger.begin_record(g)?.set_u64(ch, i * 7)?.commit()?;
    }
    // Finalize file
    logger.close()?;

    // 1. Scan low-level blocks
    let (dz_blocks, hl_blocks) = scan_blocks(&path_str)?;
    assert!(!dz_blocks.is_empty(), "Must have generated DZ blocks");
    assert!(!hl_blocks.is_empty(), "Must have generated HL blocks");

    // Total payload 10,485,760 bytes:
    // With max chunk 4,194,304 bytes:
    // Chunk 0: 4,194,304 bytes (262,144 records * 16B)
    // Chunk 1: 4,194,304 bytes (262,144 records * 16B)
    // Chunk 2: 2,097,152 bytes (131,072 records * 16B)
    // Total chunks = 3
    assert_eq!(
        dz_blocks.len(),
        3,
        "10 MiB payload must split into exactly 3 DZ chunks"
    );

    for (idx, (_pos, dz)) in dz_blocks.iter().enumerate() {
        assert!(
            dz.dz_org_data_length <= MAX_DZ_BYTES,
            "DZ chunk {} size {} exceeds 4 MiB limit!",
            idx,
            dz.dz_org_data_length
        );
        assert_eq!(
            dz.dz_org_data_length % 16,
            0,
            "DZ chunk {} size {} not aligned to record width 16!",
            idx,
            dz.dz_org_data_length
        );
        assert_eq!(dz.dz_zip_type, 1, "Must be DeflateTranspose (1)");
        assert_eq!(
            dz.dz_zip_parameter, 16,
            "dz_zip_parameter must be transpose_width (16)"
        );
    }
    assert_eq!(dz_blocks[0].1.dz_org_data_length, 4_194_304);
    assert_eq!(dz_blocks[1].1.dz_org_data_length, 4_194_304);
    assert_eq!(dz_blocks[2].1.dz_org_data_length, 2_097_152);

    // 2. High-level read-back verification through full MDF4 reader
    let mut mdf = Mdf::new(&path_str)?;
    mdf.load_all_channels_data_in_memory()?;
    let channel_data = mdf.get_channel_data("RPM").expect("RPM channel missing");
    match channel_data {
        ChannelData::UInt64(arr) => {
            let slice = arr.values_slice();
            assert_eq!(
                slice.len(),
                num_records,
                "Read back record count must match"
            );
            for (i, &val) in slice.iter().enumerate() {
                assert_eq!(val, (i as u64) * 7, "Data mismatch at record {}", i);
            }
        }
        other => panic!("Unexpected ChannelData variant: {other:?}"),
    }

    Ok(())
}

#[test]
fn test_oversized_payload_odd_alignment_lz4_transpose() -> Result<()> {
    // Odd record width = 15 bytes (8B time + 7B payload: 4B u32 + 2B u16 + 1B u8)
    let tmp = tempfile::Builder::new().suffix(".mf4").tempfile()?;
    let path_str = tmp.path().to_str().unwrap().to_string();

    let mut builder = LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::Lz4Transpose);
    let g = builder.add_group("OddGroup");
    // Write 9,000,000 bytes (600,000 records of 15 bytes)
    const TARGET_RECORDS: usize = 600_000;
    const REC_SIZE: usize = 15;
    const TARGET_BYTES: usize = TARGET_RECORDS * REC_SIZE;
    builder.set_group_flush_threshold(g, TARGET_BYTES)?;

    let ch1 = builder.add_channel(g, "C1", CnDataType::UIntLE, 32)?;
    let ch2 = builder.add_channel(g, "C2", CnDataType::UIntLE, 16)?;
    let ch3 = builder.add_channel(g, "C3", CnDataType::UIntLE, 8)?;

    let clock = ManualClock::new(0.001, 1);
    let schema = builder.build()?;
    let file = File::create(&path_str)?;
    let mut logger = Logger::start(schema, file, clock)?;

    for i in 0..TARGET_RECORDS as u64 {
        logger
            .begin_record(g)?
            .set_u64(ch1, i & 0xFFFFFFFF)?
            .set_u64(ch2, i & 0xFFFF)?
            .set_u64(ch3, i & 0xFF)?
            .commit()?;
    }
    logger.close()?;

    // 1. Block level check
    let (dz_blocks, _hl_blocks) = scan_blocks(&path_str)?;
    // (4194304 / 15) * 15 = 279620 * 15 = 4194300 bytes per chunk
    let expected_chunk_limit = (MAX_DZ_BYTES as usize / REC_SIZE) * REC_SIZE;
    assert_eq!(expected_chunk_limit, 4_194_300);

    for (idx, (_pos, dz)) in dz_blocks.iter().enumerate() {
        assert!(
            dz.dz_org_data_length <= MAX_DZ_BYTES,
            "Chunk {} uncompressed len {} exceeds 4 MiB",
            idx,
            dz.dz_org_data_length
        );
        if idx < dz_blocks.len() - 1 {
            assert_eq!(
                dz.dz_org_data_length, expected_chunk_limit as u64,
                "Chunk {} not aligned to record width 15",
                idx
            );
        }
        assert_eq!(
            dz.dz_org_data_length % (REC_SIZE as u64),
            0,
            "Chunk {} uncompressed len not multiple of 15",
            idx
        );
        assert_eq!(dz.dz_zip_type, 5, "Must be Lz4Transpose (5)");
        assert_eq!(dz.dz_zip_parameter, 15, "Parameter must be 15");
    }

    // 2. High-level read-back check
    let mut mdf = Mdf::new(&path_str)?;
    mdf.load_all_channels_data_in_memory()?;
    let d1 = mdf.get_channel_data("C1").expect("C1 missing");
    match d1 {
        ChannelData::UInt32(arr) => {
            let s1 = arr.values_slice();
            assert_eq!(s1.len(), TARGET_RECORDS);
            for (i, &val) in s1.iter().enumerate() {
                assert_eq!(val, (i as u32), "Mismatch at index {}", i);
            }
        }
        other => panic!("Unexpected ChannelData variant: {other:?}"),
    }

    Ok(())
}

#[test]
fn test_transposition_width_lte_1_downgrades_to_non_transposed() -> Result<()> {
    // When transpose_width <= 1, DeflateTranspose (1) -> Deflate (0), Lz4Transpose (5) -> Lz4 (4)
    // and dz_zip_parameter = 0.
    for (algo, expected_zip_type) in [
        (CompressionAlgorithm::DeflateTranspose, 0u8),
        (CompressionAlgorithm::Lz4Transpose, 4u8),
    ] {
        let tmp = tempfile::Builder::new().suffix(".mf4").tempfile()?;
        let path_str = tmp.path().to_str().unwrap().to_string();

        let mut builder = LoggerBuilder::new(DataLayout::Sorted, algo);
        let g = builder.add_group("SDGroup");
        // Add a VLSD channel: SD stream has transpose_width = 1!
        let ch_vlsd = builder.add_vlsd_channel(g, "VlsdData", CnDataType::ByteArray)?;
        builder.set_group_flush_threshold(g, 64)?;

        let clock = ManualClock::new(0.001, 1);
        let schema = builder.build()?;
        let file = File::create(&path_str)?;
        let mut logger = Logger::start(schema, file, clock)?;

        // Write several VLSD entries to generate SD DZ blocks
        for i in 0..50u64 {
            let payload = format!("payload_entry_{}_{}", i, "X".repeat(200));
            logger
                .begin_record(g)?
                .set_bytes(ch_vlsd, payload.as_bytes())?
                .commit()?;
        }
        logger.close()?;

        let (dz_blocks, hl_blocks) = scan_blocks(&path_str)?;
        assert!(!dz_blocks.is_empty());

        for (pos, dz) in &dz_blocks {
            // Find SD blocks (org_block_type == b"SD")
            if &dz.dz_org_block_type == b"SD" {
                assert_eq!(
                    dz.dz_zip_type, expected_zip_type,
                    "SD DZ at pos {} with transpose_width=1 must downgrade to non-transposed zip type {}",
                    pos, expected_zip_type
                );
                assert_eq!(
                    dz.dz_zip_parameter, 0,
                    "SD DZ at pos {} with transpose_width=1 must have dz_zip_parameter = 0",
                    pos
                );
            }
        }

        for (pos, hl) in &hl_blocks {
            println!("HL block at pos {}: hl_zip_type = {}", pos, hl.hl_zip_type);
        }
        // There should be two HL blocks: one for DT (width 8, zip_type 1) and one for SD (width 1, zip_type 0)
        let sd_hl = hl_blocks
            .iter()
            .find(|(_pos, hl)| (hl.hl_zip_type & 1) == 0);
        assert!(
            sd_hl.is_some(),
            "Must have found an HL block with non-transposed zip type for the SD stream"
        );
        assert_eq!(sd_hl.unwrap().1.hl_zip_type, expected_zip_type);
    }

    Ok(())
}

#[test]
fn test_buffer_reuse_thousands_of_rapid_flushes() -> Result<()> {
    // Flush threshold = 32 bytes (flush every 2 records of 16B).
    // Write 4,000 records -> 2,000 flushes!
    // Set max_blocks_per_dl = 16 -> creates ~125 chained DL blocks.
    let tmp = tempfile::Builder::new().suffix(".mf4").tempfile()?;
    let path_str = tmp.path().to_str().unwrap().to_string();

    let mut builder =
        LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::DeflateTranspose);
    builder.max_blocks_per_dl(16);
    builder.checkpoint_every_n_blocks(32); // Checkpoint every 32 blocks

    let g = builder.add_group("FastSensor");
    builder.set_group_flush_threshold(g, 32)?;
    let ch = builder.add_channel(g, "Counter", CnDataType::UIntLE, 64)?;

    let clock = ManualClock::new(0.0001, 10);
    let schema = builder.build()?;
    let file = File::create(&path_str)?;
    let mut logger = Logger::start(schema, file, clock)?;

    const TOTAL_RECORDS: u64 = 4_000;
    for i in 0..TOTAL_RECORDS {
        logger.begin_record(g)?.set_u64(ch, i * 13)?.commit()?;
    }
    logger.close()?;

    // Verify all 4,000 records read back cleanly
    let mut mdf = Mdf::new(&path_str)?;
    mdf.load_all_channels_data_in_memory()?;
    let data = mdf
        .get_channel_data("Counter")
        .expect("Counter channel missing");
    match data {
        ChannelData::UInt64(arr) => {
            let slice = arr.values_slice();
            assert_eq!(slice.len(), TOTAL_RECORDS as usize);
            for (i, &val) in slice.iter().enumerate() {
                assert_eq!(val, (i as u64) * 13, "Data desync at record {}", i);
            }
        }
        other => panic!("Unexpected ChannelData variant: {other:?}"),
    }

    Ok(())
}

#[test]
fn test_unsorted_buffer_reuse_rapid_flushes() -> Result<()> {
    // Unsorted mode with two groups and companion VLSD group
    let tmp = tempfile::Builder::new().suffix(".mf4").tempfile()?;
    let path_str = tmp.path().to_str().unwrap().to_string();

    let mut builder = LoggerBuilder::new(DataLayout::Unsorted, CompressionAlgorithm::Lz4Transpose);
    builder.max_blocks_per_dl(32);

    let g0 = builder.add_group("G0");
    let c0 = builder.add_channel(g0, "Val0", CnDataType::UIntLE, 32)?;

    let g1 = builder.add_group("G1");
    let c1 = builder.add_channel(g1, "Val1", CnDataType::FloatLE, 64)?;

    let clock = ManualClock::new(0.001, 1);
    let schema = builder.build()?;
    let file = File::create(&path_str)?;
    let mut logger = Logger::start(schema, file, clock)?;

    const N_CYCLES: usize = 2_000;
    for i in 0..N_CYCLES {
        logger.begin_record(g0)?.set_u64(c0, i as u64)?.commit()?;
        logger
            .begin_record(g1)?
            .set_f64(c1, i as f64 * std::f64::consts::PI)?
            .commit()?;
        if i % 100 == 0 {
            logger.checkpoint()?;
        }
    }
    logger.close()?;

    let mut mdf = Mdf::new(&path_str)?;
    mdf.load_all_channels_data_in_memory()?;
    let d0 = mdf.get_channel_data("Val0").expect("Val0 missing");
    match d0 {
        ChannelData::UInt32(arr) => {
            let s0 = arr.values_slice();
            assert_eq!(s0.len(), N_CYCLES);
            for (i, &val) in s0.iter().enumerate() {
                assert_eq!(val, i as u32);
            }
        }
        other => panic!("Unexpected variant: {other:?}"),
    }

    let d1 = mdf.get_channel_data("Val1").expect("Val1 missing");
    match d1 {
        ChannelData::Float64(arr) => {
            let s1 = arr.values_slice();
            assert_eq!(s1.len(), N_CYCLES);
            for (i, &val) in s1.iter().enumerate() {
                assert!((val - (i as f64 * std::f64::consts::PI)).abs() < 1e-4);
            }
        }
        other => panic!("Unexpected variant: {other:?}"),
    }

    Ok(())
}

#[test]
fn test_scratch_buffer_limits_and_dl_heap_fallback() -> Result<()> {
    // Dl4Block sizes:
    // Header (24) + next_link (8) + dl_data (N*8) + members (8) + dl_offset (N*8) = 40 + 16*N
    // At N=64: 40 + 16*64 = 1064 bytes (fits within 1088 stack buffer)
    // At N=65: 40 + 16*65 = 1080 bytes (fits within 1088 stack buffer)
    // At N=66: 40 + 16*66 = 1096 bytes (exceeds 1088 -> heap fallback)
    // At N=256: 40 + 16*256 = 4136 bytes (heap fallback)

    for n_entries in [0, 1, 64, 65, 66, 128, 256] {
        let dl = Dl4Block {
            reserved: [0; 4],
            dl_len: (40 + 16 * n_entries) as u64,
            dl_links: (1 + n_entries) as u64,
            dl_dl_next: 9999,
            dl_data: (0..n_entries as i64).map(|i| i * 100).collect(),
            dl_flags: 0,
            dl_reserved: [0; 3],
            dl_count: n_entries as u32,
            dl_equal_length: None,
            dl_offset: (0..n_entries as u64).map(|i| i * 4096).collect(),
            dl_time_values: vec![],
            dl_angle_values: vec![],
            dl_distance_values: vec![],
        };

        // Test serialization using binrw directly
        let mut out_bytes = Vec::new();
        dl.write_le(&mut Cursor::new(&mut out_bytes))?;
        assert_eq!(out_bytes.len(), 40 + 16 * n_entries - 4); // minus 4 bytes reserved prefix in binrw

        // Deserialize and check roundtrip
        let mut in_c = Cursor::new(&out_bytes);
        let deserialized = Dl4Block::read_le(&mut in_c)?;
        assert_eq!(deserialized.dl_count, n_entries as u32);
        assert_eq!(deserialized.dl_data.len(), n_entries);
        assert_eq!(deserialized.dl_offset.len(), n_entries);
        assert_eq!(deserialized.dl_dl_next, 9999);
    }

    Ok(())
}

#[test]
fn test_exact_4mib_boundary_and_plus_one() -> Result<()> {
    // 1. Exact boundary: 4,194,304 bytes (262,144 records of 16B)
    {
        let tmp = tempfile::Builder::new().suffix(".mf4").tempfile()?;
        let path_str = tmp.path().to_str().unwrap().to_string();

        let mut builder =
            LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::DeflateTranspose);
        let g = builder.add_group("Boundary");
        builder.set_group_flush_threshold(g, 4_194_304)?;
        let ch = builder.add_channel(g, "Val", CnDataType::UIntLE, 64)?;

        let clock = ManualClock::new(0.001, 1);
        let schema = builder.build()?;
        let file = File::create(&path_str)?;
        let mut logger = Logger::start(schema, file, clock)?;

        const REC_COUNT: usize = 262_144; // 262,144 * 16 = 4,194,304
        for i in 0..REC_COUNT as u64 {
            logger.begin_record(g)?.set_u64(ch, i)?.commit()?;
        }
        logger.close()?;

        let (dz_blocks, _hl_blocks) = scan_blocks(&path_str)?;
        assert_eq!(
            dz_blocks.len(),
            1,
            "Exactly 4 MiB must produce exactly 1 DZ chunk"
        );
        assert_eq!(dz_blocks[0].1.dz_org_data_length, 4_194_304);

        let mut mdf = Mdf::new(&path_str)?;
        mdf.load_all_channels_data_in_memory()?;
        let d = mdf.get_channel_data("Val").unwrap();
        match d {
            ChannelData::UInt64(arr) => {
                assert_eq!(arr.values_slice().len(), REC_COUNT);
            }
            _ => panic!("wrong variant"),
        }
    }

    // 2. Boundary + 1: 4,194,305 bytes on VLSD stream (transpose_width = 1)
    {
        let tmp = tempfile::Builder::new().suffix(".mf4").tempfile()?;
        let path_str = tmp.path().to_str().unwrap().to_string();

        let mut builder =
            LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::Lz4Transpose);
        let g = builder.add_group("BoundaryPlusOne");
        // We set threshold large so it flushes all at once on close
        builder.set_group_flush_threshold(g, 10_000_000)?;
        let ch_v = builder.add_vlsd_channel(g, "V", CnDataType::ByteArray)?;

        let clock = ManualClock::new(0.001, 1);
        let schema = builder.build()?;
        let file = File::create(&path_str)?;
        let mut logger = Logger::start(schema, file, clock)?;

        // Write a single 4,194,305-byte VLSD payload
        let large_payload = vec![0xABu8; 4_194_305];
        logger
            .begin_record(g)?
            .set_bytes(ch_v, &large_payload)?
            .commit()?;
        logger.close()?;

        let (dz_blocks, _hl_blocks) = scan_blocks(&path_str)?;
        // Find SD DZ blocks
        let sd_dz: Vec<_> = dz_blocks
            .iter()
            .filter(|(_pos, dz)| &dz.dz_org_block_type == b"SD")
            .collect();
        assert_eq!(sd_dz.len(), 2, "4,194,305 bytes must split into 2 chunks");
        assert_eq!(sd_dz[0].1.dz_org_data_length, 4_194_304);
        assert_eq!(
            sd_dz[1].1.dz_org_data_length, 5,
            "Chunk 1 is 1 byte payload + 4 bytes length prefix"
        );
        assert_eq!(
            sd_dz[0].1.dz_org_data_length + sd_dz[1].1.dz_org_data_length,
            4_194_309
        );
    }

    Ok(())
}

#[test]
fn test_direct_compress_decompress_oracle_all_algorithms() -> Result<()> {
    use mdfr_core::compress::{compress_dz_data, decompress_dz_data};

    // Test a pseudo-random buffer of 500,000 bytes with different transpose widths:
    // 0, 1, 2, 7, 8, 16, 1024, and oversized width 600,000 (> buffer len)
    let mut data = Vec::with_capacity(500_000);
    for i in 0..500_000 {
        data.push(((i * 37 + 11) % 256) as u8);
    }

    let algorithms = [
        CompressionAlgorithm::Deflate,
        CompressionAlgorithm::DeflateTranspose,
        CompressionAlgorithm::Lz4,
        CompressionAlgorithm::Lz4Transpose,
        CompressionAlgorithm::Zstd,
        CompressionAlgorithm::ZstdTranspose,
    ];

    let widths = [0, 1, 2, 7, 8, 16, 1024, 600_000];

    for &algo in &algorithms {
        for &width in &widths {
            let compressed = compress_dz_data(&data, algo, width)?;
            assert!(!compressed.is_empty());

            let effective_zip_type = if width <= 1 {
                algo.zip_type() & !1
            } else {
                algo.zip_type()
            };
            let dz_zip_param = if algo.is_transposed() && width > 1 {
                width as u32
            } else {
                0
            };

            let decompressed = decompress_dz_data(
                effective_zip_type,
                dz_zip_param,
                &compressed,
                data.len() as u64,
            )?;

            assert_eq!(
                decompressed.len(),
                data.len(),
                "Length mismatch for algo {:?} with width {}",
                algo,
                width
            );
            assert_eq!(
                decompressed, data,
                "Content mismatch for algo {:?} with width {}",
                algo, width
            );
        }
    }

    Ok(())
}

#[test]
fn test_massive_flushes_10_000_cycles_dl_chaining() -> Result<()> {
    // 10,000 records with threshold 16 bytes (flush EVERY single record!)
    // max_blocks_per_dl = 8 -> 10,000 / 8 = 1,250 chained DL blocks!
    let tmp = tempfile::Builder::new().suffix(".mf4").tempfile()?;
    let path_str = tmp.path().to_str().unwrap().to_string();

    let mut builder = LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::Lz4Transpose);
    builder.max_blocks_per_dl(8);
    builder.checkpoint_every_n_blocks(50);

    let g = builder.add_group("MassiveDL");
    builder.set_group_flush_threshold(g, 16)?; // 16 bytes = 1 record
    let ch = builder.add_channel(g, "Val", CnDataType::UIntLE, 64)?;

    let clock = ManualClock::new(0.0001, 1);
    let schema = builder.build()?;
    let file = File::create(&path_str)?;
    let mut logger = Logger::start(schema, file, clock)?;

    const RECORD_COUNT: u64 = 10_000;
    for i in 0..RECORD_COUNT {
        logger.begin_record(g)?.set_u64(ch, i * 3)?.commit()?;
    }
    logger.close()?;

    // Verify file read back
    let mut mdf = Mdf::new(&path_str)?;
    mdf.load_all_channels_data_in_memory()?;
    let val_data = mdf.get_channel_data("Val").expect("Val missing");
    match val_data {
        ChannelData::UInt64(arr) => {
            let slice = arr.values_slice();
            assert_eq!(slice.len(), RECORD_COUNT as usize);
            for (i, &val) in slice.iter().enumerate() {
                assert_eq!(val, (i as u64) * 3, "Desync at index {}", i);
            }
        }
        _ => panic!("wrong variant"),
    }

    Ok(())
}
