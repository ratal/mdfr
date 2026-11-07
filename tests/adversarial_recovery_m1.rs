//! Adversarial Stress Testing Suite for MDF 4.3 Crash Recovery (Milestone 1)
//!
//! Empirically challenges and stress-tests:
//! 1. Truncated `Hl4Block` and `Dz4Block` headers
//! 2. Unfinalized DL chain with mixed chunk lengths (DT, DZ, VD) and zeroed dl_offset
//! 3. DL slot containing 0 entries with chunks written beyond `dl_pos + 1064`
//! 4. `##VD` blocks interleaved with `##DT` and `##DZ` blocks in sorted and unsorted layout
//! 5. Forward-scan boundary conditions (up to and beyond 64 chunks)
//! 6. Fuzzing and corrupted byte streams (truncated files, loop guards, invalid offsets)
#![allow(clippy::field_reassign_with_default, clippy::useless_vec)]

use anyhow::Result;
use binrw::{BinReaderExt, BinWrite, io::Cursor};
use mdfr_core::blocks::cg_block::{CG_F_VLSC, CG_F_VLSD, Cg4Block};
use mdfr_core::blocks::cn_block::CnDataType;
use mdfr_core::blocks::data_block::{Dl4Block, Dz4Block};
use mdfr_core::blocks::dg_block::Dg4Block;
use mdfr_core::blocks::hd_block::Hd4;
use mdfr_core::blocks::header::Blockheader4;
use mdfr_core::blocks::id_block::{
    FINALIZED_ID, IdBlock, UNFIN_LAST_DL_BLOCK, UNFIN_VLSD_OFFSET, UNFINALIZED_ID,
};
use mdfr_core::compress::{CompressionAlgorithm, compress_dz_data};
use mdfr_core::io::{SeekFrom, Sink, VecSink};
use mdfr_core::logger::recover::ReadWriteSeek;
use mdfr_core::logger::{DataLayout, Logger, LoggerBuilder, recover_in_place};
use mdfr_core::time::ManualClock;

/// Helper to write a Blockheader4
fn write_header(sink: &mut VecSink, id: &[u8; 4], len: u64, links: u64) -> Result<()> {
    let hdr = Blockheader4 {
        hdr_id: *id,
        hdr_gap: [0; 4],
        hdr_len: len,
        hdr_links: links,
    };
    let mut buf = Vec::new();
    hdr.write_le(&mut Cursor::new(&mut buf))?;
    sink.write_all(&buf)?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 1: Truncated Hl4Block and Dz4Block Headers
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_adversarial_truncated_hl4_block() -> Result<()> {
    // Construct unfinalized file where DG points to a truncated HL block
    let mut b = LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::NoCompression);
    let g = b.add_group("G");
    let _ch = b.add_channel(g, "V", CnDataType::UIntLE, 32).unwrap();
    let schema = b.build().unwrap();

    let sink = VecSink::default();
    let mut logger = Logger::start(schema, sink, ManualClock::new(0.01, 1)).unwrap();
    logger.checkpoint().unwrap();
    let mut sink = logger.into_sink_unfinalized();

    // Append an HL block at EOF, and point DG to it
    let hl_pos = sink.as_slice().len() as u64;
    // Update DG.dg_data (offset 168 + 24 is dg_data in DG block)
    sink.seek(SeekFrom::Start(168 + 24))?;
    sink.write_all(&(hl_pos as i64).to_le_bytes())?;

    // Subcase 1a: File ends right after b"##HL" (only 4 bytes)
    sink.seek(SeekFrom::Start(hl_pos))?;
    sink.write_all(b"##HL")?;
    // Truncate to hl_pos + 4
    sink.data.truncate((hl_pos + 4) as usize);

    // recover_in_place must not panic and return clean Result
    let res = recover_in_place(&mut sink);
    assert!(
        res.is_ok(),
        "Truncated ##HL (4 bytes) must not panic: {:?}",
        res
    );

    // Subcase 1b: File ends at hl_pos + 20 (partial header < 40 bytes)
    sink.data.truncate(hl_pos as usize);
    sink.seek(SeekFrom::Start(hl_pos))?;
    sink.write_all(b"##HL")?;
    sink.write_all(&[0u8; 16])?; // 20 bytes total

    // Reset ID to unfinalized for re-test
    sink.seek(SeekFrom::Start(0))?;
    sink.write_all(&UNFINALIZED_ID)?;

    let res = recover_in_place(&mut sink);
    assert!(
        res.is_ok(),
        "Truncated ##HL (20 bytes) must not panic: {:?}",
        res
    );

    // Subcase 1c: File ends at hl_pos + 39 (1 byte short of full 40-byte Hl4Block)
    sink.data.truncate(hl_pos as usize);
    sink.seek(SeekFrom::Start(hl_pos))?;
    sink.write_all(b"##HL")?;
    sink.write_all(&[0u8; 35])?; // 39 bytes total

    sink.seek(SeekFrom::Start(0))?;
    sink.write_all(&UNFINALIZED_ID)?;

    let res = recover_in_place(&mut sink);
    assert!(
        res.is_ok(),
        "Truncated ##HL (39 bytes) must not panic: {:?}",
        res
    );

    Ok(())
}

#[test]
fn test_adversarial_truncated_dz4_block() -> Result<()> {
    // Construct unfinalized file where DG points to a truncated DZ block
    let mut b = LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::NoCompression);
    let g = b.add_group("G");
    let _ch = b.add_channel(g, "V", CnDataType::UIntLE, 32).unwrap();
    let schema = b.build().unwrap();

    let sink = VecSink::default();
    let mut logger = Logger::start(schema, sink, ManualClock::new(0.01, 1)).unwrap();
    logger.checkpoint().unwrap();
    let mut sink = logger.into_sink_unfinalized();

    let dz_pos = sink.as_slice().len() as u64;
    // Update DG.dg_data (offset 168 + 24)
    sink.seek(SeekFrom::Start(168 + 24))?;
    sink.write_all(&(dz_pos as i64).to_le_bytes())?;

    // Subcase 2a: Only b"##DZ" (4 bytes)
    sink.seek(SeekFrom::Start(dz_pos))?;
    sink.write_all(b"##DZ")?;
    sink.data.truncate((dz_pos + 4) as usize);

    let res = recover_in_place(&mut sink);
    assert!(
        res.is_ok(),
        "Truncated ##DZ (4 bytes) must not panic: {:?}",
        res
    );

    // Subcase 2b: Truncated header (28 bytes total, < 48 bytes)
    sink.data.truncate(dz_pos as usize);
    sink.seek(SeekFrom::Start(dz_pos))?;
    sink.write_all(b"##DZ")?;
    sink.write_all(&[0u8; 24])?; // 28 bytes total

    sink.seek(SeekFrom::Start(0))?;
    sink.write_all(&UNFINALIZED_ID)?;

    let res = recover_in_place(&mut sink);
    assert!(
        res.is_ok(),
        "Truncated ##DZ (28 bytes) must not panic: {:?}",
        res
    );

    // Subcase 2c: Truncated header (47 bytes total, 1 byte short of 48)
    sink.data.truncate(dz_pos as usize);
    sink.seek(SeekFrom::Start(dz_pos))?;
    sink.write_all(b"##DZ")?;
    sink.write_all(&[0u8; 43])?;

    sink.seek(SeekFrom::Start(0))?;
    sink.write_all(&UNFINALIZED_ID)?;

    let res = recover_in_place(&mut sink);
    assert!(
        res.is_ok(),
        "Truncated ##DZ (47 bytes) must not panic: {:?}",
        res
    );

    // Subcase 2d: Full 48-byte DZ header claiming 10,000 bytes compressed data, but file has only 10 bytes
    sink.data.truncate(dz_pos as usize);
    sink.seek(SeekFrom::Start(dz_pos))?;
    sink.write_all(b"##DZ")?;
    let dz = Dz4Block {
        reserved: [0; 4],
        len: 48 + 10_000,
        dz_links: 0,
        dz_org_block_type: *b"DT",
        dz_zip_type: 4, // LZ4
        dz_reserved: 0,
        dz_zip_parameter: 0,
        dz_org_data_length: 50_000,
        dz_data_length: 10_000,
    };
    let mut dz_bytes = Vec::new();
    dz.write_le(&mut Cursor::new(&mut dz_bytes))?;
    sink.write_all(&dz_bytes)?;
    sink.write_all(&[0xAA; 10])?; // only 10 bytes of payload instead of 10,000

    sink.seek(SeekFrom::Start(0))?;
    sink.write_all(&UNFINALIZED_ID)?;

    let res = recover_in_place(&mut sink);
    assert!(
        res.is_ok(),
        "Truncated DZ compressed payload must not panic: {:?}",
        res
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 2: Unfinalized DL Chain with Mixed Chunk Lengths (DT, DZ, VD)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_adversarial_unfinalized_dl_chain_mixed_chunk_lengths() -> Result<()> {
    let mut sink = VecSink::new();

    // ID Block with UNFIN_LAST_DL_BLOCK
    let mut id = IdBlock::default();
    id.id_file_id = UNFINALIZED_ID;
    id.id_unfin_flags = UNFIN_LAST_DL_BLOCK | UNFIN_VLSD_OFFSET;
    let mut id_bytes = Vec::new();
    id.write_le(&mut Cursor::new(&mut id_bytes))?;
    sink.write_all(&id_bytes)?;

    // HD Block at 64
    let mut hd = Hd4::default();
    hd.hd_dg_first = 168;
    let mut hd_bytes = Vec::new();
    hd.write_le(&mut Cursor::new(&mut hd_bytes))?;
    sink.write_all(&hd_bytes)?;

    // DG Block at 168 (Sorted layout, rec_id_size = 0)
    let dl_pos = 168 + 64 + 104; // after DG and CG
    let mut dg = Dg4Block::default();
    dg.dg_cg_first = 168 + 64;
    dg.dg_data = dl_pos as i64;
    let mut dg_bytes = Vec::new();
    dg.write_le(&mut Cursor::new(&mut dg_bytes))?;
    sink.write_all(&dg_bytes)?;

    // CG Block at 168 + 64 = 232 (rec_len = 8 bytes)
    sink.seek(SeekFrom::Start(232))?;
    sink.write_all(b"##CG")?;
    sink.write_all(&[0u8; 4])?; // gap
    sink.write_all(&104u64.to_le_bytes())?; // len
    let mut cg = Cg4Block::default();
    cg.cg_data_bytes = 8;
    let mut cg_bytes = Vec::new();
    cg.write_le(&mut Cursor::new(&mut cg_bytes))?;
    sink.write_all(&cg_bytes)?;

    // Now write DL Block at dl_pos
    // We will point dl_data to 4 mixed chunks:
    // Chunk 1: DT (80 payload bytes)
    // Chunk 2: DZ (240 uncompressed bytes, Deflate)
    // Chunk 3: VD (160 payload bytes)
    // Chunk 4: DT (320 payload bytes)
    // Total uncompressed payload = 80 + 240 + 160 + 320 = 800 bytes (100 cycles of 8-byte records)
    let c1_pos = dl_pos + 1064;
    let c1_len = 24 + 80;

    let c2_pos = c1_pos + c1_len;
    let c2_raw = vec![0x42u8; 240];
    let c2_comp = compress_dz_data(&c2_raw, CompressionAlgorithm::Deflate, 1)?;
    let c2_len = 48 + c2_comp.len() as u64;

    let c3_pos = c2_pos + c2_len;
    let c3_len = 24 + 160;

    let c4_pos = c3_pos + c3_len;
    let c4_len = 24 + 320;

    sink.seek(SeekFrom::Start(dl_pos))?;
    sink.write_all(b"##DL")?;
    let dl = Dl4Block {
        reserved: [0; 4],
        dl_len: 24 + 8 + 4 * 8 + 8 + 4 * 8, // 104 bytes
        dl_links: 5,
        dl_dl_next: 0,
        dl_data: vec![c1_pos as i64, c2_pos as i64, c3_pos as i64, c4_pos as i64],
        dl_flags: 0, // variable-length offsets
        dl_reserved: [0; 3],
        dl_count: 4,
        dl_equal_length: None,
        dl_offset: vec![0, 0, 0, 0], // UNFINALIZED ALL ZERO
        dl_time_values: vec![],
        dl_angle_values: vec![],
        dl_distance_values: vec![],
    };
    let mut dl_bytes = Vec::new();
    dl.write_le(&mut Cursor::new(&mut dl_bytes))?;
    sink.write_all(&dl_bytes)?;

    // Pad up to c1_pos
    let current = sink.as_slice().len() as u64;
    sink.write_all(&vec![0u8; (c1_pos - current) as usize])?;

    // Write Chunk 1: DT
    sink.seek(SeekFrom::Start(c1_pos))?;
    write_header(&mut sink, b"##DT", c1_len, 0)?;
    sink.write_all(&vec![1u8; 80])?;

    // Write Chunk 2: DZ
    sink.seek(SeekFrom::Start(c2_pos))?;
    sink.write_all(b"##DZ")?;
    let dz = Dz4Block {
        reserved: [0; 4],
        len: c2_len,
        dz_links: 0,
        dz_org_block_type: *b"DT",
        dz_zip_type: 0, // Deflate
        dz_reserved: 0,
        dz_zip_parameter: 0,
        dz_org_data_length: 240,
        dz_data_length: c2_comp.len() as u64,
    };
    let mut dz_bytes = Vec::new();
    dz.write_le(&mut Cursor::new(&mut dz_bytes))?;
    sink.write_all(&dz_bytes)?;
    sink.write_all(&c2_comp)?;

    // Write Chunk 3: VD
    sink.seek(SeekFrom::Start(c3_pos))?;
    write_header(&mut sink, b"##VD", c3_len, 0)?;
    sink.write_all(&vec![3u8; 160])?;

    // Write Chunk 4: DT
    sink.seek(SeekFrom::Start(c4_pos))?;
    write_header(&mut sink, b"##DT", c4_len, 0)?;
    sink.write_all(&vec![4u8; 320])?;

    // Run recover_in_place
    let patched = recover_in_place(&mut sink)?;
    assert!(patched > 0, "Must patch at least 1 CG");

    // Verify ID Block finalized
    assert_eq!(&sink.as_slice()[0..8], &FINALIZED_ID);

    // Verify DL block offsets reconstructed accurately:
    // Offset 0: 0
    // Offset 1: 80
    // Offset 2: 80 + 240 = 320
    // Offset 3: 320 + 160 = 480
    sink.seek(SeekFrom::Start(dl_pos + 4))?;
    let mut recovered_dl_buf = vec![0u8; (dl.dl_len - 4) as usize];
    sink.read_exact(&mut recovered_dl_buf)?;
    let recovered_dl: Dl4Block = Cursor::new(recovered_dl_buf).read_le()?;
    assert_eq!(recovered_dl.dl_offset, vec![0, 80, 320, 480]);

    // Verify CG cycle count: 800 bytes / 8 bytes per record = 100 cycles
    sink.seek(SeekFrom::Start(232 + 16 + 8 + 6 * 8 + 8))?;
    let mut cycles_buf = [0u8; 8];
    sink.read_exact(&mut cycles_buf)?;
    let cycles = u64::from_le_bytes(cycles_buf);
    assert_eq!(cycles, 100, "CG cycle count must be 100 cycles");

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 3: DL Slot Containing 0 Entries & Forward Scan Beyond dl_pos + 1064
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_adversarial_dl_zero_entries_forward_scan_with_interleaved_vd_dz_dt() -> Result<()> {
    let mut sink = VecSink::new();

    // ID Block
    let mut id = IdBlock::default();
    id.id_file_id = UNFINALIZED_ID;
    id.id_unfin_flags = UNFIN_LAST_DL_BLOCK;
    let mut id_bytes = Vec::new();
    id.write_le(&mut Cursor::new(&mut id_bytes))?;
    sink.write_all(&id_bytes)?;

    // HD Block at 64
    let mut hd = Hd4::default();
    hd.hd_dg_first = 168;
    let mut hd_bytes = Vec::new();
    hd.write_le(&mut Cursor::new(&mut hd_bytes))?;
    sink.write_all(&hd_bytes)?;

    // DG Block at 168
    let dl_pos = 168 + 64 + 104;
    let mut dg = Dg4Block::default();
    dg.dg_cg_first = 168 + 64;
    dg.dg_data = dl_pos as i64;
    let mut dg_bytes = Vec::new();
    dg.write_le(&mut Cursor::new(&mut dg_bytes))?;
    sink.write_all(&dg_bytes)?;

    // CG Block at 232 (rec_len = 4 bytes)
    sink.seek(SeekFrom::Start(232))?;
    sink.write_all(b"##CG")?;
    sink.write_all(&[0u8; 4])?;
    sink.write_all(&104u64.to_le_bytes())?;
    let mut cg = Cg4Block::default();
    cg.cg_data_bytes = 4;
    let mut cg_bytes = Vec::new();
    cg.write_le(&mut Cursor::new(&mut cg_bytes))?;
    sink.write_all(&cg_bytes)?;

    // Unpatched empty DL block at dl_pos: dl_count = 0, dl_len = 40, dl_links = 1
    sink.seek(SeekFrom::Start(dl_pos))?;
    sink.write_all(b"##DL")?;
    let dl = Dl4Block {
        reserved: [0; 4],
        dl_len: 40,
        dl_links: 1,
        dl_dl_next: 0,
        dl_data: vec![],
        dl_flags: 0,
        dl_reserved: [0; 3],
        dl_count: 0,
        dl_equal_length: None,
        dl_offset: vec![],
        dl_time_values: vec![],
        dl_angle_values: vec![],
        dl_distance_values: vec![],
    };
    let mut dl_bytes = Vec::new();
    dl.write_le(&mut Cursor::new(&mut dl_bytes))?;
    sink.write_all(&dl_bytes)?;

    // Pad reserved slot up to dl_pos + 1064
    let slot_end = dl_pos + 1064;
    let curr = sink.as_slice().len() as u64;
    sink.write_all(&vec![0u8; (slot_end - curr) as usize])?;

    // Now write 4 blocks beyond dl_pos + 1064:
    // 1. ##DT: 40 bytes (16 bytes payload)
    let c1_pos = slot_end;
    sink.seek(SeekFrom::Start(c1_pos))?;
    write_header(&mut sink, b"##DT", 40, 0)?;
    sink.write_all(&[0x11; 16])?;

    // 2. ##VD: 48 bytes (24 bytes payload)
    let c2_pos = c1_pos + 40;
    sink.seek(SeekFrom::Start(c2_pos))?;
    write_header(&mut sink, b"##VD", 48, 0)?;
    sink.write_all(&[0x22; 24])?;

    // 3. ##DZ: Deflate compressed 64 original bytes
    let c3_pos = c2_pos + 48;
    let raw3 = vec![0x33u8; 64];
    let comp3 = compress_dz_data(&raw3, CompressionAlgorithm::Deflate, 1)?;
    let dz3_len = 48 + comp3.len() as u64;
    sink.seek(SeekFrom::Start(c3_pos))?;
    sink.write_all(b"##DZ")?;
    let dz = Dz4Block {
        reserved: [0; 4],
        len: dz3_len,
        dz_links: 0,
        dz_org_block_type: *b"DT",
        dz_zip_type: 0,
        dz_reserved: 0,
        dz_zip_parameter: 0,
        dz_org_data_length: 64,
        dz_data_length: comp3.len() as u64,
    };
    let mut dz_bytes = Vec::new();
    dz.write_le(&mut Cursor::new(&mut dz_bytes))?;
    sink.write_all(&dz_bytes)?;
    sink.write_all(&comp3)?;

    // 4. ##DT: Partially written / truncated block! Header claims 100 bytes (76 bytes payload),
    // but file ends after only 20 bytes payload (total 44 bytes).
    // The recovery must clamp this block's header in place!
    let c4_pos = (c3_pos + dz3_len + 7) & !7;
    // Align file up to c4_pos
    let curr = sink.as_slice().len() as u64;
    if c4_pos > curr {
        sink.write_all(&vec![0u8; (c4_pos - curr) as usize])?;
    }
    sink.seek(SeekFrom::Start(c4_pos))?;
    write_header(&mut sink, b"##DT", 100, 0)?; // claims 100
    sink.write_all(&[0x44; 20])?; // only 20 bytes provided (file len = c4_pos + 44)

    // Run recover_in_place
    let patched = recover_in_place(&mut sink)?;
    assert!(patched > 0);

    // Verify forward-scan restored DL block
    sink.seek(SeekFrom::Start(dl_pos + 4))?;
    let mut rec_dl_buf = vec![0u8; 1060];
    sink.read_exact(&mut rec_dl_buf)?;
    let rec_dl: Dl4Block = Cursor::new(rec_dl_buf).read_le()?;

    assert_eq!(rec_dl.dl_count, 4);
    assert_eq!(rec_dl.dl_data.len(), 4);
    assert_eq!(rec_dl.dl_data[0], c1_pos as i64);
    assert_eq!(rec_dl.dl_data[1], c2_pos as i64);
    assert_eq!(rec_dl.dl_data[2], c3_pos as i64);
    assert_eq!(rec_dl.dl_data[3], c4_pos as i64);

    // Clamped payload of chunk 4: 20 bytes (aligned to 4-byte rec_len: 20 / 4 * 4 = 20)
    // Total payload: 16 + 24 + 64 + 20 = 124 bytes
    // Expected offsets: [0, 16, 40, 104]
    assert_eq!(rec_dl.dl_offset, vec![0, 16, 40, 104]);

    // Verify CG cycle count: 124 / 4 = 31 cycles
    sink.seek(SeekFrom::Start(232 + 16 + 8 + 6 * 8 + 8))?;
    let mut cycles_buf = [0u8; 8];
    sink.read_exact(&mut cycles_buf)?;
    let cycles = u64::from_le_bytes(cycles_buf);
    assert_eq!(cycles, 31, "CG cycle count must be 31");

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 4: Forward Scan Boundary Condition — 64 Chunks Cap
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_adversarial_forward_scan_64_chunk_limit() -> Result<()> {
    let mut sink = VecSink::new();

    let mut id = IdBlock::default();
    id.id_file_id = UNFINALIZED_ID;
    id.id_unfin_flags = UNFIN_LAST_DL_BLOCK;
    let mut id_bytes = Vec::new();
    id.write_le(&mut Cursor::new(&mut id_bytes))?;
    sink.write_all(&id_bytes)?;

    let mut hd = Hd4::default();
    hd.hd_dg_first = 168;
    let mut hd_bytes = Vec::new();
    hd.write_le(&mut Cursor::new(&mut hd_bytes))?;
    sink.write_all(&hd_bytes)?;

    let dl_pos = 168 + 64 + 104;
    let mut dg = Dg4Block::default();
    dg.dg_cg_first = 168 + 64;
    dg.dg_data = dl_pos as i64;
    let mut dg_bytes = Vec::new();
    dg.write_le(&mut Cursor::new(&mut dg_bytes))?;
    sink.write_all(&dg_bytes)?;

    sink.seek(SeekFrom::Start(232))?;
    sink.write_all(b"##CG")?;
    sink.write_all(&[0u8; 4])?;
    sink.write_all(&104u64.to_le_bytes())?;
    let mut cg = Cg4Block::default();
    cg.cg_data_bytes = 8;
    let mut cg_bytes = Vec::new();
    cg.write_le(&mut Cursor::new(&mut cg_bytes))?;
    sink.write_all(&cg_bytes)?;

    // Write empty DL block
    sink.seek(SeekFrom::Start(dl_pos))?;
    sink.write_all(b"##DL")?;
    let dl = Dl4Block {
        reserved: [0; 4],
        dl_len: 40,
        dl_links: 1,
        dl_dl_next: 0,
        dl_data: vec![],
        dl_flags: 0,
        dl_reserved: [0; 3],
        dl_count: 0,
        dl_equal_length: None,
        dl_offset: vec![],
        dl_time_values: vec![],
        dl_angle_values: vec![],
        dl_distance_values: vec![],
    };
    let mut dl_bytes = Vec::new();
    dl.write_le(&mut Cursor::new(&mut dl_bytes))?;
    sink.write_all(&dl_bytes)?;

    // Pad reserved slot up to dl_pos + 1064
    let slot_end = dl_pos + 1064;
    let curr = sink.as_slice().len() as u64;
    sink.write_all(&vec![0u8; (slot_end - curr) as usize])?;

    // Now write 70 chunks (more than the 64-chunk DL limit)
    let mut scan_start = slot_end;
    for _ in 0..70 {
        sink.seek(SeekFrom::Start(scan_start))?;
        write_header(&mut sink, b"##DT", 32, 0)?; // 8 bytes payload
        sink.write_all(&[0x55; 8])?;
        scan_start += 32;
    }

    // Recover
    let patched = recover_in_place(&mut sink)?;
    assert!(patched > 0);

    // Verify DL block was capped at 64 without panic
    sink.seek(SeekFrom::Start(dl_pos + 4))?;
    let mut rec_dl_buf = vec![0u8; 1060];
    sink.read_exact(&mut rec_dl_buf)?;
    let rec_dl: Dl4Block = Cursor::new(rec_dl_buf).read_le()?;
    assert_eq!(rec_dl.dl_count, 64, "Must cap at 64 entries in single DL");
    assert_eq!(rec_dl.dl_data.len(), 64);
    assert_eq!(rec_dl.dl_offset.len(), 64);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 5: Interleaved ##VD with ##DT and ##DZ in Unsorted Layout
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_adversarial_unsorted_interleaved_vd_dt_dz() -> Result<()> {
    let mut b = LoggerBuilder::new(DataLayout::Unsorted, CompressionAlgorithm::NoCompression);
    let g1 = b.add_group("FixedGroup");
    let ch1 = b.add_channel(g1, "F1", CnDataType::UIntLE, 16).unwrap();
    let g2 = b.add_group("VlsdGroup");
    let ch_vlsd = b.add_vlsd_channel(g2, "V", CnDataType::ByteArray).unwrap();
    let schema = b.build().unwrap();

    let sink = VecSink::default();
    let mut logger = Logger::start(schema, sink, ManualClock::new(0.01, 1)).unwrap();

    // Commit records in both groups
    for i in 0..10 {
        logger
            .begin_record(g1)
            .unwrap()
            .set_u64(ch1, i)
            .unwrap()
            .commit()
            .unwrap();
    }
    for i in 0..5 {
        let payload = format!("vlsd_string_payload_{i}");
        logger
            .begin_record(g2)
            .unwrap()
            .set_bytes(ch_vlsd, payload.as_bytes())
            .unwrap()
            .commit()
            .unwrap();
    }
    logger.checkpoint().unwrap();
    let mut sink = logger.into_sink_unfinalized();

    assert_eq!(&sink.as_slice()[0..8], &UNFINALIZED_ID);

    let patched = recover_in_place(&mut sink)?;
    assert!(patched >= 2, "Both fixed and VLSD CGs should be patched");
    assert_eq!(&sink.as_slice()[0..8], &FINALIZED_ID);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 6: Adversarial Stream Fuzzing & Malformed Invariants
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_adversarial_malformed_stream_invariants() {
    // 1. Zero bytes
    let mut sink = VecSink::new();
    assert!(recover_in_place(&mut sink).is_err());

    // 2. Exactly 100 bytes (< 168 bytes minimum)
    let mut sink = VecSink::new();
    sink.write_all(&[0u8; 100]).unwrap();
    assert!(recover_in_place(&mut sink).is_err());

    // 3. Exactly 168 bytes of random noise with unfinalized ID
    let mut sink = VecSink::new();
    sink.write_all(&UNFINALIZED_ID).unwrap();
    sink.write_all(&[0xFF; 160]).unwrap();
    // Must return an error or handle cleanly without panic!
    let _ = recover_in_place(&mut sink);

    // 4. File already finalized
    let mut sink = VecSink::new();
    sink.write_all(&FINALIZED_ID).unwrap();
    sink.write_all(&[0u8; 160]).unwrap();
    let res = recover_in_place(&mut sink).unwrap();
    assert_eq!(res, 0, "Already finalized stream returns 0 patched CGs");
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 7: VLSC Companion CG and ##VD Block Recovery in Unsorted Stream
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_adversarial_vlsc_and_vlsd_with_vd_block_in_unsorted_stream() -> Result<()> {
    let mut sink = VecSink::new();

    // 1. ID Block
    let mut id = IdBlock::default();
    id.id_file_id = UNFINALIZED_ID;
    id.id_unfin_flags = UNFIN_LAST_DL_BLOCK | UNFIN_VLSD_OFFSET;
    let mut id_bytes = Vec::new();
    id.write_le(&mut Cursor::new(&mut id_bytes))?;
    sink.write_all(&id_bytes)?;

    // 2. HD Block at 64
    let mut hd = Hd4::default();
    hd.hd_dg_first = 168;
    let mut hd_bytes = Vec::new();
    hd.write_le(&mut Cursor::new(&mut hd_bytes))?;
    sink.write_all(&hd_bytes)?;

    // 3. DG Block at 168 (Unsorted: dg_rec_id_size = 1)
    let cg1_pos = 168 + 64; // 232
    let cg2_pos = cg1_pos + 104; // 336 (VLSC companion)
    let vd_pos = cg2_pos + 104; // 440

    let mut dg = Dg4Block::default();
    dg.dg_cg_first = cg1_pos as i64;
    dg.dg_data = vd_pos as i64;
    dg.dg_rec_id_size = 1;
    let mut dg_bytes = Vec::new();
    dg.write_le(&mut Cursor::new(&mut dg_bytes))?;
    sink.write_all(&dg_bytes)?;

    // 4. CG 1 at cg1_pos: Fixed-length record (rec_id = 1, rec_len = 4 bytes)
    sink.seek(SeekFrom::Start(cg1_pos))?;
    sink.write_all(b"##CG")?;
    sink.write_all(&[0u8; 4])?;
    sink.write_all(&104u64.to_le_bytes())?;
    let mut cg1 = Cg4Block::default();
    cg1.cg_cg_next = cg2_pos as i64;
    cg1.cg_record_id = 1;
    cg1.cg_data_bytes = 4;
    let mut cg1_bytes = Vec::new();
    cg1.write_le(&mut Cursor::new(&mut cg1_bytes))?;
    sink.write_all(&cg1_bytes)?;

    // 5. CG 2 at cg2_pos: VLSC Companion CG (rec_id = 2, CG_F_VLSC | CG_F_VLSD)
    sink.seek(SeekFrom::Start(cg2_pos))?;
    sink.write_all(b"##CG")?;
    sink.write_all(&[0u8; 4])?;
    sink.write_all(&104u64.to_le_bytes())?;
    let mut cg2 = Cg4Block::default();
    cg2.cg_cg_next = 0;
    cg2.cg_record_id = 2;
    cg2.cg_flags = CG_F_VLSC | CG_F_VLSD;
    cg2.cg_data_bytes = 0;
    cg2.cg_inval_bytes = 0;
    let mut cg2_bytes = Vec::new();
    cg2.write_le(&mut Cursor::new(&mut cg2_bytes))?;
    sink.write_all(&cg2_bytes)?;

    // 6. Write ##VD block directly at vd_pos
    // In unsorted stream, records are multiplexed:
    // Record A (CG 1): rec_id(1 byte = 1) + 4 bytes data = 5 bytes
    // Record B (CG 2): rec_id(1 byte = 2) + len(4 bytes = 12) + 12 bytes data = 17 bytes
    // Record C (CG 1): rec_id(1 byte = 1) + 4 bytes data = 5 bytes
    // Record D (CG 2): rec_id(1 byte = 2) + len(4 bytes = 8) + 8 bytes data = 13 bytes
    // Total VD payload = 5 + 17 + 5 + 13 = 40 bytes
    // Total VD block = 24 + 40 = 64 bytes
    sink.seek(SeekFrom::Start(vd_pos))?;
    write_header(&mut sink, b"##VD", 64, 0)?;

    // Record A (CG 1)
    sink.write_all(&[1u8])?;
    sink.write_all(&[0xAA, 0xBB, 0xCC, 0xDD])?;

    // Record B (CG 2 - VLSC)
    sink.write_all(&[2u8])?;
    sink.write_all(&12u32.to_le_bytes())?;
    sink.write_all(b"VLSC_CHUNK_1")?;

    // Record C (CG 1)
    sink.write_all(&[1u8])?;
    sink.write_all(&[0x11, 0x22, 0x33, 0x44])?;

    // Record D (CG 2 - VLSC)
    sink.write_all(&[2u8])?;
    sink.write_all(&8u32.to_le_bytes())?;
    sink.write_all(b"VLSC_CH2")?;

    // Run recover_in_place
    let patched = recover_in_place(&mut sink)?;
    assert_eq!(patched, 2, "Both CG1 and CG2 should be patched");

    // Check CG1: 2 cycles
    sink.seek(SeekFrom::Start(cg1_pos + 16 + 8 + 6 * 8 + 8))?;
    let mut cycles_buf = [0u8; 8];
    sink.read_exact(&mut cycles_buf)?;
    assert_eq!(u64::from_le_bytes(cycles_buf), 2);

    // Check CG2 (VLSC): 2 cycles, total vlsd_bytes = 12 + 8 = 20
    sink.seek(SeekFrom::Start(cg2_pos + 16 + 8 + 6 * 8 + 8))?;
    sink.read_exact(&mut cycles_buf)?;
    assert_eq!(u64::from_le_bytes(cycles_buf), 2);

    // Verify lower 32 bits at cg2_pos + 96 (cg_data_bytes) and upper at cg2_pos + 100 (cg_inval_bytes)
    sink.seek(SeekFrom::Start(cg2_pos + 96))?;
    let mut low_buf = [0u8; 4];
    sink.read_exact(&mut low_buf)?;
    assert_eq!(u32::from_le_bytes(low_buf), 20);

    sink.seek(SeekFrom::Start(cg2_pos + 100))?;
    let mut high_buf = [0u8; 4];
    sink.read_exact(&mut high_buf)?;
    assert_eq!(u32::from_le_bytes(high_buf), 0);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 8: Loop Guards & Corrupted Pointer Resilience
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_adversarial_dl_chain_loop_guard_and_corrupt_pointers() -> Result<()> {
    let mut sink = VecSink::new();

    // 1. ID Block
    let mut id = IdBlock::default();
    id.id_file_id = UNFINALIZED_ID;
    id.id_unfin_flags = UNFIN_LAST_DL_BLOCK;
    let mut id_bytes = Vec::new();
    id.write_le(&mut Cursor::new(&mut id_bytes))?;
    sink.write_all(&id_bytes)?;

    // 2. HD Block at 64
    let mut hd = Hd4::default();
    hd.hd_dg_first = 168;
    let mut hd_bytes = Vec::new();
    hd.write_le(&mut Cursor::new(&mut hd_bytes))?;
    sink.write_all(&hd_bytes)?;

    // 3. DG Block at 168
    let dl_pos = 168 + 64 + 104;
    let mut dg = Dg4Block::default();
    dg.dg_cg_first = 168 + 64;
    dg.dg_data = dl_pos as i64;
    let mut dg_bytes = Vec::new();
    dg.write_le(&mut Cursor::new(&mut dg_bytes))?;
    sink.write_all(&dg_bytes)?;

    // 4. CG Block at 232
    sink.seek(SeekFrom::Start(232))?;
    sink.write_all(b"##CG")?;
    sink.write_all(&[0u8; 4])?;
    sink.write_all(&104u64.to_le_bytes())?;
    let mut cg = Cg4Block::default();
    cg.cg_data_bytes = 8;
    let mut cg_bytes = Vec::new();
    cg.write_le(&mut Cursor::new(&mut cg_bytes))?;
    sink.write_all(&cg_bytes)?;

    // 5. DL block pointing to itself (infinite loop: dl_dl_next == dl_pos)
    // AND containing corrupt/invalid child pointers: [-1, 0, 999999]
    sink.seek(SeekFrom::Start(dl_pos))?;
    sink.write_all(b"##DL")?;
    let dl = Dl4Block {
        reserved: [0; 4],
        dl_len: 24 + 8 + 3 * 8 + 8 + 3 * 8,
        dl_links: 4,
        dl_dl_next: dl_pos as i64,     // Self-referential loop!
        dl_data: vec![-1, 0, 999_999], // Invalid child pointers!
        dl_flags: 0,
        dl_reserved: [0; 3],
        dl_count: 3,
        dl_equal_length: None,
        dl_offset: vec![0, 0, 0],
        dl_time_values: vec![],
        dl_angle_values: vec![],
        dl_distance_values: vec![],
    };
    let mut dl_bytes = Vec::new();
    dl.write_le(&mut Cursor::new(&mut dl_bytes))?;
    sink.write_all(&dl_bytes)?;

    // recover_in_place must survive loop guard and corrupt pointers without hanging or panicking!
    let res = recover_in_place(&mut sink);
    assert!(
        res.is_ok(),
        "Must break on loop guard and corrupt pointers cleanly: {:?}",
        res
    );

    Ok(())
}

#[test]
fn test_adversarial_forward_scan_garbage_at_slot_boundary() -> Result<()> {
    let mut sink = VecSink::new();

    let mut id = IdBlock::default();
    id.id_file_id = UNFINALIZED_ID;
    id.id_unfin_flags = UNFIN_LAST_DL_BLOCK;
    let mut id_bytes = Vec::new();
    id.write_le(&mut Cursor::new(&mut id_bytes))?;
    sink.write_all(&id_bytes)?;

    let mut hd = Hd4::default();
    hd.hd_dg_first = 168;
    let mut hd_bytes = Vec::new();
    hd.write_le(&mut Cursor::new(&mut hd_bytes))?;
    sink.write_all(&hd_bytes)?;

    let dl_pos = 168 + 64 + 104;
    let mut dg = Dg4Block::default();
    dg.dg_cg_first = 168 + 64;
    dg.dg_data = dl_pos as i64;
    let mut dg_bytes = Vec::new();
    dg.write_le(&mut Cursor::new(&mut dg_bytes))?;
    sink.write_all(&dg_bytes)?;

    sink.seek(SeekFrom::Start(232))?;
    sink.write_all(b"##CG")?;
    sink.write_all(&[0u8; 4])?;
    sink.write_all(&104u64.to_le_bytes())?;
    let mut cg = Cg4Block::default();
    cg.cg_data_bytes = 4;
    let mut cg_bytes = Vec::new();
    cg.write_le(&mut Cursor::new(&mut cg_bytes))?;
    sink.write_all(&cg_bytes)?;

    sink.seek(SeekFrom::Start(dl_pos))?;
    sink.write_all(b"##DL")?;
    let dl = Dl4Block {
        reserved: [0; 4],
        dl_len: 40,
        dl_links: 1,
        dl_dl_next: 0,
        dl_data: vec![],
        dl_flags: 0,
        dl_reserved: [0; 3],
        dl_count: 0,
        dl_equal_length: None,
        dl_offset: vec![],
        dl_time_values: vec![],
        dl_angle_values: vec![],
        dl_distance_values: vec![],
    };
    let mut dl_bytes = Vec::new();
    dl.write_le(&mut Cursor::new(&mut dl_bytes))?;
    sink.write_all(&dl_bytes)?;

    // Pad to dl_pos + 1064
    let slot_end = dl_pos + 1064;
    let curr = sink.as_slice().len() as u64;
    sink.write_all(&vec![0u8; (slot_end - curr) as usize])?;

    // Write completely unrecognized block header: ##XX
    sink.seek(SeekFrom::Start(slot_end))?;
    write_header(&mut sink, b"##XX", 64, 0)?;
    sink.write_all(&[0xEE; 40])?;

    let res = recover_in_place(&mut sink);
    assert!(
        res.is_ok(),
        "Must break cleanly on unknown block ID: {:?}",
        res
    );

    // DL should remain dl_count = 0
    sink.seek(SeekFrom::Start(dl_pos + 4))?;
    let mut rec_dl_buf = vec![0u8; 36];
    sink.read_exact(&mut rec_dl_buf)?;
    let rec_dl: Dl4Block = Cursor::new(rec_dl_buf).read_le()?;
    assert_eq!(rec_dl.dl_count, 0);

    Ok(())
}
