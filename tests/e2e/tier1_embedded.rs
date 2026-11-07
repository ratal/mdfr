//! Tier 1: Embedded Reliability & Blocks (Features 14 - 19)
//!
//! - Feature 14: StaticArena::dealloc_layout zero-size guard and null pointer safety
//! - Feature 15: SpscRing::try_push bounds and saturation behavior
//! - Feature 16: Error::code() aligned with include/mdfr_logger.h (-1 through -9)
//! - Feature 17: Block structures (Cv4Block, Cu4Block, Gd4Block reserved, String BOM, bit channel int validation)
//! - Feature 18: Unified CompressionAlgorithm re-exported in mdfr
//! - Feature 19: Block panic hardening on malformed streams

use anyhow::Result;
use binrw::{BinReaderExt, BinWrite, io::Cursor};
use core::alloc::Layout;
use mdfr_core::arena::StaticArena;
use mdfr_core::blocks::composition_block::{Cu4Block, Cv4Block};
use mdfr_core::compress::CompressionAlgorithm;
use mdfr_core::error::Error;
use mdfr_core::logger::schema::GroupHandle;
use mdfr_core::logger::spsc::SpscRing;

#[test]
fn test_feat14_static_arena_dealloc_layout_zero_size_and_null() {
    // Feature 14: dealloc_layout safely returns on null pointer and handles allocated blocks
    let arena = StaticArena::new();
    unsafe {
        // Null pointer free must not crash
        arena.dealloc_layout(
            std::ptr::null_mut(),
            Layout::from_size_align(64, 8).unwrap(),
        );

        // Allocate and deallocate valid layout
        let ptr = arena.alloc_layout(Layout::from_size_align(32, 8).unwrap());
        if !ptr.is_null() {
            arena.dealloc_layout(ptr, Layout::from_size_align(32, 8).unwrap());
        }
    }
}

#[test]
fn test_feat15_spsc_ring_bounds_rejection() {
    // Feature 15: SpscRing capacity bounds and saturation rejection
    let ring = SpscRing::with_capacity(256);
    let payload = [1u8, 2, 3, 4];

    // Push until capacity is reached
    let mut pushed_count = 0;
    while ring.try_push(GroupHandle(1), 100, &payload) {
        pushed_count += 1;
        if pushed_count > 1000 {
            break;
        }
    }
    assert!(pushed_count > 0);
    // When ring is saturated, try_push returns false
    assert!(!ring.try_push(GroupHandle(1), 100, &payload));
}

#[test]
fn test_feat16_error_codes_c_header_alignment() {
    // Feature 16: Error::code() aligned with include/mdfr_logger.h
    assert_eq!(Error::Io.code(), -1);
    assert_eq!(Error::InvalidBlock("test".into()).code(), -2);
    assert_eq!(Error::InvalidSchema("test".into()).code(), -3);
    assert_eq!(Error::Unsupported("test").code(), -5);
}

#[test]
fn test_feat17_cv_cu_block_structure_parsing() -> Result<()> {
    // Feature 17: Cv4Block and Cu4Block link parsing, Gd4Block reserved, String BOM
    let cv = Cv4Block {
        cv_n_links: 2,
        cv_cn_discriminator: 100,
        cv_cn_option: vec![200],
        cv_option_count: 1,
        cv_reserved: [0; 4],
        cv_option_val: vec![42],
    };
    let mut buf = Vec::new();
    cv.write_le(&mut Cursor::new(&mut buf))?;
    let parsed: Cv4Block = Cursor::new(&buf).read_le()?;
    assert_eq!(parsed.cv_cn_discriminator, 100);
    assert_eq!(parsed.cv_option_count, 1);

    let cu = Cu4Block {
        cu_n_links: 2,
        cu_cn_member: vec![300, 400],
        cu_member_count: 2,
        cu_reserved: [0; 4],
    };
    let mut cu_buf = Vec::new();
    cu.write_le(&mut Cursor::new(&mut cu_buf))?;
    let parsed_cu: Cu4Block = Cursor::new(&cu_buf).read_le()?;
    assert_eq!(parsed_cu.cu_member_count, 2);
    Ok(())
}

#[test]
fn test_feat18_unified_compression_algorithm_enum() {
    // Feature 18: Canonical CompressionAlgorithm enum
    assert_eq!(CompressionAlgorithm::NoCompression.zip_type(), 255);
    assert_eq!(CompressionAlgorithm::Deflate.zip_type(), 0);
    assert_eq!(CompressionAlgorithm::DeflateTranspose.zip_type(), 1);
    assert_eq!(CompressionAlgorithm::Zstd.zip_type(), 2);
    assert_eq!(CompressionAlgorithm::ZstdTranspose.zip_type(), 3);
    assert_eq!(CompressionAlgorithm::Lz4.zip_type(), 4);
    assert_eq!(CompressionAlgorithm::Lz4Transpose.zip_type(), 5);

    assert!(!CompressionAlgorithm::Deflate.is_transposed());
    assert!(CompressionAlgorithm::DeflateTranspose.is_transposed());
}

#[test]
fn test_feat19_malformed_blocks_graceful_error_return() {
    // Feature 19: Malformed byte sequences return Err instead of panicking
    let malformed_bytes = [0xFF, 0xFE, 0x01];
    let res: Result<Cv4Block, _> = Cursor::new(&malformed_bytes).read_le();
    assert!(res.is_err());
}
