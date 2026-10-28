//! Compression and transposition helpers for MDF 4.3 DZ blocks (`no_std + alloc`).
//!
//! Supports:
//! - Transposition (`transpose` crate, `no_std`)
//! - Deflate / Zlib (`miniz_oxide`, behind `deflate` feature)
//! - LZ4 Frame format (`lz4_flex` block encoder + 7-byte LZ4 frame wrapper, behind `lz4` feature)
//! - Zstd (`zstd` crate, behind `zstd` feature which implies `std`)
use alloc::vec;
use alloc::vec::Vec;

use crate::error::{Error, Result};

/// Compression algorithm used in DZ blocks (MDF 4.3 Table 57).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum CompressionAlgorithm {
    /// No compression (use plain DT/DV/SD blocks).
    #[default]
    NoCompression = 255,
    /// Zlib/Deflate without transposition (`dz_zip_type = 0`).
    Deflate = 0,
    /// Transposition + Zlib/Deflate (`dz_zip_type = 1`).
    DeflateTranspose = 1,
    /// Zstandard without transposition (`dz_zip_type = 2`, MDF 4.3).
    Zstd = 2,
    /// Transposition + Zstandard (`dz_zip_type = 3`, MDF 4.3).
    ZstdTranspose = 3,
    /// LZ4 frame without transposition (`dz_zip_type = 4`, MDF 4.3).
    Lz4 = 4,
    /// Transposition + LZ4 frame (`dz_zip_type = 5`, MDF 4.3).
    Lz4Transpose = 5,
}

impl CompressionAlgorithm {
    /// Returns `true` if this variant applies byte transposition before compression.
    #[inline]
    pub const fn is_transposed(self) -> bool {
        matches!(
            self,
            Self::DeflateTranspose | Self::ZstdTranspose | Self::Lz4Transpose
        )
    }

    /// Returns the `dz_zip_type` byte stored in a DZBLOCK/HLBLOCK.
    #[inline]
    pub const fn zip_type(self) -> u8 {
        self as u8
    }
}

/// Transposes `rows × cols` bytes (`rows = bytes.len() / cols`) out-of-place.
/// Any trailing remainder (`bytes.len() % cols`) is appended untouched, per MDF4 spec §5.26.2.
pub fn transpose_for_compression(bytes: &[u8], cols: usize) -> Vec<u8> {
    if cols <= 1 || bytes.len() < cols {
        return bytes.to_vec();
    }
    let rows = bytes.len() / cols;
    let body_len = rows * cols;
    let mut out = vec![0u8; bytes.len()];
    transpose::transpose(&bytes[..body_len], &mut out[..body_len], cols, rows);
    if body_len < bytes.len() {
        out[body_len..].copy_from_slice(&bytes[body_len..]);
    }
    out
}

/// Inverse transposition (`cols × rows` back to `rows × cols`), preserving any trailing remainder.
pub fn untranspose_after_decompression(bytes: &[u8], cols: usize) -> Vec<u8> {
    if cols <= 1 || bytes.len() < cols {
        return bytes.to_vec();
    }
    let rows = bytes.len() / cols;
    let body_len = rows * cols;
    let mut out = vec![0u8; bytes.len()];
    transpose::transpose(&bytes[..body_len], &mut out[..body_len], rows, cols);
    if body_len < bytes.len() {
        out[body_len..].copy_from_slice(&bytes[body_len..]);
    }
    out
}

/// Compresses `bytes` according to `algo` and `record_width` (used when `algo.is_transposed()`).
pub fn compress_dz_data(
    bytes: &[u8],
    algo: CompressionAlgorithm,
    record_width: usize,
) -> Result<Vec<u8>> {
    if algo == CompressionAlgorithm::NoCompression {
        return Ok(bytes.to_vec());
    }
    let transposed;
    let input: &[u8] = if algo.is_transposed() && record_width > 1 {
        transposed = transpose_for_compression(bytes, record_width);
        &transposed
    } else {
        bytes
    };

    match algo {
        CompressionAlgorithm::NoCompression => Ok(bytes.to_vec()),
        CompressionAlgorithm::Deflate | CompressionAlgorithm::DeflateTranspose => {
            compress_deflate(input)
        }
        CompressionAlgorithm::Lz4 | CompressionAlgorithm::Lz4Transpose => compress_lz4_frame(input),
        CompressionAlgorithm::Zstd | CompressionAlgorithm::ZstdTranspose => compress_zstd(input),
    }
}

#[cfg(feature = "deflate")]
fn compress_deflate(input: &[u8]) -> Result<Vec<u8>> {
    // Level 6 (default) zlib stream as required by MDF4 DZBLOCK zip_type 0/1
    Ok(miniz_oxide::deflate::compress_to_vec_zlib(input, 6))
}

#[cfg(not(feature = "deflate"))]
fn compress_deflate(_input: &[u8]) -> Result<Vec<u8>> {
    Err(Error::Unsupported("deflate feature not enabled"))
}

/// Minimal `no_std` LZ4 Frame encoder (independent blocks, max 4 MB block size, no block/content checksums).
/// Produces a valid LZ4 Frame stream readable by the standard `lz4` / `lz4_flex::frame` decoders.
#[cfg(feature = "lz4")]
fn compress_lz4_frame(input: &[u8]) -> Result<Vec<u8>> {
    const LZ4F_MAGIC: [u8; 4] = 0x184D_2204u32.to_le_bytes();
    // FLG = 0x60 (version 01, independent blocks = 1, no checksums/content size)
    // BD  = 0x70 (max block size = 7 -> 4 MB)
    // HC  = (xxh32([0x60, 0x70], 0) >> 8) & 0xFF = 0x73
    const HEADER: [u8; 7] = [
        LZ4F_MAGIC[0],
        LZ4F_MAGIC[1],
        LZ4F_MAGIC[2],
        LZ4F_MAGIC[3],
        0x60,
        0x70,
        0x73,
    ];
    const MAX_BLOCK: usize = 4 * 1024 * 1024;
    const UNCOMPRESSED_BIT: u32 = 0x8000_0000;

    let mut out = Vec::with_capacity(HEADER.len() + input.len() / 2 + 16);
    out.extend_from_slice(&HEADER);

    for chunk in input.chunks(MAX_BLOCK) {
        let compressed = lz4_flex::block::compress(chunk);
        if compressed.len() < chunk.len() {
            let len = compressed.len() as u32;
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(&compressed);
        } else {
            let len = (chunk.len() as u32) | UNCOMPRESSED_BIT;
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(chunk);
        }
    }
    // EndMark (4 zero bytes)
    out.extend_from_slice(&0u32.to_le_bytes());
    Ok(out)
}

#[cfg(not(feature = "lz4"))]
fn compress_lz4_frame(_input: &[u8]) -> Result<Vec<u8>> {
    Err(Error::Unsupported("lz4 feature not enabled"))
}

#[cfg(feature = "zstd")]
fn compress_zstd(input: &[u8]) -> Result<Vec<u8>> {
    zstd::encode_all(input, 0).map_err(|_| Error::Io)
}

#[cfg(not(feature = "zstd"))]
fn compress_zstd(_input: &[u8]) -> Result<Vec<u8>> {
    Err(Error::Unsupported("zstd feature not enabled"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_roundtrip_with_tail() {
        let orig: Vec<u8> = (0..23u8).collect();
        let t = transpose_for_compression(&orig, 5);
        assert_ne!(t, orig);
        let back = untranspose_after_decompression(&t, 5);
        assert_eq!(back, orig);
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn deflate_transpose_roundtrip() {
        let orig: Vec<u8> = (0..1000u32).flat_map(|v| v.to_le_bytes()).collect();
        let comp = compress_dz_data(&orig, CompressionAlgorithm::DeflateTranspose, 4).unwrap();
        assert!(comp.len() < orig.len());
        let decomp = miniz_oxide::inflate::decompress_to_vec_zlib(&comp).unwrap();
        let back = untranspose_after_decompression(&decomp, 4);
        assert_eq!(back, orig);
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn lz4_frame_header_and_compress() {
        let orig: Vec<u8> = (0..1000u32).flat_map(|v| v.to_le_bytes()).collect();
        let comp = compress_dz_data(&orig, CompressionAlgorithm::Lz4Transpose, 4).unwrap();
        assert_eq!(&comp[0..4], &0x184D_2204u32.to_le_bytes());
        assert!(comp.len() < orig.len());
    }
}
