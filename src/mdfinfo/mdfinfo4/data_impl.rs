//! Data blocks (DT, DL, DZ, LD, HL, GD) for MDF4 — spec sections 6.12-6.17, Tables 55-63
//!
//! These blocks store raw channel data. DTBLOCK holds plain data, DZBLOCK holds
//! compressed data (zlib/LZ4/zstd, transposition), DLBLOCK chains data blocks,
//! HLBLOCK is a header list for DZBLOCK chains, LDBLOCK is a list for sorted data,
//! and GDBLOCK guards MDF 4.3 features in unsorted data.
use anyhow::{Context, Result, bail};
use std::fs::File;
use std::io::{BufReader, Read, Seek};

#[cfg(feature = "numpy")]
use pyo3::prelude::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "numpy", pyclass(eq, eq_int, from_py_object))]
pub enum CompressionAlgorithm {
    #[default]
    NoCompression = 255,
    Deflate = 0,
    DeflateTranspose = 1,
    Zstd = 2,
    ZstdTranspose = 3,
    Lz4 = 4,
    Lz4Transpose = 5,
}

impl From<CompressionAlgorithm> for mdfr_core::compress::CompressionAlgorithm {
    fn from(c: CompressionAlgorithm) -> Self {
        match c {
            CompressionAlgorithm::NoCompression => Self::NoCompression,
            CompressionAlgorithm::Deflate => Self::Deflate,
            CompressionAlgorithm::DeflateTranspose => Self::DeflateTranspose,
            CompressionAlgorithm::Zstd => Self::Zstd,
            CompressionAlgorithm::ZstdTranspose => Self::ZstdTranspose,
            CompressionAlgorithm::Lz4 => Self::Lz4,
            CompressionAlgorithm::Lz4Transpose => Self::Lz4Transpose,
        }
    }
}

impl From<mdfr_core::compress::CompressionAlgorithm> for CompressionAlgorithm {
    fn from(c: mdfr_core::compress::CompressionAlgorithm) -> Self {
        match c {
            mdfr_core::compress::CompressionAlgorithm::NoCompression => Self::NoCompression,
            mdfr_core::compress::CompressionAlgorithm::Deflate => Self::Deflate,
            mdfr_core::compress::CompressionAlgorithm::DeflateTranspose => Self::DeflateTranspose,
            mdfr_core::compress::CompressionAlgorithm::Zstd => Self::Zstd,
            mdfr_core::compress::CompressionAlgorithm::ZstdTranspose => Self::ZstdTranspose,
            mdfr_core::compress::CompressionAlgorithm::Lz4 => Self::Lz4,
            mdfr_core::compress::CompressionAlgorithm::Lz4Transpose => Self::Lz4Transpose,
        }
    }
}

pub use mdfr_core::blocks::data::{Dl4Block, Dt4Block, Dz4Block, Gd4Block, Hl4Block, Ld4Block};

/// parses Data List block
/// pointing to DT, SD, RD or DZ blocks
pub fn parser_dl4_block(
    rdr: &mut BufReader<&File>,
    target: i64,
    mut position: i64,
) -> Result<(Dl4Block, i64)> {
    rdr.seek_relative(target - position)
        .context("Could not reach position to read Dl4Block")?;
    let block: Dl4Block =
        binrw::BinReaderExt::read_le(rdr).context("Could not read into Dl4Block struct")?;
    position = target + block.dl_len as i64;
    Ok((block, position))
}

/// Helper function to decompress data using various algorithms
pub fn decompress_data(
    zip_type: u8,
    zip_parameter: u32,
    buf: Vec<u8>,
    org_data_length: u64,
) -> Result<Vec<u8>> {
    mdfr_core::compress::decompress_dz_data(zip_type, zip_parameter, &buf, org_data_length)
        .map_err(|e| anyhow::anyhow!("DZ decompression failed: {e}"))
}

/// Number of bytes left between the current position and the end of the file.
fn bytes_left(rdr: &mut BufReader<&File>) -> Result<u64> {
    let file_size = rdr
        .get_ref()
        .metadata()
        .context("Could not read file metadata")?
        .len();
    let position = rdr
        .stream_position()
        .context("Could not get stream position")?;
    Ok(file_size.saturating_sub(position))
}

/// Checks a length field read from a DZ block against the bytes still in the file.
fn validate_dz_data_length(block: &Dz4Block, rdr: &mut BufReader<&File>) -> Result<()> {
    let available = bytes_left(rdr)?;
    if block.dz_data_length > available {
        bail!(
            "DZ block declares {} compressed bytes but only {available} are left in the file",
            block.dz_data_length
        );
    }
    Ok(())
}

/// parses DZBlock
pub fn parse_dz(rdr: &mut BufReader<&File>) -> Result<(Vec<u8>, Dz4Block)> {
    let mut block: Dz4Block =
        binrw::BinReaderExt::read_le(rdr).context("Could not read into Dz4Block struct")?;
    validate_dz_data_length(&block, rdr)?;
    let mut buf = vec![0u8; block.dz_data_length as usize];
    rdr.read_exact(&mut buf).context("Could not read Dz data")?;
    // decompress data
    let data = decompress_data(
        block.dz_zip_type,
        block.dz_zip_parameter,
        buf,
        block.dz_org_data_length,
    )?;
    block.dz_org_data_length = data.len() as u64;
    Ok((data, block))
}

/// Reads a DZBlock header and raw compressed bytes without decompressing.
/// Use `Dz4Block::decompress` in a parallel step to decompress independently.
pub fn read_dz_raw(rdr: &mut BufReader<&File>) -> Result<(Dz4Block, Vec<u8>)> {
    let block: Dz4Block =
        binrw::BinReaderExt::read_le(rdr).context("Could not read into Dz4Block struct")?;
    validate_dz_data_length(&block, rdr)?;
    let mut buf = vec![0u8; block.dz_data_length as usize];
    rdr.read_exact(&mut buf)
        .context("Could not read Dz raw data")?;
    Ok((block, buf))
}

/// Extension on [`Dz4Block`] for decompression.
pub trait Dz4BlockExt {
    /// Decompresses `buf` using this block's zip algorithm and parameters.
    fn decompress(self, buf: Vec<u8>) -> Result<Vec<u8>>;
}

impl Dz4BlockExt for Dz4Block {
    fn decompress(self, buf: Vec<u8>) -> Result<Vec<u8>> {
        decompress_data(
            self.dz_zip_type,
            self.dz_zip_parameter,
            buf,
            self.dz_org_data_length,
        )
    }
}

/// parse List Data block
/// equivalent ot DLBlock but unsorted data is not allowed
/// pointing to DV/DI and RV/RI blocks
pub fn parser_ld4_block(
    rdr: &mut BufReader<&File>,
    target: i64,
    mut position: i64,
) -> Result<(Ld4Block, i64)> {
    rdr.seek_relative(target - position)
        .context("Could not reach Ld4Block position")?;
    let block: Ld4Block =
        binrw::BinReaderExt::read_le(rdr).context("Could not read buffer into Ld4Block struct")?;
    position = target + block.ld_len as i64;
    Ok((block, position))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_compress_decompress_interop() {
        let orig: Vec<u8> = (0..1000u32).flat_map(|v| v.to_le_bytes()).collect();
        for (algo, zip_type) in [
            (mdfr_core::compress::CompressionAlgorithm::Deflate, 0u8),
            (
                mdfr_core::compress::CompressionAlgorithm::DeflateTranspose,
                1u8,
            ),
            (mdfr_core::compress::CompressionAlgorithm::Lz4, 4u8),
            (mdfr_core::compress::CompressionAlgorithm::Lz4Transpose, 5u8),
        ] {
            let comp = mdfr_core::compress::compress_dz_data(&orig, algo, 4).unwrap();
            let decomp = decompress_data(zip_type, 4, comp, orig.len() as u64).unwrap();
            assert_eq!(decomp, orig, "failed for {algo:?}");
        }
    }
}
