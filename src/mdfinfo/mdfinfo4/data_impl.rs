//! Data blocks (DT, DL, DZ, LD, HL, GD) for MDF4 — spec sections 6.12-6.17, Tables 55-63
//!
//! These blocks store raw channel data. DTBLOCK holds plain data, DZBLOCK holds
//! compressed data (zlib/LZ4/zstd, transposition), DLBLOCK chains data blocks,
//! HLBLOCK is a header list for DZBLOCK chains, LDBLOCK is a list for sorted data,
//! and GDBLOCK guards MDF 4.3 features in unsorted data.
use anyhow::{Context, Result, bail};
use std::io::Seek;

use crate::mdfinfo::sym_buf_reader::SliceCursor;

pub use mdfr_core::compress::CompressionAlgorithm;

pub use mdfr_core::blocks::data::{Dl4Block, Dt4Block, Dz4Block, Gd4Block, Hl4Block, Ld4Block};

/// parses Data List block
/// pointing to DT, SD, RD or DZ blocks
pub fn parser_dl4_block(
    rdr: &mut SliceCursor<'_>,
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
    buf: &[u8],
    org_data_length: u64,
) -> Result<Vec<u8>> {
    mdfr_core::compress::decompress_dz_data(zip_type, zip_parameter, buf, org_data_length)
        .map_err(|e| anyhow::anyhow!("DZ decompression failed: {e}"))
}

/// Number of bytes left between the current position and the end of the file.
fn bytes_left(rdr: &mut SliceCursor<'_>) -> Result<u64> {
    let file_size = rdr.file_size()?;
    let position = rdr
        .stream_position()
        .context("Could not get stream position")?;
    Ok(file_size.saturating_sub(position))
}

/// Checks a length field read from a DZ block against the bytes still in the file.
fn validate_dz_data_length(block: &Dz4Block, rdr: &mut SliceCursor<'_>) -> Result<()> {
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
pub fn parse_dz(rdr: &mut SliceCursor<'_>) -> Result<(Vec<u8>, Dz4Block)> {
    let mut block: Dz4Block =
        binrw::BinReaderExt::read_le(rdr).context("Could not read into Dz4Block struct")?;
    validate_dz_data_length(&block, rdr)?;
    let slice = rdr
        .read_slice(block.dz_data_length as usize)
        .context("Could not read Dz data")?;
    // decompress data directly from mapped slice
    let data = decompress_data(
        block.dz_zip_type,
        block.dz_zip_parameter,
        slice,
        block.dz_org_data_length,
    )?;
    block.dz_org_data_length = data.len() as u64;
    Ok((data, block))
}

/// Reads a DZBlock header and borrows raw compressed bytes without copying or decompressing.
/// Use `Dz4Block::decompress` in a parallel step to decompress independently.
pub fn read_dz_raw<'a>(rdr: &mut SliceCursor<'a>) -> Result<(Dz4Block, &'a [u8])> {
    let block: Dz4Block =
        binrw::BinReaderExt::read_le(rdr).context("Could not read into Dz4Block struct")?;
    validate_dz_data_length(&block, rdr)?;
    let slice = rdr
        .read_slice(block.dz_data_length as usize)
        .context("Could not read Dz raw data")?;
    Ok((block, slice))
}

/// Extension on [`Dz4Block`] for decompression.
pub trait Dz4BlockExt {
    /// Decompresses `buf` using this block's zip algorithm and parameters.
    fn decompress(self, buf: &[u8]) -> Result<Vec<u8>>;
}

impl Dz4BlockExt for Dz4Block {
    fn decompress(self, buf: &[u8]) -> Result<Vec<u8>> {
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
    rdr: &mut SliceCursor<'_>,
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
            let decomp = decompress_data(zip_type, 4, &comp, orig.len() as u64).unwrap();
            assert_eq!(decomp, orig, "failed for {algo:?}");
        }
    }
}
