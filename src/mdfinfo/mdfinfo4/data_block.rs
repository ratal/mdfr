//! Data blocks (DT, DL, DZ, LD, HL, GD) for MDF4 — spec sections 6.12-6.17, Tables 55-63
//!
//! These blocks store raw channel data. DTBLOCK holds plain data, DZBLOCK holds
//! compressed data (zlib/LZ4/zstd, transposition), DLBLOCK chains data blocks,
//! HLBLOCK is a header list for DZBLOCK chains, LDBLOCK is a list for sorted data,
//! and GDBLOCK guards MDF 4.3 features in unsorted data.
use anyhow::{Context, Result, bail};
use flate2::read::{DeflateDecoder, ZlibDecoder};
use log::warn;
use lz4::Decoder as Lz4Decoder;
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek};
use zstd::Decoder as ZstdDecoder;

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
    let mut data = Vec::<u8>::with_capacity(org_data_length as usize);
    match zip_type {
        0 | 1 => {
            let reader = Cursor::new(&buf);
            let mut decoder = ZlibDecoder::new(reader);
            let zlib_ok = decoder.read_to_end(&mut data).is_ok();
            if !zlib_ok {
                warn!(
                    "zlib stream has invalid checksum or is truncated, falling back to raw deflate"
                );
                data.clear();
                let reader: Cursor<&[u8]> = if buf.len() >= 2
                    && ((buf[0] as u16) << 8 | buf[1] as u16).is_multiple_of(31)
                {
                    Cursor::new(&buf[2..])
                } else {
                    Cursor::new(&buf)
                };
                let mut decoder = DeflateDecoder::new(reader);
                decoder
                    .read_to_end(&mut data)
                    .context("Error decompressing Deflate data")?;
            }
        }
        2 | 3 => {
            // zstd algorithm
            let reader = Cursor::new(buf);
            let mut decoder =
                ZstdDecoder::new(reader).context("Error creating Zstd decoder from read vector")?;
            let _nbbytesread = decoder
                .read_to_end(&mut data)
                .context("error reading the compressed bytes")?;
        }
        4 | 5 => {
            // lz4 algorithm
            let reader = Cursor::new(buf);
            let mut decoder =
                Lz4Decoder::new(reader).context("Error creating Lz4 decoder from read vector")?;
            let _nbbytesread = decoder
                .read_to_end(&mut data)
                .context("error reading the compressed bytes")?;
        }
        254 => {
            // MDF 4.3 custom/vendor-specific compression
            warn!("Custom compression (zip_type=254) not supported - data will be empty");
            return Ok(data);
        }
        _ => {
            bail!("not implemented compression algorithm: {zip_type}")
        }
    };
    // transpose data
    if matches!(zip_type, 1 | 3 | 5) && zip_parameter > 0 {
        // org_data_length comes from the file: clamp the transposed area to what
        // was actually decompressed, otherwise split_off panics
        let m =
            (org_data_length / zip_parameter as u64).min(data.len() as u64 / zip_parameter as u64);
        let transposed_len = (m * zip_parameter as u64) as usize;
        let tail: Vec<u8> = data.split_off(transposed_len);
        let mut output = vec![0u8; transposed_len];
        transpose::transpose(&data, &mut output, m as usize, zip_parameter as usize);
        data = output;
        if !tail.is_empty() {
            data.extend(tail);
        }
    }
    Ok(data)
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

    // ── Dz4Block tests ──

    #[test]
    fn test_dz_default() {
        let dz = Dz4Block::default();
        assert_eq!(dz.dz_org_block_type, [68, 86]); // "DV"
        assert_eq!(dz.dz_zip_type, 0);
        assert_eq!(dz.dz_org_data_length, 0);
        assert_eq!(dz.dz_data_length, 0);
    }

    #[test]
    fn test_dz_get_compression_str() {
        let mut dz = Dz4Block::default();
        let expected = [
            (0, "Deflate"),
            (1, "Deflate+Transpose"),
            (2, "Zstd"),
            (3, "Zstd+Transpose"),
            (4, "LZ4"),
            (5, "LZ4+Transpose"),
            (254, "Custom"),
            (255, "Unknown"),
        ];
        for (val, name) in expected {
            dz.dz_zip_type = val;
            assert_eq!(dz.get_compression_str(), name, "dz_zip_type={val}");
        }
    }

    #[test]
    fn test_dz_display() {
        let dz = Dz4Block {
            dz_org_data_length: 1000,
            dz_data_length: 500,
            ..Default::default()
        };

        let display = format!("{dz}");
        assert!(display.contains("DZ:"));
        assert!(display.contains("Deflate"));
        assert!(display.contains("org_type=DV"));
        assert!(display.contains("compressed=500"));
        assert!(display.contains("original=1000"));
        assert!(display.contains("ratio=50.0%"));

        // Edge case: org_size=0
        let dz = Dz4Block::default();
        let display = format!("{dz}");
        assert!(display.contains("ratio=0.0%"));
    }

    // ── Ld4Block tests ──

    #[test]
    fn test_ld_default() {
        let ld = Ld4Block::default();
        assert_eq!(ld.ld_len, 56);
        assert_eq!(ld.ld_n_links, 2);
        assert_eq!(ld.ld_count, 1);
        assert_eq!(ld.ld_flags, 0);
        assert_eq!(ld.ld_flags_ext, 0);
        assert!(ld.ld_links.is_empty());
    }

    #[test]
    fn test_ld_data_simple() {
        // Non-interleaved: links returned as-is
        let ld = Ld4Block {
            ld_count: 3,
            ld_links: vec![100, 200, 300],
            ..Default::default()
        };
        let data = ld.ld_data();
        assert_eq!(data, vec![100, 200, 300]);
    }

    #[test]
    fn test_ld_data_interleaved_by_flag() {
        // Invalid data present via ld_flags_ext bit 7: first N links are data, next N are invalid
        let ld = Ld4Block {
            ld_count: 2,
            ld_flags_ext: 1u8 << 7,
            ld_links: vec![100, 200, 101, 201],
            ..Default::default()
        };
        let data = ld.ld_data();
        assert_eq!(data, vec![100, 200]);
    }

    #[test]
    fn test_ld_data_interleaved_by_count() {
        // Invalid data present detected by links.len() == 2 * ld_count
        let ld = Ld4Block {
            ld_count: 2,
            ld_links: vec![100, 200, 101, 201],
            ..Default::default()
        };
        let data = ld.ld_data();
        assert_eq!(data, vec![100, 200]);
    }

    #[test]
    fn test_ld_invalid_data() {
        // Non-invalid: empty result
        let ld = Ld4Block {
            ld_count: 2,
            ld_links: vec![100, 200],
            ..Default::default()
        };
        assert!(ld.ld_invalid_data().is_empty());

        // Invalid data present: second half (links[N..2*N]) are invalid blocks
        let ld = Ld4Block {
            ld_count: 2,
            ld_flags_ext: 1u8 << 7,
            ld_links: vec![100, 200, 101, 201],
            ..Default::default()
        };
        let invalid = ld.ld_invalid_data();
        assert_eq!(invalid, vec![101, 201]);
    }

    #[test]
    fn test_ld_display() {
        // With equal_sample_count
        let ld = Ld4Block {
            ld_equal_sample_count: Some(1000),
            ld_flags: 0b1,
            ..Default::default()
        };
        let display = format!("{ld}");
        assert!(display.contains("LD:"));
        assert!(display.contains("sample_count=1000"));

        // With offsets
        let ld2 = Ld4Block {
            ld_count: 3,
            ld_sample_offset: vec![0, 100, 200],
            ..Default::default()
        };
        let display2 = format!("{ld2}");
        assert!(display2.contains("3 offsets"));
    }

    // ── Hl4Block tests ──

    #[test]
    fn test_hl_get_zip_type_str() {
        let expected = [
            (0, "Deflate"),
            (1, "Deflate+Transpose"),
            (2, "Zstd"),
            (3, "Zstd+Transpose"),
            (4, "LZ4"),
            (5, "LZ4+Transpose"),
            (255, "Unknown"),
        ];
        for (val, name) in expected {
            let hl = Hl4Block {
                hl_zip_type: val,
                ..Default::default()
            };
            assert_eq!(hl.get_zip_type_str(), name, "hl_zip_type={val}");
        }
    }

    #[test]
    fn test_hl_display() {
        let hl = Hl4Block {
            hl_flags: 0x0001,
            hl_zip_type: 2,
            ..Default::default()
        };
        let display = format!("{hl}");
        assert!(display.contains("HL:"));
        assert!(display.contains("flags=0x0001"));
        assert!(display.contains("zip_type=Zstd"));
    }

    // ── Gd4Block tests ──

    #[test]
    fn test_gd_display() {
        let gd = Gd4Block {
            gd_version: 430,
            ..Default::default()
        };
        let display = format!("{gd}");
        assert!(display.contains("GD:"));
        assert!(display.contains("min_version=4.3.0"));

        let gd = Gd4Block {
            gd_version: 410,
            ..Default::default()
        };
        let display = format!("{gd}");
        assert!(display.contains("min_version=4.1.0"));
    }
}
