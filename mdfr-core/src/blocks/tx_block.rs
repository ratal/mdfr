//! Text (`##TX`) and XML Metadata (`##MD`) block serialization for MDF4 — spec section 6.2.
//!
//! Both block types share the same binary layout:
//! - 24-byte [`Blockheader4`] (`hdr_id = b"##TX"` or `b"##MD"`, `hdr_links = 0`)
//! - Null-terminated UTF-8 string bytes
//! - Zero-padding to 8-byte alignment
use alloc::vec;
use alloc::vec::Vec;
use binrw::{BinWrite, io::Cursor};

use super::header::Blockheader4;
use crate::error::{Error, Result};

/// Computes the 8-byte aligned size of `n` bytes (`ceil(n / 8) * 8`).
#[inline]
pub const fn align8(n: u64) -> u64 {
    (n + 7) & !7
}

/// Computes the serialized byte length of a TX or MD block containing `text` (without trailing NUL in `text`).
#[inline]
pub const fn tx_block_len(text: &[u8]) -> u64 {
    24 + align8(text.len() as u64 + 1)
}

/// Encodes a `##TX` or `##MD` block (`id = *b"##TX"` or `*b"##MD"`) with null-termination and 8-byte alignment.
pub fn encode_tx_or_md(id: [u8; 4], text: &[u8]) -> Result<Vec<u8>> {
    // Strip any trailing NUL bytes already present so we add exactly one
    let trimmed_len = text
        .iter()
        .rposition(|&b| b != 0)
        .map(|i| i + 1)
        .unwrap_or(0);
    let body = &text[..trimmed_len];
    let payload_aligned = align8(body.len() as u64 + 1) as usize;
    let total_len = 24 + payload_aligned;

    let hdr = Blockheader4 {
        hdr_id: id,
        hdr_gap: [0u8; 4],
        hdr_len: total_len as u64,
        hdr_links: 0,
    };
    let mut out = vec![0u8; total_len];
    {
        let mut cur = Cursor::new(&mut out[..24]);
        hdr.write_le(&mut cur).map_err(|_| Error::Io)?;
    }
    out[24..24 + body.len()].copy_from_slice(body);
    // Remaining bytes (including the NUL terminator and 8-byte padding) are already 0.
    Ok(out)
}

/// Encodes a `##TX` block from a UTF-8 string.
#[inline]
pub fn encode_tx(text: &str) -> Result<Vec<u8>> {
    encode_tx_or_md(*b"##TX", text.as_bytes())
}

/// Encodes a `##MD` block from an XML string.
#[inline]
pub fn encode_md(xml: &str) -> Result<Vec<u8>> {
    encode_tx_or_md(*b"##MD", xml.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tx_is_8_byte_aligned_and_null_terminated() {
        let b = encode_tx("rpm").unwrap();
        // "rpm\0" is 4 bytes -> aligned to 8 -> 24 + 8 = 32 bytes
        assert_eq!(b.len(), 32);
        assert_eq!(&b[0..4], b"##TX");
        assert_eq!(u64::from_le_bytes(b[8..16].try_into().unwrap()), 32);
        assert_eq!(&b[24..28], b"rpm\0");
        assert_eq!(&b[28..32], &[0, 0, 0, 0]);
    }
}
