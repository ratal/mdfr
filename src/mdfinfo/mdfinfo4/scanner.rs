//! Block scanner for corrupted MDF4 files — fallback when link chains are broken.
//!
//! Scans the raw file bytes for MDF4 block signatures (`##XX`) at 8-byte-aligned
//! positions. Returns block addresses even when DG→CG→CN link pointers are corrupt.
//!
//! Used as a fallback in `parse_mdf4()` when the primary DG chain yields zero data
//! groups but the file is non-empty.

use crate::mdfinfo::sym_buf_reader::SliceCursor;

/// A discovered block with its file offset and 4-byte identifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoundBlock {
    /// Byte offset from the start of the file.
    pub offset: i64,
    /// Block type identifier, e.g. `*b"##DG"`.
    pub id: [u8; 4],
    /// Block length as reported in the header (`hdr_len`), or 0 if the header
    /// could not be fully read.
    pub len: u64,
}

/// Scans the entire file for MDF4 block signatures at 8-byte-aligned positions.
///
/// A candidate is accepted when:
/// 1. The first two bytes are `##` (0x23 0x23).
/// 2. The next two bytes form a plausible ASCII block type (`A`–`Z`).
/// 3. The 8-byte `hdr_len` field at offset 8 satisfies `24 ≤ len ≤ file_size`.
pub fn scan_blocks(rdr: &mut SliceCursor<'_>, _chunk_size: usize) -> Vec<FoundBlock> {
    let data = rdr.as_slice();
    let file_size = data.len() as u64;

    if data.len() < 24 {
        return Vec::new();
    }

    let mut results: Vec<FoundBlock> = Vec::new();
    let mut i = 0usize;
    while i + 24 <= data.len() {
        if data[i] == 0x23 && data[i + 1] == 0x23 {
            let b2 = data[i + 2];
            let b3 = data[i + 3];
            if b2.is_ascii_uppercase() && b3.is_ascii_uppercase() {
                let len = u64::from_le_bytes(data[i + 8..i + 16].try_into().unwrap());
                if len >= 24 && len <= file_size {
                    results.push(FoundBlock {
                        offset: i as i64,
                        id: [0x23, 0x23, b2, b3],
                        len,
                    });
                }
            }
        }
        i += 8;
    }

    results
}

/// Filters `scan_blocks` results to only DG blocks, returning their file offsets.
pub fn find_dg_blocks(rdr: &mut SliceCursor<'_>) -> Vec<i64> {
    scan_blocks(rdr, 65536)
        .into_iter()
        .filter(|b| &b.id == b"##DG")
        .map(|b| b.offset)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mdfinfo::sym_buf_reader::MappedFile;
    use std::io::Write;

    /// Builds a minimal byte sequence with an embedded block signature and checks
    /// that the scanner finds it.
    #[test]
    fn test_scan_finds_dg_block() {
        use tempfile::NamedTempFile;

        let mut tmp = NamedTempFile::new().unwrap();

        // Write 64 zero bytes (simulating IDBLOCK), then a plausible ##DG header
        let mut payload = vec![0u8; 64];

        // ##DG at offset 64 (8-byte aligned)
        payload.extend_from_slice(b"##DG"); // id
        payload.extend_from_slice(&[0u8; 4]); // reserved
        payload.extend_from_slice(&64u64.to_le_bytes()); // hdr_len = 64
        payload.extend_from_slice(&[0u8; 48]); // rest of block

        tmp.write_all(&payload).unwrap();
        tmp.flush().unwrap();

        let mapped = MappedFile::open(tmp.path().to_str().unwrap()).unwrap();
        let mut rdr = SliceCursor::new(&mapped);
        let blocks = scan_blocks(&mut rdr, 256);
        let dg: Vec<_> = blocks.iter().filter(|b| &b.id == b"##DG").collect();
        assert!(!dg.is_empty(), "expected at least one ##DG block");
        assert_eq!(dg[0].offset, 64);
        assert_eq!(dg[0].len, 64);
    }
}
