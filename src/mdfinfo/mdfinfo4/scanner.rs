//! Block scanner for corrupted MDF4 files — fallback when link chains are broken.
//!
//! Scans the raw file bytes for MDF4 block signatures (`##XX`) at 8-byte-aligned
//! positions. Returns block addresses even when DG→CG→CN link pointers are corrupt.
//!
//! Used as a fallback in `parse_mdf4()` when the primary DG chain yields zero data
//! groups but the file is non-empty.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use crate::mdfinfo::sym_buf_reader::SymBufReader;

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
///
/// `chunk_size` controls how many bytes are read per I/O call (default 65536).
pub fn scan_blocks(rdr: &mut SymBufReader<&File>, chunk_size: usize) -> Vec<FoundBlock> {
    let chunk_size = chunk_size.max(64); // at least one block header worth

    let file_size = match rdr.file_size() {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };

    if file_size < 24 {
        return Vec::new();
    }

    // Rewind to the beginning
    if rdr.seek(SeekFrom::Start(0)).is_err() {
        return Vec::new();
    }

    let mut results: Vec<FoundBlock> = Vec::new();

    // We read overlapping chunks so we don't miss a signature that straddles a boundary.
    // The overlap is 23 bytes (max bytes we need to look behind).
    let overlap = 23usize;
    let mut buf = vec![0u8; chunk_size + overlap];
    let mut file_offset: u64 = 0; // byte offset of the first byte in `buf[overlap..]`

    loop {
        if file_offset >= file_size {
            break;
        }

        // Fill overlap from previous tail
        if file_offset > 0 {
            // bytes [chunk_size .. chunk_size + overlap] from previous iteration
            // are already in buf[0..overlap] from the copy below
        }

        // Read new data into buf[overlap..]
        let to_read = ((file_size - file_offset) as usize).min(chunk_size);
        let n = match rdr.read(&mut buf[overlap..overlap + to_read]) {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };

        let usable = overlap + n;

        // Scan for `##` at 8-byte-aligned positions relative to the file
        // The file position of buf[overlap] is file_offset.
        // buf[i] corresponds to file position: file_offset + i - overlap  (for i >= overlap)
        // We also check i < overlap for the overlap region from last chunk.
        let scan_start = if file_offset == 0 { overlap } else { 0 };
        let mut i = scan_start;
        while i + 23 <= usable {
            // Compute file position of buf[i]
            let file_pos = file_offset as i64 + i as i64 - overlap as i64;

            if file_pos < 0 || !(file_pos as u64).is_multiple_of(8) {
                i += 1;
                continue;
            }

            // Check `##` signature
            if buf[i] != 0x23 || buf[i + 1] != 0x23 {
                i += 8; // advance by alignment unit
                continue;
            }

            // Check that bytes 2-3 are uppercase ASCII (block type)
            let b2 = buf[i + 2];
            let b3 = buf[i + 3];
            if !b2.is_ascii_uppercase() || !b3.is_ascii_uppercase() {
                i += 8;
                continue;
            }

            // Read hdr_len at offset 8 within the block header
            if i + 16 > usable {
                i += 8;
                continue;
            }
            let len = u64::from_le_bytes([
                buf[i + 8],
                buf[i + 9],
                buf[i + 10],
                buf[i + 11],
                buf[i + 12],
                buf[i + 13],
                buf[i + 14],
                buf[i + 15],
            ]);

            if len >= 24 && len <= file_size {
                results.push(FoundBlock {
                    offset: file_pos,
                    id: [0x23, 0x23, b2, b3],
                    len,
                });
            }

            i += 8;
        }

        // Copy tail to overlap region for next iteration
        let tail_start = usable.saturating_sub(overlap);
        let tail_len = usable - tail_start;
        buf.copy_within(tail_start..usable, 0);
        // Zero out the rest of the overlap
        if tail_len < overlap {
            for b in &mut buf[tail_len..overlap] {
                *b = 0;
            }
        }

        file_offset += n as u64;
    }

    results
}

/// Filters `scan_blocks` results to only DG blocks, returning their file offsets.
pub fn find_dg_blocks(rdr: &mut SymBufReader<&File>) -> Vec<i64> {
    scan_blocks(rdr, 65536)
        .into_iter()
        .filter(|b| &b.id == b"##DG")
        .map(|b| b.offset)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
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

        let file = File::open(tmp.path()).unwrap();
        let mut rdr = SymBufReader::new(&file);
        let blocks = scan_blocks(&mut rdr, 256);
        let dg: Vec<_> = blocks.iter().filter(|b| &b.id == b"##DG").collect();
        assert!(!dg.is_empty(), "expected at least one ##DG block");
        assert_eq!(dg[0].offset, 64);
        assert_eq!(dg[0].len, 64);
    }
}
