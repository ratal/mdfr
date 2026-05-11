//! Unfinalized MDF4 file recovery — spec section 4.4.2
//!
//! When a recording session ends abruptly (power loss, crash), the MDF writer may not
//! have updated several block fields before closing. The `id_unfin_flags` bits describe
//! which fields need correction. This module fixes the most impactful ones in memory
//! before data reading begins, so callers get correct channel data without requiring
//! a file copy.
//!
//! Supported corrections:
//! - Bit 0 (`UNFIN_CG_CYCLE_COUNTERS`): `cg_cycle_count` set to actual records in data
//! - Bits 2/4 (`UNFIN_LAST_DT_LENGTH` / `UNFIN_LAST_DL_BLOCK`): detected via DL chain
//!   walk; the last DT block's available byte count is used to clamp cycle count.
//!
//! Not yet supported (deferred): bits 5-8 (VLSD/VLSC byte and offset corrections).

use anyhow::{Context, Result};
use binrw::BinReaderExt;
use log::{info, warn};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;

use super::cg_block::CG_F_VLSD;
use super::{Dg4, Dl4Block, Dt4Block};
use crate::mdfinfo::sym_buf_reader::SymBufReader;

/// Reads and returns the 4-byte block id at `addr` without changing the caller's
/// conceptual position tracker.
fn read_block_id(rdr: &mut SymBufReader<&File>, addr: i64, position: i64) -> Result<[u8; 4]> {
    rdr.seek_relative(addr - position)
        .context("finalize: could not seek to block")?;
    let mut id = [0u8; 4];
    rdr.read_exact(&mut id)
        .context("finalize: could not read block id")?;
    Ok(id)
}

/// Walks the DL chain starting at `dl_addr` and returns the address and available
/// data byte count for the *last* DT/SD block referenced by the last DL link.
///
/// Also returns the total data bytes across all DT blocks except the last (whose
/// length may be wrong in an unfinalized file).
fn walk_dl_chain(
    rdr: &mut SymBufReader<&File>,
    dl_addr: i64,
    file_size: u64,
) -> Result<(u64, i64)> {
    let mut position = dl_addr;
    let mut total_preceding_bytes: u64 = 0;

    // Read and follow DL chain
    let mut current_dl_addr = dl_addr;
    let mut last_dl: Option<Dl4Block> = None;

    // Walk to last DL block in chain (up to 65536 links to prevent runaway)
    for _ in 0..65536u32 {
        rdr.seek_relative(current_dl_addr - position)
            .context("finalize: could not seek to DL block")?;
        let dl: Dl4Block = rdr
            .read_le()
            .context("finalize: could not parse DL block")?;
        let dl_end = current_dl_addr + dl.dl_len as i64;

        // Sum sizes of all DT blocks in this DL that are NOT the last one
        let is_final_dl = dl.dl_dl_next <= 0;
        let last_idx = if is_final_dl {
            // In the final DL, last data pointer points to the unfinalized DT
            dl.dl_data.len().saturating_sub(1)
        } else {
            dl.dl_data.len() // all blocks are complete in non-final DL nodes
        };

        for &dt_ptr in dl.dl_data.iter().take(last_idx) {
            if dt_ptr <= 0 {
                continue;
            }
            rdr.seek_relative(dt_ptr - dl_end)
                .context("finalize: could not seek to DT block in DL")?;
            // Skip the 4-byte id we just identified to be at position dt_ptr
            let mut id_buf = [0u8; 4];
            rdr.read_exact(&mut id_buf)
                .context("finalize: could not read DT id in DL")?;
            let dt: Dt4Block = rdr
                .read_le()
                .context("finalize: could not parse DT block in DL")?;
            if dt.len >= 24 {
                total_preceding_bytes += dt.len - 24;
            }
        }

        if is_final_dl {
            last_dl = Some(dl);
            break;
        }
        position = dl_end;
        current_dl_addr = dl.dl_dl_next;
    }

    let Some(final_dl) = last_dl else {
        warn!("finalize: DL chain exceeded limit or is empty");
        return Ok((total_preceding_bytes, 0));
    };

    // Find last DT pointer in the final DL
    let last_dt_ptr = final_dl
        .dl_data
        .iter()
        .rev()
        .find(|&&p| p > 0)
        .copied()
        .unwrap_or(0);

    if last_dt_ptr <= 0 {
        return Ok((total_preceding_bytes, 0));
    }

    // Available bytes in the last (potentially unfinalized) DT block
    let header_size: i64 = 4 + 20; // 4-byte id already consumed + rest of header (20 bytes)
    let last_dt_data_start = last_dt_ptr + header_size;
    let available = if (file_size as i64) > last_dt_data_start {
        file_size as i64 - last_dt_data_start
    } else {
        0
    };

    Ok((total_preceding_bytes, available))
}

/// Fixes `cg_cycle_count` in each CG block based on the actual data available in the
/// file. Should be called after metadata parsing when `is_unfinalized` is true.
///
/// Returns the number of channel groups whose cycle count was corrected.
pub fn fix_cycle_counts(
    dg: &mut BTreeMap<i64, Dg4>,
    rdr: &mut SymBufReader<&File>,
) -> Result<usize> {
    let file_size = rdr
        .file_size()
        .context("finalize: could not determine file size")?;

    let mut n_fixed = 0usize;

    // Position tracker for seeks (starts at 0; we always use absolute seeks via seek_relative)
    let mut position: i64 = 0;

    for dg_struct in dg.values_mut() {
        let dg_data = dg_struct.block.dg_data;
        if dg_data <= 0 {
            continue;
        }

        // --- determine total data bytes available ---
        let id = read_block_id(rdr, dg_data, position)?;
        position = dg_data + 4;

        let total_data_bytes: u64 = match &id {
            b"##DT" | b"##SD" => {
                // Simple DT/SD: the data starts at dg_data + 24 (header size)
                // The hdr_len field may be wrong (unfinalized), so use file_size instead.
                let dt: Dt4Block = rdr
                    .read_le()
                    .context("finalize: could not parse DT block header")?;
                position = dg_data + 24;

                let dt_data_start = dg_data + 24;
                if dt.len >= 24 && (dt.len as i64) < file_size as i64 - dg_data {
                    // len looks correct — trust it
                    dt.len - 24
                } else {
                    // len is wrong (unfinalized) — use available bytes up to EOF
                    let available = (file_size as i64) - dt_data_start;
                    if available > 0 { available as u64 } else { 0 }
                }
            }
            b"##DL" => {
                // DL chain: walk it
                let (preceding, last_available) = walk_dl_chain(rdr, dg_data, file_size)
                    .context("finalize: DL chain walk failed")?;
                position = 0; // position lost after DL walk; reset to force absolute seek
                preceding + last_available.max(0) as u64
            }
            b"##HL" => {
                // HL wraps a DL chain — skip for now; can't easily determine uncompressed size
                warn!(
                    "finalize: HL block at 0x{dg_data:x} (compressed data list) — \
                     cycle count correction not supported for compressed chains"
                );
                continue;
            }
            other => {
                warn!(
                    "finalize: unexpected block type {:?} at dg_data=0x{dg_data:x}, skipping",
                    other
                );
                continue;
            }
        };

        // --- fix cg_cycle_count for each channel group ---
        for cg in dg_struct.cg.values_mut() {
            // Skip VLSD channel groups — their cycle count semantics differ
            if (cg.block.cg_flags & CG_F_VLSD) != 0 {
                continue;
            }

            let record_length = cg.record_length as u64;
            if record_length == 0 {
                continue;
            }

            let actual_count = total_data_bytes / record_length;
            if actual_count != cg.block.cg_cycle_count {
                info!(
                    "finalize: CG at 0x{:x} corrected cg_cycle_count {} → {}",
                    cg.block_position, cg.block.cg_cycle_count, actual_count
                );
                cg.block.cg_cycle_count = actual_count;
                n_fixed += 1;
            }
        }
    }

    Ok(n_fixed)
}
