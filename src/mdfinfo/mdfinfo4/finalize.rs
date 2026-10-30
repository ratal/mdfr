//! Unfinalized MDF4 file recovery — spec section 4.4.2
//!
//! When a recording session ends abruptly (power loss, crash), the MDF writer may not
//! have updated several block fields before closing. The `id_unfin_flags` bits describe
//! which fields need correction. This module fixes all standard bits (0..=8) in memory
//! before data reading begins, and also provides [`finalize_file_in_place`] to patch the
//! file in place on disk without copying.
//!
//! Supported corrections:
//! - Bit 0 (`UNFIN_CG_CYCLE_COUNTERS`): `cg_cycle_count` set to actual records in data
//!   (for both sorted DGs, compressed `HL`→`DL`→`DZ` chains, and unsorted DGs via record-ID scan)
//! - Bits 2/4 (`UNFIN_LAST_DT_LENGTH` / `UNFIN_LAST_DL_BLOCK`): detected via DL chain
//!   walk; the last DT/DZ block's available byte count is used to clamp cycle count.
//! - Bits 5–8 (`UNFIN_VLSD_CG_BYTES`, `UNFIN_VLSD_OFFSET`, `UNFIN_VLSC_CG_BYTES`,
//!   `UNFIN_VLSC_OFFSET`): VLSD/VLSC companion `CGBLOCK` `cg_cycle_count`, `cg_data_bytes`
//!   and `cg_inval_bytes` (64-bit total variable-length byte count) are reconstructed by
//!   scanning the unsorted record stream or sorted `SDBLOCK` chain.

use anyhow::{Context, Result};
use log::{info, warn};
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};

use super::cg_impl::{CG_F_VLSC, CG_F_VLSD};
use super::{Dg4, Dl4Block, Dt4Block, Dz4Block, Hl4Block};
use crate::mdfinfo::IdBlock;
use crate::mdfinfo::sym_buf_reader::SymBufReader;

/// Reads and returns the 4-byte block id at `addr`.
fn read_block_id(rdr: &mut SymBufReader<&File>, addr: i64) -> Result<[u8; 4]> {
    rdr.seek(SeekFrom::Start(addr as u64))
        .context("finalize: could not seek to block")?;
    let mut id = [0u8; 4];
    rdr.read_exact(&mut id)
        .context("finalize: could not read block id")?;
    Ok(id)
}

/// Walks a `DLBLOCK` chain starting at `dl_addr` (whose blocks may be `##DT`, `##SD`, or `##DZ`)
/// and returns `(total_preceding_uncompressed_bytes, last_block_uncompressed_bytes)`.
fn walk_dl_chain(
    rdr: &mut SymBufReader<&File>,
    dl_addr: i64,
    file_size: u64,
) -> Result<(u64, i64)> {
    let mut total_preceding_bytes: u64 = 0;
    let mut current_dl_addr = dl_addr;
    let mut last_dl: Option<Dl4Block> = None;

    for _ in 0..65536u32 {
        if current_dl_addr <= 0 || (current_dl_addr as u64) + 24 > file_size {
            break;
        }
        let id = read_block_id(rdr, current_dl_addr)?;
        if &id != b"##DL" {
            break;
        }
        let dl: Dl4Block =
            binrw::BinReaderExt::read_le(rdr).context("finalize: could not parse DL block")?;

        let is_final_dl = dl.dl_dl_next <= 0;
        let last_idx = if is_final_dl {
            dl.dl_data.len().saturating_sub(1)
        } else {
            dl.dl_data.len()
        };

        for &dt_ptr in dl.dl_data.iter().take(last_idx) {
            if dt_ptr <= 0 {
                continue;
            }
            let block_bytes = read_data_block_uncompressed_len(rdr, dt_ptr, file_size)?;
            total_preceding_bytes += block_bytes;
        }

        if is_final_dl {
            last_dl = Some(dl);
            break;
        }
        current_dl_addr = dl.dl_dl_next;
    }

    let Some(final_dl) = last_dl else {
        warn!("finalize: DL chain exceeded limit or is empty");
        return Ok((total_preceding_bytes, 0));
    };

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

    let last_bytes = read_data_block_uncompressed_len(rdr, last_dt_ptr, file_size)?;
    Ok((total_preceding_bytes, last_bytes as i64))
}

/// Returns the uncompressed payload length of a `##DT`, `##SD`, or `##DZ` block at `ptr`,
/// clamping to available bytes up to `file_size` if the block was truncated.
fn read_data_block_uncompressed_len(
    rdr: &mut SymBufReader<&File>,
    ptr: i64,
    file_size: u64,
) -> Result<u64> {
    if ptr <= 0 || (ptr as u64) + 24 > file_size {
        return Ok(0);
    }
    let id = read_block_id(rdr, ptr)?;
    match &id {
        b"##DT" | b"##SD" | b"##DV" => {
            let dt: Dt4Block = binrw::BinReaderExt::read_le(rdr)
                .context("finalize: could not parse DT/SD block")?;
            let available = file_size.saturating_sub(ptr as u64 + 24);
            if dt.len >= 24 && (dt.len - 24) <= available {
                Ok(dt.len - 24)
            } else {
                Ok(available)
            }
        }
        b"##DZ" => {
            if (ptr as u64) + 48 > file_size {
                return Ok(0);
            }
            let dz: Dz4Block =
                binrw::BinReaderExt::read_le(rdr).context("finalize: could not parse DZ block")?;
            let available_comp = file_size.saturating_sub(ptr as u64 + 48);
            if dz.dz_data_length <= available_comp {
                Ok(dz.dz_org_data_length)
            } else {
                // Truncated DZ block cannot be decompressed
                Ok(0)
            }
        }
        _ => Ok(0),
    }
}

/// Visits the raw uncompressed byte stream of a DG's data chain chunk by chunk (for unsorted record scanning).
fn visit_dg_stream_chunks<F>(
    rdr: &mut SymBufReader<&File>,
    dg_data: i64,
    file_size: u64,
    mut visitor: F,
) -> Result<()>
where
    F: FnMut(&[u8]),
{
    if dg_data <= 0 || (dg_data as u64) + 24 > file_size {
        return Ok(());
    }
    let id = read_block_id(rdr, dg_data)?;
    match &id {
        b"##DT" => {
            let dt: Dt4Block = binrw::BinReaderExt::read_le(rdr)?;
            let avail = file_size.saturating_sub(dg_data as u64 + 24);
            let n = if dt.len >= 24 && (dt.len - 24) <= avail {
                (dt.len - 24) as usize
            } else {
                avail as usize
            };
            let mut remaining = n;
            let mut buf = vec![0u8; 64 * 1024];
            while remaining > 0 {
                let to_read = remaining.min(buf.len());
                rdr.read_exact(&mut buf[..to_read])?;
                remaining -= to_read;
                visitor(&buf[..to_read]);
            }
        }
        b"##DL" => {
            visit_dl_stream_chunks(rdr, dg_data, file_size, visitor)?;
        }
        b"##HL" => {
            let hl: Hl4Block = binrw::BinReaderExt::read_le(rdr)?;
            if hl.hl_dl_first > 0 {
                visit_dl_stream_chunks(rdr, hl.hl_dl_first, file_size, visitor)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn visit_dl_stream_chunks<F>(
    rdr: &mut SymBufReader<&File>,
    mut dl_addr: i64,
    file_size: u64,
    mut visitor: F,
) -> Result<()>
where
    F: FnMut(&[u8]),
{
    let mut chunk_buf = vec![0u8; 64 * 1024];
    for _ in 0..65536u32 {
        if dl_addr <= 0 || (dl_addr as u64) + 24 > file_size {
            break;
        }
        let id = read_block_id(rdr, dl_addr)?;
        if &id != b"##DL" {
            break;
        }
        let dl: Dl4Block = binrw::BinReaderExt::read_le(rdr)?;

        for &ptr in &dl.dl_data {
            if ptr <= 0 || (ptr as u64) + 24 > file_size {
                continue;
            }
            let bid = read_block_id(rdr, ptr)?;
            match &bid {
                b"##DT" | b"##SD" => {
                    let dt: Dt4Block = binrw::BinReaderExt::read_le(rdr)?;
                    let avail = file_size.saturating_sub(ptr as u64 + 24);
                    let n = if dt.len >= 24 && (dt.len - 24) <= avail {
                        (dt.len - 24) as usize
                    } else {
                        avail as usize
                    };
                    let mut remaining = n;
                    while remaining > 0 {
                        let to_read = remaining.min(chunk_buf.len());
                        rdr.read_exact(&mut chunk_buf[..to_read])?;
                        remaining -= to_read;
                        visitor(&chunk_buf[..to_read]);
                    }
                }
                b"##DZ" => {
                    if (ptr as u64) + 48 > file_size {
                        continue;
                    }
                    let dz: Dz4Block = binrw::BinReaderExt::read_le(rdr)?;
                    let avail = file_size.saturating_sub(ptr as u64 + 48);
                    if dz.dz_data_length <= avail {
                        let mut comp = vec![0u8; dz.dz_data_length as usize];
                        rdr.read_exact(&mut comp)?;
                        if let Ok(decomp) = super::data_impl::decompress_data(
                            dz.dz_zip_type,
                            dz.dz_zip_parameter,
                            comp,
                            dz.dz_org_data_length,
                        ) {
                            visitor(&decomp);
                        }
                    }
                }
                _ => {}
            }
        }
        if dl.dl_dl_next <= 0 {
            break;
        }
        dl_addr = dl.dl_dl_next;
    }
    Ok(())
}

/// Fixes `cg_cycle_count` (bit 0) and VLSD/VLSC `cg_data_bytes`/`cg_inval_bytes` (bits 5–8)
/// in each CG block based on the actual data available in the file.
///
/// Returns the number of channel groups whose cycle/byte count was corrected.
pub fn fix_cycle_counts(
    dg: &mut BTreeMap<i64, Dg4>,
    rdr: &mut SymBufReader<&File>,
) -> Result<usize> {
    let file_size = rdr
        .file_size()
        .context("finalize: could not determine file size")?;

    let mut n_fixed = 0usize;

    for dg_struct in dg.values_mut() {
        let dg_data = dg_struct.block.dg_data;
        if dg_data <= 0 || (dg_data as u64) + 24 > file_size {
            continue;
        }

        // Unsorted DG (multiple CGs or rec_id_size > 0): scan the record stream to count cycles per record ID
        // and total VLSD/VLSC payload bytes (bits 5–8) in a streaming fashion without allocating the whole stream.
        if dg_struct.block.dg_rec_id_size > 0 || dg_struct.cg.len() > 1 {
            let rec_id_size = dg_struct.block.dg_rec_id_size as usize;
            if rec_id_size > 0 {
                let mut cycles_per_id: FxHashMap<u64, u64> = FxHashMap::default();
                let mut vlsd_bytes_per_id: FxHashMap<u64, u64> = FxHashMap::default();
                let mut carry = Vec::new();

                visit_dg_stream_chunks(rdr, dg_data, file_size, |chunk| {
                    let mut combined;
                    let slice: &[u8] = if carry.is_empty() {
                        chunk
                    } else {
                        combined = std::mem::take(&mut carry);
                        combined.extend_from_slice(chunk);
                        &combined
                    };

                    let mut pos = 0usize;
                    while pos + rec_id_size <= slice.len() {
                        let rec_id: u64 = match rec_id_size {
                            1 => slice[pos] as u64,
                            2 => u16::from_le_bytes(slice[pos..pos + 2].try_into().unwrap()) as u64,
                            4 => u32::from_le_bytes(slice[pos..pos + 4].try_into().unwrap()) as u64,
                            8 => u64::from_le_bytes(slice[pos..pos + 8].try_into().unwrap()),
                            _ => break,
                        };
                        let Some(cg) = dg_struct.cg.get(&rec_id) else {
                            break;
                        };
                        if (cg.block.cg_flags & (CG_F_VLSD | CG_F_VLSC)) != 0 {
                            if pos + rec_id_size + 4 > slice.len() {
                                break;
                            }
                            let vlen = u32::from_le_bytes(
                                slice[pos + rec_id_size..pos + rec_id_size + 4]
                                    .try_into()
                                    .unwrap(),
                            ) as usize;
                            if pos + rec_id_size + 4 + vlen > slice.len() {
                                break;
                            }
                            *cycles_per_id.entry(rec_id).or_insert(0) += 1;
                            *vlsd_bytes_per_id.entry(rec_id).or_insert(0) += vlen as u64;
                            pos += rec_id_size + 4 + vlen;
                        } else {
                            // `cg.record_length` includes `dg_rec_id_size` in `mdfr`
                            let full_rec_len = cg.record_length as usize;
                            if full_rec_len == 0 || pos + full_rec_len > slice.len() {
                                break;
                            }
                            *cycles_per_id.entry(rec_id).or_insert(0) += 1;
                            pos += full_rec_len;
                        }
                    }

                    if pos < slice.len() {
                        carry.extend_from_slice(&slice[pos..]);
                    }
                })?;

                for (&rec_id, cg) in dg_struct.cg.iter_mut() {
                    let actual_cycles = cycles_per_id.get(&rec_id).copied().unwrap_or(0);
                    if actual_cycles != cg.block.cg_cycle_count {
                        cg.block.cg_cycle_count = actual_cycles;
                        n_fixed += 1;
                    }
                    if (cg.block.cg_flags & (CG_F_VLSD | CG_F_VLSC)) != 0 {
                        let total_vlsd = vlsd_bytes_per_id.get(&rec_id).copied().unwrap_or(0);
                        let low = (total_vlsd & 0xFFFF_FFFF) as u32;
                        let high = (total_vlsd >> 32) as u32;
                        if cg.block.cg_data_bytes != low || cg.block.cg_inval_bytes != high {
                            cg.block.cg_data_bytes = low;
                            cg.block.cg_inval_bytes = high;
                            n_fixed += 1;
                        }
                    }
                }
            }
            continue;
        }

        // Sorted DG: determine total data bytes available
        let id = read_block_id(rdr, dg_data)?;

        let total_data_bytes: u64 = match &id {
            b"##DT" | b"##SD" | b"##DV" => {
                let dt: Dt4Block = binrw::BinReaderExt::read_le(rdr)
                    .context("finalize: could not parse DT block header")?;

                let dt_data_start = dg_data + 24;
                if dt.len >= 24 && (dt.len as i64) < file_size as i64 - dg_data {
                    dt.len - 24
                } else {
                    let available = (file_size as i64) - dt_data_start;
                    if available > 0 { available as u64 } else { 0 }
                }
            }
            b"##DL" => {
                let (preceding, last_available) = walk_dl_chain(rdr, dg_data, file_size)
                    .context("finalize: DL chain walk failed")?;
                preceding + last_available.max(0) as u64
            }
            b"##HL" => {
                let hl: Hl4Block = binrw::BinReaderExt::read_le(rdr)
                    .context("finalize: could not parse HL block")?;
                if hl.hl_dl_first > 0 {
                    let (preceding, last_available) = walk_dl_chain(rdr, hl.hl_dl_first, file_size)
                        .context("finalize: HL->DL chain walk failed")?;
                    preceding + last_available.max(0) as u64
                } else {
                    0
                }
            }
            other => {
                warn!(
                    "finalize: unexpected block type {:?} at dg_data=0x{dg_data:x}, skipping",
                    other
                );
                continue;
            }
        };

        for cg in dg_struct.cg.values_mut() {
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

/// Recovers and finalizes an unfinalized MDF 4.x file **in place** on disk (no file copy).
///
/// Parses the metadata, runs [`fix_cycle_counts`] to compute exact cycle and VLSD/VLSC byte
/// counters from the data blocks present on disk, patches each `CGBLOCK` in place, and rewrites
/// the 64-byte [`IdBlock`] at offset 0 as `"MDF     "` with `id_unfin_flags = 0`.
#[allow(dead_code)]
pub fn finalize_file_in_place(file_name: &str) -> Result<usize> {
    // 1. Parse metadata (which also runs `fix_cycle_counts` in memory)
    let info = crate::mdfinfo::MdfInfo::new(file_name)?;
    let crate::mdfinfo::MdfInfo::V4(info4) = info else {
        return Ok(0);
    };

    // 2. Open with write access and patch each CG's `cg_cycle_count` (+80) and VLSD `cg_data_bytes`/`cg_inval_bytes` (+96..+104)
    let mut f = OpenOptions::new()
        .read(true)
        .write(true)
        .open(file_name)
        .with_context(|| format!("finalize_file_in_place: cannot open {file_name}"))?;

    let mut patched = 0usize;
    for dg in info4.dg.values() {
        for cg in dg.cg.values() {
            let cg_pos = cg.block_position as u64;
            if cg_pos == 0 {
                continue;
            }
            let cycle_count_offset = 32 + cg.block.cg_links * 8;
            f.seek(SeekFrom::Start(cg_pos + cycle_count_offset))?;
            f.write_all(&cg.block.cg_cycle_count.to_le_bytes())?;
            if (cg.block.cg_flags & (CG_F_VLSD | CG_F_VLSC)) != 0 {
                let vlsd_offset = cycle_count_offset + 16;
                f.seek(SeekFrom::Start(cg_pos + vlsd_offset))?;
                f.write_all(&cg.block.cg_data_bytes.to_le_bytes())?;
                f.write_all(&cg.block.cg_inval_bytes.to_le_bytes())?;
            }
            patched += 1;
        }
    }

    // 3. Patch IDBlock at offset 0 to finalized `"MDF     "` with `id_unfin_flags = 0`
    let mut id: IdBlock = info4.id_block;
    id.id_file_id = *b"MDF     ";
    id.id_unfin_flags = 0;
    id.id_custom_unfin_flags = 0;
    f.seek(SeekFrom::Start(0))?;
    let mut id_bytes = Vec::with_capacity(64);
    binrw::BinWrite::write_le(&id, &mut binrw::io::Cursor::new(&mut id_bytes))?;
    f.write_all(&id_bytes)?;
    f.flush()?;

    Ok(patched)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mdfr_core::blocks::cn::CnDataType;
    use mdfr_core::compress::CompressionAlgorithm;
    use mdfr_core::logger::{DataLayout, Logger, LoggerBuilder};
    use mdfr_core::time::ManualClock;

    #[test]
    fn test_finalize_unfinalized_sorted_and_unsorted() -> Result<()> {
        for (layout, algo) in [
            (DataLayout::Sorted, CompressionAlgorithm::NoCompression),
            (DataLayout::Sorted, CompressionAlgorithm::DeflateTranspose),
            (DataLayout::Unsorted, CompressionAlgorithm::NoCompression),
            (DataLayout::Unsorted, CompressionAlgorithm::DeflateTranspose),
        ] {
            println!("Testing layout={layout:?}, algo={algo:?}");
            let mut b = LoggerBuilder::new(layout, algo);
            let g0 = b.add_group("Engine");
            let ch_rpm = b.add_channel(g0, "RPM", CnDataType::UIntLE, 16)?;
            let ch_vlsd = b.add_vlsd_channel(g0, "Msg", CnDataType::StringUtf8)?;

            let schema = b.build()?;
            let tmp = tempfile::Builder::new().suffix(".mf4").tempfile()?;
            let file = tmp.reopen()?;
            let clock = ManualClock::new(1e-3, 10);
            let mut logger = Logger::start(schema, file, clock)?;

            for i in 0..20u64 {
                logger
                    .begin_record(g0)?
                    .set_u64(ch_rpm, 2000 + i * 20)?
                    .set_bytes(ch_vlsd, format!("msg_{i}").as_bytes())?
                    .commit()?;
            }

            // Checkpoint so all data is on disk and unfinalized flags are set
            logger.checkpoint()?;
            // Intentionally DO NOT call logger.close() to simulate a crash/unfinalized file!
            drop(logger);

            // Verify the file was left as "UnFinMF "
            let path_str = tmp.path().to_str().unwrap();
            let mut head = [0u8; 8];
            File::open(path_str)?.read_exact(&mut head)?;
            assert_eq!(&head, b"UnFinMF ");

            // Run in-place finalization
            let patched = finalize_file_in_place(path_str)?;
            assert!(patched >= 1);

            // Verify header is now "MDF     "
            File::open(path_str)?.read_exact(&mut head)?;
            assert_eq!(&head, b"MDF     ");

            // Open with reader and verify cycles
            let mut mdf = crate::mdfreader::Mdf::new(path_str)?;
            mdf.load_all_channels_data_in_memory()?;
            assert!(mdf.get_channel_data("RPM").is_some());
        }
        Ok(())
    }

    #[test]
    fn test_finalize_nonexistent_and_invalid() {
        assert!(finalize_file_in_place("/nonexistent/path/to/file.mf4").is_err());
    }
}
