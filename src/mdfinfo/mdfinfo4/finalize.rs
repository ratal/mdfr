//! Unfinalized MDF4 file recovery — spec section 4.4.2
//!
//! When a recording session ends abruptly (power loss, crash), the MDF writer may not
//! have updated several block fields before closing. The `id_unfin_flags` bits describe
//! which fields need correction. This module fixes all standard bits (0..=8) in memory
//! before data reading begins, and also provides [`finalize_file_in_place`] to patch the
//! file in place on disk without copying.
//!
//! Both read-only recovery (`fix_cycle_counts`) and in-place patching (`finalize_file_in_place`)
//! share the single `mdfr_core::logger::recover` implementation.

use anyhow::{Context, Result};
use log::info;
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};

use super::Dg4;
use crate::mdfinfo::sym_buf_reader::SymBufReader;
use mdfr_core::logger::recover::{IoReadSeek, recover_in_place, scan_unfinalized_counts};

/// Fixes `cg_cycle_count` (bit 0), VLSD/VLSC `cg_data_bytes`/`cg_inval_bytes` (bits 5–8),
/// and unpatched `DLBLOCK` pointers in memory based on the actual data available in the file.
///
/// Returns the number of channel groups whose cycle/byte count was corrected.
pub fn fix_cycle_counts(
    dg: &mut BTreeMap<i64, Dg4>,
    rdr: &mut SymBufReader<&File>,
) -> Result<usize> {
    let file_size = rdr
        .file_size()
        .context("finalize: could not determine file size")?;
    let mut io = IoReadSeek::new(rdr, Some(file_size));
    let scan = scan_unfinalized_counts(&mut io)
        .map_err(|e| anyhow::anyhow!("finalize: recovery scan failed: {e}"))?;

    let cycles_by_cg: FxHashMap<u64, u64> = scan.cg_cycles.into_iter().collect();
    let vlsd_by_cg: FxHashMap<u64, u64> = scan.vlsd_bytes.into_iter().collect();

    let mut n_fixed = 0usize;
    for dg_struct in dg.values_mut() {
        for &(dl_pos, ref ptrs) in &scan.recovered_dl_data {
            dg_struct
                .recovered_dl_data
                .insert(dl_pos as i64, ptrs.clone());
        }
        for cg in dg_struct.cg.values_mut() {
            let cg_pos = cg.block_position as u64;
            let mut changed = false;
            if let Some(&cycles) = cycles_by_cg.get(&cg_pos)
                && cg.block.cg_cycle_count != cycles
            {
                info!(
                    "finalize: CG at 0x{cg_pos:x}: cycle count {} -> {cycles}",
                    cg.block.cg_cycle_count
                );
                cg.block.cg_cycle_count = cycles;
                changed = true;
            }
            if let Some(&total_vlsd) = vlsd_by_cg.get(&cg_pos) {
                let low = (total_vlsd & 0xFFFF_FFFF) as u32;
                let high = (total_vlsd >> 32) as u32;
                if cg.block.cg_data_bytes != low || cg.block.cg_inval_bytes != high {
                    cg.block.cg_data_bytes = low;
                    cg.block.cg_inval_bytes = high;
                    changed = true;
                }
            }
            if changed {
                n_fixed += 1;
            }
        }
    }

    Ok(n_fixed)
}

/// Finalizes an unfinalized (`"UnFinMF "`) MDF4 file **in place on disk**.
///
/// Delegates to [`mdfr_core::logger::recover::recover_in_place`], patching `DLBLOCK`s,
/// `CGBLOCK` cycle counts / VLSD byte counters, and rewriting the 64-byte `IDBLOCK`
/// at offset 0 as `"MDF     "` with `id_unfin_flags = 0`.
#[allow(dead_code)]
pub fn finalize_file_in_place(file_name: &str) -> Result<usize> {
    let mut f = OpenOptions::new()
        .read(true)
        .write(true)
        .open(file_name)
        .with_context(|| format!("finalize_file_in_place: cannot open {file_name}"))?;
    recover_in_place(&mut f).map_err(|e| anyhow::anyhow!("finalize_file_in_place: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mdfr_core::blocks::cn::CnDataType;
    use mdfr_core::compress::CompressionAlgorithm;
    use mdfr_core::logger::{DataLayout, Logger, LoggerBuilder};
    use mdfr_core::time::ManualClock;
    use std::io::Read;

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
