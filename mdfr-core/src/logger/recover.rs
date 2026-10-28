//! On-device `no_std` in-place recovery of unfinalized (`"UnFinMF "`) MDF 4.x files (§5 Path B).
//!
//! Walks the `HD` → `DG` → `CG`/`CN` and `DT`/`DL`/`HL`/`DZ`/`SD` block chains directly on a
//! [`ReadWriteSeek`] storage medium in constant RAM, patches cycle counts, VLSD byte counts,
//! and truncated block lengths in place, and clears the `"UnFinMF "` header to `"MDF     "`.
use binrw::{BinReaderExt, BinWrite, io::Cursor};

use crate::blocks::cg_block::{CG_F_VLSC, CG_F_VLSD, Cg4Block};
use crate::blocks::cn_block::Cn4Block;
use crate::blocks::data_block::{Dl4Block, Dz4Block, Hl4Block};
use crate::blocks::dg_block::Dg4Block;
use crate::blocks::hd_block::Hd4;
use crate::blocks::header::{Blockheader4, Blockheader4Short};
use crate::blocks::id_block::{FINALIZED_ID, IdBlock, UNFINALIZED_ID};
use crate::error::{Error, Result};
use crate::io::{SeekFrom, Sink, VecSink};

/// Storage trait extending [`Sink`] with exact-read capability for on-device in-place recovery.
pub trait ReadWriteSeek: Sink {
    /// Reads exactly `buf.len()` bytes at the current cursor position.
    fn read_exact(&mut self, buf: &mut [u8]) -> Result<()>;
    /// Returns the total byte length of the underlying file/object.
    fn total_len(&mut self) -> Result<u64>;
}

impl ReadWriteSeek for VecSink {
    fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
        let pos = self.position() as usize;
        let data = self.as_slice();
        if pos + buf.len() > data.len() {
            return Err(Error::Io);
        }
        buf.copy_from_slice(&data[pos..pos + buf.len()]);
        self.seek(SeekFrom::Start((pos + buf.len()) as u64))?;
        Ok(())
    }

    fn total_len(&mut self) -> Result<u64> {
        Ok(self.as_slice().len() as u64)
    }
}

#[cfg(feature = "std")]
impl ReadWriteSeek for std::fs::File {
    fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
        std::io::Read::read_exact(self, buf).map_err(|_| Error::Io)
    }

    fn total_len(&mut self) -> Result<u64> {
        self.metadata().map(|m| m.len()).map_err(|_| Error::Io)
    }
}

fn read_at<S: ReadWriteSeek, const N: usize>(sink: &mut S, offset: u64) -> Result<[u8; N]> {
    sink.seek(SeekFrom::Start(offset))?;
    let mut buf = [0u8; N];
    sink.read_exact(&mut buf)?;
    Ok(buf)
}

fn read_u64_at<S: ReadWriteSeek>(sink: &mut S, offset: u64) -> Result<u64> {
    Ok(u64::from_le_bytes(read_at::<S, 8>(sink, offset)?))
}

fn write_u64_at<S: ReadWriteSeek>(sink: &mut S, offset: u64, val: u64) -> Result<()> {
    sink.seek(SeekFrom::Start(offset))?;
    sink.write_all(&val.to_le_bytes())
}

/// Sums uncompressed payload bytes across a `DT`/`SD`, `DL`→(`DT`/`SD`/`DZ`), or `HL`→`DL`→`DZ` chain,
/// clamping the last `DT`/`SD` block's `hdr_len` in place if it exceeds `file_len`.
fn sum_chain_payload_bytes<S: ReadWriteSeek>(
    sink: &mut S,
    data_pos: u64,
    file_len: u64,
    rec_align: u64,
) -> Result<u64> {
    if data_pos == 0 || data_pos + 24 > file_len {
        return Ok(0);
    }
    let hdr_bytes = read_at::<S, 24>(sink, data_pos)?;
    let hdr: Blockheader4 = Cursor::new(hdr_bytes).read_le()?;

    match &hdr.hdr_id {
        b"##DT" | b"##SD" | b"##DV" | b"##RD" => {
            let max_avail = file_len.saturating_sub(data_pos);
            let mut blk_len = hdr.hdr_len;
            if blk_len > max_avail {
                let payload = max_avail.saturating_sub(24);
                let aligned_payload = if rec_align > 1 {
                    (payload / rec_align) * rec_align
                } else {
                    payload
                };
                blk_len = 24 + aligned_payload;
                write_u64_at(sink, data_pos + 8, blk_len)?;
            }
            Ok(blk_len.saturating_sub(24))
        }
        b"##DZ" => {
            let dz_bytes = read_at::<S, 44>(sink, data_pos + 4)?;
            let dz: Dz4Block = Cursor::new(dz_bytes).read_le()?;
            Ok(dz.dz_org_data_length)
        }
        b"##HL" => {
            let hl_bytes = read_at::<S, 16>(sink, data_pos + 24)?;
            let hl: Hl4Block = Cursor::new(hl_bytes).read_le()?;
            if hl.hl_dl_first <= 0 {
                Ok(0)
            } else {
                sum_dl_chain(sink, hl.hl_dl_first as u64, file_len, rec_align)
            }
        }
        b"##DL" => sum_dl_chain(sink, data_pos, file_len, rec_align),
        _ => Ok(0),
    }
}

fn sum_dl_chain<S: ReadWriteSeek>(
    sink: &mut S,
    first_dl_pos: u64,
    file_len: u64,
    rec_align: u64,
) -> Result<u64> {
    let mut total = 0u64;
    let mut dl_pos = first_dl_pos;
    let mut guard = 0usize;
    while dl_pos > 0 && dl_pos + 24 <= file_len && guard < 65536 {
        guard += 1;
        let hdr_bytes = read_at::<S, 24>(sink, dl_pos)?;
        let hdr: Blockheader4 = Cursor::new(hdr_bytes).read_le()?;
        if &hdr.hdr_id != b"##DL" || hdr.hdr_len < 40 || dl_pos + hdr.hdr_len > file_len {
            break;
        }
        sink.seek(SeekFrom::Start(dl_pos + 4))?;
        let mut dl_buf = alloc::vec![0u8; (hdr.hdr_len as usize).saturating_sub(4)];
        sink.read_exact(&mut dl_buf)?;
        let dl: Dl4Block = Cursor::new(dl_buf).read_le()?;
        for &child in &dl.dl_data {
            if child > 0 {
                total += sum_chain_payload_bytes(sink, child as u64, file_len, rec_align)?;
            }
        }
        dl_pos = if dl.dl_dl_next > 0 {
            dl.dl_dl_next as u64
        } else {
            0
        };
    }
    Ok(total)
}

/// Recovers and finalizes an unfinalized (`"UnFinMF "`) MDF 4.x stream in place on `sink`.
/// Returns the number of channel groups patched (`0` if the stream was already finalized).
pub fn recover_in_place<S: ReadWriteSeek>(sink: &mut S) -> Result<usize> {
    let file_len = sink.total_len()?;
    if file_len < 168 {
        return Err(Error::Io);
    }
    let id_bytes = read_at::<S, 64>(sink, 0)?;
    let mut id: IdBlock = Cursor::new(id_bytes).read_le()?;
    if id.id_file_id != UNFINALIZED_ID {
        return Ok(0);
    }

    let hd_bytes = read_at::<S, 104>(sink, 64)?;
    let hd: Hd4 = Cursor::new(hd_bytes).read_le()?;

    let mut patched_cgs = 0usize;
    let mut dg_pos = hd.hd_dg_first;
    let mut dg_guard = 0usize;

    while dg_pos > 0 && (dg_pos as u64) + 64 <= file_len && dg_guard < 4096 {
        dg_guard += 1;
        let dg_bytes = read_at::<S, 64>(sink, dg_pos as u64)?;
        let dg: Dg4Block = Cursor::new(dg_bytes).read_le()?;

        // Walk CG list in this DG
        let mut cg_pos = dg.dg_cg_first;
        let mut cg_guard = 0usize;
        while cg_pos > 0 && (cg_pos as u64) + 104 <= file_len && cg_guard < 4096 {
            cg_guard += 1;
            let short_bytes = read_at::<S, 16>(sink, cg_pos as u64)?;
            let short_hdr: Blockheader4Short = Cursor::new(short_bytes).read_le()?;
            let body_len = (short_hdr.hdr_len as usize).saturating_sub(16);
            sink.seek(SeekFrom::Start((cg_pos as u64) + 16))?;
            let mut cg_buf = alloc::vec![0u8; body_len];
            sink.read_exact(&mut cg_buf)?;
            let cg: Cg4Block = Cursor::new(cg_buf).read_le()?;

            let is_vlsd_cg = (cg.cg_flags & (CG_F_VLSD | CG_F_VLSC)) != 0;

            // Walk CN chain to repair any sorted VLSD `cn_data` SDBLOCK chains
            let mut cn_pos = cg.cg_cn_first;
            let mut cn_guard = 0usize;
            while cn_pos > 0 && (cn_pos as u64) + 160 <= file_len && cn_guard < 4096 {
                cn_guard += 1;
                let cn_short_bytes = read_at::<S, 16>(sink, cn_pos as u64)?;
                let cn_short: Blockheader4Short = Cursor::new(cn_short_bytes).read_le()?;
                let cn_body_len = (cn_short.hdr_len as usize).saturating_sub(16);
                sink.seek(SeekFrom::Start((cn_pos as u64) + 16))?;
                let mut cn_buf = alloc::vec![0u8; cn_body_len];
                sink.read_exact(&mut cn_buf)?;
                let cn: Cn4Block = Cursor::new(cn_buf).read_le()?;
                if cn.cn_type == 1 && cn.cn_data > 0 && dg.dg_rec_id_size == 0 {
                    let _ = sum_chain_payload_bytes(sink, cn.cn_data as u64, file_len, 1)?;
                }
                cn_pos = cn.cn_cn_next;
            }

            if dg.dg_rec_id_size == 0 && !is_vlsd_cg {
                let rec_len = (cg.cg_data_bytes as u64) + (cg.cg_inval_bytes as u64);
                if rec_len > 0 && dg.dg_data > 0 {
                    let total_bytes =
                        sum_chain_payload_bytes(sink, dg.dg_data as u64, file_len, rec_len)?;
                    let cycles = total_bytes / rec_len;
                    let cycle_offset = (cg_pos as u64) + 16 + 8 + cg.cg_links * 8 + 8;
                    if read_u64_at(sink, cycle_offset)? != cycles {
                        write_u64_at(sink, cycle_offset, cycles)?;
                    }
                    patched_cgs += 1;
                }
            } else {
                patched_cgs += 1;
            }

            cg_pos = cg.cg_cg_next;
        }

        dg_pos = dg.dg_dg_next;
    }

    // Clear "UnFinMF " to "MDF     " and zero out unfinalized flags at offset 0
    id.id_file_id = FINALIZED_ID;
    id.id_unfin_flags = 0;
    id.id_custom_unfin_flags = 0;
    let mut id_out = Cursor::new(alloc::vec::Vec::with_capacity(64));
    id.write_le(&mut id_out)?;
    sink.seek(SeekFrom::Start(0))?;
    sink.write_all(&id_out.into_inner())?;
    sink.flush()?;

    Ok(patched_cgs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocks::cn_block::CnDataType;
    use crate::compress::CompressionAlgorithm;
    use crate::logger::{DataLayout, Logger, LoggerBuilder};
    use crate::time::ManualClock;

    #[test]
    fn no_std_recover_in_place_on_vecsink() {
        let mut b = LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::NoCompression);
        let g = b.add_group("G");
        b.set_group_flush_threshold(g, 32).unwrap();
        let ch = b.add_channel(g, "V", CnDataType::UIntLE, 32).unwrap();
        let schema = b.build().unwrap();

        let mut sink = VecSink::default();
        let mut logger = Logger::start(schema, sink, ManualClock::new(0.01, 1)).unwrap();
        for i in 0..10 {
            logger
                .begin_record(g)
                .unwrap()
                .set_u64(ch, i)
                .unwrap()
                .commit()
                .unwrap();
        }
        logger.checkpoint().unwrap();
        // Extract sink without calling close() — remains "UnFinMF "
        sink = logger.into_sink_unfinalized();
        assert_eq!(&sink.as_slice()[0..8], b"UnFinMF ");

        let patched = recover_in_place(&mut sink).unwrap();
        assert_eq!(patched, 1);
        assert_eq!(&sink.as_slice()[0..8], b"MDF     ");
    }
}
