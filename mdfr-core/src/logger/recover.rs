//! On-device `no_std` in-place recovery of unfinalized (`"UnFinMF "`) MDF 4.x files (§5 Path B).
//!
//! Walks the `HD` → `DG` → `CG`/`CN` and `DT`/`DL`/`HL`/`DZ`/`SD` block chains directly on a
//! [`ReadWriteSeek`] storage medium in constant RAM, patches cycle counts, VLSD byte counts,
//! and truncated block lengths in place, and clears the `"UnFinMF "` header to `"MDF     "`.
use alloc::vec::Vec;
use binrw::{BinReaderExt, BinWrite, io::Cursor};

use crate::blocks::cg_block::{CG_F_VLSC, CG_F_VLSD, Cg4Block};
use crate::blocks::cn_block::Cn4Block;
use crate::blocks::data_block::{Dl4Block, Dz4Block, Hl4Block};
use crate::blocks::dg_block::Dg4Block;
use crate::blocks::hd_block::Hd4;
use crate::blocks::header::{Blockheader4, Blockheader4Short};
use crate::blocks::id_block::{
    FINALIZED_ID, IdBlock, UNFIN_LAST_DL_BLOCK, UNFIN_VLSC_OFFSET, UNFIN_VLSD_OFFSET,
    UNFINALIZED_ID,
};
use crate::blocks::tx_block::align8;
use crate::error::{Error, Result};
use crate::io::{SeekFrom, Sink, VecSink};

/// Read-only seekable storage trait for scanning unfinalized MDF 4.x streams.
pub trait ReadSeek {
    /// Seeks to `pos`.
    fn seek(&mut self, pos: SeekFrom) -> Result<u64>;
    /// Reads exactly `buf.len()` bytes at the current cursor position.
    fn read_exact(&mut self, buf: &mut [u8]) -> Result<()>;
    /// Returns the total byte length of the underlying file/object.
    fn total_len(&mut self) -> Result<u64>;
}

/// Storage trait extending [`Sink`] and [`ReadSeek`] for on-device in-place recovery.
pub trait ReadWriteSeek: Sink + ReadSeek {}

impl ReadSeek for VecSink {
    fn seek(&mut self, pos: SeekFrom) -> Result<u64> {
        Sink::seek(self, pos)
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
        let pos = self.position() as usize;
        let data = self.as_slice();
        if pos + buf.len() > data.len() {
            return Err(Error::Io);
        }
        buf.copy_from_slice(&data[pos..pos + buf.len()]);
        Sink::seek(self, SeekFrom::Start((pos + buf.len()) as u64))?;
        Ok(())
    }

    fn total_len(&mut self) -> Result<u64> {
        Ok(self.as_slice().len() as u64)
    }
}

impl ReadWriteSeek for VecSink {}

#[cfg(feature = "std")]
impl ReadSeek for std::fs::File {
    fn seek(&mut self, pos: SeekFrom) -> Result<u64> {
        Sink::seek(self, pos)
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
        std::io::Read::read_exact(self, buf).map_err(|_| Error::Io)
    }

    fn total_len(&mut self) -> Result<u64> {
        self.metadata().map(|m| m.len()).map_err(|_| Error::Io)
    }
}

#[cfg(feature = "std")]
impl ReadWriteSeek for std::fs::File {}

/// Adapter wrapping any `std::io::Read + std::io::Seek` as [`ReadSeek`] for read-only scanning.
#[cfg(feature = "std")]
pub struct IoReadSeek<'a, R: std::io::Read + std::io::Seek> {
    pub inner: &'a mut R,
    pub known_len: Option<u64>,
}

#[cfg(feature = "std")]
impl<'a, R: std::io::Read + std::io::Seek> IoReadSeek<'a, R> {
    pub fn new(inner: &'a mut R, known_len: Option<u64>) -> Self {
        Self { inner, known_len }
    }
}

#[cfg(feature = "std")]
impl<R: std::io::Read + std::io::Seek> ReadSeek for IoReadSeek<'_, R> {
    fn seek(&mut self, pos: SeekFrom) -> Result<u64> {
        let std_pos = match pos {
            SeekFrom::Start(p) => std::io::SeekFrom::Start(p),
            SeekFrom::End(p) => std::io::SeekFrom::End(p),
        };
        std::io::Seek::seek(self.inner, std_pos).map_err(|_| Error::Io)
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
        std::io::Read::read_exact(self.inner, buf).map_err(|_| Error::Io)
    }

    fn total_len(&mut self) -> Result<u64> {
        if let Some(len) = self.known_len {
            return Ok(len);
        }
        let cur = self.inner.stream_position().map_err(|_| Error::Io)?;
        let end =
            std::io::Seek::seek(self.inner, std::io::SeekFrom::End(0)).map_err(|_| Error::Io)?;
        std::io::Seek::seek(self.inner, std::io::SeekFrom::Start(cur)).map_err(|_| Error::Io)?;
        Ok(end)
    }
}

/// Results from a read-only recovery scan of an unfinalized MDF 4.x stream.
#[derive(Debug, Clone, Default)]
pub struct UnfinalizedScanResult {
    /// `(cg_pos, recovered_cycle_count)` for every channel group.
    pub cg_cycles: Vec<(u64, u64)>,
    /// `(cg_pos, recovered_vlsd_bytes)` for VLSD/VLSC channel groups.
    pub vlsd_bytes: Vec<(u64, u64)>,
    /// `(dl_pos, recovered_dl_data)` for unpatched `DLBLOCK`s (`dl_count == 0` on disk).
    pub recovered_dl_data: Vec<(u64, Vec<i64>)>,
}

trait RecoveryIo: ReadSeek {
    fn can_write(&self) -> bool;
    fn write_bytes_at(&mut self, offset: u64, data: &[u8]) -> Result<()>;
    fn flush(&mut self) -> Result<()>;
}

struct ReadOnlyIo<'a, R: ReadSeek>(&'a mut R);

impl<R: ReadSeek> ReadSeek for ReadOnlyIo<'_, R> {
    fn seek(&mut self, pos: SeekFrom) -> Result<u64> {
        self.0.seek(pos)
    }
    fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
        self.0.read_exact(buf)
    }
    fn total_len(&mut self) -> Result<u64> {
        self.0.total_len()
    }
}

impl<R: ReadSeek> RecoveryIo for ReadOnlyIo<'_, R> {
    fn can_write(&self) -> bool {
        false
    }
    fn write_bytes_at(&mut self, _offset: u64, _data: &[u8]) -> Result<()> {
        Ok(())
    }
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

struct ReadWriteIo<'a, S: ReadWriteSeek>(&'a mut S);

impl<S: ReadWriteSeek> ReadSeek for ReadWriteIo<'_, S> {
    fn seek(&mut self, pos: SeekFrom) -> Result<u64> {
        ReadSeek::seek(self.0, pos)
    }
    fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
        self.0.read_exact(buf)
    }
    fn total_len(&mut self) -> Result<u64> {
        self.0.total_len()
    }
}

impl<S: ReadWriteSeek> RecoveryIo for ReadWriteIo<'_, S> {
    fn can_write(&self) -> bool {
        true
    }
    fn write_bytes_at(&mut self, offset: u64, data: &[u8]) -> Result<()> {
        ReadSeek::seek(self.0, SeekFrom::Start(offset))?;
        self.0.write_all(data)
    }
    fn flush(&mut self) -> Result<()> {
        Sink::flush(self.0)
    }
}

fn read_at<I: ReadSeek, const N: usize>(io: &mut I, offset: u64) -> Result<[u8; N]> {
    io.seek(SeekFrom::Start(offset))?;
    let mut buf = [0u8; N];
    io.read_exact(&mut buf)?;
    Ok(buf)
}

fn read_u64_at<I: ReadSeek>(io: &mut I, offset: u64) -> Result<u64> {
    Ok(u64::from_le_bytes(read_at::<I, 8>(io, offset)?))
}

fn write_u64_at<I: RecoveryIo>(io: &mut I, offset: u64, val: u64) -> Result<()> {
    io.write_bytes_at(offset, &val.to_le_bytes())
}

/// Sums uncompressed payload bytes across a `DT`/`SD`, `DL`→(`DT`/`SD`/`DZ`), or `HL`→`DL`→`DZ` chain,
/// clamping the last `DT`/`SD` block's `hdr_len` in place if it exceeds `file_len`.
fn sum_chain_payload_bytes<I: RecoveryIo>(
    io: &mut I,
    data_pos: u64,
    file_len: u64,
    rec_align: u64,
    unfin_flags: u16,
    recovered_dls: &mut Vec<(u64, Vec<i64>)>,
) -> Result<u64> {
    if data_pos == 0 || data_pos + 24 > file_len {
        return Ok(0);
    }
    let hdr_bytes = read_at::<I, 24>(io, data_pos)?;
    let hdr: Blockheader4 = Cursor::new(hdr_bytes).read_le()?;

    match &hdr.hdr_id {
        b"##DT" | b"##SD" | b"##VD" | b"##DV" | b"##RD" => {
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
                if io.can_write() {
                    write_u64_at(io, data_pos + 8, blk_len)?;
                }
            }
            Ok(blk_len.saturating_sub(24))
        }
        b"##DZ" => {
            if data_pos + 48 > file_len {
                return Ok(0);
            }
            let dz_bytes = read_at::<I, 44>(io, data_pos + 4)?;
            let dz: Dz4Block = Cursor::new(dz_bytes).read_le()?;
            Ok(dz.dz_org_data_length)
        }
        b"##HL" => {
            if data_pos + 40 > file_len {
                return Ok(0);
            }
            let hl_bytes = read_at::<I, 36>(io, data_pos + 4)?;
            let hl: Hl4Block = Cursor::new(hl_bytes).read_le()?;
            if hl.hl_dl_first <= 0 {
                Ok(0)
            } else {
                sum_dl_chain(
                    io,
                    hl.hl_dl_first as u64,
                    file_len,
                    rec_align,
                    unfin_flags,
                    recovered_dls,
                )
            }
        }
        b"##DL" => sum_dl_chain(
            io,
            data_pos,
            file_len,
            rec_align,
            unfin_flags,
            recovered_dls,
        ),
        _ => Ok(0),
    }
}

/// Finds the start of the first data block following an empty `DLBLOCK` (`dl_count == 0`).
fn find_empty_dl_slot_end<I: RecoveryIo>(
    io: &mut I,
    dl_pos: u64,
    hdr_len: u64,
    file_len: u64,
) -> Option<u64> {
    if hdr_len > 40 {
        let candidate = dl_pos + hdr_len;
        return (candidate + 24 <= file_len).then_some(candidate);
    }
    let start = dl_pos + 40;
    let max_scan = file_len.min(start + 65536);
    let mut pos = start;
    while pos + 24 <= max_scan {
        let Ok(tag) = read_at::<I, 4>(io, pos) else {
            break;
        };
        if &tag == b"##DT"
            || &tag == b"##DZ"
            || &tag == b"##SD"
            || &tag == b"##VD"
            || &tag == b"##DV"
            || &tag == b"##RD"
        {
            return Some(pos);
        }
        pos += 8;
    }
    None
}

fn recover_dl_block<I: RecoveryIo>(
    io: &mut I,
    dl_pos: u64,
    file_len: u64,
    rec_align: u64,
    unfin_flags: u16,
    cum_offset: &mut u64,
    recovered_dls: &mut Vec<(u64, Vec<i64>)>,
) -> Result<Dl4Block> {
    let hdr_bytes = read_at::<I, 24>(io, dl_pos)?;
    let hdr: Blockheader4 = Cursor::new(hdr_bytes).read_le()?;
    if &hdr.hdr_id != b"##DL" || hdr.hdr_len < 40 || dl_pos + hdr.hdr_len > file_len {
        return Err(Error::InvalidBlock(alloc::string::String::from(
            "invalid DL block",
        )));
    }
    io.seek(SeekFrom::Start(dl_pos + 4))?;
    let mut dl_buf = alloc::vec![0u8; (hdr.hdr_len as usize).saturating_sub(4)];
    io.read_exact(&mut dl_buf)?;
    let mut dl: Dl4Block = Cursor::new(dl_buf).read_le()?;

    let mut dl_modified = false;
    let mut reserved_slot_len = hdr.hdr_len;

    // Forward-scan recovery for unpatched DLBLOCK slot (`dl_count == 0`)
    if dl.dl_count == 0 {
        if let Some(first_child_pos) = find_empty_dl_slot_end(io, dl_pos, hdr.hdr_len, file_len) {
            reserved_slot_len = first_child_pos - dl_pos;
            let max_entries = ((reserved_slot_len.saturating_sub(40)) / 16) as usize;
            let mut scan_pos = first_child_pos;
            while scan_pos + 24 <= file_len && dl.dl_data.len() < max_entries {
                let child_hdr_bytes = read_at::<I, 24>(io, scan_pos)?;
                let child_hdr: Blockheader4 = Cursor::new(child_hdr_bytes).read_le()?;
                match &child_hdr.hdr_id {
                    b"##DT" | b"##SD" | b"##VD" | b"##DV" | b"##RD" => {
                        let max_avail = file_len.saturating_sub(scan_pos);
                        let mut blk_len = child_hdr.hdr_len;
                        if blk_len > max_avail || blk_len < 24 {
                            let payload = max_avail.saturating_sub(24);
                            let aligned_payload = if rec_align > 1 {
                                (payload / rec_align) * rec_align
                            } else {
                                payload
                            };
                            blk_len = 24 + aligned_payload;
                            if io.can_write() {
                                write_u64_at(io, scan_pos + 8, blk_len)?;
                            }
                        }
                        let uncompressed_len = blk_len.saturating_sub(24);
                        dl.dl_data.push(scan_pos as i64);
                        dl.dl_offset.push(*cum_offset);
                        *cum_offset += uncompressed_len;
                        scan_pos += align8(blk_len);
                    }
                    b"##DZ" => {
                        if scan_pos + 48 > file_len {
                            break;
                        }
                        let dz_bytes = read_at::<I, 44>(io, scan_pos + 4)?;
                        let dz: Dz4Block = Cursor::new(dz_bytes).read_le()?;
                        let blk_len = 48 + dz.dz_data_length;
                        if scan_pos + blk_len > file_len {
                            break;
                        }
                        dl.dl_data.push(scan_pos as i64);
                        dl.dl_offset.push(*cum_offset);
                        *cum_offset += dz.dz_org_data_length;
                        scan_pos += align8(blk_len);
                    }
                    _ => break,
                }
            }
            if !dl.dl_data.is_empty() {
                dl.dl_count = dl.dl_data.len() as u32;
                dl.dl_links = 1 + dl.dl_count as u64;
                dl.dl_len = 24 + 8 + (dl.dl_count as u64) * 8 + 8 + (dl.dl_count as u64) * 8;
                dl.dl_flags = 0;
                dl_modified = true;
                recovered_dls.push((dl_pos, dl.dl_data.clone()));
            }
        }
    } else if (dl.dl_flags & 1) == 0 {
        // Reconstruct and patch cumulative `dl_offset` entries across chained DLBLOCKs
        let needs_offset_check =
            (unfin_flags & (UNFIN_LAST_DL_BLOCK | UNFIN_VLSD_OFFSET | UNFIN_VLSC_OFFSET)) != 0
                || dl.dl_offset.len() != dl.dl_count as usize
                || (dl.dl_count > 1 && dl.dl_offset.iter().all(|&o| o == 0))
                || (*cum_offset > 0 && dl.dl_offset.first().copied() == Some(0));

        let mut expected_offsets = Vec::with_capacity(dl.dl_data.len());
        for &child in &dl.dl_data {
            expected_offsets.push(*cum_offset);
            if child > 0 && (child as u64) + 24 <= file_len {
                let child_pos = child as u64;
                let ch_bytes = read_at::<I, 24>(io, child_pos)?;
                let ch_hdr: Blockheader4 = Cursor::new(ch_bytes).read_le()?;
                match &ch_hdr.hdr_id {
                    b"##DZ" => {
                        if child_pos + 48 <= file_len {
                            let dz_bytes = read_at::<I, 44>(io, child_pos + 4)?;
                            if let Ok(dz) = Cursor::new(dz_bytes).read_le::<Dz4Block>() {
                                *cum_offset += dz.dz_org_data_length;
                            }
                        }
                    }
                    b"##DT" | b"##SD" | b"##VD" | b"##DV" | b"##RD" => {
                        let avail = file_len.saturating_sub(child_pos);
                        let blk_len = ch_hdr.hdr_len.min(avail);
                        *cum_offset += blk_len.saturating_sub(24);
                    }
                    _ => {}
                }
            }
        }
        if needs_offset_check && dl.dl_offset != expected_offsets {
            dl.dl_offset = expected_offsets;
            dl.dl_count = dl.dl_data.len() as u32;
            dl.dl_len = 24 + 8 + (dl.dl_count as u64) * 8 + 8 + (dl.dl_count as u64) * 8;
            dl.dl_links = 1 + dl.dl_count as u64;
            dl_modified = true;
        }
    }

    if dl_modified && io.can_write() {
        let fits_slot = dl.dl_len <= reserved_slot_len
            || dl
                .dl_data
                .first()
                .is_some_and(|&first| first > dl_pos as i64 && dl_pos + dl.dl_len <= first as u64);
        if fits_slot {
            let mut dl_out = alloc::vec![0u8; dl.dl_len as usize];
            dl_out[..4].copy_from_slice(b"##DL");
            dl.write_le(&mut Cursor::new(&mut dl_out[4..]))?;
            io.write_bytes_at(dl_pos, &dl_out)?;
        }
    }

    Ok(dl)
}

fn sum_dl_chain<I: RecoveryIo>(
    io: &mut I,
    first_dl_pos: u64,
    file_len: u64,
    rec_align: u64,
    unfin_flags: u16,
    recovered_dls: &mut Vec<(u64, Vec<i64>)>,
) -> Result<u64> {
    let mut total = 0u64;
    let mut dl_pos = first_dl_pos;
    let mut cum_offset = 0u64;
    let mut guard = 0usize;
    while dl_pos > 0 && dl_pos + 24 <= file_len && guard < 65536 {
        guard += 1;
        let dl = match recover_dl_block(
            io,
            dl_pos,
            file_len,
            rec_align,
            unfin_flags,
            &mut cum_offset,
            recovered_dls,
        ) {
            Ok(dl) => dl,
            Err(_) => break,
        };
        for &child in &dl.dl_data {
            if child > 0 {
                total += sum_chain_payload_bytes(
                    io,
                    child as u64,
                    file_len,
                    rec_align,
                    unfin_flags,
                    recovered_dls,
                )?;
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

struct CgRecEntry {
    rec_id: u64,
    rec_len: u64,
    is_vlsd: bool,
    cycle_offset: u64,
    cycle_count: u64,
    cg_pos: u64,
    vlsd_bytes: u64,
}

fn parse_unsorted_carry(carry: &mut Vec<u8>, rec_id_size: usize, cg_entries: &mut [CgRecEntry]) {
    let mut pos = 0usize;
    while pos + rec_id_size <= carry.len() {
        let rec_id = match rec_id_size {
            1 => carry[pos] as u64,
            2 => u16::from_le_bytes(carry[pos..pos + 2].try_into().unwrap()) as u64,
            4 => u32::from_le_bytes(carry[pos..pos + 4].try_into().unwrap()) as u64,
            8 => u64::from_le_bytes(carry[pos..pos + 8].try_into().unwrap()),
            _ => break,
        };
        let Some(entry) = cg_entries.iter_mut().find(|e| e.rec_id == rec_id) else {
            carry.clear();
            return;
        };
        if entry.is_vlsd {
            if pos + rec_id_size + 4 > carry.len() {
                break;
            }
            let vlen = u32::from_le_bytes(
                carry[pos + rec_id_size..pos + rec_id_size + 4]
                    .try_into()
                    .unwrap(),
            ) as usize;
            if pos + rec_id_size + 4 + vlen > carry.len() {
                break;
            }
            entry.cycle_count += 1;
            entry.vlsd_bytes += vlen as u64;
            pos += rec_id_size + 4 + vlen;
        } else {
            let full_len = rec_id_size + entry.rec_len as usize;
            if pos + full_len > carry.len() {
                break;
            }
            entry.cycle_count += 1;
            pos += full_len;
        }
    }
    if pos > 0 {
        carry.drain(..pos);
    }
}

fn scan_unsorted_data_block<I: RecoveryIo>(
    io: &mut I,
    blk_pos: u64,
    file_len: u64,
    rec_id_size: usize,
    cg_entries: &mut [CgRecEntry],
    carry: &mut Vec<u8>,
) -> Result<()> {
    if blk_pos == 0 || blk_pos + 24 > file_len {
        return Ok(());
    }
    let hdr_bytes = read_at::<I, 24>(io, blk_pos)?;
    let hdr: Blockheader4 = Cursor::new(hdr_bytes).read_le()?;

    match &hdr.hdr_id {
        b"##DT" | b"##SD" | b"##VD" | b"##DV" | b"##RD" => {
            let max_avail = file_len.saturating_sub(blk_pos);
            let mut remaining = if hdr.hdr_len >= 24 && hdr.hdr_len <= max_avail {
                hdr.hdr_len - 24
            } else {
                max_avail.saturating_sub(24)
            };
            io.seek(SeekFrom::Start(blk_pos + 24))?;
            let mut chunk = [0u8; 4096];
            while remaining > 0 {
                let to_read = (remaining as usize).min(chunk.len());
                io.read_exact(&mut chunk[..to_read])?;
                remaining -= to_read as u64;
                carry.extend_from_slice(&chunk[..to_read]);
                parse_unsorted_carry(carry, rec_id_size, cg_entries);
            }
        }
        b"##DZ" => {
            if blk_pos + 48 > file_len {
                return Ok(());
            }
            let dz_bytes = read_at::<I, 44>(io, blk_pos + 4)?;
            let dz: Dz4Block = Cursor::new(dz_bytes).read_le()?;
            let avail = file_len.saturating_sub(blk_pos + 48);
            if dz.dz_data_length > avail {
                return Ok(());
            }
            io.seek(SeekFrom::Start(blk_pos + 48))?;
            let mut comp = alloc::vec![0u8; dz.dz_data_length as usize];
            io.read_exact(&mut comp)?;
            let decomp = crate::compress::decompress_dz_data(
                dz.dz_zip_type,
                dz.dz_zip_parameter,
                &comp,
                dz.dz_org_data_length,
            )?;
            carry.extend_from_slice(&decomp);
            parse_unsorted_carry(carry, rec_id_size, cg_entries);
        }
        _ => {}
    }
    Ok(())
}

struct UnsortedScanCtx<'a> {
    file_len: u64,
    rec_id_size: usize,
    cg_entries: &'a mut [CgRecEntry],
    unfin_flags: u16,
    recovered_dls: &'a mut Vec<(u64, Vec<i64>)>,
    carry: &'a mut Vec<u8>,
}

fn scan_unsorted_chain<I: RecoveryIo>(
    io: &mut I,
    data_pos: u64,
    ctx: &mut UnsortedScanCtx<'_>,
) -> Result<()> {
    if data_pos == 0 || data_pos + 24 > ctx.file_len {
        return Ok(());
    }
    let hdr_bytes = read_at::<I, 24>(io, data_pos)?;
    let hdr: Blockheader4 = Cursor::new(hdr_bytes).read_le()?;

    match &hdr.hdr_id {
        b"##DT" | b"##SD" | b"##VD" | b"##DV" | b"##RD" | b"##DZ" => {
            scan_unsorted_data_block(
                io,
                data_pos,
                ctx.file_len,
                ctx.rec_id_size,
                ctx.cg_entries,
                ctx.carry,
            )?;
        }
        b"##HL" => {
            if data_pos + 40 > ctx.file_len {
                return Ok(());
            }
            let hl_bytes = read_at::<I, 36>(io, data_pos + 4)?;
            let hl: Hl4Block = Cursor::new(hl_bytes).read_le()?;
            if hl.hl_dl_first > 0 {
                scan_unsorted_chain(io, hl.hl_dl_first as u64, ctx)?;
            }
        }
        b"##DL" => {
            let mut dl_pos = data_pos;
            let mut cum_offset = 0u64;
            let mut guard = 0usize;
            while dl_pos > 0 && dl_pos + 24 <= ctx.file_len && guard < 65536 {
                guard += 1;
                let dl = match recover_dl_block(
                    io,
                    dl_pos,
                    ctx.file_len,
                    1,
                    ctx.unfin_flags,
                    &mut cum_offset,
                    ctx.recovered_dls,
                ) {
                    Ok(dl) => dl,
                    Err(_) => break,
                };
                for &ptr in &dl.dl_data {
                    if ptr > 0 && (ptr as u64) + 24 <= ctx.file_len {
                        let _ = scan_unsorted_data_block(
                            io,
                            ptr as u64,
                            ctx.file_len,
                            ctx.rec_id_size,
                            ctx.cg_entries,
                            ctx.carry,
                        );
                    }
                }
                dl_pos = if dl.dl_dl_next > 0 {
                    dl.dl_dl_next as u64
                } else {
                    0
                };
            }
        }
        _ => {}
    }
    Ok(())
}

fn run_recovery<I: RecoveryIo>(io: &mut I) -> Result<(usize, UnfinalizedScanResult)> {
    let file_len = io.total_len()?;
    if file_len < 168 {
        return Err(Error::Io);
    }
    let id_bytes = read_at::<I, 64>(io, 0)?;
    let mut id: IdBlock = Cursor::new(id_bytes).read_le()?;
    if id.id_file_id != UNFINALIZED_ID && id.id_unfin_flags == 0 {
        return Ok((0, UnfinalizedScanResult::default()));
    }
    let unfin_flags = id.id_unfin_flags;

    let hd_bytes = read_at::<I, 104>(io, 64)?;
    let hd: Hd4 = Cursor::new(hd_bytes).read_le()?;

    let mut result = UnfinalizedScanResult::default();
    let mut patched_cgs = 0usize;
    let mut dg_pos = hd.hd_dg_first;
    let mut dg_guard = 0usize;

    while dg_pos > 0 && (dg_pos as u64) + 64 <= file_len && dg_guard < 4096 {
        dg_guard += 1;
        let dg_bytes = read_at::<I, 64>(io, dg_pos as u64)?;
        let dg: Dg4Block = Cursor::new(dg_bytes).read_le()?;

        if dg.dg_rec_id_size > 0 && dg.dg_data > 0 {
            let mut cg_entries = Vec::new();
            let mut cg_pos = dg.dg_cg_first;
            let mut cg_guard = 0usize;
            while cg_pos > 0 && (cg_pos as u64) + 104 <= file_len && cg_guard < 4096 {
                cg_guard += 1;
                let cg_short_bytes = read_at::<I, 16>(io, cg_pos as u64)?;
                let cg_short: Blockheader4Short = Cursor::new(cg_short_bytes).read_le()?;
                let cg_body_len = (cg_short.hdr_len as usize).saturating_sub(16);
                io.seek(SeekFrom::Start((cg_pos as u64) + 16))?;
                let mut cg_buf = alloc::vec![0u8; cg_body_len];
                io.read_exact(&mut cg_buf)?;
                let cg: Cg4Block = Cursor::new(cg_buf).read_le()?;
                let is_vlsd = (cg.cg_flags & (CG_F_VLSD | CG_F_VLSC)) != 0;
                let cycle_offset = (cg_pos as u64) + 16 + 8 + cg.cg_links * 8 + 8;
                cg_entries.push(CgRecEntry {
                    rec_id: cg.cg_record_id,
                    rec_len: (cg.cg_data_bytes as u64) + (cg.cg_inval_bytes as u64),
                    is_vlsd,
                    cycle_offset,
                    cycle_count: 0,
                    cg_pos: cg_pos as u64,
                    vlsd_bytes: 0,
                });
                cg_pos = cg.cg_cg_next;
            }

            let mut carry = Vec::new();
            let mut ctx = UnsortedScanCtx {
                file_len,
                rec_id_size: dg.dg_rec_id_size as usize,
                cg_entries: &mut cg_entries,
                unfin_flags,
                recovered_dls: &mut result.recovered_dl_data,
                carry: &mut carry,
            };
            let _ = scan_unsorted_chain(io, dg.dg_data as u64, &mut ctx);

            for entry in cg_entries {
                result.cg_cycles.push((entry.cg_pos, entry.cycle_count));
                if entry.is_vlsd {
                    result.vlsd_bytes.push((entry.cg_pos, entry.vlsd_bytes));
                }
                if io.can_write() {
                    if read_u64_at(io, entry.cycle_offset)? != entry.cycle_count {
                        write_u64_at(io, entry.cycle_offset, entry.cycle_count)?;
                    }
                    if entry.is_vlsd {
                        let mut bytes_buf = [0u8; 8];
                        bytes_buf[0..4].copy_from_slice(
                            &((entry.vlsd_bytes & 0xFFFF_FFFF) as u32).to_le_bytes(),
                        );
                        bytes_buf[4..8]
                            .copy_from_slice(&((entry.vlsd_bytes >> 32) as u32).to_le_bytes());
                        io.write_bytes_at(entry.cycle_offset + 16, &bytes_buf)?;
                    }
                }
                patched_cgs += 1;
            }
        } else {
            let mut cg_pos = dg.dg_cg_first;
            let mut cg_guard = 0usize;
            while cg_pos > 0 && (cg_pos as u64) + 104 <= file_len && cg_guard < 4096 {
                cg_guard += 1;
                let short_bytes = read_at::<I, 16>(io, cg_pos as u64)?;
                let short_hdr: Blockheader4Short = Cursor::new(short_bytes).read_le()?;
                let body_len = (short_hdr.hdr_len as usize).saturating_sub(16);
                io.seek(SeekFrom::Start((cg_pos as u64) + 16))?;
                let mut cg_buf = alloc::vec![0u8; body_len];
                io.read_exact(&mut cg_buf)?;
                let cg: Cg4Block = Cursor::new(cg_buf).read_le()?;

                let is_vlsd_cg = (cg.cg_flags & (CG_F_VLSD | CG_F_VLSC)) != 0;

                let mut cn_pos = cg.cg_cn_first;
                let mut cn_guard = 0usize;
                while cn_pos > 0 && (cn_pos as u64) + 160 <= file_len && cn_guard < 4096 {
                    cn_guard += 1;
                    let cn_short_bytes = read_at::<I, 16>(io, cn_pos as u64)?;
                    let cn_short: Blockheader4Short = Cursor::new(cn_short_bytes).read_le()?;
                    let cn_body_len = (cn_short.hdr_len as usize).saturating_sub(16);
                    io.seek(SeekFrom::Start((cn_pos as u64) + 16))?;
                    let mut cn_buf = alloc::vec![0u8; cn_body_len];
                    io.read_exact(&mut cn_buf)?;
                    let cn: Cn4Block = Cursor::new(cn_buf).read_le()?;
                    if (cn.cn_type == 1 || cn.cn_type == 7)
                        && cn.cn_data > 0
                        && dg.dg_rec_id_size == 0
                    {
                        let _ = sum_chain_payload_bytes(
                            io,
                            cn.cn_data as u64,
                            file_len,
                            1,
                            unfin_flags,
                            &mut result.recovered_dl_data,
                        )?;
                    }
                    cn_pos = cn.cn_cn_next;
                }

                if dg.dg_rec_id_size == 0 && !is_vlsd_cg {
                    let rec_len = (cg.cg_data_bytes as u64) + (cg.cg_inval_bytes as u64);
                    if rec_len > 0 && dg.dg_data > 0 {
                        let total_bytes = sum_chain_payload_bytes(
                            io,
                            dg.dg_data as u64,
                            file_len,
                            rec_len,
                            unfin_flags,
                            &mut result.recovered_dl_data,
                        )?;
                        let cycles = total_bytes / rec_len;
                        result.cg_cycles.push((cg_pos as u64, cycles));
                        if io.can_write() {
                            let cycle_offset = (cg_pos as u64) + 16 + 8 + cg.cg_links * 8 + 8;
                            if read_u64_at(io, cycle_offset)? != cycles {
                                write_u64_at(io, cycle_offset, cycles)?;
                            }
                        }
                        patched_cgs += 1;
                    }
                } else {
                    patched_cgs += 1;
                }

                cg_pos = cg.cg_cg_next;
            }
        }

        dg_pos = dg.dg_dg_next;
    }

    if io.can_write() {
        id.id_file_id = FINALIZED_ID;
        id.id_unfin_flags = 0;
        id.id_custom_unfin_flags = 0;
        let mut id_out = Vec::with_capacity(64);
        id.write_le(&mut Cursor::new(&mut id_out))?;
        io.write_bytes_at(0, &id_out)?;
        io.flush()?;
    }

    Ok((patched_cgs, result))
}

/// Scans an unfinalized (`"UnFinMF "`) MDF 4.x stream in read-only mode, returning recovered
/// cycle counts, VLSD byte counts, and forward-scanned `DLBLOCK` data pointers.
pub fn scan_unfinalized_counts<R: ReadSeek>(rdr: &mut R) -> Result<UnfinalizedScanResult> {
    let mut io = ReadOnlyIo(rdr);
    let (_, result) = run_recovery(&mut io)?;
    Ok(result)
}

/// Recovers and finalizes an unfinalized (`"UnFinMF "`) MDF 4.x stream in place on `sink`.
/// Returns the number of channel groups patched (`0` if the stream was already finalized).
pub fn recover_in_place<S: ReadWriteSeek>(sink: &mut S) -> Result<usize> {
    let mut io = ReadWriteIo(sink);
    let (patched, _) = run_recovery(&mut io)?;
    Ok(patched)
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

    #[test]
    fn no_std_recover_in_place_unsorted() {
        let mut b = LoggerBuilder::new(DataLayout::Unsorted, CompressionAlgorithm::NoCompression);
        let g1 = b.add_group("G1");
        let ch1 = b.add_channel(g1, "V1", CnDataType::UIntLE, 16).unwrap();
        let g2 = b.add_group("G2");
        let ch2 = b.add_channel(g2, "V2", CnDataType::UIntLE, 32).unwrap();
        let schema = b.build().unwrap();

        let mut sink = VecSink::default();
        let mut logger = Logger::start(schema, sink, ManualClock::new(0.01, 1)).unwrap();
        for i in 0..5 {
            logger
                .begin_record(g1)
                .unwrap()
                .set_u64(ch1, i)
                .unwrap()
                .commit()
                .unwrap();
        }
        for i in 0..7 {
            logger
                .begin_record(g2)
                .unwrap()
                .set_u64(ch2, i)
                .unwrap()
                .commit()
                .unwrap();
        }
        logger.checkpoint().unwrap();
        sink = logger.into_sink_unfinalized();
        assert_eq!(&sink.as_slice()[0..8], b"UnFinMF ");

        let patched = recover_in_place(&mut sink).unwrap();
        assert_eq!(patched, 2);
        assert_eq!(&sink.as_slice()[0..8], b"MDF     ");
    }

    #[test]
    fn test_recover_vlsd_unsorted_records() {
        let mut b = LoggerBuilder::new(DataLayout::Unsorted, CompressionAlgorithm::NoCompression);
        let g = b.add_group("G");
        let ch_vlsd = b.add_vlsd_channel(g, "V", CnDataType::ByteArray).unwrap();
        let schema = b.build().unwrap();

        let mut sink = VecSink::default();
        let mut logger = Logger::start(schema, sink, ManualClock::new(0.01, 1)).unwrap();
        for i in 0..5 {
            let data = alloc::format!("payload_{i}");
            logger
                .begin_record(g)
                .unwrap()
                .set_bytes(ch_vlsd, data.as_bytes())
                .unwrap()
                .commit()
                .unwrap();
        }
        logger.checkpoint().unwrap();
        sink = logger.into_sink_unfinalized();
        assert_eq!(&sink.as_slice()[0..8], b"UnFinMF ");

        let patched = recover_in_place(&mut sink).unwrap();
        assert!(patched > 0);
        assert_eq!(&sink.as_slice()[0..8], b"MDF     ");
    }

    #[test]
    fn test_recover_unpatched_dl_block_forward_scan() {
        let mut sink = VecSink::new();
        // IDBLOCK
        let id = IdBlock {
            id_unfin_flags: UNFIN_LAST_DL_BLOCK,
            ..Default::default()
        };
        let mut id_bytes = alloc::vec![];
        id.write_le(&mut Cursor::new(&mut id_bytes)).unwrap();
        sink.write_all(&id_bytes).unwrap();

        let dl_pos = sink.as_slice().len() as u64;
        // Write unpatched DLBLOCK: dl_count = 0, dl_len = 40, dl_links = 1
        sink.write_all(b"##DL").unwrap();
        let dl = Dl4Block {
            reserved: [0; 4],
            dl_len: 40,
            dl_links: 1,
            dl_dl_next: 0,
            dl_data: alloc::vec![],
            dl_flags: 0,
            dl_reserved: [0; 3],
            dl_count: 0,
            dl_equal_length: None,
            dl_offset: alloc::vec![],
            dl_time_values: alloc::vec![],
            dl_angle_values: alloc::vec![],
            dl_distance_values: alloc::vec![],
        };
        let mut dl_bytes = alloc::vec![];
        dl.write_le(&mut Cursor::new(&mut dl_bytes)).unwrap();
        sink.write_all(&dl_bytes).unwrap();

        // Pad to dl_pos + 1064
        let current_len = sink.as_slice().len() as u64;
        let padding = (dl_pos + 1064).saturating_sub(current_len);
        sink.write_all(&alloc::vec![0u8; padding as usize]).unwrap();

        // Write two DT blocks after dl_pos + 1064
        // First DT: 24 bytes header + 16 bytes payload = 40 bytes
        let dt1_pos = sink.as_slice().len() as u64;
        let dt1_hdr = Blockheader4 {
            hdr_id: *b"##DT",
            hdr_gap: [0; 4],
            hdr_len: 40,
            hdr_links: 0,
        };
        let mut dt1_hdr_bytes = alloc::vec![];
        dt1_hdr
            .write_le(&mut Cursor::new(&mut dt1_hdr_bytes))
            .unwrap();
        sink.write_all(&dt1_hdr_bytes).unwrap();
        sink.write_all(&[0xAA; 16]).unwrap();

        // Second DT: 24 bytes header + 32 bytes payload = 56 bytes
        let dt2_pos = sink.as_slice().len() as u64;
        let dt2_hdr = Blockheader4 {
            hdr_id: *b"##DT",
            hdr_gap: [0; 4],
            hdr_len: 56,
            hdr_links: 0,
        };
        let mut dt2_hdr_bytes = alloc::vec![];
        dt2_hdr
            .write_le(&mut Cursor::new(&mut dt2_hdr_bytes))
            .unwrap();
        sink.write_all(&dt2_hdr_bytes).unwrap();
        sink.write_all(&[0xBB; 32]).unwrap();

        let file_len = sink.as_slice().len() as u64;

        // Run recover_dl_block
        let mut io = ReadWriteIo(&mut sink);
        let mut cum_offset = 0u64;
        let mut recovered_dls = Vec::new();
        let recovered_dl = recover_dl_block(
            &mut io,
            dl_pos,
            file_len,
            1,
            UNFIN_LAST_DL_BLOCK,
            &mut cum_offset,
            &mut recovered_dls,
        )
        .unwrap();
        assert_eq!(recovered_dl.dl_count, 2);
        assert_eq!(recovered_dl.dl_data.len(), 2);
        assert_eq!(recovered_dl.dl_data[0], dt1_pos as i64);
        assert_eq!(recovered_dl.dl_data[1], dt2_pos as i64);
        assert_eq!(recovered_dl.dl_offset.len(), 2);
        assert_eq!(recovered_dl.dl_offset[0], 0);
        assert_eq!(recovered_dl.dl_offset[1], 16);
    }
}
