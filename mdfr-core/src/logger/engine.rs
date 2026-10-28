//! Streaming MDF 4.3 logger engine (`no_std + alloc`).
//!
//! Writes a valid MDF 4.30 file to any [`Sink`]:
//! - Starts in unfinalized state (`"UnFinMF "` with `UNFIN_*` flags set)
//! - Supports both [`DataLayout::Sorted`] (one DG per CG) and [`DataLayout::Unsorted`] (single DG with record IDs)
//! - Supports uncompressed `DT`/`SD` chains (`DLBLOCK`) and compressed `DZ` chains (`HLBLOCK` → `DLBLOCK` → `DZBLOCK`)
//!   with transposition enabled by default (`DeflateTranspose` / `Lz4Transpose` / `ZstdTranspose`)
//! - Supports fixed-length scalars, bit-packed integers, N-D arrays (`CABLOCK`), conversions (`CCBLOCK`),
//!   source info (`SIBLOCK`), and variable-length signal data (`VLSD` via `SDBLOCK` in sorted mode or companion VLSD `CGBLOCK` in unsorted mode)
//! - `checkpoint()` flushes buffers and patches `DL`/`CG` links & cycle counts in place (no copy)
//! - `close()` / `rotate()` finalizes `"MDF     "` in place and clears `id_unfin_flags`.
use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use binrw::{BinWrite, io::Cursor};

use crate::blocks::{
    ca::Ca4Block,
    cc::{Cc4Block, CcVal},
    cg::{CG_F_VLSD, Cg4Block},
    cn::{Cn4Block, CnDataType},
    data::{Dl4Block, Dz4Block, Hl4Block},
    dg::Dg4Block,
    fh::FhBlock,
    hd::{HD_END_POSITION, Hd4},
    header::{Blockheader4, Blockheader4Short},
    id::{
        IdBlock, UNFIN_CG_CYCLE_COUNTERS, UNFIN_LAST_DL_BLOCK, UNFIN_LAST_DT_LENGTH,
        UNFIN_VLSD_CG_BYTES,
    },
    si::Si4Block,
    tx::{align8, encode_md, encode_tx},
};
use crate::compress::{CompressionAlgorithm, compress_dz_data};
use crate::error::{Error, Result};
use crate::io::{SeekFrom, Sink};
use crate::time::TimeSource;

use super::schema::{
    ChannelDef, ChannelHandle, ConversionDef, DataLayout, GroupHandle, Schema, SourceDef,
};

/// State of an active `DLBLOCK` chain for a data stream (a DG's record stream or a sorted VLSD channel's SD stream).
#[derive(Debug, Clone)]
struct DlChainState {
    /// File offset of the link field that points to the head of this chain (`dg_data` or `cn_data`).
    owner_link_pos: u64,
    /// `[b'D', b'T']` for record data or `[b'S', b'D']` for sorted VLSD data.
    org_block_type: [u8; 2],
    /// Record width in bytes used as `dz_zip_parameter` when transposing (`1` for unstructured SD bytes).
    transpose_width: usize,
    /// File offset of the currently open `DLBLOCK` (0 if none allocated yet).
    current_dl_pos: u64,
    /// Data block file offsets in the currently open `DLBLOCK`.
    current_dl_blocks: Vec<i64>,
    /// Cumulative uncompressed byte offsets at the start of each block in `current_dl_blocks`.
    current_dl_offsets: Vec<u64>,
    /// Total uncompressed bytes written across all blocks in this chain so far.
    total_uncompressed_bytes: u64,
    /// Whether an `HLBLOCK` has already been written at the head of this chain (for compressed streams).
    hl_written: bool,
}

impl DlChainState {
    fn new(owner_link_pos: u64, org_block_type: [u8; 2], transpose_width: usize) -> Self {
        Self {
            owner_link_pos,
            org_block_type,
            transpose_width,
            current_dl_pos: 0,
            current_dl_blocks: Vec::new(),
            current_dl_offsets: Vec::new(),
            total_uncompressed_bytes: 0,
            hl_written: false,
        }
    }
}

/// Per-VLSD-channel runtime state.
#[derive(Debug, Clone)]
struct VlsdChannelState {
    /// Channel index within its group.
    channel_idx: usize,
    /// In sorted mode: dedicated `SDBLOCK` chain + buffer.
    /// In unsorted mode: companion VLSD `CGBLOCK` record ID + total VLSD payload bytes + file offset of VLSD `CGBLOCK`.
    sorted_sd_chain: Option<DlChainState>,
    sorted_sd_buf: Vec<u8>,
    /// Total bytes accumulated in `SDBLOCK` (sorted) or VLSD CG values (unsorted, excluding 4-byte length prefixes).
    total_vlsd_bytes: u64,
    /// Unsorted mode: record ID of the companion VLSD `CGBLOCK` (`group_idx + 1 + vlsd_ordinal`).
    unsorted_vlsd_record_id: u64,
    /// Unsorted mode: file offset of the companion VLSD `CGBLOCK` (so we can patch `cg_cycle_count`, `cg_data_bytes`, `cg_inval_bytes`).
    unsorted_vlsd_cg_pos: u64,
    /// Unsorted mode: number of VLSD samples written.
    unsorted_vlsd_cycles: u64,
}

/// Per-group runtime state.
#[derive(Debug, Clone)]
struct GroupRuntime {
    /// File offset of this group's `CGBLOCK` (short header starts at `cg_pos`).
    /// `cg_cycle_count` is at `cg_pos + 16 + 8 + 6*8 + 8 = cg_pos + 80`.
    cg_pos: u64,
    /// Record ID used in unsorted mode (`1, 2, ...`; `0` in sorted mode).
    record_id: u64,
    /// Number of records committed to this group.
    cycle_count: u64,
    /// Plain record size in bytes (`cg_data_bytes`).
    record_size: usize,
    /// Staging buffer for the next record being assembled via the dynamic `RecordWriter` API.
    staging_record: Vec<u8>,
    /// In sorted mode: per-group record buffer and `DLBLOCK` chain.
    sorted_buf: Vec<u8>,
    sorted_chain: Option<DlChainState>,
    /// VLSD channels belonging to this group.
    vlsd_channels: Vec<VlsdChannelState>,
}

/// Active MDF 4.3 streaming logger over a [`Sink`] and [`TimeSource`].
pub struct Logger<S: Sink, T: TimeSource> {
    schema: Schema,
    sink: S,
    clock: T,
    /// Current end-of-file offset where new blocks are appended.
    eof_pos: u64,
    /// Whether `hd_start_time_ns` has been written with a valid non-zero timestamp.
    hd_time_valid: bool,
    /// Per-group state.
    groups: Vec<GroupRuntime>,
    /// Unsorted mode: shared record stream buffer and `DLBLOCK` chain.
    unsorted_buf: Vec<u8>,
    unsorted_chain: Option<DlChainState>,
    /// Unsorted mode: flush threshold in bytes.
    unsorted_flush_threshold: usize,
    /// Number of data blocks flushed since the last `checkpoint()`.
    blocks_since_checkpoint: usize,
    /// Sequence index incremented across `rotate()` calls.
    sequence_index: u64,
}

impl<S: Sink, T: TimeSource> Logger<S, T> {
    /// Initializes a new MDF 4.3 file on `sink` and writes all metadata blocks (`ID`, `HD`, `FH`, `DG`, `CG`, `CN`, `CC`, `SI`, `CA`, `TX`, `MD`).
    pub fn start(schema: Schema, mut sink: S, mut clock: T) -> Result<Self> {
        let (groups, unsorted_chain, eof_pos, hd_time_valid) =
            Self::write_metadata(&schema, &mut sink, &mut clock, 0)?;
        let unsorted_flush_threshold = schema
            .groups
            .first()
            .map(|g| g.flush_threshold_bytes)
            .unwrap_or(32 * 1024);
        Ok(Self {
            schema,
            sink,
            clock,
            eof_pos,
            hd_time_valid,
            groups,
            unsorted_buf: Vec::new(),
            unsorted_chain,
            unsorted_flush_threshold,
            blocks_since_checkpoint: 0,
            sequence_index: 0,
        })
    }

    fn write_metadata(
        schema: &Schema,
        sink: &mut S,
        clock: &mut T,
        seq_index: u64,
    ) -> Result<(Vec<GroupRuntime>, Option<DlChainState>, u64, bool)> {
        sink.seek(SeekFrom::Start(0))?;

        // 1. IDBlock (64 bytes at offset 0) — unfinalized
        let mut unfin_flags = UNFIN_CG_CYCLE_COUNTERS | UNFIN_LAST_DT_LENGTH | UNFIN_LAST_DL_BLOCK;
        if schema
            .groups
            .iter()
            .any(|g| g.channels.iter().any(|c| c.is_vlsd))
        {
            unfin_flags |= UNFIN_VLSD_CG_BYTES;
        }
        let id = IdBlock::unfinalized(unfin_flags);
        write_binrw(sink, &id)?;

        // 2. Prepare HD MD comment (written immediately after HD at offset 168 so hd_md_comment is known when HD is read)
        let wall_ns = clock.wall_clock_ns().unwrap_or(0);
        let hd_md_pos = HD_END_POSITION as u64;
        let hd_xml = format!(
            "<HDcomment><TX>MDF 4.3 embedded recording</TX><common_properties><tree name=\"Recorder\"><e name=\"FileIndex\">{seq_index}</e><e name=\"SequenceIndex\">{seq_index}</e></tree></common_properties></HDcomment>"
        );
        let hd_md = encode_md(&hd_xml)?;
        let fh_pos = hd_md_pos + hd_md.len() as u64;

        let mut hd = Hd4::new_at(wall_ns);
        hd.hd_fh_first = fh_pos as i64;
        hd.hd_md_comment = hd_md_pos as i64;
        write_binrw(sink, &hd)?;
        sink.write_all(&hd_md)?;

        // 3. FhBlock (56 bytes at fh_pos) + MD comment
        let fh_comment_pos = fh_pos + 56;
        let fh = FhBlock {
            fh_fh_next: 0,
            fh_md_comment: fh_comment_pos as i64,
            fh_time_ns: wall_ns,
            ..Default::default()
        };
        write_binrw(sink, &fh)?;
        let fh_xml = "<FHcomment><TX>Created by mdfr-core embedded MDF 4.3 logger</TX><tool_id>mdfr-core</tool_id><tool_vendor>mdfr</tool_vendor><tool_version>0.8.0</tool_version></FHcomment>";
        let fh_md = encode_md(fh_xml)?;
        sink.write_all(&fh_md)?;
        let mut pos = fh_comment_pos + fh_md.len() as u64;

        let sec_per_tick = clock.seconds_per_tick();
        let mut group_runtimes = Vec::with_capacity(schema.groups.len());
        let mut unsorted_chain = None;

        let first_dg_pos: i64;

        match schema.layout {
            DataLayout::Sorted => {
                first_dg_pos = if schema.groups.is_empty() {
                    0
                } else {
                    pos as i64
                };
                let n_groups = schema.groups.len();
                let mut prev_dg_pos: Option<u64> = None;

                for (g_idx, g_def) in schema.groups.iter().enumerate() {
                    let dg_pos = pos;
                    if let Some(prev) = prev_dg_pos {
                        // Patch previous DG's dg_dg_next (at offset prev + 24)
                        patch_i64(sink, prev + 24, dg_pos as i64, pos)?;
                    }
                    prev_dg_pos = Some(dg_pos);

                    let cg_pos = dg_pos + 64;
                    let dg = Dg4Block {
                        dg_dg_next: 0,
                        dg_cg_first: cg_pos as i64,
                        dg_data: 0,
                        dg_rec_id_size: 0,
                        ..Default::default()
                    };
                    write_binrw(sink, &dg)?;
                    pos += 64;

                    let (g_rt, new_pos) = Self::write_group_hierarchy(
                        sink,
                        pos,
                        g_idx,
                        0, // sorted record_id = 0
                        g_def,
                        sec_per_tick,
                        schema.compression,
                        Some(dg_pos + 40), // dg_data link is at dg_pos + 24 (hdr) + 16 (dg_dg_next, dg_cg_first) = dg_pos + 40
                    )?;
                    pos = new_pos;
                    let _ = n_groups;
                    group_runtimes.push(g_rt);
                }
            }
            DataLayout::Unsorted => {
                let dg_pos = pos;
                first_dg_pos = dg_pos as i64;
                let cg_first_pos = dg_pos + 64;
                // Use 1-byte record IDs (supports up to 255 CGs including VLSD companion CGs)
                let dg = Dg4Block {
                    dg_dg_next: 0,
                    dg_cg_first: cg_first_pos as i64,
                    dg_data: 0,
                    dg_rec_id_size: 1,
                    ..Default::default()
                };
                write_binrw(sink, &dg)?;
                pos += 64;

                unsorted_chain = Some(DlChainState::new(
                    dg_pos + 40,
                    *b"DT",
                    1, // unsorted stream mixes record sizes, so transpose width = 1 unless single-group
                ));

                // Assign record IDs starting at 1; companion VLSD CGs get subsequent IDs
                let mut next_rec_id: u64 = 1;
                let mut prev_cg_next_link_pos: Option<u64> = None;

                for (g_idx, g_def) in schema.groups.iter().enumerate() {
                    let rec_id = next_rec_id;
                    next_rec_id += 1;

                    if let Some(link_pos) = prev_cg_next_link_pos {
                        patch_i64(sink, link_pos, pos as i64, pos)?;
                    }

                    let cg_pos = pos;
                    let (mut g_rt, new_pos) = Self::write_group_hierarchy(
                        sink,
                        pos,
                        g_idx,
                        rec_id,
                        g_def,
                        sec_per_tick,
                        schema.compression,
                        None,
                    )?;
                    pos = new_pos;
                    // cg_cg_next link is at cg_pos + 16 (short hdr) + 8 (cg_links) = cg_pos + 24
                    prev_cg_next_link_pos = Some(cg_pos + 24);

                    // For each VLSD channel in this group, emit a companion VLSD CGBLOCK in the same DG
                    // and patch the channel's `cn_data` link to point to that VLSD CGBLOCK.
                    for vlsd_state in &mut g_rt.vlsd_channels {
                        let vlsd_cg_pos = pos;
                        if let Some(link_pos) = prev_cg_next_link_pos {
                            patch_i64(sink, link_pos, vlsd_cg_pos as i64, pos)?;
                        }
                        let vlsd_rec_id = next_rec_id;
                        next_rec_id += 1;

                        let vlsd_cg_hdr = Blockheader4Short {
                            hdr_id: *b"##CG",
                            hdr_gap: [0; 4],
                            hdr_len: 104,
                        };
                        let vlsd_cg = Cg4Block {
                            cg_links: 6,
                            cg_cg_next: 0,
                            cg_cn_first: 0,
                            cg_tx_acq_name: 0,
                            cg_si_acq_source: 0,
                            cg_sr_first: 0,
                            cg_md_comment: 0,
                            cg_cg_master: None,
                            cg_record_id: vlsd_rec_id,
                            cg_cycle_count: 0,
                            cg_flags: CG_F_VLSD,
                            cg_path_separator: 0,
                            cg_reserved: [0; 4],
                            cg_data_bytes: 0,
                            cg_inval_bytes: 0,
                        };
                        write_binrw(sink, &vlsd_cg_hdr)?;
                        write_binrw(sink, &vlsd_cg)?;
                        pos += 104;
                        prev_cg_next_link_pos = Some(vlsd_cg_pos + 24);

                        // Patch the VLSD channel's `cn_data` link to point to `vlsd_cg_pos`
                        if let Some(chain) = vlsd_state.sorted_sd_chain.take() {
                            patch_i64(sink, chain.owner_link_pos, vlsd_cg_pos as i64, pos)?;
                        }
                        vlsd_state.unsorted_vlsd_record_id = vlsd_rec_id;
                        vlsd_state.unsorted_vlsd_cg_pos = vlsd_cg_pos;
                    }

                    group_runtimes.push(g_rt);
                }
            }
        }

        // Patch HD's hd_dg_first (at offset 64 + 24 = 88)
        patch_i64(sink, 88, first_dg_pos, pos)?;

        Ok((group_runtimes, unsorted_chain, pos, wall_ns != 0))
    }

    #[allow(clippy::too_many_arguments)]
    fn write_group_hierarchy(
        sink: &mut S,
        mut pos: u64,
        _g_idx: usize,
        record_id: u64,
        g_def: &super::schema::GroupDef,
        sec_per_tick: f64,
        _compression: CompressionAlgorithm,
        sorted_dg_data_link_pos: Option<u64>,
    ) -> Result<(GroupRuntime, u64)> {
        let cg_pos = pos;
        let cg_hdr = Blockheader4Short {
            hdr_id: *b"##CG",
            hdr_gap: [0; 4],
            hdr_len: 104,
        };
        let mut cg = Cg4Block {
            cg_links: 6,
            cg_cg_next: 0,
            cg_cn_first: 0,
            cg_tx_acq_name: 0,
            cg_si_acq_source: 0,
            cg_sr_first: 0,
            cg_md_comment: 0,
            cg_cg_master: None,
            cg_record_id: record_id,
            cg_cycle_count: 0,
            cg_flags: g_def.flags,
            cg_path_separator: g_def.path_separator,
            cg_reserved: [0; 4],
            cg_data_bytes: g_def.record_data_bytes,
            cg_inval_bytes: 0,
        };
        write_binrw(sink, &cg_hdr)?;
        write_binrw(sink, &cg)?;
        pos += 104;

        // Write group acquisition name TX
        let acq_tx_pos = pos;
        let acq_tx = encode_tx(&g_def.name)?;
        sink.write_all(&acq_tx)?;
        pos += acq_tx.len() as u64;
        cg.cg_tx_acq_name = acq_tx_pos as i64;

        // Optional group comment
        if let Some(ref comment) = g_def.comment {
            let c_pos = pos;
            let c_tx = encode_tx(comment)?;
            sink.write_all(&c_tx)?;
            pos += c_tx.len() as u64;
            cg.cg_md_comment = c_pos as i64;
        }

        // Optional group SI block
        if let Some(ref si_def) = g_def.source {
            let (si_pos, new_pos) = Self::write_si_block(sink, pos, si_def)?;
            pos = new_pos;
            cg.cg_si_acq_source = si_pos as i64;
        }

        // Write channels in reverse or forward linked list
        let mut prev_cn_next_link_pos: Option<u64> = None;
        let mut vlsd_channels = Vec::new();

        for (c_idx, c_def) in g_def.channels.iter().enumerate() {
            let cn_pos = pos;
            if c_idx == 0 {
                cg.cg_cn_first = cn_pos as i64;
            }
            if let Some(prev_link) = prev_cn_next_link_pos {
                patch_i64(sink, prev_link, cn_pos as i64, pos)?;
            }

            let cn_hdr = Blockheader4Short {
                hdr_id: *b"##CN",
                hdr_gap: [0; 4],
                hdr_len: 160,
            };
            let mut cn = Cn4Block {
                cn_links: 8,
                cn_cn_next: 0,
                cn_composition: 0,
                cn_tx_name: 0,
                cn_si_source: 0,
                cn_cc_conversion: 0,
                cn_data: 0,
                cn_md_unit: 0,
                cn_md_comment: 0,
                links: Vec::new(),
                cn_type: c_def.cn_type as u8,
                cn_sync_type: c_def.sync_type as u8,
                cn_data_type: c_def.data_type as u8,
                cn_bit_offset: c_def.bit_offset,
                cn_byte_offset: c_def.byte_offset,
                cn_bit_count: c_def.bit_count,
                cn_flags: c_def.flags,
                cn_inval_bit_pos: 0,
                cn_precision: 0,
                cn_alignment: 0,
                cn_attachment_count: 0,
                cn_val_range_min: 0.0,
                cn_val_range_max: 0.0,
                cn_limit_min: 0.0,
                cn_limit_max: 0.0,
                cn_limit_ext_min: 0.0,
                cn_limit_ext_max: 0.0,
            };
            write_binrw(sink, &cn_hdr)?;
            write_binrw(sink, &cn)?;
            pos += 160;
            // cn_cn_next link is at cn_pos + 16 (short hdr) + 8 (cn_links) = cn_pos + 24
            prev_cn_next_link_pos = Some(cn_pos + 24);

            // Channel name TX
            let name_pos = pos;
            let name_tx = encode_tx(&c_def.name)?;
            sink.write_all(&name_tx)?;
            pos += name_tx.len() as u64;
            cn.cn_tx_name = name_pos as i64;

            // Optional unit TX
            if let Some(ref unit) = c_def.unit {
                let u_pos = pos;
                let u_tx = encode_tx(unit)?;
                sink.write_all(&u_tx)?;
                pos += u_tx.len() as u64;
                cn.cn_md_unit = u_pos as i64;
            }

            // Optional comment TX
            if let Some(ref comment) = c_def.comment {
                let m_pos = pos;
                let m_tx = encode_tx(comment)?;
                sink.write_all(&m_tx)?;
                pos += m_tx.len() as u64;
                cn.cn_md_comment = m_pos as i64;
            }

            // Optional channel SI
            if let Some(ref si_def) = c_def.source {
                let (si_pos, new_pos) = Self::write_si_block(sink, pos, si_def)?;
                pos = new_pos;
                cn.cn_si_source = si_pos as i64;
            }

            // Optional array CABLOCK
            if let Some(ref dims) = c_def.array_dims {
                let ca_pos = pos;
                let mut ca = Ca4Block {
                    ca_ndim: dims.len() as u16,
                    ca_dim_size: dims.clone(),
                    ca_byte_offset_base: (c_def.bit_count / 8) as i32,
                    ..Default::default()
                };
                ca.prepare_for_write();
                let mut ca_bytes = Vec::with_capacity(ca.ca_len as usize);
                ca.write_to(&mut Cursor::new(&mut ca_bytes))?;
                sink.write_all(&ca_bytes)?;
                pos += ca_bytes.len() as u64;
                cn.cn_composition = ca_pos as i64;
            }

            // Conversion block (or automatic linear conversion on master time channel 0)
            let effective_conv = if c_idx == 0 && c_def.conversion.is_none() {
                Some(ConversionDef::Linear {
                    p1: 0.0,
                    p2: sec_per_tick,
                })
            } else {
                c_def.conversion.clone()
            };
            if let Some(ref conv) = effective_conv {
                let (cc_pos, new_pos) = Self::write_cc_block(sink, pos, conv)?;
                pos = new_pos;
                cn.cn_cc_conversion = cc_pos as i64;
            }

            // Re-write the populated CN block at `cn_pos`
            sink.seek(SeekFrom::Start(cn_pos))?;
            write_binrw(sink, &cn_hdr)?;
            write_binrw(sink, &cn)?;
            sink.seek(SeekFrom::Start(pos))?;

            if c_def.is_vlsd {
                // `cn_data` link is the 6th link in CNBLOCK:
                // cn_pos + 16 (short hdr) + 8 (cn_links) + 5 * 8 = cn_pos + 64
                let cn_data_link_pos = cn_pos + 64;
                vlsd_channels.push(VlsdChannelState {
                    channel_idx: c_idx,
                    sorted_sd_chain: Some(DlChainState::new(cn_data_link_pos, *b"SD", 1)),
                    sorted_sd_buf: Vec::new(),
                    total_vlsd_bytes: 0,
                    unsorted_vlsd_record_id: 0,
                    unsorted_vlsd_cg_pos: 0,
                    unsorted_vlsd_cycles: 0,
                });
            }
        }

        // Re-write the populated CG block at `cg_pos`
        sink.seek(SeekFrom::Start(cg_pos))?;
        write_binrw(sink, &cg_hdr)?;
        write_binrw(sink, &cg)?;
        sink.seek(SeekFrom::Start(pos))?;

        let rec_size = g_def.record_data_bytes as usize;
        let sorted_chain = sorted_dg_data_link_pos
            .map(|link_pos| DlChainState::new(link_pos, *b"DT", rec_size.max(1)));

        Ok((
            GroupRuntime {
                cg_pos,
                record_id,
                cycle_count: 0,
                record_size: rec_size,
                staging_record: vec![0u8; rec_size],
                sorted_buf: Vec::new(),
                sorted_chain,
                vlsd_channels,
            },
            pos,
        ))
    }

    fn write_si_block(sink: &mut S, mut pos: u64, def: &SourceDef) -> Result<(u64, u64)> {
        let name_pos = pos;
        let name_tx = encode_tx(&def.name)?;
        sink.write_all(&name_tx)?;
        pos += name_tx.len() as u64;

        let path_pos = if let Some(ref p) = def.path {
            let pp = pos;
            let p_tx = encode_tx(p)?;
            sink.write_all(&p_tx)?;
            pos += p_tx.len() as u64;
            pp as i64
        } else {
            0
        };

        let si_pos = pos;
        let hdr = Blockheader4Short {
            hdr_id: *b"##SI",
            hdr_gap: [0; 4],
            hdr_len: 56,
        };
        let si = Si4Block {
            si_links: 3,
            si_tx_name: name_pos as i64,
            si_tx_path: path_pos,
            si_md_comment: 0,
            si_type: def.si_type,
            si_bus_type: def.si_bus_type,
            si_flags: def.si_flags,
            si_reserved: [0; 5],
        };
        write_binrw(sink, &hdr)?;
        write_binrw(sink, &si)?;
        pos += 56;
        Ok((si_pos, pos))
    }

    fn write_cc_block(sink: &mut S, mut pos: u64, conv: &ConversionDef) -> Result<(u64, u64)> {
        let cc = match conv {
            ConversionDef::Identity => {
                Cc4Block::new_with_val(0, CcVal::Real(Vec::new()), Vec::new())
            }
            ConversionDef::Linear { p1, p2 } => {
                Cc4Block::new_with_val(1, CcVal::Real(vec![*p1, *p2]), Vec::new())
            }
            ConversionDef::Rational(params) => {
                Cc4Block::new_with_val(2, CcVal::Real(params.to_vec()), Vec::new())
            }
            ConversionDef::ValueToText {
                entries,
                default_text,
            } => {
                let mut refs = Vec::with_capacity(entries.len() + 1);
                let mut vals = Vec::with_capacity(entries.len());
                for (k, label) in entries {
                    vals.push(*k);
                    let tx_pos = pos;
                    let tx = encode_tx(label)?;
                    sink.write_all(&tx)?;
                    pos += tx.len() as u64;
                    refs.push(tx_pos as i64);
                }
                let def_pos = pos;
                let def_tx = encode_tx(default_text)?;
                sink.write_all(&def_tx)?;
                pos += def_tx.len() as u64;
                refs.push(def_pos as i64);
                Cc4Block::new_with_val(7, CcVal::Real(vals), refs)
            }
        };

        let cc_pos = pos;
        let mut body = Vec::new();
        cc.write_le(&mut Cursor::new(&mut body))
            .map_err(|_| Error::Io)?;
        let hdr = Blockheader4Short {
            hdr_id: *b"##CC",
            hdr_gap: [0; 4],
            hdr_len: 16 + body.len() as u64,
        };
        write_binrw(sink, &hdr)?;
        sink.write_all(&body)?;
        pos += 16 + body.len() as u64;
        Ok((cc_pos, pos))
    }

    // ── Record Push APIs (Raw & Dynamic) ──

    /// Pushes a complete pre-formatted plain record (`group.record_data_bytes` bytes, including the
    /// 8-byte master timestamp at `0..8`) using a timestamp automatically sampled from the logger's [`TimeSource`].
    ///
    /// Byte range `0..8` of `record_bytes` is overwritten with `clock.now_ticks().to_le_bytes()`;
    /// use [`Logger::push_raw_at`] if the caller already stamped hardware ticks into `0..8`.
    pub fn push_raw(&mut self, group: GroupHandle, record_bytes: &[u8]) -> Result<()> {
        let ticks = self.clock.now_ticks();
        self.push_raw_with_ticks(group, ticks, record_bytes, true)
    }

    /// Pushes a complete pre-formatted plain record using an explicit tick timestamp (e.g. hardware CAN/Ethernet timestamp).
    pub fn push_raw_at(
        &mut self,
        group: GroupHandle,
        ticks: u64,
        record_bytes: &[u8],
    ) -> Result<()> {
        self.push_raw_with_ticks(group, ticks, record_bytes, true)
    }

    /// Pushes a payload slice containing only the non-master channel bytes (`record_size - 8` bytes);
    /// the 8-byte master timestamp is prepended automatically from [`TimeSource`].
    pub fn push_payload(&mut self, group: GroupHandle, payload_without_time: &[u8]) -> Result<()> {
        let ticks = self.clock.now_ticks();
        let g_idx = group.0 as usize;
        let g = self
            .groups
            .get_mut(g_idx)
            .ok_or_else(|| Error::InvalidSchema(format!("unknown group {}", group.0)))?;
        if payload_without_time.len() + 8 != g.record_size {
            return Err(Error::RecordSize {
                expected: g.record_size - 8,
                got: payload_without_time.len(),
            });
        }
        g.staging_record[0..8].copy_from_slice(&ticks.to_le_bytes());
        g.staging_record[8..].copy_from_slice(payload_without_time);
        self.commit_staged_record(group)
    }

    fn push_raw_with_ticks(
        &mut self,
        group: GroupHandle,
        ticks: u64,
        record_bytes: &[u8],
        overwrite_time: bool,
    ) -> Result<()> {
        let g_idx = group.0 as usize;
        let g = self
            .groups
            .get_mut(g_idx)
            .ok_or_else(|| Error::InvalidSchema(format!("unknown group {}", group.0)))?;
        if record_bytes.len() != g.record_size {
            return Err(Error::RecordSize {
                expected: g.record_size,
                got: record_bytes.len(),
            });
        }
        g.staging_record.copy_from_slice(record_bytes);
        if overwrite_time {
            g.staging_record[0..8].copy_from_slice(&ticks.to_le_bytes());
        }
        self.commit_staged_record(group)
    }

    /// Begins assembling a record for `group` using the dynamic channel-handle setter API.
    /// Zeroes the staging buffer and stamps the master time channel from [`TimeSource`].
    pub fn begin_record(&mut self, group: GroupHandle) -> Result<RecordWriter<'_, S, T>> {
        let ticks = self.clock.now_ticks();
        self.begin_record_at(group, ticks)
    }

    /// Begins assembling a record for `group` with an explicit tick timestamp.
    pub fn begin_record_at(
        &mut self,
        group: GroupHandle,
        ticks: u64,
    ) -> Result<RecordWriter<'_, S, T>> {
        let g = self
            .groups
            .get_mut(group.0 as usize)
            .ok_or_else(|| Error::InvalidSchema(format!("unknown group {}", group.0)))?;
        g.staging_record.fill(0);
        g.staging_record[0..8].copy_from_slice(&ticks.to_le_bytes());
        Ok(RecordWriter {
            logger: self,
            group,
        })
    }

    /// Writes an unsigned integer (including bit-packed sub-byte or unaligned channels up to 64 bits)
    /// into the group's staging record.
    pub fn set_channel_u64(&mut self, ch: ChannelHandle, val: u64) -> Result<()> {
        let c_def = self.channel_def(ch)?;
        let byte_off = c_def.byte_offset as usize;
        let bit_off = c_def.bit_offset;
        let bit_cnt = c_def.bit_count;
        let is_be = matches!(
            c_def.data_type,
            CnDataType::UIntBE | CnDataType::IntBE | CnDataType::FloatBE
        );
        let g = &mut self.groups[ch.group as usize];
        write_bits(
            &mut g.staging_record,
            byte_off,
            bit_off,
            bit_cnt,
            val,
            is_be,
        );
        Ok(())
    }

    /// Writes a signed integer into the group's staging record.
    pub fn set_channel_i64(&mut self, ch: ChannelHandle, val: i64) -> Result<()> {
        self.set_channel_u64(ch, val as u64)
    }

    /// Writes an `f64` (or `f32` if the channel's `bit_count == 32`) into the group's staging record.
    pub fn set_channel_f64(&mut self, ch: ChannelHandle, val: f64) -> Result<()> {
        let bit_cnt = self.channel_def(ch)?.bit_count;
        if bit_cnt == 32 {
            self.set_channel_u64(ch, (val as f32).to_bits() as u64)
        } else {
            self.set_channel_u64(ch, val.to_bits())
        }
    }

    /// Writes raw bytes (for a fixed-size `ByteArray`, string or `CABLOCK` array channel) or
    /// appends variable-length bytes (for a `VLSD` channel) and writes the 64-bit VLSD offset into the staging record.
    pub fn set_channel_bytes(&mut self, ch: ChannelHandle, data: &[u8]) -> Result<()> {
        let c_def = self.channel_def(ch)?;
        let is_vlsd = c_def.is_vlsd;
        let byte_off = c_def.byte_offset as usize;
        let fixed_len = (c_def.bit_count / 8) as usize;
        let array_total_elems: usize = c_def
            .array_dims
            .as_ref()
            .map(|d| d.iter().product::<u64>() as usize)
            .unwrap_or(1);
        let max_fixed = fixed_len * array_total_elems;

        if !is_vlsd {
            let g = &mut self.groups[ch.group as usize];
            let copy_len = data.len().min(max_fixed);
            g.staging_record[byte_off..byte_off + copy_len].copy_from_slice(&data[..copy_len]);
            if copy_len < max_fixed {
                g.staging_record[byte_off + copy_len..byte_off + max_fixed].fill(0);
            }
            return Ok(());
        }

        // VLSD channel: record the current cumulative byte offset into the staging record,
        // then append `[u32_le(len)] + data` to the sorted SD buffer or unsorted stream.
        let layout = self.schema.layout;
        let g = &mut self.groups[ch.group as usize];
        let vlsd = g
            .vlsd_channels
            .iter_mut()
            .find(|v| v.channel_idx == ch.channel as usize)
            .ok_or_else(|| Error::InvalidSchema("VLSD state not found".into()))?;

        let offset_val = vlsd.total_vlsd_bytes;
        g.staging_record[byte_off..byte_off + 8].copy_from_slice(&offset_val.to_le_bytes());

        let len_u32 = data.len() as u32;
        match layout {
            DataLayout::Sorted => {
                vlsd.sorted_sd_buf.extend_from_slice(&len_u32.to_le_bytes());
                vlsd.sorted_sd_buf.extend_from_slice(data);
                vlsd.total_vlsd_bytes += 4 + data.len() as u64;
            }
            DataLayout::Unsorted => {
                // In unsorted mode, emit a VLSD CG record (`[rec_id: u8][len: u32][data]`) into `unsorted_buf`
                // before the main record is committed (per MDF4 spec §6.14.2).
                let rec_id = vlsd.unsorted_vlsd_record_id as u8;
                self.unsorted_buf.push(rec_id);
                self.unsorted_buf.extend_from_slice(&len_u32.to_le_bytes());
                self.unsorted_buf.extend_from_slice(data);
                vlsd.total_vlsd_bytes += data.len() as u64;
                vlsd.unsorted_vlsd_cycles += 1;
                // Note: in unsorted VLSD CG mode, the CN record still stores the cumulative offset in SD-style
                // if `cn_data` points to SD, or when `cn_data` points to VLSD CG, MDF4 stores the 64-bit cumulative
                // byte offset (including 4-byte length prefixes) or sequential VLSD records. Wait: MDF 4.2 §6.6.3:
                // "Even for unsorted data, the VLSD channel contains the uint64 offset in the virtual SDBLOCK".
                // Wait: in `mdfreader4.rs`, let's keep `offset_val` advancing by `4 + data.len()` so both match!
            }
        }
        Ok(())
    }

    /// Commits the group's staging record to the group buffer (or shared unsorted buffer),
    /// flushing a `DT`/`DZ` block if the buffer reaches its threshold.
    pub fn commit_staged_record(&mut self, group: GroupHandle) -> Result<()> {
        let g_idx = group.0 as usize;
        if g_idx >= self.groups.len() {
            return Err(Error::InvalidSchema(format!("unknown group {}", group.0)));
        }
        let threshold = self.schema.groups[g_idx].flush_threshold_bytes;

        match self.schema.layout {
            DataLayout::Sorted => {
                let g = &mut self.groups[g_idx];
                g.sorted_buf.extend_from_slice(&g.staging_record);
                g.cycle_count += 1;
                let need_record_flush = g.sorted_buf.len() >= threshold;
                let need_vlsd_flush = g
                    .vlsd_channels
                    .iter()
                    .any(|v| v.sorted_sd_buf.len() >= threshold);
                if need_record_flush {
                    self.flush_sorted_group(g_idx)?;
                }
                if need_vlsd_flush {
                    self.flush_sorted_vlsd(g_idx)?;
                }
            }
            DataLayout::Unsorted => {
                let g = &mut self.groups[g_idx];
                self.unsorted_buf.push(g.record_id as u8);
                self.unsorted_buf.extend_from_slice(&g.staging_record);
                g.cycle_count += 1;
                if self.unsorted_buf.len() >= self.unsorted_flush_threshold {
                    self.flush_unsorted()?;
                }
            }
        }

        if self.schema.checkpoint_every_n_blocks > 0
            && self.blocks_since_checkpoint >= self.schema.checkpoint_every_n_blocks
        {
            self.checkpoint()?;
        }
        Ok(())
    }

    fn channel_def(&self, ch: ChannelHandle) -> Result<&ChannelDef> {
        let g = self
            .schema
            .groups
            .get(ch.group as usize)
            .ok_or_else(|| Error::InvalidSchema(format!("unknown group {}", ch.group)))?;
        g.channels
            .get(ch.channel as usize)
            .ok_or_else(|| Error::InvalidSchema(format!("unknown channel {}", ch.channel)))
    }

    // ── Block Flushing & DL/HL Chaining ──

    fn flush_sorted_group(&mut self, g_idx: usize) -> Result<()> {
        let buf = core::mem::take(&mut self.groups[g_idx].sorted_buf);
        if buf.is_empty() {
            return Ok(());
        }
        let mut chain = self.groups[g_idx]
            .sorted_chain
            .take()
            .expect("sorted group has chain");
        let res = Self::write_data_chunk(
            &mut self.sink,
            &mut self.eof_pos,
            &mut chain,
            &buf,
            self.schema.compression,
            self.schema.max_blocks_per_dl,
        );
        self.groups[g_idx].sorted_chain = Some(chain);
        res?;
        self.blocks_since_checkpoint += 1;
        Ok(())
    }

    fn flush_sorted_vlsd(&mut self, g_idx: usize) -> Result<()> {
        let n_vlsd = self.groups[g_idx].vlsd_channels.len();
        for v_idx in 0..n_vlsd {
            let buf = core::mem::take(&mut self.groups[g_idx].vlsd_channels[v_idx].sorted_sd_buf);
            if buf.is_empty() {
                continue;
            }
            let mut chain = self.groups[g_idx].vlsd_channels[v_idx]
                .sorted_sd_chain
                .take()
                .expect("sorted VLSD has chain");
            let res = Self::write_data_chunk(
                &mut self.sink,
                &mut self.eof_pos,
                &mut chain,
                &buf,
                self.schema.compression,
                self.schema.max_blocks_per_dl,
            );
            self.groups[g_idx].vlsd_channels[v_idx].sorted_sd_chain = Some(chain);
            res?;
        }
        Ok(())
    }

    fn flush_unsorted(&mut self) -> Result<()> {
        let buf = core::mem::take(&mut self.unsorted_buf);
        if buf.is_empty() {
            return Ok(());
        }
        let mut chain = self
            .unsorted_chain
            .take()
            .expect("unsorted logger has chain");
        let res = Self::write_data_chunk(
            &mut self.sink,
            &mut self.eof_pos,
            &mut chain,
            &buf,
            self.schema.compression,
            self.schema.max_blocks_per_dl,
        );
        self.unsorted_chain = Some(chain);
        res?;
        self.blocks_since_checkpoint += 1;
        Ok(())
    }

    fn write_data_chunk(
        sink: &mut S,
        eof_pos: &mut u64,
        chain: &mut DlChainState,
        uncompressed: &[u8],
        algo: CompressionAlgorithm,
        max_blocks_per_dl: usize,
    ) -> Result<()> {
        let use_dz = algo != CompressionAlgorithm::NoCompression;

        // If this is the very first block in the chain, allocate the initial HL (if compressed) and DL block
        if chain.current_dl_pos == 0 {
            if use_dz && !chain.hl_written {
                let hl_pos = *eof_pos;
                let dl_pos = hl_pos + 40;
                sink.seek(SeekFrom::Start(hl_pos))?;
                sink.write_all(b"##HL")?;
                let hl = Hl4Block {
                    reserved: [0; 4],
                    hl_len: 40,
                    hl_links: 1,
                    hl_dl_first: dl_pos as i64,
                    hl_flags: 0, // bit 0 = equal length (0 = offset table in DL)
                    hl_zip_type: algo.zip_type(),
                    hl_reserved: [0; 5],
                };
                write_binrw(sink, &hl)?;
                *eof_pos = dl_pos;
                patch_i64(sink, chain.owner_link_pos, hl_pos as i64, *eof_pos)?;
                chain.hl_written = true;
            }
            let dl_pos = Self::allocate_empty_dl(sink, eof_pos, max_blocks_per_dl)?;
            if !chain.hl_written {
                patch_i64(sink, chain.owner_link_pos, dl_pos as i64, *eof_pos)?;
            }
            chain.current_dl_pos = dl_pos;
        } else if chain.current_dl_blocks.len() >= max_blocks_per_dl {
            // Current DL is full: patch it, allocate a new DL, and link `dl_dl_next`
            Self::patch_dl_block(sink, chain, max_blocks_per_dl, *eof_pos)?;
            let prev_dl = chain.current_dl_pos;
            let new_dl = Self::allocate_empty_dl(sink, eof_pos, max_blocks_per_dl)?;
            // `dl_dl_next` is at `prev_dl + 24`
            patch_i64(sink, prev_dl + 24, new_dl as i64, *eof_pos)?;
            chain.current_dl_pos = new_dl;
            chain.current_dl_blocks.clear();
            chain.current_dl_offsets.clear();
        }

        // Write the DT/SD or DZ block at `*eof_pos`
        let block_pos = *eof_pos;
        sink.seek(SeekFrom::Start(block_pos))?;

        if use_dz {
            let comp = compress_dz_data(uncompressed, algo, chain.transpose_width)?;
            sink.write_all(b"##DZ")?;
            let total_len = 48 + comp.len() as u64;
            let dz = Dz4Block {
                reserved: [0; 4],
                len: total_len,
                dz_links: 0,
                dz_org_block_type: chain.org_block_type,
                dz_zip_type: algo.zip_type(),
                dz_reserved: 0,
                dz_zip_parameter: if algo.is_transposed() {
                    chain.transpose_width as u32
                } else {
                    0
                },
                dz_org_data_length: uncompressed.len() as u64,
                dz_data_length: comp.len() as u64,
            };
            write_binrw(sink, &dz)?;
            sink.write_all(&comp)?;
            let aligned = align8(total_len);
            let pad = (aligned - total_len) as usize;
            if pad > 0 {
                let zeros = [0u8; 8];
                sink.write_all(&zeros[..pad])?;
            }
            *eof_pos = block_pos + aligned;
        } else {
            let hdr_id = [b'#', b'#', chain.org_block_type[0], chain.org_block_type[1]];
            let total_len = 24 + uncompressed.len() as u64;
            let hdr = Blockheader4 {
                hdr_id,
                hdr_gap: [0; 4],
                hdr_len: total_len,
                hdr_links: 0,
            };
            write_binrw(sink, &hdr)?;
            sink.write_all(uncompressed)?;
            let aligned = align8(total_len);
            let pad = (aligned - total_len) as usize;
            if pad > 0 {
                let zeros = [0u8; 8];
                sink.write_all(&zeros[..pad])?;
            }
            *eof_pos = block_pos + aligned;
        }

        chain.current_dl_blocks.push(block_pos as i64);
        chain
            .current_dl_offsets
            .push(chain.total_uncompressed_bytes);
        chain.total_uncompressed_bytes += uncompressed.len() as u64;
        Ok(())
    }

    /// Reserves a fixed-capacity `DLBLOCK` at `*eof_pos` for up to `max_blocks_per_dl` entries,
    /// so later `checkpoint()` / `close()` can overwrite it in place without moving subsequent data blocks.
    fn allocate_empty_dl(sink: &mut S, eof_pos: &mut u64, max_blocks: usize) -> Result<u64> {
        let dl_pos = *eof_pos;
        // Reserved size: 24 (hdr) + 8 (dl_dl_next) + max_blocks*8 (dl_data) + 8 (flags/reserved/count) + max_blocks*8 (dl_offset)
        let reserved_len = 24 + 8 + (max_blocks as u64) * 8 + 8 + (max_blocks as u64) * 8;
        sink.seek(SeekFrom::Start(dl_pos))?;
        let zeros = vec![0u8; reserved_len as usize];
        sink.write_all(&zeros)?;
        // Write valid empty DL header so even before first patch the block ID is `##DL`
        sink.seek(SeekFrom::Start(dl_pos))?;
        sink.write_all(b"##DL")?;
        let dl = Dl4Block {
            reserved: [0; 4],
            dl_len: 40, // 24 + 8 (dl_dl_next) + 8 (flags/count) when count == 0
            dl_links: 1,
            dl_dl_next: 0,
            dl_data: Vec::new(),
            dl_flags: 0,
            dl_reserved: [0; 3],
            dl_count: 0,
            dl_equal_length: None,
            dl_offset: Vec::new(),
            dl_time_values: Vec::new(),
            dl_angle_values: Vec::new(),
            dl_distance_values: Vec::new(),
        };
        write_binrw(sink, &dl)?;
        *eof_pos = dl_pos + reserved_len;
        sink.seek(SeekFrom::Start(*eof_pos))?;
        Ok(dl_pos)
    }

    fn patch_dl_block(
        sink: &mut S,
        chain: &DlChainState,
        _max_blocks: usize,
        restore_pos: u64,
    ) -> Result<()> {
        if chain.current_dl_pos == 0 {
            return Ok(());
        }
        let n = chain.current_dl_blocks.len();
        let actual_len = 24 + 8 + (n as u64) * 8 + 8 + (n as u64) * 8;
        let dl = Dl4Block {
            reserved: [0; 4],
            dl_len: actual_len,
            dl_links: 1 + n as u64,
            dl_dl_next: 0,
            dl_data: chain.current_dl_blocks.clone(),
            dl_flags: 0, // bit 0 = 0 -> explicit `dl_offset` list
            dl_reserved: [0; 3],
            dl_count: n as u32,
            dl_equal_length: None,
            dl_offset: chain.current_dl_offsets.clone(),
            dl_time_values: Vec::new(),
            dl_angle_values: Vec::new(),
            dl_distance_values: Vec::new(),
        };
        sink.seek(SeekFrom::Start(chain.current_dl_pos))?;
        sink.write_all(b"##DL")?;
        write_binrw(sink, &dl)?;
        sink.seek(SeekFrom::Start(restore_pos))?;
        Ok(())
    }

    // ── Checkpointing, In-Place Finalization & Rotation ──

    /// Flushes all pending record & VLSD buffers, patches `DLBLOCK`s and `CGBLOCK` cycle/byte counters
    /// in place, and flushes the underlying [`Sink`].
    ///
    /// Leaves the file marked as `"UnFinMF "` while recording continues, but all data up to this
    /// checkpoint is consistent on disk.
    pub fn checkpoint(&mut self) -> Result<()> {
        match self.schema.layout {
            DataLayout::Sorted => {
                for g_idx in 0..self.groups.len() {
                    self.flush_sorted_group(g_idx)?;
                    self.flush_sorted_vlsd(g_idx)?;
                    if let Some(ref chain) = self.groups[g_idx].sorted_chain {
                        Self::patch_dl_block(
                            &mut self.sink,
                            chain,
                            self.schema.max_blocks_per_dl,
                            self.eof_pos,
                        )?;
                    }
                    for v in &self.groups[g_idx].vlsd_channels {
                        if let Some(ref chain) = v.sorted_sd_chain {
                            Self::patch_dl_block(
                                &mut self.sink,
                                chain,
                                self.schema.max_blocks_per_dl,
                                self.eof_pos,
                            )?;
                        }
                    }
                }
            }
            DataLayout::Unsorted => {
                self.flush_unsorted()?;
                if let Some(ref chain) = self.unsorted_chain {
                    Self::patch_dl_block(
                        &mut self.sink,
                        chain,
                        self.schema.max_blocks_per_dl,
                        self.eof_pos,
                    )?;
                }
            }
        }

        // Patch each group's `cg_cycle_count` (at `cg_pos + 80`) and any unsorted VLSD companion CGs
        for g in &self.groups {
            patch_u64(&mut self.sink, g.cg_pos + 80, g.cycle_count, self.eof_pos)?;
            for v in &g.vlsd_channels {
                if v.unsorted_vlsd_cg_pos != 0 {
                    // Patch companion VLSD CGBLOCK:
                    // cg_cycle_count at +80, cg_data_bytes (u32) at +96, cg_inval_bytes (u32) at +100
                    patch_u64(
                        &mut self.sink,
                        v.unsorted_vlsd_cg_pos + 80,
                        v.unsorted_vlsd_cycles,
                        self.eof_pos,
                    )?;
                    let low = (v.total_vlsd_bytes & 0xFFFF_FFFF) as u32;
                    let high = (v.total_vlsd_bytes >> 32) as u32;
                    self.sink
                        .seek(SeekFrom::Start(v.unsorted_vlsd_cg_pos + 96))?;
                    self.sink.write_all(&low.to_le_bytes())?;
                    self.sink.write_all(&high.to_le_bytes())?;
                    self.sink.seek(SeekFrom::Start(self.eof_pos))?;
                }
            }
        }

        // If wall clock wasn't available at start(), check if the TimeSource has locked since then
        if !self.hd_time_valid
            && let Some(wall_ns) = self.clock.wall_clock_ns()
            && wall_ns != 0
        {
            // hd_start_time_ns is at offset 64 (HD start) + 24 (hdr) + 48 (6 links) = 136
            patch_u64(&mut self.sink, 136, wall_ns, self.eof_pos)?;
            self.hd_time_valid = true;
        }

        self.blocks_since_checkpoint = 0;
        self.sink.flush()
    }

    /// Finalizes the MDF 4.3 file in place (`checkpoint()` + rewrites [`IdBlock`] as `"MDF     "` with
    /// `id_unfin_flags = 0`) and returns the underlying `(Sink, TimeSource)`.
    pub fn close(mut self) -> Result<(S, T)> {
        self.finalize_in_place()?;
        Ok((self.sink, self.clock))
    }

    fn finalize_in_place(&mut self) -> Result<()> {
        self.checkpoint()?;
        let id = IdBlock::default();
        self.sink.seek(SeekFrom::Start(0))?;
        write_binrw(&mut self.sink, &id)?;
        self.sink.seek(SeekFrom::Start(self.eof_pos))?;
        self.sink.flush()
    }

    /// Finalizes the current file in place and starts a new MDF 4.3 file on `new_sink`
    /// (incrementing the `recorder_file_index` in `HDcomment`), optionally replacing the [`Schema`].
    /// Returns the closed old sink.
    pub fn rotate(&mut self, mut new_sink: S, new_schema: Option<Schema>) -> Result<S> {
        self.finalize_in_place()?;
        if let Some(s) = new_schema {
            self.schema = s;
        }
        self.sequence_index += 1;
        let (groups, unsorted_chain, eof_pos, hd_time_valid) = Self::write_metadata(
            &self.schema,
            &mut new_sink,
            &mut self.clock,
            self.sequence_index,
        )?;
        let old_sink = core::mem::replace(&mut self.sink, new_sink);
        self.groups = groups;
        self.unsorted_chain = unsorted_chain;
        self.unsorted_buf.clear();
        self.eof_pos = eof_pos;
        self.hd_time_valid = hd_time_valid;
        self.blocks_since_checkpoint = 0;
        Ok(old_sink)
    }

    /// Borrows the underlying sink (e.g. to inspect a `VecSink` during tests).
    pub fn sink(&self) -> &S {
        &self.sink
    }
}

/// Ergonomic dynamic record builder returned by [`Logger::begin_record`].
pub struct RecordWriter<'a, S: Sink, T: TimeSource> {
    logger: &'a mut Logger<S, T>,
    group: GroupHandle,
}

impl<'a, S: Sink, T: TimeSource> RecordWriter<'a, S, T> {
    /// Sets an unsigned integer channel in this record.
    pub fn set_u64(self, ch: ChannelHandle, val: u64) -> Result<Self> {
        self.logger.set_channel_u64(ch, val)?;
        Ok(self)
    }

    /// Sets a signed integer channel in this record.
    pub fn set_i64(self, ch: ChannelHandle, val: i64) -> Result<Self> {
        self.logger.set_channel_i64(ch, val)?;
        Ok(self)
    }

    /// Sets a floating-point (`f32` or `f64`) channel in this record.
    pub fn set_f64(self, ch: ChannelHandle, val: f64) -> Result<Self> {
        self.logger.set_channel_f64(ch, val)?;
        Ok(self)
    }

    /// Sets a byte array, string, fixed-size `CABLOCK` array, or variable-length `VLSD` channel in this record.
    pub fn set_bytes(self, ch: ChannelHandle, data: &[u8]) -> Result<Self> {
        self.logger.set_channel_bytes(ch, data)?;
        Ok(self)
    }

    /// Commits the record to the logger.
    pub fn commit(self) -> Result<()> {
        self.logger.commit_staged_record(self.group)
    }
}

// ── Low-level helpers ──

fn write_binrw<S: Sink, B: for<'a> BinWrite<Args<'a> = ()>>(sink: &mut S, block: &B) -> Result<()> {
    let mut c = Cursor::new(Vec::new());
    block.write_le(&mut c).map_err(|_| Error::Io)?;
    sink.write_all(&c.into_inner())
}

fn patch_i64<S: Sink>(sink: &mut S, offset: u64, val: i64, restore_pos: u64) -> Result<()> {
    sink.seek(SeekFrom::Start(offset))?;
    sink.write_all(&val.to_le_bytes())?;
    sink.seek(SeekFrom::Start(restore_pos))?;
    Ok(())
}

fn patch_u64<S: Sink>(sink: &mut S, offset: u64, val: u64, restore_pos: u64) -> Result<()> {
    sink.seek(SeekFrom::Start(offset))?;
    sink.write_all(&val.to_le_bytes())?;
    sink.seek(SeekFrom::Start(restore_pos))?;
    Ok(())
}

/// Writes `bit_count` bits (`1..=64`) of `val` into `record[byte_offset..]` at `bit_offset` (`0..=7`).
fn write_bits(
    record: &mut [u8],
    byte_offset: usize,
    bit_offset: u8,
    bit_count: u32,
    val: u64,
    is_big_endian: bool,
) {
    if bit_offset == 0 && bit_count.is_multiple_of(8) {
        let n_bytes = (bit_count / 8) as usize;
        if is_big_endian {
            let be = val.to_be_bytes();
            record[byte_offset..byte_offset + n_bytes].copy_from_slice(&be[8 - n_bytes..]);
        } else {
            let le = val.to_le_bytes();
            record[byte_offset..byte_offset + n_bytes].copy_from_slice(&le[..n_bytes]);
        }
        return;
    }
    // Bit-packed little-endian or big-endian field fitting inside 8 contiguous bytes
    let n_bytes = ((bit_offset as u32 + bit_count).div_ceil(8) as usize).min(8);
    let mask: u64 = if bit_count >= 64 {
        u64::MAX
    } else {
        (1u64 << bit_count) - 1
    };
    let shifted = (val & mask) << bit_offset;
    let clear_mask = !(mask << bit_offset);

    let mut raw = [0u8; 8];
    if is_big_endian {
        raw[8 - n_bytes..].copy_from_slice(&record[byte_offset..byte_offset + n_bytes]);
        let cur = u64::from_be_bytes(raw);
        let updated = (cur & clear_mask) | shifted;
        let out = updated.to_be_bytes();
        record[byte_offset..byte_offset + n_bytes].copy_from_slice(&out[8 - n_bytes..]);
    } else {
        raw[..n_bytes].copy_from_slice(&record[byte_offset..byte_offset + n_bytes]);
        let cur = u64::from_le_bytes(raw);
        let updated = (cur & clear_mask) | shifted;
        let out = updated.to_le_bytes();
        record[byte_offset..byte_offset + n_bytes].copy_from_slice(&out[..n_bytes]);
    }
}
