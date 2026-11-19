//! data read and load in memory based in MdfInfo4's metadata
use crate::data_holder::channel_builder::ChannelBuilder;
use crate::mdfinfo::MdfInfo;
use crate::mdfinfo::block_chain::BlockChain;
use crate::mdfinfo::mdfinfo4::{
    CG_F_VLSC, CG_F_VLSD, Dl4Block, Dt4Block, Dz4Block, Dz4BlockExt, Gd4Block, Hl4Block, Ld4Block,
    parse_dz, parser_dl4_block, parser_ld4_block, read_dz_raw,
};
use crate::mdfinfo::mdfinfo4::{Ca4Block, Cg4, Cn4, Compo, Composition, Dg4, Ds4Block};
use crate::mdfinfo::sym_buf_reader::SliceCursor;
use crate::mdfreader::data_read4::{
    read_channels_from_bytes, read_channels_from_bytes_with_inval, read_one_channel_array,
    read_one_channel_array_chunk,
};
use crate::mdfreader::datastream_decoder;
use anyhow::{Context, Error, Result, bail};
use encoding_rs::{Decoder, GB18030, UTF_8, UTF_16BE, UTF_16LE, WINDOWS_1252};
use log::warn;
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::HashSet;
use std::io::Read;
use std::str;
use unicode_bom::Bom;

use super::Mdf;

/// The following constant represents the size of data chunk to be read and processed.
/// a big chunk will improve performance but consume more memory
/// a small chunk will not consume too much memory but will cause many read calls, penalising performance
pub const CHUNK_SIZE_READING_4: usize = 4_194_304; // can be tuned according to architecture

/// Reads the file data based on headers information contained in info parameter
/// Hashset of channel names parameter allows to filter which channels to read
pub fn mdfreader4<'a>(
    rdr: &mut SliceCursor<'a>,
    mdf: &mut Mdf,
    channel_names: &HashSet<String>,
) -> Result<(), Error> {
    match &mut mdf.mdf_info {
        MdfInfo::V4(info) => {
            let mut position: i64 = 0;
            let mut sorted: bool;
            let mut decoder: Dec = Dec {
                windows_1252: WINDOWS_1252.new_decoder(),
                utf_16_be: UTF_16BE.new_decoder(),
                utf_16_le: UTF_16LE.new_decoder(),
            };
            for dg in info.dg.values_mut() {
                for cg in dg.cg.values_mut() {
                    for cn in cg.cn.values_mut() {
                        cn.should_read = channel_names.contains(&*cn.unique_name);
                    }
                }
            }
            // read file data
            for dg in info.dg.values_mut() {
                let has_readable = dg
                    .cg
                    .values()
                    .any(|cg| cg.cn.values().any(|cn| cn.should_read));
                if !has_readable {
                    continue;
                }
                if dg.block.dg_data != 0 {
                    // header block
                    rdr.seek_relative(dg.block.dg_data - position)
                        .context("Could not position buffer")?; // change buffer position
                    let mut id = [0u8; 4];
                    rdr.read_exact(&mut id).context("could not read block id")?;
                    sorted = dg.block.dg_rec_id_size == 0 && dg.cg.len() == 1;
                    position = read_data(rdr, id, dg, dg.block.dg_data, sorted, &mut decoder)
                        .with_context(|| format!("failed reading data for dg {dg:?}"))?;
                    apply_bit_mask_offset(dg).context("failed applying bit mask offset")?;
                } else {
                    // dg_data == 0: virtual channels or independent channel columns
                    for channel_group in dg.cg.values_mut() {
                        let cycle_count = channel_group.block.cg_cycle_count;
                        if cycle_count > 0 {
                            initialise_arrays(channel_group, &cycle_count)
                                .context("failed initialising arrays for dg_data == 0")?;
                        }
                    }
                }

                // Check for independent channels with cn_data != 0 and DG template arrays
                for channel_group in dg.cg.values_mut() {
                    let cycle_count = channel_group.block.cg_cycle_count as usize;
                    for cn in channel_group.cn.values_mut() {
                        if !cn.should_read {
                            continue;
                        }
                        // Check for DG template array storage (ca_storage == 2)
                        let opt_ca = if let Some(compo) = &cn.composition
                            && let Compo::CA(ca) = &compo.block
                            && ca.ca_storage == 2
                        {
                            Some((**ca).clone())
                        } else {
                            None
                        };
                        if let Some(ca) = opt_ca {
                            position = read_dg_template_array(rdr, cn, &ca, cycle_count, position)?;
                            continue;
                        }
                        // Check for independent column-oriented data (only when data group has no record block: dg_data == 0)
                        if dg.block.dg_data == 0
                            && cn.block.cn_data != 0
                            && (cn.block.cn_flags & crate::mdfinfo::mdfinfo4::CN_F_EVENT_SIGNAL
                                == 0)
                            && cn.event_template.is_none()
                        {
                            if cn.block.cn_type == 0 || cn.block.cn_type == 2 {
                                position =
                                    read_independent_channel_data(rdr, cn, cycle_count, position)?;
                            } else if cn.block.cn_type == 1 {
                                rdr.seek_relative(cn.block.cn_data - position)
                                    .context("Could not position buffer for VLSD channel")?;
                                position = cn.block.cn_data;
                                let mut id = [0u8; 4];
                                rdr.read_exact(&mut id)
                                    .context("could not read block id for VLSD")?;
                                if "##SD".as_bytes() == id {
                                    let block_header: Dt4Block = binrw::BinReaderExt::read_le(rdr)
                                        .context("Could not read SD block struct")?;
                                    let data = rdr
                                        .read_slice(block_header.len as usize - 24)
                                        .context("could not read SD data buffer")?;
                                    position += block_header.len as i64;
                                    read_vlsd_from_slice(data, cn, 0, &mut decoder)?;
                                } else if "##DZ".as_bytes() == id {
                                    let (data, block_header) = parse_dz(rdr)?;
                                    position += block_header.len as i64;
                                    read_vlsd_from_slice(&data, cn, 0, &mut decoder)?;
                                }
                                if let Some(b) = cn.builder.take() {
                                    cn.data = b.finish();
                                }
                            }
                        }
                    }
                    channel_group
                        .process_all_channel_invalid_bits()
                        .context("failed processing all channel invalid bits")?;
                    // Process Channel Variants (CV) - merge option data based on discriminator
                    channel_group
                        .process_channel_variants()
                        .context("failed processing channel variants")?;
                    // Process Channel Unions (CU) - create UnionArray from member channels
                    channel_group
                        .process_channel_unions()
                        .context("failed processing channel unions")?;
                    // Process Maximum Length Signal Data (MLSD) - truncate to size channel values
                    channel_group
                        .process_mlsd_channels()
                        .context("failed processing mlsd channels")?;
                    for cn in channel_group.cn.values_mut() {
                        if let Some(b) = cn.builder.take() {
                            cn.data = b.finish();
                        }
                    }
                }
                // Defer conversion of channels to physical values (handled lazily on access)
            }
        }
        MdfInfo::V3(_) => {}
    };
    Ok(())
}

/// Reads all kind of data layout : simple DT or DV, sorted or unsorted, Data List,
/// compressed data blocks DZ or Sample DATA
fn read_data<'a>(
    rdr: &mut SliceCursor<'a>,
    id: [u8; 4],
    dg: &mut Dg4,
    mut position: i64,
    sorted: bool,
    decoder: &mut Dec,
) -> Result<i64> {
    // block header is already read
    let mut vlsd_channels: Vec<(u8, i64)> = Vec::new();
    match id {
        [35, 35, 68, 84] => {
            // ##DT
            let block_header: Dt4Block = binrw::BinReaderExt::read_le(rdr)
                .context("could not read into Dt4Block structure")?;
            // simple data block
            if sorted {
                // sorted data group
                for channel_group in dg.cg.values_mut() {
                    vlsd_channels = read_all_channels_sorted(rdr, channel_group)
                        .context("failed reading all channels sorted")?;
                    position += block_header.len as i64;
                }
                position = process_dynamic_channels(rdr, dg, &vlsd_channels, position, decoder)?;
            } else if !dg.cg.is_empty() {
                // unsorted data
                // initialises all arrays
                for channel_group in dg.cg.values_mut() {
                    initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone())
                        .context("failed intialising arrays")?;
                }
                read_all_channels_unsorted(rdr, dg, block_header.len as i64)
                    .context("failed reading all channels unsorted")?;
                position += block_header.len as i64;
            }
        }
        [35, 35, 68, 90] => {
            // ##DZ
            let (data, block_header) = parse_dz(rdr)?;
            // compressed data
            if sorted {
                // sorted data group
                for channel_group in dg.cg.values_mut() {
                    vlsd_channels = read_all_channels_sorted_from_bytes(&data, channel_group)
                        .context("failed reading all channels sorted from bytes")?;
                }
                position += block_header.len as i64;
                position = process_dynamic_channels(rdr, dg, &vlsd_channels, position, decoder)?;
            } else if !dg.cg.is_empty() {
                // unsorted data
                // initialises all arrays
                for channel_group in dg.cg.values_mut() {
                    initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone())
                        .context("failed intialising arrays")?;
                }
                // initialise record counter
                let mut unsorted_state = UnsortedState::new(dg);
                unsorted_state
                    .process_slice(&data, dg, decoder)
                    .context("failed reading all channels sorted from bytes")?;
                unsorted_state
                    .finish(dg)
                    .context("failed finishing unsorted DZ data")?;
                position += block_header.len as i64;
            }
        }
        [35, 35, 72, 76] => {
            // ##HL
            let (pos, id) = read_hl(rdr, position)?;
            position = pos;
            // Read DL Blocks
            position = read_data(rdr, id, dg, position, sorted, decoder)
                .context("failed reading data from HL block")?;
        }
        [35, 35, 68, 76] => {
            // ##DL
            // data list
            let dl_start_pos = position;
            if sorted {
                // sorted data group
                for channel_group in dg.cg.values_mut() {
                    let (mut dl_blocks, pos) = parser_dl4(rdr, position)?;
                    if let Some(first_dl) = dl_blocks.first_mut()
                        && first_dl.dl_count == 0
                        && let Some(rec) = dg.recovered_dl_data.get(&dl_start_pos)
                    {
                        first_dl.dl_data = rec.clone();
                        first_dl.dl_count = rec.len() as u32;
                    }
                    let rec_pos = channel_group
                        .cn
                        .iter()
                        .find(|(_, cn)| cn.block.cn_type == 1)
                        .map(|(pos, _)| *pos)
                        .or_else(|| channel_group.cn.keys().next().copied())
                        .unwrap_or(0);
                    let (pos, vlsd) =
                        parser_dl4_sorted(rdr, dl_blocks, pos, channel_group, decoder, &rec_pos)
                            .context("failed parsing DL4 sorted")?;
                    position = pos;
                    vlsd_channels = vlsd;
                }
                position = process_dynamic_channels(rdr, dg, &vlsd_channels, position, decoder)?;
            } else if !dg.cg.is_empty() {
                // unsorted data
                // initialises all arrays
                for channel_group in dg.cg.values_mut() {
                    initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone())
                        .context("failed intialising arrays")?;
                }
                let (mut dl_blocks, pos) = parser_dl4(rdr, position)?;
                if let Some(first_dl) = dl_blocks.first_mut()
                    && first_dl.dl_count == 0
                    && let Some(rec) = dg.recovered_dl_data.get(&dl_start_pos)
                {
                    first_dl.dl_data = rec.clone();
                    first_dl.dl_count = rec.len() as u32;
                }
                let pos = parser_dl4_unsorted(rdr, dg, dl_blocks, pos)
                    .context("failed parsing DL4 block unsorted")?;
                position = pos;
            }
        }
        [35, 35, 76, 68] => {
            // ##LD
            // list data, cannot be used for unsorted data
            for channel_group in dg.cg.values_mut() {
                let pos =
                    parser_ld4(rdr, position, channel_group).context("failed parsing DL4 block")?;
                position = pos;
                for (rec_pos, cn) in &channel_group.cn {
                    if cn.should_read && cn.block.cn_type == 1 && cn.block.cn_data != 0 {
                        vlsd_channels.push((cn.block.cn_type, *rec_pos));
                    }
                }
            }
            position = process_dynamic_channels(rdr, dg, &vlsd_channels, position, decoder)?;
        }
        [35, 35, 68, 86] => {
            // ##DV
            // data values
            // sorted data group only, no record id, no invalid bytes
            let block_header: Dt4Block = binrw::BinReaderExt::read_le(rdr)
                .context("could not read into Dv4Block structure")?;
            for channel_group in dg.cg.values_mut() {
                vlsd_channels = read_all_channels_sorted(rdr, channel_group)
                    .context("failed reading all channels sorted")?;
            }
            position += block_header.len as i64;
            position = process_dynamic_channels(rdr, dg, &vlsd_channels, position, decoder)?;
        }
        [35, 35, 68, 71] => {
            // ##DG
            bail!("Weird, a DG block type {id:?}") // should never happen
        }
        [35, 35, 71, 68] => {
            // ##GD - Guard Block (MDF 4.3)
            // GDBLOCK is used to safeguard newly introduced features against older readers
            let block: Gd4Block = binrw::BinReaderExt::read_le(rdr)
                .context("could not read into Gd4Block structure")?;

            // Gd4Block struct size (without id): 30 bytes up to gd_version, plus 6 bytes of gd_reserved when gd_len >= 40.
            let struct_size: i64 = if block.gd_len >= 40 { 36 } else { 30 };
            let current_pos = position + 4 + struct_size;

            // Check if we support the required MDF version (430 for MDF 4.3.0)
            if block.gd_version > 430 {
                warn!(
                    "GDBLOCK requires MDF version {} which is not supported (max 430). Skipping guarded data.",
                    block.gd_version
                );
                position = current_pos;
            } else {
                // Follow the guarded link to the actual data block
                rdr.seek_relative(block.gd_link - current_pos)
                    .context("Could not reach guarded block from GDBLOCK")?;
                position = block.gd_link;

                // Read the id of the guarded block
                let mut guarded_id = [0u8; 4];
                rdr.read_exact(&mut guarded_id)
                    .context("could not read guarded block id")?;

                // Recursively read the guarded data block
                position = read_data(rdr, guarded_id, dg, position, sorted, decoder)
                    .context("failed reading guarded data block from GDBLOCK")?;
            }
        }
        _ => bail!("Unknown data block type {id:?}"), // should never happen
    }
    Ok(position)
}

/// Maximum uncompressed bytes per parallel decompression batch for `##DL`/`##LD` chains.
const MAX_DECOMP_BATCH_BYTES: usize = 32 * 1024 * 1024;

#[inline]
fn raw_block_est_len(raw: &RawCompBlock<'_>) -> usize {
    match raw {
        RawCompBlock::Plain(s) => s.len(),
        RawCompBlock::Dz(h, s) => (h.dz_org_data_length as usize).max(s.len()),
    }
}

#[inline]
fn raw_block_slice<'a>(raw: &RawCompBlock<'a>) -> &'a [u8] {
    match raw {
        RawCompBlock::Plain(s) | RawCompBlock::Dz(_, s) => s,
    }
}

fn take_raw_batch<T>(
    iter: &mut std::iter::Peekable<std::vec::IntoIter<T>>,
    est_len: impl Fn(&T) -> usize,
) -> Vec<T> {
    let mut batch = Vec::new();
    let mut batch_est = 0usize;
    while let Some(next_item) = iter.peek() {
        let est = est_len(next_item);
        if !batch.is_empty() && batch_est.saturating_add(est) > MAX_DECOMP_BATCH_BYTES {
            break;
        }
        batch_est += est;
        batch.push(iter.next().unwrap());
    }
    batch
}

fn single_channel_sample_bytes(cn: &Cn4) -> usize {
    let elem_count = if matches!(cn.block.cn_data_type, 15 | 16) {
        cn.list_size / 2
    } else {
        cn.list_size
    };
    cn.n_bytes as usize * elem_count
}

fn stream_single_channel_chunk(
    cn: &mut Cn4,
    block_data: &[u8],
    remainder: &mut Vec<u8>,
    sample_idx: &mut usize,
    cycle_count: usize,
    sample_bytes: usize,
) -> Result<()> {
    if sample_bytes == 0 || *sample_idx >= cycle_count {
        return Ok(());
    }
    let mut rest = block_data;
    if !remainder.is_empty() {
        let need = sample_bytes - remainder.len();
        if rest.len() < need {
            remainder.extend_from_slice(rest);
            return Ok(());
        }
        remainder.extend_from_slice(&rest[..need]);
        read_one_channel_array_chunk(remainder, cn, *sample_idx, 1)?;
        *sample_idx += 1;
        remainder.clear();
        rest = &rest[need..];
    }
    let avail = rest.len() / sample_bytes;
    let take = avail.min(cycle_count - *sample_idx);
    if take > 0 {
        read_one_channel_array_chunk(&rest[..take * sample_bytes], cn, *sample_idx, take)?;
        *sample_idx += take;
    }
    if *sample_idx < cycle_count {
        let tail = &rest[take * sample_bytes..];
        if !tail.is_empty() {
            remainder.extend_from_slice(tail);
        }
    }
    Ok(())
}

/// Collects raw (plain slice or compressed DZ) entries from a chain of `Dl4Block`s.
fn collect_dl_raw_entries<'a>(
    rdr: &mut SliceCursor<'a>,
    dl_blocks: &[Dl4Block],
    mut position: i64,
) -> Result<(Vec<RawCompBlock<'a>>, i64)> {
    let total: usize = dl_blocks.iter().map(|dl| dl.dl_data.len()).sum();
    let mut raw_entries: Vec<RawCompBlock<'a>> = Vec::with_capacity(total);
    for dl in dl_blocks {
        for &data_ptr in &dl.dl_data {
            if data_ptr == 0 {
                continue;
            }
            rdr.seek_relative(data_ptr - position)?;
            let mut blk_id = [0u8; 4];
            rdr.read_exact(&mut blk_id)?;
            if blk_id == *b"##DZ" {
                let (h, b) = read_dz_raw(rdr)?;
                position = data_ptr + h.len as i64;
                raw_entries.push(RawCompBlock::Dz(h, b));
            } else {
                let block_header: Dt4Block = binrw::BinReaderExt::read_le(rdr)?;
                let file_size = rdr.file_size().unwrap_or(u64::MAX);
                let data_len = if block_header.len >= 24 {
                    let claimed = block_header.len - 24;
                    let available = (file_size as i64).saturating_sub(data_ptr + 24);
                    if available > 0 {
                        claimed.min(available as u64) as usize
                    } else {
                        0
                    }
                } else {
                    warn!(
                        "DT block at 0x{data_ptr:x} has invalid len={} (unfinalized?), skipping",
                        block_header.len
                    );
                    0
                };
                let b = rdr.read_slice(data_len)?;
                position = data_ptr + 24 + data_len as i64;
                raw_entries.push(RawCompBlock::Plain(b));
            }
        }
    }
    Ok((raw_entries, position))
}

/// Decompresses collected `RawCompBlock` entries in parallel, borrowing plain slices zero-copy.
fn decompress_raw_blocks<'a>(
    raw_entries: Vec<RawCompBlock<'a>>,
    err_ctx: &'static str,
) -> Result<Vec<Cow<'a, [u8]>>> {
    raw_entries
        .into_par_iter()
        .map(|raw| match raw {
            RawCompBlock::Plain(d) => Ok(Cow::Borrowed(d)),
            RawCompBlock::Dz(h, b) => h.decompress(b).map(Cow::Owned).context(err_ctx),
        })
        .collect()
}

/// Concatenates decompressed `Cow` chunks into a single `Cow<[u8]>`, avoiding any copy when
/// there is only a single plain chunk.
fn concat_cow_chunks<'a>(chunks: Vec<Cow<'a, [u8]>>) -> Cow<'a, [u8]> {
    if chunks.len() == 1 {
        return chunks.into_iter().next().unwrap();
    }
    let total: usize = chunks.iter().map(|b| b.len()).sum();
    let mut all = Vec::with_capacity(total);
    for b in chunks {
        all.extend_from_slice(&b);
    }
    Cow::Owned(all)
}

/// Reads and concatenates data from any data block type (##DT, ##SD, ##VD, ##RD, ##RV, ##DZ, ##DL, ##HL).
/// The block id must already have been read. Returns borrowed/owned raw bytes and updated position.
fn read_all_blocks_to_bytes<'a>(
    rdr: &mut SliceCursor<'a>,
    id: [u8; 4],
    mut position: i64,
) -> Result<Option<(Cow<'a, [u8]>, i64)>> {
    // ##DT, ##SD, ##VD are regular data blocks; ##RD, ##RV are reduction data blocks (same format)
    if id == *b"##DT" || id == *b"##SD" || id == *b"##VD" || id == *b"##RD" || id == *b"##RV" {
        let block_header: Dt4Block =
            binrw::BinReaderExt::read_le(rdr).context("Could not read data block header")?;
        let slice = rdr
            .read_slice(block_header.len as usize - 24)
            .context("could not read data block buffer")?;
        position += block_header.len as i64;
        Ok(Some((Cow::Borrowed(slice), position)))
    } else if id == *b"##DZ" {
        let (buf, block_header) = parse_dz(rdr)?;
        position += block_header.len as i64;
        Ok(Some((Cow::Owned(buf), position)))
    } else if id == *b"##HL" || id == *b"##DL" {
        let current_pos = if id == *b"##HL" {
            let (pos, _id) = read_hl(rdr, position)?;
            pos
        } else {
            position
        };
        let (dl_blocks, pos) = parser_dl4(rdr, current_pos)?;
        let (raw_entries, pos) = collect_dl_raw_entries(rdr, &dl_blocks, pos)?;
        let chunks = decompress_raw_blocks(raw_entries, "failed decompressing DZ block in DL")?;
        Ok(Some((concat_cow_chunks(chunks), pos)))
    } else {
        Ok(None)
    }
}

/// Header List block reader
/// This HL Block references Data List Blocks that are listing DZ Blocks
/// It is existing to add complementary information about compression in DZ
fn read_hl(rdr: &mut SliceCursor<'_>, mut position: i64) -> Result<(i64, [u8; 4])> {
    // compressed data in datal list
    let block: Hl4Block = binrw::BinReaderExt::read_le(rdr).context("could not read HL block")?;
    position += block.hl_len as i64;
    // Read Id of pointed DL Block
    rdr.seek_relative(block.hl_dl_first - position)
        .context("Could not reach DL block from HL block")?;
    position = block.hl_dl_first;
    let mut id = [0u8; 4];
    rdr.read_exact(&mut id)
        .context("could not read DL block id")?;
    Ok((position, id))
}

/// Reads VLSD data from a chain of DL sub-blocks (##SD or ##DZ) without reinitialising arrays.
/// Used by read_sd to process DL-chained VLSD data for a single channel.
fn read_vlsd_from_dl_blocks(
    rdr: &mut SliceCursor<'_>,
    dl_blocks: Vec<Dl4Block>,
    mut position: i64,
    cn: &mut Cn4,
    decoder: &mut Dec,
) -> Result<i64> {
    let mut previous_index: usize = 0;
    for dl in dl_blocks {
        for data_pointer in dl.dl_data {
            if data_pointer == 0 {
                continue;
            }
            rdr.seek_relative(data_pointer - position)
                .context("Could not reach VLSD sub-block from DL")?;
            let mut id = [0u8; 4];
            rdr.read_exact(&mut id)
                .context("could not read VLSD sub-block id")?;
            if id == *b"##DZ" {
                let (dt, block_header) = parse_dz(rdr)?;
                position = data_pointer + block_header.len as i64;
                let (idx, _) = read_vlsd_from_slice(&dt, cn, previous_index, decoder)?;
                previous_index = idx;
            } else {
                // ##SD block (same header layout as Dt4Block)
                let block_header: Dt4Block = binrw::BinReaderExt::read_le(rdr)
                    .context("Could not read VLSD sub-block header")?;
                let slice = rdr
                    .read_slice((block_header.len - 24) as usize)
                    .context("Could not read VLSD sub-block data")?;
                position = data_pointer + block_header.len as i64;
                let (idx, _) = read_vlsd_from_slice(slice, cn, previous_index, decoder)?;
                previous_index = idx;
            }
        }
    }
    Ok(position)
}

/// Reads Signal Data Block containing VLSD channel, pointed by cn_data
fn read_sd(
    rdr: &mut SliceCursor<'_>,
    dg: &mut Dg4,
    vlsd_channels: &[(u8, i64)],
    mut position: i64,
    decoder: &mut Dec,
) -> Result<i64> {
    for channel_group in dg.cg.values_mut() {
        for (cn_type, rec_pos) in vlsd_channels {
            // Only process VLSD channels (cn_type == 1)
            if *cn_type != 1 {
                continue;
            }
            if let Some(cn) = channel_group.cn.get_mut(rec_pos) {
                // header block
                rdr.seek_relative(cn.block.cn_data - position)
                    .context("Could not position buffer")?; // change buffer position
                position = cn.block.cn_data;
                let mut id = [0u8; 4];
                rdr.read_exact(&mut id).context("could not read block id")?;
                if "##SD".as_bytes() == id {
                    // SD (Signal Data) block - each value prefixed with u32 length
                    let block_header: Dt4Block = binrw::BinReaderExt::read_le(rdr)
                        .context("Could not read SD block struct")?;
                    let data = rdr
                        .read_slice(block_header.len as usize - 24)
                        .context("could not read SD data buffer")?;
                    position += block_header.len as i64;
                    read_vlsd_from_slice(data, cn, 0, decoder)?;
                } else if "##DZ".as_bytes() == id {
                    let (data, block_header) = parse_dz(rdr)?;
                    position += block_header.len as i64;
                    read_vlsd_from_slice(&data, cn, 0, decoder)?;
                } else if "##HL".as_bytes() == id {
                    let (pos, _id) = read_hl(rdr, position)?;
                    position = pos;
                    let (dl_blocks, pos) = parser_dl4(rdr, position)?;
                    position = read_vlsd_from_dl_blocks(rdr, dl_blocks, pos, cn, decoder)?;
                } else if "##DL".as_bytes() == id {
                    let (dl_blocks, pos) = parser_dl4(rdr, position)?;
                    position = read_vlsd_from_dl_blocks(rdr, dl_blocks, pos, cn, decoder)?;
                } else {
                    eprintln!(
                        "read_sd: UNEXPECTED ID: {:?} at cn_data={}",
                        String::from_utf8_lossy(&id),
                        cn.block.cn_data
                    );
                }
                if let Some(b) = cn.builder.take() {
                    cn.data = b.finish();
                }
            }
        }
    }
    Ok(position)
}

/// Reads Variable Data Block containing VLSC channel data (cn_type = 7)
/// The DT/DZ block has already been read, so:
/// - offsets are in the VLSC channel cn.data (read as UInt64)
/// - sizes are in the cn_cn_size channel cn.data (read as UInt8/16/32/64)
/// - VD block (pointed by cn.block.cn_data) contains raw data without length prefixes
fn read_vd(
    rdr: &mut SliceCursor<'_>,
    channel_group: &mut Cg4,
    vlsc_channels: &[(u8, i64)],
    mut position: i64,
    decoder: &mut Dec,
) -> Result<i64> {
    for (cn_type, rec_pos) in vlsc_channels {
        // Only process VLSC channels (cn_type == 7)
        if *cn_type != 7 {
            continue;
        }
        // First pass: get offsets, sizes and cn_data position (immutable borrow)
        let (offsets, sizes, cn_data, cn_data_type) = {
            let cn = match channel_group.cn.get(rec_pos) {
                Some(cn) => cn,
                None => continue,
            };

            let cn_data = cn.block.cn_data;
            if cn_data == 0 {
                continue;
            }

            // Get offsets from the VLSC channel data
            let offsets: Vec<u64> = cn.to_u64_vec().unwrap_or_default();

            if offsets.is_empty() {
                continue;
            }

            // Get sizes from cn_cn_size channel
            let sizes: Vec<u64> = cn
                .block
                .cn_cn_size()
                .and_then(|size_pos| {
                    channel_group
                        .cn
                        .values()
                        .find(|cn| cn.block_position == size_pos)
                        .and_then(|size_cn| size_cn.to_u64_vec())
                })
                .unwrap_or_default();

            if sizes.is_empty() {
                continue;
            }

            (offsets, sizes, cn_data, cn.block.cn_data_type)
        };

        // Second pass: read VD block and update cn.data (mutable borrow)
        rdr.seek_relative(cn_data - position)
            .context("Could not position buffer for VD block")?;
        position = cn_data;

        let mut id = [0u8; 4];
        rdr.read_exact(&mut id)
            .context("could not read VD block id")?;

        let data: Cow<'_, [u8]> = match read_all_blocks_to_bytes(rdr, id, position)? {
            Some((buf, pos)) => {
                position = pos;
                buf
            }
            None => continue,
        };

        // Now update cn.data with the actual variable length data
        if let Some(cn) = channel_group.cn.get_mut(rec_pos) {
            // Reinitialize cn.data for the actual variable length data
            // cn_data_type: 6=SBC, 7=UTF-8, 8=UTF-16 LE, 9=UTF-16 BE, 10=byte array, 17=UTF-8 with BOM
            cn.builder = match cn_data_type {
                6..=9 | 17 => Some(Box::new(ChannelBuilder::Utf8(
                    arrow::array::LargeStringBuilder::new(),
                ))),
                10 => Some(Box::new(ChannelBuilder::VariableSizeByteArray(
                    arrow::array::LargeBinaryBuilder::new(),
                ))),
                _ => continue,
            };

            read_vlsc_from_bytes(&data, cn, &offsets, &sizes, decoder)?;
            if let Some(b) = cn.builder.take() {
                cn.data = b.finish();
            }
        }
    }
    Ok(position)
}

/// Reads Variable Length Signal Data from a slice of an SD block.
/// Returns `(new_index, bytes_consumed)`.
fn read_vlsd_from_slice(
    data: &[u8],
    cn: &mut Cn4,
    previous_index: usize,
    decoder: &mut Dec,
) -> Result<(usize, usize)> {
    let mut position: usize = 0;
    let data_length = data.len();
    let mut remaining: usize = data_length;
    let mut nrecord: usize = 0;
    let mut str_buf = String::new();
    if cn.builder.is_none() {
        cn.init_builder(0)?;
    }
    let builder = cn.builder.as_deref_mut().unwrap();
    match builder {
        ChannelBuilder::Utf8(array) => {
            let cn_data_type = cn.block.cn_data_type;
            while remaining >= 4 {
                let len = &data[position..position + std::mem::size_of::<u32>()];
                let length: usize =
                    u32::from_le_bytes(len.try_into().context("Could not read length")?) as usize;
                if position.saturating_add(4).saturating_add(length) <= data_length {
                    position += std::mem::size_of::<u32>();
                    // From MDF 4.3, null terminator is optional in VLSD strings.
                    // Strip trailing \0 only if actually present (check the last byte).
                    let record_len = match cn_data_type {
                        6 | 7 => {
                            if length > 0 && data[position + length - 1] == 0 {
                                length - 1
                            } else {
                                length
                            }
                        }
                        _ => length,
                    };
                    let record = &data[position..position + record_len];
                    array.append_value(decode_string_bytes(
                        record,
                        cn_data_type,
                        decoder,
                        &mut str_buf,
                    )?);
                    position += length;
                    remaining = data_length - position;
                    nrecord += 1;
                } else {
                    break;
                }
            }
        }
        ChannelBuilder::VariableSizeByteArray(array) => {
            while remaining >= 4 {
                let len = &data[position..position + std::mem::size_of::<u32>()];
                let length: usize =
                    u32::from_le_bytes(len.try_into().context("Could not read length")?) as usize;
                if position.saturating_add(4).saturating_add(length) <= data_length {
                    position += std::mem::size_of::<u32>();
                    let record = &data[position..position + length];
                    array.append_value(record);
                    position += length;
                    remaining = data_length - position;
                    nrecord += 1;
                } else {
                    break;
                }
            }
        }
        _ => {}
    }
    Ok((nrecord + previous_index, position))
}

/// Reads Variable Length Signal Data from bytes of a SD Block, compacting any incomplete tail.
/// It shall contain data of only one VLSD channel.
/// Each record is starting from its length headed by a u32.
fn read_vlsd_from_bytes(
    data: &mut Vec<u8>,
    cn: &mut Cn4,
    previous_index: usize,
    decoder: &mut Dec,
) -> Result<usize> {
    let (new_index, consumed) = read_vlsd_from_slice(data, cn, previous_index, decoder)?;
    let remaining = data.len().saturating_sub(consumed);
    if remaining > 0 {
        data.copy_within(consumed.., 0);
        data.truncate(remaining);
    } else {
        data.clear();
    }
    Ok(new_index)
}

/// Reads Variable Length Signal data with Size Channel from bytes of a VD Block
/// Unlike VLSD (SD Block), the sizes come from a separate size channel in the record triplet (time, size, offset)
/// offsets: array of offsets into the VD block data for each record
/// sizes: array of sizes (in bytes) for each record value
/// Returns the maximum position reached in the data buffer (max of offset + size)
fn read_vlsc_from_bytes(
    data: &[u8],
    cn: &mut Cn4,
    offsets: &[u64],
    sizes: &[u64],
    decoder: &mut Dec,
) -> Result<usize> {
    let data_length = data.len();
    let mut max_position: usize = 0;
    let mut str_buf = String::new();
    let builder = match cn.builder.as_deref_mut() {
        Some(b) => b,
        None => return Ok(0),
    };
    match builder {
        ChannelBuilder::Utf8(array) => {
            let cn_data_type = cn.block.cn_data_type;
            for (offset, size) in offsets.iter().zip(sizes.iter()) {
                let start = *offset as usize;
                let length = *size as usize;
                if start + length <= data_length && length > 0 {
                    let record = &data[start..start + length];
                    array.append_value(decode_string_bytes(
                        record,
                        cn_data_type,
                        decoder,
                        &mut str_buf,
                    )?);
                    max_position = max_position.max(start + length);
                } else if length == 0 {
                    array.append_value("");
                } else {
                    array.append_null();
                }
            }
        }
        ChannelBuilder::VariableSizeByteArray(array) => {
            for (offset, size) in offsets.iter().zip(sizes.iter()) {
                let start = *offset as usize;
                let length = *size as usize;
                if start + length <= data_length {
                    let record = &data[start..start + length];
                    array.append_value(record);
                    max_position = max_position.max(start + length);
                } else {
                    array.append_null();
                }
            }
        }
        _ => {}
    }
    Ok(max_position)
}

/// Reads independent column-oriented data linked directly from CNBLOCK (`cn_data`).
/// Supports LD4, DV4/DT4, DZ4, DL4, and HL4 blocks per MDF 4.3.0 §5.4 & §6.6.2.
fn read_independent_channel_data<'a>(
    rdr: &mut SliceCursor<'a>,
    cn: &mut Cn4,
    cycle_count: usize,
    mut position: i64,
) -> Result<i64> {
    let cn_data = cn.block.cn_data;
    if cn_data == 0 || cycle_count == 0 {
        return Ok(position);
    }

    rdr.seek_relative(cn_data - position)
        .context("Could not reach independent channel data position from cn_data")?;
    position = cn_data;
    let mut id = [0u8; 4];
    rdr.read_exact(&mut id)
        .context("Could not read block ID at cn_data")?;

    if id == *b"##LD" {
        let mut ld_blocks: Vec<Ld4Block> = Vec::new();
        let (block, pos) = parser_ld4_block(rdr, position, position)?;
        position = pos;
        let mut next_ld = block.ld_ld_next();
        ld_blocks.push(block);
        while next_ld > 0 {
            rdr.seek_relative(next_ld - position)
                .context("Could not reach next LD block position")?;
            position = next_ld;
            let mut ld_id = [0u8; 4];
            rdr.read_exact(&mut ld_id)
                .context("Could not read LD block ID")?;
            let (blk, pos) = parser_ld4_block(rdr, position, position)?;
            position = pos;
            next_ld = blk.ld_ld_next();
            ld_blocks.push(blk);
        }

        let mut raw_data: Vec<RawCompBlock<'a>> = Vec::new();
        let mut raw_invalid: Vec<Option<RawCompBlock<'a>>> = Vec::new();
        for ld in &ld_blocks {
            for &data_ptr in &ld.ld_data() {
                rdr.seek_relative(data_ptr - position)
                    .context("Could not reach DV/DZ block position in independent LD")?;
                let mut blk_id = [0u8; 4];
                rdr.read_exact(&mut blk_id)?;
                if blk_id == *b"##DZ" {
                    let (hdr, buf) = read_dz_raw(rdr)?;
                    position = data_ptr + hdr.len as i64;
                    raw_data.push(RawCompBlock::Dz(hdr, buf));
                } else {
                    let hdr: Dt4Block = binrw::BinReaderExt::read_le(rdr)?;
                    let buf = rdr.read_slice((hdr.len - 24) as usize)?;
                    position = data_ptr + hdr.len as i64;
                    raw_data.push(RawCompBlock::Plain(buf));
                }
            }
            for &inval_ptr in &ld.ld_invalid_data() {
                if inval_ptr <= 0 {
                    raw_invalid.push(None);
                    continue;
                }
                rdr.seek_relative(inval_ptr - position)
                    .context("Could not reach DI/DZ invalid block in independent LD")?;
                let mut blk_id = [0u8; 4];
                rdr.read_exact(&mut blk_id)?;
                if blk_id == *b"##DZ" {
                    let (hdr, buf) = read_dz_raw(rdr)?;
                    position = inval_ptr + hdr.len as i64;
                    raw_invalid.push(Some(RawCompBlock::Dz(hdr, buf)));
                } else {
                    let hdr: Dt4Block = binrw::BinReaderExt::read_le(rdr)?;
                    let buf = rdr.read_slice((hdr.len - 24) as usize)?;
                    position = inval_ptr + hdr.len as i64;
                    raw_invalid.push(Some(RawCompBlock::Plain(buf)));
                }
            }
        }

        let sample_bytes = single_channel_sample_bytes(cn);
        let mut chunk_sample_counts: Vec<usize> = Vec::with_capacity(raw_data.len());
        let mut remainder: Vec<u8> = Vec::new();
        let mut sample_idx = 0usize;
        let mut raw_iter = raw_data.into_iter().peekable();
        while raw_iter.peek().is_some() {
            let batch = take_raw_batch(&mut raw_iter, raw_block_est_len);
            let advise_slices: Vec<&'a [u8]> = batch.iter().map(raw_block_slice).collect();
            let decompressed =
                decompress_raw_blocks(batch, "failed decompressing DV DZ block in independent LD")?;
            for block_data in decompressed {
                chunk_sample_counts.push(block_data.len().checked_div(sample_bytes).unwrap_or(0));
                stream_single_channel_chunk(
                    cn,
                    &block_data,
                    &mut remainder,
                    &mut sample_idx,
                    cycle_count,
                    sample_bytes,
                )?;
            }
            for s in advise_slices {
                rdr.advise_dontneed_slice(s);
            }
        }
        if let Some(b) = cn.builder.take() {
            cn.data = b.finish();
        }

        if !raw_invalid.is_empty()
            && raw_invalid.iter().any(|r| r.is_some())
            && let Some(mask_state) = &mut cn.invalid_mask
        {
            let (_invalid_byte_pos, invalid_byte_mask) = mask_state.byte_pos_and_mask();
            // invalid_byte_mask == 0 means CN_F_ALL_VALUES_INVALID (pre-filled false); keep as-is.
            if invalid_byte_mask != 0 {
                let (mask, _, _) = mask_state.materialize_mut(cycle_count);
                let mut sample_offset = 0usize;
                let mut chunk_idx = 0usize;
                let mut inv_iter = raw_invalid.into_iter().peekable();
                while inv_iter.peek().is_some() {
                    let batch =
                        take_raw_batch(&mut inv_iter, |r| r.as_ref().map_or(0, raw_block_est_len));
                    let advise_slices: Vec<&'a [u8]> = batch
                        .iter()
                        .filter_map(|r| r.as_ref().map(raw_block_slice))
                        .collect();
                    let decomp_inval: Vec<Option<Cow<'a, [u8]>>> = batch
                        .into_par_iter()
                        .map(|r| match r {
                            None => Ok(None),
                            Some(RawCompBlock::Plain(d)) => Ok(Some(Cow::Borrowed(d))),
                            Some(RawCompBlock::Dz(h, b)) => h
                                .decompress(b)
                                .map(|v| Some(Cow::Owned(v)))
                                .context("failed decompressing DI DZ block in independent LD"),
                        })
                        .collect::<Result<_>>()?;
                    for opt_inval in decomp_inval {
                        let chunk_samples = chunk_sample_counts
                            .get(chunk_idx)
                            .copied()
                            .unwrap_or(cycle_count.saturating_sub(sample_offset));
                        if let Some(inval_bytes) = opt_inval {
                            for local_idx in 0..chunk_samples {
                                let s_idx = sample_offset + local_idx;
                                if s_idx >= mask.len() {
                                    break;
                                }
                                let byte_idx = local_idx / 8;
                                let bit_idx = local_idx % 8;
                                if byte_idx < inval_bytes.len() {
                                    let is_invalid = (inval_bytes[byte_idx] & (1 << bit_idx)) != 0;
                                    mask.set_bit(s_idx, !is_invalid);
                                }
                            }
                        }
                        sample_offset = sample_offset.saturating_add(chunk_samples);
                        chunk_idx += 1;
                    }
                    for s in advise_slices {
                        rdr.advise_dontneed_slice(s);
                    }
                }
                cn.data.set_validity(mask)?;
            }
            mask_state.clear_builder();
        }
    } else if id == *b"##DV" || id == *b"##DT" {
        let hdr: Dt4Block = binrw::BinReaderExt::read_le(rdr)
            .context("Could not read independent DV/DT block header")?;
        let slice = rdr
            .read_slice((hdr.len - 24) as usize)
            .context("Could not read independent DV/DT block data")?;
        position = cn_data + hdr.len as i64;
        read_one_channel_array(slice, cn, cycle_count)?;
        rdr.advise_dontneed_slice(slice);
    } else if id == *b"##DZ" {
        let (dt, hdr) = parse_dz(rdr)?;
        position = cn_data + hdr.len as i64;
        read_one_channel_array(&dt, cn, cycle_count)?;
    } else if id == *b"##DL" || id == *b"##HL" {
        if id == *b"##HL" {
            let (pos, _id) = read_hl(rdr, position)?;
            position = pos;
        }
        let (dl_blocks, pos) = parser_dl4(rdr, position)?;
        let (raw_entries, pos) = collect_dl_raw_entries(rdr, &dl_blocks, pos)?;
        position = pos;
        let sample_bytes = single_channel_sample_bytes(cn);
        let mut remainder: Vec<u8> = Vec::new();
        let mut sample_idx = 0usize;
        let mut raw_iter = raw_entries.into_iter().peekable();
        while raw_iter.peek().is_some() {
            let batch = take_raw_batch(&mut raw_iter, raw_block_est_len);
            let advise_slices: Vec<&'a [u8]> = batch.iter().map(raw_block_slice).collect();
            let decompressed =
                decompress_raw_blocks(batch, "failed decompressing DZ block in independent DL/HL")?;
            for block_data in decompressed {
                stream_single_channel_chunk(
                    cn,
                    &block_data,
                    &mut remainder,
                    &mut sample_idx,
                    cycle_count,
                    sample_bytes,
                )?;
            }
            for s in advise_slices {
                rdr.advise_dontneed_slice(s);
            }
        }
        if let Some(b) = cn.builder.take() {
            cn.data = b.finish();
        }
    } else {
        position += 4;
    }
    Ok(position)
}

/// Reads array channel stored as DG template (ca_storage == 2).
/// Each element k has its data block pointed by ca_data[k].
fn read_dg_template_array<'a>(
    rdr: &mut SliceCursor<'a>,
    cn: &mut Cn4,
    ca: &Ca4Block,
    cycle_count: usize,
    mut position: i64,
) -> Result<i64> {
    let ca_data_links = match &ca.ca_data {
        Some(links) if !links.is_empty() => links,
        _ => return Ok(position),
    };
    if cycle_count == 0 {
        return Ok(position);
    }
    let m = ca_data_links.len();
    let elem_size = cn.n_bytes as usize;
    if elem_size == 0 {
        return Ok(position);
    }

    let mut element_data: Vec<Cow<'a, [u8]>> = Vec::with_capacity(m);
    for (k, &ptr) in ca_data_links.iter().enumerate() {
        if ptr == 0 {
            // No data recorded for this element
            element_data.push(Cow::Owned(vec![0u8; cycle_count * elem_size]));
            continue;
        }
        rdr.seek_relative(ptr - position)
            .context("Could not seek to ca_data block")?;
        position = ptr;
        let mut id = [0u8; 4];
        rdr.read_exact(&mut id)
            .context("Could not read ca_data block ID")?;
        let elem_cycle_count = ca
            .ca_cycle_count
            .as_ref()
            .and_then(|c| c.get(k))
            .copied()
            .map(|c| c as usize)
            .unwrap_or(cycle_count);

        if id == *b"##DZ" {
            let (dt, hdr) = parse_dz(rdr)?;
            position = ptr + hdr.len as i64;
            element_data.push(Cow::Owned(dt));
        } else if id == *b"##DT" || id == *b"##DV" {
            let hdr: Dt4Block = binrw::BinReaderExt::read_le(rdr)?;
            let slice = rdr.read_slice((hdr.len - 24) as usize)?;
            position = ptr + hdr.len as i64;
            element_data.push(Cow::Borrowed(slice));
        } else if id == *b"##DL" || id == *b"##HL" {
            if id == *b"##HL" {
                let (pos, _) = read_hl(rdr, position)?;
                position = pos;
            }
            let (dl_blocks, pos) = parser_dl4(rdr, position)?;
            let (raw_entries, pos) = collect_dl_raw_entries(rdr, &dl_blocks, pos)?;
            position = pos;
            let decomp =
                decompress_raw_blocks(raw_entries, "failed decompressing DZ block in CA template")?;
            element_data.push(concat_cow_chunks(decomp));
        } else {
            // Unknown or unsupported block type
            element_data.push(Cow::Owned(vec![0u8; elem_cycle_count * elem_size]));
        }
    }

    // Interleave element columns into sample-major tensor order
    let mut interleaved = vec![0u8; cycle_count * m * elem_size];
    for (k, elem_buf) in element_data.into_iter().enumerate() {
        for t in 0..cycle_count {
            let src_start = t * elem_size;
            let dst_start = (t * m + k) * elem_size;
            if src_start + elem_size <= elem_buf.len() && dst_start + elem_size <= interleaved.len()
            {
                interleaved[dst_start..dst_start + elem_size]
                    .copy_from_slice(&elem_buf[src_start..src_start + elem_size]);
            }
        }
    }

    read_one_channel_array(&interleaved, cn, cycle_count)?;
    Ok(position)
}

/// Reads all DL Blocks and returns a vect of them
fn parser_ld4(
    rdr: &mut SliceCursor<'_>,
    mut position: i64,
    channel_group: &mut Cg4,
) -> Result<i64> {
    let mut ld_blocks: Vec<Ld4Block> = Vec::new();
    let mut chain = BlockChain::new("LD");
    chain.visit(position);
    let (block, pos) = parser_ld4_block(rdr, position, position)?;
    position = pos;
    ld_blocks.push(block.clone());
    let mut next_ld = block.ld_ld_next();
    while next_ld > 0 {
        if !chain.visit(next_ld) {
            break;
        }
        rdr.seek_relative(next_ld - position)
            .context("Could not reach LD block position")?;
        position = next_ld;
        let mut id = [0u8; 4];
        rdr.read_exact(&mut id)
            .context("could not read LD block id")?;
        let (block, pos) =
            parser_ld4_block(rdr, position, position).context("failed parsing ld4 block")?;
        position = pos;
        ld_blocks.push(block.clone());
        next_ld = block.ld_ld_next();
    }
    if ld_blocks.len() == 1 && ld_blocks[0].ld_data().len() == 1 && channel_group.cn.len() == 1 {
        // only one DV block, reading can be optimised
        // Reads DV or DZ block id
        let ld_data = ld_blocks[0].ld_data()[0];
        rdr.seek_relative(ld_data - position)
            .context("Could not reach DV or DZ block position from LD block")?;
        let mut id = [0u8; 4];
        rdr.read_exact(&mut id)
            .context("could not read data block id from ld4 invalid")?;
        initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone())
            .context("failed initialising arrays")?;
        if id == "##DZ".as_bytes() {
            let (dt, block_header) =
                parse_dz(rdr).context("failed parsing dz block pointed by ld4 block")?;
            for cn in channel_group.cn.values_mut() {
                read_one_channel_array(&dt, cn, channel_group.block.cg_cycle_count as usize)
                    .context("failed reading one channel array from DZ")?;
            }
            position = ld_data + block_header.len as i64;
        } else {
            let block_header: Dt4Block =
                binrw::BinReaderExt::read_le(rdr).context("Could not read DV block header")?;
            let slice = rdr
                .read_slice(block_header.len as usize - 24)
                .context("Could not read Dt4 block")?;
            for cn in channel_group.cn.values_mut() {
                read_one_channel_array(slice, cn, channel_group.block.cg_cycle_count as usize)
                    .context("failed reading one channel array")?;
            }
            position = ld_data + block_header.len as i64;
        }
        // For VLSD/VLSC, cg_inval_bytes is the high part of VL data size, not invalidation bytes
        if channel_group.block.cg_inval_bytes > 0
            && (channel_group.block.cg_flags & (CG_F_VLSD | CG_F_VLSC)) == 0
        {
            // Reads invalid DI or DZ block (0 = NIL pointer per MDF 4.3.0 §6.36 -> all samples valid)
            let ld_invalid_data_vec = ld_blocks[0].ld_invalid_data();
            let ld_invalid_data = if !ld_invalid_data_vec.is_empty() {
                ld_invalid_data_vec[0]
            } else {
                bail!("no invalid block (di or dz) pointer found in ld4 block")
            };
            if ld_invalid_data > 0 {
                rdr.seek_relative(ld_invalid_data - position)
                    .context("Could not reach DI or DZ block position")?;
                let mut id = [0u8; 4];
                rdr.read_exact(&mut id)
                    .context("could not read data block id from ld4 invalid")?;
                if id == "##DZ".as_bytes() {
                    let (dt, block_header) = parse_dz(rdr)?;
                    channel_group.invalid_bytes = Some(dt);
                    position = ld_invalid_data + block_header.len as i64;
                } else {
                    let block_header: Dt4Block = binrw::BinReaderExt::read_le(rdr)
                        .context("Could not read into DZ or DI block header")?;
                    let slice = rdr
                        .read_slice((block_header.len - 24) as usize)
                        .context("Could not read data block")?;
                    channel_group.invalid_bytes = Some(slice.to_vec());
                    position = ld_invalid_data + block_header.len as i64;
                }
            }
        }
    } else {
        // several DV, LD or channels per DG
        position = read_dv_di(rdr, position, channel_group, ld_blocks)?;
    }
    Ok(position)
}

/// reads DV and DI block containing several channels.
///
/// Three-phase approach: (1) zero-copy slice / raw DZ collection, (2) parallel decompression,
/// (3) sequential record assembly — mirrors `parser_dl4_sorted`.
fn read_dv_di<'a>(
    rdr: &mut SliceCursor<'a>,
    mut position: i64,
    channel_group: &mut Cg4,
    ld_blocks: Vec<Ld4Block>,
) -> Result<i64, Error> {
    let cg_cycle_count = channel_group.block.cg_cycle_count as usize;
    // For VLSD/VLSC, cg_inval_bytes is the high part of VL data size, not invalidation bytes
    let cg_inval_bytes = if (channel_group.block.cg_flags & (CG_F_VLSD | CG_F_VLSC)) != 0 {
        0
    } else {
        channel_group.block.cg_inval_bytes as usize
    };
    initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone())
        .context("failed initialising arrays for dv di blocks")?;
    if cg_inval_bytes > 0
        && ld_blocks
            .iter()
            .any(|ld| ld.ld_invalid_data().iter().any(|&p| p > 0))
    {
        channel_group.invalid_bytes = Some(vec![0u8; cg_inval_bytes * cg_cycle_count]);
    }

    // Phase 1: Zero-copy slice collection of all raw data and invalid blocks
    let total_data_blocks: usize = ld_blocks.iter().map(|ld| ld.ld_data().len()).sum();
    let total_invalid_blocks: usize = ld_blocks.iter().map(|ld| ld.ld_invalid_data().len()).sum();
    let mut raw_data: Vec<RawCompBlock<'a>> = Vec::with_capacity(total_data_blocks);
    let mut raw_invalid: Vec<Option<RawCompBlock<'a>>> = Vec::with_capacity(total_invalid_blocks);
    for ld in &ld_blocks {
        for &data_pointer in &ld.ld_data() {
            rdr.seek_relative(data_pointer - position)
                .context("Could not reach DV or DZ block position")?;
            let mut id = [0u8; 4];
            rdr.read_exact(&mut id)
                .context("could not read data block id from LD4")?;
            if id == *b"##DZ" {
                let (header, buf) = read_dz_raw(rdr)?;
                position = data_pointer + header.len as i64;
                raw_data.push(RawCompBlock::Dz(header, buf));
            } else {
                let block_header: Dt4Block = binrw::BinReaderExt::read_le(rdr)
                    .context("Could not read DV block structure")?;
                let slice = rdr
                    .read_slice((block_header.len - 24) as usize)
                    .context("Could not read DV data")?;
                position = data_pointer + block_header.len as i64;
                raw_data.push(RawCompBlock::Plain(slice));
            }
        }
        for &data_pointer in &ld.ld_invalid_data() {
            if data_pointer <= 0 {
                // NIL pointer in ld_invalid_data means all samples in the corresponding DV block are valid
                raw_invalid.push(None);
                continue;
            }
            rdr.seek_relative(data_pointer - position)
                .context("Could not reach invalid block position")?;
            let mut id = [0u8; 4];
            rdr.read_exact(&mut id)
                .context("could not read data block id from ld4 invalid")?;
            if id == *b"##DZ" {
                let (header, buf) = read_dz_raw(rdr)?;
                position = data_pointer + header.len as i64;
                raw_invalid.push(Some(RawCompBlock::Dz(header, buf)));
            } else {
                let block_header: Dt4Block = binrw::BinReaderExt::read_le(rdr)
                    .context("Could not read invalid block header")?;
                let slice = rdr
                    .read_slice((block_header.len - 24) as usize)
                    .context("Could not read invalid data")?;
                position = data_pointer + block_header.len as i64;
                raw_invalid.push(Some(RawCompBlock::Plain(slice)));
            }
        }
    }

    // Phase 2 & 3: Bounded-batch parallel decompression + sequential record assembly
    let record_length = channel_group.block.cg_data_bytes as usize;
    let is_single_channel = channel_group.cn.len() == 1;
    let mut dv_record_counts: Vec<usize> = Vec::with_capacity(raw_data.len());
    let mut data: Vec<u8> = Vec::new();
    let mut previous_index: usize = 0;

    let mut raw_iter = raw_data.into_iter().peekable();
    while raw_iter.peek().is_some() {
        let batch = take_raw_batch(&mut raw_iter, raw_block_est_len);
        let advise_slices: Vec<&'a [u8]> = batch.iter().map(raw_block_slice).collect();
        let decompressed_data = decompress_raw_blocks(batch, "failed decompressing DV DZ block")?;

        for block_data in &decompressed_data {
            let n_rec = block_data.len().checked_div(record_length).unwrap_or(0);
            dv_record_counts.push(n_rec);
        }

        if is_single_channel {
            let cn = channel_group.cn.values_mut().next().unwrap();
            let sample_bytes = single_channel_sample_bytes(cn);
            for block_data in decompressed_data {
                stream_single_channel_chunk(
                    cn,
                    &block_data,
                    &mut data,
                    &mut previous_index,
                    cg_cycle_count,
                    sample_bytes,
                )?;
            }
        } else {
            for block_data in decompressed_data {
                if previous_index >= cg_cycle_count {
                    break;
                }
                if data.is_empty() {
                    let block_length = block_data.len();
                    let n_record_chunk = block_length.checked_div(record_length).unwrap_or(0);
                    if n_record_chunk > 0 {
                        let records_to_read = n_record_chunk.min(cg_cycle_count - previous_index);
                        read_channels_from_bytes(
                            &block_data[..record_length * records_to_read],
                            &mut channel_group.cn,
                            record_length,
                            previous_index,
                            false,
                        )
                        .context("failed reading channels from dv di blocks")?;
                        previous_index += records_to_read;
                    }
                    let remaining = block_length % record_length;
                    if remaining > 0 {
                        data.extend_from_slice(&block_data[record_length * n_record_chunk..]);
                    }
                } else {
                    data.extend_from_slice(&block_data);
                    let block_length = data.len();
                    let n_record_chunk = block_length.checked_div(record_length).unwrap_or(0);
                    if n_record_chunk == 0 {
                        continue;
                    }
                    let records_to_read = n_record_chunk.min(cg_cycle_count - previous_index);
                    read_channels_from_bytes(
                        &data[..record_length * records_to_read],
                        &mut channel_group.cn,
                        record_length,
                        previous_index,
                        false,
                    )
                    .context("failed reading channels from dv di blocks")?;
                    let remaining = block_length % record_length;
                    if remaining > 0 {
                        data.copy_within(record_length * n_record_chunk.., 0);
                        data.truncate(remaining);
                    } else {
                        data.clear();
                    }
                    previous_index += records_to_read;
                }
            }
        }
        for s in advise_slices {
            rdr.advise_dontneed_slice(s);
        }
    }

    // Phase 3b: Stream invalid blocks in bounded batches into the pre-allocated buffer
    // DVBLOCK only contains data bytes (no invalidation bytes; those are in DIBLOCK).
    if let Some(invalid) = &mut channel_group.invalid_bytes {
        let mut previous_invalid_pos: usize = 0;
        let mut dv_idx = 0usize;
        let mut inv_iter = raw_invalid.into_iter().peekable();
        while inv_iter.peek().is_some() {
            let batch = take_raw_batch(&mut inv_iter, |r| r.as_ref().map_or(0, raw_block_est_len));
            let advise_slices: Vec<&'a [u8]> = batch
                .iter()
                .filter_map(|r| r.as_ref().map(raw_block_slice))
                .collect();
            let decompressed_invalid: Vec<Option<Cow<'a, [u8]>>> = batch
                .into_par_iter()
                .map(|opt_raw| -> Result<Option<Cow<'a, [u8]>>> {
                    match opt_raw {
                        None => Ok(None),
                        Some(RawCompBlock::Plain(d)) => Ok(Some(Cow::Borrowed(d))),
                        Some(RawCompBlock::Dz(header, buf)) => header
                            .decompress(buf)
                            .map(|v| Some(Cow::Owned(v)))
                            .context("failed decompressing DI DZ block"),
                    }
                })
                .collect::<Result<_>>()?;
            for opt_block_data in decompressed_invalid {
                match opt_block_data {
                    Some(block_data) => {
                        let block_length = block_data.len();
                        let end = (previous_invalid_pos + block_length).min(invalid.len());
                        if end > previous_invalid_pos {
                            invalid[previous_invalid_pos..end]
                                .copy_from_slice(&block_data[..end - previous_invalid_pos]);
                        }
                        previous_invalid_pos += block_length;
                    }
                    None => {
                        if let Some(&n_rec) = dv_record_counts.get(dv_idx) {
                            previous_invalid_pos += n_rec * cg_inval_bytes;
                        }
                    }
                }
                dv_idx += 1;
            }
            for s in advise_slices {
                rdr.advise_dontneed_slice(s);
            }
        }
    }

    Ok(position)
}

/// Reads all DL Blocks and returns a vect of them
fn parser_dl4(rdr: &mut SliceCursor<'_>, mut position: i64) -> Result<(Vec<Dl4Block>, i64)> {
    let mut dl_blocks: Vec<Dl4Block> = Vec::new();
    let mut chain = BlockChain::new("DL");
    chain.visit(position);
    let (block, pos) = parser_dl4_block(rdr, position, position)?;
    position = pos;
    dl_blocks.push(block.clone());
    let mut next_dl = block.dl_dl_next;
    while next_dl > 0 {
        if !chain.visit(next_dl) {
            break;
        }
        rdr.seek_relative(next_dl - position)
            .context("Could not reach DL4 block position")?;
        position = next_dl;
        let mut id = [0u8; 4];
        rdr.read_exact(&mut id)
            .context("could not read DL block id")?;
        let (block, pos) = parser_dl4_block(rdr, position, position)?;
        position = pos;
        dl_blocks.push(block.clone());
        next_dl = block.dl_dl_next;
    }
    Ok((dl_blocks, position))
}

/// Raw (possibly compressed) DZ or plain DT slice borrowed from mapped file.
enum RawCompBlock<'a> {
    Plain(&'a [u8]),
    Dz(Dz4Block, &'a [u8]),
}

/// Reads all sorted data blocks pointed by DL4 Blocks.
///
/// Three-phase approach: (1) zero-copy slice / DZ raw collection, (2) Rayon
/// decompresses all DZ blocks in parallel (plain blocks are borrowed zero-copy),
/// (3) sequential record assembly.
fn parser_dl4_sorted<'a>(
    rdr: &mut SliceCursor<'a>,
    dl_blocks: Vec<Dl4Block>,
    mut position: i64,
    channel_group: &mut Cg4,
    decoder: &mut Dec,
    rec_pos: &i64,
) -> Result<(i64, Vec<(u8, i64)>)> {
    initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone())
        .context("failed initialising arrays for sorted dl4 block")?;

    let cg_cycle_count = channel_group.block.cg_cycle_count as usize;
    let record_length = channel_group.record_length as usize;
    let mut vlsd_channels: Vec<(u8, i64)> = Vec::new();

    // Phase 1: Zero-copy slice collection
    let total_entries: usize = dl_blocks.iter().map(|dl| dl.dl_data.len()).sum();
    let mut raw_entries: Vec<(bool, RawCompBlock<'a>)> = Vec::with_capacity(total_entries); // (is_sd, block)
    for dl in &dl_blocks {
        for &data_pointer in &dl.dl_data {
            if data_pointer == 0 {
                continue;
            }
            rdr.seek_relative(data_pointer - position)
                .context("Could not reach DV or DZ block position from DL4")?;
            let mut id = [0u8; 4];
            rdr.read_exact(&mut id)
                .context("could not read data block id")?;
            if id == *b"##DZ" {
                let (header, compressed) = read_dz_raw(rdr)?;
                let is_sd = header.dz_org_block_type == *b"SD";
                position = data_pointer + header.len as i64;
                raw_entries.push((is_sd, RawCompBlock::Dz(header, compressed)));
            } else {
                let block_header: Dt4Block =
                    binrw::BinReaderExt::read_le(rdr).context("Could not read DT block header")?;
                // Guard against unfinalized files where the last DT block's len field
                // was never updated (len == 0 or garbage). Clamp to available file bytes.
                let data_len = if block_header.len >= 24 {
                    let file_size = rdr.file_size().unwrap_or(u64::MAX);
                    let claimed = block_header.len - 24;
                    let available = (file_size as i64).saturating_sub(data_pointer + 24);
                    if available > 0 {
                        claimed.min(available as u64) as usize
                    } else {
                        0
                    }
                } else {
                    // len < 24 — block header not finalized; read nothing for this block
                    warn!(
                        "DT block at 0x{data_pointer:x} has invalid len={} (unfinalized?), skipping",
                        block_header.len
                    );
                    0
                };
                let slice = rdr
                    .read_slice(data_len)
                    .context("Could not read DT block data")?;
                position = data_pointer + 24 + data_len as i64;
                let is_sd = &id[2..4] == b"SD";
                raw_entries.push((is_sd, RawCompBlock::Plain(slice)));
            }
        }
    }

    // Phase 2 & 3: Bounded-batch parallel decompression + sequential record assembly
    let mut data: Vec<u8> = Vec::new();
    let mut previous_index: usize = 0;
    let mut raw_iter = raw_entries.into_iter().peekable();
    while raw_iter.peek().is_some() {
        let batch = take_raw_batch(&mut raw_iter, |(_, raw)| raw_block_est_len(raw));
        let advise_slices: Vec<&'a [u8]> =
            batch.iter().map(|(_, raw)| raw_block_slice(raw)).collect();
        let decompressed: Vec<(bool, Cow<'a, [u8]>)> = batch
            .into_par_iter()
            .map(|(is_sd, raw)| -> Result<(bool, Cow<'a, [u8]>)> {
                let data = match raw {
                    RawCompBlock::Plain(d) => Cow::Borrowed(d),
                    RawCompBlock::Dz(header, buf) => Cow::Owned(
                        header
                            .decompress(buf)
                            .context("failed decompressing DZ block in DL4 sorted")?,
                    ),
                };
                Ok((is_sd, data))
            })
            .collect::<Result<_>>()?;

        for (is_sd, block_data) in decompressed {
            if is_sd {
                if data.is_empty() {
                    if let Some(cn) = channel_group.cn.get_mut(rec_pos) {
                        let (idx, consumed) =
                            read_vlsd_from_slice(&block_data, cn, previous_index, decoder)?;
                        previous_index = idx;
                        if consumed < block_data.len() {
                            data.extend_from_slice(&block_data[consumed..]);
                        }
                    }
                } else {
                    data.extend_from_slice(&block_data);
                    if let Some(cn) = channel_group.cn.get_mut(rec_pos) {
                        previous_index =
                            read_vlsd_from_bytes(&mut data, cn, previous_index, decoder)?;
                    }
                }
            } else if data.is_empty() {
                // Fast path when no partial record tail exists from the previous block
                let block_length = block_data.len();
                let n_record_chunk = block_length.checked_div(record_length).unwrap_or(0);
                if previous_index < cg_cycle_count && n_record_chunk > 0 {
                    let records_to_read = n_record_chunk.min(cg_cycle_count - previous_index);
                    let inval_bytes =
                        if (channel_group.block.cg_flags & (CG_F_VLSD | CG_F_VLSC)) == 0 {
                            channel_group.block.cg_inval_bytes as usize
                        } else {
                            0
                        };
                    vlsd_channels = read_channels_from_bytes_with_inval(
                        &block_data[..record_length * records_to_read],
                        &mut channel_group.cn,
                        record_length,
                        previous_index,
                        inval_bytes > 0,
                        inval_bytes,
                    )
                    .context("could not read channels from bytes")?;
                    previous_index += records_to_read;
                }
                let remaining = block_length % record_length;
                if remaining > 0 {
                    data.extend_from_slice(&block_data[record_length * n_record_chunk..]);
                }
            } else {
                data.extend_from_slice(&block_data);
                // Use data.len() so a partial-record tail from the previous block is included.
                let block_length = data.len();
                let n_record_chunk = block_length.checked_div(record_length).unwrap_or(0);
                if previous_index >= cg_cycle_count || n_record_chunk == 0 {
                    continue;
                }
                let records_to_read = n_record_chunk.min(cg_cycle_count - previous_index);
                let inval_bytes = if (channel_group.block.cg_flags & (CG_F_VLSD | CG_F_VLSC)) == 0 {
                    channel_group.block.cg_inval_bytes as usize
                } else {
                    0
                };
                vlsd_channels = read_channels_from_bytes_with_inval(
                    &data[..record_length * records_to_read],
                    &mut channel_group.cn,
                    record_length,
                    previous_index,
                    inval_bytes > 0,
                    inval_bytes,
                )
                .context("could not read channels from bytes")?;
                let remaining = block_length % record_length;
                if remaining > 0 {
                    data.copy_within(record_length * n_record_chunk.., 0);
                    data.truncate(remaining);
                } else {
                    data.clear();
                }
                previous_index += records_to_read;
            }
        }
        for s in advise_slices {
            rdr.advise_dontneed_slice(s);
        }
    }
    Ok((position, vlsd_channels))
}

/// Reads all unsorted data blocks pointed by DL4 Blocks
fn parser_dl4_unsorted<'a>(
    rdr: &mut SliceCursor<'a>,
    dg: &mut Dg4,
    dl_blocks: Vec<Dl4Block>,
    position: i64,
) -> Result<i64> {
    // Phase 1: Zero-copy slice / raw DZ collection
    let (raw_entries, position) = collect_dl_raw_entries(rdr, &dl_blocks, position)
        .context("Could not collect DL4 unsorted data blocks")?;

    // Phase 2 & 3: Bounded-batch parallel decompression + streaming record dispatch
    let mut data: Vec<u8> = Vec::new();
    let mut decoder: Dec = Dec {
        windows_1252: WINDOWS_1252.new_decoder(),
        utf_16_be: UTF_16BE.new_decoder(),
        utf_16_le: UTF_16LE.new_decoder(),
    };
    let mut unsorted_state = UnsortedState::new(dg);
    let mut raw_iter = raw_entries.into_iter().peekable();
    while raw_iter.peek().is_some() {
        let batch = take_raw_batch(&mut raw_iter, raw_block_est_len);
        let advise_slices: Vec<&'a [u8]> = batch.iter().map(raw_block_slice).collect();
        let decompressed =
            decompress_raw_blocks(batch, "failed decompressing DZ block in DL4 unsorted")?;
        for block_data in decompressed {
            if data.is_empty() {
                let consumed = unsorted_state.process_slice(&block_data, dg, &mut decoder)?;
                if consumed < block_data.len() {
                    data.extend_from_slice(&block_data[consumed..]);
                }
            } else {
                data.extend_from_slice(&block_data);
                unsorted_state.process_chunk(&mut data, dg, &mut decoder)?;
            }
        }
        for s in advise_slices {
            rdr.advise_dontneed_slice(s);
        }
    }
    unsorted_state.finish(dg)?;
    Ok(position)
}

/// Returns chunk size and corresponding number of records from a channel group
fn generate_chunks(channel_group: &Cg4) -> Vec<(usize, usize)> {
    let record_length = channel_group.record_length as usize;
    let cg_cycle_count = channel_group.block.cg_cycle_count as usize;
    if record_length == 0 || cg_cycle_count == 0 {
        return Vec::new();
    }
    let n_record_chunk = (CHUNK_SIZE_READING_4 / record_length).max(1);
    let n_chunks = cg_cycle_count / n_record_chunk;
    let mut chunks = vec![(n_record_chunk, record_length * n_record_chunk); n_chunks];
    let remainder = cg_cycle_count % n_record_chunk;
    if remainder > 0 {
        chunks.push((remainder, record_length * remainder));
    }
    chunks
}

/// Reads all channels from given channel group having sorted data blocks
fn read_all_channels_sorted(
    rdr: &mut SliceCursor<'_>,
    channel_group: &mut Cg4,
) -> Result<Vec<(u8, i64)>> {
    let chunks = generate_chunks(channel_group);
    // initialises the arrays
    initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone())
        .context("failed initialising arrays")?;
    // read by chunks directly from mapped slice and store in channel array
    let mut previous_index: usize = 0;
    let mut vlsd_channels: Vec<(u8, i64)> = Vec::new();
    let inval_bytes = if (channel_group.block.cg_flags & (CG_F_VLSD | CG_F_VLSC)) == 0 {
        channel_group.block.cg_inval_bytes as usize
    } else {
        0
    };
    for (n_record_chunk, chunk_size) in chunks {
        let data_chunk = rdr
            .read_slice(chunk_size)
            .context("Could not read data chunk")?;
        vlsd_channels = read_channels_from_bytes_with_inval(
            data_chunk,
            &mut channel_group.cn,
            channel_group.record_length as usize,
            previous_index,
            inval_bytes > 0,
            inval_bytes,
        )
        .context("could not read channels from bytes")?;
        previous_index += n_record_chunk;
        rdr.advise_dontneed_slice(data_chunk);
    }
    Ok(vlsd_channels)
}

/// copies complete sorted data block (not chunk) into each channel array
fn read_all_channels_sorted_from_bytes(
    data: &[u8],
    channel_group: &mut Cg4,
) -> Result<Vec<(u8, i64)>> {
    // initialises the arrays
    initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone())
        .context("failed initilising arrays")?;
    let record_length = channel_group.record_length as usize;
    let cg_cycle_count = channel_group.block.cg_cycle_count as usize;
    let n_records = data
        .len()
        .checked_div(record_length)
        .unwrap_or(0)
        .min(cg_cycle_count);
    let inval_bytes = if (channel_group.block.cg_flags & (CG_F_VLSD | CG_F_VLSC)) == 0 {
        channel_group.block.cg_inval_bytes as usize
    } else {
        0
    };
    let vlsd_channels: Vec<(u8, i64)> = read_channels_from_bytes_with_inval(
        &data[..n_records * record_length],
        &mut channel_group.cn,
        record_length,
        0,
        inval_bytes > 0,
        inval_bytes,
    )
    .context("failed initilising arrays")?;
    Ok(vlsd_channels)
}

/// Initial staging buffer capacity cap per active CG in unsorted reading.
const UNSORTED_CG_INIT_CAP_BYTES: usize = 64 * 1024;
/// Flush threshold per active CG buffer in unsorted reading to bound peak staging RAM.
const UNSORTED_CG_FLUSH_BYTES: usize = 1024 * 1024;

/// Reads unsorted data block directly from mapped slice in bounded 4 MiB streaming windows
fn read_all_channels_unsorted(
    rdr: &mut SliceCursor<'_>,
    dg: &mut Dg4,
    block_length: i64,
) -> Result<()> {
    let payload_len = (block_length as usize).saturating_sub(24);
    if payload_len == 0 {
        return Ok(());
    }
    let data = rdr
        .read_slice(payload_len)
        .context("Could not read data chunk")?;
    let mut unsorted_state = UnsortedState::new(dg);
    let mut decoder: Dec = Dec {
        windows_1252: WINDOWS_1252.new_decoder(),
        utf_16_be: UTF_16BE.new_decoder(),
        utf_16_le: UTF_16LE.new_decoder(),
    };
    let mut offset = 0usize;
    let mut advised_start = 0usize;
    while offset < data.len() {
        let mut window_end = (offset + CHUNK_SIZE_READING_4).min(data.len());
        loop {
            let consumed =
                unsorted_state.process_slice(&data[offset..window_end], dg, &mut decoder)?;
            if consumed > 0 {
                offset += consumed;
                break;
            }
            if window_end == data.len() {
                offset = data.len();
                break;
            }
            window_end = (window_end + CHUNK_SIZE_READING_4).min(data.len());
        }
        if offset.saturating_sub(advised_start) >= CHUNK_SIZE_READING_4 {
            rdr.advise_dontneed_slice(&data[advised_start..offset]);
            advised_start = offset;
        }
    }
    if advised_start < data.len() {
        rdr.advise_dontneed_slice(&data[advised_start..]);
    }
    unsorted_state.finish(dg)?;
    Ok(())
}

/// Compact record-ID-to-slot mapping: direct table for dense small IDs, FxHashMap for sparse/large IDs.
enum RecordIdMap {
    Direct(Vec<u16>),
    Sparse(rustc_hash::FxHashMap<u64, u16>),
}

impl RecordIdMap {
    #[inline(always)]
    fn get(&self, rec_id: u64) -> Option<usize> {
        match self {
            Self::Direct(table) => {
                if rec_id < table.len() as u64 {
                    let slot = table[rec_id as usize];
                    if slot != u16::MAX {
                        return Some(slot as usize);
                    }
                }
                None
            }
            Self::Sparse(map) => map.get(&rec_id).copied().map(|s| s as usize),
        }
    }
}

/// Lookup entry for fast CG access by record ID.
#[derive(Clone, Copy)]
struct CgLookupEntry {
    record_length: u32,
    is_vlsd: bool,
    should_read: bool,
    has_cg_template: bool,
    vlsd_cg: Option<(u64, i64)>,
}

/// Per-CG staging buffer for unsorted demuxing.
struct UnsortedBuffer {
    rec_id: u64,
    write_index: usize,
    data: Vec<u8>,
    has_cg_template: bool,
}

/// Flushes a single CG's accumulated unsorted records into its channel builders.
fn flush_cg_buffer(
    dg: &mut Dg4,
    rec_id: u64,
    write_index: &mut usize,
    record_data: &mut Vec<u8>,
) -> Result<(), Error> {
    if record_data.is_empty() {
        return Ok(());
    }
    if let Some(channel_group) = dg.cg.get_mut(&rec_id) {
        let record_length = channel_group.record_length as usize;
        let cg_cycle_count = channel_group.block.cg_cycle_count as usize;
        if record_length > 0 && *write_index < cg_cycle_count {
            let n_records = record_data.len() / record_length;
            let records_to_read = n_records.min(cg_cycle_count - *write_index);
            if records_to_read > 0 {
                let inval_bytes = if (channel_group.block.cg_flags & (CG_F_VLSD | CG_F_VLSC)) == 0 {
                    channel_group.block.cg_inval_bytes as usize
                } else {
                    0
                };
                read_channels_from_bytes_with_inval(
                    &record_data[..records_to_read * record_length],
                    &mut channel_group.cn,
                    record_length,
                    *write_index,
                    inval_bytes > 0,
                    inval_bytes,
                )
                .context("failed reading channels from bytes after reading unsorted data")?;
                *write_index += records_to_read;
            }
        }
    }
    record_data.clear();
    Ok(())
}

/// State for unsorted data reading — uses compact dense slots with bounded staging buffers
/// and O(1) record-ID lookup, skipping unread CGs without allocating or copying.
struct UnsortedState {
    id_map: RecordIdMap,
    buffers: Vec<UnsortedBuffer>,
    lookup: Vec<CgLookupEntry>,
}

impl UnsortedState {
    fn new(dg: &Dg4) -> Self {
        let mut max_record_id = dg
            .cg
            .values()
            .map(|cg| cg.block.cg_record_id)
            .max()
            .unwrap_or(0);

        // If any CG has ca_storage == 1, records auto-increment up to cg_record_id + M - 1
        let mut total_slots = dg.cg.len();
        for cg in dg.cg.values() {
            for cn in cg.cn.values() {
                if let Some(compo) = &cn.composition
                    && let Compo::CA(ca) = &compo.block
                    && ca.ca_storage == 1
                {
                    let m = ca.ca_dim_size.iter().product::<u64>().max(1);
                    max_record_id = max_record_id.max(cg.block.cg_record_id.saturating_add(m));
                    total_slots += m.saturating_sub(1) as usize;
                }
            }
        }

        let use_direct = max_record_id < 4096 && total_slots < u16::MAX as usize;
        let mut id_map = if use_direct {
            RecordIdMap::Direct(vec![u16::MAX; max_record_id as usize + 1])
        } else {
            RecordIdMap::Sparse(rustc_hash::FxHashMap::with_capacity_and_hasher(
                total_slots,
                Default::default(),
            ))
        };

        let mut buffers = Vec::with_capacity(total_slots);
        let mut lookup = Vec::with_capacity(total_slots);

        let insert_slot = |id_map: &mut RecordIdMap,
                           buffers: &mut Vec<UnsortedBuffer>,
                           lookup: &mut Vec<CgLookupEntry>,
                           rec_id: u64,
                           entry: CgLookupEntry,
                           capacity: usize| {
            let slot_idx = buffers.len() as u16;
            match id_map {
                RecordIdMap::Direct(table) => {
                    table[rec_id as usize] = slot_idx;
                }
                RecordIdMap::Sparse(map) => {
                    map.insert(rec_id, slot_idx);
                }
            }
            buffers.push(UnsortedBuffer {
                rec_id,
                write_index: 0,
                data: Vec::with_capacity(capacity),
                has_cg_template: entry.has_cg_template,
            });
            lookup.push(entry);
        };

        for cg in dg.cg.values() {
            let rec_id = cg.block.cg_record_id;
            let is_vlsd = (cg.block.cg_flags & (CG_F_VLSD | CG_F_VLSC)) != 0;
            let has_readable_cg_template = cg.cn.values().any(|cn| {
                cn.should_read
                    && matches!(&cn.composition, Some(compo) if matches!(&compo.block, Compo::CA(ca) if ca.ca_storage == 1))
            });
            let should_read = if is_vlsd {
                cg.vlsd_cg.is_some_and(|(target_rec_id, target_rec_pos)| {
                    dg.cg
                        .get(&target_rec_id)
                        .and_then(|tcg| tcg.cn.get(&target_rec_pos))
                        .is_some_and(|tcn| tcn.should_read)
                })
            } else {
                cg.cn.values().any(|cn| cn.should_read)
            };

            let capacity = if is_vlsd || !should_read {
                0
            } else {
                let full_bytes =
                    (cg.block.cg_cycle_count as usize).saturating_mul(cg.record_length as usize);
                if has_readable_cg_template {
                    full_bytes
                } else {
                    full_bytes.min(UNSORTED_CG_INIT_CAP_BYTES)
                }
            };

            insert_slot(
                &mut id_map,
                &mut buffers,
                &mut lookup,
                rec_id,
                CgLookupEntry {
                    record_length: cg.record_length,
                    is_vlsd,
                    should_read,
                    has_cg_template: has_readable_cg_template,
                    vlsd_cg: cg.vlsd_cg,
                },
                capacity,
            );

            // Register auto-incremented record IDs for ca_storage == 1
            for cn in cg.cn.values() {
                if let Some(compo) = &cn.composition
                    && let Compo::CA(ca) = &compo.block
                    && ca.ca_storage == 1
                {
                    let m = ca.ca_dim_size.iter().product::<u64>().max(1) as usize;
                    for k in 1..m {
                        let elem_rec_id = rec_id + k as u64;
                        let elem_capacity = if cn.should_read {
                            let cycles = ca
                                .ca_cycle_count
                                .as_ref()
                                .and_then(|c| c.get(k))
                                .copied()
                                .unwrap_or(cg.block.cg_cycle_count)
                                as usize;
                            cycles.saturating_mul(cg.record_length as usize)
                        } else {
                            0
                        };
                        insert_slot(
                            &mut id_map,
                            &mut buffers,
                            &mut lookup,
                            elem_rec_id,
                            CgLookupEntry {
                                record_length: cg.record_length,
                                is_vlsd: false,
                                should_read: cn.should_read,
                                has_cg_template: cn.should_read,
                                vlsd_cg: None,
                            },
                            elem_capacity,
                        );
                    }
                }
            }
        }

        Self {
            id_map,
            buffers,
            lookup,
        }
    }

    fn process_slice(
        &mut self,
        data: &[u8],
        dg: &mut Dg4,
        decoder: &mut Dec,
    ) -> Result<usize, Error> {
        read_all_channels_unsorted_from_slice(
            data,
            dg,
            &self.id_map,
            &mut self.buffers,
            &self.lookup,
            decoder,
        )
    }

    fn process_chunk(
        &mut self,
        data: &mut Vec<u8>,
        dg: &mut Dg4,
        decoder: &mut Dec,
    ) -> Result<(), Error> {
        let position = read_all_channels_unsorted_from_slice(
            data,
            dg,
            &self.id_map,
            &mut self.buffers,
            &self.lookup,
            decoder,
        )?;
        let remaining_len = data.len() - position;
        data.copy_within(position.., 0);
        data.truncate(remaining_len);
        Ok(())
    }

    /// Flushes all remaining buffered records and assembles `ca_storage == 1` CG-template arrays.
    fn finish(mut self, dg: &mut Dg4) -> Result<(), Error> {
        for buf in &mut self.buffers {
            if buf.data.is_empty() {
                continue;
            }
            if buf.has_cg_template {
                if let Some(channel_group) = dg.cg.get_mut(&buf.rec_id) {
                    let record_length = channel_group.record_length as usize;
                    let cg_cycle_count = channel_group.block.cg_cycle_count as usize;
                    if record_length > 0 && buf.write_index < cg_cycle_count {
                        let n_records = buf.data.len() / record_length;
                        let records_to_read = n_records.min(cg_cycle_count - buf.write_index);
                        if records_to_read > 0 {
                            let inval_bytes =
                                if (channel_group.block.cg_flags & (CG_F_VLSD | CG_F_VLSC)) == 0 {
                                    channel_group.block.cg_inval_bytes as usize
                                } else {
                                    0
                                };
                            read_channels_from_bytes_with_inval(
                                &buf.data[..records_to_read * record_length],
                                &mut channel_group.cn,
                                record_length,
                                buf.write_index,
                                inval_bytes > 0,
                                inval_bytes,
                            )
                            .context(
                                "failed reading channels from bytes after reading unsorted data",
                            )?;
                            buf.write_index += records_to_read;
                        }
                    }
                }
            } else {
                flush_cg_buffer(dg, buf.rec_id, &mut buf.write_index, &mut buf.data)?;
            }
        }

        // Assemble CG template arrays (ca_storage == 1) once all data blocks are demuxed
        for (rec_id, channel_group) in &mut dg.cg {
            for cn in channel_group.cn.values_mut() {
                if !cn.should_read {
                    continue;
                }
                if let Some(compo) = &cn.composition
                    && let Compo::CA(ca) = &compo.block
                    && ca.ca_storage == 1
                {
                    let m = ca.ca_dim_size.iter().product::<u64>().max(1) as usize;
                    let cycle_count = channel_group.block.cg_cycle_count as usize;
                    let elem_size = cn.n_bytes as usize;
                    let byte_offset = cn.pos_byte_beg as usize;
                    let record_length = channel_group.record_length as usize;
                    let mut interleaved = vec![0u8; cycle_count * m * elem_size];

                    for k in 0..m {
                        let elem_rec_id = *rec_id + k as u64;
                        if let Some(slot) = self.id_map.get(elem_rec_id) {
                            let record_data = &mut self.buffers[slot].data;
                            let n_rec = record_data.len().checked_div(record_length).unwrap_or(0);
                            for t in 0..cycle_count.min(n_rec) {
                                let src_start = t * record_length + byte_offset;
                                let dst_start = (t * m + k) * elem_size;
                                if src_start + elem_size <= record_data.len()
                                    && dst_start + elem_size <= interleaved.len()
                                {
                                    interleaved[dst_start..dst_start + elem_size].copy_from_slice(
                                        &record_data[src_start..src_start + elem_size],
                                    );
                                }
                            }
                            record_data.clear();
                        }
                    }
                    read_one_channel_array(&interleaved, cn, cycle_count)?;
                }
            }
        }
        Ok(())
    }
}

/// Demuxes records from an unsorted slice into bounded per-CG staging buffers,
/// flushing individual CG buffers once they reach `UNSORTED_CG_FLUSH_BYTES`.
fn read_all_channels_unsorted_from_slice(
    data: &[u8],
    dg: &mut Dg4,
    id_map: &RecordIdMap,
    record_buffers: &mut [UnsortedBuffer],
    cg_lookup: &[CgLookupEntry],
    decoder: &mut Dec,
) -> Result<usize, Error> {
    let mut position: usize = 0;
    let data_length = data.len();
    let dg_rec_id_size = dg.block.dg_rec_id_size as usize;
    let vlsd_data_start_offset = dg_rec_id_size + std::mem::size_of::<u32>();
    // reusable string buffer for VLSC string decoding
    let mut dst = String::new();

    // unsorted data into sorted data blocks, except for VLSD CG.
    let mut remaining: usize = data_length - position;
    while remaining > 0 {
        // reads record id
        let rec_id: u64 = if dg_rec_id_size == 1 && remaining >= 1 {
            data[position].into()
        } else if dg_rec_id_size == 2 && remaining >= 2 {
            let rec = &data[position..position + std::mem::size_of::<u16>()];
            u16::from_le_bytes(rec.try_into().unwrap()) as u64
        } else if dg_rec_id_size == 4 && remaining >= 4 {
            let rec = &data[position..position + std::mem::size_of::<u32>()];
            u32::from_le_bytes(rec.try_into().unwrap()) as u64
        } else if dg_rec_id_size == 8 && remaining >= 8 {
            let rec = &data[position..position + std::mem::size_of::<u64>()];
            u64::from_le_bytes(rec.try_into().unwrap())
        } else {
            break; // not enough data remaining
        };

        // O(1) lookup for this record ID
        let Some(slot) = id_map.get(rec_id) else {
            bail!("could not find the record id");
        };
        let cg_entry = cg_lookup[slot];

        if cg_entry.is_vlsd {
            // VLSD or VLSC channel (Variable Length Signal Data/Size Channel)
            if remaining < vlsd_data_start_offset {
                break; // not enough data remaining
            }
            let len = &data[position + dg_rec_id_size..position + vlsd_data_start_offset];
            let length: usize = u32::from_le_bytes(len.try_into().unwrap()) as usize;
            let Some(total_len) = vlsd_data_start_offset.checked_add(length) else {
                break;
            };
            if remaining < total_len {
                break; // not enough data remaining
            }
            if !cg_entry.should_read {
                position += total_len;
            } else {
                position += vlsd_data_start_offset;
                let record = &data[position..position + length];
                if let Some((target_rec_id, target_rec_pos)) = cg_entry.vlsd_cg {
                    if let Some(target_cg) = dg.cg.get_mut(&target_rec_id) {
                        if let Some(target_cn) = target_cg.cn.get_mut(&target_rec_pos) {
                            let nrecord = &mut record_buffers[slot].write_index;
                            // For VLSC channels (cn_type == 7) in unsorted data,
                            // reinitialize from UInt (offset storage) to actual data type on first record
                            if *nrecord == 0 && target_cn.block.cn_type == 7 {
                                match target_cn.block.cn_data_type {
                                    6..=9 | 17 => {
                                        target_cn.builder = Some(Box::new(ChannelBuilder::Utf8(
                                            arrow::array::LargeStringBuilder::new(),
                                        )));
                                    }
                                    10 => {
                                        target_cn.builder =
                                            Some(Box::new(ChannelBuilder::VariableSizeByteArray(
                                                arrow::array::LargeBinaryBuilder::new(),
                                            )));
                                    }
                                    _ => {}
                                }
                            }
                            if target_cn.builder.is_none() {
                                target_cn.init_builder(0)?;
                            }
                            match target_cn.builder.as_deref_mut().unwrap() {
                                ChannelBuilder::Utf8(array) => {
                                    if target_cn.block.cn_data_type == 7 {
                                        // UTF-8: no decoding needed, use &str directly
                                        array.append_value(
                                            str::from_utf8(record)
                                                .context("Found invalid UTF-8 from VLSD record")?
                                                .split('\0')
                                                .next()
                                                .unwrap_or(""),
                                        );
                                    } else {
                                        dst.clear();
                                        if target_cn.block.cn_data_type == 6 {
                                            let (_result, _size, _replacement) = decoder
                                                .windows_1252
                                                .decode_to_string(record, &mut dst, false);
                                        } else if target_cn.block.cn_data_type == 8 {
                                            let (_result, _size, _replacement) = decoder
                                                .utf_16_le
                                                .decode_to_string(record, &mut dst, false);
                                        } else if target_cn.block.cn_data_type == 9 {
                                            let (_result, _size, _replacement) = decoder
                                                .utf_16_be
                                                .decode_to_string(record, &mut dst, false);
                                        } else if target_cn.block.cn_data_type == 17 {
                                            // Unicode with BOM
                                            let bom = Bom::from(record);
                                            let mut bom_decoder = match bom {
                                                Bom::Utf8 => UTF_8.new_decoder(),
                                                Bom::Utf16Be => UTF_16BE.new_decoder(),
                                                Bom::Utf16Le => UTF_16LE.new_decoder(),
                                                Bom::Gb18030 => GB18030.new_decoder(),
                                                _ => {
                                                    bail!("not implemented BOM type");
                                                }
                                            };
                                            let (_result, _size, _replacement) = bom_decoder
                                                .decode_to_string(record, &mut dst, false);
                                        } else {
                                            bail!("channel data type is not correct for a text")
                                        };
                                        array.append_value(dst.split('\0').next().unwrap_or(""));
                                    }
                                }
                                ChannelBuilder::VariableSizeByteArray(array) => {
                                    array.append_value(record);
                                }
                                _ => {
                                    bail!("data type of VLSD is not possible");
                                }
                            }
                            *nrecord += 1;
                        } else {
                            bail!("could not find the target record position");
                        }
                    } else {
                        bail!("could not find the target record id");
                    }
                } else {
                    bail!("no VLSD/VLSC target in CG, wrong cg_flags");
                }
                position += length;
            }
        } else if remaining >= cg_entry.record_length as usize {
            let rec_len = cg_entry.record_length as usize;
            if cg_entry.should_read {
                let record = &data[position..position + rec_len];
                let buf = &mut record_buffers[slot];
                buf.data.extend_from_slice(record);
                if !buf.has_cg_template && buf.data.len() >= UNSORTED_CG_FLUSH_BYTES {
                    flush_cg_buffer(dg, buf.rec_id, &mut buf.write_index, &mut buf.data)?;
                }
            }
            position += rec_len;
        } else {
            break; // not enough data remaining
        }
        remaining = data_length - position;
    }

    Ok(position)
}

/// decoder for String SBC and UTF16 Le & Be
struct Dec {
    windows_1252: Decoder,
    utf_16_be: Decoder,
    utf_16_le: Decoder,
}

/// Decodes a byte slice to a String based on MDF4 cn_data_type.
/// cn_data_type: 6=SBC/Windows-1252, 7=UTF-8, 8=UTF-16 LE, 9=UTF-16 BE, 17=BOM-prefixed
fn decode_string_bytes<'a>(
    record: &'a [u8],
    cn_data_type: u8,
    decoder: &mut Dec,
    buf: &'a mut String,
) -> Result<&'a str> {
    match cn_data_type {
        6 => {
            buf.clear();
            buf.reserve(record.len());
            let _ = decoder.windows_1252.decode_to_string(record, buf, false);
            let s = buf.split('\0').next().unwrap_or("");
            Ok(s)
        }
        7 => {
            let s = str::from_utf8(record).context("Found invalid UTF-8")?;
            Ok(s.split('\0').next().unwrap_or(""))
        }
        8 => {
            buf.clear();
            buf.reserve(record.len());
            let _ = decoder.utf_16_le.decode_to_string(record, buf, false);
            let s = buf.split('\0').next().unwrap_or("");
            Ok(s)
        }
        9 => {
            buf.clear();
            buf.reserve(record.len());
            let _ = decoder.utf_16_be.decode_to_string(record, buf, false);
            let s = buf.split('\0').next().unwrap_or("");
            Ok(s)
        }
        17 => {
            if record.len() >= 3 && record[0] == 0xEF && record[1] == 0xBB && record[2] == 0xBF {
                let s = str::from_utf8(&record[3..]).context("Found invalid UTF-8 with BOM")?;
                Ok(s.split('\0').next().unwrap_or(""))
            } else if record.len() >= 2 && record[0] == 0xFF && record[1] == 0xFE {
                buf.clear();
                buf.reserve(record.len());
                let _ = decoder.utf_16_le.decode_to_string(&record[2..], buf, false);
                let s = buf.split('\0').next().unwrap_or("");
                Ok(s)
            } else if record.len() >= 2 && record[0] == 0xFE && record[1] == 0xFF {
                buf.clear();
                buf.reserve(record.len());
                let _ = decoder.utf_16_be.decode_to_string(&record[2..], buf, false);
                let s = buf.split('\0').next().unwrap_or("");
                Ok(s)
            } else {
                let s = str::from_utf8(record).context("Found invalid UTF-8 (no BOM)")?;
                Ok(s.split('\0').next().unwrap_or(""))
            }
        }
        _ => {
            buf.clear();
            buf.push_str(&String::from_utf8_lossy(record));
            Ok(buf.as_str())
        }
    }
}

/// initialise ndarrays for the data group/block
fn initialise_arrays(channel_group: &mut Cg4, cg_cycle_count: &u64) -> Result<(), Error> {
    let readable_sample = channel_group
        .cn
        .values()
        .filter(|cn| cn.should_read)
        .take(3)
        .count();
    if readable_sample == 0 {
        return Ok(());
    }
    if readable_sample <= 2 || *cg_cycle_count < 16_384 {
        for cn in channel_group.cn.values_mut() {
            if cn.should_read {
                cn.init_builder(*cg_cycle_count as usize).with_context(|| {
                    format!("Zeros initialisation of channel {} failed", cn.unique_name)
                })?;
            }
        }
        return Ok(());
    }
    // creates zeroed array in parallel for each channel contained in channel group
    channel_group
        .cn
        .par_iter_mut()
        .filter(|(_cn_record_position, cn)| cn.should_read)
        .try_for_each(
            |(_cn_record_position, cn): (&i64, &mut Cn4)| -> Result<(), Error> {
                cn.init_builder(*cg_cycle_count as usize).with_context(|| {
                    format!("Zeros initialisation of channel {} failed", cn.unique_name)
                })?;
                Ok(())
            },
        )
        .with_context(|| {
            format!(
                "Zeros initialisation of channel group with master {:?} failed",
                channel_group.master_channel_name
            )
        })?;
    Ok(())
}

/// Applies bit-offset masking/shifting and finishes builders for a single channel.
fn finalize_cn4_channel(cn: &mut Cn4) {
    let bit_offset = cn.block.cn_bit_offset as u32;
    let bit_count = cn.block.cn_bit_count;
    if cn.block.cn_data_type <= 3
        && bit_count > 0
        && cn.n_bytes != 9
        && (bit_offset > 0 || bit_count != cn.n_bytes * 8)
    {
        macro_rules! shift_slice {
            ($array:expr, $reg_bits:expr) => {{
                let reg_bits: u32 = $reg_bits;
                if bit_offset + bit_count <= reg_bits && bit_count < reg_bits {
                    let left_shift = reg_bits - bit_offset - bit_count;
                    let right_shift = reg_bits - bit_count;
                    let a = $array.values_slice_mut();
                    if left_shift > 0 && right_shift > 0 {
                        a.iter_mut()
                            .for_each(|x| *x = (*x << left_shift) >> right_shift);
                    } else if left_shift > 0 {
                        a.iter_mut().for_each(|x| *x <<= left_shift);
                    } else if right_shift > 0 {
                        a.iter_mut().for_each(|x| *x >>= right_shift);
                    }
                }
            }};
        }
        if let Some(builder) = cn.builder.as_deref_mut() {
            match builder {
                ChannelBuilder::Int8(array) => shift_slice!(array, 8),
                ChannelBuilder::UInt8(array) => shift_slice!(array, 8),
                ChannelBuilder::Int16(array) => shift_slice!(array, 16),
                ChannelBuilder::UInt16(array) => shift_slice!(array, 16),
                ChannelBuilder::Int32(array) => shift_slice!(array, 32),
                ChannelBuilder::UInt32(array) => shift_slice!(array, 32),
                ChannelBuilder::Float32(_) => (),
                ChannelBuilder::Int64(array) => shift_slice!(array, 64),
                ChannelBuilder::UInt64(array) => shift_slice!(array, 64),
                ChannelBuilder::Float64(_) => (),
                ChannelBuilder::Complex32(_) => (),
                ChannelBuilder::Complex64(_) => (),
                ChannelBuilder::Utf8(_) => (),
                ChannelBuilder::VariableSizeByteArray(_) => (),
                ChannelBuilder::FixedSizeByteArray(_) => (),
                ChannelBuilder::ArrayDInt8(array) => shift_slice!(array, 8),
                ChannelBuilder::ArrayDUInt8(array) => shift_slice!(array, 8),
                ChannelBuilder::ArrayDInt16(array) => shift_slice!(array, 16),
                ChannelBuilder::ArrayDUInt16(array) => shift_slice!(array, 16),
                ChannelBuilder::ArrayDInt32(array) => shift_slice!(array, 32),
                ChannelBuilder::ArrayDUInt32(array) => shift_slice!(array, 32),
                ChannelBuilder::ArrayDFloat32(_) => (),
                ChannelBuilder::ArrayDInt64(array) => shift_slice!(array, 64),
                ChannelBuilder::ArrayDUInt64(array) => shift_slice!(array, 64),
                ChannelBuilder::ArrayDFloat64(_) => (),
                ChannelBuilder::Union(_) => (),
            }
        }
    }
    if let Some(b) = cn.builder.take() {
        cn.data = b.finish();
    }
}

/// Applies bit-offset masking and finishes channel builders for one channel group.
fn finalize_cg4_channels(channel_group: &mut Cg4) {
    let readable_sample = channel_group
        .cn
        .values()
        .filter(|cn| cn.should_read)
        .take(3)
        .count();
    if readable_sample == 0 {
        return;
    }
    if readable_sample <= 2 {
        for cn in channel_group.cn.values_mut() {
            if cn.should_read {
                finalize_cn4_channel(cn);
            }
        }
    } else {
        channel_group
            .cn
            .par_iter_mut()
            .filter(|(_cn_record_position, cn)| cn.should_read)
            .for_each(|(_rec_pos, cn)| finalize_cn4_channel(cn));
    }
}

/// applies bit mask if required in channel block
pub fn apply_bit_mask_offset(dg: &mut Dg4) -> Result<(), Error> {
    // apply bit shift and masking per MDF 4.3.0 §6.26.5.2 (Figures 45 & 46):
    // A signal spans `cn_n_bytes` bytes in the record (`n_bytes = ceil((cn_bit_offset + cn_bit_count) / 8)`).
    // Those `n_bytes` bytes are decoded into the target integer register of width `REG_BITS` (8, 16, 32, or 64)
    // in the lower `n_bytes * 8` bits of the register.
    // Within that register:
    //   - `cn_bit_offset` is the LSB bit offset (0..7)
    //   - `cn_bit_count` is the signal width in bits
    // Therefore:
    //   - `reg_left_shift = REG_BITS - cn_bit_offset - cn_bit_count` clears all bits above the signal MSB
    //     (and positions the sign bit at the register MSB for signed types),
    //   - `reg_right_shift = REG_BITS - cn_bit_count` shifts the signal down to bit 0
    //     (with arithmetic sign extension for signed types or zero fill for unsigned types).
    if dg.cg.len() <= 1 {
        for channel_group in dg.cg.values_mut() {
            finalize_cg4_channels(channel_group);
        }
    } else {
        dg.cg
            .par_iter_mut()
            .for_each(|(_, channel_group)| finalize_cg4_channels(channel_group));
    }
    Ok(())
}

/// Helper to process dynamic channels (VLSD, VLSC, DS)
fn process_dynamic_channels(
    rdr: &mut SliceCursor<'_>,
    dg: &mut Dg4,
    vlsd_channels: &[(u8, i64)],
    mut position: i64,
    decoder: &mut Dec,
) -> Result<i64> {
    if vlsd_channels.is_empty() {
        return Ok(position);
    }
    let mut handled_types = HashSet::new();
    for channel in vlsd_channels {
        if handled_types.insert(channel.0) {
            match channel.0 {
                1 => {
                    position = read_sd(rdr, dg, vlsd_channels, position, decoder)
                        .context("failed reading sd block")?;
                }
                7 => {
                    for channel_group in dg.cg.values_mut() {
                        position = read_vd(rdr, channel_group, vlsd_channels, position, decoder)
                            .context("failed reading vd block")?;
                    }
                }
                _ => {
                    for channel_group in dg.cg.values_mut() {
                        position = read_ds(rdr, channel_group, vlsd_channels, position, decoder)
                            .context("failed reading ds block")?;
                    }
                }
            }
        }
    }
    Ok(position)
}

/// Reads data from Data Storage (DSBLOCK) pointed to by Data Stream channels (cn_type = 8)
/// or Dynamic Arrays (ca_storage = 5).
fn read_ds(
    rdr: &mut SliceCursor<'_>,
    channel_group: &mut Cg4,
    vlsd_channels: &[(u8, i64)],
    mut position: i64,
    decoder: &mut Dec,
) -> Result<i64> {
    use crate::mdfinfo::mdfinfo4::Compo;
    for (_cn_type, rec_pos) in vlsd_channels {
        if let Some(cn) = channel_group.cn.get_mut(rec_pos) {
            let mut ds_data_pointer = 0i64;
            // Check for DSBLOCK in composition
            if let Some(composition) = &cn.composition {
                match &composition.block {
                    Compo::DS(ds) => {
                        ds_data_pointer = ds.ds_data();
                    }
                    Compo::CA(ca) if ca.ca_storage == 5 => {
                        if let Some(sub_compo) = &composition.compo
                            && let Compo::DS(ds) = &sub_compo.block
                        {
                            ds_data_pointer = ds.ds_data();
                        }
                    }
                    _ => {}
                }
            }

            if ds_data_pointer != 0 {
                rdr.seek_relative(ds_data_pointer - position)
                    .context("Could not position buffer for DS data block")?;
                position = ds_data_pointer;

                let mut id = [0u8; 4];
                rdr.read_exact(&mut id)
                    .context("could not read DS data block id")?;

                let data: Cow<'_, [u8]> = match read_all_blocks_to_bytes(rdr, id, position)? {
                    Some((buf, pos)) => {
                        position = pos;
                        buf
                    }
                    None => continue,
                };

                // Check if this is data stream mode (ds_mode == 0) with composition
                // We need to extract what we need from cn before dropping the borrow
                let ds_decode_info: Option<(Ds4Block, Box<Composition>)> =
                    if let Some(composition) = &cn.composition {
                        if let Compo::DS(ds_block) = &composition.block {
                            if ds_block.ds_mode == 0 {
                                // Clone the ds_block (dereferencing the Box) and the composition
                                composition
                                    .compo
                                    .as_ref()
                                    .map(|c| ((**ds_block).clone(), c.clone()))
                            } else if ds_block.ds_mode == 1 {
                                // Data description mode - data layout is described by an external
                                // attachment file (e.g., FIBEX, DBC, ARXML) pointed to by ds_cn_composition.
                                // This mode is not yet fully supported.
                                warn!(
                                    "Channel '{}' uses data description mode (ds_mode=1). \
                                    Data layout is described by an external attachment. \
                                    This mode requires external description file parsing \
                                    (FIBEX, DBC, ARXML) which is not yet implemented. \
                                    Data will be stored as raw bytes.",
                                    cn.unique_name
                                );
                                None
                            } else {
                                warn!(
                                    "Channel '{}' has unknown DSBLOCK mode: {}",
                                    cn.unique_name, ds_block.ds_mode
                                );
                                None
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                if let Some((ds_block, child_composition)) = ds_decode_info {
                    // Parse record offsets and sizes from length-prefixed data
                    let (record_offsets, record_sizes) = parse_vlsd_record_offsets(&data)?;

                    // Decode using composition
                    let decoded = datastream_decoder::decode_datastream_blob(
                        &data,
                        &ds_block,
                        &child_composition,
                        &channel_group.cn,
                        &record_offsets,
                        &record_sizes,
                    )?;

                    // Store decoded child channel data
                    for (rec_pos, values) in decoded {
                        if let Some(child_cn) = channel_group.cn.get_mut(&rec_pos) {
                            store_decoded_values_in_channel(child_cn, values, decoder)?;
                        }
                    }
                } else {
                    // Fallback: read as length-prefixed samples for dynamic data
                    if let Some(cn) = channel_group.cn.get_mut(rec_pos) {
                        read_vlsd_from_slice(&data, cn, 0, decoder)?;
                    }
                }
            }
        }
    }
    Ok(position)
}

/// Parses VLSD record offsets and sizes from a length-prefixed data blob.
/// Returns (offsets, sizes) vectors where each element corresponds to one record.
fn parse_vlsd_record_offsets(data: &[u8]) -> Result<(Vec<u64>, Vec<u64>)> {
    let mut offsets = Vec::new();
    let mut sizes = Vec::new();
    let mut position: usize = 0;

    while position + 4 <= data.len() {
        let length = u32::from_le_bytes(
            data[position..position + 4]
                .try_into()
                .context("Could not read VLSD length prefix")?,
        ) as usize;

        // Record starts after the 4-byte length prefix
        offsets.push((position + 4) as u64);
        sizes.push(length as u64);

        // Move to next record
        position += 4 + length;

        if position > data.len() {
            break;
        }
    }

    Ok((offsets, sizes))
}

/// Decodes an unsigned integer of `1..=8` bytes into a `u64`.
fn decode_u64_var_bytes(bytes: &[u8], is_big: bool) -> u64 {
    let len = bytes.len().min(8);
    let mut buf = [0u8; 8];
    if is_big {
        buf[8 - len..8].copy_from_slice(&bytes[..len]);
        u64::from_be_bytes(buf)
    } else {
        buf[..len].copy_from_slice(&bytes[..len]);
        u64::from_le_bytes(buf)
    }
}

/// Decodes a signed integer of `1..=8` bytes into an `i64` with sign extension.
fn decode_i64_var_bytes(bytes: &[u8], is_big: bool) -> i64 {
    let len = bytes.len().min(8);
    if len == 0 {
        return 0;
    }
    let val = decode_u64_var_bytes(bytes, is_big);
    let bits = len * 8;
    if bits >= 64 {
        val as i64
    } else {
        let shift = 64 - bits;
        ((val as i64) << shift) >> shift
    }
}

/// Stores decoded byte values into a channel's data structure
fn store_decoded_values_in_channel(
    cn: &mut Cn4,
    values: Vec<Vec<u8>>,
    decoder: &mut Dec,
) -> Result<()> {
    if cn.builder.is_none() {
        cn.init_builder(values.len())?;
    }
    let mut str_buf = String::new();
    let is_big = cn.endian.is_big();
    let cn_data_type = cn.block.cn_data_type;
    let builder = cn
        .builder
        .as_deref_mut()
        .ok_or_else(|| anyhow::anyhow!("builder should be initialized"))?;
    for (i, value_bytes) in values.into_iter().enumerate() {
        if value_bytes.is_empty()
            && !matches!(
                builder,
                ChannelBuilder::Utf8(_) | ChannelBuilder::VariableSizeByteArray(_)
            )
        {
            continue;
        }
        match builder {
            ChannelBuilder::Int8(b) => {
                b.values_slice_mut()[i] = value_bytes[0] as i8;
            }
            ChannelBuilder::UInt8(b) => {
                b.values_slice_mut()[i] = value_bytes[0];
            }
            ChannelBuilder::Int16(b) => {
                b.values_slice_mut()[i] = decode_i64_var_bytes(&value_bytes, is_big) as i16;
            }
            ChannelBuilder::UInt16(b) => {
                b.values_slice_mut()[i] = decode_u64_var_bytes(&value_bytes, is_big) as u16;
            }
            ChannelBuilder::Int32(b) => {
                b.values_slice_mut()[i] = decode_i64_var_bytes(&value_bytes, is_big) as i32;
            }
            ChannelBuilder::UInt32(b) => {
                b.values_slice_mut()[i] = decode_u64_var_bytes(&value_bytes, is_big) as u32;
            }
            ChannelBuilder::Float32(b) if value_bytes.len() >= 4 => {
                let val = if is_big {
                    f32::from_be_bytes(value_bytes[..4].try_into()?)
                } else {
                    f32::from_le_bytes(value_bytes[..4].try_into()?)
                };
                b.values_slice_mut()[i] = val;
            }
            ChannelBuilder::Float32(b) if value_bytes.len() == 2 => {
                let bytes: [u8; 2] = value_bytes[..2].try_into()?;
                let val = if is_big {
                    half::f16::from_be_bytes(bytes).to_f32()
                } else {
                    half::f16::from_le_bytes(bytes).to_f32()
                };
                b.values_slice_mut()[i] = val;
            }
            ChannelBuilder::Int64(b) => {
                b.values_slice_mut()[i] = decode_i64_var_bytes(&value_bytes, is_big);
            }
            ChannelBuilder::UInt64(b) => {
                b.values_slice_mut()[i] = decode_u64_var_bytes(&value_bytes, is_big);
            }
            ChannelBuilder::Float64(b) if value_bytes.len() >= 8 => {
                let val = if is_big {
                    f64::from_be_bytes(value_bytes[..8].try_into()?)
                } else {
                    f64::from_le_bytes(value_bytes[..8].try_into()?)
                };
                b.values_slice_mut()[i] = val;
            }
            ChannelBuilder::Complex32(b) if value_bytes.len() >= 8 => {
                let (re, im) = if is_big {
                    (
                        f32::from_be_bytes(value_bytes[..4].try_into()?),
                        f32::from_be_bytes(value_bytes[4..8].try_into()?),
                    )
                } else {
                    (
                        f32::from_le_bytes(value_bytes[..4].try_into()?),
                        f32::from_le_bytes(value_bytes[4..8].try_into()?),
                    )
                };
                let slice = b.values_slice_mut();
                slice[i * 2] = re;
                slice[i * 2 + 1] = im;
            }
            ChannelBuilder::Complex32(b) if value_bytes.len() == 4 => {
                let (re, im) = if is_big {
                    (
                        half::f16::from_be_bytes(value_bytes[..2].try_into()?).to_f32(),
                        half::f16::from_be_bytes(value_bytes[2..4].try_into()?).to_f32(),
                    )
                } else {
                    (
                        half::f16::from_le_bytes(value_bytes[..2].try_into()?).to_f32(),
                        half::f16::from_le_bytes(value_bytes[2..4].try_into()?).to_f32(),
                    )
                };
                let slice = b.values_slice_mut();
                slice[i * 2] = re;
                slice[i * 2 + 1] = im;
            }
            ChannelBuilder::Complex64(b) if value_bytes.len() >= 16 => {
                let (re, im) = if is_big {
                    (
                        f64::from_be_bytes(value_bytes[..8].try_into()?),
                        f64::from_be_bytes(value_bytes[8..16].try_into()?),
                    )
                } else {
                    (
                        f64::from_le_bytes(value_bytes[..8].try_into()?),
                        f64::from_le_bytes(value_bytes[8..16].try_into()?),
                    )
                };
                let slice = b.values_slice_mut();
                slice[i * 2] = re;
                slice[i * 2 + 1] = im;
            }
            ChannelBuilder::Utf8(b) => {
                b.append_value(decode_string_bytes(
                    &value_bytes,
                    cn_data_type,
                    decoder,
                    &mut str_buf,
                )?);
            }
            ChannelBuilder::VariableSizeByteArray(b) => {
                b.append_value(&value_bytes);
            }
            _ => {
                // For other types (complex, tensor, etc.), skip for now
            }
        }
    }
    cn.data = cn.builder.take().unwrap().finish();
    Ok(())
}

// =============================================================================
// Sample Reduction Data Reading (RDBLOCK/RVBLOCK/RIBLOCK)
// =============================================================================
