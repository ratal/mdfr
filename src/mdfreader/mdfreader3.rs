//! data read and load in memory based in MdfInfo3's metadata
use rayon::prelude::*;

use crate::mdfinfo::MdfInfo;
use crate::mdfinfo::mdfinfo3::{Cg3, Cn3, Dg3};
use crate::mdfinfo::sym_buf_reader::SliceCursor;
use anyhow::{Context, Error, Result};
use std::collections::{HashMap, HashSet};
use std::io::Seek;

use crate::data_holder::channel_builder::ChannelBuilder;
use crate::mdfreader::data_read3::read_channels_from_bytes;

use super::Mdf;

/// The following constant represents the size of data chunk to be read and processed.
/// a big chunk will improve performance but consume more memory
/// a small chunk will not consume too much memory but will cause many read calls, penalising performance
pub const CHUNK_SIZE_READING_3: usize = 524288; // can be tuned according to architecture

/// Reads the file data based on headers information contained in info parameter
/// Hashset of channel names parameter allows to filter which channels to read
pub fn mdfreader3(
    rdr: &mut SliceCursor<'_>,
    mdf: &mut Mdf,
    channel_names: &HashSet<String>,
) -> Result<(), Error> {
    match &mut mdf.mdf_info {
        MdfInfo::V3(info) => {
            let mut position: i64 = 0;
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
                if dg.block.dg_data != 0 && has_readable {
                    // header block
                    rdr.seek_relative(dg.block.dg_data as i64 - position)
                        .context("Could not position buffer")?; // change buffer position
                    if dg.cg.len() == 1 {
                        // sorted data group
                        for channel_group in dg.cg.values_mut() {
                            read_all_channels_sorted(rdr, channel_group)?;
                            position = dg.block.dg_data as i64
                                + (channel_group.record_length as i64)
                                    * (channel_group.block.cg_cycle_count as i64);
                        }
                    } else if !dg.cg.is_empty() {
                        // unsorted data
                        // initialises all arrays
                        let mut block_length: i64 = 0;
                        for channel_group in dg.cg.values_mut() {
                            initialise_arrays(
                                channel_group,
                                &channel_group.block.cg_cycle_count.clone(),
                            )
                            .context("failed initialising arrays for unsorted channels")?;
                            block_length += channel_group.record_length as i64
                                * channel_group.block.cg_cycle_count as i64;
                        }
                        position = dg.block.dg_data as i64 + block_length;
                        read_all_channels_unsorted(rdr, dg, block_length)?;
                    }

                    // Defer conversion of channels to physical values (handled lazily on access)
                }
            }
        }
        MdfInfo::V4(_) => {}
    };
    Ok(())
}

/// initialise ndarrays for the data group/block
fn initialise_arrays(channel_group: &mut Cg3, cg_cycle_count: &u32) -> Result<(), Error> {
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
                    format!(
                        "failed intialising with zeros the channel's array named {}",
                        cn.unique_name
                    )
                })?;
            }
        }
        return Ok(());
    }
    // creates zeroed array in parallel for each channel contained in channel group
    channel_group
        .cn
        .par_iter_mut()
        .filter(|(_cn_position, cn)| cn.should_read)
        .try_for_each(
            |(_cn_position, cn): (&u32, &mut Cn3)| -> Result<(), Error> {
                cn.init_builder(*cg_cycle_count as usize).with_context(|| {
                    format!(
                        "failed intialising with zeros the channel's array named {}",
                        cn.unique_name
                    )
                })?;
                Ok(())
            },
        )
        .context("Failed initialising channel group with zeros arrays")?;
    Ok(())
}

/// Returns chunk size and corresponding number of records from a channel group
fn generate_chunks(channel_group: &Cg3) -> Vec<(usize, usize)> {
    let record_length = channel_group.record_length as usize;
    let cg_cycle_count = channel_group.block.cg_cycle_count as usize;
    if record_length == 0 || cg_cycle_count == 0 {
        return Vec::new();
    }
    let n_record_chunk = (CHUNK_SIZE_READING_3 / record_length).max(1);
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
    channel_group: &mut Cg3,
) -> Result<(), Error> {
    let chunks = generate_chunks(channel_group);
    // initialises the arrays
    initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone())
        .context("failed initialising arrays for sorted channels")?;
    // read by chunks directly from mapped slice and store in channel array
    let mut previous_index: usize = 0;
    let mut chunk_start_pos: usize = rdr.stream_position().unwrap_or(0) as usize;
    let mut advised_bytes: usize = 0;
    for (n_record_chunk, chunk_size) in chunks {
        let data_chunk = rdr
            .read_slice(chunk_size)
            .context("Could not read data chunk")?;
        read_channels_from_bytes(
            data_chunk,
            &mut channel_group.cn,
            channel_group.record_length as usize,
            previous_index,
        )
        .context("failed reading channels from bytes")?;
        previous_index += n_record_chunk;
        advised_bytes += chunk_size;
        if advised_bytes >= 4 * 1024 * 1024 {
            rdr.advise_dontneed_range(chunk_start_pos, advised_bytes);
            chunk_start_pos += advised_bytes;
            advised_bytes = 0;
        }
    }
    if advised_bytes >= 1_048_576 {
        rdr.advise_dontneed_range(chunk_start_pos, advised_bytes);
    }
    apply_bit_mask_offset_3(&mut channel_group.cn)?;
    for cn in channel_group.cn.values_mut() {
        if let Some(b) = cn.builder.take() {
            cn.data = b.finish();
        }
    }
    Ok(())
}

const UNSORTED_CG_INIT_CAP_BYTES_3: usize = 64 * 1024;
const UNSORTED_CG_FLUSH_BYTES_3: usize = 1024 * 1024;

enum RecordIdMap3 {
    Direct(Vec<u16>),
    Sparse(rustc_hash::FxHashMap<u16, u16>),
}

impl RecordIdMap3 {
    #[inline(always)]
    fn get(&self, rec_id: u16) -> Option<usize> {
        match self {
            Self::Direct(table) => {
                let slot = *table.get(rec_id as usize)?;
                if slot != u16::MAX {
                    Some(slot as usize)
                } else {
                    None
                }
            }
            Self::Sparse(map) => map.get(&rec_id).copied().map(|s| s as usize),
        }
    }
}

struct Cg3Slot {
    rec_id: u16,
    record_length: usize,
    cycle_count: usize,
    should_read: bool,
    write_index: usize,
    data: Vec<u8>,
}

fn flush_cg3_slot(dg: &mut Dg3, slot: &mut Cg3Slot) -> Result<()> {
    if slot.data.is_empty() {
        return Ok(());
    }
    if let Some(channel_group) = dg.cg.get_mut(&slot.rec_id) {
        let record_length = slot.record_length;
        if record_length > 0 && slot.write_index < slot.cycle_count {
            let n_records = slot.data.len() / record_length;
            let records_to_read = n_records.min(slot.cycle_count - slot.write_index);
            if records_to_read > 0 {
                read_channels_from_bytes(
                    &slot.data[..records_to_read * record_length],
                    &mut channel_group.cn,
                    record_length,
                    slot.write_index,
                )
                .context("failed reading channels from bytes")?;
                slot.write_index += records_to_read;
            }
        }
    }
    slot.data.clear();
    Ok(())
}

/// Reads unsorted data block chunk by chunk directly from mapped slice
fn read_all_channels_unsorted(
    rdr: &mut SliceCursor<'_>,
    dg: &mut Dg3,
    block_length: i64,
) -> Result<()> {
    let data_block_length = block_length as usize;
    let data = rdr
        .read_slice(data_block_length)
        .context("Could not read data chunk")?;

    let max_rec_id = dg
        .cg
        .values()
        .map(|cg| cg.block.cg_record_id)
        .max()
        .unwrap_or(0);
    let use_direct = max_rec_id < 4096 && dg.cg.len() < u16::MAX as usize;
    let mut id_map = if use_direct {
        RecordIdMap3::Direct(vec![u16::MAX; max_rec_id as usize + 1])
    } else {
        RecordIdMap3::Sparse(rustc_hash::FxHashMap::with_capacity_and_hasher(
            dg.cg.len(),
            Default::default(),
        ))
    };
    let mut slots = Vec::with_capacity(dg.cg.len());
    for cg in dg.cg.values() {
        let rec_id = cg.block.cg_record_id;
        let slot_idx = slots.len() as u16;
        match &mut id_map {
            RecordIdMap3::Direct(table) => table[rec_id as usize] = slot_idx,
            RecordIdMap3::Sparse(map) => {
                map.insert(rec_id, slot_idx);
            }
        }
        let record_length = cg.record_length as usize;
        let cycle_count = cg.block.cg_cycle_count as usize;
        let should_read = cg.cn.values().any(|cn| cn.should_read);
        let init_cap = if should_read {
            cycle_count
                .saturating_mul(record_length)
                .min(UNSORTED_CG_INIT_CAP_BYTES_3)
        } else {
            0
        };
        slots.push(Cg3Slot {
            rec_id,
            record_length,
            cycle_count,
            should_read,
            write_index: 0,
            data: Vec::with_capacity(init_cap),
        });
    }

    let is_2byte_id = dg.block.dg_n_record_ids == 2;
    let mut offset: usize = 0;
    let mut advised_start: usize = 0;
    while offset < data.len() {
        let mut window_end = (offset + CHUNK_SIZE_READING_3).min(data.len());
        loop {
            let consumed = read_all_channels_unsorted_from_bytes(
                &data[offset..window_end],
                dg,
                &id_map,
                &mut slots,
                is_2byte_id,
            )
            .context("failed reading channels from bytes")?;
            if consumed > 0 {
                offset += consumed;
                break;
            }
            if window_end == data.len() {
                offset = data.len();
                break;
            }
            window_end = data.len();
        }
        if offset.saturating_sub(advised_start) >= 4 * 1024 * 1024 {
            rdr.advise_dontneed_slice(&data[advised_start..offset]);
            advised_start = offset;
        }
    }
    if advised_start < data.len() {
        rdr.advise_dontneed_slice(&data[advised_start..]);
    }
    for slot in &mut slots {
        flush_cg3_slot(dg, slot)?;
    }
    for cg in dg.cg.values_mut() {
        apply_bit_mask_offset_3(&mut cg.cn)?;
        for cn in cg.cn.values_mut() {
            if let Some(b) = cn.builder.take() {
                cn.data = b.finish();
            }
        }
    }
    Ok(())
}

/// Demuxes records from an unsorted MDF3 byte slice into bounded per-CG staging buffers,
/// skipping unread CGs and flushing individual CG buffers once they reach `UNSORTED_CG_FLUSH_BYTES_3`.
fn read_all_channels_unsorted_from_bytes(
    data: &[u8],
    dg: &mut Dg3,
    id_map: &RecordIdMap3,
    slots: &mut [Cg3Slot],
    is_2byte_id: bool,
) -> Result<usize, Error> {
    let mut position: usize = 0;
    let data_length = data.len();
    let mut remaining: usize = data_length - position;

    while remaining > 0 {
        let rec_id: u16 = if is_2byte_id {
            if remaining < 2 {
                break; // not enough data remaining for 2-byte record ID
            }
            u16::from_le_bytes([data[position], data[position + 1]])
        } else {
            if remaining < 1 {
                break; // not enough data remaining for 1-byte record ID
            }
            data[position] as u16
        };

        let Some(slot_idx) = id_map.get(rec_id) else {
            // Cleanly break on unknown rec_id to prevent infinite loop
            break;
        };
        let slot = &mut slots[slot_idx];
        let record_length = slot.record_length;
        if remaining >= record_length {
            if slot.should_read {
                let record = &data[position..position + record_length];
                slot.data.extend_from_slice(record);
                if slot.data.len() >= UNSORTED_CG_FLUSH_BYTES_3 {
                    flush_cg3_slot(dg, slot)?;
                }
            }
            position += record_length;
        } else {
            break; // not enough data remaining
        }
        remaining = data_length - position;
    }

    Ok(position)
}

/// Applies bit-offset masking and shifting to a single MDF3 integer channel.
fn apply_bit_mask_offset_cn3(cn: &mut Cn3) {
    let bit_offset = (cn.block2.cn_bit_offset % 8) as u32;
    let bit_count = cn.block2.cn_bit_count as u32;
    let dt = cn.block2.cn_data_type;
    let is_int = dt <= 1 || (9..=10).contains(&dt) || (13..=14).contains(&dt);
    if is_int && bit_count > 0 && (bit_offset > 0 || bit_count != cn.n_bytes as u32 * 8) {
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
                ChannelBuilder::Int8(a) => shift_slice!(a, 8),
                ChannelBuilder::UInt8(a) => shift_slice!(a, 8),
                ChannelBuilder::Int16(a) => shift_slice!(a, 16),
                ChannelBuilder::UInt16(a) => shift_slice!(a, 16),
                ChannelBuilder::Int32(a) => shift_slice!(a, 32),
                ChannelBuilder::UInt32(a) => shift_slice!(a, 32),
                ChannelBuilder::Int64(a) => shift_slice!(a, 64),
                ChannelBuilder::UInt64(a) => shift_slice!(a, 64),
                _ => {}
            }
        }
    }
}

pub fn apply_bit_mask_offset_3(channels: &mut HashMap<u32, Cn3>) -> Result<(), Error> {
    let readable_sample = channels
        .values()
        .filter(|cn| cn.should_read)
        .take(3)
        .count();
    if readable_sample == 0 {
        return Ok(());
    }
    if readable_sample <= 2 {
        for cn in channels.values_mut() {
            if cn.should_read {
                apply_bit_mask_offset_cn3(cn);
            }
        }
    } else {
        channels
            .par_iter_mut()
            .filter(|(_, cn)| cn.should_read)
            .for_each(|(_, cn)| apply_bit_mask_offset_cn3(cn));
    }
    Ok(())
}
