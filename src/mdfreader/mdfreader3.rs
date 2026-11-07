//! data read and load in memory based in MdfInfo3's metadata
use rayon::prelude::*;

use crate::mdfinfo::MdfInfo;
use crate::mdfinfo::mdfinfo3::{Cg3, Cn3, Dg3};
use anyhow::{Context, Error, Result};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read};

use crate::data_holder::channel_builder::ChannelBuilder;
use crate::mdfreader::data_read3::read_channels_from_bytes;

use super::Mdf;

/// The following constant represents the size of data chunk to be read and processed.
/// a big chunk will improve performance but consume more memory
/// a small chunk will not consume too much memory but will cause many read calls, penalising performance
pub const CHUNK_SIZE_READING_3: usize = 524288; // can be tuned according to architecture

/// Reads the file data based on headers information contained in info parameter
/// Hashset of channel names parameter allows to filter which channels to read
pub fn mdfreader3<'a>(
    rdr: &'a mut BufReader<&File>,
    mdf: &'a mut Mdf,
    channel_names: &HashSet<String>,
) -> Result<(), Error> {
    match &mut mdf.mdf_info {
        MdfInfo::V3(info) => {
            let mut position: i64 = 0;
            for dg in info.dg.values_mut() {
                for cg in dg.cg.values_mut() {
                    for cn in cg.cn.values_mut() {
                        cn.should_read = channel_names.contains(&cn.unique_name);
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
    let n_chunks = (record_length * cg_cycle_count) / CHUNK_SIZE_READING_3 + 1; // number of chunks
    let chunk_length = (record_length * cg_cycle_count) / n_chunks; // chunks length
    let n_record_chunk = chunk_length / record_length; // number of records in chunk
    let chunck = (n_record_chunk, record_length * n_record_chunk);
    let mut chunks = vec![chunck; n_chunks];
    let n_record_chunk = cg_cycle_count - n_record_chunk * n_chunks;
    if n_record_chunk > 0 {
        chunks.push((n_record_chunk, record_length * n_record_chunk))
    }
    chunks
}

/// Reads all channels from given channel group having sorted data blocks
fn read_all_channels_sorted(
    rdr: &mut BufReader<&File>,
    channel_group: &mut Cg3,
) -> Result<(), Error> {
    let chunks = generate_chunks(channel_group);
    // initialises the arrays
    initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone())
        .context("failed initialising arrays for sorted channels")?;
    // read by chunks and store in channel array
    let mut previous_index: usize = 0;
    for (n_record_chunk, chunk_size) in chunks {
        let mut data_chunk = vec![0u8; chunk_size];
        rdr.read_exact(&mut data_chunk)
            .context("Could not read data chunk")?;
        read_channels_from_bytes(
            &data_chunk,
            &mut channel_group.cn,
            channel_group.record_length as usize,
            previous_index,
        )
        .context("failed reading channels from bytes")?;
        previous_index += n_record_chunk;
    }
    apply_bit_mask_offset_3(&mut channel_group.cn)?;
    for cn in channel_group.cn.values_mut() {
        if let Some(b) = cn.builder.take() {
            cn.data = b.finish();
        }
    }
    Ok(())
}

/// Reads unsorted data block chunk by chunk
fn read_all_channels_unsorted(
    rdr: &mut BufReader<&File>,
    dg: &mut Dg3,
    block_length: i64,
) -> Result<()> {
    let data_block_length = block_length as usize;
    let mut position: usize = 0;
    let mut record_counter: HashMap<u16, (usize, Vec<u8>)> = HashMap::new();

    // initialise record counter that will contain sorted data blocks for each channel group
    for cg in dg.cg.values_mut() {
        record_counter.insert(cg.block.cg_record_id, (0, Vec::new()));
    }

    let mut trailing_buf: Vec<u8> = Vec::new();

    while position < data_block_length {
        let to_read = (data_block_length - position).min(CHUNK_SIZE_READING_3);
        let mut read_buf = vec![0u8; to_read];
        rdr.read_exact(&mut read_buf)
            .context("Could not read data chunk")?;
        position += to_read;

        let mut data_chunk = Vec::with_capacity(trailing_buf.len() + read_buf.len());
        data_chunk.extend_from_slice(&trailing_buf);
        data_chunk.extend_from_slice(&read_buf);

        read_all_channels_unsorted_from_bytes(&mut data_chunk, dg, &mut record_counter)
            .context("failed reading channels from bytes")?;

        trailing_buf = data_chunk; // Preserves unconsumed trailing bytes for next chunk
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

/// read record by record from unsorted data block into sorted data block, then copy data into channel arrays
fn read_all_channels_unsorted_from_bytes(
    data: &mut Vec<u8>,
    dg: &mut Dg3,
    record_counter: &mut HashMap<u16, (usize, Vec<u8>)>,
) -> Result<(), Error> {
    let mut position: usize = 0;
    let data_length = data.len();
    // unsort data into sorted data blocks, except for VLSD CG.
    let mut remaining: usize = data_length - position;
    let is_2byte_id = dg.block.dg_n_record_ids == 2;

    while remaining > 0 {
        // reads record id
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

        // reads record based on record id
        let Some(cg) = dg.cg.get_mut(&rec_id) else {
            // Cleanly break on unknown rec_id to prevent infinite loop
            break;
        };

        let record_length = cg.record_length as usize;
        if remaining >= record_length {
            let record = &data[position..position + record_length];
            if let Some((_nrecord, record_data)) = record_counter.get_mut(&rec_id) {
                record_data.extend_from_slice(record);
            }
            position += record_length;
        } else {
            break; // not enough data remaining
        }
        remaining = data_length - position;
    }

    // removes consumed records from data and leaves remaining that could not be processed.
    let remaining_vect = data[position..].to_owned();
    data.clear(); // removes data but keeps capacity
    data.extend(remaining_vect);

    // From sorted data block, copies data in channels arrays
    for (rec_id, (index, record_data)) in record_counter.iter_mut() {
        if let Some(channel_group) = dg.cg.get_mut(rec_id) {
            let record_length = channel_group.record_length as usize;
            if record_length > 0 && !record_data.is_empty() {
                let n_chunk_records = record_data.len() / record_length;
                read_channels_from_bytes(record_data, &mut channel_group.cn, record_length, *index)
                    .context("failed reading channels from bytes")?;
                *index += n_chunk_records;
                record_data.clear(); // clears data for new block, keeping capacity
            }
        }
    }
    Ok(())
}

pub fn apply_bit_mask_offset_3(channels: &mut HashMap<u32, Cn3>) -> Result<(), Error> {
    channels
        .par_iter_mut()
        .filter(|(_, cn)| cn.should_read)
        .for_each(|(_, cn)| {
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
                if let Some(builder) = &mut cn.builder {
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
        });
    Ok(())
}
