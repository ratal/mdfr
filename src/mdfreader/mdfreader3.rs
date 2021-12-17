use rayon::prelude::*;

use crate::mdfinfo::mdfinfo3::{MdfInfo3, Cg3, Dg3};
use std::collections::{HashSet, HashMap};
use std::fs::File;
use std::io::{BufReader, Read};

use crate::mdfreader::data_read3::read_channels_from_bytes;
use crate::mdfreader::conversions3::convert_all_channels;

// The following constant represents the size of data chunk to be read and processed.
// a big chunk will improve performance but consume more memory
// a small chunk will not consume too much memory but will cause many read calls, penalising performance
const CHUNK_SIZE_READING: usize = 524288; // can be tuned according to architecture

pub fn mdfreader3<'a>(
    rdr: &'a mut BufReader<&File>,
    info: &'a mut MdfInfo3,
    channel_names: HashSet<String>,
) {
    let mut position: i64 = 0;
    let mut channel_names_present_in_dg: HashSet<String>;
    // read file data
    for (data_position, dg) in info.dg.iter_mut() {
        // Let's find channel names
        channel_names_present_in_dg = HashSet::new();
        for channel_group in dg.cg.values() {
            let cn = channel_group.channel_names.clone();
            channel_names_present_in_dg.par_extend(cn);
        }
        let channel_names_to_read_in_dg: HashSet<String> = channel_names_present_in_dg
            .into_par_iter()
            .filter(|v| channel_names.contains(v))
            .collect();
        if dg.block.dg_data != 0 && !channel_names_to_read_in_dg.is_empty() {
            // header block
            rdr.seek_relative(*data_position as i64 - position)
                .expect("Could not position buffer"); // change buffer position
            if dg.cg.len() == 1 {
                // sorted data group
                for channel_group in dg.cg.values_mut() {
                    read_all_channels_sorted(rdr, channel_group, &channel_names_to_read_in_dg);
                    position = *data_position as i64 + (channel_group.record_length as i64) * (channel_group.block.cg_cycle_count as i64);
                }
            } else if !dg.cg.is_empty() {
                // unsorted data
                // initialises all arrays
                let mut block_length:i64 = 0;
                for channel_group in dg.cg.values_mut() {
                    initialise_arrays(
                        channel_group,
                        &channel_group.block.cg_cycle_count.clone(),
                        &channel_names_to_read_in_dg,
                    );
                    block_length += channel_group.record_length as i64 * channel_group.block.cg_cycle_count as i64;
                }
                position = *data_position as i64 + block_length;
                read_all_channels_unsorted(
                    rdr,
                    dg,
                    block_length,
                    &channel_names_to_read_in_dg,
                );
            }
            // conversion of all channels to physical values
            convert_all_channels(dg, &info.sharable);
        }
    }
}

/// initialise ndarrays for the data group/block
fn initialise_arrays(
    channel_group: &mut Cg3,
    cg_cycle_count: &u32,
    channel_names_to_read_in_dg: &HashSet<String>,
) {
    // creates zeroed array in parallel for each channel contained in channel group
    channel_group
        .cn
        .par_iter_mut()
        .filter(|(_cn_position, cn)| channel_names_to_read_in_dg.contains(&cn.unique_name))
        .for_each(|(_cn_position, cn)| {
            cn.data = cn
                .data
                .zeros(0, *cg_cycle_count as u64, cn.n_bytes as u32, 0);
        })
}

/// Returns chunk size and corresponding number of records from a channel group
fn generate_chunks(channel_group: &Cg3) -> Vec<(usize, usize)> {
    let record_length = channel_group.record_length as usize;
    let cg_cycle_count = channel_group.block.cg_cycle_count as usize;
    let n_chunks = (record_length * cg_cycle_count) / CHUNK_SIZE_READING + 1; // number of chunks
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
    channel_names_to_read_in_dg: &HashSet<String>,
) {
    let chunks = generate_chunks(channel_group);
    // initialises the arrays
    initialise_arrays(
        channel_group,
        &channel_group.block.cg_cycle_count.clone(),
        channel_names_to_read_in_dg,
    );
    // read by chunks and store in channel array
    let mut previous_index: usize = 0;
    for (n_record_chunk, chunk_size) in chunks {
        let mut data_chunk = vec![0u8; chunk_size];
        rdr.read_exact(&mut data_chunk)
            .expect("Could not read data chunk");
        read_channels_from_bytes(
            &data_chunk,
            &mut channel_group.cn,
            channel_group.record_length as usize,
            previous_index,
            channel_names_to_read_in_dg,
        );
        previous_index += n_record_chunk;
    }
}

/// Reads unsorted data block chunk by chunk
fn read_all_channels_unsorted(
    rdr: &mut BufReader<&File>,
    dg: &mut Dg3,
    block_length: i64,
    channel_names_to_read_in_dg: &HashSet<String>,
) {
    let data_block_length = block_length as usize;
    let mut position: usize = 0;
    let mut record_counter: HashMap<u16, (usize, Vec<u8>)> = HashMap::new();

    // initialise record counter that will contain sorted data blocks for each channel group
    for cg in dg.cg.values_mut() {
        record_counter.insert(cg.block.cg_record_id, (0, Vec::new()));
    }

    // reads the sorted data block into chunks
    let mut data_chunk: Vec<u8>;
    while position < data_block_length {
        if (data_block_length - position) > CHUNK_SIZE_READING {
            // not last chunk of data
            data_chunk = vec![0u8; CHUNK_SIZE_READING];
            position += CHUNK_SIZE_READING;
        } else {
            // last chunk of data
            data_chunk = vec![0u8; data_block_length - position];
            position += data_block_length - position;
        }
        rdr.read_exact(&mut data_chunk)
            .expect("Could not read data chunk");
        read_all_channels_unsorted_from_bytes(
            &mut data_chunk,
            dg,
            &mut record_counter,
            channel_names_to_read_in_dg,
        );
    }
}

/// read record by record from unsorted data block into sorted data block, then copy data into channel arrays
fn read_all_channels_unsorted_from_bytes(
    data: &mut Vec<u8>,
    dg: &mut Dg3,
    record_counter: &mut HashMap<u16, (usize, Vec<u8>)>,
    channel_names_to_read_in_dg: &HashSet<String>,
) {
    let mut position: usize = 0;
    let data_length = data.len();
    // unsort data into sorted data blocks, except for VLSD CG.
    let mut remaining: usize = data_length - position;
    while remaining > 0 {
        // reads record id
        let rec_id: u16;
        if remaining >= 1 {
            rec_id = data[position]
                .try_into()
                .expect("Could not convert record id u8");
        } else {
            break; // not enough data remaining
        }
        // reads record based on record id
        if let Some(cg) = dg.cg.get_mut(&rec_id) {
            let record_length = cg.record_length as usize;
            if remaining >= record_length {
                let record = &data[position..position + cg.record_length as usize];
                if let Some((_nrecord, data)) = record_counter.get_mut(&rec_id) {
                    data.extend(record);
                }
                position += record_length;
            } else {
                break; // not enough data remaining
            }
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
            read_channels_from_bytes(
                &record_data,
                &mut channel_group.cn,
                channel_group.record_length as usize,
                *index,
                channel_names_to_read_in_dg,
            );
            record_data.clear(); // clears data for new block, keeping capacity
        }
    }
}