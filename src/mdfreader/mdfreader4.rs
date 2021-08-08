use crate::mdfinfo::mdfinfo4::{parse_block_header, Cg4, Cn4, CnType, Dg4, MdfInfo4};
use crate::mdfinfo::mdfinfo4::{
    parse_dz, parser_dl4_block, parser_ld4_block, Dl4Block, Dt4Block, Hl4Block, Ld4Block,
};
use crate::mdfreader::converions4::convert_all_channels;
use binread::BinReaderExt;
use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use encoding_rs::{Decoder, UTF_16BE, UTF_16LE, WINDOWS_1252};
use half::f16;
use ndarray::{Array, Array1, ArrayBase, Dim, OwnedRepr};
use num::Complex;
use rayon::prelude::*;
use std::fs::File;
use std::str;
use std::string::String;
use std::sync::{Mutex, Arc};
use std::{
    collections::HashMap,
    convert::TryInto,
    io::{BufReader, Read},
    usize,
};

// The following constant represents the size of data chunk to be read and processed.
// a big chunk will improve performance but consume more memory
// a small chunk will not consume too much memory but will cause many read calls, penalising performance
const CHUNK_SIZE_READING: usize = 524288; // can be tuned according to architecture

/// Reads the file data based on headers information contained in info parameter
pub fn mdfreader4<'a>(rdr: &'a mut BufReader<&File>, info: &'a mut MdfInfo4) {
    let mut position: i64 = 0;
    let mut sorted : bool;
    // read file data
    for (_dg_position, dg) in info.dg.iter_mut() {
        if dg.block.dg_data != 0 {
            // header block
            rdr.seek_relative(dg.block.dg_data - position)
                .expect("Could not position buffer"); // change buffer position
            let mut id = [0u8; 4];
            rdr.read_exact(&mut id).expect("could not read block id");
            if dg.cg.len() == 1 {
                sorted = true;
            } else {sorted = false}
            position = read_data(rdr, id, dg, dg.block.dg_data, sorted);
        }
        apply_bit_mask_offset(dg);
        // channel_group invalid bits calculation
        for channel_group in dg.cg.values_mut() {
            channel_group.process_all_channel_invalid_bits();
        }
        // conversion of all channels to physical values
        convert_all_channels(dg, &info.sharable.cc);
    }
}

/// Reads all kind of data layout : simple DT or DV, sorted or unsorted, Data List,
/// compressed data blocks DZ or Sample DATA
fn read_data(rdr: &mut BufReader<&File>, id: [u8; 4], dg: &mut Dg4, mut position: i64, sorted: bool) -> i64 {
    // block header is already read
    let mut decoder: Dec = Dec {
        windows_1252: WINDOWS_1252.new_decoder(),
        utf_16_be: UTF_16BE.new_decoder(),
        utf_16_le: UTF_16LE.new_decoder(),
    };
    let mut vlsd_channels: Vec<u32> = Vec::new();
    if "##DT".as_bytes() == id {
        let block_header: Dt4Block = rdr.read_le().unwrap();
        // simple data block
        if sorted {
            // sorted data group
            for channel_group in dg.cg.values_mut() {
                vlsd_channels = read_all_channels_sorted(rdr, channel_group);
                position += block_header.len as i64;
            }
            position = read_sd(rdr, dg, &vlsd_channels, position);
        } else if !dg.cg.is_empty() {
            // unsorted data
            // initialises all arrays
            for channel_group in dg.cg.values_mut() {
                initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone());
            }
            read_all_channels_unsorted(rdr, dg, block_header.len as i64);
            position += block_header.len as i64;
        }   
    } else if "##DZ".as_bytes() == id {
        let (mut data, block_header) = parse_dz(rdr);
        // compressed data
        if sorted {
            // sorted data group
            for channel_group in dg.cg.values_mut() {
                vlsd_channels = read_all_channels_sorted_from_bytes(&data, channel_group);
            }
            position += block_header.len as i64;
            position = read_sd(rdr, dg, &vlsd_channels, position);
        } else if !dg.cg.is_empty() {
            // unsorted data
            // initialises all arrays
            for channel_group in dg.cg.values_mut() {
                initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone());
            }
            // initialise record counter
            let mut record_counter: HashMap<u64, (usize, Vec<u8>)> = HashMap::new();
            for cg in dg.cg.values_mut() {
                record_counter.insert(
                    cg.block.cg_record_id,
                    (
                        0,
                        Vec::with_capacity(
                            (cg.record_length as u64 * cg.block.cg_cycle_count) as usize,
                        ),
                    ),
                );
            }
            read_all_channels_unsorted_from_bytes(&mut data, dg, &mut record_counter, &mut decoder);
            position += block_header.len as i64;
        }
    } else if "##HL".as_bytes() == id {
        let (pos, id) = read_hl(rdr, position);
        position = pos;
        // Read DL Blocks
        position = read_data(rdr, id, dg, position, sorted);
    } else if "##DL".as_bytes() == id {
        // data list
        if sorted {
            // sorted data group
            for channel_group in dg.cg.values_mut() {
                let (dl_blocks, pos) = parser_dl4(rdr, position);
                let (pos, vlsd) = parser_dl4_sorted(rdr, dl_blocks, pos, channel_group);
                position = pos;
                vlsd_channels = vlsd;
            }
            position = read_sd(rdr, dg, &vlsd_channels, position);
        } else if !dg.cg.is_empty() {
            // unsorted data
            // initialises all arrays
            for channel_group in dg.cg.values_mut() {
                initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone());
            }
            let (dl_blocks, pos) = parser_dl4(rdr, position);
            let pos = parser_dl4_unsorted(rdr, dg, dl_blocks, pos);
            position = pos;
        }
    } else if "##LD".as_bytes() == id {
        // list data, cannot be used for unsorted data
        for channel_group in dg.cg.values_mut() {
            let pos = parser_ld4(rdr, position, channel_group);
            position = pos;
        }
    } else if "##DV".as_bytes() == id {
        // data values
        // sorted data group only, no record id, no invalid bytes
        let block_header: Dt4Block = rdr.read_le().unwrap();
        for channel_group in dg.cg.values_mut() {
            match channel_group.cn.len() {
                l if l > 1 => {
                    read_all_channels_sorted(rdr, channel_group);},
                l if l == 1 => {
                    let cycle_count = channel_group.block.cg_cycle_count;
                    // only one channel, can be optimised
                    for (_rec_pos, cn) in channel_group.cn.iter_mut() {
                        read_one_channel_array(rdr, cn, cycle_count as usize);
                    }
                }
                _ => (),
            }
        }
        position += block_header.len as i64;
    }
    position
}

fn read_hl(rdr: &mut BufReader<&File>, mut position: i64) -> (i64, [u8; 4]) {
    // compressed data in datal list
    let block: Hl4Block = rdr.read_le().expect("could not read HL block");
    position += block.hl_len as i64;
    // Read Id of pointed DL Block
    rdr.seek_relative(block.hl_dl_first - position)
        .expect("Could not reach DL block from HL block");
    position = block.hl_dl_first;
    let mut id = [0u8; 4];
    rdr.read_exact(&mut id).expect("could not read DL block id");
    (position, id)
}

// reads Signal Data Block containing VLSD channel, pointed by cn_data
fn read_sd(rdr: &mut BufReader<&File>, dg: &mut Dg4, vlsd_channels: &Vec<u32>, mut position: i64) -> i64 {
    for channel_group in dg.cg.values_mut() {
        for rec_pos in vlsd_channels {
            if let Some(cn) = channel_group.cn.get_mut(&rec_pos) {
                // header block
                rdr.seek_relative(cn.block.cn_data - position)
                    .expect("Could not position buffer"); // change buffer position
                let mut id = [0u8; 4];
                rdr.read_exact(&mut id).expect("could not read block id");
                if "##SD".as_bytes() == id {
                    let block_header: Dt4Block = rdr.read_le().unwrap();

                } else if "##DZ".as_bytes() == id {
                    let (mut data, block_header) = parse_dz(rdr);

                } else if "##HL".as_bytes() == id {
                    let (pos, id) = read_hl(rdr, position);
                    position = pos;

                }else if "##DL".as_bytes() == id {
                    let (dl_blocks, pos) = parser_dl4(rdr, position);

                }

            }
        }
    }
    position
}

fn read_vlsd_from_bytes(data: &mut Vec<u8>, cn: &mut Cn4) {
    let mut decoder: Dec = Dec {
        windows_1252: WINDOWS_1252.new_decoder(),
        utf_16_be: UTF_16BE.new_decoder(),
        utf_16_le: UTF_16LE.new_decoder(),
    };
    let mut position: usize = 0;
    let data_length = data.len();
    let mut remaining: usize = data_length - position;
    let mut nrecord: usize = 0;
    match &mut cn.data {
        ChannelData::Int8(_) => {}
        ChannelData::UInt8(_) => {}
        ChannelData::Int16(_) => {}
        ChannelData::UInt16(_) => {}
        ChannelData::Float16(_) => {}
        ChannelData::Int24(_) => {}
        ChannelData::UInt24(_) => {}
        ChannelData::Int32(_) => {}
        ChannelData::UInt32(_) => {}
        ChannelData::Float32(_) => {}
        ChannelData::Int48(_) => {}
        ChannelData::UInt48(_) => {}
        ChannelData::Int64(_) => {}
        ChannelData::UInt64(_) => {}
        ChannelData::Float64(_) => {}
        ChannelData::Complex16(_) => {}
        ChannelData::Complex32(_) => {}
        ChannelData::Complex64(_) => {}
        ChannelData::StringSBC(array) => {
            while remaining > 0 {
                let len = &data[position..position + std::mem::size_of::<u32>()];
                let length: usize =
                    u32::from_le_bytes(len.try_into().expect("Could not read length")) as usize;
                position += std::mem::size_of::<u32>();
                let record = &data[position..position + length];
                let (_result, _size, _replacement) =
                    decoder
                        .windows_1252
                        .decode_to_string(&record, &mut array[nrecord], false);
                remaining = data_length - position;
                nrecord +=1 ;
            }
        }
        ChannelData::StringUTF8(array) => {
            while remaining > 0 {
                let len = &data[position..position + std::mem::size_of::<u32>()];
                let length: usize =
                    u32::from_le_bytes(len.try_into().expect("Could not read length")) as usize;
                position += std::mem::size_of::<u32>();
                let record = &data[position..position + length];
                array[nrecord] = str::from_utf8(&record)
                    .expect("Found invalid UTF-8")
                    .to_string();
                remaining = data_length - position;
                nrecord +=1 ;
            }
        }
        ChannelData::StringUTF16(array) => {
            if cn.endian {
                while remaining > 0 {
                    let len = &data[position..position + std::mem::size_of::<u32>()];
                    let length: usize =
                        u32::from_le_bytes(len.try_into().expect("Could not read length")) as usize;
                    position += std::mem::size_of::<u32>();
                    let record = &data[position..position + length];
                    let (_result, _size, _replacement) =
                        decoder
                            .utf_16_be
                            .decode_to_string(&record, &mut array[nrecord], false);
                    remaining = data_length - position;
                    nrecord +=1 ;
                }    
            } else {
                while remaining > 0 {
                    let len = &data[position..position + std::mem::size_of::<u32>()];
                    let length: usize =
                        u32::from_le_bytes(len.try_into().expect("Could not read length")) as usize;
                    position += std::mem::size_of::<u32>();
                    let record = &data[position..position + length];
                    let (_result, _size, _replacement) =
                        decoder
                            .utf_16_le
                            .decode_to_string(&record, &mut array[nrecord], false);
                    remaining = data_length - position;
                    nrecord +=1 ;
                }
            };
        }
        ChannelData::ByteArray(_) => {},
    }
    

}

/// Reads all DL Blocks and returns a vect of them
fn parser_ld4(rdr: &mut BufReader<&File>, mut position: i64, channel_group: &mut Cg4) -> i64 {
    let mut ld_blocks: Vec<Ld4Block> = Vec::new();
    let (block, pos) = parser_ld4_block(rdr, position, position);
    position = pos;
    ld_blocks.push(block.clone());
    let mut next_ld = block.ld_ld_next;
    while next_ld > 0 {
        rdr.seek_relative(next_ld - position).unwrap();
        position = next_ld;
        let mut id = [0u8; 4];
        rdr.read_exact(&mut id).expect("could not read LD block id");
        let (block, pos) = parser_ld4_block(rdr, position, position);
        position = pos;
        ld_blocks.push(block.clone());
        next_ld = block.ld_ld_next;
    }
    if ld_blocks.len() == 1 && ld_blocks[0].ld_data.len() == 1 && channel_group.cn.len() == 1 {
        // only one DV block, reading can be optimised
        // Reads DV or DZ block id
        rdr.seek_relative(ld_blocks[0].ld_data[0] - position)
            .unwrap();
        let mut id = [0u8; 4];
        rdr.read_exact(&mut id)
            .expect("could not read data block id from ld4 invalid");
        if id == "##DZ".as_bytes() {
            initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone());
            let (dt, block_header) = parse_dz(rdr);
            read_channels_from_bytes(
                &dt,
                &mut channel_group.cn,
                channel_group.record_length as usize,
                0,
            );
            position = ld_blocks[0].ld_data[0] + block_header.len as i64;
        } else {
            let block_header: Dt4Block = rdr.read_le().unwrap();
            for (_rec_pos, cn) in channel_group.cn.iter_mut() {
                read_one_channel_array(rdr, cn, channel_group.block.cg_cycle_count as usize);
            }
            position = ld_blocks[0].ld_data[0] + block_header.len as i64;
        }
        if channel_group.block.cg_inval_bytes > 0 {
            // Reads invalid DI or DZ block
            rdr.seek_relative(ld_blocks[0].ld_invalid_data[0] - position)
                .unwrap();
            let mut id = [0u8; 4];
            rdr.read_exact(&mut id)
                .expect("could not read data block id from ld4 invalid");
            if id == "##DZ".as_bytes() {
                let (dt, block_header) = parse_dz(rdr);
                channel_group.invalid_bytes = Some(dt);
                position = ld_blocks[0].ld_invalid_data[0] + block_header.len as i64;
            } else {
                let block_header: Dt4Block = rdr.read_le().unwrap();
                let mut buf = vec![0u8; (block_header.len - 24) as usize];
                rdr.read_exact(&mut buf).unwrap();
                channel_group.invalid_bytes = Some(buf);
                position = ld_blocks[0].ld_invalid_data[0] + block_header.len as i64;
            }
        }
    } else {
        // several DV, LD or channels per DG
        position = read_dv_di(rdr, position, channel_group, ld_blocks);
    }
    position
}

/// reads DV and DI block containing several channels
fn read_dv_di(
    rdr: &mut BufReader<&File>,
    mut position: i64,
    channel_group: &mut Cg4,
    ld_blocks: Vec<Ld4Block>,
) -> i64 {
    let cg_cycle_count = channel_group.block.cg_cycle_count as usize;
    let cg_inval_bytes = channel_group.block.cg_inval_bytes as usize;
    // initialises the arrays
    initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone());
    for ld in &ld_blocks {
        if !ld.ld_invalid_data.is_empty() {
            // initialises the invalid bytes vector
            channel_group.invalid_bytes = Some(vec![0u8; cg_inval_bytes * cg_cycle_count]);
        }
    }
    // Read all data blocks
    let mut data: Vec<u8> = Vec::new();
    let mut invalid_data: Vec<u8> = Vec::new();
    let mut previous_index: usize = 0;
    let mut previous_invalid_pos: usize = 0;
    for ld in ld_blocks {
        for data_pointer in ld.ld_data {
            // Reads DV or DZ block id
            rdr.seek_relative(data_pointer - position).unwrap();
            let mut id = [0u8; 4];
            rdr.read_exact(&mut id)
                .expect("could not read data block id from LD4");
            let block_length: usize;
            if id == "##DZ".as_bytes() {
                let (dt, block_header) = parse_dz(rdr);
                data.extend(dt);
                block_length = block_header.dz_org_data_length as usize;
                position = data_pointer + block_header.len as i64;
            } else {
                let block_header: Dt4Block = rdr.read_le().unwrap();
                let mut buf = vec![0u8; (block_header.len - 24) as usize];
                rdr.read_exact(&mut buf).unwrap();
                data.extend(buf);
                block_length = (block_header.len - 24) as usize;
                position = data_pointer + block_header.len as i64;
            }
            // Copies full sized records in block into channels arrays
            let record_length = channel_group.record_length as usize;
            let n_record_chunk = block_length / record_length;
            if previous_index + n_record_chunk < cg_cycle_count {
                read_channels_from_bytes(
                    &data[..record_length * n_record_chunk],
                    &mut channel_group.cn,
                    record_length,
                    previous_index,
                );
            } else {
                // Some implementation are pre allocating equal length blocks
                read_channels_from_bytes(
                    &data[..record_length * (cg_cycle_count - previous_index)],
                    &mut channel_group.cn,
                    record_length,
                    previous_index,
                );
            }
            // drop what has ben copied and keep remaining to be extended
            let remaining = block_length % record_length;
            if remaining > 0 {
                // copies tail part at beginnning of vect
                data.copy_within(record_length * n_record_chunk.., 0);
                // clears the last part
                data.truncate(remaining);
            } else {
                data.clear()
            }
            previous_index += n_record_chunk;
        }
        // Invalid data reading
        for data_pointer in ld.ld_invalid_data {
            // Reads DV or DZ block id
            rdr.seek_relative(data_pointer - position).unwrap();
            let mut id = [0u8; 4];
            rdr.read_exact(&mut id)
                .expect("could not read data block id from ld4 invalid");
            let block_length: usize;
            if id == "##DZ".as_bytes() {
                let (dt, block_header) = parse_dz(rdr);
                invalid_data.extend(dt);
                block_length = block_header.dz_org_data_length as usize;
                position = data_pointer + block_header.len as i64;
            } else {
                let block_header: Dt4Block = rdr.read_le().unwrap();
                let mut buf = vec![0u8; (block_header.len - 24) as usize];
                rdr.read_exact(&mut buf).unwrap();
                invalid_data.extend(buf);
                block_length = (block_header.len - 24) as usize;
                position = data_pointer + block_header.len as i64;
            }
            // Copies invalid data
            if let Some(invalid) = &mut channel_group.invalid_bytes {
                invalid[previous_invalid_pos..previous_invalid_pos + block_length]
                    .copy_from_slice(&invalid_data);
                previous_invalid_pos += block_length;
            }
            invalid_data.clear();
        }
    }
    position
}

/// Reads all DL Blocks and returns a vect of them
fn parser_dl4(rdr: &mut BufReader<&File>, mut position: i64) -> (Vec<Dl4Block>, i64) {
    let mut dl_blocks: Vec<Dl4Block> = Vec::new();
    let (block, pos) = parser_dl4_block(rdr, position, position);
    position = pos;
    dl_blocks.push(block.clone());
    let mut next_dl = block.dl_dl_next;
    while next_dl > 0 {
        rdr.seek_relative(next_dl - position).unwrap();
        position = next_dl;
        let mut id = [0u8; 4];
        rdr.read_exact(&mut id).expect("could not read DL block id");
        let (block, pos) = parser_dl4_block(rdr, position, position);
        position = pos;
        dl_blocks.push(block.clone());
        next_dl = block.dl_dl_next;
    }
    (dl_blocks, position)
}

/// Reads all sorted data blocks pointed by DL4 Blocks
fn parser_dl4_sorted(
    rdr: &mut BufReader<&File>,
    dl_blocks: Vec<Dl4Block>,
    mut position: i64,
    channel_group: &mut Cg4,
) -> (i64, Vec<u32>) {
    // initialises the arrays
    initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone());
    // Read all data blocks
    let mut data: Vec<u8> = Vec::new();
    let mut previous_index: usize = 0;
    let cg_cycle_count = channel_group.block.cg_cycle_count as usize;
    let mut vlsd_channels: Vec<u32> = Vec::new();
    for dl in dl_blocks {
        for data_pointer in dl.dl_data {
            // Reads DT or DZ block id
            rdr.seek_relative(data_pointer - position).unwrap();
            let mut id = [0u8; 4];
            rdr.read_exact(&mut id)
                .expect("could not read data block id");
            let block_length: usize;
            if id == "##DZ".as_bytes() {
                let (dt, block_header) = parse_dz(rdr);
                data.extend(dt);
                block_length = block_header.dz_org_data_length as usize;
                position = data_pointer + block_header.len as i64;
            } else {
                let block_header: Dt4Block = rdr.read_le().unwrap();
                let mut buf = vec![0u8; (block_header.len - 24) as usize];
                rdr.read_exact(&mut buf).unwrap();
                data.extend(buf);
                block_length = (block_header.len - 24) as usize;
                position = data_pointer + block_header.len as i64;
            }
            // Copies full sized records in block into channels arrays
            let record_length = channel_group.record_length as usize;
            let n_record_chunk = block_length / record_length;
            if previous_index + n_record_chunk < cg_cycle_count {
                vlsd_channels = read_channels_from_bytes(
                    &data[..record_length * n_record_chunk],
                    &mut channel_group.cn,
                    record_length,
                    previous_index,
                );
            } else {
                // Some implementation are pre allocating equal length blocks
                vlsd_channels = read_channels_from_bytes(
                    &data[..record_length * (cg_cycle_count - previous_index)],
                    &mut channel_group.cn,
                    record_length,
                    previous_index,
                );
            }
            // drop what has ben copied and keep remaining to be extended
            let remaining = block_length % record_length;
            if remaining > 0 {
                // copies tail part at beginnning of vect
                data.copy_within(record_length * n_record_chunk.., 0);
                // clears the last part
                data.truncate(remaining);
            } else {
                data.clear()
            }
            previous_index += n_record_chunk;
        }
    }
    (position, vlsd_channels)
}

/// Reads all unsorted data blocks pointed by DL4 Blocks
fn parser_dl4_unsorted(
    rdr: &mut BufReader<&File>,
    dg: &mut Dg4,
    dl_blocks: Vec<Dl4Block>,
    mut position: i64,
) -> i64 {
    // Read all data blocks
    let mut data: Vec<u8> = Vec::new();
    let mut decoder: Dec = Dec {
        windows_1252: WINDOWS_1252.new_decoder(),
        utf_16_be: UTF_16BE.new_decoder(),
        utf_16_le: UTF_16LE.new_decoder(),
    };
    // initialise record counter
    let mut record_counter: HashMap<u64, (usize, Vec<u8>)> = HashMap::new();
    for cg in dg.cg.values_mut() {
        record_counter.insert(cg.block.cg_record_id, (0, Vec::new()));
    }
    for dl in dl_blocks {
        for data_pointer in dl.dl_data {
            rdr.seek_relative(data_pointer - position).unwrap();
            let header = parse_block_header(rdr);
            if header.hdr_id == "##DZ".as_bytes() {
                let (dt, _block) = parse_dz(rdr);
                data.extend(dt);
            } else {
                let mut buf = vec![0u8; (header.hdr_len - 24) as usize];
                rdr.read_exact(&mut buf).unwrap();
                data.extend(buf);
            }
            // saves records as much as possible
            read_all_channels_unsorted_from_bytes(&mut data, dg, &mut record_counter, &mut decoder);
            position = data_pointer + header.hdr_len as i64;
        }
    }
    position
}

/// Returns chunk size and corresponding number of records from a channel group
fn generate_chunks(channel_group: &Cg4) -> Vec<(usize, usize)> {
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
fn read_all_channels_sorted(rdr: &mut BufReader<&File>, channel_group: &mut Cg4) -> Vec<u32> {
    let chunks = generate_chunks(channel_group);
    // initialises the arrays
    initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone());
    // read by chunks and store in channel array
    let mut previous_index: usize = 0;
    let mut vlsd_channels: Vec<u32> = Vec::new();
    for (n_record_chunk, chunk_size) in chunks {
        let mut data_chunk = vec![0u8; chunk_size];
        rdr.read_exact(&mut data_chunk)
            .expect("Could not read data chunk");
        vlsd_channels = read_channels_from_bytes(
                        &data_chunk,
                        &mut channel_group.cn,
                        channel_group.record_length as usize,
                        previous_index,
                    );
        previous_index += n_record_chunk;
    }
    vlsd_channels
}

// reads file if data block contains only one channel in a single DV
fn read_one_channel_array(rdr: &mut BufReader<&File>, cn: &mut Cn4, cycle_count: usize) {
    if cn.block.cn_type == 0
        || cn.block.cn_type == 2
        || cn.block.cn_type == 4
        || cn.block.cn_type == 5
    {
        // cn_type == 5 : Maximum length data channel, removing no valid bytes done by another size channel pointed by cn_data
        // cn_type == 0 : fixed length data channel
        // cn_type == 2 : master channel
        // cn_type == 4 : synchronisation channel
        let n_bytes = cn.n_bytes as usize;
        match &mut cn.data {
            ChannelData::Int8(data) => {
                let mut buf = vec![0; cycle_count];
                rdr.read_i8_into(&mut buf).expect("Could not read i8 array");
                *data = Array::from_vec(buf);
            }
            ChannelData::UInt8(data) => {
                let mut buf = vec![0; cycle_count];
                rdr.read_exact(&mut buf).expect("Could not read u8 array");
                *data = Array::from_vec(buf);
            }
            ChannelData::Int16(data) => {
                let mut buf = vec![0; cycle_count];
                if cn.endian {
                    rdr.read_i16_into::<BigEndian>(&mut buf)
                        .expect("Could not read be i16 array");
                } else {
                    rdr.read_i16_into::<LittleEndian>(&mut buf)
                        .expect("Could not read le i16 array");
                }
                *data = Array::from_vec(buf);
            }
            ChannelData::UInt16(data) => {
                let mut buf = vec![0; cycle_count];
                if cn.endian {
                    rdr.read_u16_into::<BigEndian>(&mut buf)
                        .expect("Could not read be u16 array");
                } else {
                    rdr.read_u16_into::<LittleEndian>(&mut buf)
                        .expect("Could not read le 16 array");
                }
                *data = Array::from_vec(buf);
            }
            ChannelData::Float16(data) => {
                let mut buf = vec![0u8; cycle_count * std::mem::size_of::<f16>()];
                rdr.read_exact(&mut buf).expect("Could not read f16 array");
                *data = Array1::<f32>::zeros((cycle_count,)); // initialisation
                if cn.endian {
                    for (i, value) in buf.chunks(std::mem::size_of::<f16>()).enumerate() {
                        data[i] =
                            f16::from_be_bytes(value.try_into().expect("Could not read be f16"))
                                .to_f32();
                    }
                } else {
                    for (i, value) in buf.chunks(std::mem::size_of::<f16>()).enumerate() {
                        data[i] =
                            f16::from_le_bytes(value.try_into().expect("Could not read le f16"))
                                .to_f32();
                    }
                }
            }
            ChannelData::Int24(data) => {
                let mut buf = vec![0u8; cycle_count * n_bytes];
                rdr.read_exact(&mut buf).expect("Could not read i24 array");
                *data = Array1::<i32>::zeros((cycle_count,)); // initialisation
                if cn.endian {
                    for (i, mut value) in buf.chunks(n_bytes).enumerate() {
                        data[i] = value
                            .read_i24::<BigEndian>()
                            .expect("Could not read be i24");
                    }
                } else {
                    for (i, mut value) in buf.chunks(n_bytes).enumerate() {
                        data[i] = value
                            .read_i24::<LittleEndian>()
                            .expect("Could not read le i24");
                    }
                }
            }
            ChannelData::UInt24(data) => {
                let mut buf = vec![0u8; cycle_count * n_bytes];
                rdr.read_exact(&mut buf).expect("Could not read u24 array");
                *data = Array1::<u32>::zeros((cycle_count,)); // initialisation
                if cn.endian {
                    for (i, mut value) in buf.chunks(n_bytes).enumerate() {
                        data[i] = value
                            .read_u24::<BigEndian>()
                            .expect("Could not read be u24");
                    }
                } else {
                    for (i, mut value) in buf.chunks(n_bytes).enumerate() {
                        data[i] = value
                            .read_u24::<LittleEndian>()
                            .expect("Could not read le u24");
                    }
                }
            }
            ChannelData::Int32(data) => {
                let mut buf = vec![0; cycle_count];
                if cn.endian {
                    rdr.read_i32_into::<BigEndian>(&mut buf)
                        .expect("Could not read be i32 array");
                } else {
                    rdr.read_i32_into::<LittleEndian>(&mut buf)
                        .expect("Could not read le i32 array");
                }
                *data = Array::from_vec(buf);
            }
            ChannelData::UInt32(data) => {
                let mut buf = vec![0; cycle_count];
                if cn.endian {
                    rdr.read_u32_into::<BigEndian>(&mut buf)
                        .expect("Could not read be u32 array");
                } else {
                    rdr.read_u32_into::<LittleEndian>(&mut buf)
                        .expect("Could not read le u32 array");
                }
                *data = Array::from_vec(buf);
            }
            ChannelData::Float32(data) => {
                let mut buf = vec![0f32; cycle_count];
                if cn.endian {
                    rdr.read_f32_into::<BigEndian>(&mut buf)
                        .expect("Could not read be f32 array");
                } else {
                    rdr.read_f32_into::<LittleEndian>(&mut buf)
                        .expect("Could not read le f32 array");
                }
                *data = Array::from_vec(buf);
            }
            ChannelData::Int48(data) => {
                let mut buf = vec![0u8; cycle_count * n_bytes];
                rdr.read_exact(&mut buf).expect("Could not read i48 array");
                *data = Array1::<i64>::zeros((cycle_count,)); // initialisation
                if cn.endian {
                    for (i, mut value) in buf.chunks(n_bytes).enumerate() {
                        data[i] = value
                            .read_i48::<BigEndian>()
                            .expect("Could not read be i48");
                    }
                } else {
                    for (i, mut value) in buf.chunks(n_bytes).enumerate() {
                        data[i] = value
                            .read_i48::<LittleEndian>()
                            .expect("Could not read le i48");
                    }
                }
            }
            ChannelData::UInt48(data) => {
                let mut buf = vec![0u8; cycle_count * n_bytes];
                rdr.read_exact(&mut buf).expect("Could not read u48 array");
                *data = Array1::<u64>::zeros((cycle_count,)); // initialisation
                if cn.endian {
                    // big endian
                    if n_bytes == 6 {
                        for (i, mut value) in buf.chunks(n_bytes).enumerate() {
                            data[i] = value
                                .read_u48::<BigEndian>()
                                .expect("Could not read be u48");
                        }
                    } else {
                        // n_bytes = 5
                        let mut temp = [0u8; 6];
                        for (i, value) in buf.chunks(n_bytes).enumerate() {
                            temp[0..5].copy_from_slice(&value[0..n_bytes]);
                            data[i] = Box::new(&temp[..])
                                .read_u48::<BigEndian>()
                                .expect("Could not read be u48 from 5 bytes");
                        }
                    }
                } else if n_bytes == 6 {
                    // little endian
                    for (i, mut value) in buf.chunks(n_bytes).enumerate() {
                        data[i] = value
                            .read_u48::<LittleEndian>()
                            .expect("Could not read le u48");
                    }
                } else {
                    // n_bytes = 5
                    let mut temp = [0u8; 6];
                    for (i, value) in buf.chunks(n_bytes).enumerate() {
                        temp[0..5].copy_from_slice(&value[0..n_bytes]);
                        data[i] = Box::new(&temp[..])
                            .read_u48::<LittleEndian>()
                            .expect("Could not read le u48 from 5 bytes");
                    }
                }
            }
            ChannelData::Int64(data) => {
                let mut buf = vec![0; cycle_count];
                if cn.endian {
                    rdr.read_i64_into::<BigEndian>(&mut buf)
                        .expect("Could not read be i64 array");
                } else {
                    rdr.read_i64_into::<LittleEndian>(&mut buf)
                        .expect("Could not read le i64 array");
                }
                *data = Array::from_vec(buf);
            }
            ChannelData::UInt64(data) => {
                if n_bytes == 8 {
                    let mut buf = vec![0; cycle_count];
                    if cn.endian {
                        rdr.read_u64_into::<BigEndian>(&mut buf)
                            .expect("Could not read be u64 array");
                    } else {
                        rdr.read_u64_into::<LittleEndian>(&mut buf)
                            .expect("Could not read le u64 array");
                    }
                    *data = Array::from_vec(buf);
                } else {
                    // n_bytes = 7
                    let mut buf = vec![0u8; cycle_count * n_bytes];
                    rdr.read_exact(&mut buf).expect("Could not read u64 array");
                    *data = Array1::<u64>::zeros((cycle_count,));
                    let mut temp = [0u8; std::mem::size_of::<u64>()];
                    if cn.endian {
                        for (i, value) in buf.chunks(n_bytes).enumerate() {
                            temp[0..7].copy_from_slice(&value[0..7]);
                            data[i] = u64::from_be_bytes(temp);
                        }
                    } else {
                        for (i, value) in buf.chunks(n_bytes).enumerate() {
                            temp[0..7].copy_from_slice(&value[0..7]);
                            data[i] = u64::from_le_bytes(temp);
                        }
                    }
                }
            }
            ChannelData::Float64(data) => {
                let mut buf = vec![0f64; cycle_count];
                if cn.endian {
                    rdr.read_f64_into::<BigEndian>(&mut buf)
                        .expect("Could not read be f64 array");
                } else {
                    rdr.read_f64_into::<LittleEndian>(&mut buf)
                        .expect("Could not read le f64 array");
                }
                *data = Array::from_vec(buf);
            }
            ChannelData::Complex16(data) => {
                let mut buf = vec![0u8; cycle_count * n_bytes];
                rdr.read_exact(&mut buf)
                    .expect("Could not read complex16 array");
                let mut re: f32;
                let mut im: f32;
                let mut re_val: &[u8];
                let mut im_val: &[u8];
                *data = Array1::<Complex<f32>>::zeros((cycle_count,)); // initialisation
                if cn.endian {
                    for (i, value) in buf.chunks(std::mem::size_of::<f16>() * 2).enumerate() {
                        re_val = &value[0..std::mem::size_of::<f16>()];
                        im_val = &value[std::mem::size_of::<f16>()..2 * std::mem::size_of::<f16>()];
                        re = f16::from_be_bytes(
                            re_val
                                .try_into()
                                .expect("Could not read be real f16 complex"),
                        )
                        .to_f32();
                        im = f16::from_be_bytes(
                            im_val
                                .try_into()
                                .expect("Could not read be img f16 complex"),
                        )
                        .to_f32();
                        data[i] = Complex::new(re, im);
                    }
                } else {
                    for (i, value) in buf.chunks(std::mem::size_of::<f16>() * 2).enumerate() {
                        re_val = &value[0..std::mem::size_of::<f16>()];
                        im_val = &value[std::mem::size_of::<f16>()..2 * std::mem::size_of::<f16>()];
                        re = f16::from_le_bytes(
                            re_val
                                .try_into()
                                .expect("Could not read le real f16 complex"),
                        )
                        .to_f32();
                        im = f16::from_le_bytes(
                            im_val
                                .try_into()
                                .expect("Could not read le img f16 complex"),
                        )
                        .to_f32();
                        data[i] = Complex::new(re, im);
                    }
                }
            }
            ChannelData::Complex32(data) => {
                let mut buf = vec![0u8; cycle_count * n_bytes];
                rdr.read_exact(&mut buf)
                    .expect("Could not read complex32 array");
                let mut re: f32;
                let mut im: f32;
                let mut re_val: &[u8];
                let mut im_val: &[u8];
                *data = Array1::<Complex<f32>>::zeros((cycle_count,)); // initialisation
                if cn.endian {
                    for (i, value) in buf.chunks(std::mem::size_of::<f32>() * 2).enumerate() {
                        re_val = &value[0..std::mem::size_of::<f32>()];
                        im_val = &value[std::mem::size_of::<f32>()..2 * std::mem::size_of::<f32>()];
                        re = f32::from_be_bytes(
                            re_val
                                .try_into()
                                .expect("Could not read be real f32 complex"),
                        );
                        im = f32::from_be_bytes(
                            im_val
                                .try_into()
                                .expect("Could not read be img f32 complex"),
                        );
                        data[i] = Complex::new(re, im);
                    }
                } else {
                    for (i, value) in buf.chunks(std::mem::size_of::<f32>() * 2).enumerate() {
                        re_val = &value[0..std::mem::size_of::<f32>()];
                        im_val = &value[std::mem::size_of::<f32>()..2 * std::mem::size_of::<f32>()];
                        re = f32::from_le_bytes(
                            re_val
                                .try_into()
                                .expect("Could not read le real f32 complex"),
                        );
                        im = f32::from_le_bytes(
                            im_val
                                .try_into()
                                .expect("Could not read le img f32 complex"),
                        );
                        data[i] = Complex::new(re, im);
                    }
                }
            }
            ChannelData::Complex64(data) => {
                let mut buf = vec![0u8; cycle_count * n_bytes];
                rdr.read_exact(&mut buf)
                    .expect("Could not read complex64 array");
                let mut re: f64;
                let mut im: f64;
                let mut re_val: &[u8];
                let mut im_val: &[u8];
                *data = Array1::<Complex<f64>>::zeros((cycle_count,)); // initialisation
                if cn.endian {
                    for (i, value) in buf.chunks(std::mem::size_of::<f64>() * 2).enumerate() {
                        re_val = &value[0..std::mem::size_of::<f64>()];
                        im_val = &value[std::mem::size_of::<f64>()..2 * std::mem::size_of::<f64>()];
                        re = f64::from_be_bytes(re_val.try_into().expect("Could not array"));
                        im = f64::from_be_bytes(im_val.try_into().expect("Could not array"));
                        data[i] = Complex::new(re, im);
                    }
                } else {
                    for (i, value) in buf.chunks(std::mem::size_of::<f64>() * 2).enumerate() {
                        re_val = &value[0..std::mem::size_of::<f64>()];
                        im_val = &value[std::mem::size_of::<f64>()..2 * std::mem::size_of::<f64>()];
                        re = f64::from_le_bytes(
                            re_val
                                .try_into()
                                .expect("Could not read le real f64 complex"),
                        );
                        im = f64::from_le_bytes(
                            im_val
                                .try_into()
                                .expect("Could not read le img f64 complex"),
                        );
                        data[i] = Complex::new(re, im);
                    }
                }
            }
            ChannelData::StringSBC(data) => {
                let mut buf = vec![0u8; cycle_count * n_bytes];
                rdr.read_exact(&mut buf).expect("Could not read SBC String");
                let mut decoder = WINDOWS_1252.new_decoder();
                *data = vec![String::new(); cycle_count]; // initialisation
                for (i, value) in buf.chunks(n_bytes).enumerate() {
                    let (_result, _size, _replacement) =
                        decoder.decode_to_string(&value, &mut data[i], false);
                    data[i] = data[i].trim_end_matches('\0').to_string();
                }
            }
            ChannelData::StringUTF8(data) => {
                let mut buf = vec![0u8; cycle_count * n_bytes];
                rdr.read_exact(&mut buf)
                    .expect("Could not read String UTF8");
                *data = vec![String::new(); cycle_count]; // initialisation
                for (i, value) in buf.chunks(n_bytes).enumerate() {
                    data[i] = str::from_utf8(&value)
                        .expect("Found invalid UTF-8")
                        .trim_end_matches('\0')
                        .to_string();
                }
            }
            ChannelData::StringUTF16(data) => {
                let mut buf = vec![0u8; cycle_count * n_bytes];
                rdr.read_exact(&mut buf)
                    .expect("Could not read String UTF16");
                *data = vec![String::new(); cycle_count]; // initialisation
                if cn.endian {
                    let mut decoder = UTF_16BE.new_decoder();
                    for (i, value) in buf.chunks(n_bytes).enumerate() {
                        let (_result, _size, _replacement) =
                            decoder.decode_to_string(&value, &mut data[i], false);
                        data[i] = data[i].trim_end_matches('\0').to_string();
                    }
                } else {
                    let mut decoder = UTF_16LE.new_decoder();
                    for (i, value) in buf.chunks(n_bytes).enumerate() {
                        let (_result, _size, _replacement) =
                            decoder.decode_to_string(&value, &mut data[i], false);
                        data[i] = data[i].trim_end_matches('\0').to_string();
                    }
                }
            }
            ChannelData::ByteArray(data) => {
                *data = vec![0u8; cycle_count * n_bytes];
                rdr.read_exact(data).expect("Could not read byte array");
            }
        }
    }
    // Other channel types : virtual channels cn_type 3 & 6 are handled at initialisation
    // cn_type == 1 VLSD not possible for sorted data
}

// copies data from data_chunk into each channel array
fn read_channels_from_bytes(
    data_chunk: &[u8],
    channels: &mut CnType,
    record_length: usize,
    previous_index: usize,
) -> Vec<u32> {
    let vlsd_channels: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
    // iterates for each channel in parallel with rayon crate
    channels.par_iter_mut().for_each(|(rec_pos, cn)| {
        if cn.block.cn_type == 0
            || cn.block.cn_type == 2
            || cn.block.cn_type == 4
            || cn.block.cn_type == 5
        {
            // cn_type == 5 : Maximum length data channel, removing no valid bytes done by another size channel pointed by cn_data
            // cn_type == 0 : fixed length data channel
            // cn_type == 2 : master channel
            // cn_type == 4 : synchronisation channel
            let mut value: &[u8]; // value of channel at record
            let pos_byte_beg = cn.pos_byte_beg as usize;
            let n_bytes = cn.n_bytes as usize;
            match &mut cn.data {
                ChannelData::Int8(data) => {
                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i8>()];
                        data[i + previous_index] =
                            i8::from_le_bytes(value.try_into().expect("Could not read i8"));
                    }
                }
                ChannelData::UInt8(data) => {
                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u8>()];
                        data[i + previous_index] =
                            u8::from_le_bytes(value.try_into().expect("Could not read u8"));
                    }
                }
                ChannelData::Int16(data) => {
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i16>()];
                            data[i + previous_index] = i16::from_be_bytes(
                                value.try_into().expect("Could not read be i16"),
                            );
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i16>()];
                            data[i + previous_index] = i16::from_le_bytes(
                                value.try_into().expect("Could not read le i16"),
                            );
                        }
                    }
                }
                ChannelData::UInt16(data) => {
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u16>()];
                            data[i + previous_index] = u16::from_be_bytes(
                                value.try_into().expect("Could not read be u16"),
                            );
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u16>()];
                            data[i + previous_index] = u16::from_le_bytes(
                                value.try_into().expect("Could not read le u16"),
                            );
                        }
                    }
                }
                ChannelData::Float16(data) => {
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f16>()];
                            data[i + previous_index] = f16::from_be_bytes(
                                value.try_into().expect("Could not read be f16"),
                            )
                            .to_f32();
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f16>()];
                            data[i + previous_index] = f16::from_le_bytes(
                                value.try_into().expect("Could not read le f16"),
                            )
                            .to_f32();
                        }
                    }
                }
                ChannelData::Int24(data) => {
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            data[i + previous_index] = value
                                .read_i24::<BigEndian>()
                                .expect("Could not read be i24");
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            data[i + previous_index] = value
                                .read_i24::<LittleEndian>()
                                .expect("Could not read le i24");
                        }
                    }
                }
                ChannelData::UInt24(data) => {
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            data[i + previous_index] = value
                                .read_u24::<BigEndian>()
                                .expect("Could not read be u24");
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            data[i + previous_index] = value
                                .read_u24::<LittleEndian>()
                                .expect("Could not read le u24");
                        }
                    }
                }
                ChannelData::Int32(data) => {
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i32>()];
                            data[i + previous_index] = i32::from_be_bytes(
                                value.try_into().expect("Could not read be i32"),
                            );
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i32>()];
                            data[i + previous_index] = i32::from_le_bytes(
                                value.try_into().expect("Could not read le i32"),
                            );
                        }
                    }
                }
                ChannelData::UInt32(data) => {
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u32>()];
                            data[i + previous_index] = u32::from_be_bytes(
                                value.try_into().expect("Could not read be u32"),
                            );
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u32>()];
                            data[i + previous_index] = u32::from_le_bytes(
                                value.try_into().expect("Could not read le u32"),
                            );
                        }
                    }
                }
                ChannelData::Float32(data) => {
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f32>()];
                            data[i + previous_index] = f32::from_be_bytes(
                                value.try_into().expect("Could not read be u32"),
                            );
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f32>()];
                            data[i + previous_index] = f32::from_le_bytes(
                                value.try_into().expect("Could not read le u32"),
                            );
                        }
                    }
                }
                ChannelData::Int48(data) => {
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            data[i + previous_index] = value
                                .read_i48::<BigEndian>()
                                .expect("Could not read be i48");
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            data[i + previous_index] = value
                                .read_i48::<LittleEndian>()
                                .expect("Could not read le i48");
                        }
                    }
                }
                ChannelData::UInt48(data) => {
                    if cn.endian {
                        if n_bytes == 6 {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                data[i + previous_index] = value
                                    .read_u48::<BigEndian>()
                                    .expect("Could not read be u48");
                            }
                        } else {
                            // n_bytes = 5
                            let mut buf = [0u8; 6];
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                buf[0..5]
                                    .copy_from_slice(&record[pos_byte_beg..pos_byte_beg + n_bytes]);
                                data[i + previous_index] = Box::new(&buf[..])
                                    .read_u48::<BigEndian>()
                                    .expect("Could not read be u48 from 5 bytes");
                            }
                        }
                    } else if n_bytes == 6 {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            data[i + previous_index] = value
                                .read_u48::<LittleEndian>()
                                .expect("Could not read le u48");
                        }
                    } else {
                        // n_bytes = 5
                        let mut buf = [0u8; 6];
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            buf[0..5].copy_from_slice(&record[pos_byte_beg..pos_byte_beg + 5]);
                            data[i + previous_index] = Box::new(&buf[..])
                                .read_u48::<LittleEndian>()
                                .expect("Could not read le u48 from 5 bytes");
                        }
                    }
                }
                ChannelData::Int64(data) => {
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            data[i + previous_index] = i64::from_be_bytes(
                                value.try_into().expect("Could not read be i64"),
                            );
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            data[i + previous_index] = i64::from_le_bytes(
                                value.try_into().expect("Could not read le i64"),
                            );
                        }
                    }
                }
                ChannelData::UInt64(data) => {
                    if cn.endian {
                        if n_bytes == 8 {
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                                data[i + previous_index] = u64::from_le_bytes(
                                    value.try_into().expect("Could not read be u64"),
                                );
                            }
                        } else {
                            // n_bytes = 7
                            let mut buf = [0u8; std::mem::size_of::<u64>()];
                            for (i, record) in data_chunk.chunks(record_length).enumerate() {
                                buf[0..7].copy_from_slice(&record[pos_byte_beg..pos_byte_beg + 7]);
                                data[i + previous_index] = u64::from_le_bytes(buf);
                            }
                        }
                    } else if n_bytes == 8 {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            data[i + previous_index] = u64::from_le_bytes(
                                value.try_into().expect("Could not read le u64"),
                            );
                        }
                    } else {
                        // n_bytes = 7
                        let mut buf = [0u8; std::mem::size_of::<u64>()];
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            buf[0..7].copy_from_slice(&record[pos_byte_beg..pos_byte_beg + 7]);
                            data[i + previous_index] = u64::from_le_bytes(buf);
                        }
                    }
                }
                ChannelData::Float64(data) => {
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f64>()];
                            data[i + previous_index] = f64::from_be_bytes(
                                value.try_into().expect("Could not read be f64"),
                            );
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f64>()];
                            data[i + previous_index] = f64::from_le_bytes(
                                value.try_into().expect("Could not read le f64"),
                            );
                        }
                    }
                }
                ChannelData::Complex16(data) => {
                    let mut re: f32;
                    let mut im: f32;
                    let mut re_val: &[u8];
                    let mut im_val: &[u8];
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            re_val =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f16>()];
                            im_val = &record[pos_byte_beg + std::mem::size_of::<f16>()
                                ..pos_byte_beg + 2 * std::mem::size_of::<f16>()];
                            re = f16::from_be_bytes(
                                re_val
                                    .try_into()
                                    .expect("Could not read be real f16 complex"),
                            )
                            .to_f32();
                            im = f16::from_be_bytes(
                                im_val
                                    .try_into()
                                    .expect("Could not read be img f16 complex"),
                            )
                            .to_f32();
                            data[i + previous_index] = Complex::new(re, im);
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            re_val =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f16>()];
                            im_val = &record[pos_byte_beg + std::mem::size_of::<f16>()
                                ..pos_byte_beg + 2 * std::mem::size_of::<f16>()];
                            re = f16::from_le_bytes(
                                re_val
                                    .try_into()
                                    .expect("Could not read le real f16 complex"),
                            )
                            .to_f32();
                            im = f16::from_le_bytes(
                                im_val
                                    .try_into()
                                    .expect("Could not read le img f16 complex"),
                            )
                            .to_f32();
                            data[i + previous_index] = Complex::new(re, im);
                        }
                    }
                }
                ChannelData::Complex32(data) => {
                    let mut re: f32;
                    let mut im: f32;
                    let mut re_val: &[u8];
                    let mut im_val: &[u8];
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            re_val =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f32>()];
                            im_val = &record[pos_byte_beg + std::mem::size_of::<f32>()
                                ..pos_byte_beg + 2 * std::mem::size_of::<f32>()];
                            re = f32::from_be_bytes(
                                re_val
                                    .try_into()
                                    .expect("Could not read be real f32 complex"),
                            );
                            im = f32::from_be_bytes(
                                im_val
                                    .try_into()
                                    .expect("Could not read be img f32 complex"),
                            );
                            data[i + previous_index] = Complex::new(re, im);
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            re_val =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f32>()];
                            im_val = &record[pos_byte_beg + std::mem::size_of::<f32>()
                                ..pos_byte_beg + 2 * std::mem::size_of::<f32>()];
                            re = f32::from_le_bytes(
                                re_val
                                    .try_into()
                                    .expect("Could not read le real f32 complex"),
                            );
                            im = f32::from_le_bytes(
                                im_val
                                    .try_into()
                                    .expect("Could not read le img f32 complex"),
                            );
                            data[i + previous_index] = Complex::new(re, im);
                        }
                    }
                }
                ChannelData::Complex64(data) => {
                    let mut re: f64;
                    let mut im: f64;
                    let mut re_val: &[u8];
                    let mut im_val: &[u8];
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            re_val =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f64>()];
                            im_val = &record[pos_byte_beg + std::mem::size_of::<f64>()
                                ..pos_byte_beg + 2 * std::mem::size_of::<f64>()];
                            re = f64::from_be_bytes(re_val.try_into().expect("Could not array"));
                            im = f64::from_be_bytes(im_val.try_into().expect("Could not array"));
                            data[i + previous_index] = Complex::new(re, im);
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            re_val =
                                &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f64>()];
                            im_val = &record[pos_byte_beg + std::mem::size_of::<f64>()
                                ..pos_byte_beg + 2 * std::mem::size_of::<f64>()];
                            re = f64::from_le_bytes(
                                re_val
                                    .try_into()
                                    .expect("Could not read le real f64 complex"),
                            );
                            im = f64::from_le_bytes(
                                im_val
                                    .try_into()
                                    .expect("Could not read le img f64 complex"),
                            );
                            data[i + previous_index] = Complex::new(re, im);
                        }
                    }
                }
                ChannelData::StringSBC(data) => {
                    let n_bytes = cn.n_bytes as usize;
                    let mut decoder = WINDOWS_1252.new_decoder();
                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                        let (_result, _size, _replacement) =
                            decoder.decode_to_string(&value, &mut data[i + previous_index], false);
                        data[i + previous_index] =
                            data[i + previous_index].trim_end_matches('\0').to_string();
                    }
                }
                ChannelData::StringUTF8(data) => {
                    let n_bytes = cn.n_bytes as usize;
                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                        data[i + previous_index] = str::from_utf8(&value)
                            .expect("Found invalid UTF-8")
                            .trim_end_matches('\0')
                            .to_string();
                    }
                }
                ChannelData::StringUTF16(data) => {
                    let n_bytes = cn.n_bytes as usize;
                    if cn.endian {
                        let mut decoder = UTF_16BE.new_decoder();
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            let (_result, _size, _replacement) = decoder.decode_to_string(
                                &value,
                                &mut data[i + previous_index],
                                false,
                            );
                            data[i + previous_index] =
                                data[i + previous_index].trim_end_matches('\0').to_string();
                        }
                    } else {
                        let mut decoder = UTF_16LE.new_decoder();
                        for (i, record) in data_chunk.chunks(record_length).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                            let (_result, _size, _replacement) = decoder.decode_to_string(
                                &value,
                                &mut data[i + previous_index],
                                false,
                            );
                            data[i + previous_index] =
                                data[i + previous_index].trim_end_matches('\0').to_string();
                        }
                    }
                }
                ChannelData::ByteArray(data) => {
                    let n_bytes = cn.n_bytes as usize;
                    for (i, record) in data_chunk.chunks(record_length).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + n_bytes];
                        let index = (i + previous_index) * n_bytes;
                        data[index..index + n_bytes].copy_from_slice(value);
                    }
                }
            }
        } else if cn.block.cn_type == 1 {
            // SD Block attached as data block is sorted
            if cn.block.cn_data != 0 {
                let c_vlsd_channel = Arc::clone(&vlsd_channels);
                let mut vlsd_channel = c_vlsd_channel.lock().expect("Could not get lock from vlsd channel arc vec");
                vlsd_channel.push(*rec_pos);
            }
        }
        // Other channel types : virtual channels cn_type 3 & 6 are handled at initialisation
    });
    let lock = vlsd_channels.lock().expect("Could not get lock from vlsd channel arc vec");
    lock.clone()
}

// copies complete sorted data block (not chunk) into each channel array
fn read_all_channels_sorted_from_bytes(data: &[u8], channel_group: &mut Cg4) -> Vec<u32>{
    // initialises the arrays
    initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone());
    let mut vlsd_channels: Vec<u32>= Vec::new();
    for nrecord in 0..channel_group.block.cg_cycle_count {
        vlsd_channels = read_channels_from_bytes(
            &data[(nrecord * channel_group.record_length as u64) as usize
                ..((nrecord + 1) * channel_group.record_length as u64) as usize],
            &mut channel_group.cn,
            channel_group.record_length as usize,
            0,
        );
    }
    vlsd_channels
}

/// Reads unsorted data block chunk by chunk
fn read_all_channels_unsorted(rdr: &mut BufReader<&File>, dg: &mut Dg4, block_length: i64) {
    let data_block_length = block_length as usize;
    let mut position: usize = 24;
    let mut record_counter: HashMap<u64, (usize, Vec<u8>)> = HashMap::new();
    let mut decoder: Dec = Dec {
        windows_1252: WINDOWS_1252.new_decoder(),
        utf_16_be: UTF_16BE.new_decoder(),
        utf_16_le: UTF_16LE.new_decoder(),
    };
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
            &mut decoder,
        );
    }
}

/// stores a vlsd record into channel vect (ChannelData)
#[inline]
fn save_vlsd(
    data: &mut ChannelData,
    record: &[u8],
    nrecord: &usize,
    decoder: &mut Dec,
    endian: bool,
) {
    match data {
        ChannelData::Int8(_) => {}
        ChannelData::UInt8(_) => {}
        ChannelData::Int16(_) => {}
        ChannelData::UInt16(_) => {}
        ChannelData::Float16(_) => {}
        ChannelData::Int24(_) => {}
        ChannelData::UInt24(_) => {}
        ChannelData::Int32(_) => {}
        ChannelData::UInt32(_) => {}
        ChannelData::Float32(_) => {}
        ChannelData::Int48(_) => {}
        ChannelData::UInt48(_) => {}
        ChannelData::Int64(_) => {}
        ChannelData::UInt64(_) => {}
        ChannelData::Float64(_) => {}
        ChannelData::Complex16(_) => {}
        ChannelData::Complex32(_) => {}
        ChannelData::Complex64(_) => {}
        ChannelData::StringSBC(array) => {
            let (_result, _size, _replacement) =
                decoder
                    .windows_1252
                    .decode_to_string(&record, &mut array[*nrecord], false);
        }
        ChannelData::StringUTF8(array) => {
            array[*nrecord] = str::from_utf8(&record)
                .expect("Found invalid UTF-8")
                .to_string();
        }
        ChannelData::StringUTF16(array) => {
            if endian {
                let (_result, _size, _replacement) =
                    decoder
                        .utf_16_be
                        .decode_to_string(&record, &mut array[*nrecord], false);
            } else {
                let (_result, _size, _replacement) =
                    decoder
                        .utf_16_le
                        .decode_to_string(&record, &mut array[*nrecord], false);
            };
        }
        ChannelData::ByteArray(_) => {},
    }
}

/// read record by record from unsorted data block into sorted data block, then copy data into channel arrays
fn read_all_channels_unsorted_from_bytes(
    data: &mut Vec<u8>,
    dg: &mut Dg4,
    record_counter: &mut HashMap<u64, (usize, Vec<u8>)>,
    decoder: &mut Dec,
) {
    let mut position: usize = 0;
    let data_length = data.len();
    let dg_rec_id_size = dg.block.dg_rec_id_size as usize;
    // unsort data into sorted data blocks, except for VLSD CG.
    let mut remaining: usize = data_length - position;
    while remaining > 0 {
        // reads record id
        let rec_id: u64;
        if dg_rec_id_size == 1 && remaining >= 1 {
            rec_id = data[position]
                .try_into()
                .expect("Could not convert record id u8");
        } else if dg_rec_id_size == 2 && remaining >= 2 {
            let rec = &data[position..position + std::mem::size_of::<u16>()];
            rec_id =
                u16::from_le_bytes(rec.try_into().expect("Could not convert record id u16")) as u64;
        } else if dg_rec_id_size == 4 && remaining >= 4 {
            let rec = &data[position..position + std::mem::size_of::<u32>()];
            rec_id =
                u32::from_le_bytes(rec.try_into().expect("Could not convert record id u32")) as u64;
        } else if dg_rec_id_size == 8 && remaining >= 8 {
            let rec = &data[position..position + std::mem::size_of::<u64>()];
            rec_id =
                u64::from_le_bytes(rec.try_into().expect("Could not convert record id u64")) as u64;
        } else {
            break; // not enough data remaining
        }
        // reads record based on record id
        if let Some(cg) = dg.cg.get_mut(&rec_id) {
            let record_length = cg.record_length as usize;
            if remaining >= record_length {
                if (cg.block.cg_flags & 0b1) != 0 {
                    // VLSD channel
                    position += dg_rec_id_size;
                    let len = &data[position..position + std::mem::size_of::<u32>()];
                    let length: usize =
                        u32::from_le_bytes(len.try_into().expect("Could not read length")) as usize;
                    position += std::mem::size_of::<u32>();
                    let record = &data[position..position + length];
                    if let Some((target_rec_id, target_rec_pos)) = cg.vlsd_cg {
                        if let Some(target_cg) = dg.cg.get_mut(&target_rec_id) {
                            if let Some(target_cn) = target_cg.cn.get_mut(&target_rec_pos) {
                                if let Some((nrecord, _)) = record_counter.get_mut(&rec_id) {
                                    save_vlsd(
                                        &mut target_cn.data,
                                        record,
                                        nrecord,
                                        decoder,
                                        target_cn.endian,
                                    );
                                    *nrecord += 1;
                                }
                            }
                        }
                    }
                    position += length;
                } else {
                    // Not VLSD channel
                    let record = &data[position..position + cg.record_length as usize];
                    if let Some((_nrecord, data)) = record_counter.get_mut(&rec_id) {
                        data.extend(record);
                    }
                    position += record_length;
                }
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
            );
            record_data.clear(); // clears data for new block, keeping capacity
        }
    }
}

/// decoder for String SBC and UTF16 Le & Be
struct Dec {
    windows_1252: Decoder,
    utf_16_be: Decoder,
    utf_16_le: Decoder,
}

/// initialise ndarrays for the data group/block
fn initialise_arrays(channel_group: &mut Cg4, n_record_chunk: &u64) {
    // creates zeroed array in parallel for each channel contained in channel group
    channel_group
        .cn
        .par_iter_mut()
        .for_each(|(_cn_record_position, cn)| {
            cn.data = data_init(
                cn.block.cn_type,
                cn.block.cn_data_type,
                cn.n_bytes,
                *n_record_chunk,
            );
        })
}

/// applies bit mask if required in channel block
fn apply_bit_mask_offset(dg: &mut Dg4) {
    // apply bit shift and masking
    for channel_group in dg.cg.values_mut() {
        channel_group.cn.par_iter_mut().for_each(|(_rec_pos, cn)| {
            if cn.block.cn_data_type <= 3 {
                let left_shift =
                    cn.n_bytes * 8 - (cn.block.cn_bit_offset as u32) - cn.block.cn_bit_count;
                let right_shift = left_shift + (cn.block.cn_bit_offset as u32);
                if left_shift > 0 || right_shift > 0 {
                    match &mut cn.data {
                        ChannelData::Int8(a) => {
                            if left_shift > 0 {
                                a.map_inplace(|x| *x <<= left_shift)
                            };
                            if right_shift > 0 {
                                a.map_inplace(|x| *x >>= right_shift)
                            };
                        }
                        ChannelData::UInt8(a) => {
                            if left_shift > 0 {
                                a.map_inplace(|x| *x <<= left_shift)
                            };
                            if right_shift > 0 {
                                a.map_inplace(|x| *x >>= right_shift)
                            };
                        }
                        ChannelData::Int16(a) => {
                            if left_shift > 0 {
                                a.map_inplace(|x| *x <<= left_shift)
                            };
                            if right_shift > 0 {
                                a.map_inplace(|x| *x >>= right_shift)
                            };
                        }
                        ChannelData::UInt16(a) => {
                            if left_shift > 0 {
                                a.map_inplace(|x| *x <<= left_shift)
                            };
                            if right_shift > 0 {
                                a.map_inplace(|x| *x >>= right_shift)
                            };
                        }
                        ChannelData::Float16(_) => (),
                        ChannelData::Int24(a) => {
                            if left_shift > 0 {
                                a.map_inplace(|x| *x <<= left_shift)
                            };
                            if right_shift > 0 {
                                a.map_inplace(|x| *x >>= right_shift)
                            };
                        }
                        ChannelData::UInt24(a) => {
                            if left_shift > 0 {
                                a.map_inplace(|x| *x <<= left_shift)
                            };
                            if right_shift > 0 {
                                a.map_inplace(|x| *x >>= right_shift)
                            };
                        }
                        ChannelData::Int32(a) => {
                            if left_shift > 0 {
                                a.map_inplace(|x| *x <<= left_shift)
                            };
                            if right_shift > 0 {
                                a.map_inplace(|x| *x >>= right_shift)
                            };
                        }
                        ChannelData::UInt32(a) => {
                            if left_shift > 0 {
                                a.map_inplace(|x| *x <<= left_shift)
                            };
                            if right_shift > 0 {
                                a.map_inplace(|x| *x >>= right_shift)
                            };
                        }
                        ChannelData::Float32(_) => (),
                        ChannelData::Int48(a) => {
                            if left_shift > 0 {
                                a.map_inplace(|x| *x <<= left_shift)
                            };
                            if right_shift > 0 {
                                a.map_inplace(|x| *x >>= right_shift)
                            };
                        }
                        ChannelData::UInt48(a) => {
                            if left_shift > 0 {
                                a.map_inplace(|x| *x <<= left_shift)
                            };
                            if right_shift > 0 {
                                a.map_inplace(|x| *x >>= right_shift)
                            };
                        }
                        ChannelData::Int64(a) => {
                            if left_shift > 0 {
                                a.map_inplace(|x| *x <<= left_shift)
                            };
                            if right_shift > 0 {
                                a.map_inplace(|x| *x >>= right_shift)
                            };
                        }
                        ChannelData::UInt64(a) => {
                            if left_shift > 0 {
                                a.map_inplace(|x| *x <<= left_shift)
                            };
                            if right_shift > 0 {
                                a.map_inplace(|x| *x >>= right_shift)
                            };
                        }
                        ChannelData::Float64(_) => (),
                        ChannelData::Complex16(_) => (),
                        ChannelData::Complex32(_) => (),
                        ChannelData::Complex64(_) => (),
                        ChannelData::StringSBC(_) => (),
                        ChannelData::StringUTF8(_) => (),
                        ChannelData::StringUTF16(_) => (),
                        ChannelData::ByteArray(_) => (),
                    }
                }
            }
        })
    }
}

/// channel data type enum
#[derive(Debug, Clone)]
pub enum ChannelData {
    Int8(Array1<i8>),
    UInt8(Array1<u8>),
    Int16(Array1<i16>),
    UInt16(Array1<u16>),
    Float16(Array1<f32>),
    Int24(Array1<i32>),
    UInt24(Array1<u32>),
    Int32(Array1<i32>),
    UInt32(Array1<u32>),
    Float32(Array1<f32>),
    Int48(Array1<i64>),
    UInt48(Array1<u64>),
    Int64(Array1<i64>),
    UInt64(Array1<u64>),
    Float64(Array1<f64>),
    Complex16(Array1<Complex<f32>>),
    Complex32(Array1<Complex<f32>>),
    Complex64(Array1<Complex<f64>>),
    StringSBC(Vec<String>),
    StringUTF8(Vec<String>),
    StringUTF16(Vec<String>),
    ByteArray(Vec<u8>),
}

impl ChannelData {
    pub fn zeros(&self, cycle_count: u64, n_bytes: u32) -> ChannelData {
        match self {
            ChannelData::Int8(_) => ChannelData::Int8(
                ArrayBase::<OwnedRepr<i8>, Dim<[usize; 1]>>::zeros((cycle_count as usize,)),
            ),
            ChannelData::UInt8(_) => ChannelData::UInt8(
                ArrayBase::<OwnedRepr<u8>, Dim<[usize; 1]>>::zeros((cycle_count as usize,)),
            ),
            ChannelData::Int16(_) => ChannelData::Int16(
                ArrayBase::<OwnedRepr<i16>, Dim<[usize; 1]>>::zeros((cycle_count as usize,)),
            ),
            ChannelData::UInt16(_) => {
                ChannelData::UInt16(ArrayBase::<OwnedRepr<u16>, Dim<[usize; 1]>>::zeros((
                    cycle_count as usize,
                )))
            }
            ChannelData::Float16(_) => {
                ChannelData::Float16(ArrayBase::<OwnedRepr<f32>, Dim<[usize; 1]>>::zeros((
                    cycle_count as usize,
                )))
            }
            ChannelData::Int24(_) => ChannelData::Int24(
                ArrayBase::<OwnedRepr<i32>, Dim<[usize; 1]>>::zeros((cycle_count as usize,)),
            ),
            ChannelData::UInt24(_) => {
                ChannelData::UInt24(ArrayBase::<OwnedRepr<u32>, Dim<[usize; 1]>>::zeros((
                    cycle_count as usize,
                )))
            }
            ChannelData::Int32(_) => ChannelData::Int32(
                ArrayBase::<OwnedRepr<i32>, Dim<[usize; 1]>>::zeros((cycle_count as usize,)),
            ),
            ChannelData::UInt32(_) => {
                ChannelData::UInt32(ArrayBase::<OwnedRepr<u32>, Dim<[usize; 1]>>::zeros((
                    cycle_count as usize,
                )))
            }
            ChannelData::Float32(_) => {
                ChannelData::Float32(ArrayBase::<OwnedRepr<f32>, Dim<[usize; 1]>>::zeros((
                    cycle_count as usize,
                )))
            }
            ChannelData::Int48(_) => ChannelData::Int48(
                ArrayBase::<OwnedRepr<i64>, Dim<[usize; 1]>>::zeros((cycle_count as usize,)),
            ),
            ChannelData::UInt48(_) => {
                ChannelData::UInt48(ArrayBase::<OwnedRepr<u64>, Dim<[usize; 1]>>::zeros((
                    cycle_count as usize,
                )))
            }
            ChannelData::Int64(_) => ChannelData::Int64(
                ArrayBase::<OwnedRepr<i64>, Dim<[usize; 1]>>::zeros((cycle_count as usize,)),
            ),
            ChannelData::UInt64(_) => {
                ChannelData::UInt64(ArrayBase::<OwnedRepr<u64>, Dim<[usize; 1]>>::zeros((
                    cycle_count as usize,
                )))
            }
            ChannelData::Float64(_) => {
                ChannelData::Float64(ArrayBase::<OwnedRepr<f64>, Dim<[usize; 1]>>::zeros((
                    cycle_count as usize,
                )))
            }
            ChannelData::Complex16(_) => ChannelData::Complex16(ArrayBase::<
                OwnedRepr<Complex<f32>>,
                Dim<[usize; 1]>,
            >::zeros((
                cycle_count as usize,
            ))),
            ChannelData::Complex32(_) => ChannelData::Complex32(ArrayBase::<
                OwnedRepr<Complex<f32>>,
                Dim<[usize; 1]>,
            >::zeros((
                cycle_count as usize,
            ))),
            ChannelData::Complex64(_) => ChannelData::Complex64(ArrayBase::<
                OwnedRepr<Complex<f64>>,
                Dim<[usize; 1]>,
            >::zeros((
                cycle_count as usize,
            ))),
            ChannelData::StringSBC(_) => {
                ChannelData::StringSBC(vec![String::new(); cycle_count as usize])
            }
            ChannelData::StringUTF8(_) => {
                ChannelData::StringUTF8(vec![String::new(); cycle_count as usize])
            }
            ChannelData::StringUTF16(_) => {
                ChannelData::StringUTF16(vec![String::new(); cycle_count as usize])
            }
            ChannelData::ByteArray(_) => {
                ChannelData::ByteArray(vec![0u8; (n_bytes as u64 * cycle_count) as usize])
            }
        }
    }
}

impl Default for ChannelData {
    fn default() -> Self {
        ChannelData::UInt8(Array1::<u8>::zeros((0,)))
    }
}

/// Initialises a channel array with cycle_count zeroes and correct depending of cn_type, cn_data_type and number of bytes
pub fn data_init(cn_type: u8, cn_data_type: u8, n_bytes: u32, cycle_count: u64) -> ChannelData {
    let data_type: ChannelData;
    if cn_type != 3 || cn_type != 6 {
        if cn_data_type == 0 || cn_data_type == 1 {
            // unsigned int
            if n_bytes <= 1 {
                data_type = ChannelData::UInt8(Array1::<u8>::zeros((cycle_count as usize,)));
            } else if n_bytes == 2 {
                data_type = ChannelData::UInt16(Array1::<u16>::zeros((cycle_count as usize,)));
            } else if n_bytes == 3 {
                data_type = ChannelData::UInt24(Array1::<u32>::zeros((cycle_count as usize,)));
            } else if n_bytes == 4 {
                data_type = ChannelData::UInt32(Array1::<u32>::zeros((cycle_count as usize,)));
            } else if n_bytes <= 6 {
                data_type = ChannelData::UInt48(Array1::<u64>::zeros((cycle_count as usize,)));
            } else {
                data_type = ChannelData::UInt64(Array1::<u64>::zeros((cycle_count as usize,)));
            }
        } else if cn_data_type == 2 || cn_data_type == 3 {
            // signed int
            if n_bytes <= 1 {
                data_type = ChannelData::Int8(Array1::<i8>::zeros((cycle_count as usize,)));
            } else if n_bytes == 2 {
                data_type = ChannelData::Int16(Array1::<i16>::zeros((cycle_count as usize,)));
            } else if n_bytes == 3 {
                data_type = ChannelData::Int24(Array1::<i32>::zeros((cycle_count as usize,)));
            } else if n_bytes == 4 {
                data_type = ChannelData::Int32(Array1::<i32>::zeros((cycle_count as usize,)));
            } else if n_bytes <= 6 {
                data_type = ChannelData::Int48(Array1::<i64>::zeros((cycle_count as usize,)));
            } else {
                data_type = ChannelData::Int64(Array1::<i64>::zeros((cycle_count as usize,)));
            }
        } else if cn_data_type == 4 || cn_data_type == 5 {
            // float
            if n_bytes <= 2 {
                data_type = ChannelData::Float16(Array1::<f32>::zeros((cycle_count as usize,)));
            } else if n_bytes <= 4 {
                data_type = ChannelData::Float32(Array1::<f32>::zeros((cycle_count as usize,)));
            } else {
                data_type = ChannelData::Float64(Array1::<f64>::zeros((cycle_count as usize,)));
            }
        } else if cn_data_type == 15 || cn_data_type == 16 {
            // complex
            if n_bytes <= 2 {
                data_type =
                    ChannelData::Complex16(Array1::<Complex<f32>>::zeros((cycle_count as usize,)));
            } else if n_bytes <= 4 {
                data_type =
                    ChannelData::Complex32(Array1::<Complex<f32>>::zeros((cycle_count as usize,)));
            } else {
                data_type =
                    ChannelData::Complex64(Array1::<Complex<f64>>::zeros((cycle_count as usize,)));
            }
        } else if cn_data_type == 6 {
            // SBC ISO-8859-1 to be converted into UTF8
            data_type = ChannelData::StringSBC(vec![String::new(); cycle_count as usize]);
        } else if cn_data_type == 7 {
            // String UTF8
            data_type = ChannelData::StringUTF8(vec![String::new(); cycle_count as usize]);
        } else if cn_data_type == 8 || cn_data_type == 9 {
            // String UTF16 to be converted into UTF8
            data_type = ChannelData::StringUTF16(vec![String::new(); cycle_count as usize]);
        } else {
            // bytearray
            data_type = ChannelData::ByteArray(vec![0u8; (n_bytes as u64 * cycle_count) as usize]);
        }
    } else {
        // virtual channels, cn_bit_count = 0 -> n_bytes = 0, must be LE unsigned int
        data_type = ChannelData::UInt64(Array1::<u64>::from_iter(0..cycle_count));
    }
    data_type
}

