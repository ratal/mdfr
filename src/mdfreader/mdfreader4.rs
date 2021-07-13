
use crate::mdfinfo::mdfinfo4::{Dl4Block, parser_dl4_block, parse_dz, Hl4Block, Dt4Block};
use crate::mdfinfo::mdfinfo4::{Cg4, Dg4, MdfInfo4, parse_block_header, Cc4Block, Cn4};
use std::{collections::HashMap, convert::TryInto, io::{BufReader, Read}, usize};
use std::fs::File;
use std::str;
use std::string::String;
use binread::BinReaderExt;
use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use num::Complex;
use half::f16;
use encoding_rs::{Decoder, UTF_16BE, UTF_16LE, WINDOWS_1252};
use ndarray::{Array1, OwnedRepr, ArrayBase, Dim, Zip};

// use ndarray::parallel::prelude::*;
const CHUNK_SIZE_READING: usize = 524288; // can be tuned according to architecture

pub fn mdfreader4<'a>(rdr: &'a mut BufReader<&File>, info: &'a mut MdfInfo4) {
    let mut position: i64 = 0;
    // read file data
    for (_dg_position, dg) in info.dg.iter_mut() {
        if dg.block.dg_data != 0 {
            // header block
            rdr.seek_relative(dg.block.dg_data - position).expect("Could not position buffer");  // change buffer position
            let mut id = [0u8; 4];
            rdr.read_exact(&mut id).expect("could not read block id");
            position = read_data(rdr, id, dg, dg.block.dg_data);
        }
        apply_bit_mask_offset(dg);
        // process all invalid bits
        for channel_group in dg.cg.values_mut() {
            // channel_group.process_invalid_bits();
        }
        // conversion of all channels
        convert_all_channels(dg, &info.sharable.cc);
    }
}

fn read_data(rdr: &mut BufReader<&File>, id: [u8; 4], dg: &mut Dg4, mut position: i64) -> i64 {
    // block header is already read
    let mut decoder: Dec = Dec {windows_1252: WINDOWS_1252.new_decoder(), utf_16_be: UTF_16BE.new_decoder(), utf_16_le: UTF_16LE.new_decoder()};
    if "##DT".as_bytes() == id {
        let block_header: Dt4Block = rdr.read_le().unwrap();
        // simple data block
        if dg.cg.len() == 1 {
            // sorted data group
            for channel_group in dg.cg.values_mut() {
                read_all_channels_sorted(rdr, channel_group);
            }
        } else if !dg.cg.is_empty() {
            // unsorted data
            // initialises all arrays
            for channel_group in dg.cg.values_mut() {
                initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone());
            }
            read_all_channels_unsorted(rdr, dg, block_header.len as i64);
        }
        position += block_header.len as i64;
    } else if "##DZ".as_bytes() == id {
        let (mut data, block_header) = parse_dz(rdr);
        // compressed data
        if dg.cg.len() == 1 {
            // sorted data group
            for channel_group in dg.cg.values_mut() {
                read_all_channels_sorted_from_bytes(&data, channel_group);
            }
        } else if !dg.cg.is_empty() {
            // unsorted data
            // initialises all arrays
            for channel_group in dg.cg.values_mut() {
                initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone());
            }
            // initialise record counter
            let mut record_counter: HashMap<u64, (usize, Vec<u8>)> = HashMap::new();
            for cg in dg.cg.values_mut() {
                record_counter.insert(cg.block.cg_record_id, (0, Vec::with_capacity((cg.record_length as u64 * cg.block.cg_cycle_count) as usize)));
            }
            read_all_channels_unsorted_from_bytes(&mut data, dg, &mut record_counter, &mut decoder);
        }
        position += block_header.len as i64;
    } else if "##HL".as_bytes() == id {
        // compressed data in datal list
        let block: Hl4Block = rdr.read_le().expect("could not read HL block");
        position += block.hl_len as i64;
        rdr.seek_relative(block.hl_dl_first - position).expect("Could not reach HL block");
        let mut id = [0u8; 4];
        rdr.read_exact(&mut id).expect("could not read DL block id");
        position = read_data(rdr, id, dg, position);
    } else if "##SD".as_bytes() == id {
        // signal data for VLSD
        let block_header: Dt4Block = rdr.read_le().unwrap();
        todo!();
    } else if "##DL".as_bytes() == id {
        // data list
        if dg.cg.len() == 1 {
            // sorted data group
            for channel_group in dg.cg.values_mut() {
                let (dl_blocks, pos) = parser_dl4(rdr, position);
                let pos = parser_dl4_sorted(rdr, dl_blocks, pos, channel_group);
                position = pos;
            }
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
        // list data
        todo!();
    }else if "##DV".as_bytes() == id {
        // data values
        // sorted data group only, no record id
        let block_header: Dt4Block = rdr.read_le().unwrap();
        for channel_group in dg.cg.values_mut() {
            read_all_channels_sorted(rdr, channel_group);
        }
        position += block_header.len as i64;
    }
    position
}

fn parser_dl4(rdr: &mut BufReader<&File>, mut position: i64) -> (Vec<Dl4Block>, i64) {
    // Read all DL Blocks
    let mut dl_blocks: Vec<Dl4Block> = Vec::new();
    let (block, pos) = parser_dl4_block(rdr, position, position);
    position = pos;
    dl_blocks.push(block.clone());
    let mut next_dl = block.dl_dl_next;
    while next_dl > 0 {
        let mut id = [0u8; 4];
        rdr.read_exact(&mut id).expect("could not read DL block id");
        position += 4;
        let (block, pos) = parser_dl4_block(rdr, block.dl_dl_next + 4, position);
        position = pos;
        dl_blocks.push(block.clone());
        next_dl = block.dl_dl_next;
    }
    (dl_blocks, position)
}

fn parser_dl4_sorted(rdr: &mut BufReader<&File>, dl_blocks: Vec<Dl4Block>, mut position: i64, channel_group: &mut Cg4) -> i64 {
    // initialises the arrays
    initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone());
    // Read all data blocks
    let mut decoder: Dec = Dec {windows_1252: WINDOWS_1252.new_decoder(), utf_16_be: UTF_16BE.new_decoder(), utf_16_le: UTF_16LE.new_decoder()};
    let mut data: Vec<u8> = Vec::new();
    let mut previous_index: usize = 0;
    for dl in dl_blocks {
        for data_pointer in dl.dl_data {
            rdr.seek_relative(data_pointer - position).unwrap();
            let mut id = [0u8; 4];
            rdr.read_exact(&mut id).expect("could not read data block id");
            let block_length: u64;
            if id == "##DZ".as_bytes() {
                let (dt, block_header) = parse_dz(rdr);
                data.extend(dt);
                block_length = block_header.dz_data_length;
                position = data_pointer + block_header.len as i64;
            } else {
                let block_header: Dt4Block = rdr.read_le().unwrap();
                let mut buf= vec![0u8; (block_header.len - 24) as usize];
                rdr.read_exact(&mut buf).unwrap();
                data.extend(buf);
                block_length = block_header.len - 24;
                position = data_pointer + block_header.len as i64;
            }
            let record_length = channel_group.record_length as u64;
            let n_record_chunk = block_length / record_length;
            for nrecord in 0..(n_record_chunk - 1) {
                read_channels_from_bytes(&data[(nrecord * channel_group.record_length as u64) as usize..
                    ((nrecord + 1) * channel_group.record_length as u64) as usize], channel_group, previous_index, &mut decoder);
            }
            data = data[(record_length * n_record_chunk) as usize..].to_owned();
            previous_index += n_record_chunk as usize;
        }
    }
    position
}

fn parser_dl4_unsorted(rdr: &mut BufReader<&File>, dg: &mut Dg4, dl_blocks: Vec<Dl4Block>, mut position: i64) -> i64 {
    // Read all data blocks
    let mut data: Vec<u8> = Vec::new();
    let mut decoder: Dec = Dec {windows_1252: WINDOWS_1252.new_decoder(), utf_16_be: UTF_16BE.new_decoder(), utf_16_le: UTF_16LE.new_decoder()};
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
                let mut buf= vec![0u8; (header.hdr_len - 24) as usize];
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

fn generate_chunks(channel_group: &Cg4) -> Vec<(usize, usize)>{
    let record_length = channel_group.record_length as usize;
    let cg_cycle_count = channel_group.block.cg_cycle_count as usize;
    let n_chunks = (record_length * cg_cycle_count) / CHUNK_SIZE_READING + 1;
    let chunk_length = (record_length * cg_cycle_count) / n_chunks;
    let n_record_chunk = chunk_length / record_length;
    let chunck = (n_record_chunk, record_length * n_record_chunk);
    let mut chunks = vec![chunck; n_chunks];
    let n_record_chunk = cg_cycle_count - n_record_chunk * n_chunks;
    if n_record_chunk > 0 {
        chunks.push((n_record_chunk, record_length * n_record_chunk))
    }
    chunks
}

//#[inline]
fn read_all_channels_sorted(rdr: &mut BufReader<&File>, channel_group: &mut Cg4) {
    let chunks =  generate_chunks(channel_group);
    // initialises the arrays
    initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone());
    // read by chunks and store in cg struct
    let mut previous_index: usize = 0;
    let mut decoder: Dec = Dec {windows_1252: WINDOWS_1252.new_decoder(), utf_16_be: UTF_16BE.new_decoder(), utf_16_le: UTF_16LE.new_decoder()};
    
    for (n_record_chunk, chunk_size) in chunks {
        let mut data_chunk= vec![0u8; chunk_size];
        rdr.read_exact(&mut data_chunk).expect("Could not read data chunk");
        read_channels_from_bytes(&data_chunk, channel_group, previous_index, &mut decoder);
        previous_index += n_record_chunk;
    }
}

fn read_channels_from_bytes(data_chunk: &[u8], channel_group: &mut Cg4, previous_index: usize, decoder: &mut Dec) {
    let mut value: &[u8];
    for (_rec_pos, cn) in channel_group.cn.iter_mut() {
        if cn.block.cn_type == 0 || cn.block.cn_type == 2 || cn.block.cn_type == 4 {
            // fixed length data channel, master channel of synchronisation channel
            let pos_byte_beg = cn.pos_byte_beg as usize;
            match &mut cn.data {
                ChannelData::Int8(data) => {
                        for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i8>()];
                            data[i + previous_index] = i8::from_le_bytes(value.try_into().expect("Could not read i8"));
                        }
                    },
                ChannelData::UInt8(data) => {
                        for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u8>()];
                            data[i + previous_index] = u8::from_le_bytes(value.try_into().expect("Could not read u8"));
                        }
                    },
                ChannelData::Int16(data) => {
                        if cn.endian {
                            for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i16>()];
                                data[i + previous_index] = i16::from_be_bytes(value.try_into().expect("Could not read be i16"));
                            }
                        } else {
                            for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i16>()];
                                data[i + previous_index] = i16::from_le_bytes(value.try_into().expect("Could not read le i16"));
                            }
                        }
                    },
                ChannelData::UInt16(data) => {
                        if cn.endian {
                            for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u16>()];
                                data[i + previous_index] = u16::from_be_bytes(value.try_into().expect("Could not read be u16"));
                            }
                        } else {
                            for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u16>()];
                                data[i + previous_index] = u16::from_le_bytes(value.try_into().expect("Could not read le u16"));
                            }
                        }
                    },
                ChannelData::Float16(data) => {
                        if cn.endian {
                            for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f16>()];
                                data[i + previous_index] = f16::from_be_bytes(value.try_into().expect("Could not read be f16")).to_f32();
                            }
                        } else {
                            for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f16>()];
                                data[i + previous_index] = f16::from_le_bytes(value.try_into().expect("Could not read le f16")).to_f32();
                            }
                        }
                    },
                ChannelData::Int24(data) => {
                        if cn.endian {
                            for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + 3];
                                data[i + previous_index] = value.read_i24::<BigEndian>().expect("Could not read be i24");
                            }
                        } else {
                            for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + 3];
                                data[i + previous_index] = value.read_i24::<LittleEndian>().expect("Could not read le i24");
                            }
                        }
                    },
                ChannelData::UInt24(data) => {
                        if cn.endian {
                            for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + 3];
                                data[i + previous_index] = value.read_u24::<BigEndian>().expect("Could not read be u24");
                            }
                        } else {
                            for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + 3];
                                data[i + previous_index] = value.read_u24::<LittleEndian>().expect("Could not read le u24");
                            }
                        }
                    },
                ChannelData::Int32(data) => {
                        if cn.endian {
                            for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i32>()];
                                data[i + previous_index] = i32::from_be_bytes(value.try_into().expect("Could not read be i32"));
                            }
                        } else {
                            for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i32>()];
                                data[i + previous_index] = i32::from_le_bytes(value.try_into().expect("Could not read le i32"));
                            }
                        }
                    },
                ChannelData::UInt32(data) => {
                        if cn.endian {
                            for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u32>()];
                                data[i + previous_index] = u32::from_be_bytes(value.try_into().expect("Could not read be u32"));
                            }
                        } else {
                            for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u32>()];
                                data[i + previous_index] = u32::from_le_bytes(value.try_into().expect("Could not read le u32"));
                            }
                        }
                    },
                ChannelData::Float32(data) => {
                        if cn.endian {
                            for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f32>()];
                                data[i + previous_index] = f32::from_be_bytes(value.try_into().expect("Could not read be u32"));
                            }
                        } else {
                            for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f32>()];
                                data[i + previous_index] = f32::from_le_bytes(value.try_into().expect("Could not read le u32"));
                            }
                        }
                    },
                ChannelData::Int48(data) => {
                        if cn.endian {
                            for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + 5];
                                data[i + previous_index] = value.read_i48::<BigEndian>().expect("Could not read be i48");
                            }
                        } else {
                            for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + 5];
                                data[i + previous_index] = value.read_i48::<LittleEndian>().expect("Could not read le i48");
                            }
                        }
                    },
                ChannelData::UInt48(data) => {
                        if cn.endian {
                            for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + 5];
                                data[i + previous_index] = value.read_u48::<BigEndian>().expect("Could not read be u48");
                            }
                        } else {
                            for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + 5];
                                data[i + previous_index] = value.read_u48::<LittleEndian>().expect("Could not read le u48");
                            }
                        }
                    },
                ChannelData::Int64(data) => {
                        if cn.endian {
                            for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i64>()];
                                data[i + previous_index] = i64::from_be_bytes(value.try_into().expect("Could not read be i64"));
                            }
                        } else {
                            for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<i64>()];
                                data[i + previous_index] = i64::from_le_bytes(value.try_into().expect("Could not read le i64"));
                            }
                        }
                    },
                ChannelData::UInt64(data) => {
                        if cn.endian {
                            for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u64>()];
                                data[i + previous_index] = u64::from_be_bytes(value.try_into().expect("Could not read be u64"));
                            }
                        } else {
                            for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<u64>()];
                                data[i + previous_index] = u64::from_le_bytes(value.try_into().expect("Could not read le u64"));
                            }
                        }
                    },
                ChannelData::Float64(data) => {
                        if cn.endian {
                            for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f64>()];
                                data[i + previous_index] = f64::from_be_bytes(value.try_into().expect("Could not read be f64"));
                            }
                        } else {
                            for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                                value = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f64>()];
                                data[i + previous_index] = f64::from_le_bytes(value.try_into().expect("Could not read le f64"));
                            }
                        }
                    },
                ChannelData::Complex16(data) => {
                    let mut re: f32;
                    let mut im: f32;
                    let mut re_val: &[u8];
                    let mut im_val: &[u8];
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                            re_val = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f16>()];
                            im_val = &record[pos_byte_beg + std::mem::size_of::<f16>()..pos_byte_beg + 2 * std::mem::size_of::<f16>()];
                            re = f16::from_be_bytes(re_val.try_into().expect("Could not read be real f16 complex")).to_f32();
                            im = f16::from_be_bytes(im_val.try_into().expect("Could not read be img f16 complex")).to_f32();
                            data[i + previous_index] = Complex::new(re, im);
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                            re_val = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f16>()];
                            im_val = &record[pos_byte_beg + std::mem::size_of::<f16>()..pos_byte_beg + 2 * std::mem::size_of::<f16>()];
                            re = f16::from_le_bytes(re_val.try_into().expect("Could not read le real f16 complex")).to_f32();
                            im = f16::from_le_bytes(im_val.try_into().expect("Could not read le img f16 complex")).to_f32();
                            data[i + previous_index] = Complex::new(re, im);}
                        }
                    },
                ChannelData::Complex32(data) => {
                    let mut re: f32;
                    let mut im: f32;
                    let mut re_val: &[u8];
                    let mut im_val: &[u8];
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                            re_val = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f32>()];
                            im_val = &record[pos_byte_beg + std::mem::size_of::<f32>()..pos_byte_beg + 2 * std::mem::size_of::<f32>()];
                            re = f32::from_be_bytes(re_val.try_into().expect("Could not read be real f32 complex"));
                            im = f32::from_be_bytes(im_val.try_into().expect("Could not read be img f32 complex"));
                            data[i + previous_index] = Complex::new(re, im);
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                            re_val = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f32>()];
                            im_val = &record[pos_byte_beg + std::mem::size_of::<f32>()..pos_byte_beg + 2 * std::mem::size_of::<f32>()];
                            re = f32::from_le_bytes(re_val.try_into().expect("Could not read le real f32 complex"));
                            im = f32::from_le_bytes(im_val.try_into().expect("Could not read le img f32 complex"));
                            data[i + previous_index] = Complex::new(re, im);}
                        }
                    },
                ChannelData::Complex64(data) => {
                    let mut re: f64;
                    let mut im: f64;
                    let mut re_val: &[u8];
                    let mut im_val: &[u8];
                    if cn.endian {
                        for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                            re_val = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f64>()];
                            im_val = &record[pos_byte_beg + std::mem::size_of::<f64>()..pos_byte_beg + 2 * std::mem::size_of::<f64>()];
                            re = f64::from_be_bytes(re_val.try_into().expect("Could not array"));
                            im = f64::from_be_bytes(im_val.try_into().expect("Could not array"));
                            data[i + previous_index] = Complex::new(re, im);
                        }
                    } else {
                        for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                            re_val = &record[pos_byte_beg..pos_byte_beg + std::mem::size_of::<f64>()];
                            im_val = &record[pos_byte_beg + std::mem::size_of::<f64>()..pos_byte_beg + 2 * std::mem::size_of::<f64>()];
                            re = f64::from_le_bytes(re_val.try_into().expect("Could not read le real f64 complex"));
                            im = f64::from_le_bytes(im_val.try_into().expect("Could not read le img f64 complex"));
                            data[i + previous_index] = Complex::new(re, im);}
                        }
                    },
                ChannelData::StringSBC(data) => {
                    for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + cn.n_bytes as usize];
                        let(_result, _size, _replacement) = decoder.windows_1252.decode_to_string(&value, &mut data[(i + previous_index )as usize], false);}
                    },
                ChannelData::StringUTF8(data) => {
                    for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + cn.n_bytes as usize];
                        data[(i + previous_index )as usize] = str::from_utf8(&value).expect("Found invalid UTF-8").to_string();}
                    },
                ChannelData::StringUTF16(data) => {
                    if cn.endian{
                        for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + cn.n_bytes as usize];
                            let(_result, _size, _replacement) = decoder.utf_16_be.decode_to_string(&value, &mut data[(i + previous_index )as usize], false);}
                    } else {
                        for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                            value = &record[pos_byte_beg..pos_byte_beg + cn.n_bytes as usize];
                            let(_result, _size, _replacement) = decoder.utf_16_le.decode_to_string(&value, &mut data[(i + previous_index )as usize], false);}
                    }},
                ChannelData::ByteArray(data) => {
                    for (i, record) in data_chunk.chunks(channel_group.record_length as usize).enumerate() {
                        value = &record[pos_byte_beg..pos_byte_beg + cn.n_bytes as usize];
                        data[(i + previous_index )as usize] = value.to_vec();}
                    },
            }
        } else if cn.block.cn_type == 5 {
            // Maximum length data channel
            todo!();
        }
        // virtual channels cn_type 3 & 6 are handled at initialisation
        // cn_type == 1 VLSD not possible for sorted data
    }
}

fn read_all_channels_sorted_from_bytes(data: &[u8], channel_group: &mut Cg4) {
    // initialises the arrays
    initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone());
    let mut decoder: Dec = Dec {windows_1252: WINDOWS_1252.new_decoder(), utf_16_be: UTF_16BE.new_decoder(), utf_16_le: UTF_16LE.new_decoder()};
    for nrecord in 0..channel_group.block.cg_cycle_count {
        read_channels_from_bytes(&data[(nrecord * channel_group.record_length as u64) as usize..
            ((nrecord + 1) * channel_group.record_length as u64) as usize], channel_group, 0, &mut decoder);
    }
}

fn read_all_channels_unsorted(rdr: &mut BufReader<&File>, dg: &mut Dg4, block_length: i64) {
    let data_block_length = block_length as usize;
    let mut position: usize = 24;
    let mut record_counter: HashMap<u64, (usize, Vec<u8>)> = HashMap::new();
    let mut decoder: Dec = Dec {windows_1252: WINDOWS_1252.new_decoder(), utf_16_be: UTF_16BE.new_decoder(), utf_16_le: UTF_16LE.new_decoder()};
    // initialise record counter
    for cg in dg.cg.values_mut() {
        record_counter.insert(cg.block.cg_record_id, (0, Vec::new()));
    }
    let mut data_chunk: Vec<u8>;
    while position < data_block_length {
        if (data_block_length - position) > CHUNK_SIZE_READING {
            data_chunk= vec![0u8; CHUNK_SIZE_READING];
            position += CHUNK_SIZE_READING;
        } else {
            data_chunk= vec![0u8; data_block_length - position];
            position += data_block_length - position;
        }
        rdr.read_exact(&mut data_chunk).expect("Could not read data chunk");
        read_all_channels_unsorted_from_bytes(&mut data_chunk, dg, &mut record_counter, &mut decoder);
    }
}

fn save_vlsd(data: &mut ChannelData, record: &[u8], nrecord: &usize, decoder: &mut Dec, endian: bool) {
    match data {
        ChannelData::Int8(_) => {},
        ChannelData::UInt8(_) => {},
        ChannelData::Int16(_) => {},
        ChannelData::UInt16(_) => {},
        ChannelData::Float16(_) => {},
        ChannelData::Int24(_) => {},
        ChannelData::UInt24(_) => {},
        ChannelData::Int32(_) => {},
        ChannelData::UInt32(_) => {},
        ChannelData::Float32(_) => {},
        ChannelData::Int48(_) => {},
        ChannelData::UInt48(_) => {},
        ChannelData::Int64(_) => {},
        ChannelData::UInt64(_) => {},
        ChannelData::Float64(_) => {},
        ChannelData::Complex16(_) => {},
        ChannelData::Complex32(_) => {},
        ChannelData::Complex64(_) => {},
        ChannelData::StringSBC(array) => {
            let(_result, _size, _replacement) = decoder.windows_1252.decode_to_string(&record, &mut array[*nrecord], false);},
        ChannelData::StringUTF8(array) => {
            array[*nrecord as usize] = str::from_utf8(&record).expect("Found invalid UTF-8").to_string();
        },
        ChannelData::StringUTF16(array) => {
            if endian{
                let(_result, _size, _replacement) = decoder.utf_16_be.decode_to_string(&record, &mut array[*nrecord], false);
            } else {
                let(_result, _size, _replacement) = decoder.utf_16_le.decode_to_string(&record, &mut array[*nrecord], false);
            };
        },
        ChannelData::ByteArray(_) => todo!(),
    }
}

fn read_all_channels_unsorted_from_bytes(data: &mut Vec<u8>, dg: &mut Dg4, record_counter: &mut HashMap<u64, (usize, Vec<u8>)>, decoder: &mut Dec) {
    let mut position: usize = 0;
    let data_length = data.len();
    // record records
    let mut remaining: usize = data_length - position;
    while remaining > 0 {
        // reads record id
        let rec_id: u64;
        let dg_rec_id_size = dg.block.dg_rec_id_size as usize;
        if dg_rec_id_size == 1 && remaining >= 1 {
            rec_id = data[position].try_into().expect("Could not convert record id u8");
        } else if dg_rec_id_size == 2 && remaining >= 2 {
            let rec = &data[position..position + std::mem::size_of::<u16>()];
            rec_id = u16::from_le_bytes(rec.try_into().expect("Could not convert record id u16")) as u64;
        } else if dg_rec_id_size == 4 && remaining >= 4 {
            let rec = &data[position..position + std::mem::size_of::<u32>()];
            rec_id = u32::from_le_bytes(rec.try_into().expect("Could not convert record id u32")) as u64;
        } else if dg_rec_id_size == 8 && remaining >= 8 {
            let rec = &data[position..position + std::mem::size_of::<u64>()];
            rec_id = u64::from_le_bytes(rec.try_into().expect("Could not convert record id u64")) as u64;
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
                    let length: usize = u32::from_le_bytes(len.try_into().expect("Could not read length")) as usize;
                    position += std::mem::size_of::<u32>();
                    let record = &data[position..position+ length];
                    if let Some((target_rec_id, target_rec_pos)) = cg.vlsd {
                        if let Some(target_cg) = dg.cg.get_mut(&target_rec_id) {
                            if let Some(target_cn) = target_cg.cn.get_mut(&target_rec_pos) {
                                if let Some((nrecord, _)) = record_counter.get_mut(&rec_id) {
                                    save_vlsd(&mut target_cn.data, record, nrecord, decoder, target_cn.endian);
                                    *nrecord += 1;
                                }
                            }
                        }
                    }
                    position += length;
                } else {
                    let record = &data[position..position + cg.block.cg_data_bytes as usize];
                    if let Some((nrecord, data)) = record_counter.get_mut(&rec_id){
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

    // removes consumed records from data and leave remaining that could not be processed.
    let remaining = data[position..].to_owned();
    data.clear(); // empty data but keeps capacity
    {
        let (left, _) = data.split_at_mut(remaining.len());
        left.copy_from_slice(&remaining);
    }

    for (rec_id, (index, data)) in record_counter.iter_mut() {
        if let Some(channel_group) = dg.cg.get_mut(rec_id) {
            read_channels_from_bytes(&data, channel_group, *index, decoder);
            data.clear(); // clears data for new block, keeping capacity
        }
    }
}

struct Dec {
    windows_1252: Decoder,
    utf_16_be: Decoder,
    utf_16_le: Decoder,
}

//#[inline]
fn read_record_inplace(channel_group: &mut Cg4, record: &[u8], nrecord: u64, previous_index: &u64, decoder: &mut Dec) {
    // read record for each channel
    for (_cn_record_position, cn) in channel_group.cn.iter_mut() {
        if cn.block.cn_type == 0 || cn.block.cn_type == 2 || cn.block.cn_type == 4 || cn.block.cn_type == 5 {
            let mut value = &record[cn.pos_byte_beg as usize .. 
                    (cn.pos_byte_beg + cn.n_bytes) as usize];
            match &mut cn.data {
                ChannelData::Int8(array) => 
                    array[(nrecord + previous_index )as usize] = value.read_i8().expect("Could not convert i8"),
                ChannelData::UInt8(array) => 
                    array[(nrecord + previous_index )as usize] = value.read_u8().expect("Could not convert u8"),
                ChannelData::Int16(array) => if cn.endian {
                    array[(nrecord + previous_index )as usize] = value.read_i16::<BigEndian>().expect("Could not convert be i16")
                    } else {array[(nrecord + previous_index )as usize] = value.read_i16::<LittleEndian>().expect("Could not convert le i16")},
                ChannelData::UInt16(array) => if cn.endian {
                    array[(nrecord + previous_index )as usize] = value.read_u16::<BigEndian>().expect("Could not convert be u16")
                    } else {array[(nrecord + previous_index )as usize] = value.read_u16::<LittleEndian>().expect("Could not convert le u16")},
                ChannelData::Float16(array) => {
                    if cn.endian {
                        array[(nrecord + previous_index )as usize] = f16::from_be_bytes(value.try_into().expect("Could not convert be f16")).to_f32();
                    } else {
                        array[(nrecord + previous_index )as usize] = f16::from_le_bytes(value.try_into().expect("Could not convert le f16")).to_f32();}
                    },
                ChannelData::Int32(array) => if cn.endian {
                    array[(nrecord + previous_index )as usize] = value.read_i32::<BigEndian>().expect("Could not convert be i32")
                    } else {array[(nrecord + previous_index )as usize] = value.read_i32::<LittleEndian>().expect("Could not convert le i32")},
                ChannelData::UInt32(array) => if cn.endian {
                    array[(nrecord + previous_index )as usize] = value.read_u32::<BigEndian>().expect("Could not convert be u32")
                    } else {array[(nrecord + previous_index )as usize] = value.read_u32::<LittleEndian>().expect("Could not convert le u32")},
                ChannelData::Float32(array) => if cn.endian {
                    array[(nrecord + previous_index )as usize] = value.read_f32::<BigEndian>().expect("Could not convert be f32")
                    } else {array[(nrecord + previous_index )as usize] = value.read_f32::<LittleEndian>().expect("Could not convert le f32")},
                ChannelData::Int64(array) => if cn.endian {
                    array[(nrecord + previous_index )as usize] = value.read_i64::<BigEndian>().expect("Could not convert be i64")
                    } else {array[(nrecord + previous_index )as usize] = value.read_i64::<LittleEndian>().expect("Could not convert le i64")},
                ChannelData::UInt64(array) => if cn.endian {
                    array[(nrecord + previous_index )as usize] = value.read_u64::<BigEndian>().expect("Could not convert be u64")
                    } else {array[(nrecord + previous_index )as usize] = value.read_u64::<LittleEndian>().expect("Could not convert le u64")},
                ChannelData::Float64(array) => if cn.endian {
                    array[(nrecord + previous_index )as usize] = value.read_f64::<BigEndian>().expect("Could not convert be f64")
                    } else {array[(nrecord + previous_index )as usize] = value.read_f64::<LittleEndian>().expect("Could not convert le f64")},
                ChannelData::Complex16(array) => {
                    let re_val = &value[0..1];
                    let im_val = &value[2..3];
                    let re: f32;
                    let im: f32;
                    if cn.endian {
                        re = f16::from_be_bytes(re_val.try_into().expect("Could not array")).to_f32();
                        im = f16::from_be_bytes(im_val.try_into().expect("Could not array")).to_f32();
                    } else {
                        re = f16::from_le_bytes(re_val.try_into().expect("Could not array")).to_f32();
                        im = f16::from_le_bytes(im_val.try_into().expect("Could not array")).to_f32();}
                    let comp = Complex::new(re, im);
                    array[(nrecord + previous_index )as usize] = comp;},
                ChannelData::Complex32(array) => {
                    let mut re_val = &value[0..1];
                    let mut im_val = &value[2..3];
                    let re: f32;
                    let im: f32;
                    if cn.endian {
                        re = re_val.read_f32::<BigEndian>().expect("Could not convert be f32 for real complex");
                        im = im_val.read_f32::<BigEndian>().expect("Could not convert be f32 for img complex");
                    } else {
                        re = re_val.read_f32::<LittleEndian>().expect("Could not convert le f32 for real complex");
                        im = im_val.read_f32::<LittleEndian>().expect("Could not convert le f32 for img complex");}
                    let comp = Complex::new(re, im);
                    array[(nrecord + previous_index )as usize] = comp;},
                ChannelData::Complex64(array) => {
                    let mut re_val = &value[0..3];
                    let mut im_val = &value[4..7];
                    let re: f64;
                    let im: f64;
                    if cn.endian {
                        re = re_val.read_f64::<BigEndian>().expect("Could not convert be f64 for real complex");
                        im = im_val.read_f64::<BigEndian>().expect("Could not convert be f64 for img complex");
                    } else {
                        re = re_val.read_f64::<LittleEndian>().expect("Could not convert le f64 for real complex");
                        im = im_val.read_f64::<LittleEndian>().expect("Could not convert le f64 for img complex");}
                    let comp = Complex::new(re, im);
                    array[(nrecord + previous_index )as usize] = comp;},
                ChannelData::StringSBC(array) => {
                    let(_result, _size, _replacement) = decoder.windows_1252.decode_to_string(&value, &mut array[(nrecord + previous_index )as usize], false);
                },
                ChannelData::StringUTF8(array) => {
                    array[(nrecord + previous_index )as usize] = str::from_utf8(&value).expect("Found invalid UTF-8").to_string();
                },
                ChannelData::StringUTF16(array) => {
                    if cn.endian{
                        let(_result, _size, _replacement) = decoder.utf_16_be.decode_to_string(&value, &mut array[(nrecord + previous_index )as usize], false);
                    } else {
                        let(_result, _size, _replacement) = decoder.utf_16_le.decode_to_string(&value, &mut array[(nrecord + previous_index )as usize], false);
                    };
                },
                ChannelData::ByteArray(array) => {
                    array[(nrecord + previous_index )as usize] = value.to_vec();
                },
                ChannelData::Int24(array) => if cn.endian {
                    array[(nrecord + previous_index )as usize] = value.read_i24::<BigEndian>().expect("Could not convert be i24")
                    } else {array[(nrecord + previous_index )as usize] = value.read_i24::<LittleEndian>().expect("Could not convert le i24")},
                ChannelData::UInt24(array) => if cn.endian {
                    array[(nrecord + previous_index )as usize] = value.read_u24::<BigEndian>().expect("Could not convert be u24")
                    } else {array[(nrecord + previous_index )as usize] = value.read_u24::<LittleEndian>().expect("Could not convert le u24")},
                ChannelData::Int48(array) => if cn.endian {
                    array[(nrecord + previous_index )as usize] = value.read_i48::<BigEndian>().expect("Could not convert be i48")
                    } else {array[(nrecord + previous_index )as usize] = value.read_i48::<LittleEndian>().expect("Could not convert le i48")},
                ChannelData::UInt48(array) => if cn.endian {
                    array[(nrecord + previous_index )as usize] = value.read_u48::<BigEndian>().expect("Could not convert be u48")
                    } else {array[(nrecord + previous_index )as usize] = value.read_u48::<LittleEndian>().expect("Could not convert le u48")},
            }
        }
    }
}

//#[inline]
fn initialise_arrays(channel_group: &mut Cg4, n_record_chunk: &u64) {
    // initialise ndarrays for the data group/block
    for (_cn_record_position, cn) in channel_group.cn.iter_mut() {
        cn.data = data_init(cn.block.cn_type, cn.block.cn_data_type, cn.n_bytes, *n_record_chunk);
    }
}

//#[inline]
fn apply_bit_mask_offset(dg: &mut Dg4) {
    // apply bit shift and masking
    for channel_group in dg.cg.values_mut() {
        // initialise ndarrays for the data group/block
        for (_cn_record_position, cn) in channel_group.cn.iter_mut() {
            if cn.block.cn_data_type <= 3 {
                let left_shift = cn.n_bytes * 8 - (cn.block.cn_bit_offset as u32) - cn.block.cn_bit_count;
                let right_shift = left_shift + (cn.block.cn_bit_offset as u32);
                if left_shift > 0 || right_shift > 0 {
                    match &mut cn.data {
                        ChannelData::Int8(a) => {
                            if left_shift > 0 {a.map_inplace(|x| *x <<= left_shift)};
                            if right_shift > 0 {a.map_inplace(|x| *x >>= right_shift)};
                        },
                        ChannelData::UInt8(a) => {
                            if left_shift > 0 {a.map_inplace(|x| *x <<= left_shift)};
                            if right_shift > 0 {a.map_inplace(|x| *x >>= right_shift)};
                        },
                        ChannelData::Int16(a) => {
                            if left_shift > 0 {a.map_inplace(|x| *x <<= left_shift)};
                            if right_shift > 0 {a.map_inplace(|x| *x >>= right_shift)};
                        },
                        ChannelData::UInt16(a) => {
                            if left_shift > 0 {a.map_inplace(|x| *x <<= left_shift)};
                            if right_shift > 0 {a.map_inplace(|x| *x >>= right_shift)};
                        },
                        ChannelData::Float16(_) => (),
                        ChannelData::Int24(a) => {
                            if left_shift > 0 {a.map_inplace(|x| *x <<= left_shift)};
                            if right_shift > 0 {a.map_inplace(|x| *x >>= right_shift)};
                        },
                        ChannelData::UInt24(a) => {
                            if left_shift > 0 {a.map_inplace(|x| *x <<= left_shift)};
                            if right_shift > 0 {a.map_inplace(|x| *x >>= right_shift)};
                        },
                        ChannelData::Int32(a) => {
                            if left_shift > 0 {a.map_inplace(|x| *x <<= left_shift)};
                            if right_shift > 0 {a.map_inplace(|x| *x >>= right_shift)};
                        },
                        ChannelData::UInt32(a) => {
                            if left_shift > 0 {a.map_inplace(|x| *x <<= left_shift)};
                            if right_shift > 0 {a.map_inplace(|x| *x >>= right_shift)};
                        },
                        ChannelData::Float32(_) => (),
                        ChannelData::Int48(a) => {
                            if left_shift > 0 {a.map_inplace(|x| *x <<= left_shift)};
                            if right_shift > 0 {a.map_inplace(|x| *x >>= right_shift)};
                        },
                        ChannelData::UInt48(a) => {
                            if left_shift > 0 {a.map_inplace(|x| *x <<= left_shift)};
                            if right_shift > 0 {a.map_inplace(|x| *x >>= right_shift)};
                        },
                        ChannelData::Int64(a) => {
                            if left_shift > 0 {a.map_inplace(|x| *x <<= left_shift)};
                            if right_shift > 0 {a.map_inplace(|x| *x >>= right_shift)};
                        },
                        ChannelData::UInt64(a) => {
                            if left_shift > 0 {a.map_inplace(|x| *x <<= left_shift)};
                            if right_shift > 0 {a.map_inplace(|x| *x >>= right_shift)};
                        },
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
        }
    }
}

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
    ByteArray(Vec<Vec<u8>>),
    // CANopenDate([u8; 7]),
    // CANopenTime([u8; 6]),
}

impl ChannelData {
    pub fn zeros(&self, cycle_count: u64) -> ChannelData {
        match self {
            ChannelData::Int8(_) => ChannelData::Int8(ArrayBase::<OwnedRepr<i8>, Dim<[usize; 1]>>::zeros((cycle_count as usize,))),
            ChannelData::UInt8(_) => ChannelData::UInt8(ArrayBase::<OwnedRepr<u8>, Dim<[usize; 1]>>::zeros((cycle_count as usize,))),
            ChannelData::Int16(_) => ChannelData::Int16(ArrayBase::<OwnedRepr<i16>, Dim<[usize; 1]>>::zeros((cycle_count as usize,))),
            ChannelData::UInt16(_) => ChannelData::UInt16(ArrayBase::<OwnedRepr<u16>, Dim<[usize; 1]>>::zeros((cycle_count as usize,))),
            ChannelData::Float16(_) => ChannelData::Float16(ArrayBase::<OwnedRepr<f32>, Dim<[usize; 1]>>::zeros((cycle_count as usize,))),
            ChannelData::Int24(_) => ChannelData::Int24(ArrayBase::<OwnedRepr<i32>, Dim<[usize; 1]>>::zeros((cycle_count as usize,))),
            ChannelData::UInt24(_) => ChannelData::UInt24(ArrayBase::<OwnedRepr<u32>, Dim<[usize; 1]>>::zeros((cycle_count as usize,))),
            ChannelData::Int32(_) => ChannelData::Int32(ArrayBase::<OwnedRepr<i32>, Dim<[usize; 1]>>::zeros((cycle_count as usize,))),
            ChannelData::UInt32(_) => ChannelData::UInt32(ArrayBase::<OwnedRepr<u32>, Dim<[usize; 1]>>::zeros((cycle_count as usize,))),
            ChannelData::Float32(_) => ChannelData::Float32(ArrayBase::<OwnedRepr<f32>, Dim<[usize; 1]>>::zeros((cycle_count as usize,))),
            ChannelData::Int48(_) => ChannelData::Int48(ArrayBase::<OwnedRepr<i64>, Dim<[usize; 1]>>::zeros((cycle_count as usize,))),
            ChannelData::UInt48(_) => ChannelData::UInt48(ArrayBase::<OwnedRepr<u64>, Dim<[usize; 1]>>::zeros((cycle_count as usize,))),
            ChannelData::Int64(_) => ChannelData::Int64(ArrayBase::<OwnedRepr<i64>, Dim<[usize; 1]>>::zeros((cycle_count as usize,))),
            ChannelData::UInt64(_) => ChannelData::UInt64(ArrayBase::<OwnedRepr<u64>, Dim<[usize; 1]>>::zeros((cycle_count as usize,))),
            ChannelData::Float64(_) => ChannelData::Float64(ArrayBase::<OwnedRepr<f64>, Dim<[usize; 1]>>::zeros((cycle_count as usize,))),
            ChannelData::Complex16(_) => ChannelData::Complex16(ArrayBase::<OwnedRepr<Complex<f32>>, Dim<[usize; 1]>>::zeros((cycle_count as usize,))),
            ChannelData::Complex32(_) => ChannelData::Complex32(ArrayBase::<OwnedRepr<Complex<f32>>, Dim<[usize; 1]>>::zeros((cycle_count as usize,))),
            ChannelData::Complex64(_) => ChannelData::Complex64(ArrayBase::<OwnedRepr<Complex<f64>>, Dim<[usize; 1]>>::zeros((cycle_count as usize,))),
            ChannelData::StringSBC(_) => ChannelData::StringSBC(vec![String::new(); cycle_count as usize]),
            ChannelData::StringUTF8(_) => ChannelData::StringUTF8(vec![String::new(); cycle_count as usize]),
            ChannelData::StringUTF16(_) => ChannelData::StringUTF16(vec![String::new(); cycle_count as usize]),
            ChannelData::ByteArray(data) => {let n_bytes = data[0].len();
            ChannelData::ByteArray(vec![vec![0u8; n_bytes as usize]; cycle_count as usize])},
        }
    }
}

impl Default for ChannelData {
    fn default() -> Self { ChannelData::UInt8(Array1::<u8>::zeros((0, ))) }
}

pub fn data_init(cn_type: u8, cn_data_type: u8, n_bytes: u32, cycle_count: u64) -> ChannelData {
    let data_type: ChannelData;
    if cn_type != 3 || cn_type != 6 {
        if cn_data_type == 0 || cn_data_type == 1 {
            // unsigned int
            if n_bytes <= 1 {
                data_type = ChannelData::UInt8(Array1::<u8>::zeros((cycle_count as usize, )));
            } else if n_bytes == 2 {
                data_type = ChannelData::UInt16(Array1::<u16>::zeros((cycle_count as usize, )));
            } else if n_bytes == 3 {
                data_type = ChannelData::UInt24(Array1::<u32>::zeros((cycle_count as usize, )));
            } else if n_bytes == 4 {
                data_type = ChannelData::UInt32(Array1::<u32>::zeros((cycle_count as usize, )));
            } else if n_bytes <= 6 {
                data_type = ChannelData::UInt48(Array1::<u64>::zeros((cycle_count as usize, )));
            } else {
                data_type = ChannelData::UInt64(Array1::<u64>::zeros((cycle_count as usize, )));
            }
        } else if cn_data_type == 2 || cn_data_type == 3 {
            // signed int
            if n_bytes <= 1 {
                data_type = ChannelData::Int8(Array1::<i8>::zeros((cycle_count as usize, )));
            } else if n_bytes == 2 {
                data_type = ChannelData::Int16(Array1::<i16>::zeros((cycle_count as usize, )));
            }  else if n_bytes == 3 {
                data_type = ChannelData::Int24(Array1::<i32>::zeros((cycle_count as usize, )));
            } else if n_bytes == 4 {
                data_type = ChannelData::Int32(Array1::<i32>::zeros((cycle_count as usize, )));
            } else if n_bytes <= 6 {
                data_type = ChannelData::Int48(Array1::<i64>::zeros((cycle_count as usize, )));
            }else {
                data_type = ChannelData::Int64(Array1::<i64>::zeros((cycle_count as usize, )));
            }
        } else if cn_data_type == 4 || cn_data_type == 5 {
            // float
            if n_bytes <= 2 {
                data_type = ChannelData::Float16(Array1::<f32>::zeros((cycle_count as usize, )));
            } else if n_bytes <= 4 {
                data_type = ChannelData::Float32(Array1::<f32>::zeros((cycle_count as usize, )));
            } else {
                data_type = ChannelData::Float64(Array1::<f64>::zeros((cycle_count as usize, )));
            } 
        } else if cn_data_type == 15 || cn_data_type == 16 {
            // complex
            if n_bytes <= 2 {
                data_type = ChannelData::Complex16(Array1::<Complex<f32>>::zeros((cycle_count as usize, )));
            } else if n_bytes <= 4 {
                data_type = ChannelData::Complex32(Array1::<Complex<f32>>::zeros((cycle_count as usize, )));
            } else {
                data_type = ChannelData::Complex64(Array1::<Complex<f64>>::zeros((cycle_count as usize, )));
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
            data_type = ChannelData::ByteArray(vec![vec![0u8; n_bytes as usize]; cycle_count as usize]);
        }
    } else {
        // virtual channels, cn_bit_count = 0 -> n_bytes = 0, must be LE unsigned int
        data_type = ChannelData::UInt64(Array1::<u64>::from_iter(0..cycle_count));
    }
    data_type
}

fn convert_all_channels(dg: &mut Dg4, cc: &HashMap<i64, Cc4Block>) {
    for channel_group in dg.cg.values_mut() {
        for (_cn_record_position, cn) in channel_group.cn.iter_mut() {
            if let Some(conv) = cc.get(&cn.block.cn_cc_conversion) {
                match conv.cc_type {
                    1 => linear_conversion(cn, &conv.cc_val, &channel_group.block.cg_cycle_count),
                    2 => rational_conversion(cn, &conv.cc_val, &channel_group.block.cg_cycle_count),
                    _ => {},
                }
            }
        }
    }
}

fn linear_conversion(cn: &mut Cn4, cc_val: &[f64], cycle_count: &u64) {
    let p1 = cc_val[0];
    let p2 = cc_val[1];
    if !(p1 == 0.0 && (p2 - 1.0) < 1e-12) {
        let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
        match &mut cn.data {
            ChannelData::UInt8(a) => {
                Zip::from(&mut new_array).and(a).par_for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data =  ChannelData::Float64(new_array);
                },
            ChannelData::Int8(a) => {
                Zip::from(&mut new_array).and(a).par_for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data =  ChannelData::Float64(new_array);
                },
            ChannelData::Int16(a) => {
                Zip::from(&mut new_array).and(a).par_for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data =  ChannelData::Float64(new_array);
                },
            ChannelData::UInt16(a) => {
                Zip::from(&mut new_array).and(a).par_for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data =  ChannelData::Float64(new_array);
                },
            ChannelData::Float16(a) => {
                Zip::from(&mut new_array).and(a).par_for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data =  ChannelData::Float64(new_array);
                },
            ChannelData::Int24(a) => {
                Zip::from(&mut new_array).and(a).par_for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data =  ChannelData::Float64(new_array);
                },
            ChannelData::UInt24(a) => {
                Zip::from(&mut new_array).and(a).par_for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data =  ChannelData::Float64(new_array);
                },
            ChannelData::Int32(a) => {
                Zip::from(&mut new_array).and(a).par_for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data =  ChannelData::Float64(new_array);
                },
            ChannelData::UInt32(a) => {
                Zip::from(&mut new_array).and(a).par_for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data =  ChannelData::Float64(new_array);
                },
            ChannelData::Float32(a) => {
                Zip::from(&mut new_array).and(a).par_for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data =  ChannelData::Float64(new_array);
                },
            ChannelData::Int48(a) => {
                Zip::from(&mut new_array).and(a).par_for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data =  ChannelData::Float64(new_array);
                },
            ChannelData::UInt48(a) => {
                Zip::from(&mut new_array).and(a).par_for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data =  ChannelData::Float64(new_array);
                },
            ChannelData::Int64(a) => {
                Zip::from(&mut new_array).and(a).par_for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data =  ChannelData::Float64(new_array);
                },
            ChannelData::UInt64(a) => {
                Zip::from(&mut new_array).and(a).par_for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data =  ChannelData::Float64(new_array);
                },
            ChannelData::Float64(a) => {
                Zip::from(&mut new_array).and(a).par_for_each(|new_array, a| *new_array = *a * p2 + p1);
                cn.data =  ChannelData::Float64(new_array);
            },
            ChannelData::Complex16(_) => todo!(),
            ChannelData::Complex32(_) => todo!(),
            ChannelData::Complex64(_) => todo!(),
            ChannelData::StringSBC(_) => {},
            ChannelData::StringUTF8(_) => {},
            ChannelData::StringUTF16(_) => {},
            ChannelData::ByteArray(_) => {},
        }
    }
}

fn rational_conversion(cn: &mut Cn4, cc_val: &[f64], cycle_count: &u64) {
    let p1 = cc_val[0];
    let p2 = cc_val[1];
    let p3 = cc_val[2];
    let p4 = cc_val[3];
    let p5 = cc_val[4];
    let p6 = cc_val[5];
    let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
    match &mut cn.data {
        ChannelData::UInt8(a) => {
            Zip::from(&mut new_array).and(a).par_for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)});
            cn.data =  ChannelData::Float64(new_array);
            },
        ChannelData::Int8(a) => {
            Zip::from(&mut new_array).and(a).par_for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)});
            cn.data =  ChannelData::Float64(new_array);
            },
        ChannelData::Int16(a) => {
            Zip::from(&mut new_array).and(a).par_for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)});
            cn.data =  ChannelData::Float64(new_array);
            },
        ChannelData::UInt16(a) => {
            Zip::from(&mut new_array).and(a).par_for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)});
            cn.data =  ChannelData::Float64(new_array);
            },
        ChannelData::Float16(a) => {
            Zip::from(&mut new_array).and(a).par_for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)});
            cn.data =  ChannelData::Float64(new_array);
            },
        ChannelData::Int24(a) => {
            Zip::from(&mut new_array).and(a).par_for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)});
            cn.data =  ChannelData::Float64(new_array);
            },
        ChannelData::UInt24(a) => {
            Zip::from(&mut new_array).and(a).par_for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)});
            cn.data =  ChannelData::Float64(new_array);
            },
        ChannelData::Int32(a) => {
            Zip::from(&mut new_array).and(a).par_for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)});
            cn.data =  ChannelData::Float64(new_array);
            },
        ChannelData::UInt32(a) => {
            Zip::from(&mut new_array).and(a).par_for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)});
            cn.data =  ChannelData::Float64(new_array);
            },
        ChannelData::Float32(a) => {
            Zip::from(&mut new_array).and(a).par_for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)});
            cn.data =  ChannelData::Float64(new_array);
            },
        ChannelData::Int48(a) => {
            Zip::from(&mut new_array).and(a).par_for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)});
            cn.data =  ChannelData::Float64(new_array);
            },
        ChannelData::UInt48(a) => {
            Zip::from(&mut new_array).and(a).par_for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)});
            cn.data =  ChannelData::Float64(new_array);
            },
        ChannelData::Int64(a) => {
            Zip::from(&mut new_array).and(a).par_for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)});
            cn.data =  ChannelData::Float64(new_array);
            },
        ChannelData::UInt64(a) => {
            Zip::from(&mut new_array).and(a).par_for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)});
            cn.data =  ChannelData::Float64(new_array);
            },
        ChannelData::Float64(a) => {
            Zip::from(&mut new_array).and(a).par_for_each(|new_array, a| {
                let m_2 = f64::powi(*a, 2);
                *new_array = (m_2 * p1 + *a * p2 + p1) / (m_2 * p4 + *a * p5 + p6)});
            cn.data =  ChannelData::Float64(new_array);
        },
        ChannelData::Complex16(_) => todo!(),
        ChannelData::Complex32(_) => todo!(),
        ChannelData::Complex64(_) => todo!(),
        ChannelData::StringSBC(_) => {},
        ChannelData::StringUTF8(_) => {},
        ChannelData::StringUTF16(_) => {},
        ChannelData::ByteArray(_) => {},
    }
}