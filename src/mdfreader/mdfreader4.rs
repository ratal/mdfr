
use crate::mdfinfo::mdfinfo4::{Blockheader4, Cg4, Dg4, MdfInfo4, parse_block_header};
use std::{collections::HashMap, convert::TryInto, io::{BufReader, Read}, usize};
use std::fs::File;
use std::str;
use std::string::String;
use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use num::Complex;
use half::f16;
use encoding_rs::{WINDOWS_1252, UTF_16LE, UTF_16BE};
use ndarray::{Array1, OwnedRepr, ArrayBase, Dim};
// use ndarray::parallel::prelude::*;

pub fn mdfreader4<'a>(rdr: &'a mut BufReader<&File>, info: &'a mut MdfInfo4) {
    let mut position: i64 = 0;
    // read file data
    for (_dg_position, dg) in info.dg.iter_mut() {
        if dg.block.dg_data != 0 {
            // header block
            rdr.seek_relative(dg.block.dg_data - position).expect("Could not position buffer");  // change buffer position
            let block_header = parse_block_header(rdr);
            position = dg.block.dg_data + 24;
            position = read_data(rdr, block_header, dg, position);
        }
    }
}

fn read_data(rdr: &mut BufReader<&File>, block_header: Blockheader4, dg: &mut Dg4, mut position: i64) -> i64 {
    if "##DT".as_bytes() == block_header.hdr_id {
        // simple data block
        if dg.cg.len() == 1 {
            // sorted data group
            // initialises all arrays
            for channel_group in dg.cg.values_mut() {
                read_all_channels_sorted(rdr, channel_group);
                position += (channel_group.record_length as i64) * (channel_group.block.cg_cycle_count as i64);
            }
            apply_bit_mask_offset(dg);
        } else if !dg.cg.is_empty() {
            // unsorted data
            // initialises all arrays
            for channel_group in dg.cg.values_mut() {
                initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone());
            }
            read_all_channels_unsorted(rdr, dg, block_header.hdr_len as i64);
            position += block_header.hdr_len as i64 - 24;
            apply_bit_mask_offset(dg);
        }
    } else if "##DZ".as_bytes() == block_header.hdr_id {
        // compressed data
        todo!();
    } else if "##HL".as_bytes() == block_header.hdr_id {
        // compressed data in datal list
        todo!();
    } else if "##DZ".as_bytes() == block_header.hdr_id {
        // compressed data
        todo!();
    } else if "##SD".as_bytes() == block_header.hdr_id {
        // signal data for VLSD
        todo!();
    } else if "##DL".as_bytes() == block_header.hdr_id {
        // data list
        todo!();
    } else if "##DV".as_bytes() == block_header.hdr_id {
        // data values
        todo!();
    }
    position
}

fn generate_chunks(channel_group: &Cg4) -> Vec<(u64, u64)>{
    const CHUNK_SIZE_READING: u64 = 524288; // can be tuned according to architecture
    let record_length = channel_group.record_length as u64;
    let n_chunks = (record_length * channel_group.block.cg_cycle_count) / CHUNK_SIZE_READING + 1;
    let chunk_length = (record_length * channel_group.block.cg_cycle_count) / n_chunks;
    let n_record_chunk = chunk_length / record_length;
    let chunck = (n_record_chunk, record_length * n_record_chunk);
    let mut chunks = vec![chunck; n_chunks as usize];
    let n_record_chunk = channel_group.block.cg_cycle_count - n_record_chunk * n_chunks;
    if n_record_chunk > 0 {
        chunks.push((n_record_chunk, record_length * n_record_chunk))
    }
    chunks
}

fn read_all_channels_sorted(rdr: &mut BufReader<&File>, channel_group: &mut Cg4) {
    let chunks =  generate_chunks(channel_group);
    // initialises the arrays
    initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone());
    // read by chunks and store in cg struct
    let mut previous_index: u64 = 0;
    for (n_record_chunk, chunk_size) in chunks {
        let mut data_chunk= vec![0u8; chunk_size as usize];
        rdr.read_exact(&mut data_chunk).expect("Could not read data chunk");
        for nrecord in 0..n_record_chunk {
            read_record_inplace(channel_group, &data_chunk, nrecord, &previous_index);
        }
        previous_index += n_record_chunk;
    }
}

fn read_all_channels_unsorted(rdr: &mut BufReader<&File>, dg: &mut Dg4, block_length: i64) {
    let mut position: i64 = 24;
    let mut record_counter: HashMap<u64, u64> = HashMap::new();
    // initialise record counter
    for cg in dg.cg.values_mut() {
        record_counter.insert(cg.block.cg_record_id, 0);
    }
    // record records
    while position < block_length {
        // reads record id
        let mut record_id= vec![0u8; dg.block.dg_rec_id_size as usize];
        rdr.read_exact(&mut record_id).expect("Could not read record id");
        let rec_id: u64;
        if dg.block.dg_rec_id_size == 1 {
            let id = (&record_id[..]).read_u8().expect("Could not convert record id u8");
            rec_id = id as u64;
        } else if dg.block.dg_rec_id_size == 2 {
            let id = (&record_id[..]).read_u16::<LittleEndian>().expect("Could not convert record id u16");
            rec_id = id as u64;
        } else if dg.block.dg_rec_id_size == 4 {
            let id = (&record_id[..]).read_u32::<LittleEndian>().expect("Could not convert record id u32");
            rec_id = id as u64;
        } else if dg.block.dg_rec_id_size == 8 {
            let id = (&record_id[..]).read_u64::<LittleEndian>().expect("Could not convert record id u64");
            rec_id = id;
        } else {rec_id = 0}
        // reads record based on record id
        if let Some(cg) = dg.cg.get_mut(&rec_id) {
            let mut record= vec![0u8; cg.record_length as usize];
            rdr.read_exact(&mut record).expect("Could not read record id");
            if let Some(nrecord) = record_counter.get_mut(&rec_id){
                read_record_inplace(cg, &record, *nrecord, &0);
                *nrecord += 1;
            }
            position += cg.record_length as i64;
        }
    }
}

//#[inline]
fn read_record_inplace(channel_group: &mut Cg4, record: &Vec<u8>, nrecord: u64, previous_index: &u64) {
    // read record for each channel
    // let mut sbc_decoder = WINDOWS_1252.new_decoder();
    // let mut utf16be_decoder = UTF_16BE.new_decoder();
    // let mut utf16le_decoder = UTF_16LE.new_decoder();
    for (_cn_record_position, cn) in channel_group.cn.iter_mut() {
        let mut value = &record[((nrecord * channel_group.record_length as u64) + cn.pos_byte_beg as u64) as usize .. 
                ((cn.pos_byte_beg + cn.n_bytes) as u64 + (nrecord * channel_group.record_length as u64)) as usize];
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
                let mut sbc_decoder = WINDOWS_1252.new_decoder();
                sbc_decoder.decode_to_string(&value, &mut array[(nrecord + previous_index )as usize], false);
            },
            ChannelData::StringUTF8(array) => {
                array[(nrecord + previous_index )as usize] = str::from_utf8(&value).expect("Found invalid UTF-8").to_string();
            },
            ChannelData::StringUTF16(array) => {
                if cn.endian{
                    let mut utf16be_decoder = UTF_16BE.new_decoder();
                    utf16be_decoder.decode_to_string(&value, &mut array[(nrecord + previous_index )as usize], false);
                } else {
                    let mut utf16le_decoder = UTF_16LE.new_decoder();
                    utf16le_decoder.decode_to_string(&value, &mut array[(nrecord + previous_index )as usize], false);
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

//#[inline]
fn initialise_arrays(channel_group: &mut Cg4, n_record_chunk: &u64) {
    // initialise ndarrays for the data group/block
    for (_cn_record_position, cn) in channel_group.cn.iter_mut() {
        cn.data = data_init(cn.block.cn_data_type, cn.n_bytes, *n_record_chunk);
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

pub fn data_init(cn_data_type: u8, n_bytes: u32, cycle_count: u64) -> ChannelData {
    let data_type: ChannelData;
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
    data_type
}

