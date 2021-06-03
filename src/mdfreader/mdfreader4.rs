
use crate::mdfinfo::mdfinfo4::{Blockheader4, MdfInfo4, parse_block_header, Dg4};
use std::{io::{BufReader, Cursor, Read, Seek, SeekFrom}, sync::Arc, usize};
use std::fs::File;
use std::str;
use std::string::String;
use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use dashmap::DashMap;
use num::Complex;
use half::f16;
use encoding_rs::{WINDOWS_1252, UTF_16LE, UTF_16BE};
use ndarray::{Array1, ArrayBase, Ix1};

pub fn mdfreader4(rdr: &mut BufReader<&File>, info: &mut MdfInfo4) {
    let mut position: i64 = 0;
    // read file data
    for (dg_position, dg) in info.dg.iter_mut() {
        if dg.block.dg_data != 0 {
            // header block
            rdr.seek_relative(dg.block.dg_data - position).unwrap();  // change buffer position
            let block_header = parse_block_header(rdr);
            position += 24;
            read_dt(rdr, block_header, dg, position);
        }
    }
}

fn read_dt(rdr: &mut BufReader<&File>, block_header: Blockheader4, dg: &mut Dg4, mut position: i64) {
    if "##DT".as_bytes() == block_header.hdr_id {
        // DT block
        let mut sbc_decoder = WINDOWS_1252.new_decoder();
        let mut utf16be_decoder = UTF_16BE.new_decoder();
        let mut utf16le_decoder = UTF_16LE.new_decoder();
        if dg.cg.len() == 1 {
            // sorted data group
            // if let Some(channel_group) = dg.cg.get(&dg.block.dg_cg_first){
            for (_, channel_group) in &dg.cg {
                // initialise ndarrays
                let channels = DashMap::new();
                for (cn_position, cn) in channel_group.cn.iter() {
                    channels.insert(cn_position, data_init(cn.block.cn_data_type, cn.n_bytes, channel_group.block.cg_cycle_count));
                }
                for nrecord in 0..channel_group.block.cg_cycle_count {
                    // read records
                    let mut buf= vec![0u8; channel_group.record_length as usize];
                    rdr.read_exact(&mut buf).unwrap();
                    let mut record = Cursor::new(buf);
                    for (cn_position, cn) in channel_group.cn.iter() {
                        record.seek(SeekFrom::Start(*cn_position as u64)).unwrap();
                        let c = &mut *channels.get_mut(cn_position).unwrap();
                        match &mut c.0 {
                            ChannelData::Int8(array) => if c.1 {
                                array[nrecord as usize] = record.read_i8().unwrap()
                                } else {array[nrecord as usize] = record.read_i8().unwrap()},
                            ChannelData::UInt8(array) => if c.1 {
                                array[nrecord as usize] = record.read_u8().unwrap()
                                } else {array[nrecord as usize] = record.read_u8().unwrap()},
                            ChannelData::Int16(array) => if c.1 {
                                array[nrecord as usize] = record.read_i16::<BigEndian>().unwrap()
                                } else {array[nrecord as usize] = record.read_i16::<LittleEndian>().unwrap()},
                            ChannelData::UInt16(array) => if c.1 {
                                array[nrecord as usize] = record.read_u16::<BigEndian>().unwrap()
                                } else {array[nrecord as usize] = record.read_u16::<LittleEndian>().unwrap()},
                            ChannelData::Float16(array) => {
                                let mut buf = [0u8; 2];
                                record.read_exact(&mut buf).unwrap();
                                if c.1 {
                                    array[nrecord as usize] = f16::from_be_bytes(buf).to_f32();
                                } else {
                                    array[nrecord as usize] = f16::from_le_bytes(buf).to_f32();}
                                },
                            ChannelData::Int32(array) => if c.1 {
                                array[nrecord as usize] = record.read_i32::<BigEndian>().unwrap()
                                } else {array[nrecord as usize] = record.read_i32::<LittleEndian>().unwrap()},
                            ChannelData::UInt32(array) => if c.1 {
                                array[nrecord as usize] = record.read_u32::<BigEndian>().unwrap()
                                } else {array[nrecord as usize] = record.read_u32::<LittleEndian>().unwrap()},
                            ChannelData::Float32(array) => if c.1 {
                                array[nrecord as usize] = record.read_f32::<BigEndian>().unwrap()
                                } else {array[nrecord as usize] = record.read_f32::<LittleEndian>().unwrap()},
                            ChannelData::Int64(array) => if c.1 {
                                array[nrecord as usize] = record.read_i64::<BigEndian>().unwrap()
                                } else {array[nrecord as usize] = record.read_i64::<LittleEndian>().unwrap()},
                            ChannelData::UInt64(array) => if c.1 {
                                array[nrecord as usize] = record.read_u64::<BigEndian>().unwrap()
                                } else {array[nrecord as usize] = record.read_u64::<LittleEndian>().unwrap()},
                            ChannelData::Float64(array) => if c.1 {
                                array[nrecord as usize] = record.read_f64::<BigEndian>().unwrap()
                                } else {array[nrecord as usize] = record.read_f64::<LittleEndian>().unwrap()},
                            ChannelData::Complex16(array) => {
                                let re: f32;
                                let im: f32;
                                let mut buf = [0u8; 2];
                                if c.1 {
                                    record.read_exact(&mut buf).unwrap();
                                    re = f16::from_be_bytes(buf).to_f32();
                                    record.read_exact(&mut buf).unwrap();
                                    im = f16::from_be_bytes(buf).to_f32();
                                } else {
                                    record.read_exact(&mut buf).unwrap();
                                    re = f16::from_le_bytes(buf).to_f32();
                                    record.read_exact(&mut buf).unwrap();
                                    im = f16::from_le_bytes(buf).to_f32();}
                                let comp = Complex::new(re, im);
                                array[nrecord as usize] = comp;},
                            ChannelData::Complex32(array) => {
                                let re: f32;
                                let im: f32;
                                if c.1 {
                                    re = record.read_f32::<BigEndian>().unwrap();
                                    im = record.read_f32::<BigEndian>().unwrap();
                                } else {
                                    re = record.read_f32::<LittleEndian>().unwrap();
                                    im = record.read_f32::<LittleEndian>().unwrap();}
                                let comp = Complex::new(re, im);
                                array[nrecord as usize] = comp;},
                            ChannelData::Complex64(array) => {
                                let re: f64;
                                let im: f64;
                                if c.1 {
                                    re = record.read_f64::<BigEndian>().unwrap();
                                    im = record.read_f64::<BigEndian>().unwrap();
                                } else {
                                    re = record.read_f64::<LittleEndian>().unwrap();
                                    im = record.read_f64::<LittleEndian>().unwrap();}
                                let comp = Complex::new(re, im);
                                array[nrecord as usize] = comp;},
                            ChannelData::StringSBC(array) => {
                                let mut buf = vec![0u8; cn.n_bytes as usize];
                                record.read_exact(&mut buf).unwrap();
                                sbc_decoder.decode_to_string(&buf, &mut array[nrecord as usize], false);
                            },
                            ChannelData::StringUTF8(array) => {
                                let mut buf = vec![0u8; cn.n_bytes as usize];
                                record.read_exact(&mut buf).unwrap();
                                array[nrecord as usize] = str::from_utf8(&buf).expect("Found invalid UTF-8").to_string();
                            },
                            ChannelData::StringUTF16(array) => {
                                let mut buf = vec![0u8; cn.n_bytes as usize];
                                record.read_exact(&mut buf).unwrap();
                                if c.1{
                                    utf16be_decoder.decode_to_string(&buf, &mut array[nrecord as usize], false);
                                } else {
                                    utf16le_decoder.decode_to_string(&buf, &mut array[nrecord as usize], false);
                                };
                            },
                            ChannelData::ByteArray(array) => {
                                let mut buf = vec![0u8; cn.n_bytes as usize];
                                record.read_exact(&mut buf).unwrap();
                                array[nrecord as usize] = buf;
                            },
                            ChannelData::Int24(array) => if c.1 {
                                array[nrecord as usize] = record.read_i24::<BigEndian>().unwrap()
                                } else {array[nrecord as usize] = record.read_i24::<LittleEndian>().unwrap()},
                            ChannelData::UInt24(array) => if c.1 {
                                array[nrecord as usize] = record.read_u24::<BigEndian>().unwrap()
                                } else {array[nrecord as usize] = record.read_u24::<LittleEndian>().unwrap()},
                            ChannelData::Int48(array) => if c.1 {
                                array[nrecord as usize] = record.read_i48::<BigEndian>().unwrap()
                                } else {array[nrecord as usize] = record.read_i48::<LittleEndian>().unwrap()},
                            ChannelData::UInt48(array) => if c.1 {
                                array[nrecord as usize] = record.read_u48::<BigEndian>().unwrap()
                                } else {array[nrecord as usize] = record.read_u48::<LittleEndian>().unwrap()},
                        }
                    }
                }
                position += (channel_group.record_length as i64) * (channel_group.block.cg_cycle_count as i64);
            }
        } else if dg.cg.len() >= 1 {
            // unsorted data
        }
    }
}

enum ChannelData {
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


fn data_init(cn_data_type: u8, n_bytes: u32, cycle_count: u64) -> (ChannelData, bool) {
    let data_type: ChannelData;
    let mut endian: bool = false; // Little endian by default
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
    if cn_data_type == 0 || cn_data_type == 2 || cn_data_type == 4 || cn_data_type == 8 || cn_data_type == 15 {
        endian = false; // little endian
    } else if  cn_data_type == 1 || cn_data_type == 3 || cn_data_type == 5 || cn_data_type == 9 || cn_data_type == 16 {
        endian = true;  // big endian
    }
    (data_type, endian)
}
