
use crate::mdfinfo::mdfinfo4::{Blockheader4, MdfInfo4, parse_block_header, Dg4};
use std::{io::{BufReader, Cursor, Read}, sync::Arc, usize};
use std::fs::File;
use dashmap::DashMap;
use num::Complex;
use half::f16;
use encoding_rs;
use ndarray::Array1;

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
        if dg.cg.len() == 1 {
            // sorted data group
            if let Some(channel_group) = dg.cg.get(&dg.block.dg_cg_first){
                // initialise ndarrays
                let channels = DashMap::new();
                for (cn_position, cn) in channel_group.cn.iter() {
                    channels.insert(cn_position, data_init(cn.block.cn_data_type, cn.n_bytes, channel_group.block.cg_cycle_count));
                }
                for nrecord in 0..channel_group.block.cg_cycle_count {
                    // read records
                    let mut buf= vec![0u8; channel_group.record_length as usize];
                    rdr.read_exact(&mut buf).unwrap();
                    let record = Cursor::new(buf);

                }
                position += (channel_group.record_length as i64) * (channel_group.block.cg_cycle_count as i64);
            }
        } else if dg.cg.len() >= 1 {
            // unsorted data
        }
    }
}

enum Channel_Data_Type {
    Int8(Array1<i8>),
    UInt8(Array1<u8>),
    Int16(Array1<i16>),
    UInt16(Array1<u16>),
    Float16(Array1<f16>),
    Int32(Array1<i32>),
    UInt32(Array1<u32>),
    Float32(Array1<f32>),
    Int64(Array1<i64>),
    UInt64(Array1<u64>),
    Float64(Array1<f64>),
    Complex16(Array1<Complex<f16>>),
    Complex32(Array1<Complex<f32>>),
    Complex64(Array1<Complex<f64>>),
    StringSBC(Vec<String>),
    StringUTF8(Vec<String>),
    StringUTF16(Vec<String>),
    ByteArray(Vec<Vec<u8>>),
    // CANopenDate([u8; 7]),
    // CANopenTime([u8; 6]),
}


fn data_init(cn_data_type: u8, n_bytes: u32, cycle_count: u64) -> (Channel_Data_Type, bool) {
    let data_type: Channel_Data_Type;
    let mut endian: bool = false; // Little endian by default
    if cn_data_type == 0 || cn_data_type == 1 {
        // unsigned int
        if n_bytes <= 1 {
            data_type = Channel_Data_Type::UInt8(Array1::<u8>::zeros((cycle_count as usize, )));
        } else if n_bytes == 2 {
            data_type = Channel_Data_Type::UInt16(Array1::<u16>::zeros((cycle_count as usize, )));
        } else if n_bytes <= 4 {
            data_type = Channel_Data_Type::UInt32(Array1::<u32>::zeros((cycle_count as usize, )));
        } else {
            data_type = Channel_Data_Type::UInt64(Array1::<u64>::zeros((cycle_count as usize, )));
        }
    } else if cn_data_type == 2 || cn_data_type == 3 {
        // signed int
        if n_bytes <= 1 {
            data_type = Channel_Data_Type::Int8(Array1::<i8>::zeros((cycle_count as usize, )));
        } else if n_bytes == 2 {
            data_type = Channel_Data_Type::Int16(Array1::<i16>::zeros((cycle_count as usize, )));
        } else if n_bytes <= 4 {
            data_type = Channel_Data_Type::Int32(Array1::<i32>::zeros((cycle_count as usize, )));
        } else {
            data_type = Channel_Data_Type::Int64(Array1::<i64>::zeros((cycle_count as usize, )));
        }
    } else if cn_data_type == 4 || cn_data_type == 5 {
        // float
        if n_bytes <= 2 {
            data_type = Channel_Data_Type::UInt16(Array1::<u16>::zeros((cycle_count as usize, )));
        } else if n_bytes <= 4 {
            data_type = Channel_Data_Type::Float32(Array1::<f32>::zeros((cycle_count as usize, )));
        } else {
            data_type = Channel_Data_Type::Float64(Array1::<f64>::zeros((cycle_count as usize, )));
        } 
    } else if cn_data_type == 15 || cn_data_type == 16 {
        // complex
        if n_bytes <= 2 {
            data_type = Channel_Data_Type::UInt32(Array1::<u32>::zeros((cycle_count as usize, )));
        } else if n_bytes <= 4 {
            data_type = Channel_Data_Type::Complex32(Array1::<Complex<f32>>::zeros((cycle_count as usize, )));
        } else {
            data_type = Channel_Data_Type::Complex64(Array1::<Complex<f64>>::zeros((cycle_count as usize, )));
        } 
    } else if cn_data_type == 6 {
        // SBC ISO-8859-1 to be converted into UTF8
        data_type = Channel_Data_Type::StringSBC(vec![String::new(); cycle_count as usize]);
    } else if cn_data_type == 7 {
        // String UTF8
        data_type = Channel_Data_Type::StringUTF8(vec![String::new(); cycle_count as usize]);
    } else if cn_data_type == 8 || cn_data_type == 9 {
        // String UTF16 to be converted into UTF8
        data_type = Channel_Data_Type::StringUTF16(vec![String::new(); cycle_count as usize]);
    } else {
        // bytearray
        data_type = Channel_Data_Type::ByteArray(vec![vec![0u8; n_bytes as usize]; cycle_count as usize]);
    }
    if cn_data_type == 0 || cn_data_type == 2 || cn_data_type == 4 || cn_data_type == 8 || cn_data_type == 15 {
        endian = false; // little endian
    } else if  cn_data_type == 1 || cn_data_type == 3 || cn_data_type == 5 || cn_data_type == 9 || cn_data_type == 16 {
        endian = true;  // big endian
    }
    (data_type, endian)
}
