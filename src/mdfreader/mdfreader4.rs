use crate::mdfinfo::mdfinfo4::{Blockheader4, MdfInfo4, parse_block_header};
use std::{io::{BufReader, Cursor}, sync::Arc};
use std::fs::File;

pub fn mdfreader4(rdr: &mut BufReader<&File>, info: &mut MdfInfo4) {
    let mut position: i64 = 0;
    // read file data
    for (dg_position, dg) in info.dg.iter_mut() {
        // header block
        rdr.seek_relative(dg.block.dg_data - position).unwrap();  // change buffer position
        let block = parse_block_header(rdr);
        position += 24;

        for (cg_position, cg) in dg.cg.iter_mut() {
            if cg.block.cg_cg_next ==0 {
                // sorted channel group
            }
            for (cn_position, cn) in cg.cn.iter_mut() {
            }
        }
    }
}

fn read_data(rdr: &mut BufReader<&File>, info: &mut MdfInfo4, block_header: Blockheader4, position: i64) {
    if "##DT".as_bytes() == block_header.hdr_id {
        // DT block
    }
}