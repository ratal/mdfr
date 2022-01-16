use std::{io::BufWriter, collections::{HashMap, HashSet}};

use crate::{mdfinfo::{
    mdfinfo4::{Hd4, MdfInfo4, FhBlock, MDBlock, Dg4Block, Cg4Block, Cn4Block, TXBlock, Cg4, Cn4, Dg4},
    IdBlock,
}, mdfreader::channel_data::ChannelData};
use binrw::BinWriterExt;
use std::fs::File;

/// writes file on hard drive
pub fn mdfwriter4<'a>(writer: &'a mut BufWriter<&File>, info: &'a mut MdfInfo4) {
    let mut new_info = MdfInfo4::default();
    new_info.id_block = IdBlock::default();
    let mut pointer:i64 = 64;
    new_info.hd_block = Hd4::default();
    pointer += 104;
    new_info.hd_block.hd_fh_first = pointer;
    new_info.fh = Vec::new();
    let mut fh = FhBlock::default();
    pointer += 56;
    fh.fh_md_comment = pointer;
    let mut fh_comments_header = MDBlock::default();
    fh_comments_header.fh();
    pointer += fh_comments_header.md_len as i64;

    new_info.hd_block.hd_dg_first = pointer;
    for (_dg_position, dg) in info.dg.iter() {
        for (_record_id, cg) in dg.cg.iter() {
            let cn_master_record_position: i64 = 0;
            let mut cg_cg_master: i64 = 0;
            
            // find master channel and start to write blocks for it
            if let Some((_master_name, _dg_master_position,
                (_cg_master_block_position, _record_id), 
                (_cn_master_block_position, cn_master_record_position))) 
                = info.get_channel_id(&cg.master_channel_name) {
                if let Some(cn_master) = cg.cn.get(&cn_master_record_position) {
                    // Writing master channel
                    cg_cg_master = pointer + 64; // after DGBlock

                    pointer = create_blocks(&mut new_info, info, pointer, cg, cn_master, cg_cg_master, true);
                }
            }

            // create the other non master channel blocks
            for (_cn_record_position, cn) in cg.cn.iter() {
                // not master channel
                if cn_master_record_position != 0 {
                    pointer = create_blocks(&mut new_info, info, pointer, cg, cn, cg_cg_master, false);
                }
            }
        }
    }

    writer
        .write_le(&new_info.id_block)
        .expect("Could not write IdBlock");
    writer
        .write_le(&new_info.hd_block)
        .expect("Could not write HDBlock");
    writer
        .write_le(&fh)
        .expect("Could not write FHBlock");
    writer
        .write_le(&fh_comments_header)
        .expect("Could not write FH MD comments header");

    // writes the channel blocks
    for (_position, dg) in new_info.dg.iter() {
        writer
            .write_le(&dg.block)
            .expect("Could not write CGBlock");
        for (_position, cg) in dg.cg.iter() {
            writer
                .write_le(&cg.block)
                .expect("Could not write CGBlock");
            for (_position, cn) in cg.cn.iter() {
                writer
                    .write_le(&cn.block)
                    .expect("Could not write CNBlock");
                writer
                    .write_le(&cn.block)
                    .expect("Could not write TXBlock for channel name");
                //if new_info.sharable.tx.contains(&cn.cn_cmd_unit) {
                //    writer
                //        .write_le(&cn.tx_block)
                //        .expect("Could not write TXBlock for channel unit");
                //}
                //if new_info.sharable.tx.contains(&cn.cn_md_comment) {
                //    writer
                //        .write_le(&cn.tx_block)
                //        .expect("Could not write TXBlock for channel comments");
                //}
            }
        }
    }

    // writes the channels data
}

fn create_blocks(new_info: &mut MdfInfo4, info: &MdfInfo4, mut pointer: i64,
    cg: &Cg4, cn: &Cn4, cg_cg_master: i64, master_flag: bool) -> i64 {
    let mut dg_block = Dg4Block::default();
    let mut cg_block = Cg4Block::default();
    let mut cn_block = Cn4Block::default();
    // DG Block
    pointer += dg_block.dg_len as i64;
    let dg_position = pointer;

    // CG Block
    if cg_cg_master != 0 {
        cg_block.cg_len = 7; // with cg_cg_master
        cg_block.cg_cg_master = pointer;
        cg_block.cg_flags = 0b1000;
    }
    cg_block.cg_cycle_count = cg.block.cg_cycle_count;
    let bit_count = cn.data.bit_count();
    cg_block.cg_data_bytes = bit_count / 8;
    pointer += cg_block.cg_len as i64;
    let cg_position = pointer;
    cg_block.cg_cn_first = pointer;
    
    // CN Block
    if master_flag {
        cn_block.cn_type = 2; // master channel
    }
    cn_block.cn_data_type = cn.data.data_type(cn.endian);
    cn_block.cn_bit_count = bit_count;
    let cn_position = pointer;
    pointer += cn_block.cn_len as i64;

    // channel name TX
    let mut tx_name_block = TXBlock::default();
    tx_name_block.data(cn.unique_name.clone());
    pointer += tx_name_block.tx_len as i64;

    // channel unit
    if let Some(str) = info.sharable.tx.get(&cn.block.cn_md_unit) {
        let mut tx_unit_block = TXBlock::default();
        cn_block.cn_md_unit = pointer;
        tx_unit_block.data(str.0.clone());
        new_info.sharable.tx.insert(pointer, (str.0.clone(), false));
        pointer += tx_unit_block.tx_len as i64;
    }

    // channel comment
    if let Some(str) = info.sharable.tx.get(&cn.block.cn_md_unit) {
        let mut tx_comment_block = TXBlock::default();
        cn_block.cn_md_comment = pointer;
        tx_comment_block.data(str.0.clone());
        new_info.sharable.tx.insert(pointer, (str.0.clone(), false));
        pointer += tx_comment_block.tx_len as i64;
    }


    dg_block.dg_dg_next = pointer;
    // saves the blocks in the mdfinfo4 structure
    let new_cn = Cn4 {
        block: cn_block,
        unique_name: cn.unique_name.clone(),
        data: ChannelData::default(),
        endian: cn.endian,
        block_position: cn_position,
        pos_byte_beg: 0,
        n_bytes: cg_block.cg_data_bytes,
        composition: None,
        invalid_mask: None,
        channel_data_valid: false,
    };
    let mut new_cg = Cg4 {
        block: cg_block,
        master_channel_name: cg.master_channel_name.clone(),
        cn: HashMap::new(),
        block_position: cg_position,
        channel_names: HashSet::new(),
        record_length: cg_block.cg_data_bytes,
        vlsd_cg: None,
        invalid_bytes: None,
    };
    new_cg.cn.insert(0, new_cn);
    let mut new_dg = Dg4 {
        block: dg_block,
        comments: HashMap::new(),
        cg: HashMap::new(),
    };
    new_dg.cg.insert(0, new_cg);
    new_info.dg.insert(dg_position, new_dg);
    
    pointer
}