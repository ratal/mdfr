use std::{
    collections::{HashMap, HashSet},
    io::BufWriter,
};

use crate::{
    mdfinfo::mdfinfo4::{
        BlockType, Cg4, Cg4Block, Cn4, Cn4Block, Dg4, Dg4Block, FhBlock, MdfInfo4, MetaData,
        MetaDataBlockType,
    },
    mdfreader::channel_data::ChannelData,
};
use binrw::BinWriterExt;
use std::fs::File;

/// writes file on hard drive
pub fn mdfwriter4<'a>(writer: &'a mut BufWriter<&File>, info: &'a mut MdfInfo4) {
    let mut new_info = MdfInfo4::default();
    // IDBlock
    // HDblock
    let mut pointer: i64 = 168;
    new_info.hd_block.hd_fh_first = pointer;
    new_info.fh = Vec::new();
    let mut fh = FhBlock::default();
    pointer += 56;
    fh.fh_md_comment = pointer;
    let mut fh_comments = MetaData::new(MetaDataBlockType::MdBlock, BlockType::FH);
    fh_comments.create_fh();
    pointer += fh_comments.block.hdr_len as i64;

    new_info.hd_block.hd_dg_first = pointer;
    for (_dg_position, dg) in info.dg.iter() {
        for (_record_id, cg) in dg.cg.iter() {
            let cn_master_record_position: i64 = 0;
            let mut cg_cg_master: i64 = 0;

            // find master channel and start to write blocks for it
            if let Some((
                _master_name,
                _dg_master_position,
                (_cg_master_block_position, _record_id),
                (_cn_master_block_position, cn_master_record_position),
            )) = info.get_channel_id(&cg.master_channel_name)
            {
                if let Some(cn_master) = cg.cn.get(cn_master_record_position) {
                    // Writing master channel
                    cg_cg_master = pointer + 64; // after DGBlock

                    pointer = create_blocks(
                        &mut new_info,
                        info,
                        pointer,
                        cg,
                        cn_master,
                        cg_cg_master,
                        true,
                    );
                }
            }

            // create the other non master channel blocks
            for (_cn_record_position, cn) in cg.cn.iter() {
                // not master channel
                if cn_master_record_position != 0 {
                    pointer =
                        create_blocks(&mut new_info, info, pointer, cg, cn, cg_cg_master, false);
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
    writer.write_le(&fh).expect("Could not write FHBlock");
    fh_comments.write(writer);

    // writes the channel blocks
    for (_position, dg) in new_info.dg.iter() {
        writer.write_le(&dg.block).expect("Could not write CGBlock");
        for (_position, cg) in dg.cg.iter() {
            writer.write_le(&cg.block).expect("Could not write CGBlock");
            for (_position, cn) in cg.cn.iter() {
                writer.write_le(&cn.block).expect("Could not write CNBlock");
                writer
                    .write_le(&cn.block)
                    .expect("Could not write TXBlock for channel name");
                // TX Block channel name
                if let Some(tx_name_metadata) = new_info.sharable.md_tx.get(&cn.block.cn_tx_name) {
                    tx_name_metadata.write(writer);
                }
                if let Some(tx_unit_metadata) = new_info.sharable.md_tx.get(&cn.block.cn_md_unit) {
                    tx_unit_metadata.write(writer);
                }
                if let Some(tx_comment_metadata) =
                    new_info.sharable.md_tx.get(&cn.block.cn_md_comment)
                {
                    tx_comment_metadata.write(writer);
                }
            }
        }
    }

    // writes the channels data
}

fn create_blocks(
    new_info: &mut MdfInfo4,
    info: &MdfInfo4,
    mut pointer: i64,
    cg: &Cg4,
    cn: &Cn4,
    cg_cg_master: i64,
    master_flag: bool,
) -> i64 {
    let mut dg_block = Dg4Block::default();
    let mut cg_block = Cg4Block::default();
    let mut cn_block = Cn4Block::default();
    // DG Block
    pointer += dg_block.dg_len as i64;
    let dg_position = pointer;
    dg_block.dg_cg_first = pointer;

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
    let mut tx_name_block = MetaData::new(MetaDataBlockType::TX, BlockType::CN);
    tx_name_block.set_data_buffer(cn.unique_name.clone());
    let tx_position = pointer;
    pointer += tx_name_block.block.hdr_len as i64;
    new_info.sharable.md_tx.insert(tx_position, tx_name_block);

    // channel unit
    if let Some(str) = info.sharable.md_tx.get(&cn.block.cn_md_unit) {
        let mut tx_unit_block = MetaData::new(MetaDataBlockType::TX, BlockType::CN);
        cn_block.cn_md_unit = pointer;
        tx_unit_block.raw_data = str.raw_data.clone();
        tx_unit_block.block.hdr_len = tx_unit_block.raw_data.len() as u64 + 24;
        pointer += tx_unit_block.block.hdr_len as i64;
        new_info.sharable.md_tx.insert(pointer, tx_unit_block);
    }

    // channel comment
    if let Some(str) = info.sharable.md_tx.get(&cn.block.cn_md_unit) {
        let mut tx_comment_block = MetaData::new(MetaDataBlockType::TX, BlockType::CN);
        cn_block.cn_md_comment = pointer;
        tx_comment_block.raw_data = str.raw_data.clone();
        tx_comment_block.block.hdr_len = tx_comment_block.raw_data.len() as u64 + 24;
        pointer += tx_comment_block.block.hdr_len as i64;
        new_info.sharable.md_tx.insert(pointer, tx_comment_block);
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
        cg: HashMap::new(),
    };
    new_dg.cg.insert(0, new_cg);
    new_info.dg.insert(dg_position, new_dg);

    pointer
}
