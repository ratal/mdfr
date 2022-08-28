//! Writer of data in memory into mdf4.2 file
use std::{
    collections::{HashMap, HashSet},
    fs::OpenOptions,
    io::{BufWriter, Cursor, Seek, SeekFrom, Write},
    ops::Deref,
    sync::Arc,
    thread,
};

use crate::{
    mdfinfo::{
        mdfinfo4::{
            BlockType, Blockheader4, Ca4Block, Ca4BlockMembers, Cg4, Cg4Block, Cn4, Cn4Block,
            Compo, Composition, Dg4, Dg4Block, Dz4Block, FhBlock, Ld4Block, MdfInfo4, MetaData,
            MetaDataBlockType,
        },
        MdfInfo,
    },
    mdfreader::{
        arrow::{bit_count, to_bytes},
        channel_data::data_type_init,
        Mdf,
    },
};
use arrow2::{array::Array, bitmap::Bitmap};
use binrw::BinWriterExt;
use crossbeam_channel::bounded;
use parking_lot::Mutex;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use std::fs::File;
use yazi::{compress, CompressionLevel, Format};

use super::mdfwriter3::convert3to4;

/// writes file on hard drive
pub fn mdfwriter4(mdf: &Mdf, file_name: &str, compression: bool) -> Mdf {
    let info: MdfInfo4 = match &mdf.mdf_info {
        MdfInfo::V3(mdfinfo3) => convert3to4(mdfinfo3, file_name),
        MdfInfo::V4(mdfinfo4) => mdfinfo4.deref().clone(),
    };
    let n_channels = mdf.mdf_info.get_channel_names_set().len();
    let mut new_info = MdfInfo4::new(file_name, n_channels);
    let mut pointer: i64 = 168; // after HD block
                                // FH block
    new_info.fh = Vec::new();
    let mut fh = FhBlock::default();
    new_info.hd_block.hd_fh_first = pointer;
    pointer += 56;
    // Writes FH comments
    fh.fh_md_comment = pointer;
    let mut fh_comments = MetaData::new(MetaDataBlockType::MdBlock, BlockType::FH);
    fh_comments.create_fh();
    pointer += fh_comments.block.hdr_len as i64;
    let mut last_dg_pointer: i64 = pointer;
    new_info.hd_block.hd_dg_first = pointer;

    // builds meta data blocks for the new file
    for (_dg_position, dg) in info.dg.iter() {
        for (_record_id, cg) in dg.cg.iter() {
            let mut cg_cg_master: i64 = 0;

            // find master channel and start to write blocks for it
            if let Some(master_channel_name) = &cg.master_channel_name {
                if let Some((
                    _master_name,
                    _dg_master_position,
                    (_cg_master_block_position, _record_id),
                    (_cn_master_block_position, cn_master_record_position),
                )) = info.get_channel_id(master_channel_name)
                {
                    if let Some(cn_master) = cg.cn.get(cn_master_record_position) {
                        // Writing master channel
                        cg_cg_master = pointer + 64; // after DGBlock
                        last_dg_pointer = pointer;
                        pointer = create_blocks(
                            &mut new_info,
                            &info,
                            pointer,
                            cg,
                            cn_master,
                            &cg_cg_master,
                            true,
                        );
                    }
                }
            }

            // create the other non master channel blocks
            for (_cn_record_position, cn) in cg.cn.iter() {
                // not master channel
                if cn.block.cn_type != 2 && cn.block.cn_type != 3 {
                    last_dg_pointer = pointer;
                    pointer =
                        create_blocks(&mut new_info, &info, pointer, cg, cn, &cg_cg_master, false);
                }
            }
        }
    }
    // last DG must point to null DGBlock
    if let Some(mut last_dg) = new_info.dg.get_mut(&last_dg_pointer) {
        last_dg.block.dg_dg_next = 0;
    }

    // thread writing the channels data first as block size can be unknown due to compression
    let (tx, rx) = bounded::<Vec<u8>>(n_channels);
    let fname = Arc::new(Mutex::new(file_name.to_string()));
    let sfname = Arc::clone(&fname);
    thread::spawn(move || {
        let file_name = Arc::clone(&sfname);
        let file = file_name.lock();
        let f: File = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&*file)
            .expect("Cannot create the file");
        let mut writer = BufWriter::new(&f);

        writer
            .seek(SeekFrom::Start(pointer as u64))
            .expect("Could not reach position to write data blocks");
        for buffer in rx {
            writer
                .write_all(&buffer)
                .expect("Could not write data blocks buffer");
        }
    });

    let data_pointer = Arc::new(Mutex::new(pointer));
    new_info
        .dg
        .par_iter_mut()
        .for_each(|(_dg_block_position, dg)| {
            for (_rec_id, cg) in dg.cg.iter_mut() {
                for (_rec_pos, cn) in cg.cn.iter() {
                    let dt = mdf.get_channel_data(&cn.unique_name);
                    if let Some(data) = dt {
                        let m = data.validity();
                        if !data.is_empty() && bit_count(data) > 0 {
                            // empty strings are not written
                            let mut offset: i64 = 0;
                            let mut ld_block: Option<Ld4Block> = None;
                            if compression || m.is_some() {
                                ld_block = create_ld(m, &mut offset);
                            }

                            let data_block: (DataBlock, usize, Vec<u8>) = if compression {
                                create_dz_dv(data.clone(), &mut offset)
                            } else {
                                create_dv(data.clone(), &mut offset)
                            };

                            // invalid mask existing
                            let mut invalid_block: Option<(DataBlock, Vec<u8>)> = None;
                            if let Some(mask) = m {
                                cg.block.cg_inval_bytes = 1; // on byte (u8) for invalid mask
                                if let Some(ref mut ld) = ld_block {
                                    ld.ld_links.push(offset);
                                }
                                if compression {
                                    invalid_block = create_dz_di(mask, &mut offset);
                                } else {
                                    invalid_block = create_di(mask, &mut offset);
                                }
                            }
                            {
                                let data_pointer = Arc::clone(&data_pointer);
                                let mut locked_data_pointer = data_pointer.lock();
                                dg.block.dg_data = *locked_data_pointer;
                                *locked_data_pointer += offset;
                                let buffer = write_data_blocks(
                                    dg.block.dg_data,
                                    ld_block,
                                    data_block,
                                    invalid_block,
                                    offset as usize,
                                );
                                tx.send(buffer).expect("Channel disconnected");
                                drop(locked_data_pointer);
                            }
                        }
                    }
                }
            }
        });
    drop(tx);

    let file_name = Arc::clone(&fname);
    let file = file_name.lock();
    let f: File = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&*file)
        .expect("Cannot create the file");
    let mut writer = BufWriter::new(f);
    let mut buffer = Cursor::new(Vec::<u8>::with_capacity(pointer as usize));
    // IDBlock
    buffer
        .write_le(&new_info.id_block)
        .expect("Could not write IdBlock");
    // Writes HDblock
    buffer
        .write_le(&new_info.hd_block)
        .expect("Could not write HDBlock");
    // Writes FHBlock
    buffer.write_le(&fh).expect("Could not write FHBlock");
    fh_comments.write(&mut buffer); // FH comments

    // Writes DG+CG+CN blocks
    for (_position, dg) in new_info.dg.iter() {
        buffer.write_le(&dg.block).expect("Could not write CGBlock");
        for (_red_id, cg) in dg.cg.iter() {
            buffer.write_le(&cg.block).expect("Could not write CGBlock");
            for (_rec_pos, cn) in cg.cn.iter() {
                buffer.write_le(&cn.block).expect("Could not write CNBlock");
                // TX Block channel name
                if let Some(tx_name_metadata) = new_info.sharable.md_tx.get(&cn.block.cn_tx_name) {
                    tx_name_metadata.write(&mut buffer);
                }
                if let Some(tx_unit_metadata) = new_info.sharable.md_tx.get(&cn.block.cn_md_unit) {
                    tx_unit_metadata.write(&mut buffer);
                }
                if let Some(tx_comment_metadata) =
                    new_info.sharable.md_tx.get(&cn.block.cn_md_comment)
                {
                    tx_comment_metadata.write(&mut buffer);
                }
                // channel array
                if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(c) => {
                            let mut header = Blockheader4::default();
                            header.hdr_id = [35, 35, 67, 65]; // ##CA
                            header.hdr_len = c.ca_len as u64;
                            header.hdr_links = 1;
                            buffer
                                .write_le(&header)
                                .expect("Could not write CABlock header");
                            let ca_composition: u64 = 0;
                            buffer
                                .write_le(&ca_composition)
                                .expect("Could not write CABlock ca_composition");
                            let mut ca_block = Ca4BlockMembers::default();
                            ca_block.ca_ndim = c.ca_ndim;
                            ca_block.ca_dim_size = c.ca_dim_size.clone();
                            buffer
                                .write_le(&ca_composition)
                                .expect("Could not write CABlock members");
                        }
                        Compo::CN(_) => {}
                    }
                }
            }
        }
    }
    writer
        .write_all(&buffer.into_inner())
        .expect("Could not write DG+CG+CN blocks");
    writer.flush().expect("Could not flush file");
    Mdf {
        mdf_info: MdfInfo::V4(Box::new(new_info)),
        arrow_data: mdf.arrow_data.clone(),
        arrow_schema: mdf.arrow_schema.clone(),
        channel_indexes: mdf.channel_indexes.clone(),
    }
}

/// Writes the data blocks
fn write_data_blocks(
    position: i64,
    mut ld_block: Option<Ld4Block>,
    data_block: (DataBlock, usize, Vec<u8>),
    invalid_block: Option<(DataBlock, Vec<u8>)>,
    offset: usize,
) -> Vec<u8> {
    let mut buffer = Cursor::new(vec![0u8; offset]);
    // Writes LD block
    if let Some(ref mut ld) = ld_block {
        let id_ld: [u8; 4] = [35, 35, 76, 68]; // ##LD
        buffer.write_le(&id_ld).expect("Could not write LDBlock id");
        ld.ld_links
            .iter_mut()
            .skip(1) // spare ld_next for offset
            .for_each(|x| *x += position);
        buffer.write_le(ld).expect("Could not write LDBlock");
    }

    // Writes DV or DZ block
    match data_block.0 {
        DataBlock::DvDi(dv_block) => {
            buffer.write_le(&dv_block).expect("Could not write DVBlock");
        }
        DataBlock::DZ(dz_block) => {
            let id_dz: [u8; 4] = [35, 35, 68, 90]; // ##DZ
            buffer
                .write_le(&id_dz)
                .expect("Could not write DZDVBlock id");
            buffer
                .write_le(&dz_block)
                .expect("Could not write DZDVBlock");
        }
    }
    buffer
        .write_all(&data_block.2)
        .expect("Could not write data in DVBlock or DZBlock");
    // 8 byte align
    buffer
        .write_all(&vec![0; data_block.1])
        .expect("Could not align written data to 8 bytes");

    // invalid mask existing
    if let Some((invalid_block, invalid_bytes)) = invalid_block {
        match invalid_block {
            DataBlock::DvDi(di_block) => {
                buffer.write_le(&di_block).expect("Could not write DIBlock");
            }
            DataBlock::DZ(dz_di_block) => {
                let id_dz: [u8; 4] = [35, 35, 68, 90]; // ##DZ
                buffer
                    .write_le(&id_dz)
                    .expect("Could not write DZDIBlock id");
                buffer
                    .write_le(&dz_di_block)
                    .expect("Could not write DZDIBlock");
            }
        }
        buffer
            .write_all(&invalid_bytes)
            .expect("Could not write invalid data");
    }
    buffer.into_inner()
}

/// Create a LDBlock
fn create_ld(m: Option<&Bitmap>, offset: &mut i64) -> Option<Ld4Block> {
    let mut ld_block = Ld4Block::default();
    ld_block.ld_count = 1;
    ld_block.ld_sample_offset.push(0);
    if m.is_some() {
        ld_block.ld_n_links = (ld_block.ld_count * 2 + 1) as u64;
        ld_block.ld_flags = 1u32 << 31;
    } else {
        ld_block.ld_n_links = (ld_block.ld_count + 1) as u64;
        ld_block.ld_flags = 0b0;
    }
    ld_block.ld_len = 40 + (ld_block.ld_n_links * 8) as u64;
    *offset = ld_block.ld_len as i64;
    ld_block.ld_links.push(*offset);
    Some(ld_block)
}

/// Create a DV Block
fn create_dv(data: Box<dyn Array>, offset: &mut i64) -> (DataBlock, usize, Vec<u8>) {
    let mut dv_block = Blockheader4::default();
    dv_block.hdr_id = [35, 35, 68, 86]; // ##DV
    let data_bytes: Vec<u8> = to_bytes(&data);
    let data_bytes_len = data_bytes.len();
    dv_block.hdr_len += data_bytes_len as u64;
    let byte_aligned = 8 - data_bytes_len % 8;

    *offset += dv_block.hdr_len as i64 + byte_aligned as i64;

    (DataBlock::DvDi(dv_block), byte_aligned, data_bytes)
}

/// Enumeration of data block types
#[derive(Debug, Clone)]
enum DataBlock {
    DZ(Dz4Block),
    DvDi(Blockheader4),
}

/// Create a DZ Block of DV type
fn create_dz_dv(data: Box<dyn Array>, offset: &mut i64) -> (DataBlock, usize, Vec<u8>) {
    let mut dz_block = Dz4Block::default();
    let mut data_bytes = compress(&to_bytes(&data), Format::Zlib, CompressionLevel::Default)
        .expect("Could not compress invalid data");
    dz_block.dz_data_length = data_bytes.len() as u64;
    let dv_dz_block: DataBlock;
    let byte_aligned: usize;
    dz_block.dz_org_data_length = (data.len() * bit_count(&data) as usize / 8) as u64;
    if dz_block.dz_org_data_length < dz_block.dz_data_length {
        (dv_dz_block, byte_aligned, data_bytes) = create_dv(data, offset);
    } else {
        byte_aligned = (8 - dz_block.dz_data_length % 8) as usize;
        dz_block.dz_data_length = data_bytes.len() as u64;
        dz_block.len = dz_block.dz_data_length + 48;
        *offset += dz_block.len as i64 + byte_aligned as i64;
        dv_dz_block = DataBlock::DZ(dz_block);
    };
    (dv_dz_block, byte_aligned, data_bytes)
}

/// Create a DI Block
fn create_di(mask: &Bitmap, offset: &mut i64) -> Option<(DataBlock, Vec<u8>)> {
    let mut dv_invalid_block = Blockheader4::default();
    dv_invalid_block.hdr_id = [35, 35, 68, 73]; // ##DI
    let mask_length = mask.len();
    dv_invalid_block.hdr_len += mask_length as u64;
    let byte_aligned = 8 - mask_length % 8;
    let invalid_data: Vec<u8> = [
        mask.iter().map(|v| v as u8).collect::<Vec<u8>>(),
        vec![0; byte_aligned],
    ]
    .concat();
    *offset += dv_invalid_block.hdr_len as i64 + byte_aligned as i64;
    Some((DataBlock::DvDi(dv_invalid_block), invalid_data))
}

/// Create a DZ Block of DI type
fn create_dz_di(mask: &Bitmap, offset: &mut i64) -> Option<(DataBlock, Vec<u8>)> {
    let mut dz_invalid_block = Dz4Block::default();
    dz_invalid_block.dz_org_data_length = mask.len() as u64;
    let mut data_bytes = compress(
        mask.iter().map(|v| v as u8).collect::<Vec<u8>>().as_slice(),
        Format::Zlib,
        CompressionLevel::Default,
    )
    .expect("Could not compress invalid data");
    dz_invalid_block.dz_data_length = data_bytes.len() as u64;
    if dz_invalid_block.dz_org_data_length < dz_invalid_block.dz_data_length {
        create_di(mask, offset)
    } else {
        dz_invalid_block.len = data_bytes.len() as u64 + 48;
        let byte_aligned = 8 - data_bytes.len() % 8;
        data_bytes = [data_bytes, vec![0; byte_aligned as usize]].concat();
        dz_invalid_block.dz_org_block_type = [68, 73]; // DI
        *offset += dz_invalid_block.len as i64 + byte_aligned as i64;
        Some((DataBlock::DZ(dz_invalid_block), data_bytes))
    }
}

/// Creates the dg and following data strutures in the new mdfinfo from the old one
fn create_blocks(
    new_info: &mut MdfInfo4,
    info: &MdfInfo4,
    mut pointer: i64,
    cg: &Cg4,
    cn: &Cn4,
    cg_cg_master: &i64,
    master_flag: bool,
) -> i64 {
    let bit_count = cn.data.bit_count();
    if !cn.data.is_empty() && bit_count > 0 {
        // no empty strings
        let mut dg_block = Dg4Block::default();
        let mut cg_block = Cg4Block::default();
        let mut cn_block = Cn4Block::default();
        // DG Block
        let dg_position = pointer;
        pointer += dg_block.dg_len as i64;
        dg_block.dg_cg_first = pointer;

        // CG Block
        let cg_position = pointer;
        if *cg_cg_master != 0 && !master_flag {
            cg_block.cg_links = 7; // with cg_cg_master
            cg_block.cg_len = 112;
            cg_block.cg_cg_master = Some(*cg_cg_master);
            cg_block.cg_flags = 0b1000;
        }
        cg_block.cg_cycle_count = cg.block.cg_cycle_count;

        cg_block.cg_data_bytes = cn.data.byte_count();
        if cg.block.cg_inval_bytes > 0 {
            // One byte for invalid data as only one channel per CG
            cg_block.cg_inval_bytes = 1;
        }
        pointer += cg_block.cg_len as i64;
        cg_block.cg_cn_first = pointer;

        // CN Block
        let cn_position = pointer;
        if master_flag {
            cn_block.cn_type = cn_type_writer(cn.block.cn_type); // master channel
            if cn.block.cn_sync_type != 0 {
                cn_block.cn_sync_type = cn.block.cn_sync_type;
            } else {
                cn_block.cn_sync_type = 1; // Default is time
            }
        }

        let machine_endian: bool = cfg!(target_endian = "big");

        cn_block.cn_data_type = cn.data.data_type(machine_endian);

        cn_block.cn_bit_count = bit_count;

        pointer += cn_block.cn_len as i64;

        // channel name TX
        let mut tx_name_block = MetaData::new(MetaDataBlockType::TX, BlockType::CN);
        tx_name_block.set_data_buffer(cn.unique_name.clone().as_bytes());
        cn_block.cn_tx_name = pointer;
        let tx_name_position = pointer;
        pointer += tx_name_block.block.hdr_len as i64;
        new_info
            .sharable
            .md_tx
            .insert(tx_name_position, tx_name_block);

        // channel unit
        if let Some(unit) = info.sharable.md_tx.get(&cn.block.cn_md_unit) {
            if let Some(unit_str) = unit.get_tx_bytes() {
                let mut tx_unit_block = MetaData::new(MetaDataBlockType::TX, BlockType::CN);
                tx_unit_block.set_data_buffer(unit_str);
                cn_block.cn_md_unit = pointer;
                pointer += tx_unit_block.block.hdr_len as i64;
                new_info
                    .sharable
                    .md_tx
                    .insert(cn_block.cn_md_unit, tx_unit_block);
            }
        }

        // channel comment
        if let Some(comment) = info.sharable.md_tx.get(&cn.block.cn_md_comment) {
            if let Some(comment_str) = comment.get_tx_bytes() {
                let mut tx_comment_block = MetaData::new(MetaDataBlockType::TX, BlockType::CN);
                tx_comment_block.set_data_buffer(comment_str);
                cn_block.cn_md_comment = pointer;
                pointer += tx_comment_block.block.hdr_len as i64;
                new_info
                    .sharable
                    .md_tx
                    .insert(cn_block.cn_md_comment, tx_comment_block);
            }
        }

        // Channel array
        let data_ndim = cn.data.ndim() - 1;
        let mut composition: Option<Composition> = None;
        if data_ndim > 0 {
            let data_dim_size = cn
                .data
                .shape()
                .0
                .iter()
                .skip(1)
                .map(|x| *x as u64)
                .collect::<Vec<_>>();
            // data_dim_size.remove(0);
            let mut ca_block = Ca4Block::default();
            for x in data_dim_size.clone() {
                ca_block.snd += x as usize;
                ca_block.pnd *= x as usize;
            }
            cg_block.cg_data_bytes = ca_block.pnd as u32 * cn.data.byte_count();

            cn_block.cn_composition = pointer;
            ca_block.ca_ndim = data_ndim as u16;
            ca_block.ca_dim_size = data_dim_size.clone();
            ca_block.shape.0 = data_dim_size
                .iter()
                .map(|x| *x as usize)
                .collect::<Vec<_>>();
            ca_block.ca_len = 48 + 8 * data_ndim as u64;
            pointer += ca_block.ca_len as i64;
            composition = Some(Composition {
                block: Compo::CA(Box::new(ca_block)),
                compo: None,
            });
        }

        dg_block.dg_dg_next = pointer;
        // saves the blocks in the mdfinfo4 structure
        let new_cn = Cn4 {
            unique_name: cn.unique_name.clone(),
            data: data_type_init(
                cn_block.cn_type,
                cn_block.cn_data_type,
                cg_block.cg_data_bytes,
                data_ndim > 0,
            ),
            block: cn_block,
            endian: machine_endian,
            block_position: cn_position,
            pos_byte_beg: 0,
            n_bytes: cg_block.cg_data_bytes,
            composition,
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
        new_cg.channel_names.insert(cn.unique_name.clone());
        let mut new_dg = Dg4 {
            block: dg_block,
            cg: HashMap::new(),
        };
        new_dg.cg.insert(0, new_cg);
        new_info.dg.insert(dg_position, new_dg);
        new_info.channel_names_set.insert(
            cn.unique_name.clone(),
            (
                cg.master_channel_name.clone(), // computes at second step master channel because of cg_cg_master
                dg_position,
                (cg_position, 0),
                (cn_position, 0),
            ),
        );
    }

    pointer
}

/// supports only data and master channels
fn cn_type_writer(cn_type: u8) -> u8 {
    // not all types are supported
    match cn_type {
        0 => 0,
        1 => 0,
        2 => 2,
        3 => 2,
        4 => 0,
        5 => 0,
        6 => 0,
        _ => panic!("Unknown CN type"),
    }
}
