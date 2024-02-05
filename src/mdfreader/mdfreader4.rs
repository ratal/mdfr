//! data read and load in memory based in MdfInfo4's metadata
use crate::export::tensor::Order;
use crate::mdfinfo::mdfinfo4::{
    parse_dz, parser_dl4_block, parser_ld4_block, Dl4Block, Dt4Block, Hl4Block, Ld4Block,
};
use crate::mdfinfo::mdfinfo4::{Blockheader4, Cg4, Cn4, Compo, Dg4};
use crate::mdfinfo::MdfInfo;
use crate::mdfreader::channel_data::ChannelData;
use crate::mdfreader::conversions4::convert_all_channels;
use crate::mdfreader::data_read4::read_channels_from_bytes;
use crate::mdfreader::data_read4::read_one_channel_array;
use anyhow::{bail, Context, Error, Result};
use binrw::BinReaderExt;
use encoding_rs::{Decoder, UTF_16BE, UTF_16LE, WINDOWS_1252};
use rayon::prelude::*;
use std::fs::File;
use std::io::Cursor;
use std::str;
use std::string::String;
use std::{
    collections::{HashMap, HashSet},
    convert::TryInto,
    io::{BufReader, Read},
    usize,
};

use super::Mdf;

/// The following constant represents the size of data chunk to be read and processed.
/// a big chunk will improve performance but consume more memory
/// a small chunk will not consume too much memory but will cause many read calls, penalising performance
pub const CHUNK_SIZE_READING_4: usize = 524288; // can be tuned according to architecture

/// Reads the file data based on headers information contained in info parameter
/// Hashset of channel names parameter allows to filter which channels to read
pub fn mdfreader4<'a>(
    rdr: &'a mut BufReader<&File>,
    mdf: &'a mut Mdf,
    channel_names: &HashSet<String>,
) -> Result<(), Error> {
    match &mut mdf.mdf_info {
        MdfInfo::V4(info) => {
            let mut position: i64 = 0;
            let mut sorted: bool;
            let mut channel_names_present_in_dg: HashSet<String>;
            let mut decoder: Dec = Dec {
                windows_1252: WINDOWS_1252.new_decoder(),
                utf_16_be: UTF_16BE.new_decoder(),
                utf_16_le: UTF_16LE.new_decoder(),
            };
            // read file data
            for (_dg_position, dg) in info.dg.iter_mut() {
                // Let's find channel names
                channel_names_present_in_dg = HashSet::new();
                for channel_group in dg.cg.values() {
                    let cn = channel_group.channel_names.clone();
                    channel_names_present_in_dg.par_extend(cn);
                }
                let channel_names_to_read_in_dg: HashSet<_> = channel_names_present_in_dg
                    .into_par_iter()
                    .filter(|v| channel_names.contains(v))
                    .collect();
                if dg.block.dg_data != 0 && !channel_names_to_read_in_dg.is_empty() {
                    // header block
                    rdr.seek_relative(dg.block.dg_data - position)
                        .context("Could not position buffer")?; // change buffer position
                    let mut id = [0u8; 4];
                    rdr.read_exact(&mut id).context("could not read block id")?;
                    sorted = dg.cg.len() == 1;
                    position = read_data(
                        rdr,
                        id,
                        dg,
                        dg.block.dg_data,
                        sorted,
                        &channel_names_to_read_in_dg,
                        &mut decoder,
                    )
                    .with_context(|| format!("failed reading data for dg {:?}", dg))?;
                    apply_bit_mask_offset(dg, &channel_names_to_read_in_dg)
                        .context("failed applying bit mask offset")?;
                    // channel_group invalid bits calculation (only for DIBlocks)
                    for channel_group in dg.cg.values_mut() {
                        channel_group
                            .process_all_channel_invalid_bits()
                            .context("failed processing all channel invalid bits")?;
                    }
                    // conversion of all channels to physical values
                    convert_all_channels(dg, &info.sharable)
                        .context("failed converting all channels")?;
                }
            }
        }
        MdfInfo::V3(_) => {}
    };
    Ok(())
}

/// Reads all kind of data layout : simple DT or DV, sorted or unsorted, Data List,
/// compressed data blocks DZ or Sample DATA
fn read_data(
    rdr: &mut BufReader<&File>,
    id: [u8; 4],
    dg: &mut Dg4,
    mut position: i64,
    sorted: bool,
    channel_names_to_read_in_dg: &HashSet<String>,
    decoder: &mut Dec,
) -> Result<i64> {
    // block header is already read
    let mut vlsd_channels: Vec<i32> = Vec::new();
    match id {
        [35, 35, 68, 84] => {
            // ##DT
            let block_header: Dt4Block = rdr
                .read_le()
                .context("could not read into Dt4Block structure")?;
            // simple data block
            if sorted {
                // sorted data group
                for channel_group in dg.cg.values_mut() {
                    vlsd_channels =
                        read_all_channels_sorted(rdr, channel_group, channel_names_to_read_in_dg)
                            .context("failed reading all channels sorted")?;
                    position += block_header.len as i64;
                }
                if !vlsd_channels.is_empty() {
                    position = read_sd(
                        rdr,
                        dg,
                        &vlsd_channels,
                        position,
                        decoder,
                        channel_names_to_read_in_dg,
                    )
                    .context("failed reading sd block")?;
                }
            } else if !dg.cg.is_empty() {
                // unsorted data
                // initialises all arrays
                for channel_group in dg.cg.values_mut() {
                    initialise_arrays(
                        channel_group,
                        &channel_group.block.cg_cycle_count.clone(),
                        channel_names_to_read_in_dg,
                    )
                    .context("failed intialising arrays")?;
                }
                read_all_channels_unsorted(
                    rdr,
                    dg,
                    block_header.len as i64,
                    channel_names_to_read_in_dg,
                )
                .context("failed reading all channels unsorted")?;
                position += block_header.len as i64;
            }
        }
        [35, 35, 68, 90] => {
            // ##DZ
            let (mut data, block_header) = parse_dz(rdr)?;
            // compressed data
            if sorted {
                // sorted data group
                for channel_group in dg.cg.values_mut() {
                    vlsd_channels = read_all_channels_sorted_from_bytes(
                        &data,
                        channel_group,
                        channel_names_to_read_in_dg,
                    )
                    .context("failed reading all channels sorted from bytes")?;
                }
                position += block_header.len as i64;
                if !vlsd_channels.is_empty() {
                    position = read_sd(
                        rdr,
                        dg,
                        &vlsd_channels,
                        position,
                        decoder,
                        channel_names_to_read_in_dg,
                    )
                    .context("failed reading SD block")?;
                }
            } else if !dg.cg.is_empty() {
                // unsorted data
                // initialises all arrays
                for channel_group in dg.cg.values_mut() {
                    initialise_arrays(
                        channel_group,
                        &channel_group.block.cg_cycle_count.clone(),
                        channel_names_to_read_in_dg,
                    )
                    .context("failed intialising arrays")?;
                }
                // initialise record counter
                let mut record_counter: HashMap<u64, (usize, Vec<u8>)> =
                    HashMap::with_capacity(dg.cg.len());
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
                read_all_channels_unsorted_from_bytes(
                    &mut data,
                    dg,
                    &mut record_counter,
                    decoder,
                    channel_names_to_read_in_dg,
                )
                .context("failed reading all channels sorted from bytes")?;
                position += block_header.len as i64;
            }
        }
        [35, 35, 72, 76] => {
            // ##HL
            let (pos, id) = read_hl(rdr, position)?;
            position = pos;
            // Read DL Blocks
            position = read_data(
                rdr,
                id,
                dg,
                position,
                sorted,
                channel_names_to_read_in_dg,
                decoder,
            )
            .context("failed reading data from HL block")?;
        }
        [35, 35, 68, 76] => {
            // ##DL
            // data list
            if sorted {
                // sorted data group
                for channel_group in dg.cg.values_mut() {
                    let (dl_blocks, pos) = parser_dl4(rdr, position)?;
                    let (pos, vlsd) = parser_dl4_sorted(
                        rdr,
                        dl_blocks,
                        pos,
                        channel_group,
                        decoder,
                        &0i32,
                        channel_names_to_read_in_dg,
                    )
                    .context("failed parsing DL4 sorted")?;
                    position = pos;
                    vlsd_channels = vlsd;
                }
                if !vlsd_channels.is_empty() {
                    position = read_sd(
                        rdr,
                        dg,
                        &vlsd_channels,
                        position,
                        decoder,
                        channel_names_to_read_in_dg,
                    )
                    .context("failed reading SD block")?;
                }
            } else if !dg.cg.is_empty() {
                // unsorted data
                // initialises all arrays
                for channel_group in dg.cg.values_mut() {
                    initialise_arrays(
                        channel_group,
                        &channel_group.block.cg_cycle_count.clone(),
                        channel_names_to_read_in_dg,
                    )
                    .context("failed intialising arrays")?;
                }
                let (dl_blocks, pos) = parser_dl4(rdr, position)?;
                let pos = parser_dl4_unsorted(rdr, dg, dl_blocks, pos, channel_names_to_read_in_dg)
                    .context("failed parsing DL4 block unsorted")?;
                position = pos;
            }
        }
        [35, 35, 76, 68] => {
            // ##LD
            // list data, cannot be used for unsorted data
            for channel_group in dg.cg.values_mut() {
                let pos = parser_ld4(rdr, position, channel_group, channel_names_to_read_in_dg)
                    .context("failed parsing DL4 block")?;
                position = pos;
            }
        }
        [35, 35, 68, 86] => {
            // ##DV
            // data values
            // sorted data group only, no record id, no invalid bytes
            let block_header: Dt4Block = rdr
                .read_le()
                .context("could not read into Dv4Block structure")?;
            for channel_group in dg.cg.values_mut() {
                read_all_channels_sorted(rdr, channel_group, channel_names_to_read_in_dg)
                    .context("failed reading all channels sorted")?;
            }
            position += block_header.len as i64;
        }
        [35, 35, 68, 71] => {
            // ##DG
            bail!("Weird, a DG block type {:?}", id) // should never happen
        }
        _ => bail!("Unknown data block type {:?}", id), // should never happen
    }
    Ok(position)
}

/// Header List block reader
/// This HL Block references Data List Blocks that are listing DZ Blocks
/// It is existing to add complementary information about compression in DZ
fn read_hl(rdr: &mut BufReader<&File>, mut position: i64) -> Result<(i64, [u8; 4])> {
    // compressed data in datal list
    let block: Hl4Block = rdr.read_le().context("could not read HL block")?;
    position += block.hl_len as i64;
    // Read Id of pointed DL Block
    rdr.seek_relative(block.hl_dl_first - position)
        .context("Could not reach DL block from HL block")?;
    position = block.hl_dl_first;
    let mut id = [0u8; 4];
    rdr.read_exact(&mut id)
        .context("could not read DL block id")?;
    Ok((position, id))
}

/// Reads Signal Data Block containing VLSD channel, pointed by cn_data
fn read_sd(
    rdr: &mut BufReader<&File>,
    dg: &mut Dg4,
    vlsd_channels: &[i32],
    mut position: i64,
    decoder: &mut Dec,
    channel_names_to_read_in_dg: &HashSet<String>,
) -> Result<i64> {
    for channel_group in dg.cg.values_mut() {
        for rec_pos in vlsd_channels {
            if let Some(cn) = channel_group.cn.get_mut(rec_pos) {
                // header block
                rdr.seek_relative(cn.block.cn_data - position)
                    .context("Could not position buffer")?; // change buffer position
                position = cn.block.cn_data;
                let mut id = [0u8; 4];
                rdr.read_exact(&mut id).context("could not read block id")?;
                if "##SD".as_bytes() == id {
                    let block_header: Dt4Block =
                        rdr.read_le().context("Could not read Sd4Block struct")?;
                    let mut data = vec![0u8; block_header.len as usize - 24];
                    rdr.read_exact(&mut data)
                        .context("could not read SD data buffer")?;
                    position += block_header.len as i64;
                    read_vlsd_from_bytes(&mut data, cn, 0, decoder)?;
                } else if "##DZ".as_bytes() == id {
                    let (mut data, block_header) = parse_dz(rdr)?;
                    position += block_header.len as i64;
                    read_vlsd_from_bytes(&mut data, cn, 0, decoder)?;
                } else if "##HL".as_bytes() == id {
                    let (pos, _id) = read_hl(rdr, position)?;
                    position = pos;
                    let (dl_blocks, pos) = parser_dl4(rdr, position)?;
                    let (pos, _vlsd) = parser_dl4_sorted(
                        rdr,
                        dl_blocks,
                        pos,
                        channel_group,
                        decoder,
                        rec_pos,
                        channel_names_to_read_in_dg,
                    )?;
                    position = pos;
                } else if "##DL".as_bytes() == id {
                    let (dl_blocks, pos) = parser_dl4(rdr, position)?;
                    let (pos, _vlsd) = parser_dl4_sorted(
                        rdr,
                        dl_blocks,
                        pos,
                        channel_group,
                        decoder,
                        rec_pos,
                        channel_names_to_read_in_dg,
                    )?;
                    position = pos;
                }
            }
        }
    }
    Ok(position)
}

/// Reads Variable Length Signal Data from bytes of a SD Block
/// It shall contain data of only one VLSD channel
/// Each reacord is starting from its length headed by a u32
fn read_vlsd_from_bytes(
    data: &mut Vec<u8>,
    cn: &mut Cn4,
    previous_index: usize,
    decoder: &mut Dec,
) -> Result<usize> {
    let mut position: usize = 0;
    let data_length = data.len();
    let mut remaining: usize = data_length - position;
    let mut nrecord: usize = 0;
    match &mut cn.data {
        ChannelData::Int8(_) => {}
        ChannelData::UInt8(_) => {}
        ChannelData::Int16(_) => {}
        ChannelData::UInt16(_) => {}
        ChannelData::Int32(_) => {}
        ChannelData::UInt32(_) => {}
        ChannelData::Float32(_) => {}
        ChannelData::Int64(_) => {}
        ChannelData::UInt64(_) => {}
        ChannelData::Float64(_) => {}
        ChannelData::Complex32(_) => {}
        ChannelData::Complex64(_) => {}
        ChannelData::Utf8(array) => {
            if cn.block.cn_data_type == 6 {
                while remaining > 0 {
                    let len = &data[position..position + std::mem::size_of::<u32>()];
                    let length: usize =
                        u32::from_le_bytes(len.try_into().context("Could not read length")?)
                            as usize;
                    if (position + length + 4) <= data_length {
                        position += std::mem::size_of::<u32>();
                        let record = &data[position..position + length - 1]; // do not take null terminated character
                        let mut dst = String::with_capacity(record.len());
                        let (_result, _size, _replacement) = decoder
                            .windows_1252
                            .decode_to_string(record, &mut dst, false);
                        array.push(Some(dst));
                        position += length;
                        remaining = data_length - position;
                        nrecord += 1;
                    } else {
                        remaining = data_length - position;
                        // copies tail part at beginnning of vect
                        data.copy_within(position.., 0);
                        // clears the last part
                        data.truncate(remaining);
                        break;
                    }
                }
            } else if cn.block.cn_data_type == 7 {
                while remaining > 0 {
                    let len = &data[position..position + std::mem::size_of::<u32>()];
                    let length: usize =
                        u32::from_le_bytes(len.try_into().context("Could not read length")?)
                            as usize;
                    if (position + length + 4) <= data_length {
                        position += std::mem::size_of::<u32>();
                        let record = &data[position..position + length - 1]; // do not take null terminated character
                        let dst = str::from_utf8(record).context("Found invalid UTF-8")?;
                        array.push(Some(dst));
                        position += length;
                        remaining = data_length - position;
                        nrecord += 1;
                    } else {
                        remaining = data_length - position;
                        // copies tail part at beginnning of vect
                        data.copy_within(position.., 0);
                        // clears the last part
                        data.truncate(remaining);
                        break;
                    }
                }
            } else if cn.block.cn_data_type == 8 {
                while remaining > 0 {
                    let len = &data[position..position + std::mem::size_of::<u32>()];
                    let length: usize =
                        u32::from_le_bytes(len.try_into().context("Could not read length")?)
                            as usize;
                    if (position + length + 4) <= data_length {
                        position += std::mem::size_of::<u32>();
                        let record = &data[position..position + length];
                        let mut dst = String::with_capacity(record.len());
                        let (_result, _size, _replacement) =
                            decoder.utf_16_le.decode_to_string(record, &mut dst, false);
                        dst = dst.trim_end_matches('\0').to_owned();
                        array.push(Some(dst));
                        position += length;
                        remaining = data_length - position;
                        nrecord += 1;
                    } else {
                        remaining = data_length - position;
                        // copies tail part at beginnning of vect
                        data.copy_within(position.., 0);
                        // clears the last part
                        data.truncate(remaining);
                        break;
                    }
                }
            } else if cn.block.cn_data_type == 9 {
                while remaining > 0 {
                    let len = &data[position..position + std::mem::size_of::<u32>()];
                    let length: usize =
                        u32::from_le_bytes(len.try_into().context("Could not read length")?)
                            as usize;
                    if (position + length + 4) <= data_length {
                        position += std::mem::size_of::<u32>();
                        let record = &data[position..position + length];
                        let mut dst = String::with_capacity(record.len());
                        let (_result, _size, _replacement) =
                            decoder.utf_16_be.decode_to_string(record, &mut dst, false);
                        dst = dst.trim_end_matches('\0').to_owned();
                        array.push(Some(dst));
                        position += length;
                        remaining = data_length - position;
                        nrecord += 1;
                    } else {
                        remaining = data_length - position;
                        // copies tail part at beginnning of vect
                        data.copy_within(position.., 0);
                        // clears the last part
                        data.truncate(remaining);
                        break;
                    }
                }
            };
            if remaining == 0 {
                data.clear()
            }
        }
        ChannelData::VariableSizeByteArray(array) => {
            while remaining > 0 {
                let len = &data[position..position + std::mem::size_of::<u32>()];
                let length: usize =
                    u32::from_le_bytes(len.try_into().context("Could not read length")?) as usize;
                if (position + length + 4) <= data_length {
                    position += std::mem::size_of::<u32>();
                    let record = &data[position..position + length];
                    array.push(Some(record));
                    position += length;
                    remaining = data_length - position;
                    nrecord += 1;
                } else {
                    remaining = data_length - position;
                    // copies tail part at beginnning of vect
                    data.copy_within(position.., 0);
                    // clears the last part
                    data.truncate(remaining);
                    break;
                }
            }
            if remaining == 0 {
                data.clear()
            }
        }
        ChannelData::FixedSizeByteArray(_) => {}
        ChannelData::ArrayDInt8(_) => {}
        ChannelData::ArrayDUInt8(_) => {}
        ChannelData::ArrayDInt16(_) => {}
        ChannelData::ArrayDUInt16(_) => {}
        ChannelData::ArrayDInt32(_) => {}
        ChannelData::ArrayDUInt32(_) => {}
        ChannelData::ArrayDFloat32(_) => {}
        ChannelData::ArrayDInt64(_) => {}
        ChannelData::ArrayDUInt64(_) => {}
        ChannelData::ArrayDFloat64(_) => {}
    }
    Ok(nrecord + previous_index)
}

/// Reads all DL Blocks and returns a vect of them
fn parser_ld4(
    rdr: &mut BufReader<&File>,
    mut position: i64,
    channel_group: &mut Cg4,
    channel_names_to_read_in_dg: &HashSet<String>,
) -> Result<i64> {
    let mut ld_blocks: Vec<Ld4Block> = Vec::new();
    let (block, pos) = parser_ld4_block(rdr, position, position)?;
    position = pos;
    ld_blocks.push(block.clone());
    let mut next_ld = block.ld_ld_next();
    while next_ld > 0 {
        rdr.seek_relative(next_ld - position)
            .context("Could not reach LD block position")?;
        position = next_ld;
        let mut id = [0u8; 4];
        rdr.read_exact(&mut id)
            .context("could not read LD block id")?;
        let (block, pos) =
            parser_ld4_block(rdr, position, position).context("failed parsing ld4 block")?;
        position = pos;
        ld_blocks.push(block.clone());
        next_ld = block.ld_ld_next();
    }
    if ld_blocks.len() == 1 && ld_blocks[0].ld_data().len() == 1 && channel_group.cn.len() == 1 {
        // only one DV block, reading can be optimised
        // Reads DV or DZ block id
        let ld_data = ld_blocks[0].ld_data()[0];
        rdr.seek_relative(ld_data - position)
            .context("Could not reach DV or DZ block position from LD block")?;
        let mut id = [0u8; 4];
        rdr.read_exact(&mut id)
            .context("could not read data block id from ld4 invalid")?;
        initialise_arrays(
            channel_group,
            &channel_group.block.cg_cycle_count.clone(),
            channel_names_to_read_in_dg,
        )
        .context("failed initialising arrays")?;
        if id == "##DZ".as_bytes() {
            let (dt, block_header) =
                parse_dz(rdr).context("failed parsing dz block pointed by ld4 block")?;
            for (_rec_pos, cn) in channel_group.cn.iter_mut() {
                let shape = if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(ca) => Some(ca.shape.clone()),
                        Compo::CN(_) => None,
                    }
                } else {
                    None
                };
                read_one_channel_array(&dt, cn, channel_group.block.cg_cycle_count as usize, shape)
                    .context("failed reading one channel array from DZ")?;
            }
            position = ld_data + block_header.len as i64;
        } else {
            let block_header: Dt4Block = rdr.read_le().context("Could not read DV block header")?;
            let mut buf = vec![0u8; block_header.len as usize - 24];
            rdr.read_exact(&mut buf)
                .context("Could not read Dt4 block")?;
            for (_rec_pos, cn) in channel_group.cn.iter_mut() {
                let shape = if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(ca) => Some(ca.shape.clone()),
                        Compo::CN(_) => None,
                    }
                } else {
                    None
                };
                read_one_channel_array(
                    &buf,
                    cn,
                    channel_group.block.cg_cycle_count as usize,
                    shape,
                )
                .context("failed reading one channel array")?;
            }
            position = ld_data + block_header.len as i64;
        }
        if channel_group.block.cg_inval_bytes > 0 {
            // Reads invalid DI or DZ block
            let ld_invalid_data_vec = ld_blocks[0].ld_invalid_data();
            let ld_invalid_data = if !ld_invalid_data_vec.is_empty() {
                ld_invalid_data_vec[0]
            } else {
                bail!("no invalid block (di or dz) pointer found in ld4 block")
            };
            rdr.seek_relative(ld_invalid_data - position)
                .context("Could not reach DI or DZ block position")?;
            let mut id = [0u8; 4];
            rdr.read_exact(&mut id)
                .context("could not read data block id from ld4 invalid")?;
            if id == "##DZ".as_bytes() {
                let (dt, block_header) = parse_dz(rdr)?;
                channel_group.invalid_bytes = Some(dt);
                position = ld_invalid_data + block_header.len as i64;
            } else {
                let block_header: Dt4Block = rdr
                    .read_le()
                    .context("Could not read into DZ or DI block header")?;
                let mut buf = vec![0u8; (block_header.len - 24) as usize];
                rdr.read_exact(&mut buf)
                    .context("Could not read data block")?;
                channel_group.invalid_bytes = Some(buf);
                position = ld_invalid_data + block_header.len as i64;
            }
        }
    } else {
        // several DV, LD or channels per DG
        position = read_dv_di(
            rdr,
            position,
            channel_group,
            ld_blocks,
            channel_names_to_read_in_dg,
        )?;
    }
    Ok(position)
}

/// reads DV and DI block containing several channels
fn read_dv_di(
    rdr: &mut BufReader<&File>,
    mut position: i64,
    channel_group: &mut Cg4,
    ld_blocks: Vec<Ld4Block>,
    channel_names_to_read_in_dg: &HashSet<String>,
) -> Result<i64, Error> {
    let cg_cycle_count = channel_group.block.cg_cycle_count as usize;
    let cg_inval_bytes = channel_group.block.cg_inval_bytes as usize;
    // initialises the arrays
    initialise_arrays(
        channel_group,
        &channel_group.block.cg_cycle_count.clone(),
        channel_names_to_read_in_dg,
    )
    .context("failed initialising arrays for dv di blocks")?;
    for ld in &ld_blocks {
        if !ld.ld_invalid_data().is_empty() {
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
        for data_pointer in ld.ld_data() {
            // Reads DV or DZ block id
            rdr.seek_relative(data_pointer - position)
                .context("Could not reach DV or DZ block position")?;
            let mut id = [0u8; 4];
            rdr.read_exact(&mut id)
                .context("could not read data block id from LD4")?;
            let block_length: usize;
            if id == "##DZ".as_bytes() {
                let (dt, block_header) = parse_dz(rdr)?;
                data.extend(dt);
                block_length = block_header.dz_org_data_length as usize;
                position = data_pointer + block_header.len as i64;
            } else {
                let block_header: Dt4Block =
                    rdr.read_le().context("Could not read DV block structure")?;
                let mut buf = vec![0u8; (block_header.len - 24) as usize];
                rdr.read_exact(&mut buf).context("Could not read DV data")?;
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
                    channel_names_to_read_in_dg,
                    false,
                )
                .context("failed reading channels from dv di blocks")?;
            } else {
                // Some implementation are pre allocating equal length blocks
                read_channels_from_bytes(
                    &data[..record_length * (cg_cycle_count - previous_index)],
                    &mut channel_group.cn,
                    record_length,
                    previous_index,
                    channel_names_to_read_in_dg,
                    false,
                )
                .context("failed reading channels from from dv di blocks")?;
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
        for data_pointer in ld.ld_invalid_data() {
            // Reads DV or DZ block id
            rdr.seek_relative(data_pointer - position)
                .context("Could not reach invalid block position")?;
            let mut id = [0u8; 4];
            rdr.read_exact(&mut id)
                .context("could not read data block id from ld4 invalid")?;
            let block_length: usize;
            if id == "##DZ".as_bytes() {
                let (dt, block_header) = parse_dz(rdr)?;
                invalid_data.extend(dt);
                block_length = block_header.dz_org_data_length as usize;
                position = data_pointer + block_header.len as i64;
            } else {
                let block_header: Dt4Block = rdr
                    .read_le()
                    .context("Could not read invalid block header")?;
                let mut buf = vec![0u8; (block_header.len - 24) as usize];
                rdr.read_exact(&mut buf)
                    .context("Could not read invalid data")?;
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
    Ok(position)
}

/// Reads all DL Blocks and returns a vect of them
fn parser_dl4(rdr: &mut BufReader<&File>, mut position: i64) -> Result<(Vec<Dl4Block>, i64)> {
    let mut dl_blocks: Vec<Dl4Block> = Vec::new();
    let (block, pos) = parser_dl4_block(rdr, position, position)?;
    position = pos;
    dl_blocks.push(block.clone());
    let mut next_dl = block.dl_dl_next;
    while next_dl > 0 {
        rdr.seek_relative(next_dl - position)
            .context("Could not reach DL4 block position")?;
        position = next_dl;
        let mut id = [0u8; 4];
        rdr.read_exact(&mut id)
            .context("could not read DL block id")?;
        let (block, pos) = parser_dl4_block(rdr, position, position)?;
        position = pos;
        dl_blocks.push(block.clone());
        next_dl = block.dl_dl_next;
    }
    Ok((dl_blocks, position))
}

/// Reads all sorted data blocks pointed by DL4 Blocks
fn parser_dl4_sorted(
    rdr: &mut BufReader<&File>,
    dl_blocks: Vec<Dl4Block>,
    mut position: i64,
    channel_group: &mut Cg4,
    decoder: &mut Dec,
    rec_pos: &i32,
    channel_names_to_read_in_dg: &HashSet<String>,
) -> Result<(i64, Vec<i32>)> {
    // initialises the arrays
    initialise_arrays(
        channel_group,
        &channel_group.block.cg_cycle_count.clone(),
        channel_names_to_read_in_dg,
    )
    .context("failed initialising arrays for sorted dl4 block")?;
    // Read all data blocks
    let mut data: Vec<u8> = Vec::new();
    let mut previous_index: usize = 0;
    let cg_cycle_count = channel_group.block.cg_cycle_count as usize;
    let record_length = channel_group.record_length as usize;
    let mut vlsd_channels: Vec<i32> = Vec::new();
    for dl in dl_blocks {
        for data_pointer in dl.dl_data {
            // Reads DT or DZ block id
            rdr.seek_relative(data_pointer - position)
                .context("Could not reach DV or DZ block position from DL4")?;
            let mut id = [0u8; 4];
            rdr.read_exact(&mut id)
                .context("could not read data block id")?;
            let block_length: usize;
            if id == "##DZ".as_bytes() {
                let (dt, block_header) = parse_dz(rdr)?;
                data.extend(dt);
                block_length = block_header.dz_org_data_length as usize;
                position = data_pointer + block_header.len as i64;
                id[2..].copy_from_slice(&block_header.dz_org_block_type[..]);
            } else {
                let block_header: Dt4Block = rdr.read_le().context("Could not DT block header")?;
                let mut buf = vec![0u8; (block_header.len - 24) as usize];
                rdr.read_exact(&mut buf)
                    .context("Could not read DT block data")?;
                data.extend(buf);
                block_length = (block_header.len - 24) as usize;
                position = data_pointer + block_header.len as i64;
            }
            // Copies full sized records in block into channels arrays

            if id == "##SD".as_bytes() {
                if let Some(cn) = channel_group.cn.get_mut(rec_pos) {
                    previous_index = read_vlsd_from_bytes(&mut data, cn, previous_index, decoder)?;
                }
            } else {
                let n_record_chunk = block_length / record_length;
                if previous_index + n_record_chunk < cg_cycle_count {
                    vlsd_channels = read_channels_from_bytes(
                        &data[..record_length * n_record_chunk],
                        &mut channel_group.cn,
                        record_length,
                        previous_index,
                        channel_names_to_read_in_dg,
                        true,
                    )
                    .context("could not read channels from bytes")?;
                } else {
                    // Some implementation are pre allocating equal length blocks
                    vlsd_channels = read_channels_from_bytes(
                        &data[..record_length * (cg_cycle_count - previous_index)],
                        &mut channel_group.cn,
                        record_length,
                        previous_index,
                        channel_names_to_read_in_dg,
                        true,
                    )
                    .context("could not read channels from bytes")?;
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
    }
    Ok((position, vlsd_channels))
}

/// Reads all unsorted data blocks pointed by DL4 Blocks
fn parser_dl4_unsorted(
    rdr: &mut BufReader<&File>,
    dg: &mut Dg4,
    dl_blocks: Vec<Dl4Block>,
    mut position: i64,
    channel_names_to_read_in_dg: &HashSet<String>,
) -> Result<i64> {
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
            rdr.seek_relative(data_pointer - position)
                .context("Could not reach DT or DZ position from DL")?;
            let mut buf = [0u8; 24];
            rdr.read_exact(&mut buf)
                .context("could not read blockheader4 Id")?;
            let mut block = Cursor::new(buf);
            let header: Blockheader4 = block.read_le().context("could not parse blockheader4")?;
            if header.hdr_id == "##DZ".as_bytes() {
                let (dt, _block) = parse_dz(rdr)?;
                data.extend(dt);
            } else {
                let mut buf = vec![0u8; (header.hdr_len - 24) as usize];
                rdr.read_exact(&mut buf)
                    .context("Could not read DT block data")?;
                data.extend(buf);
            }
            // saves records as much as possible
            read_all_channels_unsorted_from_bytes(
                &mut data,
                dg,
                &mut record_counter,
                &mut decoder,
                channel_names_to_read_in_dg,
            )?;
            position = data_pointer + header.hdr_len as i64;
        }
    }
    Ok(position)
}

/// Returns chunk size and corresponding number of records from a channel group
fn generate_chunks(channel_group: &Cg4) -> Vec<(usize, usize)> {
    let record_length = channel_group.record_length as usize;
    let cg_cycle_count = channel_group.block.cg_cycle_count as usize;
    let n_chunks = (record_length * cg_cycle_count) / CHUNK_SIZE_READING_4 + 1; // number of chunks
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
    channel_group: &mut Cg4,
    channel_names_to_read_in_dg: &HashSet<String>,
) -> Result<Vec<i32>> {
    let chunks = generate_chunks(channel_group);
    // initialises the arrays
    initialise_arrays(
        channel_group,
        &channel_group.block.cg_cycle_count.clone(),
        channel_names_to_read_in_dg,
    )
    .context("failed initilising arrays")?;
    // read by chunks and store in channel array
    let mut previous_index: usize = 0;
    let mut vlsd_channels: Vec<i32> = Vec::new();
    for (n_record_chunk, chunk_size) in chunks {
        let mut data_chunk = vec![0u8; chunk_size];
        rdr.read_exact(&mut data_chunk)
            .context("Could not read data chunk")?;
        vlsd_channels = read_channels_from_bytes(
            &data_chunk,
            &mut channel_group.cn,
            channel_group.record_length as usize,
            previous_index,
            channel_names_to_read_in_dg,
            true,
        )
        .context("could not read channels from bytes")?;
        previous_index += n_record_chunk;
    }
    Ok(vlsd_channels)
}

/// copies complete sorted data block (not chunk) into each channel array
fn read_all_channels_sorted_from_bytes(
    data: &[u8],
    channel_group: &mut Cg4,
    channel_names_to_read_in_dg: &HashSet<String>,
) -> Result<Vec<i32>> {
    // initialises the arrays
    initialise_arrays(
        channel_group,
        &channel_group.block.cg_cycle_count.clone(),
        channel_names_to_read_in_dg,
    )
    .context("failed initilising arrays")?;
    let vlsd_channels: Vec<i32> = read_channels_from_bytes(
        data,
        &mut channel_group.cn,
        channel_group.record_length as usize,
        0,
        channel_names_to_read_in_dg,
        true,
    )
    .context("failed initilising arrays")?;
    Ok(vlsd_channels)
}

/// Reads unsorted data block chunk by chunk
fn read_all_channels_unsorted(
    rdr: &mut BufReader<&File>,
    dg: &mut Dg4,
    block_length: i64,
    channel_names_to_read_in_dg: &HashSet<String>,
) -> Result<()> {
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
    let mut data: Vec<u8> = Vec::new();
    let mut data_chunk: Vec<u8>;
    while position < data_block_length {
        if (data_block_length - position) > CHUNK_SIZE_READING_4 {
            // not last chunk of data
            data_chunk = vec![0u8; CHUNK_SIZE_READING_4];
            position += CHUNK_SIZE_READING_4;
        } else {
            // last chunk of data
            data_chunk = vec![0u8; data_block_length - position];
            position += data_block_length - position;
        }
        rdr.read_exact(&mut data_chunk)
            .context("Could not read data chunk")?;
        data.extend(data_chunk);
        read_all_channels_unsorted_from_bytes(
            &mut data,
            dg,
            &mut record_counter,
            &mut decoder,
            channel_names_to_read_in_dg,
        )?;
    }
    Ok(())
}

/// read record by record from unsorted data block into sorted data block, then copy data into channel arrays
fn read_all_channels_unsorted_from_bytes(
    data: &mut Vec<u8>,
    dg: &mut Dg4,
    record_counter: &mut HashMap<u64, (usize, Vec<u8>)>,
    decoder: &mut Dec,
    channel_names_to_read_in_dg: &HashSet<String>,
) -> Result<(), Error> {
    let mut position: usize = 0;
    let data_length = data.len();
    let dg_rec_id_size = dg.block.dg_rec_id_size as usize;
    let vlsd_data_start_offset = dg_rec_id_size + std::mem::size_of::<u32>();
    // unsort data into sorted data blocks, except for VLSD CG.
    let mut remaining: usize = data_length - position;
    while remaining > 0 {
        // reads record id
        let rec_id: u64;
        if dg_rec_id_size == 1 && remaining >= 1 {
            rec_id = data[position].into();
        } else if dg_rec_id_size == 2 && remaining >= 2 {
            let rec = &data[position..position + std::mem::size_of::<u16>()];
            rec_id = u16::from_le_bytes(rec.try_into().context("Could not convert record id u16")?)
                as u64;
        } else if dg_rec_id_size == 4 && remaining >= 4 {
            let rec = &data[position..position + std::mem::size_of::<u32>()];
            rec_id = u32::from_le_bytes(rec.try_into().context("Could not convert record id u32")?)
                as u64;
        } else if dg_rec_id_size == 8 && remaining >= 8 {
            let rec = &data[position..position + std::mem::size_of::<u64>()];
            rec_id = u64::from_le_bytes(rec.try_into().context("Could not convert record id u64")?);
        } else {
            break; // not enough data remaining
        }
        // reads record based on record id
        if let Some(cg) = dg.cg.get_mut(&rec_id) {
            let record_length = cg.record_length as usize;
            if (cg.block.cg_flags & 0b1) != 0 {
                // VLSD channel
                if remaining >= 4 + dg_rec_id_size {
                    let len = &data[position + dg_rec_id_size..position + vlsd_data_start_offset];
                    let length: usize =
                        u32::from_le_bytes(len.try_into().context("Could not read length")?)
                            as usize;
                    remaining = data_length - position - vlsd_data_start_offset;
                    if remaining >= length {
                        position += vlsd_data_start_offset;
                        let record = &data[position..position + length];
                        if let Some((target_rec_id, target_rec_pos)) = cg.vlsd_cg {
                            if let Some(target_cg) = dg.cg.get_mut(&target_rec_id) {
                                if let Some(target_cn) = target_cg.cn.get_mut(&target_rec_pos) {
                                    if let Some((nrecord, _)) = record_counter.get_mut(&rec_id) {
                                        match &mut target_cn.data {
                                            ChannelData::Utf8(array) => {
                                                let mut dst = String::with_capacity(record.len());
                                                if target_cn.block.cn_data_type == 6 {
                                                    let (_result, _size, _replacement) = decoder
                                                        .windows_1252
                                                        .decode_to_string(record, &mut dst, false);
                                                } else if target_cn.block.cn_data_type == 7 {
                                                    dst = str::from_utf8(record)
                                                        .context(
                                                            "Found invalid UTF-8 from VLSD record",
                                                        )?
                                                        .to_string();
                                                } else if target_cn.block.cn_data_type == 8 {
                                                    let (_result, _size, _replacement) = decoder
                                                        .utf_16_le
                                                        .decode_to_string(record, &mut dst, false);
                                                } else if target_cn.block.cn_data_type == 9 {
                                                    let (_result, _size, _replacement) = decoder
                                                        .utf_16_be
                                                        .decode_to_string(record, &mut dst, false);
                                                } else {
                                                    bail!("channel data type is not correct for a text")
                                                };
                                                dst = dst.trim_end_matches('\0').to_owned();
                                                array.push(Some(dst));
                                            }
                                            ChannelData::VariableSizeByteArray(array) => {
                                                array.push(Some(record));
                                            }
                                            _ => {
                                                bail!("data type of VLSD is not possible");
                                            }
                                        }
                                        *nrecord += 1;
                                    } else {
                                        bail!("could not find the record id");
                                    }
                                } else {
                                    bail!("could not find the target record position");
                                }
                            } else {
                                bail!("could not find the target record id");
                            }
                        } else {
                            bail!("no vsld in CG, wrong cg_flags");
                        }
                        position += length;
                    } else {
                        break; // not enough data remaining
                    }
                } else {
                    break; // not enough data remaining
                }
            } else if remaining >= record_length {
                // Not VLSD channel
                let record = &data[position..position + cg.record_length as usize];
                if let Some((_nrecord, data)) = record_counter.get_mut(&rec_id) {
                    data.extend(record);
                } else {
                    bail!("could not find the record id");
                }
                position += record_length;
            } else {
                break; // not enough data remaining
            }
        } else {
            bail!("could not find the record id");
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
                record_data,
                &mut channel_group.cn,
                channel_group.record_length as usize,
                *index,
                channel_names_to_read_in_dg,
                true,
            )
            .context("failed reading channels from bytes after reading unsorted data")?;
            record_data.clear(); // clears data for new block, keeping capacity
        }
    }
    Ok(())
}

/// decoder for String SBC and UTF16 Le & Be
struct Dec {
    windows_1252: Decoder,
    utf_16_be: Decoder,
    utf_16_le: Decoder,
}

/// initialise ndarrays for the data group/block
fn initialise_arrays(
    channel_group: &mut Cg4,
    cg_cycle_count: &u64,
    channel_names_to_read_in_dg: &HashSet<String>,
) -> Result<(), Error> {
    // creates zeroed array in parallel for each channel contained in channel group
    channel_group
        .cn
        .par_iter_mut()
        .filter(|(_cn_record_position, cn)| channel_names_to_read_in_dg.contains(&cn.unique_name))
        .try_for_each(
            |(_cn_record_position, cn): (&i32, &mut Cn4)| -> Result<(), Error> {
                let mut shape = (Vec::new(), Order::RowMajor);
                if let Some(compo) = &cn.composition {
                    match &compo.block {
                        Compo::CA(ca) => shape = ca.shape.clone(),
                        Compo::CN(_) => (),
                    }
                }
                cn.data = cn
                    .data
                    .zeros(cn.block.cn_type, *cg_cycle_count, cn.n_bytes, shape)
                    .with_context(|| {
                        format!("Zeros initialisation of channel {} failed", cn.unique_name)
                    })?;
                Ok(())
            },
        )
        .with_context(|| {
            format!(
                "Zeros initialisation of channel group with master {:?} failed",
                channel_group.master_channel_name
            )
        })?;
    Ok(())
}

/// applies bit mask if required in channel block
fn apply_bit_mask_offset(
    dg: &mut Dg4,
    channel_names_to_read_in_dg: &HashSet<String>,
) -> Result<(), Error> {
    // apply bit shift and masking
    for channel_group in dg.cg.values_mut() {
        channel_group
            .cn
            .par_iter_mut()
            .filter(|(_cn_record_position, cn)| {
                channel_names_to_read_in_dg.contains(&cn.unique_name)
            })
            .try_for_each(|(_rec_pos, cn): (&i32, &mut Cn4)| -> Result<(), Error> {
                if cn.block.cn_data_type <= 3 {
                    let left_shift =
                        cn.n_bytes * 8 - (cn.block.cn_bit_offset as u32) - cn.block.cn_bit_count;
                    let right_shift = left_shift + (cn.block.cn_bit_offset as u32);
                    if left_shift > 0 || right_shift > 0 {
                        match &mut cn.data {
                            ChannelData::Int8(array) => {
                                let a = array
                                    .get_mut_values()
                                    .context(
                                        "bit mask offset application could not get mutable values for primitive array i8",
                                    )?;
                                if left_shift > 0 {
                                    a.iter_mut().for_each(|x| *x <<= left_shift)
                                };
                                if right_shift > 0 {
                                    a.iter_mut().for_each(|x| *x >>= right_shift)
                                };
                            }
                            ChannelData::UInt8(array) => {
                                let a = array
                                    .get_mut_values()
                                    .context(
                                        "bit mask offset application could not get mutable values for primitive array u8",
                                    )?;
                                if left_shift > 0 {
                                    a.iter_mut().for_each(|x| *x <<= left_shift)
                                };
                                if right_shift > 0 {
                                    a.iter_mut().for_each(|x| *x >>= right_shift)
                                };
                            }
                            ChannelData::Int16(array) => {
                                let a = array
                                    .get_mut_values()
                                    .context(
                                        "bit mask offset application could not get mutable values for primitive array i16",
                                    )?;
                                if left_shift > 0 {
                                    a.iter_mut().for_each(|x| *x <<= left_shift)
                                };
                                if right_shift > 0 {
                                    a.iter_mut().for_each(|x| *x >>= right_shift)
                                };
                            }
                            ChannelData::UInt16(array) => {
                                let a = array
                                    .get_mut_values()
                                    .context(
                                        "bit mask offset application could not get mutable values for primitive array u16",
                                    )?;
                                if left_shift > 0 {
                                    a.iter_mut().for_each(|x| *x <<= left_shift)
                                };
                                if right_shift > 0 {
                                    a.iter_mut().for_each(|x| *x >>= right_shift)
                                };
                            }
                            ChannelData::Int32(array) => {
                                let a = array
                                    .get_mut_values()
                                    .context(
                                        "bit mask offset application could not get mutable values for primitive array i32",
                                    )?;
                                if left_shift > 0 {
                                    a.iter_mut().for_each(|x| *x <<= left_shift)
                                };
                                if right_shift > 0 {
                                    a.iter_mut().for_each(|x| *x >>= right_shift)
                                };
                            }
                            ChannelData::UInt32(array) => {
                                let a = array
                                    .get_mut_values()
                                    .context(
                                        "bit mask offset application could not get mutable values for primitive array u32",
                                    )?;
                                if left_shift > 0 {
                                    a.iter_mut().for_each(|x| *x <<= left_shift)
                                };
                                if right_shift > 0 {
                                    a.iter_mut().for_each(|x| *x >>= right_shift)
                                };
                            }
                            ChannelData::Float32(_) => (),
                            ChannelData::Int64(array) => {
                                let a = array
                                    .get_mut_values()
                                    .context(
                                        "bit mask offset application could not get mutable values for primitive array i64",
                                    )?;
                                let left_shift =
                                    64 - (cn.block.cn_bit_offset as u32) - cn.block.cn_bit_count;
                                let right_shift = left_shift + (cn.block.cn_bit_offset as u32);
                                if left_shift > 0 {
                                    a.iter_mut().for_each(|x| *x <<= left_shift)
                                };
                                if right_shift > 0 {
                                    a.iter_mut().for_each(|x| *x >>= right_shift)
                                };
                            }
                            ChannelData::UInt64(array) => {
                                let a = array
                                    .get_mut_values()
                                    .context(
                                        "bit mask offset application could not get mutable values for primitive array u64",
                                    )?;
                                let left_shift =
                                    64 - (cn.block.cn_bit_offset as u32) - cn.block.cn_bit_count;
                                let right_shift = left_shift + (cn.block.cn_bit_offset as u32);
                                if left_shift > 0 {
                                    a.iter_mut().for_each(|x| *x <<= left_shift)
                                };
                                if right_shift > 0 {
                                    a.iter_mut().for_each(|x| *x >>= right_shift)
                                };
                            }
                            ChannelData::Float64(_) => (),
                            ChannelData::Complex32(_) => (),
                            ChannelData::Complex64(_) => (),
                            ChannelData::Utf8(_) => (),
                            ChannelData::VariableSizeByteArray(_) => (),
                            ChannelData::FixedSizeByteArray(_) => (),
                            ChannelData::ArrayDInt8(array) => {
                                let a = array
                                    .get_mut_values()
                                    .context(
                                        "bit mask offset application could not get mutable values for tensor i8",
                                    )?;
                                if left_shift > 0 {
                                    a.iter_mut().for_each(|x| *x <<= left_shift)
                                };
                                if right_shift > 0 {
                                    a.iter_mut().for_each(|x| *x >>= right_shift)
                                };
                            }
                            ChannelData::ArrayDUInt8(array) => {
                                let a = array
                                    .get_mut_values()
                                    .context(
                                        "bit mask offset application could not get mutable values for tensor u8",
                                    )?;
                                if left_shift > 0 {
                                    a.iter_mut().for_each(|x| *x <<= left_shift)
                                };
                                if right_shift > 0 {
                                    a.iter_mut().for_each(|x| *x >>= right_shift)
                                };
                            }
                            ChannelData::ArrayDInt16(array) => {
                                let a = array
                                    .get_mut_values()
                                    .context(
                                        "bit mask offset application could not get mutable values for tensor i16",
                                    )?;
                                if left_shift > 0 {
                                    a.iter_mut().for_each(|x| *x <<= left_shift)
                                };
                                if right_shift > 0 {
                                    a.iter_mut().for_each(|x| *x >>= right_shift)
                                };
                            }
                            ChannelData::ArrayDUInt16(array) => {
                                let a = array
                                    .get_mut_values()
                                    .context(
                                        "bit mask offset application could not get mutable values for tensor u16",
                                    )?;
                                if left_shift > 0 {
                                    a.iter_mut().for_each(|x| *x <<= left_shift)
                                };
                                if right_shift > 0 {
                                    a.iter_mut().for_each(|x| *x >>= right_shift)
                                };
                            }
                            ChannelData::ArrayDInt32(array) => {
                                let a = array
                                    .get_mut_values()
                                    .context(
                                        "bit mask offset application could not get mutable values for tensor i32",
                                    )?;
                                if left_shift > 0 {
                                    a.iter_mut().for_each(|x| *x <<= left_shift)
                                };
                                if right_shift > 0 {
                                    a.iter_mut().for_each(|x| *x >>= right_shift)
                                };
                            }
                            ChannelData::ArrayDUInt32(array) => {
                                let a = array
                                    .get_mut_values()
                                    .context(
                                        "bit mask offset application could not get mutable values for tensor u32",
                                    )?;
                                if left_shift > 0 {
                                    a.iter_mut().for_each(|x| *x <<= left_shift)
                                };
                                if right_shift > 0 {
                                    a.iter_mut().for_each(|x| *x >>= right_shift)
                                };
                            }
                            ChannelData::ArrayDFloat32(_) => (),
                            ChannelData::ArrayDInt64(array) => {
                                let a = array
                                    .get_mut_values()
                                    .context(
                                        "bit mask offset application could not get mutable values for tensor i64",
                                    )?;
                                let left_shift =
                                    64 - (cn.block.cn_bit_offset as u32) - cn.block.cn_bit_count;
                                let right_shift = left_shift + (cn.block.cn_bit_offset as u32);
                                if left_shift > 0 {
                                    a.iter_mut().for_each(|x| *x <<= left_shift)
                                };
                                if right_shift > 0 {
                                    a.iter_mut().for_each(|x| *x >>= right_shift)
                                };
                            }
                            ChannelData::ArrayDUInt64(array) => {
                                let a = array
                                    .get_mut_values()
                                    .context(
                                        "bit mask offset application could not get mutable values for tensor u64",
                                    )?;
                                let left_shift =
                                    64 - (cn.block.cn_bit_offset as u32) - cn.block.cn_bit_count;
                                let right_shift = left_shift + (cn.block.cn_bit_offset as u32);
                                if left_shift > 0 {
                                    a.iter_mut().for_each(|x| *x <<= left_shift)
                                };
                                if right_shift > 0 {
                                    a.iter_mut().for_each(|x| *x >>= right_shift)
                                };
                            }
                            ChannelData::ArrayDFloat64(_) => (),
                        }
                    }
                }
                Ok(())
            }).with_context(|| {
                format!(
                    "bit mask application failed for channel group {:?}",
                    channel_group
                )
            })?
    }
    Ok(())
}
