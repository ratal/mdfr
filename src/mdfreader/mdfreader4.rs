//! data read and load in memory based in MdfInfo4's metadata
use crate::data_holder::channel_data::ChannelData;
use crate::mdfinfo::MdfInfo;
use crate::mdfinfo::mdfinfo4::{Blockheader4, Cg4, Cn4, Composition, Dg4, Ds4Block};
use crate::mdfinfo::mdfinfo4::{
    CG_F_VLSC, CG_F_VLSD, Dl4Block, Dt4Block, Dz4Block, Gd4Block, Hl4Block, Ld4Block, parse_dz,
    parser_dl4_block, parser_ld4_block, read_dz_raw,
};
use crate::mdfreader::data_read4::read_channels_from_bytes;
use crate::mdfreader::data_read4::read_one_channel_array;
use crate::mdfreader::datastream_decoder;
use anyhow::{Context, Error, Result, bail};
use binrw::BinReaderExt;
use encoding_rs::{Decoder, GB18030, UTF_8, UTF_16BE, UTF_16LE, WINDOWS_1252};
use log::warn;
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, Cursor, Read};
use std::str;
use unicode_bom::Bom;

use super::Mdf;

/// The following constant represents the size of data chunk to be read and processed.
/// a big chunk will improve performance but consume more memory
/// a small chunk will not consume too much memory but will cause many read calls, penalising performance
pub const CHUNK_SIZE_READING_4: usize = 4_194_304; // can be tuned according to architecture

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
            let mut decoder: Dec = Dec {
                windows_1252: WINDOWS_1252.new_decoder(),
                utf_16_be: UTF_16BE.new_decoder(),
                utf_16_le: UTF_16LE.new_decoder(),
            };
            for dg in info.dg.values_mut() {
                for cg in dg.cg.values_mut() {
                    for cn in cg.cn.values_mut() {
                        cn.should_read = channel_names.contains(&cn.unique_name);
                    }
                }
            }
            // read file data
            for dg in info.dg.values_mut() {
                let has_readable = dg
                    .cg
                    .values()
                    .any(|cg| cg.cn.values().any(|cn| cn.should_read));
                if dg.block.dg_data != 0 && has_readable {
                    // header block
                    rdr.seek_relative(dg.block.dg_data - position)
                        .context("Could not position buffer")?; // change buffer position
                    let mut id = [0u8; 4];
                    rdr.read_exact(&mut id).context("could not read block id")?;
                    sorted = dg.cg.len() == 1;
                    position = read_data(rdr, id, dg, dg.block.dg_data, sorted, &mut decoder)
                        .with_context(|| format!("failed reading data for dg {dg:?}"))?;
                    apply_bit_mask_offset(dg).context("failed applying bit mask offset")?;
                    // channel_group invalid bits calculation (only for DIBlocks)
                    for channel_group in dg.cg.values_mut() {
                        channel_group
                            .process_all_channel_invalid_bits()
                            .context("failed processing all channel invalid bits")?;
                        // Process Channel Variants (CV) - merge option data based on discriminator
                        channel_group
                            .process_channel_variants()
                            .context("failed processing channel variants")?;
                        // Process Channel Unions (CU) - create UnionArray from member channels
                        channel_group
                            .process_channel_unions()
                            .context("failed processing channel unions")?;
                    }
                    // Defer conversion of channels to physical values (handled lazily on access)
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
    decoder: &mut Dec,
) -> Result<i64> {
    // block header is already read
    let mut vlsd_channels: Vec<(u8, i32)> = Vec::new();
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
                    vlsd_channels = read_all_channels_sorted(rdr, channel_group)
                        .context("failed reading all channels sorted")?;
                    position += block_header.len as i64;
                }
                position = process_dynamic_channels(rdr, dg, &vlsd_channels, position, decoder)?;
            } else if !dg.cg.is_empty() {
                // unsorted data
                // initialises all arrays
                for channel_group in dg.cg.values_mut() {
                    initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone())
                        .context("failed intialising arrays")?;
                }
                read_all_channels_unsorted(rdr, dg, block_header.len as i64)
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
                    vlsd_channels = read_all_channels_sorted_from_bytes(&data, channel_group)
                        .context("failed reading all channels sorted from bytes")?;
                }
                position += block_header.len as i64;
                position = process_dynamic_channels(rdr, dg, &vlsd_channels, position, decoder)?;
            } else if !dg.cg.is_empty() {
                // unsorted data
                // initialises all arrays
                for channel_group in dg.cg.values_mut() {
                    initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone())
                        .context("failed intialising arrays")?;
                }
                // initialise record counter
                let mut unsorted_state = UnsortedState::new(dg);
                read_all_channels_unsorted_from_bytes(
                    &mut data,
                    dg,
                    &mut unsorted_state.buffers,
                    decoder,
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
            position = read_data(rdr, id, dg, position, sorted, decoder)
                .context("failed reading data from HL block")?;
        }
        [35, 35, 68, 76] => {
            // ##DL
            // data list
            if sorted {
                // sorted data group
                for channel_group in dg.cg.values_mut() {
                    let (dl_blocks, pos) = parser_dl4(rdr, position)?;
                    let (pos, vlsd) =
                        parser_dl4_sorted(rdr, dl_blocks, pos, channel_group, decoder, &0i32)
                            .context("failed parsing DL4 sorted")?;
                    position = pos;
                    vlsd_channels = vlsd;
                }
                position = process_dynamic_channels(rdr, dg, &vlsd_channels, position, decoder)?;
            } else if !dg.cg.is_empty() {
                // unsorted data
                // initialises all arrays
                for channel_group in dg.cg.values_mut() {
                    initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone())
                        .context("failed intialising arrays")?;
                }
                let (dl_blocks, pos) = parser_dl4(rdr, position)?;
                let pos = parser_dl4_unsorted(rdr, dg, dl_blocks, pos)
                    .context("failed parsing DL4 block unsorted")?;
                position = pos;
            }
        }
        [35, 35, 76, 68] => {
            // ##LD
            // list data, cannot be used for unsorted data
            for channel_group in dg.cg.values_mut() {
                let pos =
                    parser_ld4(rdr, position, channel_group).context("failed parsing DL4 block")?;
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
                read_all_channels_sorted(rdr, channel_group)
                    .context("failed reading all channels sorted")?;
            }
            position += block_header.len as i64;
        }
        [35, 35, 68, 71] => {
            // ##DG
            bail!("Weird, a DG block type {id:?}") // should never happen
        }
        [35, 35, 71, 68] => {
            // ##GD - Guard Block (MDF 4.3)
            // GDBLOCK is used to safeguard newly introduced features against older readers
            let block: Gd4Block = rdr
                .read_le()
                .context("could not read into Gd4Block structure")?;

            // Gd4Block struct size (without id): 4 + 8 + 8 + 8 + 2 = 30 bytes
            // After reading: reader at position + 4 (id) + 30 (struct) = position + 34
            const GD4_STRUCT_SIZE: i64 = 30;
            let current_pos = position + 4 + GD4_STRUCT_SIZE;

            // Check if we support the required MDF version (430 for MDF 4.3.0)
            if block.gd_version > 430 {
                warn!(
                    "GDBLOCK requires MDF version {} which is not supported (max 430). Skipping guarded data.",
                    block.gd_version
                );
                position += block.gd_len as i64;
            } else {
                // Follow the guarded link to the actual data block
                rdr.seek_relative(block.gd_link - current_pos)
                    .context("Could not reach guarded block from GDBLOCK")?;
                position = block.gd_link;

                // Read the id of the guarded block
                let mut guarded_id = [0u8; 4];
                rdr.read_exact(&mut guarded_id)
                    .context("could not read guarded block id")?;

                // Recursively read the guarded data block
                position = read_data(rdr, guarded_id, dg, position, sorted, decoder)
                    .context("failed reading guarded data block from GDBLOCK")?;
            }
        }
        _ => bail!("Unknown data block type {id:?}"), // should never happen
    }
    Ok(position)
}

/// Reads and concatenates data from any data block type (##DT, ##SD, ##VD, ##RD, ##RV, ##DZ, ##DL, ##HL).
/// The block id must already have been read. Returns concatenated raw bytes and updated position.
fn read_all_blocks_to_bytes(
    rdr: &mut BufReader<&File>,
    id: [u8; 4],
    mut position: i64,
) -> Result<Option<(Vec<u8>, i64)>> {
    // ##DT, ##SD, ##VD are regular data blocks; ##RD, ##RV are reduction data blocks (same format)
    if id == *b"##DT" || id == *b"##SD" || id == *b"##VD" || id == *b"##RD" || id == *b"##RV" {
        let block_header: Dt4Block = rdr.read_le().context("Could not read data block header")?;
        let mut buf = vec![0u8; block_header.len as usize - 24];
        rdr.read_exact(&mut buf)
            .context("could not read data block buffer")?;
        position += block_header.len as i64;
        Ok(Some((buf, position)))
    } else if id == *b"##DZ" {
        let (buf, block_header) = parse_dz(rdr)?;
        position += block_header.len as i64;
        Ok(Some((buf, position)))
    } else if id == *b"##HL" || id == *b"##DL" {
        let current_pos = if id == *b"##HL" {
            let (pos, _id) = read_hl(rdr, position)?;
            pos
        } else {
            position
        };
        let (dl_blocks, mut pos) = parser_dl4(rdr, current_pos)?;
        let mut combined_data = Vec::with_capacity(CHUNK_SIZE_READING_4 * 2);
        for dl in dl_blocks {
            for data_ptr in dl.dl_data {
                if data_ptr == 0 {
                    continue;
                }
                rdr.seek_relative(data_ptr - pos)?;
                pos = data_ptr;
                let mut inner_id = [0u8; 4];
                rdr.read_exact(&mut inner_id)?;
                if inner_id == *b"##DZ" {
                    let (buf, header) = parse_dz(rdr)?;
                    pos += header.len as i64;
                    combined_data.extend(buf);
                } else {
                    // ##DT, ##SD, ##VD or any other raw data block
                    let header: Dt4Block = rdr.read_le()?;
                    let mut buf = vec![0u8; header.len as usize - 24];
                    rdr.read_exact(&mut buf)?;
                    pos += header.len as i64;
                    combined_data.extend(buf);
                }
            }
        }
        Ok(Some((combined_data, pos)))
    } else {
        Ok(None)
    }
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

/// Reads VLSD data from a chain of DL sub-blocks (##SD or ##DZ) without reinitialising arrays.
/// Used by read_sd to process DL-chained VLSD data for a single channel.
fn read_vlsd_from_dl_blocks(
    rdr: &mut BufReader<&File>,
    dl_blocks: Vec<Dl4Block>,
    mut position: i64,
    cn: &mut Cn4,
    decoder: &mut Dec,
) -> Result<i64> {
    let mut previous_index: usize = 0;
    for dl in dl_blocks {
        for data_pointer in dl.dl_data {
            rdr.seek_relative(data_pointer - position)
                .context("Could not reach VLSD sub-block from DL")?;
            let mut id = [0u8; 4];
            rdr.read_exact(&mut id)
                .context("could not read VLSD sub-block id")?;
            let mut data = if id == *b"##DZ" {
                let (dt, block_header) = parse_dz(rdr)?;
                position = data_pointer + block_header.len as i64;
                dt
            } else {
                // ##SD block (same header layout as Dt4Block)
                let block_header: Dt4Block = rdr
                    .read_le()
                    .context("Could not read VLSD sub-block header")?;
                let mut buf = vec![0u8; (block_header.len - 24) as usize];
                rdr.read_exact(&mut buf)
                    .context("Could not read VLSD sub-block data")?;
                position = data_pointer + block_header.len as i64;
                buf
            };
            previous_index = read_vlsd_from_bytes(&mut data, cn, previous_index, decoder)?;
        }
    }
    Ok(position)
}

/// Reads Signal Data Block containing VLSD channel, pointed by cn_data
fn read_sd(
    rdr: &mut BufReader<&File>,
    dg: &mut Dg4,
    vlsd_channels: &[(u8, i32)],
    mut position: i64,
    decoder: &mut Dec,
) -> Result<i64> {
    for channel_group in dg.cg.values_mut() {
        for (cn_type, rec_pos) in vlsd_channels {
            // Only process VLSD channels (cn_type == 1)
            if *cn_type != 1 {
                continue;
            }
            if let Some(cn) = channel_group.cn.get_mut(rec_pos) {
                // header block
                rdr.seek_relative(cn.block.cn_data - position)
                    .context("Could not position buffer")?; // change buffer position
                position = cn.block.cn_data;
                let mut id = [0u8; 4];
                rdr.read_exact(&mut id).context("could not read block id")?;
                if "##SD".as_bytes() == id {
                    // SD (Signal Data) block - each value prefixed with u32 length
                    let block_header: Dt4Block =
                        rdr.read_le().context("Could not read SD block struct")?;
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
                    position = read_vlsd_from_dl_blocks(rdr, dl_blocks, pos, cn, decoder)?;
                } else if "##DL".as_bytes() == id {
                    let (dl_blocks, pos) = parser_dl4(rdr, position)?;
                    position = read_vlsd_from_dl_blocks(rdr, dl_blocks, pos, cn, decoder)?;
                }
            }
        }
    }
    Ok(position)
}

/// Reads Variable Data Block containing VLSC channel data (cn_type = 7)
/// The DT/DZ block has already been read, so:
/// - offsets are in the VLSC channel cn.data (read as UInt64)
/// - sizes are in the cn_cn_size channel cn.data (read as UInt8/16/32/64)
/// - VD block (pointed by cn.block.cn_data) contains raw data without length prefixes
fn read_vd(
    rdr: &mut BufReader<&File>,
    channel_group: &mut Cg4,
    vlsc_channels: &[(u8, i32)],
    mut position: i64,
    decoder: &mut Dec,
) -> Result<i64> {
    for (cn_type, rec_pos) in vlsc_channels {
        // Only process VLSC channels (cn_type == 7)
        if *cn_type != 7 {
            continue;
        }
        // First pass: get offsets, sizes and cn_data position (immutable borrow)
        let (offsets, sizes, cn_data, cn_data_type) = {
            let cn = match channel_group.cn.get(rec_pos) {
                Some(cn) => cn,
                None => continue,
            };

            let cn_data = cn.block.cn_data;
            if cn_data == 0 {
                continue;
            }

            // Get offsets from the VLSC channel data
            let offsets: Vec<u64> = cn.data.to_u64_vec().unwrap_or_default();

            if offsets.is_empty() {
                continue;
            }

            // Get sizes from cn_cn_size channel
            let sizes: Vec<u64> = cn
                .block
                .cn_cn_size()
                .and_then(|size_pos| {
                    channel_group
                        .cn
                        .values()
                        .find(|cn| cn.block_position == size_pos)
                        .and_then(|size_cn| size_cn.data.to_u64_vec())
                })
                .unwrap_or_default();

            if sizes.is_empty() {
                continue;
            }

            (offsets, sizes, cn_data, cn.block.cn_data_type)
        };

        // Second pass: read VD block and update cn.data (mutable borrow)
        rdr.seek_relative(cn_data - position)
            .context("Could not position buffer for VD block")?;
        position = cn_data;

        let mut id = [0u8; 4];
        rdr.read_exact(&mut id)
            .context("could not read VD block id")?;

        let data: Vec<u8> = match read_all_blocks_to_bytes(rdr, id, position)? {
            Some((buf, pos)) => {
                position = pos;
                buf
            }
            None => continue,
        };

        // Now update cn.data with the actual variable length data
        if let Some(cn) = channel_group.cn.get_mut(rec_pos) {
            // Reinitialize cn.data for the actual variable length data
            // cn_data_type: 6=SBC, 7=UTF-8, 8=UTF-16 LE, 9=UTF-16 BE, 10=byte array, 17=UTF-8 with BOM
            cn.data = match cn_data_type {
                6..=9 | 17 => ChannelData::Utf8(arrow::array::LargeStringBuilder::new()),
                10 => ChannelData::VariableSizeByteArray(arrow::array::LargeBinaryBuilder::new()),
                _ => continue,
            };

            read_vlsc_from_bytes(&data, cn, &offsets, &sizes, decoder)?;
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
    let mut str_buf = String::new();
    match &mut cn.data {
        ChannelData::Utf8(array) => {
            let cn_data_type = cn.block.cn_data_type;
            while remaining > 0 {
                let len = &data[position..position + std::mem::size_of::<u32>()];
                let length: usize =
                    u32::from_le_bytes(len.try_into().context("Could not read length")?) as usize;
                if (position + length + 4) <= data_length {
                    position += std::mem::size_of::<u32>();
                    // From MDF 4.3, null terminator is optional in VLSD strings.
                    // Strip trailing \0 only if actually present (check the last byte).
                    let record_len = match cn_data_type {
                        6 | 7 => {
                            if length > 0 && data[position + length - 1] == 0 {
                                length - 1
                            } else {
                                length
                            }
                        }
                        _ => length,
                    };
                    let record = &data[position..position + record_len];
                    array.append_value(decode_string_bytes(
                        record,
                        cn_data_type,
                        decoder,
                        &mut str_buf,
                    )?);
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
        ChannelData::VariableSizeByteArray(array) => {
            while remaining > 0 {
                let len = &data[position..position + std::mem::size_of::<u32>()];
                let length: usize =
                    u32::from_le_bytes(len.try_into().context("Could not read length")?) as usize;
                if (position + length + 4) <= data_length {
                    position += std::mem::size_of::<u32>();
                    let record = &data[position..position + length];
                    array.append_value(record);
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
        _ => {}
    }
    Ok(nrecord + previous_index)
}

/// Reads Variable Length Signal data with Size Channel from bytes of a VD Block
/// Unlike VLSD (SD Block), the sizes come from a separate size channel in the record triplet (time, size, offset)
/// offsets: array of offsets into the VD block data for each record
/// sizes: array of sizes (in bytes) for each record value
/// Returns the maximum position reached in the data buffer (max of offset + size)
fn read_vlsc_from_bytes(
    data: &[u8],
    cn: &mut Cn4,
    offsets: &[u64],
    sizes: &[u64],
    decoder: &mut Dec,
) -> Result<usize> {
    let data_length = data.len();
    let mut max_position: usize = 0;
    let mut str_buf = String::new();
    match &mut cn.data {
        ChannelData::Utf8(array) => {
            let cn_data_type = cn.block.cn_data_type;
            for (offset, size) in offsets.iter().zip(sizes.iter()) {
                let start = *offset as usize;
                let length = *size as usize;
                if start + length <= data_length && length > 0 {
                    let record = &data[start..start + length];
                    array.append_value(decode_string_bytes(
                        record,
                        cn_data_type,
                        decoder,
                        &mut str_buf,
                    )?);
                    max_position = max_position.max(start + length);
                } else if length == 0 {
                    array.append_value("");
                } else {
                    array.append_null();
                }
            }
        }
        ChannelData::VariableSizeByteArray(array) => {
            for (offset, size) in offsets.iter().zip(sizes.iter()) {
                let start = *offset as usize;
                let length = *size as usize;
                if start + length <= data_length {
                    let record = &data[start..start + length];
                    array.append_value(record);
                    max_position = max_position.max(start + length);
                } else {
                    array.append_null();
                }
            }
        }
        _ => {}
    }
    Ok(max_position)
}

/// Reads all DL Blocks and returns a vect of them
fn parser_ld4(
    rdr: &mut BufReader<&File>,
    mut position: i64,
    channel_group: &mut Cg4,
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
        initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone())
            .context("failed initialising arrays")?;
        if id == "##DZ".as_bytes() {
            let (dt, block_header) =
                parse_dz(rdr).context("failed parsing dz block pointed by ld4 block")?;
            for cn in channel_group.cn.values_mut() {
                read_one_channel_array(&dt, cn, channel_group.block.cg_cycle_count as usize)
                    .context("failed reading one channel array from DZ")?;
            }
            position = ld_data + block_header.len as i64;
        } else {
            let block_header: Dt4Block = rdr.read_le().context("Could not read DV block header")?;
            let mut buf = vec![0u8; block_header.len as usize - 24];
            rdr.read_exact(&mut buf)
                .context("Could not read Dt4 block")?;
            for cn in channel_group.cn.values_mut() {
                read_one_channel_array(&buf, cn, channel_group.block.cg_cycle_count as usize)
                    .context("failed reading one channel array")?;
            }
            position = ld_data + block_header.len as i64;
        }
        // For VLSD/VLSC, cg_inval_bytes is the high part of VL data size, not invalidation bytes
        if channel_group.block.cg_inval_bytes > 0
            && (channel_group.block.cg_flags & (CG_F_VLSD | CG_F_VLSC)) == 0
        {
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
        position = read_dv_di(rdr, position, channel_group, ld_blocks)?;
    }
    Ok(position)
}

/// reads DV and DI block containing several channels.
///
/// Three-phase approach: (1) sequential I/O, (2) parallel decompression,
/// (3) sequential record assembly — mirrors `parser_dl4_sorted`.
fn read_dv_di(
    rdr: &mut BufReader<&File>,
    mut position: i64,
    channel_group: &mut Cg4,
    ld_blocks: Vec<Ld4Block>,
) -> Result<i64, Error> {
    let cg_cycle_count = channel_group.block.cg_cycle_count as usize;
    // For VLSD/VLSC, cg_inval_bytes is the high part of VL data size, not invalidation bytes
    let cg_inval_bytes = if (channel_group.block.cg_flags & (CG_F_VLSD | CG_F_VLSC)) != 0 {
        0
    } else {
        channel_group.block.cg_inval_bytes as usize
    };
    initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone())
        .context("failed initialising arrays for dv di blocks")?;
    for ld in &ld_blocks {
        if !ld.ld_invalid_data().is_empty() && cg_inval_bytes > 0 {
            channel_group.invalid_bytes = Some(vec![0u8; cg_inval_bytes * cg_cycle_count]);
        }
    }

    // Phase 1: Sequential I/O — collect all raw data and invalid blocks
    let mut raw_data: Vec<RawCompBlock> = Vec::new();
    let mut raw_invalid: Vec<RawCompBlock> = Vec::new();
    for ld in &ld_blocks {
        for &data_pointer in &ld.ld_data() {
            rdr.seek_relative(data_pointer - position)
                .context("Could not reach DV or DZ block position")?;
            let mut id = [0u8; 4];
            rdr.read_exact(&mut id)
                .context("could not read data block id from LD4")?;
            if id == *b"##DZ" {
                let (header, buf) = read_dz_raw(rdr)?;
                position = data_pointer + header.len as i64;
                raw_data.push(RawCompBlock::Dz(header, buf));
            } else {
                let block_header: Dt4Block =
                    rdr.read_le().context("Could not read DV block structure")?;
                let mut buf = vec![0u8; (block_header.len - 24) as usize];
                rdr.read_exact(&mut buf).context("Could not read DV data")?;
                position = data_pointer + block_header.len as i64;
                raw_data.push(RawCompBlock::Plain(buf));
            }
        }
        for &data_pointer in &ld.ld_invalid_data() {
            rdr.seek_relative(data_pointer - position)
                .context("Could not reach invalid block position")?;
            let mut id = [0u8; 4];
            rdr.read_exact(&mut id)
                .context("could not read data block id from ld4 invalid")?;
            if id == *b"##DZ" {
                let (header, buf) = read_dz_raw(rdr)?;
                position = data_pointer + header.len as i64;
                raw_invalid.push(RawCompBlock::Dz(header, buf));
            } else {
                let block_header: Dt4Block = rdr
                    .read_le()
                    .context("Could not read invalid block header")?;
                let mut buf = vec![0u8; (block_header.len - 24) as usize];
                rdr.read_exact(&mut buf)
                    .context("Could not read invalid data")?;
                position = data_pointer + block_header.len as i64;
                raw_invalid.push(RawCompBlock::Plain(buf));
            }
        }
    }

    // Phase 2: Parallel decompression
    let decompressed_data: Vec<Vec<u8>> = raw_data
        .into_par_iter()
        .map(|raw| -> Result<Vec<u8>> {
            match raw {
                RawCompBlock::Plain(d) => Ok(d),
                RawCompBlock::Dz(header, buf) => header
                    .decompress(buf)
                    .context("failed decompressing DV DZ block"),
            }
        })
        .collect::<Result<_>>()?;
    let decompressed_invalid: Vec<Vec<u8>> = raw_invalid
        .into_par_iter()
        .map(|raw| -> Result<Vec<u8>> {
            match raw {
                RawCompBlock::Plain(d) => Ok(d),
                RawCompBlock::Dz(header, buf) => header
                    .decompress(buf)
                    .context("failed decompressing DI DZ block"),
            }
        })
        .collect::<Result<_>>()?;

    // Phase 3: Sequential record assembly — use actual decompressed length (matches original
    // parse_dz which updated dz_org_data_length = data.len() after decompression).
    let record_length = channel_group.record_length as usize;
    let mut data: Vec<u8> = Vec::new();
    let mut previous_index: usize = 0;
    for block_data in decompressed_data {
        let block_length = block_data.len();
        data.extend(block_data);
        let n_record_chunk = block_length / record_length;
        if previous_index + n_record_chunk < cg_cycle_count {
            read_channels_from_bytes(
                &data[..record_length * n_record_chunk],
                &mut channel_group.cn,
                record_length,
                previous_index,
                false,
            )
            .context("failed reading channels from dv di blocks")?;
        } else {
            read_channels_from_bytes(
                &data[..record_length * (cg_cycle_count - previous_index)],
                &mut channel_group.cn,
                record_length,
                previous_index,
                false,
            )
            .context("failed reading channels from dv di blocks")?;
        }
        let remaining = block_length % record_length;
        if remaining > 0 {
            data.copy_within(record_length * n_record_chunk.., 0);
            data.truncate(remaining);
        } else {
            data.clear();
        }
        previous_index += n_record_chunk;
    }

    // Phase 3b: Copy invalid data into the pre-allocated buffer
    let mut previous_invalid_pos: usize = 0;
    for block_data in decompressed_invalid {
        let block_length = block_data.len();
        if let Some(invalid) = &mut channel_group.invalid_bytes {
            invalid[previous_invalid_pos..previous_invalid_pos + block_length]
                .copy_from_slice(&block_data);
            previous_invalid_pos += block_length;
        }
    }

    Ok(position)
}

/// Reads all DL Blocks and returns a vect of them
fn parser_dl4(rdr: &mut BufReader<&File>, mut position: i64) -> Result<(Vec<Dl4Block>, i64)> {
    let mut dl_blocks: Vec<Dl4Block> = Vec::new();
    let mut visited: HashSet<i64> = HashSet::new();
    let start = position;
    visited.insert(start);
    let (block, pos) = parser_dl4_block(rdr, position, position)?;
    position = pos;
    dl_blocks.push(block.clone());
    let mut next_dl = block.dl_dl_next;
    while next_dl > 0 {
        if !visited.insert(next_dl) {
            warn!("DL block cycle detected at 0x{next_dl:x}, stopping chain walk");
            break;
        }
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

/// Raw (possibly compressed) DZ or plain DT block collected during sequential I/O.
enum RawCompBlock {
    Plain(Vec<u8>),
    Dz(Dz4Block, Vec<u8>),
}

/// Reads all sorted data blocks pointed by DL4 Blocks.
///
/// Three-phase approach: (1) sequential I/O collects raw bytes, (2) Rayon
/// decompresses all DZ blocks in parallel, (3) sequential record assembly.
fn parser_dl4_sorted(
    rdr: &mut BufReader<&File>,
    dl_blocks: Vec<Dl4Block>,
    mut position: i64,
    channel_group: &mut Cg4,
    decoder: &mut Dec,
    rec_pos: &i32,
) -> Result<(i64, Vec<(u8, i32)>)> {
    initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone())
        .context("failed initialising arrays for sorted dl4 block")?;

    let cg_cycle_count = channel_group.block.cg_cycle_count as usize;
    let record_length = channel_group.record_length as usize;
    let mut vlsd_channels: Vec<(u8, i32)> = Vec::new();

    // Phase 1: Sequential I/O — collect raw blocks
    let mut raw_entries: Vec<(bool, RawCompBlock)> = Vec::new(); // (is_sd, block)
    for dl in &dl_blocks {
        for &data_pointer in &dl.dl_data {
            rdr.seek_relative(data_pointer - position)
                .context("Could not reach DV or DZ block position from DL4")?;
            let mut id = [0u8; 4];
            rdr.read_exact(&mut id)
                .context("could not read data block id")?;
            if id == *b"##DZ" {
                let (header, compressed) = read_dz_raw(rdr)?;
                let is_sd = header.dz_org_block_type == *b"SD";
                position = data_pointer + header.len as i64;
                raw_entries.push((is_sd, RawCompBlock::Dz(header, compressed)));
            } else {
                let block_header: Dt4Block =
                    rdr.read_le().context("Could not read DT block header")?;
                // Guard against unfinalized files where the last DT block's len field
                // was never updated (len == 0 or garbage). Clamp to available file bytes.
                let data_len = if block_header.len >= 24 {
                    let file_size = rdr
                        .get_ref()
                        .metadata()
                        .map(|m| m.len())
                        .unwrap_or(u64::MAX);
                    let claimed = block_header.len - 24;
                    let available = (file_size as i64).saturating_sub(data_pointer + 24);
                    if available > 0 {
                        claimed.min(available as u64) as usize
                    } else {
                        0
                    }
                } else {
                    // len < 24 — block header not finalized; read nothing for this block
                    warn!(
                        "DT block at 0x{data_pointer:x} has invalid len={} (unfinalized?), skipping",
                        block_header.len
                    );
                    0
                };
                let mut buf = vec![0u8; data_len];
                rdr.read_exact(&mut buf)
                    .context("Could not read DT block data")?;
                position = data_pointer + 24 + data_len as i64;
                let is_sd = &id[2..4] == b"SD";
                raw_entries.push((is_sd, RawCompBlock::Plain(buf)));
            }
        }
    }

    // Phase 2: Parallel decompression
    let decompressed: Vec<(bool, Vec<u8>)> = raw_entries
        .into_par_iter()
        .map(|(is_sd, raw)| -> Result<(bool, Vec<u8>)> {
            let data = match raw {
                RawCompBlock::Plain(d) => d,
                RawCompBlock::Dz(header, buf) => header
                    .decompress(buf)
                    .context("failed decompressing DZ block in DL4 sorted")?,
            };
            Ok((is_sd, data))
        })
        .collect::<Result<_>>()?;

    // Phase 3: Sequential record assembly (same logic as original)
    let mut data: Vec<u8> = Vec::new();
    let mut previous_index: usize = 0;
    for (is_sd, block_data) in decompressed {
        data.extend(block_data);
        // Use data.len() so a partial-record tail from the previous block is included.
        let block_length = data.len();
        if is_sd {
            if let Some(cn) = channel_group.cn.get_mut(rec_pos) {
                previous_index = read_vlsd_from_bytes(&mut data, cn, previous_index, decoder)?;
            }
        } else {
            let n_record_chunk = block_length.checked_div(record_length).unwrap_or(0);
            if previous_index >= cg_cycle_count || n_record_chunk == 0 {
                continue;
            }
            if previous_index + n_record_chunk < cg_cycle_count {
                vlsd_channels = read_channels_from_bytes(
                    &data[..record_length * n_record_chunk],
                    &mut channel_group.cn,
                    record_length,
                    previous_index,
                    true,
                )
                .context("could not read channels from bytes")?;
            } else {
                // Some implementations pre-allocate equal-length blocks
                vlsd_channels = read_channels_from_bytes(
                    &data[..record_length * (cg_cycle_count - previous_index)],
                    &mut channel_group.cn,
                    record_length,
                    previous_index,
                    true,
                )
                .context("could not read channels from bytes")?;
            }
            let remaining = block_length % record_length;
            if remaining > 0 {
                data.copy_within(record_length * n_record_chunk.., 0);
                data.truncate(remaining);
            } else {
                data.clear();
            }
            previous_index += n_record_chunk;
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
) -> Result<i64> {
    // Read all data blocks
    let mut data: Vec<u8> = Vec::with_capacity(CHUNK_SIZE_READING_4 * 2);
    let mut decoder: Dec = Dec {
        windows_1252: WINDOWS_1252.new_decoder(),
        utf_16_be: UTF_16BE.new_decoder(),
        utf_16_le: UTF_16LE.new_decoder(),
    };
    // initialise record counter
    let mut unsorted_state = UnsortedState::new(dg);
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
                &mut unsorted_state.buffers,
                &mut decoder,
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
) -> Result<Vec<(u8, i32)>> {
    let chunks = generate_chunks(channel_group);
    // initialises the arrays
    initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone())
        .context("failed initialising arrays")?;
    // read by chunks and store in channel array
    let mut previous_index: usize = 0;
    let mut vlsd_channels: Vec<(u8, i32)> = Vec::new();
    // Allocate buffer once and reuse across chunks
    let max_chunk_size = chunks.iter().map(|c| c.1).max().unwrap_or(0);
    let mut data_chunk = vec![0u8; max_chunk_size];
    for (n_record_chunk, chunk_size) in chunks {
        rdr.read_exact(&mut data_chunk[..chunk_size])
            .context("Could not read data chunk")?;
        vlsd_channels = read_channels_from_bytes(
            &data_chunk[..chunk_size],
            &mut channel_group.cn,
            channel_group.record_length as usize,
            previous_index,
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
) -> Result<Vec<(u8, i32)>> {
    // initialises the arrays
    initialise_arrays(channel_group, &channel_group.block.cg_cycle_count.clone())
        .context("failed initilising arrays")?;
    let vlsd_channels: Vec<(u8, i32)> = read_channels_from_bytes(
        data,
        &mut channel_group.cn,
        channel_group.record_length as usize,
        0,
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
) -> Result<()> {
    let data_block_length = block_length as usize;
    let mut position: usize = 24;
    let mut unsorted_state = UnsortedState::new(dg);
    let mut decoder: Dec = Dec {
        windows_1252: WINDOWS_1252.new_decoder(),
        utf_16_be: UTF_16BE.new_decoder(),
        utf_16_le: UTF_16LE.new_decoder(),
    };

    // reads the sorted data block into chunks
    let mut data: Vec<u8> = Vec::with_capacity(CHUNK_SIZE_READING_4 * 2);
    let mut data_chunk = vec![0u8; CHUNK_SIZE_READING_4];
    while position < data_block_length {
        let chunk_size = if (data_block_length - position) > CHUNK_SIZE_READING_4 {
            position += CHUNK_SIZE_READING_4;
            CHUNK_SIZE_READING_4
        } else {
            let remaining = data_block_length - position;
            position += remaining;
            remaining
        };
        rdr.read_exact(&mut data_chunk[..chunk_size])
            .context("Could not read data chunk")?;
        data.extend_from_slice(&data_chunk[..chunk_size]);
        unsorted_state.process_chunk(&mut data, dg, &mut decoder)?;
    }
    Ok(())
}

/// Lookup entry for fast CG access by record ID.
struct CgLookupEntry {
    rec_id: u64,
    record_length: u32,
    is_vlsd: bool,
    vlsd_cg: Option<(u64, i32)>,
}

/// State for unsorted data reading — uses Vec-indexed lookup instead of HashMap
/// for O(1) access by record ID with better cache locality.
struct UnsortedState {
    /// Indexed by record ID — each entry holds (write_index, sorted_data_buffer)
    buffers: Vec<Option<(usize, Vec<u8>)>>,
}

impl UnsortedState {
    fn new(dg: &Dg4) -> Self {
        let max_record_id = dg
            .cg
            .values()
            .map(|cg| cg.block.cg_record_id as usize)
            .max()
            .unwrap_or(0);
        let mut buffers: Vec<Option<(usize, Vec<u8>)>> = Vec::with_capacity(max_record_id + 1);
        buffers.resize_with(max_record_id + 1, || None);

        for cg in dg.cg.values() {
            let rec_id = cg.block.cg_record_id as usize;
            let capacity = if (cg.block.cg_flags & (CG_F_VLSD | CG_F_VLSC)) != 0 {
                0
            } else {
                cg.block.cg_cycle_count as usize * cg.record_length as usize
            };
            buffers[rec_id] = Some((0, Vec::with_capacity(capacity)));
        }

        Self { buffers }
    }

    fn process_chunk(
        &mut self,
        data: &mut Vec<u8>,
        dg: &mut Dg4,
        decoder: &mut Dec,
    ) -> Result<(), Error> {
        read_all_channels_unsorted_from_bytes(data, dg, &mut self.buffers, decoder)
    }
}

/// read record by record from unsorted data block into sorted data block, then copy data into channel arrays
fn read_all_channels_unsorted_from_bytes(
    data: &mut Vec<u8>,
    dg: &mut Dg4,
    record_buffers: &mut [Option<(usize, Vec<u8>)>],
    decoder: &mut Dec,
) -> Result<(), Error> {
    let mut position: usize = 0;
    let data_length = data.len();
    let dg_rec_id_size = dg.block.dg_rec_id_size as usize;
    let vlsd_data_start_offset = dg_rec_id_size + std::mem::size_of::<u32>();
    // reusable string buffer for VLSC string decoding
    let mut dst = String::new();

    // Pre-build a Vec of (rec_id, record_length, is_vlsd, vlsd_target_info)
    // for fast lookup in the hot loop
    let mut cg_lookup: Vec<CgLookupEntry> = Vec::new();
    for cg in dg.cg.values() {
        let is_vlsd = (cg.block.cg_flags & (CG_F_VLSD | CG_F_VLSC)) != 0;
        cg_lookup.push(CgLookupEntry {
            rec_id: cg.block.cg_record_id,
            record_length: cg.record_length,
            is_vlsd,
            vlsd_cg: cg.vlsd_cg,
        });
    }

    // unsorted data into sorted data blocks, except for VLSD CG.
    let mut remaining: usize = data_length - position;
    while remaining > 0 {
        // reads record id
        let rec_id: u64 = if dg_rec_id_size == 1 && remaining >= 1 {
            data[position].into()
        } else if dg_rec_id_size == 2 && remaining >= 2 {
            let rec = &data[position..position + std::mem::size_of::<u16>()];
            u16::from_le_bytes(rec.try_into().unwrap()) as u64
        } else if dg_rec_id_size == 4 && remaining >= 4 {
            let rec = &data[position..position + std::mem::size_of::<u32>()];
            u32::from_le_bytes(rec.try_into().unwrap()) as u64
        } else if dg_rec_id_size == 8 && remaining >= 8 {
            let rec = &data[position..position + std::mem::size_of::<u64>()];
            u64::from_le_bytes(rec.try_into().unwrap())
        } else {
            break; // not enough data remaining
        };

        // Find the CG for this record ID using linear search (fast for small N)
        let cg_info = cg_lookup.iter().find(|entry| entry.rec_id == rec_id);

        if let Some(cg_entry) = cg_info {
            if cg_entry.is_vlsd {
                // VLSD or VLSC channel (Variable Length Signal Data/Size Channel)
                if remaining >= 4 + dg_rec_id_size {
                    let len = &data[position + dg_rec_id_size..position + vlsd_data_start_offset];
                    let length: usize = u32::from_le_bytes(len.try_into().unwrap()) as usize;
                    remaining = data_length - position - vlsd_data_start_offset;
                    if remaining >= length {
                        position += vlsd_data_start_offset;
                        let record = &data[position..position + length];
                        if let Some((target_rec_id, target_rec_pos)) = cg_entry.vlsd_cg {
                            if let Some(target_cg) = dg.cg.get_mut(&target_rec_id) {
                                if let Some(target_cn) = target_cg.cn.get_mut(&target_rec_pos) {
                                    if let Some((nrecord, _)) =
                                        record_buffers[rec_id as usize].as_mut()
                                    {
                                        // For VLSC channels (cn_type == 7) in unsorted data,
                                        // reinitialize from UInt (offset storage) to actual data type on first record
                                        if *nrecord == 0 && target_cn.block.cn_type == 7 {
                                            match target_cn.block.cn_data_type {
                                                6..=9 | 17 => {
                                                    target_cn.data = ChannelData::Utf8(
                                                        arrow::array::LargeStringBuilder::new(),
                                                    );
                                                }
                                                10 => {
                                                    target_cn.data =
                                                        ChannelData::VariableSizeByteArray(
                                                            arrow::array::LargeBinaryBuilder::new(),
                                                        );
                                                }
                                                _ => {}
                                            }
                                        }
                                        match &mut target_cn.data {
                                            ChannelData::Utf8(array) => {
                                                if target_cn.block.cn_data_type == 7 {
                                                    // UTF-8: no decoding needed, use &str directly
                                                    array.append_value(
                                                        str::from_utf8(record)
                                                            .context("Found invalid UTF-8 from VLSD record")?
                                                            .trim_end_matches('\0'),
                                                    );
                                                } else {
                                                    dst.clear();
                                                    if target_cn.block.cn_data_type == 6 {
                                                        let (_result, _size, _replacement) =
                                                            decoder.windows_1252.decode_to_string(
                                                                record, &mut dst, false,
                                                            );
                                                    } else if target_cn.block.cn_data_type == 8 {
                                                        let (_result, _size, _replacement) =
                                                            decoder.utf_16_le.decode_to_string(
                                                                record, &mut dst, false,
                                                            );
                                                    } else if target_cn.block.cn_data_type == 9 {
                                                        let (_result, _size, _replacement) =
                                                            decoder.utf_16_be.decode_to_string(
                                                                record, &mut dst, false,
                                                            );
                                                    } else if target_cn.block.cn_data_type == 17 {
                                                        // Unicode with BOM
                                                        let bom = Bom::from(record);
                                                        let mut bom_decoder = match bom {
                                                            Bom::Utf8 => UTF_8.new_decoder(),
                                                            Bom::Utf16Be => UTF_16BE.new_decoder(),
                                                            Bom::Utf16Le => UTF_16LE.new_decoder(),
                                                            Bom::Gb18030 => GB18030.new_decoder(),
                                                            _ => {
                                                                bail!("not implemented BOM type");
                                                            }
                                                        };
                                                        let (_result, _size, _replacement) =
                                                            bom_decoder.decode_to_string(
                                                                record, &mut dst, false,
                                                            );
                                                    } else {
                                                        bail!(
                                                            "channel data type is not correct for a text"
                                                        )
                                                    };
                                                    array.append_value(dst.trim_end_matches('\0'));
                                                }
                                            }
                                            ChannelData::VariableSizeByteArray(array) => {
                                                array.append_value(record);
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
                            bail!("no VLSD/VLSC target in CG, wrong cg_flags");
                        }
                        position += length;
                    } else {
                        break; // not enough data remaining
                    }
                } else {
                    break; // not enough data remaining
                }
            } else if remaining >= cg_entry.record_length as usize {
                // Not VLSD channel
                let record = &data[position..position + cg_entry.record_length as usize];
                if let Some((_nrecord, data)) = record_buffers[rec_id as usize].as_mut() {
                    data.extend(record);
                } else {
                    bail!("could not find the record id");
                }
                position += cg_entry.record_length as usize;
            } else {
                break; // not enough data remaining
            }
        } else {
            bail!("could not find the record id");
        }
        remaining = data_length - position;
    }

    // removes consumed records from data and leaves remaining that could not be processed.
    let remaining_len = data.len() - position;
    data.copy_within(position.., 0);
    data.truncate(remaining_len);

    // From sorted data block, copies data in channels arrays
    for (rec_id, buffer) in record_buffers.iter_mut().enumerate() {
        if let Some((index, record_data)) = buffer
            && let Some(channel_group) = dg.cg.get_mut(&(rec_id as u64))
        {
            let record_length = channel_group.record_length as usize;
            let n_records = record_data.len().checked_div(record_length).unwrap_or(0);
            read_channels_from_bytes(
                record_data,
                &mut channel_group.cn,
                record_length,
                *index,
                true,
            )
            .context("failed reading channels from bytes after reading unsorted data")?;
            *index += n_records; // advance write position for next DL block
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

/// Decodes a byte slice to a String based on MDF4 cn_data_type.
/// cn_data_type: 6=SBC/Windows-1252, 7=UTF-8, 8=UTF-16 LE, 9=UTF-16 BE, 17=BOM-prefixed
fn decode_string_bytes<'a>(
    record: &'a [u8],
    cn_data_type: u8,
    decoder: &mut Dec,
    buf: &'a mut String,
) -> Result<&'a str> {
    match cn_data_type {
        6 => {
            buf.clear();
            buf.reserve(record.len());
            let _ = decoder.windows_1252.decode_to_string(record, buf, false);
            Ok(buf.as_str())
        }
        7 => Ok(str::from_utf8(record).context("Found invalid UTF-8")?),
        8 => {
            buf.clear();
            buf.reserve(record.len());
            let _ = decoder.utf_16_le.decode_to_string(record, buf, false);
            Ok(buf.trim_end_matches('\0'))
        }
        9 => {
            buf.clear();
            buf.reserve(record.len());
            let _ = decoder.utf_16_be.decode_to_string(record, buf, false);
            Ok(buf.trim_end_matches('\0'))
        }
        17 => {
            if record.len() >= 3 && record[0] == 0xEF && record[1] == 0xBB && record[2] == 0xBF {
                Ok(str::from_utf8(&record[3..]).context("Found invalid UTF-8 with BOM")?)
            } else if record.len() >= 2 && record[0] == 0xFF && record[1] == 0xFE {
                buf.clear();
                buf.reserve(record.len());
                let _ = decoder.utf_16_le.decode_to_string(&record[2..], buf, false);
                Ok(buf.trim_end_matches('\0'))
            } else if record.len() >= 2 && record[0] == 0xFE && record[1] == 0xFF {
                buf.clear();
                buf.reserve(record.len());
                let _ = decoder.utf_16_be.decode_to_string(&record[2..], buf, false);
                Ok(buf.trim_end_matches('\0'))
            } else {
                Ok(str::from_utf8(record).context("Found invalid UTF-8 (no BOM)")?)
            }
        }
        _ => {
            buf.clear();
            buf.push_str(&String::from_utf8_lossy(record));
            Ok(buf.as_str())
        }
    }
}

/// initialise ndarrays for the data group/block
fn initialise_arrays(channel_group: &mut Cg4, cg_cycle_count: &u64) -> Result<(), Error> {
    // creates zeroed array in parallel for each channel contained in channel group
    channel_group
        .cn
        .par_iter_mut()
        .filter(|(_cn_record_position, cn)| cn.should_read)
        .try_for_each(
            |(_cn_record_position, cn): (&i32, &mut Cn4)| -> Result<(), Error> {
                cn.data = cn
                    .data
                    .zeros(
                        cn.block.cn_type,
                        *cg_cycle_count,
                        cn.n_bytes,
                        cn.shape.clone(),
                    )
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
fn apply_bit_mask_offset(dg: &mut Dg4) -> Result<(), Error> {
    // apply bit shift and masking
    dg.cg
        .par_iter_mut()
        .try_for_each(|(_, channel_group)| -> Result<(), Error> {
            channel_group
                .cn
                .par_iter_mut()
                .filter(|(_cn_record_position, cn)| cn.should_read)
                .try_for_each(|(_rec_pos, cn): (&i32, &mut Cn4)| -> Result<(), Error> {
                    if cn.block.cn_data_type <= 3 {
                        let left_shift = cn.n_bytes * 8
                            - (cn.block.cn_bit_offset as u32)
                            - cn.block.cn_bit_count;
                        let right_shift = left_shift + (cn.block.cn_bit_offset as u32);
                        if left_shift > 0 || right_shift > 0 {
                            match &mut cn.data {
                                ChannelData::Int8(array) => {
                                    let a = array.values_slice_mut();
                                    if left_shift > 0 {
                                        a.iter_mut().for_each(|x| *x <<= left_shift)
                                    };
                                    if right_shift > 0 {
                                        a.iter_mut().for_each(|x| *x >>= right_shift)
                                    };
                                }
                                ChannelData::UInt8(array) => {
                                    let a = array.values_slice_mut();
                                    if left_shift > 0 {
                                        a.iter_mut().for_each(|x| *x <<= left_shift)
                                    };
                                    if right_shift > 0 {
                                        a.iter_mut().for_each(|x| *x >>= right_shift)
                                    };
                                }
                                ChannelData::Int16(array) => {
                                    let a = array.values_slice_mut();
                                    if left_shift > 0 {
                                        a.iter_mut().for_each(|x| *x <<= left_shift)
                                    };
                                    if right_shift > 0 {
                                        a.iter_mut().for_each(|x| *x >>= right_shift)
                                    };
                                }
                                ChannelData::UInt16(array) => {
                                    let a = array.values_slice_mut();
                                    if left_shift > 0 {
                                        a.iter_mut().for_each(|x| *x <<= left_shift)
                                    };
                                    if right_shift > 0 {
                                        a.iter_mut().for_each(|x| *x >>= right_shift)
                                    };
                                }
                                ChannelData::Int32(array) => {
                                    let a = array.values_slice_mut();
                                    if left_shift > 0 {
                                        a.iter_mut().for_each(|x| *x <<= left_shift)
                                    };
                                    if right_shift > 0 {
                                        a.iter_mut().for_each(|x| *x >>= right_shift)
                                    };
                                }
                                ChannelData::UInt32(array) => {
                                    let a = array.values_slice_mut();
                                    if left_shift > 0 {
                                        a.iter_mut().for_each(|x| *x <<= left_shift)
                                    };
                                    if right_shift > 0 {
                                        a.iter_mut().for_each(|x| *x >>= right_shift)
                                    };
                                }
                                ChannelData::Float32(_) => (),
                                ChannelData::Int64(array) => {
                                    let a = array.values_slice_mut();
                                    let left_shift = 64
                                        - (cn.block.cn_bit_offset as u32)
                                        - cn.block.cn_bit_count;
                                    let right_shift = left_shift + (cn.block.cn_bit_offset as u32);
                                    if left_shift > 0 {
                                        a.iter_mut().for_each(|x| *x <<= left_shift)
                                    };
                                    if right_shift > 0 {
                                        a.iter_mut().for_each(|x| *x >>= right_shift)
                                    };
                                }
                                ChannelData::UInt64(array) => {
                                    let a = array.values_slice_mut();
                                    let left_shift = 64
                                        - (cn.block.cn_bit_offset as u32)
                                        - cn.block.cn_bit_count;
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
                                    let a = array.values_slice_mut();
                                    if left_shift > 0 {
                                        a.iter_mut().for_each(|x| *x <<= left_shift)
                                    };
                                    if right_shift > 0 {
                                        a.iter_mut().for_each(|x| *x >>= right_shift)
                                    };
                                }
                                ChannelData::ArrayDUInt8(array) => {
                                    let a = array.values_slice_mut();
                                    if left_shift > 0 {
                                        a.iter_mut().for_each(|x| *x <<= left_shift)
                                    };
                                    if right_shift > 0 {
                                        a.iter_mut().for_each(|x| *x >>= right_shift)
                                    };
                                }
                                ChannelData::ArrayDInt16(array) => {
                                    let a = array.values_slice_mut();
                                    if left_shift > 0 {
                                        a.iter_mut().for_each(|x| *x <<= left_shift)
                                    };
                                    if right_shift > 0 {
                                        a.iter_mut().for_each(|x| *x >>= right_shift)
                                    };
                                }
                                ChannelData::ArrayDUInt16(array) => {
                                    let a = array.values_slice_mut();
                                    if left_shift > 0 {
                                        a.iter_mut().for_each(|x| *x <<= left_shift)
                                    };
                                    if right_shift > 0 {
                                        a.iter_mut().for_each(|x| *x >>= right_shift)
                                    };
                                }
                                ChannelData::ArrayDInt32(array) => {
                                    let a = array.values_slice_mut();
                                    if left_shift > 0 {
                                        a.iter_mut().for_each(|x| *x <<= left_shift)
                                    };
                                    if right_shift > 0 {
                                        a.iter_mut().for_each(|x| *x >>= right_shift)
                                    };
                                }
                                ChannelData::ArrayDUInt32(array) => {
                                    let a = array.values_slice_mut();
                                    if left_shift > 0 {
                                        a.iter_mut().for_each(|x| *x <<= left_shift)
                                    };
                                    if right_shift > 0 {
                                        a.iter_mut().for_each(|x| *x >>= right_shift)
                                    };
                                }
                                ChannelData::ArrayDFloat32(_) => (),
                                ChannelData::ArrayDInt64(array) => {
                                    let a = array.values_slice_mut();
                                    let left_shift = 64
                                        - (cn.block.cn_bit_offset as u32)
                                        - cn.block.cn_bit_count;
                                    let right_shift = left_shift + (cn.block.cn_bit_offset as u32);
                                    if left_shift > 0 {
                                        a.iter_mut().for_each(|x| *x <<= left_shift)
                                    };
                                    if right_shift > 0 {
                                        a.iter_mut().for_each(|x| *x >>= right_shift)
                                    };
                                }
                                ChannelData::ArrayDUInt64(array) => {
                                    let a = array.values_slice_mut();
                                    let left_shift = 64
                                        - (cn.block.cn_bit_offset as u32)
                                        - cn.block.cn_bit_count;
                                    let right_shift = left_shift + (cn.block.cn_bit_offset as u32);
                                    if left_shift > 0 {
                                        a.iter_mut().for_each(|x| *x <<= left_shift)
                                    };
                                    if right_shift > 0 {
                                        a.iter_mut().for_each(|x| *x >>= right_shift)
                                    };
                                }
                                ChannelData::ArrayDFloat64(_) => (),
                                ChannelData::Union(_) => (),
                            }
                        }
                    }
                    Ok(())
                })
                .with_context(|| {
                    format!("bit mask application failed for channel group {channel_group:?}")
                })?;
            Ok(())
        })?;
    Ok(())
}

/// Helper to process dynamic channels (VLSD, VLSC, DS)
fn process_dynamic_channels(
    rdr: &mut BufReader<&File>,
    dg: &mut Dg4,
    vlsd_channels: &[(u8, i32)],
    mut position: i64,
    decoder: &mut Dec,
) -> Result<i64> {
    if vlsd_channels.is_empty() {
        return Ok(position);
    }
    let mut handled_types = HashSet::new();
    for channel in vlsd_channels {
        if handled_types.insert(channel.0) {
            match channel.0 {
                1 => {
                    position = read_sd(rdr, dg, vlsd_channels, position, decoder)
                        .context("failed reading sd block")?;
                }
                7 => {
                    for channel_group in dg.cg.values_mut() {
                        position = read_vd(rdr, channel_group, vlsd_channels, position, decoder)
                            .context("failed reading vd block")?;
                    }
                }
                _ => {
                    for channel_group in dg.cg.values_mut() {
                        position = read_ds(rdr, channel_group, vlsd_channels, position, decoder)
                            .context("failed reading ds block")?;
                    }
                }
            }
        }
    }
    Ok(position)
}

/// Reads data from Data Storage (DSBLOCK) pointed to by Data Stream channels (cn_type = 8)
/// or Dynamic Arrays (ca_storage = 5).
fn read_ds(
    rdr: &mut BufReader<&File>,
    channel_group: &mut Cg4,
    vlsd_channels: &[(u8, i32)],
    mut position: i64,
    decoder: &mut Dec,
) -> Result<i64> {
    use crate::mdfinfo::mdfinfo4::Compo;
    for (_cn_type, rec_pos) in vlsd_channels {
        if let Some(cn) = channel_group.cn.get_mut(rec_pos) {
            let mut ds_data_pointer = 0i64;
            // Check for DSBLOCK in composition
            if let Some(composition) = &cn.composition {
                match &composition.block {
                    Compo::DS(ds) => {
                        ds_data_pointer = ds.ds_data();
                    }
                    Compo::CA(ca) if ca.ca_storage == 5 => {
                        if let Some(sub_compo) = &composition.compo
                            && let Compo::DS(ds) = &sub_compo.block
                        {
                            ds_data_pointer = ds.ds_data();
                        }
                    }
                    _ => {}
                }
            }

            if ds_data_pointer != 0 {
                rdr.seek_relative(ds_data_pointer - position)
                    .context("Could not position buffer for DS data block")?;
                position = ds_data_pointer;

                let mut id = [0u8; 4];
                rdr.read_exact(&mut id)
                    .context("could not read DS data block id")?;

                let mut data: Vec<u8> = match read_all_blocks_to_bytes(rdr, id, position)? {
                    Some((buf, pos)) => {
                        position = pos;
                        buf
                    }
                    None => continue,
                };

                // Check if this is data stream mode (ds_mode == 0) with composition
                // We need to extract what we need from cn before dropping the borrow
                let ds_decode_info: Option<(Ds4Block, Box<Composition>)> =
                    if let Some(composition) = &cn.composition {
                        if let Compo::DS(ds_block) = &composition.block {
                            if ds_block.ds_mode == 0 {
                                // Clone the ds_block (dereferencing the Box) and the composition
                                composition
                                    .compo
                                    .as_ref()
                                    .map(|c| ((**ds_block).clone(), c.clone()))
                            } else if ds_block.ds_mode == 1 {
                                // Data description mode - data layout is described by an external
                                // attachment file (e.g., FIBEX, DBC, ARXML) pointed to by ds_cn_composition.
                                // This mode is not yet fully supported.
                                warn!(
                                    "Channel '{}' uses data description mode (ds_mode=1). \
                                    Data layout is described by an external attachment. \
                                    This mode requires external description file parsing \
                                    (FIBEX, DBC, ARXML) which is not yet implemented. \
                                    Data will be stored as raw bytes.",
                                    cn.unique_name
                                );
                                None
                            } else {
                                warn!(
                                    "Channel '{}' has unknown DSBLOCK mode: {}",
                                    cn.unique_name, ds_block.ds_mode
                                );
                                None
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                if let Some((ds_block, child_composition)) = ds_decode_info {
                    // Parse record offsets and sizes from length-prefixed data
                    let (record_offsets, record_sizes) = parse_vlsd_record_offsets(&data)?;

                    // Decode using composition
                    let decoded = datastream_decoder::decode_datastream_blob(
                        &data,
                        &ds_block,
                        &child_composition,
                        &channel_group.cn,
                        &record_offsets,
                        &record_sizes,
                    )?;

                    // Store decoded child channel data
                    for (rec_pos, values) in decoded {
                        if let Some(child_cn) = channel_group.cn.get_mut(&rec_pos) {
                            store_decoded_values_in_channel(child_cn, values, decoder)?;
                        }
                    }
                } else {
                    // Fallback: read as length-prefixed samples for dynamic data
                    if let Some(cn) = channel_group.cn.get_mut(rec_pos) {
                        read_vlsd_from_bytes(&mut data, cn, 0, decoder)?;
                    }
                }
            }
        }
    }
    Ok(position)
}

/// Parses VLSD record offsets and sizes from a length-prefixed data blob.
/// Returns (offsets, sizes) vectors where each element corresponds to one record.
fn parse_vlsd_record_offsets(data: &[u8]) -> Result<(Vec<u64>, Vec<u64>)> {
    let mut offsets = Vec::new();
    let mut sizes = Vec::new();
    let mut position: usize = 0;

    while position + 4 <= data.len() {
        let length = u32::from_le_bytes(
            data[position..position + 4]
                .try_into()
                .context("Could not read VLSD length prefix")?,
        ) as usize;

        // Record starts after the 4-byte length prefix
        offsets.push((position + 4) as u64);
        sizes.push(length as u64);

        // Move to next record
        position += 4 + length;

        if position > data.len() {
            break;
        }
    }

    Ok((offsets, sizes))
}

/// Stores decoded byte values into a channel's data structure
fn store_decoded_values_in_channel(
    cn: &mut Cn4,
    values: Vec<Vec<u8>>,
    decoder: &mut Dec,
) -> Result<()> {
    let mut str_buf = String::new();
    for value_bytes in values {
        match &mut cn.data {
            ChannelData::Int8(builder) if !value_bytes.is_empty() => {
                builder.append_value(value_bytes[0] as i8);
            }
            ChannelData::UInt8(builder) if !value_bytes.is_empty() => {
                builder.append_value(value_bytes[0]);
            }
            ChannelData::Int16(builder) if value_bytes.len() >= 2 => {
                let val = if cn.endian.is_big() {
                    i16::from_be_bytes(value_bytes[..2].try_into()?)
                } else {
                    i16::from_le_bytes(value_bytes[..2].try_into()?)
                };
                builder.append_value(val);
            }
            ChannelData::UInt16(builder) if value_bytes.len() >= 2 => {
                let val = if cn.endian.is_big() {
                    u16::from_be_bytes(value_bytes[..2].try_into()?)
                } else {
                    u16::from_le_bytes(value_bytes[..2].try_into()?)
                };
                builder.append_value(val);
            }
            ChannelData::Int32(builder) if value_bytes.len() >= 4 => {
                let val = if cn.endian.is_big() {
                    i32::from_be_bytes(value_bytes[..4].try_into()?)
                } else {
                    i32::from_le_bytes(value_bytes[..4].try_into()?)
                };
                builder.append_value(val);
            }
            ChannelData::UInt32(builder) if value_bytes.len() >= 4 => {
                let val = if cn.endian.is_big() {
                    u32::from_be_bytes(value_bytes[..4].try_into()?)
                } else {
                    u32::from_le_bytes(value_bytes[..4].try_into()?)
                };
                builder.append_value(val);
            }
            ChannelData::Float32(builder) if value_bytes.len() >= 4 => {
                let val = if cn.endian.is_big() {
                    f32::from_be_bytes(value_bytes[..4].try_into()?)
                } else {
                    f32::from_le_bytes(value_bytes[..4].try_into()?)
                };
                builder.append_value(val);
            }
            ChannelData::Int64(builder) if value_bytes.len() >= 8 => {
                let val = if cn.endian.is_big() {
                    i64::from_be_bytes(value_bytes[..8].try_into()?)
                } else {
                    i64::from_le_bytes(value_bytes[..8].try_into()?)
                };
                builder.append_value(val);
            }
            ChannelData::UInt64(builder) if value_bytes.len() >= 8 => {
                let val = if cn.endian.is_big() {
                    u64::from_be_bytes(value_bytes[..8].try_into()?)
                } else {
                    u64::from_le_bytes(value_bytes[..8].try_into()?)
                };
                builder.append_value(val);
            }
            ChannelData::Float64(builder) if value_bytes.len() >= 8 => {
                let val = if cn.endian.is_big() {
                    f64::from_be_bytes(value_bytes[..8].try_into()?)
                } else {
                    f64::from_le_bytes(value_bytes[..8].try_into()?)
                };
                builder.append_value(val);
            }
            ChannelData::Utf8(builder) => {
                builder.append_value(decode_string_bytes(
                    &value_bytes,
                    cn.block.cn_data_type,
                    decoder,
                    &mut str_buf,
                )?);
            }
            ChannelData::VariableSizeByteArray(builder) => {
                builder.append_value(&value_bytes);
            }
            _ => {
                // For other types (complex, tensor, etc.), skip for now
            }
        }
    }
    Ok(())
}

// =============================================================================
// Sample Reduction Data Reading (RDBLOCK/RVBLOCK/RIBLOCK)
// =============================================================================
