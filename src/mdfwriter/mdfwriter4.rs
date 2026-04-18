//! MDF 4.2 file writer — spec sections 5-6
//!
//! Writes in-memory channel data to an MDF 4.2 file. The writer always produces
//! sorted data (one DG per channel, `dg_rec_id_size=0`), converting VLSC channels
//! to fixed-length. Two-phase approach:
//! - **Phase 1** (sequential): builds the metadata block hierarchy (HD→FH, EV, AT, SI, DG→CG→CN)
//!   and calculates file positions for each block.
//! - **Phase 2** (parallel): writes raw channel data blocks (DV/DZ/SD) via crossbeam channel,
//!   using rayon for parallel compression.
use std::{
    collections::{HashMap, HashSet},
    fs::OpenOptions,
    io::{BufWriter, Cursor, Seek, SeekFrom, Write},
    ops::Deref,
    sync::Arc,
    thread,
};

use super::mdfwriter3::convert3to4;
use crate::mdfinfo::mdfinfo4::Blockheader4Short;
use crate::{
    data_holder::channel_data::{ChannelData, data_type_init},
    mdfinfo::{
        MdfInfo,
        mdfinfo4::{
            At4Block, BlockType, Blockheader4, Ca4Block, Cg4, Cg4Block, Ch4Block, Cn4, Cn4Block,
            Compo, Composition, Dg4, Dg4Block, Dz4Block, Endianness, Ev4Block, FhBlock, Ld4Block,
            MdfInfo4, MetaData, MetaDataBlockType, Si4Block, default_short_header,
        },
    },
    mdfreader::Mdf,
};
use anyhow::{Context, Error, Result, bail};
use arrow::array::Array;
use arrow::buffer::NullBuffer;
use binrw::BinWriterExt;
use crossbeam_channel::bounded;
use flate2::Compression;
use flate2::write::ZlibEncoder;
use parking_lot::Mutex;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use std::fs::File;

/// Main entry point: writes an MDF 4.2 file from in-memory `Mdf` data.
///
/// Converts MDF3 sources to MDF4 if needed. Preserves file history (FH), events (EV),
/// attachments (AT), source info (SI), and channel hierarchy (CH) from the source file.
/// Supports optional zlib compression for data blocks.
pub fn mdfwriter4(mdf: &Mdf, file_name: &str, compression: bool) -> Result<Mdf> {
    let info: MdfInfo4 = match &mdf.mdf_info {
        MdfInfo::V3(mdfinfo3) => convert3to4(mdfinfo3, file_name)
            .context("failed converting mdf version 3 into version 4")?,
        MdfInfo::V4(mdfinfo4) => mdfinfo4.deref().clone(),
    };
    let n_channels = mdf.mdf_info.get_channel_names_set().len();
    let mut new_info = MdfInfo4::new(file_name, n_channels);
    let mut pointer: i64 = 168; // after HD block

    // Copy existing FH blocks from source and add a new one for mdfr
    // FH blocks are stored as a linked list
    new_info.hd_block.hd_fh_first = pointer;
    let mut fh_blocks: Vec<(FhBlock, MetaData)> = Vec::new();

    // Copy existing FH blocks from source file
    for fh in info.fh.iter() {
        let mut new_fh = *fh;
        let _fh_position = pointer;
        pointer += 56; // FH block size

        // Get the comment MD block from source
        let fh_comment_md = if fh.fh_md_comment != 0 {
            info.sharable.md_tx.get(&fh.fh_md_comment).cloned()
        } else {
            None
        };

        // Create comment block
        let comment_md = if let Some(md) = fh_comment_md {
            new_fh.fh_md_comment = pointer;
            pointer += md.block.hdr_len as i64;
            md
        } else {
            // Create empty comment if source had none
            let mut empty_md = MetaData::new(MetaDataBlockType::MdBlock, BlockType::FH);
            empty_md.set_data_buffer(b"");
            new_fh.fh_md_comment = pointer;
            pointer += empty_md.block.hdr_len as i64;
            empty_md
        };

        // Set next pointer (will be updated below)
        new_fh.fh_fh_next = pointer;

        fh_blocks.push((new_fh, comment_md));
    }

    // Add new FH block documenting mdfr modification
    let mut mdfr_fh = FhBlock::default();
    let mdfr_fh_position = pointer;
    pointer += 56;

    // Update the last copied FH block to point to mdfr FH block
    if let Some((last_fh, _)) = fh_blocks.last_mut() {
        last_fh.fh_fh_next = mdfr_fh_position;
    }

    // Create mdfr comment
    mdfr_fh.fh_md_comment = pointer;
    let mut mdfr_fh_comments = MetaData::new(MetaDataBlockType::MdBlock, BlockType::FH);
    mdfr_fh_comments.create_fh();
    pointer += mdfr_fh_comments.block.hdr_len as i64;
    mdfr_fh.fh_fh_next = 0; // End of list

    fh_blocks.push((mdfr_fh, mdfr_fh_comments));

    // Store FH blocks in new_info
    for (fh, _) in &fh_blocks {
        new_info.fh.push(*fh);
    }

    // Copy events from the source file
    // Events are stored as a linked list starting from hd_ev_first
    let mut ev_blocks: Vec<(i64, Ev4Block, Option<MetaData>, Option<MetaData>)> = Vec::new();
    if !info.ev.is_empty() {
        new_info.hd_block.hd_ev_first = pointer;

        for (orig_pos, ev) in info.ev.iter() {
            let ev_block_position = pointer;

            // Create copies of TX/MD blocks for event name and comment
            let ev_name_md: Option<MetaData> = if ev.ev_tx_name != 0 {
                info.sharable.md_tx.get(&ev.ev_tx_name).cloned()
            } else {
                None
            };

            let ev_comment_md: Option<MetaData> = if ev.ev_md_comment != 0 {
                info.sharable.md_tx.get(&ev.ev_md_comment).cloned()
            } else {
                None
            };

            // Calculate block size: 16 (short header) + 8 (link count) + 8*links + data members
            // Links: ev_ev_next, ev_ev_parent, ev_ev_range, ev_tx_name, ev_md_comment + scope/attachment links
            let n_links = 5 + ev.ev_scope_count as u64 + ev.ev_attachment_count as u64;
            let block_size: i64 = 16 + 8 + (n_links * 8) as i64 + 32; // 32 bytes for data members

            // Create a modified event block with updated pointers
            let mut new_ev = ev.clone();

            // Set name pointer (will be right after the event block)
            let mut current_offset = block_size;
            if let Some(md) = &ev_name_md {
                new_ev.ev_tx_name = pointer + current_offset;
                current_offset += md.block.hdr_len as i64;
            } else {
                new_ev.ev_tx_name = 0;
            }

            // Set comment pointer
            if let Some(md) = &ev_comment_md {
                new_ev.ev_md_comment = pointer + current_offset;
                current_offset += md.block.hdr_len as i64;
            } else {
                new_ev.ev_md_comment = 0;
            }

            // Clear parent and range links for simplicity (would need mapping for full support)
            new_ev.ev_ev_parent = 0;
            new_ev.ev_ev_range = 0;

            // Update the previous event's next pointer
            if let Some(last) = ev_blocks.last_mut() {
                last.1.ev_ev_next = ev_block_position;
            }

            // Advance pointer for this event block and its metadata
            pointer += current_offset;

            ev_blocks.push((*orig_pos, new_ev, ev_name_md, ev_comment_md));
        }

        // Last event's next pointer should be 0
        if let Some(last) = ev_blocks.last_mut() {
            last.1.ev_ev_next = 0;
        }

        // Copy events to new_info
        for (orig_pos, ev, _, _) in &ev_blocks {
            new_info.ev.insert(*orig_pos, ev.clone());
        }
    }

    // Copy attachments from the source file
    // Attachments are stored as a linked list starting from hd_at_first
    type AtBlockEntry = (
        i64,
        At4Block,
        Option<Vec<u8>>,
        Option<MetaData>,
        Option<MetaData>,
        Option<MetaData>,
    );
    let mut at_blocks: Vec<AtBlockEntry> = Vec::new();
    if !info.at.is_empty() {
        new_info.hd_block.hd_at_first = pointer;

        for (orig_pos, (at, embedded_data)) in info.at.iter() {
            let at_block_position = pointer;

            // Create copies of TX/MD blocks for filename, mimetype, and comment
            let at_filename_md: Option<MetaData> = if at.at_tx_filename != 0 {
                info.sharable.md_tx.get(&at.at_tx_filename).cloned()
            } else {
                None
            };

            let at_mimetype_md: Option<MetaData> = if at.at_tx_mimetype != 0 {
                info.sharable.md_tx.get(&at.at_tx_mimetype).cloned()
            } else {
                None
            };

            let at_comment_md: Option<MetaData> = if at.at_md_comment != 0 {
                info.sharable.md_tx.get(&at.at_md_comment).cloned()
            } else {
                None
            };

            // Create a modified attachment block with updated pointers
            let mut new_at = *at;

            // Calculate offsets for metadata blocks (they follow the AT block)
            let mut current_offset = at.at_len as i64;

            // Set filename pointer
            if let Some(md) = &at_filename_md {
                new_at.at_tx_filename = pointer + current_offset;
                current_offset += md.block.hdr_len as i64;
            } else {
                new_at.at_tx_filename = 0;
            }

            // Set mimetype pointer
            if let Some(md) = &at_mimetype_md {
                new_at.at_tx_mimetype = pointer + current_offset;
                current_offset += md.block.hdr_len as i64;
            } else {
                new_at.at_tx_mimetype = 0;
            }

            // Set comment pointer
            if let Some(md) = &at_comment_md {
                new_at.at_md_comment = pointer + current_offset;
                current_offset += md.block.hdr_len as i64;
            } else {
                new_at.at_md_comment = 0;
            }

            // Update the previous attachment's next pointer
            if let Some(last) = at_blocks.last_mut() {
                last.1.at_at_next = at_block_position;
            }

            // Advance pointer for this attachment block and its metadata
            pointer += current_offset;

            at_blocks.push((
                *orig_pos,
                new_at,
                embedded_data.clone(),
                at_filename_md,
                at_mimetype_md,
                at_comment_md,
            ));
        }

        // Last attachment's next pointer should be 0
        if let Some(last) = at_blocks.last_mut() {
            last.1.at_at_next = 0;
        }

        // Copy attachments to new_info
        for (orig_pos, at, embedded_data, _, _, _) in &at_blocks {
            new_info.at.insert(*orig_pos, (*at, embedded_data.clone()));
        }
    }

    // Copy SI blocks from source file
    // SI blocks are shared - multiple channels can reference the same SI block
    // We need to create a mapping from old positions to new positions
    type SiBlockEntry = (
        i64,
        Si4Block,
        Option<MetaData>,
        Option<MetaData>,
        Option<MetaData>,
    );
    let mut si_blocks: Vec<SiBlockEntry> = Vec::new();
    let mut si_position_map: HashMap<i64, i64> = HashMap::new();

    for (orig_pos, si) in info.sharable.si.iter() {
        let new_si_position = pointer;
        si_position_map.insert(*orig_pos, new_si_position);

        // Get the TX/MD blocks from source
        let si_name_md = if si.si_tx_name != 0 {
            info.sharable.md_tx.get(&si.si_tx_name).cloned()
        } else {
            None
        };

        let si_path_md = if si.si_tx_path != 0 {
            info.sharable.md_tx.get(&si.si_tx_path).cloned()
        } else {
            None
        };

        let si_comment_md = if si.si_md_comment != 0 {
            info.sharable.md_tx.get(&si.si_md_comment).cloned()
        } else {
            None
        };

        // Calculate block size and update pointer
        pointer += si.calculate_block_size();

        // Add TX/MD block sizes
        if let Some(ref md) = si_name_md {
            pointer += md.block.hdr_len as i64;
        }
        if let Some(ref md) = si_path_md {
            pointer += md.block.hdr_len as i64;
        }
        if let Some(ref md) = si_comment_md {
            pointer += md.block.hdr_len as i64;
        }

        si_blocks.push((*orig_pos, *si, si_name_md, si_path_md, si_comment_md));
    }

    let mut last_dg_pointer: i64 = pointer;
    new_info.hd_block.hd_dg_first = pointer;

    // builds meta data blocks for the new file
    for (_dg_position, dg) in info.dg.iter() {
        for (_record_id, cg) in dg.cg.iter() {
            let mut cg_cg_master: i64 = 0;

            // find master channel and start to write blocks for it
            if let Some(master_channel_name) = &cg.master_channel_name
                && let Some((
                    _master_name,
                    _dg_master_position,
                    (_cg_master_block_position, _record_id),
                    (_cn_master_block_position, cn_master_record_position),
                )) = info.get_channel_id(master_channel_name)
                && let Some(cn_master) = cg.cn.get(cn_master_record_position)
                && let Some(data) = mdf.get_channel_data(&cn_master.unique_name)
            {
                // Writing master channel
                cg_cg_master = pointer + 64; // after DGBlock
                last_dg_pointer = pointer;
                pointer = create_blocks(
                    &mut new_info,
                    &info,
                    pointer,
                    cg,
                    cn_master,
                    data,
                    &cg_cg_master,
                    true,
                    &si_position_map,
                )?;
            }

            // create the other non master channel blocks
            for (_cn_record_position, cn) in cg.cn.iter() {
                // not master channel
                if cn.block.cn_type != 2
                    && cn.block.cn_type != 3
                    && let Some(data) = mdf.get_channel_data(&cn.unique_name)
                {
                    last_dg_pointer = pointer;
                    pointer = create_blocks(
                        &mut new_info,
                        &info,
                        pointer,
                        cg,
                        cn,
                        data,
                        &cg_cg_master,
                        false,
                        &si_position_map,
                    )?;
                }
            }
        }
    }
    // last DG must point to null DGBlock
    if let Some(last_dg) = new_info.dg.get_mut(&last_dg_pointer) {
        last_dg.block.dg_dg_next = 0;
    }

    // Build position maps for DG/CG/CN remapping (needed for CH block element triplets)
    let mut dg_position_map: HashMap<i64, i64> = HashMap::new();
    let mut cg_position_map: HashMap<i64, i64> = HashMap::new();
    let mut cn_position_map: HashMap<i64, i64> = HashMap::new();
    for (name, (_, old_dg, (old_cg, _), (old_cn, _))) in info.channel_names_set.iter() {
        if let Some((_, new_dg, (new_cg, _), (new_cn, _))) = new_info.channel_names_set.get(name) {
            dg_position_map.insert(*old_dg, *new_dg);
            cg_position_map.insert(*old_cg, *new_cg);
            cn_position_map.insert(*old_cn, *new_cn);
        }
    }

    // Copy CH blocks (channel hierarchy) from source file
    type ChBlockEntry = (i64, Ch4Block, Option<MetaData>, Option<MetaData>);
    let mut ch_blocks: Vec<ChBlockEntry> = Vec::new();
    let mut ch_position_map: HashMap<i64, i64> = HashMap::new();

    if !info.ch.is_empty() {
        new_info.hd_block.hd_ch_first = pointer;

        // Sort source CH blocks by position for deterministic sibling/child link order
        let mut sorted_ch: Vec<(i64, &Ch4Block)> = info.ch.iter().map(|(k, v)| (*k, v)).collect();
        sorted_ch.sort_by_key(|(pos, _)| *pos);

        // Pass 1: Assign positions and collect metadata
        for (orig_pos, ch) in &sorted_ch {
            let new_ch_position = pointer;
            ch_position_map.insert(*orig_pos, new_ch_position);

            let ch_name_md = if ch.ch_tx_name != 0 {
                info.sharable.md_tx.get(&ch.ch_tx_name).cloned()
            } else {
                None
            };
            let ch_comment_md = if ch.ch_md_comment != 0 {
                info.sharable.md_tx.get(&ch.ch_md_comment).cloned()
            } else {
                None
            };

            pointer += ch.calculate_block_size();
            if let Some(ref md) = ch_name_md {
                pointer += md.block.hdr_len as i64;
            }
            if let Some(ref md) = ch_comment_md {
                pointer += md.block.hdr_len as i64;
            }

            ch_blocks.push((*orig_pos, (*ch).clone(), ch_name_md, ch_comment_md));
        }

        // Pass 2: Remap all links
        for (orig_pos, ch, name_md, comment_md) in ch_blocks.iter_mut() {
            // Remap sibling and child links
            ch.ch_ch_next = if ch.ch_ch_next > 0 {
                ch_position_map.get(&ch.ch_ch_next).copied().unwrap_or(0)
            } else {
                0
            };
            ch.ch_ch_first = if ch.ch_ch_first > 0 {
                ch_position_map.get(&ch.ch_ch_first).copied().unwrap_or(0)
            } else {
                0
            };

            // Remap TX name and MD comment to positions after the CH block itself
            let new_ch_pos = ch_position_map.get(orig_pos).copied().unwrap_or(0);
            let block_size = ch.calculate_block_size();
            let mut meta_offset = new_ch_pos + block_size;

            if let Some(md) = name_md.as_ref() {
                ch.ch_tx_name = meta_offset;
                meta_offset += md.block.hdr_len as i64;
            } else {
                ch.ch_tx_name = 0;
            }
            ch.ch_md_comment = if comment_md.is_some() { meta_offset } else { 0 };

            // Remap element triplets (DG, CG, CN positions)
            for i in 0..ch.ch_element_count as usize {
                let base = i * 3;
                if base + 2 < ch.ch_element.len() {
                    ch.ch_element[base] = dg_position_map
                        .get(&ch.ch_element[base])
                        .copied()
                        .unwrap_or(0);
                    ch.ch_element[base + 1] = cg_position_map
                        .get(&ch.ch_element[base + 1])
                        .copied()
                        .unwrap_or(0);
                    ch.ch_element[base + 2] = cn_position_map
                        .get(&ch.ch_element[base + 2])
                        .copied()
                        .unwrap_or(0);
                }
            }
        }

        // Store in new_info
        for (orig_pos, ch, _, _) in &ch_blocks {
            let new_pos = ch_position_map.get(orig_pos).copied().unwrap_or(0);
            new_info.ch.insert(new_pos, ch.clone());
        }
    }

    // thread writing the channels data first as block size can be unknown due to compression
    let (tx, rx) = bounded::<Vec<u8>>(n_channels);
    let fname = Arc::new(Mutex::new(file_name.to_string()));
    let sfname = Arc::clone(&fname);
    let writer_handle = thread::spawn(move || -> Result<(), Error> {
        let file_name = Arc::clone(&sfname);
        let file = file_name.lock();
        let f: File = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&*file)
            .context("Cannot create the file")?;
        let mut writer = BufWriter::new(&f);

        writer
            .seek(SeekFrom::Start(pointer as u64))
            .context("Could not reach position to write data blocks")?;
        for buffer in rx {
            writer
                .write_all(&buffer)
                .context("Could not write data blocks buffer")?;
        }
        writer.flush().context("Could not flush data blocks")?;
        Ok(())
    });

    let data_pointer = Arc::new(Mutex::new(pointer));
    new_info
        .dg
        .par_iter_mut()
        .try_for_each(|(_dg_block_position, dg)| -> Result<(), Error> {
            for (_rec_id, cg) in dg.cg.iter_mut() {
                for (_rec_pos, cn) in cg.cn.iter_mut() {
                    let dt = mdf.get_channel_data(&cn.unique_name);
                    if let Some(data) = dt {
                        let m = data.validity();
                        if !data.is_empty() && data.bit_count() > 0 {
                            // Check if this is a VLSD channel
                            let is_vlsd = cn.block.cn_type == 1 && is_vlsd_data(data);

                            if is_vlsd {
                                // VLSD channel: write SD block, set cn_data
                                let mut offset: i64 = 0;

                                let data_pointer = Arc::clone(&data_pointer);
                                let mut locked_data_pointer = data_pointer.lock();
                                cn.block.cn_data = *locked_data_pointer;
                                // For VLSD, dg_data is not used (set to 0)
                                dg.block.dg_data = 0;
                                let data_block = if compression {
                                    create_dz_sd(data, &mut offset)
                                        .context("failed creating dz or sd block")?
                                } else {
                                    create_sd(data, &mut offset)
                                        .context("failed creating sd block")?
                                };
                                *locked_data_pointer += offset;
                                let buffer =
                                    write_sd_block(cn.block.cn_data, data_block, offset as usize)?;
                                tx.send(buffer).context("Channel disconnected")?;
                                drop(locked_data_pointer);
                            } else {
                                // Regular channel: write DV/DZ block
                                let mut offset: i64 = 0;
                                let mut ld_block: Option<Ld4Block> = None;
                                if compression || m.is_some() {
                                    ld_block = create_ld(&m, &mut offset);
                                }

                                let data_block = if compression {
                                    create_dz_dv(data, &mut offset)
                                        .context("failed creating dz or dv block")?
                                } else {
                                    create_dv(data, &mut offset)
                                        .context("failed creating dv block")?
                                };

                                // invalid mask existing
                                let mut invalid_block: Option<(DataBlock, Vec<u8>)> = None;
                                if let Some(mask) = m {
                                    cg.block.cg_inval_bytes = 1; // one byte (u8) for invalid mask
                                    if let Some(ref mut ld) = ld_block {
                                        ld.ld_links.push(offset);
                                    }
                                    if compression {
                                        invalid_block = create_dz_di(&mask, &mut offset)
                                            .context("failed creating dz or di block")?;
                                    } else {
                                        invalid_block = create_di(&mask, &mut offset)
                                            .context("failed creating di block")?;
                                    }
                                }

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
                                )?;
                                tx.send(buffer).context("Channel disconnected")?;
                                drop(locked_data_pointer);
                            }
                        }
                    }
                }
            }
            Ok(())
        })?;
    drop(tx);
    writer_handle
        .join()
        .map_err(|e| anyhow::anyhow!("Data writer thread panicked: {:?}", e))
        .and_then(|r| r)?;

    let file_name = Arc::clone(&fname);
    let file = file_name.lock();
    let f: File = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&*file)
        .context("Cannot create the file")?;
    let mut writer = BufWriter::new(f);
    let mut buffer = Cursor::new(Vec::<u8>::with_capacity(pointer as usize));
    // IDBlock
    buffer
        .write_le(&new_info.id_block)
        .context("Could not write IdBlock")?;
    // Writes HDblock
    buffer
        .write_le(&new_info.hd_block)
        .context("Could not write HDBlock")?;
    // Writes FHBlocks (file history chain)
    for (fh, fh_comment) in &fh_blocks {
        buffer.write_le(fh).context("Could not write FHBlock")?;
        fh_comment.write(&mut buffer)?;
    }

    // Writes EVBLOCKs (events)
    for (_orig_pos, ev, ev_name_md, ev_comment_md) in &ev_blocks {
        // Write event block header
        let ev_header = Blockheader4Short {
            hdr_id: [35, 35, 69, 86], // ##EV
            hdr_gap: [0u8; 4],
            hdr_len: ev.calculate_block_size() as u64,
        };
        buffer
            .write_le(&ev_header)
            .context("Could not write EVBlock header")?;

        // Write the event block body
        buffer.write_le(ev).context("Could not write EVBlock")?;

        // Write event name TX block if present
        if let Some(name_md) = ev_name_md {
            name_md
                .write(&mut buffer)
                .context("Failed writing event name")?;
        }

        // Write event comment MD block if present
        if let Some(comment_md) = ev_comment_md {
            comment_md
                .write(&mut buffer)
                .context("Failed writing event comment")?;
        }
    }

    // Writes ATBLOCKs (attachments)
    for (_orig_pos, at, embedded_data, at_filename_md, at_mimetype_md, at_comment_md) in &at_blocks
    {
        // Write attachment block (AT4Block includes its own header)
        buffer.write_le(at).context("Could not write ATBlock")?;

        // Write embedded data if present
        if let Some(data) = embedded_data {
            buffer
                .write_all(data)
                .context("Could not write embedded attachment data")?;
        }

        // Write filename TX block if present
        if let Some(filename_md) = at_filename_md {
            filename_md
                .write(&mut buffer)
                .context("Failed writing attachment filename")?;
        }

        // Write mimetype TX block if present
        if let Some(mimetype_md) = at_mimetype_md {
            mimetype_md
                .write(&mut buffer)
                .context("Failed writing attachment mimetype")?;
        }

        // Write comment MD block if present
        if let Some(comment_md) = at_comment_md {
            comment_md
                .write(&mut buffer)
                .context("Failed writing attachment comment")?;
        }
    }

    // Writes SIBLOCKs (source information)
    for (orig_pos, si, si_name_md, si_path_md, si_comment_md) in &si_blocks {
        // Write SI block header
        let si_header = Blockheader4Short {
            hdr_id: [35, 35, 83, 73], // ##SI
            hdr_gap: [0u8; 4],
            hdr_len: si.calculate_block_size() as u64,
        };
        buffer
            .write_le(&si_header)
            .context("Could not write SIBlock header")?;

        // Write SI block links and data
        let si_links: u64 = 3; // always 3 links: name, path, comment
        buffer
            .write_le(&si_links)
            .context("Could not write SIBlock link count")?;

        // Calculate link positions
        let mut link_offset = si.calculate_block_size();
        let new_si_name = if si_name_md.is_some() {
            let pos = si_position_map.get(orig_pos).unwrap_or(&0) + link_offset;
            link_offset += si_name_md.as_ref().unwrap().block.hdr_len as i64;
            pos
        } else {
            0
        };
        let new_si_path = if si_path_md.is_some() {
            let pos = si_position_map.get(orig_pos).unwrap_or(&0) + link_offset;
            link_offset += si_path_md.as_ref().unwrap().block.hdr_len as i64;
            pos
        } else {
            0
        };
        let new_si_comment = if si_comment_md.is_some() {
            si_position_map.get(orig_pos).unwrap_or(&0) + link_offset
        } else {
            0
        };

        buffer
            .write_le(&new_si_name)
            .context("Could not write SIBlock si_tx_name")?;
        buffer
            .write_le(&new_si_path)
            .context("Could not write SIBlock si_tx_path")?;
        buffer
            .write_le(&new_si_comment)
            .context("Could not write SIBlock si_md_comment")?;

        // Write SI data members (type, bus_type, flags, reserved)
        buffer
            .write_le(&si.si_type)
            .context("Could not write SIBlock si_type")?;
        buffer
            .write_le(&si.si_bus_type)
            .context("Could not write SIBlock si_bus_type")?;
        buffer
            .write_le(&si.si_flags)
            .context("Could not write SIBlock si_flags")?;
        let si_reserved: [u8; 5] = [0u8; 5];
        buffer
            .write_all(&si_reserved)
            .context("Could not write SIBlock si_reserved")?;

        // Write SI name TX block if present
        if let Some(name_md) = si_name_md {
            name_md
                .write(&mut buffer)
                .context("Failed writing SI name")?;
        }

        // Write SI path TX block if present
        if let Some(path_md) = si_path_md {
            path_md
                .write(&mut buffer)
                .context("Failed writing SI path")?;
        }

        // Write SI comment MD block if present
        if let Some(comment_md) = si_comment_md {
            comment_md
                .write(&mut buffer)
                .context("Failed writing SI comment")?;
        }
    }

    // Writes CHBLOCKs (channel hierarchy)
    for (_orig_pos, ch, ch_name_md, ch_comment_md) in &ch_blocks {
        // Write CH block short header
        let ch_header = Blockheader4Short {
            hdr_id: [35, 35, 67, 72], // ##CH
            hdr_gap: [0u8; 4],
            hdr_len: ch.calculate_block_size() as u64,
        };
        buffer
            .write_le(&ch_header)
            .context("Could not write CHBlock header")?;

        // Write link count
        buffer
            .write_le(&ch.ch_links)
            .context("Could not write CHBlock link count")?;

        // Write fixed links
        buffer
            .write_le(&ch.ch_ch_next)
            .context("Could not write ch_ch_next")?;
        buffer
            .write_le(&ch.ch_ch_first)
            .context("Could not write ch_ch_first")?;
        buffer
            .write_le(&ch.ch_tx_name)
            .context("Could not write ch_tx_name")?;
        buffer
            .write_le(&ch.ch_md_comment)
            .context("Could not write ch_md_comment")?;

        // Write element links (DG/CG/CN triplets)
        for elem in &ch.ch_element {
            buffer
                .write_le(elem)
                .context("Could not write ch_element link")?;
        }

        // Write data members
        buffer
            .write_le(&ch.ch_element_count)
            .context("Could not write ch_element_count")?;
        buffer
            .write_le(&ch.ch_type)
            .context("Could not write ch_type")?;
        buffer
            .write_all(&ch.ch_reserved)
            .context("Could not write ch_reserved")?;

        // Write TX name block if present
        if let Some(name_md) = ch_name_md {
            name_md
                .write(&mut buffer)
                .context("Failed writing CH name")?;
        }
        // Write MD comment block if present
        if let Some(comment_md) = ch_comment_md {
            comment_md
                .write(&mut buffer)
                .context("Failed writing CH comment")?;
        }
    }

    // Writes DG+CG+CN blocks
    for (_position, dg) in new_info.dg.iter() {
        buffer
            .write_le(&dg.block)
            .context("Could not write CGBlock")?;
        for (_red_id, cg) in dg.cg.iter() {
            buffer
                .write_le(&cg.header)
                .context("Could not write CGBlock header")?;
            buffer
                .write_le(&cg.block)
                .context("Could not write CGBlock")?;
            for (_rec_pos, cn) in cg.cn.iter() {
                buffer
                    .write_le(&cn.header)
                    .context("Could not write CNBlock header")?;
                buffer
                    .write_le(&cn.block)
                    .context("Could not write CNBlock")?;
                // TX Block channel name
                if let Some(tx_name_metadata) = new_info.sharable.md_tx.get(&cn.block.cn_tx_name) {
                    tx_name_metadata
                        .write(&mut buffer)
                        .context("Failed writing tx name")?;
                }
                if let Some(tx_unit_metadata) = new_info.sharable.md_tx.get(&cn.block.cn_md_unit) {
                    tx_unit_metadata
                        .write(&mut buffer)
                        .context("Failed writing tx unit")?;
                }
                if let Some(tx_comment_metadata) =
                    new_info.sharable.md_tx.get(&cn.block.cn_md_comment)
                {
                    tx_comment_metadata
                        .write(&mut buffer)
                        .context("Failed writing tx comment")?;
                }
                // channel composition
                if let Some(compo) = &cn.composition {
                    write_composition(&mut buffer, compo)?;
                }
            }
        }
    }
    writer
        .write_all(&buffer.into_inner())
        .context("Could not write DG+CG+CN blocks")?;
    writer.flush().context("Could not flush file")?;
    Ok(Mdf {
        mdf_info: MdfInfo::V4(Box::new(new_info)),
    })
}

/// Writes a composition block (CA, DS, CL, CU, CV, CN) and any nested composition recursively.
/// Each block type has its own header format and serialization logic.
fn write_composition(buffer: &mut Cursor<Vec<u8>>, compo: &Composition) -> Result<()> {
    match &compo.block {
        Compo::CA(c) => {
            c.write_to(buffer).context("Could not write CA block")?;
        }
        Compo::DS(ds) => {
            let header = Blockheader4Short {
                hdr_id: [35, 35, 68, 83], // ##DS
                hdr_len: 16 + 8 + ds.ds_links * 8 + 8,
                ..Default::default()
            };
            buffer
                .write_le(&header)
                .context("Could not write DSBlock header")?;
            buffer.write_le(ds).context("Could not write DSBlock")?;
        }
        Compo::CL(cl) => {
            let header = Blockheader4Short {
                hdr_id: [35, 35, 67, 76], // ##CL
                hdr_len: 48,
                ..Default::default()
            };
            buffer
                .write_le(&header)
                .context("Could not write CLBlock header")?;
            buffer.write_le(cl).context("Could not write CLBlock")?;
        }
        Compo::CU(cu) => {
            let header = Blockheader4Short {
                hdr_id: [35, 35, 67, 85], // ##CU
                hdr_len: 16 + 8 + cu.cu_n_links * 8 + 4 + 4,
                ..Default::default()
            };
            buffer
                .write_le(&header)
                .context("Could not write CUBlock header")?;
            buffer.write_le(cu).context("Could not write CUBlock")?;
        }
        Compo::CV(cv) => {
            let header = Blockheader4Short {
                hdr_id: [35, 35, 67, 86], // ##CV
                hdr_len: 16 + 8 + cv.cv_n_links * 8 + 4 + 4 + cv.cv_option_count as u64 * 8,
                ..Default::default()
            };
            buffer
                .write_le(&header)
                .context("Could not write CVBlock header")?;
            buffer.write_le(cv).context("Could not write CVBlock")?;
        }
        Compo::CN(cn) => {
            // Nested CN composition: write the CN block header + block data
            buffer
                .write_le(&cn.header)
                .context("Could not write composition CN header")?;
            buffer
                .write_le(&cn.block)
                .context("Could not write composition CN block")?;
        }
    }
    // Handle recursive nested compositions
    if let Some(nested) = &compo.compo {
        write_composition(buffer, nested)?;
    }
    Ok(())
}

/// Calculate the serialized size of a composition tree (for position tracking)
fn calculate_composition_size(compo: &Composition) -> i64 {
    let block_size: i64 = match &compo.block {
        Compo::CA(c) => c.ca_len as i64,
        Compo::DS(ds) => (16 + 8 + ds.ds_links * 8 + 8) as i64,
        Compo::CL(_) => 48,
        Compo::CV(cv) => {
            (16 + 8 + cv.cv_n_links * 8 + 4 + 4 + cv.cv_option_count as u64 * 8) as i64
        }
        Compo::CU(cu) => (16 + 8 + cu.cu_n_links * 8 + 4 + 4) as i64,
        Compo::CN(cn) => cn.header.hdr_len as i64,
    };
    let nested = compo
        .compo
        .as_ref()
        .map_or(0, |n| calculate_composition_size(n));
    block_size + nested
}

/// Zero internal CN-referencing links in a composition tree.
/// Preserves block types, flags, and data members but zeroes links to CN blocks
/// from the old file (whose positions are invalid after writing).
fn zero_composition_links(compo: &mut Composition) {
    match &mut compo.block {
        Compo::DS(ds) => ds.links.iter_mut().for_each(|l| *l = 0),
        Compo::CL(cl) => {
            cl.cl_composition = 0;
            cl.cl_cn_size = 0;
        }
        Compo::CV(cv) => {
            cv.cv_cn_discriminator = 0;
            cv.cv_cn_option.iter_mut().for_each(|l| *l = 0);
        }
        Compo::CU(cu) => cu.cu_cn_member.iter_mut().for_each(|l| *l = 0),
        Compo::CN(cn) => {
            cn.block.cn_composition = 0;
            cn.block.cn_tx_name = 0;
            cn.block.cn_md_unit = 0;
            cn.block.cn_md_comment = 0;
            cn.block.cn_data = 0;
            cn.block.cn_cc_conversion = 0;
            cn.block.set_si_source(0);
        }
        Compo::CA(ca) => ca.prepare_for_write(),
    }
    if let Some(ref mut nested) = compo.compo {
        zero_composition_links(nested);
    }
}

/// Serializes DV/DZ data blocks (and optional LD/DI blocks) into a byte buffer.
/// The LD block links are adjusted to absolute file positions.
fn write_data_blocks(
    position: i64,
    mut ld_block: Option<Ld4Block>,
    data_block: (DataBlock, usize, Vec<u8>),
    invalid_block: Option<(DataBlock, Vec<u8>)>,
    offset: usize,
) -> Result<Vec<u8>> {
    let mut buffer = Cursor::new(vec![0u8; offset]);
    // Writes LD block
    if let Some(ref mut ld) = ld_block {
        let id_ld: [u8; 4] = [35, 35, 76, 68]; // ##LD
        buffer
            .write_le(&id_ld)
            .context("Could not write LDBlock id")?;
        ld.ld_links.iter_mut().for_each(|x| *x += position);
        ld.ld_n_links = ld.ld_links.len() as u64 + 1;
        buffer.write_le(ld).context("Could not write LDBlock")?;
    }

    // Writes DV or DZ block
    match data_block.0 {
        DataBlock::DvDi(dv_block) => {
            buffer
                .write_le(&dv_block)
                .context("Could not write DVBlock")?;
        }
        DataBlock::DZ(dz_block) => {
            let id_dz: [u8; 4] = [35, 35, 68, 90]; // ##DZ
            buffer
                .write_le(&id_dz)
                .context("Could not write DZDVBlock id")?;
            buffer
                .write_le(&dz_block)
                .context("Could not write DZDVBlock")?;
        }
    }
    buffer
        .write_all(&data_block.2)
        .context("Could not write data in DVBlock or DZBlock")?;
    // 8 byte align
    buffer
        .write_all(&vec![0; data_block.1])
        .context("Could not align written data to 8 bytes")?;

    // invalid mask existing
    if let Some((invalid_block, invalid_bytes)) = invalid_block {
        match invalid_block {
            DataBlock::DvDi(di_block) => {
                buffer
                    .write_le(&di_block)
                    .context("Could not write DIBlock")?;
            }
            DataBlock::DZ(dz_di_block) => {
                let id_dz: [u8; 4] = [35, 35, 68, 90]; // ##DZ
                buffer
                    .write_le(&id_dz)
                    .context("Could not write DZDIBlock id")?;
                buffer
                    .write_le(&dz_di_block)
                    .context("Could not write DZDIBlock")?;
            }
        }
        buffer
            .write_all(&invalid_bytes)
            .context("Could not write invalid data")?;
    }
    Ok(buffer.into_inner())
}

/// Serializes an SD/DZ block for VLSD channels into a byte buffer.
fn write_sd_block(
    _position: i64,
    data_block: (DataBlock, usize, Vec<u8>),
    offset: usize,
) -> Result<Vec<u8>> {
    let mut buffer = Cursor::new(vec![0u8; offset]);

    // Write SD or DZ block
    match data_block.0 {
        DataBlock::DvDi(sd_block) => {
            // SD block (uses same header format as DV/DI)
            buffer
                .write_le(&sd_block)
                .context("Could not write SDBlock")?;
        }
        DataBlock::DZ(dz_block) => {
            let id_dz: [u8; 4] = [35, 35, 68, 90]; // ##DZ
            buffer
                .write_le(&id_dz)
                .context("Could not write DZSDBlock id")?;
            buffer
                .write_le(&dz_block)
                .context("Could not write DZSDBlock")?;
        }
    }
    buffer
        .write_all(&data_block.2)
        .context("Could not write SD data")?;
    // 8 byte align
    buffer
        .write_all(&vec![0; data_block.1])
        .context("Could not align SD data to 8 bytes")?;

    Ok(buffer.into_inner())
}

/// Creates an LDBLOCK (list data block, spec Table 61) to wrap DV+DI block pairs.
/// Sets `ld_flags` bit 7 if an invalidation mask is present.
fn create_ld(m: &Option<NullBuffer>, offset: &mut i64) -> Option<Ld4Block> {
    let mut ld_block = Ld4Block::default();
    ld_block.ld_count = 1;
    ld_block.ld_sample_offset.push(0);
    if m.is_some() {
        ld_block.ld_n_links = (ld_block.ld_count * 2 + 1) as u64;
        ld_block.ld_flags = 1u8 << 7;
    } else {
        ld_block.ld_n_links = (ld_block.ld_count + 1) as u64;
        ld_block.ld_flags = 0b0;
    }
    ld_block.ld_len = 40 + (ld_block.ld_n_links * 8);
    *offset = ld_block.ld_len as i64;
    ld_block.ld_links.push(*offset);
    Some(ld_block)
}

/// Creates an uncompressed DVBLOCK (data values, spec Table 55 with ##DV ID).
fn create_dv(data: &ChannelData, offset: &mut i64) -> Result<(DataBlock, usize, Vec<u8>), Error> {
    let mut dv_block = Blockheader4 {
        hdr_id: [35, 35, 68, 86], // ##DV
        ..Default::default()
    };
    let data_bytes: Vec<u8> = data
        .to_bytes()
        .context("failed converting arraw data into bytes for dv block")?;
    let data_bytes_len = data_bytes.len();
    dv_block.hdr_len += data_bytes_len as u64;
    let byte_aligned = 8 - data_bytes_len % 8;

    *offset += dv_block.hdr_len as i64 + byte_aligned as i64;

    Ok((DataBlock::DvDi(dv_block), byte_aligned, data_bytes))
}

/// Writer-internal enum: either a compressed DZBLOCK or an uncompressed DV/DI/SD block.
#[derive(Debug, Clone)]
enum DataBlock {
    DZ(Dz4Block),
    DvDi(Blockheader4),
}

/// Creates a zlib-compressed DZBLOCK wrapping DV data (spec Table 57).
/// Falls back to uncompressed DV if compression increases size.
fn create_dz_dv(
    data: &ChannelData,
    offset: &mut i64,
) -> Result<(DataBlock, usize, Vec<u8>), Error> {
    let mut dz_block = Dz4Block::default();
    let bytes = data
        .to_bytes()
        .context("failed converting array data into bytes for dz or dv block")?;
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::best());
    encoder.write_all(&bytes).expect("Could not compress data");
    let mut data_bytes = encoder.finish().expect("failed finishing to compress data");
    dz_block.dz_data_length = data_bytes.len() as u64;
    let dv_dz_block: DataBlock;
    let byte_aligned: usize;
    let length = data.len();
    dz_block.dz_org_data_length = (length * data.byte_count() as usize) as u64;
    if dz_block.dz_org_data_length < dz_block.dz_data_length {
        (dv_dz_block, byte_aligned, data_bytes) = create_dv(data, offset)?;
    } else {
        byte_aligned = (8 - dz_block.dz_data_length % 8) as usize;
        dz_block.len = dz_block.dz_data_length + 48;
        *offset += dz_block.len as i64 + byte_aligned as i64;
        dv_dz_block = DataBlock::DZ(dz_block);
    };
    Ok((dv_dz_block, byte_aligned, data_bytes))
}

/// Creates an uncompressed DIBLOCK (data invalidation, ##DI) for the validity mask.
fn create_di(mask: &NullBuffer, offset: &mut i64) -> Result<Option<(DataBlock, Vec<u8>)>> {
    let mut dv_invalid_block = Blockheader4 {
        hdr_id: [35, 35, 68, 73], // ##DI
        ..Default::default()
    };
    let mask_length = mask.len();
    dv_invalid_block.hdr_len += mask_length as u64;
    let byte_aligned = 8 - mask_length % 8;
    let invalid_data: Vec<u8> = [
        mask.iter().map(|v| v as u8).collect::<Vec<u8>>(),
        vec![0; byte_aligned],
    ]
    .concat();
    *offset += dv_invalid_block.hdr_len as i64 + byte_aligned as i64;
    Ok(Some((DataBlock::DvDi(dv_invalid_block), invalid_data)))
}

/// Creates a zlib-compressed DZBLOCK wrapping DI invalidation data.
/// Falls back to uncompressed DI if compression increases size.
fn create_dz_di(
    mask: &NullBuffer,
    offset: &mut i64,
) -> Result<Option<(DataBlock, Vec<u8>)>, Error> {
    let mut dz_invalid_block = Dz4Block::default();
    dz_invalid_block.dz_org_data_length = mask.len() as u64;
    let mask_bytes: Vec<u8> = mask.iter().map(|v| v as u8).collect();
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::best());
    encoder
        .write_all(&mask_bytes)
        .expect("Could not compress invalid data");
    let mut data_bytes = encoder
        .finish()
        .expect("failed finishing to compress invalid data");
    dz_invalid_block.dz_data_length = data_bytes.len() as u64;
    if dz_invalid_block.dz_org_data_length < dz_invalid_block.dz_data_length {
        Ok(create_di(mask, offset)?)
    } else {
        dz_invalid_block.len = dz_invalid_block.dz_data_length + 48;
        let byte_aligned = 8 - dz_invalid_block.dz_data_length as usize % 8;
        data_bytes = [data_bytes, vec![0; byte_aligned]].concat();
        dz_invalid_block.dz_org_block_type = [68, 73]; // DI
        *offset += dz_invalid_block.len as i64 + byte_aligned as i64;
        Ok(Some((DataBlock::DZ(dz_invalid_block), data_bytes)))
    }
}

/// Creates one DG→CG→CN block set for a single channel (sorted layout).
/// Handles master/data channel distinction, VLSD detection, TX/MD metadata,
/// SI source remapping, CA array composition, and DS/CL/CV/CU preservation.
/// Skips channels with `bit_count==0` or empty data.
#[allow(clippy::too_many_arguments)]
fn create_blocks(
    new_info: &mut MdfInfo4,
    info: &MdfInfo4,
    mut pointer: i64,
    cg: &Cg4,
    cn: &Cn4,
    data: &ChannelData,
    cg_cg_master: &i64,
    master_flag: bool,
    si_position_map: &HashMap<i64, i64>,
) -> Result<i64> {
    let bit_count = data.bit_count();
    if !data.is_empty() && bit_count > 0 {
        let byte_count = data.byte_count();
        // no empty strings
        let mut dg_block = Dg4Block::default();
        let mut cg_block_header = default_short_header(BlockType::CG);
        let mut cg_block = Cg4Block::default();
        let cn_block_header = default_short_header(BlockType::CN);
        let mut cn_block = Cn4Block::default();
        // DG Block
        let dg_position = pointer;
        pointer += dg_block.dg_len as i64;
        dg_block.dg_cg_first = pointer;

        // CG Block
        let cg_position = pointer;
        if *cg_cg_master != 0 && !master_flag {
            cg_block.cg_links = 7; // with cg_cg_master
            cg_block_header.hdr_len = 112;
            cg_block.cg_cg_master = Some(*cg_cg_master);
            cg_block.cg_flags = 0b1000;
        }
        cg_block.cg_cycle_count = cg.block.cg_cycle_count;

        // CN Block
        let cn_position = pointer + cg_block_header.hdr_len as i64;
        let is_vlsd = is_vlsd_data(data);

        // For VLSD channels, cg_data_bytes = 0 (no fixed record, all data in SDBLOCK)
        if is_vlsd {
            cg_block.cg_data_bytes = 0;
        } else {
            cg_block.cg_data_bytes = byte_count;
        }
        if data.validity().is_some() {
            // One byte for invalid data as only one channel per CG
            cg_block.cg_inval_bytes = 1;
        }
        pointer += cg_block_header.hdr_len as i64;
        cg_block.cg_cn_first = pointer;

        // CN Block setup
        if master_flag {
            cn_block.cn_type = cn_type_writer(cn.block.cn_type, false)?; // master channel never VLSD
            if cn.block.cn_sync_type != 0 {
                cn_block.cn_sync_type = cn.block.cn_sync_type;
            } else {
                cn_block.cn_sync_type = 1; // Default is time
            }
        } else if is_vlsd {
            cn_block.cn_type = 1; // VLSD channel type
        }

        let machine_endian: bool = cfg!(target_endian = "big");

        cn_block.cn_data_type = data.data_type(machine_endian);

        // For VLSD, cn_bit_count can be 0 (variable length)
        if is_vlsd {
            cn_block.cn_bit_count = 0;
        } else {
            cn_block.cn_bit_count = bit_count;
        }

        pointer += cn_block_header.hdr_len as i64;

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
        if let Some(unit) = info.sharable.md_tx.get(&cn.block.cn_md_unit)
            && let Some(unit_str) = unit.get_tx_bytes()
        {
            let mut tx_unit_block = MetaData::new(MetaDataBlockType::TX, BlockType::CN);
            tx_unit_block.set_data_buffer(unit_str);
            cn_block.cn_md_unit = pointer;
            pointer += tx_unit_block.block.hdr_len as i64;
            new_info
                .sharable
                .md_tx
                .insert(cn_block.cn_md_unit, tx_unit_block);
        }

        // channel comment
        if let Some(comment) = info.sharable.md_tx.get(&cn.block.cn_md_comment)
            && let Some(comment_str) = comment.get_tx_bytes()
        {
            let mut tx_comment_block = MetaData::new(MetaDataBlockType::TX, BlockType::CN);
            tx_comment_block.set_data_buffer(comment_str);
            cn_block.cn_md_comment = pointer;
            pointer += tx_comment_block.block.hdr_len as i64;
            new_info
                .sharable
                .md_tx
                .insert(cn_block.cn_md_comment, tx_comment_block);
        }

        // Source information - map old SI position to new position
        let old_si_source = cn.block.get_si_source();
        if old_si_source != 0
            && let Some(&new_si_pos) = si_position_map.get(&old_si_source)
        {
            cn_block.set_si_source(new_si_pos);
        }

        // Channel array
        let data_ndim = data.ndim();
        let mut composition: Option<Composition> = None;
        if data_ndim > 1 {
            let data_dim_size: Vec<u64> = cn
                .data
                .shape()
                .0
                .iter()
                .skip(1)
                .map(|x| *x as u64)
                .collect();

            // Preserve source CA block if available, else create default
            let mut ca_block = if let Some(ref source_compo) = cn.composition
                && let Compo::CA(ref source_ca) = source_compo.block
            {
                let mut ca = (**source_ca).clone();
                ca.prepare_for_write();
                ca
            } else {
                Ca4Block::default()
            };

            cg_block.cg_data_bytes = cn.list_size as u32 * byte_count;
            cn_block.cn_composition = pointer;

            // Override dims from actual data shape (data_ndim includes sample count, ca_ndim does not)
            ca_block.ca_ndim = (data_ndim - 1) as u16;
            ca_block.ca_dim_size = data_dim_size;
            // Recalculate after dim changes
            ca_block.ca_len = ca_block.calculate_block_len();

            pointer += ca_block.ca_len as i64;
            composition = Some(Composition {
                block: Compo::CA(Box::new(ca_block)),
                compo: None,
            });
        } else if let Some(ref source_compo) = cn.composition {
            // Preserve non-CA composition from source (DS, CL, CV, CU, CN)
            match &source_compo.block {
                Compo::CA(_) => {} // Already handled above for data_ndim > 1
                _ => {
                    let mut cloned = source_compo.clone();
                    zero_composition_links(&mut cloned);
                    let compo_size = calculate_composition_size(&cloned);
                    cn_block.cn_composition = pointer;
                    pointer += compo_size;
                    composition = Some(cloned);
                }
            }
        }

        dg_block.dg_dg_next = pointer;
        // saves the blocks in the mdfinfo4 structure
        let new_cn = Cn4 {
            header: cn_block_header,
            unique_name: cn.unique_name.clone(),
            data: data_type_init(
                cn_block.cn_type,
                cn_block.cn_data_type,
                cg_block.cg_data_bytes,
                cn.list_size,
                cn_block.cn_flags,
            )
            .with_context(|| format!("failed initilising array for channel {}", cn.unique_name))?,
            block: cn_block,
            endian: Endianness::from(machine_endian),
            block_position: cn_position,
            pos_byte_beg: 0,
            n_bytes: cg_block.cg_data_bytes,
            composition,
            list_size: cn.list_size,
            shape: cn.shape.clone(),
            invalid_mask: None,
            event_template: None,
        };
        let mut new_cg = Cg4 {
            header: cg_block_header,
            block: cg_block,
            master_channel_name: cg.master_channel_name.clone(),
            cn: HashMap::new(),
            block_position: cg_position,
            channel_names: HashSet::new(),
            record_length: cg_block.cg_data_bytes,
            vlsd_cg: None,
            invalid_bytes: None,
            sr: Vec::new(),
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
    Ok(pointer)
}

/// Maps source cn_type to writer cn_type (spec Table 23).
/// VLSC (7) and other special types are converted to fixed-length (0).
/// Master channel types (2,3) are mapped to virtual master (2).
fn cn_type_writer(cn_type: u8, is_vlsd: bool) -> Result<u8> {
    // not all types are supported
    match cn_type {
        0 => Ok(0),
        1 => {
            if is_vlsd {
                Ok(1) // Keep VLSD type
            } else {
                Ok(0) // Fallback to fixed-length
            }
        }
        2 => Ok(2),
        3 => Ok(2),
        4 => Ok(0),
        5 => Ok(0),
        6 => Ok(0),
        7 => Ok(0), // VLSC converted to fixed-length (complex to write)
        _ => bail!("Unknown CN type"),
    }
}

/// Returns true if the channel data is variable-length (Utf8 or VariableSizeByteArray),
/// requiring VLSD storage with an SDBLOCK instead of inline record data.
fn is_vlsd_data(data: &ChannelData) -> bool {
    matches!(
        data,
        ChannelData::Utf8(_) | ChannelData::VariableSizeByteArray(_)
    )
}

/// Converts ChannelData to SDBLOCK format: each value is stored as a u32 length prefix
/// followed by the raw bytes (with null terminator for UTF-8 strings).
fn to_sd_bytes(data: &ChannelData) -> Result<Vec<u8>, Error> {
    match data {
        ChannelData::Utf8(a) => {
            let array = a.finish_cloned();
            let mut result = Vec::new();
            for i in 0..array.len() {
                let value = array.value(i);
                let bytes = value.as_bytes();
                // Add null terminator for UTF-8 strings (cn_data_type 7)
                let len_with_null = bytes.len() + 1;
                result.extend_from_slice(&(len_with_null as u32).to_le_bytes());
                result.extend_from_slice(bytes);
                result.push(0); // null terminator
            }
            Ok(result)
        }
        ChannelData::VariableSizeByteArray(a) => {
            let array = a.finish_cloned();
            let mut result = Vec::new();
            for i in 0..array.len() {
                let value = array.value(i);
                result.extend_from_slice(&(value.len() as u32).to_le_bytes());
                result.extend_from_slice(value);
            }
            Ok(result)
        }
        _ => bail!("to_sd_bytes called on non-VLSD data type"),
    }
}

/// Calculate the size of SDBLOCK data (for position calculation)
#[allow(dead_code)]
fn calculate_sd_size(data: &ChannelData) -> usize {
    match data {
        ChannelData::Utf8(a) => {
            let array = a.finish_cloned();
            let mut size = 0usize;
            for i in 0..array.len() {
                let value = array.value(i);
                // 4 bytes for length + string bytes + 1 for null terminator
                size += 4 + value.len() + 1;
            }
            size
        }
        ChannelData::VariableSizeByteArray(a) => {
            let array = a.finish_cloned();
            let mut size = 0usize;
            for i in 0..array.len() {
                let value = array.value(i);
                // 4 bytes for length + data bytes
                size += 4 + value.len();
            }
            size
        }
        _ => 0,
    }
}

/// Creates an uncompressed SDBLOCK (signal data, ##SD) for VLSD channels.
fn create_sd(data: &ChannelData, offset: &mut i64) -> Result<(DataBlock, usize, Vec<u8>), Error> {
    let mut sd_block = Blockheader4 {
        hdr_id: [35, 35, 83, 68], // ##SD
        ..Default::default()
    };
    let data_bytes = to_sd_bytes(data).context("failed converting data to SD format")?;
    let data_bytes_len = data_bytes.len();
    sd_block.hdr_len += data_bytes_len as u64;
    let byte_aligned = (8 - data_bytes_len % 8) % 8;

    *offset += sd_block.hdr_len as i64 + byte_aligned as i64;

    Ok((DataBlock::DvDi(sd_block), byte_aligned, data_bytes))
}

/// Creates a zlib-compressed DZBLOCK wrapping SD signal data.
/// Falls back to uncompressed SD if compression increases size.
fn create_dz_sd(
    data: &ChannelData,
    offset: &mut i64,
) -> Result<(DataBlock, usize, Vec<u8>), Error> {
    let mut dz_block = Dz4Block::default();
    let bytes = to_sd_bytes(data).context("failed converting data to SD format")?;
    dz_block.dz_org_data_length = bytes.len() as u64;

    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::best());
    encoder.write_all(&bytes).expect("Could not compress data");
    let mut data_bytes = encoder.finish().expect("failed finishing to compress data");
    dz_block.dz_data_length = data_bytes.len() as u64;

    let dz_sd_block: DataBlock;
    let byte_aligned: usize;

    if dz_block.dz_org_data_length < dz_block.dz_data_length {
        // Compression not beneficial, use uncompressed
        (dz_sd_block, byte_aligned, data_bytes) = create_sd(data, offset)?;
    } else {
        byte_aligned = (8 - dz_block.dz_data_length % 8) as usize % 8;
        dz_block.len = dz_block.dz_data_length + 48;
        dz_block.dz_org_block_type = [83, 68]; // SD
        *offset += dz_block.len as i64 + byte_aligned as i64;
        dz_sd_block = DataBlock::DZ(dz_block);
    }
    Ok((dz_sd_block, byte_aligned, data_bytes))
}
