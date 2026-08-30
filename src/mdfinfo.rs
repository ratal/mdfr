//! This module is reading the mdf file blocks (metadata)
//! mdfinfo module

use anyhow::Error;
use anyhow::{Context, Result, bail};
use arrow::array::Array;
use binrw::{BinReaderExt, binrw};
use codepage::to_encoding;
use encoding_rs::Encoding;
use log::{info, warn};
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::Read;
use std::path::PathBuf;
use std::str;
use std::sync::Arc;

pub mod mdfinfo3;

/// Map from channel name to its position tuple: (master_name, dg_pos, cg_pos, rec_id, cn_pos, rec_pos).
/// MDF3 files always have rec_pos = 0.
pub type ChannelsDb = HashMap<String, (Option<String>, i64, i64, u64, i64, i32)>;
pub mod mdfinfo4;
pub mod sym_buf_reader;

use binrw::io::Cursor;
use mdfinfo3::{MdfInfo3, SharableBlocks3, hd3_comment_parser, hd3_parser, parse_dg3};
use mdfinfo4::finalize::fix_cycle_counts;
use mdfinfo4::scanner;
use mdfinfo4::{
    MdfInfo4, SharableBlocks, build_channel_db, hd4_parser, parse_at4, parse_ch4, parse_dg4,
    parse_ev4, parse_fh,
};

use crate::data_holder::channel_data::ChannelData;
use crate::mdfwriter::mdfwriter3::convert3to4;

use self::mdfinfo3::build_channel_db3;
use self::mdfinfo4::{At4Block, Ch4Block, Ev4Block, FhBlock, Si4Block, Sr4Block};
use self::sym_buf_reader::SymBufReader;
use crate::mdfreader::{DataSignature, MasterSignature};

/// "UnFinMF " identifier bytes for unfinalized MDF files
const UNFINALIZED_ID: [u8; 8] = [85, 110, 70, 105, 110, 77, 70, 32];

/// Standard unfinalization flag bits (id_unfin_flags)
/// Bit 0: Update of cycle counters for CG-/CABLOCK required
const UNFIN_CG_CYCLE_COUNTERS: u16 = 1 << 0;
/// Bit 1: Update of cycle counters for SRBLOCKs required
const UNFIN_SR_CYCLE_COUNTERS: u16 = 1 << 1;
/// Bit 2: Update of length for last DTBLOCK required
const UNFIN_LAST_DT_LENGTH: u16 = 1 << 2;
/// Bit 3: Update of length for last RDBLOCK required
const UNFIN_LAST_RD_LENGTH: u16 = 1 << 3;
/// Bit 4: Update of last DLBLOCK in each chained list required
const UNFIN_LAST_DL_BLOCK: u16 = 1 << 4;
/// Bit 5: Update of cg_data_bytes and cg_inval_bytes in VLSD CGBLOCK required
const UNFIN_VLSD_CG_BYTES: u16 = 1 << 5;
/// Bit 6: Update of offset values for VLSD channel required
const UNFIN_VLSD_OFFSET: u16 = 1 << 6;
/// Bit 7: Update of cg_data_bytes and cg_inval_bytes in VLSC CGBLOCK required
const UNFIN_VLSC_CG_BYTES: u16 = 1 << 7;
/// Bit 8: Update of offset values for VLSC channel required
const UNFIN_VLSC_OFFSET: u16 = 1 << 8;

/// joins mdf versions 3.x and 4.x
#[derive(Debug)]
#[repr(C)]
pub enum MdfInfo {
    V3(Box<MdfInfo3>), // version 3.x
    V4(Box<MdfInfo4>), // version 4.x
}

/// Common Id block structure for both versions 2 and 3
#[derive(Debug, PartialEq, Eq, Clone)]
#[binrw]
#[allow(dead_code)]
#[repr(C)]
pub struct IdBlock {
    /// "MDF
    pub id_file_id: [u8; 8],
    /// version in char
    pub id_vers: [u8; 8],
    // logger id
    pub id_prog: [u8; 8],
    /// 0 Little endian, >= 1 Big endian, only valid for 3.x
    pub id_default_byteorder: u16,
    /// default floating point number. 0: IEEE754, 1: G_Float, 2: D_Float, only valid for 3.x
    id_floatingpointformat: u16,
    /// version number, valid for both 3.x and 4.x
    pub id_ver: u16,
    /// code page (only for version 3.3)
    pub id_codepage: u16,
    /// check
    id_check: [u8; 2],
    id_fill: [u8; 26],
    /// only valid for 4.x but can exist in 3.x
    id_unfin_flags: u16,
    /// only valid for 4.x but can exist in 3.x
    id_custom_unfin_flags: u16,
}

impl Default for IdBlock {
    fn default() -> Self {
        IdBlock {
            id_file_id: [77, 68, 70, 32, 32, 32, 32, 32], // "MDF     "
            id_vers: [52, 46, 51, 48, 32, 32, 32, 32],    // "4.30    "
            id_prog: [109, 100, 102, 114, 32, 32, 32, 32], // "mdfr    "
            id_ver: 430,
            id_default_byteorder: 0,
            id_floatingpointformat: 0,
            id_codepage: 0,
            id_check: [0, 0],
            id_fill: [0u8; 26],
            id_unfin_flags: 0,
            id_custom_unfin_flags: 0,
        }
    }
}

/// implements MdfInfo creation and manipulation functions
#[allow(dead_code)]
impl MdfInfo {
    /// creates new MdfInfo from file
    pub fn new(file_name: &str) -> Result<MdfInfo, Error> {
        let f: File = OpenOptions::new()
            .read(true)
            .write(false)
            .open(file_name)
            .with_context(|| format!("Cannot find the file {file_name}"))?;
        info!("Opened file {file_name}");

        let mut rdr = SymBufReader::new(&f);
        // Read beginning of ID Block
        let mut buf = [0u8; 64]; // reserved
        rdr.read_exact(&mut buf)
            .context("Could not read IdBlock buffer")?;
        let mut block = Cursor::new(buf);
        let id: IdBlock = block
            .read_le()
            .context("Could not parse buffer into IdBlock structure")?;
        info!("Read IdBlock");

        // Check for unfinalized MDF file (MDF 4.x feature, version-independent)
        let id_unfin_flags = id.id_unfin_flags;
        let id_custom_unfin_flags = id.id_custom_unfin_flags;
        let is_unfinalized = id.id_file_id == UNFINALIZED_ID || id_unfin_flags != 0;
        if is_unfinalized {
            warn!(
                "Unfinalized MDF file detected (id_unfin_flags=0x{id_unfin_flags:04X}, id_custom_unfin_flags=0x{id_custom_unfin_flags:04X}). \
                 Data may be incomplete or metadata may be inaccurate.",
            );
            if id_unfin_flags & UNFIN_CG_CYCLE_COUNTERS != 0 {
                warn!("  Bit 0: CG/CA cycle counters may be incorrect");
            }
            if id_unfin_flags & UNFIN_SR_CYCLE_COUNTERS != 0 {
                warn!("  Bit 1: SR cycle counters may be incorrect");
            }
            if id_unfin_flags & UNFIN_LAST_DT_LENGTH != 0 {
                warn!("  Bit 2: Last DT block length may be incorrect");
            }
            if id_unfin_flags & UNFIN_LAST_RD_LENGTH != 0 {
                warn!("  Bit 3: Last RD block length may be incorrect");
            }
            if id_unfin_flags & UNFIN_LAST_DL_BLOCK != 0 {
                warn!("  Bit 4: Last DL block may have incorrect dl_count or NIL links");
            }
            if id_unfin_flags & UNFIN_VLSD_CG_BYTES != 0 {
                warn!("  Bit 5: VLSD CG data_bytes/inval_bytes may be incorrect");
            }
            if id_unfin_flags & UNFIN_VLSD_OFFSET != 0 {
                warn!("  Bit 6: VLSD channel offsets may be incorrect");
            }
            if id_unfin_flags & UNFIN_VLSC_CG_BYTES != 0 {
                warn!("  Bit 7: VLSC CG data_bytes/inval_bytes may be incorrect");
            }
            if id_unfin_flags & UNFIN_VLSC_OFFSET != 0 {
                warn!("  Bit 8: VLSC channel offsets may be incorrect");
            }
            if id_custom_unfin_flags != 0 {
                warn!(
                    "  Custom finalization flags set (tool-specific): 0x{id_custom_unfin_flags:04X}"
                );
            }
        }

        // Depending of version different blocks
        let mdf_info: MdfInfo = if id.id_ver < 400 {
            let mut sharable: SharableBlocks3 = SharableBlocks3 {
                cc: HashMap::new(),
                ce: HashMap::new(),
            };
            // define encoding if any
            let encoding: &Encoding =
                to_encoding(id.id_codepage).unwrap_or(encoding_rs::WINDOWS_1252);

            // Read HD Block
            let (hd, position) =
                hd3_parser(&mut rdr, id.id_ver, encoding).context("failed parsing HD3 block")?;
            let (hd_comment, position) = hd3_comment_parser(&mut rdr, &hd, position, encoding)
                .context("failed parsing HD3 block comments")?;

            // Read DG Block
            let (mut dg, _, n_cg, n_cn) = parse_dg3(
                &mut rdr,
                hd.hd_dg_first,
                position,
                &mut sharable,
                id.id_default_byteorder,
                encoding,
            )
            .context("failed parsing mdf3 data")?;

            // make channel names unique, list channels and create master dictionnary
            let channel_names_set = build_channel_db3(&mut dg, &sharable, n_cg, n_cn);

            MdfInfo::V3(Box::new(MdfInfo3 {
                file_name: file_name.to_string(),
                id_block: id,
                encoding,
                hd_block: hd,
                hd_comment,
                dg,
                sharable,
                channel_names_set,
            }))
        } else {
            let mut sharable: SharableBlocks = SharableBlocks {
                md_tx: FxHashMap::default(),
                cc: FxHashMap::default(),
                si: FxHashMap::default(),
            };

            // Read HD block
            let (hd, position) =
                hd4_parser(&mut rdr, &mut sharable).context("failed parsing HD4 block")?;
            // parse HD metadata
            sharable.parse_hd_comments(hd.hd_md_comment);

            // FH block
            let (fh, position) = parse_fh(&mut rdr, &mut sharable, hd.hd_fh_first, position)
                .context("failed parsing File History")?;

            // AT Block read
            let (at, position) = parse_at4(&mut rdr, &mut sharable, hd.hd_at_first, position)
                .context("failed parsing attachments")?;

            // EV Block read
            let (ev, position) = parse_ev4(&mut rdr, &mut sharable, hd.hd_ev_first, position)
                .context("failed parsing events")?;

            // CH Block read
            let (ch, position) = parse_ch4(&mut rdr, &mut sharable, hd.hd_ch_first, position)
                .context("failed parsing channel hierarchy")?;

            // Read DG Block
            let (mut dg, _, n_cg, n_cn) =
                parse_dg4(&mut rdr, hd.hd_dg_first, position, &mut sharable)
                    .context("failed parsing mdf4 data")?;

            // Fallback: if the HD→DG link is nil and the chain yielded nothing,
            // scan the raw file bytes for ##DG signatures and re-parse from there.
            if dg.is_empty() && hd.hd_dg_first <= 0 {
                warn!("No DG blocks found via HD chain; attempting raw block scan");
                let dg_offsets = scanner::find_dg_blocks(&mut rdr);
                if !dg_offsets.is_empty() {
                    warn!("Scanner found {} DG block(s); re-parsing", dg_offsets.len());
                    for offset in dg_offsets {
                        if dg.contains_key(&offset) {
                            continue;
                        }
                        match parse_dg4(&mut rdr, offset, 0, &mut sharable) {
                            Ok((scanned_dg, _, _, _)) => dg.extend(scanned_dg),
                            Err(e) => warn!("Scanner: could not parse DG at 0x{offset:x}: {e:#}"),
                        }
                    }
                }
            }

            // If the file was not properly finalized, fix cycle counts in memory before
            // data reading begins. This corrects cg_cycle_count so the reader knows how
            // many records to expect from the truncated data blocks.
            if is_unfinalized
                && (id_unfin_flags
                    & (UNFIN_CG_CYCLE_COUNTERS | UNFIN_LAST_DT_LENGTH | UNFIN_LAST_DL_BLOCK))
                    != 0
            {
                match fix_cycle_counts(&mut dg, &mut rdr) {
                    Ok(n) => info!("finalize: corrected cycle counts for {n} channel groups"),
                    Err(e) => warn!("finalize: could not fix cycle counts: {e:#}"),
                }
            }

            // make channel names unique, list channels and create master dictionnary
            let channel_names_set = build_channel_db(&mut dg, &sharable, n_cg, n_cn);

            MdfInfo::V4(Box::new(MdfInfo4 {
                file_name: file_name.to_string(),
                id_block: id,
                hd_block: hd,
                fh,
                at,
                ev,
                dg,
                sharable,
                channel_names_set,
                ch,
                is_unfinalized,
            }))
        };
        info!("Finished reading metadata");
        Ok(mdf_info)
    }
    /// gets the version of mdf file
    pub fn get_version(&self) -> u16 {
        match self {
            MdfInfo::V3(mdfinfo3) => mdfinfo3.id_block.id_ver,
            MdfInfo::V4(mdfinfo4) => mdfinfo4.id_block.id_ver,
        }
    }
    /// returns true if the file was marked as unfinalized
    pub fn is_unfinalized(&self) -> bool {
        match self {
            MdfInfo::V3(_) => false,
            MdfInfo::V4(mdfinfo4) => mdfinfo4.is_unfinalized,
        }
    }
    /// returns the standard and custom unfinalization flags (0, 0) if finalized or MDF3
    pub fn get_unfin_flags(&self) -> (u16, u16) {
        match self {
            MdfInfo::V3(_) => (0, 0),
            MdfInfo::V4(mdfinfo4) => (
                mdfinfo4.id_block.id_unfin_flags,
                mdfinfo4.id_block.id_custom_unfin_flags,
            ),
        }
    }
    /// Index of this file within a recording sequence (MDF 4.3). None for MDF3 or missing.
    pub fn get_recorder_sequence_index(&self) -> Option<u64> {
        if let MdfInfo::V4(mdfinfo4) = self {
            mdfinfo4
                .sharable
                .get_hd_comments(mdfinfo4.hd_block.hd_md_comment)?
                .recorder_sequence_index()
        } else {
            None
        }
    }
    /// Index of this file within the recorder's file set (MDF 4.3). None for MDF3 or missing.
    pub fn get_recorder_file_index(&self) -> Option<u64> {
        if let MdfInfo::V4(mdfinfo4) = self {
            mdfinfo4
                .sharable
                .get_hd_comments(mdfinfo4.hd_block.hd_md_comment)?
                .recorder_file_index()
        } else {
            None
        }
    }
    /// True if this is the last file in the recorder sequence (MDF 4.3). None for MDF3 or missing.
    pub fn get_recorder_file_last(&self) -> Option<bool> {
        if let MdfInfo::V4(mdfinfo4) = self {
            mdfinfo4
                .sharable
                .get_hd_comments(mdfinfo4.hd_block.hd_md_comment)?
                .recorder_file_last()
        } else {
            None
        }
    }
    /// UUID of the recorder device (MDF 4.3). None for MDF3 or missing.
    pub fn get_recorder_uuid(&self) -> Option<String> {
        if let MdfInfo::V4(mdfinfo4) = self {
            mdfinfo4
                .sharable
                .get_hd_comments(mdfinfo4.hd_block.hd_md_comment)?
                .recorder_uuid()
                .map(str::to_string)
        } else {
            None
        }
    }
    /// UUID identifying the measurement (MDF 4.3). None for MdfInfo3 or missing.
    pub fn get_measurement_uuid(&self) -> Option<String> {
        if let MdfInfo::V4(mdfinfo4) = self {
            mdfinfo4
                .sharable
                .get_hd_comments(mdfinfo4.hd_block.hd_md_comment)?
                .measurement_uuid()
                .map(str::to_string)
        } else {
            None
        }
    }
    /// Author field from HD common_properties (MDF 4.3). None for MDF3 or missing.
    pub fn get_author(&self) -> Option<String> {
        if let MdfInfo::V4(mdfinfo4) = self {
            mdfinfo4
                .sharable
                .get_hd_comments(mdfinfo4.hd_block.hd_md_comment)?
                .author()
                .map(str::to_string)
        } else {
            None
        }
    }
    /// Department field from HD common_properties (MDF 4.3). None for MDF3 or missing.
    pub fn get_department(&self) -> Option<String> {
        if let MdfInfo::V4(mdfinfo4) = self {
            mdfinfo4
                .sharable
                .get_hd_comments(mdfinfo4.hd_block.hd_md_comment)?
                .department()
                .map(str::to_string)
        } else {
            None
        }
    }
    /// Project field from HD common_properties (MDF 4.3). None for MDF3 or missing.
    pub fn get_project(&self) -> Option<String> {
        if let MdfInfo::V4(mdfinfo4) = self {
            mdfinfo4
                .sharable
                .get_hd_comments(mdfinfo4.hd_block.hd_md_comment)?
                .project()
                .map(str::to_string)
        } else {
            None
        }
    }
    /// Subject field from HD common_properties (MDF 4.3). None for MDF3 or missing.
    pub fn get_subject(&self) -> Option<String> {
        if let MdfInfo::V4(mdfinfo4) = self {
            mdfinfo4
                .sharable
                .get_hd_comments(mdfinfo4.hd_block.hd_md_comment)?
                .subject()
                .map(str::to_string)
        } else {
            None
        }
    }
    /// returns channel's unit string
    pub fn get_channel_unit(&self, channel_name: &str) -> Result<Option<String>> {
        let unit: Option<String> = match self {
            MdfInfo::V3(mdfinfo3) => mdfinfo3.get_channel_unit(channel_name),
            MdfInfo::V4(mdfinfo4) => mdfinfo4
                .get_channel_unit(channel_name)
                .context("failed getting channel unit")?,
        };
        Ok(unit)
    }
    /// returns channel's description string
    pub fn get_channel_desc(&self, channel_name: &str) -> Result<Option<String>> {
        let desc: Option<String> = match self {
            MdfInfo::V3(mdfinfo3) => mdfinfo3.get_channel_desc(channel_name),
            MdfInfo::V4(mdfinfo4) => mdfinfo4
                .get_channel_desc(channel_name)
                .context("failed getting channel description")?,
        };
        Ok(desc)
    }
    /// returns channel's associated master channel name string
    pub fn get_channel_master(&self, channel_name: &str) -> Option<String> {
        let master: Option<String> = match self {
            MdfInfo::V3(mdfinfo3) => mdfinfo3.get_channel_master(channel_name),
            MdfInfo::V4(mdfinfo4) => mdfinfo4.get_channel_master(channel_name),
        };
        master
    }
    /// returns channel's associated master channel type string
    /// 0 = None (normal data channels), 1 = Time (seconds), 2 = Angle (radians),
    /// 3 = Distance (meters), 4 = Index (zero-based index values)
    pub fn get_channel_master_type(&self, channel_name: &str) -> u8 {
        let master: u8 = match self {
            MdfInfo::V3(mdfinfo3) => mdfinfo3.get_channel_master_type(channel_name),
            MdfInfo::V4(mdfinfo4) => mdfinfo4.get_channel_master_type(channel_name),
        };
        master
    }
    /// Returns event signal information for an event signal channel.
    /// Returns `None` if the channel is not an event signal channel or if the template is not available.
    pub fn get_event_signal_info(&self, channel_name: &str) -> Option<&mdfinfo4::Ev4Block> {
        match self {
            MdfInfo::V3(_) => None,
            MdfInfo::V4(mdfinfo4) => mdfinfo4.get_event_signal_info(channel_name),
        }
    }
    /// Returns true if the channel is a synchronization channel.
    /// Sync channels reference an ATBLOCK (attachment) rather than containing raw data.
    pub fn is_sync_channel(&self, channel_name: &str) -> bool {
        match self {
            MdfInfo::V3(_) => false,
            MdfInfo::V4(mdfinfo4) => mdfinfo4.is_sync_channel(channel_name),
        }
    }
    /// Returns the precision of a channel if the precision flag is set.
    /// Returns `None` if the channel is not found or precision is not specified.
    pub fn get_channel_precision(&self, channel_name: &str) -> Option<u8> {
        match self {
            MdfInfo::V3(_) => None,
            MdfInfo::V4(mdfinfo4) => mdfinfo4.get_channel_precision(channel_name),
        }
    }
    /// Returns the minimum value of the valid range for a channel.
    /// Returns `None` if the channel is not found or range is not specified.
    pub fn get_channel_range_min(&self, channel_name: &str) -> Option<f64> {
        match self {
            MdfInfo::V3(mdfinfo3) => mdfinfo3.get_channel_range_min(channel_name),
            MdfInfo::V4(mdfinfo4) => mdfinfo4.get_channel_range_min(channel_name),
        }
    }
    /// Returns the maximum value of the valid range for a channel.
    /// Returns `None` if the channel is not found or range is not specified.
    pub fn get_channel_range_max(&self, channel_name: &str) -> Option<f64> {
        match self {
            MdfInfo::V3(mdfinfo3) => mdfinfo3.get_channel_range_max(channel_name),
            MdfInfo::V4(mdfinfo4) => mdfinfo4.get_channel_range_max(channel_name),
        }
    }
    /// Returns the sampling rate in seconds if set (non-zero).
    /// Returns `None` if the channel is not found or sampling rate is not specified.
    pub fn get_channel_sampling_rate(&self, channel_name: &str) -> Option<f64> {
        match self {
            MdfInfo::V3(mdfinfo3) => mdfinfo3.get_channel_sampling_rate(channel_name),
            MdfInfo::V4(_) => None,
        }
    }
    /// Returns the minimum limit for a channel.
    /// Returns `None` if the channel is not found or limit is not specified.
    pub fn get_channel_limit_min(&self, channel_name: &str) -> Option<f64> {
        match self {
            MdfInfo::V3(_) => None,
            MdfInfo::V4(mdfinfo4) => mdfinfo4.get_channel_limit_min(channel_name),
        }
    }
    /// Returns the maximum limit for a channel.
    /// Returns `None` if the channel is not found or limit is not specified.
    pub fn get_channel_limit_max(&self, channel_name: &str) -> Option<f64> {
        match self {
            MdfInfo::V3(_) => None,
            MdfInfo::V4(mdfinfo4) => mdfinfo4.get_channel_limit_max(channel_name),
        }
    }
    /// Returns the minimum extended limit for a channel.
    /// Returns `None` if the channel is not found or extended limit is not specified.
    pub fn get_channel_limit_ext_min(&self, channel_name: &str) -> Option<f64> {
        match self {
            MdfInfo::V3(_) => None,
            MdfInfo::V4(mdfinfo4) => mdfinfo4.get_channel_limit_ext_min(channel_name),
        }
    }
    /// Returns the maximum extended limit for a channel.
    /// Returns `None` if the channel is not found or extended limit is not specified.
    pub fn get_channel_limit_ext_max(&self, channel_name: &str) -> Option<f64> {
        match self {
            MdfInfo::V3(_) => None,
            MdfInfo::V4(mdfinfo4) => mdfinfo4.get_channel_limit_ext_max(channel_name),
        }
    }
    /// Returns a reference to the CNBLOCK for a channel (MDF4 only, internal use).
    pub(crate) fn get_cn_block(&self, channel_name: &str) -> Option<&mdfinfo4::Cn4> {
        match self {
            MdfInfo::V3(_) => None,
            MdfInfo::V4(mdfinfo4) => mdfinfo4.get_cn_block(channel_name),
        }
    }
    /// returns the full channel-name → position-tuple map.
    /// Tuple: (master_name, dg_pos, cg_pos, rec_id, cn_pos, rec_pos)
    /// MDF3: rec_pos is always 0.
    pub fn get_channels_db(&self) -> ChannelsDb {
        match self {
            MdfInfo::V3(m) => m
                .channel_names_set
                .iter()
                .map(|(name, (master, dg, (cg, rec_id), cn))| {
                    (
                        name.clone(),
                        (
                            master.clone(),
                            *dg as i64,
                            *cg as i64,
                            *rec_id as u64,
                            *cn as i64,
                            0i32,
                        ),
                    )
                })
                .collect(),
            MdfInfo::V4(m) => m
                .channel_names_set
                .iter()
                .map(|(name, (master, dg, (cg, rec_id), (cn, rec_pos)))| {
                    (
                        name.clone(),
                        (master.clone(), *dg, *cg, *rec_id, *cn, *rec_pos),
                    )
                })
                .collect(),
        }
    }
    /// returns measurement start timestamp in nanoseconds since Unix epoch (1970-01-01 UTC)
    pub fn get_start_time_ns(&self) -> u64 {
        match self {
            MdfInfo::V4(m) => m.hd_block.hd_start_time_ns,
            MdfInfo::V3(m) => m.hd_block.hd_start_time_ns.unwrap_or(0),
        }
    }
    /// returns timezone + DST offset in minutes; 0 when not recorded or not valid
    pub fn get_tz_offset_min(&self) -> i16 {
        match self {
            MdfInfo::V4(m) => {
                if m.hd_block.hd_time_flags & 0b10 != 0 {
                    m.hd_block.hd_tz_offset_min + m.hd_block.hd_dst_offset_min
                } else {
                    0
                }
            }
            MdfInfo::V3(m) => m.hd_block.hd_time_offset.unwrap_or(0),
        }
    }
    /// returns a set of all channel names contained in file
    pub fn get_channel_names_set(&self) -> HashSet<String> {
        let channel_list: HashSet<String> = match self {
            MdfInfo::V3(mdfinfo3) => mdfinfo3.get_channel_names_set(),
            MdfInfo::V4(mdfinfo4) => mdfinfo4.get_channel_names_set(),
        };
        channel_list
    }
    /// returns the set of channel names that are in same channel group as input channel name
    pub fn get_channel_names_cg_set(&self, channel_name: &str) -> HashSet<String> {
        let channel_list: HashSet<String> = match self {
            MdfInfo::V3(mdfinfo3) => mdfinfo3.get_channel_names_cg_set(channel_name),
            MdfInfo::V4(mdfinfo4) => mdfinfo4.get_channel_names_cg_set(channel_name),
        };
        channel_list
    }
    /// returns a dict of master names keys for which values are a set of associated channel names
    pub fn get_master_channel_names_set(&self) -> HashMap<Option<String>, HashSet<String>> {
        let channel_master_list: HashMap<Option<String>, HashSet<String>> = match self {
            MdfInfo::V3(mdfinfo3) => mdfinfo3.get_master_channel_names_set(),
            MdfInfo::V4(mdfinfo4) => mdfinfo4.get_master_channel_names_set(),
        };
        channel_master_list
    }
    /// Clears all data arrays
    pub fn clear_channel_data_from_memory(&mut self, channel_names: HashSet<String>) -> Result<()> {
        match self {
            MdfInfo::V3(mdfinfo3) => {
                mdfinfo3
                    .clear_channel_data_from_memory(channel_names.clone())
                    .with_context(|| {
                        format!(
                            "failed clearing channel data from memory for {:?}",
                            channel_names.clone(),
                        )
                    })?;
            }
            MdfInfo::V4(mdfinfo4) => {
                mdfinfo4
                    .clear_channel_data_from_memory(channel_names.clone())
                    .with_context(|| {
                        format!(
                            "failed clearing channel data from memory for {:?}",
                            channel_names.clone(),
                        )
                    })?;
            }
        }
        Ok(())
    }
    /// directly replaces a single channel's ChannelData (bypasses try_from, preserves Complex/TensorArrow)
    pub fn replace_channel_data(&mut self, channel_name: &str, data: ChannelData) -> Result<()> {
        match self {
            MdfInfo::V3(m) => m.replace_channel_data(channel_name, data)?,
            MdfInfo::V4(m) => m.replace_channel_data(channel_name, data)?,
        }
        Ok(())
    }
    /// replaces each channel's data with the slice [start_idx, end_idx)
    pub fn slice_channels(
        &mut self,
        channel_names: &HashSet<String>,
        start_idx: usize,
        end_idx: usize,
    ) -> Result<()> {
        match self {
            MdfInfo::V3(m) => m.slice_channels(channel_names, start_idx, end_idx)?,
            MdfInfo::V4(m) => m.slice_channels(channel_names, start_idx, end_idx)?,
        }
        Ok(())
    }
    /// Eagerly converts all channels to physical values.
    pub fn convert_all_channels(&mut self) -> Result<(), Error> {
        match self {
            MdfInfo::V3(mdfinfo3) => mdfinfo3.convert_all_channels(),
            MdfInfo::V4(mdfinfo4) => mdfinfo4.convert_all_channels(),
        }
    }
    /// Converts one channels to physical values.
    pub fn convert_channel<'a>(&'a mut self, channel_name: &'a str) -> Result<(), Error> {
        match self {
            MdfInfo::V3(mdfinfo3) => mdfinfo3.convert_channel(channel_name),
            MdfInfo::V4(mdfinfo4) => mdfinfo4.convert_channel(channel_name),
        }
    }
    /// returns channel's data ndarray. If data were not yet converted, keeps converted data into memory
    pub fn get_channel_data<'a>(&'a mut self, channel_name: &'a str) -> Option<&'a ChannelData> {
        match self {
            MdfInfo::V3(mdfinfo3) => mdfinfo3.get_channel_data(channel_name),
            MdfInfo::V4(mdfinfo4) => mdfinfo4.get_channel_data(channel_name),
        }
    }
    /// returns channel's data ndarray, does not modify
    pub fn get_channel_converted_data<'a>(&'a self, channel_name: &'a str) -> Option<ChannelData> {
        match self {
            MdfInfo::V3(mdfinfo3) => mdfinfo3.get_channel_converted_data(channel_name),
            MdfInfo::V4(mdfinfo4) => mdfinfo4.get_channel_converted_data(channel_name),
        }
    }
    /// Adds a new channel in memory (no file modification)
    pub fn add_channel(
        &mut self,
        channel_name: String,
        data: ChannelData,
        data_signature: DataSignature,
        master: MasterSignature,
        unit: Option<String>,
        description: Option<String>,
    ) -> Result<(), Error> {
        match self {
            MdfInfo::V3(mdfinfo3) => {
                let mut file_name = PathBuf::from(mdfinfo3.file_name.as_str());
                file_name.set_extension("mf4");
                let mut mdf4 = convert3to4(mdfinfo3, &file_name.to_string_lossy())
                    .context("failed converting mdf3 into mdf4 when adding new channel")?;
                mdf4.add_channel(
                    channel_name,
                    data,
                    data_signature,
                    master,
                    unit,
                    description,
                )?;
            }
            MdfInfo::V4(mdfinfo4) => {
                mdfinfo4.add_channel(
                    channel_name,
                    data,
                    data_signature,
                    master,
                    unit,
                    description,
                )?;
            }
        }
        Ok(())
    }
    /// Convert mdf verion 3.x to 4.2
    /// Require file name parameter but no file written
    pub fn convert3to4(&mut self, file_name: &str) -> Result<MdfInfo, Error> {
        match self {
            MdfInfo::V3(mdfinfo3) => Ok(MdfInfo::V4(Box::new(
                convert3to4(mdfinfo3, file_name).context("failed converting mdf version 3 to 4")?,
            ))),
            MdfInfo::V4(_) => bail!("file is already a mdf version 4.x"),
        }
    }
    /// defines channel's data in memory
    pub fn set_channel_data(
        &mut self,
        channel_name: &str,
        data: Arc<dyn Array>,
    ) -> Result<(), Error> {
        match self {
            MdfInfo::V3(mdfinfo3) => {
                let mut file_name = PathBuf::from(mdfinfo3.file_name.as_str());
                file_name.set_extension("mf4");
                let mut mdf4 = convert3to4(mdfinfo3, &file_name.to_string_lossy())?;
                mdf4.set_channel_data(channel_name, data)
                    .context("failed setting channel data")?;
            }
            MdfInfo::V4(mdfinfo4) => mdfinfo4
                .set_channel_data(channel_name, data)
                .context("failed setting channel data")?,
        }
        Ok(())
    }
    /// Sets the channel's related master channel type in memory
    pub fn set_channel_master_type(
        &mut self,
        master_name: &str,
        master_type: u8,
    ) -> Result<(), Error> {
        match self {
            MdfInfo::V3(mdfinfo3) => {
                let mut file_name = PathBuf::from(mdfinfo3.file_name.as_str());
                file_name.set_extension("mf4");
                let mut mdf4 = convert3to4(mdfinfo3, &file_name.to_string_lossy()).context(
                    "failed converting mdf3 into mdf4 when setting new channel master type",
                )?;
                mdf4.set_channel_master_type(master_name, master_type);
            }
            MdfInfo::V4(mdfinfo4) => mdfinfo4.set_channel_master_type(master_name, master_type),
        }
        Ok(())
    }
    /// Removes a channel in memory (no file modification)
    pub fn remove_channel(&mut self, channel_name: &str) {
        match self {
            MdfInfo::V3(mdfinfo3) => mdfinfo3.remove_channel(channel_name),
            MdfInfo::V4(mdfinfo4) => mdfinfo4.remove_channel(channel_name),
        }
    }
    /// Renames a channel's name in memory
    pub fn rename_channel(&mut self, channel_name: &str, new_name: &str) {
        match self {
            MdfInfo::V3(mdfinfo3) => mdfinfo3.rename_channel(channel_name, new_name),
            MdfInfo::V4(mdfinfo4) => mdfinfo4.rename_channel(channel_name, new_name),
        }
    }
    /// Sets the channel unit in memory
    pub fn set_channel_unit(&mut self, channel_name: &str, unit: &str) {
        match self {
            MdfInfo::V3(mdfinfo3) => mdfinfo3.set_channel_unit(channel_name, unit),
            MdfInfo::V4(mdfinfo4) => mdfinfo4.set_channel_unit(channel_name, unit),
        }
    }
    /// Sets the channel description in memory
    pub fn set_channel_desc(&mut self, channel_name: &str, desc: &str) {
        match self {
            MdfInfo::V3(mdfinfo3) => mdfinfo3.set_channel_desc(channel_name, desc),
            MdfInfo::V4(mdfinfo4) => mdfinfo4.set_channel_desc(channel_name, desc),
        }
    }
    /// get typed comment from position
    pub fn get_md_comment(
        &mut self,
        position: i64,
    ) -> Option<&crate::mdfinfo::mdfinfo4::MdComment> {
        match self {
            MdfInfo::V3(_mdfinfo3) => None,
            MdfInfo::V4(mdfinfo4) => mdfinfo4.sharable.get_md_comment(position),
        }
    }
    /// get tx from position
    pub fn get_tx(&self, position: i64) -> Result<Option<String>> {
        match self {
            MdfInfo::V3(_mdfinfo3) => Ok(None),
            MdfInfo::V4(mdfinfo4) => mdfinfo4.sharable.get_tx(position),
        }
    }
    /// list attachments
    pub fn list_attachments(&mut self) -> String {
        match self {
            MdfInfo::V3(_) => "".to_string(),
            MdfInfo::V4(mdfinfo4) => mdfinfo4.list_attachments(),
        }
    }
    /// get attachment block
    pub fn get_attachment_block(&self, position: i64) -> Option<At4Block> {
        match self {
            MdfInfo::V3(_) => None,
            MdfInfo::V4(mdfinfo4) => mdfinfo4.get_attachment_block(position),
        }
    }
    /// get all attachement blocks
    pub fn get_attachement_blocks(&self) -> Option<HashMap<i64, At4Block>> {
        match self {
            MdfInfo::V3(_) => None,
            MdfInfo::V4(mdfinfo4) => Some(mdfinfo4.get_attachment_blocks()),
        }
    }
    /// get embedded data in attachment
    pub fn get_attachment_embedded_data(&self, position: i64) -> Option<Vec<u8>> {
        match self {
            MdfInfo::V3(_) => None,
            MdfInfo::V4(mdfinfo4) => mdfinfo4.get_attachment_embedded_data(position),
        }
    }
    /// list events
    pub fn list_events(&mut self) -> String {
        match self {
            MdfInfo::V3(_) => "".to_string(),
            MdfInfo::V4(mdfinfo4) => mdfinfo4.list_events(),
        }
    }
    /// get event block
    pub fn get_event_block(&self, position: i64) -> Option<Ev4Block> {
        match self {
            MdfInfo::V3(_) => None,
            MdfInfo::V4(mdfinfo4) => mdfinfo4.get_event_block(position),
        }
    }
    /// get all event blocks
    pub fn get_event_blocks(&self) -> Option<HashMap<i64, Ev4Block>> {
        match self {
            MdfInfo::V3(_) => None,
            MdfInfo::V4(mdfinfo4) => Some(mdfinfo4.get_event_blocks()),
        }
    }
    /// list file history entries
    pub fn list_file_history(&mut self) -> String {
        match self {
            MdfInfo::V3(_) => String::new(),
            MdfInfo::V4(mdfinfo4) => mdfinfo4.list_file_history(),
        }
    }
    /// get file history blocks
    pub fn get_file_history_blocks(&self) -> Option<Vec<FhBlock>> {
        match self {
            MdfInfo::V3(_) => None,
            MdfInfo::V4(mdfinfo4) => Some(mdfinfo4.fh.clone()),
        }
    }
    /// Get a channel hierarchy block from its position (MDF 4.x only)
    pub fn get_channel_hierarchy_block(&self, position: i64) -> Option<Ch4Block> {
        match self {
            MdfInfo::V3(_) => None,
            MdfInfo::V4(mdfinfo4) => mdfinfo4.get_channel_hierarchy_block(position),
        }
    }
    /// Get all channel hierarchy blocks (MDF 4.x only)
    pub fn get_channel_hierarchy_blocks(&self) -> Option<HashMap<i64, Ch4Block>> {
        match self {
            MdfInfo::V3(_) => None,
            MdfInfo::V4(mdfinfo4) => Some(mdfinfo4.get_channel_hierarchy_blocks()),
        }
    }
    /// List channel hierarchy in a human-readable format (MDF 4.x only)
    pub fn list_channel_hierarchy(&self) -> String {
        match self {
            MdfInfo::V3(_) => String::new(),
            MdfInfo::V4(mdfinfo4) => mdfinfo4.list_channel_hierarchy(),
        }
    }
    /// List source information blocks (MDF 4.x only)
    pub fn list_source_information(&self) -> String {
        match self {
            MdfInfo::V3(_) => String::new(),
            MdfInfo::V4(mdfinfo4) => mdfinfo4.list_source_information(),
        }
    }
    /// List sample reduction blocks for all channel groups (MDF 4.x only)
    pub fn list_sample_reductions(&self) -> String {
        match self {
            MdfInfo::V3(_) => String::new(),
            MdfInfo::V4(mdfinfo4) => mdfinfo4.list_sample_reductions(),
        }
    }
    /// Get all source information blocks (MDF 4.x only)
    pub fn get_source_information_blocks(&self) -> Option<FxHashMap<i64, Si4Block>> {
        match self {
            MdfInfo::V3(_) => None,
            MdfInfo::V4(mdfinfo4) => Some(mdfinfo4.get_source_information_blocks()),
        }
    }
    /// Get all sample reduction blocks across all channel groups (MDF 4.x only)
    /// Returns a vector of (dg_position, rec_id, sr_blocks) tuples
    pub fn get_sample_reduction_blocks(&self) -> Option<Vec<(i64, u64, Vec<Sr4Block>)>> {
        match self {
            MdfInfo::V3(_) => None,
            MdfInfo::V4(mdfinfo4) => Some(mdfinfo4.get_sample_reduction_blocks()),
        }
    }
}

impl fmt::Display for MdfInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MdfInfo::V3(mdfinfo3) => write!(f, "{mdfinfo3}"),
            MdfInfo::V4(mdfinfo4) => write!(f, "{mdfinfo4}"),
        }
    }
}
