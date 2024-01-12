//! This module is reading the mdf file blocks (metadata)
//! mdfinfo module

use anyhow::{bail, Context, Result};
use arrow2::array::Array;
use binrw::{binrw, BinReaderExt};
use codepage::to_encoding;
use encoding_rs::Encoding;
use log::info;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::Read;
use std::path::PathBuf;
use std::str;

pub mod mdfinfo3;
pub mod mdfinfo4;
pub mod sym_buf_reader;

use binrw::io::Cursor;
use mdfinfo3::{hd3_comment_parser, hd3_parser, parse_dg3, MdfInfo3, SharableBlocks3};
use mdfinfo4::{
    build_channel_db, hd4_parser, parse_at4, parse_dg4, parse_ev4, parse_fh, MdfInfo4,
    SharableBlocks,
};

use crate::mdfwriter::mdfwriter3::convert3to4;

use self::mdfinfo3::build_channel_db3;
use self::mdfinfo4::{At4Block, Ev4Block, FhBlock};
use self::sym_buf_reader::SymBufReader;
use crate::mdfreader::{DataSignature, MasterSignature};

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
            id_vers: [52, 46, 50, 48, 32, 32, 32, 32],    // "4.20    "
            id_prog: [109, 100, 102, 114, 32, 32, 32, 32], // "mdfr    "
            id_ver: 420,
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
    pub fn new(file_name: &str) -> Result<MdfInfo> {
        let f: File = OpenOptions::new()
            .read(true)
            .write(false)
            .open(file_name)
            .with_context(|| format!("Cannot find the file {file_name}"))?;
        info!("Opened file {}", file_name);

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
                hd3_parser(&mut rdr, id.id_ver, encoding).context("Failed reading HD3 Block")?;
            let (hd_comment, position) = hd3_comment_parser(&mut rdr, &hd, position, encoding)
                .context("Failed reading HD3 comment Block")?;

            // Read DG Block
            let (mut dg, _, n_cg, n_cn) = parse_dg3(
                &mut rdr,
                hd.hd_dg_first,
                position,
                &mut sharable,
                id.id_default_byteorder,
                encoding,
            )
            .context("Failed reading DG3 Block and related blocks including record data")?;

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
                md_tx: HashMap::new(),
                cc: HashMap::new(),
                si: HashMap::new(),
            };

            // Read HD block
            let (hd, position) =
                hd4_parser(&mut rdr, &mut sharable).context("Failed reading HD4 Block")?;
            // parse HD metadata
            sharable.parse_hd_comments(hd.hd_md_comment);

            // FH block
            let (fh, position) = parse_fh(&mut rdr, &mut sharable, hd.hd_fh_first, position)
                .context("Failed reading FH4 Blocks")?;

            // AT Block read
            let (at, position) = parse_at4(&mut rdr, &mut sharable, hd.hd_at_first, position)
                .context("Failed reading AT4 Blocks")?;

            // EV Block read
            let (ev, position) = parse_ev4(&mut rdr, &mut sharable, hd.hd_ev_first, position)
                .context("Failed reading EV4 Blocks")?;

            // Read DG Block
            let (mut dg, _, n_cg, n_cn) =
                parse_dg4(&mut rdr, hd.hd_dg_first, position, &mut sharable).context(
                    "Failed reading DG4 Blocks and related blocks including record data",
                )?;

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
    /// returns channel's unit string
    pub fn get_channel_unit(&self, channel_name: &str) -> Result<Option<String>> {
        let unit: Option<String> = match self {
            MdfInfo::V3(mdfinfo3) => mdfinfo3.get_channel_unit(channel_name),
            MdfInfo::V4(mdfinfo4) => mdfinfo4.get_channel_unit(channel_name)?,
        };
        Ok(unit)
    }
    /// returns channel's description string
    pub fn get_channel_desc(&self, channel_name: &str) -> Result<Option<String>> {
        let desc: Option<String> = match self {
            MdfInfo::V3(mdfinfo3) => mdfinfo3.get_channel_desc(channel_name),
            MdfInfo::V4(mdfinfo4) => mdfinfo4.get_channel_desc(channel_name)?,
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
                mdfinfo3.clear_channel_data_from_memory(channel_names);
            }
            MdfInfo::V4(mdfinfo4) => {
                mdfinfo4.clear_channel_data_from_memory(channel_names)?;
            }
        }
        Ok(())
    }
    /// returns channel's data ndarray.
    pub fn get_channel_data(&self, channel_name: &str) -> Option<Box<dyn Array>> {
        let data = match self {
            MdfInfo::V3(mdfinfo3) => {
                // mdfinfo3.get_channel_data(channel_name)
                None
            }
            MdfInfo::V4(mdfinfo4) => mdfinfo4.get_channel_data(channel_name),
        };
        data
    }
    /// Adds a new channel in memory (no file modification)
    pub fn add_channel(
        &mut self,
        channel_name: String,
        data: DataSignature,
        master: MasterSignature,
        unit: Option<String>,
        description: Option<String>,
    ) -> Result<()> {
        match self {
            MdfInfo::V3(mdfinfo3) => {
                let mut file_name = PathBuf::from(mdfinfo3.file_name.as_str());
                file_name.set_extension("mf4");
                let mut mdf4 = convert3to4(mdfinfo3, &file_name.to_string_lossy())?;
                mdf4.add_channel(channel_name, data, master, unit, description)?;
            }
            MdfInfo::V4(mdfinfo4) => {
                mdfinfo4.add_channel(channel_name, data, master, unit, description)?;
            }
        }
        Ok(())
    }
    /// Convert mdf verion 3.x to 4.2
    /// Require file name parameter but no file written
    pub fn convert3to4(&mut self, file_name: &str) -> Result<MdfInfo> {
        match self {
            MdfInfo::V3(mdfinfo3) => Ok(MdfInfo::V4(Box::new(convert3to4(mdfinfo3, file_name)?))),
            MdfInfo::V4(_) => bail!("file is already a mdf version 4.x"),
        }
    }
    /// defines channel's data in memory
    pub fn set_channel_data(&mut self, channel_name: &str, data: Box<dyn Array>) -> Result<()> {
        match self {
            MdfInfo::V3(mdfinfo3) => {
                let mut file_name = PathBuf::from(mdfinfo3.file_name.as_str());
                file_name.set_extension("mf4");
                let mut mdf4 = convert3to4(mdfinfo3, &file_name.to_string_lossy())?;
                mdf4.set_channel_data(channel_name, data);
            }
            MdfInfo::V4(mdfinfo4) => mdfinfo4.set_channel_data(channel_name, data),
        }
        Ok(())
    }
    /// Sets the channel's related master channel type in memory
    pub fn set_channel_master_type(&mut self, master_name: &str, master_type: u8) -> Result<()> {
        match self {
            MdfInfo::V3(mdfinfo3) => {
                let mut file_name = PathBuf::from(mdfinfo3.file_name.as_str());
                file_name.set_extension("mf4");
                let mut mdf4 = convert3to4(mdfinfo3, &file_name.to_string_lossy())?;
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
    /// get comment from position
    pub fn get_comments(&mut self, position: i64) -> Option<HashMap<String, String>> {
        match self {
            MdfInfo::V3(_mdfinfo3) => None,
            MdfInfo::V4(mdfinfo4) => Some(mdfinfo4.sharable.get_comments(position)),
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
    /// get file history blocks
    pub fn get_file_history_blocks(&self) -> Option<Vec<FhBlock>> {
        match self {
            MdfInfo::V3(_) => None,
            MdfInfo::V4(mdfinfo4) => Some(mdfinfo4.fh.clone()),
        }
    }
}

impl fmt::Display for MdfInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MdfInfo::V3(mdfinfo3) => {
                writeln!(f, "Version : {}\n", mdfinfo3.id_block.id_ver)?;
                writeln!(
                    f,
                    "Header :\n Author: {}  Organisation:{}\n",
                    mdfinfo3.hd_block.hd_author, mdfinfo3.hd_block.hd_organization
                )?;
                writeln!(
                    f,
                    "Project: {}  Subject:{}\n",
                    mdfinfo3.hd_block.hd_project, mdfinfo3.hd_block.hd_subject
                )?;
                writeln!(
                    f,
                    "Date: {:?}  Time:{:?}\n",
                    mdfinfo3.hd_block.hd_date, mdfinfo3.hd_block.hd_time
                )?;
                writeln!(f, "Comments: {}", mdfinfo3.hd_comment)?;
                writeln!(f, "\n")
            }
            MdfInfo::V4(mdfinfo4) => {
                writeln!(f, "Version : {}", mdfinfo4.id_block.id_ver)?;
                writeln!(f, "{}\n", mdfinfo4.hd_block)?;
                let comments = &mdfinfo4
                    .sharable
                    .get_hd_comments(mdfinfo4.hd_block.hd_md_comment);
                for c in comments.iter() {
                    writeln!(f, "{} {}", c.0, c.1)?;
                }
                writeln!(f, "\n")
            }
        }
    }
}
