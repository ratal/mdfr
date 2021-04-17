
use byteorder::{LittleEndian, ReadBytesExt};
use roxmltree;
use std::io::BufReader;
use std::fs::File;
use std::io::prelude::*;
use std::str;
use std::option::Option;
use std::default::Default;
use std::convert::TryFrom;
use binread::{BinRead, BinReaderExt};
use std::collections::HashMap;

#[derive(Debug)]
pub struct MdfInfo4 {
    pub ver: u16,
    pub prog: [u8; 8],
    pub id_block: Id4,
    pub hd_block: Hd4,
}

/// MDF4 - common Header
#[derive(BinRead)]
#[br(little)]
pub struct Blockheader4 {
    hdr_id: [u8; 4],   // '##XX'
    hdr_gap: [u8; 4],   // reserved, must be 0
    hdr_len: u64,   // Length of block in bytes
    hdr_links: u64 // # of links 
}

pub fn parse_block_header(rdr: &mut BufReader<&File>) -> Blockheader4 {
    let header: Blockheader4 = rdr.read_le().unwrap();
    return header
}

/// Id4 block structure
#[derive(Debug, PartialEq, Default)]
pub struct Id4 {
    id_file_id: [u8; 8], // "MDF     "
    id_vers: [u8; 4],
    id_prog: [u8; 8],
    pub id_ver: u16,
    id_unfin_flags: u16,
    id_custom_unfin_flags: u16
}

/// Reads the ID4 Block structure int he file
pub fn parse_id4(rdr: &mut BufReader<&File>, id_file_id: [u8; 8], id_vers: [u8; 4], id_prog: [u8; 8]) -> Id4 {
    let _ = rdr.read_u32::<LittleEndian>().unwrap();  // reserved
    let id_ver = rdr.read_u16::<LittleEndian>().unwrap();
    let mut gap = [0u8; 30];
    rdr.read_exact(&mut  gap).unwrap();
    let id_unfin_flags = rdr.read_u16::<LittleEndian>().unwrap();
    let id_custom_unfin_flags = rdr.read_u16::<LittleEndian>().unwrap();
    Id4 {id_file_id, id_vers, id_prog, id_ver, id_unfin_flags, id_custom_unfin_flags
    }
}

#[derive(Debug)]
#[derive(BinRead)]
#[br(little)]
pub struct Hd4 {
    hd_id: [u8; 4],  // ##HD
    hd_reserved: [u8; 4],  // reserved
    hd_len:u64,   // Length of block in bytes
    hd_link_counts:u64, // # of links 
    hd_dg_first:i64, // Pointer to the first data group block (DGBLOCK) (can be NIL)
    hd_fh_first:i64, // Pointer to first file history block (FHBLOCK) 
                // There must be at least one FHBLOCK with information about the application which created the MDF file.
    hd_ch_first:i64, // Pointer to first channel hierarchy block (CHBLOCK) (can be NIL).
    hd_at_first:i64, // Pointer to first attachment block (ATBLOCK) (can be NIL)
    hd_ev_first:i64, // Pointer to first event block (EVBLOCK) (can be NIL)
    hd_md_comment:i64, // Pointer to the measurement file comment (TXBLOCK or MDBLOCK) (can be NIL) For MDBLOCK contents, see Table 14.

    // Data members
    hd_start_time_ns:u64,  // Time stamp in nanoseconds elapsed since 00:00:00 01.01.1970 (UTC time or local time, depending on "local time" flag, see [UTC]).
    hd_tz_offset_min:i16,  // Time zone offset in minutes. The value must be in range [-720,720], i.e. it can be negative! For example a value of 60 (min) means UTC+1 time zone = Central European Time (CET). Only valid if "time offsets valid" flag is set in time flags.
    hd_dst_offset_min:i16, // Daylight saving time (DST) offset in minutes for start time stamp. During the summer months, most regions observe a DST offset of 60 min (1 hour). Only valid if "time offsets valid" flag is set in time flags.
    hd_time_flags:u8,     // Time flags The value contains the following bit flags (see HD_TF_xxx)
    hd_time_class:u8,    // Time quality class (see HD_TC_xxx)
    hd_flags:u8,          // Flags The value contains the following bit flags (see HD_FL_xxx):
    hd_reserved2: u8,    // reserved
    hd_start_angle_rad:f64, // Start angle in radians at start of measurement (only for angle synchronous measurements) Only valid if "start angle valid" flag is set. All angle values for angle synchronized master channels or events are relative to this start angle.
    hd_start_distance_m:f64, // Start distance in meters at start of measurement (only for distance synchronous measurements) Only valid if "start distance valid" flag is set. All distance values for distance synchronized master channels or events are relative to this start distance.
}

pub fn hd4_parser(rdr: &mut BufReader<&File>) -> Hd4 {
    let block: Hd4 = rdr.read_le().unwrap();
    return block
}

#[derive(Debug)]
pub struct Hd4Comment {
    hd_comment: Option<String>,  // for TX Block comment
    hd_author: Option<String>,     // Author's name
    hd_organization: Option<String>,    // name of the organization or department
    hd_project: Option<String>,          // project name
    hd_subject: Option<String>, // subject or measurement object
}

pub fn hd4_comment_parser(rdr: &mut BufReader<&File>, hd4_block: &Hd4) -> (HashMap<String, String>, i64) {
    let mut position:i64 = 168;
    let mut comments: HashMap<String, String> = HashMap::new();
    // parsing HD comment block
    if hd4_block.hd_md_comment != 0 {
        let (block_header, comment, offset) = parse_md(rdr, hd4_block.hd_md_comment - position);
        position += offset;
        if block_header.hdr_id == "##TX".as_bytes() {
            // TX Block
            comments.insert(String::from("comment"), comment);
        } else {
            // MD Block, reading xml
            let md = roxmltree::Document::parse(&comment).expect("Could not parse HD MD block");
            for node in md.root().descendants().filter(|p| p.has_tag_name("e")){
                if let (Some(value), Some(text)) = (node.attribute("name"), node.text()) {
                comments.insert(value.to_string(), text.to_string());
                }
            }
        }
    }
    (comments, position)
}

pub fn parse_md(rdr: &mut BufReader<&File>, offset: i64) -> (Blockheader4, String, i64) {
    rdr.seek_relative(offset).unwrap();
    let block_header: Blockheader4 = parse_block_header(rdr);
    // reads xml file
    let mut comment_raw = vec![0; (block_header.hdr_len - 24) as usize];
    rdr.read(&mut comment_raw).unwrap();
    let comment:String = str::from_utf8(&comment_raw).unwrap().parse().unwrap();
    let comment:String = comment.trim_end_matches(char::from(0)).into();
    let offset = offset + i64::try_from(block_header.hdr_len).unwrap();
    return (block_header, comment, offset)
}