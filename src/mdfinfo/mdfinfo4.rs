
use byteorder::{LittleEndian, ReadBytesExt};
use roxmltree;
use std::io::{BufRead, Seek, SeekFrom};
use std::io::prelude::*;
use std::str;
use std::option::Option;
use std::default::Default;
use std::convert::TryFrom;
use nom::number::streaming::*;
use nom::bytes::streaming::take;
use nom::IResult;

#[derive(Debug)]
pub struct MdfInfo4 {
    pub ver: u16,
    pub prog: [u8; 8],
    pub idblock: Id4,
    pub hdblock: Hd4,
}

/// MDF4 - common Header
#[derive(Debug, PartialEq, Default)]
pub struct Blockheader4 {
    hdr_id: [u8; 4],   // '##XX'
    hdr_gap: [u8; 4],   // reserved, must be 0
    hdr_len: u64,   // Length of block in bytes
    hdr_links: u64 // # of links 
}

pub fn parse_block_header(i: &[u8]) -> IResult<&[u8], Blockheader4> {
    let (i, id) = take(4usize)(i)?;  // ##HD and reserved, must be 0 
    let mut hdr_id: [u8; 4] = [0; 4];
    hdr_id.clone_from_slice(id);
    let (i, gap) = take(4usize)(i)?;   // reserved, must be 0
    let mut hdr_gap: [u8; 4] = [0; 4];
    hdr_gap.clone_from_slice(gap);
    let (i, hdr_len) = le_u64(i)?;   // Length of block in bytes
    let (i, hdr_links) = le_u64(i)?; // # of links
    Ok((i, Blockheader4 {hdr_id, hdr_gap, hdr_len, hdr_links}))
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
pub fn parse_id4(i: &[u8], id_file_id: [u8; 8], id_vers: [u8; 4], id_prog: [u8; 8]) -> IResult<&[u8], Id4> {
    let (i, _) = take(4usize)(i)?; // reserved
    let (i, id_ver) = le_u16(i)?;
    let (i, _) = take(30usize)(i)?; // reserved
    let (i, id_unfin_flags) = le_u16(i)?;
    let (i, id_custom_unfin_flags) = le_u16(i)?;
    Ok((i, Id4 {id_file_id, id_vers, id_prog, id_ver, id_unfin_flags, id_custom_unfin_flags
    }))
}

#[derive(Debug)]
pub struct Hd4 {
    hd_id: [u8; 4],  // ##HD
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
    hd_start_angle_rad:f64, // Start angle in radians at start of measurement (only for angle synchronous measurements) Only valid if "start angle valid" flag is set. All angle values for angle synchronized master channels or events are relative to this start angle.
    hd_start_distance_m:f64, // Start distance in meters at start of measurement (only for distance synchronous measurements) Only valid if "start distance valid" flag is set. All distance values for distance synchronized master channels or events are relative to this start distance.
    hd_comment: Option<String>,  // for TX Block comment
    hd_author: Option<String>,     // Author's name
    hd_organization: Option<String>,    // name of the organization or department
    hd_project: Option<String>,          // project name
    hd_subject: Option<String>, // subject or measurement object
}

pub fn hd4_parser<R: Seek + BufRead>(f: &mut R) -> Hd4 {
    let mut hd_id = [0; 4]; // ##HD and reserved, must be 0
    f.read_exact(&mut hd_id).unwrap();
    let _ = f.read_u32::<LittleEndian>().unwrap();  // reserved
    let hd_len = f.read_u64::<LittleEndian>().unwrap();   // Length of block in bytes
    let hd_link_counts = f.read_u64::<LittleEndian>().unwrap(); // # of links 
    let hd_dg_first = f.read_i64::<LittleEndian>().unwrap(); // Pointer to the first data group block (DGBLOCK) (can be NIL)
    let hd_fh_first = f.read_i64::<LittleEndian>().unwrap(); // Pointer to first file history block (FHBLOCK) 
                // There must be at least one FHBLOCK with information about the application which created the MDF file.
    let hd_ch_first = f.read_i64::<LittleEndian>().unwrap(); // Pointer to first channel hierarchy block (CHBLOCK) (can be NIL).
    let hd_at_first = f.read_i64::<LittleEndian>().unwrap(); // Pointer to first attachment block (ATBLOCK) (can be NIL)
    let hd_ev_first = f.read_i64::<LittleEndian>().unwrap(); // Pointer to first event block (EVBLOCK) (can be NIL)
    let hd_md_comment = f.read_i64::<LittleEndian>().unwrap(); // Pointer to the measurement file comment (TXBLOCK or MDBLOCK) (can be NIL) For MDBLOCK contents, see Table 14.

    // Data members
    let hd_start_time_ns = f.read_u64::<LittleEndian>().unwrap();  // Time stamp in nanoseconds elapsed since 00:00:00 01.01.1970 (UTC time or local time, depending on "local time" flag, see [UTC]).
    let hd_tz_offset_min = f.read_i16::<LittleEndian>().unwrap();  // Time zone offset in minutes. The value must be in range [-720,720], i.e. it can be negative! For example a value of 60 (min) means UTC+1 time zone = Central European Time (CET). Only valid if "time offsets valid" flag is set in time flags.
    let hd_dst_offset_min = f.read_i16::<LittleEndian>().unwrap(); // Daylight saving time (DST) offset in minutes for start time stamp. During the summer months, most regions observe a DST offset of 60 min (1 hour). Only valid if "time offsets valid" flag is set in time flags.
    let hd_time_flags = f.read_u8().unwrap();     // Time flags The value contains the following bit flags (see HD_TF_xxx)
    let hd_time_class = f.read_u8().unwrap();    // Time quality class (see HD_TC_xxx)
    let hd_flags = f.read_u8().unwrap();           // Flags The value contains the following bit flags (see HD_FL_xxx):
    let _ = f.read_u8().unwrap();       // Reserved
    let hd_start_angle_rad = f.read_f64::<LittleEndian>().unwrap(); // Start angle in radians at start of measurement (only for angle synchronous measurements) Only valid if "start angle valid" flag is set. All angle values for angle synchronized master channels or events are relative to this start angle.
    let hd_start_distance_m = f.read_f64::<LittleEndian>().unwrap(); // Start distance in meters at start of measurement (only for distance synchronous measurements) Only valid if "start distance valid" flag is set. All distance values for distance synchronized master channels or events are relative to this start distance.
    let hd_comment : Option<String>;
    let hd_author: Option<String>;
    let hd_organization: Option<String>;    // name of the organization or department
    let hd_project: Option<String>;          // project name
    let hd_subject: Option<String>;
    // parsing HD comment block
    if hd_md_comment != 0 {
        let (block_header, comment) = parse_comment(f, hd_md_comment);
        if block_header.hdr_id == "##HD".as_bytes() {
            // TX Block
            hd_comment = Some(comment);
            hd_author = None;
            hd_organization = None;
            hd_project = None;
            hd_subject = None;
        } else {
            // MD Block, reading xml
            hd_comment = None;
            let md = roxmltree::Document::parse(&comment). expect("Could not parse HD MD block");
            let prop = md.descendants().find(|p| p.has_tag_name("common_properties")).expect("No common_properties found in HD comment block");
            let prope = prop.document().descendants().find(|p| p.has_tag_name("e")).expect("No common_properties found in HD comment block");
            if prope.has_attribute("author") {
                hd_author = prope.attribute("author").map(str::to_string);
                hd_organization = prope.attribute("organization").map(str::to_string);
                hd_project = prope.attribute("project").map(str::to_string);
                hd_subject = prope.attribute("subject").map(str::to_string);
            } else {
                hd_author = None;
                hd_organization = None;
                hd_project = None;
                hd_subject = None;
            }
        }
    } else {
        hd_comment = None;
        hd_author = None;
        hd_organization = None;
        hd_project = None;
        hd_subject = None;
    }
    Hd4 {hd_id, hd_len, hd_link_counts, hd_dg_first, hd_fh_first, hd_ch_first,
        hd_at_first, hd_ev_first, hd_md_comment, hd_start_time_ns,
        hd_tz_offset_min, hd_dst_offset_min, hd_time_flags,  hd_time_class, hd_flags,
        hd_start_angle_rad, hd_start_distance_m, hd_comment,
        hd_author, hd_organization, hd_project, hd_subject
    }
}

/* pub fn hd4_reader<R: Seek + BufRead>(f: &mut R) -> Hd4 {
    let s = structure("<4sI2Q6qQ2h4B2d");
    let mut buf = [0; 104];
    f.take(104).read(&mut buf).unwrap();
    let (hd_id, hd_len, hd_link_counts, hd_dg_first, hd_fh_first, hd_ch_first,
        hd_at_first, hd_ev_first, hd_md_comment, hd_start_time_ns,
        hd_tz_offset_min, hd_dst_offset_min, hd_time_flags,  hd_time_class, hd_flags,
        hd_start_angle_rad, hd_start_distance_m) 
        = s.unpack(buf);
    let hd_comment = Comment::NoComment;
    let hd_author = None;
    let hd_organization = None;
    let hd_project = None;
    let hd_subject = None;
    // parsing HD comment block
    if hd_md_comment != 0 {
        let (block_header, comment) = parse_comment(f, hd_md_comment);
        if block_header.hdr_id == *b"##HD" {
            // TX Block
            hd_comment = Comment::TX(comment);
        } else {
            // MD Block, reading xml
            let md = match roxmltree::Document::parse(&comment) {
                Ok(c) => c,
                Err(e) => {
                    println!("Error, could not parse HD Block xml comment: {}.", e);
                    std::process::exit(1);
                }
            };
            hd_comment = Comment::MD(md);
        }
    }
    Hd4 {hd_id, hd_len, hd_link_counts, hd_dg_first, hd_fh_first, hd_ch_first,
        hd_at_first, hd_ev_first, hd_md_comment, hd_start_time_ns,
        hd_tz_offset_min, hd_dst_offset_min, hd_time_flags,  hd_time_class, hd_flags,
        hd_start_angle_rad, hd_start_distance_m, hd_comment,
        hd_author, hd_organization, hd_project, hd_subject
    }
} */

pub fn parse_comment<R: Seek + BufRead>(f: &mut R, position: i64) -> (Blockheader4, String) {
    f.by_ref().seek(SeekFrom::Start(u64::try_from(position).unwrap())).unwrap();
    let header = [0; 24];
    let block_header: Blockheader4 = match parse_block_header(&header).map(|x| x.1){
        Ok(i) => i,
        Err(e) => panic!("Failed parsing the file MD Block : {:?}", e),
    };
    // reads xml file
    let mut comment_raw = vec![0; (block_header.hdr_len - 24) as usize];
    f.take(4).read(&mut comment_raw).unwrap();
    let comment:String = str::from_utf8(&comment_raw).unwrap().parse().unwrap();
    return (block_header, comment)
}