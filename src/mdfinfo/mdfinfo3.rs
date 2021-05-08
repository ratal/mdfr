
use encoding::all::{ISO_8859_1, ASCII};
use std::default::Default;
use std::io::BufReader;
use std::fs::File;
use std::io::prelude::*;
use encoding::{Encoding, DecoderTrap};
use chrono::NaiveDate;
use std::convert::TryFrom;
use byteorder::{LittleEndian, ReadBytesExt};
use binread::{BinRead, BinReaderExt};

#[derive(Debug)]
pub struct MdfInfo3 {
    pub ver: u16,
    pub prog: [u8; 8],
    pub idblock: Id3,
    pub hdblock: Hd3,
    pub hd_comment: String,
}

/// MDF4 - common Header
#[derive(Debug)]
#[derive(BinRead)]
#[br(little)]
pub struct Blockheader3 {
    hdr_id: [u8; 2],   // 'XX' Block type identifier
    hdr_len: u16,   // block size
}

pub fn parse_block_header(rdr: &mut BufReader<&File>) -> Blockheader3 {
    let header: Blockheader3 = rdr.read_le().unwrap();
    header
}

/// Id3 block structure
#[derive(Debug, PartialEq, Default)]
pub struct Id3 {
    id_file_id: [u8; 8], // "MDF    
    id_vers: [u8; 4],  // version in char
    id_prog: [u8; 8],  // logger id
    id_byteorder: u16,  // 0 Little endian, >= 1 Big endian
    id_floatingpointformat: u16,  // default floating point number. 0: IEEE754, 1: G_Float, 2: D_Float
    pub id_ver: u16,  // version number
    id_codepagenumber: u16,  // if 0, code page unknown. If position, corresponds to to extended ascii code page
}

/// Reads the Id3 block structure in the file
pub fn parse_id3(rdr: &mut BufReader<&File>, id_file_id: [u8; 8], id_vers: [u8; 4], id_prog: [u8; 8]) -> Id3 {
    let id_byteorder = rdr.read_u16::<LittleEndian>().unwrap();
    let id_floatingpointformat = rdr.read_u16::<LittleEndian>().unwrap();
    let id_ver = rdr.read_u16::<LittleEndian>().unwrap();
    let id_codepagenumber = rdr.read_u16::<LittleEndian>().unwrap();
    let mut reserved = [0u8; 32];  // reserved
    rdr.read_exact(&mut reserved).unwrap();
    Id3 {id_file_id, id_vers, id_prog, id_byteorder, id_floatingpointformat,
        id_ver, id_codepagenumber}
}

/// HD3 block strucutre
#[derive(Debug, PartialEq)]
pub struct Hd3 {
    hd_id: [u8; 2],  // HD
    hd_len:u16,      // Length of block in bytes
    hd_dg_first:u32,    // Pointer to the first data group block (DGBLOCK) (can be NIL)
    hd_md_comment:u32,  // Pointer to the measurement file comment (TXBLOCK) (can be NIL)
    hd_pr:u32,          // Program block

    // Data members
    hd_n_datagroups:u16,  // Time stamp in nanoseconds elapsed since 00:00:00 01.01.1970 (UTC time or local time, depending on "local time" flag, see [UTC]).
    hd_date: (u32, u32, i32),  // Date at which the recording was started in "DD:MM:YYYY" format
    hd_time: (u32, u32, u32), // Time at which the recording was started in "HH:MM:SS" format
    hd_author: String,     // Author's name
    hd_organization: String,    // name of the organization or department
    hd_project: String,          // project name
    hd_subject: String, // subject or measurement object
    hd_start_time_ns: Option<u64>, // time stamp at which recording was started in nanosecond
    hd_time_offset: Option<i16>, // UTC time offset
    hd_time_quality: Option<u16>, // time quality class
    hd_time_identifier: Option<String> // timer identification or time source
}

pub fn hd3_parser(rdr: &mut BufReader<&File>, ver:u16) -> (Hd3, i64) {
    let mut hd_id = [0; 2];
    rdr.read_exact(&mut hd_id).unwrap();
    let hd_len = rdr.read_u16::<LittleEndian>().unwrap();    // Length of block in bytes
    let hd_dg_first = rdr.read_u32::<LittleEndian>().unwrap(); // Pointer to the first data group block (DGBLOCK) (can be NIL)
    let hd_md_comment = rdr.read_u32::<LittleEndian>().unwrap();  // TXblock link
    let hd_pr = rdr.read_u32::<LittleEndian>().unwrap();  // PRblock link
    let hd_n_datagroups = rdr.read_u16::<LittleEndian>().unwrap();  // number of datagroups
    let mut date = [0; 10];
    rdr.read_exact(&mut date).unwrap();  // date
    let mut datestr = String::new();
    ASCII.decode_to(&date, DecoderTrap::Replace, &mut datestr).unwrap();
    let mut dateiter = datestr.split(":");
    let day:u32 = dateiter.next().unwrap().parse::<u32>().unwrap();
    let month:u32 = dateiter.next().unwrap().parse::<u32>().unwrap();
    let year:i32 = dateiter.next().unwrap().parse::<i32>().unwrap();
    let hd_date = (day, month, year);
    let mut time = [0u8; 8];
    rdr.read_exact(&mut time).unwrap();  // time
    let mut timestr = String::new();
    ASCII.decode_to(&time, DecoderTrap::Replace, &mut timestr).unwrap();
    let mut timeiter = timestr.split(":");
    let hour:u32 = timeiter.next().unwrap().parse::<u32>().unwrap();
    let minute:u32 = timeiter.next().unwrap().parse::<u32>().unwrap();
    let sec:u32 = timeiter.next().unwrap().parse::<u32>().unwrap();
    let hd_time = (hour, minute, sec);
    let mut author = [0u8; 32];
    rdr.read_exact(&mut author).unwrap(); // author
    let mut hd_author = String::new();
    ISO_8859_1.decode_to(&author, DecoderTrap::Replace, &mut hd_author).unwrap();
    let mut organisation = [0u8; 32];
    rdr.read_exact(&mut organisation).unwrap(); // author
    let mut hd_organization = String::new();
    ISO_8859_1.decode_to(&organisation, DecoderTrap::Replace, &mut hd_organization).unwrap();
    let mut project = [0u8; 32];
    rdr.read_exact(&mut  project).unwrap(); // author
    let mut hd_project = String::new();
    ISO_8859_1.decode_to(&project, DecoderTrap::Replace, &mut hd_project).unwrap();
    let mut subject = [0u8; 32];
    rdr.read_exact(&mut subject).unwrap(); // author
    let mut hd_subject = String::new();
    ISO_8859_1.decode_to(&subject, DecoderTrap::Replace, &mut hd_subject).unwrap();
    let hd_start_time_ns: Option<u64>;
    let hd_time_offset: Option<i16>;
    let hd_time_quality: Option<u16>;
    let hd_time_identifier: Option<String>;
    let position: i64;
    if ver >= 320 {
        hd_start_time_ns = Some(rdr.read_u64::<LittleEndian>().unwrap());  // time stamp
        hd_time_offset = Some(rdr.read_i16::<LittleEndian>().unwrap());  // time offset
        hd_time_quality = Some(rdr.read_u16::<LittleEndian>().unwrap());  // time quality
        let mut time_identifier = [0u8; 32];
        rdr.read_exact(&mut time_identifier).unwrap(); // time identification
        let mut ti = String::new();
        ISO_8859_1.decode_to(&time_identifier, DecoderTrap::Replace, &mut ti).unwrap();
        hd_time_identifier = Some(ti);
        position = 208 + 64; // position after reading ID and HD
    } else {
        // calculate hd_start_time_ns
        hd_start_time_ns = Some(u64::try_from(NaiveDate::from_ymd(hd_date.2, hd_date.1, hd_date.0)
            .and_hms(hd_time.0, hd_time.1, hd_time.2)
            .timestamp_nanos()).unwrap());
        hd_time_offset = None;
        hd_time_quality = None;
        hd_time_identifier = None;
        position = 164 + 64; // position after reading ID and HD
    }
    (Hd3 {hd_id, hd_len, hd_dg_first, hd_md_comment, hd_pr,
        hd_n_datagroups, hd_date, hd_time,  hd_author, hd_organization,
        hd_project, hd_subject, hd_start_time_ns, hd_time_offset, hd_time_quality, hd_time_identifier
    }, position)
}

pub fn hd3_comment_parser(rdr: &mut BufReader<&File>, hd3_block: &Hd3, mut position: i64) -> (String, i64) {
    let (_, comment, offset) = parse_tx(rdr, i64::try_from(hd3_block.hd_md_comment).unwrap() - position);
    position += offset;
    (comment, position)
}

pub fn parse_tx(rdr: &mut BufReader<&File>, offset: i64) -> (Blockheader3, String, i64) {
    rdr.seek_relative(offset).unwrap();
    let block_header: Blockheader3 = parse_block_header(rdr);  // reads header
    // reads comment
    let mut comment_raw = vec![0; (block_header.hdr_len - 2) as usize];
    rdr.read_exact(&mut comment_raw).unwrap();
    let mut comment:String = String::new();
    ISO_8859_1.decode_to(&comment_raw, DecoderTrap::Replace, &mut comment).unwrap();
    let comment:String = comment.trim_end_matches(char::from(0)).into();
    let offset = offset + i64::try_from(block_header.hdr_len).unwrap();
    (block_header, comment, offset)
}

#[derive(Debug)]
#[derive(BinRead)]
#[br(little)]
pub struct Dg3Block {
    dg_id: [u8; 2],  // DG
    dg_len: u16,      // Length of block in bytes
    dg_dg_next: u32, // Pointer to next data group block (DGBLOCK) (can be NIL)
    dg_cg_first: u32, // Pointer to first channel group block (CGBLOCK) (can be NIL)
    dg_tr: u32,       // Pointer to trigger block
    dg_data: u32,     // Pointer to data block (DTBLOCK or DZBLOCK for this block type) or data list block (DLBLOCK of data blocks or its HLBLOCK)  (can be NIL)
    dg_n_dg: u16,    // number of channel groups
    dg_n_record_ids: u16,      // number of record ids
    reserved: u32   // reserved
}

pub fn parse_dg3(rdr: &mut BufReader<&File>, target: i64, position: i64) -> (Dg3Block, i64) {
    rdr.seek_relative(target - position).unwrap();
    let block: Dg3Block = rdr.read_le().unwrap();
    (block, position + 28)
}