
use encoding::all::{ISO_8859_1, ASCII};
use std::{io::Read, option::Option};
use std::default::Default;
use std::io::{BufRead, Seek};
use encoding::{Encoding, DecoderTrap};
use nom::number::streaming::*;
use nom::bytes::streaming::take;
use nom::IResult;
use chrono::NaiveDate;
use std::convert::TryFrom;
use byteorder::LittleEndian;

#[derive(Debug)]
pub struct MdfInfo3 {
    ver: u16,
    prog: [u8; 8],
    idblock: Id3,
    hdblock: Hd3,
}

/// Id3 block structure
#[derive(Debug, PartialEq, Default)]
pub struct Id3 {
    id_file_id: [u8; 8], // "MDF    
    id_vers: [u8; 4],  // version in char
    id_prog: [u8; 8],
    id_byteorder: u16,
    id_floatingpointformat: u16,
    pub id_ver: u16,
    id_codepagenumber: u16,
}

/// Reads the Id3 block structure in the file
pub fn parse_id3(i: &[u8], id_file_id: [u8; 8], id_vers: [u8; 4], id_prog: [u8; 8]) -> IResult<&[u8], Id3> {
    let (i, id_byteorder) = le_u16(i)?;
    let (i, id_floatingpointformat) = le_u16(i)?;
    let (i, id_ver) = le_u16(i)?;
    let (i, id_codepagenumber) = le_u16(i)?;
    let (i, _) = take(32usize)(i)?;  // reserved
    Ok((i, Id3 {id_file_id, id_vers, id_prog, id_byteorder, id_floatingpointformat,
        id_ver, id_codepagenumber
    }))
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

pub fn hd3_parser<R: Seek + BufRead>(f: &mut R, ver:u16) -> IResult<&[u8], Hd3> {
    let mut i: [u8; 104];
    f.take(104).read(&mut i).unwrap();
    let (i, id) = take(2usize)(i)?;  // HD and block size
    let mut hd_id: [u8; 2] = [0; 2];
    hd_id.clone_from_slice(id);
    let (i, hd_len) = le_u16(i)?;    // Length of block in bytes
    let (i, hd_dg_first) = le_u32(i)?; // Pointer to the first data group block (DGBLOCK) (can be NIL)
    let (i, hd_md_comment) = le_u32(i)?;  // TXblock link
    let (i, hd_pr) = le_u32(i)?;  // PRblock link
    let (i, hd_n_datagroups) = le_u16(i)?;  // number of datagroups
    let (i, date) = take(10usize)(i)?;  // date
    let mut datestr = String::new();
    ASCII.decode_to(date, DecoderTrap::Replace, &mut datestr).unwrap();
    let mut dateiter = datestr.split(":");
    let day:u32 = dateiter.next().unwrap().parse::<u32>().unwrap();
    let month:u32 = dateiter.next().unwrap().parse::<u32>().unwrap();
    let year:i32 = dateiter.next().unwrap().parse::<i32>().unwrap();
    let hd_date = (day, month, year);
    let (i, time) = take(8usize)(i)?;  // time
    let mut timestr = String::new();
    ASCII.decode_to(time, DecoderTrap::Replace, &mut timestr).unwrap();
    let mut timeiter = timestr.split(":");
    let hour:u32 = timeiter.next().unwrap().parse::<u32>().unwrap();
    let minute:u32 = timeiter.next().unwrap().parse::<u32>().unwrap();
    let sec:u32 = timeiter.next().unwrap().parse::<u32>().unwrap();
    let hd_time = (hour, minute, sec);
    let (i, author) = take(32usize)(i)?; // author
    let mut hd_author = String::new();
    ISO_8859_1.decode_to(author, DecoderTrap::Replace, &mut hd_author).unwrap();
    let (i, organisation) = take(32usize)(i)?; // author
    let mut hd_organization = String::new();
    ISO_8859_1.decode_to(organisation, DecoderTrap::Replace, &mut hd_organization).unwrap();
    let (i, project) = take(32usize)(i)?; // author
    let mut hd_project = String::new();
    ISO_8859_1.decode_to(project, DecoderTrap::Replace, &mut hd_project).unwrap();
    let (i, subject) = take(32usize)(i)?; // author
    let mut hd_subject = String::new();
    ISO_8859_1.decode_to(subject, DecoderTrap::Replace, &mut hd_subject).unwrap();
    let hd_start_time_ns: Option<u64> = None;
    let hd_time_offset: Option<i16> = None;
    let hd_time_quality: Option<u16> = None;
    let hd_time_identifier: Option<String> = None;
    if ver >= 320 {
        let mut _hdtimebuffer = [0; 44];
        f.read(&mut _hdtimebuffer).unwrap();
        let (_hdtimebuffer, start_time_ns) = le_u64(&_hdtimebuffer[..])?;  // time stamp
        hd_start_time_ns = Some(start_time_ns);
        let (_hdtimebuffer, time_offset) = le_i16(_hdtimebuffer)?;  // time offset
        hd_time_offset = Some(time_offset);
        let (_hdtimebuffer, time_quality) = le_u16(_hdtimebuffer)?;  // time quality
        hd_time_quality = Some(time_quality);
        let (_hdtimebuffer, time_identifier) = take(32usize)(_hdtimebuffer)?; // time identification
        let mut ti = String::new();
        ISO_8859_1.decode_to(time_identifier, DecoderTrap::Replace, &mut ti).unwrap();
        hd_time_identifier = Some(ti);
    } else {
        // calculate hd_start_time_ns
        hd_start_time_ns = Some(u64::try_from(NaiveDate::from_ymd(hd_date.2, hd_date.1, hd_date.0)
            .and_hms(hd_time.0, hd_time.1, hd_time.2)
            .timestamp_nanos()).unwrap());
    }
    Ok((i, Hd3 {hd_id, hd_len, hd_dg_first, hd_md_comment, hd_pr,
        hd_n_datagroups, hd_date, hd_time,  hd_author, hd_organization,
        hd_project, hd_subject, hd_start_time_ns, hd_time_offset, hd_time_quality, hd_time_identifier
    }))
}

