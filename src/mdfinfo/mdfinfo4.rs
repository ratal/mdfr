
use byteorder::{LittleEndian, ReadBytesExt};
use roxmltree;
use std::io::BufReader;
use std::fs::File;
use std::io::prelude::*;
use std::str;
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
    pub hd_comment: HashMap<String, String>,
    pub fh: Vec<(FhBlock, HashMap<String, String>)>,
}

/// MDF4 - common Header
#[derive(Debug)]
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
    pub id_prog: [u8; 8],
    pub id_ver: u16,
    pub id_unfin_flags: u16,
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
    pub hd_dg_first:i64, // Pointer to the first data group block (DGBLOCK) (can be NIL)
    pub hd_fh_first:i64, // Pointer to first file history block (FHBLOCK) 
                // There must be at least one FHBLOCK with information about the application which created the MDF file.
    hd_ch_first:i64, // Pointer to first channel hierarchy block (CHBLOCK) (can be NIL).
    pub hd_at_first:i64, // Pointer to first attachment block (ATBLOCK) (can be NIL)
    pub hd_ev_first:i64, // Pointer to first event block (EVBLOCK) (can be NIL)
    pub hd_md_comment:i64, // Pointer to the measurement file comment (TXBLOCK or MDBLOCK) (can be NIL) For MDBLOCK contents, see Table 14.

    // Data members
    pub hd_start_time_ns:u64,  // Time stamp in nanoseconds elapsed since 00:00:00 01.01.1970 (UTC time or local time, depending on "local time" flag, see [UTC]).
    pub hd_tz_offset_min:i16,  // Time zone offset in minutes. The value must be in range [-720,720], i.e. it can be negative! For example a value of 60 (min) means UTC+1 time zone = Central European Time (CET). Only valid if "time offsets valid" flag is set in time flags.
    pub hd_dst_offset_min:i16, // Daylight saving time (DST) offset in minutes for start time stamp. During the summer months, most regions observe a DST offset of 60 min (1 hour). Only valid if "time offsets valid" flag is set in time flags.
    pub hd_time_flags:u8,     // Time flags The value contains the following bit flags (see HD_TF_xxx)
    pub  hd_time_class:u8,    // Time quality class (see HD_TC_xxx)
    pub hd_flags:u8,          // Flags The value contains the following bit flags (see HD_FL_xxx):
    pub hd_reserved2: u8,    // reserved
    pub hd_start_angle_rad:f64, // Start angle in radians at start of measurement (only for angle synchronous measurements) Only valid if "start angle valid" flag is set. All angle values for angle synchronized master channels or events are relative to this start angle.
    pub hd_start_distance_m:f64, // Start distance in meters at start of measurement (only for distance synchronous measurements) Only valid if "start distance valid" flag is set. All distance values for distance synchronized master channels or events are relative to this start distance.
}

pub fn hd4_parser(rdr: &mut BufReader<&File>) -> Hd4 {
    let block: Hd4 = rdr.read_le().unwrap();
    return block
}

pub fn hd4_comment_parser(rdr: &mut BufReader<&File>, hd4_block: &Hd4) -> (HashMap<String, String>, i64) {
    let mut position:i64 = 168;
    let mut comments: HashMap<String, String> = HashMap::new();
    // parsing HD comment block
    if hd4_block.hd_md_comment != 0 {
        let (block_header, comment, offset) = parse_comment(rdr, hd4_block.hd_md_comment - position);
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

pub fn parse_comment(rdr: &mut BufReader<&File>, offset: i64) -> (Blockheader4, String, i64) {
    rdr.seek_relative(offset).unwrap();  // change buffer position
    let block_header: Blockheader4 = parse_block_header(rdr);  // reads header
    // reads comment
    let mut comment_raw = vec![0; (block_header.hdr_len - 24) as usize];
    rdr.read(&mut comment_raw).unwrap();
    let comment:String = str::from_utf8(&comment_raw).unwrap().parse().unwrap();
    let comment:String = comment.trim_end_matches(char::from(0)).into();
    let ofst = offset + i64::try_from(block_header.hdr_len).unwrap();
    return (block_header, comment, ofst)
}

#[derive(Debug)]
#[derive(BinRead)]
#[br(little)]
pub struct FhBlock {
    fh_id: [u8; 4],   // '##FH'
    fh_gap: [u8; 4],   // reserved, must be 0
    fh_len: u64,   // Length of block in bytes
    fh_links: u64, // # of links
    pub fh_fh_next: i64, // Link to next FHBLOCK (can be NIL if list finished)
    pub fh_md_comment: i64, // Link to MDBLOCK containing comment about the creation or modification of the MDF file.
    pub fh_time_ns: u64,  // time stamp in nanosecs
    pub fh_tz_offset_min: i16, // time zone offset on minutes
    pub fh_dst_offset_min: i16, // daylight saving time offset in minutes for start time stamp
    pub fh_time_flags: u8,  // time flags, but 1 local, bit 2 time offsets
    fh_reserved: [u8; 3], // reserved
}

pub fn parse_fh_block(rdr: &mut BufReader<&File>, offset: i64) -> (FhBlock, i64) {
    rdr.seek_relative(offset).unwrap();  // change buffer position
    let fh: FhBlock = match rdr.read_le() {
        Ok(v) => v,
        Err(e) => panic!("Error reading fh block \n{}", e),
    };  // reads the fh block
    let offset = offset + 56; 
    return (fh, offset)
}

pub fn parse_fh_comment(rdr: &mut BufReader<&File>, fh_block: &FhBlock, mut offset: i64) -> (HashMap<String, String>, i64){
    let mut comments: HashMap<String, String> = HashMap::new();
    if fh_block.fh_md_comment != 0 {
        let (block_header, comment, of) = parse_comment(rdr, offset);
        offset = of;
        if block_header.hdr_id == "##TX".as_bytes() {
            // TX Block
            comments.insert(String::from("comment"), comment);
        } else {
            // MD Block, reading xml
            let comment:String = comment.trim_end_matches(|c| c == '\n' || c == '\r' || c == ' ').into(); // removes ending spaces
            match roxmltree::Document::parse(&comment) {
                Ok(md) => {
                    for node in md.root().descendants() {
                        let text = match node.text() {
                            Some(text) => text.to_string(),
                            None => String::new(),
                        };
                        comments.insert(node.tag_name().name().to_string(), text);
                    }
                    comments = HashMap::new();
                },
                Err(e) => {
                    println!("Error parsing FH comment : \n{}\n{}", comment, e);
                    comments = HashMap::new();
                },
            };
        }
    }
    return (comments, offset)
}

pub fn parse_fh(rdr: &mut BufReader<&File>, of: i64) -> (Vec<(FhBlock, HashMap<String, String>)>, i64) {
    let mut fh: Vec<(FhBlock, HashMap<String, String>)> = Vec::new();
    let (block, offset) = parse_fh_block(rdr, of);
    let (comment_temp, offset) = 
        parse_fh_comment(rdr, &block, block.fh_md_comment - offset);
    let mut next_pointer = block.fh_fh_next;
    fh.push((block, comment_temp));
    while next_pointer != 0 {
        let (block, offset) = parse_fh_block(rdr, next_pointer - &offset);
        next_pointer = block.fh_fh_next;
        let (comment_temp, offset) = 
            parse_fh_comment(rdr, &block, block.fh_md_comment - offset);
        fh.push((block, comment_temp));
    } 
    return (fh, offset)
}
#[derive(Debug)]
#[derive(BinRead)]
#[br(little)]
pub struct At4Block {
    at_id: [u8; 4],  // DG
    reserved: [u8; 4],  // reserved
    at_len: u64,      // Length of block in bytes
    at_links: u64,         // # of links 
    at_at_next: i64, // Link to next ATBLOCK (linked list) (can be NIL)
    at_tx_filename: i64, // Link to TXBLOCK with the path and file name of the embedded or referenced file (can only be NIL if data is embedded). The path of the file can be relative or absolute. If relative, it is relative to the directory of the MDF file. If no path is given, the file must be in the same directory as the MDF file.
    at_tx_mimetype: i64, // Link to TXBLOCK with MIME content-type text that gives information about the attached data. Can be NIL if the content-type is unknown, but should be specified whenever possible. The MIME content-type string must be written in lowercase.
    at_md_comment: i64,   // Link to MDBLOCK with comment and additional information about the attachment (can be NIL).
    at_flags: u16,         // Flags The value contains the following bit flags (see AT_FL_xxx):
    at_creator_index: u16, // Creator index, i.e. zero-based index of FHBLOCK in global list of FHBLOCKs that specifies which application has created this attachment, or changed it most recently.
    at_reserved: [u8; 4],      // Reserved
    at_md5_checksum: [u8; 16],  // 128-bit value for MD5 check sum (of the uncompressed data if data is embedded and compressed). Only valid if "MD5 check sum valid" flag (bit 2) is set.
    at_original_size: u64, // Original data size in Bytes, i.e. either for external file or for uncompressed data.
    at_embedded_size: u64, // Embedded data size N, i.e. number of Bytes for binary embedded data following this element. Must be 0 if external file is referenced.
    // followed by embedded data depending of flag
}

pub fn parser_at4_block(rdr: &mut BufReader<&File>, mut offset: i64) -> (At4Block, HashMap<String, String>, Option<Vec<u8>>, i64) {
    rdr.seek_relative(offset).unwrap();
    let block: At4Block = rdr.read_le().unwrap();
    offset += 96;
    
    let data:Option<Vec<u8>>;
    // reads embedded if exists
    if (block.at_flags & 0b1) > 0 {
        let mut embedded_data = vec![0u8; block.at_embedded_size as usize];
        rdr.read(&mut embedded_data).unwrap();
        offset += i64::try_from(block.at_embedded_size).unwrap();
        data = Some(embedded_data);
    } else {
        data = None;
    }

    let mut comments: HashMap<String, String> = HashMap::new();
    // Reads MD
    if block.at_md_comment > 0{
        let (_, comment, of) =
            parse_comment(rdr, block.at_md_comment - offset);
        offset += of;
        let comment:String = comment.trim_end_matches(|c| c == '\n' || c == '\r' || c == ' ').into(); // removes ending spaces
            match roxmltree::Document::parse(&comment) {
                Ok(md) => {
                    for node in md.root().descendants() {
                        let text = match node.text() {
                            Some(text) => text.to_string(),
                            None => String::new(),
                        };
                        comments.insert(node.tag_name().name().to_string(), text);
                    }
                    comments = HashMap::new();
                },
                Err(e) => {
                    println!("Error parsing AT comment : \n{}\n{}", comment, e);
                    comments = HashMap::new();
                },
            };
    }

    // reads TX
    if block.at_tx_filename > 0 {
        let (_, comment, of) = 
            parse_comment(rdr, block.at_tx_filename - offset);
        comments.insert(String::from("comment"), comment);
        offset += of;
    }
    
    // Reads tx mime type
    if block.at_tx_mimetype > 0 {
        let (_, comment, of) = 
            parse_comment(rdr, block.at_tx_mimetype - offset);
        comments.insert(String::from("comment_mimetype"), comment);
        offset += of;
    }

    return (block, comments, data, offset)
}

pub fn parse_at4(rdr: &mut BufReader<&File>, mut offset: i64) -> Vec<(At4Block, HashMap<String, String>, Option<Vec<u8>>, i64)> {
    let mut at: Vec<(At4Block, HashMap<String, String>, Option<Vec<u8>>, i64)> = Vec::new();
    let (block, comments, data, of) = parser_at4_block(rdr, offset);
    offset += of;
    let mut next_pointer = block.at_at_next;
    at.push((block, comments, data, offset));
    while next_pointer >0 {
        let (block, comments, data, of) = parser_at4_block(rdr, next_pointer - offset);
        offset += of;
        next_pointer = block.at_at_next;
        at.push((block, comments, data, offset));
    }
    return at
}

#[derive(Debug)]
#[derive(BinRead)]
#[br(little)]
pub struct Ev4 {
    ev_id: [u8; 4],  // DG
    reserved: [u8; 4],  // reserved
    ev_len: u64,      // Length of block in bytes
    ev_links: u64,         // # of links
    ev_ev_next: u64,     // Link to next EVBLOCK (linked list) (can be NIL)
    ev_ev_parent: u64,   // Referencing link to EVBLOCK with parent event (can be NIL).
    ev_ev_range: u64,    // Referencing link to EVBLOCK with event that defines the beginning of a range (can be NIL, must be NIL if ev_range_type ≠ 2).
    ev_tx_name: u64,     // Pointer to TXBLOCK with event name (can be NIL) Name must be according to naming rules stated in 4.4.2 Naming Rules. If available, the name of a named trigger condition should be used as event name. Other event types may have individual names or no names.
    ev_md_comment: u64,  // Pointer to TX/MDBLOCK with event comment and additional information, e.g. trigger condition or formatted user comment text (can be NIL)
    #[br(if(ev_links - 5 > 0), little, count = ev_links - 5)]
    links: Vec<u64>,       // links

    ev_type: u8, // Event type (see EV_T_xxx)
    ev_sync_type: u8, // Sync type (see EV_S_xxx)
    ev_range_type: u8, // Range Type (see EV_R_xxx)
    ev_cause: u8,     // Cause of event (see EV_C_xxx)
    ev_flags: u8,     // flags (see EV_F_xxx)
    ev_reserved: [u8; 3], // Reserved
    ev_scope_count: u32, // Length M of ev_scope list. Can be zero.
    ev_attachment_count: u16, // Length N of ev_at_reference list, i.e. number of attachments for this event. Can be zero.
    ev_creator_index: u16, // Creator index, i.e. zero-based index of FHBLOCK in global list of FHBLOCKs that specifies which application has created or changed this event (e.g. when generating event offline).
    ev_sync_base_value: i64, // Base value for synchronization value.
    ev_sync_factor: f64,  // Factor for event synchronization value.
}

pub fn ev4_parser(rdr: &mut BufReader<&File>, mut offset: i64) -> (Ev4, i64) {
    rdr.seek_relative(offset).unwrap();
    let block: Ev4 = rdr.read_le().unwrap();
    offset += i64::try_from(block.ev_len).unwrap();
    return (block, offset)
}

#[derive(Debug)]
#[derive(BinRead)]
#[br(little)]
pub struct Dg4 {
    dg_id: [u8; 4],  // DG
    reserved: [u8; 4],  // reserved
    dg_len: u64,      // Length of block in bytes
    dg_links: u64,         // # of links 
    dg_dg_next: u64, // Pointer to next data group block (DGBLOCK) (can be NIL)
    dg_cg_first: u64, // Pointer to first channel group block (CGBLOCK) (can be NIL)
    dg_data: u64,     // Pointer to data block (DTBLOCK or DZBLOCK for this block type) or data list block (DLBLOCK of data blocks or its HLBLOCK)  (can be NIL)
    dg_comment: u64,    // comment
    dg_rec_id_size: u8,      // number of bytes used for record IDs. 0 no recordID
    reserved_2: [u8; 7],  // reserved
}

pub fn dg4_parser(rdr: &mut BufReader<&File>, offset: i64) -> (Dg4, i64) {
    rdr.seek_relative(offset).unwrap();
    let block: Dg4 = rdr.read_le().unwrap();
    return (block, offset + 64)
}