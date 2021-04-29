
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
    pub at: HashMap<i64, (At4Block, HashMap<String, String>, Option<Vec<u8>>)> ,
    pub ev: HashMap<i64, (Ev4Block, HashMap<String, String>)>,
    pub dg: HashMap<i64, Dg4>,
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

fn parse_block_header(rdr: &mut BufReader<&File>) -> Blockheader4 {
    let header: Blockheader4 = match rdr.read_le() {
        Ok(v) => v,
        Err(e) => panic!("Error reading comment block \n{}", e),
    };
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
        let (block_header, comment, pos) = parse_comment(rdr, hd4_block.hd_md_comment, position);
        position = pos;
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

fn parse_comment(rdr: &mut BufReader<&File>, target: i64, mut position: i64) -> (Blockheader4, String, i64) {
    rdr.seek_relative(target - position).unwrap();  // change buffer position
    let block_header: Blockheader4 = parse_block_header(rdr);  // reads header
    // reads comment
    let mut comment_raw = vec![0u8; (block_header.hdr_len - 24) as usize];
    rdr.read(&mut comment_raw).unwrap();
    let comment:String = match str::from_utf8(&comment_raw).unwrap().parse(){
        Ok(v) => v,
        Err(e) => panic!("Error reading comment block \n{}", e),
    };
    let comment:String = comment.trim_end_matches(char::from(0)).into();
    position = target + i64::try_from(block_header.hdr_len).unwrap();
    return (block_header, comment, position)
}

fn comment(rdr: &mut BufReader<&File>, target: i64, mut position: i64) -> (HashMap<String, String>, i64) {
    let mut comments: HashMap<String, String> = HashMap::new();
    // Reads MD
    if target > 0{
        let (block_header, comment, pos) = parse_comment(rdr, target, position);
        position = pos;
        if block_header.hdr_id == "##TX".as_bytes() {
            // TX Block
            comments.insert(String::from("comment"), comment);
        } else {
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
    }
    return (comments, position);
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

fn parse_fh_block(rdr: &mut BufReader<&File>, target:i64, position:i64) -> (FhBlock, i64) {
    rdr.seek_relative(target - position).unwrap();  // change buffer position
    let fh: FhBlock = match rdr.read_le() {
        Ok(v) => v,
        Err(e) => panic!("Error reading fh block \n{}", e),
    };  // reads the fh block
    return (fh, target + 56)
}

fn parse_fh_comment(rdr: &mut BufReader<&File>, fh_block: &FhBlock, target:i64, mut position: i64) -> (HashMap<String, String>, i64){
    let mut comments: HashMap<String, String> = HashMap::new();
    if fh_block.fh_md_comment != 0 {
        let (block_header, comment, pos) = parse_comment(rdr, target, position);
        position = pos;
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
    return (comments, position)
}

pub fn parse_fh(rdr: &mut BufReader<&File>, target: i64, position: i64) -> (Vec<(FhBlock, HashMap<String, String>)>, i64) {
    let mut fh: Vec<(FhBlock, HashMap<String, String>)> = Vec::new();
    let (block, position) = parse_fh_block(rdr, target, position);
    let (comment_temp, mut position) = 
        parse_fh_comment(rdr, &block, block.fh_md_comment, position);
    let mut next_pointer = block.fh_fh_next;
    fh.push((block, comment_temp));
    while next_pointer != 0 {
        let (block, pos) = parse_fh_block(rdr, next_pointer, position);
        position = pos;
        next_pointer = block.fh_fh_next;
        let (comment_temp, pos) = 
            parse_fh_comment(rdr, &block, block.fh_md_comment, position);
        position = pos;
        fh.push((block, comment_temp));
    } 
    return (fh, position)
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

fn parser_at4_block(rdr: &mut BufReader<&File>, target: i64, mut position: i64) -> (At4Block, HashMap<String, String>, Option<Vec<u8>>, i64) {
    rdr.seek_relative(target - position).unwrap();
    let block: At4Block = rdr.read_le().unwrap();
    position = target + 96;
    
    let data:Option<Vec<u8>>;
    // reads embedded if exists
    if (block.at_flags & 0b1) > 0 {
        let mut embedded_data = vec![0u8; block.at_embedded_size as usize];
        rdr.read(&mut embedded_data).unwrap();
        position += i64::try_from(block.at_embedded_size).unwrap();
        data = Some(embedded_data);
    } else {
        data = None;
    }

    // Reads MD
    let (mut comments, mut position) = comment(rdr, block.at_md_comment, position);

    // reads TX
    if block.at_tx_filename > 0 {
        let (_, comment, pos) = 
            parse_comment(rdr, block.at_tx_filename, position);
        position = pos;
        comments.insert(String::from("comment"), comment);
    }
    
    // Reads tx mime type
    if block.at_tx_mimetype > 0 {
        let (_, comment, pos) = 
            parse_comment(rdr, block.at_tx_mimetype, position);
        position = pos;
        comments.insert(String::from("comment_mimetype"), comment);
    }

    return (block, comments, data, position)
}

pub fn parse_at4(rdr: &mut BufReader<&File>, target: i64, mut position: i64) 
        -> (HashMap<i64, (At4Block, HashMap<String, String>, Option<Vec<u8>>)>, i64) {
    let mut at: HashMap<i64, (At4Block, HashMap<String, String>, Option<Vec<u8>>)> = HashMap::new();
    if target > 0{
        let (block, comments, data, pos) = parser_at4_block(rdr, target, position);
        let mut next_pointer = block.at_at_next;
        at.insert(target, (block, comments, data));
        position = pos;
        while next_pointer >0 {
            let block_start = next_pointer;
            let (block, comments, data, pos) = parser_at4_block(rdr, next_pointer, position);
            next_pointer = block.at_at_next;
            at.insert(block_start, (block, comments, data));
            position = pos;
        }
    }
    return (at, position)
}

#[derive(Debug)]
#[derive(BinRead)]
#[br(little)]
pub struct Ev4Block {
    ev_id: [u8; 4],  // DG
    reserved: [u8; 4],  // reserved
    ev_len: u64,      // Length of block in bytes
    ev_links: u64,         // # of links
    ev_ev_next: i64,     // Link to next EVBLOCK (linked list) (can be NIL)
    ev_ev_parent: i64,   // Referencing link to EVBLOCK with parent event (can be NIL).
    ev_ev_range: i64,    // Referencing link to EVBLOCK with event that defines the beginning of a range (can be NIL, must be NIL if ev_range_type ≠ 2).
    ev_tx_name: i64,     // Pointer to TXBLOCK with event name (can be NIL) Name must be according to naming rules stated in 4.4.2 Naming Rules. If available, the name of a named trigger condition should be used as event name. Other event types may have individual names or no names.
    ev_md_comment: i64,  // Pointer to TX/MDBLOCK with event comment and additional information, e.g. trigger condition or formatted user comment text (can be NIL)
    #[br(if(ev_links > 5), little, count = ev_links - 5)]
    links: Vec<i64>,       // links

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

fn parse_ev4_block(rdr: &mut BufReader<&File>, target: i64, mut position: i64) -> (Ev4Block, HashMap<String, String>, i64) {
    rdr.seek_relative(target - position).unwrap();
    let block: Ev4Block = match rdr.read_le() {
        Ok(v) => v,
        Err(e) => panic!("Error reading ev block \n{}", e),
    };  // reads the fh block
    position = target + i64::try_from(block.ev_len).unwrap();

    // Reads MD
    let (mut comments, pos) = comment(rdr, block.ev_md_comment, position);
    position = pos;

    // reads TX name
    if block.ev_tx_name > 0 {
        let (_, comment, pos) = 
            parse_comment(rdr, block.ev_tx_name, position);
        comments.insert(String::from("comment"), comment);
        position = pos;
    }

    return (block, comments, position)
}

pub fn parse_ev4(rdr: &mut BufReader<&File>, target: i64, mut position: i64) 
        -> (HashMap<i64, (Ev4Block, HashMap<String, String>)>, i64) {
    let mut ev: HashMap<i64, (Ev4Block, HashMap<String, String>)> = HashMap::new();
    if target > 0 {
        let (block, comments, pos) = parse_ev4_block(rdr, target, position);
        let mut next_pointer = block.ev_ev_next;
        ev.insert(target, (block, comments));
        position = pos;
        while next_pointer >0 {
            let block_start = next_pointer;
            let (block, comments, pos) = parse_ev4_block(rdr, next_pointer, position);
            next_pointer = block.ev_ev_next;
            ev.insert(block_start, (block, comments));
            position = pos;
        }
    }
    return (ev, position)
}

#[derive(Debug)]
#[derive(BinRead)]
#[br(little)]
pub struct Dg4Block {
    dg_id: [u8; 4],  // DG
    reserved: [u8; 4],  // reserved
    dg_len: u64,      // Length of block in bytes
    dg_links: u64,         // # of links 
    pub dg_dg_next: i64, // Pointer to next data group block (DGBLOCK) (can be NIL)
    pub dg_cg_first: i64, // Pointer to first channel group block (CGBLOCK) (can be NIL)
    pub dg_data: i64,     // Pointer to data block (DTBLOCK or DZBLOCK for this block type) or data list block (DLBLOCK of data blocks or its HLBLOCK)  (can be NIL)
    dg_md_comment: i64,    // comment
    pub dg_rec_id_size: u8,      // number of bytes used for record IDs. 0 no recordID
    reserved_2: [u8; 7],  // reserved
}

fn parse_dg4_block(rdr: &mut BufReader<&File>, target: i64, mut position: i64) -> (Dg4Block, HashMap<String, String>, i64) {
    rdr.seek_relative(target - position).unwrap();
    let block: Dg4Block = rdr.read_le().unwrap();
    position = target + 64;

    // Reads MD
    let (comments, position) = comment(rdr, block.dg_md_comment, position);

    return (block, comments, position)
}

#[derive(Debug)]
pub struct Dg4 {
    block: Dg4Block,  // DG Block
    comments: HashMap<String, String>,  // Comments
    cg: HashMap<i64, Cg4>,    // CG Block
}

pub fn parse_dg4(rdr: &mut BufReader<&File>, target: i64, mut position: i64) -> (HashMap<i64, Dg4>, i64) {
    let mut dg: HashMap<i64, Dg4> = HashMap::new();
    if target > 0 {
        let (block, comments, pos) = parse_dg4_block(rdr, target, position);
        position = pos;
        let mut next_pointer = block.dg_dg_next;
        let (cg, pos) = parse_cg4(rdr, block.dg_cg_first, position);
        let dg_struct = Dg4 {block, comments, cg};
        dg.insert(target, dg_struct);
        position = pos;
        while next_pointer >0 {
            let block_start = next_pointer;
            let (block, comments, pos) = parse_dg4_block(rdr, next_pointer, position);
            next_pointer = block.dg_dg_next;
            position = pos;
            let (cg, pos) = parse_cg4(rdr, block.dg_cg_first, position);
            let dg_struct = Dg4 {block, comments, cg};
            dg.insert(block_start, dg_struct);
            position = pos;
        }
    }
    return (dg, position)
}

#[derive(Debug)]
#[derive(BinRead)]
#[br(little)]
pub struct Cg4Block {
    cg_id: [u8; 4],  // DG
    reserved: [u8; 4],  // reserved
    cg_len: u64,      // Length of block in bytes
    cg_links: u64,         // # of links
    cg_cg_next: i64, // Pointer to next channel group block (CGBLOCK) (can be NIL)
    cg_cn_first: i64, // Pointer to first channel block (CNBLOCK) (can be NIL, must be NIL for VLSD CGBLOCK, i.e. if "VLSD channel group" flag (bit 0) is set)
    cg_tx_acq_name: i64, // Pointer to acquisition name (TXBLOCK) (can be NIL, must be NIL for VLSD CGBLOCK)
    cg_si_acq_source: i64, // Pointer to acquisition source (SIBLOCK) (can be NIL, must be NIL for VLSD CGBLOCK) See also rules for uniqueness explained in 4.4.3 Identification of Channels.
    cg_sr_first: i64, // Pointer to first sample reduction block (SRBLOCK) (can be NIL, must be NIL for VLSD CGBLOCK)
    cg_md_comment: i64, //Pointer to comment and additional information (TXBLOCK or MDBLOCK) (can be NIL, must be NIL for VLSD CGBLOCK)  
    #[br(if(cg_links > 6))]
    cg_cg_master: i64,
     // Data Members
    cg_record_id: u64, // Record ID, value must be less than maximum unsigned integer value allowed by dg_rec_id_size in parent DGBLOCK. Record ID must be unique within linked list of CGBLOCKs.
    cg_cycle_count: u64, // Number of cycles, i.e. number of samples for this channel group. This specifies the number of records of this type in the data block.
    cg_flags: u16, // Flags The value contains the following bit flags (see CG_F_xx):
    cg_path_separator: u16,
    cg_reserved: [u8; 4], // Reserved.
    cg_data_bytes: u32, // Normal CGBLOCK: Number of data Bytes (after record ID) used for signal values in record, i.e. size of plain data for each recorded sample of this channel group. VLSD CGBLOCK: Low part of a UINT64 value that specifies the total size in Bytes of all variable length signal values for the recorded samples of this channel group. See explanation for cg_inval_bytes.
    cg_inval_bytes: u32, // Normal CGBLOCK: Number of additional Bytes for record used for invalidation bits. Can be zero if no invalidation bits are used at all. Invalidation bits may only occur in the specified number of Bytes after the data Bytes, not within the data Bytes that contain the signal values. VLSD CGBLOCK: High part of UINT64 value that specifies the total size in Bytes of all variable length signal values for the recorded samples of this channel group, i.e. the total size in Bytes can be calculated by cg_data_bytes + (cg_inval_bytes << 32) Note: this value does not include the Bytes used to specify the length of each VLSD value!
}

fn parse_cg4_block(rdr: &mut BufReader<&File>, target: i64, mut position: i64) -> (Cg4Block, HashMap<String, String>, i64) {
    rdr.seek_relative(target - position).unwrap();
    let block: Cg4Block = rdr.read_le().unwrap();
    position = target + i64::try_from(block.cg_len).unwrap();

    // Reads MD
    let (comments, position) = comment(rdr, block.cg_md_comment, position);

    return (block, comments, position)
}

#[derive(Debug)]
pub struct Cg4 {
    block: Cg4Block,
    comments: HashMap<String, String>,  // Comments
    cn: HashMap<i64, Cn4> ,
}

pub fn parse_cg4(rdr: &mut BufReader<&File>, target: i64, mut position: i64) -> (HashMap<i64, Cg4>, i64) {
    let mut cg: HashMap<i64, Cg4> = HashMap::new();
    if target > 0 {
        let (block, comments, pos) = parse_cg4_block(rdr, target, position);
        position = pos;
        let mut next_pointer = block.cg_cg_next;
        let (cn, pos) = parse_cn4(rdr, block.cg_cn_first, position);
        let cg_struct = Cg4 {block, comments, cn};
        cg.insert(target, cg_struct);
        position = pos;
        while next_pointer >0 {
            let block_start = next_pointer;
            let (block, comments, pos) = parse_cg4_block(rdr, next_pointer, position);
            next_pointer = block.cg_cg_next;
            position = pos;
            let (cn, pos) = parse_cn4(rdr, block.cg_cn_first, position);
            let cg_struct = Cg4 {block, comments, cn};
            cg.insert(block_start, cg_struct);
            position = pos;
        }
    }
    return (cg, position)
}

#[derive(Debug)]
#[derive(BinRead)]
#[br(little)]
pub struct Cn4Block {
    cn_id: [u8; 4],  // DG
    reserved: [u8; 4],  // reserved
    cn_len: u64,      // Length of block in bytes
    cn_links: u64,         // # of links
    cn_cn_next: i64, // Pointer to next channel block (CNBLOCK) (can be NIL)
    cn_composition: i64, // Composition of channels: Pointer to channel array block (CABLOCK) or channel block (CNBLOCK) (can be NIL). Details see 4.18 Composition of Channels
    cn_tx_name: i64, // Pointer to TXBLOCK with name (identification) of channel. Name must be according to naming rules stated in 4.4.2 Naming Rules.
    cn_si_source: i64, // Pointer to channel source (SIBLOCK) (can be NIL) Must be NIL for component channels (members of a structure or array elements) because they all must have the same source and thus simply use the SIBLOCK of their parent CNBLOCK (direct child of CGBLOCK).
    cn_cc_conversion: i64, // Pointer to the conversion formula (CCBLOCK) (can be NIL, must be NIL for complex channel data types, i.e. for cn_data_type ≥ 10). If the pointer is NIL, this means that a 1:1 conversion is used (phys = int).  };
    cn_data: i64, // Pointer to channel type specific signal data For variable length data channel (cn_type = 1): unique link to signal data block (SDBLOCK) or data list block (DLBLOCK) or, only for unsorted data groups, referencing link to a VLSD channel group block (CGBLOCK). Can only be NIL if SDBLOCK would be empty. For synchronization channel (cn_type = 4): referencing link to attachment block (ATBLOCK) in global linked list of ATBLOCKs starting at hd_at_first. Cannot be NIL.
    cn_md_unit: i64, // Pointer to TXBLOCK/MDBLOCK with designation for physical unit of signal data (after conversion) or (only for channel data types "MIME sample" and "MIME stream") to MIME context-type text. (can be NIL). The unit can be used if no conversion rule is specified or to overwrite the unit specified for the conversion rule (e.g. if a conversion rule is shared between channels). If the link is NIL, then the unit from the conversion rule must be used. If the content is an empty string, no unit should be displayed. If an MDBLOCK is used, in addition the A-HDO unit definition can be stored, see Table 38. Note: for (virtual) master and synchronization channels the A-HDO definition should be omitted to avoid redundancy. Here the unit is already specified by cn_sync_type of the channel. In case of channel data types "MIME sample" and "MIME stream", the text of the unit must be the content-type text of a MIME type which specifies the content of the values of the channel (either fixed length in record or variable length in SDBLOCK). The MIME content-type string must be written in lowercase, and it must apply to the same rules as defined for at_tx_mimetype in 4.11 The Attachment Block ATBLOCK.
    cn_md_comment: i64, // Pointer to TXBLOCK/MDBLOCK with comment and additional information about the channel, see Table 37. (can be NIL)
    #[br(if(cn_links > 8))]
    links: i64,

  // Data Members
    cn_type: u8, // Channel type (see CN_T_xxx)
    cn_sync_type: u8, // Sync type: (see CN_S_xxx)
    cn_data_type: u8, // Channel data type of raw signal value (see CN_DT_xxx)
    cn_bit_offset: u8, // Bit offset (0-7): first bit (=LSB) of signal value after Byte offset has been applied (see 4.21.4.2 Reading the Signal Value). If zero, the signal value is 1-Byte aligned. A value different to zero is only allowed for Integer data types (cn_data_type ≤ 3) and if the Integer signal value fits into 8 contiguous Bytes (cn_bit_count + cn_bit_offset ≤ 64). For all other cases, cn_bit_offset must be zero.
    cn_byte_offset: u32, // Offset to first Byte in the data record that contains bits of the signal value. The offset is applied to the plain record data, i.e. skipping the record ID.
    cn_bit_count: u32, // Number of bits for signal value in record
    cn_flags: u32,     // Flags (see CN_F_xxx)
    cn_inval_bit_pos: u32, // Position of invalidation bit.
    cn_precision: u8, // Precision for display of floating point values. 0xFF means unrestricted precision (infinite). Any other value specifies the number of decimal places to use for display of floating point values. Only valid if "precision valid" flag (bit 2) is set
    cn_reserved: [u8;3], // Reserved
    cn_val_range_min: f64, // Minimum signal value that occurred for this signal (raw value) Only valid if "value range valid" flag (bit 3) is set.
    cn_val_range_max: f64, // Maximum signal value that occurred for this signal (raw value) Only valid if "value range valid" flag (bit 3) is set.
    cn_limit_min: f64,    // Lower limit for this signal (physical value for numeric conversion rule, otherwise raw value) Only valid if "limit range valid" flag (bit 4) is set.
    cn_limit_max: f64,    // Upper limit for this signal (physical value for numeric conversion rule, otherwise raw value) Only valid if "limit range valid" flag (bit 4) is set.
    cn_limit_ext_min: f64, // Lower extended limit for this signal (physical value for numeric conversion rule, otherwise raw value) Only valid if "extended limit range valid" flag (bit 5) is set.
    cn_limit_ext_max: f64, // Upper extended limit for this signal (physical value for numeric conversion rule, otherwise raw value) Only valid if "extended limit range valid" flag (bit 5) is set.
}

fn parse_cn4_block(rdr: &mut BufReader<&File>, target: i64, mut position: i64) -> (Cn4Block, String, String, i64) {
    rdr.seek_relative(target - position).unwrap();
    let block: Cn4Block = rdr.read_le().unwrap();
    position = target + i64::try_from(block.cn_len).unwrap();

    // Reads TX name
    let (_, name, position) = parse_comment(rdr, block.cn_md_comment, position);

    // Reads unit
    let (u, position) = comment(rdr, block.cn_md_comment, position);
    let unit = match u.get("TX") {
        Some(u) => u.to_string(),
        None => "".to_string(),
    };

    return (block, name, unit, position)
}

#[derive(Debug)]
pub struct Cn4 {
    block: Cn4Block,
    name: String,
    unit: String,
}

pub fn parse_cn4(rdr: &mut BufReader<&File>, target: i64, mut position: i64) -> (HashMap<i64, Cn4>, i64) {
    let mut cn: HashMap<i64, Cn4> = HashMap::new();
    if target > 0 {
        let (block, name, unit, pos) 
            = parse_cn4_block(rdr, target, position);
        let mut next_pointer = block.cn_cn_next;
        //let (cn, pos) = parse_cn4(rdr, target, position);
        let cn_struct = Cn4 {block, name, unit};
        cn.insert(target, cn_struct);
        position = pos;
        while next_pointer >0 {
            let block_start = next_pointer;
            let (block, name, unit, pos) 
                = parse_cn4_block(rdr, next_pointer, position);
            next_pointer = block.cn_cn_next;
            // let (cn, pos) = parse_cn4(rdr, target, position);
            let cn_struct = Cn4 {block, name, unit};
            cn.insert(block_start, cn_struct);
            position = pos;
        }
    }
    return (cn, position)
}