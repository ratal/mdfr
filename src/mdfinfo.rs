// mdfinfo module

//! This module is reading the mdf file blocks

use std::{collections::{HashMap, HashSet}, io::{BufReader, Read}};
use std::fs::{File, OpenOptions};
use std::str;

pub mod mdfinfo3;
pub mod mdfinfo4;

use mdfinfo3::{MdfInfo3, parse_id3, hd3_parser, hd3_comment_parser};
use mdfinfo4::{MdfInfo4, parse_id4, hd4_parser, hd4_comment_parser, extract_xml,
    parse_fh, parse_at4, parse_at4_comments, parse_ev4, parse_ev4_comments, parse_dg4};

#[derive(Debug)]
pub enum MdfInfo {
    V3(MdfInfo3),
    V4(MdfInfo4),
}

impl MdfInfo {
    pub fn get_version(&mut self) -> u16{
        match self {
            MdfInfo::V3(mdfinfo3) => mdfinfo3.ver,
            MdfInfo::V4(mdfinfo4) => mdfinfo4.ver,
        }
    }
}

pub fn mdfinfo(file_name: &str) -> MdfInfo {
    let f: File = OpenOptions::new().read(true)
                    .write(false)
                    .open(file_name)
                    .expect("Cannot find the file");
    let mut rdr = BufReader::new(&f);
    // Read beginning of ID Block
    let mut id_file_id = [0u8; 8];
    rdr.read_exact(&mut id_file_id).unwrap(); // "MDF     "
    let mut id_vers = [0u8; 4];
    rdr.read_exact(&mut id_vers).unwrap();
    let ver_char:f32 = str::from_utf8(&id_vers).unwrap().parse().unwrap();
    let mut gap = [0u8; 4];
    rdr.read_exact(&mut gap).unwrap();
    let mut prog = [0u8; 8];
    rdr.read_exact(&mut prog).unwrap();
    let ver:u16;
    let mdf_info: MdfInfo;
    let mut comments: HashMap<i64, HashMap<String, String>> = HashMap::new();
    // Depending of version different blocks
    if ver_char < 4.0 {
        let id = parse_id3(&mut rdr, id_file_id, id_vers, prog);
        ver = id.id_ver;

        // Read HD Block
        let (hd, position) = hd3_parser(&mut rdr, ver);
        let (hd_comment, position) =  hd3_comment_parser(&mut rdr, &hd, position);

        mdf_info = MdfInfo::V3(MdfInfo3{ver, prog,
            idblock: id, hdblock: hd, hd_comment,
            });
    } else {
        let mut channel_list: HashSet<String> = HashSet::new();

        let id = parse_id4(&mut rdr, id_file_id, id_vers, prog);
        ver = id.id_ver;

        // Read HD block
        let hd = hd4_parser(&mut rdr);
        let (hd_comment, position) = hd4_comment_parser(&mut rdr, &hd);
        // FH block
        let (fh, position) = parse_fh(&mut rdr, hd.hd_fh_first, position);

        // AT Block read
        let (at, position) = parse_at4(&mut rdr, hd.hd_at_first, position);
        let (c, position) = parse_at4_comments(&mut rdr, &at, position);
        comments.extend(c.into_iter());
 
        // EV Block read
        let (ev, position) = parse_ev4(&mut rdr, hd.hd_ev_first, position);
        let (c, position) = parse_ev4_comments(&mut rdr, &ev, position);
        comments.extend(c.into_iter());

        // Read DG Block
        let (dg, mut unit, mut desc, position) 
            = parse_dg4(&mut rdr, hd.hd_dg_first, position);
        extract_xml(&mut unit);
        extract_xml(&mut desc);
        
        mdf_info = MdfInfo::V4(MdfInfo4{ver, prog,
            id_block: id, hd_block: hd, hd_comment, comments, fh, at, ev, dg
            });
    };
    mdf_info
}
