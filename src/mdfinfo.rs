// mdfinfo module

//! This module is reading the mdf file blocks

use std::collections::HashMap;
use std::io::{BufReader, Read};
use std::fs::{File, OpenOptions};
use std::str;
use std::fmt;

pub mod mdfinfo3;
pub mod mdfinfo4;

use mdfinfo3::{MdfInfo3, parse_id3, hd3_parser, hd3_comment_parser};
use mdfinfo4::{MdfInfo4, parse_id4, hd4_parser, hd4_comment_parser, extract_xml,
    parse_fh, parse_at4, parse_at4_comments, parse_ev4, parse_ev4_comments, parse_dg4,
    build_channel_db, SharableBlocks};

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

impl fmt::Display for MdfInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MdfInfo::V3(mdfinfo3) => {
                writeln!(f, "Version : {}", mdfinfo3.ver)?;
                writeln!(f, "Version : {:?}", mdfinfo3.hdblock)
            },
            MdfInfo::V4(mdfinfo4) => {
                writeln!(f, "Version : {}", mdfinfo4.ver)?;
                writeln!(f, "{}\n", mdfinfo4.hd_block)?;
                let comments = &mdfinfo4.hd_comment;
                Ok(for c in comments.iter() {
                    writeln!(f, "{} {}", c.0, c.1)?;
                })
            }
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
        
        let mut sharable: SharableBlocks = SharableBlocks {md: HashMap::new(), tx: HashMap::new(), 
            cc: HashMap::new(), si: HashMap::new()};

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
        sharable.md.extend(c.into_iter());
 
        // EV Block read
        let (ev, position) = parse_ev4(&mut rdr, hd.hd_ev_first, position);
        let (c, position) = parse_ev4_comments(&mut rdr, &ev, position);
        sharable.md.extend(c.into_iter());

        // Read DG Block
        let (mut dg, _) 
            = parse_dg4(&mut rdr, hd.hd_dg_first, position, &mut sharable);
        extract_xml(&mut sharable.tx);  // extract xml from text

        // make channel names unique, list channels and create master dictionnary
        let db = build_channel_db(&mut dg, &sharable);
        //println!("{}", db);
        
        mdf_info = MdfInfo::V4(MdfInfo4{ver, prog,
            id_block: id, hd_block: hd, hd_comment, fh, at, ev, dg, sharable, db,
            });
    };
    mdf_info
}
