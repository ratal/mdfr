// mdfinfo module

//! This module is reading the mdf file blocks

use std::io::{BufReader, Read};
use std::fs::{File, OpenOptions};
use std::str;

pub mod mdfinfo3;
pub mod mdfinfo4;

use mdfinfo3::{MdfInfo3, parse_id3, hd3_parser, hd3_comment_parser};
use mdfinfo4::{MdfInfo4, parse_id4, hd4_parser, hd4_comment_parser};

#[derive(Debug)]
pub enum MdfInfo {
    V3(MdfInfo3),
    V4(MdfInfo4),
}

pub fn mdfinfo(file_name: &str) -> MdfInfo {
    let f: File = OpenOptions::new().read(true)
                    .write(false)
                    .open(file_name)
                    .expect("Cannot find the file");
    let mut rdr = BufReader::new(&f);
    // Read beginning of ID Block
    let mut id_file_id = [0u8; 8];
    rdr.read(&mut id_file_id).unwrap(); // "MDF     "
    let mut id_vers = [0u8; 4];
    rdr.read(&mut id_vers).unwrap();
    let ver_char:f32 = str::from_utf8(&id_vers).unwrap().parse().unwrap();
    let mut gap = [0u8; 4];
    rdr.read(&mut gap).unwrap();
    let mut prog = [0u8; 8];
    rdr.read(&mut prog).unwrap();
    let ver:u16;
    let mdf_info: MdfInfo;
    // Depending of version different blocks
    if ver_char < 4.0 {
        let id = parse_id3(&mut rdr, id_file_id, id_vers, prog);
        ver = id.id_ver;

        // Read HD Block
        let hd = hd3_parser(&mut rdr, ver);
        let (hd_comment, position) =  hd3_comment_parser(&mut rdr, &hd);

        mdf_info = MdfInfo::V3(MdfInfo3{ver, prog,
            idblock: id, hdblock: hd, hd_comment,
            });
    } else {

        let id = parse_id4(&mut rdr, id_file_id, id_vers, prog);
        ver = id.id_ver;

        // Read HD block
        let hd = hd4_parser(&mut rdr);
        let (hd_comment, position) = hd4_comment_parser(&mut rdr, &hd);

        // Read DG Block
        
        mdf_info = MdfInfo::V4(MdfInfo4{ver, prog,
            id_block: id, hd_block: hd, hd_comment,
            });
    };
    return mdf_info
}
