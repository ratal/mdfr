// mdfinfo module

//! This module is reading the mdf file blocks

use std::io::{BufRead, Seek, SeekFrom};
use std::io::prelude::*;
use std::str;
use rayon::prelude::*;

pub mod mdfinfo3;
pub mod mdfinfo4;

use mdfinfo3::{MdfInfo3, parse_id3, hd3_parser};
use mdfinfo4::{MdfInfo4, parse_id4, hd4_parser};

#[derive(Debug)]
pub enum MdfInfo {
    V3(MdfInfo3),
    V4(MdfInfo4),
}

pub fn mdfinfo<R: Seek + BufRead>(f: &mut R) -> MdfInfo {
    // Read beginning of ID Block
    let mut id_file_id = [0; 8];
    f.take(8).read(&mut id_file_id).unwrap(); // "MDF     "
    let mut id_vers = [0; 4];
    f.take(4).read(&mut id_vers).unwrap();
    let ver_char:f32 = str::from_utf8(&id_vers).unwrap().parse().unwrap();
    f.seek(SeekFrom::Current(4)).unwrap();
    let mut prog = [0; 8];
    f.take(8).read(&mut prog).unwrap();
    let ver:u16;
    let mdf_info: MdfInfo;
    // Depending of version different blocks
    if ver_char < 4.0 {
        let mut idbuffer = [0; 40];
        f.read(&mut idbuffer).unwrap();
        let id = match parse_id3(&idbuffer, id_file_id, id_vers, prog).map(|x| x.1) {
            Ok(i) => i,
            Err(e) => panic!("Failed parsing the file ID Block : {:?}", e),
        };
        ver = id.id_ver;

        // Read HD Block
        let mut hd = match hd3_parser(f, ver).map(|x| x.1){
            Ok(i) => i,
            Err(e) => panic!("Failed parsing the file HD Block : {:?}", e),
        };

        mdf_info = MdfInfo::V3(MdfInfo3{ver, prog,
            idblock: id, hdblock: hd, 
            });
    } else {
        let mut buffer = [0; 40];
        f.read(&mut buffer).unwrap();
        let id = match parse_id4(&buffer, id_file_id, id_vers, prog).map(|x| x.1) {
            Ok(i) => i,
            Err(e) => panic!("Failed parsing the file ID Block : {:?}", e),
        };
        ver = id.id_ver;

        // Read HD block
        let mut hd = hd4_parser(f);
        
        mdf_info = MdfInfo::V4(MdfInfo4{ver, prog,
            idblock: id, hdblock: hd, 
            });
    };
    return mdf_info
}
