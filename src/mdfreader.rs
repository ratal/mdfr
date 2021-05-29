pub mod mdfreader4;
pub mod mdfreader3;
use crate::mdfinfo::{mdfinfo, MdfInfo};
use mdfreader4::mdfreader4;
use mdfreader3::mdfreader3;
use std::{fs::OpenOptions, io::{BufReader, Cursor}, sync::Arc};
use std::fs::File;

pub fn mdfreader(file_name: &str) {
    // read file blocks
    let info = mdfinfo(file_name);

    let f: File = OpenOptions::new().read(true)
                    .write(false)
                    .open(file_name)
                    .expect("Cannot find the file");
    let mut rdr = BufReader::new(&f);

    match info {
        MdfInfo::V3(mut mdfinfo3) => mdfreader3(&mut rdr, &mut mdfinfo3),
        MdfInfo::V4(mut mdfinfo4) => mdfreader4(&mut rdr, &mut mdfinfo4),
    }
}
