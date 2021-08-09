pub mod conversions4;
pub mod mdfreader3;
pub mod mdfreader4;
pub mod channel_data;
use crate::mdfinfo::{mdfinfo, MdfInfo};
use mdfreader3::mdfreader3;
use mdfreader4::mdfreader4;
use std::fs::File;
use std::{fs::OpenOptions, io::BufReader};

pub fn mdfreader(file_name: &str) {
    // read file blocks
    let mdf = mdfinfo(file_name);
    println!("{}", mdf);
    let f: File = OpenOptions::new()
        .read(true)
        .write(false)
        .open(file_name)
        .expect("Cannot find the file");
    let mut rdr = BufReader::new(&f);

    match mdf {
        MdfInfo::V3(mut mdfinfo3) => mdfreader3(&mut rdr, &mut mdfinfo3),
        MdfInfo::V4(mut mdfinfo4) => mdfreader4(&mut rdr, &mut mdfinfo4),
    }
}
