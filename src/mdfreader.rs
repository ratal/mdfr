pub mod mdfreader4;
pub mod mdfreader3;
use crate::mdfinfo::{mdfinfo, MdfInfo};
use mdfreader4::mdfreader4;
use mdfreader3::mdfreader3;

pub fn mdfreader(file_name: &str) {
    // read file blocks
    let mut info = mdfinfo(file_name);

    match info {
        MdfInfo::V3(mut mdfinfo3) => mdfreader3(&mut mdfinfo3),
        MdfInfo::V4(mut mdfinfo4) => mdfreader4(&mut mdfinfo4),
    }
}
