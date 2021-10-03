pub mod channel_data;
pub mod conversions4;
pub mod mdfreader3;
pub mod mdfreader4;
use crate::mdfinfo::MdfInfo;

pub fn mdfreader(file_name: &str) {
    // read file blocks
    let mut mdf = MdfInfo::new(file_name);
    mdf.load_all_channels_data_in_memory();
}
//TODO add Rust graphical interface ?
