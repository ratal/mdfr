//! This module contains the data reading features
pub mod channel_data;
pub mod conversions3;
pub mod conversions4;
pub mod data_read3;
pub mod data_read4;
pub mod mdfreader3;
pub mod mdfreader4;
use crate::mdfinfo::MdfInfo;

/// reads files metadata and loads all channels data into memory
pub fn mdfreader(file_name: &str) -> MdfInfo {
    // read file blocks
    let mut mdf = MdfInfo::new(file_name);
    mdf.load_all_channels_data_in_memory();
    mdf
}
