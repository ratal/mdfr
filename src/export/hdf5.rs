//! Exporting mdf to hdf5 files.
use anyhow::{Context, Error, Result};
use hdf5::dataset::*;
use hdf5::file::*;

use crate::mdfreader::Mdf;

/// writes mdf into hdf5 file
pub fn export_to_hdf5(mdf: &Mdf, file_name: &str) -> Result<(), Error> {}

/// writes a dataframe or channel group defined by a given channel into a hdf5 file
pub fn export_dataframe_to_hdf5(
    mdf: &Mdf,
    channel_name: &str,
    file_name: &str,
) -> Result<(), Error> {
}
