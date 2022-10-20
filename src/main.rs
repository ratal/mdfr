//! command line interface to load mdf file and manipulate it.
extern crate clap;

use arrow2::error::Error;
use clap::{Arg, Command};
mod export;
mod mdfinfo;
mod mdfreader;
mod mdfwriter;

fn main() -> Result<(), Error> {
    let matches = Command::new("mdfr")
        .bin_name("mdfr")
        .version("0.1.0")
        .author("Aymeric Rateau <aymeric.rateau@gmail.com>")
        .about("reads ASAM mdf file")
        .arg(
            Arg::new("file")
                .help("Sets the input file to use")
                .required(true)
                .num_args(1)
                .value_name("FILE_NAME")
                .index(1),
        )
        .arg(
            Arg::new("write")
                .long("write")
                .short('w')
                .required(false)
                .num_args(1)
                .value_name("FILE_NAME")
                .help("write the read content into a new mdf4.2 file"),
        )
        .arg(
            Arg::new("compress")
                .long("compress")
                .short('z')
                .action(clap::ArgAction::SetTrue)
                .help("compress data when writing into a new mdf4.2 file"),
        )
        .arg(
            Arg::new("export_to_parquet")
                .long("export_to_parquet")
                .short('p')
                .required(false)
                .num_args(1)
                .value_name("FILE_NAME")
                .help("Converts mdf into parquet file"),
        )
        .arg(
            Arg::new("parquet_compression")
                .long("parquet_compression")
                .required(false)
                .num_args(1)
                .value_name("ALGORITHM")
                .help("Compression algorithm for writing data in parquet file, valid values are snappy, gzip, lzo. Default is uncompressed"),
        )
        .arg(
            Arg::new("info")
                .short('i')
                .long("file_info")
                .action(clap::ArgAction::SetTrue)
                .help("prints file information"),
        )
        .get_matches();

    let file_name = matches
        .get_one::<String>("file")
        .expect("File name missing");

    let mut mdf_file = mdfreader::Mdf::new(file_name);
    if matches.get_flag("info") {
        println!("{:?}", mdf_file.get_master_channel_names_set());
    }

    let mdf4_file_name = matches.get_one::<String>("write");
    let parquet_file_name = matches.get_one::<String>("export_to_parquet");

    if mdf4_file_name.is_some() || parquet_file_name.is_some() {
        mdf_file.load_all_channels_data_in_memory();
    }

    let compression = matches.get_flag("compress");
    if let Some(file_name) = mdf4_file_name {
        mdf_file.write(file_name, compression);
    }

    let parquet_compression = matches.get_one::<String>("parquet_compression");
    if let Some(file_name) = parquet_file_name {
        mdf_file.export_to_parquet(file_name, parquet_compression.map(|x| &**x))?;
    }

    Ok(())
}

// TODO better error handling with anyhow
// TODO add C interface
