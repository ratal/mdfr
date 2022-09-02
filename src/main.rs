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
                .takes_value(true)
                .index(1),
        )
        .arg(
            Arg::new("write")
                .long("write")
                .short('w')
                .required(false)
                .takes_value(true)
                .help("write read content into mdf4.2 file"),
        )
        .arg(
            Arg::new("compress")
                .long("compress")
                .short('z')
                .required(false)
                .takes_value(false)
                .help("write read content compressed into mdf4.2 file"),
        )
        .arg(
            Arg::new("convert3to4")
                .long("convert3to4")
                .short('c')
                .required(false)
                .takes_value(true)
                .help("Converts mdf version 3.x to 4.2"),
        )
        .arg(
            Arg::new("export_to_parquet")
                .long("export_to_parquet")
                .short('p')
                .required(false)
                .takes_value(true)
                .help("Converts mdf into parquet file"),
        )
        .arg(
            Arg::new("parquet_compression")
                .long("parquet_compression")
                .required(false)
                .takes_value(true)
                .help("Compresses data in parquet file, valid values are snappy, gzip, lzo"),
        )
        .get_matches();

    let file_name = matches.value_of("file").expect("File name missing");
    let mdf4_file_name = matches.value_of("write");

    let mut mdf_file = mdfreader::mdfreader(file_name);

    let compression = matches.is_present("compress");

    if let Some(file_name) = mdf4_file_name {
        mdf_file.write(file_name, compression);
    }

    let convert3to4_file_name = matches.value_of("convert3to4");
    if let Some(file_name) = convert3to4_file_name {
        mdf_file.mdf_info.convert3to4(file_name);
    }

    let parquet_compression = matches.value_of("parquet_compression");

    let parquet_file_name = matches.value_of("export_to_parquet");
    if let Some(file_name) = parquet_file_name {
        mdf_file.export_to_parquet(file_name, parquet_compression)?;
    }

    Ok(())
}
