//! command line interface to load mdf file and manipulate it.
extern crate clap;

use clap::{Arg, Command};
use std::io;
mod mdfinfo;
mod mdfreader;
mod mdfwriter;

fn main() -> io::Result<()> {
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
        .get_matches();

    let file_name = matches.value_of("file").expect("File name missing");
    let mdf4_file_name = matches.value_of("write");

    let mut mdf_file = mdfreader::mdfreader(file_name);

    let compression = matches.is_present("compress");

    if let Some(file_name) = mdf4_file_name {
        mdf_file.write(&file_name, compression);
    }

    let convert3to4_file_name = matches.value_of("convert3to4");
    if let Some(file_name) = convert3to4_file_name {
        mdf_file.convert3to4(file_name);
    }

    Ok(())
}
