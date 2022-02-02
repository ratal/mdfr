//! command line interface to load mdf file and manipulate it.
extern crate clap;

use clap::{App, Arg};
use std::io;
mod mdfinfo;
mod mdfreader;
mod mdfwriter;

fn main() -> io::Result<()> {
    let matches = App::new("mdfr")
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
            Arg::new("verbose")
                .long("verbose")
                .short('v')
                .takes_value(true)
                .help("Sets the level of verbosity"),
        )
        .get_matches();

    let file_name = matches.value_of("file").expect("File name missing");

    //let info = mdfinfo::mdfinfo(file_name);
    mdfreader::mdfreader(file_name);

    //println!(r#"{:?}"#, info);

    Ok(())
}
