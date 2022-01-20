//! command line interface to load mdf file and manipulate it.
extern crate clap;

use clap::{App, Arg};
use std::io;
mod mdfinfo;
mod mdfreader;
mod mdfwriter;

fn main() -> io::Result<()> {
    let matches = App::new("mdfr")
        .version("0.1.0")
        .author("Aymeric Rateau <aymeric.rateau@gmail.com>")
        .about("reads ASAM mdf file")
        .arg(
            Arg::new("file")
                .help("Sets the input file to use")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::new("v")
                .multiple_values(true)
                .help("Sets the level of verbosity"),
        )
        .get_matches();

    let file_name = matches.value_of("file").expect("File name missing");

    if let Some(matches) = matches.subcommand_matches("test") {
        if matches.is_present("debug") {
            println!("Printing debug info...");
        } else {
            println!("Printing normally...");
        }
    }

    //let info = mdfinfo::mdfinfo(file_name);
    mdfreader::mdfreader(file_name);

    //println!(r#"{:?}"#, info);

    Ok(())
}
