extern crate clap;

use clap::{Arg, App, SubCommand};
use std::io;
mod mdfinfo;
mod mdfreader;

fn main() -> io::Result<()>{
    let matches = App::new("mdfr")
                          .version("0.1.0")
                          .author("Aymeric Rateau <aymeric.rateau@gmail.com>")
                          .about("reads ASAM mdf file")
                          .arg(Arg::with_name("file")
                               .help("Sets the input file to use")
                               .required(true)
                               .index(1))
                          .arg(Arg::with_name("v")
                               .short("v")
                               .multiple(true)
                               .help("Sets the level of verbosity"))
                          .subcommand(SubCommand::with_name("test")
                                      .about("controls testing features")
                                      .version("0.1")
                                      .author("Aymeric Rateau <aymeric.rateau@gmail.com>")
                                      .arg(Arg::with_name("debug")
                                          .short("d")
                                          .help("print debug information verbosely")))
                          .get_matches();

    let file_name = matches.value_of("file")
        .expect("File name missing");

    if let Some(matches) = matches.subcommand_matches("test") {
        if matches.is_present("debug") {
            println!("Printing debug info...");
        } else {
            println!("Printing normally...");
        }
    }

    let info = mdfinfo::mdfinfo(file_name);
    mdfreader::mdfreader(file_name);

    //println!(r#"{:?}"#, info);

    Ok(())

}
