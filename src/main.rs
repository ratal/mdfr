//! command line interface to load mdf file and manipulate it.
use clap::{Arg, Command};
mod data_holder;
mod export;
mod mdfinfo;
mod mdfreader;
mod mdfwriter;
use anyhow::{Context, Error, Result};
use env_logger::Env;
use log::info;

fn main() -> Result<(), Error> {
    env_logger::Builder::from_env(Env::default().default_filter_or("warn")).init();
    let matches = Command::new("mdfr")
        .bin_name("mdfr")
        .version(env!("CARGO_PKG_VERSION"))
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
                .num_args(1)
                .value_name("FILE_NAME")
                .help("write the read content into a new mdf4.3 file"),
        )
        .arg(
            Arg::new("compress")
                .long("compress")
                .short('z')
                .action(clap::ArgAction::SetTrue)
                .help("compress data when writing into a new mdf4.3 file (defaults to deflate)"),
        )
        .arg(
            Arg::new("mdf4_compression")
                .long("mdf4_compression")
                .num_args(1)
                .value_name("ALGORITHM")
                .value_parser(["deflate", "deflate_transpose", "zstd", "zstd_transpose", "lz4", "lz4_transpose", "no_compression"])
                .help("Compression algorithm for writing data in mdf4 file. Default is deflate_transpose"),
        )
        .arg(
            Arg::new("export_to_parquet")
                .long("export_to_parquet")
                .short('p')
                .num_args(1)
                .value_name("FILE_NAME")
                .help("Converts mdf into parquet file, file name to be given without extension"),
        )
        .arg(
            Arg::new("parquet_compression")
                .long("parquet_compression")
                .num_args(1)
                .value_name("ALGORITHM")
                .value_parser(["snappy", "gzip", "lzo", "lz4", "zstd", "brotli"])
                .help("Compression algorithm for writing data in parquet file. Default is uncompressed"),
        )
        .arg(
            Arg::new("export_to_hdf5")
                .long("export_to_hdf5")
                .short('m')
                .num_args(1)
                .value_name("FILE_NAME")
                .help("Converts mdf into hdf5 file, file name to be given without extension"),
        )
        .arg(
            Arg::new("hdf5_compression")
                .long("hdf5_compression")
                .num_args(1)
                .value_name("FILTER")
                .value_parser(["deflate", "lz4", "zstd"])
                .help("Compression filter for writing data in hdf5 file. Default is uncompressed"),
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
        .context("File name missing")?;

    let mut mdf_file = mdfreader::Mdf::new(file_name)
        .with_context(|| format!("failed reading metadata from file {file_name}"))?;

    if matches.get_flag("info") {
        println!("{:?}", mdf_file.get_master_channel_names_set());
    }

    let mdf4_file_name = matches.get_one::<String>("write");
    let parquet_file_name = matches.get_one::<String>("export_to_parquet");
    let hdf5_file_name = matches.get_one::<String>("export_to_hdf5");

    if mdf4_file_name.is_some() || parquet_file_name.is_some() || hdf5_file_name.is_some() {
        mdf_file
            .load_all_channels_data_in_memory()
            .with_context(|| format!("failed reading channels data from file {file_name}"))?;
        info!("loaded all channels data in memory from file {file_name}");
    }

    let _compression = matches.get_flag("compress");
    let mdf4_compression_str = matches
        .get_one::<String>("mdf4_compression")
        .map(String::as_str);

    let compression_algo = match mdf4_compression_str {
        Some("deflate") => crate::mdfinfo::mdfinfo4::CompressionAlgorithm::Deflate,
        Some("deflate_transpose") => {
            crate::mdfinfo::mdfinfo4::CompressionAlgorithm::DeflateTranspose
        }
        Some("zstd") => crate::mdfinfo::mdfinfo4::CompressionAlgorithm::Zstd,
        Some("zstd_transpose") => crate::mdfinfo::mdfinfo4::CompressionAlgorithm::ZstdTranspose,
        Some("lz4") => crate::mdfinfo::mdfinfo4::CompressionAlgorithm::Lz4,
        Some("lz4_transpose") => crate::mdfinfo::mdfinfo4::CompressionAlgorithm::Lz4Transpose,
        Some("no_compression") => crate::mdfinfo::mdfinfo4::CompressionAlgorithm::NoCompression,
        None if _compression => crate::mdfinfo::mdfinfo4::CompressionAlgorithm::Deflate,
        _ => crate::mdfinfo::mdfinfo4::CompressionAlgorithm::DeflateTranspose,
    };

    if let Some(file_name) = mdf4_file_name {
        mdf_file.write(file_name, compression_algo)?;
        if compression_algo != crate::mdfinfo::mdfinfo4::CompressionAlgorithm::NoCompression {
            info!(
                "Wrote mdf4 file {file_name} with compression {:?}",
                compression_algo
            );
        } else {
            info!("Wrote mdf4 file {file_name} without compression");
        }
    }

    #[cfg(feature = "parquet")]
    {
        let parquet_compression = matches.get_one::<String>("parquet_compression");
        if let Some(file_name) = parquet_file_name {
            mdf_file
                .export_to_parquet(file_name, parquet_compression.map(String::as_str))
                .with_context(|| format!("failed to export into parquet file {file_name}"))?;
            info!("Wrote parquet file {file_name} with compression {parquet_compression:?}");
        }
    }

    #[cfg(feature = "hdf5")]
    {
        let hdf5_compression = matches.get_one::<String>("hdf5_compression");
        if let Some(file_name) = hdf5_file_name {
            mdf_file
                .export_to_hdf5(file_name, hdf5_compression.map(String::as_str))
                .with_context(|| format!("failed to export into hdf5 file {file_name}"))?;
            info!("Wrote hdf5 file {file_name}");
        }
    }

    Ok(())
}
