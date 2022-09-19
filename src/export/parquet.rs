//! Exporting mdf to Parquet files.
use arrow2::{
    datatypes::DataType,
    error::{Error, Result},
    io::parquet::write::{
        array_to_columns, compress, to_parquet_schema, CompressedPage, CompressionOptions, DynIter,
        DynStreamingIterator, Encoding, FallibleStreamingIterator, FileWriter, Version,
        WriteOptions,
    },
    io::parquet::{read::ParquetError, write::transverse},
};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::mdfreader::Mdf;

use std::collections::VecDeque;
use std::{fs, path::Path};

struct Bla {
    columns: VecDeque<CompressedPage>,
    current: Option<CompressedPage>,
}

impl Bla {
    pub fn new(columns: VecDeque<CompressedPage>) -> Self {
        Self {
            columns,
            current: None,
        }
    }
}

impl FallibleStreamingIterator for Bla {
    type Item = CompressedPage;
    type Error = Error;

    fn advance(&mut self) -> Result<()> {
        self.current = self.columns.pop_front();
        Ok(())
    }

    fn get(&self) -> Option<&Self::Item> {
        self.current.as_ref()
    }
}

/// writes mdf into parquet file
pub fn export_to_parquet(mdf: &mut Mdf, file_name: &str, compression: Option<&str>) -> Result<()> {
    // Create file
    let path = Path::new(file_name);

    let options = WriteOptions {
        write_statistics: false,
        version: Version::V2,
        compression: parquet_compression_from_string(compression),
    };

    let encoding_map = |data_type: &DataType| {
        match data_type.to_physical_type() {
            // remaining is plain
            _ => Encoding::Plain,
        }
    };

    // declare encodings
    let encodings = (mdf.arrow_schema.fields)
        .par_iter()
        .map(|f| transverse(&f.data_type, encoding_map))
        .collect::<Vec<_>>();

    // derive the parquet schema (physical types) from arrow's schema.
    let parquet_schema = to_parquet_schema(&mdf.arrow_schema)
        .expect("Failed to create SchemaDescriptor from Schema");

    let row_groups = mdf.arrow_data.iter().map(|batch| {
        // write batch to pages; parallelized by rayon
        let columns = batch
            .par_iter()
            .zip(parquet_schema.fields().to_vec())
            .zip(encodings.par_iter())
            .flat_map(move |((array, type_), encoding)| {
                let encoded_columns = array_to_columns(array, type_, options, &encoding)
                    .expect("Could not convert arrow array to column");
                encoded_columns
                    .into_iter()
                    .map(|encoded_pages| {
                        let encoded_pages = DynIter::new(encoded_pages.into_iter().map(|x| {
                            x.map_err(|e| ParquetError::FeatureNotSupported(e.to_string()))
                        }));
                        encoded_pages
                            .map(|page| {
                                compress(page?, vec![], options.compression).map_err(|x| x.into())
                            })
                            .collect::<Result<VecDeque<_>>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Result<Vec<VecDeque<CompressedPage>>>>()?;

        let row_group = DynIter::new(
            columns
                .into_iter()
                .map(|column| Ok(DynStreamingIterator::new(Bla::new(column)))),
        );
        Result::Ok(row_group)
    });

    let file = fs::File::create(&path).expect("Failed to create file");
    let mut writer = FileWriter::try_new(file, mdf.arrow_schema.clone(), options)
        .expect("Failed to write parquet file");

    // write data in file
    for group in row_groups {
        writer.write(group?)?;
    }
    writer.end(None).expect("Failed to write footer");
    Ok(())
}

/// converts a clap compression string into a CompressionOptions enum
pub fn parquet_compression_from_string(compression_option: Option<&str>) -> CompressionOptions {
    match compression_option {
        Some(option) => match option {
            "snappy" => CompressionOptions::Snappy,
            "gzip" => CompressionOptions::Gzip(None),
            "lzo" => CompressionOptions::Lzo,
            "brotli" => CompressionOptions::Brotli(None),
            "lz4" => CompressionOptions::Lz4,
            "lz4raw" => CompressionOptions::Lz4Raw,
            _ => CompressionOptions::Uncompressed,
        },
        None => CompressionOptions::Uncompressed,
    }
}
