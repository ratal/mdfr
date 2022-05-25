//! Exporting mdf to Parquet files.
use arrow2::{
    datatypes::PhysicalType,
    error::{ArrowError, Result},
    io::parquet::write::{
        array_to_pages, compress, to_parquet_schema, CompressedPage, CompressionOptions, DynIter,
        DynStreamingIterator, Encoding, FallibleStreamingIterator, FileWriter, Version,
        WriteOptions,
    },
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

use crate::mdfinfo::mdfinfo4::MdfInfo4;

use std::collections::VecDeque;
use std::{fs, path::Path};

use super::arrow::mdf4_data_to_arrow;

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
    type Error = ArrowError;

    fn advance(&mut self) -> Result<()> {
        self.current = self.columns.pop_front();
        Ok(())
    }

    fn get(&self) -> Option<&Self::Item> {
        self.current.as_ref()
    }
}

pub fn export_to_parquet(
    info: &mut MdfInfo4,
    file_name: &str,
    compression: Option<&str>,
) -> Result<()> {
    // create arrowchunks and schema
    let (batches, schema) = mdf4_data_to_arrow(info);
    // Create file
    let path = Path::new(file_name);

    let options = WriteOptions {
        write_statistics: true,
        version: Version::V2,
        compression: parquet_compression_from_string(compression),
    };

    // declare encodings
    let encodings = schema.fields.par_iter().map(|field| {
        match field.data_type().to_physical_type() {
            // let's be fancy and use delta-encoding for binary fields
            PhysicalType::Binary
            | PhysicalType::LargeBinary
            | PhysicalType::FixedSizeBinary
            | PhysicalType::Utf8
            | PhysicalType::LargeUtf8 => Encoding::DeltaLengthByteArray,
            // remaining is plain
            _ => Encoding::Plain,
        }
    });

    // derive the parquet schema (physical types) from arrow's schema.
    let parquet_schema = to_parquet_schema(&schema)?;

    let row_groups = batches.iter().map(|batch| {
        // write batch to pages; parallelized by rayon
        let columns = batch
            .columns()
            .par_iter()
            .zip(parquet_schema.columns().to_vec().into_par_iter())
            .zip(encodings.clone())
            .map(|((array, col_descriptor), encoding)| {
                // create encoded and compressed pages this column
                let encoded_pages =
                    array_to_pages(array.as_ref(), col_descriptor.descriptor, options, encoding)
                        .expect("could not convert array to pages");
                encoded_pages
                    .map(|page| compress(page?, vec![], options.compression).map_err(|x| x.into()))
                    .collect::<Result<VecDeque<_>>>()
            })
            .collect::<Result<Vec<VecDeque<CompressedPage>>>>()
            .expect("could not collect compressed pages");

        DynIter::new(
            columns
                .into_iter()
                .map(|column| Ok(DynStreamingIterator::new(Bla::new(column)))),
        )
    });

    let file = fs::File::create(&path)?;
    let mut writer = FileWriter::try_new(file, schema.clone(), options)?;
    writer.start()?;
    // write data in file
    for group in row_groups {
        writer.write(group)?;
    }
    writer.end(None)?;
    Ok(())
}

pub fn parquet_compression_from_string(compression_option: Option<&str>) -> CompressionOptions {
    match compression_option {
        Some(option) => match option {
            "snappy" => CompressionOptions::Snappy,
            "gzip" => CompressionOptions::Gzip,
            "lzo" => CompressionOptions::Lzo,
            "brotli" => CompressionOptions::Brotli,
            "lz4" => CompressionOptions::Lz4,
            "lz4raw" => CompressionOptions::Lz4Raw,
            _ => CompressionOptions::Uncompressed,
        },
        None => CompressionOptions::Uncompressed,
    }
}
