//! Exporting mdf to Parquet files.
use arrow2::{
    array::Array,
    datatypes::DataType,
    datatypes::{Field, Metadata, Schema},
    error::{Error, Result},
    io::parquet::write::{
        array_to_columns, compress, to_parquet_schema, CompressedPage, CompressionOptions, DynIter,
        DynStreamingIterator, Encoding, FallibleStreamingIterator, FileWriter, Version,
        WriteOptions,
    },
    io::parquet::{read::ParquetError, write::transverse},
};
use codepage::to_encoding;
use encoding_rs::Encoding as EncodingRs;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, ParallelExtend, ParallelIterator,
};

use crate::{
    mdfinfo::{
        mdfinfo4::{Cn4, MdfInfo4},
        MdfInfo,
    },
    mdfreader::Mdf,
};

use std::collections::{HashSet, VecDeque};
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
pub fn export_to_parquet(mdf: &Mdf, file_name: &str, compression: Option<&str>) -> Result<()> {
    //let _ = data_type;
    // Create file
    let path = Path::new(file_name);

    let options = WriteOptions {
        write_statistics: false,
        version: Version::V2,
        compression: parquet_compression_from_string(compression),
        data_pagesize_limit: None,
    };

    // No other encoding yet implemented, to be reviewed later if needed.
    let encoding_map = |_data_type: &DataType| Encoding::Plain;

    let (arrow_data, mut arrow_schema) = mdf_data_to_arrow(mdf);
    arrow_schema
        .metadata
        .insert("file_name".to_string(), file_name.to_string());

    // declare encodings
    let encodings = (arrow_schema.fields)
        .par_iter()
        .map(|f| transverse(&f.data_type, encoding_map))
        .collect::<Vec<_>>();

    // derive the parquet schema (physical types) from arrow's schema.
    let parquet_schema =
        to_parquet_schema(&arrow_schema).expect("Failed to create SchemaDescriptor from Schema");

    let row_groups = arrow_data.iter().map(|batch| {
        // write batch to pages; parallelized by rayon
        let columns = batch
            .par_iter()
            .zip(parquet_schema.fields().to_vec())
            .zip(encodings.par_iter())
            .flat_map(move |((array, type_), encoding)| {
                let encoded_columns = array_to_columns(array, type_, options, encoding)
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

    let file = fs::File::create(path).expect("Failed to create file");
    let mut writer = FileWriter::try_new(file, arrow_schema.clone(), options)
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

/// returns arrow field from cn
#[inline]
fn cn4_field(mdfinfo4: &MdfInfo4, cn: &Cn4, data_type: DataType, is_nullable: bool) -> Field {
    let field = Field::new(cn.unique_name.clone(), data_type, is_nullable);
    let mut metadata = Metadata::new();
    if let Ok(Some(unit)) = mdfinfo4.sharable.get_tx(cn.block.cn_md_unit) {
        metadata.insert("unit".to_string(), unit);
    };
    if let Ok(Some(desc)) = mdfinfo4.sharable.get_tx(cn.block.cn_md_comment) {
        metadata.insert("description".to_string(), desc);
    };
    if let Some((Some(master_channel_name), _dg_pos, (_cg_pos, _rec_idd), (_cn_pos, _rec_pos))) =
        mdfinfo4.channel_names_set.get(&cn.unique_name)
    {
        metadata.insert(
            "master_channel".to_string(),
            master_channel_name.to_string(),
        );
    }
    if cn.block.cn_type == 4 {
        metadata.insert(
            "sync_channel".to_string(),
            cn.block.cn_sync_type.to_string(),
        );
    }
    field.with_metadata(metadata)
}

/// takes data of channel set from MdfInfo structure and stores in mdf.arrow_data
fn mdf_data_to_arrow(mdf: &Mdf) -> (Vec<Vec<Box<dyn Array>>>, Schema) {
    let mut chunk_index: usize = 0;
    let mut array_index: usize = 0;
    let mut field_index: usize = 0;
    let mut arrow_schema = Schema::default();
    match &mdf.mdf_info {
        MdfInfo::V4(mdfinfo4) => {
            let mut arrow_data: Vec<Vec<Box<dyn Array>>> = Vec::with_capacity(mdfinfo4.dg.len());
            arrow_schema.fields = Vec::<Field>::with_capacity(mdfinfo4.channel_names_set.len());
            for (_dg_block_position, dg) in mdfinfo4.dg.iter() {
                let mut channel_names_present_in_dg = HashSet::new();
                for channel_group in dg.cg.values() {
                    let cn = channel_group.channel_names.clone();
                    channel_names_present_in_dg.par_extend(cn);
                }
                if !channel_names_present_in_dg.is_empty() {
                    dg.cg.iter().for_each(|(_rec_id, cg)| {
                        let is_nullable: bool = cg.block.cg_inval_bytes > 0;
                        let mut columns =
                            Vec::<Box<dyn Array>>::with_capacity(cg.channel_names.len());
                        cg.cn.iter().for_each(|(_rec_pos, cn)| {
                            if !cn.data.is_empty() {
                                arrow_schema.fields.push(cn4_field(
                                    mdfinfo4,
                                    cn,
                                    cn.data.data_type().clone(),
                                    is_nullable,
                                ));
                                columns.push(cn.data.clone());
                                array_index += 1;
                                field_index += 1;
                            }
                        });
                        arrow_data.push(columns);
                        chunk_index += 1;
                        array_index = 0;
                    });
                }
            }
            (arrow_data, arrow_schema)
        }
        MdfInfo::V3(mdfinfo3) => {
            let mut arrow_data: Vec<Vec<Box<dyn Array>>> = Vec::with_capacity(mdfinfo3.dg.len());
            arrow_schema.fields = Vec::<Field>::with_capacity(mdfinfo3.channel_names_set.len());
            for (_dg_block_position, dg) in mdfinfo3.dg.iter() {
                for (_rec_id, cg) in dg.cg.iter() {
                    let mut columns = Vec::<Box<dyn Array>>::with_capacity(cg.channel_names.len());
                    for (_rec_pos, cn) in cg.cn.iter() {
                        if !cn.data.is_empty() {
                            let field = Field::new(
                                cn.unique_name.clone(),
                                cn.data.data_type().clone(),
                                false,
                            );
                            columns.push(cn.data.clone());
                            let mut metadata = Metadata::new();
                            if let Some(array) =
                                mdfinfo3.sharable.cc.get(&cn.block1.cn_cc_conversion)
                            {
                                let txt = array.0.cc_unit;
                                let encoding: &'static EncodingRs =
                                    to_encoding(mdfinfo3.id_block.id_codepage)
                                        .unwrap_or(encoding_rs::WINDOWS_1252);
                                let u: String = encoding.decode(&txt).0.into();
                                metadata.insert(
                                    "unit".to_string(),
                                    u.trim_end_matches(char::from(0)).to_string(),
                                );
                            };
                            metadata.insert("description".to_string(), cn.description.clone());
                            if let Some((
                                Some(master_channel_name),
                                _dg_pos,
                                (_cg_pos, _rec_idd),
                                _cn_pos,
                            )) = mdfinfo3.channel_names_set.get(&cn.unique_name)
                            {
                                metadata.insert(
                                    "master_channel".to_string(),
                                    master_channel_name.to_string(),
                                );
                            }
                            let field = field.with_metadata(metadata);
                            arrow_schema.fields.push(field);
                            array_index += 1;
                            field_index += 1;
                        }
                    }
                    arrow_data.push(columns);
                    chunk_index += 1;
                    array_index = 0;
                }
            }
            (arrow_data, arrow_schema)
        }
    }
}
