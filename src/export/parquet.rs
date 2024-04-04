//! Exporting mdf to Parquet files.
use anyhow::{Context, Error, Result};
use arrow::{
    array::{Array, RecordBatch},
    datatypes::{Field, Schema, SchemaBuilder},
};
use codepage::to_encoding;
use encoding_rs::Encoding as EncodingRs;
use parquet::{
    arrow::arrow_writer::ArrowWriter,
    basic::{BrotliLevel, Compression, Encoding, GzipLevel, ZstdLevel},
    file::{
        metadata::KeyValue,
        properties::{WriterProperties, WriterVersion},
    },
};
use rayon::iter::ParallelExtend;

use crate::{
    mdfinfo::{
        mdfinfo3::{Cn3, MdfInfo3},
        mdfinfo4::{Cg4, Cn4, Dg4, MdfInfo4},
        MdfInfo,
    },
    mdfreader::Mdf,
};

use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::BufWriter,
    path::Path,
    sync::Arc,
};

/// writes mdf into parquet file
pub fn export_to_parquet(
    mdf: &Mdf,
    file_name: &str,
    compression: Option<&str>,
) -> Result<(), Error> {
    let parquet_compression = parquet_compression_from_string(compression);
    match &mdf.mdf_info {
        MdfInfo::V4(mdfinfo4) => {
            mdfinfo4.dg.iter().try_for_each(
                |(_dg_block_position, dg): (&i64, &Dg4)| -> Result<(), Error> {
                    let mut channel_names_present_in_dg = HashSet::new();
                    for channel_group in dg.cg.values() {
                        let cn = channel_group.channel_names.clone();
                        channel_names_present_in_dg.par_extend(cn);
                    }
                    if !channel_names_present_in_dg.is_empty() {
                        dg.cg.iter().try_for_each(
                            |(rec_id, cg): (&u64, &Cg4)| -> Result<(), Error> {
                                let mut columns =
                                    Vec::<Arc<dyn Array>>::with_capacity(cg.channel_names.len());
                                let mut fields =
                                    SchemaBuilder::with_capacity(cg.channel_names.len());
                                cg.cn
                                    .iter()
                                    .try_for_each(
                                        |(_rec_pos, cn): (&i32, &Cn4)| -> Result<(), Error> {
                                            if !cn.data.is_empty() {
                                                fields.push(mdf4_field(mdfinfo4, cn));
                                                columns.push(cn.data.finish_cloned());
                                            }
                                            Ok(())
                                        },
                                    )
                                    .context("failed extracting data")?;
                                if !columns.is_empty() {
                                    // write data in file
                                    if let Some(master_channel) = &cg.master_channel_name {
                                        fields.metadata_mut().insert("master_channel".to_owned(), master_channel.to_string());}
                                    let finalised_arrow_schema = fields.finish();
                                    write_data(
                                        cg.master_channel_name.clone(),
                                        rec_id,
                                        file_name,
                                        parquet_compression,
                                        finalised_arrow_schema,
                                        columns,
                                    )
                                    .with_context(|| {
                                        format!(
                                            "failed writing data in parquet for rec id {}, master {:?}",
                                            rec_id, cg.master_channel_name
                                        )
                                    })?;
                                }
                                Ok(())
                            },
                        )?;
                    }
                    Ok(())
                },
            )?;
        }
        MdfInfo::V3(mdfinfo3) => {
            for (_dg_block_position, dg) in mdfinfo3.dg.iter() {
                for (rec_id, cg) in dg.cg.iter() {
                    let mut columns = Vec::<Arc<dyn Array>>::with_capacity(cg.channel_names.len());
                    let mut fields = SchemaBuilder::with_capacity(cg.channel_names.len());
                    cg.cn
                        .iter()
                        .try_for_each(|(_rec_pos, cn): (&u32, &Cn3)| -> Result<(), Error> {
                            if !cn.data.is_empty() {
                                fields.push(mdf3_field(mdfinfo3, cn));
                                columns.push(cn.data.finish_cloned());
                            }
                            Ok(())
                        })
                        .context("failed extracting data")?;
                    if !columns.is_empty() {
                        // write data in file
                        if let Some(master_channel) = &cg.master_channel_name {
                            fields
                                .metadata_mut()
                                .insert("master_channel".to_owned(), master_channel.to_string());
                        }
                        let finalised_arrow_schema = fields.finish();
                        write_data(
                            cg.master_channel_name.clone(),
                            &(*rec_id as u64),
                            file_name,
                            parquet_compression,
                            finalised_arrow_schema,
                            columns,
                        )
                        .with_context(|| {
                            format!(
                                "failed writing data in parquet for rec id {}, master {:?}",
                                rec_id, cg.master_channel_name
                            )
                        })?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// converts a clap compression string into a CompressionOptions enum
pub fn parquet_compression_from_string(compression_option: Option<&str>) -> Compression {
    match compression_option {
        Some(option) => match option {
            "snappy" => Compression::SNAPPY,
            "gzip" => Compression::GZIP(GzipLevel::try_new(6).expect("Wrong Gzip level")),
            "lzo" => Compression::LZO,
            "brotli" => Compression::BROTLI(BrotliLevel::try_new(1).expect("Wrong Brotli level")),
            "lz4" => Compression::LZ4,
            "lz4raw" => Compression::LZ4_RAW,
            "zstd" => Compression::ZSTD(ZstdLevel::try_new(1).expect("Wrong Zstd level")),
            _ => Compression::UNCOMPRESSED,
        },
        None => Compression::UNCOMPRESSED,
    }
}

// create

/// Create parquet file name appending Channel Group's master channel
/// Or if no master existing, add.
/// Appending at the end of name the . parquet file extension
fn create_parquet_writer(
    file: &str,
    compression: Compression,
    finalised_arrow_schema: Schema,
    master_channel: Option<String>,
    rec_id: &u64,
) -> Result<ArrowWriter<BufWriter<File>>, Error> {
    let base_path = Path::new(file);
    let mut master_channel_name = match master_channel {
        Some(name) => name,
        None => rec_id.to_string(),
    };
    master_channel_name.insert_str(0, &r"_");
    let mut file_name = base_path
        .file_name()
        .context("no given file name")?
        .to_os_string();
    file_name.push(master_channel_name);
    let mut buf_path = base_path.with_file_name(file_name.as_os_str());
    buf_path.set_extension("parquet");
    let path = buf_path.into_boxed_path();
    let file = std::io::BufWriter::new(
        std::fs::File::create(path.clone())
            .with_context(|| format!("Failed to create file {:?}", path))?,
    );
    let options = WriterProperties::builder()
        .set_compression(compression)
        .set_writer_version(WriterVersion::PARQUET_1_0)
        .set_encoding(Encoding::PLAIN)
        .set_key_value_metadata(Some(vec![KeyValue::new(
            "file_name".to_string(),
            file_name
                .into_string()
                .expect("file name contains invalid Unicode data"),
        )]))
        .build();

    Ok(ArrowWriter::try_new(
        file,
        Arc::new(finalised_arrow_schema.clone()),
        Some(options.clone()),
    )
    .with_context(|| {
        format!(
            "Failed to write parquet file with schema {:?} and options {:?}",
            finalised_arrow_schema, options
        )
    })?)
}

/// create mdf4 channel field
fn mdf4_field(mdfinfo4: &Box<MdfInfo4>, cn: &Cn4) -> Field {
    let field = Field::new(
        cn.unique_name.clone(),
        cn.data.arrow_data_type().clone(),
        cn.data.validity().is_some(),
    );
    let mut metadata = HashMap::<String, String>::new();
    if let Ok(Some(unit)) = mdfinfo4.sharable.get_tx(cn.block.cn_md_unit) {
        if !unit.is_empty() {
            metadata.insert("unit".to_string(), unit);
        }
    };
    if let Ok(Some(desc)) = mdfinfo4.sharable.get_tx(cn.block.cn_md_comment) {
        if !desc.is_empty() {
            metadata.insert("description".to_string(), desc);
        }
    };
    if cn.block.cn_type == 4 {
        metadata.insert(
            "sync_channel".to_string(),
            cn.block.cn_sync_type.to_string(),
        );
    }
    field.with_metadata(metadata)
}

/// create mdf3 channel field
fn mdf3_field(mdfinfo3: &Box<MdfInfo3>, cn: &Cn3) -> Field {
    let field = Field::new(
        cn.unique_name.clone(),
        cn.data.arrow_data_type().clone(),
        false,
    );
    let mut metadata = HashMap::<String, String>::new();
    if let Some(array) = mdfinfo3.sharable.cc.get(&cn.block1.cn_cc_conversion) {
        let txt = array.0.cc_unit;
        let encoding: &'static EncodingRs =
            to_encoding(mdfinfo3.id_block.id_codepage).unwrap_or(encoding_rs::WINDOWS_1252);
        let u: String = encoding.decode(&txt).0.into();
        let unit = u.trim_end_matches(char::from(0)).to_string();
        if !unit.is_empty() {
            metadata.insert("unit".to_string(), unit);
        }
    };
    if !cn.description.is_empty() {
        metadata.insert("description".to_string(), cn.description.clone());
    }
    field.with_metadata(metadata)
}

/// Write columns and fields in parquet file
fn write_data(
    master_channel_name: Option<String>,
    rec_id: &u64,
    file_name: &str,
    compression: Compression,
    fields: Schema,
    columns: Vec<Arc<dyn Array>>,
) -> Result<(), Error> {
    let record_batch = RecordBatch::try_new(Arc::new(fields.clone()), columns)
        .context("Failed creating recordbatch")?;
    let mut writer = create_parquet_writer(
        file_name,
        compression,
        fields,
        master_channel_name.clone(),
        rec_id,
    )
    .context("failed creating parquet writer")?;
    writer
        .write(&record_batch)
        .with_context(|| format!("Failed writing recordbatch"))?;
    writer.close().context("Failed to write footer")?;
    Ok(())
}
