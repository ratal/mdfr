//! Exporting mdf to Parquet files.
use anyhow::{Context, Error, Result};
use arrow::{
    array::{Array, RecordBatch},
    datatypes::{Field, SchemaBuilder},
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

use crate::{mdfinfo::MdfInfo, mdfreader::Mdf};

use std::{cmp::max, path::Path};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

/// writes mdf into parquet file
pub fn export_to_parquet(
    mdf: &Mdf,
    file_name: &str,
    compression: Option<&str>,
) -> Result<(), Error> {
    // Create file
    let path = Path::new(file_name);

    let (arrow_data, mut arrow_schema, max_row_group_size) =
        mdf_data_to_arrow(mdf).context("failed creating arrow schema and recordbatches")?;

    let options = WriterProperties::builder()
        .set_compression(parquet_compression_from_string(compression))
        .set_max_row_group_size(max_row_group_size)
        .set_writer_version(WriterVersion::PARQUET_1_0)
        .set_encoding(Encoding::PLAIN)
        .set_key_value_metadata(Some(vec![KeyValue::new(
            "file_name".to_string(),
            file_name.to_string(),
        )]))
        .build();

    arrow_schema
        .metadata_mut()
        .insert("file_name".to_string(), file_name.to_string());

    let file =
        std::io::BufWriter::new(std::fs::File::create(path).context("Failed to create file")?);
    let finalised_arrow_schema = arrow_schema.finish();
    let mut writer = ArrowWriter::try_new(
        file,
        Arc::new(finalised_arrow_schema.clone()),
        Some(options.clone()),
    )
    .with_context(|| {
        format!(
            "Failed to write parquet file with schema {:?} and options {:?}",
            finalised_arrow_schema, options
        )
    })?;

    // write data in file
    for group in arrow_data {
        writer
            .write(&group)
            .with_context(|| format!("Failed wirting recordbatch {:?}", group))?;
    }
    writer.close().context("Failed to write footer")?;
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

/// takes data of channel set from MdfInfo structure and stores in mdf.arrow_data
fn mdf_data_to_arrow(mdf: &Mdf) -> Result<(Vec<RecordBatch>, SchemaBuilder, usize), Error> {
    match &mdf.mdf_info {
        MdfInfo::V4(mdfinfo4) => {
            let mut arrow_schema = SchemaBuilder::with_capacity(mdfinfo4.channel_names_set.len());
            let mut arrow_data: Vec<RecordBatch> = Vec::with_capacity(mdfinfo4.dg.len());
            let mut max_row_group_size: usize = 0;
            for (_dg_block_position, dg) in mdfinfo4.dg.iter() {
                let mut channel_names_present_in_dg = HashSet::new();
                for channel_group in dg.cg.values() {
                    let cn = channel_group.channel_names.clone();
                    channel_names_present_in_dg.par_extend(cn);
                    max_row_group_size = max(
                        max_row_group_size,
                        channel_group.block.cg_cycle_count as usize,
                    );
                }
                if !channel_names_present_in_dg.is_empty() {
                    dg.cg.iter().for_each(|(_rec_id, cg)| {
                        let mut columns =
                            Vec::<Arc<dyn Array>>::with_capacity(cg.channel_names.len());
                        let mut fields = SchemaBuilder::with_capacity(cg.channel_names.len());
                        cg.cn.iter().for_each(|(_rec_pos, cn)| {
                            if !cn.data.is_empty() {
                                let mut field = Field::new(
                                    cn.unique_name.clone(),
                                    cn.data.arrow_data_type().clone(),
                                    cn.data.validity().is_some(),
                                );
                                let mut metadata = HashMap::<String, String>::new();
                                if let Ok(Some(unit)) =
                                    mdfinfo4.sharable.get_tx(cn.block.cn_md_unit)
                                {
                                    if !unit.is_empty() {
                                        metadata.insert("unit".to_string(), unit);
                                    }
                                };
                                if let Ok(Some(desc)) =
                                    mdfinfo4.sharable.get_tx(cn.block.cn_md_comment)
                                {
                                    if !desc.is_empty() {
                                        metadata.insert("description".to_string(), desc);
                                    }
                                };
                                if let Some((
                                    Some(master_channel_name),
                                    _dg_pos,
                                    (_cg_pos, _rec_idd),
                                    (_cn_pos, _rec_pos),
                                )) = mdfinfo4.channel_names_set.get(&cn.unique_name)
                                {
                                    if !master_channel_name.is_empty() {
                                        metadata.insert(
                                            "master_channel".to_string(),
                                            master_channel_name.to_string(),
                                        );
                                    }
                                }
                                if cn.block.cn_type == 4 {
                                    metadata.insert(
                                        "sync_channel".to_string(),
                                        cn.block.cn_sync_type.to_string(),
                                    );
                                }
                                field = field.with_metadata(metadata);
                                arrow_schema.push(field.clone());
                                fields.push(field);
                                columns.push(cn.data.finish_cloned());
                            }
                        });
                        if !columns.is_empty() {
                            arrow_data.push(
                                RecordBatch::try_new(Arc::new(fields.finish()), columns)
                                    .expect("Failed creating recordbatch"),
                            );
                        }
                    });
                }
            }
            Ok((arrow_data, arrow_schema, max_row_group_size))
        }
        MdfInfo::V3(mdfinfo3) => {
            let mut arrow_schema = SchemaBuilder::with_capacity(mdfinfo3.channel_names_set.len());
            let mut arrow_data: Vec<RecordBatch> = Vec::with_capacity(mdfinfo3.dg.len());
            let mut max_row_group_size: usize = 0;
            for (_dg_block_position, dg) in mdfinfo3.dg.iter() {
                for (_rec_id, cg) in dg.cg.iter() {
                    let mut columns = Vec::<Arc<dyn Array>>::with_capacity(cg.channel_names.len());
                    let mut fields = SchemaBuilder::with_capacity(cg.channel_names.len());
                    max_row_group_size = max(max_row_group_size, cg.block.cg_cycle_count as usize);
                    for (_rec_pos, cn) in cg.cn.iter() {
                        if !cn.data.is_empty() {
                            let mut field = Field::new(
                                cn.unique_name.clone(),
                                cn.data.arrow_data_type().clone(),
                                false,
                            );
                            columns.push(cn.data.finish_cloned());
                            let mut metadata = HashMap::<String, String>::new();
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
                            field = field.with_metadata(metadata);
                            arrow_schema.push(field.clone());
                            fields.push(field);
                        }
                    }
                    if !columns.is_empty() {
                        arrow_data.push(
                            RecordBatch::try_new(Arc::new(fields.finish()), columns)
                                .expect("Failed creating recordbatch"),
                        );
                    }
                }
            }
            Ok((arrow_data, arrow_schema, max_row_group_size))
        }
    }
}
