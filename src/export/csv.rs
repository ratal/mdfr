//! Exporting MDF data to CSV files using Arrow's CSV writer.
use crate::{
    mdfinfo::{
        MdfInfo,
        mdfinfo3::{Cg3, Cn3, MdfInfo3},
        mdfinfo4::{Cg4, Cn4, MdfInfo4},
    },
    mdfreader::Mdf,
};
use anyhow::{Context, Result};
use arrow::{
    array::{Array, RecordBatch},
    datatypes::{Field, SchemaBuilder},
};
use std::{fs::File, io::BufWriter, path::Path, sync::Arc};

/// Exports all loaded channel groups to CSV — one file per channel group.
pub fn export_to_csv(mdf: &Mdf, file_name: &str) -> Result<()> {
    match &mdf.mdf_info {
        MdfInfo::V4(mdfinfo4) => {
            for dg in mdfinfo4.dg.values() {
                if dg.cg.values().any(|cg| !cg.channel_names.is_empty()) {
                    for (rec_id, cg) in &dg.cg {
                        mdf4_cg_to_csv(file_name, mdfinfo4, rec_id, cg)
                            .context("failed converting Channel Group 4 to CSV")?;
                    }
                }
            }
        }
        MdfInfo::V3(mdfinfo3) => {
            for dg in mdfinfo3.dg.values() {
                for (rec_id, cg) in &dg.cg {
                    mdf3_cg_to_csv(file_name, mdfinfo3, rec_id, cg)
                        .context("failed converting Channel Group 3 to CSV")?;
                }
            }
        }
    }
    Ok(())
}

/// Exports the channel group that contains the named channel to a CSV file.
pub fn export_dataframe_to_csv(mdf: &Mdf, channel_name: &str, file_name: &str) -> Result<()> {
    match &mdf.mdf_info {
        MdfInfo::V4(mdfinfo4) => {
            if let Some((_master, dg_pos, (_cg_pos, rec_id), (_cn_pos, _rec_pos))) =
                mdfinfo4.get_channel_id(channel_name)
                && let Some(dg) = mdfinfo4.dg.get(dg_pos)
                && let Some(cg) = dg.cg.get(rec_id)
            {
                mdf4_cg_to_csv(file_name, mdfinfo4, rec_id, cg)
                    .context("failed converting Channel Group 4 to CSV")?;
            }
        }
        MdfInfo::V3(mdfinfo3) => {
            if let Some((_master, dg_pos, (_cg_pos, rec_id), _cn_pos)) =
                mdfinfo3.get_channel_id(channel_name)
                && let Some(dg) = mdfinfo3.dg.get(dg_pos)
                && let Some(cg) = dg.cg.get(rec_id)
            {
                mdf3_cg_to_csv(file_name, mdfinfo3, rec_id, cg)
                    .context("failed converting Channel Group 3 to CSV")?;
            }
        }
    }
    Ok(())
}

fn mdf4_cg_to_csv(file_name: &str, _mdfinfo4: &MdfInfo4, rec_id: &u64, cg: &Cg4) -> Result<()> {
    let mut columns = Vec::<Arc<dyn Array>>::with_capacity(cg.channel_names.len());
    let mut fields = SchemaBuilder::with_capacity(cg.channel_names.len());
    for (_rec_pos, cn) in cg.cn.iter() {
        if !cn.data.is_empty() && is_csv_writable(cn) {
            fields.push(Field::new(
                cn.unique_name.clone(),
                cn.data.arrow_data_type().clone(),
                cn.data.validity().is_some(),
            ));
            columns.push(cn.data.finish_cloned());
        }
    }
    if !columns.is_empty() {
        let schema = Arc::new(fields.finish());
        let batch =
            RecordBatch::try_new(schema, columns).context("failed building RecordBatch for CSV")?;
        write_csv(file_name, cg.master_channel_name.clone(), *rec_id, &batch)?;
    }
    Ok(())
}

fn mdf3_cg_to_csv(file_name: &str, _mdfinfo3: &MdfInfo3, rec_id: &u16, cg: &Cg3) -> Result<()> {
    let mut columns = Vec::<Arc<dyn Array>>::with_capacity(cg.channel_names.len());
    let mut fields = SchemaBuilder::with_capacity(cg.channel_names.len());
    for (_rec_pos, cn) in cg.cn.iter() {
        if !cn.data.is_empty() && is_csv_writable_cn3(cn) {
            fields.push(Field::new(
                cn.unique_name.clone(),
                cn.data.arrow_data_type().clone(),
                false,
            ));
            columns.push(cn.data.finish_cloned());
        }
    }
    if !columns.is_empty() {
        let schema = Arc::new(fields.finish());
        let batch =
            RecordBatch::try_new(schema, columns).context("failed building RecordBatch for CSV")?;
        write_csv(
            file_name,
            cg.master_channel_name.clone(),
            u64::from(*rec_id),
            &batch,
        )?;
    }
    Ok(())
}

/// Returns true if this MDF4 channel's data type can be represented in a CSV column.
/// Arrow's CSV writer handles scalars, strings, and timestamps but not FixedSizeList
/// (Complex / TensorArrow) or Union (variant) types.
fn is_csv_writable(cn: &Cn4) -> bool {
    use arrow::datatypes::DataType;
    !matches!(
        cn.data.arrow_data_type(),
        DataType::FixedSizeList(_, _) | DataType::Union(_, _)
    )
}

fn is_csv_writable_cn3(cn: &Cn3) -> bool {
    use arrow::datatypes::DataType;
    !matches!(
        cn.data.arrow_data_type(),
        DataType::FixedSizeList(_, _) | DataType::Union(_, _)
    )
}

/// Builds the output CSV file path and writes the RecordBatch.
/// Mirrors the parquet naming convention: base_<master_or_recid>.csv
fn write_csv(
    file_name: &str,
    master_channel: Option<String>,
    rec_id: u64,
    batch: &RecordBatch,
) -> Result<()> {
    let base_path = Path::new(file_name);
    let mut suffix = master_channel.unwrap_or_else(|| rec_id.to_string());
    suffix.insert(0, '_');
    let mut stem = base_path
        .file_stem()
        .context("no file stem in given path")?
        .to_os_string();
    stem.push(&suffix);
    let csv_path = base_path.with_file_name(stem).with_extension("csv");
    let file = File::create(&csv_path)
        .with_context(|| format!("failed creating CSV file {csv_path:?}"))?;
    let mut writer = arrow::csv::WriterBuilder::new()
        .with_header(true)
        .build(BufWriter::new(file));
    writer
        .write(batch)
        .context("failed writing RecordBatch to CSV")?;
    Ok(())
}
