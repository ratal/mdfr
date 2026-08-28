//! Exporting MDF data to CSV files using Arrow's ArrayFormatter (prettyprint feature).
use crate::{
    mdfinfo::{
        MdfInfo,
        mdfinfo3::{Cg3, Cn3, MdfInfo3},
        mdfinfo4::{Cg4, MdfInfo4},
    },
    mdfreader::Mdf,
};
use anyhow::{Context, Result};
use arrow::util::display::{ArrayFormatter, FormatOptions};
use rayon::prelude::*;
use std::{fs::File, io::BufWriter, io::Write, path::Path, sync::Arc};

/// Exports all loaded channel groups to CSV — one file per channel group.
pub fn export_to_csv(mdf: &Mdf, file_name: &str) -> Result<()> {
    match &mdf.mdf_info {
        MdfInfo::V4(mdfinfo4) => {
            for dg in mdfinfo4.dg.values() {
                if dg.cg.values().any(|cg| !cg.channel_names.is_empty()) {
                    // Collect CGs to export in parallel
                    let cg_refs: Vec<(&u64, &Cg4)> = dg.cg.iter().collect();
                    cg_refs.par_iter().try_for_each(|(rec_id, cg)| {
                        mdf4_cg_to_csv(file_name, mdfinfo4, **rec_id, cg)
                            .with_context(|| {
                                format!("failed converting Channel Group 4 rec_id {rec_id} to CSV")
                            })
                    })?;
                }
            }
        }
        MdfInfo::V3(mdfinfo3) => {
            for dg in mdfinfo3.dg.values() {
                let cg_refs: Vec<(&u16, &Cg3)> = dg.cg.iter().collect();
                cg_refs.par_iter().try_for_each(|(rec_id, cg)| {
                    mdf3_cg_to_csv(file_name, mdfinfo3, **rec_id, cg).with_context(|| {
                        format!("failed converting Channel Group 3 rec_id {rec_id} to CSV")
                    })
                })?;
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
                mdf4_cg_to_csv(file_name, mdfinfo4, *rec_id, cg)
                    .context("failed converting Channel Group 4 to CSV")?;
            }
        }
        MdfInfo::V3(mdfinfo3) => {
            if let Some((_master, dg_pos, (_cg_pos, rec_id), _cn_pos)) =
                mdfinfo3.get_channel_id(channel_name)
                && let Some(dg) = mdfinfo3.dg.get(dg_pos)
                && let Some(cg) = dg.cg.get(rec_id)
            {
                mdf3_cg_to_csv(file_name, mdfinfo3, *rec_id, cg)
                    .context("failed converting Channel Group 3 to CSV")?;
            }
        }
    }
    Ok(())
}

fn mdf4_cg_to_csv(file_name: &str, _mdfinfo4: &MdfInfo4, rec_id: u64, cg: &Cg4) -> Result<()> {
    let channels: Vec<(&str, Arc<dyn arrow::array::Array>)> = cg
        .cn
        .values()
        .filter(|cn| !cn.data.is_empty())
        .map(|cn| (cn.unique_name.as_str(), cn.data.finish_cloned()))
        .collect();
    if !channels.is_empty() {
        write_csv(
            file_name,
            cg.master_channel_name.clone(),
            rec_id,
            &channels,
        )?;
    }
    Ok(())
}

fn mdf3_cg_to_csv(file_name: &str, _mdfinfo3: &MdfInfo3, rec_id: u16, cg: &Cg3) -> Result<()> {
    let channels: Vec<(&str, Arc<dyn arrow::array::Array>)> = cg
        .cn
        .values()
        .filter(|cn: &&Cn3| !cn.data.is_empty())
        .map(|cn| (cn.unique_name.as_str(), cn.data.finish_cloned()))
        .collect();
    if !channels.is_empty() {
        write_csv(
            file_name,
            cg.master_channel_name.clone(),
            u64::from(rec_id),
            &channels,
        )?;
    }
    Ok(())
}

/// Formats channel data row-by-row and writes to a CSV file.
/// Uses Arrow's `ArrayFormatter` (enabled by the `prettyprint` feature).
fn write_csv(
    file_name: &str,
    master_channel: Option<String>,
    rec_id: u64,
    channels: &[(&str, Arc<dyn arrow::array::Array>)],
) -> Result<()> {
    let base_path = Path::new(file_name);
    let mut suffix = master_channel.unwrap_or_else(|| rec_id.to_string());
    suffix.insert(0, '_');
    let mut stem = base_path
        .file_stem()
        .context("no file stem in given file_name")?
        .to_os_string();
    stem.push(&suffix);
    let csv_path = base_path.with_file_name(stem).with_extension("csv");

    let file = File::create(&csv_path)
        .with_context(|| format!("failed creating CSV file {csv_path:?}"))?;
    let mut w = BufWriter::new(file);

    // Build formatters — skip columns whose type ArrayFormatter can't handle
    let opts = FormatOptions::default();
    let formatters: Vec<(&&str, ArrayFormatter)> = channels
        .iter()
        .filter_map(|(name, arr)| {
            ArrayFormatter::try_new(arr.as_ref(), &opts)
                .ok()
                .map(|f| (name, f))
        })
        .collect();

    if formatters.is_empty() {
        return Ok(());
    }

    // Header row
    let header: Vec<&str> = formatters.iter().map(|(n, _)| **n).collect();
    writeln!(w, "{}", header.join(",")).context("failed writing CSV header")?;

    let n_rows = channels[0].1.len();
    let mut row = String::with_capacity(256);
    for i in 0..n_rows {
        row.clear();
        for (j, (_, fmt)) in formatters.iter().enumerate() {
            if j > 0 {
                row.push(',');
            }
            fmt.value(i)
                .write(&mut row)
                .context("failed formatting CSV value")?;
        }
        writeln!(w, "{row}").context("failed writing CSV row")?;
    }
    Ok(())
}
