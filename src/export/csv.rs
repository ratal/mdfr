//! Exporting MDF data to CSV files using Arrow's ArrayFormatter (prettyprint feature).
use crate::{
    mdfinfo::{
        MdfInfo,
        mdfinfo3::{Cg3, Cn3, MdfInfo3},
        mdfinfo4::{Cg4, Cn4, MdfInfo4},
    },
    mdfreader::{
        Mdf, conversions3::convert_channel_data_cow_3, conversions4::convert_channel_data_cow,
    },
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
                if dg.cg.values().any(|cg| !cg.cn.is_empty()) {
                    // Collect CGs to export in parallel
                    let cg_refs: Vec<(&u64, &Cg4)> = dg.cg.iter().collect();
                    cg_refs.par_iter().try_for_each(|(rec_id, cg)| {
                        mdf4_cg_to_csv(file_name, mdfinfo4, cg).with_context(|| {
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
                    mdf3_cg_to_csv(file_name, mdfinfo3, cg).with_context(|| {
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
                mdf4_cg_to_csv(file_name, mdfinfo4, cg)
                    .context("failed converting Channel Group 4 to CSV")?;
            }
        }
        MdfInfo::V3(mdfinfo3) => {
            if let Some((_master, dg_pos, (_cg_pos, rec_id), _cn_pos)) =
                mdfinfo3.get_channel_id(channel_name)
                && let Some(dg) = mdfinfo3.dg.get(dg_pos)
                && let Some(cg) = dg.cg.get(rec_id)
            {
                mdf3_cg_to_csv(file_name, mdfinfo3, cg)
                    .context("failed converting Channel Group 3 to CSV")?;
            }
        }
    }
    Ok(())
}

fn mdf4_cg_to_csv(file_name: &str, mdfinfo4: &MdfInfo4, cg: &Cg4) -> Result<()> {
    let mut sorted_cns: Vec<(&i64, &Cn4)> = cg.cn.iter().collect();
    sorted_cns.sort_unstable_by_key(|&(&rec_pos, _)| rec_pos);
    let mut channels = Vec::with_capacity(sorted_cns.len());
    for (_, cn) in sorted_cns {
        if !cn.data.is_empty() {
            let converted = convert_channel_data_cow(cn, &mdfinfo4.sharable)?;
            channels.push((cn.unique_name.as_ref(), converted.finish_cloned()));
        }
    }
    if !channels.is_empty() {
        let group_suffix = cg
            .master_channel_name
            .as_deref()
            .map(str::to_string)
            .unwrap_or_else(|| format!("no_master_{}", cg.block_position));
        write_csv(file_name, &group_suffix, &channels)?;
    }
    Ok(())
}

fn mdf3_cg_to_csv(file_name: &str, mdfinfo3: &MdfInfo3, cg: &Cg3) -> Result<()> {
    let mut sorted_cns: Vec<(&u32, &Cn3)> = cg.cn.iter().collect();
    sorted_cns.sort_unstable_by_key(|&(&cn_pos, _)| cn_pos);
    let mut channels = Vec::with_capacity(sorted_cns.len());
    for (_, cn) in sorted_cns {
        if !cn.data.is_empty() {
            let converted = convert_channel_data_cow_3(cn, &mdfinfo3.sharable)?;
            channels.push((cn.unique_name.as_ref(), converted.finish_cloned()));
        }
    }
    if !channels.is_empty() {
        let group_suffix = cg
            .master_channel_name
            .as_deref()
            .map(str::to_string)
            .unwrap_or_else(|| format!("no_master_{}", cg.block_position));
        write_csv(file_name, &group_suffix, &channels)?;
    }
    Ok(())
}

/// Formats channel data row-by-row and writes to a CSV file.
/// Uses Arrow's `ArrayFormatter` (enabled by the `prettyprint` feature).
fn write_csv(
    file_name: &str,
    group_suffix: &str,
    channels: &[(&str, Arc<dyn arrow::array::Array>)],
) -> Result<()> {
    let base_path = Path::new(file_name);
    let stem = if base_path
        .extension()
        .and_then(|s| s.to_str())
        .map(|e| e.eq_ignore_ascii_case("csv"))
        .unwrap_or(false)
    {
        base_path
            .file_stem()
            .context("no file stem in given file_name")?
    } else {
        base_path
            .file_name()
            .context("no file name in given file_name")?
    }
    .to_string_lossy();

    let out_file_name = format!("{stem}_{group_suffix}.csv");
    let csv_path = base_path.with_file_name(out_file_name);

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
