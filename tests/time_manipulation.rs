#[path = "common.rs"]
mod common;

use anyhow::Result;
use arrow::array::Array;
use arrow::datatypes::DataType;
use mdfr::data_holder::channel_data::ChannelData;
use mdfr::mdfreader::Mdf;
use std::collections::HashSet;
use std::sync::LazyLock;

static BASE_PATH_MDF4: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}MDF4/MDF4.3/Base_Standard/Examples/",
        common::mdfreader_tests_path()
    )
});

fn load_simple() -> Result<Mdf> {
    let file = format!(
        "{}Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4",
        BASE_PATH_MDF4.as_str()
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;
    Ok(mdf)
}

fn first_time_master(mdf: &Mdf) -> Option<String> {
    mdf.get_master_channel_names_set()
        .into_keys()
        .flatten()
        .find(|m| mdf.get_channel_master_type(m) == 1)
}

/// Returns the first Time master that has at least 2 distinct values (real spread).
fn first_time_master_with_spread(mdf: &Mdf) -> Option<String> {
    mdf.get_master_channel_names_set()
        .into_keys()
        .flatten()
        .filter(|m| mdf.get_channel_master_type(m) == 1)
        .find(|m| {
            if let Some(ChannelData::Float64(b)) = mdf.get_channel_data(m) {
                let s = b.values_slice();
                s.len() >= 2 && s.last() > s.first()
            } else {
                false
            }
        })
}

fn master_f64_values(mdf: &Mdf, master: &str) -> Vec<f64> {
    match mdf.get_channel_data(master) {
        Some(ChannelData::Float64(b)) => b.values_slice().to_vec(),
        _ => panic!("expected Float64 data for master channel '{master}'"),
    }
}

// ── cut ──────────────────────────────────────────────────────────────────────

#[test]
fn cut_trims_to_range() -> Result<()> {
    let mut mdf = load_simple()?;
    let master = match first_time_master_with_spread(&mdf) {
        Some(m) => m,
        None => {
            eprintln!("skip: no Time master with spread found in test file");
            return Ok(());
        }
    };

    let vals = master_f64_values(&mdf, &master);
    let orig_min = *vals.first().unwrap();
    let orig_max = *vals.last().unwrap();

    let start = orig_min + (orig_max - orig_min) * 0.25;
    let stop = orig_min + (orig_max - orig_min) * 0.75;
    mdf.cut(&master, start, stop)?;

    let cut_vals = master_f64_values(&mdf, &master);
    assert!(!cut_vals.is_empty(), "cut result must not be empty");
    assert!(
        *cut_vals.first().unwrap() >= start - 1e-9,
        "first value below start"
    );
    assert!(
        *cut_vals.last().unwrap() <= stop + 1e-9,
        "last value above stop"
    );

    // All channels in the same CG must match the master length
    let master_len = cut_vals.len();
    let group = mdf.mdf_info.get_channel_names_cg_set(&master);
    for name in &group {
        if let Some(ch) = mdf.get_channel_data(name) {
            assert_eq!(
                ch.len(),
                master_len,
                "channel {name} length mismatch after cut"
            );
        }
    }
    Ok(())
}

#[test]
fn cut_out_of_range_yields_empty() -> Result<()> {
    let mut mdf = load_simple()?;
    let master = match first_time_master_with_spread(&mdf) {
        Some(m) => m,
        None => {
            eprintln!("skip: no Time master with spread found in test file");
            return Ok(());
        }
    };

    let vals = master_f64_values(&mdf, &master);
    let orig_max = *vals.last().unwrap_or(&0.0);

    mdf.cut(&master, orig_max + 1.0, orig_max + 10.0)?;

    // After an empty cut the channel may return None or Some with len==0
    let len = mdf.get_channel_data(&master).map(|d| d.len()).unwrap_or(0);
    assert_eq!(len, 0, "expected zero-length result for out-of-range cut");
    Ok(())
}

// ── keep_channels ─────────────────────────────────────────────────────────────

#[test]
fn keep_channels_drops_others() -> Result<()> {
    let mut mdf = load_simple()?;
    let master = first_time_master(&mdf).expect("no Time master found");

    let group: Vec<String> = mdf
        .mdf_info
        .get_channel_names_cg_set(&master)
        .into_iter()
        .filter(|n| n != &master)
        .take(3)
        .collect();
    assert!(group.len() >= 3, "need at least 3 data channels in group");

    let (keep_a, keep_b, dropped) = (group[0].clone(), group[1].clone(), group[2].clone());

    mdf.keep_channels([keep_a.clone(), keep_b.clone()].into())?;

    assert!(
        mdf.get_channel_data(&keep_a)
            .map(|d| d.len() > 0)
            .unwrap_or(false),
        "kept channel A has no data"
    );
    assert!(
        mdf.get_channel_data(&keep_b)
            .map(|d| d.len() > 0)
            .unwrap_or(false),
        "kept channel B has no data"
    );
    let dropped_empty = mdf
        .get_channel_data(&dropped)
        .map(|d| d.len() == 0)
        .unwrap_or(true);
    assert!(dropped_empty, "dropped channel still has data");
    Ok(())
}

#[test]
fn keep_channels_preserves_master() -> Result<()> {
    let mut mdf = load_simple()?;
    let master = first_time_master(&mdf).expect("no Time master found");

    let one_ch: String = mdf
        .mdf_info
        .get_channel_names_cg_set(&master)
        .into_iter()
        .filter(|n| n != &master)
        .next()
        .expect("need at least one data channel");

    let master_len_before = mdf.get_channel_data(&master).map(|d| d.len()).unwrap_or(0);

    mdf.keep_channels([one_ch].into())?;

    let master_len_after = mdf.get_channel_data(&master).map(|d| d.len()).unwrap_or(0);
    assert_eq!(
        master_len_after, master_len_before,
        "master must be preserved when a channel in its group is kept"
    );
    Ok(())
}

// ── resample ──────────────────────────────────────────────────────────────────

#[test]
fn resample_group_uniform_spacing() -> Result<()> {
    let mut mdf = load_simple()?;
    let master = match first_time_master_with_spread(&mdf) {
        Some(m) => m,
        None => {
            eprintln!("skip: no Time master with spread found");
            return Ok(());
        }
    };

    let vals = master_f64_values(&mdf, &master);
    let (min_v, max_v) = (*vals.first().unwrap(), *vals.last().unwrap());
    let raster_s = (max_v - min_v) / 100.0; // ~100 points across the range
    assert!(raster_s > 0.0, "raster must be positive");

    mdf.resample_group(&master, raster_s)?;

    let resampled = master_f64_values(&mdf, &master);
    assert!(
        resampled.len() >= 2,
        "need at least 2 points after resample"
    );
    for w in resampled.windows(2) {
        let diff = w[1] - w[0];
        assert!(
            (diff - raster_s).abs() < raster_s * 1e-6 + 1e-12,
            "non-uniform spacing: diff={diff:.9}, raster={raster_s:.9}"
        );
    }
    Ok(())
}

#[test]
fn resample_all_time_groups_uniform() -> Result<()> {
    let mut mdf = load_simple()?;

    // Pick a raster based on the first Time master's range
    let time_masters: Vec<String> = mdf
        .get_master_channel_names_set()
        .into_keys()
        .flatten()
        .filter(|m| mdf.get_channel_master_type(m) == 1)
        .collect();
    assert!(!time_masters.is_empty(), "no Time masters found");

    let first = &time_masters[0];
    let vals = master_f64_values(&mdf, first);
    let raster_s = (vals.last().unwrap() - vals.first().unwrap()) / 100.0;
    assert!(raster_s > 0.0);

    mdf.resample(raster_s)?;

    for master in &time_masters {
        if let Some(ChannelData::Float64(b)) = mdf.get_channel_data(master) {
            let v = b.values_slice();
            if v.len() < 2 {
                continue;
            }
            for w in v.windows(2) {
                let diff = w[1] - w[0];
                assert!(
                    (diff - raster_s).abs() < raster_s * 1e-6 + 1e-12,
                    "master {master}: non-uniform spacing after resample"
                );
            }
        }
    }
    Ok(())
}

// ── concat_mdf ────────────────────────────────────────────────────────────────

#[test]
fn concat_extends_time_axis() -> Result<()> {
    let file = format!(
        "{}Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4",
        BASE_PATH_MDF4.as_str()
    );
    let mut mdf1 = Mdf::new(&file)?;
    mdf1.load_all_channels_data_in_memory()?;
    let master = first_time_master(&mdf1).expect("no Time master found");
    let len1 = mdf1.get_channel_data(&master).map(|d| d.len()).unwrap_or(0);

    let mut mdf2 = Mdf::new(&file)?;
    mdf2.load_all_channels_data_in_memory()?;
    let len2 = mdf2.get_channel_data(&master).map(|d| d.len()).unwrap_or(0);

    mdf1.concat_mdf(&mdf2)?;

    let total_len = mdf1.get_channel_data(&master).map(|d| d.len()).unwrap_or(0);
    assert_eq!(
        total_len,
        len1 + len2,
        "total length must equal sum of both"
    );

    // Master must be monotonically non-decreasing after concat
    let vals = master_f64_values(&mdf1, &master);
    for w in vals.windows(2) {
        assert!(
            w[1] >= w[0],
            "master not monotonically non-decreasing after concat"
        );
    }
    Ok(())
}

// ── merge ─────────────────────────────────────────────────────────────────────

#[test]
fn merge_adds_channels_from_other() -> Result<()> {
    let mut mdf1 = Mdf::new(&format!(
        "{}Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4",
        BASE_PATH_MDF4.as_str()
    ))?;
    mdf1.load_all_channels_data_in_memory()?;

    let mut mdf2 = Mdf::new(&format!("{}Simple/test.mf4", BASE_PATH_MDF4.as_str()))?;
    mdf2.load_all_channels_data_in_memory()?;

    let names_before: HashSet<String> = mdf1.get_channel_names_set();
    let unique_to_other: Vec<String> = mdf2
        .get_channel_names_set()
        .difference(&names_before)
        .take(3)
        .cloned()
        .collect();

    mdf1.merge(&mdf2)?;

    for ch in &unique_to_other {
        assert!(
            mdf1.get_channel_names_set().contains(ch),
            "channel '{ch}' from other not found in self after merge"
        );
    }
    Ok(())
}

// ── start_time ────────────────────────────────────────────────────────────────

#[test]
fn start_time_returns_u64() -> Result<()> {
    // ASAM reference test files often have epoch=0 (synthetic data); we verify
    // the call succeeds and returns a valid u64 without panicking.
    let mdf = load_simple()?;
    let _ts: u64 = mdf.get_start_time_ns();
    Ok(())
}

#[test]
fn start_time_nonzero_for_mdf4_file() -> Result<()> {
    // test_mdf4.mf4 is generated by mdfr's own write tests and carries a real timestamp.
    let mut mdf = Mdf::new(&format!("{}test_mdf4.mf4", common::TEST_FILES))?;
    mdf.load_all_channels_data_in_memory()?;
    let ts = mdf.get_start_time_ns();
    assert!(
        ts > 0,
        "expected nonzero start_time_ns in test_mdf4.mf4, got {ts}"
    );
    Ok(())
}

// ── get_master_channel_datetimes ──────────────────────────────────────────────

#[test]
fn datetimes_length_and_type() -> Result<()> {
    let mdf = load_simple()?;
    let master = first_time_master(&mdf).expect("no Time master found");

    let data_ch: String = mdf
        .mdf_info
        .get_channel_names_cg_set(&master)
        .into_iter()
        .filter(|n| n != &master)
        .next()
        .expect("need a data channel");

    let master_len = mdf.get_channel_data(&master).map(|d| d.len()).unwrap_or(0);

    let arr = mdf
        .get_master_channel_datetimes(&data_ch)
        .expect("expected Some(arr) for channel with Time master");

    assert_eq!(arr.len(), master_len, "datetime array length mismatch");
    assert!(
        matches!(arr.data_type(), DataType::Timestamp(_, Some(_))),
        "expected Timestamp with timezone, got {:?}",
        arr.data_type()
    );
    Ok(())
}

#[test]
fn datetimes_none_for_non_time_master() -> Result<()> {
    let mdf = load_simple()?;
    // Find a channel whose master has type != 1 (or no master)
    let candidate = mdf
        .get_channel_names_set()
        .into_iter()
        .find(|n| mdf.get_channel_master_type(n) != 1);
    if let Some(ch) = candidate {
        assert!(
            mdf.get_master_channel_datetimes(&ch).is_none(),
            "expected None for non-Time master channel"
        );
    }
    Ok(())
}
