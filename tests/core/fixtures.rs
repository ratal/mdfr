/// Synthetic MDF4 fixture verification tests.
///
/// Shared fixtures under `test_files/synthetic/` are initialized atomically via
/// `crate::common::synthetic` so tests run concurrently without `FIXTURE_LOCK`
/// or cross-binary file races.
use crate::common::synthetic;
use mdfr::data_holder::channel_data::{AsValuesSlice, ChannelData};
use mdfr::mdfreader::Mdf;

#[test]
fn create_fixtures() {
    type FixtureCase = (&'static str, fn() -> anyhow::Result<Vec<u8>>);
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let cases: [FixtureCase; 5] = [
        ("int_linear_cc.mf4", synthetic::build_int_linear_cc),
        ("unsorted_multi_cg.mf4", synthetic::build_unsorted_multi_cg),
        (
            "unsorted_vlsd_vlsc.mf4",
            synthetic::build_unsorted_vlsd_vlsc,
        ),
        ("synthetic_array.mf4", synthetic::build_synthetic_array),
        (
            "synthetic_dz_deflate.mf4",
            synthetic::build_synthetic_dz_deflate,
        ),
    ];
    for (name, build) in cases {
        let bytes = build().unwrap_or_else(|e| panic!("failed building {name}: {e}"));
        let out = tmp.path().join(name);
        std::fs::write(&out, &bytes).unwrap_or_else(|e| panic!("failed writing {name}: {e}"));
        assert!(out.exists(), "{name} fixture file not created");
    }

    assert!(std::path::Path::new(synthetic::ensure_int_linear_cc()).exists());
    assert!(std::path::Path::new(synthetic::ensure_unsorted_multi_cg()).exists());
    assert!(std::path::Path::new(synthetic::ensure_unsorted_vlsd_vlsc()).exists());
    assert!(std::path::Path::new(synthetic::ensure_synthetic_array()).exists());
    assert!(std::path::Path::new(synthetic::ensure_synthetic_dz_deflate()).exists());
}

#[test]
fn verify_synthetic_array_fixture_loads() {
    let path = synthetic::ensure_synthetic_array();
    let mut mdf = Mdf::new(path).expect("failed to open synthetic_array fixture");
    mdf.load_all_channels_data_in_memory()
        .expect("failed to load synthetic array data");

    let names = mdf.get_channel_names_set();
    assert!(names.contains("array2d"), "missing array2d channel");
    if let Some(ChannelData::ArrayDFloat64(arr)) = mdf.get_channel_data("array2d") {
        let vals = arr.values_slice();
        assert_eq!(
            vals.len(),
            12,
            "should have 12 total values (3 cycles * 4 elements)"
        );
        assert_eq!(
            arr.shape().as_slice(),
            &[2usize, 2],
            "shape should be [2, 2]"
        );
        assert_eq!(arr.len(), 3, "len should be 3");
        assert_eq!(arr.ndim(), 2, "ndim should be 2");
        for (i, &v) in vals.iter().enumerate() {
            assert_eq!(v, (i + 1) as f64);
        }
    } else {
        panic!("array2d channel is not ArrayDFloat64");
    }
}

#[test]
fn verify_synthetic_dz_fixture_loads() {
    let path = synthetic::ensure_synthetic_dz_deflate();
    let mut mdf = Mdf::new(path).expect("failed to open synthetic_dz_deflate fixture");
    mdf.load_all_channels_data_in_memory()
        .expect("failed to load synthetic dz data");

    let names = mdf.get_channel_names_set();
    assert!(names.contains("time_ch"), "missing time_ch channel");
    if let Some(ChannelData::Float64(arr)) = mdf.get_channel_data("time_ch") {
        let vals = arr.values_slice();
        assert_eq!(vals.len(), 10);
        for (i, &v) in vals.iter().enumerate() {
            assert_eq!(v, i as f64);
        }
    } else {
        panic!("time_ch channel is not Float64");
    }
}

#[test]
fn verify_unsorted_vlsd_vlsc_fixture_loads() {
    let path = synthetic::ensure_unsorted_vlsd_vlsc();
    let mut mdf = Mdf::new(path).expect("failed to open unsorted VLSD/VLSC fixture");
    mdf.load_all_channels_data_in_memory()
        .expect("failed to load unsorted VLSD/VLSC data");

    let names = mdf.get_channel_names_set();
    assert!(names.contains("vlsd_target_ch"), "missing vlsd_target_ch");
}

#[test]
fn verify_unsorted_fixture_loads() {
    let path = synthetic::ensure_unsorted_multi_cg();
    let mut mdf = Mdf::new(path).expect("failed to open unsorted fixture");
    mdf.load_all_channels_data_in_memory()
        .expect("failed to load unsorted data");

    let names = mdf.get_channel_names_set();
    let expected = [
        "cg1_ch1", "cg2_ch1", "cg2_ch2", "cg3_ch1", "cg3_ch2", "cg3_ch3", "cg4_ch1", "cg4_ch2",
        "cg4_ch3",
    ];
    for name in &expected {
        assert!(names.contains(*name), "missing {name}");
    }
    assert!(
        names.len() >= 9,
        "expected at least 9 channels, got {}",
        names.len()
    );
}
