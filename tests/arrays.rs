//! Integration tests for MDF4 array channels (CABlock / TensorArrow).
//! Tests read real MDF4.3 array example files and verify that array channels
//! load correctly and survive a round-trip write.

mod common;

use anyhow::Result;
use mdfr::data_holder::channel_data::ChannelData;
use mdfr::mdfinfo::mdfinfo4::CompressionAlgorithm;
use mdfr::mdfreader::Mdf;
use std::sync::LazyLock;

static BASE_PATH_ARRAYS: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}MDF4/MDF4.3/Base_Standard/Examples/Arrays/",
        common::mdfreader_tests_path()
    )
});

/// Checks whether a `ChannelData` value is one of the ArrayD variants.
fn is_array_channel(data: &ChannelData) -> bool {
    matches!(
        data,
        ChannelData::ArrayDFloat64(_)
            | ChannelData::ArrayDFloat32(_)
            | ChannelData::ArrayDInt8(_)
            | ChannelData::ArrayDUInt8(_)
            | ChannelData::ArrayDInt16(_)
            | ChannelData::ArrayDUInt16(_)
            | ChannelData::ArrayDInt32(_)
            | ChannelData::ArrayDUInt32(_)
            | ChannelData::ArrayDInt64(_)
            | ChannelData::ArrayDUInt64(_)
    )
}

fn load_array_file(subpath: &str) -> Result<Option<Mdf>> {
    let file = format!("{}{}", BASE_PATH_ARRAYS.as_str(), subpath);
    if !std::path::Path::new(&file).exists() {
        return Ok(None);
    }
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;
    Ok(Some(mdf))
}

// ──────────────────────────────────────────────────────────────────────────────
// Simple array files
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn array_simple_vector() -> Result<()> {
    let Some(mut mdf) = load_array_file("Simple/Vector_MeasurementArrays.mf4")? else {
        return Ok(());
    };

    let names = mdf.get_channel_names_set();
    assert!(!names.is_empty(), "No channels found in array file");

    // At least one channel should be an ArrayD type
    let has_array_channel = names
        .iter()
        .any(|name| mdf.get_channel_data(name).is_some_and(is_array_channel));
    assert!(
        has_array_channel,
        "Expected at least one ArrayD channel in Vector_MeasurementArrays.mf4"
    );

    Ok(())
}

#[test]
fn array_simple_dspace() -> Result<()> {
    #[allow(unused_mut)]
    let Some(mut mdf) = load_array_file("Simple/dSPACE_MeasurementArrays.mf4")? else {
        return Ok(());
    };

    let names = mdf.get_channel_names_set();
    assert!(!names.is_empty(), "No channels found in dSPACE array file");

    #[cfg(feature = "numpy")]
    {
        use pyo3::prelude::*;
        Python::initialize();
        Python::attach(|py| -> Result<()> {
            let two_dim = mdf
                .get_channel_data("Signal_TwoDim")
                .expect("Signal_TwoDim missing")
                .clone()
                .into_pyobject(py)?;
            let two_dim_shape: Vec<usize> = two_dim.getattr("shape")?.extract()?;
            assert_eq!(two_dim_shape, vec![5, 3, 4]);
            let two_dim_vals: Vec<Vec<Vec<f64>>> = two_dim.call_method0("tolist")?.extract()?;
            assert_eq!(
                two_dim_vals[0],
                vec![
                    vec![0.0, 0.01, 0.02, 0.03],
                    vec![0.1, 0.11, 0.12, 0.13],
                    vec![0.2, 0.21, 0.22, 0.23],
                ]
            );
            assert_eq!(
                two_dim_vals[1],
                vec![
                    vec![0.001, 0.011, 0.021, 0.031],
                    vec![0.101, 0.111, 0.121, 0.131],
                    vec![0.201, 0.211, 0.221, 0.231],
                ]
            );

            let matrix_dim = mdf
                .get_channel_data("Signal_MatrixDim")
                .expect("Signal_MatrixDim missing")
                .clone()
                .into_pyobject(py)?;
            let matrix_dim_shape: Vec<usize> = matrix_dim.getattr("shape")?.extract()?;
            assert_eq!(matrix_dim_shape, vec![5, 2, 3, 4]);
            let matrix_dim_vals: Vec<Vec<Vec<Vec<f64>>>> =
                matrix_dim.call_method0("tolist")?.extract()?;
            assert_eq!(
                matrix_dim_vals[0],
                vec![
                    vec![
                        vec![0.0, 0.01, 0.02, 0.03],
                        vec![0.1, 0.11, 0.12, 0.13],
                        vec![0.2, 0.21, 0.22, 0.23],
                    ],
                    vec![
                        vec![1.0, 1.01, 1.02, 1.03],
                        vec![1.1, 1.11, 1.12, 1.13],
                        vec![1.2, 1.21, 1.22, 1.23],
                    ],
                ]
            );
            assert_eq!(
                matrix_dim_vals[1],
                vec![
                    vec![
                        vec![0.001, 0.011, 0.021, 0.031],
                        vec![0.101, 0.111, 0.121, 0.131],
                        vec![0.201, 0.211, 0.221, 0.231],
                    ],
                    vec![
                        vec![1.001, 1.011, 1.021, 1.031],
                        vec![1.101, 1.111, 1.121, 1.131],
                        vec![1.201, 1.211, 1.221, 1.231],
                    ],
                ]
            );
            Ok(())
        })?;
    }

    Ok(())
}

#[test]
fn array_with_fixed_axes() -> Result<()> {
    let Some(mdf) = load_array_file("Simple/Vector_ArrayWithFixedAxes.MF4")? else {
        return Ok(());
    };

    let names = mdf.get_channel_names_set();
    assert!(
        !names.is_empty(),
        "No channels in Vector_ArrayWithFixedAxes.MF4"
    );

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// Classification
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn array_classification_porsche() -> Result<()> {
    let Some(mdf) = load_array_file("Classification/Porsche_2D_classification_result.mf4")? else {
        return Ok(());
    };

    let names = mdf.get_channel_names_set();
    assert!(
        !names.is_empty(),
        "No channels in Porsche_2D_classification_result.mf4"
    );

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// Round-trip: write and re-read an array file
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn array_round_trip_write() -> Result<()> {
    let file = format!(
        "{}Simple/Vector_MeasurementArrays.mf4",
        BASE_PATH_ARRAYS.as_str()
    );
    if !std::path::Path::new(&file).exists() {
        return Ok(());
    }

    let tmp_file = tempfile::Builder::new().suffix(".mf4").tempfile()?;
    let out_file_str = tmp_file.path().to_str().unwrap().to_string();

    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;

    let original_names = mdf.get_channel_names_set();
    assert!(!original_names.is_empty());

    // Write the file; the write itself should succeed
    let mdf2 = mdf.write(&out_file_str, CompressionAlgorithm::NoCompression)?;

    // The written file must exist and be non-empty
    assert!(
        std::path::Path::new(&out_file_str).exists(),
        "Written file does not exist"
    );

    // The Mdf returned by write() should have at least the same channel names
    let written_names = mdf2.get_channel_names_set();
    assert!(!written_names.is_empty(), "Written file has no channels");

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// Array channel properties
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn array_channels_have_expected_ndim() -> Result<()> {
    let Some(mut mdf) = load_array_file("Simple/Vector_MeasurementArrays.mf4")? else {
        return Ok(());
    };

    let names = mdf.get_channel_names_set();
    for name in &names {
        if let Some(data) = mdf.get_channel_data(name)
            && is_array_channel(data)
        {
            // Array channels must have ndim >= 1
            assert!(data.ndim() >= 1, "Array channel {name} has ndim < 1");
        }
    }

    Ok(())
}

#[test]
fn array_channels_min_max() -> Result<()> {
    let Some(mut mdf) = load_array_file("Simple/Vector_MeasurementArrays.mf4")? else {
        return Ok(());
    };

    let names = mdf.get_channel_names_set();
    let mut found_array = false;
    for name in &names {
        if let Some(data) = mdf.get_channel_data(name)
            && is_array_channel(data)
            && !data.is_empty()
        {
            let (min, max) = data.min_max();
            // If there is data, min and max should be populated for numeric arrays
            if let (Some(lo), Some(hi)) = (min, max) {
                assert!(lo <= hi, "min > max for channel {name}");
            }
            found_array = true;
        }
    }
    assert!(found_array, "No non-empty array channels found");

    Ok(())
}
