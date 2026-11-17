/// Integration tests exercising Int8/Int16/Float32 conversion arms in conversions4.rs.
///
/// The existing sample files all use Float64 as raw channel type, so those arms
/// were never hit. These tests load a synthetic MDF4 fixture that has Int8/Int16/Float32
/// channels with a linear CC block.
use crate::common::synthetic;
use anyhow::Result;
use mdfr::data_holder::channel_data::{AsValuesSlice, ChannelData};
use mdfr::mdfreader::Mdf;

fn load_fixture() -> Result<Mdf> {
    let path = synthetic::ensure_int_linear_cc();
    let mut mdf = Mdf::new(path)?;
    mdf.load_all_channels_data_in_memory()?;
    Ok(mdf)
}

#[test]
fn int8_linear_conversion() -> Result<()> {
    let mut mdf = load_fixture()?;
    // raw: -5, 0, 5, 10 → phys = raw * 2.0 + 0.5 → -9.5, 0.5, 10.5, 20.5
    let data = mdf.get_channel_data("int8_ch").expect("int8_ch not found");
    assert!(
        matches!(data, ChannelData::Float64(_)),
        "expected Float64 after linear CC, got {}",
        data.data_type(false)
    );
    if let ChannelData::Float64(arr) = data {
        let expected = [-9.5f64, 0.5, 10.5, 20.5];
        for (i, (&got, &exp)) in arr.values_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-9,
                "int8_ch[{i}]: expected {exp}, got {got}"
            );
        }
    }
    Ok(())
}

#[test]
fn int16_linear_conversion() -> Result<()> {
    let mut mdf = load_fixture()?;
    // raw: -100, 0, 100, 200 → -199.5, 0.5, 200.5, 400.5
    let data = mdf
        .get_channel_data("int16_ch")
        .expect("int16_ch not found");
    assert!(
        matches!(data, ChannelData::Float64(_)),
        "expected Float64 after linear CC"
    );
    if let ChannelData::Float64(arr) = data {
        let expected = [-199.5f64, 0.5, 200.5, 400.5];
        for (i, (&got, &exp)) in arr.values_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-9,
                "int16_ch[{i}]: expected {exp}, got {got}"
            );
        }
    }
    Ok(())
}

#[test]
fn float32_linear_conversion() -> Result<()> {
    let mut mdf = load_fixture()?;
    // raw: 1.5, 2.5, 3.5, 4.5 → 3.5, 5.5, 7.5, 9.5
    let data = mdf
        .get_channel_data("float32_ch")
        .expect("float32_ch not found");
    assert!(
        matches!(data, ChannelData::Float64(_)),
        "expected Float64 after linear CC"
    );
    if let ChannelData::Float64(arr) = data {
        let expected = [3.5f64, 5.5, 7.5, 9.5];
        for (i, (&got, &exp)) in arr.values_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "float32_ch[{i}]: expected {exp}, got {got}"
            );
        }
    }
    Ok(())
}

#[test]
fn master_channel_loaded() -> Result<()> {
    let mut mdf = load_fixture()?;
    let data = mdf.get_channel_data("time_ch").expect("time_ch not found");
    assert!(
        matches!(data, ChannelData::Float64(_)),
        "master should be Float64"
    );
    assert_eq!(data.len(), 4);
    Ok(())
}
