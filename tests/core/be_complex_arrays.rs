/// Integration tests for BigEndian channels, Complex32, and ArrayDFloat64 fixtures.
///
/// Synthetic MDF4 files exercise paths in `read_channels_from_bytes` and
/// `read_one_channel_array` that the existing sample suite never hits:
///
///  - `be_scalars.mf4`      → BE Int16 and BE Float64
///  - `complex_f32_le.mf4`  → Complex32 LE f32-pair arm
///  - `array_f64_le.mf4`    → ArrayDFloat64 LE + CA block parsing
///  - `ld_be_channels.mf4`  → LD-backed BE Int16/Float64/UInt16/Float32
use crate::common::synthetic;
use anyhow::Result;
use mdfr::data_holder::channel_data::{AsValuesSlice, ChannelData};
use mdfr::mdfreader::Mdf;

#[test]
fn be_int16_reads_correctly() -> Result<()> {
    let path = synthetic::ensure_be_scalars();
    let mut mdf = Mdf::new(path)?;
    mdf.load_all_channels_data_in_memory()?;

    let data = mdf.get_channel_data("be_i16").expect("be_i16 not found");
    assert!(
        matches!(data, ChannelData::Int16(_)),
        "expected Int16, got {}",
        data.data_type(false)
    );
    if let ChannelData::Int16(arr) = data {
        let expected = [-100i16, 0, 100, 200];
        for (i, (&got, &exp)) in arr.values_slice().iter().zip(expected.iter()).enumerate() {
            assert_eq!(got, exp, "be_i16[{i}]: expected {exp}, got {got}");
        }
    }
    Ok(())
}

#[test]
fn be_float64_reads_correctly() -> Result<()> {
    let path = synthetic::ensure_be_scalars();
    let mut mdf = Mdf::new(path)?;
    mdf.load_all_channels_data_in_memory()?;

    let data = mdf.get_channel_data("be_f64").expect("be_f64 not found");
    assert!(
        matches!(data, ChannelData::Float64(_)),
        "expected Float64, got {}",
        data.data_type(false)
    );
    if let ChannelData::Float64(arr) = data {
        let expected = [1.5f64, 2.5, 3.5, 4.5];
        for (i, (&got, &exp)) in arr.values_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-9,
                "be_f64[{i}]: expected {exp}, got {got}"
            );
        }
    }
    Ok(())
}

#[test]
fn complex32_le_reads_correctly() -> Result<()> {
    let path = synthetic::ensure_complex_f32_le();
    let mut mdf = Mdf::new(path)?;
    mdf.load_all_channels_data_in_memory()?;

    let data = mdf.get_channel_data("cx32_ch").expect("cx32_ch not found");
    assert!(
        matches!(data, ChannelData::Complex32(_)),
        "expected Complex32, got {}",
        data.data_type(false)
    );
    if let ChannelData::Complex32(arr) = data {
        // values_slice() returns interleaved [re0, im0, re1, im1, ...]
        let vals = arr.values_slice();
        assert_eq!(vals.len(), 8, "expected 4 complex samples = 8 floats");
        let expected_re = [1.0f32, 3.0, 5.0, 7.0];
        let expected_im = [2.0f32, 4.0, 6.0, 8.0];
        for i in 0..4 {
            assert!(
                (vals[i * 2] - expected_re[i]).abs() < 1e-6,
                "cx32_ch[{i}].re: expected {}, got {}",
                expected_re[i],
                vals[i * 2]
            );
            assert!(
                (vals[i * 2 + 1] - expected_im[i]).abs() < 1e-6,
                "cx32_ch[{i}].im: expected {}, got {}",
                expected_im[i],
                vals[i * 2 + 1]
            );
        }
    }
    Ok(())
}

#[test]
fn array_f64_le_reads_correctly() -> Result<()> {
    let path = synthetic::ensure_array_f64_le();
    let mut mdf = Mdf::new(path)?;
    mdf.load_all_channels_data_in_memory()?;

    let data = mdf.get_channel_data("arr_f64").expect("arr_f64 not found");
    assert!(
        matches!(data, ChannelData::ArrayDFloat64(_)),
        "expected ArrayDFloat64, got {}",
        data.data_type(false)
    );
    if let ChannelData::ArrayDFloat64(arr) = data {
        let vals = arr.values_slice();
        assert_eq!(
            vals.len(),
            12,
            "expected exactly 12 f64 values (4 cycles * 3 elements), got {}",
            vals.len()
        );
        let expected: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        for (i, (&got, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-9,
                "arr_f64[{i}]: expected {exp}, got {got}"
            );
        }
        assert!(arr.ndim() >= 1, "expected ndim >= 1");
    }
    Ok(())
}

fn load_ld_be() -> Result<Mdf> {
    let path = synthetic::ensure_ld_be_channels();
    let mut mdf = Mdf::new(path)?;
    mdf.load_all_channels_data_in_memory()?;
    Ok(mdf)
}

#[test]
fn ld_bei16_reads_correctly() -> Result<()> {
    let mut mdf = load_ld_be()?;
    let data = mdf.get_channel_data("bei16").expect("bei16 not found");
    assert!(
        matches!(data, ChannelData::Int16(_)),
        "expected Int16, got {}",
        data.data_type(false)
    );
    if let ChannelData::Int16(arr) = data {
        let expected = [-100i16, 0, 100, 200];
        for (i, (&got, &exp)) in arr.values_slice().iter().zip(expected.iter()).enumerate() {
            assert_eq!(got, exp, "bei16[{i}]: expected {exp}, got {got}");
        }
    }
    Ok(())
}

#[test]
fn ld_bef64_reads_correctly() -> Result<()> {
    let mut mdf = load_ld_be()?;
    let data = mdf.get_channel_data("bef64").expect("bef64 not found");
    assert!(
        matches!(data, ChannelData::Float64(_)),
        "expected Float64, got {}",
        data.data_type(false)
    );
    if let ChannelData::Float64(arr) = data {
        let expected = [1.5f64, 2.5, 3.5, 4.5];
        for (i, (&got, &exp)) in arr.values_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-9,
                "bef64[{i}]: expected {exp}, got {got}"
            );
        }
    }
    Ok(())
}

#[test]
fn ld_beu16_reads_correctly() -> Result<()> {
    let mut mdf = load_ld_be()?;
    let data = mdf.get_channel_data("beu16").expect("beu16 not found");
    assert!(
        matches!(data, ChannelData::UInt16(_)),
        "expected UInt16, got {}",
        data.data_type(false)
    );
    if let ChannelData::UInt16(arr) = data {
        let expected = [100u16, 200, 300, 400];
        for (i, (&got, &exp)) in arr.values_slice().iter().zip(expected.iter()).enumerate() {
            assert_eq!(got, exp, "beu16[{i}]: expected {exp}, got {got}");
        }
    }
    Ok(())
}

#[test]
fn ld_bef32_reads_correctly() -> Result<()> {
    let mut mdf = load_ld_be()?;
    let data = mdf.get_channel_data("bef32").expect("bef32 not found");
    assert!(
        matches!(data, ChannelData::Float32(_)),
        "expected Float32, got {}",
        data.data_type(false)
    );
    if let ChannelData::Float32(arr) = data {
        let expected = [1.5f32, 2.5, 3.5, 4.5];
        for (i, (&got, &exp)) in arr.values_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "bef32[{i}]: expected {exp}, got {got}"
            );
        }
    }
    Ok(())
}
