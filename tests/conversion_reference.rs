mod common;

use anyhow::Result;
use arrow::array::Array;
use mdfr::data_holder::channel_data::{AsValuesSlice, ChannelData};
use mdfr::mdfreader::Mdf;
use std::sync::LazyLock;

static BASE: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}MDF4/MDF4.3/Base_Standard/Examples/Conversion/",
        common::mdfreader_tests_path()
    )
});

#[test]
fn test_reference_linear_conversion() -> Result<()> {
    let file = format!("{}LinearConversion/Vector_LinearConversion.mf4", *BASE);
    if !std::path::Path::new(&file).exists() {
        return Ok(());
    }
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;
    mdf.convert_all_channels()?;

    // In Vector_LinearConversion.mf4: p1 = -4.8, p2 = -3.2 => phys = -3.2 * X - 4.8
    if let Some(ChannelData::Float64(builder)) = mdf.get_channel_data("Data channel") {
        let vals = builder.values_slice();
        assert_eq!(vals.len(), 10);
        for (i, &v) in vals.iter().enumerate() {
            let raw = i as f64;
            let expected = -3.2 * raw - 4.8;
            assert!(
                (v - expected).abs() < 1e-10,
                "Linear conversion mismatch at {i}: got {v}, expected {expected}"
            );
        }
    } else {
        panic!("Data channel not found or wrong type");
    }

    Ok(())
}

#[test]
fn test_reference_algebraic_quadratic() -> Result<()> {
    let file = format!(
        "{}TextConversion/Vector_AlgebraicConversionQuadratic.mf4",
        *BASE
    );
    if !std::path::Path::new(&file).exists() {
        return Ok(());
    }
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;
    mdf.convert_all_channels()?;

    // Formula: X * X + 1
    if let Some(ChannelData::Float64(builder)) = mdf.get_channel_data("Data channel") {
        let vals = builder.values_slice();
        assert_eq!(vals.len(), 10);
        let expected = [1.0, 2.0, 5.0, 10.0, 17.0, 26.0, 37.0, 50.0, 65.0, 82.0];
        for (i, (&v, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
            assert!(
                (v - exp).abs() < 1e-10,
                "Quadratic conversion mismatch at {i}: got {v}, expected {exp}"
            );
        }
    } else {
        panic!("Data channel not found or wrong type");
    }

    Ok(())
}

#[test]
fn test_reference_algebraic_sinus() -> Result<()> {
    let file = format!(
        "{}TextConversion/Vector_AlgebraicConversionSinus.mf4",
        *BASE
    );
    if !std::path::Path::new(&file).exists() {
        return Ok(());
    }
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;
    mdf.convert_all_channels()?;

    // Formula: sin(X)
    if let Some(ChannelData::Float64(builder)) = mdf.get_channel_data("Data channel") {
        let vals = builder.values_slice();
        assert_eq!(vals.len(), 10);
        for (i, &v) in vals.iter().enumerate() {
            let raw = i as f64;
            let expected = raw.sin();
            assert!(
                (v - expected).abs() < 1e-10,
                "Sinus conversion mismatch at {i}: got {v}, expected {expected}"
            );
        }
    } else {
        panic!("Data channel not found or wrong type");
    }

    Ok(())
}

#[test]
fn test_reference_value_to_text() -> Result<()> {
    let file = format!("{}LookUpConversion/Vector_Value2TextConversion.mf4", *BASE);
    if !std::path::Path::new(&file).exists() {
        return Ok(());
    }
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;
    mdf.convert_all_channels()?;

    // Map: 1.0->first gear, 2.0->second gear, 3.0->third gear, 4.0->fourth gear, 5.0->fifth gear, default->No match
    let expected = [
        "No match",
        "first gear",
        "second gear",
        "third gear",
        "fourth gear",
        "fifth gear",
        "No match",
        "No match",
        "No match",
        "No match",
    ];

    if let Some(ChannelData::Utf8(builder)) = mdf.get_channel_data("Data channel") {
        let array = builder;
        assert_eq!(array.len(), 10);
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(
                array.value(i),
                exp,
                "Value2Text mismatch at index {i}: got {}, expected {exp}",
                array.value(i)
            );
        }
    } else {
        panic!("Data channel not found or not Utf8");
    }

    Ok(())
}

#[test]
fn test_reference_rational_conversion() -> Result<()> {
    let file = format!(
        "{}RationalConversion/Vector_RationalConversionIntParams.mf4",
        *BASE
    );
    if !std::path::Path::new(&file).exists() {
        return Ok(());
    }
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;
    mdf.convert_all_channels()?;

    // Formula: (1 * X^2 + 2 * X + 3) / (4 * X^2 + 5 * X + 6)
    if let Some(ChannelData::Float64(builder)) = mdf.get_channel_data("Data channel") {
        let vals = builder.values_slice();
        assert_eq!(vals.len(), 10);
        for (i, &v) in vals.iter().enumerate() {
            let x = i as f64;
            let expected = (1.0 * x * x + 2.0 * x + 3.0) / (4.0 * x * x + 5.0 * x + 6.0);
            assert!(
                (v - expected).abs() < 1e-10,
                "Rational conversion mismatch at {i}: got {v}, expected {expected}"
            );
        }
    } else {
        panic!("Data channel not found or wrong type");
    }

    Ok(())
}

#[test]
fn test_reference_bitfield_text_table() -> Result<()> {
    let file = format!(
        "{}BitfieldConversion/RAC_MDF420_BitfieldTextTable.mf4",
        *BASE
    );
    if !std::path::Path::new(&file).exists() {
        return Ok(());
    }
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;
    mdf.convert_all_channels()?;

    // BitfieldTest channel is converted to Utf8
    if let Some(ChannelData::Utf8(builder)) = mdf.get_channel_data("BitfieldTest") {
        let array = builder;
        assert_eq!(array.len(), 366);
        // Ensure values were converted to text and not empty
        assert!(!array.value(0).is_empty());
        assert!(!array.value(100).is_empty());
    } else {
        panic!("BitfieldTest channel not found or not Utf8");
    }

    Ok(())
}
