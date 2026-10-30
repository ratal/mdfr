mod common;

use anyhow::Result;
use arrow::array::{AsArray, Float64Builder, LargeStringBuilder};
use mdfr::data_holder::channel_data::ChannelData;
use mdfr::mdfreader::Mdf;
use std::sync::LazyLock;

static BASE_PATH_MDF4: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}MDF4/MDF4.3/Base_Standard/Examples/",
        common::mdfreader_tests_path()
    )
});

#[test]
fn linear_conversion() -> Result<()> {
    let list_of_paths = ["Conversion/LinearConversion/".to_string()];

    // Linear conversion testing
    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_LinearConversion.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Data channel") {
        let mut vect: Vec<f64> = vec![0.; 10];
        let mut counter: f64 = 0.;
        vect.iter_mut().for_each(|v| {
            *v = counter * -3.2 - 4.8;
            counter += 1.
        });
        assert_eq!(
            &ChannelData::Float64(Float64Builder::new_from_buffer(vect.into(), None).finish()),
            data
        );
    }

    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_LinearConversionFactor0.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Data channel") {
        let vect: Vec<f64> = vec![3.; 10];
        assert_eq!(
            &ChannelData::Float64(Float64Builder::new_from_buffer(vect.into(), None).finish()),
            data
        );
    }

    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "dSPACE_LinearConversion.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    Ok(())
}

#[test]
fn rational_conversion() -> Result<()> {
    let list_of_paths = ["Conversion/RationalConversion/".to_string()];

    // Rational conversion
    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_RationalConversionIntParams.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;

    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_RationalConversionRealParams.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;

    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_RationalConversionZeroedParams.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    Ok(())
}

#[test]
fn algebraic_conversion() -> Result<()> {
    let list_of_paths = ["Conversion/TextConversion/".to_string()];

    // Text conversion (algebraic)
    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_AlgebraicConversionQuadratic.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Data channel") {
        let vect = Vec::from([1., 2., 5., 10., 17., 26., 37., 50., 65., 82.]);
        assert_eq!(
            &ChannelData::Float64(Float64Builder::new_from_buffer(vect.into(), None).finish()),
            data
        );
    }

    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_AlgebraicConversionRational.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;

    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_AlgebraicConversionSinus.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;

    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "dSPACE_AlgebraicConversion.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    Ok(())
}

#[test]
fn lookup_value_to_value_interpolation() -> Result<()> {
    let list_of_paths = ["Conversion/LookUpConversion/".to_string()];

    // Lookup conversion : Value to Value Table With Interpolation
    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_Value2ValueConversionInterpolation.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Data channel") {
        let vect = Vec::from([
            -5.,
            -5.,
            -5.,
            -5.,
            -4.5,
            -4.,
            -3.5,
            -3.,
            -2.5,
            -2.,
            -4. / 3.,
            -2. / 3.,
            0.,
            1. / 3.,
            2. / 3.,
            1.,
            1.5,
            2.,
            1.,
            0.,
            1.5,
            3.,
            4.5,
            6.,
            4.5,
            3.,
            1.5,
            0.,
            0.,
            0.,
        ]);
        assert_eq!(
            &ChannelData::Float64(Float64Builder::new_from_buffer(vect.into(), None).finish()),
            data
        );
    }

    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "dSPACE_Value2ValueConversionInterpolation.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    Ok(())
}

#[test]
fn lookup_value_to_value_no_interpolation() -> Result<()> {
    let list_of_paths = ["Conversion/LookUpConversion/".to_string()];

    // Lookup conversion : Value to Value Table Without Interpolation
    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_Value2ValueConversionNoInterpolation.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Data channel") {
        let vect = Vec::from([
            -5., -5., -5., -5., -5., -5., -5., -2., -2., -2., -2., 0., 0., 0., 1., 1., 1., 2., 2.,
            0., 0., 3., 3., 6., 6., 3., 3., 0., 0., 0.,
        ]);
        assert_eq!(
            &ChannelData::Float64(Float64Builder::new_from_buffer(vect.into(), None).finish()),
            data
        );
    }

    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "dSPACE_Value2ValueConversionNoInterpolation.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    Ok(())
}

#[test]
fn lookup_value_range_to_value() -> Result<()> {
    let list_of_paths = ["Conversion/LookUpConversion/".to_string()];

    // Lookup conversion : Value Range to Value
    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_ValueRange2ValueConversion.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Data channel") {
        // Spec 6.17.8: float data uses [lo, hi) exclusive upper bound.
        // Raw data: -15…14 (step 1).  CC ranges:
        //   [-10,-7)→-1, [-7,-5)→0, [-5,0)→1, [0,2)→2, [2,5)→3,
        //   [5,6)→5, [6,8.5)→6, [8.5,10)→7, [10,12)→8, [12,14)→9, default=-1
        // Values -15…-11 are below all ranges → default -1.
        // Value 14 equals the exclusive upper of last range → default -1.
        let vect = Vec::from([
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0,
            2.0, 3.0, 3.0, 3.0, 5.0, 6.0, 6.0, 6.0, 7.0, 8.0, 8.0, 9.0, 9.0, -1.0,
        ]);
        assert_eq!(
            &ChannelData::Float64(Float64Builder::new_from_buffer(vect.into(), None).finish()),
            data
        );
    }
    Ok(())
}

#[test]
fn lookup_value_to_text() -> Result<()> {
    let list_of_paths = ["Conversion/LookUpConversion/".to_string()];

    // Lookup conversion : Value to Text
    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_Value2TextConversion.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Data channel") {
        let mut target = LargeStringBuilder::with_capacity(10, 20);
        target.append_value("No match");
        target.append_value("first gear");
        target.append_value("second gear");
        target.append_value("third gear");
        target.append_value("fourth gear");
        target.append_value("fifth gear");
        target.append_value("No match");
        target.append_value("No match");
        target.append_value("No match");
        target.append_value("No match");
        assert_eq!(&ChannelData::Utf8(target.finish()), data);
    }

    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "dSPACE_Value2TextConversion.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    Ok(())
}

#[test]
fn lookup_value_range_to_text() -> Result<()> {
    let list_of_paths = ["Conversion/LookUpConversion/".to_string()];

    // Lookup conversion : Value range to Text
    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_ValueRange2TextConversion.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Data channel") {
        let mut target = LargeStringBuilder::with_capacity(10, 20);
        target.append_value("Out of range");
        target.append_value("very low");
        target.append_value("very low");
        target.append_value("very low");
        target.append_value("low");
        target.append_value("low");
        target.append_value("medium");
        target.append_value("medium");
        target.append_value("high");
        target.append_value("high");
        assert_eq!(&ChannelData::Utf8(target.finish()), data);
    }

    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "dSPACE_ValueRange2TextConversion.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    Ok(())
}

#[test]
fn status_string_table_conversion() -> Result<()> {
    let list_of_paths = ["Conversion/PartialConversion/".to_string()];

    // Lookup conversion : Value range to Text,
    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_StatusStringTableConversionAlgebraic.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Data channel") {
        let mut vect: Vec<f64> = vec![0.; 300];
        let mut counter: f64 = 0.;
        vect.iter_mut().for_each(|v| {
            *v = counter;
            counter += 0.1
        });
        let mut target = LargeStringBuilder::with_capacity(vect.len(), 32);
        vect.iter().for_each(|v| {
            if 9.9999 <= *v && *v <= 10.1001 {
                target.append_value("Illegal value")
            } else if 20.0 <= *v && *v <= 30.0 {
                target.append_value("Out of range")
            } else {
                target.append_value((10.0 / (v - 10.0)).to_string())
            }
        });
        let data_values = data.finish_cloned();
        let data_values = data_values
            .as_string::<i64>()
            .iter()
            .collect::<Vec<Option<&str>>>();
        let target_values = target.finish();
        let target_values = target_values.iter().collect::<Vec<Option<&str>>>();
        assert_eq!(target_values[0], data_values[0]);
        assert_eq!(target_values[299], data_values[299]);
        assert_eq!(target_values[101], data_values[101]);
    }
    Ok(())
}

#[test]
fn text_to_value_conversion() -> Result<()> {
    let list_of_paths = ["Conversion/StringConversion/".to_string()];

    // Text conversion : Text to Value
    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_Text2ValueConversion.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Data channel") {
        let vect = Vec::from([-50., 1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        assert_eq!(
            &ChannelData::Float64(Float64Builder::new_from_buffer(vect.into(), None).finish()),
            data
        );
    }
    Ok(())
}

#[test]
fn text_to_text_conversion() -> Result<()> {
    let list_of_paths = ["Conversion/StringConversion/".to_string()];

    // Text conversion : Text to Text
    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_Text2TextConversion.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Data channel") {
        let mut target = LargeStringBuilder::with_capacity(10, 20);
        target.append_value("No translation");
        target.append_value("Eins");
        target.append_value("Zwei");
        target.append_value("Drei");
        target.append_value("Vier");
        target.append_value("Fünf");
        target.append_value("Sechs");
        target.append_value("Sieben");
        target.append_value("Acht");
        target.append_value("Neun");
        assert_eq!(&ChannelData::Utf8(target.finish()), data);
    }
    Ok(())
}

#[test]
fn partial_conversion() -> Result<()> {
    let list_of_paths = ["Conversion/PartialConversion/".to_string()];

    // Partial conversion
    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_PartialConversionLinearIdentityAlgebraic.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;

    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_PartialConversionValueRange2TextRational.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;

    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_StatusStringTableConversionAlgebraic.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    Ok(())
}

#[test]
fn bitfield_conversion() -> Result<()> {
    let list_of_paths = ["Conversion/BitfieldConversion/".to_string()];

    // Bitfield conversion
    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "RAC_MDF420_BitfieldTextTable.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    Ok(())
}

mod conversions4_tests {
    use arrow::array::{Float64Builder, Int32Builder, UInt16Builder};
    use fasteval::Compiler;
    use itertools::Itertools;
    use mdfr::data_holder::channel_data::{AsValuesSlice, ChannelData};
    use mdfr::mdfinfo::mdfinfo4::{Cc4Block, CcVal, Cn4, SharableBlocks};
    use mdfr::mdfreader::conversions4::*;

    #[test]
    fn test_linear_calculation() {
        // v * p2 + p1 where p1=10.0, p2=2.0
        let p1 = 10.0;
        let p2 = 2.0;

        // Int32
        let mut builder = Int32Builder::new();
        builder.append_value(0);
        builder.append_value(1);
        builder.append_value(5);
        builder.append_value(-3);
        let result = linear_calculation(&builder.finish(), p1, p2).unwrap();
        let values = result.values_slice();
        assert_eq!(values.len(), 4);
        assert!((values[0] - 10.0).abs() < 1e-12); // 0*2+10
        assert!((values[1] - 12.0).abs() < 1e-12); // 1*2+10
        assert!((values[2] - 20.0).abs() < 1e-12); // 5*2+10
        assert!((values[3] - 4.0).abs() < 1e-12); // -3*2+10

        // Float64
        let mut builder = Float64Builder::new();
        builder.append_value(1.5);
        builder.append_value(2.5);
        let result = linear_calculation(&builder.finish(), p1, p2).unwrap();
        let values = result.values_slice();
        assert!((values[0] - 13.0).abs() < 1e-12); // 1.5*2+10
        assert!((values[1] - 15.0).abs() < 1e-12); // 2.5*2+10

        // UInt16
        let mut builder = UInt16Builder::new();
        builder.append_value(100);
        builder.append_value(0);
        let result = linear_calculation(&builder.finish(), p1, p2).unwrap();
        let values = result.values_slice();
        assert!((values[0] - 210.0).abs() < 1e-12); // 100*2+10
        assert!((values[1] - 10.0).abs() < 1e-12); // 0*2+10
    }

    #[test]
    fn test_rational_calculation() {
        // (x²*p1 + x*p2 + p3) / (x²*p4 + x*p5 + p6)
        // Simple linear: p1=0, p2=2, p3=1, p4=0, p5=0, p6=1 → (2x+1)/1
        let cc_val = vec![0.0, 2.0, 1.0, 0.0, 0.0, 1.0];

        let mut builder = Int32Builder::new();
        builder.append_value(0);
        builder.append_value(1);
        builder.append_value(5);
        let result = rational_calculation(&builder.finish(), &cc_val).unwrap();
        let values = result.values_slice();
        assert!((values[0] - 1.0).abs() < 1e-12); // (0+0+1)/1
        assert!((values[1] - 3.0).abs() < 1e-12); // (0+2+1)/1
        assert!((values[2] - 11.0).abs() < 1e-12); // (0+10+1)/1

        // Quadratic: p1=1, p2=0, p3=0, p4=0, p5=0, p6=1 → x²
        let cc_val = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let mut builder = Float64Builder::new();
        builder.append_value(3.0);
        builder.append_value(4.0);
        let result = rational_calculation(&builder.finish(), &cc_val).unwrap();
        let values = result.values_slice();
        assert!((values[0] - 9.0).abs() < 1e-12); // 3²
        assert!((values[1] - 16.0).abs() < 1e-12); // 4²
    }

    #[test]
    fn test_value_to_value_with_interpolation_primitive() {
        // Table pairs: x=0→y=0, x=10→y=100, x=20→y=200
        let keys = [0.0, 0.0, 10.0, 100.0, 20.0, 200.0];
        let val: Vec<(&f64, &f64)> = keys.iter().tuples().collect();

        let mut builder = Float64Builder::new();
        builder.append_value(0.0); // exact → 0
        builder.append_value(5.0); // interpolate → 50
        builder.append_value(10.0); // exact → 100
        builder.append_value(15.0); // interpolate → 150
        builder.append_value(-5.0); // below first → 0
        builder.append_value(25.0); // above last → 200

        let result = value_to_value_with_interpolation_primitive(&builder.finish(), &val).unwrap();
        let values = result.values_slice();
        assert_eq!(values.len(), 6);
        assert!((values[0] - 0.0).abs() < 1e-12);
        assert!((values[1] - 50.0).abs() < 1e-12);
        assert!((values[2] - 100.0).abs() < 1e-12);
        assert!((values[3] - 150.0).abs() < 1e-12);
        assert!((values[4] - 0.0).abs() < 1e-12);
        assert!((values[5] - 200.0).abs() < 1e-12);
    }

    #[test]
    fn test_value_to_value_without_interpolation_primitive() {
        // Table pairs: x=0→y=0, x=10→y=100, x=20→y=200 (MDF4 spec 6.17.7)
        let keys = [0.0, 0.0, 10.0, 100.0, 20.0, 200.0];
        let val: Vec<(&f64, &f64)> = keys.iter().tuples().collect();

        let mut builder = Float64Builder::new();
        builder.append_value(0.0); // exact match → 0
        builder.append_value(3.0); // between key[0]=0 and key[1]=10, nearer to 0 → 0
        builder.append_value(7.0); // between key[0]=0 and key[1]=10, nearer to 10 → 100
        builder.append_value(10.0); // exact match → 100
        builder.append_value(-5.0); // below first key → value[0] = 0
        builder.append_value(25.0); // above last key → value[n-1] = 200

        let result = value_to_value_without_interpolation_primitive(&builder.finish(), &val).unwrap();
        let values = result.values_slice();
        assert_eq!(values.len(), 6);
        assert!((values[0] - 0.0).abs() < 1e-12); // exact → 0
        assert!((values[1] - 0.0).abs() < 1e-12); // nearest key=0 → 0
        assert!((values[2] - 100.0).abs() < 1e-12); // nearest key=10 → 100
        assert!((values[3] - 100.0).abs() < 1e-12); // exact → 100
        assert!((values[4] - 0.0).abs() < 1e-12); // below first → value[0]=0
        assert!((values[5] - 200.0).abs() < 1e-12); // above last → value[n-1]=200
    }

    #[test]
    fn test_value_range_to_value_table() {
        // Ranges: (key_min, key_max, value)
        let val = vec![(0.0, 10.0, 100.0), (10.0, 20.0, 200.0), (20.0, 30.0, 300.0)];
        let default = -1.0;

        let mut builder = Float64Builder::new();
        builder.append_value(0.0); // exact match on key_min=0 → 100
        builder.append_value(10.0); // exact match on key_min=10 → 200
        builder.append_value(20.0); // exact match on key_min=20 → 300
        builder.append_value(-5.0); // below all ranges → default
        builder.append_value(25.0); // above last key_min but within last upper bound → 300

        let result =
            value_range_to_value_table_calculation(&builder.finish(), &val, &default, false).unwrap();
        let values = result.values_slice();
        assert_eq!(values.len(), 5);
        assert!((values[0] - 100.0).abs() < 1e-12);
        assert!((values[1] - 200.0).abs() < 1e-12);
        assert!((values[2] - 300.0).abs() < 1e-12);
        assert!((values[3] - (-1.0)).abs() < 1e-12);
        assert!((values[4] - 300.0).abs() < 1e-12);
    }

    #[test]
    fn test_value_range_float_touching_boundaries() {
        // Replicate the Vector_ValueRange2ValueConversion.mf4 CC table (float [lo, hi) semantics)
        let val = vec![
            (-10.0f64, -7.0, -1.0), // range 0
            (-7.0, -5.0, 0.0),      // range 1 (touching: min=-7 == max of range 0)
            (-5.0, 0.0, 1.0),
        ];
        let default = -1.0f64;

        let mut builder = Float64Builder::new();
        builder.append_value(-9.0); // in range 0 [-10,-7): -10<=-9<-7 → True → -1
        builder.append_value(-7.0); // at touching boundary: Ok(1)→range 1→ 0 (not range 0 since -7 is not < -7)
        builder.append_value(-6.0); // in range 1: -7<=-6<-5 → True → 0
        builder.append_value(100.0); // above all ranges → default -1

        let result =
            value_range_to_value_table_calculation(&builder.finish(), &val, &default, false).unwrap();
        let values = result.values_slice();
        assert!(
            (values[0] - (-1.0)).abs() < 1e-12,
            "raw=-9 should be -1, got {}",
            values[0]
        );
        assert!(
            (values[1] - 0.0).abs() < 1e-12,
            "raw=-7 should be 0, got {}",
            values[1]
        );
        assert!(
            (values[2] - 0.0).abs() < 1e-12,
            "raw=-6 should be 0, got {}",
            values[2]
        );
        assert!(
            (values[3] - (-1.0)).abs() < 1e-12,
            "raw=100 should be -1, got {}",
            values[3]
        );
    }

    #[test]
    fn test_algebraic_conversion_primitive() {
        // Expression: X * 2 + 1
        let parser = fasteval::Parser::new();
        let mut slab = fasteval::Slab::new();
        let compiled = parser.parse("X * 2 + 1", &mut slab.ps).unwrap();
        let compiled = compiled.from(&slab.ps).compile(&slab.ps, &mut slab.cs);

        let mut builder = Float64Builder::new();
        builder.append_value(0.0); // → 1
        builder.append_value(1.0); // → 3
        builder.append_value(5.0); // → 11
        builder.append_value(-3.0); // → -5

        let result = alegbraic_conversion_primitive(&compiled, &slab, &builder.finish()).unwrap();
        let values = result.values_slice();
        assert_eq!(values.len(), 4);
        assert!((values[0] - 1.0).abs() < 1e-12);
        assert!((values[1] - 3.0).abs() < 1e-12);
        assert!((values[2] - 11.0).abs() < 1e-12);
        assert!((values[3] - (-5.0)).abs() < 1e-12);
    }

    // ── Helper for higher-level conversion tests ──

    fn make_cn4_with_data(data: ChannelData) -> Cn4 {
        Cn4 {
            data,
            ..Default::default()
        }
    }

    // ── linear_conversion (Cn4) tests ──

    #[test]
    fn test_linear_cn4_uint8() {
        use arrow::array::UInt8Builder;
        let mut builder = UInt8Builder::new();
        builder.append_value(2);
        builder.append_value(4);
        let mut cn = make_cn4_with_data(ChannelData::UInt8(builder.finish()));
        linear_conversion(&mut cn, &[1.0, 2.0]).unwrap(); // p1=1, p2=2 → v*2+1
        if let ChannelData::Float64(ref b) = cn.data {
            let vals = b.values_slice();
            assert!((vals[0] - 5.0).abs() < 1e-12); // 2*2+1
            assert!((vals[1] - 9.0).abs() < 1e-12); // 4*2+1
        } else {
            panic!("Expected Float64 after linear conversion");
        }
    }

    #[test]
    fn test_linear_cn4_int16() {
        use arrow::array::Int16Builder;
        let mut builder = Int16Builder::new();
        builder.append_value(10);
        let mut cn = make_cn4_with_data(ChannelData::Int16(builder.finish()));
        linear_conversion(&mut cn, &[0.0, 0.5]).unwrap();
        if let ChannelData::Float64(ref b) = cn.data {
            let vals = b.values_slice();
            assert!((vals[0] - 5.0).abs() < 1e-12); // 10*0.5+0
        } else {
            panic!("Expected Float64");
        }
    }

    #[test]
    fn test_linear_cn4_array_d_int16() {
        use arrow::array::Int16Builder;
        use arrow::datatypes::Int16Type;
        use mdfr::data_holder::tensor_arrow::{Order, TensorArrow};
        let mut builder = Int16Builder::new();
        builder.append_value(1);
        builder.append_value(2);
        let tensor =
            TensorArrow::<Int16Type>::new_from_primitive(builder, None, vec![2], Order::RowMajor);
        let mut cn = make_cn4_with_data(ChannelData::ArrayDInt16(tensor));
        linear_conversion(&mut cn, &[0.0, 3.0]).unwrap();
        if let ChannelData::ArrayDFloat64(ref t) = cn.data {
            let vals = t.values_slice();
            assert!((vals[0] - 3.0).abs() < 1e-12);
            assert!((vals[1] - 6.0).abs() < 1e-12);
        } else {
            panic!("Expected ArrayDFloat64");
        }
    }

    #[test]
    fn test_linear_cn4_array_d_float64() {
        use arrow::datatypes::Float64Type;
        use mdfr::data_holder::tensor_arrow::{Order, TensorArrow};
        let mut builder = Float64Builder::new();
        builder.append_value(1.0);
        let tensor =
            TensorArrow::<Float64Type>::new_from_primitive(builder, None, vec![1], Order::RowMajor);
        let mut cn = make_cn4_with_data(ChannelData::ArrayDFloat64(tensor));
        linear_conversion(&mut cn, &[0.0, 2.0]).unwrap();
        if let ChannelData::ArrayDFloat64(ref t) = cn.data {
            let vals = t.values_slice();
            assert!((vals[0] - 2.0).abs() < 1e-12);
        } else {
            panic!("Expected ArrayDFloat64");
        }
    }

    #[test]
    fn test_linear_cn4_utf8_warn() {
        use arrow::array::LargeStringBuilder;
        // Utf8 data should produce a warn but NOT change data type and should return Ok
        let mut builder = LargeStringBuilder::new();
        let mut cn = make_cn4_with_data(ChannelData::Utf8(builder.finish()));
        let result = linear_conversion(&mut cn, &[1.0, 2.0]);
        assert!(result.is_ok());
        // data type unchanged
        assert!(matches!(cn.data, ChannelData::Utf8(_)));
    }

    #[test]
    fn test_linear_cn4_identity_no_op() {
        use arrow::array::UInt8Builder;
        let mut builder = UInt8Builder::new();
        builder.append_value(7);
        let mut cn = make_cn4_with_data(ChannelData::UInt8(builder.finish()));
        // p1=0, p2=1 is identity → no change
        linear_conversion(&mut cn, &[0.0, 1.0]).unwrap();
        // data type should remain UInt8 (identity skipped)
        assert!(matches!(cn.data, ChannelData::UInt8(_)));
    }

    // ── rational_conversion (Cn4) tests ──

    #[test]
    fn test_rational_cn4_float64() {
        // identity rational: (0*x^2 + 1*x + 0) / (0*x^2 + 0*x + 1) = x
        let cc_val = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut builder = Float64Builder::new();
        builder.append_value(2.0);
        let mut cn = make_cn4_with_data(ChannelData::Float64(builder.finish()));
        rational_conversion(&mut cn, &cc_val).unwrap();
        if let ChannelData::Float64(ref b) = cn.data {
            let vals = b.values_slice();
            assert!((vals[0] - 2.0).abs() < 1e-12);
        } else {
            panic!("Expected Float64");
        }
    }

    #[test]
    fn test_rational_cn4_uint8() {
        // (0*x^2 + 1*x + 0) / (0*x^2 + 0*x + 2) = x/2
        let cc_val = vec![0.0, 1.0, 0.0, 0.0, 0.0, 2.0];
        use arrow::array::UInt8Builder;
        let mut builder = UInt8Builder::new();
        builder.append_value(4);
        let mut cn = make_cn4_with_data(ChannelData::UInt8(builder.finish()));
        rational_conversion(&mut cn, &cc_val).unwrap();
        if let ChannelData::Float64(ref b) = cn.data {
            let vals = b.values_slice();
            assert!((vals[0] - 2.0).abs() < 1e-12);
        } else {
            panic!("Expected Float64");
        }
    }

    // ── value_to_value_with_interpolation (Cn4) tests ──

    #[test]
    fn test_vtv_interp_cn4_float64() {
        // cc_val: pairs (x, y) interleaved: 1.0→10.0, 2.0→20.0
        let cc_val = vec![1.0, 10.0, 2.0, 20.0];
        let mut builder = Float64Builder::new();
        builder.append_value(1.5); // interpolated: 15.0
        let mut cn = make_cn4_with_data(ChannelData::Float64(builder.finish()));
        value_to_value_with_interpolation(&mut cn, cc_val).unwrap();
        if let ChannelData::Float64(ref b) = cn.data {
            let vals = b.values_slice();
            assert!((vals[0] - 15.0).abs() < 1e-9);
        } else {
            panic!("Expected Float64");
        }
    }

    #[test]
    fn test_vtv_interp_cn4_uint8() {
        // cc_val: 0→0, 10→100
        let cc_val = vec![0.0, 0.0, 10.0, 100.0];
        use arrow::array::UInt8Builder;
        let mut builder = UInt8Builder::new();
        builder.append_value(5); // 50
        let mut cn = make_cn4_with_data(ChannelData::UInt8(builder.finish()));
        value_to_value_with_interpolation(&mut cn, cc_val).unwrap();
        if let ChannelData::Float64(ref b) = cn.data {
            let vals = b.values_slice();
            assert!((vals[0] - 50.0).abs() < 1e-9);
        } else {
            panic!("Expected Float64");
        }
    }

    #[test]
    fn test_vtv_interp_cn4_array_d_uint8() {
        use arrow::array::UInt8Builder;
        use arrow::datatypes::UInt8Type;
        use mdfr::data_holder::tensor_arrow::{Order, TensorArrow};
        let cc_val = vec![0.0, 0.0, 10.0, 100.0];
        let mut builder = UInt8Builder::new();
        builder.append_value(5);
        let tensor =
            TensorArrow::<UInt8Type>::new_from_primitive(builder, None, vec![1], Order::RowMajor);
        let mut cn = make_cn4_with_data(ChannelData::ArrayDUInt8(tensor));
        value_to_value_with_interpolation(&mut cn, cc_val).unwrap();
        assert!(matches!(cn.data, ChannelData::ArrayDFloat64(_)));
    }

    #[test]
    fn test_vtv_interp_cn4_int64_noop() {
        use arrow::array::Int64Builder;
        // Int64 not handled in value_to_value_with_interpolation (falls through to warn)
        let mut builder = Int64Builder::new();
        builder.append_value(5);
        let mut cn = make_cn4_with_data(ChannelData::Int64(builder.finish()));
        let result = value_to_value_with_interpolation(&mut cn, vec![0.0, 0.0, 10.0, 100.0]);
        assert!(result.is_ok());
    }

    // ── value_to_value_without_interpolation (Cn4) tests ──

    #[test]
    fn test_vtv_no_interp_cn4_float64() {
        let cc_val = vec![1.0, 100.0, 2.0, 200.0];
        let mut builder = Float64Builder::new();
        builder.append_value(1.0); // exact match → 100.0
        let mut cn = make_cn4_with_data(ChannelData::Float64(builder.finish()));
        value_to_value_without_interpolation(&mut cn, cc_val).unwrap();
        if let ChannelData::Float64(ref b) = cn.data {
            let vals = b.values_slice();
            assert!((vals[0] - 100.0).abs() < 1e-12);
        } else {
            panic!("Expected Float64");
        }
    }

    #[test]
    fn test_vtv_no_interp_cn4_uint8() {
        use arrow::array::UInt8Builder;
        let cc_val = vec![1.0, 10.0, 2.0, 20.0];
        let mut builder = UInt8Builder::new();
        builder.append_value(2); // exact match → 20.0
        let mut cn = make_cn4_with_data(ChannelData::UInt8(builder.finish()));
        value_to_value_without_interpolation(&mut cn, cc_val).unwrap();
        if let ChannelData::Float64(ref b) = cn.data {
            let vals = b.values_slice();
            assert!((vals[0] - 20.0).abs() < 1e-12);
        } else {
            panic!("Expected Float64");
        }
    }

    // ── value_range_to_value_table (Cn4) tests ──

    #[test]
    fn test_vrv_cn4_float64() {
        // cc_val groups of 3: (min, max, out_val)
        // Range [1.0, 2.0] → 100.0
        let cc_val = vec![1.0, 2.0, 100.0];
        let mut builder = Float64Builder::new();
        builder.append_value(1.5); // in [1.0, 2.0] → 100.0
        let mut cn = make_cn4_with_data(ChannelData::Float64(builder.finish()));
        value_range_to_value_table(&mut cn, cc_val).unwrap();
        if let ChannelData::Float64(ref b) = cn.data {
            let vals = b.values_slice();
            assert!((vals[0] - 100.0).abs() < 1e-12);
        } else {
            panic!("Expected Float64");
        }
    }

    #[test]
    fn test_vrv_cn4_uint8() {
        use arrow::array::UInt8Builder;
        // Range [0.0, 10.0] → 42.0
        let cc_val = vec![0.0, 10.0, 42.0];
        let mut builder = UInt8Builder::new();
        builder.append_value(5); // in [0, 10]
        let mut cn = make_cn4_with_data(ChannelData::UInt8(builder.finish()));
        value_range_to_value_table(&mut cn, cc_val).unwrap();
        if let ChannelData::Float64(ref b) = cn.data {
            let vals = b.values_slice();
            assert!((vals[0] - 42.0).abs() < 1e-12);
        } else {
            panic!("Expected Float64");
        }
    }

    #[test]
    fn test_vrv_cn4_array_d_warn() {
        use arrow::array::Int8Builder;
        use arrow::datatypes::Int8Type;
        use mdfr::data_holder::tensor_arrow::{Order, TensorArrow};
        let mut builder = Int8Builder::new();
        builder.append_value(5);
        let tensor =
            TensorArrow::<Int8Type>::new_from_primitive(builder, None, vec![1], Order::RowMajor);
        let mut cn = make_cn4_with_data(ChannelData::ArrayDInt8(tensor));
        // ArrayDInt8 falls through to warn path
        let result = value_range_to_value_table(&mut cn, vec![0.0, 10.0, 42.0]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_algebraic_conversion_large_parallel() {
        let parser = fasteval::Parser::new();
        let mut slab = fasteval::Slab::new();
        let compiled = parser.parse("X * 3.0 + 2.0", &mut slab.ps).unwrap();
        let compiled = compiled.from(&slab.ps).compile(&slab.ps, &mut slab.cs);

        let n = 1_000_000;
        let mut builder = Float64Builder::new();
        for i in 0..n {
            builder.append_value(i as f64);
        }

        let start = std::time::Instant::now();
        let result = alegbraic_conversion_primitive(&compiled, &slab, &builder.finish()).unwrap();
        let duration = start.elapsed();
        println!(
            "algebraic conversion 1,000,000 samples took: {:?}",
            duration
        );

        let values = result.values_slice();
        assert_eq!(values.len(), n);
        assert!((values[0] - 2.0).abs() < 1e-12);
        assert!((values[n - 1] - ((n - 1) as f64 * 3.0 + 2.0)).abs() < 1e-12);
    }

    #[test]
    fn test_bitfield_text_table_rounding_float() {
        use arrow::array::Float32Builder;

        let mut sharable = SharableBlocks::default();
        let cc_ptr = 1000i64;
        let cc = Cc4Block::new_with_val(7, CcVal::Real(vec![1.0]), vec![2001, 2002]);
        sharable.cc.insert(cc_ptr, cc);
        sharable.create_tx(2001, "BIT0_ON".to_string());
        sharable.create_tx(2002, "DEFAULT".to_string());

        let cc_val = vec![1u64];
        let cc_ref = vec![cc_ptr];

        // Raw float: 0.9 rounds to 1 (bit 0 set), 0.4 rounds to 0 (bit 0 clear -> default)
        let mut builder = Float32Builder::new();
        builder.append_value(0.9);
        builder.append_value(0.4);

        let mut cn = make_cn4_with_data(ChannelData::Float32(builder.finish()));
        bitfield_text_table(&mut cn, &cc_val, &cc_ref, &sharable).unwrap();

        if let ChannelData::Utf8(ref s) = cn.data {
            assert_eq!(s.value(0), " | BIT0_ON");
            assert_eq!(s.value(1), "DEFAULT");
        } else {
            panic!("Expected Utf8 channel data");
        }
    }
}
