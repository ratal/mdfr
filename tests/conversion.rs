use anyhow::Result;
use arrow::array::{AsArray, Float64Builder, LargeStringBuilder};
use mdfr::data_holder::channel_data::ChannelData;
use mdfr::mdfreader::Mdf;
use std::sync::LazyLock;

static BASE_PATH_MDF4: LazyLock<String> = LazyLock::new(|| {
    "/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/MDF4.3/Base_Standard/Examples/"
        .to_string()
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
            &ChannelData::Float64(Float64Builder::new_from_buffer(vect.into(), None)),
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
            &ChannelData::Float64(Float64Builder::new_from_buffer(vect.into(), None)),
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
            &ChannelData::Float64(Float64Builder::new_from_buffer(vect.into(), None)),
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
            &ChannelData::Float64(Float64Builder::new_from_buffer(vect.into(), None)),
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
            &ChannelData::Float64(Float64Builder::new_from_buffer(vect.into(), None)),
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
        let vect = Vec::from([
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            3.0, 3.0, 5.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 9.0, 9.0, 9.0,
        ]);
        assert_eq!(
            &ChannelData::Float64(Float64Builder::new_from_buffer(vect.into(), None)),
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
        assert_eq!(&ChannelData::Utf8(target), data);
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
        assert_eq!(&ChannelData::Utf8(target), data);
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
        let target_values = target.finish_cloned();
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
            &ChannelData::Float64(Float64Builder::new_from_buffer(vect.into(), None)),
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
        assert_eq!(&ChannelData::Utf8(target), data);
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
