use anyhow::Result;
use arrow::array::{
    FixedSizeBinaryBuilder, Float64Builder, Int16Builder, Int32Builder, Int64Builder,
    LargeStringBuilder,
};
use mdfr::data_holder::channel_data::ChannelData;
use mdfr::mdfreader::Mdf;
use std::fs;
use std::sync::LazyLock;

static BASE_PATH_MDF4: LazyLock<String> = LazyLock::new(|| {
    "/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/MDF4.3/Base_Standard/Examples/"
        .to_string()
});

static BASE_TEST_PATH: LazyLock<String> =
    LazyLock::new(|| "/home/ratal/workspace/mdfr/test_files".to_string());

#[test]
fn data_types() -> Result<()> {
    let list_of_paths = [
        "DataTypes/ByteArray/".to_string(),
        "DataTypes/CANopenTypes/".to_string(),
        "DataTypes/IntegerTypes/".to_string(),
        "DataTypes/RealTypes/".to_string(),
        "DataTypes/StringTypes/".to_string(),
        "DataTypes/Complex/".to_string(),
    ];
    let writing_mdf_file = format!("{}/data_types_test.mf4", BASE_TEST_PATH.as_str());

    // Integer testing
    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[2],
        "Vector_IntegerTypes.MF4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    let mut vect: Vec<i64> = vec![100; 201];
    let mut counter: i64 = 0;
    vect.iter_mut().for_each(|v| {
        *v -= counter;
        counter += 1
    });
    if let Some(data) = mdf.get_channel_data("Counter_INT64_BE") {
        assert_eq!(
            ChannelData::Int64(Int64Builder::new_from_buffer(vect.clone().into(), None)),
            data.clone()
        );
    }
    let mut mdf2 = mdf.write(&writing_mdf_file, false)?;
    mdf2.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf2.get_channel_data("Counter_INT64_LE") {
        assert_eq!(
            ChannelData::Int64(Int64Builder::new_from_buffer(vect.into(), None)),
            data.clone()
        );
    }
    let mut vect: Vec<i32> = vec![100; 201];
    let mut counter: i32 = 0;
    vect.iter_mut().for_each(|v| {
        *v -= counter;
        counter += 1
    });
    if let Some(data) = mdf2.get_channel_data("Counter_INT32_LE") {
        assert_eq!(
            ChannelData::Int32(Int32Builder::new_from_buffer(vect.into(), None)),
            data.clone()
        );
    }
    let mut vect: Vec<i16> = vec![100; 201];
    let mut counter: i16 = 0;
    vect.iter_mut().for_each(|v| {
        *v -= counter;
        counter += 1
    });
    if let Some(data) = mdf.get_channel_data("Counter_INT16_BE") {
        assert_eq!(
            ChannelData::Int16(Int16Builder::new_from_buffer(vect.clone().into(), None)),
            data.clone()
        );
    }
    if let Some(data) = mdf2.get_channel_data("Counter_INT16_LE") {
        assert_eq!(
            ChannelData::Int16(Int16Builder::new_from_buffer(vect.into(), None)),
            data.clone()
        );
    }

    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[2],
        "ETAS_IntegerTypes.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;

    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[2],
        "dSPACE_IntegerTypes.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;

    // Cleanup temporary file
    fs::remove_file(&writing_mdf_file).ok();
    Ok(())
}

#[test]
fn real_types() -> Result<()> {
    let list_of_paths = ["DataTypes/RealTypes/".to_string()];

    // Real types
    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_RealTypes.MF4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;

    let file_name = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "Halffloat/halffloat_sinus.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;

    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "dSPACE_RealTypes.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    Ok(())
}

#[test]
fn string_types() -> Result<()> {
    let list_of_paths = ["DataTypes/StringTypes/".to_string()];
    let writing_mdf_file = format!("{}/string_types_test.mf4", BASE_TEST_PATH.as_str());

    // StringTypes testing
    // UTF8
    let mut expected_string_result = LargeStringBuilder::with_capacity(10, 6);
    expected_string_result.append_value("zero");
    expected_string_result.append_value("one");
    expected_string_result.append_value("two");
    expected_string_result.append_value("three");
    expected_string_result.append_value("four");
    expected_string_result.append_value("five");
    expected_string_result.append_value("six");
    expected_string_result.append_value("seven");
    expected_string_result.append_value("eight");
    expected_string_result.append_value("nine");
    let expected_string_result = ChannelData::Utf8(expected_string_result);

    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_FixedLengthStringUTF8.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Time channel") {
        assert_eq!(
            ChannelData::Float64(Float64Builder::new_from_buffer(
                vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.].into(),
                None
            )),
            data.clone()
        );
    }
    if let Some(data) = mdf.get_channel_data("Data channel") {
        assert_eq!(expected_string_result, data.clone());
    }
    let mut mdf2 = mdf.write(&writing_mdf_file, false)?;
    mdf2.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf2.get_channel_data("Data channel") {
        assert_eq!(expected_string_result, data.clone());
    }

    //UTF16
    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_FixedLengthStringUTF16_BE.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Data channel") {
        assert_eq!(expected_string_result, data.clone());
    }
    let mut mdf2 = mdf.write(&writing_mdf_file, false)?;
    mdf2.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf2.get_channel_data("Data channel") {
        assert_eq!(expected_string_result, data.clone());
    }

    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_FixedLengthStringUTF16_LE.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Data channel") {
        assert_eq!(expected_string_result, data.clone());
    }

    //SBC
    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_FixedLengthStringSBC.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Data channel") {
        assert_eq!(expected_string_result, data.clone());
    }
    let mut mdf2 = mdf.write(&writing_mdf_file, false)?;
    mdf2.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf2.get_channel_data("Data channel") {
        assert_eq!(expected_string_result, data.clone());
    }

    // Cleanup temporary file
    fs::remove_file(&writing_mdf_file).ok();
    Ok(())
}

#[test]
fn byte_array_types() -> Result<()> {
    let list_of_paths = ["DataTypes/ByteArray/".to_string()];
    let writing_mdf_file = format!("{}/byte_array_test.mf4", BASE_TEST_PATH.as_str());

    // byteArray testing
    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_ByteArrayFixedLength.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    let mut byte_array = FixedSizeBinaryBuilder::with_capacity(10, 5);
    byte_array.append_value(vec![255, 255, 255, 255, 255])?;
    byte_array.append_value(vec![18, 35, 52, 69, 86])?;
    byte_array.append_value(vec![0, 1, 2, 3, 4])?;
    byte_array.append_value(vec![4, 3, 2, 1, 0])?;
    byte_array.append_value(vec![255, 254, 253, 252, 251])?;
    byte_array.append_value(vec![250, 249, 248, 247, 246])?;
    byte_array.append_value(vec![245, 244, 243, 242, 241])?;
    byte_array.append_value(vec![240, 239, 238, 237, 236])?;
    byte_array.append_value(vec![235, 234, 233, 232, 231])?;
    byte_array.append_value(vec![255, 255, 255, 255, 255])?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Time channel") {
        assert_eq!(
            ChannelData::Float64(Float64Builder::new_from_buffer(
                vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.].into(),
                None
            )),
            data.clone()
        );
    }
    if let Some(data) = mdf.get_channel_data("Data channel") {
        assert_eq!(&ChannelData::FixedSizeByteArray(byte_array), data);
    }
    let mut mdf2 = mdf.write(&writing_mdf_file, false)?;
    mdf2.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf2.get_channel_data("Data channel") {
        let mut byte_array = FixedSizeBinaryBuilder::with_capacity(10, 5);
        byte_array.append_value(vec![255, 255, 255, 255, 255])?;
        byte_array.append_value(vec![18, 35, 52, 69, 86])?;
        byte_array.append_value(vec![0, 1, 2, 3, 4])?;
        byte_array.append_value(vec![4, 3, 2, 1, 0])?;
        byte_array.append_value(vec![255, 254, 253, 252, 251])?;
        byte_array.append_value(vec![250, 249, 248, 247, 246])?;
        byte_array.append_value(vec![245, 244, 243, 242, 241])?;
        byte_array.append_value(vec![240, 239, 238, 237, 236])?;
        byte_array.append_value(vec![235, 234, 233, 232, 231])?;
        byte_array.append_value(vec![255, 255, 255, 255, 255])?;
        assert_eq!(&ChannelData::FixedSizeByteArray(byte_array), data);
    }

    // Cleanup temporary file
    fs::remove_file(&writing_mdf_file).ok();
    Ok(())
}

#[test]
fn complex_types() -> Result<()> {
    //Complex testing
    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        "DataTypes/Complex/",
        "Vector_ComplexNumbers.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    Ok(())
}
