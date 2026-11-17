use crate::common;

use anyhow::Result;
use arrow::array::{Float64Builder, LargeStringBuilder, UInt64Builder};
use mdfr::data_holder::channel_data::ChannelData;
use mdfr::mdfinfo::mdfinfo4::CompressionAlgorithm;
use mdfr::mdfreader::Mdf;
use std::fs;
use std::path::Path;
use std::sync::LazyLock;

static BASE_PATH_MDF4: LazyLock<String> = LazyLock::new(common::base_path_mdf4);

#[test]
fn info_test() -> Result<()> {
    let mut file_name = "test_files/test_basic.mf4";
    println!("reading {}", file_name);
    let mdf = Mdf::new(file_name)?;
    assert_eq!(mdf.get_version(), 410);

    file_name = "test_files/test_mdf3.mdf";
    if Path::new(file_name).exists() {
        let mdf = Mdf::new(file_name)?;
        assert_eq!(mdf.get_version(), 310);
    }

    file_name = "test_files/test_mdf4.mf4";
    if Path::new(file_name).exists() {
        let mdf = Mdf::new(file_name)?;
        assert_eq!(mdf.get_version(), 400);
    }
    Ok(())
}

#[test]
fn basic_test() -> Result<()> {
    let file = "test_files/test_basic.mf4";
    let mut mdf = Mdf::new(file)?;
    mdf.load_all_channels_data_in_memory()?;
    let tmp = tempfile::Builder::new().suffix(".mf4").tempfile()?;
    let tmp_path = tmp.path().to_str().unwrap();
    mdf.write(tmp_path, CompressionAlgorithm::Deflate)?;
    let mut reloaded = Mdf::new(tmp_path)?;
    reloaded.load_all_channels_data_in_memory()?;
    assert_eq!(
        mdf.get_channel_names_set(),
        reloaded.get_channel_names_set()
    );
    Ok(())
}

#[test]
fn record_layout() -> Result<()> {
    if !Path::new(BASE_PATH_MDF4.as_str()).is_dir() {
        return Ok(());
    }
    // Not byte aligned signals
    let file_name = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "RecordLayout/NotByteAligned/Vector_NotByteAligned.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Channel B") {
        let mut vect: Vec<u64> = vec![0; 30];
        let mut counter: u64 = 0;
        vect.iter_mut().for_each(|v| {
            *v += counter;
            counter += 1
        });
        assert_eq!(
            ChannelData::UInt64(UInt64Builder::new_from_buffer(vect.into(), None).finish()),
            data.clone()
        );
    }

    // Overlapping signals
    let file_name = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "RecordLayout/OverlappingSignals/Vector_OverlappingSignals.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    Ok(())
}

#[test]
fn unsorted_data() -> Result<()> {
    if !Path::new(BASE_PATH_MDF4.as_str()).is_dir() {
        return Ok(());
    }
    // VLSD
    let file_name = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "UnsortedData/VLSD/Vector_Unsorted_VLSD.MF4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;

    let file_name = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "UnsortedData/VLSD/RAC_MDF430_Unsorted_VLSD_Compact_Structure.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;

    //VLSC
    let file_name = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "UnsortedData/VLSC/RAC_MDF430_Unsorted_VLSC_Compact_Structure.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;

    // Construct expected data (standard pattern for these tests)
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
    let expected_string_result = ChannelData::Utf8(expected_string_result.finish());

    if let Some(data) = mdf.get_channel_data("Data channel") {
        assert_eq!(expected_string_result, data.clone());
    }
    Ok(())
}

#[test]
fn bus_logging() -> Result<()> {
    if !Path::new(BASE_PATH_MDF4.as_str()).is_dir() {
        return Ok(());
    }
    // sort bus
    let file_name = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "BusLogging/CAN/Vector_CAN_DataFrame_Sort_ID.MF4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) =
        mdf.get_channel_data("CAN_DataFrame.ID CAN_DataFrame_101 CANReplay_7_5 Message")
    {
        let vect: Vec<f64> = vec![101.; 79];
        assert_eq!(
            ChannelData::Float64(Float64Builder::new_from_buffer(vect.into(), None).finish()),
            *data
        );
    }

    let file_name = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "BusLogging/CAN/Vector_CAN_DataFrame_Sort_Bus.MF4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;

    let file_name = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "BusLogging/CAN/Vector_CAN_DataFrame_Sort_ID_SignalDesc.MF4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    Ok(())
}

#[test]
fn from_bytes_matches_file_read() -> Result<()> {
    let path = "test_files/test_basic.mf4";
    let bytes = fs::read(path)?;

    let mut mdf_file = Mdf::new(path)?;
    mdf_file.load_all_channels_data_in_memory()?;

    let mut mdf_bytes = Mdf::from_bytes(&bytes)?;
    mdf_bytes.load_all_channels_data_from_bytes(&bytes)?;

    assert_eq!(mdf_file.get_version(), mdf_bytes.get_version());
    let names = mdf_file.get_channel_names_set();
    assert_eq!(names, mdf_bytes.get_channel_names_set());
    for name in &names {
        assert_eq!(
            mdf_file.get_channel_data(name),
            mdf_bytes.get_channel_data(name),
            "channel {name} mismatch between Mdf::new and Mdf::from_bytes"
        );
    }
    Ok(())
}

#[test]
fn mdf3_from_bytes_matches_file_read() -> Result<()> {
    let path = "test_files/test_mdf3.mdf";
    if !Path::new(path).exists() {
        return Ok(());
    }
    let bytes = fs::read(path)?;

    let mut mdf_file = Mdf::new(path)?;
    mdf_file.load_all_channels_data_in_memory()?;

    let mut mdf_bytes = Mdf::from_bytes(&bytes)?;
    mdf_bytes.load_all_channels_data_from_bytes(&bytes)?;

    assert_eq!(mdf_file.get_version(), mdf_bytes.get_version());
    let names = mdf_file.get_channel_names_set();
    assert_eq!(names, mdf_bytes.get_channel_names_set());
    for name in &names {
        assert_eq!(
            mdf_file.get_channel_data(name),
            mdf_bytes.get_channel_data(name),
            "MDF3 channel {name} mismatch between Mdf::new and Mdf::from_bytes"
        );
    }
    Ok(())
}

#[test]
fn from_bytes_empty_or_truncated_returns_error() {
    assert!(Mdf::from_bytes(&[]).is_err());
    assert!(Mdf::from_bytes(&[0u8; 32]).is_err());
}
