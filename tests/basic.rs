#[path = "common.rs"]
mod common;

use anyhow::Result;
use arrow::array::{Float64Builder, LargeStringBuilder, UInt64Builder};
use mdfr::data_holder::channel_data::ChannelData;
use mdfr::mdfreader::Mdf;
use std::fs;
use std::io;
use std::path::Path;
use std::sync::LazyLock;

static BASE_PATH_MDF4: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}MDF4/MDF4.3/Base_Standard/Examples/",
        common::mdfreader_tests_path()
    )
});

fn parse_info_folder(folder: &String) -> Result<()> {
    let path = Path::new(folder);
    let valid_ext: Vec<String> = vec![
        "mf4".to_string(),
        "MF4".to_string(),
        "DAT".to_string(),
        "dat".to_string(),
        "MDF".to_string(),
        "mdf".to_string(),
    ];
    if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            if let Ok(metadata) = entry.metadata() {
                if metadata.is_file() {
                    if let Ok(ext) = entry
                        .path()
                        .extension()
                        .unwrap()
                        .to_os_string()
                        .into_string()
                        && valid_ext.contains(&ext)
                        && let Some(file_name) = entry.path().to_str()
                    {
                        println!(" Reading file : {}", file_name);
                        let mut mdf = Mdf::new(file_name)?;
                        mdf.load_all_channels_data_in_memory()?;
                    }
                } else if metadata.is_dir()
                    && let Some(path) = entry.path().to_str()
                {
                    let path_str = path.to_owned();
                    match parse_info_folder(&path_str) {
                        Ok(v) => v,
                        Err(e) => {
                            println!("Error parsing the folder {} \n {}", path_str, e)
                        }
                    };
                }
            }
        }
    }
    Ok(())
}

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
    mdf.write("test_files/test.mf4", true)?;
    Ok(())
}

#[test]
fn parse_all_folders4() -> io::Result<()> {
    if !Path::new(BASE_PATH_MDF4.as_str()).is_dir() {
        return Ok(());
    }
    let list_of_paths = [
        "Arrays/Classification".to_string(),
        "Arrays/Simple".to_string(),
        "Attachments/Embedded".to_string(),
        "Attachments/EmbeddedCompressed".to_string(),
        "Attachments/External".to_string(),
        "BusLogging/CAN".to_string(),
        "ChannelInfo/AttachmentRef".to_string(),
        "ChannelInfo/DefaultX".to_string(),
        "ChannelTypes/MasterChannels".to_string(),
        "ChannelTypes/MLSD".to_string(),
        "ChannelTypes/Monotonicity".to_string(),
        "ChannelTypes/Synchronization".to_string(),
        "ChannelTypes/VirtualData".to_string(),
        "ChannelTypes/VLSC".to_string(),
        "ChannelTypes/VLSD".to_string(),
        "CompressedData/DataList".to_string(),
        "CompressedData/MDF430_Algorithms".to_string(),
        "CompressedData/Simple".to_string(),
        "CompressedData/Unsorted".to_string(),
        "Conversion/BitfieldConversion".to_string(),
        "Conversion/LinearConversion".to_string(),
        "Conversion/LookUpConversion".to_string(),
        "Conversion/PartialConversion".to_string(),
        "Conversion/RationalConversion".to_string(),
        "Conversion/StringConversion".to_string(),
        "Conversion/TextConversion".to_string(),
        "DataList".to_string(),
        "DataTypes/ByteArray".to_string(),
        "DataTypes/CANopenTypes".to_string(),
        "DataTypes/Complex".to_string(),
        "DataTypes/IntegerTypes".to_string(),
        "DataTypes/RealTypes".to_string(),
        "DataTypes/StringTypes".to_string(),
        "DynamicData/ChannelList".to_string(),
        "Events/EventSignals".to_string(),
        "Events/Marker".to_string(),
        "Events/Recording".to_string(),
        "Events/Trigger".to_string(),
        "GnssDataStorage".to_string(),
        "Halffloat".to_string(),
        "MetaData/CustomExtensions".to_string(),
        "MetaData/HDO".to_string(),
        "RawSensorLogging/Lidar".to_string(),
        "RawSensorLogging/Video".to_string(),
        "RecordLayout/NotByteAligned".to_string(),
        "RecordLayout/OverlappingSignals".to_string(),
        "RemoteMaster".to_string(),
        "SampleReduction/Simple".to_string(),
        "Simple".to_string(),
        "Union".to_string(),
        "UnsortedData/VLSC".to_string(),
        "UnsortedData/VLSD".to_string(),
        "Variant".to_string(),
    ];
    for path in list_of_paths.iter() {
        println!("reading folder : {}", path);
        parse_info_folder(&format!("{}{}", BASE_PATH_MDF4.as_str(), &path)).unwrap();
    }
    Ok(())
}

#[test]
fn parse_all_folders3() -> io::Result<()> {
    let base_path = format!("{}mdf3/", common::mdfreader_tests_path());
    if !Path::new(&base_path).is_dir() {
        return Ok(());
    }
    parse_info_folder(&base_path).unwrap();
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
            ChannelData::UInt64(UInt64Builder::new_from_buffer(vect.into(), None)),
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
    let expected_string_result = ChannelData::Utf8(expected_string_result);

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
            ChannelData::Float64(Float64Builder::new_from_buffer(vect.into(), None)),
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
