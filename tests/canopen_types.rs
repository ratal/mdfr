mod common;

use anyhow::Result;
use mdfr::data_holder::channel_data::ChannelData;
use mdfr::mdfreader::Mdf;
use std::sync::LazyLock;

static BASE: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}MDF4/MDF4.3/Base_Standard/Examples/DataTypes/CANopenTypes/",
        common::mdfreader_tests_path()
    )
});

#[test]
fn canopen_date() -> Result<()> {
    let mut mdf = Mdf::new(&format!("{}Vector_CANOpenDate.mf4", *BASE))?;
    mdf.load_all_channels_data_in_memory()?;
    let names = mdf.get_channel_names_set();
    assert!(names.contains("ms"));
    assert!(names.contains("min"));
    assert!(names.contains("hour"));
    assert!(names.contains("day"));
    assert!(names.contains("month"));
    assert!(names.contains("year"));

    if let Some(ChannelData::UInt16(builder)) = mdf.get_channel_data("ms") {
        assert_eq!(
            builder.values_slice(),
            &[30000, 0, 30000, 0, 30000, 0, 30000, 0, 30000, 0]
        );
    } else {
        panic!("Channel 'ms' missing or wrong type");
    }

    if let Some(ChannelData::UInt8(builder)) = mdf.get_channel_data("min") {
        assert_eq!(
            builder.values_slice(),
            &[19, 20, 20, 21, 21, 22, 22, 23, 23, 24]
        );
    } else {
        panic!("Channel 'min' missing or wrong type");
    }

    if let Some(ChannelData::UInt8(builder)) = mdf.get_channel_data("hour") {
        assert_eq!(
            builder.values_slice(),
            &[11, 13, 15, 17, 19, 21, 23, 1, 2, 11]
        );
    } else {
        panic!("Channel 'hour' missing or wrong type");
    }

    if let Some(ChannelData::UInt8(builder)) = mdf.get_channel_data("day") {
        assert_eq!(
            builder.values_slice(),
            &[15, 16, 17, 18, 19, 20, 21, 22, 23, 16]
        );
    } else {
        panic!("Channel 'day' missing or wrong type");
    }

    // Month on 4 bits (LSB 0..3) per Table 28 of MDF 4.3.0
    if let Some(ChannelData::UInt8(builder)) = mdf.get_channel_data("month") {
        assert_eq!(
            builder.values_slice(),
            &[10, 10, 10, 10, 10, 10, 10, 10, 10, 6]
        );
    } else {
        panic!("Channel 'month' missing or wrong type");
    }

    if let Some(ChannelData::UInt8(builder)) = mdf.get_channel_data("year") {
        assert_eq!(
            builder.values_slice(),
            &[12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
        );
    } else {
        panic!("Channel 'year' missing or wrong type");
    }

    Ok(())
}

#[test]
fn canopen_time() -> Result<()> {
    let mut mdf = Mdf::new(&format!("{}Vector_CANOpenTime.mf4", *BASE))?;
    mdf.load_all_channels_data_in_memory()?;
    let names = mdf.get_channel_names_set();
    assert!(names.contains("ms"));
    assert!(names.contains("day"));

    if let Some(ChannelData::UInt32(builder)) = mdf.get_channel_data("ms") {
        assert_eq!(
            builder.values_slice(),
            &[
                45060000, 45120000, 45180000, 45240000, 45300000, 45360000, 45420000, 45480000,
                45540000, 45600000
            ]
        );
    } else {
        panic!("Channel 'ms' missing or wrong type");
    }

    if let Some(ChannelData::UInt16(builder)) = mdf.get_channel_data("day") {
        assert_eq!(builder.values_slice(), &[10511; 10]);
    } else {
        panic!("Channel 'day' missing or wrong type");
    }

    Ok(())
}

#[test]
fn test_days_since_unix_epoch_canopen_epoch() {
    assert_eq!(mdfr::mdfreader::data_read4::days_since_unix_epoch(1984, 1, 1), 5113);
}

#[test]
fn test_days_since_unix_epoch_unix_epoch() {
    assert_eq!(mdfr::mdfreader::data_read4::days_since_unix_epoch(1970, 1, 1), 0);
}

#[test]
fn test_days_since_unix_epoch_leap_year() {
    assert_eq!(mdfr::mdfreader::data_read4::days_since_unix_epoch(1972, 3, 1), 790);
}

#[test]
fn test_canopen_time_epoch() {
    let bytes = [0u8; 6];
    assert_eq!(mdfr::mdfreader::data_read4::canopen_time_to_unix_ms(&bytes), 441_763_200_000);
}

#[test]
fn test_canopen_time_1984_01_02_midnight() {
    let mut bytes = [0u8; 6];
    bytes[4..6].copy_from_slice(&1u16.to_le_bytes());
    let expected = 441_763_200_000 + 86_400_000;
    assert_eq!(mdfr::mdfreader::data_read4::canopen_time_to_unix_ms(&bytes), expected);
}

#[test]
fn test_canopen_date_1984_01_01_midnight() {
    let mut bytes = [0u8; 7];
    bytes[4] = 1;
    bytes[5] = 1;
    bytes[6] = 0;
    assert_eq!(mdfr::mdfreader::data_read4::canopen_date_to_unix_ms(&bytes), 441_763_200_000);
}

#[test]
fn test_canopen_date_known_value() {
    let mut bytes = [0u8; 7];
    bytes[4] = 1;
    bytes[5] = 1;
    bytes[6] = 16;
    let expected = 10957i64 * 86_400_000;
    assert_eq!(mdfr::mdfreader::data_read4::canopen_date_to_unix_ms(&bytes), expected);
}
