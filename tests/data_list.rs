#[path = "common.rs"]
mod common;

use anyhow::Result;
use mdfr::mdfreader::Mdf;
use std::sync::LazyLock;

static BASE_PATH_MDF4: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}MDF4/MDF4.3/Base_Standard/Examples/",
        common::mdfreader_tests_path()
    )
});

#[test]
fn equal_length() -> Result<()> {
    // Equal length
    let file_name = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "DataList/DT_EqualLength/Vector_DT_EqualLen.MF4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("channel1") {
        assert_eq!(data.len(), 254552);
    }
    Ok(())
}

#[test]
fn linked_list() -> Result<()> {
    // Linked list
    let file_name = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "DataList/Vector_DL_Linked_List.MF4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("channel1") {
        assert_eq!(data.len(), 254552);
    }
    Ok(())
}

#[test]
fn empty_list() -> Result<()> {
    // Empty data
    let file_name = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "DataList/EmptyList/ETAS_EmptyDL.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    Ok(())
}

#[test]
fn sd_list() -> Result<()> {
    // SD List
    let file_name = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "DataList/SD_List/Vector_SD_List.MF4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    Ok(())
}

#[test]
fn separate_invalidation_bits() -> Result<()> {
    // Separate invalidation bits
    let file_name = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "DataList/RAC_MDF420_ListData_SeparateInvalidationBits.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    Ok(())
}
