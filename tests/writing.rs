use anyhow::Result;
use arrow::array::{AsArray, Float64Array, PrimitiveBuilder};
use arrow::datatypes::Float32Type;
use mdfr::data_holder::channel_data::ChannelData;
use mdfr::mdfreader::Mdf;
use std::fs;
use std::sync::{Arc, LazyLock};

static MDFREADER_TESTS_PATH: &str = "/home/ratal/workspace/mdfreader/mdfreader/tests/";
static MDFR_PATH: &str = "/home/ratal/workspace/mdfr/";

static BASE_PATH_MDF4: LazyLock<String> = LazyLock::new(|| {
    format!("{}MDF4/MDF4.3/Base_Standard/Examples/", MDFREADER_TESTS_PATH)
});

static BASE_PATH_MDF3: LazyLock<String> =
    LazyLock::new(|| format!("{}mdf3/", MDFREADER_TESTS_PATH));

static BASE_TEST_PATH: LazyLock<String> = LazyLock::new(|| format!("{}test_files", MDFR_PATH));

#[test]
fn writing_mdf4() -> Result<()> {
    let writing_mdf_file = format!("{}/writing_mdf4_test.mf4", BASE_TEST_PATH.as_str());

    // write file with invalid channels
    let file = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        &"Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4"
    );
    let ref_channel = r"NO";
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;

    // without compression
    let mut info2 = mdf.write(&writing_mdf_file, false)?;
    info2.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data(ref_channel) {
        if let Some(data2) = info2.get_channel_data(ref_channel) {
            assert_eq!(*data2, *data);
        } else {
            panic!("Channel not found");
        }
    } else {
        panic!("Channel not found");
    }

    // with compression
    let mut info2 = mdf.write(&writing_mdf_file, true)?;
    info2.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data(ref_channel) {
        if let Some(data2) = info2.get_channel_data(ref_channel) {
            assert_eq!(*data2, *data);
        } else {
            panic!("Channel not found");
        }
    } else {
        panic!("Channel not found");
    }

    // Cleanup temporary file
    fs::remove_file(&writing_mdf_file).ok();
    Ok(())
}

#[test]
fn writing_mdf4_many_channels() -> Result<()> {
    let writing_mdf_file = format!("{}/writing_many_channels_test.mf4", BASE_TEST_PATH.as_str());

    // write file with many channels
    let file = format!("{}{}", BASE_PATH_MDF4.as_str(), &"Simple/test.mf4");
    let ref_channel = r"C90 CG21 in error.mdf";
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;

    // with compression
    let mut info2 = mdf.write(&writing_mdf_file, true)?;
    info2.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data(ref_channel) {
        if let Some(data2) = info2.get_channel_data(ref_channel) {
            assert_eq!(*data2, *data);
        } else {
            panic!("Channel not found");
        }
    } else {
        panic!("Channel not found");
    }

    // without compression
    let mut info2 = mdf.write(&writing_mdf_file, false)?;
    info2.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data(ref_channel) {
        if let Some(data2) = info2.get_channel_data(ref_channel) {
            assert_eq!(*data2, *data);
        } else {
            panic!("Channel not found");
        }
    } else {
        panic!("Channel not found");
    }

    // Cleanup temporary file
    fs::remove_file(&writing_mdf_file).ok();
    Ok(())
}

#[test]
fn mdf3_to_mdf4_conversion() -> Result<()> {
    let writing_mdf_file = format!("{}/mdf3_to_mdf4_test.mf4", BASE_TEST_PATH.as_str());

    //mdf3 conversion
    let file = format!(
        "{}{}",
        BASE_PATH_MDF3.as_str(),
        &"RJ_N16-12-363_BM-15C-0024_228_2_20170116094355_CAN.dat"
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;
    let channel_name3 = r"TEMP_FUEL";
    let mut mdf4 = mdf.write(&writing_mdf_file, true)?;
    mdf4.load_all_channels_data_in_memory()?;
    let mdf3_data = mdf.get_channel_data(channel_name3);
    let mdf4_data = mdf4.get_channel_data(channel_name3);
    assert_eq!(mdf3_data, mdf4_data);

    // Cleanup temporary file
    fs::remove_file(&writing_mdf_file).ok();
    Ok(())
}

#[test]
fn mdf_modifications() -> Result<()> {
    // write file with invalid channels
    let file = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        &"Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4"
    );
    let ref_channel = r"PANS";
    let ref_desc = r"tralala";
    let ref_unit = r"Bar";
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;

    // modify data
    if let Some(data) = mdf.get_channel_data(ref_channel) {
        let mut new_data = PrimitiveBuilder::with_capacity(data.len());
        data.finish_cloned()
            .as_primitive::<Float32Type>()
            .iter()
            .for_each(|v| new_data.append_option(v));
        new_data.values_slice_mut()[0] = 0.0f32;
        mdf.set_channel_data(ref_channel, ChannelData::Float32(new_data).as_ref())?;
        mdf.set_channel_desc(ref_channel, ref_desc);
        mdf.set_channel_unit(ref_channel, ref_unit);
        mdf.set_channel_master_type(ref_channel, 1)?;
    } else {
        panic!("channel not found");
    }
    if let Some(data) = mdf.get_channel_data(ref_channel) {
        let (minimum, _) = data.min_max();
        if let Some(min) = minimum {
            assert!(min < 1000.0f64);
        }
    } else {
        panic!("channel not found");
    }
    match mdf.get_channel_desc(ref_channel) {
        Ok(Some(desc)) => {
            assert_eq!(desc, ref_desc);
        }
        _ => {
            panic!("channel not found");
        }
    }
    match mdf.get_channel_unit(ref_channel) {
        Ok(Some(unit)) => {
            assert_eq!(unit, ref_unit);
        }
        _ => {
            panic!("channel not found");
        }
    }
    assert_eq!(mdf.get_channel_master_type(ref_channel), 1);
    Ok(())
}

#[test]
fn mdf_add_channel() -> Result<()> {
    let file = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        &"Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4"
    );
    let ref_channel = r"PANS";
    let ref_desc = r"tralala";
    let ref_unit = r"Bar";
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;

    // add new channel
    let channel_name = r"Fake_name".to_string();
    let new_channel_name = r"New fake_name".to_string();
    let new_data = Arc::new(Float64Array::try_new(vec![0f64; 3300].into(), None)?);
    let master_channel = mdf.get_channel_master(ref_channel);
    let master_type = Some(0);
    let master_flag = false;
    let unit = Some(ref_unit.to_string());
    let desc = Some(ref_desc.to_string());
    mdf.add_channel(
        channel_name.clone(),
        new_data,
        master_channel,
        master_type,
        master_flag,
        unit,
        desc,
    )?;

    if let Some(data) = mdf.get_channel_data(&channel_name) {
        let (minimum, _) = data.min_max();
        if let Some(min) = minimum {
            assert!(min == 0.0f64);
        }
    } else {
        panic!("channel not found");
    }
    match mdf.get_channel_desc(&channel_name) {
        Ok(Some(desc)) => {
            assert_eq!(desc, ref_desc.to_string());
        }
        _ => {
            panic!("channel not found");
        }
    }
    match mdf.get_channel_unit(&channel_name) {
        Ok(Some(unit)) => {
            assert_eq!(unit, ref_unit);
        }
        _ => {
            panic!("channel not found");
        }
    }
    assert_eq!(mdf.get_channel_master_type(&channel_name), 0);

    //rename
    assert!(mdf.get_channel_data(&channel_name).is_some());
    mdf.rename_channel(&channel_name, &new_channel_name);
    assert!(mdf.get_channel_data(&channel_name).is_none());

    //remove
    assert!(mdf.get_channel_data(&new_channel_name).is_some());
    mdf.remove_channel(&new_channel_name);
    assert!(mdf.get_channel_data(&new_channel_name).is_none());
    Ok(())
}
