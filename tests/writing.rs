use anyhow::Result;
use arrow::array::{AsArray, Float64Array, PrimitiveBuilder};
use arrow::datatypes::Float32Type;
use mdfr::data_holder::channel_data::ChannelData;
use mdfr::mdfinfo::MdfInfo;
use mdfr::mdfinfo::mdfinfo4::Compo;
use mdfr::mdfreader::Mdf;
use std::fs;
use std::path::Path;
use std::sync::{Arc, LazyLock};

/// SI block metadata: (type, bus_type, flags, name, path)
type SiInfo = (u8, u8, u8, Option<String>, Option<String>);
/// AT block metadata: (filename, mimetype, flags, original_size, embedded_data)
type AtInfo = (Option<String>, Option<String>, u16, u64, Option<Vec<u8>>);

static MDFREADER_TESTS_PATH: &str = "/home/ratal/workspace/mdfreader/mdfreader/tests/";
static MDFR_PATH: &str = "/home/ratal/workspace/mdfr/";

static BASE_PATH_MDF4: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}MDF4/MDF4.3/Base_Standard/Examples/",
        MDFREADER_TESTS_PATH
    )
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

#[test]
fn writing_mdf4_file_history() -> Result<()> {
    let writing_mdf_file = format!("{}/writing_fh_test.mf4", BASE_TEST_PATH.as_str());
    let file = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4"
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;

    // Capture source FH metadata
    let (source_fh_count, source_fh_timestamps) = match &mdf.mdf_info {
        MdfInfo::V4(info4) => {
            assert!(!info4.fh.is_empty(), "Source file should have FH blocks");
            let timestamps: Vec<u64> = info4.fh.iter().map(|fh| fh.fh_time_ns).collect();
            (info4.fh.len(), timestamps)
        }
        _ => panic!("Expected MDF4 file"),
    };

    // Write and re-read from disk
    let _written = mdf.write(&writing_mdf_file, false)?;
    let reread = Mdf::new(&writing_mdf_file)?;

    match &reread.mdf_info {
        MdfInfo::V4(info4) => {
            assert_eq!(
                info4.fh.len(),
                source_fh_count + 1,
                "Written file should have one more FH block than source"
            );
            // Original timestamps preserved in order
            for (i, expected_ts) in source_fh_timestamps.iter().enumerate() {
                assert_eq!(
                    info4.fh[i].fh_time_ns, *expected_ts,
                    "FH[{}] timestamp mismatch",
                    i
                );
            }
            // New FH block has valid timestamp
            let new_fh = &info4.fh[source_fh_count];
            assert!(new_fh.fh_time_ns > 0, "New FH block should have valid timestamp");
        }
        _ => panic!("Expected MDF4 file"),
    }

    fs::remove_file(&writing_mdf_file).ok();
    Ok(())
}

#[test]
fn writing_mdf4_source_information() -> Result<()> {
    let writing_mdf_file = format!("{}/writing_si_test.mf4", BASE_TEST_PATH.as_str());
    let file = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "Events/Marker/dSPACE_Bookmarks.mf4"
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;

    // Capture source SI unique content tuples using public API
    let mut source_si_set: Vec<SiInfo> =
        match &mdf.mdf_info {
            MdfInfo::V4(info4) => {
                let si_blocks = info4.get_source_information_blocks();
                assert!(
                    !si_blocks.is_empty(),
                    "Source file should have SI blocks"
                );
                si_blocks
                    .values()
                    .map(|si| {
                        let name = si.get_si_source_name(&info4.sharable).ok().flatten();
                        let path = si.get_si_path_name(&info4.sharable).ok().flatten();
                        (si.si_type, si.si_bus_type, si.si_flags, name, path)
                    })
                    .collect()
            }
            _ => panic!("Expected MDF4 file"),
        };
    source_si_set.sort();
    source_si_set.dedup();

    // Write and re-read from disk
    let _written = mdf.write(&writing_mdf_file, false)?;
    let reread = Mdf::new(&writing_mdf_file)?;

    match &reread.mdf_info {
        MdfInfo::V4(info4) => {
            let si_blocks = info4.get_source_information_blocks();
            assert!(
                !si_blocks.is_empty(),
                "Written file should have SI blocks"
            );
            let mut reread_si_set: Vec<SiInfo> = si_blocks
                .values()
                .map(|si| {
                    let name = si.get_si_source_name(&info4.sharable).ok().flatten();
                    let path = si.get_si_path_name(&info4.sharable).ok().flatten();
                    (si.si_type, si.si_bus_type, si.si_flags, name, path)
                })
                .collect();
            reread_si_set.sort();
            reread_si_set.dedup();
            // Verify all unique SI content is preserved
            assert_eq!(
                source_si_set, reread_si_set,
                "SI block unique content mismatch"
            );
        }
        _ => panic!("Expected MDF4 file"),
    }

    fs::remove_file(&writing_mdf_file).ok();
    Ok(())
}

#[test]
fn writing_mdf4_events() -> Result<()> {
    let writing_mdf_file = format!("{}/writing_events_test.mf4", BASE_TEST_PATH.as_str());
    let file = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "Events/Marker/dSPACE_Bookmarks.mf4"
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;

    // Capture source event metadata
    let (mut source_event_types, mut source_event_names) = match &mdf.mdf_info {
        MdfInfo::V4(info4) => {
            assert!(!info4.ev.is_empty(), "Source file should contain events");
            let types: Vec<(u8, u8, u8)> = info4
                .ev
                .values()
                .map(|ev| (ev.ev_type, ev.ev_sync_type, ev.ev_cause))
                .collect();
            let names: Vec<Option<String>> = info4
                .ev
                .values()
                .map(|ev| info4.sharable.get_tx(ev.ev_tx_name).ok().flatten())
                .collect();
            (types, names)
        }
        _ => panic!("Expected MDF4 file"),
    };
    let source_event_count = source_event_types.len();
    source_event_types.sort();
    source_event_names.sort();

    // Write and re-read from disk
    let _written = mdf.write(&writing_mdf_file, false)?;
    let mut reread = Mdf::new(&writing_mdf_file)?;
    reread.load_all_channels_data_in_memory()?;

    match &reread.mdf_info {
        MdfInfo::V4(info4) => {
            assert_eq!(info4.ev.len(), source_event_count, "Event count mismatch");
            let mut reread_types: Vec<(u8, u8, u8)> = info4
                .ev
                .values()
                .map(|ev| (ev.ev_type, ev.ev_sync_type, ev.ev_cause))
                .collect();
            reread_types.sort();
            assert_eq!(source_event_types, reread_types, "Event types mismatch");

            let mut reread_names: Vec<Option<String>> = info4
                .ev
                .values()
                .map(|ev| info4.sharable.get_tx(ev.ev_tx_name).ok().flatten())
                .collect();
            reread_names.sort();
            assert_eq!(source_event_names, reread_names, "Event names mismatch");
        }
        _ => panic!("Expected MDF4 file"),
    }

    // Verify channel data also preserved
    let channel_names = mdf.get_channel_names_set();
    for name in &channel_names {
        if let Some(src_data) = mdf.get_channel_data(name)
            && let Some(reread_data) = reread.get_channel_data(name)
        {
            assert_eq!(*src_data, *reread_data, "Data mismatch for channel {}", name);
        }
    }

    fs::remove_file(&writing_mdf_file).ok();
    Ok(())
}

#[test]
fn writing_mdf4_attachments() -> Result<()> {
    let writing_mdf_file = format!("{}/writing_attachments_test.mf4", BASE_TEST_PATH.as_str());
    let file = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "Attachments/Embedded/Vector_Embedded.MF4"
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;

    // Capture source attachment metadata as sorted tuples (sort by filename)
    let mut source_at_info: Vec<AtInfo> =
        match &mdf.mdf_info {
            MdfInfo::V4(info4) => {
                assert!(
                    !info4.at.is_empty(),
                    "Source file should have attachments"
                );
                info4
                    .at
                    .values()
                    .map(|(at, data)| {
                        let filename = info4.sharable.get_tx(at.at_tx_filename).ok().flatten();
                        let mimetype = info4.sharable.get_tx(at.at_tx_mimetype).ok().flatten();
                        (filename, mimetype, at.at_flags, at.at_original_size, data.clone())
                    })
                    .collect()
            }
            _ => panic!("Expected MDF4 file"),
        };
    source_at_info.sort_by(|a, b| a.0.cmp(&b.0));
    let source_at_count = source_at_info.len();

    // Write and re-read from disk
    let _written = mdf.write(&writing_mdf_file, false)?;
    let mut reread = Mdf::new(&writing_mdf_file)?;
    reread.load_all_channels_data_in_memory()?;

    match &reread.mdf_info {
        MdfInfo::V4(info4) => {
            assert_eq!(info4.at.len(), source_at_count, "Attachment count mismatch");
            let mut reread_at_info: Vec<AtInfo> =
                info4
                    .at
                    .values()
                    .map(|(at, data)| {
                        let filename = info4.sharable.get_tx(at.at_tx_filename).ok().flatten();
                        let mimetype = info4.sharable.get_tx(at.at_tx_mimetype).ok().flatten();
                        (filename, mimetype, at.at_flags, at.at_original_size, data.clone())
                    })
                    .collect();
            reread_at_info.sort_by(|a, b| a.0.cmp(&b.0));

            for (i, (src, rr)) in source_at_info.iter().zip(reread_at_info.iter()).enumerate() {
                assert_eq!(src.0, rr.0, "Attachment {} filename mismatch", i);
                assert_eq!(src.1, rr.1, "Attachment {} mimetype mismatch", i);
                assert_eq!(src.2, rr.2, "Attachment {} flags mismatch", i);
                assert_eq!(src.3, rr.3, "Attachment {} original_size mismatch", i);
                assert_eq!(src.4, rr.4, "Attachment {} embedded data mismatch", i);
            }
        }
        _ => panic!("Expected MDF4 file"),
    }

    // Verify channel data preserved
    let channel_names = mdf.get_channel_names_set();
    for name in &channel_names {
        if let Some(src_data) = mdf.get_channel_data(name)
            && let Some(reread_data) = reread.get_channel_data(name)
        {
            assert_eq!(*src_data, *reread_data, "Data mismatch for channel {}", name);
        }
    }

    fs::remove_file(&writing_mdf_file).ok();
    Ok(())
}

#[test]
fn writing_mdf4_vlsd() -> Result<()> {
    let writing_mdf_file = format!("{}/writing_vlsd_test.mf4", BASE_TEST_PATH.as_str());

    let vlsd_channel = "Data channel";
    let time_channel = "Time channel";

    // Test UTF-8 VLSD
    let file = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "ChannelTypes/VLSD/Vector_VLSD_String_UTF8.mf4"
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;
    let src_time = mdf
        .get_channel_data(time_channel)
        .expect("Source should have Time channel")
        .clone();

    // Write without compression — verify file is valid and non-VLSD data preserved
    let _written = mdf.write(&writing_mdf_file, false)?;
    let mut reread = Mdf::new(&writing_mdf_file)?;
    reread.load_all_channels_data_in_memory()?;
    let reread_names = reread.get_channel_names_set();
    assert!(
        reread_names.contains(vlsd_channel),
        "VLSD channel should be detected in re-read file (uncompressed)"
    );
    let reread_time = reread
        .get_channel_data(time_channel)
        .expect("Time channel should have data after re-read");
    assert_eq!(src_time, *reread_time, "Time data mismatch (no compression)");

    // Write with compression — verify file is valid
    let _written = mdf.write(&writing_mdf_file, true)?;
    let mut reread = Mdf::new(&writing_mdf_file)?;
    reread.load_all_channels_data_in_memory()?;
    let reread_names = reread.get_channel_names_set();
    assert!(
        reread_names.contains(vlsd_channel),
        "VLSD channel should be detected in re-read file (compressed)"
    );
    let reread_time = reread
        .get_channel_data(time_channel)
        .expect("Time channel should have data after compressed re-read");
    assert_eq!(src_time, *reread_time, "Time data mismatch (with compression)");

    // Test SBC encoding
    let file = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "ChannelTypes/VLSD/Vector_VLSD_String_SBC.mf4"
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;
    let src_time = mdf
        .get_channel_data(time_channel)
        .expect("SBC source should have Time channel")
        .clone();

    let _written = mdf.write(&writing_mdf_file, true)?;
    let mut reread = Mdf::new(&writing_mdf_file)?;
    reread.load_all_channels_data_in_memory()?;
    assert!(
        reread.get_channel_names_set().contains(vlsd_channel),
        "VLSD SBC channel should be detected in re-read file"
    );
    let reread_time = reread
        .get_channel_data(time_channel)
        .expect("SBC Time channel should have data after re-read");
    assert_eq!(src_time, *reread_time, "SBC Time data mismatch");

    fs::remove_file(&writing_mdf_file).ok();
    Ok(())
}

#[test]
fn writing_mdf4_arrays() -> Result<()> {
    let writing_mdf_file = format!("{}/writing_arrays_test.mf4", BASE_TEST_PATH.as_str());
    let file = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "Arrays/Simple/Vector_ArrayWithFixedAxes.MF4"
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;

    // Verify source has CA blocks
    let source_has_ca = match &mdf.mdf_info {
        MdfInfo::V4(info4) => info4.dg.values().any(|dg| {
            dg.cg.values().any(|cg| {
                cg.cn.values().any(|cn| {
                    cn.composition
                        .as_ref()
                        .is_some_and(|c| matches!(c.block, mdfr::mdfinfo::mdfinfo4::Compo::CA(_)))
                })
            })
        }),
        _ => false,
    };
    assert!(source_has_ca, "Source file should have CA blocks");

    // Capture source channel names
    let source_names = mdf.get_channel_names_set();
    assert!(!source_names.is_empty(), "Source file should have channels");

    // Write and verify channel data is preserved via in-memory return
    let written = mdf.write(&writing_mdf_file, false)?;
    let written_names = written.get_channel_names_set();
    assert_eq!(source_names, written_names, "Channel names should be preserved");

    for name in &source_names {
        if let Some(src_data) = mdf.get_channel_data(name)
            && let Some(written_data) = written.get_channel_data(name)
        {
            assert_eq!(
                *src_data, *written_data,
                "Array data mismatch for channel {}",
                name
            );
        }
    }

    fs::remove_file(&writing_mdf_file).ok();
    Ok(())
}

#[test]
fn writing_mdf4_channel_hierarchy() -> Result<()> {
    // CH blocks are optional in MDF4 and none of the current test files contain them.
    // This test verifies that files without CH blocks still write correctly (hd_ch_first = 0)
    // and that the CH writing code path doesn't break anything.
    let writing_mdf_file = format!("{}/writing_ch_test.mf4", BASE_TEST_PATH.as_str());
    let file = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4"
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;

    // Verify source has no CH blocks (expected with current test files)
    let source_ch_count = match &mdf.mdf_info {
        MdfInfo::V4(info4) => info4.ch.len(),
        _ => panic!("Expected MDF4 file"),
    };

    // Write and re-read
    let _written = mdf.write(&writing_mdf_file, false)?;
    let reread = Mdf::new(&writing_mdf_file)?;

    match &reread.mdf_info {
        MdfInfo::V4(info4) => {
            // CH block count should match source (both should be 0 with current test files)
            assert_eq!(
                info4.ch.len(),
                source_ch_count,
                "CH block count should be preserved"
            );
        }
        _ => panic!("Expected MDF4 file"),
    }

    fs::remove_file(&writing_mdf_file).ok();
    Ok(())
}

#[test]
fn writing_mdf4_composition_ds_cl() -> Result<()> {
    // DS/CL (Data Stream + Channel List) roundtrip test
    // DS/CL compositions describe VLSD blob layouts. After the reader decodes the
    // blob into typed child channels (x.a, x.b), the parent structure channel ("x")
    // has zero bit_count and the auxiliary VLSD channel has empty data. These are
    // metadata-only channels that carry no data, so the writer correctly skips them.
    // The decoded child channel data is preserved as independent channels.
    let writing_mdf_file = format!("{}/writing_ds_cl_test.mf4", BASE_TEST_PATH.as_str());
    let file = "/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/MDF4.3/Base_Standard/Examples/DynamicData/ChannelList/simple_list.mf4";
    if !Path::new(file).exists() {
        return Ok(());
    }

    let mut mdf = Mdf::new(file)?;
    mdf.load_all_channels_data_in_memory()?;

    // Verify source has DS/CL composition
    let has_ds_or_cl = match &mdf.mdf_info {
        MdfInfo::V4(info4) => info4.dg.values().any(|dg| {
            dg.cg.values().any(|cg| {
                cg.cn.values().any(|cn| {
                    cn.composition.as_ref().is_some_and(|c| {
                        matches!(c.block, Compo::DS(_) | Compo::CL(_))
                    })
                })
            })
        }),
        _ => false,
    };
    assert!(has_ds_or_cl, "Source should have DS or CL composition");

    // Write and verify via in-memory return
    let mut written = mdf.write(&writing_mdf_file, false)?;
    written.load_all_channels_data_in_memory()?;

    // Verify decoded child channel data is preserved through write roundtrip
    for channel_name in &["time", "x.a", "x.b", "size"] {
        if let Some(src) = mdf.get_channel_data(channel_name)
            && let Some(wr) = written.get_channel_data(channel_name)
        {
            assert_eq!(
                src.len(),
                wr.len(),
                "Length mismatch for channel {}",
                channel_name
            );
            assert_eq!(*src, *wr, "Data mismatch for channel {}", channel_name);
        } else {
            panic!(
                "Channel {} missing: source={}, written={}",
                channel_name,
                mdf.get_channel_data(channel_name).is_some(),
                written.get_channel_data(channel_name).is_some(),
            );
        }
    }

    fs::remove_file(&writing_mdf_file).ok();
    Ok(())
}

#[test]
fn writing_mdf4_composition_cv() -> Result<()> {
    // CV (Channel Variant) composition test
    let writing_mdf_file = format!("{}/writing_cv_test.mf4", BASE_TEST_PATH.as_str());
    let file = "/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/MDF4.3/Base_Standard/Examples/Variant/Etas_cv_storage_with_fixed_length.mf4";
    if !Path::new(file).exists() {
        return Ok(());
    }

    let mut mdf = Mdf::new(file)?;
    mdf.load_all_channels_data_in_memory()?;

    // Verify source has CV composition
    let source_cv_info = match &mdf.mdf_info {
        MdfInfo::V4(info4) => {
            let mut found = false;
            let mut option_count = 0u32;
            for dg in info4.dg.values() {
                for cg in dg.cg.values() {
                    for cn in cg.cn.values() {
                        if let Some(c) = &cn.composition
                            && let Compo::CV(cv) = &c.block
                        {
                            found = true;
                            option_count = cv.cv_option_count;
                        }
                    }
                }
            }
            (found, option_count)
        }
        _ => (false, 0),
    };
    assert!(source_cv_info.0, "Source should have CV composition");

    // Write and verify via in-memory return
    let written = mdf.write(&writing_mdf_file, false)?;

    // Verify CV composition preserved with same option count
    match &written.mdf_info {
        MdfInfo::V4(info4) => {
            let mut found = false;
            for dg in info4.dg.values() {
                for cg in dg.cg.values() {
                    for cn in cg.cn.values() {
                        if let Some(c) = &cn.composition
                            && let Compo::CV(cv) = &c.block
                        {
                            found = true;
                            assert_eq!(
                                cv.cv_option_count, source_cv_info.1,
                                "CV option count should be preserved"
                            );
                        }
                    }
                }
            }
            assert!(found, "Written file should preserve CV composition");
        }
        _ => panic!("Expected MDF4"),
    }

    // Verify time channel data
    if let Some(src) = mdf.get_channel_data("time")
        && let Some(wr) = written.get_channel_data("time")
    {
        assert_eq!(*src, *wr, "Time data mismatch");
    }

    fs::remove_file(&writing_mdf_file).ok();
    Ok(())
}

#[test]
fn writing_mdf4_composition_cu() -> Result<()> {
    // CU (Channel Union) composition test
    let writing_mdf_file = format!("{}/writing_cu_test.mf4", BASE_TEST_PATH.as_str());
    let file = "/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/MDF4.3/Base_Standard/Examples/Union/Etas_cu_storage_with_fixed_length.mf4";
    if !Path::new(file).exists() {
        return Ok(());
    }

    let mut mdf = Mdf::new(file)?;
    mdf.load_all_channels_data_in_memory()?;

    // Verify source has CU composition
    let source_cu_info = match &mdf.mdf_info {
        MdfInfo::V4(info4) => {
            let mut found = false;
            let mut member_count = 0u32;
            for dg in info4.dg.values() {
                for cg in dg.cg.values() {
                    for cn in cg.cn.values() {
                        if let Some(c) = &cn.composition
                            && let Compo::CU(cu) = &c.block
                        {
                            found = true;
                            member_count = cu.cu_member_count;
                        }
                    }
                }
            }
            (found, member_count)
        }
        _ => (false, 0),
    };
    assert!(source_cu_info.0, "Source should have CU composition");

    // Write and verify via in-memory return
    let written = mdf.write(&writing_mdf_file, false)?;

    // Verify CU composition preserved with same member count
    match &written.mdf_info {
        MdfInfo::V4(info4) => {
            let mut found = false;
            for dg in info4.dg.values() {
                for cg in dg.cg.values() {
                    for cn in cg.cn.values() {
                        if let Some(c) = &cn.composition
                            && let Compo::CU(cu) = &c.block
                        {
                            found = true;
                            assert_eq!(
                                cu.cu_member_count, source_cu_info.1,
                                "CU member count should be preserved"
                            );
                        }
                    }
                }
            }
            assert!(found, "Written file should preserve CU composition");
        }
        _ => panic!("Expected MDF4"),
    }

    // Verify time channel data
    if let Some(src) = mdf.get_channel_data("time")
        && let Some(wr) = written.get_channel_data("time")
    {
        assert_eq!(*src, *wr, "Time data mismatch");
    }

    fs::remove_file(&writing_mdf_file).ok();
    Ok(())
}
