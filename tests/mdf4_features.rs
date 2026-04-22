//! Integration tests for MDF4.3 specialized features:
//! Events, Attachments, Sample Reduction, MetaData, DynamicData, Variant, Union,
//! Source Information, and Channel Hierarchy.
//!
//! These tests exercise the high-level accessor APIs using real MDF4.3 example files.

#[path = "common.rs"]
mod common;

use anyhow::Result;
use mdfr::mdfinfo::MdfInfo;
use mdfr::mdfreader::Mdf;
use std::sync::LazyLock;

static BASE_PATH: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}MDF4/MDF4.3/Base_Standard/Examples/",
        common::mdfreader_tests_path()
    )
});

// ── Events ──────────────────────────────────────────────────────────────────

#[test]
fn test_events_markers() -> Result<()> {
    let file = format!("{}Events/Marker/dSPACE_Bookmarks.mf4", BASE_PATH.as_str());
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;

    // list_events exercises EV block iteration + TX/MD lookup
    let events_list = mdf.mdf_info.list_events();
    assert!(
        !events_list.is_empty(),
        "list_events should return non-empty string"
    );

    // get_event_blocks returns all EV blocks
    let event_blocks = mdf.mdf_info.get_event_blocks();
    assert!(event_blocks.is_some(), "should have event blocks");
    let event_blocks = event_blocks.unwrap();
    assert!(!event_blocks.is_empty(), "should have at least one event");

    for ev in event_blocks.values() {
        // Exercise accessor methods on each event
        let type_str = ev.get_event_type_str();
        assert!(!type_str.is_empty());
        let sync_str = ev.get_sync_type_str();
        assert!(!sync_str.is_empty());
        let cause_str = ev.get_cause_str();
        assert!(!cause_str.is_empty());
        let range_str = ev.get_range_type_str();
        assert!(!range_str.is_empty());
        let _sync_val = ev.get_sync_value();
        // Exercise Display
        let display = format!("{}", ev);
        assert!(display.contains("EV:"));
    }

    // Channels should still be accessible alongside events
    let channels = mdf.get_channel_names_set();
    assert!(!channels.is_empty(), "should have channels");

    Ok(())
}

#[test]
fn test_events_recording() -> Result<()> {
    let file = format!(
        "{}Events/Recording/dSPACE_CaptureBlocks.mf4",
        BASE_PATH.as_str()
    );
    let mut mdf = Mdf::new(&file)?;

    let events_list = mdf.mdf_info.list_events();
    assert!(!events_list.is_empty());

    let event_blocks = mdf.mdf_info.get_event_blocks();
    assert!(event_blocks.is_some());
    let event_blocks = event_blocks.unwrap();
    assert!(!event_blocks.is_empty());

    // Exercise individual event block lookup and sync value computation
    for (&pos, ev) in &event_blocks {
        let single = mdf.mdf_info.get_event_block(pos);
        assert!(single.is_some());
        let _sync_val = ev.get_sync_value();
        let _scope = ev.get_scope_links();
        let _attachments = ev.get_attachment_links();
    }

    Ok(())
}

#[test]
fn test_events_trigger() -> Result<()> {
    let file = format!(
        "{}Events/Trigger/dSPACE_HILAPITrigger.mf4",
        BASE_PATH.as_str()
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;

    let events_list = mdf.mdf_info.list_events();
    assert!(!events_list.is_empty());

    let event_blocks = mdf.mdf_info.get_event_blocks();
    assert!(event_blocks.is_some());
    let event_blocks = event_blocks.unwrap();
    assert!(!event_blocks.is_empty());

    // Check parent event relationships
    let has_parent = event_blocks.values().any(|ev| ev.ev_ev_parent != 0);
    // Trigger files often have parent relationships
    let _ = has_parent;

    Ok(())
}

// ── Attachments ─────────────────────────────────────────────────────────────

#[test]
fn test_attachments_embedded() -> Result<()> {
    let file = format!(
        "{}Attachments/Embedded/Vector_Embedded.MF4",
        BASE_PATH.as_str()
    );
    let mut mdf = Mdf::new(&file)?;

    // list_attachments exercises AT block iteration + TX/MD lookup
    let attachments_list = mdf.mdf_info.list_attachments();
    assert!(
        !attachments_list.is_empty(),
        "list_attachments should return non-empty"
    );

    let at_blocks = mdf.mdf_info.get_attachement_blocks();
    assert!(at_blocks.is_some());
    let at_blocks = at_blocks.unwrap();
    assert!(!at_blocks.is_empty(), "should have at least one attachment");

    for (&pos, at) in &at_blocks {
        // Embedded file: is_embedded should be true
        assert!(
            at.is_embedded(),
            "embedded attachment should have embedded flag set"
        );
        assert!(at.at_original_size > 0, "original size should be > 0");

        // Should be able to retrieve embedded data
        let data = mdf.mdf_info.get_attachment_embedded_data(pos);
        assert!(
            data.is_some(),
            "should have embedded data at position {}",
            pos
        );
        let data = data.unwrap();
        assert!(!data.is_empty(), "embedded data should not be empty");
        assert_eq!(
            data.len() as u64,
            at.at_original_size,
            "data length should match original size"
        );

        // Exercise display
        let display = format!("{}", at);
        assert!(display.contains("embedded"));

        // Exercise individual block lookup
        let single = mdf.mdf_info.get_attachment_block(pos);
        assert!(single.is_some());
    }

    Ok(())
}

#[test]
fn test_attachments_embedded_compressed() -> Result<()> {
    let file = format!(
        "{}Attachments/EmbeddedCompressed/Vector_EmbeddedCompressed.MF4",
        BASE_PATH.as_str()
    );
    let mdf = Mdf::new(&file)?;

    let at_blocks = mdf.mdf_info.get_attachement_blocks();
    assert!(at_blocks.is_some());
    let at_blocks = at_blocks.unwrap();
    assert!(!at_blocks.is_empty());

    for (&pos, at) in &at_blocks {
        assert!(at.is_embedded(), "should be embedded");
        assert!(at.is_compressed(), "should be compressed");

        // Compression type should be known
        let comp_str = at.get_compression_str();
        assert_ne!(
            comp_str, "None",
            "compressed attachment should have compression type"
        );
        assert_ne!(comp_str, "Unknown", "compression type should be known");

        // MD5 checksum may be valid
        let _has_md5 = at.has_md5_checksum();

        // Decompressed data should match original size
        let data = mdf.mdf_info.get_attachment_embedded_data(pos);
        assert!(data.is_some());
        let data = data.unwrap();
        assert_eq!(data.len() as u64, at.at_original_size);
    }

    Ok(())
}

#[test]
fn test_attachments_external() -> Result<()> {
    let file = format!(
        "{}Attachments/External/Vector_External.MF4",
        BASE_PATH.as_str()
    );
    let mdf = Mdf::new(&file)?;

    let at_blocks = mdf.mdf_info.get_attachement_blocks();
    assert!(at_blocks.is_some());
    let at_blocks = at_blocks.unwrap();
    assert!(!at_blocks.is_empty());

    for (&pos, at) in &at_blocks {
        // External: not embedded
        assert!(
            !at.is_embedded(),
            "external attachment should not be embedded"
        );

        // No embedded data for external
        let data = mdf.mdf_info.get_attachment_embedded_data(pos);
        assert!(
            data.is_none(),
            "external attachment should have no embedded data"
        );

        // Filename TX block should exist
        if at.at_tx_filename != 0 {
            let filename = mdf.mdf_info.get_tx(at.at_tx_filename)?;
            assert!(filename.is_some(), "should have filename TX");
            let filename = filename.unwrap();
            assert!(!filename.is_empty(), "filename should not be empty");
        }

        // Display should show external
        let display = format!("{}", at);
        assert!(display.contains("external"));
    }

    Ok(())
}

// ── Sample Reduction ────────────────────────────────────────────────────────

#[test]
fn test_sample_reduction() -> Result<()> {
    let file = format!(
        "{}SampleReduction/Simple/Vector_SampleReduction.mf4",
        BASE_PATH.as_str()
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;

    // list_sample_reductions exercises SR block iteration
    let sr_list = mdf.mdf_info.list_sample_reductions();
    assert!(
        !sr_list.is_empty(),
        "list_sample_reductions should return non-empty"
    );

    // get_sample_reduction_blocks returns structured data
    let sr_blocks = mdf.mdf_info.get_sample_reduction_blocks();
    assert!(sr_blocks.is_some());
    let sr_blocks = sr_blocks.unwrap();
    assert!(!sr_blocks.is_empty(), "should have sample reduction blocks");

    for (_dg_pos, _rec_id, sr_vec) in &sr_blocks {
        for sr in sr_vec {
            assert!(sr.sr_cycle_count > 0, "cycle count should be > 0");
            assert!(sr.sr_interval > 0.0, "interval should be > 0");

            let sync_str = sr.get_sync_type_str();
            assert!(!sync_str.is_empty());

            let _has_inval = sr.has_invalidation_bytes();

            // Exercise Display
            let display = format!("{}", sr);
            assert!(display.contains("SR:"));
        }
    }

    // Channels should be accessible
    let channels = mdf.get_channel_names_set();
    assert!(!channels.is_empty());

    Ok(())
}

// ── MetaData ────────────────────────────────────────────────────────────────

#[test]
fn test_metadata_hdo_comments() -> Result<()> {
    let file = format!(
        "{}MetaData/HDO/RAC_MDF430_HDO_Comments.mf4",
        BASE_PATH.as_str()
    );
    let mut mdf = Mdf::new(&file)?;

    // File history with MD comments
    let fh_list = mdf.mdf_info.list_file_history();
    assert!(
        !fh_list.is_empty(),
        "list_file_history should return non-empty"
    );

    let fh_blocks = mdf.mdf_info.get_file_history_blocks();
    assert!(fh_blocks.is_some());
    let fh_blocks = fh_blocks.unwrap();
    assert!(!fh_blocks.is_empty(), "should have file history blocks");

    // At least one FH should have a comment
    let has_comment = fh_blocks.iter().any(|fh| fh.fh_md_comment != 0);
    assert!(has_comment, "HDO file should have FH blocks with comments");

    // FH block Display
    for fh in &fh_blocks {
        let display = format!("{}", fh);
        assert!(!display.is_empty());
    }

    // Header comments via MdfInfo4 (need to pattern match)
    match &mut mdf.mdf_info {
        MdfInfo::V4(info4) => {
            let header_comments = info4.format_header_comments();
            assert!(
                !header_comments.is_empty(),
                "HDO file should have header comments"
            );
        }
        _ => panic!("expected MDF4"),
    }

    Ok(())
}

#[test]
fn test_metadata_custom_extensions() -> Result<()> {
    let file = format!(
        "{}MetaData/CustomExtensions/Vector_CustomExtensions_CNcomment.mf4",
        BASE_PATH.as_str()
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;

    let channels = mdf.get_channel_names_set();
    assert!(!channels.is_empty());

    // At least one channel should have a description (CN comment)
    let has_desc = channels
        .iter()
        .any(|name| matches!(mdf.get_channel_desc(name), Ok(Some(_))));
    assert!(
        has_desc,
        "CustomExtensions file should have channels with descriptions"
    );

    // Exercise source information listing
    let si_list = mdf.mdf_info.list_source_information();
    let _ = si_list;

    Ok(())
}

// ── DynamicData ─────────────────────────────────────────────────────────────

#[test]
fn test_dynamic_data_channel_list() -> Result<()> {
    let file = format!(
        "{}DynamicData/ChannelList/simple_list.mf4",
        BASE_PATH.as_str()
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;

    let channels = mdf.get_channel_names_set();
    assert!(!channels.is_empty(), "should have channels");

    // Verify at least one channel has data
    let has_data = channels.iter().any(|name| {
        if let Some(data) = mdf.get_channel_data(name) {
            !data.is_empty()
        } else {
            false
        }
    });
    assert!(has_data, "should have at least one channel with data");

    Ok(())
}

// ── Variant ─────────────────────────────────────────────────────────────────

#[test]
fn test_variant_fixed_length() -> Result<()> {
    let file = format!(
        "{}Variant/Etas_cv_storage_with_fixed_length.mf4",
        BASE_PATH.as_str()
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;

    let channels = mdf.get_channel_names_set();
    assert!(!channels.is_empty(), "should have channels");

    for name in &channels {
        if let Some(data) = mdf.get_channel_data(name) {
            let _len = data.len();
        }
    }

    Ok(())
}

#[test]
fn test_variant_vlsd_option() -> Result<()> {
    let file = format!(
        "{}Variant/Vector_V430_Variant_VLSD_Option.mf4",
        BASE_PATH.as_str()
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;

    let channels = mdf.get_channel_names_set();
    assert!(!channels.is_empty(), "should have channels");

    let has_data = channels.iter().any(|name| {
        if let Some(data) = mdf.get_channel_data(name) {
            !data.is_empty()
        } else {
            false
        }
    });
    assert!(has_data, "should have at least one channel with data");

    Ok(())
}

// ── Union ───────────────────────────────────────────────────────────────────

#[test]
fn test_union_fixed_length() -> Result<()> {
    let file = format!(
        "{}Union/Etas_cu_storage_with_fixed_length.mf4",
        BASE_PATH.as_str()
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;

    let channels = mdf.get_channel_names_set();
    assert!(!channels.is_empty(), "should have channels");

    for name in &channels {
        if let Some(data) = mdf.get_channel_data(name) {
            let _len = data.len();
        }
    }

    Ok(())
}

// ── Source Information ──────────────────────────────────────────────────────

#[test]
fn test_source_information() -> Result<()> {
    // Use a bus logging file which typically has rich SI blocks
    let file = format!(
        "{}BusLogging/CAN/Vector_CAN_DataFrame_Sort_Bus.MF4",
        BASE_PATH.as_str()
    );
    let mdf = Mdf::new(&file)?;

    let si_list = mdf.mdf_info.list_source_information();
    assert!(
        !si_list.is_empty(),
        "CAN bus logging file should have source information"
    );

    let si_blocks = mdf.mdf_info.get_source_information_blocks();
    assert!(si_blocks.is_some());
    let si_blocks = si_blocks.unwrap();
    assert!(!si_blocks.is_empty());

    for si in si_blocks.values() {
        let type_str = si.get_type_str();
        assert!(!type_str.is_empty());
        let bus_str = si.get_bus_type_str();
        assert!(!bus_str.is_empty());
        // Exercise Display
        let display = format!("{}", si);
        assert!(display.contains("SI:"));
    }

    Ok(())
}

// ── Channel Hierarchy ───────────────────────────────────────────────────────

#[test]
fn test_channel_hierarchy() -> Result<()> {
    let file = format!(
        "{}ChannelInfo/AttachmentRef/Vector_AttachmentRef.mf4",
        BASE_PATH.as_str()
    );
    let mdf = Mdf::new(&file)?;

    let ch_list = mdf.mdf_info.list_channel_hierarchy();
    let _ = ch_list;

    let ch_blocks = mdf.mdf_info.get_channel_hierarchy_blocks();
    if let Some(ch_blocks) = ch_blocks {
        for &pos in ch_blocks.keys() {
            let single = mdf.mdf_info.get_channel_hierarchy_block(pos);
            assert!(single.is_some());
        }
    }

    Ok(())
}

// ── Event Signals (channel-level event data, not EVBLOCK) ───────────────────

#[test]
fn test_event_signals_channels() -> Result<()> {
    // EventSignals files don't have EVBLOCKs but contain event-related channels
    let file = format!(
        "{}Events/EventSignals/RAC_MDF430_EventSignals_CommonProperties.mf4",
        BASE_PATH.as_str()
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;

    let channels = mdf.get_channel_names_set();
    assert!(!channels.is_empty(), "should have channels");

    let has_data = channels.iter().any(|name| {
        if let Some(data) = mdf.get_channel_data(name) {
            !data.is_empty()
        } else {
            false
        }
    });
    assert!(has_data, "should have channel data");

    // File history should be present
    let fh_list = mdf.mdf_info.list_file_history();
    assert!(!fh_list.is_empty());

    Ok(())
}

// ── MdfInfo4 API Methods ─────────────────────────────────────────────────────

#[test]
fn test_mdf4_summary_and_format() -> Result<()> {
    let file = format!("{}Simple/ETAS_SimpleSorted.mf4", BASE_PATH.as_str());
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;

    // Exercise MdfInfo4::summary() and format_channels() via pattern match
    match &mdf.mdf_info {
        MdfInfo::V4(info4) => {
            let summary = info4.summary();
            assert!(summary.contains("MDF4"));
            assert!(summary.contains("DGs"));
            assert!(summary.contains("channels"));

            // format_channels without data
            let channels_no_data = info4.format_channels(false);
            assert!(!channels_no_data.is_empty());

            // format_channels with data
            let channels_with_data = info4.format_channels(true);
            assert!(!channels_with_data.is_empty());
            // With data loaded, should contain length indicators
            assert!(channels_with_data.contains("["));

            // Display trait for MdfInfo4
            let display = format!("{}", info4);
            assert!(!display.is_empty());
            assert!(display.contains("MDF4"));
        }
        _ => panic!("expected MDF4"),
    }

    // Display trait for MdfInfo
    let info_display = format!("{}", mdf.mdf_info);
    assert!(!info_display.is_empty());

    // Display trait for Mdf
    let mdf_display = format!("{mdf}");
    assert!(!mdf_display.is_empty());

    Ok(())
}

#[test]
fn test_mdf4_channel_api_methods() -> Result<()> {
    let file = format!("{}Simple/ETAS_SimpleSorted.mf4", BASE_PATH.as_str());
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;

    let channels = mdf.get_channel_names_set();
    assert!(!channels.is_empty());

    // Pick a channel name
    let channel_name = channels.iter().next().unwrap().clone();

    // get_channel_master
    let _master = mdf.get_channel_master(&channel_name);

    // get_channel_master_type
    let _master_type = mdf.get_channel_master_type(&channel_name);

    // get_channel_unit / set_channel_unit
    let _unit = mdf.get_channel_unit(&channel_name)?;
    mdf.set_channel_unit(&channel_name, "m/s");
    let unit = mdf.get_channel_unit(&channel_name)?;
    assert_eq!(unit, Some("m/s".to_string()));

    // get_channel_desc / set_channel_desc
    let _desc = mdf.get_channel_desc(&channel_name)?;
    mdf.set_channel_desc(&channel_name, "Test description");
    let desc = mdf.get_channel_desc(&channel_name)?;
    assert_eq!(desc, Some("Test description".to_string()));

    // get_master_channel_names_set
    let master_map = mdf.get_master_channel_names_set();
    assert!(!master_map.is_empty());

    // get_channel_names_cg_set (via MdfInfo)
    let cg_set = mdf.mdf_info.get_channel_names_cg_set(&channel_name);
    assert!(!cg_set.is_empty());

    // is_unfinalized / get_unfin_flags
    let _unfinalized = mdf.is_unfinalized();
    let _flags = mdf.get_unfin_flags();

    // list_sample_reductions (via Mdf)
    let _sr = mdf.list_sample_reductions();

    Ok(())
}

#[test]
fn test_mdf4_rename_and_remove_channel() -> Result<()> {
    let file = format!("{}Simple/ETAS_SimpleSorted.mf4", BASE_PATH.as_str());
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;

    let channels = mdf.get_channel_names_set();
    let original_count = channels.len();
    assert!(original_count >= 2, "need at least 2 channels");

    // Pick a non-master channel to rename
    let channel_name = channels.iter().next().unwrap().clone();

    // rename_channel
    mdf.rename_channel(&channel_name, "renamed_channel");
    let new_channels = mdf.get_channel_names_set();
    assert!(new_channels.contains("renamed_channel"));
    assert!(!new_channels.contains(&channel_name));
    assert_eq!(new_channels.len(), original_count);

    // remove_channel
    mdf.remove_channel("renamed_channel");
    let after_remove = mdf.get_channel_names_set();
    assert!(!after_remove.contains("renamed_channel"));
    assert_eq!(after_remove.len(), original_count - 1);

    Ok(())
}

#[test]
fn test_mdf4_clear_channel_data() -> Result<()> {
    let file = format!("{}Simple/ETAS_SimpleSorted.mf4", BASE_PATH.as_str());
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;

    // Verify data is loaded
    let channels = mdf.get_channel_names_set();
    let channel_name = channels.iter().next().unwrap().clone();
    assert!(mdf.get_channel_data(&channel_name).is_some());

    // clear_all_channel_data_from_memory
    mdf.clear_all_channel_data_from_memory()?;

    // After clearing, channel data should be empty
    if let Some(data) = mdf.get_channel_data(&channel_name) {
        assert!(data.is_empty(), "data should be empty after clearing");
    }

    Ok(())
}

// ── MDF3 API Methods (exercises V3 delegator paths) ──────────────────────────

#[test]
fn test_mdf3_api_methods() -> Result<()> {
    let file = "test_files/test_mdf3.mdf";
    let mut mdf = Mdf::new(file)?;
    mdf.load_all_channels_data_in_memory()?;

    // Version check
    assert_eq!(mdf.get_version(), 310);

    // MDF3 is never unfinalized
    assert!(!mdf.is_unfinalized());
    assert_eq!(mdf.get_unfin_flags(), (0, 0));

    let channels = mdf.get_channel_names_set();
    assert!(!channels.is_empty());

    let channel_name = channels.iter().next().unwrap().clone();

    // get_channel_unit
    let _unit = mdf.get_channel_unit(&channel_name)?;

    // get_channel_desc
    let _desc = mdf.get_channel_desc(&channel_name)?;

    // get_channel_master
    let _master = mdf.get_channel_master(&channel_name);

    // get_channel_master_type
    let _master_type = mdf.get_channel_master_type(&channel_name);

    // get_channel_data
    let _data = mdf.get_channel_data(&channel_name);

    // get_channel_names_cg_set
    let cg_set = mdf.mdf_info.get_channel_names_cg_set(&channel_name);
    assert!(!cg_set.is_empty());

    // get_master_channel_names_set
    let master_map = mdf.get_master_channel_names_set();
    assert!(!master_map.is_empty());

    // V3 block accessor methods return None/empty
    assert!(mdf.mdf_info.get_event_blocks().is_none());
    assert!(mdf.mdf_info.get_attachement_blocks().is_none());
    assert!(mdf.mdf_info.get_file_history_blocks().is_none());
    assert!(mdf.mdf_info.get_source_information_blocks().is_none());
    assert!(mdf.mdf_info.get_sample_reduction_blocks().is_none());
    assert!(mdf.mdf_info.get_channel_hierarchy_blocks().is_none());

    // V3 list methods return empty strings
    let events = mdf.mdf_info.list_events();
    assert!(events.is_empty());
    let attachments = mdf.mdf_info.list_attachments();
    assert!(attachments.is_empty());
    let fh = mdf.mdf_info.list_file_history();
    assert!(fh.is_empty());
    let si = mdf.mdf_info.list_source_information();
    assert!(si.is_empty());
    let sr = mdf.mdf_info.list_sample_reductions();
    assert!(sr.is_empty());
    let ch = mdf.mdf_info.list_channel_hierarchy();
    assert!(ch.is_empty());

    // set_channel_unit (V3)
    mdf.mdf_info.set_channel_unit(&channel_name, "kg");

    // set_channel_desc (V3)
    mdf.mdf_info.set_channel_desc(&channel_name, "test desc");

    // rename_channel (V3)
    let second_channel = channels.iter().nth(1).unwrap().clone();
    mdf.rename_channel(&second_channel, "renamed_v3");
    assert!(mdf.get_channel_names_set().contains("renamed_v3"));

    // remove_channel (V3)
    mdf.remove_channel("renamed_v3");
    assert!(!mdf.get_channel_names_set().contains("renamed_v3"));

    // clear_channel_data_from_memory (V3)
    let remaining = mdf.get_channel_names_set();
    mdf.clear_channel_data_from_memory(remaining)?;

    // Display for MDF3 Mdf
    let display = format!("{mdf}");
    assert!(display.contains("Version"));

    // Display for MdfInfo V3
    let info_display = format!("{}", mdf.mdf_info);
    assert!(!info_display.is_empty());

    Ok(())
}
