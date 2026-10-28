//! End-to-end round-trip tests: `mdfr_core::logger::Logger` → `.mf4` file → `mdfr::mdfreader::Mdf`.
use anyhow::Result;
use arrow::array::Array;
use mdfr::data_holder::channel_data::ChannelData;
use mdfr::mdfreader::Mdf;
use mdfr_core::blocks::cn::CnDataType;
use mdfr_core::compress::CompressionAlgorithm;
use mdfr_core::logger::{BusPreset, ConversionDef, DataLayout, Logger, LoggerBuilder, SourceDef};
use mdfr_core::time::ManualClock;

#[test]
fn roundtrip_sorted_uncompressed_and_compressed() -> Result<()> {
    for (layout, algo) in [
        (DataLayout::Sorted, CompressionAlgorithm::NoCompression),
        (DataLayout::Sorted, CompressionAlgorithm::DeflateTranspose),
        (DataLayout::Sorted, CompressionAlgorithm::Lz4Transpose),
        (DataLayout::Unsorted, CompressionAlgorithm::NoCompression),
        (DataLayout::Unsorted, CompressionAlgorithm::DeflateTranspose),
        (DataLayout::Unsorted, CompressionAlgorithm::Lz4Transpose),
    ] {
        let mut b = LoggerBuilder::new(layout, algo);
        let g0 = b.add_group("Engine");
        b.set_group_flush_threshold(g0, 256)?;
        b.set_group_source(
            g0,
            SourceDef {
                name: "ECU_Engine".into(),
                path: Some("PT/ECU_Engine".into()),
                si_type: 1, // ECU
                si_bus_type: 2,
                si_flags: 0,
            },
        )?;
        let ch_rpm = b.add_channel(g0, "RPM", CnDataType::UIntLE, 16)?;
        b.set_channel_unit(ch_rpm, "rpm")?;
        b.set_channel_conversion(ch_rpm, ConversionDef::Linear { p1: 0.0, p2: 0.5 })?;

        let ch_temp = b.add_channel(g0, "CoolantTemp", CnDataType::FloatLE, 32)?;
        b.set_channel_unit(ch_temp, "degC")?;

        let ch_gear = b.add_channel(g0, "Gear", CnDataType::UIntLE, 8)?;
        b.set_channel_conversion(
            ch_gear,
            ConversionDef::ValueToText {
                entries: vec![
                    (0.0, "Neutral".into()),
                    (1.0, "First".into()),
                    (2.0, "Second".into()),
                ],
                default_text: "Other".into(),
            },
        )?;

        let ch_map = b.add_array_channel(g0, "TorqueMap", CnDataType::UIntLE, 16, vec![3])?;
        let ch_msg = b.add_vlsd_channel(g0, "DiagMsg", CnDataType::StringUtf8)?;

        // Add a CAN-FD bus logging group as a second group
        let can = BusPreset::can(&mut b, "CAN1")?;
        b.set_group_flush_threshold(can.group, 256)?;

        let schema = b.build()?;
        let tmp = tempfile::Builder::new().suffix(".mf4").tempfile()?;
        let file = tmp.reopen()?;
        let clock = ManualClock::new(0.001, 10); // 10 ms per step

        let mut logger = Logger::start(schema, file, clock)?;

        for i in 0..50u64 {
            let map_bytes: [u8; 6] = [
                (i as u16).to_le_bytes()[0],
                (i as u16).to_le_bytes()[1],
                ((i + 1) as u16).to_le_bytes()[0],
                ((i + 1) as u16).to_le_bytes()[1],
                ((i + 2) as u16).to_le_bytes()[0],
                ((i + 2) as u16).to_le_bytes()[1],
            ];
            let msg = format!("msg_{i}");
            logger
                .begin_record(g0)?
                .set_u64(ch_rpm, 2000 + i * 20)?
                .set_f64(ch_temp, 85.5 + i as f64 * 0.25)?
                .set_u64(ch_gear, i % 3)?
                .set_bytes(ch_map, &map_bytes)?
                .set_bytes(ch_msg, msg.as_bytes())?
                .commit()?;

            let payload = [i as u8; 8];
            logger
                .begin_record(can.group)?
                .set_u64(can.bus_channel, 1)?
                .set_u64(can.can_id, 0x123)?
                .set_u64(can.dir, 0)?
                .set_u64(can.dlc, 8)?
                .set_u64(can.data_length, 8)?
                .set_bytes(can.data_bytes, &payload)?
                .commit()?;
        }

        logger.close()?;

        // Read back with `mdfr::mdfreader::Mdf`
        let path_str = tmp.path().to_str().unwrap();
        let mut mdf = Mdf::new(path_str)?;
        assert!(!mdf.mdf_info.is_unfinalized(), "file should be finalized");
        mdf.load_all_channels_data_in_memory()?;

        // Check converted RPM (linear P2 = 0.5 -> 2000 * 0.5 = 1000.0)
        let rpm = mdf.get_channel_data("RPM").expect("RPM channel missing");
        match rpm {
            ChannelData::Float64(arr) => {
                let s = arr.values_slice();
                assert_eq!(s.len(), 50);
                assert!(
                    (s[0] - 1000.0).abs() < 1e-6,
                    "layout={layout:?} algo={algo:?}"
                );
                assert!((s[49] - (2000.0 + 49.0 * 20.0) * 0.5).abs() < 1e-6);
            }
            other => panic!("unexpected RPM ChannelData variant: {other:?}"),
        }

        // Check VLSD string channel
        let diag = mdf
            .get_channel_data("DiagMsg")
            .expect("DiagMsg channel missing");
        match diag {
            ChannelData::Utf8(arr) => {
                let built = arr.finish_cloned();
                assert_eq!(built.len(), 50);
                assert_eq!(built.value(0), "msg_0");
                assert_eq!(built.value(49), "msg_49");
            }
            other => panic!("unexpected DiagMsg variant: {other:?}"),
        }
    }
    Ok(())
}

#[test]
fn power_loss_unfinalized_recovery_and_in_place_finalize() -> Result<()> {
    for layout in [DataLayout::Sorted, DataLayout::Unsorted] {
        let mut b = LoggerBuilder::new(layout, CompressionAlgorithm::Lz4Transpose);
        b.checkpoint_every_n_blocks(1);
        let g0 = b.add_group("Sensors");
        b.set_group_flush_threshold(g0, 128)?;
        let ch_val = b.add_channel(g0, "Pressure", CnDataType::UIntLE, 32)?;
        let ch_str = b.add_vlsd_channel(g0, "Tag", CnDataType::StringUtf8)?;
        let schema = b.build()?;

        let tmp = tempfile::Builder::new().suffix(".mf4").tempfile()?;
        let file = tmp.reopen()?;
        let clock = ManualClock::new(0.01, 1);

        let mut logger = Logger::start(schema, file, clock)?;
        for i in 0..30u64 {
            logger
                .begin_record(g0)?
                .set_u64(ch_val, 100 + i)?
                .set_bytes(ch_str, format!("tag_{i}").as_bytes())?
                .commit()?;
        }
        //Checkpoint first 30 records, then zero out CG cycle count on disk to simulate an unpatched crash
        logger.checkpoint()?;
        // Drop logger WITHOUT calling close() — file remains "UnFinMF " with unfin_flags != 0!
        drop(logger);

        let path_str = tmp.path().to_str().unwrap();

        // 1. Read-only recovery in `Mdf::new`
        let mut mdf = Mdf::new(path_str)?;
        assert!(
            mdf.mdf_info.is_unfinalized(),
            "dropped logger file must be detected as unfinalized"
        );
        mdf.load_all_channels_data_in_memory()?;
        if let Some(ChannelData::UInt32(arr)) = mdf.get_channel_data("Pressure") {
            assert_eq!(arr.values_slice().len(), 30);
            assert_eq!(arr.values_slice()[29], 129);
        } else {
            panic!("Pressure channel missing after crash recovery");
        }

        // 2. In-place finalization on disk (no file copy)
        mdfr::mdfinfo::mdfinfo4::finalize::finalize_file_in_place(path_str)?;
        let mut mdf2 = Mdf::new(path_str)?;
        assert!(
            !mdf2.mdf_info.is_unfinalized(),
            "file must be finalized in place after finalize_file_in_place"
        );
        mdf2.load_all_channels_data_in_memory()?;
        if let Some(ChannelData::Utf8(arr)) = mdf2.get_channel_data("Tag") {
            let built = arr.finish_cloned();
            assert_eq!(built.len(), 30);
            assert_eq!(built.value(29), "tag_29");
        } else {
            panic!("Tag VLSD channel missing after in-place finalization");
        }
    }
    Ok(())
}

#[test]
fn file_rotation_increments_sequence_index() -> Result<()> {
    let mut b = LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::DeflateTranspose);
    let g0 = b.add_group("IMU");
    let ch_acc = b.add_channel(g0, "AccZ", CnDataType::FloatLE, 32)?;
    let schema = b.build()?;

    let tmp1 = tempfile::Builder::new().suffix(".mf4").tempfile()?;
    let tmp2 = tempfile::Builder::new().suffix(".mf4").tempfile()?;
    let clock = ManualClock::new(0.001, 1);

    let mut logger = Logger::start(schema, tmp1.reopen()?, clock)?;
    for i in 0..10 {
        logger
            .begin_record(g0)?
            .set_f64(ch_acc, i as f64 * 0.5)?
            .commit()?;
    }
    let _old_file = logger.rotate(tmp2.reopen()?, None)?;
    for i in 10..20 {
        logger
            .begin_record(g0)?
            .set_f64(ch_acc, i as f64 * 0.5)?
            .commit()?;
    }
    logger.close()?;

    let mut m1 = Mdf::new(tmp1.path().to_str().unwrap())?;
    m1.load_all_channels_data_in_memory()?;
    let mut m2 = Mdf::new(tmp2.path().to_str().unwrap())?;
    m2.load_all_channels_data_in_memory()?;

    assert!(!m1.mdf_info.is_unfinalized());
    assert!(!m2.mdf_info.is_unfinalized());
    assert_eq!(m1.get_recorder_file_index(), Some(0));
    assert_eq!(m2.get_recorder_file_index(), Some(1));
    Ok(())
}

#[test]
fn c_api_roundtrip_and_spsc_drain() -> Result<()> {
    use mdfr::c_api::*;
    use std::ffi::CString;

    let tmp1 = tempfile::Builder::new().suffix(".mf4").tempfile()?;
    let tmp2 = tempfile::Builder::new().suffix(".mf4").tempfile()?;
    let path1_c = CString::new(tmp1.path().to_str().unwrap())?;
    let path2_c = CString::new(tmp2.path().to_str().unwrap())?;

    unsafe {
        let builder = mdfr_logger_builder_new(0, 1); // Sorted, DeflateTranspose
        assert!(!builder.is_null());

        let g_name = CString::new("Telemetry")?;
        let mut g0 = 0u16;
        assert_eq!(mdfr_logger_add_group(builder, g_name.as_ptr(), &mut g0), 0);

        let v_name = CString::new("Speed")?;
        let mut ch_speed = 0u32;
        assert_eq!(
            mdfr_logger_add_channel(builder, g0, v_name.as_ptr(), 4, 64, &mut ch_speed),
            0
        );
        let unit_c = CString::new("km/h")?;
        assert_eq!(
            mdfr_logger_set_channel_unit(builder, ch_speed, unit_c.as_ptr()),
            0
        );

        let m_name = CString::new("Status")?;
        let mut ch_status = 0u32;
        assert_eq!(
            mdfr_logger_add_vlsd_channel(builder, g0, m_name.as_ptr(), 7, &mut ch_status),
            0
        );

        let can_name = CString::new("CAN_FD1")?;
        let mut g_can = 0u16;
        assert_eq!(
            mdfr_logger_add_bus_preset(builder, 1, can_name.as_ptr(), &mut g_can),
            0
        );

        let mut err = -999i32;
        let logger = mdfr_logger_open(builder, path1_c.as_ptr(), &mut err);
        assert!(!logger.is_null());
        assert_eq!(err, 0);

        for i in 0..15 {
            assert_eq!(mdfr_logger_record_begin(logger, g0), 0);
            assert_eq!(
                mdfr_logger_record_set_f64(logger, ch_speed, 50.0 + i as f64),
                0
            );
            let s = format!("ok_{i}");
            assert_eq!(
                mdfr_logger_record_set_bytes(logger, ch_status, s.as_ptr(), s.len()),
                0
            );
            assert_eq!(mdfr_logger_record_commit(logger, g0), 0);
        }

        assert_eq!(mdfr_logger_checkpoint(logger), 0);
        assert_eq!(mdfr_logger_rotate_path(logger, path2_c.as_ptr()), 0);

        for i in 15..25 {
            assert_eq!(mdfr_logger_record_begin(logger, g0), 0);
            assert_eq!(
                mdfr_logger_record_set_f64(logger, ch_speed, 50.0 + i as f64),
                0
            );
            let s = format!("ok_{i}");
            assert_eq!(
                mdfr_logger_record_set_bytes(logger, ch_status, s.as_ptr(), s.len()),
                0
            );
            assert_eq!(mdfr_logger_record_commit(logger, g0), 0);
        }

        assert_eq!(mdfr_logger_close(logger), 0);
    }

    let mut m1 = Mdf::new(tmp1.path().to_str().unwrap())?;
    m1.load_all_channels_data_in_memory()?;
    if let Some(ChannelData::Float64(arr)) = m1.get_channel_data("Speed") {
        assert_eq!(arr.values_slice().len(), 15);
        assert!((arr.values_slice()[14] - 64.0).abs() < 1e-9);
    } else {
        panic!("Speed channel missing in C API file 1");
    }

    let mut m2 = Mdf::new(tmp2.path().to_str().unwrap())?;
    m2.load_all_channels_data_in_memory()?;
    if let Some(ChannelData::Utf8(arr)) = m2.get_channel_data("Status") {
        let built = arr.finish_cloned();
        assert_eq!(built.len(), 10);
        assert_eq!(built.value(9), "ok_24");
    } else {
        panic!("Status channel missing in C API file 2");
    }

    Ok(())
}
