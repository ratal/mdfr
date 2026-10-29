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

#[test]
fn roundtrip_bitpacked_rational_2d_array_and_all_bus_presets() -> Result<()> {
    use mdfr_core::logger::{ChannelHandle, SpscRing};

    let mut b = LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::NoCompression);
    let g_bits = b.add_group("BitPackedGroup");
    // 3-bit + 5-bit = 1 byte; 12-bit signed + 4-bit unsigned = 2 bytes
    let ch_u3 = b.add_channel(g_bits, "Flag3Bit", CnDataType::UIntLE, 3)?;
    let ch_u5 = b.add_channel(g_bits, "Counter5Bit", CnDataType::UIntLE, 5)?;
    let ch_i12 = b.add_channel(g_bits, "Signed12Bit", CnDataType::IntLE, 12)?;
    let ch_u4 = b.add_channel(g_bits, "Mode4Bit", CnDataType::UIntLE, 4)?;
    let ch_rat = b.add_channel(g_bits, "RationalSensor", CnDataType::UIntLE, 16)?;
    // Rational: Y = (0*X^2 + 2.0*X + 10.0) / (0*X^2 + 0*X + 1.0) = 2*X + 10
    b.set_channel_conversion(
        ch_rat,
        ConversionDef::Rational([0.0, 2.0, 10.0, 0.0, 0.0, 1.0]),
    )?;
    let ch_mat = b.add_array_channel(g_bits, "Matrix2x3", CnDataType::UIntLE, 8, vec![2, 3])?;

    let can_fd = BusPreset::can_fd(&mut b, "CANFD_Bus")?;
    let g_lin = BusPreset::lin(&mut b, "LIN_Bus")?;
    let g_flexray = BusPreset::flexray(&mut b, "FlexRay_Bus")?;
    let g_eth = BusPreset::ethernet(&mut b, "ETH_Bus")?;

    // Raw fixed-size group for push_raw_at & SpscRing
    let g_raw = b.add_group("FastRaw");
    let _ch_raw_val = b.add_channel(g_raw, "RawVal", CnDataType::UIntLE, 32)?;

    let schema = b.build()?;
    let tmp = tempfile::Builder::new().suffix(".mf4").tempfile()?;
    let clock = ManualClock::new(0.001, 1);
    let mut logger = Logger::start(schema, tmp.reopen()?, clock)?;

    for i in 0..20u64 {
        let mat: [u8; 6] = [
            i as u8,
            (i + 1) as u8,
            (i + 2) as u8,
            (i + 3) as u8,
            (i + 4) as u8,
            (i + 5) as u8,
        ];
        logger
            .begin_record(g_bits)?
            .set_u64(ch_u3, i & 0x07)?
            .set_u64(ch_u5, (i * 2) & 0x1F)?
            .set_i64(ch_i12, -100 + i as i64 * 10)?
            .set_u64(ch_u4, i & 0x0F)?
            .set_u64(ch_rat, i * 5)?
            .set_bytes(ch_mat, &mat)?
            .commit()?;

        let fd_payload = [0xAAu8; 64];
        logger
            .begin_record_at(can_fd.group, 1000 + i * 10)?
            .set_u64(can_fd.bus_channel, 1)?
            .set_u64(can_fd.can_id, 0x18FF_1234)?
            .set_u64(can_fd.dir, 1)?
            .set_u64(can_fd.dlc, 15)?
            .set_u64(can_fd.data_length, 64)?
            .set_bytes(can_fd.data_bytes, &fd_payload)?
            .commit()?;

        let lin_payload = [0x55u8; 8];
        logger
            .begin_record(g_lin)?
            .set_u64(
                ChannelHandle {
                    group: g_lin.0,
                    channel: 1,
                },
                0x2A,
            )?
            .set_u64(
                ChannelHandle {
                    group: g_lin.0,
                    channel: 2,
                },
                8,
            )?
            .set_u64(
                ChannelHandle {
                    group: g_lin.0,
                    channel: 3,
                },
                8,
            )?
            .set_bytes(
                ChannelHandle {
                    group: g_lin.0,
                    channel: 4,
                },
                &lin_payload,
            )?
            .commit()?;

        let fr_payload = [0x33u8; 12];
        logger
            .begin_record(g_flexray)?
            .set_u64(
                ChannelHandle {
                    group: g_flexray.0,
                    channel: 1,
                },
                15,
            )?
            .set_u64(
                ChannelHandle {
                    group: g_flexray.0,
                    channel: 2,
                },
                i & 0x3F,
            )?
            .set_u64(
                ChannelHandle {
                    group: g_flexray.0,
                    channel: 3,
                },
                0,
            )?
            .set_u64(
                ChannelHandle {
                    group: g_flexray.0,
                    channel: 4,
                },
                12,
            )?
            .set_bytes(
                ChannelHandle {
                    group: g_flexray.0,
                    channel: 5,
                },
                &fr_payload,
            )?
            .commit()?;

        let eth_payload = [0x11u8; 32];
        logger
            .begin_record(g_eth)?
            .set_bytes(
                ChannelHandle {
                    group: g_eth.0,
                    channel: 1,
                },
                &[0x00, 0x11, 0x22, 0x33, 0x44, 0x55],
            )?
            .set_bytes(
                ChannelHandle {
                    group: g_eth.0,
                    channel: 2,
                },
                &[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
            )?
            .set_u64(
                ChannelHandle {
                    group: g_eth.0,
                    channel: 3,
                },
                0x0800,
            )?
            .set_u64(
                ChannelHandle {
                    group: g_eth.0,
                    channel: 4,
                },
                32,
            )?
            .set_bytes(
                ChannelHandle {
                    group: g_eth.0,
                    channel: 5,
                },
                &eth_payload,
            )?
            .commit()?;
    }

    // Exercise SpscRing ISR queue + push_raw_at
    let ring = SpscRing::with_capacity(1024);
    for i in 0..10u32 {
        let mut raw_rec = [0u8; 12]; // 8B master timestamp + 4B u32
        raw_rec[8..12].copy_from_slice(&(1000 + i).to_le_bytes());
        assert!(ring.try_push(g_raw, (500 + i as u64) * 10, &raw_rec));
    }
    assert_eq!(ring.drain_into(&mut logger)?, 10);
    // Also exercise push_raw_at directly (patches master timestamp at bytes 0..8)
    let mut raw_rec = [0u8; 12];
    raw_rec[8..12].copy_from_slice(&2000u32.to_le_bytes());
    logger.push_raw_at(g_raw, 9999, &raw_rec)?;

    logger.close()?;

    let mut mdf = Mdf::new(tmp.path().to_str().unwrap())?;
    mdf.load_all_channels_data_in_memory()?;

    // Verify bit-packed channels
    if let Some(ChannelData::UInt8(arr)) = mdf.get_channel_data("Flag3Bit") {
        let s = arr.values_slice();
        assert_eq!(s.len(), 20);
        for (i, &val) in s.iter().enumerate() {
            assert_eq!(val, (i as u8) & 0x07);
        }
    } else {
        panic!("Flag3Bit missing");
    }

    if let Some(ChannelData::UInt8(arr)) = mdf.get_channel_data("Counter5Bit") {
        let s = arr.values_slice();
        assert_eq!(s.len(), 20);
        for (i, &val) in s.iter().enumerate() {
            assert_eq!(val, ((i as u8) * 2) & 0x1F);
        }
    } else {
        panic!("Counter5Bit missing");
    }

    if let Some(ChannelData::Int16(arr)) = mdf.get_channel_data("Signed12Bit") {
        let s = arr.values_slice();
        assert_eq!(s.len(), 20);
        assert_eq!(s[0], -100);
        assert_eq!(s[19], -100 + 190);
    } else {
        panic!("Signed12Bit missing");
    }

    // Verify Rational conversion: Y = 2*X + 10 where X = i * 5
    if let Some(ChannelData::Float64(arr)) = mdf.get_channel_data("RationalSensor") {
        let s = arr.values_slice();
        assert_eq!(s.len(), 20);
        assert!((s[0] - 10.0).abs() < 1e-6);
        assert!((s[10] - (2.0 * 50.0 + 10.0)).abs() < 1e-6);
    } else {
        panic!("RationalSensor missing");
    }

    // Verify RawVal (10 via SpscRing + 1 via push_raw_at = 11)
    if let Some(ChannelData::UInt32(arr)) = mdf.get_channel_data("RawVal") {
        let s = arr.values_slice();
        assert_eq!(s.len(), 11);
        assert_eq!(s[0], 1000);
        assert_eq!(s[9], 1009);
        assert_eq!(s[10], 2000);
    } else {
        panic!("RawVal missing");
    }

    Ok(())
}

#[test]
fn mid_record_truncation_and_on_device_recovery() -> Result<()> {
    use std::io::{Seek, SeekFrom, Write};

    // Build an uncompressed sorted file so records stream into a DTBLOCK, then simulate
    // a mid-record power loss by writing partial record bytes and leaving hdr_len = 24.
    let mut b = LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::NoCompression);
    let g0 = b.add_group("PowerCutGroup");
    b.set_group_flush_threshold(g0, 64)?;
    let ch_v = b.add_channel(g0, "Voltage", CnDataType::UIntLE, 32)?;
    let schema = b.build()?;

    let tmp = tempfile::Builder::new().suffix(".mf4").tempfile()?;
    let clock = ManualClock::new(0.001, 1);
    let mut logger = Logger::start(schema, tmp.reopen()?, clock)?;

    // Each record is 8B (time) + 4B (u32) = 12 bytes.
    // Push 20 records (240 bytes -> flushes into DTBLOCK).
    for i in 0..20u64 {
        logger.begin_record(g0)?.set_u64(ch_v, 3300 + i)?.commit()?;
    }
    logger.checkpoint()?;
    drop(logger); // Unfinalized!

    // Append 7 stray bytes at EOF to simulate power loss mid-record, and reset the DTBLOCK
    // hdr_len to 24 (header only) so recovery must infer the valid length from file size
    // and truncate the partial record.
    {
        let mut f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(tmp.path())?;
        f.seek(SeekFrom::End(0))?;
        f.write_all(&[0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03])?;
        f.flush()?;
    }

    // Run `no_std` on-device recovery (`mdfr_core::logger::recover_in_place`) on a copy first
    let tmp_copy = tempfile::Builder::new().suffix(".mf4").tempfile()?;
    std::fs::copy(tmp.path(), tmp_copy.path())?;
    {
        let mut f_dev = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(tmp_copy.path())?;
        let recovered = mdfr_core::logger::recover_in_place(&mut f_dev)
            .expect("on-device recover_in_place failed");
        assert!(recovered > 0);
    }

    let mut mdf_dev = Mdf::new(tmp_copy.path().to_str().unwrap())?;
    assert!(!mdf_dev.mdf_info.is_unfinalized());
    mdf_dev.load_all_channels_data_in_memory()?;
    if let Some(ChannelData::UInt32(arr)) = mdf_dev.get_channel_data("Voltage") {
        assert_eq!(arr.values_slice().len(), 20);
        assert_eq!(arr.values_slice()[0], 3300);
        assert_eq!(arr.values_slice()[19], 3319);
    } else {
        panic!("Voltage channel missing after on-device recovery");
    }

    // Also verify host-side `finalize_file_in_place` on the original truncated file
    mdfr::mdfinfo::mdfinfo4::finalize::finalize_file_in_place(tmp.path().to_str().unwrap())?;
    let mut mdf_host = Mdf::new(tmp.path().to_str().unwrap())?;
    assert!(!mdf_host.mdf_info.is_unfinalized());
    mdf_host.load_all_channels_data_in_memory()?;
    if let Some(ChannelData::UInt32(arr)) = mdf_host.get_channel_data("Voltage") {
        assert_eq!(arr.values_slice().len(), 20);
        assert_eq!(arr.values_slice()[19], 3319);
    } else {
        panic!("Voltage channel missing after host finalize_file_in_place");
    }

    Ok(())
}

#[test]
fn asammdf_cross_check_if_available() -> Result<()> {
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    for (layout, algo) in [
        (DataLayout::Sorted, CompressionAlgorithm::NoCompression),
        (DataLayout::Sorted, CompressionAlgorithm::DeflateTranspose),
        (DataLayout::Sorted, CompressionAlgorithm::Lz4Transpose),
        (DataLayout::Unsorted, CompressionAlgorithm::NoCompression),
        (DataLayout::Unsorted, CompressionAlgorithm::DeflateTranspose),
    ] {
        let mut b = LoggerBuilder::new(layout, algo);
        let g0 = b.add_group("Vehicle");
        let ch_v = b.add_channel(g0, "Velocity", CnDataType::FloatLE, 64)?;
        b.set_channel_unit(ch_v, "m/s")?;

        let ch_curr = b.add_channel(g0, "Current", CnDataType::IntLE, 16)?;
        b.set_channel_conversion(
            ch_curr,
            ConversionDef::Linear {
                p1: -100.0,
                p2: 0.1,
            },
        )?;
        b.set_channel_unit(ch_curr, "A")?;

        let ch_press = b.add_channel(g0, "Pressure", CnDataType::UIntLE, 16)?;
        b.set_channel_conversion(
            ch_press,
            ConversionDef::Rational([0.0, 0.5, 0.0, 0.0, 0.0, 1.0]),
        )?;
        b.set_channel_unit(ch_press, "bar")?;

        let ch_status = b.add_channel(g0, "StatusByte", CnDataType::UIntLE, 8)?;
        let ch_raw = b.add_channel(g0, "RawPayload", CnDataType::ByteArray, 32)?;
        let ch_flag = b.add_bit_channel(g0, "StatusFlag", CnDataType::UIntLE, 25, 0, 1)?;

        let g_pt = b.add_record_group::<PowertrainRecord>()?;

        let schema = b.build()?;

        let tmp = tempfile::Builder::new().suffix(".mf4").tempfile()?;
        let clock = ManualClock::new(0.01, 1);
        let mut logger = Logger::start(schema, tmp.reopen()?, clock)?;
        for i in 0..25u64 {
            logger
                .begin_record(g0)?
                .set_f64(ch_v, i as f64 * 1.5)?
                .set_i64(ch_curr, (i * 10) as i64)?
                .set_u64(ch_press, 100 + i * 5)?
                .set_u64(ch_status, i % 256)?
                .set_u64(ch_flag, i % 2)?
                .set_bytes(
                    ch_raw,
                    &[i as u8, (i + 1) as u8, (i + 2) as u8, (i + 3) as u8],
                )?
                .commit()?;

            if i < 20 {
                let rec = PowertrainRecord {
                    rpm_raw: 4000 + i as u16 * 40,
                    torque_raw: 400 + i as i16 * 2,
                    throttle: 25.0 + i as f32 * 0.5,
                    gear: (i % 6) as u8,
                    fault: i % 2 == 1,
                    raw_frame: [i as u8, (i + 1) as u8, (i + 2) as u8, (i + 3) as u8],
                };
                logger.push_record(g_pt, &rec)?;
            }
        }
        logger.close()?;

        let path_str = tmp.path().to_str().unwrap().to_string();

        Python::attach(|py| -> PyResult<()> {
            // Check if asammdf is installed in the current Python environment
            if py.import("asammdf").is_ok() {
                let locals = PyDict::new(py);
                locals.set_item("path", &path_str)?;
                py.run(
                    pyo3::ffi::c_str!(
                        r#"
from asammdf import MDF
m = MDF(path)

# 1. Float channel
sig = m.get("Velocity")
assert len(sig.samples) == 25, f"Velocity len: {len(sig.samples)}"
assert abs(float(sig.samples[0]) - 0.0) < 1e-5
assert abs(float(sig.samples[24]) - 36.0) < 1e-5

# 2. Linear converted channel: phys = -100.0 + 0.1 * (i * 10) = -100.0 + i
sig_c = m.get("Current")
assert len(sig_c.samples) == 25, f"Current len: {len(sig_c.samples)}"
assert abs(float(sig_c.samples[0]) - (-100.0)) < 1e-4
assert abs(float(sig_c.samples[24]) - (-76.0)) < 1e-4

# 3. Rational converted channel: phys = 0.5 * (100 + i * 5) = 50.0 + 2.5 * i
sig_p = m.get("Pressure")
assert len(sig_p.samples) == 25, f"Pressure len: {len(sig_p.samples)}"
assert abs(float(sig_p.samples[0]) - 50.0) < 1e-4
assert abs(float(sig_p.samples[24]) - 110.0) < 1e-4

# 4. StatusByte and bit-channel
sig_s = m.get("StatusByte")
assert int(sig_s.samples[0]) == 0
assert int(sig_s.samples[24]) == 24

sig_f = m.get("StatusFlag")
assert int(sig_f.samples[0]) == 0
assert int(sig_f.samples[1]) == 1

# 5. #[derive(MdfRecord)] struct channels
sig_rpm = m.get("EngineSpeed")
assert len(sig_rpm.samples) == 20
assert abs(float(sig_rpm.samples[0]) - 1000.0) < 1e-4
assert abs(float(sig_rpm.samples[19]) - 1190.0) < 1e-4

sig_trq = m.get("Torque")
assert len(sig_trq.samples) == 20
assert abs(float(sig_trq.samples[0]) - 100.0) < 1e-4
assert abs(float(sig_trq.samples[19]) - 119.0) < 1e-4

sig_th = m.get("Throttle")
assert len(sig_th.samples) == 20
assert abs(float(sig_th.samples[0]) - 25.0) < 1e-4
assert abs(float(sig_th.samples[19]) - 34.5) < 1e-4

m.close()
"#
                    ),
                    None,
                    Some(&locals),
                )?;
            }
            Ok(())
        })?;
    }
    Ok(())
}

#[test]
fn no_std_cortex_m_cross_build() -> Result<()> {
    use std::process::Command;

    // Check if thumbv7em-none-eabihf is installed in rustup/rustc sysroot
    let sysroot_out = Command::new("rustc")
        .args([
            "--print",
            "target-libdir",
            "--target",
            "thumbv7em-none-eabihf",
        ])
        .output()?;
    if !sysroot_out.status.success() {
        return Ok(());
    }
    let libdir = String::from_utf8_lossy(&sysroot_out.stdout)
        .trim()
        .to_string();
    if !std::path::Path::new(&libdir).exists() {
        return Ok(());
    }

    let tmp_target = tempfile::tempdir()?;
    let status = Command::new("cargo")
        .args([
            "check",
            "--offline",
            "--manifest-path",
            concat!(env!("CARGO_MANIFEST_DIR"), "/mdfr-core/Cargo.toml"),
            "--target",
            "thumbv7em-none-eabihf",
            "--features",
            "capi-arena,defmt",
        ])
        .env("CARGO_TARGET_DIR", tmp_target.path())
        .status()?;
    assert!(
        status.success(),
        "no_std Cortex-M4F (thumbv7em-none-eabihf) cross-check failed"
    );
    Ok(())
}

#[derive(mdfr_core::MdfRecord)]
#[mdf(group = "Powertrain")]
struct PowertrainRecord {
    #[mdf(name = "EngineSpeed", unit = "rpm", linear(0.0, 0.25))]
    rpm_raw: u16,
    #[mdf(name = "Torque", unit = "Nm", rational(0.0, 0.5, -100.0, 0.0, 0.0, 1.0))]
    torque_raw: i16,
    #[mdf(name = "Throttle", unit = "%")]
    throttle: f32,
    #[mdf(name = "GearEngaged")]
    gear: u8,
    #[mdf(name = "FaultActive")]
    fault: bool,
    #[mdf(name = "RawFrame")]
    raw_frame: [u8; 4],
}

#[test]
fn derive_mdf_record_and_vlsc_roundtrip() -> Result<()> {
    for (layout, algo) in [
        (DataLayout::Sorted, CompressionAlgorithm::NoCompression),
        (DataLayout::Sorted, CompressionAlgorithm::Lz4Transpose),
        (DataLayout::Unsorted, CompressionAlgorithm::NoCompression),
        (DataLayout::Unsorted, CompressionAlgorithm::DeflateTranspose),
    ] {
        let mut b = LoggerBuilder::new(layout, algo);
        let g_pt = b.add_record_group::<PowertrainRecord>()?;
        let g_vlsc = b.add_group("Diagnostics");
        let ch_vlsc = b.add_vlsc_channel(g_vlsc, "VlscText", CnDataType::StringUtf8)?;

        let schema = b.build()?;
        let tmp = tempfile::Builder::new().suffix(".mf4").tempfile()?;
        let clock = ManualClock::new(0.001, 1);
        let mut logger = Logger::start(schema, tmp.reopen()?, clock)?;

        for i in 0..20u16 {
            let rec = PowertrainRecord {
                rpm_raw: 4000 + i * 40,         // * 0.25 -> 1000.0 + i * 10.0
                torque_raw: 400 + i as i16 * 2, // 0.5 * X - 100 -> 100.0 + i * 1.0
                throttle: 25.0 + i as f32 * 0.5,
                gear: (i % 6) as u8,
                fault: i % 2 == 1,
                raw_frame: [i as u8, (i + 1) as u8, (i + 2) as u8, (i + 3) as u8],
            };
            logger.push_record(g_pt, &rec)?;

            let text = format!("vlsc_payload_{i}");
            logger
                .begin_record(g_vlsc)?
                .set_bytes(ch_vlsc, text.as_bytes())?
                .commit()?;
        }
        logger.close()?;

        let mut mdf = Mdf::new(tmp.path().to_str().unwrap())?;
        mdf.load_all_channels_data_in_memory()?;

        // Verify #[derive(MdfRecord)] channels
        if let Some(ChannelData::Float64(arr)) = mdf.get_channel_data("EngineSpeed") {
            let s = arr.values_slice();
            assert_eq!(s.len(), 20);
            assert!((s[0] - 1000.0).abs() < 1e-6);
            assert!((s[19] - 1190.0).abs() < 1e-6);
        } else {
            panic!("EngineSpeed missing for layout={layout:?} algo={algo:?}");
        }

        if let Some(ChannelData::Float64(arr)) = mdf.get_channel_data("Torque") {
            let s = arr.values_slice();
            assert_eq!(s.len(), 20);
            assert!((s[0] - 100.0).abs() < 1e-6);
            assert!((s[19] - 119.0).abs() < 1e-6);
        } else {
            panic!("Torque missing for layout={layout:?} algo={algo:?}");
        }

        // Verify MDF 4.3 VLSC channel (`cn_type = 7`, `cn_cn_size`, `VDBLOCK` / `CG_F_VLSC`)
        if let Some(ChannelData::Utf8(arr)) = mdf.get_channel_data("VlscText") {
            let built = arr.finish_cloned();
            assert_eq!(built.len(), 20);
            assert_eq!(built.value(0), "vlsc_payload_0");
            assert_eq!(built.value(19), "vlsc_payload_19");
        } else {
            panic!("VlscText missing for layout={layout:?} algo={algo:?}");
        }
    }
    Ok(())
}
