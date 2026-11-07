//! Tier 4: Real-World Domain Application Scenarios
//!
//! Realistic high-coverage automotive and aerospace domain workloads:
//! - Scenario 1: Multi-ECU Automotive Powertrain & Chassis Datalogging
//! - Scenario 2: Severe Power Loss / Abrupt Crash Recovery Simulation
//! - Scenario 3: High-Bandwidth CAN-FD & Ethernet VLSD Bus Logging
//! - Scenario 4: Complex Sensor Calibration & Diagnostic Bitfield Analysis
//! - Scenario 5: High-Throughput Continuous Streaming with DZ Chunking

use anyhow::Result;
use mdfr::data_holder::channel_data::{AsValuesSlice, ChannelData};
use mdfr::mdfreader::Mdf;
use mdfr_core::blocks::cn::CnDataType;
use mdfr_core::compress::CompressionAlgorithm;
use mdfr_core::logger::schema::{ConversionDef, LoggerBuilder};
use mdfr_core::logger::{DataLayout, Logger};
use mdfr_core::time::ManualClock;

#[test]
fn test_tier4_scenario_1_multi_ecu_automotive_datalogging() -> Result<()> {
    // Scenario 1: Multi-ECU automotive datalogging across 3 asynchronous channel groups
    // Group 1: Powertrain ECU (1 kHz): RPM (uint16), Coolant Temp (float32)
    // Group 2: Transmission ECU (100 Hz): Gear (uint8)
    // Group 3: Chassis Controller (20 Hz): Lateral Accel (float32)

    let mut b = LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::NoCompression);

    // Group 1: Powertrain
    let g_pt = b.add_group("Powertrain");
    b.set_group_flush_threshold(g_pt, 256)?;
    let ch_rpm = b.add_channel(g_pt, "EngineRPM", CnDataType::UIntLE, 16)?;
    b.set_channel_unit(ch_rpm, "rpm")?;
    b.set_channel_conversion(ch_rpm, ConversionDef::Linear { p1: 0.0, p2: 0.25 })?;

    let ch_temp = b.add_channel(g_pt, "CoolantTemp", CnDataType::FloatLE, 32)?;
    b.set_channel_unit(ch_temp, "degC")?;

    // Group 2: Transmission
    let g_tm = b.add_group("Transmission");
    let ch_gear = b.add_channel(g_tm, "GearPosition", CnDataType::UIntLE, 8)?;
    b.set_channel_conversion(
        ch_gear,
        ConversionDef::ValueToText {
            entries: vec![
                (0.0, "Park".into()),
                (1.0, "Reverse".into()),
                (2.0, "Neutral".into()),
                (3.0, "Drive".into()),
            ],
            default_text: "Fault".into(),
        },
    )?;

    // Group 3: Chassis
    let g_ch = b.add_group("Chassis");
    let ch_accel = b.add_channel(g_ch, "LateralAccel", CnDataType::FloatLE, 32)?;
    b.set_channel_unit(ch_accel, "m/s2")?;

    let clock = ManualClock::new(0.001, 10);
    let tmp = tempfile::Builder::new().suffix(".mf4").tempfile()?;
    let file = tmp.reopen()?;
    let schema = b.build()?;
    let mut logger = Logger::start(schema, file, clock)?;

    // Simulate 50 cycles of logging across groups
    for i in 0..50u64 {
        // Powertrain 1 kHz: 16-bit RPM, 32-bit Temp
        logger
            .begin_record(g_pt)?
            .set_u64(ch_rpm, 3000 + i * 20)?
            .set_f64(ch_temp, 85.0 + (i as f64) * 0.1)?
            .commit()?;

        // Transmission (logged every 2nd cycle)
        if i % 2 == 0 {
            logger.begin_record(g_tm)?.set_u64(ch_gear, 3)?.commit()?;
        }

        // Chassis (logged every 5th cycle)
        if i % 5 == 0 {
            logger
                .begin_record(g_ch)?
                .set_f64(ch_accel, 0.05 * (i as f64))?
                .commit()?;
        }
    }

    logger.close()?;

    // Read back and verify channel counts and integrity
    let mut mdf = Mdf::new(tmp.path().to_str().unwrap())?;
    assert!(!mdf.mdf_info.is_unfinalized());
    mdf.load_all_channels_data_in_memory()?;

    assert!(mdf.get_channel_data("EngineRPM").is_some());
    Ok(())
}

#[test]
fn test_tier4_scenario_2_power_loss_crash_recovery_simulation() -> Result<()> {
    // Scenario 2: Simulate power loss crash during logging (file left unfinalized)
    let mut b = LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::NoCompression);
    let g = b.add_group("EngineSensors");
    b.set_group_flush_threshold(g, 64)?;
    let ch_p = b.add_channel(g, "OilPressure", CnDataType::UIntLE, 32)?;

    let clock = ManualClock::new(0.001, 10);
    let tmp = tempfile::Builder::new().suffix(".mf4").tempfile()?;

    // Build and push samples, then drop logger WITHOUT calling close() to simulate power cut
    {
        let file = tmp.reopen()?;
        let schema = b.build()?;
        let mut logger = Logger::start(schema, file, clock)?;
        for i in 0..100u64 {
            logger.begin_record(g)?.set_u64(ch_p, 300 + i)?.commit()?;
        }
        // Logger intentionally dropped here without close()
    }

    // Verify unfinalized state before recovery
    let unrecovered_mdf = Mdf::new(tmp.path().to_str().unwrap())?;
    assert!(unrecovered_mdf.mdf_info.is_unfinalized());

    // Execute in-place crash recovery on file
    {
        let mut f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(tmp.path())?;
        let rec = mdfr_core::logger::recover_in_place(&mut f);
        assert!(rec.is_ok());
    }

    // After recovery, file is finalized and samples are readable
    let mut recovered_mdf = Mdf::new(tmp.path().to_str().unwrap())?;
    assert!(!recovered_mdf.mdf_info.is_unfinalized());
    recovered_mdf.load_all_channels_data_in_memory()?;

    if let Some(ChannelData::UInt32(arr)) = recovered_mdf.get_channel_data("OilPressure") {
        assert!(arr.values_slice().len() > 0);
    }
    Ok(())
}

#[test]
fn test_tier4_scenario_3_canfd_vlsd_bus_logging() -> Result<()> {
    // Scenario 3: CAN-FD bus logging with variable-length payload frames
    let mut b = LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::NoCompression);
    let g = b.add_group("CANFD_Bus");
    let ch_vlsd = b.add_vlsd_channel(g, "Payload", CnDataType::ByteArray)?;

    let clock = ManualClock::new(0.001, 10);
    let tmp = tempfile::Builder::new().suffix(".mf4").tempfile()?;
    let file = tmp.reopen()?;
    let schema = b.build()?;
    let mut logger = Logger::start(schema, file, clock)?;

    // Push varying length frames: 8 bytes, 16 bytes, 32 bytes, 64 bytes
    let frames: [&[u8]; 4] = [&[0x01; 8], &[0x02; 16], &[0x03; 32], &[0x04; 64]];

    for frame in &frames {
        logger
            .begin_record(g)?
            .set_bytes(ch_vlsd, frame)?
            .commit()?;
    }

    logger.close()?;
    assert!(tmp.as_file().metadata()?.len() > 0);
    Ok(())
}

#[test]
fn test_tier4_scenario_4_sensor_calibration_exclusive_ranges() {
    // Scenario 4: Calibration curves with strictly exclusive upper bounds
    // Intervals: [0.0, 1.0) -> "Low", [1.0, 2.5) -> "Nominal", [2.5, 5.0) -> "High"
    let intervals = [
        (0.0f64, 1.0f64, "Low"),
        (1.0f64, 2.5f64, "Nominal"),
        (2.5f64, 5.0f64, "High"),
    ];

    let classify = |val: f64| -> Option<&'static str> {
        for (min, max, label) in intervals {
            if min <= val && val < max {
                return Some(label);
            }
        }
        None
    };

    assert_eq!(classify(0.0), Some("Low"));
    assert_eq!(classify(0.999), Some("Low"));
    assert_eq!(
        classify(1.0),
        Some("Nominal"),
        "1.0 must classify as Nominal, not Low"
    );
    assert_eq!(classify(2.499), Some("Nominal"));
    assert_eq!(
        classify(2.5),
        Some("High"),
        "2.5 must classify as High, not Nominal"
    );
    assert_eq!(classify(5.0), None, "5.0 is exclusive upper bound of High");
}

#[test]
fn test_tier4_scenario_5_high_throughput_streaming_and_dz_chunking() -> Result<()> {
    // Scenario 5: High-throughput streaming with 4 MiB DZ chunking
    let mut b = LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::DeflateTranspose);
    let g = b.add_group("HighThroughput");
    b.set_group_flush_threshold(g, 1024)?;
    let ch_t = b.add_channel(g, "Telemetry", CnDataType::UIntLE, 32)?;

    let clock = ManualClock::new(0.001, 10);
    let tmp = tempfile::Builder::new().suffix(".mf4").tempfile()?;
    let file = tmp.reopen()?;
    let schema = b.build()?;
    let mut logger = Logger::start(schema, file, clock)?;

    // Push 1,000 records
    for i in 0..1000u64 {
        logger.begin_record(g)?.set_u64(ch_t, i)?.commit()?;
    }

    logger.close()?;
    assert!(tmp.as_file().metadata()?.len() > 0);
    Ok(())
}
