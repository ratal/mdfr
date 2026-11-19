use crate::common;

use anyhow::Result;
use mdfr::mdfreader::Mdf;
use std::sync::LazyLock;

static BASE_PATH_MDF4: LazyLock<String> = LazyLock::new(common::base_path_mdf4);

static BASE_PATH_MDF3: LazyLock<String> = LazyLock::new(common::base_path_mdf3);

#[cfg(feature = "parquet")]
#[test]
fn export_to_parquet() -> Result<()> {
    if !std::path::Path::new(BASE_PATH_MDF4.as_str()).is_dir() {
        return Ok(());
    }
    let tmp_dir = tempfile::tempdir()?;
    let writing_parquet_file = tmp_dir.path().join("test_parquet");
    let writing_parquet_file = writing_parquet_file.to_str().unwrap();

    // Export mdf4 to Parquet file
    let file = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4"
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;
    for comp in ["snappy", "zstd", "gzip", "lz4"] {
        mdf.export_to_parquet(writing_parquet_file, Some(comp))
            .unwrap_or_else(|_| panic!("failed writing mdf4 parquet file with {comp}"));
    }

    // Export mdf3 to Parquet file
    let file = format!(
        "{}{}",
        BASE_PATH_MDF3.as_str(),
        "RJ_N16-12-363_BM-15C-0024_228_2_20170116094355_CAN.dat"
    );
    if !std::path::Path::new(&file).exists() {
        return Ok(());
    }
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;
    mdf.export_to_parquet(writing_parquet_file, Some("snappy"))
        .expect("failed writing mdf3 parquet file");

    // tmp_dir cleaned up on drop
    Ok(())
}

#[cfg(feature = "hdf5")]
#[test]
fn export_to_hdf5() -> Result<()> {
    if !std::path::Path::new(BASE_PATH_MDF4.as_str()).is_dir() {
        return Ok(());
    }
    let tmp_dir = tempfile::tempdir()?;
    let writing_hdf5_file = tmp_dir.path().join("test_hdf5.hdf5");
    let writing_hdf5_file = writing_hdf5_file.to_str().unwrap();

    // Export mdf4 to HDF5 file
    let file = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4"
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;
    mdf.export_to_hdf5(writing_hdf5_file, Some("lzf"))
        .expect("failed writing mdf4 hdf5 file");

    // Export mdf3 to HDF5 file
    let file = format!(
        "{}{}",
        BASE_PATH_MDF3.as_str(),
        "RJ_N16-12-363_BM-15C-0024_228_2_20170116094355_CAN.dat"
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;
    mdf.export_to_hdf5(writing_hdf5_file, Some("deflate"))
        .expect("failed writing mdf3 hdf5 file");

    // tmp_dir cleaned up on drop
    Ok(())
}

#[cfg(feature = "parquet")]
#[test]
fn export_and_verify_parquet() -> Result<()> {
    use arrow::array::Array;
    if !std::path::Path::new(BASE_PATH_MDF4.as_str()).is_dir() {
        return Ok(());
    }
    let tmp_dir = tempfile::tempdir()?;
    let writing_parquet_file = tmp_dir.path().join("test_parquet");
    let writing_parquet_file = writing_parquet_file.to_str().unwrap();

    let file = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4"
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;

    // Pick a Float64 channel to verify
    let test_channel = "Me_VE2";
    mdf.export_dataframe_to_parquet(test_channel.to_string(), writing_parquet_file, None)
        .expect("failed exporting to parquet");

    let mut exported_files = std::fs::read_dir(tmp_dir.path())?
        .filter_map(Result::ok)
        .filter(|d| d.path().extension().and_then(|s| s.to_str()) == Some("parquet"))
        .collect::<Vec<_>>();
    assert_eq!(exported_files.len(), 1);
    let parquet_path = exported_files.pop().unwrap().path();

    let file = std::fs::File::open(parquet_path)?;
    let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut total_rows = 0;
    let mut first_val: Option<f64> = None;
    let mut first_is_null: Option<bool> = None;

    for batch in reader {
        let batch = batch?;
        let rows = batch.num_rows();
        total_rows += rows;

        let schema = batch.schema();
        if let Ok(idx) = schema.index_of(test_channel) {
            let col = batch.column(idx);
            let float_col = col
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .unwrap();
            if first_val.is_none() && !float_col.is_empty() {
                first_is_null = Some(float_col.is_null(0));
                first_val = Some(float_col.value(0));
            }
        }
    }

    let orig_data = mdf.get_channel_converted_data(test_channel).unwrap();
    assert_eq!(orig_data.len(), total_rows);

    if let mdfr::data_holder::channel_data::ChannelData::Float64(b) = orig_data {
        let arr = b;
        if !arr.is_empty() {
            assert_eq!(first_is_null.unwrap(), arr.is_null(0));
            if !arr.is_null(0) {
                assert_eq!(first_val.unwrap(), arr.value(0));
            }
        }
    } else {
        panic!("Expected Float64 channel");
    }

    Ok(())
}

#[test]
fn export_and_verify_csv() -> Result<()> {
    if !std::path::Path::new(BASE_PATH_MDF4.as_str()).is_dir() {
        return Ok(());
    }
    let tmp_dir = tempfile::tempdir()?;
    let writing_csv_file = tmp_dir.path().join("test_csv");
    let writing_csv_file = writing_csv_file.to_str().unwrap();

    let file = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4"
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;

    // Pick a UInt8 channel to verify
    let test_channel = "TOEL_S";
    mdf.export_dataframe_to_csv(test_channel, writing_csv_file)
        .expect("failed exporting to csv");

    let mut exported_files = std::fs::read_dir(tmp_dir.path())?
        .filter_map(Result::ok)
        .filter(|d| d.path().extension().and_then(|s| s.to_str()) == Some("csv"))
        .collect::<Vec<_>>();
    assert_eq!(exported_files.len(), 1);
    let csv_path = exported_files.pop().unwrap().path();

    let file = std::fs::File::open(csv_path)?;
    let reader = std::io::BufReader::new(file);
    use std::io::BufRead;

    let mut lines = reader.lines();
    let header = lines.next().expect("File is empty")?;
    let cols: Vec<&str> = header.split(',').collect();
    let col_idx = cols
        .iter()
        .position(|&c| c == test_channel)
        .expect("Column not found");

    let mut row_count = 0;
    let mut first_val = None;

    for line in lines {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let vals: Vec<&str> = line.split(',').collect();
        if first_val.is_none() {
            first_val = Some(vals[col_idx].to_string());
        }
        row_count += 1;
    }

    let orig_data = mdf.get_channel_converted_data(test_channel).unwrap();
    assert_eq!(orig_data.len(), row_count);

    if let mdfr::data_holder::channel_data::ChannelData::UInt8(b) = orig_data {
        let arr = b;
        if !arr.is_empty() {
            assert_eq!(first_val.unwrap().parse::<u8>().unwrap(), arr.value(0));
        }
    } else {
        panic!("Expected UInt8 channel");
    }

    Ok(())
}

#[cfg(feature = "hdf5")]
#[test]
fn export_and_verify_hdf5() -> Result<()> {
    if !std::path::Path::new(BASE_PATH_MDF4.as_str()).is_dir() {
        return Ok(());
    }
    let tmp_dir = tempfile::tempdir()?;
    let writing_hdf5_file = tmp_dir.path().join("test_hdf5.hdf5");
    let writing_hdf5_file = writing_hdf5_file.to_str().unwrap();

    let file = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4"
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;

    let test_channel = "Me_VE2";
    mdf.export_dataframe_to_hdf5(test_channel.to_string(), writing_hdf5_file, None)
        .expect("failed writing mdf4 hdf5 file");

    let hdf5_file = rust_hdf5::H5File::open(writing_hdf5_file)?;

    let mut dataset = None;
    let root = hdf5_file.root_group();
    for group_name in root.group_names()? {
        if let Ok(ds) = hdf5_file.dataset(&format!("{}/{}", group_name, test_channel)) {
            dataset = Some(ds);
            break;
        }
    }

    let dataset = dataset.expect("Dataset not found in HDF5 file");

    let orig_data = mdf.get_channel_converted_data(test_channel).unwrap();
    assert_eq!(dataset.shape()[0], orig_data.len());

    if let mdfr::data_holder::channel_data::ChannelData::Float64(b) = orig_data {
        let arr = b;
        if !arr.is_empty() {
            let hdf5_data = dataset.read_raw::<f64>()?;
            assert_eq!(hdf5_data[0], arr.value(0));
        }
    } else {
        panic!("Expected Float64 channel");
    }

    Ok(())
}

#[test]
fn export_all_to_csv() -> Result<()> {
    if !std::path::Path::new(BASE_PATH_MDF4.as_str()).is_dir() {
        return Ok(());
    }
    let tmp_dir = tempfile::tempdir()?;
    let writing_csv_file = tmp_dir.path().join("all_cg");
    let writing_csv_file = writing_csv_file.to_str().unwrap();

    let file = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4"
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;
    mdf.export_to_csv(writing_csv_file)?;

    let count = std::fs::read_dir(tmp_dir.path())?
        .filter_map(Result::ok)
        .filter(|d| d.path().extension().and_then(|s| s.to_str()) == Some("csv"))
        .count();
    assert!(count >= 1);
    Ok(())
}

#[cfg(all(feature = "parquet", feature = "polars"))]
#[test]
fn export_dotted_master_to_parquet() -> Result<()> {
    let tmp_dir = tempfile::tempdir()?;
    let writing_parquet_file = tmp_dir.path().join("dotted_export");
    let writing_parquet_str = writing_parquet_file.to_str().unwrap();

    let df = polars::df!(
        "Time.Channel" => &[0.0f64, 0.1, 0.2],
        "Vehicle.Speed" => &[10.0f64, 20.0, 30.0]
    )?;
    let mdf = Mdf::from_polars_df(&df, Some("Time.Channel"))?;
    mdf.export_to_parquet(writing_parquet_str, None)?;

    let expected_file = tmp_dir.path().join("dotted_export_Time.Channel.parquet");
    assert!(
        expected_file.exists(),
        "Expected parquet file {:?} to exist, but was not found",
        expected_file
    );
    assert!(
        expected_file.metadata()?.len() > 0,
        "Exported parquet file must not be empty"
    );
    Ok(())
}

#[cfg(feature = "polars")]
#[test]
fn export_dotted_master_to_csv() -> Result<()> {
    let tmp_dir = tempfile::tempdir()?;
    let writing_csv_file = tmp_dir.path().join("dotted_export");
    let writing_csv_str = writing_csv_file.to_str().unwrap();

    let df = polars::df!(
        "Time.Channel" => &[0.0f64, 0.1, 0.2],
        "Vehicle.Speed" => &[10.0f64, 20.0, 30.0]
    )?;
    let mdf = Mdf::from_polars_df(&df, Some("Time.Channel"))?;
    mdf.export_to_csv(writing_csv_str)?;

    let expected_file = tmp_dir.path().join("dotted_export_Time.Channel.csv");
    assert!(
        expected_file.exists(),
        "Expected CSV file {:?} to exist, but was not found",
        expected_file
    );
    let contents = std::fs::read_to_string(&expected_file)?;
    assert!(contents.contains("Time.Channel"));
    assert!(contents.contains("Vehicle.Speed"));
    Ok(())
}

#[cfg(feature = "parquet")]
#[test]
fn export_applies_lazy_conversion_to_parquet_and_csv() -> Result<()> {
    use mdfr::CompressionAlgorithm;
    use mdfr::logger::{ConversionDef, DataLayout, Logger, LoggerBuilder};
    use mdfr_core::blocks::cn::CnDataType;
    use mdfr_core::time::ManualClock;

    let tmp_dir = tempfile::tempdir()?;
    let mf4_path = tmp_dir.path().join("converted.mf4");
    let mut b = LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::NoCompression);
    let g = b.add_group("Grp");
    let ch_lin = b.add_channel(g, "Temp", CnDataType::UIntLE, 16)?;
    b.set_channel_conversion(ch_lin, ConversionDef::Linear { p1: 10.0, p2: 2.0 })?;
    let ch_txt = b.add_channel(g, "State", CnDataType::UIntLE, 8)?;
    b.set_channel_conversion(
        ch_txt,
        ConversionDef::ValueToText {
            entries: vec![(1.0, "ON".into()), (0.0, "OFF".into())],
            default_text: "UNK".into(),
        },
    )?;
    let schema = b.build()?;
    let f = std::fs::File::create(&mf4_path)?;
    let mut logger = Logger::start(schema, f, ManualClock::new(1e-3, 0))?;
    for i in 0..3u64 {
        logger
            .begin_record(g)?
            .set_u64(ch_lin, i + 5)?
            .set_u64(ch_txt, i % 2)?
            .commit()?;
    }
    logger.close()?;

    let mut mdf = Mdf::new(mf4_path.to_str().unwrap())?;
    mdf.load_all_channels_data_in_memory()?;

    // Export WITHOUT calling get_channel_data() first (tests lazy conversion inside export)
    let pq_prefix = tmp_dir.path().join("out_pq");
    mdf.export_to_parquet(pq_prefix.to_str().unwrap(), None)?;
    let pq_file = tmp_dir.path().join("out_pq_time.parquet");
    assert!(pq_file.exists());
    let pq_reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
        std::fs::File::open(&pq_file)?,
    )?
    .build()?;
    let batches: Vec<_> = pq_reader.collect::<std::result::Result<_, _>>()?;
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    // Deterministic column order: master `time`, then `Temp`, then `State`
    assert_eq!(batch.schema().field(0).name(), "time");
    assert_eq!(batch.schema().field(1).name(), "Temp");
    assert_eq!(batch.schema().field(2).name(), "State");
    let temp_col = batch
        .column(1)
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .expect("Temp must be converted to Float64");
    assert_eq!(temp_col.values(), &[20.0, 22.0, 24.0]);
    let state_col = batch
        .column(2)
        .as_any()
        .downcast_ref::<arrow::array::LargeStringArray>()
        .expect("State must be converted to LargeString");
    assert_eq!(state_col.value(0), "OFF");
    assert_eq!(state_col.value(1), "ON");
    assert_eq!(state_col.value(2), "OFF");

    let csv_prefix = tmp_dir.path().join("out_csv");
    mdf.export_to_csv(csv_prefix.to_str().unwrap())?;
    let csv_file = tmp_dir.path().join("out_csv_time.csv");
    let csv_text = std::fs::read_to_string(&csv_file)?;
    let lines: Vec<&str> = csv_text.lines().collect();
    assert_eq!(lines[0], "time,Temp,State");
    assert!(lines[1].contains("20,OFF") || lines[1].contains("20.0,OFF"));
    assert!(lines[2].contains("22,ON") || lines[2].contains("22.0,ON"));
    Ok(())
}

#[cfg(feature = "parquet")]
#[test]
fn export_multiple_masterless_groups_no_filename_collision() -> Result<()> {
    let tmp_dir = tempfile::tempdir()?;
    let mut mdf = Mdf::new("test_files/test_basic.mf4")?;
    mdf.add_channel(
        "OrphanA".to_string(),
        std::sync::Arc::new(arrow::array::Float64Array::from(vec![1.0, 2.0])),
        None,
        None,
        false,
        None,
        None,
    )?;
    mdf.add_channel(
        "OrphanB".to_string(),
        std::sync::Arc::new(arrow::array::Float64Array::from(vec![3.0, 4.0])),
        None,
        None,
        false,
        None,
        None,
    )?;

    let csv_prefix = tmp_dir.path().join("multi");
    mdf.export_to_csv(csv_prefix.to_str().unwrap())?;
    let csv_count = std::fs::read_dir(tmp_dir.path())?
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("csv"))
        .count();
    assert_eq!(csv_count, 2, "two masterless groups must yield 2 CSV files");

    let pq_prefix = tmp_dir.path().join("multi");
    mdf.export_to_parquet(pq_prefix.to_str().unwrap(), None)?;
    let pq_count = std::fs::read_dir(tmp_dir.path())?
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("parquet"))
        .count();
    assert_eq!(
        pq_count, 2,
        "two masterless groups must yield 2 Parquet files"
    );
    Ok(())
}
