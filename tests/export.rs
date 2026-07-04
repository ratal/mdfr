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

static BASE_PATH_MDF3: LazyLock<String> =
    LazyLock::new(|| format!("{}mdf3/", common::mdfreader_tests_path()));

#[cfg(feature = "parquet")]
#[test]
fn export_to_parquet() -> Result<()> {
    let tmp_dir = tempfile::tempdir()?;
    let writing_parquet_file = tmp_dir.path().join("test_parquet");
    let writing_parquet_file = writing_parquet_file.to_str().unwrap();

    // Export mdf4 to Parquet file
    let file = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        &"Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4"
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;
    mdf.export_to_parquet(writing_parquet_file, Some("zstd"))
        .expect("failed writing mdf4 parquet file");

    // Export mdf3 to Parquet file
    let file = format!(
        "{}{}",
        BASE_PATH_MDF3.as_str(),
        &"RJ_N16-12-363_BM-15C-0024_228_2_20170116094355_CAN.dat"
    );
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
    let tmp_dir = tempfile::tempdir()?;
    let writing_hdf5_file = tmp_dir.path().join("test_hdf5.hdf5");
    let writing_hdf5_file = writing_hdf5_file.to_str().unwrap();

    // Export mdf4 to HDF5 file
    let file = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        &"Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4"
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;
    mdf.export_to_hdf5(writing_hdf5_file, Some(&"lzf"))
        .expect("failed writing mdf4 hdf5 file");

    // Export mdf3 to HDF5 file
    let file = format!(
        "{}{}",
        BASE_PATH_MDF3.as_str(),
        &"RJ_N16-12-363_BM-15C-0024_228_2_20170116094355_CAN.dat"
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;
    mdf.export_to_hdf5(writing_hdf5_file, Some(&"deflate"))
        .expect("failed writing mdf3 hdf5 file");

    // tmp_dir cleaned up on drop
    Ok(())
}

#[cfg(feature = "parquet")]
#[test]
fn export_and_verify_parquet() -> Result<()> {
    let tmp_dir = tempfile::tempdir()?;
    let writing_parquet_file = tmp_dir.path().join("test_parquet");
    let writing_parquet_file = writing_parquet_file.to_str().unwrap();

    let file = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        &"Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4"
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
    let mut reader = builder.build()?;
    
    let mut total_rows = 0;
    let mut first_val: Option<f64> = None;
    
    while let Some(batch) = reader.next() {
        let batch = batch?;
        let rows = batch.num_rows();
        total_rows += rows;
        
        let schema = batch.schema();
        if let Ok(idx) = schema.index_of(test_channel) {
            let col = batch.column(idx);
            let float_col = col.as_any().downcast_ref::<arrow::array::Float64Array>().unwrap();
            if first_val.is_none() && float_col.len() > 0 {
                first_val = Some(float_col.value(0));
            }
        }
    }
    
    let orig_data = mdf.get_channel_converted_data(test_channel).unwrap();
    assert_eq!(orig_data.len(), total_rows);
    
    if let mdfr::data_holder::channel_data::ChannelData::Float64(b) = orig_data {
        let arr = b.finish_cloned();
        if arr.len() > 0 {
            assert_eq!(first_val.unwrap(), arr.value(0));
        }
    } else {
        panic!("Expected Float64 channel");
    }

    Ok(())
}

#[test]
fn export_and_verify_csv() -> Result<()> {
    let tmp_dir = tempfile::tempdir()?;
    let writing_csv_file = tmp_dir.path().join("test_csv");
    let writing_csv_file = writing_csv_file.to_str().unwrap();

    let file = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        &"Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4"
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
    let col_idx = cols.iter().position(|&c| c == test_channel).expect("Column not found");

    let mut row_count = 0;
    let mut first_val = None;
    
    for line in lines {
        let line = line?;
        if line.is_empty() { continue; }
        let vals: Vec<&str> = line.split(',').collect();
        if first_val.is_none() {
            first_val = Some(vals[col_idx].to_string());
        }
        row_count += 1;
    }
    
    let orig_data = mdf.get_channel_converted_data(test_channel).unwrap();
    assert_eq!(orig_data.len(), row_count);
    
    if let mdfr::data_holder::channel_data::ChannelData::UInt8(b) = orig_data {
        let arr = b.finish_cloned();
        if arr.len() > 0 {
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
    let tmp_dir = tempfile::tempdir()?;
    let writing_hdf5_file = tmp_dir.path().join("test_hdf5.hdf5");
    let writing_hdf5_file = writing_hdf5_file.to_str().unwrap();

    let file = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        &"Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4"
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
        let arr = b.finish_cloned();
        if arr.len() > 0 {
            let hdf5_data = dataset.read_raw::<f64>()?;
            assert_eq!(hdf5_data[0], arr.value(0));
        }
    } else {
        panic!("Expected Float64 channel");
    }

    Ok(())
}
