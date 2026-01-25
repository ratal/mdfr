use anyhow::Result;
use glob::glob;
use mdfr::mdfreader::Mdf;
use std::fs;
use std::sync::LazyLock;

static MDFREADER_TESTS_PATH: &str = "/home/ratal/workspace/mdfreader/mdfreader/tests/";
static MDFR_PATH: &str = "/home/ratal/workspace/mdfr/";

static BASE_PATH_MDF4: LazyLock<String> = LazyLock::new(|| {
    format!("{}MDF4/MDF4.3/Base_Standard/Examples/", MDFREADER_TESTS_PATH)
});

static BASE_PATH_MDF3: LazyLock<String> =
    LazyLock::new(|| format!("{}mdf3/", MDFREADER_TESTS_PATH));

static BASE_TEST_PATH: LazyLock<String> = LazyLock::new(|| format!("{}test_files", MDFR_PATH));

#[cfg(feature = "parquet")]
static WRITING_PARQUET_FILE: LazyLock<String> =
    LazyLock::new(|| format!("{}test_files/test_parquet", MDFR_PATH));

#[cfg(feature = "hdf5")]
static WRITING_HDF5_FILE: LazyLock<String> =
    LazyLock::new(|| format!("{}test_files/test_hdf5.hdf5", MDFR_PATH));

#[cfg(feature = "parquet")]
#[test]
fn export_to_parquet() -> Result<()> {
    // Export mdf4 to Parquet file
    let file = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        &"Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4"
    );
    let extension = "*.parquet";
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;
    mdf.export_to_parquet(WRITING_PARQUET_FILE.as_str(), Some("zstd"))
        .expect("failed writing mdf4 parquet file");

    // Export mdf3 to Parquet file
    let file = format!(
        "{}{}",
        BASE_PATH_MDF3.as_str(),
        &"RJ_N16-12-363_BM-15C-0024_228_2_20170116094355_CAN.dat"
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;
    mdf.export_to_parquet(WRITING_PARQUET_FILE.as_str(), Some("snappy"))
        .expect("failed writing mdf3 parquet file");

    // remove all generated parquet files
    let pattern = format!("{}/{}", BASE_TEST_PATH.as_str(), extension);
    for path in glob(&pattern).unwrap().filter_map(Result::ok) {
        fs::remove_file(path)?;
    }
    Ok(())
}

#[cfg(feature = "hdf5")]
#[test]
fn export_to_hdf5() -> Result<()> {
    // Export mdf4 to HDF5 file
    let file = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        &"Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4"
    );
    let extension = "*.hdf5";
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;
    mdf.export_to_hdf5(WRITING_HDF5_FILE.as_str(), Some(&"lzf"))
        .expect("failed writing mdf4 hdf5 file");

    // Export mdf3 to HDF5 file
    let file = format!(
        "{}{}",
        BASE_PATH_MDF3.as_str(),
        &"RJ_N16-12-363_BM-15C-0024_228_2_20170116094355_CAN.dat"
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;
    mdf.export_to_hdf5(WRITING_HDF5_FILE.as_str(), Some(&"deflate"))
        .expect("failed writing mdf3 hdf5 file");

    // remove all generated hdf5 files
    let pattern = format!("{}/{}", BASE_TEST_PATH.as_str(), extension);
    for path in glob(&pattern).unwrap().filter_map(Result::ok) {
        fs::remove_file(path)?;
    }
    Ok(())
}
