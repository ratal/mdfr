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
