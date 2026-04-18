use anyhow::Result;
use mdfr::mdfreader::Mdf;
use std::sync::LazyLock;

static MDF3_PATH: LazyLock<String> =
    LazyLock::new(|| "/home/ratal/workspace/mdfreader/mdfreader/tests/mdf3/".to_string());

/// Helper: load a file and all its channel data
fn load_mdf3(filename: &str) -> Result<Mdf> {
    let path = format!("{}{}", MDF3_PATH.as_str(), filename);
    let mut mdf = Mdf::new(&path)?;
    mdf.load_all_channels_data_in_memory()?;
    Ok(mdf)
}

#[test]
fn mdf3_canape_loads() -> Result<()> {
    let mdf = load_mdf3("MDF_CANAPE.mdf")?;
    assert!(!mdf.get_channel_names_set().is_empty());
    Ok(())
}

#[test]
fn mdf3_ascet_loads() -> Result<()> {
    let mdf = load_mdf3("ASCET.dat")?;
    assert!(!mdf.get_channel_names_set().is_empty());
    Ok(())
}

#[test]
fn mdf3_mda71_loads() -> Result<()> {
    let mdf = load_mdf3("MDA71.dat")?;
    assert!(!mdf.get_channel_names_set().is_empty());
    Ok(())
}

#[test]
fn mdf3_rj_can_loads() -> Result<()> {
    // RJ file has linear conversion channels
    let mdf = load_mdf3("RJ_N16-12-363_BM-15C-0024_228_2_20170116094355_CAN.dat")?;
    assert!(!mdf.get_channel_names_set().is_empty());
    Ok(())
}

#[test]
fn mdf3_measure_loads() -> Result<()> {
    let mdf = load_mdf3("Measure.dat")?;
    assert!(!mdf.get_channel_names_set().is_empty());
    Ok(())
}

#[test]
fn mdf3_t3_nedc_loads() -> Result<()> {
    let mdf = load_mdf3("T3_121121_000_6NEDC.dat")?;
    assert!(!mdf.get_channel_names_set().is_empty());
    Ok(())
}

#[test]
fn mdf3_738l10_loads() -> Result<()> {
    let mdf = load_mdf3("738L10_040410 Base Acc 30km_hr.dat")?;
    assert!(!mdf.get_channel_names_set().is_empty());
    Ok(())
}

#[test]
fn mdf3_hidden_bytes_loads() -> Result<()> {
    let mdf = load_mdf3("Mdf3_hiddenBytes_NotAlignedBytes.dat")?;
    assert!(!mdf.get_channel_names_set().is_empty());
    Ok(())
}

#[test]
fn mdf3_tgt_loads() -> Result<()> {
    let mdf = load_mdf3("TGT.dat")?;
    assert!(!mdf.get_channel_names_set().is_empty());
    Ok(())
}

#[test]
fn mdf3_canoe_unsorted_loads() -> Result<()> {
    let mdf = load_mdf3("CANoe3_unsorted.mdf")?;
    assert!(!mdf.get_channel_names_set().is_empty());
    Ok(())
}
