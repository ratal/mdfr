use anyhow::Result;
use mdfr::mdfreader::Mdf;
use std::sync::LazyLock;

static BASE: LazyLock<String> = LazyLock::new(|| {
    "/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/MDF4.3/Base_Standard/Examples/DataTypes/CANopenTypes/".to_string()
});

#[test]
fn canopen_date() -> Result<()> {
    let mut mdf = Mdf::new(&format!("{}Vector_CANOpenDate.mf4", *BASE))?;
    mdf.load_all_channels_data_in_memory()?;
    let names = mdf.get_channel_names_set();
    assert!(!names.is_empty(), "No channels found in CANopen date file");
    let has_data = names
        .iter()
        .any(|n| mdf.get_channel_data(n).is_some_and(|d| !d.is_empty()));
    assert!(has_data, "All channels empty in CANopen date file");
    Ok(())
}

#[test]
fn canopen_time() -> Result<()> {
    let mut mdf = Mdf::new(&format!("{}Vector_CANOpenTime.mf4", *BASE))?;
    mdf.load_all_channels_data_in_memory()?;
    let names = mdf.get_channel_names_set();
    assert!(!names.is_empty(), "No channels found in CANopen time file");
    let has_data = names
        .iter()
        .any(|n| mdf.get_channel_data(n).is_some_and(|d| !d.is_empty()));
    assert!(has_data, "All channels empty in CANopen time file");
    Ok(())
}
