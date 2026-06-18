#[path = "common.rs"]
mod common;

use anyhow::Result;
use mdfr::mdfreader::Mdf;
use mdfr::mdfinfo::MdfInfo;
use std::sync::LazyLock;

static BASE_PATH_MDF4: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}MDF4/MDF4.3/Base_Standard/Examples/",
        common::mdfreader_tests_path()
    )
});

#[test]
fn test_lazy_conversion_trigger() -> Result<()> {
    let file = format!(
        "{}Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4",
        BASE_PATH_MDF4.as_str()
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;

    // Find a channel that has conversion rules and verify it's not converted yet
    let mut target_channel_name = None;
    if let MdfInfo::V4(ref info4) = mdf.mdf_info {
        for dg in info4.dg.values() {
            for cg in dg.cg.values() {
                for cn in cg.cn.values() {
                    // Check if it has a conversion block reference and is not empty
                    if cn.block.cn_cc_conversion != 0 && !cn.data.is_empty() {
                        assert!(!cn.is_converted, "Channel {} should not be converted initially", cn.unique_name);
                        target_channel_name = Some(cn.unique_name.clone());
                        break;
                    }
                }
                if target_channel_name.is_some() {
                    break;
                }
            }
            if target_channel_name.is_some() {
                break;
            }
        }
    }

    let channel_name = target_channel_name.expect("Should find at least one channel with conversion rules");

    // Retrieve the data, which triggers lazy conversion
    let _data = mdf.get_channel_data(&channel_name).expect("Should successfully retrieve channel data");

    // Verify it is now marked as converted
    if let MdfInfo::V4(ref info4) = mdf.mdf_info {
        for dg in info4.dg.values() {
            for cg in dg.cg.values() {
                for cn in cg.cn.values() {
                    if cn.unique_name == channel_name {
                        assert!(cn.is_converted, "Channel {} should be marked as converted after retrieval", channel_name);
                    }
                }
            }
        }
    }

    Ok(())
}

#[test]
fn test_eager_conversion_api() -> Result<()> {
    let file = format!(
        "{}Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4",
        BASE_PATH_MDF4.as_str()
    );
    let mut mdf = Mdf::new(&file)?;
    mdf.load_all_channels_data_in_memory()?;

    // Call eager conversion
    mdf.convert_all_channels()?;

    // Verify all channels with conversion rules are eagerly converted
    if let MdfInfo::V4(ref info4) = mdf.mdf_info {
        for dg in info4.dg.values() {
            for cg in dg.cg.values() {
                for cn in cg.cn.values() {
                    if cn.block.cn_cc_conversion != 0 && !cn.data.is_empty() {
                        assert!(cn.is_converted, "Channel {} should be eagerly converted", cn.unique_name);
                    }
                }
            }
        }
    }

    Ok(())
}
