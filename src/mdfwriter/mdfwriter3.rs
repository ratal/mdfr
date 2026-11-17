//! Converter of mdf version 3.x into mdf version 4.2
use crate::mdfinfo::mdfinfo3::{Cg3, Cn3, Dg3, MdfInfo3, convert_data_type_3to4};
use crate::mdfinfo::mdfinfo4::{MdfInfo4, new_fh_block};
use crate::mdfreader::{DataSignature, MasterSignature};
use anyhow::{Context, Error, Result};

/// Helper to convert a single MDF3 channel to physical values and add it to `MdfInfo4`.
fn add_cn3_to_mdf4(
    mdf3: &MdfInfo3,
    mdf4: &mut MdfInfo4,
    cg: &Cg3,
    cn: &Cn3,
    master_flag: bool,
) -> Result<(), Error> {
    let unit = mdf3._get_unit(&cn.block1.cn_cc_conversion);
    let desc = Some(cn.description.clone());
    let cycle_count = cg.block.cg_cycle_count as usize;
    let converted_data = mdf3
        .get_channel_converted_data(&cn.unique_name)
        .unwrap_or_else(|| cn.data.clone());
    let machine_endian: bool = cfg!(target_endian = "big");
    let data_type = if !converted_data.is_empty() {
        converted_data.data_type(machine_endian)
    } else {
        convert_data_type_3to4(cn.block2.cn_data_type)
    };
    let bit_count = if !converted_data.is_empty() && converted_data.bit_count() > 0 {
        converted_data.bit_count()
    } else {
        cn.block2.cn_bit_count as u32
    };
    let byte_count = if !converted_data.is_empty() && converted_data.byte_count() > 0 {
        converted_data.byte_count()
    } else {
        cn.n_bytes as u32
    };
    let (shape_vec, order) = converted_data.shape();
    let ndim = converted_data.ndim();
    let data_signature = DataSignature {
        len: cycle_count,
        data_type,
        bit_count,
        byte_count,
        ndim,
        shape: (shape_vec, order),
    };
    let master_signature = MasterSignature {
        master_channel: cg.master_channel_name.as_deref().map(str::to_string),
        master_type: Some(u8::from(master_flag)),
        master_flag,
    };
    mdf4.add_channel(
        cn.unique_name.to_string(),
        converted_data,
        data_signature,
        master_signature,
        unit,
        desc,
    )
}

/// Converts mdfinfo3 into mdfinfo4
pub fn convert3to4(mdf3: &MdfInfo3, file_name: &str) -> Result<MdfInfo4, Error> {
    let n_channels = mdf3.get_channel_names_set().len();
    let mut mdf4 = MdfInfo4::new(file_name, n_channels);
    // FH
    let fh = new_fh_block();
    mdf4.fh.push(fh);

    mdf3.dg.iter().try_for_each(
        |(_dg_block_position, dg): (&u32, &Dg3)| -> Result<(), Error> {
            dg.cg
                .iter()
                .try_for_each(|(_rec_id, cg): (&u16, &Cg3)| -> Result<(), Error> {
                    // First add master channel
                    if let Some(master_channel_name) = &cg.master_channel_name
                        && let Some((_master_channel, _dg_pos, (_cg_pos, _rec_id), cn_pos)) =
                            mdf3.channel_names_set.get(master_channel_name.as_ref())
                        && let Some(cn) = cg.cn.get(cn_pos)
                    {
                        add_cn3_to_mdf4(mdf3, &mut mdf4, cg, cn, true)
                            .context("Failed adding channel")?;
                    }
                    // then add other channels
                    cg.cn
                        .iter()
                        .filter(|(_rec_pos, cn)| {
                            cg.master_channel_name.as_deref() != Some(cn.unique_name.as_ref())
                        })
                        .try_for_each(|(_rec_pos, cn): (&u32, &Cn3)| -> Result<(), Error> {
                            add_cn3_to_mdf4(mdf3, &mut mdf4, cg, cn, false)
                        })
                        .context("Failed adding channels")?;
                    Ok(())
                })?;
            Ok(())
        },
    )?;
    Ok(mdf4)
}
