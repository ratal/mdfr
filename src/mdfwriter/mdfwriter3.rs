//! Converter of mdf version 3.x into mdf version 4.2
use crate::mdfinfo::mdfinfo3::{Cg3, Cn3, Dg3, MdfInfo3, convert_data_type_3to4};
use crate::mdfinfo::mdfinfo4::{MdfInfo4, new_fh_block};
use crate::mdfreader::conversions3::converted_output_signature_3;
use crate::mdfreader::{DataSignature, MasterSignature};
use anyhow::{Context, Error, Result};

/// Helper to transfer an MDF3 channel into `MdfInfo4`, converting upfront only when `convert_data` is true.
fn add_cn3_to_mdf4(
    mdf3: &MdfInfo3,
    mdf4: &mut MdfInfo4,
    cg: &Cg3,
    cn: &Cn3,
    master_flag: bool,
    convert_data: bool,
) -> Result<(), Error> {
    let unit = mdf3._get_unit(&cn.block1.cn_cc_conversion);
    let desc = Some(cn.description.clone());
    let cycle_count = cg.block.cg_cycle_count as usize;
    let machine_endian: bool = cfg!(target_endian = "big");
    let (data_for_mdf4, data_type, bit_count, byte_count, ndim, (shape_vec, order)) =
        if convert_data {
            let converted_data = mdf3
                .get_channel_converted_data(&cn.unique_name)
                .unwrap_or_else(|| cn.data.clone());
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
            let shape = converted_data.shape();
            let ndim = converted_data.ndim();
            (
                converted_data,
                data_type,
                bit_count,
                byte_count,
                ndim,
                shape,
            )
        } else {
            let sig = converted_output_signature_3(cn, &mdf3.sharable, machine_endian)?;
            let data_type = if !cn.data.is_empty() {
                sig.data_type
            } else {
                convert_data_type_3to4(cn.block2.cn_data_type)
            };
            let bit_count = if !cn.data.is_empty() && sig.bit_count > 0 {
                sig.bit_count
            } else {
                cn.block2.cn_bit_count as u32
            };
            let byte_count = if !cn.data.is_empty() && sig.byte_count > 0 {
                sig.byte_count
            } else {
                cn.n_bytes as u32
            };
            let shape = cn.data.shape();
            let ndim = cn.data.ndim();
            (
                cn.data.clone(),
                data_type,
                bit_count,
                byte_count,
                ndim,
                shape,
            )
        };
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
        data_for_mdf4,
        data_signature,
        master_signature,
        unit,
        desc,
    )
}

fn convert3to4_inner(
    mdf3: &MdfInfo3,
    file_name: &str,
    convert_data: bool,
) -> Result<MdfInfo4, Error> {
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
                        add_cn3_to_mdf4(mdf3, &mut mdf4, cg, cn, true, convert_data)
                            .context("Failed adding channel")?;
                    }
                    // then add other channels
                    cg.cn
                        .iter()
                        .filter(|(_rec_pos, cn)| {
                            cg.master_channel_name.as_deref() != Some(cn.unique_name.as_ref())
                        })
                        .try_for_each(|(_rec_pos, cn): (&u32, &Cn3)| -> Result<(), Error> {
                            add_cn3_to_mdf4(mdf3, &mut mdf4, cg, cn, false, convert_data)
                        })
                        .context("Failed adding channels")?;
                    Ok(())
                })?;
            Ok(())
        },
    )?;
    Ok(mdf4)
}

/// Converts mdfinfo3 into mdfinfo4 with physical values materialized.
pub fn convert3to4(mdf3: &MdfInfo3, file_name: &str) -> Result<MdfInfo4, Error> {
    convert3to4_inner(mdf3, file_name, true)
}

/// Converts mdfinfo3 metadata/headers into mdfinfo4 without eagerly converting numeric arrays,
/// allowing `mdfwriter4` to convert and stream data in bounded batches.
pub(crate) fn convert3to4_unconverted(mdf3: &MdfInfo3, file_name: &str) -> Result<MdfInfo4, Error> {
    convert3to4_inner(mdf3, file_name, false)
}
