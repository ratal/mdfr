//! Converter of mdf version 3.x into mdf version 4.2
use crate::mdfreader::{DataSignature, MasterSignature};

use crate::mdfinfo::{
    mdfinfo3::MdfInfo3,
    mdfinfo4::{FhBlock, MdfInfo4},
};
use crate::mdfreader::channel_data::Order;

/// Converts mdfinfo3 into mdfinfo4
pub fn convert3to4(mdf3: &MdfInfo3, file_name: &str) -> MdfInfo4 {
    let n_channels = mdf3.get_channel_names_set().len();
    let mut mdf4 = MdfInfo4::new(file_name, n_channels);
    // FH
    let fh = FhBlock::default();
    mdf4.fh.push(fh);

    mdf3.dg.iter().for_each(|(_dg_block_position, dg)| {
        dg.cg.iter().for_each(|(_rec_id, cg)| {
            // First add master channel
            if let Some(master_channel_name) = &cg.master_channel_name {
                if let Some((_master_channel, _dg_pos, (_cg_pos, _rec_id), cn_pos)) =
                    mdf3.channel_names_set.get(master_channel_name)
                {
                    if let Some(cn) = cg.cn.get(cn_pos) {
                        let unit = mdf3._get_unit(&cn.block1.cn_cc_conversion);
                        let desc = Some(cn.description.clone());
                        let cycle_count = cg.block.cg_cycle_count as usize;
                        let bit_count = cn.block2.cn_bit_count;
                        let data_type = convert3to4_data_type(cn.block2.cn_data_type);
                        let data_signature = DataSignature {
                            len: cycle_count,
                            data_type,
                            bit_count: bit_count as u32,
                            byte_count: cn.n_bytes as u32,
                            ndim: 1,
                            shape: (vec![cycle_count; 1], Order::RowMajor),
                        };
                        let master_signature = MasterSignature {
                            master_channel: cg.master_channel_name.clone(),
                            master_type: Some(1),
                            master_flag: true,
                        };
                        mdf4.add_channel(
                            master_channel_name.clone(),
                            data_signature,
                            master_signature,
                            unit,
                            desc,
                        );
                    }
                }
            }
            // then add other channels
            cg.cn
                .iter()
                .filter(|(_rec_pos, cn)| Some(cn.unique_name.clone()) != cg.master_channel_name)
                .for_each(|(_rec_pos, cn)| {
                    let unit = mdf3._get_unit(&cn.block1.cn_cc_conversion);
                    let desc = Some(cn.description.clone());
                    let cycle_count = cg.block.cg_cycle_count as usize;
                    let bit_count = cn.block2.cn_bit_count;
                    let data_type = convert3to4_data_type(cn.block2.cn_data_type);
                    let data_signature = DataSignature {
                        len: cycle_count,
                        data_type,
                        bit_count: bit_count as u32,
                        byte_count: cn.n_bytes as u32,
                        ndim: 1,
                        shape: (vec![cycle_count; 1], Order::RowMajor),
                    };
                    let master_signature = MasterSignature {
                        master_channel: cg.master_channel_name.clone(),
                        master_type: Some(0),
                        master_flag: false,
                    };
                    mdf4.add_channel(
                        cn.unique_name.clone(),
                        data_signature,
                        master_signature,
                        unit,
                        desc,
                    );
                });
        });
    });
    mdf4
}

fn convert3to4_data_type(data_type: u16) -> u8 {
    match data_type {
        0 | 13 => 0,
        1 | 14 => 2,
        2 | 3 | 15 | 16 => 4,
        7 => 6,
        8 => 10,
        9 => 1,
        10 => 3,
        11 | 12 => 5,
        _ => 0,
    }
}
