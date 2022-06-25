use crate::mdfinfo::mdfinfo4::FhBlock;

use crate::mdfinfo::{mdfinfo3::MdfInfo3, mdfinfo4::MdfInfo4};
use crate::mdfreader::channel_data::ChannelData;

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
                        mdf4.add_channel(
                            master_channel_name.clone(),
                            ChannelData::default(),
                            cg.master_channel_name.clone(),
                            Some(1),
                            true,
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
                    mdf4.add_channel(
                        cn.unique_name.clone(),
                        ChannelData::default(),
                        cg.master_channel_name.clone(),
                        Some(0),
                        false,
                        unit,
                        desc,
                    );
                });
        });
    });
    mdf4
}
