use crate::mdfinfo::mdfinfo4::MdfInfo4;

pub fn mdfreader4(info: &mut MdfInfo4) {

    // read file data
    for (dg_position, dg) in info.dg.iter_mut() {
        for (cg_position, cg) in dg.cg.iter_mut() {
            for (cn_position, cn) in cg.cn.iter_mut() {
            }
        }
    }
}