//! Converts ndarray data in into arrow.
use crate::mdfinfo::mdfinfo4::MdfInfo4;
use arrow2::bitmap::Bitmap;
use arrow2::datatypes::{Field, Metadata, Schema};

pub fn mdf4_data_to_arrow(mdf4: &mut MdfInfo4) {
    mdf4.dg.iter().for_each(|(_dg_block_position, dg)| {
        for (_rec_id, cg) in dg.cg.iter() {
            let is_nullable: bool = if cg.invalid_bytes.is_some() {
                true
            } else {
                false
            };
            let mut table = Vec::<Field>::with_capacity(cg.channel_names.len());
            for (_rec_pos, cn) in cg.cn.iter() {
                let mut bitmap: Option<Bitmap> = None;
                if let Some(mask) = &cn.invalid_mask {
                    bitmap = Some(Bitmap::from_u8_slice(mask, mask.len()));
                };
                let data = cn.data.to_arrow_array(bitmap);
                let field = Field::new(
                    cn.unique_name.clone(),
                    data.data_type().clone(),
                    is_nullable,
                );
                let mut metadata = Metadata::new();
                if let Some(unit) = mdf4.sharable.get_tx(cn.block.cn_md_unit) {
                    metadata.insert("unit".to_string(), unit);
                };
                if let Some(desc) = mdf4.sharable.get_tx(cn.block.cn_md_comment) {
                    metadata.insert("description".to_string(), desc);
                };
                let field = field.with_metadata(metadata);
                table.push(field);
            }
            let schema = Schema::from(table);
            let mut metadata = Metadata::new();
            if let Some(master_channel_name) = &cg.master_channel_name {
                metadata.insert("master channel".to_string(), master_channel_name.clone());
            }
            let schema = schema.with_metadata(metadata);
        }
    });
}
