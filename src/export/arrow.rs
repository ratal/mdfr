//! Converts ndarray data in into arrow.
use crate::mdfinfo::mdfinfo4::MdfInfo4;
use crate::mdfreader::channel_data::ChannelData;
use arrow2::array::{
    Array, BinaryArray, BooleanArray, FixedSizeBinaryArray, FixedSizeListArray, PrimitiveArray,
    Utf8Array,
};
use arrow2::bitmap::Bitmap;
use arrow2::buffer::Buffer;
use arrow2::chunk::Chunk;
use arrow2::datatypes::{DataType, Field, Metadata, Schema};
use std::sync::Arc;

pub fn mdf4_data_to_arrow(mdf4: &MdfInfo4) -> (Vec<Chunk<Arc<dyn Array>>>, Schema) {
    let mut row_groups = Vec::<Chunk<Arc<dyn Array>>>::with_capacity(mdf4.dg.len());
    let mut table = Vec::<Field>::with_capacity(mdf4.channel_names_set.len());
    mdf4.dg.iter().for_each(|(_dg_block_position, dg)| {
        for (_rec_id, cg) in dg.cg.iter() {
            let is_nullable: bool = cg.invalid_bytes.is_some();
            let mut columns = Vec::<Arc<dyn Array>>::with_capacity(cg.channel_names.len());
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
                columns.push(data);
                let mut metadata = Metadata::new();
                if let Some(unit) = mdf4.sharable.get_tx(cn.block.cn_md_unit) {
                    metadata.insert("unit".to_string(), unit);
                };
                if let Some(desc) = mdf4.sharable.get_tx(cn.block.cn_md_comment) {
                    metadata.insert("description".to_string(), desc);
                };
                if cn.block.cn_type == 2 || cn.block.cn_type == 3 {
                    metadata.insert(
                        "master_channel".to_string(),
                        cn.block.cn_sync_type.to_string(),
                    );
                } else if cn.block.cn_type == 4 {
                    metadata.insert(
                        "sync_channel".to_string(),
                        cn.block.cn_sync_type.to_string(),
                    );
                }
                let field = field.with_metadata(metadata);
                table.push(field);
            }
            row_groups.push(Chunk::new(columns));
        }
    });
    let schema = Schema::from(table);
    let mut metadata = Metadata::new();
    metadata.insert("file_name".to_string(), mdf4.file_name.clone());
    (row_groups, schema)
}

impl ChannelData {
    pub fn to_arrow_array(&self, bitmap: Option<Bitmap>) -> Arc<dyn Array> {
        match self {
            ChannelData::Boolean(a) => {
                let bitmap_bools = Bitmap::from_u8_slice(a, a.len() * 8);
                Arc::new(BooleanArray::new(DataType::Boolean, bitmap_bools, bitmap))
            }
            ChannelData::Int8(a) => Arc::new(PrimitiveArray::new(
                DataType::Int8,
                Buffer::from_slice(a),
                bitmap,
            )),
            ChannelData::UInt8(a) => Arc::new(PrimitiveArray::new(
                DataType::UInt8,
                Buffer::from_slice(a),
                bitmap,
            )),
            ChannelData::Int16(a) => Arc::new(PrimitiveArray::new(
                DataType::Int16,
                Buffer::from_slice(a),
                bitmap,
            )),
            ChannelData::UInt16(a) => Arc::new(PrimitiveArray::new(
                DataType::UInt16,
                Buffer::from_slice(a),
                bitmap,
            )),
            ChannelData::Float16(a) => Arc::new(PrimitiveArray::new(
                DataType::Float32,
                Buffer::from_slice(a),
                bitmap,
            )),
            ChannelData::Int24(a) => Arc::new(PrimitiveArray::new(
                DataType::Int32,
                Buffer::from_slice(a),
                bitmap,
            )),
            ChannelData::UInt24(a) => Arc::new(PrimitiveArray::new(
                DataType::UInt32,
                Buffer::from_slice(a),
                bitmap,
            )),
            ChannelData::Int32(a) => Arc::new(PrimitiveArray::new(
                DataType::Int32,
                Buffer::from_slice(a),
                bitmap,
            )),
            ChannelData::UInt32(a) => Arc::new(PrimitiveArray::new(
                DataType::UInt32,
                Buffer::from_slice(a),
                bitmap,
            )),
            ChannelData::Float32(a) => Arc::new(PrimitiveArray::new(
                DataType::Float32,
                Buffer::from_slice(a),
                bitmap,
            )),
            ChannelData::Int48(a) => Arc::new(PrimitiveArray::new(
                DataType::Int64,
                Buffer::from_slice(a),
                bitmap,
            )),
            ChannelData::UInt48(a) => Arc::new(PrimitiveArray::new(
                DataType::UInt64,
                Buffer::from_slice(a),
                bitmap,
            )),
            ChannelData::Int64(a) => Arc::new(PrimitiveArray::new(
                DataType::Int64,
                Buffer::from_slice(a),
                bitmap,
            )),
            ChannelData::UInt64(a) => Arc::new(PrimitiveArray::new(
                DataType::UInt64,
                Buffer::from_slice(a),
                bitmap,
            )),
            ChannelData::Float64(a) => Arc::new(PrimitiveArray::new(
                DataType::Float64,
                Buffer::from_slice(a),
                bitmap,
            )),
            ChannelData::Complex16(a) => {
                let array = Arc::new(PrimitiveArray::new(
                    DataType::Float32,
                    Buffer::from_slice(&a.0),
                    None,
                ));
                let field = Field::new("complex64", DataType::Float32, bitmap.is_some());
                Arc::new(FixedSizeListArray::new(
                    DataType::FixedSizeList(Box::new(field), 2),
                    array as Arc<dyn Array>,
                    bitmap,
                ))
            }
            ChannelData::Complex32(a) => {
                let array = Arc::new(PrimitiveArray::new(
                    DataType::Float32,
                    Buffer::from_slice(&a.0),
                    None,
                ));
                let field = Field::new("complex64", DataType::Float32, bitmap.is_some());
                Arc::new(FixedSizeListArray::new(
                    DataType::FixedSizeList(Box::new(field), 2),
                    array as Arc<dyn Array>,
                    bitmap,
                ))
            }
            ChannelData::Complex64(a) => {
                let array = Arc::new(PrimitiveArray::new(
                    DataType::Float64,
                    Buffer::from_slice(&a.0),
                    None,
                ));
                let field = Field::new("complex128", DataType::Float64, bitmap.is_some());
                Arc::new(FixedSizeListArray::new(
                    DataType::FixedSizeList(Box::new(field), 2),
                    array as Arc<dyn Array>,
                    bitmap,
                ))
            }
            ChannelData::StringSBC(a) => {
                let array = Utf8Array::<i64>::from_slice(a.as_slice());
                Arc::new(array.with_validity(bitmap))
            }
            ChannelData::StringUTF8(a) => {
                let array = Utf8Array::<i64>::from_slice(a.as_slice());
                Arc::new(array.with_validity(bitmap))
            }
            ChannelData::StringUTF16(a) => {
                let array = Utf8Array::<i64>::from_slice(a.as_slice());
                Arc::new(array.with_validity(bitmap))
            }
            ChannelData::VariableSizeByteArray(a) => {
                let array = BinaryArray::<i64>::from_slice(a.as_slice());
                Arc::new(array.with_validity(bitmap))
            }
            ChannelData::FixedSizeByteArray(a) => Arc::new(FixedSizeBinaryArray::new(
                DataType::FixedSizeBinary(a.1),
                Buffer::from_slice(&a.0),
                bitmap,
            )),
            ChannelData::ArrayDInt8(_) => todo!(),
            ChannelData::ArrayDUInt8(_) => todo!(),
            ChannelData::ArrayDInt16(_) => todo!(),
            ChannelData::ArrayDUInt16(_) => todo!(),
            ChannelData::ArrayDFloat16(_) => todo!(),
            ChannelData::ArrayDInt24(_) => todo!(),
            ChannelData::ArrayDUInt24(_) => todo!(),
            ChannelData::ArrayDInt32(_) => todo!(),
            ChannelData::ArrayDUInt32(_) => todo!(),
            ChannelData::ArrayDFloat32(_) => todo!(),
            ChannelData::ArrayDInt48(_) => todo!(),
            ChannelData::ArrayDUInt48(_) => todo!(),
            ChannelData::ArrayDInt64(_) => todo!(),
            ChannelData::ArrayDUInt64(_) => todo!(),
            ChannelData::ArrayDFloat64(_) => todo!(),
            ChannelData::ArrayDComplex16(_) => todo!(),
            ChannelData::ArrayDComplex32(_) => todo!(),
            ChannelData::ArrayDComplex64(_) => todo!(),
        }
    }
}
