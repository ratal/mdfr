//! Converter of mdf version 3.x into mdf version 4.2
use crate::mdfinfo::mdfinfo3::{Cg3, Cn3, Dg3};
use crate::mdfreader::channel_data::ChannelData;
use crate::mdfreader::{DataSignature, MasterSignature};

use crate::export::tensor::{Order::RowMajor, Tensor};
use crate::mdfinfo::{
    mdfinfo3::MdfInfo3,
    mdfinfo4::{FhBlock, MdfInfo4},
};
use anyhow::{Context, Error, Result};
use arrow2::array::{
    Array, BinaryArray, FixedSizeBinaryArray, FixedSizeListArray, PrimitiveArray, Utf8Array,
};
use arrow2::bitmap::Bitmap;
use arrow2::buffer::Buffer;
use arrow2::datatypes::{DataType, Field};

/// Converts mdfinfo3 into mdfinfo4
pub fn convert3to4(mdf3: &MdfInfo3, file_name: &str) -> Result<MdfInfo4> {
    let n_channels = mdf3.get_channel_names_set().len();
    let mut mdf4 = MdfInfo4::new(file_name, n_channels);
    // FH
    let fh = FhBlock::default();
    mdf4.fh.push(fh);

    mdf3.dg.iter().try_for_each(
        |(_dg_block_position, dg): (&u32, &Dg3)| -> Result<(), Error> {
            dg.cg
                .iter()
                .try_for_each(|(_rec_id, cg): (&u16, &Cg3)| -> Result<(), Error> {
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
                                    shape: (vec![cycle_count; 1], RowMajor),
                                };
                                let master_signature = MasterSignature {
                                    master_channel: cg.master_channel_name.clone(),
                                    master_type: Some(1),
                                    master_flag: true,
                                };
                                mdf4.add_channel(
                                    master_channel_name.clone(),
                                    cn.data.take_to_arrow_array(None),
                                    data_signature,
                                    master_signature,
                                    unit,
                                    desc,
                                )
                                .context("Failed adding channel")?;
                            }
                        }
                    }
                    // then add other channels
                    cg.cn
                        .iter()
                        .filter(|(_rec_pos, cn)| {
                            Some(cn.unique_name.clone()) != cg.master_channel_name
                        })
                        .try_for_each(|(_rec_pos, cn): (&u32, &Cn3)| -> Result<(), Error> {
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
                                shape: (vec![cycle_count; 1], RowMajor),
                            };
                            let master_signature = MasterSignature {
                                master_channel: cg.master_channel_name.clone(),
                                master_type: Some(0),
                                master_flag: false,
                            };
                            mdf4.add_channel(
                                cn.unique_name.clone(),
                                cn.data.take_to_arrow_array(None),
                                data_signature,
                                master_signature,
                                unit,
                                desc,
                            )?;
                            Ok(())
                        })
                        .context("Failed adding channels")?;
                    Ok(())
                })?;
            Ok(())
        },
    )?;
    Ok(mdf4)
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

impl ChannelData {
    /// takes (or replace by default) the ChannelData array and returns an arrow array
    pub fn take_to_arrow_array(&self, bitmap: Option<Bitmap>) -> Box<dyn Array> {
        match self {
            ChannelData::Int8(a) => Box::new(PrimitiveArray::new(
                DataType::Int8,
                a.clone().into(),
                bitmap,
            )),
            ChannelData::UInt8(a) => Box::new(PrimitiveArray::new(
                DataType::UInt8,
                a.clone().into(),
                bitmap,
            )),
            ChannelData::Int16(a) => Box::new(PrimitiveArray::new(
                DataType::Int16,
                a.clone().into(),
                bitmap,
            )),
            ChannelData::UInt16(a) => Box::new(PrimitiveArray::new(
                DataType::UInt16,
                a.clone().into(),
                bitmap,
            )),
            ChannelData::Float16(a) => Box::new(PrimitiveArray::new(
                DataType::Float32,
                a.clone().into(),
                bitmap,
            )),
            ChannelData::Int24(a) => Box::new(PrimitiveArray::new(
                DataType::Int32,
                a.clone().into(),
                bitmap,
            )),
            ChannelData::UInt24(a) => Box::new(PrimitiveArray::new(
                DataType::UInt32,
                a.clone().into(),
                bitmap,
            )),
            ChannelData::Int32(a) => Box::new(PrimitiveArray::new(
                DataType::Int32,
                a.clone().into(),
                bitmap,
            )),
            ChannelData::UInt32(a) => Box::new(PrimitiveArray::new(
                DataType::UInt32,
                a.clone().into(),
                bitmap,
            )),
            ChannelData::Float32(a) => Box::new(PrimitiveArray::new(
                DataType::Float32,
                a.clone().into(),
                bitmap,
            )),
            ChannelData::Int48(a) => Box::new(PrimitiveArray::new(
                DataType::Int64,
                a.clone().into(),
                bitmap,
            )),
            ChannelData::UInt48(a) => Box::new(PrimitiveArray::new(
                DataType::UInt64,
                a.clone().into(),
                bitmap,
            )),
            ChannelData::Int64(a) => Box::new(PrimitiveArray::new(
                DataType::Int64,
                a.clone().into(),
                bitmap,
            )),
            ChannelData::UInt64(a) => Box::new(PrimitiveArray::new(
                DataType::UInt64,
                a.clone().into(),
                bitmap,
            )),
            ChannelData::Float64(a) => Box::new(PrimitiveArray::new(
                DataType::Float64,
                a.clone().into(),
                bitmap,
            )),
            ChannelData::Complex16(a) => {
                let is_nullable = bitmap.is_some();
                let array = Box::new(PrimitiveArray::from_vec(a.0.clone()));
                let field = Field::new("complex32", DataType::Float32, is_nullable);
                Box::new(FixedSizeListArray::new(
                    DataType::FixedSizeList(Box::new(field), 2),
                    array as Box<dyn Array>,
                    bitmap,
                ))
            }
            ChannelData::Complex32(a) => {
                let is_nullable = bitmap.is_some();
                let array = Box::new(PrimitiveArray::from_vec(a.0.clone()));
                let field = Field::new("complex32", DataType::Float32, is_nullable);
                Box::new(FixedSizeListArray::new(
                    DataType::FixedSizeList(Box::new(field), 2),
                    array as Box<dyn Array>,
                    bitmap,
                ))
            }
            ChannelData::Complex64(a) => {
                let is_nullable = bitmap.is_some();
                let array = Box::new(PrimitiveArray::from_vec(a.0.clone()));
                let field = Field::new("complex64", DataType::Float64, is_nullable);
                Box::new(FixedSizeListArray::new(
                    DataType::FixedSizeList(Box::new(field), 2),
                    array as Box<dyn Array>,
                    bitmap,
                ))
            }
            ChannelData::StringSBC(a) => {
                let array = Utf8Array::<i64>::from_slice(a.clone().as_slice());
                Box::new(array.with_validity(bitmap))
            }
            ChannelData::StringUTF8(a) => {
                let array = Utf8Array::<i64>::from_slice(a.clone().as_slice());
                Box::new(array.with_validity(bitmap))
            }
            ChannelData::StringUTF16(a) => {
                let array = Utf8Array::<i64>::from_slice(a.clone().as_slice());
                Box::new(array.with_validity(bitmap))
            }
            ChannelData::VariableSizeByteArray(a) => {
                let array = BinaryArray::<i64>::from_slice(a.clone().as_slice());
                Box::new(array.with_validity(bitmap))
            }
            ChannelData::FixedSizeByteArray(a) => Box::new(FixedSizeBinaryArray::new(
                DataType::FixedSizeBinary(a.1),
                Buffer::<u8>::from(a.clone().0),
                bitmap,
            )),
            ChannelData::ArrayDInt8(a) => Box::new(Tensor::new(
                DataType::Extension("Tensor".to_owned(), Box::new(DataType::Int8), None),
                Buffer::<i8>::from(a.0.clone()),
                Some(a.1 .0.clone().clone()),
                Some(a.1 .1.clone().into()),
                None,
                None,
            )),
            ChannelData::ArrayDUInt8(a) => Box::new(Tensor::new(
                DataType::Extension("Tensor".to_owned(), Box::new(DataType::UInt8), None),
                Buffer::<u8>::from(a.0.clone()),
                Some(a.1 .0.clone()),
                Some(a.1 .1.clone().into()),
                None,
                None,
            )),
            ChannelData::ArrayDInt16(a) => Box::new(Tensor::new(
                DataType::Extension("Tensor".to_owned(), Box::new(DataType::Int16), None),
                Buffer::<i16>::from(a.0.clone()),
                Some(a.1 .0.clone()),
                Some(a.1 .1.clone().into()),
                None,
                None,
            )),
            ChannelData::ArrayDUInt16(a) => Box::new(Tensor::new(
                DataType::Extension("Tensor".to_owned(), Box::new(DataType::UInt16), None),
                Buffer::<u16>::from(a.0.clone()),
                Some(a.1 .0.clone()),
                Some(a.1 .1.clone().into()),
                None,
                None,
            )),
            ChannelData::ArrayDFloat16(a) => Box::new(Tensor::new(
                DataType::Extension("Tensor".to_owned(), Box::new(DataType::Float32), None),
                Buffer::<f32>::from(a.0.clone()),
                Some(a.1 .0.clone()),
                Some(a.1 .1.clone().into()),
                None,
                None,
            )),
            ChannelData::ArrayDInt24(a) => Box::new(Tensor::new(
                DataType::Extension("Tensor".to_owned(), Box::new(DataType::Int32), None),
                Buffer::<i32>::from(a.0.clone()),
                Some(a.1 .0.clone()),
                Some(a.1 .1.clone().into()),
                None,
                None,
            )),
            ChannelData::ArrayDUInt24(a) => Box::new(Tensor::new(
                DataType::Extension("Tensor".to_owned(), Box::new(DataType::UInt32), None),
                Buffer::<u32>::from(a.0.clone()),
                Some(a.1 .0.clone()),
                Some(a.1 .1.clone().into()),
                None,
                None,
            )),
            ChannelData::ArrayDInt32(a) => Box::new(Tensor::new(
                DataType::Extension("Tensor".to_owned(), Box::new(DataType::Int32), None),
                Buffer::<i32>::from(a.0.clone()),
                Some(a.1 .0.clone()),
                Some(a.1 .1.clone().into()),
                None,
                None,
            )),
            ChannelData::ArrayDUInt32(a) => Box::new(Tensor::new(
                DataType::Extension("Tensor".to_owned(), Box::new(DataType::UInt32), None),
                Buffer::<u32>::from(a.0.clone()),
                Some(a.1 .0.clone()),
                Some(a.1 .1.clone().into()),
                None,
                None,
            )),
            ChannelData::ArrayDFloat32(a) => Box::new(Tensor::new(
                DataType::Extension("Tensor".to_owned(), Box::new(DataType::Float32), None),
                Buffer::<f32>::from(a.0.clone()),
                Some(a.1 .0.clone()),
                Some(a.1 .1.clone().into()),
                None,
                None,
            )),
            ChannelData::ArrayDInt48(a) => Box::new(Tensor::new(
                DataType::Extension("Tensor".to_owned(), Box::new(DataType::Int64), None),
                Buffer::<i64>::from(a.0.clone()),
                Some(a.1 .0.clone()),
                Some(a.1 .1.clone().into()),
                None,
                None,
            )),
            ChannelData::ArrayDUInt48(a) => Box::new(Tensor::new(
                DataType::Extension("Tensor".to_owned(), Box::new(DataType::UInt64), None),
                Buffer::<u64>::from(a.0.clone()),
                Some(a.1 .0.clone()),
                Some(a.1 .1.clone().into()),
                None,
                None,
            )),
            ChannelData::ArrayDInt64(a) => Box::new(Tensor::new(
                DataType::Extension("Tensor".to_owned(), Box::new(DataType::Int64), None),
                Buffer::<i64>::from(a.0.clone()),
                Some(a.1 .0.clone()),
                Some(a.1 .1.clone().into()),
                None,
                None,
            )),
            ChannelData::ArrayDUInt64(a) => Box::new(Tensor::new(
                DataType::Extension("Tensor".to_owned(), Box::new(DataType::UInt64), None),
                Buffer::<u64>::from(a.0.clone()),
                Some(a.1 .0.clone()),
                Some(a.1 .1.clone().into()),
                None,
                None,
            )),
            ChannelData::ArrayDFloat64(a) => Box::new(Tensor::new(
                DataType::Extension("Tensor".to_owned(), Box::new(DataType::Float64), None),
                Buffer::<f64>::from(a.0.clone()),
                Some(a.1 .0.clone()),
                Some(a.1 .1.clone().into()),
                None,
                None,
            )),
            ChannelData::ArrayDComplex16(_) => todo!(),
            ChannelData::ArrayDComplex32(_) => todo!(),
            ChannelData::ArrayDComplex64(_) => todo!(),
        }
    }
}
