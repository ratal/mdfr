//! Converts ndarray data in into arrow.
use crate::mdfinfo::MdfInfo;
use crate::mdfreader::channel_data::ChannelData;
use crate::mdfreader::{ChannelIndexes, Mdf};
use arrow2::array::{
    Array, BinaryArray, FixedSizeBinaryArray, FixedSizeListArray, PrimitiveArray, Utf8Array,
};
use arrow2::bitmap::Bitmap;
use arrow2::buffer::Buffer;
use arrow2::chunk::Chunk;
use arrow2::datatypes::{DataType, Field, Metadata, PhysicalType, PrimitiveType};
use arrow2::types::NativeType;
use arrow2::{array::ArrayRef, ffi};

use encoding::all::ISO_8859_1;
use encoding::{DecoderTrap, Encoding};
use pyo3::prelude::*;
use pyo3::{ffi::Py_uintptr_t, PyAny, PyObject, PyResult};
use rayon::iter::{IntoParallelIterator, ParallelExtend, ParallelIterator};
use std::collections::HashSet;
use std::mem;
use std::sync::Arc;

impl ChannelData {
    pub fn to_arrow_array(&mut self, bitmap: Option<Bitmap>) -> Arc<dyn Array> {
        match self {
            ChannelData::Int8(a) => Arc::new(PrimitiveArray::new(
                DataType::Int8,
                mem::take(a).into(),
                bitmap,
            )),
            ChannelData::UInt8(a) => Arc::new(PrimitiveArray::new(
                DataType::UInt8,
                mem::take(a).into(),
                bitmap,
            )),
            ChannelData::Int16(a) => Arc::new(PrimitiveArray::new(
                DataType::Int16,
                mem::take(a).into(),
                bitmap,
            )),
            ChannelData::UInt16(a) => Arc::new(PrimitiveArray::new(
                DataType::UInt16,
                mem::take(a).into(),
                bitmap,
            )),
            ChannelData::Float16(a) => Arc::new(PrimitiveArray::new(
                DataType::Float32,
                mem::take(a).into(),
                bitmap,
            )),
            ChannelData::Int24(a) => Arc::new(PrimitiveArray::new(
                DataType::Int32,
                mem::take(a).into(),
                bitmap,
            )),
            ChannelData::UInt24(a) => Arc::new(PrimitiveArray::new(
                DataType::UInt32,
                mem::take(a).into(),
                bitmap,
            )),
            ChannelData::Int32(a) => Arc::new(PrimitiveArray::new(
                DataType::Int32,
                mem::take(a).into(),
                bitmap,
            )),
            ChannelData::UInt32(a) => Arc::new(PrimitiveArray::new(
                DataType::UInt32,
                mem::take(a).into(),
                bitmap,
            )),
            ChannelData::Float32(a) => Arc::new(PrimitiveArray::new(
                DataType::Float32,
                mem::take(a).into(),
                bitmap,
            )),
            ChannelData::Int48(a) => Arc::new(PrimitiveArray::new(
                DataType::Int64,
                mem::take(a).into(),
                bitmap,
            )),
            ChannelData::UInt48(a) => Arc::new(PrimitiveArray::new(
                DataType::UInt64,
                mem::take(a).into(),
                bitmap,
            )),
            ChannelData::Int64(a) => Arc::new(PrimitiveArray::new(
                DataType::Int64,
                mem::take(a).into(),
                bitmap,
            )),
            ChannelData::UInt64(a) => Arc::new(PrimitiveArray::new(
                DataType::UInt64,
                mem::take(a).into(),
                bitmap,
            )),
            ChannelData::Float64(a) => Arc::new(PrimitiveArray::new(
                DataType::Float64,
                mem::take(a).into(),
                bitmap,
            )),
            ChannelData::Complex16(a) => {
                let is_nullable = bitmap.is_some();
                let array = Arc::new(PrimitiveArray::from_vec(mem::take(&mut a.0)));
                let field = Field::new("complex32", DataType::Float32, is_nullable);
                Arc::new(FixedSizeListArray::new(
                    DataType::FixedSizeList(Box::new(field), 2),
                    array as Arc<dyn Array>,
                    bitmap,
                ))
            }
            ChannelData::Complex32(a) => {
                let is_nullable = bitmap.is_some();
                let array = Arc::new(PrimitiveArray::from_vec(mem::take(&mut a.0)));
                let field = Field::new("complex32", DataType::Float32, is_nullable);
                Arc::new(FixedSizeListArray::new(
                    DataType::FixedSizeList(Box::new(field), 2),
                    array as Arc<dyn Array>,
                    bitmap,
                ))
            }
            ChannelData::Complex64(a) => {
                let is_nullable = bitmap.is_some();
                let array = Arc::new(PrimitiveArray::from_vec(mem::take(&mut a.0)));
                let field = Field::new("complex64", DataType::Float64, is_nullable);
                Arc::new(FixedSizeListArray::new(
                    DataType::FixedSizeList(Box::new(field), 2),
                    array as Arc<dyn Array>,
                    bitmap,
                ))
            }
            ChannelData::StringSBC(a) => {
                let array = Utf8Array::<i64>::from_slice(mem::take(a).as_slice());
                Arc::new(array.with_validity(bitmap))
            }
            ChannelData::StringUTF8(a) => {
                let array = Utf8Array::<i64>::from_slice(mem::take(a).as_slice());
                Arc::new(array.with_validity(bitmap))
            }
            ChannelData::StringUTF16(a) => {
                let array = Utf8Array::<i64>::from_slice(mem::take(a).as_slice());
                Arc::new(array.with_validity(bitmap))
            }
            ChannelData::VariableSizeByteArray(a) => {
                let array = BinaryArray::<i64>::from_slice(mem::take(a).as_slice());
                Arc::new(array.with_validity(bitmap))
            }
            ChannelData::FixedSizeByteArray(a) => Arc::new(FixedSizeBinaryArray::new(
                DataType::FixedSizeBinary(a.1),
                Buffer::<u8>::from(mem::take(a).0),
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

/// takes data of channel set from MdfInfo structure and stores in arrow_data
pub fn mdf_data_to_arrow(mdf: &mut Mdf, channel_names: &HashSet<String>) {
    let mut chunk_index: usize = 0;
    let mut array_index: usize = 0;
    let mut field_index: usize = 0;
    match &mut mdf.mdf_info {
        MdfInfo::V4(mdfinfo4) => {
            mdf.arrow_data = Vec::<Chunk<Arc<dyn Array>>>::with_capacity(mdfinfo4.dg.len());
            mdf.arrow_schema.fields = Vec::<Field>::with_capacity(mdfinfo4.channel_names_set.len());
            for (_dg_block_position, dg) in mdfinfo4.dg.iter_mut() {
                let mut channel_names_present_in_dg = HashSet::new();
                for channel_group in dg.cg.values() {
                    let cn = channel_group.channel_names.clone();
                    channel_names_present_in_dg.par_extend(cn);
                }
                let channel_names_to_read_in_dg: HashSet<_> = channel_names_present_in_dg
                    .into_par_iter()
                    .filter(|v| channel_names.contains(v))
                    .collect();
                if !channel_names_to_read_in_dg.is_empty() {
                    dg.cg.iter_mut().for_each(|(_rec_id, cg)| {
                        let is_nullable: bool = cg.block.cg_inval_bytes > 0;
                        let mut columns =
                            Vec::<Arc<dyn Array>>::with_capacity(cg.channel_names.len());
                        cg.cn.iter_mut().for_each(|(_rec_pos, cn)| {
                            if !cn.data.is_empty() {
                                let data: Arc<dyn Array>;
                                if let Some(bitmap) = mem::take(&mut cn.invalid_mask) {
                                    data = cn.data.to_arrow_array(Some(Bitmap::from(bitmap.0)));
                                } else {
                                    data = cn.data.to_arrow_array(None);
                                }
                                let field = Field::new(
                                    cn.unique_name.clone(),
                                    data.data_type().clone(),
                                    is_nullable,
                                );
                                columns.push(data);
                                let mut metadata = Metadata::new();
                                if let Some(unit) = mdfinfo4.sharable.get_tx(cn.block.cn_md_unit) {
                                    metadata.insert("unit".to_string(), unit);
                                };
                                if let Some(desc) = mdfinfo4.sharable.get_tx(cn.block.cn_md_comment)
                                {
                                    metadata.insert("description".to_string(), desc);
                                };
                                if let Some((
                                    master_channel_name,
                                    _dg_pos,
                                    (_cg_pos, _rec_idd),
                                    (_cn_pos, _rec_pos),
                                )) = mdfinfo4.channel_names_set.get(&cn.unique_name)
                                {
                                    if let Some(master_channel_name) = master_channel_name {
                                        metadata.insert(
                                            "master_channel".to_string(),
                                            master_channel_name.to_string(),
                                        );
                                    }
                                }
                                if cn.block.cn_type == 4 {
                                    metadata.insert(
                                        "sync_channel".to_string(),
                                        cn.block.cn_sync_type.to_string(),
                                    );
                                }
                                let field = field.with_metadata(metadata);
                                mdf.arrow_schema.fields.push(field);
                                mdf.channel_indexes.insert(
                                    cn.unique_name.clone(),
                                    ChannelIndexes {
                                        chunk_index,
                                        array_index,
                                        field_index,
                                    },
                                );
                                array_index += 1;
                                field_index += 1;
                            }
                        });
                        mdf.arrow_data.push(Chunk::new(columns));
                        chunk_index += 1;
                        array_index = 0;
                    });
                }
            }
        }
        MdfInfo::V3(mdfinfo3) => {
            mdf.arrow_data = Vec::<Chunk<Arc<dyn Array>>>::with_capacity(mdfinfo3.dg.len());
            mdf.arrow_schema.fields = Vec::<Field>::with_capacity(mdfinfo3.channel_names_set.len());
            for (_dg_block_position, dg) in mdfinfo3.dg.iter_mut() {
                for (_rec_id, cg) in dg.cg.iter_mut() {
                    let mut columns = Vec::<Arc<dyn Array>>::with_capacity(cg.channel_names.len());
                    for (_rec_pos, cn) in cg.cn.iter_mut() {
                        if !cn.data.is_empty() {
                            let data = cn.data.to_arrow_array(None);
                            let field =
                                Field::new(cn.unique_name.clone(), data.data_type().clone(), false);
                            columns.push(data);
                            let mut metadata = Metadata::new();
                            if let Some(array) =
                                mdfinfo3.sharable.cc.get(&cn.block1.cn_cc_conversion)
                            {
                                let txt = array.0.cc_unit;
                                let mut u = String::new();
                                ISO_8859_1
                                    .decode_to(&txt, DecoderTrap::Replace, &mut u)
                                    .expect("channel description is latin1 encoded");
                                metadata.insert(
                                    "unit".to_string(),
                                    u.trim_end_matches(char::from(0)).to_string(),
                                );
                            };
                            metadata.insert("description".to_string(), cn.description.clone());
                            if let Some((
                                master_channel_name,
                                _dg_pos,
                                (_cg_pos, _rec_idd),
                                _cn_pos,
                            )) = mdfinfo3.channel_names_set.get(&cn.unique_name)
                            {
                                if let Some(master_channel_name) = master_channel_name {
                                    metadata.insert(
                                        "master_channel".to_string(),
                                        master_channel_name.to_string(),
                                    );
                                }
                            }
                            let field = field.with_metadata(metadata);
                            mdf.arrow_schema.fields.push(field);
                            mdf.channel_indexes.insert(
                                cn.unique_name.clone(),
                                ChannelIndexes {
                                    chunk_index,
                                    array_index,
                                    field_index,
                                },
                            );
                            array_index += 1;
                            field_index += 1;
                        }
                    }
                    mdf.arrow_data.push(Chunk::new(columns));
                    chunk_index += 1;
                    array_index = 0;
                }
            }
        }
    }
}

/// Take an arrow array from python and convert it to a rust arrow array.
/// This operation does not copy data.
pub(crate) fn array_to_rust(py: Python, arrow_array: &Py<PyAny>) -> PyResult<ArrayRef> {
    // prepare a pointer to receive the Array struct
    let array = Box::new(ffi::ArrowArray::empty());
    let schema = Box::new(ffi::ArrowSchema::empty());

    let array_ptr = &*array as *const ffi::ArrowArray;
    let schema_ptr = &*schema as *const ffi::ArrowSchema;

    // make the conversion through PyArrow's private API
    // this changes the pointer's memory and is thus unsafe. In particular, `_export_to_c` can go out of bounds
    arrow_array.call_method1(
        py,
        "_export_to_c",
        (array_ptr as Py_uintptr_t, schema_ptr as Py_uintptr_t),
    )?;

    unsafe {
        let field = ffi::import_field_from_c(schema.as_ref()).unwrap();
        let array = ffi::import_array_from_c(array, field.data_type).unwrap();
        Ok(array.into())
    }
}

/// Arrow array to Python.
pub(crate) fn to_py_array(
    py: Python,
    pyarrow: &PyModule,
    array: ArrayRef,
    field: &Field,
) -> PyResult<PyObject> {
    let array_ptr = Box::new(ffi::ArrowArray::empty());
    let schema_ptr = Box::new(ffi::ArrowSchema::empty());

    let array_ptr = Box::into_raw(array_ptr);
    let schema_ptr = Box::into_raw(schema_ptr);

    unsafe {
        ffi::export_field_to_c(field, schema_ptr);
        ffi::export_array_to_c(array, array_ptr);
    };

    let array = pyarrow.getattr("Array")?.call_method1(
        "_import_from_c",
        (array_ptr as Py_uintptr_t, schema_ptr as Py_uintptr_t),
    )?;

    unsafe {
        Box::from_raw(array_ptr);
        Box::from_raw(schema_ptr);
    };

    Ok(array.to_object(py))
}

pub fn bit_count(array: &Arc<dyn Array>) -> u32 {
    match array.data_type() {
        DataType::Null => 0,
        DataType::Boolean => 8,
        DataType::Int8 => 8,
        DataType::Int16 => 16,
        DataType::Int32 => 32,
        DataType::Int64 => 64,
        DataType::UInt8 => 8,
        DataType::UInt16 => 16,
        DataType::UInt32 => 32,
        DataType::UInt64 => 64,
        DataType::Float16 => 16,
        DataType::Float32 => 32,
        DataType::Float64 => 64,
        DataType::Timestamp(_, _) => 64,
        DataType::Date32 => 32,
        DataType::Date64 => 64,
        DataType::Time32(_) => 32,
        DataType::Time64(_) => 64,
        DataType::Binary => {
            let array = array
                .as_any()
                .downcast_ref::<BinaryArray<i32>>()
                .expect("could not downcast to utf8 array");
            array
                .values_iter()
                .map(|s| s.len() as u32)
                .max()
                .unwrap_or(0)
                * 8
        }
        DataType::FixedSizeBinary(size) => 8 * *size as u32,
        DataType::LargeBinary => {
            let array = array
                .as_any()
                .downcast_ref::<BinaryArray<i64>>()
                .expect("could not downcast to utf8 array");
            array
                .values_iter()
                .map(|s| s.len() as u32)
                .max()
                .unwrap_or(0)
                * 8
        }
        DataType::Utf8 => {
            let array = array
                .as_any()
                .downcast_ref::<Utf8Array<i32>>()
                .expect("could not downcast to utf8 array");
            array
                .values_iter()
                .map(|s| s.len() as u32)
                .max()
                .unwrap_or(0)
                * 8
        }
        DataType::LargeUtf8 => {
            let array = array
                .as_any()
                .downcast_ref::<Utf8Array<i64>>()
                .expect("could not downcast to long utf8 array");
            array
                .values_iter()
                .map(|s| s.len() as u32)
                .max()
                .unwrap_or(0)
                * 8
        }
        DataType::FixedSizeList(field, size) => match field.data_type.to_physical_type() {
            PhysicalType::Primitive(PrimitiveType::Float32) => 32 * *size as u32,
            PhysicalType::Primitive(PrimitiveType::Float64) => 64 * *size as u32,
            _ => todo!(),
        },
        _ => panic!("unsupported type"),
    }
}

pub fn to_bytes(array: &Arc<dyn Array>) -> Vec<u8> {
    match array.data_type() {
        DataType::Null => Vec::new(),
        DataType::Boolean => {
            let array = array
                .as_any()
                .downcast_ref::<Bitmap>()
                .expect("could not downcast to Bitmap");
            array.iter().map(|v| v as u8).collect()
        }
        DataType::Int8 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i8>>()
                .expect("could not downcast to i8 array");
            array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect()
        }
        DataType::Int16 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i16>>()
                .expect("could not downcast to i16 array");
            array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect()
        }
        DataType::Int32 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i32>>()
                .expect("could not downcast to i32 array");
            array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect()
        }
        DataType::Int64 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i64>>()
                .expect("could not downcast to i64 array");
            array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect()
        }
        DataType::UInt8 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<u8>>()
                .expect("could not downcast to u8 array");
            array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect()
        }
        DataType::UInt16 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<u16>>()
                .expect("could not downcast to u16 array");
            array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect()
        }
        DataType::UInt32 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<u32>>()
                .expect("could not downcast to u32 array");
            array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect()
        }
        DataType::UInt64 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<u64>>()
                .expect("could not downcast to u64 array");
            array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect()
        }
        DataType::Float16 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<f32>>()
                .expect("could not downcast f16 to f32 array");
            array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect()
        }
        DataType::Float32 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<f32>>()
                .expect("could not downcast to f32 array");
            array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect()
        }
        DataType::Float64 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<f64>>()
                .expect("could not downcast to f64 array");
            array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect()
        }
        DataType::Timestamp(_, _) => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i64>>()
                .expect("could not downcast timestamp to i64 array");
            array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect()
        }
        DataType::Date32 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i32>>()
                .expect("could not downcast date32 to i32 array");
            array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect()
        }
        DataType::Date64 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i64>>()
                .expect("could not downcast date64 to i64 array");
            array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect()
        }
        DataType::Time32(_) => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i32>>()
                .expect("could not downcast time32 to i32 array");
            array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect()
        }
        DataType::Time64(_) => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i64>>()
                .expect("could not downcast time64 to i64 array");
            array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect()
        }
        DataType::Binary => {
            let array = array
                .as_any()
                .downcast_ref::<BinaryArray<i32>>()
                .expect("could not downcast binary array to bytes vect");
            let maxnbytes = array
                .values_iter()
                .map(|s| s.len() as u32)
                .max()
                .unwrap_or(0) as usize;
            array
                .values_iter()
                .flat_map(|x| {
                    let bytes = x.to_vec();
                    let n_bytes = bytes.len();
                    if maxnbytes > n_bytes {
                        [bytes, vec![0u8; maxnbytes - n_bytes]].concat()
                    } else {
                        bytes
                    }
                })
                .collect()
        }
        DataType::FixedSizeBinary(_) => {
            let array = array
                .as_any()
                .downcast_ref::<BinaryArray<i64>>()
                .expect("could not downcast large binary to bytes vect");
            array.values_iter().flat_map(|x| x.to_vec()).collect()
        }
        DataType::LargeBinary => {
            let array = array
                .as_any()
                .downcast_ref::<BinaryArray<i64>>()
                .expect("could not downcast large binary to bytes vect");
            let maxnbytes = array
                .values_iter()
                .map(|s| s.len() as u32)
                .max()
                .unwrap_or(0) as usize;
            array
                .values_iter()
                .flat_map(|x| {
                    let bytes = x.to_vec();
                    let n_bytes = bytes.len();
                    if maxnbytes > n_bytes {
                        [bytes, vec![0u8; maxnbytes - n_bytes]].concat()
                    } else {
                        bytes
                    }
                })
                .collect()
        }
        DataType::Utf8 => {
            let array = array
                .as_any()
                .downcast_ref::<Utf8Array<i32>>()
                .expect("could not downcast to utf8 array");
            let nbytes = array
                .values_iter()
                .map(|s| s.len() as u32)
                .max()
                .unwrap_or(0) as usize;
            array
                .values_iter()
                .flat_map(|x| {
                    let str_bytes = x.to_string().into_bytes();
                    let n_str_bytes = str_bytes.len();
                    if nbytes > n_str_bytes {
                        [str_bytes, vec![0u8; nbytes - n_str_bytes]].concat()
                    } else {
                        str_bytes
                    }
                })
                .collect()
        }
        DataType::LargeUtf8 => {
            let array = array
                .as_any()
                .downcast_ref::<Utf8Array<i64>>()
                .expect("could not downcast to long utf8 array");
            let nbytes = array
                .values_iter()
                .map(|s| s.len() as u32)
                .max()
                .unwrap_or(0) as usize;
            array
                .values_iter()
                .flat_map(|x| {
                    let str_bytes = x.to_string().into_bytes();
                    let n_str_bytes = str_bytes.len();
                    if nbytes > n_str_bytes {
                        [str_bytes, vec![0u8; nbytes - n_str_bytes]].concat()
                    } else {
                        str_bytes
                    }
                })
                .collect()
        }
        DataType::FixedSizeList(field, _size) => match field.data_type.to_physical_type() {
            PhysicalType::Primitive(PrimitiveType::Float32) => {
                let array = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<f32>>()
                    .expect("could not downcast to f32 array");
                array.values_iter().flat_map(|x| x.to_ne_bytes()).collect()
            }
            PhysicalType::Primitive(PrimitiveType::Float64) => {
                let array = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<f64>>()
                    .expect("could not downcast to f64 array");
                array.values_iter().flat_map(|x| x.to_ne_bytes()).collect()
            }
            _ => todo!(),
        },
        _ => panic!("unsupported type"),
    }
}
