//! Converts ndarray data in into arrow.
use crate::mdfinfo::mdfinfo3::MdfInfo3;
use crate::mdfinfo::mdfinfo4::MdfInfo4;
use crate::mdfreader::channel_data::ChannelData;
use crate::mdfreader::{ChannelIndexes, Mdf};
use arrow2::array::{
    Array, BinaryArray, BooleanArray, FixedSizeBinaryArray, FixedSizeListArray, MutableArray,
    MutableFixedSizeListArray, MutablePrimitiveArray, PrimitiveArray, Utf8Array,
};
use arrow2::bitmap::Bitmap;
use arrow2::buffer::Buffer;
use arrow2::chunk::Chunk;
use arrow2::datatypes::{DataType, Field, Metadata};
use arrow2::{array::ArrayRef, ffi};

use pyo3::prelude::*;
use pyo3::{ffi::Py_uintptr_t, PyAny, PyObject, PyResult};
use rayon::iter::{IntoParallelIterator, ParallelExtend, ParallelIterator};
use std::collections::HashSet;
use std::mem;
use std::sync::Arc;

impl ChannelData {
    pub fn to_arrow_array(&mut self, bitmap: Option<Bitmap>) -> Arc<dyn Array> {
        match self {
            ChannelData::Boolean(a) => {
                let length = a.len();
                let bitmap_bools = Bitmap::from_u8_slice(a, length);
                Arc::new(BooleanArray::new(DataType::Boolean, bitmap_bools, bitmap))
            }
            ChannelData::Int8(a) => {
                Arc::new(PrimitiveArray::from_vec(mem::take(a)).with_validity(bitmap))
            }
            ChannelData::UInt8(a) => {
                Arc::new(PrimitiveArray::from_vec(mem::take(a)).with_validity(bitmap))
            }
            ChannelData::Int16(a) => {
                Arc::new(PrimitiveArray::from_vec(mem::take(a)).with_validity(bitmap))
            }
            ChannelData::UInt16(a) => {
                Arc::new(PrimitiveArray::from_vec(mem::take(a)).with_validity(bitmap))
            }
            ChannelData::Float16(a) => {
                Arc::new(PrimitiveArray::from_vec(mem::take(a)).with_validity(bitmap))
            }
            ChannelData::Int24(a) => {
                Arc::new(PrimitiveArray::from_vec(mem::take(a)).with_validity(bitmap))
            }
            ChannelData::UInt24(a) => {
                Arc::new(PrimitiveArray::from_vec(mem::take(a)).with_validity(bitmap))
            }
            ChannelData::Int32(a) => {
                Arc::new(PrimitiveArray::from_vec(mem::take(a)).with_validity(bitmap))
            }
            ChannelData::UInt32(a) => {
                Arc::new(PrimitiveArray::from_vec(mem::take(a)).with_validity(bitmap))
            }
            ChannelData::Float32(a) => {
                Arc::new(PrimitiveArray::from_vec(mem::take(a)).with_validity(bitmap))
            }
            ChannelData::Int48(a) => {
                Arc::new(PrimitiveArray::from_vec(mem::take(a)).with_validity(bitmap))
            }
            ChannelData::UInt48(a) => {
                Arc::new(PrimitiveArray::from_vec(mem::take(a)).with_validity(bitmap))
            }
            ChannelData::Int64(a) => {
                Arc::new(PrimitiveArray::from_vec(mem::take(a)).with_validity(bitmap))
            }
            ChannelData::UInt64(a) => mem::take(a).into_arc(),
            ChannelData::Float64(a) => mem::take(a).into_arc(),
            ChannelData::Complex16(a) => {
                let is_nullable = bitmap.is_some();
                let array = Arc::new(PrimitiveArray::from_vec(mem::take(&mut a.0)));
                let field = Field::new("complex64", DataType::Float32, is_nullable);
                Arc::new(FixedSizeListArray::new(
                    DataType::FixedSizeList(Box::new(field), 2),
                    array as Arc<dyn Array>,
                    bitmap,
                ))
            }
            ChannelData::Complex32(a) => {
                let is_nullable = bitmap.is_some();
                let array = Arc::new(PrimitiveArray::from_vec(mem::take(&mut a.0)));
                let field = Field::new("complex64", DataType::Float32, is_nullable);
                Arc::new(FixedSizeListArray::new(
                    DataType::FixedSizeList(Box::new(field), 2),
                    array as Arc<dyn Array>,
                    bitmap,
                ))
            }
            ChannelData::Complex64(a) => mem::replace(
                a,
                MutableFixedSizeListArray::new(MutablePrimitiveArray::<f64>::new(), 2),
            )
            .as_arc(),
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
                Buffer::<u8>::from(a.0),
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
pub fn mdf4_data_to_arrow(mdf: &mut Mdf, mdfinfo4: &mut MdfInfo4, channel_names: HashSet<String>) {
    let mut chunk_index: usize = 0;
    let mut array_index: usize = 0;
    let mut field_index: usize = 0;
    mdf.arrow_data = Vec::<Chunk<Arc<dyn Array>>>::with_capacity(mdfinfo4.dg.len());
    mdf.arrow_schema.fields = Vec::<Field>::with_capacity(mdfinfo4.channel_names_set.len());
    mdfinfo4.dg.iter_mut().for_each(|(_dg_block_position, dg)| {
        let channel_names_present_in_dg = HashSet::new();
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
                let is_nullable: bool = cg.invalid_bytes.is_some();
                let mut columns = Vec::<Arc<dyn Array>>::with_capacity(cg.channel_names.len());
                for (_rec_pos, cn) in cg.cn.iter_mut() {
                    if !cn.data.is_empty() {
                        let data = cn.data.to_arrow_array(None);
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
                        if let Some(desc) = mdfinfo4.sharable.get_tx(cn.block.cn_md_comment) {
                            metadata.insert("description".to_string(), desc);
                        };
                        if let Some(master_channel_name) =
                            mdfinfo4.get_channel_master(&cn.unique_name)
                        {
                            metadata.insert("master_channel".to_string(), master_channel_name);
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
                            cn.unique_name,
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
            });
        }
    });
}

/// takes data of channel set from MdfInfo structure and stores in arrow_data
pub fn mdf3_data_to_arrow(mdf: &mut Mdf, mdfinfo3: &mut MdfInfo3, channel_names: HashSet<String>) {
    let mut chunk_index: usize = 0;
    let mut array_index: usize = 0;
    let mut field_index: usize = 0;
    mdf.arrow_data = Vec::<Chunk<Arc<dyn Array>>>::with_capacity(mdfinfo3.dg.len());
    mdf.arrow_schema.fields = Vec::<Field>::with_capacity(mdfinfo3.channel_names_set.len());
    mdfinfo3.dg.iter_mut().for_each(|(_dg_block_position, dg)| {
        dg.cg.iter_mut().for_each(|(_rec_id, cg)| {
            let mut columns = Vec::<Arc<dyn Array>>::with_capacity(cg.channel_names.len());
            for (_rec_pos, cn) in cg.cn.iter_mut() {
                if !cn.data.is_empty() {
                    let data = cn.data.to_arrow_array(None);
                    let field = Field::new(cn.unique_name.clone(), data.data_type().clone(), false);
                    columns.push(data);
                    let mut metadata = Metadata::new();
                    if let Some(unit) = mdfinfo3._get_unit(&cn.block1.cn_cc_conversion) {
                        metadata.insert("unit".to_string(), unit);
                    };
                    metadata.insert("description".to_string(), cn.description);
                    if let Some(master_channel_name) = mdfinfo3.get_channel_master(&cn.unique_name)
                    {
                        metadata.insert("master_channel".to_string(), master_channel_name);
                    }
                    let field = field.with_metadata(metadata);
                    mdf.arrow_schema.fields.push(field);
                    mdf.channel_indexes.insert(
                        cn.unique_name,
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
        });
    });
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
