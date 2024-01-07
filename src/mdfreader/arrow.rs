//! Converts ndarray data in into arrow.
use crate::export::tensor::Order;
use crate::export::tensor::Tensor;
use crate::mdfinfo::mdfinfo4::Cn4;
use crate::mdfinfo::mdfinfo4::MdfInfo4;
use anyhow::{bail, Context, Error};
// use crate::mdfinfo::MdfInfo;
// use crate::mdfreader::channel_data::ChannelData;
// use crate::mdfreader::Mdf;
use arrow2::array::{
    Array, BinaryArray, FixedSizeBinaryArray, FixedSizeListArray, MutableArray,
    MutableFixedSizeBinaryArray, PrimitiveArray, Utf8Array,
};
use arrow2::bitmap::Bitmap;
use arrow2::buffer::Buffer;
use arrow2::datatypes::{DataType, Field, Metadata, PhysicalType, PrimitiveType};
use arrow2::ffi;
use arrow2::offset::OffsetsBuffer;
use arrow2::types::f16;
// use codepage::to_encoding;
// use encoding_rs::Encoding;
use pyo3::prelude::*;
use pyo3::{ffi::Py_uintptr_t, PyAny, PyObject, PyResult};
// use rayon::iter::{IntoParallelIterator, ParallelExtend, ParallelIterator};
// use std::collections::HashSet;
// use std::mem;

// impl ChannelData {
//     /// takes (or replace by default) the ChannelData array and returns an arrow array
//     pub fn take_to_arrow_array(&mut self, bitmap: Option<Bitmap>) -> Box<dyn Array> {
//         match self {
//             ChannelData::Int8(a) => Box::new(PrimitiveArray::new(
//                 DataType::Int8,
//                 mem::take(a).into(),
//                 bitmap,
//             )),
//             ChannelData::UInt8(a) => Box::new(PrimitiveArray::new(
//                 DataType::UInt8,
//                 mem::take(a).into(),
//                 bitmap,
//             )),
//             ChannelData::Int16(a) => Box::new(PrimitiveArray::new(
//                 DataType::Int16,
//                 mem::take(a).into(),
//                 bitmap,
//             )),
//             ChannelData::UInt16(a) => Box::new(PrimitiveArray::new(
//                 DataType::UInt16,
//                 mem::take(a).into(),
//                 bitmap,
//             )),
//             ChannelData::Float16(a) => Box::new(PrimitiveArray::new(
//                 DataType::Float32,
//                 mem::take(a).into(),
//                 bitmap,
//             )),
//             ChannelData::Int24(a) => Box::new(PrimitiveArray::new(
//                 DataType::Int32,
//                 mem::take(a).into(),
//                 bitmap,
//             )),
//             ChannelData::UInt24(a) => Box::new(PrimitiveArray::new(
//                 DataType::UInt32,
//                 mem::take(a).into(),
//                 bitmap,
//             )),
//             ChannelData::Int32(a) => Box::new(PrimitiveArray::new(
//                 DataType::Int32,
//                 mem::take(a).into(),
//                 bitmap,
//             )),
//             ChannelData::UInt32(a) => Box::new(PrimitiveArray::new(
//                 DataType::UInt32,
//                 mem::take(a).into(),
//                 bitmap,
//             )),
//             ChannelData::Float32(a) => Box::new(PrimitiveArray::new(
//                 DataType::Float32,
//                 mem::take(a).into(),
//                 bitmap,
//             )),
//             ChannelData::Int48(a) => Box::new(PrimitiveArray::new(
//                 DataType::Int64,
//                 mem::take(a).into(),
//                 bitmap,
//             )),
//             ChannelData::UInt48(a) => Box::new(PrimitiveArray::new(
//                 DataType::UInt64,
//                 mem::take(a).into(),
//                 bitmap,
//             )),
//             ChannelData::Int64(a) => Box::new(PrimitiveArray::new(
//                 DataType::Int64,
//                 mem::take(a).into(),
//                 bitmap,
//             )),
//             ChannelData::UInt64(a) => Box::new(PrimitiveArray::new(
//                 DataType::UInt64,
//                 mem::take(a).into(),
//                 bitmap,
//             )),
//             ChannelData::Float64(a) => Box::new(PrimitiveArray::new(
//                 DataType::Float64,
//                 mem::take(a).into(),
//                 bitmap,
//             )),
//             ChannelData::Complex16(a) => {
//                 let is_nullable = bitmap.is_some();
//                 let array = Box::new(PrimitiveArray::from_vec(mem::take(&mut a.0)));
//                 let field = Field::new("complex32", DataType::Float32, is_nullable);
//                 Box::new(FixedSizeListArray::new(
//                     DataType::FixedSizeList(Box::new(field), 2),
//                     array as Box<dyn Array>,
//                     bitmap,
//                 ))
//             }
//             ChannelData::Complex32(a) => {
//                 let is_nullable = bitmap.is_some();
//                 let array = Box::new(PrimitiveArray::from_vec(mem::take(&mut a.0)));
//                 let field = Field::new("complex32", DataType::Float32, is_nullable);
//                 Box::new(FixedSizeListArray::new(
//                     DataType::FixedSizeList(Box::new(field), 2),
//                     array as Box<dyn Array>,
//                     bitmap,
//                 ))
//             }
//             ChannelData::Complex64(a) => {
//                 let is_nullable = bitmap.is_some();
//                 let array = Box::new(PrimitiveArray::from_vec(mem::take(&mut a.0)));
//                 let field = Field::new("complex64", DataType::Float64, is_nullable);
//                 Box::new(FixedSizeListArray::new(
//                     DataType::FixedSizeList(Box::new(field), 2),
//                     array as Box<dyn Array>,
//                     bitmap,
//                 ))
//             }
//             ChannelData::StringSBC(a) => {
//                 let array = Utf8Array::<i64>::from_slice(mem::take(a).as_slice());
//                 Box::new(array.with_validity(bitmap))
//             }
//             ChannelData::StringUTF8(a) => {
//                 let array = Utf8Array::<i64>::from_slice(mem::take(a).as_slice());
//                 Box::new(array.with_validity(bitmap))
//             }
//             ChannelData::StringUTF16(a) => {
//                 let array = Utf8Array::<i64>::from_slice(mem::take(a).as_slice());
//                 Box::new(array.with_validity(bitmap))
//             }
//             ChannelData::VariableSizeByteArray(a) => {
//                 let array = BinaryArray::<i64>::from_slice(mem::take(a).as_slice());
//                 Box::new(array.with_validity(bitmap))
//             }
//             ChannelData::FixedSizeByteArray(a) => Box::new(FixedSizeBinaryArray::new(
//                 DataType::FixedSizeBinary(a.1),
//                 Buffer::<u8>::from(mem::take(a).0),
//                 bitmap,
//             )),
//             ChannelData::ArrayDInt8(a) => {
//                 let a = mem::replace(a, (Vec::new(), (Vec::new(), Order::RowMajor)));
//                 Box::new(Tensor::try_new(
//                     DataType::Extension("Tensor".to_owned(), Box::new(DataType::Int8), None),
//                     Buffer::<i8>::from(a.0),
//                     Some(a.1 .0),
//                     Some(a.1 .1.into()),
//                     None,
//                     None,
//                 ))
//             }
//             ChannelData::ArrayDUInt8(a) => {
//                 let a = mem::replace(a, (Vec::new(), (Vec::new(), Order::RowMajor)));
//                 Box::new(Tensor::try_new(
//                     DataType::Extension("Tensor".to_owned(), Box::new(DataType::UInt8), None),
//                     Buffer::<u8>::from(a.0),
//                     Some(a.1 .0),
//                     Some(a.1 .1.into()),
//                     None,
//                     None,
//                 ))
//             }
//             ChannelData::ArrayDInt16(a) => {
//                 let a = mem::replace(a, (Vec::new(), (Vec::new(), Order::RowMajor)));
//                 Box::new(Tensor::try_new(
//                     DataType::Extension("Tensor".to_owned(), Box::new(DataType::Int16), None),
//                     Buffer::<i16>::from(a.0),
//                     Some(a.1 .0),
//                     Some(a.1 .1.into()),
//                     None,
//                     None,
//                 ))
//             }
//             ChannelData::ArrayDUInt16(a) => {
//                 let a = mem::replace(a, (Vec::new(), (Vec::new(), Order::RowMajor)));
//                 Box::new(Tensor::try_new(
//                     DataType::Extension("Tensor".to_owned(), Box::new(DataType::UInt16), None),
//                     Buffer::<u16>::from(a.0),
//                     Some(a.1 .0),
//                     Some(a.1 .1.into()),
//                     None,
//                     None,
//                 ))
//             }
//             ChannelData::ArrayDFloat16(a) => {
//                 let a = mem::replace(a, (Vec::new(), (Vec::new(), Order::RowMajor)));
//                 Box::new(Tensor::try_new(
//                     DataType::Extension("Tensor".to_owned(), Box::new(DataType::Float32), None),
//                     Buffer::<f32>::from(a.0),
//                     Some(a.1 .0),
//                     Some(a.1 .1.into()),
//                     None,
//                     None,
//                 ))
//             }
//             ChannelData::ArrayDInt24(a) => {
//                 let a = mem::replace(a, (Vec::new(), (Vec::new(), Order::RowMajor)));
//                 Box::new(Tensor::try_new(
//                     DataType::Extension("Tensor".to_owned(), Box::new(DataType::Int32), None),
//                     Buffer::<i32>::from(a.0),
//                     Some(a.1 .0),
//                     Some(a.1 .1.into()),
//                     None,
//                     None,
//                 ))
//             }
//             ChannelData::ArrayDUInt24(a) => {
//                 let a = mem::replace(a, (Vec::new(), (Vec::new(), Order::RowMajor)));
//                 Box::new(Tensor::try_new(
//                     DataType::Extension("Tensor".to_owned(), Box::new(DataType::UInt32), None),
//                     Buffer::<u32>::from(a.0),
//                     Some(a.1 .0),
//                     Some(a.1 .1.into()),
//                     None,
//                     None,
//                 ))
//             }
//             ChannelData::ArrayDInt32(a) => {
//                 let a = mem::replace(a, (Vec::new(), (Vec::new(), Order::RowMajor)));
//                 Box::new(Tensor::try_new(
//                     DataType::Extension("Tensor".to_owned(), Box::new(DataType::Int32), None),
//                     Buffer::<i32>::from(a.0),
//                     Some(a.1 .0),
//                     Some(a.1 .1.into()),
//                     None,
//                     None,
//                 ))
//             }
//             ChannelData::ArrayDUInt32(a) => {
//                 let a = mem::replace(a, (Vec::new(), (Vec::new(), Order::RowMajor)));
//                 Box::new(Tensor::try_new(
//                     DataType::Extension("Tensor".to_owned(), Box::new(DataType::UInt32), None),
//                     Buffer::<u32>::from(a.0),
//                     Some(a.1 .0),
//                     Some(a.1 .1.into()),
//                     None,
//                     None,
//                 ))
//             }
//             ChannelData::ArrayDFloat32(a) => {
//                 let a = mem::replace(a, (Vec::new(), (Vec::new(), Order::RowMajor)));
//                 Box::new(Tensor::try_new(
//                     DataType::Extension("Tensor".to_owned(), Box::new(DataType::Float32), None),
//                     Buffer::<f32>::from(a.0),
//                     Some(a.1 .0),
//                     Some(a.1 .1.into()),
//                     None,
//                     None,
//                 ))
//             }
//             ChannelData::ArrayDInt48(a) => {
//                 let a = mem::replace(a, (Vec::new(), (Vec::new(), Order::RowMajor)));
//                 Box::new(Tensor::try_new(
//                     DataType::Extension("Tensor".to_owned(), Box::new(DataType::Int64), None),
//                     Buffer::<i64>::from(a.0),
//                     Some(a.1 .0),
//                     Some(a.1 .1.into()),
//                     None,
//                     None,
//                 ))
//             }
//             ChannelData::ArrayDUInt48(a) => {
//                 let a = mem::replace(a, (Vec::new(), (Vec::new(), Order::RowMajor)));
//                 Box::new(Tensor::try_new(
//                     DataType::Extension("Tensor".to_owned(), Box::new(DataType::UInt64), None),
//                     Buffer::<u64>::from(a.0),
//                     Some(a.1 .0),
//                     Some(a.1 .1.into()),
//                     None,
//                     None,
//                 ))
//             }
//             ChannelData::ArrayDInt64(a) => {
//                 let a = mem::replace(a, (Vec::new(), (Vec::new(), Order::RowMajor)));
//                 Box::new(Tensor::try_new(
//                     DataType::Extension("Tensor".to_owned(), Box::new(DataType::Int64), None),
//                     Buffer::<i64>::from(a.0),
//                     Some(a.1 .0),
//                     Some(a.1 .1.into()),
//                     None,
//                     None,
//                 ))
//             }
//             ChannelData::ArrayDUInt64(a) => {
//                 let a = mem::replace(a, (Vec::new(), (Vec::new(), Order::RowMajor)));
//                 Box::new(Tensor::try_new(
//                     DataType::Extension("Tensor".to_owned(), Box::new(DataType::UInt64), None),
//                     Buffer::<u64>::from(a.0),
//                     Some(a.1 .0),
//                     Some(a.1 .1.into()),
//                     None,
//                     None,
//                 ))
//             }
//             ChannelData::ArrayDFloat64(a) => {
//                 let a = mem::replace(a, (Vec::new(), (Vec::new(), Order::RowMajor)));
//                 Box::new(Tensor::try_new(
//                     DataType::Extension("Tensor".to_owned(), Box::new(DataType::Float64), None),
//                     Buffer::<f64>::from(a.0),
//                     Some(a.1 .0),
//                     Some(a.1 .1.into()),
//                     None,
//                     None,
//                 ))
//             }
//             ChannelData::ArrayDComplex16(_) => todo!(),
//             ChannelData::ArrayDComplex32(_) => todo!(),
//             ChannelData::ArrayDComplex64(_) => todo!(),
//         }
//     }
// }

/// returns arrow field from cn
#[inline]
fn cn4_field(mdfinfo4: &MdfInfo4, cn: &Cn4, data_type: DataType, is_nullable: bool) -> Field {
    let field = Field::new(cn.unique_name.clone(), data_type, is_nullable);
    let mut metadata = Metadata::new();
    if let Ok(Some(unit)) = mdfinfo4.sharable.get_tx(cn.block.cn_md_unit) {
        metadata.insert("unit".to_string(), unit);
    };
    if let Ok(Some(desc)) = mdfinfo4.sharable.get_tx(cn.block.cn_md_comment) {
        metadata.insert("description".to_string(), desc);
    };
    if let Some((Some(master_channel_name), _dg_pos, (_cg_pos, _rec_idd), (_cn_pos, _rec_pos))) =
        mdfinfo4.channel_names_set.get(&cn.unique_name)
    {
        metadata.insert(
            "master_channel".to_string(),
            master_channel_name.to_string(),
        );
    }
    if cn.block.cn_type == 4 {
        metadata.insert(
            "sync_channel".to_string(),
            cn.block.cn_sync_type.to_string(),
        );
    }
    field.with_metadata(metadata)
}

// /// takes data of channel set from MdfInfo structure and stores in mdf.arrow_data
// pub fn mdf_data_to_arrow(mdf: &mut Mdf, channel_names: &HashSet<String>) {
//     let mut chunk_index: usize = 0;
//     let mut array_index: usize = 0;
//     let mut field_index: usize = 0;
//     match &mut mdf.mdf_info {
//         MdfInfo::V4(mdfinfo4) => {
//             mdf.arrow_data = Vec::<Vec<Box<dyn Array>>>::with_capacity(mdfinfo4.dg.len());
//             mdf.arrow_schema.fields = Vec::<Field>::with_capacity(mdfinfo4.channel_names_set.len());
//             for (_dg_block_position, dg) in mdfinfo4.dg.iter_mut() {
//                 let mut channel_names_present_in_dg = HashSet::new();
//                 for channel_group in dg.cg.values() {
//                     let cn = channel_group.channel_names.clone();
//                     channel_names_present_in_dg.par_extend(cn);
//                 }
//                 let channel_names_to_read_in_dg: HashSet<_> = channel_names_present_in_dg
//                     .into_par_iter()
//                     .filter(|v| channel_names.contains(v))
//                     .collect();
//                 if !channel_names_to_read_in_dg.is_empty() {
//                     dg.cg.iter_mut().for_each(|(_rec_id, cg)| {
//                         let is_nullable: bool = cg.block.cg_inval_bytes > 0;
//                         let mut columns =
//                             Vec::<Box<dyn Array>>::with_capacity(cg.channel_names.len());
//                         cg.cn.iter_mut().for_each(|(_rec_pos, cn)| {
//                             if !cn.data.is_empty() {
//                                 let data: Box<dyn Array>;
//                                 if let Some(bitmap) = mem::take(&mut cn.invalid_mask) {
//                                     data =
//                                         cn.data.take_to_arrow_array(Some(Bitmap::from(bitmap.0)));
//                                 } else {
//                                     data = cn.data.take_to_arrow_array(None);
//                                 }
//                                 // mdf.arrow_schema.fields.push(cn4_field(
//                                 //     mdfinfo4,
//                                 //     cn,
//                                 //     data.data_type().clone(),
//                                 //     is_nullable,
//                                 // ));
//                                 columns.push(data);
//                                 mdf.channel_indexes.insert(
//                                     cn.unique_name.clone(),
//                                     ChannelIndexes {
//                                         chunk_index,
//                                         array_index,
//                                         field_index,
//                                     },
//                                 );
//                                 array_index += 1;
//                                 field_index += 1;
//                             }
//                         });
//                         mdf.arrow_data.push(columns);
//                         chunk_index += 1;
//                         array_index = 0;
//                     });
//                 }
//             }
//         }
//         MdfInfo::V3(mdfinfo3) => {
//             mdf.arrow_data = Vec::<Vec<Box<dyn Array>>>::with_capacity(mdfinfo3.dg.len());
//             mdf.arrow_schema.fields = Vec::<Field>::with_capacity(mdfinfo3.channel_names_set.len());
//             for (_dg_block_position, dg) in mdfinfo3.dg.iter_mut() {
//                 for (_rec_id, cg) in dg.cg.iter_mut() {
//                     let mut columns = Vec::<Box<dyn Array>>::with_capacity(cg.channel_names.len());
//                     for (_rec_pos, cn) in cg.cn.iter_mut() {
//                         if !cn.data.is_empty() {
//                             let data = cn.data.take_to_arrow_array(None);
//                             let field =
//                                 Field::new(cn.unique_name.clone(), data.data_type().clone(), false);
//                             columns.push(data);
//                             let mut metadata = Metadata::new();
//                             if let Some(array) =
//                                 mdfinfo3.sharable.cc.get(&cn.block1.cn_cc_conversion)
//                             {
//                                 let txt = array.0.cc_unit;
//                                 let encoding: &'static Encoding =
//                                     to_encoding(mdfinfo3.id_block.id_codepage)
//                                         .unwrap_or(encoding_rs::WINDOWS_1252);
//                                 let u: String = encoding.decode(&txt).0.into();
//                                 metadata.insert(
//                                     "unit".to_string(),
//                                     u.trim_end_matches(char::from(0)).to_string(),
//                                 );
//                             };
//                             metadata.insert("description".to_string(), cn.description.clone());
//                             if let Some((
//                                 Some(master_channel_name),
//                                 _dg_pos,
//                                 (_cg_pos, _rec_idd),
//                                 _cn_pos,
//                             )) = mdfinfo3.channel_names_set.get(&cn.unique_name)
//                             {
//                                 metadata.insert(
//                                     "master_channel".to_string(),
//                                     master_channel_name.to_string(),
//                                 );
//                             }
//                             let field = field.with_metadata(metadata);
//                             mdf.arrow_schema.fields.push(field);
//                             mdf.channel_indexes.insert(
//                                 cn.unique_name.clone(),
//                                 ChannelIndexes {
//                                     chunk_index,
//                                     array_index,
//                                     field_index,
//                                 },
//                             );
//                             array_index += 1;
//                             field_index += 1;
//                         }
//                     }
//                     mdf.arrow_data.push(columns);
//                     chunk_index += 1;
//                     array_index = 0;
//                 }
//             }
//         }
//     }
// }

/// Take an arrow array from python and convert it to a rust arrow array.
/// This operation does not copy data.
#[allow(dead_code)]
pub fn array_to_rust(arrow_array: &PyAny) -> PyResult<Box<dyn Array>> {
    // prepare a pointer to receive the Array struct
    let array = Box::new(ffi::ArrowArray::empty());
    let schema = Box::new(ffi::ArrowSchema::empty());

    let array_ptr = &*array as *const ffi::ArrowArray;
    let schema_ptr = &*schema as *const ffi::ArrowSchema;

    // make the conversion through PyArrow's private API
    // this changes the pointer's memory and is thus unsafe. In particular, `_export_to_c` can go out of bounds
    arrow_array.call_method1(
        "_export_to_c",
        (array_ptr as Py_uintptr_t, schema_ptr as Py_uintptr_t),
    )?;

    unsafe {
        let field =
            ffi::import_field_from_c(schema.as_ref()).context("field import from C failed")?;
        let array = ffi::import_array_from_c(*array, field.data_type)
            .context("array import from C failed")?;
        Ok(array)
    }
}

/// Arrow array to Python.
pub(crate) fn to_py_array(
    py: Python,
    pyarrow: &PyModule,
    array: Box<dyn Array>,
) -> PyResult<PyObject> {
    let schema = Box::new(ffi::export_field_to_c(&Field::new(
        "",
        array.data_type().clone(),
        true,
    )));
    let array = Box::new(ffi::export_array_to_c(array));

    let schema_ptr: *const ffi::ArrowSchema = &*schema;
    let array_ptr: *const ffi::ArrowArray = &*array;

    let array = pyarrow.getattr("Array")?.call_method1(
        "_import_from_c",
        (array_ptr as Py_uintptr_t, schema_ptr as Py_uintptr_t),
    )?;

    Ok(array.to_object(py))
}

/// returns the number of bits corresponding to the array's datatype
pub fn arrow_bit_count(array: Box<dyn Array>) -> u32 {
    let data_type = array.data_type();
    bit_count(array.clone(), data_type)
}

fn bit_count(array: Box<dyn Array>, data_type: &DataType) -> u32 {
    match data_type {
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
                .expect("could not downcast to Binary array");
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
                .expect("could not downcast to LargeBinary array");
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
                .expect("could not downcast to large utf8 array");
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
            _ => panic!("unsupported type"),
        },
        DataType::Extension(ext_str, dtype, _) => match ext_str.as_str() {
            "Tensor" => bit_count(array, dtype),
            _ => panic!("unsupported type"),
        },
        _ => panic!("unsupported type"),
    }
}

/// returns the number of bytes corresponding to the array's datatype
pub fn arrow_byte_count(array: Box<dyn Array>) -> u32 {
    let data_type = array.data_type();
    byte_count(array.clone(), data_type)
}
fn byte_count(array: Box<dyn Array>, data_type: &DataType) -> u32 {
    match data_type {
        DataType::Null => 0,
        DataType::Boolean => 1,
        DataType::Int8 => 1,
        DataType::Int16 => 2,
        DataType::Int32 => 4,
        DataType::Int64 => 8,
        DataType::UInt8 => 1,
        DataType::UInt16 => 2,
        DataType::UInt32 => 4,
        DataType::UInt64 => 8,
        DataType::Float16 => 2,
        DataType::Float32 => 4,
        DataType::Float64 => 8,
        DataType::Timestamp(_, _) => 8,
        DataType::Date32 => 4,
        DataType::Date64 => 8,
        DataType::Time32(_) => 4,
        DataType::Time64(_) => 8,
        DataType::Binary => {
            let array = array
                .as_any()
                .downcast_ref::<BinaryArray<i32>>()
                .expect("could not downcast to binary array");
            array
                .values_iter()
                .map(|s| s.len() as u32)
                .max()
                .unwrap_or(0)
        }
        DataType::FixedSizeBinary(size) => 8 * *size as u32,
        DataType::LargeBinary => {
            let array = array
                .as_any()
                .downcast_ref::<BinaryArray<i64>>()
                .expect("could not downcast to large binary array");
            array
                .values_iter()
                .map(|s| s.len() as u32)
                .max()
                .unwrap_or(0)
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
        }
        DataType::LargeUtf8 => {
            let array = array
                .as_any()
                .downcast_ref::<Utf8Array<i64>>()
                .expect("could not downcast to large utf8 array");
            array
                .values_iter()
                .map(|s| s.len() as u32)
                .max()
                .unwrap_or(0)
        }
        DataType::FixedSizeList(field, size) => match field.data_type.to_physical_type() {
            PhysicalType::Primitive(PrimitiveType::Float32) => 4 * *size as u32,
            PhysicalType::Primitive(PrimitiveType::Float64) => 8 * *size as u32,
            _ => panic!("unsupported type"),
        },
        DataType::Extension(ext_str, dtype, _) => match ext_str.as_str() {
            "Tensor" => byte_count(array, dtype),
            _ => panic!("unsupported type"),
        },
        _ => panic!("unsupported type"),
    }
}

/// returns mdf4 data type from arrow array
pub fn arrow_to_mdf_data_type(array: Box<dyn Array>, endian: bool) -> u8 {
    mdf_data_type(array.data_type(), endian)
}

fn mdf_data_type(data_type: &DataType, endian: bool) -> u8 {
    if endian {
        // BE
        match data_type {
            DataType::Null => 1,
            DataType::Boolean => 1,
            DataType::Int8 => 3,
            DataType::Int16 => 3,
            DataType::Int32 => 3,
            DataType::Int64 => 3,
            DataType::UInt8 => 1,
            DataType::UInt16 => 1,
            DataType::UInt32 => 1,
            DataType::UInt64 => 1,
            DataType::Float16 => 5,
            DataType::Float32 => 5,
            DataType::Float64 => 5,
            DataType::Timestamp(_, _) => 3,
            DataType::Date32 => 3,
            DataType::Date64 => 3,
            DataType::Time32(_) => 3,
            DataType::Time64(_) => 3,
            DataType::Duration(_) => 3,
            DataType::Interval(_) => 3,
            DataType::Binary => 10,
            DataType::FixedSizeBinary(_) => 10,
            DataType::LargeBinary => 10,
            DataType::Utf8 => 7,
            DataType::LargeUtf8 => 7,
            DataType::List(_) => 16,
            DataType::FixedSizeList(_, _) => 16,
            DataType::LargeList(_) => 16,
            DataType::Extension(ext_str, dtype, _) => match ext_str.as_str() {
                "Tensor" => mdf_data_type(dtype, endian),
                _ => panic!("unsupported type"),
            },
            _ => panic!("unsupported type"),
        }
    } else {
        // LE
        match data_type {
            DataType::Null => 0,
            DataType::Boolean => 0,
            DataType::Int8 => 2,
            DataType::Int16 => 2,
            DataType::Int32 => 2,
            DataType::Int64 => 2,
            DataType::UInt8 => 0,
            DataType::UInt16 => 0,
            DataType::UInt32 => 0,
            DataType::UInt64 => 0,
            DataType::Float16 => 4,
            DataType::Float32 => 4,
            DataType::Float64 => 4,
            DataType::Timestamp(_, _) => 2,
            DataType::Date32 => 2,
            DataType::Date64 => 2,
            DataType::Time32(_) => 2,
            DataType::Time64(_) => 2,
            DataType::Duration(_) => 2,
            DataType::Interval(_) => 2,
            DataType::Binary => 10,
            DataType::FixedSizeBinary(_) => 10,
            DataType::LargeBinary => 10,
            DataType::Utf8 => 7,
            DataType::LargeUtf8 => 7,
            DataType::List(_) => 15,
            DataType::FixedSizeList(_, _) => 15,
            DataType::LargeList(_) => 15,
            DataType::Extension(ext_str, dtype, _) => match ext_str.as_str() {
                "Tensor" => mdf_data_type(dtype, endian),
                _ => panic!("unsupported type"),
            },
            _ => panic!("unsupported type"),
        }
    }
}

/// returns the number of dimensions of the channel
pub fn ndim(array: Box<dyn Array>) -> usize {
    match array.data_type() {
        DataType::Extension(ext_str, dtype, _) => match ext_str.as_str() {
            "Tensor" => match &**dtype {
                DataType::Int8 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<i8>>()
                        .expect("could not downcast to i8 array");
                    array.ndim()
                }
                DataType::Int16 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<i16>>()
                        .expect("could not downcast to i16 array");
                    array.ndim()
                }
                DataType::Int32 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<i32>>()
                        .expect("could not downcast to i32 array");
                    array.ndim()
                }
                DataType::Int64 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<i64>>()
                        .expect("could not downcast to i64 array");
                    array.ndim()
                }
                DataType::UInt8 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<u8>>()
                        .expect("could not downcast to u8 array");
                    array.ndim()
                }
                DataType::UInt16 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<u16>>()
                        .expect("could not downcast to u16 array");
                    array.ndim()
                }
                DataType::UInt32 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<u32>>()
                        .expect("could not downcast to u32 array");
                    array.ndim()
                }
                DataType::UInt64 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<u64>>()
                        .expect("could not downcast to u64 array");
                    array.ndim()
                }
                DataType::Float16 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<f16>>()
                        .expect("could not downcast to f16 array");
                    array.ndim()
                }
                DataType::Float32 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<f32>>()
                        .expect("could not downcast to f32 array");
                    array.ndim()
                }
                DataType::Float64 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<f64>>()
                        .expect("could not downcast to f64 array");
                    array.ndim()
                }
                DataType::Timestamp(_, _) => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<i64>>()
                        .expect("could not downcast to i64 array");
                    array.ndim()
                }
                DataType::Date32 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<i32>>()
                        .expect("could not downcast to i64 array");
                    array.ndim()
                }
                DataType::Date64 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<i64>>()
                        .expect("could not downcast to i64 array");
                    array.ndim()
                }
                DataType::Time32(_) => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<i32>>()
                        .expect("could not downcast to i64 array");
                    array.ndim()
                }
                DataType::Time64(_) => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<i64>>()
                        .expect("could not downcast to i64 array");
                    array.ndim()
                }
                DataType::Duration(_) => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<i64>>()
                        .expect("could not downcast to i64 array");
                    array.ndim()
                }
                DataType::Interval(_) => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<i64>>()
                        .expect("could not downcast to i64 array");
                    array.ndim()
                }
                DataType::FixedSizeList(_, _) => todo!(),
                _ => panic!("unsupported type"),
            },
            _ => panic!("unsupported type"),
        },
        _ => 1,
    }
}

// fn order_convert(tensor_order: &TensorOrder) -> Order {
//     match tensor_order {
//         TensorOrder::RowMajor => Order::RowMajor,
//         TensorOrder::ColumnMajor => Order::ColumnMajor,
//     }
// }

/// returns the number of dimensions of the channel
pub fn shape(array: Box<dyn Array>) -> (Vec<usize>, Order) {
    match array.data_type() {
        DataType::Extension(ext_str, dtype, _) => match ext_str.as_str() {
            "Tensor" => match &**dtype {
                DataType::Int8 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<i8>>()
                        .expect("could not downcast to i8 array");
                    (array.shape().to_vec(), array.order().clone())
                }
                DataType::Int16 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<i16>>()
                        .expect("could not downcast to i16 array");
                    (array.shape().to_vec(), array.order().clone())
                }
                DataType::Int32 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<i32>>()
                        .expect("could not downcast to i32 array");
                    (array.shape().to_vec(), array.order().clone())
                }
                DataType::Int64 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<i64>>()
                        .expect("could not downcast to i64 array");
                    (array.shape().to_vec(), array.order().clone())
                }
                DataType::UInt8 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<u8>>()
                        .expect("could not downcast to u8 array");
                    (array.shape().to_vec(), array.order().clone())
                }
                DataType::UInt16 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<u16>>()
                        .expect("could not downcast to u16 array");
                    (array.shape().to_vec(), array.order().clone())
                }
                DataType::UInt32 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<u32>>()
                        .expect("could not downcast to u32 array");
                    (array.shape().to_vec(), array.order().clone())
                }
                DataType::UInt64 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<u64>>()
                        .expect("could not downcast to u64 array");
                    (array.shape().to_vec(), array.order().clone())
                }
                DataType::Float16 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<f16>>()
                        .expect("could not downcast to f16 array");
                    (array.shape().to_vec(), array.order().clone())
                }
                DataType::Float32 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<f32>>()
                        .expect("could not downcast to f32 array");
                    (array.shape().to_vec(), array.order().clone())
                }
                DataType::Float64 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<f64>>()
                        .expect("could not downcast to f64 array");
                    (array.shape().to_vec(), array.order().clone())
                }
                DataType::Timestamp(_, _) => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<i64>>()
                        .expect("could not downcast to i64 array");
                    (array.shape().to_vec(), array.order().clone())
                }
                DataType::Date32 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<i32>>()
                        .expect("could not downcast to i64 array");
                    (array.shape().to_vec(), array.order().clone())
                }
                DataType::Date64 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<i64>>()
                        .expect("could not downcast to i64 array");
                    (array.shape().to_vec(), array.order().clone())
                }
                DataType::Time32(_) => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<i32>>()
                        .expect("could not downcast to i64 array");
                    (array.shape().to_vec(), array.order().clone())
                }
                DataType::Time64(_) => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<i64>>()
                        .expect("could not downcast to i64 array");
                    (array.shape().to_vec(), array.order().clone())
                }
                DataType::Duration(_) => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<i64>>()
                        .expect("could not downcast to i64 array");
                    (array.shape().to_vec(), array.order().clone())
                }
                DataType::Interval(_) => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Tensor<i64>>()
                        .expect("could not downcast to i64 array");
                    (array.shape().to_vec(), array.order().clone())
                }
                DataType::FixedSizeList(_, _) => todo!(),
                _ => panic!("unsupported type"),
            },
            _ => panic!("unsupported type"),
        },
        _ => (vec![array.len(); 1], Order::RowMajor),
    }
}

/// returns the a vec<u8>, bytes vector of arrow array
pub fn arrow_to_bytes(array: Box<dyn Array>) -> Vec<u8> {
    let data_type = array.data_type();
    to_bytes(array.clone(), data_type)
}

fn to_bytes(array: Box<dyn Array>, data_type: &DataType) -> Vec<u8> {
    match data_type {
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
            _ => panic!("unsupported FixedSizeList physical type"),
        },
        DataType::Extension(ext_str, dtype, _) => match ext_str.as_str() {
            "Tensor" => to_bytes(array, dtype),
            _ => panic!("unsupported extension type Tensor"),
        },
        _ => panic!("unsupported type"),
    }
}

/// Initialises a channel arrow array type depending of cn_type, cn_data_type and if array
pub fn arrow_data_type_init(
    cn_type: u8,
    cn_data_type: u8,
    n_bytes: u32,
    is_array: bool,
) -> Result<Box<dyn Array>, Error> {
    if !is_array {
        // Not an array
        if cn_type != 3 || cn_type != 6 {
            // not virtual channel or vlsd
            match cn_data_type {
                0 | 1 => {
                    // unsigned int
                    if n_bytes <= 1 {
                        Ok(PrimitiveArray::new(DataType::UInt8, Buffer::<u8>::new(), None).boxed())
                    } else if n_bytes == 2 {
                        Ok(
                            PrimitiveArray::new(DataType::UInt16, Buffer::<u16>::new(), None)
                                .boxed(),
                        )
                    } else if n_bytes <= 4 {
                        Ok(
                            PrimitiveArray::new(DataType::UInt32, Buffer::<u32>::new(), None)
                                .boxed(),
                        )
                    } else {
                        Ok(
                            PrimitiveArray::new(DataType::UInt64, Buffer::<u64>::new(), None)
                                .boxed(),
                        )
                    }
                }
                2 | 3 => {
                    // signed int
                    if n_bytes <= 1 {
                        Ok(PrimitiveArray::new(DataType::Int8, Buffer::<i8>::new(), None).boxed())
                    } else if n_bytes == 2 {
                        Ok(
                            PrimitiveArray::new(DataType::Int16, Buffer::<i16>::new(), None)
                                .boxed(),
                        )
                    } else if n_bytes <= 4 {
                        Ok(
                            PrimitiveArray::new(DataType::Int32, Buffer::<i32>::new(), None)
                                .boxed(),
                        )
                    } else {
                        Ok(
                            PrimitiveArray::new(DataType::Int64, Buffer::<i64>::new(), None)
                                .boxed(),
                        )
                    }
                }
                4 | 5 => {
                    // float
                    if n_bytes <= 4 {
                        Ok(
                            PrimitiveArray::new(DataType::Float32, Buffer::<f32>::new(), None)
                                .boxed(),
                        )
                    } else {
                        Ok(
                            PrimitiveArray::new(DataType::Float64, Buffer::<f64>::new(), None)
                                .boxed(),
                        )
                    }
                }
                15 | 16 => {
                    // complex
                    if n_bytes <= 4 {
                        let field = Field::new("complex32", DataType::Float32, false);
                        Ok(FixedSizeListArray::new(
                            DataType::FixedSizeList(Box::new(field), 2),
                            PrimitiveArray::new(DataType::Float32, Buffer::<f32>::new(), None)
                                .boxed(),
                            None,
                        )
                        .to_boxed())
                    } else {
                        let field = Field::new("complex64", DataType::Float64, false);
                        Ok(FixedSizeListArray::new(
                            DataType::FixedSizeList(Box::new(field), 2),
                            PrimitiveArray::new(DataType::Float64, Buffer::<f64>::new(), None)
                                .boxed(),
                            None,
                        )
                        .to_boxed())
                    }
                }
                6..=9 => {
                    // 6: SBC ISO-8859-1 to be converted into UTF8
                    // 7: String UTF8
                    // 8 & 9:String UTF16 to be converted into UTF8
                    Ok(Utf8Array::<i64>::new(
                        DataType::LargeUtf8,
                        OffsetsBuffer::new(),
                        Buffer::<u8>::new(),
                        None,
                    )
                    .boxed())
                }
                _ => {
                    // bytearray
                    if cn_type == 1 {
                        // VLSD
                        Ok(Utf8Array::<i64>::new(
                            DataType::LargeUtf8,
                            OffsetsBuffer::new(),
                            Buffer::<u8>::new(),
                            None,
                        )
                        .boxed())
                    } else {
                        Ok(MutableFixedSizeBinaryArray::new(n_bytes as usize).as_box())
                    }
                }
            }
        } else {
            // virtual channels, cn_bit_count = 0 -> n_bytes = 0, must be LE unsigned int
            Ok(PrimitiveArray::new(DataType::UInt64, Buffer::<u64>::new(), None).boxed())
        }
    } else if cn_type != 3 && cn_type != 6 {
        // Array, not virtual
        match cn_data_type {
            0 | 1 => {
                // unsigned int
                if n_bytes <= 1 {
                    Ok(Tensor::try_new(
                        DataType::Extension("Tensor".to_owned(), Box::new(DataType::UInt8), None),
                        Buffer::<u8>::new(),
                        Some(Vec::new()),
                        Some(Order::RowMajor),
                        None,
                        None,
                    )?
                    .to_boxed())
                } else if n_bytes == 2 {
                    Ok(Tensor::try_new(
                        DataType::Extension("Tensor".to_owned(), Box::new(DataType::UInt16), None),
                        Buffer::<u16>::new(),
                        Some(Vec::new()),
                        Some(Order::RowMajor),
                        None,
                        None,
                    )?
                    .to_boxed())
                } else if n_bytes <= 4 {
                    Ok(Tensor::try_new(
                        DataType::Extension("Tensor".to_owned(), Box::new(DataType::UInt32), None),
                        Buffer::<u32>::new(),
                        Some(Vec::new()),
                        Some(Order::RowMajor),
                        None,
                        None,
                    )?
                    .to_boxed())
                } else {
                    Ok(Tensor::try_new(
                        DataType::Extension("Tensor".to_owned(), Box::new(DataType::UInt64), None),
                        Buffer::<u64>::new(),
                        Some(Vec::new()),
                        Some(Order::RowMajor),
                        None,
                        None,
                    )?
                    .to_boxed())
                }
            }
            2 | 3 => {
                // signed int
                if n_bytes <= 1 {
                    Ok(Tensor::try_new(
                        DataType::Extension("Tensor".to_owned(), Box::new(DataType::Int8), None),
                        Buffer::<i8>::new(),
                        Some(Vec::new()),
                        Some(Order::RowMajor),
                        None,
                        None,
                    )?
                    .to_boxed())
                } else if n_bytes == 2 {
                    Ok(Tensor::try_new(
                        DataType::Extension("Tensor".to_owned(), Box::new(DataType::Int16), None),
                        Buffer::<i16>::new(),
                        Some(Vec::new()),
                        Some(Order::RowMajor),
                        None,
                        None,
                    )?
                    .to_boxed())
                } else if n_bytes <= 4 {
                    Ok(Tensor::try_new(
                        DataType::Extension("Tensor".to_owned(), Box::new(DataType::Int32), None),
                        Buffer::<i32>::new(),
                        Some(Vec::new()),
                        Some(Order::RowMajor),
                        None,
                        None,
                    )?
                    .to_boxed())
                } else {
                    Ok(Tensor::try_new(
                        DataType::Extension("Tensor".to_owned(), Box::new(DataType::Int64), None),
                        Buffer::<i64>::new(),
                        Some(Vec::new()),
                        Some(Order::RowMajor),
                        None,
                        None,
                    )?
                    .to_boxed())
                }
            }
            4 | 5 => {
                // float
                if n_bytes <= 4 {
                    Ok(Tensor::try_new(
                        DataType::Extension("Tensor".to_owned(), Box::new(DataType::Float32), None),
                        Buffer::<f32>::new(),
                        Some(Vec::new()),
                        Some(Order::RowMajor),
                        None,
                        None,
                    )?
                    .to_boxed())
                } else {
                    Ok(Tensor::try_new(
                        DataType::Extension("Tensor".to_owned(), Box::new(DataType::Float64), None),
                        Buffer::<f64>::new(),
                        Some(Vec::new()),
                        Some(Order::RowMajor),
                        None,
                        None,
                    )?
                    .to_boxed())
                }
            }
            15 | 16 => {
                // complex
                if n_bytes <= 4 {
                    let field = Field::new("complex32", DataType::Float32, false);
                    Ok(Tensor::try_new(
                        DataType::Extension(
                            "Tensor".to_owned(),
                            Box::new(DataType::FixedSizeList(Box::new(field), 2)),
                            None,
                        ),
                        Buffer::<f32>::new(),
                        Some(Vec::new()),
                        Some(Order::RowMajor),
                        None,
                        None,
                    )?
                    .to_boxed())
                } else {
                    let field = Field::new("complex64", DataType::Float64, false);
                    Ok(Tensor::try_new(
                        DataType::Extension(
                            "Tensor".to_owned(),
                            Box::new(DataType::FixedSizeList(Box::new(field), 2)),
                            None,
                        ),
                        Buffer::<f64>::new(),
                        Some(Vec::new()),
                        Some(Order::RowMajor),
                        None,
                        None,
                    )?
                    .to_boxed())
                }
            }
            _ => {
                // strings or bytes arrays not implemented for tensors, theoritically not possible from spec
                bail!("strings or bytes arrays not implemented for tensors, should it be ?");
            }
        }
    } else {
        // virtual channels arrays not implemented, can it even exists ?
        bail!("Virtual channel arrays not implemented, should it even exist ?");
    }
}

/// based on already existing type, rewrite the array filled with zeros at needed size based on cycle_count
pub fn arrow_init_zeros(
    data: &dyn Array,
    cn_type: u8,
    cycle_count: u64,
    shape: (Vec<usize>, Order),
) -> Result<Box<dyn Array>, Error> {
    if cn_type == 3 || cn_type == 6 {
        // virtual channels, cn_bit_count = 0 -> n_bytes = 0, must be LE unsigned int
        let mut array = vec![0u64; cycle_count as usize];
        array
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = i as u64);
        Ok(PrimitiveArray::from_vec(array).boxed())
    } else {
        match data.data_type() {
            DataType::Int8 => Ok(PrimitiveArray::from_vec(vec![0i8; cycle_count as usize]).boxed()),
            DataType::UInt8 => {
                Ok(PrimitiveArray::from_vec(vec![0u8; cycle_count as usize]).boxed())
            }
            DataType::Int16 => {
                Ok(PrimitiveArray::from_vec(vec![0i16; cycle_count as usize]).boxed())
            }
            DataType::UInt16 => {
                Ok(PrimitiveArray::from_vec(vec![0u16; cycle_count as usize]).boxed())
            }
            DataType::Int32 => {
                Ok(PrimitiveArray::from_vec(vec![0i32; cycle_count as usize]).boxed())
            }
            DataType::UInt32 => {
                Ok(PrimitiveArray::from_vec(vec![0u32; cycle_count as usize]).boxed())
            }
            DataType::Int64 => {
                Ok(PrimitiveArray::from_vec(vec![0i64; cycle_count as usize]).boxed())
            }
            DataType::UInt64 => {
                Ok(PrimitiveArray::from_vec(vec![0u64; cycle_count as usize]).boxed())
            }
            DataType::Float32 => {
                Ok(PrimitiveArray::from_vec(vec![0f32; cycle_count as usize]).boxed())
            }
            DataType::Float64 => {
                Ok(PrimitiveArray::from_vec(vec![0f64; cycle_count as usize]).boxed())
            }
            DataType::FixedSizeBinary(size) => Ok(FixedSizeBinaryArray::try_new(
                DataType::FixedSizeBinary(*size),
                Buffer::<u8>::from(vec![0u8; cycle_count as usize * size]),
                None,
            )
            .context("failed initialising FixedSizeBinaryArray with zeros")?
            .boxed()),
            DataType::FixedSizeList(field, size) => {
                if field.name.eq(&"complex32".to_string()) {
                    Ok(FixedSizeListArray::new(
                        DataType::FixedSizeList(field.clone(), *size),
                        PrimitiveArray::from_vec(vec![0f32; size * cycle_count as usize]).boxed(),
                        None,
                    )
                    .to_boxed())
                } else if field.name.eq(&"complex64".to_string()) {
                    Ok(FixedSizeListArray::new(
                        DataType::FixedSizeList(field.clone(), *size),
                        PrimitiveArray::from_vec(vec![0f64; size * cycle_count as usize]).boxed(),
                        None,
                    )
                    .to_boxed())
                } else {
                    bail!("fixed size list field name not understood")
                }
            }
            DataType::LargeUtf8 => {
                // 6: SBC ISO-8859-1 to be converted into UTF8
                // 7: String UTF8
                // 8 | 9 :String UTF16 to be converted into UTF8
                Ok(Utf8Array::<i64>::new_null(DataType::LargeUtf8, cycle_count as usize).boxed())
            }
            DataType::Extension(extension_name, data_type, _) => {
                if extension_name.eq(&"Tensor".to_string()) {
                    match *data_type.clone() {
                        DataType::Int8 => Ok(Tensor::try_new(
                            DataType::Extension(
                                "Tensor".to_owned(),
                                Box::new(DataType::Int8),
                                None,
                            ),
                            Buffer::from(vec![
                                0i8;
                                (cycle_count as usize)
                                    * shape.0.iter().product::<usize>()
                            ]),
                            Some(shape.0),
                            Some(shape.1),
                            None,
                            None,
                        )?
                        .to_boxed()),
                        DataType::UInt8 => Ok(Tensor::try_new(
                            DataType::Extension(
                                "Tensor".to_owned(),
                                Box::new(DataType::UInt8),
                                None,
                            ),
                            Buffer::from(vec![
                                0u8;
                                (cycle_count as usize)
                                    * shape.0.iter().product::<usize>()
                            ]),
                            Some(shape.0),
                            Some(shape.1),
                            None,
                            None,
                        )?
                        .to_boxed()),
                        DataType::Int16 => Ok(Tensor::try_new(
                            DataType::Extension(
                                "Tensor".to_owned(),
                                Box::new(DataType::Int16),
                                None,
                            ),
                            Buffer::from(vec![
                                0i16;
                                (cycle_count as usize)
                                    * shape.0.iter().product::<usize>()
                            ]),
                            Some(shape.0),
                            Some(shape.1),
                            None,
                            None,
                        )?
                        .to_boxed()),
                        DataType::UInt16 => Ok(Tensor::try_new(
                            DataType::Extension(
                                "Tensor".to_owned(),
                                Box::new(DataType::UInt16),
                                None,
                            ),
                            Buffer::from(vec![
                                0u16;
                                (cycle_count as usize)
                                    * shape.0.iter().product::<usize>()
                            ]),
                            Some(shape.0),
                            Some(shape.1),
                            None,
                            None,
                        )?
                        .to_boxed()),
                        DataType::Int32 => Ok(Tensor::try_new(
                            DataType::Extension(
                                "Tensor".to_owned(),
                                Box::new(DataType::Int32),
                                None,
                            ),
                            Buffer::from(vec![
                                0i32;
                                (cycle_count as usize)
                                    * shape.0.iter().product::<usize>()
                            ]),
                            Some(shape.0),
                            Some(shape.1),
                            None,
                            None,
                        )?
                        .to_boxed()),
                        DataType::UInt32 => Ok(Tensor::try_new(
                            DataType::Extension(
                                "Tensor".to_owned(),
                                Box::new(DataType::UInt32),
                                None,
                            ),
                            Buffer::from(vec![
                                0u32;
                                (cycle_count as usize)
                                    * shape.0.iter().product::<usize>()
                            ]),
                            Some(shape.0),
                            Some(shape.1),
                            None,
                            None,
                        )?
                        .to_boxed()),
                        DataType::Int64 => Ok(Tensor::try_new(
                            DataType::Extension(
                                "Tensor".to_owned(),
                                Box::new(DataType::Int64),
                                None,
                            ),
                            Buffer::from(vec![
                                0i64;
                                (cycle_count as usize)
                                    * shape.0.iter().product::<usize>()
                            ]),
                            Some(shape.0),
                            Some(shape.1),
                            None,
                            None,
                        )?
                        .to_boxed()),
                        DataType::UInt64 => Ok(Tensor::try_new(
                            DataType::Extension(
                                "Tensor".to_owned(),
                                Box::new(DataType::UInt64),
                                None,
                            ),
                            Buffer::from(vec![
                                0u64;
                                (cycle_count as usize)
                                    * shape.0.iter().product::<usize>()
                            ]),
                            Some(shape.0),
                            Some(shape.1),
                            None,
                            None,
                        )?
                        .to_boxed()),
                        DataType::Float32 => Ok(Tensor::try_new(
                            DataType::Extension(
                                "Tensor".to_owned(),
                                Box::new(DataType::Float32),
                                None,
                            ),
                            Buffer::from(vec![
                                0f32;
                                (cycle_count as usize)
                                    * shape.0.iter().product::<usize>()
                            ]),
                            Some(shape.0),
                            Some(shape.1),
                            None,
                            None,
                        )?
                        .to_boxed()),
                        DataType::Float64 => Ok(Tensor::try_new(
                            DataType::Extension(
                                "Tensor".to_owned(),
                                Box::new(DataType::Float64),
                                None,
                            ),
                            Buffer::from(vec![
                                0f64;
                                (cycle_count as usize)
                                    * shape.0.iter().product::<usize>()
                            ]),
                            Some(shape.0),
                            Some(shape.1),
                            None,
                            None,
                        )?
                        .to_boxed()),
                        DataType::FixedSizeList(field, size) => {
                            if field.name.eq(&"complex32".to_string()) {
                                Ok(FixedSizeListArray::new(
                                    DataType::FixedSizeList(field.clone(), size),
                                    PrimitiveArray::from_vec(vec![
                                        0f32;
                                        size * cycle_count as usize
                                            * shape
                                                .0
                                                .iter()
                                                .product::<usize>()
                                    ])
                                    .boxed(),
                                    None,
                                )
                                .to_boxed())
                            } else if field.name.eq(&"complex64".to_string()) {
                                Ok(FixedSizeListArray::new(
                                    DataType::FixedSizeList(field.clone(), size),
                                    PrimitiveArray::from_vec(vec![
                                        0f64;
                                        size * cycle_count as usize
                                            * shape
                                                .0
                                                .iter()
                                                .product::<usize>()
                                    ])
                                    .boxed(),
                                    None,
                                )
                                .to_boxed())
                            } else {
                                bail!("fixed size list field name not understood")
                            }
                        }
                        _ => bail!("Tensor data type {:?} not properly initialised", data_type),
                    }
                } else {
                    bail!("extension {} not properly initialised", extension_name)
                }
            }
            _ => bail!("data type {:?} not properly initialised", data.data_type()),
        }
    }
}
