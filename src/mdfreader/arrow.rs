//! Converts ndarray data in into arrow.
use crate::export::tensor::Order;
use crate::export::tensor::Tensor;
use anyhow::Context;
use anyhow::Error;
use arrow2::array::{Array, BinaryArray, FixedSizeBinaryArray, PrimitiveArray, Utf8Array};
use arrow2::bitmap::Bitmap;
use arrow2::datatypes::{DataType, Field, PhysicalType, PrimitiveType};
use arrow2::ffi;
use arrow2::types::f16;
use pyo3::prelude::*;
use pyo3::{ffi::Py_uintptr_t, PyAny, PyObject, PyResult};

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
        let field = ffi::import_field_from_c(schema.as_ref()).unwrap();
        let array = ffi::import_array_from_c(*array, field.data_type).unwrap();
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
pub fn arrow_bit_count(array: &dyn Array) -> u32 {
    let data_type = array.data_type();
    bit_count(array, data_type)
}

fn bit_count(array: &dyn Array, data_type: &DataType) -> u32 {
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
pub fn arrow_byte_count(array: &dyn Array) -> u32 {
    let data_type = array.data_type();
    byte_count(array, data_type)
}
fn byte_count(array: &dyn Array, data_type: &DataType) -> u32 {
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
                .expect("could not downcast to utf8 array");
            array
                .values_iter()
                .map(|s| s.len() as u32)
                .max()
                .unwrap_or(0)
        }
        DataType::FixedSizeBinary(size) => *size as u32,
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
                .expect("could not downcast to long utf8 array");
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
pub fn arrow_to_mdf_data_type(array: &dyn Array, endian: bool) -> u8 {
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
pub fn ndim(array: &dyn Array) -> usize {
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

/// returns the number of dimensions of the channel
pub fn shape(array: &dyn Array) -> (Vec<usize>, Order) {
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
pub fn arrow_to_bytes(array: Box<dyn Array>) -> Result<Vec<u8>, Error> {
    let data_type = array.data_type();
    to_bytes(array.clone(), data_type)
}

#[inline]
fn to_bytes(array: Box<dyn Array>, data_type: &DataType) -> Result<Vec<u8>, Error> {
    // returns native endian as defined in channel block with arrow_to_mdf_data_type()
    match data_type {
        DataType::Null => Ok(Vec::new()),
        DataType::Boolean => {
            let array = array
                .as_any()
                .downcast_ref::<Bitmap>()
                .context("could not downcast to Bitmap")?;
            Ok(array.iter().map(|v| v as u8).collect())
        }
        DataType::Int8 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i8>>()
                .context("could not downcast to i8 array")?;
            Ok(array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect())
        }
        DataType::Int16 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i16>>()
                .context("could not downcast to i16 array")?;
            Ok(array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect())
        }
        DataType::Int32 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i32>>()
                .context("could not downcast to i32 array")?;
            Ok(array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect())
        }
        DataType::Int64 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i64>>()
                .context("could not downcast to i64 array")?;
            Ok(array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect())
        }
        DataType::UInt8 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<u8>>()
                .context("could not downcast to u8 array")?;
            Ok(array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect())
        }
        DataType::UInt16 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<u16>>()
                .context("could not downcast to u16 array")?;
            Ok(array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect())
        }
        DataType::UInt32 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<u32>>()
                .context("could not downcast to u32 array")?;
            Ok(array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect())
        }
        DataType::UInt64 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<u64>>()
                .context("could not downcast to u64 array")?;
            Ok(array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect())
        }
        DataType::Float16 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<f32>>()
                .context("could not downcast f16 to f32 array")?;
            Ok(array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect())
        }
        DataType::Float32 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<f32>>()
                .context("could not downcast to f32 array")?;
            Ok(array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect())
        }
        DataType::Float64 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<f64>>()
                .context("could not downcast to f64 array")?;
            Ok(array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect())
        }
        DataType::Timestamp(_, _) => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i64>>()
                .context("could not downcast timestamp to i64 array")?;
            Ok(array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect())
        }
        DataType::Date32 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i32>>()
                .context("could not downcast date32 to i32 array")?;
            Ok(array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect())
        }
        DataType::Date64 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i64>>()
                .context("could not downcast date64 to i64 array")?;
            Ok(array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect())
        }
        DataType::Time32(_) => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i32>>()
                .context("could not downcast time32 to i32 array")?;
            Ok(array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect())
        }
        DataType::Time64(_) => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i64>>()
                .context("could not downcast time64 to i64 array")?;
            Ok(array
                .values()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect())
        }
        DataType::Binary => {
            let array = array
                .as_any()
                .downcast_ref::<BinaryArray<i32>>()
                .context("could not downcast binary array to bytes vect")?;
            let maxnbytes = array
                .values_iter()
                .map(|s| s.len() as u32)
                .max()
                .unwrap_or(0) as usize;
            Ok(array
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
                .collect())
        }
        DataType::FixedSizeBinary(_) => {
            let array = array
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .context("could not downcast large binary to bytes vect")?;
            Ok(array.values().to_vec())
        }
        DataType::LargeBinary => {
            let array = array
                .as_any()
                .downcast_ref::<BinaryArray<i64>>()
                .context("could not downcast large binary to bytes vect")?;
            let maxnbytes = array
                .values_iter()
                .map(|s| s.len() as u32)
                .max()
                .unwrap_or(0) as usize;
            Ok(array
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
                .collect())
        }
        DataType::Utf8 => {
            let array = array
                .as_any()
                .downcast_ref::<Utf8Array<i32>>()
                .context("could not downcast to utf8 array")?;
            let nbytes = array.values_iter().map(|s| s.len()).max().unwrap_or(0);
            Ok(array
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
                .collect())
        }
        DataType::LargeUtf8 => {
            let array = array
                .as_any()
                .downcast_ref::<Utf8Array<i64>>()
                .context("could not downcast to large utf8 array")?;
            let nbytes = array.values_iter().map(|s| s.len()).max().unwrap_or(0);
            Ok(array
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
                .collect())
        }
        DataType::FixedSizeList(field, _size) => match field.data_type.to_physical_type() {
            PhysicalType::Primitive(PrimitiveType::Float32) => {
                let array = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<f32>>()
                    .context("could not downcast to f32 array")?;
                Ok(array.values_iter().flat_map(|x| x.to_ne_bytes()).collect())
            }
            PhysicalType::Primitive(PrimitiveType::Float64) => {
                let array = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<f64>>()
                    .context("could not downcast to f64 array")?;
                Ok(array.values_iter().flat_map(|x| x.to_ne_bytes()).collect())
            }
            _ => panic!("unsupported FixedSizeList physical type"),
        },
        DataType::Extension(ext_str, dtype, _) => match ext_str.as_str() {
            "Tensor" => to_bytes(array, dtype),
            _ => panic!("unsupported type"),
        },
        _ => panic!("unsupported type"),
    }
}
