//! Converts ndarray data in into arrow.
use std::sync::Arc;

use arrow::array::{
    make_array, Array, ArrayData, BinaryArray, LargeBinaryArray, LargeStringArray, StringArray,
};
use arrow::datatypes::DataType;
use arrow::pyarrow::PyArrowType;
use pyo3::prelude::*;
use pyo3::PyResult;

/// Take an arrow array from python and convert it to a rust arrow array.
/// This operation does not copy data.
#[allow(dead_code)]
pub fn array_to_rust(arrow_array: &PyAny) -> PyResult<Arc<dyn Array>> {
    // prepare a pointer to receive the Array struct
    let array = arrow_array.0; // Extract from PyArrowType wrapper
    Ok(make_array(array))
}

/// Arrow array to Python.
pub(crate) fn to_py_array(py: Python, array: &dyn Array) -> PyResult<PyArrowType<ArrayData>> {
    Ok(array.into())
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
                .downcast_ref::<BinaryArray>()
                .expect("could not downcast to utf8 array");
            array
                .iter()
                .map(|s| s.unwrap_or_default().len() as u32)
                .max()
                .unwrap_or(0)
                * 8
        }
        DataType::FixedSizeBinary(size) => 8 * *size as u32,
        DataType::LargeBinary => {
            let array = array
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .expect("could not downcast to utf8 array");
            array
                .iter()
                .map(|s| s.unwrap_or_default().len() as u32)
                .max()
                .unwrap_or(0)
                * 8
        }
        DataType::Utf8 => {
            let array = array
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("could not downcast to utf8 array");
            array
                .iter()
                .map(|s| s.unwrap_or_default().len() as u32)
                .max()
                .unwrap_or(0)
                * 8
        }
        DataType::LargeUtf8 => {
            let array = array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("could not downcast to long utf8 array");
            array
                .iter()
                .map(|s| s.unwrap_or_default().len() as u32)
                .max()
                .unwrap_or(0)
                * 8
        }
        DataType::FixedSizeList(field, size) => match field.data_type() {
            DataType::Float32 => 32 * *size as u32,
            DataType::Float64 => 64 * *size as u32,
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
                .downcast_ref::<BinaryArray>()
                .expect("could not downcast to utf8 array");
            array
                .iter()
                .map(|s| s.unwrap_or_default().len() as u32)
                .max()
                .unwrap_or(0)
        }
        DataType::FixedSizeBinary(size) => *size as u32,
        DataType::LargeBinary => {
            let array = array
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .expect("could not downcast to utf8 array");
            array
                .iter()
                .map(|s| s.unwrap_or_default().len() as u32)
                .max()
                .unwrap_or(0)
        }
        DataType::Utf8 => {
            let array = array
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("could not downcast to utf8 array");
            array
                .iter()
                .map(|s| s.unwrap_or_default().len() as u32)
                .max()
                .unwrap_or(0)
        }
        DataType::LargeUtf8 => {
            let array = array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("could not downcast to long utf8 array");
            array
                .iter()
                .map(|s| s.unwrap_or_default().len() as u32)
                .max()
                .unwrap_or(0)
        }
        DataType::FixedSizeList(field, size) => match field.data_type() {
            DataType::Float32 => 4 * *size as u32,
            DataType::Float64 => 8 * *size as u32,
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
            _ => panic!("unsupported type"),
        }
    }
}
