//! Converts ndarray data in into arrow.
use std::mem::size_of;

use anyhow::{bail, Error};

use arrow::array::{
    make_array, Array, ArrayBuilder, ArrayData, BinaryArray, BinaryBuilder, FixedSizeBinaryBuilder,
    FixedSizeListBuilder, LargeBinaryBuilder, LargeStringBuilder, PrimitiveBuilder, StringBuilder,
};
use arrow::buffer::{MutableBuffer, NullBuffer};
use arrow::datatypes::{
    DataType, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type,
    UInt32Type, UInt64Type, UInt8Type,
};
use arrow::pyarrow::PyArrowType;
use pyo3::prelude::*;
use pyo3::PyResult;

/// Order of the array, Row or Column Major (first)
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Order {
    #[default]
    RowMajor,
    ColumnMajor,
}

/// Take an arrow array from python and convert it to a rust arrow array.
/// This operation does not copy data.
#[allow(dead_code)]
pub fn array_to_rust(arrow_array: PyArrowType<ArrayData>) -> PyResult<Box<dyn Array>> {
    // prepare a pointer to receive the Array struct
    let array = arrow_array.0; // Extract from PyArrowType wrapper
    Ok(make_array(array))
}

pub(crate) fn to_py_array(_: Python, array: Box<dyn Array>) -> PyResult<PyArrowType<ArrayData>> {
    Ok(PyArrowType(array.into_data()))
}

/// returns the number of bits corresponding to the array's datatype
pub fn arrow_bit_count(array: Box<dyn ArrayBuilder>) -> u32 {
    let data_type = array.data_type();
    bit_count(array.clone(), data_type)
}

fn bit_count(array: Box<dyn ArrayBuilder>, data_type: &DataType) -> u32 {
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
                .downcast_ref::<BinaryBuilder>()
                .expect("could not downcast to Binary array");
            (array
                .offsets_slice()
                .iter()
                .tuple_windows::<(_, _)>()
                .map(|w| w.1 - w.0)
                .max()
                .unwrap_or(0)
                * 8) as u32
        }
        DataType::FixedSizeBinary(size) => 8 * *size as u32,
        DataType::LargeBinary => {
            let array = array
                .as_any()
                .downcast_ref::<LargeBinaryBuilder>()
                .expect("could not downcast to LargeBinary array");
            (array
                .offsets_slice()
                .iter()
                .tuple_windows::<(_, _)>()
                .map(|w| w.1 - w.0)
                .max()
                .unwrap_or(0)
                * 8) as u32
        }
        DataType::Utf8 => {
            let array = array
                .as_any()
                .downcast_ref::<StringBuilder>()
                .expect("could not downcast to utf8 array");
            (array
                .offsets_slice()
                .iter()
                .tuple_windows::<(_, _)>()
                .map(|w| w.1 - w.0)
                .max()
                .unwrap_or(0)
                * 8) as u32
        }
        DataType::LargeUtf8 => {
            let array = array
                .as_any()
                .downcast_ref::<LargeStringBuilder>()
                .expect("could not downcast to large utf8 array");
            (array
                .offsets_slice()
                .iter()
                .tuple_windows::<(_, _)>()
                .map(|w| w.1 - w.0)
                .max()
                .unwrap_or(0)
                * 8) as u32
        }
        DataType::FixedSizeList(field, size) => bit_count(array.values(), data_type),
        _ => panic!("unsupported type"),
    }
}

/// returns the number of bytes corresponding to the array's datatype
pub fn arrow_byte_count(array: Box<dyn ArrayBuilder>) -> u32 {
    let data_type = array.data_type();
    byte_count(array.clone(), data_type)
}
fn byte_count(array: Box<dyn ArrayBuilder>, data_type: &DataType) -> u32 {
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
                .downcast_ref::<BinaryBuilder>()
                .expect("could not downcast to binary array");
            (array
                .offsets_slice()
                .iter()
                .tuple_windows::<(_, _)>()
                .map(|w| w.1 - w.0)
                .max()
                .unwrap_or(0)) as u32
        }
        DataType::FixedSizeBinary(size) => *size as u32,
        DataType::LargeBinary => {
            let array = array
                .as_any()
                .downcast_ref::<LargeBinaryBuilder>()
                .expect("could not downcast to large binary array");
            (array
                .offsets_slice()
                .iter()
                .tuple_windows::<(_, _)>()
                .map(|w| w.1 - w.0)
                .max()
                .unwrap_or(0)) as u32
        }
        DataType::Utf8 => {
            let array = array
                .as_any()
                .downcast_ref::<StringBuilder>()
                .expect("could not downcast to utf8 array");
            (array
                .offsets_slice()
                .iter()
                .tuple_windows::<(_, _)>()
                .map(|w| w.1 - w.0)
                .max()
                .unwrap_or(0)) as u32
        }
        DataType::LargeUtf8 => {
            let array = array
                .as_any()
                .downcast_ref::<LargeStringBuilder>()
                .expect("could not downcast to large utf8 array");
            (array
                .offsets_slice()
                .iter()
                .tuple_windows::<(_, _)>()
                .map(|w| w.1 - w.0)
                .max()
                .unwrap_or(0)) as u32
        }
        DataType::FixedSizeList(field, size) => byte_count(array.values(), data_type),
        _ => panic!("unsupported type"),
    }
}

/// returns mdf4 data type from arrow array
pub fn arrow_to_mdf_data_type(array: Box<dyn ArrayBuilder>, endian: bool) -> u8 {
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

/// returns the a vec<u8>, bytes vector of arrow array
pub fn arrow_to_bytes(array: Box<dyn ArrayBuilder>) -> Vec<u8> {
    let data_type = array.data_type();
    to_bytes(array.clone(), data_type)
}

fn to_bytes(array: Box<dyn ArrayBuilder>, data_type: &DataType) -> Vec<u8> {
    match data_type {
        DataType::Null => Vec::new(),
        DataType::Boolean => {
            let array = array
                .as_any()
                .downcast_ref::<NullBuffer>()
                .expect("could not downcast to Bitmap");
            array.iter().map(|v| v as u8).collect()
        }
        DataType::Int8 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveBuilder<i8>>()
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
                .downcast_ref::<PrimitiveBuilder<i16>>()
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
                .downcast_ref::<PrimitiveBuilder<i32>>()
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
                .downcast_ref::<PrimitiveBuilder<i64>>()
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
                .downcast_ref::<PrimitiveBuilder<u8>>()
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
                .downcast_ref::<PrimitiveBuilder<u16>>()
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
                .downcast_ref::<PrimitiveBuilder<u32>>()
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
                .downcast_ref::<PrimitiveBuilder<u64>>()
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
                .downcast_ref::<PrimitiveBuilder<f32>>()
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
                .downcast_ref::<PrimitiveBuilder<f32>>()
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
                .downcast_ref::<PrimitiveBuilder<f64>>()
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
                .downcast_ref::<PrimitiveBuilder<i64>>()
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
                .downcast_ref::<PrimitiveBuilder<i32>>()
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
                .downcast_ref::<PrimitiveBuilder<i64>>()
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
                .downcast_ref::<PrimitiveBuilder<i32>>()
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
                .downcast_ref::<PrimitiveBuilder<i64>>()
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
                .downcast_ref::<BinaryBuilder>()
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
                .downcast_ref::<FixedSizeBinaryBuilder>()
                .expect("could not downcast large binary to bytes vect");
            array.values_iter().flat_map(|x| x.to_vec()).collect()
        }
        DataType::LargeBinary => {
            let array = array
                .as_any()
                .downcast_ref::<LargeBinaryBuilder>()
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
                .downcast_ref::<StringBuilder>()
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
                .downcast_ref::<LargeStringBuilder>()
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
        DataType::FixedSizeList(field, _size) => to_bytes(array.values(), data_type),
        _ => panic!("unsupported type"),
    }
}

/// Initialises a channel arrow array type depending of cn_type, cn_data_type and if array
pub fn arrow_data_type_init(
    cn_type: u8,
    cn_data_type: u8,
    n_bytes: u32,
    list_size: usize,
) -> Result<Box<dyn ArrayBuilder>, Error> {
    if list_size == 1 {
        // Not an array
        if cn_type != 3 || cn_type != 6 {
            // not virtual channel or vlsd
            match cn_data_type {
                0 | 1 => {
                    // unsigned int
                    if n_bytes <= 1 {
                        Ok(PrimitiveBuilder::<UInt8Type>::new().into_box_any())
                    } else if n_bytes == 2 {
                        Ok(PrimitiveBuilder::<UInt16Type>::new().into_box_any())
                    } else if n_bytes <= 4 {
                        Ok(PrimitiveBuilder::<UInt32Type>::new().into_box_any())
                    } else {
                        Ok(PrimitiveBuilder::<UInt64Type>::new().into_box_any())
                    }
                }
                2 | 3 => {
                    // signed int
                    if n_bytes <= 1 {
                        Ok(PrimitiveBuilder::<Int8Type>::new().into_box_any())
                    } else if n_bytes == 2 {
                        Ok(PrimitiveBuilder::<Int16Type>::new().into_box_any())
                    } else if n_bytes <= 4 {
                        Ok(PrimitiveBuilder::<Int32Type>::new().into_box_any())
                    } else {
                        Ok(PrimitiveBuilder::<Int64Type>::new().into_box_any())
                    }
                }
                4 | 5 => {
                    // float
                    if n_bytes <= 4 {
                        Ok(PrimitiveBuilder::<Float32Type>::new().into_box_any())
                    } else {
                        Ok(PrimitiveBuilder::<Float64Type>::new().into_box_any())
                    }
                }
                15 | 16 => {
                    // complex, should not happen as list_size == 2
                    if n_bytes <= 4 {
                        Ok(FixedSizeListBuilder::new(
                            PrimitiveBuilder::<Float32Type>::new().into_box_any(),
                            2,
                        )
                        .into_box_any())
                    } else {
                        Ok(FixedSizeListBuilder::new(
                            PrimitiveBuilder::<Float64Type>::new().into_box_any(),
                            2,
                        )
                        .into_box_any())
                    }
                }
                6..=9 => {
                    // 6: SBC ISO-8859-1 to be converted into UTF8
                    // 7: String UTF8
                    // 8 & 9:String UTF16 to be converted into UTF8
                    Ok(LargeStringBuilder::new().into_box_any())
                }
                _ => {
                    // bytearray
                    if cn_type == 1 {
                        // VLSD
                        Ok(LargeBinaryBuilder::new().into_box_any())
                    } else {
                        Ok(FixedSizeBinaryBuilder::new(n_bytes as usize).into_box_any())
                    }
                }
            }
        } else {
            // virtual channels, cn_bit_count = 0 -> n_bytes = 0, must be LE unsigned int
            Ok(Ok(PrimitiveBuilder::<UInt64Type>::new().into_box_any()))
        }
    } else if cn_type != 3 && cn_type != 6 {
        // Array or complex, not virtual
        match cn_data_type {
            0 | 1 => {
                // unsigned int
                if n_bytes <= 1 {
                    Ok(FixedSizeListBuilder::new(
                        PrimitiveBuilder::<UInt8Type>::new().into_box_any(),
                        list_size,
                    )
                    .into_box_any())
                } else if n_bytes == 2 {
                    Ok(FixedSizeListBuilder::new(
                        PrimitiveBuilder::<UInt16Type>::new().into_box_any(),
                        list_size,
                    )
                    .into_box_any())
                } else if n_bytes <= 4 {
                    Ok(FixedSizeListBuilder::new(
                        PrimitiveBuilder::<UInt32Type>::new().into_box_any(),
                        list_size,
                    )
                    .into_box_any())
                } else {
                    Ok(FixedSizeListBuilder::new(
                        PrimitiveBuilder::<UInt64Type>::new().into_box_any(),
                        list_size,
                    )
                    .into_box_any())
                }
            }
            2 | 3 => {
                // signed int
                if n_bytes <= 1 {
                    Ok(FixedSizeListBuilder::new(
                        PrimitiveBuilder::<Int8Type>::new().into_box_any(),
                        list_size,
                    )
                    .into_box_any())
                } else if n_bytes == 2 {
                    Ok(FixedSizeListBuilder::new(
                        PrimitiveBuilder::<Int16Type>::new().into_box_any(),
                        list_size,
                    )
                    .into_box_any())
                } else if n_bytes <= 4 {
                    Ok(FixedSizeListBuilder::new(
                        PrimitiveBuilder::<Int32Type>::new().into_box_any(),
                        list_size,
                    )
                    .into_box_any())
                } else {
                    Ok(FixedSizeListBuilder::new(
                        PrimitiveBuilder::<Int64Type>::new().into_box_any(),
                        list_size,
                    )
                    .into_box_any())
                }
            }
            4 | 5 => {
                // float
                if n_bytes <= 4 {
                    Ok(FixedSizeListBuilder::new(
                        PrimitiveBuilder::<Float32Type>::new().into_box_any(),
                        list_size,
                    )
                    .into_box_any())
                } else {
                    Ok(FixedSizeListBuilder::new(
                        PrimitiveBuilder::<Float64Type>::new().into_box_any(),
                        list_size,
                    )
                    .into_box_any())
                }
            }
            15 | 16 => {
                // complex
                if n_bytes <= 4 {
                    Ok(FixedSizeListBuilder::new(
                        PrimitiveBuilder::<Float32Type>::new().into_box_any(),
                        list_size,
                    )
                    .into_box_any())
                } else {
                    Ok(FixedSizeListBuilder::new(
                        PrimitiveBuilder::<Float64Type>::new().into_box_any(),
                        list_size,
                    )
                    .into_box_any())
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
    data: Box<dyn ArrayBuilder>,
    cn_type: u8,
    cycle_count: u64,
    n_bytes: u32,
    shape: (Vec<usize>, Order),
) -> Result<Box<dyn ArrayBuilder>, Error> {
    if cn_type == 3 || cn_type == 6 {
        // virtual channels, cn_bit_count = 0 -> n_bytes = 0, must be LE unsigned int
        let mut array = vec![0u64; cycle_count as usize];
        array
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = i as u64);
        Ok(PrimitiveBuilder::new_from_buffer(array.into(), None).into_box_any())
    } else {
        match data.data_type() {
            DataType::Int8 => Ok(PrimitiveBuilder::new_from_buffer(
                MutableBuffer::from_len_zeroed(cycle_count as usize * size_of::<i8>()),
                None,
            )
            .into_box_any()),
            DataType::UInt8 => Ok(PrimitiveBuilder::new_from_buffer(
                MutableBuffer::from_len_zeroed(cycle_count as usize * size_of::<u8>()),
                None,
            )
            .into_box_any()),
            DataType::Int16 => Ok(PrimitiveBuilder::new_from_buffer(
                MutableBuffer::from_len_zeroed(cycle_count as usize * size_of::<i16>()),
                None,
            )
            .into_box_any()),
            DataType::UInt16 => Ok(PrimitiveBuilder::new_from_buffer(
                MutableBuffer::from_len_zeroed(cycle_count as usize * size_of::<u16>()),
                None,
            )
            .into_box_any()),
            DataType::Int32 => Ok(PrimitiveBuilder::new_from_buffer(
                MutableBuffer::from_len_zeroed(cycle_count as usize * size_of::<i32>()),
                None,
            )
            .into_box_any()),
            DataType::UInt32 => Ok(PrimitiveBuilder::new_from_buffer(
                MutableBuffer::from_len_zeroed(cycle_count as usize * size_of::<u32>()),
                None,
            )
            .into_box_any()),
            DataType::Int64 => Ok(PrimitiveBuilder::new_from_buffer(
                MutableBuffer::from_len_zeroed(cycle_count as usize * size_of::<i64>()),
                None,
            )
            .into_box_any()),
            DataType::UInt64 => Ok(PrimitiveBuilder::new_from_buffer(
                MutableBuffer::from_len_zeroed(cycle_count as usize * size_of::<u64>()),
                None,
            )
            .into_box_any()),
            DataType::Float32 => Ok(PrimitiveBuilder::new_from_buffer(
                MutableBuffer::from_len_zeroed(cycle_count as usize * size_of::<f32>()),
                None,
            )
            .into_box_any()),
            DataType::Float64 => Ok(PrimitiveBuilder::new_from_buffer(
                MutableBuffer::from_len_zeroed(cycle_count as usize * size_of::<f64>()),
                None,
            )
            .into_box_any()),
            DataType::FixedSizeBinary(size) => Ok(FixedSizeBinaryBuilder::with_capacity(
                cycle_count as usize,
                n_bytes as i32,
            )),
            DataType::LargeUtf8 => {
                // 6: SBC ISO-8859-1 to be converted into UTF8
                // 7: String UTF8
                // 8 | 9 :String UTF16 to be converted into UTF8
                Ok(
                    LargeStringBuilder::with_capacity(cycle_count as usize, n_bytes as usize)
                        .into_box_any(),
                )
            }
            DataType::LargeBinary => Ok(LargeBinaryBuilder::with_capacity(
                cycle_count as usize,
                n_bytes as usize,
            )
            .into_box_any()),
            DataType::FixedSizeList(field, size) => match data.values().data_type() {
                DataType::Int8 => Ok(PrimitiveBuilder::new_from_buffer(
                    vec![0i8; shape.0.iter().product::<usize>()].into(),
                    None,
                )
                .into_box_any()),
                DataType::UInt8 => Ok(PrimitiveBuilder::new_from_buffer(
                    vec![0u8; shape.0.iter().product::<usize>()].into(),
                    None,
                )
                .into_box_any()),
                DataType::Int16 => Ok(PrimitiveBuilder::new_from_buffer(
                    vec![0i16; shape.0.iter().product::<usize>()].into(),
                    None,
                )
                .into_box_any()),
                DataType::UInt16 => Ok(PrimitiveBuilder::new_from_buffer(
                    vec![0u16; shape.0.iter().product::<usize>()].into(),
                    None,
                )
                .into_box_any()),
                DataType::Int32 => Ok(PrimitiveBuilder::new_from_buffer(
                    vec![0i32; shape.0.iter().product::<usize>()].into(),
                    None,
                )
                .into_box_any()),
                DataType::UInt32 => Ok(PrimitiveBuilder::new_from_buffer(
                    vec![0u32; shape.0.iter().product::<usize>()].into(),
                    None,
                )
                .into_box_any()),
                DataType::Int64 => Ok(PrimitiveBuilder::new_from_buffer(
                    vec![0i64; shape.0.iter().product::<usize>()].into(),
                    None,
                )
                .into_box_any()),
                DataType::UInt64 => Ok(PrimitiveBuilder::new_from_buffer(
                    vec![0u64; shape.0.iter().product::<usize>()].into(),
                    None,
                )
                .into_box_any()),
                DataType::Float32 => Ok(PrimitiveBuilder::new_from_buffer(
                    vec![0f32; shape.0.iter().product::<usize>()].into(),
                    None,
                )
                .into_box_any()),
                DataType::Float64 => Ok(PrimitiveBuilder::new_from_buffer(
                    vec![0f64; shape.0.iter().product::<usize>()].into(),
                    None,
                )
                .into_box_any()),
                _ => bail!(
                    "fixed size list data type {:?} not properly initialised",
                    data.values().data_type()
                ),
            },
            _ => bail!("data type {:?} not properly initialised", data.data_type()),
        }
    }
}
