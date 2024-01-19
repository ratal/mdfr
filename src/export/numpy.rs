//! this module provides methods to get directly channelData into python

use arrow2::array::{Array, BinaryArray, FixedSizeBinaryArray, PrimitiveArray, Utf8Array};
use arrow2::bitmap::Bitmap;
use arrow2::datatypes::{DataType, PhysicalType, PrimitiveType};

use numpy::npyffi::types::NPY_ORDER;
use numpy::{IntoPyArray, ToPyArray};
use pyo3::prelude::*;
use pyo3::PyObject;

use crate::export::tensor::Order as TensorOrder;

use super::tensor::Tensor;

impl From<TensorOrder> for NPY_ORDER {
    fn from(order: TensorOrder) -> Self {
        match order {
            TensorOrder::RowMajor => NPY_ORDER::NPY_CORDER,
            TensorOrder::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
        }
    }
}

/// returns a numpy array from an arrow array
#[allow(dead_code)]
pub fn arrow_to_numpy(py: Python, array: Box<dyn Array>) -> PyObject {
    match array.data_type() {
        DataType::Null => Python::None(py),
        DataType::Boolean => {
            let array = array
                .as_any()
                .downcast_ref::<Bitmap>()
                .expect("could not downcast to Bitmap");
            array.iter().collect::<Vec<_>>().to_pyarray(py).into_py(py)
        }
        DataType::Int8 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i8>>()
                .expect("could not downcast to i8 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::Int16 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i16>>()
                .expect("could not downcast to i16 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::Int32 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i32>>()
                .expect("could not downcast to i32 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::Int64 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i64>>()
                .expect("could not downcast to i64 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::UInt8 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<u8>>()
                .expect("could not downcast to u8 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::UInt16 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<u16>>()
                .expect("could not downcast to u16 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::UInt32 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<u32>>()
                .expect("could not downcast to u32 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::UInt64 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<u64>>()
                .expect("could not downcast to u64 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::Float16 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<f32>>()
                .expect("could not downcast to f16 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::Float32 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<f32>>()
                .expect("could not downcast to f32 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::Float64 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<f64>>()
                .expect("could not downcast to f64 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::Timestamp(_, _) => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i64>>()
                .expect("could not downcast timestamp to i64 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::Date32 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i32>>()
                .expect("could not downcast date32 to i32 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::Date64 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i64>>()
                .expect("could not downcast date64 to i64 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::Time32(_) => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i32>>()
                .expect("could not downcast time32 to i32 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::Time64(_) => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i64>>()
                .expect("could not downcast time64 to i64 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::Duration(_) => todo!(),
        DataType::Interval(_) => todo!(),
        DataType::Binary => {
            let array = array
                .as_any()
                .downcast_ref::<BinaryArray<i32>>()
                .expect("could not downcast binary array to bytes vect");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::FixedSizeBinary(size) => {
            let array = array
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .expect("could not downcast large binary to bytes vect");
            array
                .values()
                .to_pyarray(py)
                .reshape([array.len() / size, *size])
                .expect("failed reshaping the fixedsizebinaryarray")
                .into_py(py)
        }
        DataType::LargeBinary => {
            let array = array
                .as_any()
                .downcast_ref::<BinaryArray<i64>>()
                .expect("could not downcast large binary to bytes vect");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::Utf8 => {
            let array = array
                .as_any()
                .downcast_ref::<Utf8Array<i32>>()
                .expect("could not downcast to utf8 array");
            let mut vect_str = Vec::<PyObject>::with_capacity(array.len());
            array
                .values_iter()
                .for_each(|x| vect_str.push(x.to_object(py)));
            vect_str.to_pyarray(py).into_py(py)
        }
        DataType::LargeUtf8 => {
            let array = array
                .as_any()
                .downcast_ref::<Utf8Array<i64>>()
                .expect("could not downcast to long utf8 array");
            let mut vect_str = Vec::<PyObject>::with_capacity(array.len());
            array
                .values_iter()
                .for_each(|x| vect_str.push(x.to_object(py)));
            vect_str.to_pyarray(py).into_py(py)
        }
        DataType::FixedSizeList(field, _size) => match field.data_type.to_physical_type() {
            // Complex types
            PhysicalType::Primitive(PrimitiveType::Float32) => {
                let array = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<f32>>()
                    .expect("could not downcast to f32 array");
                array.values().to_pyarray(py).into_py(py)
            }
            PhysicalType::Primitive(PrimitiveType::Float64) => {
                let array = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<f64>>()
                    .expect("could not downcast to f64 array");
                array.values().to_pyarray(py).into_py(py)
            }
            _ => Python::None(py),
        },
        DataType::Extension(ext_str, dtype, _) => match ext_str.as_str() {
            "Tensor" => match **dtype {
                DataType::Int8 => {
                    let tensor = array
                        .as_any()
                        .downcast_ref::<Tensor<i8>>()
                        .expect("could not downcast to i8 tensor");
                    tensor
                        .values()
                        .to_vec()
                        .into_pyarray(py)
                        .reshape_with_order(tensor.shape().clone(), tensor.order().clone().into())
                        .expect("could not reshape i8 tensor")
                        .into_py(py)
                }
                DataType::Int16 => {
                    let tensor = array
                        .as_any()
                        .downcast_ref::<Tensor<i16>>()
                        .expect("could not downcast to i16 tensor");
                    tensor
                        .values()
                        .to_vec()
                        .into_pyarray(py)
                        .reshape_with_order(tensor.shape().clone(), tensor.order().clone().into())
                        .expect("could not reshape i16 tensor")
                        .into_py(py)
                }
                DataType::Int32 => {
                    let tensor = array
                        .as_any()
                        .downcast_ref::<Tensor<i32>>()
                        .expect("could not downcast to i32 tensor");
                    tensor
                        .values()
                        .to_vec()
                        .into_pyarray(py)
                        .reshape_with_order(tensor.shape().clone(), tensor.order().clone().into())
                        .expect("could not reshape i32 tensor")
                        .into_py(py)
                }
                DataType::Int64 => {
                    let tensor = array
                        .as_any()
                        .downcast_ref::<Tensor<i64>>()
                        .expect("could not downcast to i64 tensor");
                    tensor
                        .values()
                        .to_vec()
                        .into_pyarray(py)
                        .reshape_with_order(tensor.shape().clone(), tensor.order().clone().into())
                        .expect("could not reshape i64 tensor")
                        .into_py(py)
                }
                DataType::UInt8 => {
                    let tensor = array
                        .as_any()
                        .downcast_ref::<Tensor<u8>>()
                        .expect("could not downcast to u8 tensor");
                    tensor
                        .values()
                        .to_vec()
                        .into_pyarray(py)
                        .reshape_with_order(tensor.shape().clone(), tensor.order().clone().into())
                        .expect("could not reshape u8 tensor")
                        .into_py(py)
                }
                DataType::UInt16 => {
                    let tensor = array
                        .as_any()
                        .downcast_ref::<Tensor<u16>>()
                        .expect("could not downcast to u16 tensor");
                    tensor
                        .values()
                        .to_vec()
                        .into_pyarray(py)
                        .reshape_with_order(tensor.shape().clone(), tensor.order().clone().into())
                        .expect("could not reshape u16 tensor")
                        .into_py(py)
                }
                DataType::UInt32 => {
                    let tensor = array
                        .as_any()
                        .downcast_ref::<Tensor<u32>>()
                        .expect("could not downcast to u32 tensor");
                    tensor
                        .values()
                        .to_vec()
                        .into_pyarray(py)
                        .reshape_with_order(tensor.shape().clone(), tensor.order().clone().into())
                        .expect("could not reshape u32 tensor")
                        .into_py(py)
                }
                DataType::UInt64 => {
                    let tensor = array
                        .as_any()
                        .downcast_ref::<Tensor<u64>>()
                        .expect("could not downcast to u64 tensor");
                    tensor
                        .values()
                        .to_vec()
                        .into_pyarray(py)
                        .reshape_with_order(tensor.shape().clone(), tensor.order().clone().into())
                        .expect("could not reshape u64 tensor")
                        .into_py(py)
                }
                DataType::Float16 => {
                    let tensor = array
                        .as_any()
                        .downcast_ref::<Tensor<f32>>()
                        .expect("could not downcast to f16 tensor");
                    tensor
                        .values()
                        .to_vec()
                        .into_pyarray(py)
                        .reshape_with_order(tensor.shape().clone(), tensor.order().clone().into())
                        .expect("could not reshape f16(f32) tensor")
                        .into_py(py)
                }
                DataType::Float32 => {
                    let tensor = array
                        .as_any()
                        .downcast_ref::<Tensor<f32>>()
                        .expect("could not downcast to f32 tensor");
                    tensor
                        .values()
                        .to_vec()
                        .into_pyarray(py)
                        .reshape_with_order(tensor.shape().clone(), tensor.order().clone().into())
                        .expect("could not reshape f32 tensor")
                        .into_py(py)
                }
                DataType::Float64 => {
                    let tensor = array
                        .as_any()
                        .downcast_ref::<Tensor<f64>>()
                        .expect("could not downcast to f64 tensor");
                    tensor
                        .values()
                        .to_vec()
                        .into_pyarray(py)
                        .reshape_with_order(tensor.shape().clone(), tensor.order().clone().into())
                        .expect("could not reshape f64 tensor")
                        .into_py(py)
                }
                DataType::FixedSizeList(_, _) => todo!(),
                _ => Python::None(py),
            },
            _ => Python::None(py),
        },
        _ => Python::None(py),
    }
}
