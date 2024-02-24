//! this module provides methods to get directly channelData into python

use arrow::array::{
    Array, ArrayBuilder, BinaryBuilder, FixedSizeBinaryBuilder, LargeBinaryBuilder,
    LargeStringBuilder, PrimitiveBuilder, StringBuilder,
};

use arrow::buffer::NullBuffer;
use arrow::datatypes::DataType;

use numpy::npyffi::types::NPY_ORDER;
use numpy::ToPyArray;
use pyo3::prelude::*;
use pyo3::PyObject;

use crate::mdfreader::arrow_helpers::Order;

impl From<Order> for NPY_ORDER {
    fn from(order: Order) -> Self {
        match order {
            Order::RowMajor => NPY_ORDER::NPY_CORDER,
            Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
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
                .downcast_ref::<NullBuffer>()
                .expect("could not downcast to Bitmap");
            array.iter().collect::<Vec<_>>().to_pyarray(py).into_py(py)
        }
        DataType::Int8 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveBuilder<i8>>()
                .expect("could not downcast to i8 array");
            array.values_slice().to_pyarray(py).into_py(py)
        }
        DataType::Int16 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveBuilder<i16>>()
                .expect("could not downcast to i16 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::Int32 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveBuilder<i32>>()
                .expect("could not downcast to i32 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::Int64 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveBuilder<i64>>()
                .expect("could not downcast to i64 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::UInt8 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveBuilder<u8>>()
                .expect("could not downcast to u8 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::UInt16 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveBuilder<u16>>()
                .expect("could not downcast to u16 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::UInt32 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveBuilder<u32>>()
                .expect("could not downcast to u32 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::UInt64 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveBuilder<u64>>()
                .expect("could not downcast to u64 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::Float16 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveBuilder<f32>>()
                .expect("could not downcast to f16 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::Float32 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveBuilder<f32>>()
                .expect("could not downcast to f32 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::Float64 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveBuilder<f64>>()
                .expect("could not downcast to f64 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::Timestamp(_, _) => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveBuilder<i64>>()
                .expect("could not downcast timestamp to i64 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::Date32 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveBuilder<i32>>()
                .expect("could not downcast date32 to i32 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::Date64 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveBuilder<i64>>()
                .expect("could not downcast date64 to i64 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::Time32(_) => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveBuilder<i32>>()
                .expect("could not downcast time32 to i32 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::Time64(_) => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveBuilder<i64>>()
                .expect("could not downcast time64 to i64 array");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::Duration(_) => todo!(),
        DataType::Interval(_) => todo!(),
        DataType::Binary => {
            let array = array
                .as_any()
                .downcast_ref::<BinaryBuilder>()
                .expect("could not downcast binary array to bytes vect");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::FixedSizeBinary(size) => {
            let array = array
                .as_any()
                .downcast_ref::<FixedSizeBinaryBuilder>()
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
                .downcast_ref::<LargeBinaryBuilder>()
                .expect("could not downcast large binary to bytes vect");
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::Utf8 => {
            let array = array
                .as_any()
                .downcast_ref::<StringBuilder>()
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
                .downcast_ref::<LargeStringBuilder>()
                .expect("could not downcast to long utf8 array");
            let mut vect_str = Vec::<PyObject>::with_capacity(array.len());
            array
                .values_iter()
                .for_each(|x| vect_str.push(x.to_object(py)));
            vect_str.to_pyarray(py).into_py(py)
        }
        DataType::FixedSizeList(field, _size) => arrow_to_numpy(py, array.values()),
        _ => Python::None(py),
    }
}
