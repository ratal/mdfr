use std::sync::Arc;

use arrow2::array::{Array, BinaryArray, PrimitiveArray, Utf8Array};
use arrow2::bitmap::Bitmap;
use arrow2::datatypes::{DataType, PhysicalType, PrimitiveType};
//_ this module provides methods to get directly channelData into python
use num::Complex;
use numpy::npyffi::types::NPY_ORDER;
use numpy::{IntoPyArray, PyArray1, PyArrayDyn, ToPyArray};
use pyo3::prelude::*;
use pyo3::{PyAny, PyObject, PyResult};

use crate::mdfreader::channel_data::{ArrowComplex, ChannelData, Order};

pub fn arrow_to_numpy(py: Python, array: &Arc<dyn Array>) -> PyObject {
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
        DataType::FixedSizeBinary(_) => {
            let array = array
                .as_any()
                .downcast_ref::<BinaryArray<i64>>()
                .expect("could not downcast large binary to bytes vect");
            array.values().to_pyarray(py).into_py(py)
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
            array.values().to_pyarray(py).into_py(py)
        }
        DataType::LargeUtf8 => {
            let array = array
                .as_any()
                .downcast_ref::<Utf8Array<i64>>()
                .expect("could not downcast to long utf8 array");
            array.values().to_pyarray(py).into_py(py)
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
                DataType::Int8 => todo!(),
                DataType::Int16 => todo!(),
                DataType::Int32 => todo!(),
                DataType::Int64 => todo!(),
                DataType::UInt8 => todo!(),
                DataType::UInt16 => todo!(),
                DataType::UInt32 => todo!(),
                DataType::UInt64 => todo!(),
                DataType::Float16 => todo!(),
                DataType::Float32 => todo!(),
                DataType::Float64 => todo!(),
                DataType::FixedSizeList(_, _) => todo!(),
                _ => Python::None(py),
            },
            _ => Python::None(py),
        },
        _ => Python::None(py),
    }
}

impl ArrowComplex<f64> {
    pub fn to_ndarray(&self) -> Vec<Complex<f64>> {
        Vec::<Complex<f64>>::from_iter(self.clone().into_iter())
    }
    pub fn from_array(array: Vec<Complex<f64>>) -> Self {
        let mut output = Vec::with_capacity(array.len() * 2);
        array.iter().for_each(|v| {
            output.push(v.re);
            output.push(v.im);
        });
        ArrowComplex::<f64>(output)
    }
}

impl ArrowComplex<f32> {
    pub fn to_ndarray(&self) -> Vec<Complex<f32>> {
        Vec::<Complex<f32>>::from_iter(self.clone().into_iter())
    }
    pub fn from_array(array: Vec<Complex<f32>>) -> Self {
        let mut output = Vec::with_capacity(array.len() * 2);
        array.iter().for_each(|v| {
            output.push(v.re);
            output.push(v.im);
        });
        ArrowComplex::<f32>(output)
    }
}

/// IntoPy implementation to convert a ChannelData into a PyObject
impl IntoPy<PyObject> for ChannelData {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            ChannelData::Int8(array) => array.into_pyarray(py).into_py(py),
            ChannelData::UInt8(array) => array.into_pyarray(py).into_py(py),
            ChannelData::Int16(array) => array.into_pyarray(py).into_py(py),
            ChannelData::UInt16(array) => array.into_pyarray(py).into_py(py),
            ChannelData::Float16(array) => array.into_pyarray(py).into_py(py),
            ChannelData::Int24(array) => array.into_pyarray(py).into_py(py),
            ChannelData::UInt24(array) => array.into_pyarray(py).into_py(py),
            ChannelData::Int32(array) => array.into_pyarray(py).into_py(py),
            ChannelData::UInt32(array) => array.into_pyarray(py).into_py(py),
            ChannelData::Float32(array) => array.into_pyarray(py).into_py(py),
            ChannelData::Int48(array) => array.into_pyarray(py).into_py(py),
            ChannelData::UInt48(array) => array.into_pyarray(py).into_py(py),
            ChannelData::Int64(array) => array.into_pyarray(py).into_py(py),
            ChannelData::UInt64(array) => array.into_pyarray(py).into_py(py),
            ChannelData::Float64(array) => array.into_pyarray(py).into_py(py),
            ChannelData::Complex16(array) => array.to_ndarray().into_pyarray(py).into_py(py),
            ChannelData::Complex32(array) => array.to_ndarray().into_pyarray(py).into_py(py),
            ChannelData::Complex64(array) => array.to_ndarray().into_pyarray(py).into_py(py),
            ChannelData::StringSBC(array) => array.into_py(py),
            ChannelData::StringUTF8(array) => array.into_py(py),
            ChannelData::StringUTF16(array) => array.into_py(py),
            ChannelData::VariableSizeByteArray(array) => array.into_py(py),
            ChannelData::FixedSizeByteArray(array) => {
                let out: Vec<Vec<u8>> = array.0.chunks(array.1).map(|x| x.to_vec()).collect();
                out.into_py(py)
            }
            ChannelData::ArrayDInt8(array) => {
                let order = match array.1 .1 {
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array
                    .0
                    .into_pyarray(py)
                    .reshape_with_order(array.1 .0, order)
                    .expect("could not reshape i8")
                    .into_py(py)
            }
            ChannelData::ArrayDUInt8(array) => {
                let order = match array.1 .1 {
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array
                    .0
                    .into_pyarray(py)
                    .reshape_with_order(array.1 .0, order)
                    .expect("could not reshape u8")
                    .into_py(py)
            }
            ChannelData::ArrayDInt16(array) => {
                let order = match array.1 .1 {
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array
                    .0
                    .into_pyarray(py)
                    .reshape_with_order(array.1 .0, order)
                    .expect("could not reshape u16")
                    .into_py(py)
            }
            ChannelData::ArrayDUInt16(array) => {
                let order = match array.1 .1 {
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array
                    .0
                    .into_pyarray(py)
                    .reshape_with_order(array.1 .0, order)
                    .expect("could not reshape i16")
                    .into_py(py)
            }
            ChannelData::ArrayDFloat16(array) => {
                let order = match array.1 .1 {
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array
                    .0
                    .into_pyarray(py)
                    .reshape_with_order(array.1 .0, order)
                    .expect("could not reshape f16")
                    .into_py(py)
            }
            ChannelData::ArrayDInt24(array) => {
                let order = match array.1 .1 {
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array
                    .0
                    .into_pyarray(py)
                    .reshape_with_order(array.1 .0, order)
                    .expect("could not reshape i24")
                    .into_py(py)
            }
            ChannelData::ArrayDUInt24(array) => {
                let order = match array.1 .1 {
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array
                    .0
                    .into_pyarray(py)
                    .reshape_with_order(array.1 .0, order)
                    .expect("could not reshape u24")
                    .into_py(py)
            }
            ChannelData::ArrayDInt32(array) => {
                let order = match array.1 .1 {
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array
                    .0
                    .into_pyarray(py)
                    .reshape_with_order(array.1 .0, order)
                    .expect("could not reshape i32")
                    .into_py(py)
            }
            ChannelData::ArrayDUInt32(array) => {
                let order = match array.1 .1 {
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array
                    .0
                    .into_pyarray(py)
                    .reshape_with_order(array.1 .0, order)
                    .expect("could not reshape u32")
                    .into_py(py)
            }
            ChannelData::ArrayDFloat32(array) => {
                let order = match array.1 .1 {
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array
                    .0
                    .into_pyarray(py)
                    .reshape_with_order(array.1 .0, order)
                    .expect("could not reshape f32")
                    .into_py(py)
            }
            ChannelData::ArrayDInt48(array) => {
                let order = match array.1 .1 {
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array
                    .0
                    .into_pyarray(py)
                    .reshape_with_order(array.1 .0, order)
                    .expect("could not reshape i48")
                    .into_py(py)
            }
            ChannelData::ArrayDUInt48(array) => {
                let order = match array.1 .1 {
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array
                    .0
                    .into_pyarray(py)
                    .reshape_with_order(array.1 .0, order)
                    .expect("could not reshape u48")
                    .into_py(py)
            }
            ChannelData::ArrayDInt64(array) => {
                let order = match array.1 .1 {
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array
                    .0
                    .into_pyarray(py)
                    .reshape_with_order(array.1 .0, order)
                    .expect("could not reshape i64")
                    .into_py(py)
            }
            ChannelData::ArrayDUInt64(array) => {
                let order = match array.1 .1 {
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array
                    .0
                    .into_pyarray(py)
                    .reshape_with_order(array.1 .0, order)
                    .expect("could not reshape u64")
                    .into_py(py)
            }
            ChannelData::ArrayDFloat64(array) => {
                let order = match array.1 .1 {
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array
                    .0
                    .into_pyarray(py)
                    .reshape_with_order(array.1 .0, order)
                    .expect("could not reshape f64")
                    .into_py(py)
            }
            ChannelData::ArrayDComplex16(array) => {
                let order = match array.1 .1 {
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array
                    .0
                    .into_iter()
                    .collect::<Vec<Complex<f32>>>()
                    .into_pyarray(py)
                    .reshape_with_order(array.1 .0, order)
                    .expect("could not reshape complex16")
                    .into_py(py)
            }
            ChannelData::ArrayDComplex32(array) => {
                let order = match array.1 .1 {
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array
                    .0
                    .into_iter()
                    .collect::<Vec<Complex<f32>>>()
                    .into_pyarray(py)
                    .reshape_with_order(array.1 .0, order)
                    .expect("could not reshape complex32")
                    .into_py(py)
            }
            ChannelData::ArrayDComplex64(array) => {
                let order = match array.1 .1 {
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array
                    .0
                    .into_iter()
                    .collect::<Vec<Complex<f64>>>()
                    .into_pyarray(py)
                    .reshape_with_order(array.1 .0, order)
                    .expect("could not reshape complex64")
                    .into_py(py)
            }
        }
    }
}

/// ToPyObject implementation to convert a ChannelData into a PyObject
impl ToPyObject for ChannelData {
    fn to_object(&self, py: Python) -> PyObject {
        match self {
            ChannelData::Int8(array) => array.to_pyarray(py).into_py(py),
            ChannelData::UInt8(array) => array.to_pyarray(py).into_py(py),
            ChannelData::Int16(array) => array.to_pyarray(py).into_py(py),
            ChannelData::UInt16(array) => array.to_pyarray(py).into_py(py),
            ChannelData::Float16(array) => array.to_pyarray(py).into_py(py),
            ChannelData::Int24(array) => array.to_pyarray(py).into_py(py),
            ChannelData::UInt24(array) => array.to_pyarray(py).into_py(py),
            ChannelData::Int32(array) => array.to_pyarray(py).into_py(py),
            ChannelData::UInt32(array) => array.to_pyarray(py).into_py(py),
            ChannelData::Float32(array) => array.to_pyarray(py).into_py(py),
            ChannelData::Int48(array) => array.to_pyarray(py).into_py(py),
            ChannelData::UInt48(array) => array.to_pyarray(py).into_py(py),
            ChannelData::Int64(array) => array.to_pyarray(py).into_py(py),
            ChannelData::UInt64(array) => array.to_pyarray(py).into_py(py),
            ChannelData::Float64(array) => array.to_pyarray(py).into_py(py),
            ChannelData::Complex16(array) => array.to_ndarray().to_pyarray(py).into_py(py),
            ChannelData::Complex32(array) => array.to_ndarray().to_pyarray(py).into_py(py),
            ChannelData::Complex64(array) => array.to_ndarray().into_pyarray(py).into_py(py),
            ChannelData::StringSBC(array) => array.to_object(py),
            ChannelData::StringUTF8(array) => array.to_object(py),
            ChannelData::StringUTF16(array) => array.to_object(py),
            ChannelData::VariableSizeByteArray(array) => array.to_object(py),
            ChannelData::FixedSizeByteArray(array) => {
                let out: Vec<Vec<u8>> = array.0.chunks(array.1).map(|x| x.to_vec()).collect();
                out.to_object(py)
            }
            ChannelData::ArrayDInt8(array) => array.0.to_pyarray(py).into_py(py),
            ChannelData::ArrayDUInt8(array) => array.0.to_pyarray(py).into_py(py),
            ChannelData::ArrayDInt16(array) => array.0.to_pyarray(py).into_py(py),
            ChannelData::ArrayDUInt16(array) => array.0.to_pyarray(py).into_py(py),
            ChannelData::ArrayDFloat16(array) => array.0.to_pyarray(py).into_py(py),
            ChannelData::ArrayDInt24(array) => array.0.to_pyarray(py).into_py(py),
            ChannelData::ArrayDUInt24(array) => array.0.to_pyarray(py).into_py(py),
            ChannelData::ArrayDInt32(array) => array.0.to_pyarray(py).into_py(py),
            ChannelData::ArrayDUInt32(array) => array.0.to_pyarray(py).into_py(py),
            ChannelData::ArrayDFloat32(array) => array.0.to_pyarray(py).into_py(py),
            ChannelData::ArrayDInt48(array) => array.0.to_pyarray(py).into_py(py),
            ChannelData::ArrayDUInt48(array) => array.0.to_pyarray(py).into_py(py),
            ChannelData::ArrayDInt64(array) => array.0.to_pyarray(py).into_py(py),
            ChannelData::ArrayDUInt64(array) => array.0.to_pyarray(py).into_py(py),
            ChannelData::ArrayDFloat64(array) => array.0.to_pyarray(py).into_py(py),
            ChannelData::ArrayDComplex16(array) => array.0 .0.to_pyarray(py).into_py(py),
            ChannelData::ArrayDComplex32(array) => array.0 .0.to_pyarray(py).into_py(py),
            ChannelData::ArrayDComplex64(array) => array.0 .0.to_pyarray(py).into_py(py),
        }
    }
}

/// FromPyObject implementation to allow conversion from a Python object to a ChannelData
impl FromPyObject<'_> for ChannelData {
    fn extract(ob: &'_ PyAny) -> PyResult<Self> {
        let truc: NumpyArray = ob.extract()?;
        match truc {
            NumpyArray::Boolean(array) => Ok(ChannelData::UInt8(
                array.readonly().as_array().to_owned().to_vec(),
            )),
            NumpyArray::Int8(array) => Ok(ChannelData::Int8(
                array.readonly().as_array().to_owned().to_vec(),
            )),
            NumpyArray::UInt8(array) => Ok(ChannelData::UInt8(
                array.readonly().as_array().to_owned().to_vec(),
            )),
            NumpyArray::Int16(array) => Ok(ChannelData::Int16(
                array.readonly().as_array().to_owned().to_vec(),
            )),
            NumpyArray::UInt16(array) => Ok(ChannelData::UInt16(
                array.readonly().as_array().to_owned().to_vec(),
            )),
            NumpyArray::Int32(array) => Ok(ChannelData::Int32(
                array.readonly().as_array().to_owned().to_vec(),
            )),
            NumpyArray::UInt32(array) => Ok(ChannelData::UInt32(
                array.readonly().as_array().to_owned().to_vec().to_vec(),
            )),
            NumpyArray::Float32(array) => Ok(ChannelData::Float32(
                array.readonly().as_array().to_owned().to_vec(),
            )),
            NumpyArray::Int64(array) => Ok(ChannelData::Int64(
                array.readonly().as_array().to_owned().to_vec(),
            )),
            NumpyArray::UInt64(array) => Ok(ChannelData::UInt64(
                array.readonly().as_array().to_owned().to_vec(),
            )),
            NumpyArray::Float64(array) => Ok(ChannelData::Float64(
                array.readonly().as_array().to_owned().to_vec(),
            )),
            NumpyArray::Complex32(array) => {
                Ok(ChannelData::Complex32(ArrowComplex::<f32>::from_array(
                    array.readonly().as_array().to_owned().into_raw_vec(),
                )))
            }
            NumpyArray::Complex64(array) => {
                Ok(ChannelData::Complex64(ArrowComplex::<f64>::from_array(
                    array.readonly().as_array().to_owned().into_raw_vec(),
                )))
            }
            NumpyArray::ArrayDInt8(array) => {
                let order: Order = if array.is_c_contiguous() {
                    Order::RowMajor
                } else {
                    Order::ColumnMajor
                };
                Ok(ChannelData::ArrayDInt8((
                    array.readonly().as_array().to_owned().into_raw_vec(),
                    (array.shape().to_vec(), order),
                )))
            }
            NumpyArray::ArrayDUInt8(array) => {
                let order: Order = if array.is_c_contiguous() {
                    Order::RowMajor
                } else {
                    Order::ColumnMajor
                };
                Ok(ChannelData::ArrayDUInt8((
                    array.readonly().as_array().to_owned().into_raw_vec(),
                    (array.shape().to_vec(), order),
                )))
            }
            NumpyArray::ArrayDInt16(array) => {
                let order: Order = if array.is_c_contiguous() {
                    Order::RowMajor
                } else {
                    Order::ColumnMajor
                };
                Ok(ChannelData::ArrayDInt16((
                    array.readonly().as_array().to_owned().into_raw_vec(),
                    (array.shape().to_vec(), order),
                )))
            }
            NumpyArray::ArrayDUInt16(array) => {
                let order: Order = if array.is_c_contiguous() {
                    Order::RowMajor
                } else {
                    Order::ColumnMajor
                };
                Ok(ChannelData::ArrayDUInt16((
                    array.readonly().as_array().to_owned().into_raw_vec(),
                    (array.shape().to_vec(), order),
                )))
            }
            NumpyArray::ArrayDInt32(array) => {
                let order: Order = if array.is_c_contiguous() {
                    Order::RowMajor
                } else {
                    Order::ColumnMajor
                };
                Ok(ChannelData::ArrayDInt32((
                    array.readonly().as_array().to_owned().into_raw_vec(),
                    (array.shape().to_vec(), order),
                )))
            }
            NumpyArray::ArrayDUInt32(array) => {
                let order: Order = if array.is_c_contiguous() {
                    Order::RowMajor
                } else {
                    Order::ColumnMajor
                };
                Ok(ChannelData::ArrayDUInt32((
                    array.readonly().as_array().to_owned().into_raw_vec(),
                    (array.shape().to_vec(), order),
                )))
            }
            NumpyArray::ArrayDFloat32(array) => {
                let order: Order = if array.is_c_contiguous() {
                    Order::RowMajor
                } else {
                    Order::ColumnMajor
                };
                Ok(ChannelData::ArrayDFloat32((
                    array.readonly().as_array().to_owned().into_raw_vec(),
                    (array.shape().to_vec(), order),
                )))
            }
            NumpyArray::ArrayDInt64(array) => {
                let order: Order = if array.is_c_contiguous() {
                    Order::RowMajor
                } else {
                    Order::ColumnMajor
                };
                Ok(ChannelData::ArrayDInt64((
                    array.readonly().as_array().to_owned().into_raw_vec(),
                    (array.shape().to_vec(), order),
                )))
            }
            NumpyArray::ArrayDUInt64(array) => {
                let order: Order = if array.is_c_contiguous() {
                    Order::RowMajor
                } else {
                    Order::ColumnMajor
                };
                Ok(ChannelData::ArrayDUInt64((
                    array.readonly().as_array().to_owned().into_raw_vec(),
                    (array.shape().to_vec(), order),
                )))
            }
            NumpyArray::ArrayDFloat64(array) => {
                let order: Order = if array.is_c_contiguous() {
                    Order::RowMajor
                } else {
                    Order::ColumnMajor
                };
                Ok(ChannelData::ArrayDFloat64((
                    array.readonly().as_array().to_owned().into_raw_vec(),
                    (array.shape().to_vec(), order),
                )))
            }
            NumpyArray::ArrayDComplex32(array) => {
                let order: Order = if array.is_c_contiguous() {
                    Order::RowMajor
                } else {
                    Order::ColumnMajor
                };
                Ok(ChannelData::ArrayDComplex32((
                    ArrowComplex::<f32>::from_array(
                        array.readonly().as_array().to_owned().into_raw_vec(),
                    ),
                    (array.shape().to_vec(), order),
                )))
            }
            NumpyArray::ArrayDComplex64(array) => {
                let order: Order = if array.is_c_contiguous() {
                    Order::RowMajor
                } else {
                    Order::ColumnMajor
                };
                Ok(ChannelData::ArrayDComplex64((
                    ArrowComplex::<f64>::from_array(
                        array.readonly().as_array().to_owned().into_raw_vec(),
                    ),
                    (array.shape().to_vec(), order),
                )))
            }
        }
    }
}

/// Enum to identify the dtype of numpy array
#[derive(Clone, FromPyObject)]
enum NumpyArray<'a> {
    Boolean(&'a PyArray1<u8>),
    Int8(&'a PyArray1<i8>),
    UInt8(&'a PyArray1<u8>),
    Int16(&'a PyArray1<i16>),
    UInt16(&'a PyArray1<u16>),
    Int32(&'a PyArray1<i32>),
    UInt32(&'a PyArray1<u32>),
    Float32(&'a PyArray1<f32>),
    Int64(&'a PyArray1<i64>),
    UInt64(&'a PyArray1<u64>),
    Float64(&'a PyArray1<f64>),
    Complex32(&'a PyArray1<Complex<f32>>),
    Complex64(&'a PyArray1<Complex<f64>>),
    ArrayDInt8(&'a PyArrayDyn<i8>),
    ArrayDUInt8(&'a PyArrayDyn<u8>),
    ArrayDInt16(&'a PyArrayDyn<i16>),
    ArrayDUInt16(&'a PyArrayDyn<u16>),
    ArrayDInt32(&'a PyArrayDyn<i32>),
    ArrayDUInt32(&'a PyArrayDyn<u32>),
    ArrayDFloat32(&'a PyArrayDyn<f32>),
    ArrayDInt64(&'a PyArrayDyn<i64>),
    ArrayDUInt64(&'a PyArrayDyn<u64>),
    ArrayDFloat64(&'a PyArrayDyn<f64>),
    ArrayDComplex32(&'a PyArrayDyn<Complex<f32>>),
    ArrayDComplex64(&'a PyArrayDyn<Complex<f64>>),
}
