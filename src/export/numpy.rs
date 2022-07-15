use numpy::{IntoPyArray, PyArray1, PyArrayDyn, ToPyArray, Complex32};
use numpy::npyffi::types::NPY_ORDER;
use pyo3::prelude::*;

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
                let order = match array.1.1{
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array.0.into_pyarray(py).reshape_with_order(array.1.0, order).expect("could not reshape i8").into_py(py)},
            ChannelData::ArrayDUInt8(array) => {
                let order = match array.1.1{
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array.0.into_pyarray(py).reshape_with_order(array.1.0, order).expect("could not reshape u8").into_py(py)},
            ChannelData::ArrayDInt16(array) => {
                let order = match array.1.1{
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array.0.into_pyarray(py).reshape_with_order(array.1.0, order).expect("could not reshape u16").into_py(py)},
            ChannelData::ArrayDUInt16(array) => {
                let order = match array.1.1{
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array.0.into_pyarray(py).reshape_with_order(array.1.0, order).expect("could not reshape i16").into_py(py)},
            ChannelData::ArrayDFloat16(array) => {
                let order = match array.1.1{
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array.0.into_pyarray(py).reshape_with_order(array.1.0, order).expect("could not reshape f16").into_py(py)},
            ChannelData::ArrayDInt24(array) => {
                let order = match array.1.1{
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array.0.into_pyarray(py).reshape_with_order(array.1.0, order).expect("could not reshape i24").into_py(py)},
            ChannelData::ArrayDUInt24(array) => {
                let order = match array.1.1{
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array.0.into_pyarray(py).reshape_with_order(array.1.0, order).expect("could not reshape u24").into_py(py)},
            ChannelData::ArrayDInt32(array) => {
                let order = match array.1.1{
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array.0.into_pyarray(py).reshape_with_order(array.1.0, order).expect("could not reshape i32").into_py(py)},
            ChannelData::ArrayDUInt32(array) => {
                let order = match array.1.1{
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array.0.into_pyarray(py).reshape_with_order(array.1.0, order).expect("could not reshape u32").into_py(py)},
            ChannelData::ArrayDFloat32(array) => {
                let order = match array.1.1{
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array.0.into_pyarray(py).reshape_with_order(array.1.0, order).expect("could not reshape f32").into_py(py)},
            ChannelData::ArrayDInt48(array) => {
                let order = match array.1.1{
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array.0.into_pyarray(py).reshape_with_order(array.1.0, order).expect("could not reshape i48").into_py(py)},
            ChannelData::ArrayDUInt48(array) => {
                let order = match array.1.1{
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array.0.into_pyarray(py).reshape_with_order(array.1.0, order).expect("could not reshape u48").into_py(py)},
            ChannelData::ArrayDInt64(array) => {
                let order = match array.1.1{
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array.0.into_pyarray(py).reshape_with_order(array.1.0, order).expect("could not reshape i64").into_py(py)},
            ChannelData::ArrayDUInt64(array) => {
                let order = match array.1.1{
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array.0.into_pyarray(py).reshape_with_order(array.1.0, order).expect("could not reshape u64").into_py(py)},
            ChannelData::ArrayDFloat64(array) => {
                let order = match array.1.1{
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array.0.into_pyarray(py).reshape_with_order(array.1.0, order).expect("could not reshape f64").into_py(py)},
            ChannelData::ArrayDComplex16(array) => {
                let order = match array.1.1{
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array.0.into_iter().collect::<Complex32>().into_pyarray(py).reshape_with_order(array.1.0, order).expect("could not reshape complex16").into_py(py)
            },
            ChannelData::ArrayDComplex32(array) => {
                let order = match array.1.1{
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array.0.into_iter().collect::<Complex32>().into_pyarray(py).reshape_with_order(array.1.0, order).expect("could not reshape complex32").into_py(py)
            },
            ChannelData::ArrayDComplex64(array) => {
                let order = match array.1.1{
                    Order::RowMajor => NPY_ORDER::NPY_CORDER,
                    Order::ColumnMajor => NPY_ORDER::NPY_FORTRANORDER,
                };
                array.0.into_iter().collect::<Complex64>().into_pyarray(py).reshape_with_order(array.1.0, order).expect("could not reshape complex64").into_py(py)
            },
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
            ChannelData::ArrayDComplex16(array) => array.0.to_pyarray(py).into_py(py),
            ChannelData::ArrayDComplex32(array) => array.0.to_pyarray(py).into_py(py),
            ChannelData::ArrayDComplex64(array) => array.0.to_pyarray(py).into_py(py),
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
            NumpyArray::Complex32(array) => Ok(ChannelData::Complex32(
                ArrowComplex::<f32>::from_ndarray(array.readonly().as_array().to_owned()),
            )),
            NumpyArray::Complex64(array) => Ok(ChannelData::Complex64(
                ArrowComplex::<f64>::from_ndarray(array.readonly().as_array().to_owned()),
            )),
            NumpyArray::ArrayDInt8(array) => {
                let order:Order = if array.is_c_contiguous() {Order::RowMajor} 
                    else {Order::ColumnMajor};
                Ok(ChannelData::ArrayDInt8((
                array.readonly().as_array().to_owned().into_raw_vec(),
                (array.shape().to_vec(), order),
            )))},
            NumpyArray::ArrayDUInt8(array) => {
                let order:Order = if array.is_c_contiguous() {Order::RowMajor} 
                    else {Order::ColumnMajor};
                Ok(ChannelData::ArrayDUInt8((
                array.readonly().as_array().to_owned().into_raw_vec(),
                (array.shape().to_vec(), order),
            )))},
            NumpyArray::ArrayDInt16(array) => {
                let order:Order = if array.is_c_contiguous() {Order::RowMajor} 
                    else {Order::ColumnMajor};
                Ok(ChannelData::ArrayDInt16((
                array.readonly().as_array().to_owned().into_raw_vec(),
                (array.shape().to_vec(), order),
            )))},
            NumpyArray::ArrayDUInt16(array) => {
                let order:Order = if array.is_c_contiguous() {Order::RowMajor} 
                    else {Order::ColumnMajor};
                Ok(ChannelData::ArrayDUInt16((
                array.readonly().as_array().to_owned().into_raw_vec(),
                (array.shape().to_vec(), order),
            )))},
            NumpyArray::ArrayDInt32(array) => {
                let order:Order = if array.is_c_contiguous() {Order::RowMajor} 
                    else {Order::ColumnMajor};
                Ok(ChannelData::ArrayDInt32((
                array.readonly().as_array().to_owned().into_raw_vec(),
                (array.shape().to_vec(), order),
            )))},
            NumpyArray::ArrayDUInt32(array) => {
                let order:Order = if array.is_c_contiguous() {Order::RowMajor} 
                    else {Order::ColumnMajor};
                Ok(ChannelData::ArrayDUInt32((
                array.readonly().as_array().to_owned().into_raw_vec(),
                (array.shape().to_vec(), order),
            )))},
            NumpyArray::ArrayDFloat32(array) => {
                let order:Order = if array.is_c_contiguous() {Order::RowMajor} 
                    else {Order::ColumnMajor};
                Ok(ChannelData::ArrayDFloat32((
                array.readonly().as_array().to_owned().into_raw_vec(),
                (array.shape().to_vec(), order),
            )))},
            NumpyArray::ArrayDInt64(array) => {
                let order:Order = if array.is_c_contiguous() {Order::RowMajor} 
                    else {Order::ColumnMajor};
                Ok(ChannelData::ArrayDInt64((
                array.readonly().as_array().to_owned().into_raw_vec(),
                (array.shape().to_vec(), order),
            )))},
            NumpyArray::ArrayDUInt64(array) => {
                let order:Order = if array.is_c_contiguous() {Order::RowMajor} 
                    else {Order::ColumnMajor};
                Ok(ChannelData::ArrayDUInt64((
                array.readonly().as_array().to_owned().into_raw_vec(),
                (array.shape().to_vec(), order),
            )))},
            NumpyArray::ArrayDFloat64(array) => {
                let order:Order = if array.is_c_contiguous() {Order::RowMajor} 
                    else {Order::ColumnMajor};
                Ok(ChannelData::ArrayDFloat64((
                array.readonly().as_array().to_owned().into_raw_vec(),
                (array.shape().to_vec(), order),
            )))},
            NumpyArray::ArrayDComplex32(array) => {
                let order:Order = if array.is_c_contiguous() {Order::RowMajor} 
                    else {Order::ColumnMajor};
                Ok(ChannelData::ArrayDComplex32((
                array.readonly().as_array().to_owned().into_raw_vec(),
                (array.shape().to_vec(), order),
            )))},
            NumpyArray::ArrayDComplex64(array) => {
                let order:Order = if array.is_c_contiguous() {Order::RowMajor} 
                    else {Order::ColumnMajor};
                Ok(ChannelData::ArrayDComplex64((
                array.readonly().as_array().to_owned().into_raw_vec(),
                (array.shape().to_vec(), order),
            )))},
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