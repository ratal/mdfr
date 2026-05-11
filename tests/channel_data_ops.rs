//! Integration tests for ChannelData methods.
//! These tests exercise public API methods that are not covered by the internal unit tests,
//! or cover additional variants to improve overall code coverage.

use arrow::array::{
    Array, FixedSizeBinaryBuilder, Float64Builder, Int8Builder, Int16Builder, Int32Builder,
    Int64Builder, LargeStringBuilder, UInt8Builder, UInt16Builder, UInt32Builder, UInt64Builder,
};
use arrow::buffer::MutableBuffer;
use arrow::datatypes::{Float64Type, Int8Type, Int16Type, UInt8Type, UInt32Type, UInt64Type};
use mdfr::data_holder::channel_data::ChannelData;
use mdfr::data_holder::complex_arrow::ComplexArrow;
use mdfr::data_holder::tensor_arrow::{Order, TensorArrow};

// ──────────────────────────────────────────────────────────────────────────────
// Group 1: zeros() for variants not previously covered
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_zeros_virtual_channel_cn_type_3() {
    // cn_type 3 always returns UInt64 counter regardless of self type.
    let cd = ChannelData::Float64(Float64Builder::new());
    let result = cd.zeros(3, 5, 8, (vec![1], Order::RowMajor)).unwrap();
    assert!(matches!(result, ChannelData::UInt64(_)));
    assert_eq!(result.len(), 5);
    assert_eq!(result.to_u64_vec(), Some(vec![0, 1, 2, 3, 4]));
}

#[test]
fn test_zeros_virtual_channel_cn_type_6() {
    let cd = ChannelData::Float64(Float64Builder::new());
    let result = cd.zeros(6, 3, 8, (vec![1], Order::RowMajor)).unwrap();
    assert!(matches!(result, ChannelData::UInt64(_)));
    assert_eq!(result.len(), 3);
    assert_eq!(result.to_u64_vec(), Some(vec![0, 1, 2]));
}

#[test]
fn test_zeros_array_d_int16() {
    let cd = ChannelData::ArrayDInt16(TensorArrow::new());
    // shape=[2,3] → product=6, buffer = 6 i16 = 12 bytes
    let result = cd.zeros(0, 4, 6, (vec![2, 3], Order::RowMajor)).unwrap();
    assert!(matches!(result, ChannelData::ArrayDInt16(_)));
    // len = buffer_bytes / shape_product = 12 / 6 = 2
    assert!(!result.is_empty());
    // All values should be zero
    if let ChannelData::ArrayDInt16(ta) = &result {
        assert!(ta.values_slice().iter().all(|&v| v == 0));
    }
}

#[test]
fn test_zeros_array_d_uint8() {
    let cd = ChannelData::ArrayDUInt8(TensorArrow::new());
    let result = cd.zeros(0, 5, 1, (vec![4], Order::RowMajor)).unwrap();
    assert!(matches!(result, ChannelData::ArrayDUInt8(_)));
    assert!(!result.is_empty());
}

#[test]
fn test_zeros_fixed_size_byte() {
    let cd = ChannelData::FixedSizeByteArray(FixedSizeBinaryBuilder::with_capacity(1, 4));
    let result = cd.zeros(0, 3, 4, (vec![1], Order::RowMajor)).unwrap();
    assert!(matches!(result, ChannelData::FixedSizeByteArray(_)));
    assert_eq!(result.len(), 0); // zeros creates empty builder with capacity
}

#[test]
fn test_zeros_complex32() {
    let cd = ChannelData::Complex32(ComplexArrow::new());
    // zeros creates vec![0f32; cycle_count*2] = 2*2=4 f32 = 16 bytes
    // ComplexArrow::new_from_buffer: len = byte_len / 2 = 16 / 2 = 8
    let result = cd.zeros(0, 2, 8, (vec![1], Order::RowMajor)).unwrap();
    assert!(matches!(result, ChannelData::Complex32(_)));
    assert!(!result.is_empty());
}

#[test]
fn test_zeros_complex64() {
    let cd = ChannelData::Complex64(ComplexArrow::new());
    // zeros creates vec![0f64; cycle_count*2] = 3*2=6 f64 = 48 bytes
    // ComplexArrow::new_from_buffer: len = byte_len / 2 = 48 / 2 = 24
    let result = cd.zeros(0, 3, 16, (vec![1], Order::RowMajor)).unwrap();
    assert!(matches!(result, ChannelData::Complex64(_)));
    assert!(!result.is_empty());
}

#[test]
fn test_zeros_utf8() {
    let mut b = LargeStringBuilder::new();
    b.append_value("x");
    let cd = ChannelData::Utf8(b);
    let result = cd.zeros(0, 5, 10, (vec![1], Order::RowMajor)).unwrap();
    assert!(matches!(result, ChannelData::Utf8(_)));
    assert_eq!(result.len(), 0); // empty builder after zeros
}

#[test]
fn test_zeros_array_d_float64() {
    let cd = ChannelData::ArrayDFloat64(TensorArrow::new());
    // For ArrayDFloat64, zeros uses cycle_count * shape product
    // cycle_count=2, shape=[1] → buffer = 2*1 f64 = 16 bytes, len = 16/1 = 16
    let result = cd.zeros(0, 2, 8, (vec![1], Order::RowMajor)).unwrap();
    assert!(matches!(result, ChannelData::ArrayDFloat64(_)));
    assert!(!result.is_empty());
}

#[test]
fn test_zeros_array_d_int32() {
    let cd = ChannelData::ArrayDInt32(TensorArrow::new());
    let result = cd.zeros(0, 3, 4, (vec![3], Order::RowMajor)).unwrap();
    assert!(matches!(result, ChannelData::ArrayDInt32(_)));
    if let ChannelData::ArrayDInt32(ta) = &result {
        assert!(ta.values_slice().iter().all(|&v| v == 0));
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Group 2: len() for ArrayD variant using new_from_buffer
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_len_array_d_from_buffer() {
    // 4 f64 values, shape=[1]: buffer = 4*8 = 32 bytes, len = 32/1 = 32
    let buf = MutableBuffer::from_iter([1.0f64, 2.0, 3.0, 4.0].iter().copied());
    let ta = TensorArrow::<Float64Type>::new_from_buffer(buf, vec![1], Order::RowMajor);
    let cd = ChannelData::ArrayDFloat64(ta);
    // internal len formula: byte_len / shape_product = 32 / 1 = 32
    assert_eq!(cd.len(), 32);
    assert!(!cd.is_empty());
}

#[test]
fn test_len_array_d_int8_from_buffer() {
    // 6 i8 values, shape=[3]: buffer = 6 bytes, len = 6/3 = 2
    let buf = MutableBuffer::from_iter([1i8, 2, 3, 4, 5, 6].iter().copied());
    let ta = TensorArrow::<Int8Type>::new_from_buffer(buf, vec![3], Order::RowMajor);
    let cd = ChannelData::ArrayDInt8(ta);
    assert_eq!(cd.len(), 2);
    assert!(!cd.is_empty());
}

#[test]
fn test_len_array_d_uint8_from_buffer() {
    // 8 u8 values, shape=[2]: buffer = 8 bytes, len = 8/2 = 4
    let buf = MutableBuffer::from_iter([1u8, 2, 3, 4, 5, 6, 7, 8].iter().copied());
    let ta = TensorArrow::<UInt8Type>::new_from_buffer(buf, vec![2], Order::RowMajor);
    let cd = ChannelData::ArrayDUInt8(ta);
    assert_eq!(cd.len(), 4);
}

// ──────────────────────────────────────────────────────────────────────────────
// Group 3: min_max() for variants with new coverage
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_min_max_array_d_int16() {
    let buf = MutableBuffer::from_iter([10i16, -5, 20, 0].iter().copied());
    let ta = TensorArrow::<Int16Type>::new_from_buffer(buf, vec![1], Order::RowMajor);
    let cd = ChannelData::ArrayDInt16(ta);
    let (min, max) = cd.min_max();
    assert!(min.is_some());
    assert!(max.is_some());
    assert_eq!(min.unwrap(), -5.0);
    assert_eq!(max.unwrap(), 20.0);
}

#[test]
fn test_min_max_array_d_uint32() {
    let buf = MutableBuffer::from_iter([100u32, 50, 200, 1].iter().copied());
    let ta = TensorArrow::<UInt32Type>::new_from_buffer(buf, vec![1], Order::RowMajor);
    let cd = ChannelData::ArrayDUInt32(ta);
    let (min, max) = cd.min_max();
    assert_eq!(min.unwrap(), 1.0);
    assert_eq!(max.unwrap(), 200.0);
}

#[test]
fn test_min_max_array_d_uint64() {
    let buf = MutableBuffer::from_iter([5u64, 1, 100].iter().copied());
    let ta = TensorArrow::<UInt64Type>::new_from_buffer(buf, vec![1], Order::RowMajor);
    let cd = ChannelData::ArrayDUInt64(ta);
    let (min, max) = cd.min_max();
    assert_eq!(min.unwrap(), 1.0);
    assert_eq!(max.unwrap(), 100.0);
}

// ──────────────────────────────────────────────────────────────────────────────
// Group 4: to_u64_vec() for all integer variants not covered by internal tests
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_to_u64_vec_int8() {
    let mut b = Int8Builder::new();
    b.append_value(10);
    b.append_value(-1); // cast to u64: 0xFFFFFFFFFFFFFFFF as u64
    let cd = ChannelData::Int8(b);
    let v = cd.to_u64_vec().unwrap();
    assert_eq!(v.len(), 2);
    assert_eq!(v[0], 10u64);
    // -1i8 as u64 = 18446744073709551615
    assert_eq!(v[1], (-1i8) as u64);
}

#[test]
fn test_to_u64_vec_uint8() {
    let mut b = UInt8Builder::new();
    b.append_value(255u8);
    b.append_value(0u8);
    let cd = ChannelData::UInt8(b);
    assert_eq!(cd.to_u64_vec().unwrap(), vec![255u64, 0u64]);
}

#[test]
fn test_to_u64_vec_uint16() {
    let mut b = UInt16Builder::new();
    b.append_value(1000u16);
    let cd = ChannelData::UInt16(b);
    assert_eq!(cd.to_u64_vec().unwrap(), vec![1000u64]);
}

#[test]
fn test_to_u64_vec_uint32() {
    let mut b = UInt32Builder::new();
    b.append_value(u32::MAX);
    let cd = ChannelData::UInt32(b);
    assert_eq!(cd.to_u64_vec().unwrap(), vec![u32::MAX as u64]);
}

#[test]
fn test_to_u64_vec_uint64() {
    let mut b = UInt64Builder::new();
    b.append_value(u64::MAX);
    let cd = ChannelData::UInt64(b);
    assert_eq!(cd.to_u64_vec().unwrap(), vec![u64::MAX]);
}

#[test]
fn test_to_u64_vec_int64() {
    let mut b = Int64Builder::new();
    b.append_value(42i64);
    b.append_value(-100i64);
    let cd = ChannelData::Int64(b);
    let v = cd.to_u64_vec().unwrap();
    assert_eq!(v[0], 42u64);
    assert_eq!(v[1], (-100i64) as u64);
}

#[test]
fn test_to_u64_vec_int32() {
    let mut b = Int32Builder::new();
    b.append_value(7i32);
    b.append_value(-2i32);
    let cd = ChannelData::Int32(b);
    let v = cd.to_u64_vec().unwrap();
    assert_eq!(v[0], 7u64);
    assert_eq!(v[1], (-2i32) as u64);
}

#[test]
fn test_to_u64_vec_float64_returns_none() {
    let mut b = Float64Builder::new();
    b.append_value(1.0);
    let cd = ChannelData::Float64(b);
    assert!(cd.to_u64_vec().is_none());
}

#[test]
fn test_to_u64_vec_utf8_returns_none() {
    let mut b = LargeStringBuilder::new();
    b.append_value("abc");
    let cd = ChannelData::Utf8(b);
    assert!(cd.to_u64_vec().is_none());
}

// ──────────────────────────────────────────────────────────────────────────────
// Group 5: ndim() for more ArrayD variants
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_ndim_array_d_int16_2d() {
    let buf = MutableBuffer::from_iter([1i16, 2, 3, 4, 5, 6].iter().copied());
    let ta = TensorArrow::<Int16Type>::new_from_buffer(buf, vec![2, 3], Order::RowMajor);
    let cd = ChannelData::ArrayDInt16(ta);
    assert_eq!(cd.ndim(), 2);
}

#[test]
fn test_ndim_complex_is_1() {
    let cd = ChannelData::Complex32(ComplexArrow::new());
    assert_eq!(cd.ndim(), 1);
}

#[test]
fn test_ndim_fixed_size_byte_is_1() {
    let b = FixedSizeBinaryBuilder::with_capacity(1, 4);
    let cd = ChannelData::FixedSizeByteArray(b);
    assert_eq!(cd.ndim(), 1);
}

// ──────────────────────────────────────────────────────────────────────────────
// Group 6: Clone for FixedSizeByteArray with nulls
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_clone_fixed_size_byte_with_nulls() {
    let mut b = FixedSizeBinaryBuilder::with_capacity(3, 4);
    b.append_value(b"AAAA").unwrap();
    b.append_null();
    b.append_value(b"CCCC").unwrap();
    let cd = ChannelData::FixedSizeByteArray(b);
    let cloned = cd.clone();

    assert_eq!(cloned.len(), cd.len());
    assert!(matches!(cloned, ChannelData::FixedSizeByteArray(_)));
    if let ChannelData::FixedSizeByteArray(mut b) = cloned {
        let arr = b.finish();
        assert_eq!(arr.len(), 3);
        assert!(arr.is_null(1));
        assert!(!arr.is_null(0));
        assert!(!arr.is_null(2));
        assert_eq!(arr.value(0), b"AAAA");
        assert_eq!(arr.value(2), b"CCCC");
    }
}

#[test]
fn test_clone_utf8() {
    let mut b = LargeStringBuilder::new();
    b.append_value("hello");
    b.append_value("world");
    let cd = ChannelData::Utf8(b);
    let cloned = cd.clone();
    assert_eq!(cloned.len(), 2);
}

#[test]
fn test_clone_complex32() {
    let cd = ChannelData::Complex32(ComplexArrow::new());
    let cloned = cd.clone();
    assert_eq!(cloned.len(), 0);
    assert!(matches!(cloned, ChannelData::Complex32(_)));
}

// ──────────────────────────────────────────────────────────────────────────────
// Group 7: bit_count() and byte_count() for variants not covered by internal tests
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_bit_count_int8() {
    let mut b = Int8Builder::new();
    b.append_value(1);
    let cd = ChannelData::Int8(b);
    assert_eq!(cd.bit_count(), 8);
}

#[test]
fn test_bit_count_complex64() {
    let cd = ChannelData::Complex64(ComplexArrow::new());
    assert_eq!(cd.bit_count(), 128);
}

#[test]
fn test_bit_count_array_d_int8() {
    let cd = ChannelData::ArrayDInt8(TensorArrow::new());
    assert_eq!(cd.bit_count(), 8);
}

#[test]
fn test_bit_count_array_d_uint16() {
    let cd = ChannelData::ArrayDUInt16(TensorArrow::new());
    assert_eq!(cd.bit_count(), 16);
}

#[test]
fn test_byte_count_complex32() {
    let cd = ChannelData::Complex32(ComplexArrow::new());
    assert_eq!(cd.byte_count(), 8);
}

#[test]
fn test_byte_count_complex64() {
    let cd = ChannelData::Complex64(ComplexArrow::new());
    assert_eq!(cd.byte_count(), 16);
}

#[test]
fn test_byte_count_array_d_int16() {
    let cd = ChannelData::ArrayDInt16(TensorArrow::new());
    assert_eq!(cd.byte_count(), 2);
}

// ──────────────────────────────────────────────────────────────────────────────
// Group 8: is_empty() for variants not covered by internal tests
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_is_empty_array_d_float32() {
    let cd = ChannelData::ArrayDFloat32(TensorArrow::new());
    assert!(cd.is_empty());
}

#[test]
fn test_is_empty_array_d_int16() {
    let cd = ChannelData::ArrayDInt16(TensorArrow::new());
    assert!(cd.is_empty());
}

#[test]
fn test_is_not_empty_array_d() {
    let buf = MutableBuffer::from_iter([1.0f64, 2.0].iter().copied());
    let ta = TensorArrow::<Float64Type>::new_from_buffer(buf, vec![1], Order::RowMajor);
    let cd = ChannelData::ArrayDFloat64(ta);
    assert!(!cd.is_empty());
}

#[test]
fn test_is_empty_complex32() {
    let cd = ChannelData::Complex32(ComplexArrow::new());
    assert!(cd.is_empty());
}

#[test]
fn test_is_empty_fixed_size_byte() {
    let cd = ChannelData::FixedSizeByteArray(FixedSizeBinaryBuilder::with_capacity(0, 4));
    assert!(cd.is_empty());
}

// ──────────────────────────────────────────────────────────────────────────────
// Group 9: data_type() for uncovered variants
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_data_type_fixed_size_byte_array() {
    let b = FixedSizeBinaryBuilder::with_capacity(1, 4);
    let cd = ChannelData::FixedSizeByteArray(b);
    // LE: FixedSizeByteArray → 10; BE: also 10
    assert_eq!(cd.data_type(false), 10);
    assert_eq!(cd.data_type(true), 10);
}

#[test]
fn test_data_type_array_d_int8() {
    let cd = ChannelData::ArrayDInt8(TensorArrow::new());
    // LE: signed int → 2; BE: → 3
    assert_eq!(cd.data_type(false), 2);
    assert_eq!(cd.data_type(true), 3);
}

#[test]
fn test_data_type_array_d_uint8() {
    let cd = ChannelData::ArrayDUInt8(TensorArrow::new());
    // LE: unsigned int → 0; BE: → 1
    assert_eq!(cd.data_type(false), 0);
    assert_eq!(cd.data_type(true), 1);
}

#[test]
fn test_data_type_complex32() {
    let cd = ChannelData::Complex32(ComplexArrow::new());
    // LE: 15; BE: 16
    assert_eq!(cd.data_type(false), 15);
    assert_eq!(cd.data_type(true), 16);
}

#[test]
fn test_data_type_complex64() {
    let cd = ChannelData::Complex64(ComplexArrow::new());
    assert_eq!(cd.data_type(false), 15);
    assert_eq!(cd.data_type(true), 16);
}

#[test]
fn test_data_type_array_d_float32() {
    let cd = ChannelData::ArrayDFloat32(TensorArrow::new());
    // LE: 4; BE: 5
    assert_eq!(cd.data_type(false), 4);
    assert_eq!(cd.data_type(true), 5);
}

#[test]
fn test_data_type_array_d_int32() {
    let cd = ChannelData::ArrayDInt32(TensorArrow::new());
    assert_eq!(cd.data_type(false), 2);
    assert_eq!(cd.data_type(true), 3);
}

// ──────────────────────────────────────────────────────────────────────────────
// Group 10: as_u64_slice() for non-UInt64 variants
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_as_u64_slice_non_uint64_returns_none() {
    let mut b = Int32Builder::new();
    b.append_value(1);
    let cd = ChannelData::Int32(b);
    assert!(cd.as_u64_slice().is_none());
}

#[test]
fn test_as_u64_slice_float64_returns_none() {
    let mut b = Float64Builder::new();
    b.append_value(1.0);
    let cd = ChannelData::Float64(b);
    assert!(cd.as_u64_slice().is_none());
}

// ──────────────────────────────────────────────────────────────────────────────
// Group 11: byte_count() for additional scalar types
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_byte_count_int8() {
    let cd = ChannelData::Int8(Int8Builder::new());
    assert_eq!(cd.byte_count(), 1);
}

#[test]
fn test_byte_count_uint8() {
    let cd = ChannelData::UInt8(UInt8Builder::new());
    assert_eq!(cd.byte_count(), 1);
}

#[test]
fn test_byte_count_int16() {
    let cd = ChannelData::Int16(Int16Builder::new());
    assert_eq!(cd.byte_count(), 2);
}

#[test]
fn test_byte_count_int64() {
    let cd = ChannelData::Int64(Int64Builder::new());
    assert_eq!(cd.byte_count(), 8);
}

#[test]
fn test_byte_count_uint64() {
    let cd = ChannelData::UInt64(UInt64Builder::new());
    assert_eq!(cd.byte_count(), 8);
}

// ──────────────────────────────────────────────────────────────────────────────
// Group 12: shape() for additional variants
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_shape_fixed_size_byte() {
    let mut b = FixedSizeBinaryBuilder::with_capacity(2, 4);
    b.append_value(b"AAAA").unwrap();
    b.append_value(b"BBBB").unwrap();
    let cd = ChannelData::FixedSizeByteArray(b);
    let (shape, order) = cd.shape();
    assert_eq!(shape, vec![2]);
    assert_eq!(order, Order::RowMajor);
}

#[test]
fn test_shape_complex32_with_data() {
    use arrow::datatypes::Float32Type;
    // 4 f32 values = 16 bytes; ComplexArrow::new_from_buffer: len = 16/2 = 8
    let buf = MutableBuffer::from_iter([1.0f32, 2.0, 3.0, 4.0].iter().copied());
    let ca = mdfr::data_holder::complex_arrow::ComplexArrow::<Float32Type>::new_from_buffer(
        buf,
        vec![1],
        Order::RowMajor,
    );
    let cd = ChannelData::Complex32(ca);
    let (shape, order) = cd.shape();
    assert_eq!(shape, vec![8]); // len = byte_len / 2 = 16/2 = 8
    assert_eq!(order, Order::RowMajor);
}
