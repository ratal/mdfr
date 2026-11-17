//! Adversarial empirical stress tests for Milestone 4 (Features 28 through 32):
//! - Feature 28: TensorArrow per-sample dimensions with various tensor shapes (1D, 2D, 3D),
//!   cycle counts (0, 1, 100), shape() returning [d1, d2, ...], len() returning cycle count,
//!   and to_ndarray reshaping to (len, d1, d2, ...).
//! - Feature 29: NumPy export reshaping across all 10 ArrayD* types, verifying that shape is
//!   (N, d1, ...) and ArrayDUInt32 reports "u4".
//! - Feature 30: VariableSizeByteArray export with variable-length rows (empty, 1-byte, multi-KB)
//!   and verify row count and content match exactly in Python list of bytes.
//! - Feature 31: Polars export null preservation (primitives, strings, byte arrays, complex, tensors),
//!   ChannelData::try_from with BooleanArray (including nulls) and NullArray, and cut_channel_group
//!   zero-copy null buffer slicing.
//! - Feature 32: to_bytes memory copy with all numeric primitive types and multi-dimensional tensor arrays,
//!   verifying byte equality against .to_ne_bytes().

use anyhow::Result;
#[allow(unused_imports)]
use arrow::array::{
    Array, BooleanArray, BooleanBufferBuilder, FixedSizeBinaryArray, FixedSizeBinaryBuilder,
    Float32Array, Float32Builder, Float64Array, Float64Builder, Int8Array, Int8Builder, Int16Array,
    Int16Builder, Int32Array, Int32Builder, Int64Array, Int64Builder, LargeBinaryArray,
    LargeBinaryBuilder, LargeStringArray, LargeStringBuilder, NullArray, PrimitiveArray,
    UInt8Array, UInt8Builder, UInt16Array, UInt16Builder, UInt32Array, UInt32Builder, UInt64Array,
    UInt64Builder,
};
use arrow::buffer::{MutableBuffer, ScalarBuffer};
use arrow::datatypes::{
    Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type,
    UInt32Type, UInt64Type,
};
use mdfr::data_holder::channel_data::ChannelData;
use mdfr::data_holder::complex_arrow::ComplexArrow;
use mdfr::data_holder::tensor_arrow::{Order, TensorArrow};

// ══════════════════════════════════════════════════════════════════════════════
// Feature 28: TensorArrow Per-Sample Dimensions & ndarray Reshaping
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "ndarray")]
mod feature_28_tensor_arrow_stress {
    use super::*;

    macro_rules! verify_tensor_shape_and_ndarray {
        ($arrow_type:ty, $native:ty, $shape:expr, $cycle_count:expr, $order:expr, $val_generator:expr) => {{
            let shape: Vec<usize> = $shape;
            let cycle_count: usize = $cycle_count;
            let order: Order = $order;
            let elem_per_sample: usize = shape.iter().product();
            let total_elems = cycle_count * elem_per_sample;
            let values: Vec<$native> = (0..total_elems).map($val_generator).collect();

            let mut buf =
                MutableBuffer::with_capacity(total_elems * std::mem::size_of::<$native>());
            for v in &values {
                buf.extend_from_slice(std::slice::from_ref(v));
            }

            let tensor =
                TensorArrow::<$arrow_type>::new_from_buffer(buf, shape.clone(), order.clone());

            // 1. Verify shape() returns per-sample dimensions [d1, d2, ...] without cycle count
            assert_eq!(
                tensor.shape(),
                &shape,
                "Tensor shape() must return per-sample dimensions [d1, ...]"
            );
            assert_eq!(
                tensor.ndim(),
                shape.len(),
                "Tensor ndim() must match per-sample dimension count"
            );

            // 2. Verify len() returns cycle count
            assert_eq!(
                tensor.len(),
                cycle_count,
                "Tensor len() must return cycle count"
            );
            assert_eq!(
                tensor.is_empty(),
                cycle_count == 0,
                "Tensor is_empty() must reflect cycle count == 0"
            );

            // 3. Verify to_ndarray reshapes to (cycle_count, d1, d2, ...)
            let nd = tensor.to_ndarray().expect("to_ndarray must succeed");
            let mut expected_nd_shape = Vec::with_capacity(shape.len() + 1);
            expected_nd_shape.push(cycle_count);
            expected_nd_shape.extend_from_slice(&shape);

            assert_eq!(
                nd.shape(),
                expected_nd_shape.as_slice(),
                "to_ndarray shape must be (len, d1, d2, ...)"
            );

            // 4. Verify elements in ndarray match input buffer in order
            if cycle_count > 0 {
                let nd_flat: Vec<$native> = nd.iter().copied().collect();
                assert_eq!(
                    nd_flat, values,
                    "Flattened ndarray elements must match original values in order"
                );
            } else {
                assert_eq!(nd.len(), 0, "0-cycle ndarray must contain 0 elements");
            }

            // 5. Test slicing if cycle_count >= 10
            if cycle_count >= 10 {
                let slice_offset = 2;
                let slice_len = cycle_count - 5;
                let sliced = tensor.slice(slice_offset, slice_len);

                assert_eq!(
                    sliced.shape(),
                    &shape,
                    "Sliced tensor must preserve per-sample shape"
                );
                assert_eq!(
                    sliced.len(),
                    slice_len,
                    "Sliced tensor len must match requested slice length"
                );
                let sliced_nd = sliced.to_ndarray().expect("sliced to_ndarray");
                let mut expected_sliced_shape = Vec::with_capacity(shape.len() + 1);
                expected_sliced_shape.push(slice_len);
                expected_sliced_shape.extend_from_slice(&shape);
                assert_eq!(sliced_nd.shape(), expected_sliced_shape.as_slice());

                let expected_slice_values = &values
                    [slice_offset * elem_per_sample..(slice_offset + slice_len) * elem_per_sample];
                let sliced_flat: Vec<$native> = sliced_nd.iter().copied().collect();
                assert_eq!(sliced_flat, expected_slice_values);
            }
        }};
    }

    #[test]
    fn test_tensor_1d_shapes_across_cycle_counts() {
        let shapes = vec![vec![1], vec![5], vec![17]];
        let cycle_counts = [0, 1, 100];

        for shape in shapes {
            for &cycles in &cycle_counts {
                // Float64
                verify_tensor_shape_and_ndarray!(
                    Float64Type,
                    f64,
                    shape.clone(),
                    cycles,
                    Order::RowMajor,
                    |i| (i as f64) * 0.5 + 1.25
                );
                // Int32
                verify_tensor_shape_and_ndarray!(
                    Int32Type,
                    i32,
                    shape.clone(),
                    cycles,
                    Order::ColumnMajor,
                    |i| (i as i32) - 50
                );
            }
        }
    }

    #[test]
    fn test_tensor_2d_shapes_across_cycle_counts() {
        let shapes = vec![vec![2, 3], vec![4, 4], vec![1, 8], vec![7, 1]];
        let cycle_counts = [0, 1, 100];

        for shape in shapes {
            for &cycles in &cycle_counts {
                verify_tensor_shape_and_ndarray!(
                    Float32Type,
                    f32,
                    shape.clone(),
                    cycles,
                    Order::RowMajor,
                    |i| (i as f32) * 1.5 - 10.0
                );
                verify_tensor_shape_and_ndarray!(
                    UInt8Type,
                    u8,
                    shape.clone(),
                    cycles,
                    Order::ColumnMajor,
                    |i| (i % 255) as u8
                );
            }
        }
    }

    #[test]
    fn test_tensor_3d_shapes_across_cycle_counts() {
        let shapes = vec![vec![2, 3, 4], vec![1, 2, 5], vec![3, 1, 2]];
        let cycle_counts = [0, 1, 100];

        for shape in shapes {
            for &cycles in &cycle_counts {
                verify_tensor_shape_and_ndarray!(
                    Int16Type,
                    i16,
                    shape.clone(),
                    cycles,
                    Order::RowMajor,
                    |i| (i as i16) % 1000
                );
                verify_tensor_shape_and_ndarray!(
                    UInt32Type,
                    u32,
                    shape.clone(),
                    cycles,
                    Order::ColumnMajor,
                    |i| (i as u32) * 42
                );
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Feature 29: NumPy Export Reshaping & ArrayDUInt32 "u4" Dtype
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "numpy")]
mod feature_29_numpy_export_stress {
    use super::*;
    use pyo3::prelude::*;

    fn init_python() {
        Python::initialize();
    }

    fn make_tensor_channel<T, N>(
        shape: Vec<usize>,
        cycles: usize,
        wrap: impl Fn(TensorArrow<T>) -> ChannelData,
        val_fn: impl Fn(usize) -> N,
    ) -> ChannelData
    where
        T: arrow::datatypes::ArrowPrimitiveType<Native = N>,
        N: arrow::datatypes::ArrowNativeType,
    {
        let total = cycles * shape.iter().product::<usize>();
        let mut buf = MutableBuffer::with_capacity(total * std::mem::size_of::<N>());
        for i in 0..total {
            let v = val_fn(i);
            buf.extend_from_slice(std::slice::from_ref(&v));
        }
        let ta = TensorArrow::<T>::new_from_buffer(buf, shape, Order::RowMajor);
        wrap(ta)
    }

    #[test]
    fn test_numpy_get_dtype_all_10_types() {
        let shape = vec![2, 3];
        let n = 5;

        // 1. ArrayDInt8 -> i1
        let cd =
            make_tensor_channel::<Int8Type, i8>(shape.clone(), n, ChannelData::ArrayDInt8, |i| {
                i as i8
            });
        let dt = cd.get_dtype();
        assert_eq!(dt.shape, vec![n, 2, 3]);
        assert_eq!(dt.kind, "i1");

        // 2. ArrayDUInt8 -> u1
        let cd =
            make_tensor_channel::<UInt8Type, u8>(shape.clone(), n, ChannelData::ArrayDUInt8, |i| {
                i as u8
            });
        let dt = cd.get_dtype();
        assert_eq!(dt.shape, vec![n, 2, 3]);
        assert_eq!(dt.kind, "u1");

        // 3. ArrayDInt16 -> i2
        let cd = make_tensor_channel::<Int16Type, i16>(
            shape.clone(),
            n,
            ChannelData::ArrayDInt16,
            |i| i as i16,
        );
        let dt = cd.get_dtype();
        assert_eq!(dt.shape, vec![n, 2, 3]);
        assert_eq!(dt.kind, "i2");

        // 4. ArrayDUInt16 -> u2
        let cd = make_tensor_channel::<UInt16Type, u16>(
            shape.clone(),
            n,
            ChannelData::ArrayDUInt16,
            |i| i as u16,
        );
        let dt = cd.get_dtype();
        assert_eq!(dt.shape, vec![n, 2, 3]);
        assert_eq!(dt.kind, "u2");

        // 5. ArrayDInt32 -> i4
        let cd = make_tensor_channel::<Int32Type, i32>(
            shape.clone(),
            n,
            ChannelData::ArrayDInt32,
            |i| i as i32,
        );
        let dt = cd.get_dtype();
        assert_eq!(dt.shape, vec![n, 2, 3]);
        assert_eq!(dt.kind, "i4");

        // 6. ArrayDUInt32 -> MUST BE "u4"
        let cd = make_tensor_channel::<UInt32Type, u32>(
            shape.clone(),
            n,
            ChannelData::ArrayDUInt32,
            |i| i as u32,
        );
        let dt = cd.get_dtype();
        assert_eq!(
            dt.shape,
            vec![n, 2, 3],
            "ArrayDUInt32 shape must be (N, d1, d2)"
        );
        assert_eq!(
            dt.kind, "u4",
            "CRITICAL: ArrayDUInt32 get_dtype must report 'u4', not 'i4'"
        );

        // 7. ArrayDFloat32 -> f4
        let cd = make_tensor_channel::<Float32Type, f32>(
            shape.clone(),
            n,
            ChannelData::ArrayDFloat32,
            |i| i as f32,
        );
        let dt = cd.get_dtype();
        assert_eq!(dt.shape, vec![n, 2, 3]);
        assert_eq!(dt.kind, "f4");

        // 8. ArrayDInt64 -> i8
        let cd = make_tensor_channel::<Int64Type, i64>(
            shape.clone(),
            n,
            ChannelData::ArrayDInt64,
            |i| i as i64,
        );
        let dt = cd.get_dtype();
        assert_eq!(dt.shape, vec![n, 2, 3]);
        assert_eq!(dt.kind, "i8");

        // 9. ArrayDUInt64 -> u8
        let cd = make_tensor_channel::<UInt64Type, u64>(
            shape.clone(),
            n,
            ChannelData::ArrayDUInt64,
            |i| i as u64,
        );
        let dt = cd.get_dtype();
        assert_eq!(dt.shape, vec![n, 2, 3]);
        assert_eq!(dt.kind, "u8");

        // 10. ArrayDFloat64 -> f8
        let cd = make_tensor_channel::<Float64Type, f64>(
            shape.clone(),
            n,
            ChannelData::ArrayDFloat64,
            |i| i as f64,
        );
        let dt = cd.get_dtype();
        assert_eq!(dt.shape, vec![n, 2, 3]);
        assert_eq!(dt.kind, "f8");
    }

    #[test]
    fn test_numpy_python_export_reshaping_all_10_types() {
        init_python();

        let shape = vec![3, 2];
        let n = 4;

        Python::attach(|py| {
            let verify_py_ndarray = |cd: ChannelData,
                                     expected_kind: &str,
                                     expected_itemsize: usize,
                                     expected_name: &str| {
                let obj = cd.into_pyobject(py).expect("into_pyobject failed");
                let shape_out: Vec<usize> = obj
                    .getattr("shape")
                    .expect("shape")
                    .extract()
                    .expect("shape extract");
                assert_eq!(
                    shape_out,
                    vec![n, 3, 2],
                    "NumPy ndarray shape must be prepended with cycle count N"
                );

                let dtype = obj.getattr("dtype").expect("dtype");
                let kind: String = dtype
                    .getattr("kind")
                    .expect("kind")
                    .extract()
                    .expect("extract kind");
                let itemsize: usize = dtype
                    .getattr("itemsize")
                    .expect("itemsize")
                    .extract()
                    .expect("extract itemsize");
                let name: String = dtype
                    .getattr("name")
                    .expect("name")
                    .extract()
                    .expect("extract name");

                assert_eq!(
                    kind, expected_kind,
                    "NumPy dtype kind mismatch for {expected_name}"
                );
                assert_eq!(
                    itemsize, expected_itemsize,
                    "NumPy itemsize mismatch for {expected_name}"
                );
                assert_eq!(
                    name, expected_name,
                    "NumPy dtype name mismatch for {expected_name}"
                );
            };

            // 1. ArrayDInt8
            verify_py_ndarray(
                make_tensor_channel::<Int8Type, i8>(
                    shape.clone(),
                    n,
                    ChannelData::ArrayDInt8,
                    |i| i as i8,
                ),
                "i",
                1,
                "int8",
            );
            // 2. ArrayDUInt8
            verify_py_ndarray(
                make_tensor_channel::<UInt8Type, u8>(
                    shape.clone(),
                    n,
                    ChannelData::ArrayDUInt8,
                    |i| i as u8,
                ),
                "u",
                1,
                "uint8",
            );
            // 3. ArrayDInt16
            verify_py_ndarray(
                make_tensor_channel::<Int16Type, i16>(
                    shape.clone(),
                    n,
                    ChannelData::ArrayDInt16,
                    |i| i as i16,
                ),
                "i",
                2,
                "int16",
            );
            // 4. ArrayDUInt16
            verify_py_ndarray(
                make_tensor_channel::<UInt16Type, u16>(
                    shape.clone(),
                    n,
                    ChannelData::ArrayDUInt16,
                    |i| i as u16,
                ),
                "u",
                2,
                "uint16",
            );
            // 5. ArrayDInt32
            verify_py_ndarray(
                make_tensor_channel::<Int32Type, i32>(
                    shape.clone(),
                    n,
                    ChannelData::ArrayDInt32,
                    |i| i as i32,
                ),
                "i",
                4,
                "int32",
            );
            // 6. ArrayDUInt32 (Critical Feature 29 verification: shape (N, d1, d2) and dtype "u4")
            verify_py_ndarray(
                make_tensor_channel::<UInt32Type, u32>(
                    shape.clone(),
                    n,
                    ChannelData::ArrayDUInt32,
                    |i| i as u32,
                ),
                "u",
                4,
                "uint32",
            );
            // 7. ArrayDFloat32
            verify_py_ndarray(
                make_tensor_channel::<Float32Type, f32>(
                    shape.clone(),
                    n,
                    ChannelData::ArrayDFloat32,
                    |i| i as f32,
                ),
                "f",
                4,
                "float32",
            );
            // 8. ArrayDInt64
            verify_py_ndarray(
                make_tensor_channel::<Int64Type, i64>(
                    shape.clone(),
                    n,
                    ChannelData::ArrayDInt64,
                    |i| i as i64,
                ),
                "i",
                8,
                "int64",
            );
            // 9. ArrayDUInt64
            verify_py_ndarray(
                make_tensor_channel::<UInt64Type, u64>(
                    shape.clone(),
                    n,
                    ChannelData::ArrayDUInt64,
                    |i| i as u64,
                ),
                "u",
                8,
                "uint64",
            );
            // 10. ArrayDFloat64
            verify_py_ndarray(
                make_tensor_channel::<Float64Type, f64>(
                    shape.clone(),
                    n,
                    ChannelData::ArrayDFloat64,
                    |i| i as f64,
                ),
                "f",
                8,
                "float64",
            );
        });
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Feature 30: VariableSizeByteArray Export with Variable-Length Rows
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "numpy")]
mod feature_30_variable_size_byte_array_stress {
    use super::*;
    use pyo3::prelude::*;

    fn init_python() {
        Python::initialize();
    }

    #[test]
    fn test_variable_size_byte_array_rows_exact_match() {
        init_python();

        // Construct distinct rows spanning edge cases:
        // - Row 0: empty (0 bytes)
        // - Row 1: 1 byte
        // - Row 2: 32 bytes binary with null bytes and 0xFF
        // - Row 3: 4096 bytes (4 KB)
        // - Row 4: empty (0 bytes)
        // - Row 5: 8192 bytes (8 KB)
        // - Row 6: 1 byte 0x00
        // - Row 7: 65536 bytes (64 KB)
        // - Row 8: ASCII string bytes
        let test_rows: Vec<Vec<u8>> = vec![
            vec![],     // 0 bytes
            vec![0x7A], // 1 byte
            (0..32)
                .map(|i| if i % 2 == 0 { 0x00 } else { 0xFF })
                .collect(), // 32 bytes with nulls
            vec![0xA5; 4096], // 4 KB
            vec![],     // empty row
            (0..8192).map(|i| (i % 251) as u8).collect(), // 8 KB pseudo-random pattern
            vec![0x00], // 1 byte 0x00
            vec![0x3C; 65536], // 64 KB
            b"arbitrary utf-8 compatible trailing byte row".to_vec(),
        ];

        let mut builder = LargeBinaryBuilder::with_capacity(test_rows.len(), 100_000);
        for row in &test_rows {
            builder.append_value(row.as_slice());
        }
        let array = builder.finish();
        let cd = ChannelData::VariableSizeByteArray(array);

        Python::attach(|py| {
            let py_obj = cd
                .into_pyobject(py)
                .expect("into_pyobject for VariableSizeByteArray");

            // Verify type is list
            let type_name: String = py_obj.get_type().name().expect("type name").to_string();
            assert_eq!(type_name, "list", "Must convert into a Python list");

            // Verify length
            let list_len: usize = py_obj
                .call_method0("__len__")
                .expect("len")
                .extract()
                .expect("len extract");
            assert_eq!(
                list_len,
                test_rows.len(),
                "List length must exactly equal number of rows in VariableSizeByteArray"
            );

            for (i, expected_row) in test_rows.iter().enumerate() {
                let item = py_obj.get_item(i).expect("get_item");
                let item_type: String = item.get_type().name().expect("item type").to_string();
                assert_eq!(item_type, "bytes", "List element at row {i} must be bytes");

                let actual_bytes: Vec<u8> = item.extract().expect("extract bytes");
                assert_eq!(
                    actual_bytes.len(),
                    expected_row.len(),
                    "Row {i} byte length mismatch (expected {}, got {})",
                    expected_row.len(),
                    actual_bytes.len()
                );
                assert_eq!(&actual_bytes, expected_row, "Row {i} byte content mismatch");
            }
        });
    }

    #[test]
    fn test_variable_size_byte_array_empty_array() {
        init_python();

        let mut builder = LargeBinaryBuilder::new();
        let array = builder.finish();
        let cd = ChannelData::VariableSizeByteArray(array);

        Python::attach(|py| {
            let py_obj = cd
                .into_pyobject(py)
                .expect("into_pyobject for empty VariableSizeByteArray");
            let type_name: String = py_obj.get_type().name().expect("type name").to_string();
            assert_eq!(type_name, "list");
            let list_len: usize = py_obj
                .call_method0("__len__")
                .expect("len")
                .extract()
                .expect("len extract");
            assert_eq!(
                list_len, 0,
                "Empty VariableSizeByteArray must produce 0-length PyList"
            );
        });
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Feature 31: Polars Export Null Preservation, Arrow Types & cut_channel_group Slicing
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "polars")]
mod feature_31_polars_null_preservation_and_slicing_stress {
    use super::*;
    use mdfr::export::polars::channel_data_to_series;
    use mdfr::mdfreader::Mdf;
    use polars::prelude::df;

    #[test]
    fn test_polars_null_preservation_primitives() -> Result<()> {
        macro_rules! test_primitive_nulls {
            ($builder_type:ident, $variant:ident, [$($val:expr),*], [$($is_some:expr),*]) => {{
                let mut b = $builder_type::new();
                let vals = [$($val),*];
                let somes = [$($is_some),*];
                for (&v, &is_some) in vals.iter().zip(somes.iter()) {
                    if is_some {
                        b.append_value(v);
                    } else {
                        b.append_null();
                    }
                }
                let arr = b.finish();
                let cd = ChannelData::$variant(arr);
                let series = channel_data_to_series("test_prim", &cd)?;

                assert_eq!(series.len(), vals.len());
                let expected_null_count = somes.iter().filter(|&&s| !s).count();
                assert_eq!(series.null_count(), expected_null_count, "Null count mismatch for {}", stringify!($variant));

                let null_mask = series.is_null();
                for (i, &is_some) in somes.iter().enumerate() {
                    let is_null = null_mask.get(i) == Some(true);
                    assert_eq!(is_null, !is_some, "is_null mismatch at index {i} for {}", stringify!($variant));
                }
            }};
        }

        test_primitive_nulls!(
            Int8Builder,
            Int8,
            [1i8, 0, -5, 120, -10],
            [true, false, true, false, true]
        );
        test_primitive_nulls!(
            UInt8Builder,
            UInt8,
            [10u8, 0, 255, 42, 7],
            [false, true, true, false, true]
        );
        test_primitive_nulls!(
            Int16Builder,
            Int16,
            [-1000i16, 0, 5000, -1, 300],
            [true, false, false, true, true]
        );
        test_primitive_nulls!(
            UInt16Builder,
            UInt16,
            [1000u16, 0, 5000, 1, 300],
            [false, false, true, true, true]
        );
        test_primitive_nulls!(
            Int32Builder,
            Int32,
            [100000i32, -2, 0, 42, -999],
            [true, false, true, false, true]
        );
        test_primitive_nulls!(
            UInt32Builder,
            UInt32,
            [100000u32, 2, 0, 42, 999],
            [true, true, false, false, true]
        );
        test_primitive_nulls!(
            Float32Builder,
            Float32,
            [1.5f32, -2.5, 0.0, 99.9, -0.01],
            [true, false, true, false, true]
        );
        test_primitive_nulls!(
            Int64Builder,
            Int64,
            [100000000i64, -2, 0, 42, -999],
            [false, true, true, false, false]
        );
        test_primitive_nulls!(
            UInt64Builder,
            UInt64,
            [100000000u64, 2, 0, 42, 999],
            [true, false, true, false, true]
        );
        test_primitive_nulls!(
            Float64Builder,
            Float64,
            [1.5e10f64, -2.5e-3, 0.0, 99.9, -0.01],
            [true, false, false, true, false]
        );

        Ok(())
    }

    #[test]
    fn test_polars_null_preservation_strings_and_byte_arrays() -> Result<()> {
        // 1. Strings (Utf8)
        let mut str_b = LargeStringBuilder::new();
        str_b.append_value("alpha");
        str_b.append_null();
        str_b.append_value("beta");
        str_b.append_null();
        str_b.append_value("");
        let cd_str = ChannelData::Utf8(str_b.finish());
        let s_str = channel_data_to_series("str", &cd_str)?;
        assert_eq!(s_str.len(), 5);
        assert_eq!(s_str.null_count(), 2);
        let mask = s_str.is_null();
        assert_eq!(mask.get(0), Some(false));
        assert_eq!(mask.get(1), Some(true));
        assert_eq!(mask.get(2), Some(false));
        assert_eq!(mask.get(3), Some(true));
        assert_eq!(mask.get(4), Some(false));

        // 2. FixedSizeByteArray
        let mut fb = FixedSizeBinaryBuilder::new(3);
        fb.append_value(b"ABC")?;
        fb.append_null();
        fb.append_value(b"XYZ")?;
        fb.append_null();
        let cd_fb = ChannelData::FixedSizeByteArray(fb.finish());
        let s_fb = channel_data_to_series("fixed_bytes", &cd_fb)?;
        assert_eq!(s_fb.len(), 4);
        assert_eq!(s_fb.null_count(), 2);
        let mask = s_fb.is_null();
        assert_eq!(mask.get(0), Some(false));
        assert_eq!(mask.get(1), Some(true));
        assert_eq!(mask.get(2), Some(false));
        assert_eq!(mask.get(3), Some(true));

        // 3. VariableSizeByteArray
        let mut vb = LargeBinaryBuilder::new();
        vb.append_value(b"one");
        vb.append_null();
        vb.append_value(b"three_long_payload");
        vb.append_null();
        vb.append_value(b"");
        let cd_vb = ChannelData::VariableSizeByteArray(vb.finish());
        let s_vb = channel_data_to_series("var_bytes", &cd_vb)?;
        assert_eq!(s_vb.len(), 5);
        assert_eq!(s_vb.null_count(), 2);
        let mask = s_vb.is_null();
        assert_eq!(mask.get(0), Some(false));
        assert_eq!(mask.get(1), Some(true));
        assert_eq!(mask.get(2), Some(false));
        assert_eq!(mask.get(3), Some(true));
        assert_eq!(mask.get(4), Some(false));

        Ok(())
    }

    #[test]
    fn test_polars_null_preservation_complex_and_tensors() -> Result<()> {
        // Complex32 null preservation
        let cx_vals: Vec<f32> = vec![
            1.0, 2.0, // sample 0: valid
            0.0, 0.0, // sample 1: null
            5.0, 6.0, // sample 2: valid
            0.0, 0.0, // sample 3: null
        ];
        let cx_buf = MutableBuffer::from_iter(cx_vals.iter().copied());
        let mut cx_mask = BooleanBufferBuilder::new(4);
        cx_mask.append(true);
        cx_mask.append(false);
        cx_mask.append(true);
        cx_mask.append(false);
        let cx_null_buf = cx_mask.finish();

        let cx = ComplexArrow::<Float32Type>::new_from_array(
            PrimitiveArray::<Float32Type>::new(ScalarBuffer::<f32>::from(cx_buf), None),
            Some(cx_null_buf),
            vec![1],
            Order::RowMajor,
        );
        let cd_cx = ChannelData::Complex32(cx);
        let s_cx = channel_data_to_series("complex_series", &cd_cx)?;

        assert_eq!(s_cx.len(), 4);
        assert_eq!(s_cx.null_count(), 2);
        let mask = s_cx.is_null();
        assert_eq!(mask.get(0), Some(false));
        assert_eq!(mask.get(1), Some(true));
        assert_eq!(mask.get(2), Some(false));
        assert_eq!(mask.get(3), Some(true));

        // TensorArray (ArrayDFloat64) null preservation: 4 samples of shape [2, 2] = 16 elements
        let tensor_vals: Vec<f64> = (0..16).map(|x| x as f64).collect();
        let t_buf = MutableBuffer::from_iter(tensor_vals.iter().copied());
        let mut t_mask = BooleanBufferBuilder::new(4);
        t_mask.append(true); // sample 0: valid
        t_mask.append(false); // sample 1: null
        t_mask.append(false); // sample 2: null
        t_mask.append(true); // sample 3: valid
        let t_null_buf = t_mask.finish();

        let ta = TensorArrow::<Float64Type>::new_from_array(
            PrimitiveArray::<Float64Type>::new(ScalarBuffer::<f64>::from(t_buf), None),
            Some(t_null_buf),
            vec![2, 2],
            Order::RowMajor,
        );
        let cd_ta = ChannelData::ArrayDFloat64(ta);
        let s_ta = channel_data_to_series("tensor_series", &cd_ta)?;

        assert_eq!(s_ta.len(), 4);
        assert_eq!(s_ta.null_count(), 2);
        let mask = s_ta.is_null();
        assert_eq!(mask.get(0), Some(false));
        assert_eq!(mask.get(1), Some(true));
        assert_eq!(mask.get(2), Some(true));
        assert_eq!(mask.get(3), Some(false));

        Ok(())
    }

    #[test]
    fn test_channel_data_try_from_boolean_and_null_arrays() -> Result<()> {
        // 1. BooleanArray with mixed nulls
        let bool_vals = vec![Some(true), Some(false), None, Some(true), None, Some(false)];
        let bool_arr = BooleanArray::from(bool_vals);

        let cd = mdfr::data_holder::channel_data::try_from(&bool_arr as &dyn Array)?;
        match cd {
            ChannelData::UInt8(u8_arr) => {
                assert_eq!(u8_arr.len(), 6);
                assert_eq!(u8_arr.null_count(), 2);
                assert_eq!(u8_arr.value(0), 1);
                assert!(!u8_arr.is_null(0));
                assert_eq!(u8_arr.value(1), 0);
                assert!(!u8_arr.is_null(1));
                assert!(u8_arr.is_null(2));
                assert_eq!(u8_arr.value(3), 1);
                assert!(!u8_arr.is_null(3));
                assert!(u8_arr.is_null(4));
                assert_eq!(u8_arr.value(5), 0);
                assert!(!u8_arr.is_null(5));
            }
            other => panic!("Expected UInt8 variant for BooleanArray, got {other:?}"),
        }

        // 2. NullArray
        let null_arr = NullArray::new(24);
        let cd_null = mdfr::data_holder::channel_data::try_from(&null_arr as &dyn Array)?;
        match cd_null {
            ChannelData::UInt8(u8_arr) => {
                assert_eq!(u8_arr.len(), 24);
                assert_eq!(u8_arr.null_count(), 24);
                for i in 0..24 {
                    assert!(u8_arr.is_null(i));
                }
            }
            other => panic!("Expected UInt8 variant for NullArray, got {other:?}"),
        }

        Ok(())
    }

    #[test]
    fn test_cut_channel_group_zero_copy_null_buffer_slicing() -> Result<()> {
        let time_col: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let i32_col: Vec<Option<i32>> = vec![
            Some(10),
            None,
            Some(30),
            None,
            Some(50),
            Some(60),
            None,
            Some(80),
            None,
            Some(100),
        ];
        let str_col: Vec<Option<&str>> = vec![
            Some("s0"),
            None,
            Some("s2"),
            None,
            Some("s4"),
            Some("s5"),
            None,
            Some("s7"),
            None,
            Some("s9"),
        ];

        let df = df!(
            "Time" => &time_col,
            "Value_I32" => &i32_col,
            "Value_Str" => &str_col,
        )?;

        let mut mdf = Mdf::from_polars_df(&df, Some("Time"))?;

        // Slice from 2.0 s to 7.0 s inclusive (indices 2 to 7 inclusive: length = 6 samples)
        mdf.cut_channel_group("Time", 2.0, 7.0)?;

        // Verify master Time channel sliced
        let time_data = mdf.get_channel_data("Time").expect("Time channel exists");
        assert_eq!(time_data.len(), 6);
        if let ChannelData::Float64(t_arr) = time_data {
            assert_eq!(t_arr.values().as_ref(), &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        } else {
            panic!("Expected Float64 for master Time");
        }

        // Verify Value_I32 channel sliced with exact null offset preservation
        let i32_data = mdf.get_channel_data("Value_I32").expect("Value_I32 exists");
        assert_eq!(i32_data.len(), 6);
        if let ChannelData::Int32(arr) = i32_data {
            assert_eq!(arr.null_count(), 2);
            assert_eq!(arr.value(0), 30);
            assert!(!arr.is_null(0));
            assert!(arr.is_null(1));
            assert_eq!(arr.value(2), 50);
            assert!(!arr.is_null(2));
            assert_eq!(arr.value(3), 60);
            assert!(!arr.is_null(3));
            assert!(arr.is_null(4));
            assert_eq!(arr.value(5), 80);
            assert!(!arr.is_null(5));
        } else {
            panic!("Expected Int32 for Value_I32");
        }

        // Verify Value_Str channel sliced with exact null offset preservation
        let str_data = mdf.get_channel_data("Value_Str").expect("Value_Str exists");
        assert_eq!(str_data.len(), 6);
        if let ChannelData::Utf8(arr) = str_data {
            assert_eq!(arr.null_count(), 2);
            assert_eq!(arr.value(0), "s2");
            assert!(!arr.is_null(0));
            assert!(arr.is_null(1));
            assert_eq!(arr.value(2), "s4");
            assert!(!arr.is_null(2));
            assert_eq!(arr.value(3), "s5");
            assert!(!arr.is_null(3));
            assert!(arr.is_null(4));
            assert_eq!(arr.value(5), "s7");
            assert!(!arr.is_null(5));
        } else {
            panic!("Expected Utf8 for Value_Str");
        }

        // Direct test of slice_range on complex and tensor channels to verify zero-copy null slicing
        let cx_vals: Vec<f64> = (0..20).map(|x| x as f64).collect(); // 10 pairs
        let cx_buf = MutableBuffer::from_iter(cx_vals.iter().copied());
        let mut cx_mask = BooleanBufferBuilder::new(10);
        for i in 0..10 {
            cx_mask.append(i % 3 != 0); // null at 0, 3, 6, 9
        }
        let ca = ComplexArrow::<Float64Type>::new_from_array(
            PrimitiveArray::<Float64Type>::new(ScalarBuffer::<f64>::from(cx_buf), None),
            Some(cx_mask.finish()),
            vec![1],
            Order::RowMajor,
        );
        let cd_cx = ChannelData::Complex64(ca);
        let sliced_cx = cd_cx.slice_range(2, 8)?; // slice [2..8) len 6. Orig nulls at 3, 6 -> new nulls at 1, 4
        assert_eq!(sliced_cx.len(), 6);
        if let ChannelData::Complex64(c) = sliced_cx {
            let nulls = c.nulls().expect("nulls exist");
            assert_eq!(nulls.len(), 6);
            assert!(nulls.value(0)); // orig 2 (valid)
            assert!(!nulls.value(1)); // orig 3 (null)
            assert!(nulls.value(2)); // orig 4 (valid)
            assert!(nulls.value(3)); // orig 5 (valid)
            assert!(!nulls.value(4)); // orig 6 (null)
            assert!(nulls.value(5)); // orig 7 (valid)
        }

        Ok(())
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Feature 32: to_bytes Memory Copy Verification against .to_ne_bytes()
// ══════════════════════════════════════════════════════════════════════════════

mod feature_32_to_bytes_memory_copy_stress {
    use super::*;

    #[test]
    fn test_to_bytes_all_10_primitives() -> Result<()> {
        macro_rules! test_prim_to_bytes {
            ($builder:ident, $variant:ident, $native:ty, $count:expr, $val_expr:expr) => {{
                let mut b = $builder::with_capacity($count);
                let mut expected_bytes: Vec<u8> =
                    Vec::with_capacity($count * std::mem::size_of::<$native>());
                for i in 0usize..$count {
                    let v: $native = $val_expr(i);
                    b.append_value(v);
                    expected_bytes.extend_from_slice(&v.to_ne_bytes());
                }
                let cd = ChannelData::$variant(b.finish());
                let actual_bytes = cd.to_bytes()?;
                assert_eq!(
                    actual_bytes,
                    expected_bytes,
                    "to_bytes mismatch for {}",
                    stringify!($variant)
                );
            }};
        }

        test_prim_to_bytes!(Int8Builder, Int8, i8, 256, |i| (i as i8).wrapping_mul(7));
        test_prim_to_bytes!(UInt8Builder, UInt8, u8, 256, |i| (i as u8).wrapping_mul(13));
        test_prim_to_bytes!(Int16Builder, Int16, i16, 500, |i| {
            (i as i16).wrapping_mul(-73) ^ 0x1234
        });
        test_prim_to_bytes!(UInt16Builder, UInt16, u16, 500, |i| {
            (i as u16).wrapping_mul(101) ^ 0xABCD
        });
        test_prim_to_bytes!(Int32Builder, Int32, i32, 1000, |i| {
            (i as i32).wrapping_mul(-123456).wrapping_add(42)
        });
        test_prim_to_bytes!(UInt32Builder, UInt32, u32, 1000, |i| {
            (i as u32).wrapping_mul(987654) ^ 0xDEADBEEF
        });
        test_prim_to_bytes!(Int64Builder, Int64, i64, 1000, |i| {
            (i as i64).wrapping_mul(-9876543210123).wrapping_add(77)
        });
        test_prim_to_bytes!(UInt64Builder, UInt64, u64, 1000, |i| {
            (i as u64).wrapping_mul(1234567890987) ^ 0xCAFEBABE_FEEDFACE
        });
        test_prim_to_bytes!(Float32Builder, Float32, f32, 1000, |i| (i as f32)
            * std::f32::consts::PI
            - 100.5);
        test_prim_to_bytes!(Float64Builder, Float64, f64, 1000, |i| (i as f64)
            * std::f64::consts::E
            + 1234.5678);

        Ok(())
    }

    #[test]
    fn test_to_bytes_complex_types() -> Result<()> {
        let pairs_count = 500;
        let mut cx32_vals: Vec<f32> = Vec::with_capacity(pairs_count * 2);
        let mut cx32_expected: Vec<u8> = Vec::with_capacity(pairs_count * 8);
        for i in 0..pairs_count {
            let re = (i as f32) * 1.5;
            let im = (i as f32) * -0.75;
            cx32_vals.push(re);
            cx32_vals.push(im);
            cx32_expected.extend_from_slice(&re.to_ne_bytes());
            cx32_expected.extend_from_slice(&im.to_ne_bytes());
        }
        let ca32 = ComplexArrow::<Float32Type>::new_from_buffer(
            MutableBuffer::from_iter(cx32_vals.iter().copied()),
            vec![1],
            Order::RowMajor,
        );
        let cd_cx32 = ChannelData::Complex32(ca32);
        assert_eq!(cd_cx32.to_bytes()?, cx32_expected);

        let mut cx64_vals: Vec<f64> = Vec::with_capacity(pairs_count * 2);
        let mut cx64_expected: Vec<u8> = Vec::with_capacity(pairs_count * 16);
        for i in 0..pairs_count {
            let re = (i as f64) * std::f64::consts::PI;
            let im = (i as f64) * std::f64::consts::LN_2;
            cx64_vals.push(re);
            cx64_vals.push(im);
            cx64_expected.extend_from_slice(&re.to_ne_bytes());
            cx64_expected.extend_from_slice(&im.to_ne_bytes());
        }
        let ca64 = ComplexArrow::<Float64Type>::new_from_buffer(
            MutableBuffer::from_iter(cx64_vals.iter().copied()),
            vec![1],
            Order::RowMajor,
        );
        let cd_cx64 = ChannelData::Complex64(ca64);
        assert_eq!(cd_cx64.to_bytes()?, cx64_expected);

        Ok(())
    }

    #[test]
    fn test_to_bytes_all_10_tensor_types() -> Result<()> {
        let shape = vec![2, 3, 4]; // 24 elements per cycle
        let cycle_count = 50;
        let total_elements = cycle_count * 24;

        macro_rules! test_tensor_to_bytes {
            ($variant:ident, $arrow_type:ty, $native:ty, $val_expr:expr) => {{
                let mut vals: Vec<$native> = Vec::with_capacity(total_elements);
                let mut expected_bytes: Vec<u8> =
                    Vec::with_capacity(total_elements * std::mem::size_of::<$native>());
                for i in 0..total_elements {
                    let v: $native = $val_expr(i);
                    vals.push(v);
                    expected_bytes.extend_from_slice(&v.to_ne_bytes());
                }
                let buf = MutableBuffer::from_iter(vals.iter().copied());
                let ta = TensorArrow::<$arrow_type>::new_from_buffer(
                    buf,
                    shape.clone(),
                    Order::RowMajor,
                );
                let cd = ChannelData::$variant(ta);
                let actual_bytes = cd.to_bytes()?;
                assert_eq!(
                    actual_bytes,
                    expected_bytes,
                    "to_bytes mismatch for tensor variant {}",
                    stringify!($variant)
                );
            }};
        }

        test_tensor_to_bytes!(ArrayDInt8, Int8Type, i8, |i| (i as i8).wrapping_mul(3));
        test_tensor_to_bytes!(ArrayDUInt8, UInt8Type, u8, |i| (i as u8).wrapping_mul(17));
        test_tensor_to_bytes!(ArrayDInt16, Int16Type, i16, |i| (i as i16)
            .wrapping_mul(-19));
        test_tensor_to_bytes!(ArrayDUInt16, UInt16Type, u16, |i| (i as u16)
            .wrapping_mul(47));
        test_tensor_to_bytes!(ArrayDInt32, Int32Type, i32, |i| (i as i32)
            .wrapping_mul(-12345));
        test_tensor_to_bytes!(ArrayDUInt32, UInt32Type, u32, |i| (i as u32)
            .wrapping_mul(67890));
        test_tensor_to_bytes!(ArrayDInt64, Int64Type, i64, |i| (i as i64)
            .wrapping_mul(-987654321));
        test_tensor_to_bytes!(ArrayDUInt64, UInt64Type, u64, |i| (i as u64)
            .wrapping_mul(123456789));
        test_tensor_to_bytes!(ArrayDFloat32, Float32Type, f32, |i| (i as f32)
            * std::f32::consts::PI);
        test_tensor_to_bytes!(ArrayDFloat64, Float64Type, f64, |i| (i as f64)
            * std::f64::consts::E);

        Ok(())
    }

    #[test]
    fn test_to_bytes_empty_and_large_arrays() -> Result<()> {
        // Empty array
        let empty_buf = MutableBuffer::new(0);
        let ta_empty =
            TensorArrow::<Float64Type>::new_from_buffer(empty_buf, vec![3, 3], Order::RowMajor);
        let cd_empty = ChannelData::ArrayDFloat64(ta_empty);
        assert_eq!(cd_empty.to_bytes()?, Vec::<u8>::new());

        // Large array (50,000 f64 values)
        let count = 50_000;
        let mut expected = Vec::with_capacity(count * 8);
        let mut buf = MutableBuffer::with_capacity(count * 8);
        for i in 0..count {
            let v = (i as f64) * 0.123456789;
            expected.extend_from_slice(&v.to_ne_bytes());
            buf.extend_from_slice(std::slice::from_ref(&v));
        }
        let ta_large = TensorArrow::<Float64Type>::new_from_buffer(buf, vec![1], Order::RowMajor);
        let cd_large = ChannelData::ArrayDFloat64(ta_large);
        let actual = cd_large.to_bytes()?;
        assert_eq!(actual.len(), count * 8);
        assert_eq!(actual, expected);

        Ok(())
    }
}
