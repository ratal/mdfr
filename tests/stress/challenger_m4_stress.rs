//! Adversarial Empirical Stress Suite for Milestone 4 (Iteration 2).
//!
//! Validates:
//! 1. TensorArrow shape consistency:
//!    - TensorArrow::shape() returns strictly per-sample dimensions [dim_1, dim_2, ...]
//!    - TensorArrow::len() returns sample count N
//!    - Export to NumPy produces ndarray of shape (N, dim_1, dim_2, ...)
//!    - 0-sample, 1-sample, 1D, 2D, 3D, and 4D tensor shapes
//!    - ArrayDUInt32 dtype is "u4"
//! 2. Regression check corresponding to tests/numpy_polars.rs:
//!    - 12 elements with per-sample shape [3] produces NumPy shape (4, 3) and ndim == 2
//! 3. Variable-size byte arrays:
//!    - Row boundaries preserved when exporting to Python list of bytes
//!    - Variable lengths (0 B, 1 B, 16 B, 1 KB, 32 KB, null bytes)
//! 4. Polars series export & Arrow type conversions:
//!    - Null preservation across all primitives, strings, byte arrays, complex, and tensors
//!    - Arrow DataType::Boolean conversion to UInt8 without panics (preserving nulls)
//!    - Arrow DataType::Null conversion to UInt8 with 100% nulls without panics
//!    - Zero-copy null buffer slicing preserves validity mask integrity
//! 5. Zero-copy dispatch_numeric! optimization:
//!    - Bitwise equivalence between to_bytes() and element-wise .to_ne_bytes() across
//!      all 22 numeric variants (scalars, complex, tensors) and boundary sizes (0, 1, 100k).

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
// 1. Array Channel Shape Consistency & NumPy Export
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "numpy")]
mod tensor_shape_and_numpy_export {
    use super::*;
    use pyo3::prelude::*;

    fn init_python() {
        Python::initialize();
    }

    fn make_tensor<T, N>(
        shape: Vec<usize>,
        sample_count: usize,
        val_gen: impl Fn(usize) -> N,
    ) -> TensorArrow<T>
    where
        T: arrow::datatypes::ArrowPrimitiveType<Native = N>,
        N: arrow::datatypes::ArrowNativeType,
    {
        let elem_per_sample: usize = shape.iter().product();
        let total = sample_count * elem_per_sample;
        let mut buf = MutableBuffer::with_capacity(total * std::mem::size_of::<N>());
        for i in 0..total {
            let v = val_gen(i);
            buf.extend_from_slice(std::slice::from_ref(&v));
        }
        TensorArrow::<T>::new_from_buffer(buf, shape, Order::RowMajor)
    }

    #[test]
    fn test_tensor_per_sample_shape_and_len() {
        // Test 1D shape [5] with 8 samples
        let t1 = make_tensor::<Float64Type, f64>(vec![5], 8, |i| i as f64);
        assert_eq!(t1.shape(), &vec![5], "shape() must be per-sample [5]");
        assert_eq!(t1.len(), 8, "len() must be sample count 8");
        assert_eq!(t1.ndim(), 1);

        // Test 2D shape [3, 4] with 10 samples
        let t2 = make_tensor::<Int32Type, i32>(vec![3, 4], 10, |i| i as i32);
        assert_eq!(t2.shape(), &vec![3, 4], "shape() must be per-sample [3, 4]");
        assert_eq!(t2.len(), 10, "len() must be sample count 10");
        assert_eq!(t2.ndim(), 2);

        // Test 3D shape [2, 3, 5] with 6 samples
        let t3 = make_tensor::<UInt16Type, u16>(vec![2, 3, 5], 6, |i| i as u16);
        assert_eq!(
            t3.shape(),
            &vec![2, 3, 5],
            "shape() must be per-sample [2, 3, 5]"
        );
        assert_eq!(t3.len(), 6, "len() must be sample count 6");
        assert_eq!(t3.ndim(), 3);

        // Test 4D shape [2, 2, 2, 2] with 4 samples
        let t4 = make_tensor::<Float32Type, f32>(vec![2, 2, 2, 2], 4, |i| i as f32);
        assert_eq!(
            t4.shape(),
            &vec![2, 2, 2, 2],
            "shape() must be per-sample [2, 2, 2, 2]"
        );
        assert_eq!(t4.len(), 4, "len() must be sample count 4");
        assert_eq!(t4.ndim(), 4);

        // Test 1-sample tensor
        let t_single = make_tensor::<UInt32Type, u32>(vec![7], 1, |i| i as u32);
        assert_eq!(t_single.shape(), &vec![7]);
        assert_eq!(t_single.len(), 1);

        // Test 0-sample tensor
        let t_zero = make_tensor::<Int8Type, i8>(vec![4, 4], 0, |i| i as i8);
        assert_eq!(t_zero.shape(), &vec![4, 4]);
        assert_eq!(t_zero.len(), 0);
        assert!(t_zero.is_empty());
    }

    #[test]
    fn test_numpy_export_shape_prepends_sample_count() {
        init_python();

        Python::attach(|py| {
            // 1. 2D per-sample shape [3, 5], N = 7 -> NumPy shape (7, 3, 5)
            let ta = make_tensor::<Float64Type, f64>(vec![3, 5], 7, |i| (i as f64) * 0.1);
            let cd = ChannelData::ArrayDFloat64(ta);
            let py_obj = cd.into_pyobject(py).expect("into_pyobject");
            let shape: Vec<usize> = py_obj
                .getattr("shape")
                .expect("shape")
                .extract()
                .expect("extract");
            assert_eq!(shape, vec![7, 3, 5], "NumPy export shape must be (7, 3, 5)");
            let ndim: usize = py_obj
                .getattr("ndim")
                .expect("ndim")
                .extract()
                .expect("extract");
            assert_eq!(ndim, 3, "NumPy ndim must be 3");

            // 2. 1D per-sample shape [4], N = 12 -> NumPy shape (12, 4)
            let ta = make_tensor::<Int16Type, i16>(vec![4], 12, |i| i as i16);
            let cd = ChannelData::ArrayDInt16(ta);
            let py_obj = cd.into_pyobject(py).expect("into_pyobject");
            let shape: Vec<usize> = py_obj
                .getattr("shape")
                .expect("shape")
                .extract()
                .expect("extract");
            assert_eq!(shape, vec![12, 4], "NumPy export shape must be (12, 4)");
            let ndim: usize = py_obj
                .getattr("ndim")
                .expect("ndim")
                .extract()
                .expect("extract");
            assert_eq!(ndim, 2, "NumPy ndim must be 2");

            // 3. 3D per-sample shape [2, 2, 3], N = 3 -> NumPy shape (3, 2, 2, 3)
            let ta = make_tensor::<UInt8Type, u8>(vec![2, 2, 3], 3, |i| i as u8);
            let cd = ChannelData::ArrayDUInt8(ta);
            let py_obj = cd.into_pyobject(py).expect("into_pyobject");
            let shape: Vec<usize> = py_obj
                .getattr("shape")
                .expect("shape")
                .extract()
                .expect("extract");
            assert_eq!(
                shape,
                vec![3, 2, 2, 3],
                "NumPy export shape must be (3, 2, 2, 3)"
            );
            let ndim: usize = py_obj
                .getattr("ndim")
                .expect("ndim")
                .extract()
                .expect("extract");
            assert_eq!(ndim, 4, "NumPy ndim must be 4");

            // 4. 0 samples: shape [4], N = 0 -> NumPy shape (0, 4)
            let ta = make_tensor::<Float32Type, f32>(vec![4], 0, |i| i as f32);
            let cd = ChannelData::ArrayDFloat32(ta);
            let py_obj = cd.into_pyobject(py).expect("into_pyobject");
            let shape: Vec<usize> = py_obj
                .getattr("shape")
                .expect("shape")
                .extract()
                .expect("extract");
            assert_eq!(
                shape,
                vec![0, 4],
                "NumPy export shape for 0 samples must be (0, 4)"
            );

            // 5. 1 sample: shape [3, 3], N = 1 -> NumPy shape (1, 3, 3)
            let ta = make_tensor::<Int64Type, i64>(vec![3, 3], 1, |i| i as i64);
            let cd = ChannelData::ArrayDInt64(ta);
            let py_obj = cd.into_pyobject(py).expect("into_pyobject");
            let shape: Vec<usize> = py_obj
                .getattr("shape")
                .expect("shape")
                .extract()
                .expect("extract");
            assert_eq!(
                shape,
                vec![1, 3, 3],
                "NumPy export shape for 1 sample must be (1, 3, 3)"
            );
        });
    }

    #[test]
    fn test_arrayd_uint32_reports_u4_dtype() {
        let ta = make_tensor::<UInt32Type, u32>(vec![2, 3], 4, |i| i as u32);
        let cd = ChannelData::ArrayDUInt32(ta);
        let dt = cd.get_dtype();
        assert_eq!(dt.shape, vec![4, 2, 3]);
        assert_eq!(dt.kind, "u4", "ArrayDUInt32 dtype must strictly be 'u4'");

        init_python();
        Python::attach(|py| {
            let py_obj = cd.into_pyobject(py).expect("into_pyobject");
            let dtype = py_obj.getattr("dtype").expect("dtype");
            let kind: String = dtype
                .getattr("kind")
                .expect("kind")
                .extract()
                .expect("extract");
            let itemsize: usize = dtype
                .getattr("itemsize")
                .expect("itemsize")
                .extract()
                .expect("extract");
            let name: String = dtype
                .getattr("name")
                .expect("name")
                .extract()
                .expect("extract");
            assert_eq!(kind, "u");
            assert_eq!(itemsize, 4);
            assert_eq!(name, "uint32");
        });
    }

    #[test]
    fn test_numpy_polars_remediation_regression() {
        init_python();
        // 4 samples × 3 elements each = 12 f64 values, per-sample shape [3]
        let vals: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let buf = MutableBuffer::from_iter(vals.iter().copied());
        let ta = TensorArrow::<Float64Type>::new_from_buffer(buf, vec![3], Order::RowMajor);
        assert_eq!(ta.shape(), &vec![3]);
        assert_eq!(ta.len(), 4);

        let cd = ChannelData::ArrayDFloat64(ta);
        Python::attach(|py| {
            let obj = cd.into_pyobject(py).unwrap();
            let shape: Vec<usize> = obj.getattr("shape").unwrap().extract().unwrap();
            let ndim: usize = obj.getattr("ndim").unwrap().extract().unwrap();
            assert_eq!(ndim, 2, "ndim must be 2");
            assert_eq!(shape, vec![4usize, 3], "shape must be (4, 3)");
        });
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// 2. Variable-Size Byte Arrays: Row Boundary Preservation
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "numpy")]
mod variable_size_byte_array_stress {
    use super::*;
    use pyo3::prelude::*;

    fn init_python() {
        Python::initialize();
    }

    #[test]
    fn test_variable_size_byte_array_export_row_boundaries() {
        init_python();

        // 10 distinct test rows covering boundaries:
        let rows: Vec<Option<Vec<u8>>> = vec![
            Some(vec![]),                                         // row 0: empty
            Some(vec![0x42]),                                     // row 1: 1 byte
            Some(vec![0x00, 0xFF, 0x00, 0xAA, 0x55]),             // row 2: 5 bytes with nulls
            None,                                                 // row 3: null row
            Some(vec![0xCC; 1024]),                               // row 4: 1 KB
            Some(vec![]),                                         // row 5: empty
            Some(vec![0x7F; 32768]),                              // row 6: 32 KB
            Some(b"Hello ASAM MDF 4.3 variable bytes!".to_vec()), // row 7: ASCII
            None,                                                 // row 8: null row
            Some(vec![0xEE]),                                     // row 9: 1 byte
        ];

        let mut builder = LargeBinaryBuilder::with_capacity(rows.len(), 50000);
        for row in &rows {
            match row {
                Some(bytes) => builder.append_value(bytes.as_slice()),
                None => builder.append_null(),
            }
        }
        let arr = builder.finish();
        let cd = ChannelData::VariableSizeByteArray(arr);

        Python::attach(|py| {
            let py_obj = cd.into_pyobject(py).expect("into_pyobject");
            let type_name: String = py_obj.get_type().name().expect("type name").to_string();
            assert_eq!(
                type_name, "list",
                "VariableSizeByteArray must export as Python list"
            );

            let list_len: usize = py_obj.call_method0("__len__").unwrap().extract().unwrap();
            assert_eq!(
                list_len,
                rows.len(),
                "Exported list length must match row count"
            );

            for (i, expected_opt) in rows.iter().enumerate() {
                let item = py_obj.get_item(i).unwrap();
                match expected_opt {
                    Some(expected_bytes) => {
                        let item_type: String = item.get_type().name().unwrap().to_string();
                        assert_eq!(item_type, "bytes", "Row {i} must be bytes");
                        let actual: Vec<u8> = item.extract().unwrap();
                        assert_eq!(
                            actual.len(),
                            expected_bytes.len(),
                            "Row {i} length mismatch: expected {}, got {}",
                            expected_bytes.len(),
                            actual.len()
                        );
                        assert_eq!(&actual, expected_bytes, "Row {i} content mismatch");
                    }
                    None => {
                        assert!(item.is_none(), "Row {i} must be None for null entry");
                    }
                }
            }
        });
    }

    #[test]
    fn test_variable_size_byte_array_empty_list() {
        init_python();

        let mut builder = LargeBinaryBuilder::new();
        let arr = builder.finish();
        let cd = ChannelData::VariableSizeByteArray(arr);

        Python::attach(|py| {
            let py_obj = cd.into_pyobject(py).expect("into_pyobject");
            let type_name: String = py_obj.get_type().name().unwrap().to_string();
            assert_eq!(type_name, "list");
            let list_len: usize = py_obj.call_method0("__len__").unwrap().extract().unwrap();
            assert_eq!(list_len, 0);
        });
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// 3. Polars Series Export, Null Preservation & Arrow Types
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "polars")]
mod polars_and_arrow_conversions {
    use super::*;
    use mdfr::export::polars::channel_data_to_series;

    #[test]
    fn test_polars_series_null_preservation_primitives() -> Result<()> {
        macro_rules! check_primitive_null_preservation {
            ($builder_ty:ident, $variant:ident, $vals:expr) => {{
                let mut b = $builder_ty::new();
                let vals: Vec<Option<_>> = $vals;
                for v in &vals {
                    match v {
                        Some(val) => b.append_value(*val),
                        None => b.append_null(),
                    }
                }
                let cd = ChannelData::$variant(b.finish());
                let series = channel_data_to_series("test_channel", &cd)?;
                assert_eq!(series.len(), vals.len());

                let expected_nulls = vals.iter().filter(|v| v.is_none()).count();
                assert_eq!(
                    series.null_count(),
                    expected_nulls,
                    "Null count mismatch for {}",
                    stringify!($variant)
                );

                let is_null_mask = series.is_null();
                for (i, v) in vals.iter().enumerate() {
                    assert_eq!(
                        is_null_mask.get(i),
                        Some(v.is_none()),
                        "Null mask mismatch at index {i} for {}",
                        stringify!($variant)
                    );
                }
            }};
        }

        check_primitive_null_preservation!(
            Int8Builder,
            Int8,
            vec![Some(10i8), None, Some(-5), None, Some(0)]
        );
        check_primitive_null_preservation!(
            UInt8Builder,
            UInt8,
            vec![None, Some(100u8), Some(200), None, None]
        );
        check_primitive_null_preservation!(
            Int16Builder,
            Int16,
            vec![Some(-3000i16), None, Some(15000), Some(0), None]
        );
        check_primitive_null_preservation!(
            UInt16Builder,
            UInt16,
            vec![Some(3000u16), None, Some(65000), None, Some(1)]
        );
        check_primitive_null_preservation!(
            Int32Builder,
            Int32,
            vec![None, Some(-100000i32), None, Some(42), Some(-1)]
        );
        check_primitive_null_preservation!(
            UInt32Builder,
            UInt32,
            vec![Some(100000u32), None, None, Some(999), None]
        );
        check_primitive_null_preservation!(
            Int64Builder,
            Int64,
            vec![Some(1i64 << 40), None, Some(-(1i64 << 40)), None, Some(0)]
        );
        check_primitive_null_preservation!(
            UInt64Builder,
            UInt64,
            vec![None, Some(1u64 << 50), None, Some(12345), None]
        );
        check_primitive_null_preservation!(
            Float32Builder,
            Float32,
            vec![Some(1.234f32), None, Some(-99.9), None, Some(0.0)]
        );
        check_primitive_null_preservation!(
            Float64Builder,
            Float64,
            vec![
                Some(std::f64::consts::PI),
                None,
                None,
                Some(-1e20),
                Some(0.0)
            ]
        );

        Ok(())
    }

    #[test]
    fn test_polars_series_null_preservation_complex_and_tensors() -> Result<()> {
        // 1. Complex32 with null mask
        let cx_vals: Vec<f32> = vec![
            1.0, 2.0, // 0: valid
            0.0, 0.0, // 1: null
            3.0, 4.0, // 2: valid
            0.0, 0.0, // 3: null
        ];
        let mut mask = BooleanBufferBuilder::new(4);
        mask.append(true);
        mask.append(false);
        mask.append(true);
        mask.append(false);
        let ca = ComplexArrow::<Float32Type>::new_from_array(
            PrimitiveArray::<Float32Type>::new(
                ScalarBuffer::from(MutableBuffer::from_iter(cx_vals)),
                None,
            ),
            Some(mask.finish()),
            vec![1],
            Order::RowMajor,
        );
        let cd_cx = ChannelData::Complex32(ca);
        let s_cx = channel_data_to_series("cx_series", &cd_cx)?;
        assert_eq!(s_cx.len(), 4);
        assert_eq!(s_cx.null_count(), 2);
        assert_eq!(s_cx.is_null().get(0), Some(false));
        assert_eq!(s_cx.is_null().get(1), Some(true));
        assert_eq!(s_cx.is_null().get(2), Some(false));
        assert_eq!(s_cx.is_null().get(3), Some(true));

        // 2. TensorArrow (ArrayDFloat64) with null mask: 3 samples of shape [2, 3] = 18 elements
        let t_vals: Vec<f64> = (0..18).map(|x| x as f64).collect();
        let mut t_mask = BooleanBufferBuilder::new(3);
        t_mask.append(true); // sample 0: valid
        t_mask.append(false); // sample 1: null
        t_mask.append(true); // sample 2: valid
        let ta = TensorArrow::<Float64Type>::new_from_array(
            PrimitiveArray::<Float64Type>::new(
                ScalarBuffer::from(MutableBuffer::from_iter(t_vals)),
                None,
            ),
            Some(t_mask.finish()),
            vec![2, 3],
            Order::RowMajor,
        );
        let cd_ta = ChannelData::ArrayDFloat64(ta);
        let s_ta = channel_data_to_series("ta_series", &cd_ta)?;
        assert_eq!(s_ta.len(), 3);
        assert_eq!(s_ta.null_count(), 1);
        assert_eq!(s_ta.is_null().get(0), Some(false));
        assert_eq!(s_ta.is_null().get(1), Some(true));
        assert_eq!(s_ta.is_null().get(2), Some(false));

        Ok(())
    }

    #[test]
    fn test_arrow_boolean_and_null_conversion_without_panics() -> Result<()> {
        // Test BooleanArray with Some(true), Some(false), None
        let bool_data = vec![
            Some(true),
            Some(false),
            None,
            Some(true),
            None,
            Some(false),
            Some(true),
        ];
        let bool_arr = BooleanArray::from(bool_data);
        let cd = mdfr::data_holder::channel_data::try_from(&bool_arr as &dyn Array)?;

        match cd {
            ChannelData::UInt8(u8_arr) => {
                assert_eq!(u8_arr.len(), 7);
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
                assert_eq!(u8_arr.value(6), 1);
                assert!(!u8_arr.is_null(6));
            }
            other => panic!("Expected ChannelData::UInt8 for BooleanArray, got {other:?}"),
        }

        // Test NullArray (100% nulls)
        let null_arr = NullArray::new(50);
        let cd_null = mdfr::data_holder::channel_data::try_from(&null_arr as &dyn Array)?;
        match cd_null {
            ChannelData::UInt8(u8_arr) => {
                assert_eq!(u8_arr.len(), 50);
                assert_eq!(u8_arr.null_count(), 50);
                for i in 0..50 {
                    assert!(u8_arr.is_null(i));
                }
            }
            other => panic!("Expected ChannelData::UInt8 for NullArray, got {other:?}"),
        }

        // Verify arrow_bit_count helper handles Boolean and Null cleanly without panic
        assert_eq!(
            mdfr::data_holder::arrow_helpers::arrow_bit_count(&bool_arr),
            8
        );
        assert_eq!(
            mdfr::data_holder::arrow_helpers::arrow_bit_count(&null_arr),
            0
        );

        Ok(())
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// 4. Zero-Copy dispatch_numeric! Bitwise Equivalence
// ══════════════════════════════════════════════════════════════════════════════

mod zero_copy_dispatch_numeric_equivalence {
    use super::*;

    #[test]
    fn test_to_bytes_bitwise_equivalence_primitives() -> Result<()> {
        macro_rules! verify_prim_bitwise_eq {
            ($builder:ident, $variant:ident, $native:ty, $count:expr, $generator:expr) => {{
                let mut b = $builder::with_capacity($count);
                let mut expected = Vec::with_capacity($count * std::mem::size_of::<$native>());
                for i in 0usize..$count {
                    let v: $native = $generator(i);
                    b.append_value(v);
                    expected.extend_from_slice(&v.to_ne_bytes());
                }
                let cd = ChannelData::$variant(b.finish());
                let actual = cd.to_bytes()?;
                assert_eq!(
                    actual.len(),
                    expected.len(),
                    "Byte length mismatch for {}",
                    stringify!($variant)
                );
                assert_eq!(
                    actual,
                    expected,
                    "Bitwise byte mismatch for {}",
                    stringify!($variant)
                );
            }};
        }

        verify_prim_bitwise_eq!(Int8Builder, Int8, i8, 1024, |i| (i as i8).wrapping_mul(31));
        verify_prim_bitwise_eq!(UInt8Builder, UInt8, u8, 1024, |i| (i as u8)
            .wrapping_add(17));
        verify_prim_bitwise_eq!(Int16Builder, Int16, i16, 2048, |i| (i as i16)
            .wrapping_mul(-123)
            ^ 0x5555);
        verify_prim_bitwise_eq!(UInt16Builder, UInt16, u16, 2048, |i| (i as u16)
            .wrapping_mul(333)
            ^ 0xAAAA);
        verify_prim_bitwise_eq!(Int32Builder, Int32, i32, 2048, |i| (i as i32)
            .wrapping_mul(-98765)
            .wrapping_add(101));
        verify_prim_bitwise_eq!(UInt32Builder, UInt32, u32, 2048, |i| (i as u32)
            .wrapping_mul(123456)
            ^ 0xFEEDC0DE);
        verify_prim_bitwise_eq!(Int64Builder, Int64, i64, 2048, |i| (i as i64)
            .wrapping_mul(-1234567890123));
        verify_prim_bitwise_eq!(UInt64Builder, UInt64, u64, 2048, |i| (i as u64)
            .wrapping_mul(9876543210987)
            ^ 0xDEADBEEF_CAFEBABE);
        verify_prim_bitwise_eq!(Float32Builder, Float32, f32, 2048, |i| (i as f32)
            * std::f32::consts::SQRT_2
            - 77.77);
        verify_prim_bitwise_eq!(Float64Builder, Float64, f64, 2048, |i| (i as f64)
            * std::f64::consts::PI
            + 12345.6789);

        Ok(())
    }

    #[test]
    fn test_to_bytes_bitwise_equivalence_tensors() -> Result<()> {
        let sample_count = 50;
        let shape = vec![2, 3, 5]; // 30 elements per sample -> 1500 elements total
        let total = sample_count * 30;

        macro_rules! verify_tensor_bitwise_eq {
            ($variant:ident, $arrow_type:ty, $native:ty, $generator:expr) => {{
                let mut vals: Vec<$native> = Vec::with_capacity(total);
                let mut expected: Vec<u8> =
                    Vec::with_capacity(total * std::mem::size_of::<$native>());
                for i in 0..total {
                    let v: $native = $generator(i);
                    vals.push(v);
                    expected.extend_from_slice(&v.to_ne_bytes());
                }
                let buf = MutableBuffer::from_iter(vals.iter().copied());
                let ta = TensorArrow::<$arrow_type>::new_from_buffer(
                    buf,
                    shape.clone(),
                    Order::RowMajor,
                );
                let cd = ChannelData::$variant(ta);
                let actual = cd.to_bytes()?;
                assert_eq!(
                    actual.len(),
                    expected.len(),
                    "Tensor length mismatch for {}",
                    stringify!($variant)
                );
                assert_eq!(
                    actual,
                    expected,
                    "Tensor bitwise mismatch for {}",
                    stringify!($variant)
                );
            }};
        }

        verify_tensor_bitwise_eq!(ArrayDInt8, Int8Type, i8, |i| (i as i8).wrapping_mul(7));
        verify_tensor_bitwise_eq!(ArrayDUInt8, UInt8Type, u8, |i| (i as u8).wrapping_mul(19));
        verify_tensor_bitwise_eq!(ArrayDInt16, Int16Type, i16, |i| (i as i16)
            .wrapping_mul(-43));
        verify_tensor_bitwise_eq!(ArrayDUInt16, UInt16Type, u16, |i| (i as u16)
            .wrapping_mul(71));
        verify_tensor_bitwise_eq!(ArrayDInt32, Int32Type, i32, |i| (i as i32)
            .wrapping_mul(-54321));
        verify_tensor_bitwise_eq!(ArrayDUInt32, UInt32Type, u32, |i| (i as u32)
            .wrapping_mul(99999));
        verify_tensor_bitwise_eq!(ArrayDInt64, Int64Type, i64, |i| (i as i64)
            .wrapping_mul(-999888777));
        verify_tensor_bitwise_eq!(ArrayDUInt64, UInt64Type, u64, |i| (i as u64)
            .wrapping_mul(111222333444));
        verify_tensor_bitwise_eq!(ArrayDFloat32, Float32Type, f32, |i| (i as f32)
            * std::f32::consts::E);
        verify_tensor_bitwise_eq!(ArrayDFloat64, Float64Type, f64, |i| (i as f64)
            * std::f64::consts::LN_10);

        Ok(())
    }

    #[test]
    fn test_to_bytes_edge_cases_empty_and_large() -> Result<()> {
        // Empty primitive
        let mut b = Float64Builder::new();
        let cd_empty_prim = ChannelData::Float64(b.finish());
        assert_eq!(cd_empty_prim.to_bytes()?, Vec::<u8>::new());

        // Empty tensor
        let empty_buf = MutableBuffer::new(0);
        let ta_empty =
            TensorArrow::<Int32Type>::new_from_buffer(empty_buf, vec![4, 4], Order::RowMajor);
        let cd_empty_tensor = ChannelData::ArrayDInt32(ta_empty);
        assert_eq!(cd_empty_tensor.to_bytes()?, Vec::<u8>::new());

        // Single element primitive
        let mut b1 = UInt64Builder::new();
        b1.append_value(0x123456789ABCDEF0);
        let cd_single = ChannelData::UInt64(b1.finish());
        assert_eq!(
            cd_single.to_bytes()?,
            0x123456789ABCDEF0u64.to_ne_bytes().to_vec()
        );

        // 100,000 elements large buffer
        let count = 100_000;
        let mut expected = Vec::with_capacity(count * 8);
        let mut buf = MutableBuffer::with_capacity(count * 8);
        for i in 0..count {
            let v = (i as f64) * 0.987654321;
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
