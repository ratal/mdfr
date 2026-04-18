//! Smoke tests for numpy and polars exports using embedded Python (PyO3).
//! Both modules require a live Python interpreter; we use pyo3::prepare_freethreaded_python().
//! No maturin/wheel build is needed — cargo test suffices.

#[cfg(feature = "numpy")]
mod numpy_tests {
    use arrow::array::{Float64Builder, Int16Builder, LargeStringBuilder};
    use arrow::buffer::MutableBuffer;
    use arrow::datatypes::{Float32Type, Float64Type};
    use mdfr::data_holder::channel_data::ChannelData;
    use mdfr::data_holder::complex_arrow::ComplexArrow;
    use mdfr::data_holder::tensor_arrow::{Order, TensorArrow};
    use pyo3::prelude::*;

    fn init() {
        Python::initialize();
    }

    #[test]
    fn numpy_float64_correct_values() {
        init();
        let mut b = Float64Builder::with_capacity(4);
        b.append_slice(&[1.0_f64, 2.0, 3.0, 4.0]);
        let cd = ChannelData::Float64(b);
        Python::attach(|py| {
            let obj = cd.into_pyobject(py).unwrap();
            let dtype_name: String = obj
                .getattr("dtype")
                .unwrap()
                .getattr("name")
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(dtype_name, "float64");
            let size: usize = obj.getattr("size").unwrap().extract().unwrap();
            assert_eq!(size, 4);
            let values: Vec<f64> = obj.call_method0("tolist").unwrap().extract().unwrap();
            assert_eq!(values, vec![1.0_f64, 2.0, 3.0, 4.0]);
        });
    }

    #[test]
    fn numpy_int16_correct_dtype() {
        init();
        let mut b = Int16Builder::with_capacity(4);
        b.append_slice(&[-100_i16, 0, 100, 200]);
        let cd = ChannelData::Int16(b);
        Python::attach(|py| {
            let obj = cd.into_pyobject(py).unwrap();
            let dtype_name: String = obj
                .getattr("dtype")
                .unwrap()
                .getattr("name")
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(dtype_name, "int16");
        });
    }

    #[test]
    fn numpy_complex32_correct_dtype() {
        init();
        // 4 complex32 samples stored as interleaved f32 pairs: (1+2j, 3+4j, 5+6j, 7+8j)
        let buf = MutableBuffer::from_iter(
            [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].iter().copied(),
        );
        let ca = ComplexArrow::<Float32Type>::new_from_buffer(buf);
        let cd = ChannelData::Complex32(ca);
        Python::attach(|py| {
            let obj = cd.into_pyobject(py).unwrap();
            // values_slice() returns &[f32], so numpy dtype is float32 (interleaved re/im)
            let dtype_name: String = obj
                .getattr("dtype")
                .unwrap()
                .getattr("name")
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(dtype_name, "float32");
            let size: usize = obj.getattr("size").unwrap().extract().unwrap();
            assert_eq!(size, 8); // 4 pairs × 2 components
        });
    }

    #[test]
    fn numpy_arrayd_float64_correct_shape() {
        init();
        // 4 samples × 3 elements each = 12 f64 values, shape [4, 3]
        let vals: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let buf = MutableBuffer::from_iter(vals.iter().copied());
        let ta = TensorArrow::<Float64Type>::new_from_buffer(buf, vec![4, 3], Order::RowMajor);
        let cd = ChannelData::ArrayDFloat64(ta);
        Python::attach(|py| {
            let obj = cd.into_pyobject(py).unwrap();
            let ndim: usize = obj.getattr("ndim").unwrap().extract().unwrap();
            assert_eq!(ndim, 2);
            let shape: Vec<usize> = obj.getattr("shape").unwrap().extract().unwrap();
            assert_eq!(shape, vec![4usize, 3]);
        });
    }

    #[test]
    fn numpy_utf8_converts_to_list() {
        init();
        let mut b = LargeStringBuilder::new();
        b.append_value("hello");
        b.append_value("world");
        b.append_value("!");
        let cd = ChannelData::Utf8(b);
        Python::attach(|py| {
            let obj = cd.into_pyobject(py).unwrap();
            let len: usize = obj.call_method0("__len__").unwrap().extract().unwrap();
            assert_eq!(len, 3);
        });
    }
}

#[cfg(feature = "polars")]
mod polars_tests {
    use arrow::array::{Float64Array, Int64Array};
    use mdfr::export::polars::rust_arrow_to_py_series;
    use pyo3::prelude::*;
    use std::sync::Arc;

    fn init() {
        Python::initialize();
    }

    #[test]
    fn polars_series_name_and_length() {
        init();
        let arr: Arc<dyn arrow::array::Array> =
            Arc::new(Float64Array::from(vec![1.0f64, 2.0, 3.0, 4.0]));
        let result = rust_arrow_to_py_series(arr, "ch".to_string()).unwrap();
        Python::attach(|py| {
            let obj = result.bind(py);
            // rust_arrow_to_py_series returns a polars Series
            let name: String = obj.getattr("name").unwrap().extract().unwrap();
            assert_eq!(name, "ch");
            let len: usize = obj.call_method0("__len__").unwrap().extract().unwrap();
            assert_eq!(len, 4);
        });
    }

    #[test]
    fn polars_series_values_match() {
        init();
        let arr: Arc<dyn arrow::array::Array> =
            Arc::new(Int64Array::from(vec![1_i64, 2, 3, 4]));
        let result = rust_arrow_to_py_series(arr, "vals".to_string()).unwrap();
        Python::attach(|py| {
            let obj = result.bind(py);
            let values: Vec<i64> = obj.call_method0("to_list").unwrap().extract().unwrap();
            assert_eq!(values, vec![1_i64, 2, 3, 4]);
        });
    }
}
