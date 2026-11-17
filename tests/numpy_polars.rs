//! Smoke tests for numpy and polars exports using embedded Python (PyO3).
//! Both modules require a live Python interpreter; we use pyo3::prepare_freethreaded_python().
//! No maturin/wheel build is needed — cargo test suffices.

mod common;

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
        let cd = ChannelData::Float64(b.finish());
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
        let cd = ChannelData::Int16(b.finish());
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
        let buf =
            MutableBuffer::from_iter([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].iter().copied());
        let ca = ComplexArrow::<Float32Type>::new_from_buffer(
            buf,
            vec![1],
            mdfr::data_holder::tensor_arrow::Order::RowMajor,
        );
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
        let ta = TensorArrow::<Float64Type>::new_from_buffer(buf, vec![3], Order::RowMajor);
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
    fn numpy_arrayd_column_major_per_sample() {
        init();
        // 2D: 2 samples of shape [3, 4] stored column-major per sample:
        // sample s, element [i, j] = s * 100 + i * 10 + j
        let mut raw_2d = Vec::with_capacity(2 * 3 * 4);
        let mut expected_2d = Vec::with_capacity(2);
        for s in 0..2 {
            let mut sample = vec![vec![0.0_f64; 4]; 3];
            for j in 0..4 {
                for (i, row) in sample.iter_mut().enumerate() {
                    let v = (s * 100 + i * 10 + j) as f64;
                    raw_2d.push(v);
                    row[j] = v;
                }
            }
            expected_2d.push(sample);
        }
        let buf_2d = MutableBuffer::from_iter(raw_2d);
        let ta_2d =
            TensorArrow::<Float64Type>::new_from_buffer(buf_2d, vec![3, 4], Order::ColumnMajor);
        let cd_2d = ChannelData::ArrayDFloat64(ta_2d);

        // 3D: 2 samples of shape [2, 3, 4] stored column-major per sample:
        // sample s, element [i, j, k] = s * 1000 + i * 100 + j * 10 + k
        let mut raw_3d = Vec::with_capacity(2 * 2 * 3 * 4);
        let mut expected_3d = Vec::with_capacity(2);
        for s in 0..2 {
            let mut sample = vec![vec![vec![0.0_f64; 4]; 3]; 2];
            for k in 0..4 {
                for j in 0..3 {
                    for (i, plane) in sample.iter_mut().enumerate() {
                        let v = (s * 1000 + i * 100 + j * 10 + k) as f64;
                        raw_3d.push(v);
                        plane[j][k] = v;
                    }
                }
            }
            expected_3d.push(sample);
        }
        let buf_3d = MutableBuffer::from_iter(raw_3d);
        let ta_3d =
            TensorArrow::<Float64Type>::new_from_buffer(buf_3d, vec![2, 3, 4], Order::ColumnMajor);
        let cd_3d = ChannelData::ArrayDFloat64(ta_3d);

        // 1D ColumnMajor: 2 samples × 3 elements — must not transpose across samples
        let buf_1d = MutableBuffer::from_iter([1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let ta_1d =
            TensorArrow::<Float64Type>::new_from_buffer(buf_1d, vec![3], Order::ColumnMajor);
        let cd_1d = ChannelData::ArrayDFloat64(ta_1d);

        // Empty 2D ColumnMajor: 0 samples × [3, 4]
        let ta_empty = TensorArrow::<Float64Type>::new_from_buffer(
            MutableBuffer::new(0),
            vec![3, 4],
            Order::ColumnMajor,
        );
        let cd_empty = ChannelData::ArrayDFloat64(ta_empty);

        Python::attach(|py| {
            let obj_2d = cd_2d.into_pyobject(py).unwrap();
            let shape_2d: Vec<usize> = obj_2d.getattr("shape").unwrap().extract().unwrap();
            assert_eq!(shape_2d, vec![2, 3, 4]);
            let vals_2d: Vec<Vec<Vec<f64>>> =
                obj_2d.call_method0("tolist").unwrap().extract().unwrap();
            assert_eq!(vals_2d, expected_2d);

            let obj_3d = cd_3d.into_pyobject(py).unwrap();
            let shape_3d: Vec<usize> = obj_3d.getattr("shape").unwrap().extract().unwrap();
            assert_eq!(shape_3d, vec![2, 2, 3, 4]);
            let vals_3d: Vec<Vec<Vec<Vec<f64>>>> =
                obj_3d.call_method0("tolist").unwrap().extract().unwrap();
            assert_eq!(vals_3d, expected_3d);

            let obj_1d = cd_1d.into_pyobject(py).unwrap();
            let shape_1d: Vec<usize> = obj_1d.getattr("shape").unwrap().extract().unwrap();
            assert_eq!(shape_1d, vec![2, 3]);
            let vals_1d: Vec<Vec<f64>> = obj_1d.call_method0("tolist").unwrap().extract().unwrap();
            assert_eq!(vals_1d, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

            let obj_empty = cd_empty.into_pyobject(py).unwrap();
            let shape_empty: Vec<usize> = obj_empty.getattr("shape").unwrap().extract().unwrap();
            assert_eq!(shape_empty, vec![0, 3, 4]);
        });
    }

    #[test]
    fn numpy_utf8_converts_to_list() {
        init();
        let mut b = LargeStringBuilder::new();
        b.append_value("hello");
        b.append_value("world");
        b.append_value("!");
        let cd = ChannelData::Utf8(b.finish());
        Python::attach(|py| {
            let obj = cd.into_pyobject(py).unwrap();
            let len: usize = obj.call_method0("__len__").unwrap().extract().unwrap();
            assert_eq!(len, 3);
        });
    }
}

// ─── Rust-native polars tests (no PyO3) ─────────────────────────────────────

#[cfg(feature = "polars")]
mod rust_polars_tests {
    use crate::common;
    use mdfr::mdfreader::Mdf;

    fn load_mdf() -> Mdf {
        let path = common::synthetic::ensure_be_scalars();
        let mut mdf = Mdf::new(path).expect("open failed");
        mdf.load_all_channels_data_in_memory().expect("load failed");
        mdf
    }

    #[test]
    fn rust_polars_f64_series_values() {
        let mut mdf = load_mdf();
        let s = mdf
            .get_channel_polars_series("be_f64")
            .expect("be_f64 failed");
        let vals: Vec<f64> = s.f64().unwrap().into_no_null_iter().collect();
        for (got, exp) in vals.iter().zip([1.5, 2.5, 3.5, 4.5]) {
            assert!((got - exp).abs() < 1e-9, "got {got} expected {exp}");
        }
    }

    #[test]
    fn rust_polars_i16_series_values() {
        let mut mdf = load_mdf();
        let s = mdf
            .get_channel_polars_series("be_i16")
            .expect("be_i16 failed");
        assert_eq!(s.dtype().to_string(), "i16");
        let vals: Vec<i16> = s.i16().unwrap().into_no_null_iter().collect();
        assert_eq!(vals, vec![-100i16, 0, 100, 200]);
    }

    #[test]
    fn rust_polars_series_name_matches_channel() {
        let mut mdf = load_mdf();
        let s = mdf
            .get_channel_polars_series("be_f64")
            .expect("be_f64 failed");
        assert_eq!(s.name().as_str(), "be_f64");
    }

    #[test]
    fn rust_polars_dataframe_shape_and_column_values() {
        let mut mdf = load_mdf();
        // "master" is the master channel; the group contains master + be_i16 + be_f64
        let df = mdf
            .get_channel_polars_dataframe(Some("master"))
            .expect("dataframe failed");
        assert_eq!(df.height(), 4);
        assert_eq!(df.width(), 3); // master, be_i16, be_f64
        let col = df.column("be_i16").expect("be_i16 column");
        let vals: Vec<i16> = col.i16().unwrap().into_no_null_iter().collect();
        assert_eq!(vals, vec![-100i16, 0, 100, 200]);
    }

    #[test]
    fn rust_polars_to_dataframes_all_groups() {
        let mut mdf = load_mdf();
        let frames = mdf.get_polars_dataframes().expect("dataframes failed");
        // be_scalars.mf4 has one group with master "master"
        assert_eq!(frames.len(), 1);
        let df = frames
            .get(&Some("master".to_string()))
            .expect("master group");
        assert_eq!(df.height(), 4);
        assert!(df.column("be_f64").is_ok());
        assert!(df.column("be_i16").is_ok());
    }

    #[test]
    fn test_from_polars_df_and_write_roundtrip() {
        use mdfr::mdfinfo::mdfinfo4::CompressionAlgorithm;
        use polars::df;
        let df = df!(
            "time" => [0.0f64, 0.1, 0.2, 0.3],
            "speed" => [10.0f64, 20.0, 30.0, 40.0],
            "gear" => [1i32, 2, 2, 3],
            "label" => ["idle", "run", "run", "fast"],
        )
        .expect("df creation");

        let tmp_file = tempfile::Builder::new().suffix(".mf4").tempfile().unwrap();
        let file_path = tmp_file.path().to_str().unwrap();

        let _written = Mdf::write_polars(
            &df,
            file_path,
            Some("time"),
            CompressionAlgorithm::NoCompression,
        )
        .expect("write_polars failed");

        // Read back the written file
        let mut read_mdf = Mdf::new(file_path).expect("open failed");
        read_mdf
            .load_all_channels_data_in_memory()
            .expect("load failed");

        let names = read_mdf.get_channel_names_set();
        eprintln!("Channels in written file: {:?}", names);
        for name in &names {
            let master = read_mdf.get_channel_master(name);
            let has_data = read_mdf.get_channel_data(name).is_some();
            eprintln!("Channel {name}: master={master:?}, data_is_some={has_data}");
        }

        let read_df = read_mdf
            .get_channel_polars_dataframe(Some("time"))
            .expect("get df failed");
        assert_eq!(read_df.height(), 4);
        assert_eq!(read_df.width(), 4);

        let speed_col = read_df.column("speed").expect("speed col");
        let speed_vals: Vec<f64> = speed_col.f64().unwrap().iter().flatten().collect();
        assert_eq!(speed_vals, vec![10.0, 20.0, 30.0, 40.0]);

        let gear_col = read_df.column("gear").expect("gear col");
        let gear_vals: Vec<i32> = gear_col.i32().unwrap().iter().flatten().collect();
        assert_eq!(gear_vals, vec![1, 2, 2, 3]);

        let label_col = read_df.column("label").expect("label col");
        let label_vals: Vec<Option<&str>> = label_col.str().unwrap().iter().collect();
        assert_eq!(
            label_vals,
            vec![Some("idle"), Some("run"), Some("run"), Some("fast")]
        );
    }
}

#[cfg(feature = "polars")]
mod polars_tests {
    use arrow::array::{Float64Array, Int64Array};
    use mdfr::export::polars::rust_arrow_to_py_series;
    use pyo3::prelude::*;
    use std::sync::Arc;

    fn polars_pyarrow_available() -> bool {
        Python::attach(|py| py.import("pyarrow").is_ok() && py.import("polars").is_ok())
    }

    #[test]
    fn polars_series_name_and_length() {
        Python::initialize();
        if !polars_pyarrow_available() {
            eprintln!("SKIP: polars or pyarrow not installed — run `pip install polars pyarrow`");
            return;
        }
        let arr: Arc<dyn arrow::array::Array> =
            Arc::new(Float64Array::from(vec![1.0f64, 2.0, 3.0, 4.0]));
        let result = rust_arrow_to_py_series(arr, "ch").unwrap();
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
        Python::initialize();
        if !polars_pyarrow_available() {
            eprintln!("SKIP: polars or pyarrow not installed — run `pip install polars pyarrow`");
            return;
        }
        let arr: Arc<dyn arrow::array::Array> = Arc::new(Int64Array::from(vec![1_i64, 2, 3, 4]));
        let result = rust_arrow_to_py_series(arr, "vals").unwrap();
        Python::attach(|py| {
            let obj = result.bind(py);
            let values: Vec<i64> = obj.call_method0("to_list").unwrap().extract().unwrap();
            assert_eq!(values, vec![1_i64, 2, 3, 4]);
        });
    }
}
