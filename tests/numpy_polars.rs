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

// ─── Rust-native polars tests (no PyO3) ─────────────────────────────────────

#[cfg(feature = "polars")]
mod rust_polars_tests {
    use anyhow::Result;
    use mdfr::mdfreader::Mdf;
    use std::sync::LazyLock;

    // ── byte-push helpers (same as be_complex_array.rs) ──
    fn pu8(b: &mut Vec<u8>, v: u8) {
        b.push(v);
    }
    fn pu16(b: &mut Vec<u8>, v: u16) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    fn pu32(b: &mut Vec<u8>, v: u32) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    fn pu64(b: &mut Vec<u8>, v: u64) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    fn pi64(b: &mut Vec<u8>, v: i64) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    fn pf64(b: &mut Vec<u8>, v: f64) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    fn pi16_be(b: &mut Vec<u8>, v: i16) {
        b.extend_from_slice(&v.to_be_bytes());
    }
    fn pf64_be(b: &mut Vec<u8>, v: f64) {
        b.extend_from_slice(&v.to_be_bytes());
    }
    fn zeros(b: &mut Vec<u8>, n: usize) {
        b.extend(std::iter::repeat_n(0u8, n));
    }

    fn id_block(b: &mut Vec<u8>) {
        b.extend_from_slice(b"MDF     ");
        b.extend_from_slice(b"4.30    ");
        b.extend_from_slice(b"mdfr    ");
        pu16(b, 0);
        pu16(b, 0);
        pu16(b, 430);
        pu16(b, 0);
        zeros(b, 2);
        zeros(b, 26);
        pu16(b, 0);
        pu16(b, 0);
    }
    fn hd4(b: &mut Vec<u8>, dg: i64, fh: i64) {
        b.extend_from_slice(b"##HD");
        zeros(b, 4);
        pu64(b, 104);
        pu64(b, 6);
        pi64(b, dg);
        pi64(b, fh);
        pi64(b, 0);
        pi64(b, 0);
        pi64(b, 0);
        pi64(b, 0);
        pu64(b, 0);
        b.extend_from_slice(&0i16.to_le_bytes());
        b.extend_from_slice(&0i16.to_le_bytes());
        pu8(b, 0);
        pu8(b, 0);
        pu8(b, 0);
        pu8(b, 0);
        pf64(b, 0.0);
        pf64(b, 0.0);
    }
    fn fh(b: &mut Vec<u8>) {
        b.extend_from_slice(b"##FH");
        zeros(b, 4);
        pu64(b, 56);
        pu64(b, 2);
        pi64(b, 0);
        pi64(b, 0);
        pu64(b, 0);
        b.extend_from_slice(&0i16.to_le_bytes());
        b.extend_from_slice(&0i16.to_le_bytes());
        pu8(b, 0);
        zeros(b, 3);
    }
    fn dg4(b: &mut Vec<u8>, cg: i64, data: i64) {
        b.extend_from_slice(b"##DG");
        zeros(b, 4);
        pu64(b, 64);
        pu64(b, 4);
        pi64(b, 0);
        pi64(b, cg);
        pi64(b, data);
        pi64(b, 0);
        pu8(b, 0);
        zeros(b, 7);
    }
    fn cg4(b: &mut Vec<u8>, cn: i64, cycles: u64, data_bytes: u32) {
        b.extend_from_slice(b"##CG");
        zeros(b, 4);
        pu64(b, 104);
        pu64(b, 6);
        pi64(b, 0);
        pi64(b, cn);
        pi64(b, 0);
        pi64(b, 0);
        pi64(b, 0);
        pi64(b, 0);
        pu64(b, 0);
        pu64(b, cycles);
        pu16(b, 0);
        pu16(b, 0);
        zeros(b, 4);
        pu32(b, data_bytes);
        pu32(b, 0);
    }
    fn cn4(b: &mut Vec<u8>, desc: (u8, u8, u8), span: (u32, u32), refs: (i64, i64, i64)) {
        let (cn_type, sync, dtype) = desc;
        let (byte_off, bits) = span;
        let (next, tx, cc) = refs;
        b.extend_from_slice(b"##CN");
        zeros(b, 4);
        pu64(b, 160);
        pu64(b, 8);
        pi64(b, next);
        pi64(b, 0);
        pi64(b, tx);
        pi64(b, 0);
        pi64(b, cc);
        pi64(b, 0);
        pi64(b, 0);
        pi64(b, 0);
        pu8(b, cn_type);
        pu8(b, sync);
        pu8(b, dtype);
        pu8(b, 0);
        pu32(b, byte_off);
        pu32(b, bits);
        pu32(b, 0);
        pu32(b, 0);
        pu8(b, 0xff);
        pu8(b, 0);
        pu16(b, 0);
        pf64(b, 0.0);
        pf64(b, 0.0);
        pf64(b, 0.0);
        pf64(b, 0.0);
        pf64(b, 0.0);
        pf64(b, 0.0);
    }
    fn tx(b: &mut Vec<u8>, text: &str) {
        let t = text.as_bytes();
        let len = 24u64 + t.len() as u64 + 1;
        b.extend_from_slice(b"##TX");
        zeros(b, 4);
        pu64(b, len);
        pu64(b, 0);
        b.extend_from_slice(t);
        b.push(0);
    }
    fn dt(b: &mut Vec<u8>, records: &[u8]) {
        let len = 24u64 + records.len() as u64;
        b.extend_from_slice(b"##DT");
        zeros(b, 4);
        pu64(b, len);
        pu64(b, 0);
        b.extend_from_slice(records);
    }

    fn create_be_scalars() -> Result<()> {
        const PATH: &str = "test_files/synthetic/be_scalars.mf4";
        if std::path::Path::new(PATH).exists() {
            return Ok(());
        }
        std::fs::create_dir_all("test_files/synthetic")?;
        let mut b: Vec<u8> = Vec::with_capacity(1061);
        id_block(&mut b);
        hd4(&mut b, 224, 168);
        fh(&mut b);
        dg4(&mut b, 288, 965);
        cg4(&mut b, 392, 4, 18);
        cn4(&mut b, (2, 1, 4), (0, 64), (552, 872, 0));
        cn4(&mut b, (0, 0, 3), (8, 16), (712, 903, 0));
        cn4(&mut b, (0, 0, 5), (10, 64), (0, 934, 0));
        tx(&mut b, "master");
        tx(&mut b, "be_i16");
        tx(&mut b, "be_f64");
        let raw_i16_be: [i16; 4] = [-100, 0, 100, 200];
        let raw_f64_be: [f64; 4] = [1.5, 2.5, 3.5, 4.5];
        let mut recs: Vec<u8> = Vec::with_capacity(72);
        for i in 0..4 {
            pf64(&mut recs, i as f64);
            pi16_be(&mut recs, raw_i16_be[i]);
            pf64_be(&mut recs, raw_f64_be[i]);
        }
        dt(&mut b, &recs);
        let tmp = format!("{}.tmp.{}", PATH, std::process::id());
        std::fs::write(&tmp, &b)?;
        if std::fs::rename(&tmp, PATH).is_err() {
            std::fs::remove_file(&tmp).ok();
        }
        Ok(())
    }

    const BE_SCALARS_PATH: &str = "test_files/synthetic/be_scalars.mf4";
    static FIXTURE_BE: LazyLock<()> = LazyLock::new(|| {
        create_be_scalars().expect("failed to create be_scalars fixture");
    });

    fn load_mdf() -> Mdf {
        LazyLock::force(&FIXTURE_BE);
        let mut mdf = Mdf::new(BE_SCALARS_PATH).expect("open failed");
        mdf.load_all_channels_data_in_memory().expect("load failed");
        mdf
    }

    #[test]
    fn rust_polars_f64_series_values() {
        let mdf = load_mdf();
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
        let mdf = load_mdf();
        let s = mdf
            .get_channel_polars_series("be_i16")
            .expect("be_i16 failed");
        assert_eq!(s.dtype().to_string(), "i16");
        let vals: Vec<i16> = s.i16().unwrap().into_no_null_iter().collect();
        assert_eq!(vals, vec![-100i16, 0, 100, 200]);
    }

    #[test]
    fn rust_polars_series_name_matches_channel() {
        let mdf = load_mdf();
        let s = mdf
            .get_channel_polars_series("be_f64")
            .expect("be_f64 failed");
        assert_eq!(s.name().as_str(), "be_f64");
    }

    #[test]
    fn rust_polars_dataframe_shape_and_column_values() {
        let mdf = load_mdf();
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
        let mdf = load_mdf();
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
}

#[cfg(feature = "polars")]
mod polars_tests {
    use arrow::array::{Float64Array, Int64Array};
    use mdfr::export::polars::rust_arrow_to_py_series;
    use pyo3::prelude::*;
    use std::sync::Arc;

    fn pyarrow_available() -> bool {
        Python::attach(|py| py.import("pyarrow").is_ok())
    }

    #[test]
    fn polars_series_name_and_length() {
        Python::initialize();
        if !pyarrow_available() {
            eprintln!("SKIP: pyarrow not installed — run `pip install pyarrow`");
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
        if !pyarrow_available() {
            eprintln!("SKIP: pyarrow not installed — run `pip install pyarrow`");
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
