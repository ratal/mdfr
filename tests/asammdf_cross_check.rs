mod common;

use anyhow::Result;
use arrow::array::Array;
use mdfr::data_holder::channel_data::{AsValuesSlice, ChannelData};
use mdfr::mdfreader::Mdf;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict};
use std::path::Path;
use std::sync::LazyLock;

static BASE_PATH_MDF4: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}MDF4/MDF4.3/Base_Standard/Examples/",
        common::mdfreader_tests_path()
    )
});

fn is_asammdf_and_reference_available(subpath: &str) -> Option<String> {
    let full_path = format!("{}{subpath}", *BASE_PATH_MDF4);
    if !Path::new(&full_path).exists() {
        eprintln!("Reference file not found on disk: {full_path}");
        return None;
    }
    let asammdf_ok = Python::attach(|py| py.import("asammdf").is_ok());
    if !asammdf_ok {
        eprintln!("asammdf is not installed in Python environment, skipping cross-check");
        return None;
    }
    Some(full_path)
}

#[test]
fn test_asammdf_reference_data_types() -> Result<()> {
    // 1. Integer Types: LE and BE, signed and unsigned
    if let Some(file) =
        is_asammdf_and_reference_available("DataTypes/IntegerTypes/Vector_IntegerTypes.MF4")
    {
        let mut mdf = Mdf::new(&file)?;
        mdf.load_all_channels_data_in_memory()?;

        Python::attach(|py| -> PyResult<()> {
            let locals = PyDict::new(py);
            locals.set_item("path", &file)?;
            py.run(
                pyo3::ffi::c_str!(
                    r#"
from asammdf import MDF
import numpy as np

m = MDF(path)

# Counter_UINT8
sig_u8 = m.get("Counter_UINT8")
u8_vals = list(sig_u8.samples)

# Counter_INT64_BE
sig_i64_be = m.get("Counter_INT64_BE")
i64_be_vals = list(sig_i64_be.samples)

# Counter_INT64_LE
sig_i64_le = m.get("Counter_INT64_LE")
i64_le_vals = list(sig_i64_le.samples)

# Time
t_vals = list(sig_u8.timestamps)

m.close()
"#
                ),
                None,
                Some(&locals),
            )?;

            // Verify Counter_UINT8
            if let Some(ChannelData::UInt8(b)) = mdf.get_channel_data("Counter_UINT8") {
                let mdfr_vals = b.values_slice();
                let py_vals: Vec<u8> = locals.get_item("u8_vals")?.unwrap().extract()?;
                assert_eq!(mdfr_vals.len(), py_vals.len());
                assert_eq!(mdfr_vals, &py_vals[..]);
            } else {
                panic!("Counter_UINT8 channel missing or wrong type in mdfr");
            }

            // Verify Counter_INT64_BE
            if let Some(ChannelData::Int64(b)) = mdf.get_channel_data("Counter_INT64_BE") {
                let mdfr_vals = b.values_slice();
                let py_vals: Vec<i64> = locals.get_item("i64_be_vals")?.unwrap().extract()?;
                assert_eq!(mdfr_vals.len(), py_vals.len());
                assert_eq!(mdfr_vals, &py_vals[..]);
            } else {
                panic!("Counter_INT64_BE channel missing or wrong type in mdfr");
            }

            // Verify Counter_INT64_LE
            if let Some(ChannelData::Int64(b)) = mdf.get_channel_data("Counter_INT64_LE") {
                let mdfr_vals = b.values_slice();
                let py_vals: Vec<i64> = locals.get_item("i64_le_vals")?.unwrap().extract()?;
                assert_eq!(mdfr_vals.len(), py_vals.len());
                assert_eq!(mdfr_vals, &py_vals[..]);
            } else {
                panic!("Counter_INT64_LE channel missing or wrong type in mdfr");
            }

            // Verify Master Timestamps
            if let Some(ChannelData::Float64(b)) = mdf.get_channel_data("t") {
                let mdfr_t = b.values_slice();
                let py_t: Vec<f64> = locals.get_item("t_vals")?.unwrap().extract()?;
                assert_eq!(mdfr_t.len(), py_t.len());
                for (a, b) in mdfr_t.iter().zip(py_t.iter()) {
                    assert!((a - b).abs() < 1e-5);
                }
            }

            Ok(())
        })?;
    }

    // 2. Real / Float Types: Float32 and Float64, LE and BE
    if let Some(file) =
        is_asammdf_and_reference_available("DataTypes/RealTypes/Vector_RealTypes.MF4")
    {
        let mut mdf = Mdf::new(&file)?;
        mdf.load_all_channels_data_in_memory()?;

        Python::attach(|py| -> PyResult<()> {
            let locals = PyDict::new(py);
            locals.set_item("path", &file)?;
            py.run(
                pyo3::ffi::c_str!(
                    r#"
from asammdf import MDF
m = MDF(path)

f32_le = [float(x) for x in m.get("Counter_FLOAT_LE").samples]
f32_be = [float(x) for x in m.get("Counter_FLOAT_BE").samples]
f64_le = [float(x) for x in m.get("Counter_DOUBLE_LE").samples]
f64_be = [float(x) for x in m.get("Counter_DOUBLE_BE").samples]

m.close()
"#
                ),
                None,
                Some(&locals),
            )?;

            // Float32 LE
            if let Some(ChannelData::Float32(b)) = mdf.get_channel_data("Counter_FLOAT_LE") {
                let mdfr_vals = b.values_slice();
                let py_vals: Vec<f64> = locals.get_item("f32_le")?.unwrap().extract()?;
                assert_eq!(mdfr_vals.len(), py_vals.len());
                for (&m, &p) in mdfr_vals.iter().zip(py_vals.iter()) {
                    assert!((m as f64 - p).abs() < 1e-4);
                }
            }

            // Float32 BE
            if let Some(ChannelData::Float32(b)) = mdf.get_channel_data("Counter_FLOAT_BE") {
                let mdfr_vals = b.values_slice();
                let py_vals: Vec<f64> = locals.get_item("f32_be")?.unwrap().extract()?;
                assert_eq!(mdfr_vals.len(), py_vals.len());
                for (&m, &p) in mdfr_vals.iter().zip(py_vals.iter()) {
                    assert!((m as f64 - p).abs() < 1e-4);
                }
            }

            // Float64 LE
            if let Some(ChannelData::Float64(b)) = mdf.get_channel_data("Counter_DOUBLE_LE") {
                let mdfr_vals = b.values_slice();
                let py_vals: Vec<f64> = locals.get_item("f64_le")?.unwrap().extract()?;
                assert_eq!(mdfr_vals.len(), py_vals.len());
                for (&m, &p) in mdfr_vals.iter().zip(py_vals.iter()) {
                    assert!((m - p).abs() < 1e-5);
                }
            }

            // Float64 BE
            if let Some(ChannelData::Float64(b)) = mdf.get_channel_data("Counter_DOUBLE_BE") {
                let mdfr_vals = b.values_slice();
                let py_vals: Vec<f64> = locals.get_item("f64_be")?.unwrap().extract()?;
                assert_eq!(mdfr_vals.len(), py_vals.len());
                for (&m, &p) in mdfr_vals.iter().zip(py_vals.iter()) {
                    assert!((m - p).abs() < 1e-5);
                }
            }

            Ok(())
        })?;
    }

    // 3. String Types: Fixed length UTF-8
    if let Some(file) =
        is_asammdf_and_reference_available("DataTypes/StringTypes/Vector_FixedLengthStringUTF8.mf4")
    {
        let mut mdf = Mdf::new(&file)?;
        mdf.load_all_channels_data_in_memory()?;

        Python::attach(|py| -> PyResult<()> {
            let locals = PyDict::new(py);
            locals.set_item("path", &file)?;
            py.run(
                pyo3::ffi::c_str!(
                    r#"
from asammdf import MDF
m = MDF(path)
sig = m.get("Data channel")
str_vals = [s.decode('utf-8', errors='replace').rstrip('\x00') if isinstance(s, bytes) else str(s).rstrip('\x00') for s in sig.samples]
m.close()
"#
                ),
                None,
                Some(&locals),
            )?;

            if let Some(ChannelData::Utf8(b)) = mdf.get_channel_data("Data channel") {
                let arr = b;
                let py_vals: Vec<String> = locals.get_item("str_vals")?.unwrap().extract()?;
                assert_eq!(arr.len(), py_vals.len());
                for (i, py_val) in py_vals.iter().enumerate().take(arr.len()) {
                    assert_eq!(arr.value(i), py_val);
                }
            } else {
                panic!("Data channel string missing in mdfr");
            }
            Ok(())
        })?;
    }

    // 4. Byte Array Types: Fixed length byte array
    if let Some(file) =
        is_asammdf_and_reference_available("DataTypes/ByteArray/Vector_ByteArrayFixedLength.mf4")
    {
        let mut mdf = Mdf::new(&file)?;
        mdf.load_all_channels_data_in_memory()?;

        Python::attach(|py| -> PyResult<()> {
            let locals = PyDict::new(py);
            locals.set_item("path", &file)?;
            py.run(
                pyo3::ffi::c_str!(
                    r#"
from asammdf import MDF
m = MDF(path)
sig = m.get("Data channel")
# Convert numpy array of bytes to list of byte lists
raw_bytes = [list(row) for row in sig.samples]
m.close()
"#
                ),
                None,
                Some(&locals),
            )?;

            if let Some(ChannelData::FixedSizeByteArray(b)) = mdf.get_channel_data("Data channel") {
                let arr = b;
                let py_vals: Vec<Vec<u8>> = locals.get_item("raw_bytes")?.unwrap().extract()?;
                assert_eq!(arr.len(), py_vals.len());
                for (i, py_val) in py_vals.iter().enumerate().take(arr.len()) {
                    assert_eq!(arr.value(i), &py_val[..]);
                }
            } else {
                panic!("Data channel byte array missing in mdfr");
            }
            Ok(())
        })?;
    }

    Ok(())
}

#[test]
fn test_asammdf_reference_channel_types() -> Result<()> {
    // 1. Master Channels: Distance, Angle, Time
    if let Some(file) = is_asammdf_and_reference_available(
        "ChannelTypes/MasterChannels/Vector_DifferentMasterChannels.mf4",
    ) {
        let mut mdf = Mdf::new(&file)?;
        mdf.load_all_channels_data_in_memory()?;

        Python::attach(|py| -> PyResult<()> {
            let locals = PyDict::new(py);
            locals.set_item("path", &file)?;
            py.run(
                pyo3::ffi::c_str!(
                    r#"
from asammdf import MDF
m = MDF(path)

data_sig = m.get("Data channel")
data_vals = [float(x) for x in data_sig.samples]
dist_vals = [float(x) for x in m.get("Master channel distance").samples]
angle_vals = [float(x) for x in m.get("Master channel angle").samples]
time_vals = [float(x) for x in m.get("Master channel time").samples]

m.close()
"#
                ),
                None,
                Some(&locals),
            )?;

            // Data channel
            if let Some(ChannelData::Float64(b)) = mdf.get_channel_data("Data channel") {
                let mdfr_vals = b.values_slice();
                let py_vals: Vec<f64> = locals.get_item("data_vals")?.unwrap().extract()?;
                assert_eq!(mdfr_vals.len(), py_vals.len());
                for (&m, &p) in mdfr_vals.iter().zip(py_vals.iter()) {
                    assert!((m - p).abs() < 1e-5);
                }
            }

            // Distance master
            if let Some(ChannelData::Float64(b)) = mdf.get_channel_data("Master channel distance") {
                let mdfr_vals = b.values_slice();
                let py_vals: Vec<f64> = locals.get_item("dist_vals")?.unwrap().extract()?;
                assert_eq!(mdfr_vals.len(), py_vals.len());
                for (&m, &p) in mdfr_vals.iter().zip(py_vals.iter()) {
                    assert!((m - p).abs() < 1e-5);
                }
            }

            // Angle master
            if let Some(ChannelData::Float64(b)) = mdf.get_channel_data("Master channel angle") {
                let mdfr_vals = b.values_slice();
                let py_vals: Vec<f64> = locals.get_item("angle_vals")?.unwrap().extract()?;
                assert_eq!(mdfr_vals.len(), py_vals.len());
                for (&m, &p) in mdfr_vals.iter().zip(py_vals.iter()) {
                    assert!((m - p).abs() < 1e-5);
                }
            }

            // Time master
            if let Some(ChannelData::Float64(b)) = mdf.get_channel_data("Master channel time") {
                let mdfr_vals = b.values_slice();
                let py_vals: Vec<f64> = locals.get_item("time_vals")?.unwrap().extract()?;
                assert_eq!(mdfr_vals.len(), py_vals.len());
                for (&m, &p) in mdfr_vals.iter().zip(py_vals.iter()) {
                    assert!((m - p).abs() < 1e-5);
                }
            }

            Ok(())
        })?;
    }

    // 2. VLSD (Variable Length Signal Data): UTF-8 Strings
    if let Some(file) =
        is_asammdf_and_reference_available("ChannelTypes/VLSD/Vector_VLSD_String_UTF8.mf4")
    {
        let mut mdf = Mdf::new(&file)?;
        mdf.load_all_channels_data_in_memory()?;

        Python::attach(|py| -> PyResult<()> {
            let locals = PyDict::new(py);
            locals.set_item("path", &file)?;
            py.run(
                pyo3::ffi::c_str!(
                    r#"
from asammdf import MDF
m = MDF(path)
sig = m.get("Data channel")
str_vals = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in sig.samples]
m.close()
"#
                ),
                None,
                Some(&locals),
            )?;

            if let Some(ChannelData::Utf8(b)) = mdf.get_channel_data("Data channel") {
                let arr = b;
                let py_vals: Vec<String> = locals.get_item("str_vals")?.unwrap().extract()?;
                assert_eq!(arr.len(), py_vals.len());
                for (i, py_val) in py_vals.iter().enumerate().take(arr.len()) {
                    assert_eq!(arr.value(i), py_val);
                }
            } else {
                panic!("VLSD Data channel missing or wrong type in mdfr");
            }

            Ok(())
        })?;
    }

    Ok(())
}

#[test]
fn test_asammdf_reference_conversions() -> Result<()> {
    // 1. Linear Conversion (p1 = -4.8, p2 = -3.2 -> phys = -3.2 * X - 4.8)
    if let Some(file) = is_asammdf_and_reference_available(
        "Conversion/LinearConversion/Vector_LinearConversion.mf4",
    ) {
        let mut mdf = Mdf::new(&file)?;
        mdf.load_all_channels_data_in_memory()?;
        mdf.convert_all_channels()?;

        Python::attach(|py| -> PyResult<()> {
            let locals = PyDict::new(py);
            locals.set_item("path", &file)?;
            py.run(
                pyo3::ffi::c_str!(
                    r#"
from asammdf import MDF
m = MDF(path)
sig = m.get("Data channel", raw=False)
conv_vals = [float(x) for x in sig.samples]
m.close()
"#
                ),
                None,
                Some(&locals),
            )?;

            if let Some(ChannelData::Float64(b)) = mdf.get_channel_data("Data channel") {
                let mdfr_vals = b.values_slice();
                let py_vals: Vec<f64> = locals.get_item("conv_vals")?.unwrap().extract()?;
                assert_eq!(mdfr_vals.len(), py_vals.len());
                for (&m, &p) in mdfr_vals.iter().zip(py_vals.iter()) {
                    assert!(
                        (m - p).abs() < 1e-5,
                        "Linear mismatch: mdfr={m}, asammdf={p}"
                    );
                }
            } else {
                panic!("Linear Data channel missing in mdfr");
            }

            Ok(())
        })?;
    }

    // 2. Rational Conversion ((1*X^2 + 2*X + 3) / (4*X^2 + 5*X + 6))
    if let Some(file) = is_asammdf_and_reference_available(
        "Conversion/RationalConversion/Vector_RationalConversionIntParams.mf4",
    ) {
        let mut mdf = Mdf::new(&file)?;
        mdf.load_all_channels_data_in_memory()?;
        mdf.convert_all_channels()?;

        Python::attach(|py| -> PyResult<()> {
            let locals = PyDict::new(py);
            locals.set_item("path", &file)?;
            py.run(
                pyo3::ffi::c_str!(
                    r#"
from asammdf import MDF
m = MDF(path)
sig = m.get("Data channel", raw=False)
conv_vals = [float(x) for x in sig.samples]
m.close()
"#
                ),
                None,
                Some(&locals),
            )?;

            if let Some(ChannelData::Float64(b)) = mdf.get_channel_data("Data channel") {
                let mdfr_vals = b.values_slice();
                let py_vals: Vec<f64> = locals.get_item("conv_vals")?.unwrap().extract()?;
                assert_eq!(mdfr_vals.len(), py_vals.len());
                for (&m, &p) in mdfr_vals.iter().zip(py_vals.iter()) {
                    assert!(
                        (m - p).abs() < 1e-5,
                        "Rational mismatch: mdfr={m}, asammdf={p}"
                    );
                }
            } else {
                panic!("Rational Data channel missing in mdfr");
            }

            Ok(())
        })?;
    }

    // 3. Value to Text Lookup Conversion
    if let Some(file) = is_asammdf_and_reference_available(
        "Conversion/LookUpConversion/Vector_Value2TextConversion.mf4",
    ) {
        let mut mdf = Mdf::new(&file)?;
        mdf.load_all_channels_data_in_memory()?;
        mdf.convert_all_channels()?;

        Python::attach(|py| -> PyResult<()> {
            let locals = PyDict::new(py);
            locals.set_item("path", &file)?;
            py.run(
                pyo3::ffi::c_str!(
                    r#"
from asammdf import MDF
m = MDF(path)
sig = m.get("Data channel", raw=False)
text_vals = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in sig.samples]
m.close()
"#
                ),
                None,
                Some(&locals),
            )?;

            if let Some(ChannelData::Utf8(b)) = mdf.get_channel_data("Data channel") {
                let arr = b;
                let py_vals: Vec<String> = locals.get_item("text_vals")?.unwrap().extract()?;
                assert_eq!(arr.len(), py_vals.len());
                for (i, py_val) in py_vals.iter().enumerate().take(arr.len()) {
                    assert_eq!(
                        arr.value(i),
                        py_val,
                        "Value2Text mismatch at {i}: mdfr='{}', asammdf='{py_val}'",
                        arr.value(i)
                    );
                }
            } else {
                panic!("Value2Text Data channel missing in mdfr");
            }

            Ok(())
        })?;
    }

    // 4. Algebraic Conversion (Formula: X * X + 1)
    if let Some(file) = is_asammdf_and_reference_available(
        "Conversion/TextConversion/Vector_AlgebraicConversionQuadratic.mf4",
    ) {
        let mut mdf = Mdf::new(&file)?;
        mdf.load_all_channels_data_in_memory()?;
        mdf.convert_all_channels()?;

        Python::attach(|py| -> PyResult<()> {
            let locals = PyDict::new(py);
            locals.set_item("path", &file)?;
            py.run(
                pyo3::ffi::c_str!(
                    r#"
from asammdf import MDF
m = MDF(path)
sig = m.get("Data channel", raw=False)
conv_vals = [float(x) for x in sig.samples]
m.close()
"#
                ),
                None,
                Some(&locals),
            )?;

            if let Some(ChannelData::Float64(b)) = mdf.get_channel_data("Data channel") {
                let mdfr_vals = b.values_slice();
                let py_vals: Vec<f64> = locals.get_item("conv_vals")?.unwrap().extract()?;
                assert_eq!(mdfr_vals.len(), py_vals.len());
                for (&m, &p) in mdfr_vals.iter().zip(py_vals.iter()) {
                    assert!(
                        (m - p).abs() < 1e-5,
                        "Algebraic mismatch: mdfr={m}, asammdf={p}"
                    );
                }
            } else {
                panic!("Algebraic Data channel missing in mdfr");
            }

            Ok(())
        })?;
    }

    Ok(())
}
