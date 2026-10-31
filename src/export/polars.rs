//! this module provides methods to get directly from arrow into polars (rust or python)
use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::Array;
use pyo3::{Py, PyAny, PyResult, types::PyList};

use crate::data_holder::channel_data::ChannelData;
use crate::export::numpy::to_py_array;
use crate::mdfreader::Mdf;

use polars::prelude::{
    Column, DataFrame, DataType, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, IntoSeries, ListBuilderTrait, ListPrimitiveChunkedBuilder, NamedFrom, PolarsError,
    PolarsResult, Series, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};

/// Build a `List<T>` Series from a flat TensorArrow, one row per sample.
macro_rules! tensor_to_list {
    ($name:expr, $ta:expr, $polars_type:ty, $dtype:expr) => {{
        let vals = $ta.values_slice();
        let n = $ta.len();
        let row_len = vals.len().checked_div(n).unwrap_or(0);
        let mut bldr =
            ListPrimitiveChunkedBuilder::<$polars_type>::new($name.into(), n, vals.len(), $dtype);
        for row in vals.chunks_exact(row_len) {
            bldr.append_slice(row);
        }
        bldr.finish().into_series()
    }};
}

/// Build a `List<T>` Series from a ComplexArrow, one `[re, im]` pair per row.
macro_rules! complex_to_list {
    ($name:expr, $cx:expr, $polars_type:ty, $dtype:expr) => {{
        let vals = $cx.values_slice();
        let n = $cx.len();
        let mut bldr =
            ListPrimitiveChunkedBuilder::<$polars_type>::new($name.into(), n, vals.len(), $dtype);
        for pair in vals.chunks_exact(2) {
            bldr.append_slice(pair);
        }
        bldr.finish().into_series()
    }};
}

/// Converts an iterator of `Option<&[u8]>` items into a polars Binary `Series`.
fn bytes_iter_to_series<'a>(name: &str, iter: impl Iterator<Item = Option<&'a [u8]>>) -> Series {
    let v: Vec<Vec<u8>> = iter.map(|x| x.unwrap_or(&[]).to_vec()).collect();
    Series::new(name.into(), v)
}

/// Converts a [`ChannelData`] value into a Polars [`Series`].
///
/// **Supported types**
/// - Scalar numerics: Int8/16/32/64, UInt8/16/32/64, Float32/64
/// - Utf8 strings (LargeString)
/// - Byte arrays: FixedSizeByteArray, VariableSizeByteArray → Binary series
/// - Complex32/64 → `List<Float32/Float64>` (each row is `[re, im]`)
/// - Tensor arrays (ArrayD*) → `List<T>` (each row is a flat slice of the sample)
///
/// # Errors
///
/// Returns [`PolarsError::InvalidOperation`] for the `Union` variant.
pub fn channel_data_to_series(name: &str, data: &ChannelData) -> PolarsResult<Series> {
    use ChannelData::*;
    Ok(match data {
        // ── scalar numerics ──────────────────────────────────────────────────
        Int8(b) => Series::new(name.into(), b.values().as_ref()),
        UInt8(b) => Series::new(name.into(), b.values().as_ref()),
        Int16(b) => Series::new(name.into(), b.values().as_ref()),
        UInt16(b) => Series::new(name.into(), b.values().as_ref()),
        Int32(b) => Series::new(name.into(), b.values().as_ref()),
        UInt32(b) => Series::new(name.into(), b.values().as_ref()),
        Float32(b) => Series::new(name.into(), b.values().as_ref()),
        Int64(b) => Series::new(name.into(), b.values().as_ref()),
        UInt64(b) => Series::new(name.into(), b.values().as_ref()),
        Float64(b) => Series::new(name.into(), b.values().as_ref()),

        // ── strings ──────────────────────────────────────────────────────────
        Utf8(b) => {
            let v: Vec<Option<&str>> = b.iter().collect();
            Series::new(name.into(), v)
        }

        // ── byte arrays ──────────────────────────────────────────────────────
        FixedSizeByteArray(b) => {
            bytes_iter_to_series(name, b.iter())
        }
        VariableSizeByteArray(b) => {
            bytes_iter_to_series(name, b.iter())
        }

        // ── complex ([re, im] per row) ────────────────────────────────────────
        Complex32(cx) => complex_to_list!(name, cx, Float32Type, DataType::Float32),
        Complex64(cx) => complex_to_list!(name, cx, Float64Type, DataType::Float64),

        // ── tensor arrays (flat sample per row) ───────────────────────────────
        ArrayDInt8(ta) => tensor_to_list!(name, ta, Int8Type, DataType::Int8),
        ArrayDUInt8(ta) => tensor_to_list!(name, ta, UInt8Type, DataType::UInt8),
        ArrayDInt16(ta) => tensor_to_list!(name, ta, Int16Type, DataType::Int16),
        ArrayDUInt16(ta) => tensor_to_list!(name, ta, UInt16Type, DataType::UInt16),
        ArrayDInt32(ta) => tensor_to_list!(name, ta, Int32Type, DataType::Int32),
        ArrayDUInt32(ta) => tensor_to_list!(name, ta, UInt32Type, DataType::UInt32),
        ArrayDFloat32(ta) => tensor_to_list!(name, ta, Float32Type, DataType::Float32),
        ArrayDInt64(ta) => tensor_to_list!(name, ta, Int64Type, DataType::Int64),
        ArrayDUInt64(ta) => tensor_to_list!(name, ta, UInt64Type, DataType::UInt64),
        ArrayDFloat64(ta) => tensor_to_list!(name, ta, Float64Type, DataType::Float64),

        // ── union (unsupported) ───────────────────────────────────────────────
        Union(_) => {
            return Err(PolarsError::InvalidOperation(
                format!("channel_data_to_series: Union type not supported for channel '{name}'")
                    .into(),
            ));
        }
    })
}

/// Converts all channels sharing `master_channel_name` into a Polars [`DataFrame`].
///
/// Pass `None` for channels that have no master (orphan channels). All channels
/// in the master's group (including the master itself) become columns. Channels
/// that cannot be converted to a Series are silently skipped.
///
/// All channels must already be loaded in memory.
///
/// # Errors
///
/// Returns [`PolarsError::ColumnNotFound`] if `master_channel_name` is not a
/// known master in the file, or a polars error if DataFrame construction fails.
pub fn mdf_master_to_dataframe(
    mdf: &mut Mdf,
    master_channel_name: Option<&str>,
) -> PolarsResult<DataFrame> {
    let groups = mdf.get_master_channel_names_set();
    let key = master_channel_name.map(ToString::to_string);
    let channels = groups.get(&key).ok_or_else(|| {
        PolarsError::ColumnNotFound(
            format!(
                "master channel '{}' not found",
                master_channel_name.unwrap_or("<none>")
            )
            .into(),
        )
    })?;
    let cols: Vec<Column> = channels
        .iter()
        .filter_map(|ch| {
            mdf.get_channel_data(ch)
                .and_then(|d| channel_data_to_series(ch, d).ok())
                .map(Column::from)
        })
        .collect();
    DataFrame::new_infer_height(cols)
}

/// Converts every channel group in the MDF into its own Polars [`DataFrame`].
///
/// Returns a map from master channel name (`None` = no master) to the
/// corresponding DataFrame. Each DataFrame contains all convertible channels
/// of that group as columns.
///
/// All channels must already be loaded in memory.
///
/// # Errors
///
/// Returns a polars error if any DataFrame construction fails.
pub fn mdf_to_dataframes(mdf: &mut Mdf) -> PolarsResult<HashMap<Option<String>, DataFrame>> {
    mdf.get_master_channel_names_set()
        .into_iter()
        .map(|(master, channels)| {
            let cols: Vec<Column> = channels
                .iter()
                .filter_map(|ch| {
                    mdf.get_channel_data(ch)
                        .and_then(|d| channel_data_to_series(ch, d).ok())
                        .map(Column::from)
                })
                .collect();
            Ok((master, DataFrame::new_infer_height(cols)?))
        })
        .collect()
}

/// Converts a Polars [`Series`] into an Apache Arrow [`Array`].
pub fn series_to_arrow_array(series: &Series) -> anyhow::Result<Arc<dyn Array>> {
    use arrow::array::*;
    use arrow::datatypes::*;
    use polars::prelude::DataType as PDT;

    let s = series.rechunk();
    match s.dtype() {
        PDT::Float64 => {
            let ca = s.f64().map_err(|e| anyhow::anyhow!("{e}"))?;
            let arr: PrimitiveArray<Float64Type> = ca.iter().collect();
            Ok(Arc::new(arr))
        }
        PDT::Float32 => {
            let ca = s.f32().map_err(|e| anyhow::anyhow!("{e}"))?;
            let arr: PrimitiveArray<Float32Type> = ca.iter().collect();
            Ok(Arc::new(arr))
        }
        PDT::Int64 => {
            let ca = s.i64().map_err(|e| anyhow::anyhow!("{e}"))?;
            let arr: PrimitiveArray<Int64Type> = ca.iter().collect();
            Ok(Arc::new(arr))
        }
        PDT::Int32 => {
            let ca = s.i32().map_err(|e| anyhow::anyhow!("{e}"))?;
            let arr: PrimitiveArray<Int32Type> = ca.iter().collect();
            Ok(Arc::new(arr))
        }
        PDT::Int16 => {
            let ca = s.i16().map_err(|e| anyhow::anyhow!("{e}"))?;
            let arr: PrimitiveArray<Int16Type> = ca.iter().collect();
            Ok(Arc::new(arr))
        }
        PDT::Int8 => {
            let ca = s.i8().map_err(|e| anyhow::anyhow!("{e}"))?;
            let arr: PrimitiveArray<Int8Type> = ca.iter().collect();
            Ok(Arc::new(arr))
        }
        PDT::UInt64 => {
            let ca = s.u64().map_err(|e| anyhow::anyhow!("{e}"))?;
            let arr: PrimitiveArray<UInt64Type> = ca.iter().collect();
            Ok(Arc::new(arr))
        }
        PDT::UInt32 => {
            let ca = s.u32().map_err(|e| anyhow::anyhow!("{e}"))?;
            let arr: PrimitiveArray<UInt32Type> = ca.iter().collect();
            Ok(Arc::new(arr))
        }
        PDT::UInt16 => {
            let ca = s.u16().map_err(|e| anyhow::anyhow!("{e}"))?;
            let arr: PrimitiveArray<UInt16Type> = ca.iter().collect();
            Ok(Arc::new(arr))
        }
        PDT::UInt8 => {
            let ca = s.u8().map_err(|e| anyhow::anyhow!("{e}"))?;
            let arr: PrimitiveArray<UInt8Type> = ca.iter().collect();
            Ok(Arc::new(arr))
        }
        PDT::Boolean => {
            let ca = s.bool().map_err(|e| anyhow::anyhow!("{e}"))?;
            let arr: PrimitiveArray<UInt8Type> = ca
                .iter()
                .map(|opt| opt.map(|b| if b { 1u8 } else { 0u8 }))
                .collect();
            Ok(Arc::new(arr))
        }
        PDT::String => {
            let ca = s.str().map_err(|e| anyhow::anyhow!("{e}"))?;
            let arr: LargeStringArray = ca.iter().collect();
            Ok(Arc::new(arr))
        }
        PDT::Binary => {
            let ca = s.binary().map_err(|e| anyhow::anyhow!("{e}"))?;
            let arr: LargeBinaryArray = ca.iter().collect();
            Ok(Arc::new(arr))
        }
        PDT::Datetime(_time_unit, _tz) => {
            let ca = s.datetime().map_err(|e| anyhow::anyhow!("{e}"))?;
            let arr: PrimitiveArray<Int64Type> = ca.physical().iter().collect();
            Ok(Arc::new(arr))
        }
        PDT::Date => {
            let ca = s.date().map_err(|e| anyhow::anyhow!("{e}"))?;
            let arr: PrimitiveArray<Int32Type> = ca.physical().iter().collect();
            Ok(Arc::new(arr))
        }
        other => anyhow::bail!("Unsupported Polars data type: {other:?}"),
    }
}

/// Creates an in-memory MDF structure from a Polars [`DataFrame`].
pub fn from_polars_df(df: &DataFrame, time_col: Option<&str>) -> anyhow::Result<Mdf> {
    let height = df.height();
    if height == 0 {
        anyhow::bail!("Cannot create MDF from an empty DataFrame");
    }

    // Determine master time column name
    let master_name = if let Some(tc) = time_col {
        if df.column(tc).is_err() {
            anyhow::bail!("Specified time column '{tc}' not found in DataFrame");
        }
        tc.to_string()
    } else {
        // Check common candidates
        let candidates = ["time", "t", "timestamp", "zeit", "Time", "Timestamp"];
        let mut found = None;
        for cand in candidates {
            if df.column(cand).is_ok() {
                found = Some(cand.to_string());
                break;
            }
        }
        if let Some(name) = found {
            name
        } else {
            // Check if first column is numeric
            let first_col = df
                .columns()
                .first()
                .ok_or_else(|| anyhow::anyhow!("No columns in DataFrame"))?;
            if first_col.dtype().is_numeric() {
                first_col.name().to_string()
            } else {
                "time".to_string()
            }
        }
    };

    let num_cols = df.width() + 1;
    let mut mdf = Mdf {
        mdf_info: crate::mdfinfo::MdfInfo::V4(Box::new(
            crate::mdfinfo::mdfinfo4::MdfInfo4::new("polars_export.mf4", num_cols),
        )),
    };

    let has_master_in_df = df.column(&master_name).is_ok();

    if has_master_in_df {
        let master_col = df.column(&master_name)?;
        let arrow_arr = series_to_arrow_array(master_col.as_materialized_series())?;
        mdf.add_channel(
            master_name.clone(),
            arrow_arr,
            None,
            Some(1),
            true,
            Some("s".to_string()),
            None,
        )?;
    } else {
        let time_vals: Vec<f64> = (0..height).map(|i| i as f64).collect();
        let time_arr = Arc::new(arrow::array::Float64Array::from(time_vals));
        mdf.add_channel(
            master_name.clone(),
            time_arr,
            None,
            Some(1), // sync_type = Time
            true,    // master_flag = true
            Some("s".to_string()),
            Some("Synthetic master time channel".to_string()),
        )?;
    }

    for col in df.columns() {
        let col_name = col.name().to_string();
        if col_name == master_name {
            continue;
        }
        let series = col.as_materialized_series();
        let arrow_arr = series_to_arrow_array(series)?;

        mdf.add_channel(
            col_name,
            arrow_arr,
            Some(master_name.clone()),
            None,
            false,
            None,
            None,
        )?;
    }

    Ok(mdf)
}

/// Directly writes a Polars [`DataFrame`] to an MDF 4.3 file.
pub fn write_polars(
    df: &DataFrame,
    file_name: &str,
    time_col: Option<&str>,
    compression: crate::mdfinfo::mdfinfo4::CompressionAlgorithm,
) -> anyhow::Result<Mdf> {
    let mut mdf = from_polars_df(df, time_col)?;
    mdf.write(file_name, compression)
}

/// Converts a rust Apache Arrow array into a Python polars Series via PyArrow FFI.
#[allow(dead_code)] // used from src/mdfr.rs (PyO3 bindings), invisible to the bin target
pub fn rust_arrow_to_py_series(array: Arc<dyn Array>, name: &str) -> PyResult<Py<PyAny>> {
    pyo3::Python::attach(|py| {
        let pyarrow_array = to_py_array(py, array)?;
        let polars = py.import("polars")?;
        let pyname = PyList::new(py, [name])?;
        let out = polars
            .unbind()
            .call_method1(py, "from_arrow", (pyarrow_array, pyname))?;
        Ok(out)
    })
}
