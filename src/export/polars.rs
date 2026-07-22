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
        Int8(b) => Series::new(name.into(), b.finish_cloned().values().as_ref()),
        UInt8(b) => Series::new(name.into(), b.finish_cloned().values().as_ref()),
        Int16(b) => Series::new(name.into(), b.finish_cloned().values().as_ref()),
        UInt16(b) => Series::new(name.into(), b.finish_cloned().values().as_ref()),
        Int32(b) => Series::new(name.into(), b.finish_cloned().values().as_ref()),
        UInt32(b) => Series::new(name.into(), b.finish_cloned().values().as_ref()),
        Float32(b) => Series::new(name.into(), b.finish_cloned().values().as_ref()),
        Int64(b) => Series::new(name.into(), b.finish_cloned().values().as_ref()),
        UInt64(b) => Series::new(name.into(), b.finish_cloned().values().as_ref()),
        Float64(b) => Series::new(name.into(), b.finish_cloned().values().as_ref()),

        // ── strings ──────────────────────────────────────────────────────────
        Utf8(b) => {
            let arr = b.finish_cloned();
            let v: Vec<Option<&str>> = arr.iter().collect();
            Series::new(name.into(), v)
        }

        // ── byte arrays ──────────────────────────────────────────────────────
        FixedSizeByteArray(b) => {
            let arr = b.finish_cloned();
            bytes_iter_to_series(name, arr.iter())
        }
        VariableSizeByteArray(b) => {
            let arr = b.finish_cloned();
            bytes_iter_to_series(name, arr.iter())
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
