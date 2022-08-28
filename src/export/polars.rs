//! this module provides methods to get directly from arrow into polars (rust or python)
use arrow2::array::Array;
use arrow2::compute::cast;
use arrow2::temporal_conversions::*;
use polars::chunked_array::ChunkedArray;
use polars::prelude::*;
use polars::series::Series;
use pyo3::exceptions::PyValueError;
use pyo3::{PyAny, PyObject, PyResult, ToPyObject};

use crate::mdfreader::arrow::{array_to_rust, to_py_array};
use crate::mdfreader::Mdf;

impl Mdf {
    /// returns the channel polars serie
    pub fn get_polars_series(&self, name: &str) -> Option<Series> {
        let data = self.get_channel_data(name);
        let mut out: Option<Series> = None;
        if let Some(data) = data {
            let dtype = data.data_type().clone();
            let chunks = vec![data.clone()];
            out = match dtype {
                ArrowDataType::Null => {
                    // we don't support null types yet so we use a small digit type filled with nulls
                    let len = chunks.iter().fold(0, |acc, array| acc + array.len());
                    Some(Int8Chunked::full_null(name, len).into_series())
                }
                ArrowDataType::Boolean => {
                    Some(ChunkedArray::<BooleanType>::from_chunks(name, chunks).into_series())
                }
                ArrowDataType::Int8 => {
                    Some(ChunkedArray::<Int8Type>::from_chunks(name, chunks).into_series())
                }
                ArrowDataType::Int16 => {
                    Some(ChunkedArray::<Int16Type>::from_chunks(name, chunks).into_series())
                }
                ArrowDataType::Int32 => {
                    Some(ChunkedArray::<Int32Type>::from_chunks(name, chunks).into_series())
                }
                ArrowDataType::Int64 => {
                    Some(ChunkedArray::<Int64Type>::from_chunks(name, chunks).into_series())
                }
                ArrowDataType::UInt8 => {
                    Some(ChunkedArray::<UInt8Type>::from_chunks(name, chunks).into_series())
                }
                ArrowDataType::UInt16 => {
                    Some(ChunkedArray::<UInt16Type>::from_chunks(name, chunks).into_series())
                }
                ArrowDataType::UInt32 => {
                    Some(ChunkedArray::<UInt32Type>::from_chunks(name, chunks).into_series())
                }
                ArrowDataType::UInt64 => {
                    Some(ChunkedArray::<UInt64Type>::from_chunks(name, chunks).into_series())
                }
                ArrowDataType::Float16 => {
                    let chunks = cast_chunks(&chunks, &DataType::Float32).unwrap();
                    Some(Float32Chunked::from_chunks(name, chunks).into_series())
                }
                ArrowDataType::Float32 => {
                    Some(ChunkedArray::<Float32Type>::from_chunks(name, chunks).into_series())
                }
                ArrowDataType::Float64 => {
                    Some(ChunkedArray::<Float64Type>::from_chunks(name, chunks).into_series())
                }
                ArrowDataType::Timestamp(tu, tz) => {
                    let mut time_zone = tz.clone();
                    if time_zone.as_deref() == Some("") {
                        time_zone = None;
                    }
                    // we still drop timezone for now
                    let chunks = cast_chunks(&chunks, &DataType::Int64).unwrap();
                    let s = Int64Chunked::from_chunks(name, chunks)
                        .into_datetime(TimeUnit::from(&tu), time_zone)
                        .into_series();
                    Some(match tu {
                        ArrowTimeUnit::Second => &s * MILLISECONDS,
                        ArrowTimeUnit::Millisecond => s,
                        ArrowTimeUnit::Microsecond => s,
                        ArrowTimeUnit::Nanosecond => s,
                    })
                }
                ArrowDataType::Date32 => {
                    let chunks = cast_chunks(&chunks, &DataType::Int32).unwrap();
                    Some(
                        Int32Chunked::from_chunks(name, chunks)
                            .into_date()
                            .into_series(),
                    )
                }
                ArrowDataType::Date64 => {
                    let chunks = cast_chunks(&chunks, &DataType::Int64).unwrap();
                    let ca = Int64Chunked::from_chunks(name, chunks);
                    Some(ca.into_datetime(TimeUnit::Milliseconds, None).into_series())
                }
                ArrowDataType::Time64(tu) | ArrowDataType::Time32(tu) => {
                    let mut chunks = chunks;
                    if matches!(dtype, ArrowDataType::Time32(_)) {
                        chunks = cast_chunks(&chunks, &DataType::Int32).unwrap();
                    }
                    let chunks = cast_chunks(&chunks, &DataType::Int64).unwrap();
                    let s = Int64Chunked::from_chunks(name, chunks)
                        .into_time()
                        .into_series();
                    Some(match tu {
                        ArrowTimeUnit::Second => &s * NANOSECONDS,
                        ArrowTimeUnit::Millisecond => &s * 1_000_000,
                        ArrowTimeUnit::Microsecond => &s * 1_000,
                        ArrowTimeUnit::Nanosecond => s,
                    })
                }
                ArrowDataType::Duration(tu) => {
                    let chunks = cast_chunks(&chunks, &DataType::Int64).unwrap();
                    let s = Int64Chunked::from_chunks(name, chunks)
                        .into_duration(TimeUnit::from(&tu))
                        .into_series();
                    Some(match tu {
                        ArrowTimeUnit::Second => &s * MILLISECONDS,
                        ArrowTimeUnit::Millisecond => s,
                        ArrowTimeUnit::Microsecond => s,
                        ArrowTimeUnit::Nanosecond => s,
                    })
                }
                ArrowDataType::Utf8 => {
                    let chunks = cast_chunks(&chunks, &DataType::Utf8).unwrap();
                    Some(Utf8Chunked::from_chunks(name, chunks).into_series())
                }
                ArrowDataType::LargeUtf8 => {
                    Some(Utf8Chunked::from_chunks(name, chunks).into_series())
                }
                ArrowDataType::FixedSizeList(_, _) => todo!(),
                ArrowDataType::Extension(_ext_str, _dtype, _) => todo!(),
                _ => None,
            };
        }
        out
    }
}

/// converts an array to a new datatype
fn cast_chunks(chunks: &[ArrayRef], dtype: &DataType) -> Result<Vec<ArrayRef>> {
    let chunks = chunks
        .iter()
        .map(|arr| cast::cast(arr.as_ref(), &dtype.to_arrow(), Default::default()))
        .map(|arr| arr.map(|x| x))
        .collect::<arrow2::error::Result<Vec<_>>>()?;
    Ok(chunks)
}

/// converts a python polars series into a polars rust series
pub fn py_series_to_rust_series(series: &PyAny) -> PyResult<Series> {
    // rechunk series so that they have a single arrow array
    let series = series.call_method0("rechunk")?;

    let name = series.getattr("name")?.extract::<String>()?;

    // retrieve pyarrow array
    let array = series.call_method0("to_arrow")?;

    // retrieve rust arrow array
    let array = array_to_rust(array)?;

    Series::try_from((name.as_str(), array)).map_err(|e| PyValueError::new_err(format!("{}", e)))
}

pub fn rust_series_to_py_series(series: &Series) -> PyResult<PyObject> {
    // ensure we have a single chunk
    let series = series.rechunk();
    let array = series.to_arrow(0);

    // acquire the gil
    pyo3::Python::with_gil(|py| {
        // import pyarrow
        let pyarrow = py.import("pyarrow")?;

        // pyarrow array
        let pyarrow_array = to_py_array(py, pyarrow, &array)?;

        // import polars
        let polars = py.import("polars")?;
        let out = polars.call_method1("from_arrow", (pyarrow_array,))?;
        Ok(out.to_object(py))
    })
}

pub fn rust_arrow_to_py_series(array: &Box<dyn Array>) -> PyResult<PyObject> {
    // ensure we have a single chunk

    // acquire the gil
    pyo3::Python::with_gil(|py| {
        // import pyarrow
        let pyarrow = py.import("pyarrow").expect("could not import pyarrow");

        // pyarrow array
        let pyarrow_array = to_py_array(py, pyarrow, array)
            .expect("failed to convert arrow array to pyarrow array");

        // import polars
        let polars = py.import("polars").expect("could not import polars");
        let out = polars
            .call_method1("from_arrow", (pyarrow_array,))
            .expect("method from_arrow not existing");
        Ok(out.to_object(py))
    })
}
