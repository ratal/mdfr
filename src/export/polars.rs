use arrow2::array::{Array, ArrayRef};
use arrow2::compute::cast;
use arrow2::temporal_conversions::*;
use polars::chunked_array::ChunkedArray;
use polars::prelude::*;
use polars::series::Series;
use pyo3::exceptions::PyValueError;
use pyo3::{PyAny, PyObject, PyResult, Python, ToPyObject};

use crate::mdfreader::arrow::{array_to_rust, to_py_array};
use crate::mdfreader::Mdf;

impl Mdf {
    pub fn get_polars_series(&self, name: &str) -> Option<Result<Series>> {
        let data = self.get_channel_data(name);
        let mut out: Option<Result<Series>> = None;
        if let Some(data) = data {
            let dtype = data.data_type().clone();
            let chunks = vec![data];
            out = match dtype {
                ArrowDataType::Null => {
                    // we don't support null types yet so we use a small digit type filled with nulls
                    let len = chunks.iter().fold(0, |acc, array| acc + array.len());
                    Some(Ok(Int8Chunked::full_null(name, len).into_series()))
                }
                ArrowDataType::Boolean => Some(Ok(ChunkedArray::<BooleanType>::from_chunks(
                    name, chunks,
                )
                .into_series())),
                ArrowDataType::Int8 => Some(Ok(ChunkedArray::<Int8Type>::from_chunks(
                    name, chunks,
                )
                .into_series())),
                ArrowDataType::Int16 => Some(Ok(ChunkedArray::<Int16Type>::from_chunks(
                    name, chunks,
                )
                .into_series())),
                ArrowDataType::Int32 => Some(Ok(ChunkedArray::<Int32Type>::from_chunks(
                    name, chunks,
                )
                .into_series())),
                ArrowDataType::Int64 => Some(Ok(ChunkedArray::<Int64Type>::from_chunks(
                    name, chunks,
                )
                .into_series())),
                ArrowDataType::UInt8 => Some(Ok(ChunkedArray::<UInt8Type>::from_chunks(
                    name, chunks,
                )
                .into_series())),
                ArrowDataType::UInt16 => Some(Ok(ChunkedArray::<UInt16Type>::from_chunks(
                    name, chunks,
                )
                .into_series())),
                ArrowDataType::UInt32 => Some(Ok(ChunkedArray::<UInt32Type>::from_chunks(
                    name, chunks,
                )
                .into_series())),
                ArrowDataType::UInt64 => Some(Ok(ChunkedArray::<UInt64Type>::from_chunks(
                    name, chunks,
                )
                .into_series())),
                ArrowDataType::Float16 => {
                    let chunks = cast_chunks(&chunks, &DataType::Float32).unwrap();
                    Some(Ok(Float32Chunked::from_chunks(name, chunks).into_series()))
                }
                ArrowDataType::Float32 => Some(Ok(ChunkedArray::<Float32Type>::from_chunks(
                    name, chunks,
                )
                .into_series())),
                ArrowDataType::Float64 => Some(Ok(ChunkedArray::<Float64Type>::from_chunks(
                    name, chunks,
                )
                .into_series())),
                ArrowDataType::Timestamp(tu, tz) => {
                    let mut tz = tz.clone();
                    if tz.as_deref() == Some("") {
                        tz = None;
                    }
                    // we still drop timezone for now
                    let chunks = cast_chunks(&chunks, &DataType::Int64).unwrap();
                    let s = Int64Chunked::from_chunks(name, chunks)
                        .into_datetime(TimeUnit::from(&tu), tz)
                        .into_series();
                    Some(Ok(match tu {
                        ArrowTimeUnit::Second => &s * MILLISECONDS,
                        ArrowTimeUnit::Millisecond => s,
                        ArrowTimeUnit::Microsecond => s,
                        ArrowTimeUnit::Nanosecond => s,
                    }))
                }
                ArrowDataType::Date32 => {
                    let chunks = cast_chunks(&chunks, &DataType::Int32).unwrap();
                    Some(Ok(Int32Chunked::from_chunks(name, chunks)
                        .into_date()
                        .into_series()))
                }
                ArrowDataType::Date64 => {
                    let chunks = cast_chunks(&chunks, &DataType::Int64).unwrap();
                    let ca = Int64Chunked::from_chunks(name, chunks);
                    Some(Ok(ca
                        .into_datetime(TimeUnit::Milliseconds, None)
                        .into_series()))
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
                    Some(Ok(match tu {
                        ArrowTimeUnit::Second => &s * NANOSECONDS,
                        ArrowTimeUnit::Millisecond => &s * 1_000_000,
                        ArrowTimeUnit::Microsecond => &s * 1_000,
                        ArrowTimeUnit::Nanosecond => s,
                    }))
                }
                ArrowDataType::Duration(tu) => {
                    let chunks = cast_chunks(&chunks, &DataType::Int64).unwrap();
                    let s = Int64Chunked::from_chunks(name, chunks)
                        .into_duration(TimeUnit::from(&tu))
                        .into_series();
                    Some(Ok(match tu {
                        ArrowTimeUnit::Second => &s * MILLISECONDS,
                        ArrowTimeUnit::Millisecond => s,
                        ArrowTimeUnit::Microsecond => s,
                        ArrowTimeUnit::Nanosecond => s,
                    }))
                }
                ArrowDataType::Utf8 => {
                    let chunks = cast_chunks(&chunks, &DataType::Utf8).unwrap();
                    Some(Ok(Utf8Chunked::from_chunks(name, chunks).into_series()))
                }
                ArrowDataType::LargeUtf8 => {
                    Some(Ok(Utf8Chunked::from_chunks(name, chunks).into_series()))
                }
                ArrowDataType::FixedSizeList(_, _) => todo!(),
                ArrowDataType::Extension(_, _, _) => todo!(),
                _ => None,
            };
        }
        out
    }
}

fn cast_chunks(chunks: &[ArrayRef], dtype: &DataType) -> Result<Vec<ArrayRef>> {
    let chunks = chunks
        .iter()
        .map(|arr| cast::cast(arr.as_ref(), &dtype.to_arrow(), Default::default()))
        .map(|arr| arr.map(|x| x.into()))
        .collect::<arrow2::error::Result<Vec<_>>>()?;
    Ok(chunks)
}

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
    let gil = Python::acquire_gil();
    let py = gil.python();
    // import pyarrow
    let pyarrow = py.import("pyarrow")?;

    // pyarrow array
    let pyarrow_array = to_py_array(py, pyarrow, array)?;

    // import polars
    let polars = py.import("polars")?;
    let out = polars.call_method1("from_arrow", (pyarrow_array,))?;
    Ok(out.to_object(py))
}

pub fn rust_arrow_to_py_series(array: Arc<dyn Array>) -> PyResult<PyObject> {
    // ensure we have a single chunk

    // acquire the gil
    let gil = Python::acquire_gil();
    let py = gil.python();
    // import pyarrow
    let pyarrow = py.import("pyarrow").expect("could not import pyarrow");

    // pyarrow array
    let pyarrow_array =
        to_py_array(py, pyarrow, array).expect("failed to convert arrow array to pyarrow array");

    // import polars
    let polars = py.import("polars").expect("could not import polars");
    let out = polars
        .call_method1("from_arrow", (pyarrow_array,))
        .expect("method from_arrow not existing");
    Ok(out.to_object(py))
}
