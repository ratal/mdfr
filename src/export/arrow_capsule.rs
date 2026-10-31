//! Arrow PyCapsule zero-copy protocol implementation.
//!
//! Implements `__arrow_c_array__` and `__arrow_c_stream__` PyCapsule export and import
//! for seamless zero-copy interoperability with PyArrow, Polars, DuckDB, and other
//! Arrow-compatible analytical tools.
#![allow(dead_code)]

use arrow::array::{ArrayData, RecordBatchReader};
use arrow::ffi::{to_ffi, FFI_ArrowArray, FFI_ArrowSchema};
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use pyo3::prelude::*;

/// Destructor for the `arrow_schema` capsule.
/// Drops `Box<FFI_ArrowSchema>`, releasing C resources if unconsumed, or freeing memory if consumed.
unsafe extern "C" fn schema_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    unsafe {
        let name = pyo3::ffi::PyCapsule_GetName(capsule);
        let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, name) as *mut FFI_ArrowSchema;
        if ptr.is_null() {
            pyo3::ffi::PyErr_Clear();
            return;
        }
        let _ = Box::from_raw(ptr);
    }
}

/// Destructor for the `arrow_array` capsule.
/// Drops `Box<FFI_ArrowArray>`, releasing C resources if unconsumed, or freeing memory if consumed.
unsafe extern "C" fn array_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    unsafe {
        let name = pyo3::ffi::PyCapsule_GetName(capsule);
        let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, name) as *mut FFI_ArrowArray;
        if ptr.is_null() {
            pyo3::ffi::PyErr_Clear();
            return;
        }
        let _ = Box::from_raw(ptr);
    }
}

/// Destructor for the `arrow_array_stream` capsule.
/// Drops `Box<FFI_ArrowArrayStream>`, releasing C resources if unconsumed, or freeing memory if consumed.
unsafe extern "C" fn stream_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    unsafe {
        let name = pyo3::ffi::PyCapsule_GetName(capsule);
        let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, name) as *mut FFI_ArrowArrayStream;
        if ptr.is_null() {
            pyo3::ffi::PyErr_Clear();
            return;
        }
        let _ = Box::from_raw(ptr);
    }
}

/// Exports an [`ArrayData`] instance into a pair of Python Capsules:
/// `(PyCapsule("arrow_schema"), PyCapsule("arrow_array"))`.
pub fn export_array_to_pycapsules<'py>(
    py: Python<'py>,
    data: &ArrayData,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let (ffi_array, ffi_schema) =
        to_ffi(data).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let schema_box = Box::into_raw(Box::new(ffi_schema));
    let array_box = Box::into_raw(Box::new(ffi_array));

    let schema_capsule = unsafe {
        pyo3::ffi::PyCapsule_New(
            schema_box as *mut _,
            c"arrow_schema".as_ptr(),
            Some(schema_capsule_destructor),
        )
    };
    if schema_capsule.is_null() {
        unsafe {
            drop(Box::from_raw(schema_box));
            drop(Box::from_raw(array_box));
        }
        return Err(PyErr::fetch(py));
    }
    let schema_bound = unsafe { Bound::from_owned_ptr(py, schema_capsule) };

    let array_capsule = unsafe {
        pyo3::ffi::PyCapsule_New(
            array_box as *mut _,
            c"arrow_array".as_ptr(),
            Some(array_capsule_destructor),
        )
    };
    if array_capsule.is_null() {
        unsafe {
            drop(Box::from_raw(array_box));
        }
        return Err(PyErr::fetch(py));
    }
    let array_bound = unsafe { Bound::from_owned_ptr(py, array_capsule) };

    Ok((schema_bound, array_bound))
}

/// Exports a [`RecordBatchReader`] into a Python Capsule:
/// `PyCapsule("arrow_array_stream")`.
pub fn export_reader_to_pycapsule<'py>(
    py: Python<'py>,
    reader: Box<dyn RecordBatchReader + Send>,
) -> PyResult<Bound<'py, PyAny>> {
    let stream = FFI_ArrowArrayStream::new(reader);
    let stream_box = Box::into_raw(Box::new(stream));

    let capsule = unsafe {
        pyo3::ffi::PyCapsule_New(
            stream_box as *mut _,
            c"arrow_array_stream".as_ptr(),
            Some(stream_capsule_destructor),
        )
    };
    if capsule.is_null() {
        unsafe {
            drop(Box::from_raw(stream_box));
        }
        return Err(PyErr::fetch(py));
    }
    Ok(unsafe { Bound::from_owned_ptr(py, capsule) })
}

/// Imports an [`ArrowArrayStreamReader`] from an `arrow_array_stream` Python Capsule.
pub fn import_stream_from_pycapsule(
    stream_capsule: &Bound<'_, PyAny>,
) -> PyResult<ArrowArrayStreamReader> {
    unsafe {
        let ptr = pyo3::ffi::PyCapsule_GetPointer(
            stream_capsule.as_ptr(),
            c"arrow_array_stream".as_ptr(),
        ) as *mut FFI_ArrowArrayStream;
        if ptr.is_null() {
            return Err(PyErr::fetch(stream_capsule.py()));
        }
        let stream = FFI_ArrowArrayStream::from_raw(ptr);
        ArrowArrayStreamReader::try_new(stream)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}
