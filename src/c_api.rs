//! C API
use crate::mdfreader::Mdf;
use arrow::ffi::{FFI_ArrowArray, to_ffi};
use libc::c_char;
use std::ffi::{CStr, CString, c_uchar, c_ushort};

/// create a new mdf from a file and its metadata.
/// Returns a heap-allocated Mdf pointer, or null on error.
/// Caller must free the returned pointer with `free_mdf()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn new_mdf(file_name: *const c_char) -> *mut Mdf {
    unsafe {
        // # Safety
        //
        // It is the caller's guarantee to ensure `file_name`:
        //
        // - is not a null pointer
        // - points to valid, initialized data
        // - points to memory ending in a null byte
        // - won't be mutated for the duration of this function call
        if file_name.is_null() {
            return std::ptr::null_mut();
        }
        let f = match CStr::from_ptr(file_name).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        };
        match Mdf::new(f) {
            Ok(mdf) => Box::into_raw(Box::new(mdf)),
            Err(_) => std::ptr::null_mut(),
        }
    }
}

/// frees an Mdf object previously returned by `new_mdf()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn free_mdf(mdf: *mut Mdf) {
    unsafe {
        if !mdf.is_null() {
            // SAFETY: mdf was created by Box::into_raw in new_mdf(); we reconstruct and drop it.
            drop(Box::from_raw(mdf));
        }
    }
}

/// returns mdf file version. Returns 0 on null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_version(mdf: *const Mdf) -> c_ushort {
    unsafe {
        // SAFETY: caller guarantees mdf is either null or a valid pointer returned by new_mdf().
        // as_ref() handles the null case safely.
        if let Some(mdf) = mdf.as_ref() {
            mdf.get_version()
        } else {
            0
        }
    }
}

/// returns channel's unit string
/// if no unit is existing for this channel, returns a null pointer.
/// The returned pointer is heap-allocated; caller must free it with `libc::free()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_channel_unit(
    mdf: *const Mdf,
    channel_name: *const c_char,
) -> *const c_char {
    unsafe {
        // SAFETY: caller guarantees both pointers are either null or valid null-terminated C strings.
        // Null is checked before dereferencing. CStr::from_ptr requires a valid, null-terminated string.
        if mdf.is_null() || channel_name.is_null() {
            return std::ptr::null();
        }
        let name = match CStr::from_ptr(channel_name).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null(),
        };
        if let Some(mdf) = mdf.as_ref() {
            match mdf.get_channel_unit(name) {
                Ok(Some(unit)) => match CString::new(unit) {
                    Ok(cs) => cs.into_raw(),
                    Err(_) => std::ptr::null(),
                },
                _ => std::ptr::null(),
            }
        } else {
            std::ptr::null()
        }
    }
}

/// returns channel's description string
/// if no description is existing for this channel, returns null pointer.
/// The returned pointer is heap-allocated; caller must free it with `libc::free()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_channel_desc(
    mdf: *const Mdf,
    channel_name: *const libc::c_char,
) -> *const c_char {
    unsafe {
        // SAFETY: caller guarantees both pointers are either null or valid null-terminated C strings.
        if mdf.is_null() || channel_name.is_null() {
            return std::ptr::null();
        }
        let name = match CStr::from_ptr(channel_name).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null(),
        };
        if let Some(mdf) = mdf.as_ref() {
            match mdf.get_channel_desc(name) {
                Ok(Some(desc)) => match CString::new(desc) {
                    Ok(cs) => cs.into_raw(),
                    Err(_) => std::ptr::null(),
                },
                _ => std::ptr::null(),
            }
        } else {
            std::ptr::null()
        }
    }
}

/// returns channel's associated master channel name string
/// if no master channel existing, returns null pointer.
/// The returned pointer is heap-allocated; caller must free it with `libc::free()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_channel_master(
    mdf: *const Mdf,
    channel_name: *const libc::c_char,
) -> *const c_char {
    unsafe {
        // SAFETY: caller guarantees both pointers are either null or valid null-terminated C strings.
        if mdf.is_null() || channel_name.is_null() {
            return std::ptr::null();
        }
        let name = match CStr::from_ptr(channel_name).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null(),
        };
        if let Some(mdf) = mdf.as_ref() {
            match mdf.get_channel_master(name) {
                Some(st) => match CString::new(st) {
                    Ok(cs) => cs.into_raw(),
                    Err(_) => std::ptr::null(),
                },
                None => std::ptr::null(),
            }
        } else {
            std::ptr::null()
        }
    }
}

/// returns channel's associated master channel type string
/// 0 = None (normal data channels), 1 = Time (seconds), 2 = Angle (radians),
/// 3 = Distance (meters), 4 = Index (zero-based index values)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_channel_master_type(
    mdf: *const Mdf,
    channel_name: *const libc::c_char,
) -> c_uchar {
    unsafe {
        // SAFETY: caller guarantees both pointers are either null or valid null-terminated C strings.
        if mdf.is_null() || channel_name.is_null() {
            return 0;
        }
        let name = match CStr::from_ptr(channel_name).to_str() {
            Ok(s) => s,
            Err(_) => return 0,
        };
        if let Some(mdf) = mdf.as_ref() {
            mdf.get_channel_master_type(name)
        } else {
            0
        }
    }
}

/// returns a sorted array of strings of all channel names contained in file.
/// The returned pointer is heap-allocated; call `free_channel_names_set(ptr, len)` to free it,
/// where `len` is the number of channel names (obtained separately, e.g. via a count function).
/// Returns null on null input.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_channel_names_set(mdf: *const Mdf) -> *const *mut c_char {
    unsafe {
        // SAFETY: caller guarantees mdf is either null or a valid pointer from new_mdf().
        if let Some(mdf) = mdf.as_ref() {
            let set = mdf.get_channel_names_set();
            let mut s = set.into_iter().collect::<Vec<String>>();
            s.sort();
            let mut cstring_vec = s
                .iter()
                .filter_map(|e| CString::new(e.as_str()).ok())
                .map(std::ffi::CString::into_raw)
                .collect::<Vec<*mut c_char>>();
            cstring_vec.shrink_to_fit();
            let p = cstring_vec.as_ptr();
            // SAFETY: We intentionally leak the Vec here to transfer ownership of the
            // backing allocation to the caller. The pointer remains valid until
            // free_channel_names_set() is called with the correct length.
            std::mem::forget(cstring_vec);
            p
        } else {
            std::ptr::null()
        }
    }
}

/// frees a channel names array returned by `get_channel_names_set()`.
/// `len` must match the number of channels returned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn free_channel_names_set(ptr: *mut *mut c_char, len: usize) {
    unsafe {
        if ptr.is_null() {
            return;
        }
        // SAFETY: ptr and each element were allocated by Rust (CString::into_raw / Vec::into_raw_parts).
        let slice = std::slice::from_raw_parts_mut(ptr, len);
        for p in slice.iter_mut() {
            if !p.is_null() {
                drop(CString::from_raw(*p));
            }
        }
        drop(Vec::from_raw_parts(ptr, len, len));
    }
}

/// load all channels data in memory
#[unsafe(no_mangle)]
pub unsafe extern "C" fn load_all_channels_data_in_memory(mdf: *mut Mdf) {
    unsafe {
        // SAFETY: caller guarantees mdf is either null or a valid pointer from new_mdf().
        if let Some(mdf) = mdf.as_mut()
            && let Err(e) = mdf.load_all_channels_data_in_memory()
        {
            log::error!("load_all_channels_data_in_memory failed: {e}");
        }
    }
}

/// returns channel's arrow Array as a heap-allocated pointer.
/// Caller must free it with `free_channel_array()`.
/// Returns null pointer if channel not found.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_channel_array(
    mdf: *const Mdf,
    channel_name: *const libc::c_char,
) -> *mut FFI_ArrowArray {
    unsafe {
        // SAFETY: caller guarantees both pointers are either null or valid null-terminated C strings,
        // and mdf is a valid pointer from new_mdf().
        if mdf.is_null() || channel_name.is_null() {
            return std::ptr::null_mut();
        }
        let name = match CStr::from_ptr(channel_name).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        };
        if let Some(mdf) = mdf.as_ref() {
            match mdf.get_channel_data(name) {
                Some(data) => match to_ffi(&data.to_data()) {
                    Ok((array, _)) => Box::into_raw(Box::new(array)),
                    Err(e) => {
                        log::error!("get_channel_array: FFI conversion failed: {e}");
                        std::ptr::null_mut()
                    }
                },
                None => std::ptr::null_mut(),
            }
        } else {
            std::ptr::null_mut()
        }
    }
}

/// frees an FFI_ArrowArray returned by `get_channel_array()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn free_channel_array(array: *mut FFI_ArrowArray) {
    unsafe {
        if !array.is_null() {
            // SAFETY: array was created by Box::into_raw in get_channel_array().
            drop(Box::from_raw(array));
        }
    }
}

// export to Parquet file
// Compression can be one of the following strings
// "snappy", "gzip", "lzo", "brotli", "lz4", "lz4raw"
//  or null pointer if no compression wanted
#[unsafe(no_mangle)]
#[cfg(feature = "parquet")]
pub unsafe extern "C" fn export_to_parquet(
    mdf: *const Mdf,
    file_name: *const c_char,
    compression: *const c_char,
) {
    unsafe {
        // # Safety
        //
        // It is the caller's guarantee to ensure `file_name`:
        //
        // - is not a null pointer
        // - points to valid, initialized data
        // - points to memory ending in a null byte
        // - won't be mutated for the duration of this function call
        if mdf.is_null() || file_name.is_null() {
            return;
        }
        let name = match CStr::from_ptr(file_name).to_str() {
            Ok(s) => s,
            Err(_) => return,
        };
        let comp = if compression.is_null() {
            None
        } else {
            match CStr::from_ptr(compression).to_str() {
                Ok(s) => Some(s),
                Err(_) => return,
            }
        };
        if let Some(mdf) = mdf.as_ref()
            && let Err(e) = mdf.export_to_parquet(name, comp)
        {
            log::error!("export_to_parquet failed: {e}");
        }
    }
}

// export to hdf5 file
// Compression can be one of the following strings
// "deflate", "lzf"
//  or null pointer if no compression wanted
#[unsafe(no_mangle)]
#[cfg(feature = "hdf5")]
pub unsafe extern "C" fn export_to_hdf5(
    mdf: *const Mdf,
    file_name: *const c_char,
    compression: *const c_char,
) {
    // # Safety
    //
    // It is the caller's guarantee to ensure `file_name`:
    //
    // - is not a null pointer
    // - points to valid, initialized data
    // - points to memory ending in a null byte
    // - won't be mutated for the duration of this function call
    unsafe {
        if mdf.is_null() || file_name.is_null() {
            return;
        }
        let name = match CStr::from_ptr(file_name).to_str() {
            Ok(s) => s,
            Err(_) => return,
        };
        let comp = if compression.is_null() {
            None
        } else {
            match CStr::from_ptr(compression).to_str() {
                Ok(s) => Some(s),
                Err(_) => return,
            }
        };
        if let Some(mdf) = mdf.as_ref() {
            if let Err(e) = mdf.export_to_hdf5(name, comp) {
                log::error!("export_to_hdf5 failed: {e}");
            }
        }
    }
}

/// Returns the recording sequence index (MDF 4.3 `Recorder.SequenceIndex` common_property).
/// Writes the value into `*out` and returns true if the property is present; false otherwise.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_recorder_sequence_index(mdf: *const Mdf, out: *mut u64) -> bool {
    unsafe {
        if mdf.is_null() || out.is_null() {
            return false;
        }
        if let Some(mdf) = mdf.as_ref()
            && let Some(v) = mdf.get_recorder_sequence_index()
        {
            *out = v;
            return true;
        }
        false
    }
}

/// Returns the recorder file index (MDF 4.3 `Recorder.FileIndex` common_property).
/// Writes the value into `*out` and returns true if the property is present; false otherwise.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_recorder_file_index(mdf: *const Mdf, out: *mut u64) -> bool {
    unsafe {
        if mdf.is_null() || out.is_null() {
            return false;
        }
        if let Some(mdf) = mdf.as_ref()
            && let Some(v) = mdf.get_recorder_file_index()
        {
            *out = v;
            return true;
        }
        false
    }
}

/// Returns whether this is the last file in the recorder sequence
/// (MDF 4.3 `Recorder.FileLast` common_property).
/// Writes 0 (false) or 1 (true) into `*out` and returns true if the property is present.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_recorder_file_last(mdf: *const Mdf, out: *mut c_uchar) -> bool {
    unsafe {
        if mdf.is_null() || out.is_null() {
            return false;
        }
        if let Some(mdf) = mdf.as_ref()
            && let Some(v) = mdf.get_recorder_file_last()
        {
            *out = v as c_uchar;
            return true;
        }
        false
    }
}

/// Returns the recorder UUID string (MDF 4.3 `Recorder.UUID` common_property).
/// Returns a heap-allocated null-terminated C string; caller must free with `libc::free()`.
/// Returns null pointer if the property is absent or on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_recorder_uuid(mdf: *const Mdf) -> *const c_char {
    unsafe {
        if mdf.is_null() {
            return std::ptr::null();
        }
        if let Some(mdf) = mdf.as_ref()
            && let Some(s) = mdf.get_recorder_uuid()
            && let Ok(cs) = CString::new(s)
        {
            return cs.into_raw();
        }
        std::ptr::null()
    }
}

/// Returns the measurement UUID string (MDF 4.3 `MeasurementUUID` common_property).
/// Returns a heap-allocated null-terminated C string; caller must free with `libc::free()`.
/// Returns null pointer if the property is absent or on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_measurement_uuid(mdf: *const Mdf) -> *const c_char {
    unsafe {
        if mdf.is_null() {
            return std::ptr::null();
        }
        if let Some(mdf) = mdf.as_ref()
            && let Some(s) = mdf.get_measurement_uuid()
            && let Ok(cs) = CString::new(s)
        {
            return cs.into_raw();
        }
        std::ptr::null()
    }
}

/// Returns the author string from HD common_properties (MDF 4.3).
/// Returns a heap-allocated null-terminated C string; caller must free with `libc::free()`.
/// Returns null pointer if the property is absent or on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_author(mdf: *const Mdf) -> *const c_char {
    unsafe {
        if mdf.is_null() {
            return std::ptr::null();
        }
        if let Some(mdf) = mdf.as_ref()
            && let Some(s) = mdf.get_author()
            && let Ok(cs) = CString::new(s)
        {
            return cs.into_raw();
        }
        std::ptr::null()
    }
}

/// Returns the department string from HD common_properties (MDF 4.3).
/// Returns a heap-allocated null-terminated C string; caller must free with `libc::free()`.
/// Returns null pointer if the property is absent or on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_department(mdf: *const Mdf) -> *const c_char {
    unsafe {
        if mdf.is_null() {
            return std::ptr::null();
        }
        if let Some(mdf) = mdf.as_ref()
            && let Some(s) = mdf.get_department()
            && let Ok(cs) = CString::new(s)
        {
            return cs.into_raw();
        }
        std::ptr::null()
    }
}

/// Returns the project string from HD common_properties (MDF 4.3).
/// Returns a heap-allocated null-terminated C string; caller must free with `libc::free()`.
/// Returns null pointer if the property is absent or on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_project(mdf: *const Mdf) -> *const c_char {
    unsafe {
        if mdf.is_null() {
            return std::ptr::null();
        }
        if let Some(mdf) = mdf.as_ref()
            && let Some(s) = mdf.get_project()
            && let Ok(cs) = CString::new(s)
        {
            return cs.into_raw();
        }
        std::ptr::null()
    }
}

/// Returns the subject string from HD common_properties (MDF 4.3).
/// Returns a heap-allocated null-terminated C string; caller must free with `libc::free()`.
/// Returns null pointer if the property is absent or on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_subject(mdf: *const Mdf) -> *const c_char {
    unsafe {
        if mdf.is_null() {
            return std::ptr::null();
        }
        if let Some(mdf) = mdf.as_ref()
            && let Some(s) = mdf.get_subject()
            && let Ok(cs) = CString::new(s)
        {
            return cs.into_raw();
        }
        std::ptr::null()
    }
}
