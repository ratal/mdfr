//! C API
#![allow(clippy::missing_safety_doc)]
use crate::mdfreader::Mdf;
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema, to_ffi};
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
/// The returned pointer is heap-allocated; caller must free it with `free_c_string()`.
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
/// The returned pointer is heap-allocated; caller must free it with `free_c_string()`.
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
/// The returned pointer is heap-allocated; caller must free it with `free_c_string()`.
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

/// returns total number of unique channel names in the mdf file. Returns 0 on null input.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_channel_count(mdf: *const Mdf) -> usize {
    unsafe {
        if let Some(mdf) = mdf.as_ref() {
            mdf.get_channel_names_set().len()
        } else {
            0
        }
    }
}

/// returns a sorted array of strings of all channel names contained in file.
/// The returned pointer is heap-allocated; call `free_channel_names_set(ptr, len)` to free it,
/// where `len` is the number of channel names (obtained via `get_channel_count()`).
/// Returns null on null input.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_channel_names_set(mdf: *const Mdf) -> *const *mut c_char {
    unsafe {
        // SAFETY: caller guarantees mdf is either null or a valid pointer from new_mdf().
        if let Some(mdf) = mdf.as_ref() {
            let set = mdf.get_channel_names_set();
            let mut s = set.into_iter().collect::<Vec<String>>();
            s.sort();
            let cstring_vec = s
                .iter()
                .filter_map(|e| CString::new(e.as_str()).ok())
                .map(std::ffi::CString::into_raw)
                .collect::<Vec<*mut c_char>>();
            let boxed = cstring_vec.into_boxed_slice();
            let p = boxed.as_ptr();
            // SAFETY: Box::into_raw transfers ownership of the slice allocation to caller.
            // Caller frees with free_channel_names_set(ptr, len).
            let _ = Box::into_raw(boxed);
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
        // SAFETY: ptr and each element were allocated by Rust (CString::into_raw / Box::into_raw).
        let slice_ptr = std::ptr::slice_from_raw_parts_mut(ptr, len);
        let boxed = Box::from_raw(slice_ptr);
        for &p in boxed.iter() {
            if !p.is_null() {
                drop(CString::from_raw(p));
            }
        }
        drop(boxed);
    }
}

/// Frees a null-terminated C string allocated by any `get_*` function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn free_c_string(ptr: *mut c_char) {
    unsafe {
        if !ptr.is_null() {
            drop(CString::from_raw(ptr));
        }
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
            match mdf.get_channel_converted_data(name) {
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

/// returns channel's arrow Schema as a heap-allocated pointer.
/// Caller must free it with `free_channel_schema()`.
/// Returns null pointer if channel not found.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_channel_schema(
    mdf: *const Mdf,
    channel_name: *const libc::c_char,
) -> *mut FFI_ArrowSchema {
    unsafe {
        if mdf.is_null() || channel_name.is_null() {
            return std::ptr::null_mut();
        }
        let name = match CStr::from_ptr(channel_name).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        };
        if let Some(mdf) = mdf.as_ref() {
            match mdf.get_channel_converted_data(name) {
                Some(data) => match to_ffi(&data.to_data()) {
                    Ok((_, schema)) => Box::into_raw(Box::new(schema)),
                    Err(e) => {
                        log::error!("get_channel_schema: FFI conversion failed: {e}");
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

/// frees an FFI_ArrowSchema returned by `get_channel_schema()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn free_channel_schema(schema: *mut FFI_ArrowSchema) {
    unsafe {
        if !schema.is_null() {
            drop(Box::from_raw(schema));
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
        if let Some(mdf) = mdf.as_ref()
            && let Err(e) = mdf.export_to_hdf5(name, comp)
        {
            log::error!("export_to_hdf5 failed: {e}");
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
/// Returns a heap-allocated null-terminated C string; caller must free with `free_c_string()`.
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
/// Returns a heap-allocated null-terminated C string; caller must free with `free_c_string()`.
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
/// Returns a heap-allocated null-terminated C string; caller must free with `free_c_string()`.
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
/// Returns a heap-allocated null-terminated C string; caller must free with `free_c_string()`.
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
/// Returns a heap-allocated null-terminated C string; caller must free with `free_c_string()`.
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
/// Returns a heap-allocated null-terminated C string; caller must free with `free_c_string()`.
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

/// Uniquely identifies a channel by its file position.
/// master_name is heap-allocated; free it with `free_channel_id_master()`.
/// master_name is NULL when the channel has no master.
#[repr(C)]
pub struct ChannelIdC {
    pub master_name: *mut c_char,
    pub dg_pos: i64,
    pub cg_pos: i64,
    pub rec_id: u64,
    pub cn_pos: i64,
    pub rec_pos: i32,
}

/// Fills `out` with the channel id for `channel_name`.
/// Returns true if the channel was found, false otherwise.
/// On success the caller owns `out.master_name` and must free it with
/// `free_channel_id_master()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_channel_id(
    mdf: *const Mdf,
    channel_name: *const c_char,
    out: *mut ChannelIdC,
) -> bool {
    unsafe {
        if mdf.is_null() || channel_name.is_null() || out.is_null() {
            return false;
        }
        let Ok(name) = CStr::from_ptr(channel_name).to_str() else {
            return false;
        };
        let Some(mdf) = mdf.as_ref() else {
            return false;
        };
        let db = mdf.get_channels_db();
        let Some((master, dg_pos, cg_pos, rec_id, cn_pos, rec_pos)) = db.get(name) else {
            return false;
        };
        let master_ptr = match master {
            Some(s) => match CString::new(s.as_str()) {
                Ok(cs) => cs.into_raw(),
                Err(_) => return false,
            },
            None => std::ptr::null_mut(),
        };
        (*out) = ChannelIdC {
            master_name: master_ptr,
            dg_pos: *dg_pos,
            cg_pos: *cg_pos,
            rec_id: *rec_id,
            cn_pos: *cn_pos,
            rec_pos: *rec_pos,
        };
        true
    }
}

/// Frees the `master_name` string inside a `ChannelIdC` filled by `get_channel_id()`.
/// Does nothing when `master_name` is NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn free_channel_id_master(id: *mut ChannelIdC) {
    unsafe {
        if id.is_null() {
            return;
        }
        let master = (*id).master_name;
        if !master.is_null() {
            drop(CString::from_raw(master));
            (*id).master_name = std::ptr::null_mut();
        }
    }
}

/// Returns the number of file history (FH) blocks.
/// Returns 0 on null pointer or MDF3.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_file_history_count(mdf: *const Mdf) -> usize {
    unsafe {
        if let Some(mdf) = mdf.as_ref() {
            mdf.mdf_info
                .get_file_history_blocks()
                .map(|v| v.len())
                .unwrap_or(0)
        } else {
            0
        }
    }
}

/// Returns the timestamp (nanoseconds since Unix epoch) of the FH block at `index`.
/// Writes the value into `*out` and returns true on success.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_file_history_time_ns(
    mdf: *const Mdf,
    index: usize,
    out: *mut u64,
) -> bool {
    unsafe {
        if mdf.is_null() || out.is_null() {
            return false;
        }
        if let Some(mdf) = mdf.as_ref()
            && let Some(fh_blocks) = mdf.mdf_info.get_file_history_blocks()
            && let Some(fh) = fh_blocks.get(index)
        {
            *out = fh.fh_time_ns;
            return true;
        }
        false
    }
}

/// Returns the number of event (EV) blocks.
/// Returns 0 on null pointer or MDF3.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_event_count(mdf: *const Mdf) -> usize {
    unsafe {
        if let Some(mdf) = mdf.as_ref() {
            mdf.mdf_info
                .get_event_blocks()
                .map(|v| v.len())
                .unwrap_or(0)
        } else {
            0
        }
    }
}

/// Returns the event type of the EV block at `index`.
/// Returns 255 on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_event_type(mdf: *const Mdf, index: usize) -> u8 {
    unsafe {
        if let Some(mdf) = mdf.as_ref()
            && let Some(ev_blocks) = mdf.mdf_info.get_event_blocks()
            && let Some(ev) = ev_blocks.values().nth(index)
        {
            return ev.ev_type;
        }
        255
    }
}

/// Returns the number of channel hierarchy (CH) blocks.
/// Returns 0 on null pointer or MDF3.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_channel_hierarchy_count(mdf: *const Mdf) -> usize {
    unsafe {
        if let Some(mdf) = mdf.as_ref() {
            mdf.mdf_info
                .get_channel_hierarchy_blocks()
                .map(|v| v.len())
                .unwrap_or(0)
        } else {
            0
        }
    }
}

/// Returns a human-readable list of channel hierarchy.
/// Returns a heap-allocated null-terminated C string; caller must free with `free_c_string()`.
/// Returns null pointer on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn list_channel_hierarchy(mdf: *const Mdf) -> *const c_char {
    unsafe {
        if mdf.is_null() {
            return std::ptr::null();
        }
        if let Some(mdf) = mdf.as_ref() {
            let s = mdf.mdf_info.list_channel_hierarchy();
            if let Ok(cs) = CString::new(s) {
                return cs.into_raw();
            }
        }
        std::ptr::null()
    }
}

/// Returns the number of source information (SI) blocks.
/// Returns 0 on null pointer or MDF3.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_source_information_count(mdf: *const Mdf) -> usize {
    unsafe {
        if let Some(mdf) = mdf.as_ref() {
            mdf.mdf_info
                .get_source_information_blocks()
                .map(|v| v.len())
                .unwrap_or(0)
        } else {
            0
        }
    }
}

/// Returns a human-readable list of source information blocks.
/// Returns a heap-allocated null-terminated C string; caller must free with `free_c_string()`.
/// Returns null pointer on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn list_source_information(mdf: *const Mdf) -> *const c_char {
    unsafe {
        if mdf.is_null() {
            return std::ptr::null();
        }
        if let Some(mdf) = mdf.as_ref() {
            let s = mdf.mdf_info.list_source_information();
            if let Ok(cs) = CString::new(s) {
                return cs.into_raw();
            }
        }
        std::ptr::null()
    }
}

pub use mdfr_core::capi::*;

extern "C" fn std_file_sink_write(ctx: *mut core::ffi::c_void, buf: *const u8, len: usize) -> i32 {
    if ctx.is_null() || (buf.is_null() && len > 0) {
        return -3;
    }
    let file = unsafe { &mut *(ctx as *mut std::fs::File) };
    let slice = if len == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(buf, len) }
    };
    match std::io::Write::write_all(file, slice) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

extern "C" fn std_file_sink_seek(ctx: *mut core::ffi::c_void, pos: u64) -> i32 {
    if ctx.is_null() {
        return -3;
    }
    let file = unsafe { &mut *(ctx as *mut std::fs::File) };
    match std::io::Seek::seek(file, std::io::SeekFrom::Start(pos)) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

extern "C" fn std_file_sink_flush(ctx: *mut core::ffi::c_void) -> i32 {
    if ctx.is_null() {
        return -3;
    }
    let file = unsafe { &mut *(ctx as *mut std::fs::File) };
    match std::io::Write::flush(file) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

extern "C" fn std_file_sink_close(ctx: *mut core::ffi::c_void) -> i32 {
    if !ctx.is_null() {
        unsafe {
            let mut f = Box::from_raw(ctx as *mut std::fs::File);
            let _ = std::io::Write::flush(&mut *f);
            drop(f);
        }
    }
    0
}

extern "C" fn std_clock_now_ticks(_ctx: *mut core::ffi::c_void) -> u64 {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let start = EPOCH.get_or_init(std::time::Instant::now);
    start.elapsed().as_micros() as u64
}

extern "C" fn std_clock_wall_ns(_ctx: *mut core::ffi::c_void, out_ns: *mut u64) -> i32 {
    if out_ns.is_null() {
        return -3;
    }
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => {
            unsafe {
                *out_ns = d.as_nanos() as u64;
            }
            0
        }
        Err(_) => -1,
    }
}

fn make_file_sink_vtable(path_str: &str) -> Result<MdfrSinkVtable, i32> {
    let f = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path_str)
        .map_err(|_| -1i32)?;
    let ctx = Box::into_raw(Box::new(f)) as *mut core::ffi::c_void;
    Ok(MdfrSinkVtable {
        ctx,
        write: Some(std_file_sink_write),
        seek: Some(std_file_sink_seek),
        flush: Some(std_file_sink_flush),
        close: Some(std_file_sink_close),
    })
}

/// Starts an MDF 4.3 streaming logger writing directly to a filesystem `path` using the OS clock (`1 us` tick resolution).
/// Consumes `builder`. Returns null and sets `*out_err` on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdfr_logger_open(
    builder: *mut MdfrBuilderHandle,
    path: *const c_char,
    out_err: *mut i32,
) -> *mut MdfrLoggerHandle {
    if builder.is_null() || path.is_null() {
        if !out_err.is_null() {
            unsafe {
                *out_err = -3;
            }
        }
        return std::ptr::null_mut();
    }
    let Ok(path_str) = (unsafe { CStr::from_ptr(path).to_str() }) else {
        unsafe { mdfr_logger_builder_free(builder) };
        if !out_err.is_null() {
            unsafe {
                *out_err = -3;
            }
        }
        return std::ptr::null_mut();
    };
    let sink = match make_file_sink_vtable(path_str) {
        Ok(s) => s,
        Err(code) => {
            unsafe { mdfr_logger_builder_free(builder) };
            if !out_err.is_null() {
                unsafe {
                    *out_err = code;
                }
            }
            return std::ptr::null_mut();
        }
    };
    let clock = MdfrClockVtable {
        ctx: std::ptr::null_mut(),
        now_ticks: Some(std_clock_now_ticks),
        tick_ns: 1_000,
        wall_clock_ns: Some(std_clock_wall_ns),
    };
    unsafe { mdfr_logger_start(builder, sink, clock, out_err) }
}

/// Rotates an active logger to a new filesystem `new_path`, finalizing the previous file in place.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdfr_logger_rotate_path(
    logger: *mut MdfrLoggerHandle,
    new_path: *const c_char,
) -> i32 {
    if logger.is_null() || new_path.is_null() {
        return -3;
    }
    let Ok(path_str) = (unsafe { CStr::from_ptr(new_path).to_str() }) else {
        return -3;
    };
    let sink = match make_file_sink_vtable(path_str) {
        Ok(s) => s,
        Err(code) => return code,
    };
    unsafe { mdfr_logger_rotate(logger, sink) }
}

/// Finalizes an unfinalized MDF 4.x file (`"UnFinMF "`) in place on disk without copying.
/// Returns `0` on success, negative error code on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdfr_finalize_file_in_place(file_name: *const c_char) -> i32 {
    if file_name.is_null() {
        return -3;
    }
    let Ok(path_str) = (unsafe { CStr::from_ptr(file_name).to_str() }) else {
        return -3;
    };
    match crate::mdfinfo::mdfinfo4::finalize::finalize_file_in_place(path_str) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_c_api_null_safety() {
        unsafe {
            assert!(new_mdf(std::ptr::null()).is_null());
            free_mdf(std::ptr::null_mut());
            assert_eq!(get_version(std::ptr::null()), 0);
            assert!(get_channel_unit(std::ptr::null(), std::ptr::null()).is_null());
            assert!(get_channel_desc(std::ptr::null(), std::ptr::null()).is_null());
            assert!(get_channel_master(std::ptr::null(), std::ptr::null()).is_null());
            assert_eq!(get_channel_master_type(std::ptr::null(), std::ptr::null()), 0);
            assert_eq!(get_channel_count(std::ptr::null()), 0);
            assert!(get_channel_names_set(std::ptr::null()).is_null());
            free_channel_names_set(std::ptr::null_mut(), 0);
            free_c_string(std::ptr::null_mut());
            load_all_channels_data_in_memory(std::ptr::null_mut());
            assert!(get_channel_array(std::ptr::null(), std::ptr::null()).is_null());
            free_channel_array(std::ptr::null_mut());
            assert!(get_channel_schema(std::ptr::null(), std::ptr::null()).is_null());
            free_channel_schema(std::ptr::null_mut());

            #[cfg(feature = "parquet")]
            export_to_parquet(std::ptr::null(), std::ptr::null(), std::ptr::null());

            #[cfg(feature = "hdf5")]
            export_to_hdf5(std::ptr::null(), std::ptr::null(), std::ptr::null());

            let mut u64_val = 0u64;
            let mut u8_val = 0u8;
            assert!(!get_recorder_sequence_index(std::ptr::null(), &mut u64_val));
            assert!(!get_recorder_file_index(std::ptr::null(), &mut u64_val));
            assert!(!get_recorder_file_last(std::ptr::null(), &mut u8_val));
            assert!(get_recorder_uuid(std::ptr::null()).is_null());
            assert!(get_measurement_uuid(std::ptr::null()).is_null());
            assert!(get_author(std::ptr::null()).is_null());
            assert!(get_department(std::ptr::null()).is_null());
            assert!(get_project(std::ptr::null()).is_null());
            assert!(get_subject(std::ptr::null()).is_null());

            let mut ch_id = ChannelIdC {
                master_name: std::ptr::null_mut(),
                dg_pos: 0,
                cg_pos: 0,
                rec_id: 0,
                cn_pos: 0,
                rec_pos: 0,
            };
            assert!(!get_channel_id(std::ptr::null(), std::ptr::null(), &mut ch_id));
            free_channel_id_master(std::ptr::null_mut());

            assert_eq!(get_file_history_count(std::ptr::null()), 0);
            assert!(!get_file_history_time_ns(std::ptr::null(), 0, &mut u64_val));
            assert_eq!(get_event_count(std::ptr::null()), 0);
            assert_eq!(get_event_type(std::ptr::null(), 0), 255);
            assert_eq!(get_channel_hierarchy_count(std::ptr::null()), 0);
            assert!(list_channel_hierarchy(std::ptr::null()).is_null());
            assert_eq!(get_source_information_count(std::ptr::null()), 0);
            assert!(list_source_information(std::ptr::null()).is_null());

            assert!(mdfr_logger_open(std::ptr::null_mut(), std::ptr::null(), std::ptr::null_mut()).is_null());
            assert_eq!(mdfr_logger_rotate_path(std::ptr::null_mut(), std::ptr::null()), -3);
            assert_eq!(mdfr_finalize_file_in_place(std::ptr::null()), -3);
        }
    }

    #[test]
    fn test_c_api_roundtrip_file_and_reader() {
        unsafe {
            let tmp1 = tempfile::Builder::new().suffix(".mf4").tempfile().unwrap();
            let p1 = CString::new(tmp1.path().to_str().unwrap()).unwrap();

            let tmp2 = tempfile::Builder::new().suffix(".mf4").tempfile().unwrap();
            let p2 = CString::new(tmp2.path().to_str().unwrap()).unwrap();

            // Create and write with C logger API
            let builder = mdfr_logger_builder_new(0, 255); // Sorted, NoCompression
            assert!(!builder.is_null());

            let mut g0 = 0u16;
            let group_name = CString::new("Powertrain").unwrap();
            assert_eq!(mdfr_logger_add_group(builder, group_name.as_ptr(), &mut g0), 0);

            let mut ch_rpm = 0u32;
            let ch_name = CString::new("EngineSpeed").unwrap();
            assert_eq!(mdfr_logger_add_channel(builder, g0, ch_name.as_ptr(), 0, 16, &mut ch_rpm), 0);

            let unit = CString::new("rpm").unwrap();
            assert_eq!(mdfr_logger_set_channel_unit(builder, ch_rpm, unit.as_ptr()), 0);
            assert_eq!(mdfr_logger_set_conversion_linear(builder, ch_rpm, 0.0, 0.25), 0);

            let mut err = 0;
            let logger = mdfr_logger_open(builder, p1.as_ptr(), &mut err);
            assert!(!logger.is_null());
            assert_eq!(err, 0);

            // Record
            assert_eq!(mdfr_logger_record_begin(logger, g0), 0);
            assert_eq!(mdfr_logger_record_set_u64(logger, ch_rpm, 4000), 0);
            assert_eq!(mdfr_logger_record_commit(logger, g0), 0);

            // Rotate to p2
            assert_eq!(mdfr_logger_rotate_path(logger, p2.as_ptr()), 0);

            // Close
            assert_eq!(mdfr_logger_close(logger), 0);

            // Now open the rotated p1 file with C Reader API
            let mdf = new_mdf(p1.as_ptr());
            assert!(!mdf.is_null());

            let ver = get_version(mdf);
            assert!(ver >= 400);

            let ch_names_ptr = get_channel_names_set(mdf);
            assert!(!ch_names_ptr.is_null());
            // Free the channel names set (we know it contains at least "time" and "EngineSpeed")
            free_channel_names_set(ch_names_ptr as *mut *mut c_char, 2);

            let u_ptr = get_channel_unit(mdf, ch_name.as_ptr());
            assert!(!u_ptr.is_null());
            assert_eq!(CStr::from_ptr(u_ptr).to_str().unwrap(), "rpm");
            free_c_string(u_ptr as *mut c_char);

            let desc_ptr = get_channel_desc(mdf, ch_name.as_ptr());
            if !desc_ptr.is_null() {
                free_c_string(desc_ptr as *mut c_char);
            }

            let m_ptr = get_channel_master(mdf, ch_name.as_ptr());
            assert!(!m_ptr.is_null());
            assert_eq!(CStr::from_ptr(m_ptr).to_str().unwrap(), "time");
            free_c_string(m_ptr as *mut c_char);

            let m_type = get_channel_master_type(mdf, ch_name.as_ptr());
            assert_eq!(m_type, 0); // 0 = None (normal data channel)

            let time_name = CString::new("time").unwrap();
            let time_m_type = get_channel_master_type(mdf, time_name.as_ptr());
            assert_eq!(time_m_type, 1); // 1 = Time (master channel)

            load_all_channels_data_in_memory(mdf);

            let arr_ptr = get_channel_array(mdf, ch_name.as_ptr());
            assert!(!arr_ptr.is_null());
            free_channel_array(arr_ptr);

            let schema_ptr = get_channel_schema(mdf, ch_name.as_ptr());
            assert!(!schema_ptr.is_null());
            free_channel_schema(schema_ptr);

            let mut cid = ChannelIdC {
                master_name: std::ptr::null_mut(),
                dg_pos: 0,
                cg_pos: 0,
                rec_id: 0,
                cn_pos: 0,
                rec_pos: 0,
            };
            assert!(get_channel_id(mdf, ch_name.as_ptr(), &mut cid));
            assert!(!cid.master_name.is_null());
            assert_eq!(CStr::from_ptr(cid.master_name).to_str().unwrap(), "time");
            free_channel_id_master(&mut cid);

            let mut out_u64 = 0u64;
            let mut out_u8 = 0u8;
            let _ = get_recorder_sequence_index(mdf, &mut out_u64);
            let _ = get_recorder_file_index(mdf, &mut out_u64);
            let _ = get_recorder_file_last(mdf, &mut out_u8);

            let r_uuid = get_recorder_uuid(mdf);
            if !r_uuid.is_null() {
                free_c_string(r_uuid as *mut c_char);
            }
            let m_uuid = get_measurement_uuid(mdf);
            if !m_uuid.is_null() {
                free_c_string(m_uuid as *mut c_char);
            }

            let author = get_author(mdf);
            if !author.is_null() {
                free_c_string(author as *mut c_char);
            }
            let dept = get_department(mdf);
            if !dept.is_null() {
                free_c_string(dept as *mut c_char);
            }
            let proj = get_project(mdf);
            if !proj.is_null() {
                free_c_string(proj as *mut c_char);
            }
            let subj = get_subject(mdf);
            if !subj.is_null() {
                free_c_string(subj as *mut c_char);
            }

            let fh_count = get_file_history_count(mdf);
            if fh_count > 0 {
                let mut fh_time = 0u64;
                let _ = get_file_history_time_ns(mdf, 0, &mut fh_time);
            }

            let _ = get_event_count(mdf);
            let _ = get_channel_hierarchy_count(mdf);
            let hier = list_channel_hierarchy(mdf);
            if !hier.is_null() {
                free_c_string(hier as *mut c_char);
            }
            let _ = get_source_information_count(mdf);
            let si = list_source_information(mdf);
            if !si.is_null() {
                free_c_string(si as *mut c_char);
            }

            #[cfg(feature = "parquet")]
            {
                let pq_tmp = tempfile::Builder::new().suffix(".parquet").tempfile().unwrap();
                let pq_path = CString::new(pq_tmp.path().to_str().unwrap()).unwrap();
                export_to_parquet(mdf, pq_path.as_ptr(), std::ptr::null());
            }

            free_mdf(mdf);

            // Finalize file in place check
            assert_eq!(mdfr_finalize_file_in_place(p2.as_ptr()), 0);
        }
    }
}
