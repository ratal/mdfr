//! C API
use crate::mdfreader::Mdf;
use arrow2::ffi::{export_array_to_c, ArrowArray};
use libc::c_char;
use std::ffi::{c_uchar, c_ushort, CStr, CString};

/// create a new mdf from a file and its metadata
#[no_mangle]
pub unsafe extern "C" fn new_mdf(file_name: *const c_char) -> *mut Mdf {
    // # Safety
    //
    // It is the caller's guarantee to ensure `file_name`:
    //
    // - is not a null pointer
    // - points to valid, initialized data
    // - points to memory ending in a null byte
    // - won't be mutated for the duration of this function call
    let f = CStr::from_ptr(file_name)
        .to_str()
        .expect("Could not convert into utf8 the file name string");
    match Mdf::new(f) {
        Ok(mut mdf) => {
            let p: *mut Mdf = &mut mdf;
            std::mem::forget(mdf);
            p
        }
        Err(e) => panic!("{e:?}"),
    }
}

/// returns mdf file version
#[no_mangle]
pub unsafe extern "C" fn get_version(mdf: *const Mdf) -> c_ushort {
    if let Some(mdf) = mdf.as_ref() {
        mdf.get_version()
    } else {
        panic!("Null pointer given for Mdf Rust object")
    }
}

/// returns channel's unit string
/// if no unit is existing for this channel, returns a null pointer
#[no_mangle]
pub unsafe extern "C" fn get_channel_unit(
    mdf: *const Mdf,
    channel_name: *const c_char,
) -> *const c_char {
    let name = CStr::from_ptr(channel_name)
        .to_str()
        .expect("Could not convert into utf8 the file name string");
    if let Some(mdf) = mdf.as_ref() {
        match mdf.get_channel_unit(name) {
            Ok(unit) => match unit {
                Some(unit) => CString::new(unit)
                    .expect("CString::new failed because of internal 0 byte")
                    .into_raw(),
                None => std::ptr::null::<c_char>(), // null pointer
            },
            Err(e) => panic!("{}", e),
        }
    } else {
        panic!("Null pointer given for Mdf Rust object")
    }
}

/// returns channel's description string
/// if no description is existing for this channel, returns null pointer
#[no_mangle]
pub unsafe extern "C" fn get_channel_desc(
    mdf: *const Mdf,
    channel_name: *const libc::c_char,
) -> *const c_char {
    let name = CStr::from_ptr(channel_name)
        .to_str()
        .expect("Could not convert into utf8 the file name string");
    if let Some(mdf) = mdf.as_ref() {
        match mdf.get_channel_desc(name) {
            Ok(desc) => {
                match desc {
                    Some(desc) => CString::new(desc)
                        .expect("CString::new failed because of internal 0 byte")
                        .into_raw(),
                    None => std::ptr::null::<c_char>(), // null pointer
                }
            }
            Err(e) => panic!("{}", e),
        }
    } else {
        panic!("Null pointer given for Mdf Rust object")
    }
}

/// returns channel's associated master channel name string
/// if no master channel existing, returns null pointer
#[no_mangle]
pub unsafe extern "C" fn get_channel_master(
    mdf: *const Mdf,
    channel_name: *const libc::c_char,
) -> *const c_char {
    let name = CStr::from_ptr(channel_name)
        .to_str()
        .expect("Could not convert into utf8 the file name string");
    if let Some(mdf) = mdf.as_ref() {
        match mdf.get_channel_master(name) {
            Some(st) => CString::new(st)
                .expect("CString::new failed because of internal 0 byte")
                .into_raw(),
            None => std::ptr::null::<c_char>(), // null pointer
        }
    } else {
        panic!("Null pointer given for Mdf Rust object")
    }
}

/// returns channel's associated master channel type string
/// 0 = None (normal data channels), 1 = Time (seconds), 2 = Angle (radians),
/// 3 = Distance (meters), 4 = Index (zero-based index values)
#[no_mangle]
pub unsafe extern "C" fn get_channel_master_type(
    mdf: *const Mdf,
    channel_name: *const libc::c_char,
) -> c_uchar {
    let name = CStr::from_ptr(channel_name)
        .to_str()
        .expect("Could not convert into utf8 the file name string");
    if let Some(mdf) = mdf.as_ref() {
        mdf.get_channel_master_type(name)
    } else {
        panic!("Null pointer given for Mdf Rust object")
    }
}

/// returns a sorted array of strings of all channel names contained in file
#[no_mangle]
pub unsafe extern "C" fn get_channel_names_set(mdf: *const Mdf) -> *const *mut c_char {
    if let Some(mdf) = mdf.as_ref() {
        let set = mdf.get_channel_names_set();
        let mut s = set.into_iter().collect::<Vec<String>>();
        s.sort();
        let cstring_vec = s
            .iter()
            .map(|e| {
                CString::new(e.to_string())
                    .expect("CString::new failed because of internal 0 byte")
                    .into_raw()
            })
            .collect::<Vec<*mut c_char>>();
        let p = cstring_vec.as_ptr();
        std::mem::forget(cstring_vec);
        p
    } else {
        panic!("Null pointer given for Mdf Rust object")
    }
}

/// load all channels data in memory
#[no_mangle]
pub unsafe extern "C" fn load_all_channels_data_in_memory(mdf: *mut Mdf) {
    if let Some(mdf) = mdf.as_mut() {
        match mdf.load_all_channels_data_in_memory() {
            Ok(_) => {}
            Err(e) => panic!("{}", e),
        }
    } else {
        panic!("Null pointer given for Mdf Rust object")
    }
}

/// returns channel's arrow Array.
/// null pointer returned if not found
#[no_mangle]
pub unsafe extern "C" fn get_channel_array(
    mdf: *const Mdf,
    channel_name: *const libc::c_char,
) -> *const ArrowArray {
    let name = CStr::from_ptr(channel_name)
        .to_str()
        .expect("Could not convert into utf8 the file name string");
    if let Some(mdf) = mdf.as_ref() {
        match mdf.get_channel_data(name) {
            Some(data) => {
                let array = Box::new(export_array_to_c(data.clone()));
                let array_ptr: *const ArrowArray = &*array;
                array_ptr
            }
            None => std::ptr::null::<ArrowArray>(), // null pointers
        }
    } else {
        panic!("Null pointer given for Mdf Rust object")
    }
}

/// export to Parquet file
/// Compression can be one of the following strings
/// "snappy", "gzip", "lzo", "brotli", "lz4", "lz4raw"
///  or null pointer if no compression wanted
#[no_mangle]
pub unsafe extern "C" fn export_to_parquet(
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
    let name = CStr::from_ptr(file_name)
        .to_str()
        .expect("Could not convert into utf8 the file name string");
    let comp = if compression.is_null() {
        None
    } else {
        Some(
            CStr::from_ptr(compression)
                .to_str()
                .expect("Could not convert into utf8 the compression string"),
        )
    };
    if let Some(mdf) = mdf.as_ref() {
        match mdf.export_to_parquet(name, comp) {
            Ok(_) => {}
            Err(e) => panic!("{}", e),
        }
    } else {
        panic!("Null pointer given for Mdf Rust object")
    }
}
