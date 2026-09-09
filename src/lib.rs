//! # mdfr — ASAM MDF file reader and writer
//!
//! Rust library for reading and writing ASAM Measurement Data Format (MDF) files,
//! supporting versions 3.x and 4.x (up to MDF 4.3).
//!
//! ## Modules
//!
//! - [`mdfreader`] — High-level API for reading MDF files into memory (`Mdf` struct)
//! - [`mdfwriter`] — Writing in-memory data back to MDF 4.2 files
//! - [`mdfinfo`] — MDF file metadata: block structures, parsing, and version-agnostic `MdfInfo` enum
//! - [`data_holder`] — Channel data storage using Apache Arrow arrays
//! - [`export`] — Exporting channel data to Parquet and HDF5 formats
//! - [`mdfr`] — Python bindings via PyO3 (requires `numpy` feature)

#![allow(unstable_name_collisions)]
//#![forbid(unsafe_code)]
mod c_api;
pub mod data_holder;
pub mod export;
pub mod mdfinfo;
#[cfg(feature = "numpy")]
pub mod mdfr;

pub mod mdfreader;
pub mod mdfwriter;
