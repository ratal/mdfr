//! # mdfr-core — `no_std` core of mdfr
//!
//! Shared MDF 4.3 block model, codecs and the embedded logger.
//! Requires `alloc`; enable the `std` feature for `std::fs` sinks.
#![no_std]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

pub mod blocks;
pub mod error;
pub mod io;
