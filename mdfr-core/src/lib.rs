//! # mdfr-core — `no_std` core of mdfr
//!
//! Shared MDF 4.3 block model, codecs and the embedded logger.
//! Requires `alloc`; enable the `std` feature for `std::fs` sinks.
#![no_std]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

pub mod blocks;
pub mod compress;
pub mod error;
pub mod io;
pub mod logger;
pub mod time;

#[cfg(feature = "capi")]
pub mod capi;

pub use error::{Error, Result};
pub use io::{SeekFrom, Sink, VecSink};
pub use time::{ManualClock, TimeSource};
