//! Consolidated ASAM Reference & Conformance Test Suite.
//!
//! Runs all ASAM MDF4/MDF3 example-based integration tests—including the 53
//! per-folder MDF4 scans and MDF3 folder scan—concurrently across Cargo test
//! worker threads.

pub mod asam;
mod common;
