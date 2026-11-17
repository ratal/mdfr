//! Consolidated Core Integration Test Suite.
//!
//! Runs core MDF reading, writing, exporting, synthetic fixture, and embedded
//! logger tests in parallel within a single test binary with isolated temp files
//! and atomic fixture initialization.

mod common;
pub mod core;
