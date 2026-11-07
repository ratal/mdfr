//! Comprehensive E2E Test Suite for ASAM MDF 4.3 & 3.x (Tiers 1-4)
//!
//! Evaluates opaque-box functional correctness, boundary conditions, cross-feature
//! interactions, and real-world domain scenarios across `mdfr-core` and `mdfr`.
#![allow(
    clippy::assertions_on_constants,
    clippy::manual_div_ceil,
    clippy::field_reassign_with_default,
    clippy::len_zero,
    clippy::useless_vec,
    clippy::unusual_byte_groupings,
    clippy::bool_assert_comparison
)]

pub mod e2e;
