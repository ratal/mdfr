//! Modular tier implementations and shared fixtures for the E2E test suite.

pub mod tier1_data_holders;
pub mod tier1_embedded;
pub mod tier1_engine;
pub mod tier1_mdf_rw;
pub mod tier1_recovery;
pub mod tier2_boundaries;
pub mod tier3_combinations;
pub mod tier4_scenarios;

use anyhow::Result;
use tempfile::NamedTempFile;

/// Helper to create a temporary `.mf4` file that cleans up on drop.
pub fn create_temp_mf4() -> Result<NamedTempFile> {
    tempfile::Builder::new()
        .prefix("mdfr_e2e_")
        .suffix(".mf4")
        .tempfile()
        .map_err(|e| anyhow::anyhow!("Failed to create temp mf4 file: {}", e))
}

/// Helper to create a temporary `.mdf` file for MDF3 testing.
pub fn create_temp_mdf3() -> Result<NamedTempFile> {
    tempfile::Builder::new()
        .prefix("mdfr3_e2e_")
        .suffix(".mdf")
        .tempfile()
        .map_err(|e| anyhow::anyhow!("Failed to create temp mdf file: {}", e))
}
