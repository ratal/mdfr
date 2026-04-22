/// Returns path to external ASAM MDF reference test files.
/// Override with `MDFREADER_TESTS_PATH` env var; falls back to developer default.
#[allow(dead_code)]
pub fn mdfreader_tests_path() -> String {
    std::env::var("MDFREADER_TESTS_PATH")
        .unwrap_or_else(|_| "/home/ratal/workspace/mdfreader/mdfreader/tests/".to_string())
}

/// Relative path to mdfr's own test_files directory (relative to workspace root).
#[allow(dead_code)]
pub const TEST_FILES: &str = "test_files/";

/// Relative path to synthetic test fixtures directory.
#[allow(dead_code)]
pub const TEST_FILES_SYNTHETIC: &str = "test_files/synthetic/";
