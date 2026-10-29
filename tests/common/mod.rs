/// Returns path to external ASAM MDF reference test files.
/// Priority:
/// 1. `MDFREADER_TESTS_PATH` env var (if set)
/// 2. Local workspace `test_files/mdfreader/` (if `test_files/Examples/` is present)
/// 3. Developer fallback `/home/ratal/workspace/mdfreader/mdfreader/tests/`
#[allow(dead_code)]
pub fn mdfreader_tests_path() -> String {
    if let Ok(path) = std::env::var("MDFREADER_TESTS_PATH") {
        return path;
    }
    if std::path::Path::new("test_files/Examples").is_dir() {
        return "test_files/mdfreader/".to_string();
    }
    "/home/ratal/workspace/mdfreader/mdfreader/tests/".to_string()
}

/// Returns the base directory containing the ASAM MDF4 Examples/ test files.
#[allow(dead_code)]
pub fn base_path_mdf4() -> String {
    if std::path::Path::new("test_files/Examples").is_dir() {
        return "test_files/Examples/".to_string();
    }
    format!("{}MDF4/MDF4.3/Base_Standard/Examples/", mdfreader_tests_path())
}

/// Relative path to mdfr's own test_files directory (relative to workspace root).
#[allow(dead_code)]
pub const TEST_FILES: &str = "test_files/";

/// Relative path to synthetic test fixtures directory.
#[allow(dead_code)]
pub const TEST_FILES_SYNTHETIC: &str = "test_files/synthetic/";
