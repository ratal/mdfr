pub mod synthetic;

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
    format!(
        "{}MDF4/MDF4.3/Base_Standard/Examples/",
        mdfreader_tests_path()
    )
}

/// Returns the base directory containing the MDF3 test files.
#[allow(dead_code)]
pub fn base_path_mdf3() -> String {
    if std::path::Path::new("test_files/Examples/mdf3").is_dir() {
        return "test_files/Examples/mdf3/".to_string();
    }
    format!("{}mdf3/", mdfreader_tests_path())
}

/// Relative path to mdfr's own test_files directory (relative to workspace root).
#[allow(dead_code)]
pub const TEST_FILES: &str = "test_files/";

/// Relative path to synthetic test fixtures directory.
#[allow(dead_code)]
pub const TEST_FILES_SYNTHETIC: &str = "test_files/synthetic/";

/// Prepends `.venv/lib/*/site-packages` to Python's `sys.path` at most once per process.
#[allow(dead_code)]
pub fn ensure_python_path_once(py: pyo3::Python<'_>) -> pyo3::PyResult<()> {
    use pyo3::types::PyAnyMethods;
    static INIT: std::sync::Once = std::sync::Once::new();
    let mut err: Option<pyo3::PyErr> = None;
    INIT.call_once(|| {
        let res = (|| -> pyo3::PyResult<()> {
            let sys = py.import("sys")?;
            let sys_path = sys.getattr("path")?;
            let venv_lib = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".venv/lib");
            if let Ok(entries) = std::fs::read_dir(&venv_lib) {
                for entry in entries.flatten() {
                    let site = entry.path().join("site-packages");
                    if site.exists() {
                        sys_path.call_method1("insert", (0, site.to_str().unwrap()))?;
                    }
                }
            }
            Ok(())
        })();
        if let Err(e) = res {
            err = Some(e);
        }
    });
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}
