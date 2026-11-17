//! Smoke test to verify C/C++ header syntax and consistency for `mdfr.h` and `include/mdfr_logger.h`.

use std::process::Command;

#[test]
fn c_header_syntax_check() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let header_path = format!("{manifest_dir}/mdfr.h");
    let logger_header_path = format!("{manifest_dir}/include/mdfr_logger.h");

    // Check if a C compiler is available
    let has_cc = Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !has_cc {
        eprintln!("Skipping C header check: no C compiler (cc) found");
        return;
    }

    // Verify mdfr.h with C compiler
    let status_c = Command::new("cc")
        .args([
            "-fsyntax-only",
            "-Wall",
            "-Wextra",
            "-pedantic",
            &header_path,
        ])
        .status()
        .expect("Failed to run cc on mdfr.h");
    assert!(status_c.success(), "cc -fsyntax-only failed on mdfr.h");

    // Verify mdfr_logger.h with C compiler
    let status_logger = Command::new("cc")
        .args([
            "-fsyntax-only",
            "-Wall",
            "-Wextra",
            "-pedantic",
            &logger_header_path,
        ])
        .status()
        .expect("Failed to run cc on mdfr_logger.h");
    assert!(
        status_logger.success(),
        "cc -fsyntax-only failed on include/mdfr_logger.h"
    );

    // If g++ is available, check C++ compatibility as well
    let has_cxx = Command::new("c++")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if has_cxx {
        let status_cxx = Command::new("c++")
            .args([
                "-fsyntax-only",
                "-Wall",
                "-Wextra",
                "-pedantic",
                "-x",
                "c++",
                &header_path,
            ])
            .status()
            .expect("Failed to run c++ on mdfr.h");
        assert!(status_cxx.success(), "c++ -fsyntax-only failed on mdfr.h");
    }
}
