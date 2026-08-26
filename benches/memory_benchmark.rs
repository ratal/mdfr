//! Memory footprint benchmark — measures peak RSS for loading MDF files.
//!
//! Uses `/proc/self/status` on Linux to read VmPeak / VmRSS before and after
//! `Mdf::new()` + `load_all_channels_data_in_memory()`.
//!
//! Run with: cargo bench --bench memory_benchmark

use mdfr::mdfreader::Mdf;
use std::fs;

static LOCAL_FILES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_files/");

fn read_proc_status() -> std::collections::HashMap<String, u64> {
    let mut map = std::collections::HashMap::new();
    if let Ok(content) = fs::read_to_string("/proc/self/status") {
        for line in content.lines() {
            if let Some(rest) = line.strip_prefix("VmPeak:") {
                if let Some(val) = rest.trim().split_whitespace().next() {
                    if let Ok(kb) = val.parse::<u64>() {
                        map.insert("VmPeak".to_string(), kb);
                    }
                }
            } else if let Some(rest) = line.strip_prefix("VmRSS:") {
                if let Some(val) = rest.trim().split_whitespace().next() {
                    if let Ok(kb) = val.parse::<u64>() {
                        map.insert("VmRSS".to_string(), kb);
                    }
                }
            }
        }
    }
    map
}

fn main() {
    for (name, rel) in [
        ("array_f64_le", "synthetic/array_f64_le.mf4"),
        ("be_scalars", "synthetic/be_scalars.mf4"),
        ("complex_f32_le", "synthetic/complex_f32_le.mf4"),
        ("int_linear_cc", "synthetic/int_linear_cc.mf4"),
        ("ld_be_channels", "synthetic/ld_be_channels.mf4"),
    ] {
        let path = format!("{}{rel}", LOCAL_FILES);
        let before = read_proc_status();
        let mut mdf = Mdf::new(&path).expect("failed to open file");
        mdf.load_all_channels_data_in_memory()
            .expect("failed to load data");
        let after = read_proc_status();
        let peak_delta = after.get("VmPeak").unwrap_or(&0) - before.get("VmPeak").unwrap_or(&0);
        let rss_delta = after.get("VmRSS").unwrap_or(&0) - before.get("VmRSS").unwrap_or(&0);
        println!(
            "{name}: peak_delta={peak_delta} kB, rss_delta={rss_delta} kB"
        );
    }
}
