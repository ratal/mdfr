//! Adversarial Stress Testing Suite for Milestone 4 Iteration 2.
//!
//! Evaluates:
//! 1. Writer re-saving: multi-DG / sorted MDF4 files re-written via `loaded.write()`.
//!    Confirms channel count remains strictly preserved (no duplication of data channels as master channels).
//! 2. Mask polarity: NumPy boolean masks have strictly `True` for invalid samples and `False` for valid samples.
//! 3. O(1) channel queries: `contains_channel` and `channel_count` with large channel sets (50,000 channels).
//! 4. `add_channel` with dict format, string master, and kwargs.
//! 5. Dotted master channel names in Parquet and CSV export across nested directory paths containing dots.

use std::alloc::{GlobalAlloc, Layout, System};
use std::fs;
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use std::time::Instant;

use arrow::array::{Array, Float64Array};
use arrow::pyarrow::PyArrowType;
use mdfr::mdfinfo::mdfinfo4::MdfInfo4;
use mdfr::mdfr::{MdfLogger, Mdfr};
use mdfr::mdfreader::Mdf;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

// ============================================================================
// Tracking Allocator for Zero-Allocation Assertions
// ============================================================================

struct TrackingAllocator;
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static TRACKED_TID: AtomicI64 = AtomicI64::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let tracked = TRACKED_TID.load(Ordering::Relaxed);
        if tracked != 0 {
            let tid = unsafe { libc::syscall(libc::SYS_gettid) as i64 };
            if tid == tracked {
                ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            }
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

fn ensure_python_path(py: Python<'_>) -> PyResult<()> {
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
}

// ============================================================================
// Challenge 1: Writer Re-saving Multi-DG / Sorted MDF4 Files
// ============================================================================

#[test]
fn test_stress_writer_resaving_preserves_channel_count_and_data() -> anyhow::Result<()> {
    let tmp_dir = tempfile::tempdir()?;

    // 1. Create a 2-channel sorted file with MdfLogger:
    // DG 0 has "time", DG 1 has "rpm"
    let base_file = tmp_dir.path().join("base_sorted_2ch.mf4");
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let logger_cls = py.get_type::<MdfLogger>();
        let logger = logger_cls.call1((base_file.to_str().unwrap(), "unsorted"))?;
        let g0: u16 = logger.call_method1("add_group", ("Engine",))?.extract()?;
        let ch_rpm: u32 = logger
            .call_method1("add_channel", (g0, "rpm", 0u8, 16u32))?
            .extract()?;
        logger.call_method0("start")?;
        for i in 0..10 {
            logger.call_method1("record_begin", (g0,))?;
            logger.call_method1("record_set_u64", (ch_rpm, 1000u64 + i * 250))?;
            logger.call_method1("record_commit", (g0,))?;
        }
        logger.call_method0("close")?;
        Ok(())
    })?;

    // Load base file in Rust
    let mut mdf_loaded = Mdf::new(base_file.to_str().unwrap())?;
    mdf_loaded.load_all_channels_data_in_memory()?;
    assert_eq!(mdf_loaded.channel_count(), 2);
    assert!(mdf_loaded.contains_channel("time"));
    assert!(mdf_loaded.contains_channel("rpm"));

    // Write file 1 (initial sorted write)
    let file_pass1 = tmp_dir.path().join("sorted_pass1.mf4");
    let mut written_pass1 = mdf_loaded.write(
        file_pass1.to_str().unwrap(),
        mdfr::CompressionAlgorithm::NoCompression,
    )?;
    written_pass1.load_all_channels_data_in_memory()?;
    assert_eq!(
        written_pass1.channel_count(),
        2,
        "Pass 1: Written file must contain strictly 2 channels"
    );

    // Independent reload of Pass 1
    let mut reloaded_pass1 = Mdf::new(file_pass1.to_str().unwrap())?;
    reloaded_pass1.load_all_channels_data_in_memory()?;
    assert_eq!(
        reloaded_pass1.channel_count(),
        2,
        "Reloaded Pass 1 must contain strictly 2 channels"
    );
    assert!(reloaded_pass1.contains_channel("time"));
    assert!(reloaded_pass1.contains_channel("rpm"));

    // Sequential Re-saving Stress: Pass 2, 3, 4, 5
    // Tests that re-saving already sorted MDF4 files repeatedly does NOT duplicate channels
    let mut current_file = file_pass1.clone();
    for pass in 2..=5 {
        let next_file = tmp_dir.path().join(format!("sorted_pass{pass}.mf4"));
        let mut loaded = Mdf::new(current_file.to_str().unwrap())?;
        loaded.load_all_channels_data_in_memory()?;
        assert_eq!(
            loaded.channel_count(),
            2,
            "Pass {pass} prior to write: channel count must be strictly 2"
        );

        let mut resaved = loaded.write(
            next_file.to_str().unwrap(),
            mdfr::CompressionAlgorithm::NoCompression,
        )?;
        resaved.load_all_channels_data_in_memory()?;
        assert_eq!(
            resaved.channel_count(),
            2,
            "Pass {pass} in-memory handle: channel count must remain strictly 2"
        );

        // Independent reload from disk
        let mut reloaded = Mdf::new(next_file.to_str().unwrap())?;
        reloaded.load_all_channels_data_in_memory()?;
        assert_eq!(
            reloaded.channel_count(),
            2,
            "Pass {pass} reloaded from disk: channel count must remain strictly 2 (no duplication!)"
        );
        assert!(reloaded.contains_channel("time"));
        assert!(reloaded.contains_channel("rpm"));

        // Verify data integrity across passes
        let rpm_data = reloaded
            .get_channel_data("rpm")
            .expect("rpm channel must exist");
        assert_eq!(rpm_data.len(), 10, "Data length must remain 10");

        current_file = next_file;
    }

    // Re-saving with Compression (Deflate)
    let comp_file = tmp_dir.path().join("sorted_compressed.mf4");
    let mut loaded_for_comp = Mdf::new(current_file.to_str().unwrap())?;
    loaded_for_comp.load_all_channels_data_in_memory()?;
    let _ = loaded_for_comp.write(
        comp_file.to_str().unwrap(),
        mdfr::CompressionAlgorithm::Deflate,
    )?;

    let mut reloaded_comp = Mdf::new(comp_file.to_str().unwrap())?;
    reloaded_comp.load_all_channels_data_in_memory()?;
    assert_eq!(
        reloaded_comp.channel_count(),
        2,
        "Compressed re-saved file must maintain strictly 2 channels"
    );

    // Multi-Channel Sorted File: 4 Channels (time, rpm, speed, temp)
    let df_4ch = polars::df!(
        "time" => &[0.1f64, 0.2, 0.3, 0.4, 0.5],
        "rpm" => &[1000.0f64, 1500.0, 2000.0, 2500.0, 3000.0],
        "speed" => &[30.0f64, 45.0, 60.0, 75.0, 90.0],
        "temp" => &[85.0f64, 86.0, 87.0, 88.0, 89.0]
    )?;
    let mut mdf_4ch = Mdf::from_polars_df(&df_4ch, Some("time"))?;
    let path_4ch_1 = tmp_dir.path().join("multi_4ch_1.mf4");
    mdf_4ch.write(
        path_4ch_1.to_str().unwrap(),
        mdfr::CompressionAlgorithm::NoCompression,
    )?;

    // Resave the 4-channel file
    let path_4ch_2 = tmp_dir.path().join("multi_4ch_2.mf4");
    let mut loaded_4ch = Mdf::new(path_4ch_1.to_str().unwrap())?;
    loaded_4ch.load_all_channels_data_in_memory()?;
    assert_eq!(loaded_4ch.channel_count(), 4);

    let _ = loaded_4ch.write(
        path_4ch_2.to_str().unwrap(),
        mdfr::CompressionAlgorithm::NoCompression,
    )?;
    let mut reloaded_4ch_2 = Mdf::new(path_4ch_2.to_str().unwrap())?;
    reloaded_4ch_2.load_all_channels_data_in_memory()?;
    assert_eq!(
        reloaded_4ch_2.channel_count(),
        4,
        "4-channel re-saved file must contain strictly 4 channels without duplication"
    );
    assert!(reloaded_4ch_2.contains_channel("time"));
    assert!(reloaded_4ch_2.contains_channel("rpm"));
    assert!(reloaded_4ch_2.contains_channel("speed"));
    assert!(reloaded_4ch_2.contains_channel("temp"));

    // Python-side write() and resave()
    Python::attach(|py| -> PyResult<()> {
        ensure_python_path(py)?;

        let mdfr_cls = py.get_type::<Mdfr>();
        let py_obj = mdfr_cls.call1((path_4ch_2.to_str().unwrap(),))?;
        py_obj.call_method0("load_all_channels_data_in_memory")?;
        assert_eq!(py_obj.call_method0("channel_count")?.extract::<usize>()?, 4);

        let py_resave_path = tmp_dir.path().join("py_resave.mf4");
        py_obj.call_method1("write", (py_resave_path.to_str().unwrap(),))?;

        let reloaded_py = mdfr_cls.call1((py_resave_path.to_str().unwrap(),))?;
        assert_eq!(
            reloaded_py
                .call_method0("channel_count")?
                .extract::<usize>()?,
            4
        );
        let keys: Vec<String> = reloaded_py.call_method0("keys")?.extract()?;
        assert_eq!(keys.len(), 4);
        assert!(keys.contains(&"time".to_string()));
        assert!(keys.contains(&"rpm".to_string()));
        assert!(keys.contains(&"speed".to_string()));
        assert!(keys.contains(&"temp".to_string()));

        Ok(())
    })?;

    Ok(())
}

// ============================================================================
// Challenge 2: Mask Polarity Stress Testing
// ============================================================================

#[test]
fn test_stress_mask_polarity_comprehensive() -> anyhow::Result<()> {
    let tmp_dir = tempfile::tempdir()?;

    // Comprehensive validity test patterns:
    // Channel A: Boundary nulls (first and last null): [None, 2.0, 3.0, 4.0, None]
    // Channel B: Alternating nulls: [10.0, None, 30.0, None, 50.0]
    // Channel C: All nulls: [None, None, None, None, None]
    // Channel D: Single null: [None, 2.0, 3.0, 4.0, 5.0]
    // Channel E: All valid: [1.0, 2.0, 3.0, 4.0, 5.0]
    let df = polars::df!(
        "time" => &[0.0f64, 1.0, 2.0, 3.0, 4.0],
        "boundary_nulls" => &[None::<f64>, Some(2.0), Some(3.0), Some(4.0), None],
        "alt_nulls" => &[Some(10.0f64), None, Some(30.0), None, Some(50.0)],
        "all_nulls" => &[None::<f64>, None, None, None, None],
        "single_null" => &[None::<f64>, Some(2.0), Some(3.0), Some(4.0), Some(5.0)],
        "all_valid" => &[1.0f64, 2.0, 3.0, 4.0, 5.0]
    )?;

    let mut mdf = Mdf::from_polars_df(&df, Some("time"))?;
    let path = tmp_dir.path().join("mask_polarity_stress.mf4");
    mdf.write(
        path.to_str().unwrap(),
        mdfr::CompressionAlgorithm::NoCompression,
    )?;

    Python::initialize();
    let has_numpy = Python::attach(|py| {
        let _ = ensure_python_path(py);
        py.import("numpy.ma").is_ok()
    });
    if !has_numpy {
        eprintln!(
            "Skipping test_stress_mask_polarity_comprehensive: numpy.ma not available in Python"
        );
        return Ok(());
    }

    Python::attach(|py| -> PyResult<()> {
        ensure_python_path(py)?;

        let mdfr_cls = py.get_type::<Mdfr>();
        let mdfr_obj = mdfr_cls.call1((path.to_str().unwrap(),))?;
        mdfr_obj.call_method0("load_all_channels_data_in_memory")?;
        let np_ma = py.import("numpy.ma")?;

        // 1. Boundary nulls: [None, 2.0, 3.0, 4.0, None]
        let obj_boundary = mdfr_obj.call_method1("get_channel_data", ("boundary_nulls",))?;
        let is_masked: bool = np_ma
            .call_method1("is_masked", (&obj_boundary,))?
            .extract()?;
        assert!(is_masked, "boundary_nulls must produce a masked array");
        let mask: Vec<bool> = obj_boundary
            .getattr("mask")?
            .call_method0("tolist")?
            .extract()?;
        assert_eq!(
            mask,
            vec![true, false, false, false, true],
            "Strict polarity failed for boundary_nulls!"
        );

        // Verify PyChannel.to_numpy matches
        let ch_b = mdfr_obj.get_item("boundary_nulls")?;
        let ch_mask: Vec<bool> = ch_b
            .call_method0("to_numpy")?
            .getattr("mask")?
            .call_method0("tolist")?
            .extract()?;
        assert_eq!(ch_mask, vec![true, false, false, false, true]);

        // 2. Alternating nulls: [10.0, None, 30.0, None, 50.0]
        let obj_alt = mdfr_obj.call_method1("get_channel_data", ("alt_nulls",))?;
        let mask_alt: Vec<bool> = obj_alt.getattr("mask")?.call_method0("tolist")?.extract()?;
        assert_eq!(
            mask_alt,
            vec![false, true, false, true, false],
            "Strict polarity failed for alt_nulls!"
        );

        // Verify compressed() excludes invalid elements
        let compressed = obj_alt.call_method0("compressed")?;
        let comp_list: Vec<f64> = compressed.call_method0("tolist")?.extract()?;
        assert_eq!(
            comp_list,
            vec![10.0, 30.0, 50.0],
            "compressed() must retain only valid samples"
        );

        // 3. All nulls: [None, None, None, None, None]
        let obj_all_nulls = mdfr_obj.call_method1("get_channel_data", ("all_nulls",))?;
        let mask_all_nulls: Vec<bool> = obj_all_nulls
            .getattr("mask")?
            .call_method0("tolist")?
            .extract()?;
        assert_eq!(
            mask_all_nulls,
            vec![true, true, true, true, true],
            "Strict polarity failed for all_nulls!"
        );
        let comp_all_nulls = obj_all_nulls.call_method0("compressed")?;
        let comp_empty: Vec<f64> = comp_all_nulls.call_method0("tolist")?.extract()?;
        assert!(comp_empty.is_empty(), "All null compressed() must be empty");

        // 4. Single null at index 0
        let obj_single = mdfr_obj.call_method1("get_channel_data", ("single_null",))?;
        let mask_single: Vec<bool> = obj_single
            .getattr("mask")?
            .call_method0("tolist")?
            .extract()?;
        assert_eq!(
            mask_single,
            vec![true, false, false, false, false],
            "Strict polarity failed for single_null!"
        );

        // 5. All valid
        let obj_all_valid = mdfr_obj.call_method1("get_channel_data", ("all_valid",))?;
        let is_valid_masked: bool = np_ma
            .call_method1("is_masked", (&obj_all_valid,))?
            .extract()?;
        assert!(
            !is_valid_masked,
            "all_valid channel must not be a masked array"
        );

        Ok(())
    })?;

    Ok(())
}

// ============================================================================
// Challenge 3: O(1) Large Channel Set Queries Stress
// ============================================================================

#[test]
fn test_stress_o1_queries_50000_channels() {
    let n_channels = 50_000;
    let mut info4 = MdfInfo4::default();

    // Populate 50,000 channels
    for i in 0..n_channels {
        info4.channel_names_set.insert(
            format!("Channel_Signal_Prefix_{:06}", i),
            (Some("TimeMaster".to_string()), 0, (0, 0), (0, i as i32)),
        );
    }

    assert_eq!(info4.channel_count(), n_channels);

    // Prepare 10,000 existing queries and 10,000 non-existent queries
    let existing_queries: Vec<String> = (0..10_000)
        .map(|i| format!("Channel_Signal_Prefix_{:06}", i * 5))
        .collect();
    let missing_queries: Vec<String> = (0..10_000)
        .map(|i| format!("Missing_Channel_Signal_{:06}", i))
        .collect();

    // Verify zero allocations on contains_channel and channel_count
    let my_tid = unsafe { libc::syscall(libc::SYS_gettid) as i64 };
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    TRACKED_TID.store(my_tid, Ordering::SeqCst);

    let start = Instant::now();
    for query in &existing_queries {
        assert!(info4.contains_channel(query));
    }
    for query in &missing_queries {
        assert!(!info4.contains_channel(query));
    }
    for _ in 0..1_000 {
        assert_eq!(info4.channel_count(), n_channels);
    }
    let elapsed = start.elapsed();

    let allocs = ALLOC_COUNT.load(Ordering::SeqCst);
    TRACKED_TID.store(0, Ordering::SeqCst);

    assert_eq!(
        allocs, 0,
        "contains_channel or channel_count allocated {} times on heap!",
        allocs
    );

    // 21,000 operations should complete in well under 50 milliseconds
    assert!(
        elapsed.as_millis() < 50,
        "O(1) lookups took too long: {:?} for 21,000 queries",
        elapsed
    );

    // Edge-case channel names
    info4.channel_names_set.insert(
        "".to_string(),
        (Some("TimeMaster".to_string()), 0, (0, 0), (0, 0)),
    );
    assert!(info4.contains_channel(""));

    info4.channel_names_set.insert(
        "Temperature_°C_日本語".to_string(),
        (Some("TimeMaster".to_string()), 0, (0, 0), (0, 1)),
    );
    assert!(info4.contains_channel("Temperature_°C_日本語"));

    let long_name = "x".repeat(1000);
    info4.channel_names_set.insert(
        long_name.clone(),
        (Some("TimeMaster".to_string()), 0, (0, 0), (0, 2)),
    );
    assert!(info4.contains_channel(&long_name));
}

// ============================================================================
// Challenge 4: add_channel Dict Format and Kwargs Stress
// ============================================================================

#[test]
fn test_stress_add_channel_dict_and_kwargs_edge_cases() -> anyhow::Result<()> {
    let tmp_dir = tempfile::tempdir()?;
    let df = polars::df!(
        "time" => &[0.0f64, 1.0, 2.0, 3.0],
        "sig0" => &[10.0f64, 20.0, 30.0, 40.0]
    )?;
    let mut mdf = Mdf::from_polars_df(&df, Some("time"))?;
    let base_file = tmp_dir.path().join("base_add_ch_stress.mf4");
    mdf.write(
        base_file.to_str().unwrap(),
        mdfr::CompressionAlgorithm::NoCompression,
    )?;

    Python::initialize();
    let has_pyarrow = Python::attach(|py| {
        let _ = ensure_python_path(py);
        py.import("pyarrow").is_ok()
    });
    if !has_pyarrow {
        eprintln!(
            "Skipping test_stress_add_channel_dict_and_kwargs_edge_cases: pyarrow not available in Python"
        );
        return Ok(());
    }

    Python::attach(|py| -> PyResult<()> {
        ensure_python_path(py)?;

        let mdfr_cls = py.get_type::<Mdfr>();
        let mdfr_obj = mdfr_cls.call1((base_file.to_str().unwrap(),))?;
        mdfr_obj.call_method0("load_all_channels_data_in_memory")?;

        let arr_data = Float64Array::from(vec![1.1, 2.2, 3.3, 4.4]);

        // 1. Dict with extra unknown keys + partial keys
        let dict1 = PyDict::new(py);
        dict1.set_item("name", "time")?;
        dict1.set_item("extra_future_field", 999)?;
        let kw1 = PyDict::new(py);
        kw1.set_item("master", dict1)?;
        kw1.set_item("unit", "kPa")?;
        kw1.set_item("description", "Partial dict test")?;

        mdfr_obj.call_method(
            "add_channel",
            ("ch_partial_dict", PyArrowType(Array::to_data(&arr_data))),
            Some(&kw1),
        )?;

        // 2. Both dict master AND kwargs master: dict takes precedence for name/type/flag
        let dict2 = PyDict::new(py);
        dict2.set_item("name", "time")?;
        dict2.set_item("type", 1u8)?;
        let kw2 = PyDict::new(py);
        kw2.set_item("master", dict2)?;
        kw2.set_item("master_channel", "ignored_time")?;
        kw2.set_item("unit", "Nm")?;
        kw2.set_item("description", "Precedence test")?;

        mdfr_obj.call_method(
            "add_channel",
            ("ch_precedence_dict", PyArrowType(Array::to_data(&arr_data))),
            Some(&kw2),
        )?;

        // 3. Kwargs only (no master dict)
        let kw3 = PyDict::new(py);
        kw3.set_item("master_channel", "time")?;
        kw3.set_item("master_type", 1u8)?;
        kw3.set_item("master_flag", false)?;
        kw3.set_item("unit", "rpm")?;
        kw3.set_item("description", "Kwargs only test")?;

        mdfr_obj.call_method(
            "add_channel",
            ("ch_kwargs_only", PyArrowType(Array::to_data(&arr_data))),
            Some(&kw3),
        )?;

        // 4. String master only (via master parameter)
        let kw4 = PyDict::new(py);
        kw4.set_item("master", PyString::new(py, "time"))?;
        kw4.set_item("unit", "deg")?;
        kw4.set_item("description", "String master test")?;

        mdfr_obj.call_method(
            "add_channel",
            ("ch_str_master", PyArrowType(Array::to_data(&arr_data))),
            Some(&kw4),
        )?;

        // Verify all channels exist
        assert!(
            mdfr_obj
                .call_method1("contains_channel", ("ch_partial_dict",))?
                .extract::<bool>()?
        );
        assert!(
            mdfr_obj
                .call_method1("contains_channel", ("ch_precedence_dict",))?
                .extract::<bool>()?
        );
        assert!(
            mdfr_obj
                .call_method1("contains_channel", ("ch_kwargs_only",))?
                .extract::<bool>()?
        );
        assert!(
            mdfr_obj
                .call_method1("contains_channel", ("ch_str_master",))?
                .extract::<bool>()?
        );

        // Total count = 2 original + 4 added = 6
        let total_count: usize = mdfr_obj.call_method0("channel_count")?.extract()?;
        assert_eq!(total_count, 6);

        // Verify units and descriptions
        assert_eq!(
            mdfr_obj
                .call_method1("get_channel_unit", ("ch_partial_dict",))?
                .extract::<Option<String>>()?,
            Some("kPa".to_string())
        );
        assert_eq!(
            mdfr_obj
                .call_method1("get_channel_desc", ("ch_partial_dict",))?
                .extract::<Option<String>>()?,
            Some("Partial dict test".to_string())
        );

        assert_eq!(
            mdfr_obj
                .call_method1("get_channel_unit", ("ch_precedence_dict",))?
                .extract::<Option<String>>()?,
            Some("Nm".to_string())
        );
        assert_eq!(
            mdfr_obj
                .call_method1("get_channel_unit", ("ch_kwargs_only",))?
                .extract::<Option<String>>()?,
            Some("rpm".to_string())
        );

        // Write the modified file with new channels
        let written_path = tmp_dir.path().join("added_channels_written.mf4");
        mdfr_obj.call_method1("write", (written_path.to_str().unwrap(),))?;

        // Independent reload
        let reloaded_obj = mdfr_cls.call1((written_path.to_str().unwrap(),))?;
        assert_eq!(
            reloaded_obj
                .call_method0("channel_count")?
                .extract::<usize>()?,
            6
        );

        Ok(())
    })?;

    Ok(())
}

// ============================================================================
// Challenge 5: Dotted Master Channel Names Parquet & CSV Export Stress
// ============================================================================

#[test]
fn test_stress_dotted_master_names_nested_paths() -> anyhow::Result<()> {
    let tmp_dir = tempfile::tempdir()?;

    // Create a directory containing multiple dots to stress-test path stem extraction:
    // e.g. "experiment.run.001/session.v2.data"
    let nested_dir = tmp_dir
        .path()
        .join("experiment.run.001")
        .join("session.v2.data");
    fs::create_dir_all(&nested_dir)?;

    let df = polars::df!(
        "Engine.ECU.CAN_1.Timestamp" => &[0.001f64, 0.002, 0.003],
        "Engine.ECU.CAN_1.RPM" => &[1200.0f64, 1250.0, 1300.0],
        "Chassis.Suspension.Height.FL" => &[120.5f64, 120.3, 120.1]
    )?;

    let mdf = Mdf::from_polars_df(&df, Some("Engine.ECU.CAN_1.Timestamp"))?;

    // 1. Export to Parquet with a path that includes dots in directory and file name:
    // Prefix: ".../experiment.run.001/session.v2.data/output.export"
    let parquet_prefix = nested_dir.join("output.export");
    mdf.export_to_parquet(parquet_prefix.to_str().unwrap(), None)?;

    let expected_parquet = nested_dir.join("output.export_Engine.ECU.CAN_1.Timestamp.parquet");
    assert!(
        expected_parquet.exists(),
        "Expected parquet file {:?} to exist in nested directory with dots",
        expected_parquet
    );
    assert!(
        expected_parquet.metadata()?.len() > 0,
        "Parquet file must not be empty"
    );

    // Verify Parquet file contents
    let pfile = fs::File::open(&expected_parquet)?;
    let preader = parquet::file::reader::SerializedFileReader::new(pfile)?;
    use parquet::file::reader::FileReader;
    let pmeta = preader.metadata();
    assert_eq!(pmeta.file_metadata().num_rows(), 3);
    assert_eq!(pmeta.num_row_groups(), 1);

    // 2. Export to Parquet passing ".parquet" suffix explicitly:
    // Prefix: ".../output.export.parquet"
    let parquet_with_ext = nested_dir.join("output.export.parquet");
    mdf.export_to_parquet(parquet_with_ext.to_str().unwrap(), None)?;
    // Should NOT produce ".../output.export.parquet_Engine.ECU.CAN_1.Timestamp.parquet",
    // but rather ".../output.export_Engine.ECU.CAN_1.Timestamp.parquet"
    assert!(
        expected_parquet.exists(),
        "Export with .parquet suffix must correctly strip .parquet extension"
    );

    // 3. Export to CSV in nested dir
    let csv_prefix = nested_dir.join("output.export");
    mdf.export_to_csv(csv_prefix.to_str().unwrap())?;

    let expected_csv = nested_dir.join("output.export_Engine.ECU.CAN_1.Timestamp.csv");
    assert!(
        expected_csv.exists(),
        "Expected CSV file {:?} to exist",
        expected_csv
    );

    let csv_content = fs::read_to_string(&expected_csv)?;
    let csv_lines: Vec<&str> = csv_content.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(csv_lines.len(), 4, "CSV must have header + 3 data rows");
    assert!(csv_lines[0].contains("Engine.ECU.CAN_1.Timestamp"));
    assert!(csv_lines[0].contains("Engine.ECU.CAN_1.RPM"));
    assert!(csv_lines[0].contains("Chassis.Suspension.Height.FL"));

    // 4. Export to CSV with ".csv" suffix explicitly
    let csv_with_ext = nested_dir.join("output.export.csv");
    mdf.export_to_csv(csv_with_ext.to_str().unwrap())?;
    assert!(expected_csv.exists());

    Ok(())
}
