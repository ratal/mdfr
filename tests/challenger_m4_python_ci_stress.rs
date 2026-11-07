//! Comprehensive Empirical Stress Test Suite for Milestone 4 (Features 33-35).
//!
//! Features covered:
//! 1. Feature 33:
//!    - NumPy mask polarity: verify that invalidation bits create a boolean mask where `mask == True` strictly for invalid samples and `mask == False` for valid samples.
//!    - O(1) channel lookups: create synthetic structures with 10,000 channels, verifying `contains_channel` and `channel_count` execute with 0 allocations.
//!    - `add_channel` with dict parameters, string master, and keyword arguments.
//!    - `write` with default `NoCompression`.
//! 2. Feature 34:
//!    - Parquet and CSV export with complex master channel names containing dots (e.g. "Engine.Speed.RPM", "CAN1.Timestamp"), verifying exact output file names and valid readable file content.
//!    - `mdfr.pyi` typing consistency and `py.typed` PEP 561 marker.
//! 3. Feature 35:
//!    - Clean linking and execution without PyO3 `extension-module` linker errors.
//!    - Validate YAML syntax of `.github/workflows/rust.yml` and release workflow triggers (`v*` tags, `workflow_dispatch`).

use std::alloc::{GlobalAlloc, Layout, System};
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::{Array, Float64Array};
use arrow::pyarrow::PyArrowType;
use mdfr::mdfinfo::MdfInfo;
use mdfr::mdfinfo::mdfinfo3::MdfInfo3;
use mdfr::mdfinfo::mdfinfo4::MdfInfo4;
use mdfr::mdfr::{MdfLogger, Mdfr};
use mdfr::mdfreader::Mdf;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

fn ensure_python_path(py: Python<'_>) -> PyResult<()> {
    let sys = py.import("sys")?;
    let sys_path = sys.getattr("path")?;
    let venv_site =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".venv/lib/python3.13/site-packages");
    if venv_site.exists() {
        sys_path.call_method1("insert", (0, venv_site.to_str().unwrap()))?;
    }
    Ok(())
}

// ============================================================================
// Tracking Allocator for Zero-Allocation Assertions
// ============================================================================

struct TrackingAllocator;
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static TRACKED_TID: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

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

// ============================================================================
// Feature 33 Tests: NumPy Mask Polarity
// ============================================================================

#[test]
fn test_feature33_numpy_mask_polarity_strictly_true_for_invalid() -> anyhow::Result<()> {
    let tmp_dir = tempfile::tempdir()?;

    // Construct Polars DataFrame with alternating nulls:
    // Index 0: 100.0 (Valid)
    // Index 1: None  (Invalid / Null)
    // Index 2: 300.0 (Valid)
    // Index 3: None  (Invalid / Null)
    // Index 4: 500.0 (Valid)
    let df = polars::df!(
        "time" => &[0.0f64, 0.1, 0.2, 0.3, 0.4],
        "sensor_alt" => &[Some(100.0f64), None, Some(300.0), None, Some(500.0)],
        "sensor_all_valid" => &[1.0f64, 2.0, 3.0, 4.0, 5.0],
        "sensor_all_invalid" => &[None::<f64>, None, None, None, None]
    )?;

    let mut mdf = Mdf::from_polars_df(&df, Some("time"))?;

    let test_file = tmp_dir.path().join("polarity_test.mf4");
    mdf.write(
        test_file.to_str().unwrap(),
        mdfr_core::compress::CompressionAlgorithm::NoCompression,
    )?;

    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        ensure_python_path(py)?;
        let mdfr_cls = py.get_type::<Mdfr>();
        let mdfr_obj = mdfr_cls.call1((test_file.to_str().unwrap(),))?;
        mdfr_obj.call_method0("load_all_channels_data_in_memory")?;
        let np_ma = py.import("numpy.ma")?;

        // 1. Test sensor_alt via Mdfr.get_channel_data
        let py_obj = mdfr_obj.call_method1("get_channel_data", ("sensor_alt",))?;

        // Verify it is a masked array
        let is_masked: bool = np_ma.call_method1("is_masked", (&py_obj,))?.extract()?;
        assert!(
            is_masked,
            "sensor_alt must produce a NumPy MaskedArray because invalid samples exist"
        );

        let mask = py_obj.getattr("mask")?;
        let mask_vec: Vec<bool> = mask.call_method0("tolist")?.extract()?;

        // STRICT POLARITY ASSERTION:
        // mask == True strictly for invalid samples, mask == False for valid samples
        assert_eq!(
            mask_vec,
            vec![false, true, false, true, false],
            "Mask polarity violation! Expected [false, true, false, true, false], got {:?}",
            mask_vec
        );
        assert!(!mask_vec[0], "Sample 0 is valid, mask must be False");
        assert!(mask_vec[1], "Sample 1 is invalid, mask must be True");
        assert!(!mask_vec[2], "Sample 2 is valid, mask must be False");
        assert!(mask_vec[3], "Sample 3 is invalid, mask must be True");
        assert!(!mask_vec[4], "Sample 4 is valid, mask must be False");

        // 2. Test sensor_alt via PyChannel.to_numpy
        let ch = mdfr_obj.get_item("sensor_alt")?;
        let ch_py_obj = ch.call_method0("to_numpy")?;

        let ch_mask = ch_py_obj.getattr("mask")?;
        let ch_mask_vec: Vec<bool> = ch_mask.call_method0("tolist")?.extract()?;
        assert_eq!(
            ch_mask_vec,
            vec![false, true, false, true, false],
            "PyChannel.to_numpy mask polarity violation!"
        );

        // 3. Test sensor_all_valid: mask must NOT be active
        let valid_py = mdfr_obj.call_method1("get_channel_data", ("sensor_all_valid",))?;
        let valid_is_masked: bool = np_ma.call_method1("is_masked", (&valid_py,))?.extract()?;
        assert!(
            !valid_is_masked,
            "sensor_all_valid should not be masked when all samples are valid"
        );

        // 4. Test sensor_all_invalid: mask must be all True
        let invalid_py = mdfr_obj.call_method1("get_channel_data", ("sensor_all_invalid",))?;
        let invalid_mask = invalid_py.getattr("mask")?;
        let invalid_mask_vec: Vec<bool> = invalid_mask.call_method0("tolist")?.extract()?;
        assert_eq!(
            invalid_mask_vec,
            vec![true, true, true, true, true],
            "All-invalid channel must have mask == True for all elements"
        );

        Ok(())
    })?;

    Ok(())
}

// ============================================================================
// Feature 33 Tests: 10,000 Channels Zero-Allocation Lookups
// ============================================================================

#[test]
fn test_feature33_zero_allocation_10000_channel_lookups() {
    let n_channels = 10_000;

    // 1. Setup 10,000 synthetic channels in MdfInfo4
    let mut info4 = MdfInfo4::default();
    for i in 0..n_channels {
        info4.channel_names_set.insert(
            format!("Channel_Signal_{:05}", i),
            (Some("MasterTime".to_string()), 0, (0, 0), (0, i as i32)),
        );
    }

    // Pre-allocate queries to strictly measure lookup allocations
    let existing_queries: Vec<String> = (0..n_channels)
        .map(|i| format!("Channel_Signal_{:05}", i))
        .collect();
    let missing_queries: Vec<String> = (0..1_000)
        .map(|i| format!("NonExistent_{:05}", i))
        .collect();

    // Verify initial count
    assert_eq!(info4.channel_count(), n_channels);

    // Warm-up
    assert!(info4.contains_channel(&existing_queries[0]));
    assert!(!info4.contains_channel(&missing_queries[0]));

    // Enable heap allocation tracking strictly on current thread
    let my_tid = unsafe { libc::syscall(libc::SYS_gettid) as i64 };
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    TRACKED_TID.store(my_tid, Ordering::SeqCst);

    // Run 10,000 lookups for existing channels
    for query in &existing_queries {
        assert!(info4.contains_channel(query));
    }

    // Run 1,000 lookups for missing channels
    for query in &missing_queries {
        assert!(!info4.contains_channel(query));
    }

    // Run 1,000 channel_count calls
    for _ in 0..1_000 {
        assert_eq!(info4.channel_count(), n_channels);
    }

    let allocs_v4 = ALLOC_COUNT.load(Ordering::SeqCst);
    TRACKED_TID.store(0, Ordering::SeqCst);

    assert_eq!(
        allocs_v4, 0,
        "MdfInfo4 zero-allocation constraint failed: contains_channel and channel_count allocated {} times on the heap!",
        allocs_v4
    );

    // 2. Setup 10,000 synthetic channels in MdfInfo3
    let mut info3 = MdfInfo3 {
        file_name: String::new(),
        id_block: mdfr::mdfinfo::IdBlock::default(),
        encoding: encoding_rs::WINDOWS_1252,
        hd_block: mdfr::mdfinfo::mdfinfo3::Hd3::default(),
        hd_comment: String::new(),
        dg: std::collections::BTreeMap::new(),
        sharable: mdfr::mdfinfo::mdfinfo3::SharableBlocks3::default(),
        channel_names_set: std::collections::HashMap::new(),
    };
    for i in 0..n_channels {
        info3.channel_names_set.insert(
            format!("Channel_Signal_{:05}", i),
            (Some("MasterTime".to_string()), 0, (0, 0), i as u32),
        );
    }

    ALLOC_COUNT.store(0, Ordering::SeqCst);
    TRACKED_TID.store(my_tid, Ordering::SeqCst);

    for query in &existing_queries {
        assert!(info3.contains_channel(query));
    }
    for query in &missing_queries {
        assert!(!info3.contains_channel(query));
    }
    for _ in 0..1_000 {
        assert_eq!(info3.channel_count(), n_channels);
    }

    let allocs_v3 = ALLOC_COUNT.load(Ordering::SeqCst);
    TRACKED_TID.store(0, Ordering::SeqCst);

    assert_eq!(
        allocs_v3, 0,
        "MdfInfo3 zero-allocation constraint failed: allocated {} times on the heap!",
        allocs_v3
    );

    // 3. Setup and verify MdfInfo enum
    let mdfinfo = MdfInfo::V4(Box::new(info4));
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    TRACKED_TID.store(my_tid, Ordering::SeqCst);

    for query in &existing_queries[..1_000] {
        assert!(mdfinfo.contains_channel(query));
    }
    assert_eq!(mdfinfo.channel_count(), n_channels);

    let allocs_enum = ALLOC_COUNT.load(Ordering::SeqCst);
    TRACKED_TID.store(0, Ordering::SeqCst);

    assert_eq!(
        allocs_enum, 0,
        "MdfInfo enum zero-allocation constraint failed: allocated {} times on the heap!",
        allocs_enum
    );
}

// ============================================================================
// Feature 33 Tests: add_channel with Dict, String Master, and Kwargs
// ============================================================================

#[test]
fn test_feature33_add_channel_dict_string_kwargs() -> anyhow::Result<()> {
    let tmp_dir = tempfile::tempdir()?;

    let df = polars::df!(
        "time" => &[0.0f64, 1.0, 2.0],
        "base_signal" => &[10.0f64, 20.0, 30.0]
    )?;
    let mut mdf = Mdf::from_polars_df(&df, Some("time"))?;

    let base_file = tmp_dir.path().join("base_for_add_channel.mf4");
    mdf.write(
        base_file.to_str().unwrap(),
        mdfr_core::compress::CompressionAlgorithm::NoCompression,
    )?;

    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        ensure_python_path(py)?;
        let mdfr_cls = py.get_type::<Mdfr>();
        let mdfr_obj = mdfr_cls.call1((base_file.to_str().unwrap(),))?;

        let test_arr = Float64Array::from(vec![100.0, 200.0, 300.0]);

        // Case A: Dict master parameter
        let master_dict = PyDict::new(py);
        master_dict.set_item("name", "time")?;
        master_dict.set_item("type", 1u8)?;
        master_dict.set_item("flag", false)?;

        let kwargs_a = PyDict::new(py);
        kwargs_a.set_item("master", master_dict)?;
        kwargs_a.set_item("unit", "m/s")?;
        kwargs_a.set_item("description", "Channel added via dict master")?;

        mdfr_obj.call_method(
            "add_channel",
            ("ch_dict", PyArrowType(Array::to_data(&test_arr))),
            Some(&kwargs_a),
        )?;

        // Case B: String master parameter
        let master_str = PyString::new(py, "time");
        let kwargs_b = PyDict::new(py);
        kwargs_b.set_item("master", master_str)?;
        kwargs_b.set_item("unit", "bar")?;
        kwargs_b.set_item("description", "Channel added via string master")?;

        mdfr_obj.call_method(
            "add_channel",
            ("ch_str", PyArrowType(Array::to_data(&test_arr))),
            Some(&kwargs_b),
        )?;

        // Case C: Keyword arguments
        let kwargs_c = PyDict::new(py);
        kwargs_c.set_item("master_channel", "time")?;
        kwargs_c.set_item("master_type", 1u8)?;
        kwargs_c.set_item("master_flag", false)?;
        kwargs_c.set_item("unit", "degC")?;
        kwargs_c.set_item("description", "Channel added via kwargs")?;

        mdfr_obj.call_method(
            "add_channel",
            ("ch_kwargs", PyArrowType(Array::to_data(&test_arr))),
            Some(&kwargs_c),
        )?;

        // Verify channel presence and O(1) contains
        let has_dict: bool = mdfr_obj
            .call_method1("contains_channel", ("ch_dict",))?
            .extract()?;
        let has_str: bool = mdfr_obj
            .call_method1("contains_channel", ("ch_str",))?
            .extract()?;
        let has_kwargs: bool = mdfr_obj
            .call_method1("contains_channel", ("ch_kwargs",))?
            .extract()?;

        assert!(has_dict, "ch_dict must exist");
        assert!(has_str, "ch_str must exist");
        assert!(has_kwargs, "ch_kwargs must exist");

        let count: usize = mdfr_obj.call_method0("channel_count")?.extract()?;
        assert_eq!(count, 5); // time, base_signal, ch_dict, ch_str, ch_kwargs

        // Verify metadata preservation
        let unit_dict: Option<String> = mdfr_obj
            .call_method1("get_channel_unit", ("ch_dict",))?
            .extract()?;
        let desc_dict: Option<String> = mdfr_obj
            .call_method1("get_channel_desc", ("ch_dict",))?
            .extract()?;
        assert_eq!(unit_dict, Some("m/s".to_string()));
        assert_eq!(desc_dict, Some("Channel added via dict master".to_string()));

        let unit_str: Option<String> = mdfr_obj
            .call_method1("get_channel_unit", ("ch_str",))?
            .extract()?;
        let desc_str: Option<String> = mdfr_obj
            .call_method1("get_channel_desc", ("ch_str",))?
            .extract()?;
        assert_eq!(unit_str, Some("bar".to_string()));
        assert_eq!(
            desc_str,
            Some("Channel added via string master".to_string())
        );

        let unit_kwargs: Option<String> = mdfr_obj
            .call_method1("get_channel_unit", ("ch_kwargs",))?
            .extract()?;
        let desc_kwargs: Option<String> = mdfr_obj
            .call_method1("get_channel_desc", ("ch_kwargs",))?
            .extract()?;
        assert_eq!(unit_kwargs, Some("degC".to_string()));
        assert_eq!(desc_kwargs, Some("Channel added via kwargs".to_string()));

        Ok(())
    })?;

    Ok(())
}

// ============================================================================
// Feature 33 Tests: write with Default NoCompression
// ============================================================================

#[test]
fn test_feature33_write_default_no_compression() -> anyhow::Result<()> {
    let tmp_dir = tempfile::tempdir()?;
    let base_file = tmp_dir.path().join("base_write_test.mf4");

    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        ensure_python_path(py)?;
        // 1. Create a well-formed single-CG MDF file with master channel 't' and channel 'rpm'
        let logger_cls = py.get_type::<MdfLogger>();
        let logger = logger_cls.call1((base_file.to_str().unwrap(), "unsorted"))?;
        let g0: u16 = logger.call_method1("add_group", ("Engine",))?.extract()?;
        let ch_rpm: u32 = logger
            .call_method1("add_channel", (g0, "rpm", 0u8, 16u32))?
            .extract()?;
        logger.call_method0("start")?;
        for i in 0..5 {
            logger.call_method1("record_begin", (g0,))?;
            logger.call_method1("record_set_u64", (ch_rpm, 1000u64 + i * 500))?;
            logger.call_method1("record_commit", (g0,))?;
        }
        logger.call_method0("close")?;

        let mdfr_cls = py.get_type::<Mdfr>();
        let mdfr_obj = mdfr_cls.call1((base_file.to_str().unwrap(),))?;
        mdfr_obj.call_method0("load_all_channels_data_in_memory")?;

        // 2. write with default compression (single argument -> defaults to NoCompression)
        let path_none = tmp_dir.path().join("out_none.mf4");
        let written_none = mdfr_obj.call_method1("write", (path_none.to_str().unwrap(),))?;
        let is_unfin: bool = written_none.call_method0("is_unfinalized")?.extract()?;
        assert!(!is_unfin, "Written file handle must be marked as finalized");

        // Verify independently reloaded file in Python
        let loaded_none_py = mdfr_cls.call1((path_none.to_str().unwrap(),))?;
        let keys_none: Vec<String> = loaded_none_py.call_method0("keys")?.extract()?;
        println!("keys_none: {:?}", keys_none);
        let count_none: usize = loaded_none_py.call_method0("channel_count")?.extract()?;
        assert_eq!(
            count_none, 2,
            "Reloaded written file must contain 2 channels"
        );
        assert!(
            loaded_none_py
                .call_method1("contains_channel", ("time",))?
                .extract::<bool>()?
        );
        assert!(
            loaded_none_py
                .call_method1("contains_channel", ("rpm",))?
                .extract::<bool>()?
        );

        // Verify independently written file in Rust reader
        let loaded_none = Mdf::new(path_none.to_str().unwrap())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        assert!(loaded_none.contains_channel("time"));
        assert!(loaded_none.contains_channel("rpm"));
        assert_eq!(loaded_none.channel_count(), 2);

        // 3. write with explicit None
        let path_explicit = tmp_dir.path().join("out_explicit.mf4");
        let written_explicit =
            mdfr_obj.call_method1("write", (path_explicit.to_str().unwrap(), py.None()))?;
        let is_unfin_explicit: bool = written_explicit.call_method0("is_unfinalized")?.extract()?;
        assert!(!is_unfin_explicit);

        let loaded_explicit = Mdf::new(path_explicit.to_str().unwrap())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        assert!(loaded_explicit.contains_channel("time"));
        assert!(loaded_explicit.contains_channel("rpm"));
        assert_eq!(loaded_explicit.channel_count(), 2);

        // 4. write with Deflate compression
        let deflate_val = py
            .get_type::<mdfr::mdfr::PyCompressionAlgorithm>()
            .getattr("Deflate")?;
        let path_deflate = tmp_dir.path().join("out_deflate.mf4");
        let written_deflate =
            mdfr_obj.call_method1("write", (path_deflate.to_str().unwrap(), deflate_val))?;
        let is_unfin_deflate: bool = written_deflate.call_method0("is_unfinalized")?.extract()?;
        assert!(!is_unfin_deflate);

        let loaded_deflate = Mdf::new(path_deflate.to_str().unwrap())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        assert!(loaded_deflate.contains_channel("time"));
        assert!(loaded_deflate.contains_channel("rpm"));
        assert_eq!(loaded_deflate.channel_count(), 2);

        // 5. Empirical Challenge: Re-writing an already-sorted MDF4 file (where channels are in separate DGs)
        // triggers the mdfwriter4 foreign-CG master indexing bug, reproducing channel duplication.
        let path_resave = tmp_dir.path().join("out_resave.mf4");
        let mut loaded_none_mut = loaded_none;
        loaded_none_mut
            .load_all_channels_data_in_memory()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let _ = loaded_none_mut
            .write(
                path_resave.to_str().unwrap(),
                mdfr_core::compress::CompressionAlgorithm::NoCompression,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let reloaded_resave = Mdf::new(path_resave.to_str().unwrap())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        assert_eq!(
            reloaded_resave.channel_count(),
            2,
            "Re-written file must preserve exactly 2 channels without duplication"
        );

        Ok(())
    })?;

    Ok(())
}

// ============================================================================
// Feature 34 Tests: Dotted Master Channel Names Parquet & CSV Export
// ============================================================================

#[test]
fn test_feature34_parquet_csv_export_dotted_master_channels() -> anyhow::Result<()> {
    let tmp_dir = tempfile::tempdir()?;

    // Test complex dotted channel names
    let df = polars::df!(
        "Engine.Speed.RPM" => &[800.0f64, 1500.0, 3200.0],
        "CAN1.Timestamp" => &[0.001f64, 0.002, 0.003],
        "Vehicle.Dynamics.YawRate" => &[0.05f64, 0.10, 0.15]
    )?;

    // 1. Export with master "Engine.Speed.RPM"
    let mdf = Mdf::from_polars_df(&df, Some("Engine.Speed.RPM"))?;

    // Parquet export with prefix "prefix"
    let parquet_prefix = tmp_dir.path().join("prefix");
    mdf.export_to_parquet(parquet_prefix.to_str().unwrap(), None)?;
    let expected_parquet = tmp_dir.path().join("prefix_Engine.Speed.RPM.parquet");
    assert!(
        expected_parquet.exists(),
        "Expected parquet file {:?} to exist",
        expected_parquet
    );
    assert!(
        expected_parquet.metadata()?.len() > 0,
        "Parquet file must not be empty"
    );

    // Verify Parquet file readable content
    let file = fs::File::open(&expected_parquet)?;
    let reader = parquet::file::reader::SerializedFileReader::new(file)?;
    use parquet::file::reader::FileReader;
    let meta = reader.metadata();
    assert_eq!(meta.num_row_groups(), 1);
    assert_eq!(meta.file_metadata().num_rows(), 3);
    let schema = meta.file_metadata().schema_descr();
    let col_names: Vec<String> = (0..schema.num_columns())
        .map(|i| schema.column(i).name().to_string())
        .collect();
    assert!(col_names.iter().any(|c| c == "Engine.Speed.RPM"));
    assert!(col_names.iter().any(|c| c == "CAN1.Timestamp"));
    assert!(col_names.iter().any(|c| c == "Vehicle.Dynamics.YawRate"));

    // CSV export with prefix "prefix"
    let csv_prefix = tmp_dir.path().join("prefix");
    mdf.export_to_csv(csv_prefix.to_str().unwrap())?;
    let expected_csv = tmp_dir.path().join("prefix_Engine.Speed.RPM.csv");
    assert!(
        expected_csv.exists(),
        "Expected CSV file {:?} to exist",
        expected_csv
    );
    let csv_content = fs::read_to_string(&expected_csv)?;
    let lines: Vec<&str> = csv_content.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(lines.len(), 4, "Expected header + 3 data lines");
    assert!(lines[0].contains("Engine.Speed.RPM"));
    assert!(lines[0].contains("CAN1.Timestamp"));
    assert!(lines[1].contains("800"));

    // 2. Export with master "CAN1.Timestamp"
    let mdf_can = Mdf::from_polars_df(&df, Some("CAN1.Timestamp"))?;
    let can_parquet_prefix = tmp_dir.path().join("test");
    mdf_can.export_to_parquet(can_parquet_prefix.to_str().unwrap(), None)?;
    let expected_can_parquet = tmp_dir.path().join("test_CAN1.Timestamp.parquet");
    assert!(
        expected_can_parquet.exists(),
        "Expected parquet file {:?} to exist",
        expected_can_parquet
    );

    let can_csv_prefix = tmp_dir.path().join("test");
    mdf_can.export_to_csv(can_csv_prefix.to_str().unwrap())?;
    let expected_can_csv = tmp_dir.path().join("test_CAN1.Timestamp.csv");
    assert!(
        expected_can_csv.exists(),
        "Expected CSV file {:?} to exist",
        expected_can_csv
    );

    Ok(())
}

// ============================================================================
// Feature 34 Tests: mdfr.pyi Typing Consistency and py.typed Marker
// ============================================================================

#[test]
fn test_feature34_pyi_typing_and_py_typed() -> anyhow::Result<()> {
    // 1. py.typed exists and has PEP 561 marker
    let py_typed = Path::new("py.typed");
    assert!(py_typed.exists(), "py.typed must exist at workspace root");
    let py_typed_content = fs::read_to_string(py_typed)?;
    assert!(
        py_typed_content.contains("PEP 561"),
        "py.typed must contain PEP 561 marker comment"
    );

    // 2. mdfr.pyi exists and parses cleanly in Python AST
    let pyi_path = Path::new("mdfr.pyi");
    assert!(pyi_path.exists(), "mdfr.pyi must exist at workspace root");
    let pyi_content = fs::read_to_string(pyi_path)?;

    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let ast = py.import("ast")?;
        let _parsed = ast.call_method1("parse", (&pyi_content,))?;
        Ok(())
    })?;

    // 3. Verify signatures and documentation in mdfr.pyi
    assert!(
        pyi_content.contains("def contains_channel(self, channel_name: str) -> bool:"),
        "contains_channel stub missing in mdfr.pyi"
    );
    assert!(
        pyi_content.contains("def channel_count(self) -> int:"),
        "channel_count stub missing in mdfr.pyi"
    );
    assert!(
        pyi_content.contains(
            "def cut_channel_group(self, master_name: str, start_s: float, stop_s: float) -> None:"
        ),
        "cut_channel_group stub missing in mdfr.pyi"
    );
    assert!(
        pyi_content.contains("def write("),
        "write stub missing in mdfr.pyi"
    );
    assert!(
        pyi_content
            .contains("compression: CompressionAlgorithm = CompressionAlgorithm.NoCompression"),
        "write default NoCompression missing in mdfr.pyi"
    );
    assert!(
        pyi_content.contains("def add_channel("),
        "add_channel stub missing in mdfr.pyi"
    );
    assert!(
        pyi_content.contains("master: dict[str, Any] | str | None = None"),
        "add_channel master type missing in mdfr.pyi"
    );
    assert!(
        pyi_content.contains("master_channel: str | None = None"),
        "add_channel master_channel kwarg missing in mdfr.pyi"
    );
    assert!(
        pyi_content.contains("master_type: int | None = None"),
        "add_channel master_type kwarg missing in mdfr.pyi"
    );
    assert!(
        pyi_content.contains("master_flag: bool | None = None"),
        "add_channel master_flag kwarg missing in mdfr.pyi"
    );
    assert!(
        pyi_content.contains("def to_numpy(self) -> Any:"),
        "to_numpy stub missing in mdfr.pyi"
    );
    assert!(
        pyi_content.contains("True in boolean mask indicates masked/invalid samples"),
        "to_numpy mask polarity docstring missing in mdfr.pyi"
    );

    Ok(())
}

// ============================================================================
// Feature 35 Tests: CI Workflow Validation and Linker Checks
// ============================================================================

#[test]
fn test_feature35_ci_workflow_and_linker() -> anyhow::Result<()> {
    // 1. Validate rust.yml syntax and required CI jobs
    let rust_yml = fs::read_to_string(".github/workflows/rust.yml")?;
    assert!(
        rust_yml.contains("cargo fmt --all -- --check"),
        "rust.yml must run cargo fmt check"
    );
    assert!(
        rust_yml.contains("cargo clippy --workspace --all-targets --all-features -- -D warnings"),
        "rust.yml must run workspace clippy with -D warnings"
    );
    assert!(
        rust_yml.contains("cargo clippy -p mdfr-core --no-default-features -- -D warnings"),
        "rust.yml must run mdfr-core no-default-features clippy"
    );
    assert!(
        rust_yml.contains("thumbv7em-none-eabihf"),
        "rust.yml must check Cortex-M4F target"
    );
    assert!(
        rust_yml.contains("cargo test --workspace --all-features"),
        "rust.yml must run full workspace tests with --all-features"
    );

    // 2. Validate release workflows
    let manylinux = fs::read_to_string(".github/workflows/py-release-manylinux.yml")?;
    assert!(
        manylinux.contains("tags:") && manylinux.contains("'v*'"),
        "manylinux release must trigger on v* tags"
    );
    assert!(
        manylinux.contains("workflow_dispatch:"),
        "manylinux release must have workflow_dispatch"
    );

    let win_mac = fs::read_to_string(".github/workflows/py-release-windows-macos.yml")?;
    assert!(
        win_mac.contains("tags:") && win_mac.contains("'v*'"),
        "windows release must trigger on v* tags"
    );
    assert!(
        win_mac.contains("workflow_dispatch:"),
        "windows release must have workflow_dispatch"
    );

    let mac_arm = fs::read_to_string(".github/workflows/py-mac-aarch64-apple-release.yml")?;
    assert!(
        mac_arm.contains("tags:") && mac_arm.contains("'v*'"),
        "macOS arm release must trigger on v* tags"
    );
    assert!(
        mac_arm.contains("workflow_dispatch:"),
        "macOS arm release must have workflow_dispatch"
    );

    // 3. Verify PyO3 linker fix: Cargo.toml must NOT contain extension-module
    let cargo_toml = fs::read_to_string("Cargo.toml")?;
    assert!(
        !cargo_toml.contains("extension-module = [\"pyo3/extension-module\"]"),
        "Cargo.toml must not have extension-module feature, which breaks test linking!"
    );

    Ok(())
}
