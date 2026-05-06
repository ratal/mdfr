use core::time::Duration;
use criterion::{Criterion, criterion_group, criterion_main};
use mdfr::mdfreader::Mdf;
use std::hint::black_box;
use std::sync::LazyLock;

static TESTS_PATH: &str = "/home/ratal/workspace/mdfreader/mdfreader/tests/";
const LOCAL_FILES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_files/");

static MDF4_EXAMPLES: LazyLock<String> = LazyLock::new(|| {
    format!("{}MDF4/MDF4.3/Base_Standard/Examples/", TESTS_PATH)
});
static MDF3_PATH: LazyLock<String> = LazyLock::new(|| format!("{}mdf3/", TESTS_PATH));

fn read_and_load(path: &str) {
    let mut mdf = Mdf::new(black_box(path)).expect("failed to open file");
    mdf.load_all_channels_data_in_memory()
        .expect("failed to load data");
}

/// Large files (≥50 MB) — sample_size(10) with enough measurement_time to avoid the
/// "Unable to complete N samples" warning (10 × ~2 s/iter < 25 s).
fn bench_large_files(c: &mut Criterion) {
    let base = MDF4_EXAMPLES.as_str();
    let mut group = c.benchmark_group("large_files");
    group.warm_up_time(Duration::from_secs(1));
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(25));

    let mdf4_path = format!("{base}Simple/error.mf4");
    group.bench_function("mdf4_sorted_184mb", |b| b.iter(|| read_and_load(&mdf4_path)));

    for (name, file) in [
        (
            "mdf3_can_84mb",
            "RJ_N16-12-363_BM-15C-0024_228_2_20170116094355_CAN.dat",
        ),
        ("mdf3_nedc_63mb", "738L10 cold nedc 05101001-1.dat"),
    ] {
        let path = format!("{}{file}", MDF3_PATH.as_str());
        group.bench_function(name, |b| b.iter(|| read_and_load(&path)));
    }

    group.finish();
}

/// Medium files (5–50 MB) — covers sorted, linked-list, and equal-length DT layouts.
fn bench_medium_files(c: &mut Criterion) {
    let base = MDF4_EXAMPLES.as_str();
    let mut group = c.benchmark_group("medium_files");
    group.warm_up_time(Duration::from_secs(1));
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(12));

    for (name, rel) in [
        ("mdf4_measure_47mb", "Simple/measure2.mf4"),
        ("mdf4_data_list_linked_11mb", "DataList/Vector_DL_Linked_List.MF4"),
        ("mdf4_data_list_equal_8mb", "DataList/DT_EqualLength/Vector_DT_EqualLen.MF4"),
    ] {
        let path = format!("{base}{rel}");
        group.bench_function(name, |b| b.iter(|| read_and_load(&path)));
    }

    group.finish();
}

/// Deflate vs transpose-deflate decompression on equivalent content —
/// single-block and data-list layouts compared directly.
fn bench_decompression(c: &mut Criterion) {
    let base = MDF4_EXAMPLES.as_str();
    let mut group = c.benchmark_group("decompression");
    group.warm_up_time(Duration::from_secs(1));
    group.sample_size(30);
    group.measurement_time(Duration::from_secs(5));

    for (name, rel) in [
        (
            "single_block_deflate",
            "CompressedData/Simple/Vector_SingleDZ_Deflate.mf4",
        ),
        (
            "single_block_transpose_deflate",
            "CompressedData/Simple/Vector_SingleDZ_TransposeDeflate.mf4",
        ),
        (
            "data_list_deflate",
            "CompressedData/DataList/Vector_DataList_Deflate.mf4",
        ),
        (
            "data_list_transpose_deflate",
            "CompressedData/DataList/Vector_DataList_TransposeDeflate.mf4",
        ),
    ] {
        let path = format!("{base}{rel}");
        group.bench_function(name, |b| b.iter(|| read_and_load(&path)));
    }

    group.finish();
}

/// Local synthetic files — always available, each exercises a distinct data path.
fn bench_local_synthetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("local_synthetic");
    group.warm_up_time(Duration::from_secs(1));
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(5));

    for (name, rel) in [
        ("array_f64_le", "synthetic/array_f64_le.mf4"),     // f64 array channels, LE
        ("be_scalars", "synthetic/be_scalars.mf4"),         // big-endian byte-swap path
        ("complex_f32_le", "synthetic/complex_f32_le.mf4"), // complex number channels
        ("int_linear_cc", "synthetic/int_linear_cc.mf4"),   // integer + linear CC conversion
        ("ld_be_channels", "synthetic/ld_be_channels.mf4"), // linked data blocks, BE
    ] {
        let path = format!("{LOCAL_FILES}{rel}");
        group.bench_function(name, |b| b.iter(|| read_and_load(&path)));
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_large_files,
    bench_medium_files,
    bench_decompression,
    bench_local_synthetic,
);
criterion_main!(benches);
