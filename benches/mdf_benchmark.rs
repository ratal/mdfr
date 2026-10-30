use core::time::Duration;
use criterion::{Criterion, criterion_group, criterion_main};
use mdfr::mdfreader::Mdf;
use std::hint::black_box;
use std::sync::LazyLock;

static TESTS_PATH: &str = "/home/ratal/workspace/mdfreader/mdfreader/tests/";
const LOCAL_FILES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_files/");

static MDF4_EXAMPLES: LazyLock<String> = LazyLock::new(|| {
    let local_examples = format!("{}Examples/", LOCAL_FILES);
    if std::path::Path::new(&local_examples).is_dir() {
        return local_examples;
    }
    let mdfreader_examples =
        format!("{}mdfreader/MDF4/MDF4.3/Base_Standard/Examples/", LOCAL_FILES);
    if std::path::Path::new(&mdfreader_examples).is_dir() {
        return mdfreader_examples;
    }
    format!("{}MDF4/MDF4.3/Base_Standard/Examples/", TESTS_PATH)
});
static MDF3_PATH: LazyLock<String> = LazyLock::new(|| {
    let local_mdf3 = format!("{}mdf3/", LOCAL_FILES);
    if std::path::Path::new(&local_mdf3).is_dir() {
        return local_mdf3;
    }
    format!("{}mdf3/", TESTS_PATH)
});

fn read_and_load(path: &str) {
    let mut mdf = Mdf::new(black_box(path)).expect("failed to open file");
    mdf.load_all_channels_data_in_memory()
        .expect("failed to load data");
}

fn read_load_convert(path: &str) {
    let mut mdf = Mdf::new(black_box(path)).expect("failed to open file");
    mdf.load_all_channels_data_in_memory()
        .expect("failed to load data");
    mdf.convert_all_channels()
        .expect("failed to convert channels");
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
    if std::path::Path::new(&mdf4_path).exists() {
        group.bench_function("mdf4_sorted_184mb", |b| {
            b.iter(|| read_and_load(&mdf4_path))
        });
    }

    for (name, file) in [
        (
            "mdf3_can_84mb",
            "RJ_N16-12-363_BM-15C-0024_228_2_20170116094355_CAN.dat",
        ),
        ("mdf3_nedc_63mb", "738L10 cold nedc 05101001-1.dat"),
    ] {
        let path = format!("{}{file}", MDF3_PATH.as_str());
        if std::path::Path::new(&path).exists() {
            group.bench_function(name, |b| b.iter(|| read_and_load(&path)));
        }
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
        (
            "mdf4_data_list_linked_11mb",
            "DataList/Vector_DL_Linked_List.MF4",
        ),
        (
            "mdf4_data_list_equal_8mb",
            "DataList/DT_EqualLength/Vector_DT_EqualLen.MF4",
        ),
    ] {
        let path = format!("{base}{rel}");
        if std::path::Path::new(&path).exists() {
            group.bench_function(name, |b| b.iter(|| read_and_load(&path)));
        }
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
        if std::path::Path::new(&path).exists() {
            group.bench_function(name, |b| b.iter(|| read_and_load(&path)));
        }
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
        ("array_f64_le", "synthetic/array_f64_le.mf4"), // f64 array channels, LE
        ("be_scalars", "synthetic/be_scalars.mf4"),     // big-endian byte-swap path
        ("complex_f32_le", "synthetic/complex_f32_le.mf4"), // complex number channels
        ("int_linear_cc", "synthetic/int_linear_cc.mf4"), // integer + linear CC conversion
        ("ld_be_channels", "synthetic/ld_be_channels.mf4"), // linked data blocks, BE
    ] {
        let path = format!("{LOCAL_FILES}{rel}");
        if std::path::Path::new(&path).exists() {
            group.bench_function(name, |b| b.iter(|| read_and_load(&path)));
        }
    }

    group.finish();
}

/// Metadata-only parsing — calls only `Mdf::new()` (no data load) to isolate
/// the cost of block header parsing and eager `parse_xml()` calls.
fn bench_metadata(c: &mut Criterion) {
    let base = MDF4_EXAMPLES.as_str();
    let mut group = c.benchmark_group("metadata");
    group.warm_up_time(Duration::from_secs(1));
    group.sample_size(30);
    group.measurement_time(Duration::from_secs(6));

    for (name, rel) in [
        ("mdf4_sorted_184mb", "Simple/error.mf4"),
        ("mdf4_measure_47mb", "Simple/measure2.mf4"),
        (
            "mdf4_compressed",
            "CompressedData/Simple/Vector_SingleDZ_Deflate.mf4",
        ),
    ] {
        let path = format!("{base}{rel}");
        if std::path::Path::new(&path).exists() {
            group.bench_function(name, |b| {
                b.iter(|| Mdf::new(black_box(&path)).expect("failed to open file"))
            });
        }
    }

    group.finish();
}

/// Unsorted data benchmark — measures performance of the unsorted data reading path
/// where records from multiple channel groups are interleaved and must be sorted.
fn bench_unsorted_data(c: &mut Criterion) {
    let mut group = c.benchmark_group("unsorted_data");
    group.warm_up_time(Duration::from_secs(1));
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));

    // Synthetic large file with 4 channel groups, ~50MB (fixed-length records)
    let p_multi_cg = "test_files/synthetic/unsorted_multi_cg.mf4";
    if std::path::Path::new(p_multi_cg).exists() {
        group.bench_function("unsorted_multi_cg_50mb", |b| {
            b.iter(|| read_and_load(p_multi_cg))
        });
    }

    // Synthetic large file with VLSD (Variable Length Signal Data), ~30MB
    let p_vlsd = "test_files/synthetic/unsorted_vlsd_vlsc.mf4";
    if std::path::Path::new(p_vlsd).exists() {
        group.bench_function("unsorted_vlsd_30mb", |b| {
            b.iter(|| read_and_load(p_vlsd))
        });
    }

    // Existing reference files from mdfreader tests
    let base = MDF4_EXAMPLES.as_str();
    let p_ref_vlsd = format!("{base}UnsortedData/VLSD/Vector_Unsorted_VLSD.MF4");
    if std::path::Path::new(&p_ref_vlsd).exists() {
        group.bench_function("mdf4_vlsd_60kb", |b| {
            b.iter(|| read_and_load(&p_ref_vlsd))
        });
    }

    let p_ref_vlsc =
        format!("{base}UnsortedData/VLSC/RAC_MDF430_Unsorted_VLSC_Compact_Structure.mf4");
    if std::path::Path::new(&p_ref_vlsc).exists() {
        group.bench_function("mdf4_vlsc_5kb", |b| {
            b.iter(|| read_and_load(&p_ref_vlsc))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_large_files,
    bench_medium_files,
    bench_decompression,
    bench_local_synthetic,
    bench_metadata,
    bench_unsorted_data,
    bench_conversion,
    bench_export,
    bench_embedded_logger,
);
criterion_main!(benches);

/// Conversion benchmark — measures performance of converting raw channel data
/// to physical values (applies CC formulas, bit masks, etc.).
fn bench_conversion(c: &mut Criterion) {
    let base = MDF4_EXAMPLES.as_str();
    let mut group = c.benchmark_group("conversion");
    group.warm_up_time(Duration::from_secs(1));
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));

    // Full pipeline: load + conversion
    let path_err = format!("{base}Simple/error.mf4");
    if std::path::Path::new(&path_err).exists() {
        group.bench_function("convert_184mb", |b| {
            b.iter(|| read_load_convert(&path_err))
        });
    }

    let path_m2 = format!("{base}Simple/measure2.mf4");
    if std::path::Path::new(&path_m2).exists() {
        group.bench_function("convert_47mb", |b| {
            b.iter(|| read_load_convert(&path_m2))
        });

        // Conversion only (data already loaded)
        group.bench_function("convert_only_47mb", |b| {
            let mut mdf = Mdf::new(&path_m2).expect("failed to open file");
            mdf.load_all_channels_data_in_memory()
                .expect("failed to load data");
            b.iter(|| {
                mdf.convert_all_channels().expect("failed to convert");
                black_box(&mdf);
            })
        });
    }

    // Integer + linear CC conversion
    let path_int = format!("{LOCAL_FILES}synthetic/int_linear_cc.mf4");
    if std::path::Path::new(&path_int).exists() {
        group.bench_function("convert_int_linear_cc", |b| {
            b.iter(|| read_load_convert(&path_int))
        });
    }

    group.finish();
}

/// Export benchmark — measures performance of exporting channel data
/// to various file formats (Parquet, HDF5, CSV).
fn bench_export(c: &mut Criterion) {
    let base = MDF4_EXAMPLES.as_str();
    let mut group = c.benchmark_group("export");
    group.warm_up_time(Duration::from_secs(1));
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));

    let path = format!("{base}Simple/measure2.mf4");
    if std::path::Path::new(&path).exists() {
        // Parquet export
        #[cfg(feature = "parquet")]
        {
            group.bench_function("export_parquet_47mb", |b| {
                let mut mdf = Mdf::new(&path).expect("failed to open file");
                mdf.load_all_channels_data_in_memory()
                    .expect("failed to load data");
                mdf.convert_all_channels().expect("failed to convert");
                let output = "/tmp/bench_export.parquet";
                b.iter(|| {
                    mdf.export_to_parquet(black_box(output), None)
                        .expect("failed to export");
                })
            });

            // Parquet export with compression
            group.bench_function("export_parquet_compressed_47mb", |b| {
                let mut mdf = Mdf::new(&path).expect("failed to open file");
                mdf.load_all_channels_data_in_memory()
                    .expect("failed to load data");
                mdf.convert_all_channels().expect("failed to convert");
                let output = "/tmp/bench_export_compressed.parquet";
                b.iter(|| {
                    mdf.export_to_parquet(black_box(output), Some("snappy"))
                        .expect("failed to export");
                })
            });
        }

        // HDF5 export
        #[cfg(feature = "hdf5")]
        {
            group.bench_function("export_hdf5_47mb", |b| {
                let mut mdf = Mdf::new(&path).expect("failed to open file");
                mdf.load_all_channels_data_in_memory()
                    .expect("failed to load data");
                mdf.convert_all_channels().expect("failed to convert");
                let output = "/tmp/bench_export.h5";
                b.iter(|| {
                    mdf.export_to_hdf5(black_box(output), None)
                        .expect("failed to export");
                })
            });
        }

        // CSV export
        group.bench_function("export_csv_47mb", |b| {
            let mut mdf = Mdf::new(&path).expect("failed to open file");
            mdf.load_all_channels_data_in_memory()
                .expect("failed to load data");
            mdf.convert_all_channels().expect("failed to convert");
            let output = "/tmp/bench_export.csv";
            b.iter(|| {
                mdf.export_to_csv(black_box(output))
                    .expect("failed to export");
            })
        });
    }

    group.finish();
}

/// Embedded streaming logger (`mdfr-core`) throughput benchmarks — measures `push_raw`,
/// dynamic `RecordWriter`, and `SpscRing` ISR queue drain into an in-memory `VecSink`.
fn bench_embedded_logger(c: &mut Criterion) {
    use mdfr_core::blocks::cn::CnDataType;
    use mdfr_core::compress::CompressionAlgorithm;
    use mdfr_core::io::VecSink;
    use mdfr_core::logger::{BusPreset, DataLayout, Logger, LoggerBuilder, SpscRing};
    use mdfr_core::time::ManualClock;

    let mut group = c.benchmark_group("embedded_logger");
    group.warm_up_time(Duration::from_secs(1));
    group.sample_size(30);
    group.measurement_time(Duration::from_secs(5));

    // 1. 10k CAN frames via zero-overhead push_raw (Sorted, Uncompressed)
    group.bench_function("push_raw_10k_sorted_uncompressed", |b| {
        let mut builder =
            LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::NoCompression);
        let can = BusPreset::can(&mut builder, "CAN1").unwrap();
        let schema = builder.build().unwrap();
        let rec_len = schema.groups[can.group.0 as usize].record_data_bytes as usize;
        let raw_frame = vec![0x55u8; rec_len];

        b.iter(|| {
            let sink = VecSink::with_capacity(512 * 1024);
            let mut logger =
                Logger::start(schema.clone(), sink, ManualClock::new(1e-6, 100)).unwrap();
            for _ in 0..10_000 {
                logger.push_raw(can.group, black_box(&raw_frame)).unwrap();
            }
            black_box(logger.close().unwrap());
        })
    });

    // 2. 10k CAN frames via push_raw (Sorted, Lz4Transpose)
    group.bench_function("push_raw_10k_sorted_lz4_transpose", |b| {
        let mut builder =
            LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::Lz4Transpose);
        let can = BusPreset::can(&mut builder, "CAN1").unwrap();
        let schema = builder.build().unwrap();
        let rec_len = schema.groups[can.group.0 as usize].record_data_bytes as usize;
        let raw_frame = vec![0x55u8; rec_len];

        b.iter(|| {
            let sink = VecSink::with_capacity(256 * 1024);
            let mut logger =
                Logger::start(schema.clone(), sink, ManualClock::new(1e-6, 100)).unwrap();
            for _ in 0..10_000 {
                logger.push_raw(can.group, black_box(&raw_frame)).unwrap();
            }
            black_box(logger.close().unwrap());
        })
    });

    // 3. 10k interleaved frames across 2 groups (Unsorted, DeflateTranspose)
    group.bench_function("push_raw_10k_unsorted_deflate_transpose", |b| {
        let mut builder =
            LoggerBuilder::new(DataLayout::Unsorted, CompressionAlgorithm::DeflateTranspose);
        let can1 = BusPreset::can(&mut builder, "CAN1").unwrap();
        let can2 = BusPreset::can(&mut builder, "CAN2").unwrap();
        let schema = builder.build().unwrap();
        let rec_len = schema.groups[can1.group.0 as usize].record_data_bytes as usize;
        let raw_frame = vec![0x42u8; rec_len];

        b.iter(|| {
            let sink = VecSink::with_capacity(256 * 1024);
            let mut logger =
                Logger::start(schema.clone(), sink, ManualClock::new(1e-6, 100)).unwrap();
            for _ in 0..5_000 {
                logger.push_raw(can1.group, black_box(&raw_frame)).unwrap();
                logger.push_raw(can2.group, black_box(&raw_frame)).unwrap();
            }
            black_box(logger.close().unwrap());
        })
    });

    // 4. 10k records via dynamic RecordWriter (begin_record + set_u64 + set_f64 + commit)
    group.bench_function("record_writer_10k_dynamic", |b| {
        let mut builder =
            LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::Lz4Transpose);
        let g0 = builder.add_group("Sensors");
        let ch_rpm = builder
            .add_channel(g0, "RPM", CnDataType::UIntLE, 16)
            .unwrap();
        let ch_temp = builder
            .add_channel(g0, "Temp", CnDataType::FloatLE, 32)
            .unwrap();
        let schema = builder.build().unwrap();

        b.iter(|| {
            let sink = VecSink::with_capacity(256 * 1024);
            let mut logger =
                Logger::start(schema.clone(), sink, ManualClock::new(1e-3, 1)).unwrap();
            for i in 0..10_000u64 {
                logger
                    .begin_record(g0)
                    .unwrap()
                    .set_u64(ch_rpm, black_box(2500 + (i & 0xFF)))
                    .unwrap()
                    .set_f64(ch_temp, black_box(85.0))
                    .unwrap()
                    .commit()
                    .unwrap();
            }
            black_box(logger.close().unwrap());
        })
    });

    // 5. SpscRing ISR push + drain_into logger (10k records in batches of 100)
    group.bench_function("spsc_ring_10k_push_and_drain", |b| {
        let mut builder =
            LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::NoCompression);
        let g0 = builder.add_group("ISRGroup");
        let _ch = builder
            .add_channel(g0, "ADC", CnDataType::UIntLE, 32)
            .unwrap();
        let schema = builder.build().unwrap();
        let ring = SpscRing::with_capacity(4096);
        let raw_rec = [0x11u8; 12];

        b.iter(|| {
            let sink = VecSink::with_capacity(256 * 1024);
            let mut logger =
                Logger::start(schema.clone(), sink, ManualClock::new(1e-6, 10)).unwrap();
            for _ in 0..100 {
                for _ in 0..100 {
                    ring.try_push(g0, 100, black_box(&raw_rec));
                }
                ring.drain_into(&mut logger).unwrap();
            }
            black_box(logger.close().unwrap());
        })
    });

    group.finish();
}
