//! Standalone Criterion push-throughput benchmarks for the `mdfr-core` embedded MDF 4.3 logger.
//!
//! Run with:
//! ```bash
//! cargo bench --bench logger_benchmark
//! ```
use core::time::Duration;
use criterion::{Criterion, criterion_group, criterion_main};
use mdfr_core::blocks::cn::CnDataType;
use mdfr_core::compress::CompressionAlgorithm;
use mdfr_core::io::VecSink;
use mdfr_core::logger::{BusPreset, DataLayout, Logger, LoggerBuilder, MdfRecord, SpscRing};
use mdfr_core::time::ManualClock;
use std::hint::black_box;

#[derive(MdfRecord)]
#[mdf(group = "Engine")]
struct EngineSample {
    #[mdf(unit = "rpm", linear(0.0, 0.5))]
    rpm: u16,
    #[mdf(name = "CoolantTemp", unit = "degC")]
    temp: f32,
    gear: u8,
    flags: u8,
}

fn bench_logger_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("logger_throughput");
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

    // 3. 10k typed #[derive(MdfRecord)] structs via push_record (Sorted, Lz4Transpose)
    group.bench_function("push_record_derive_10k_sorted_lz4", |b| {
        let mut builder =
            LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::Lz4Transpose);
        let g_eng = builder.add_record_group::<EngineSample>().unwrap();
        let schema = builder.build().unwrap();
        let sample = EngineSample {
            rpm: 3000,
            temp: 88.5,
            gear: 4,
            flags: 0x03,
        };

        b.iter(|| {
            let sink = VecSink::with_capacity(256 * 1024);
            let mut logger =
                Logger::start(schema.clone(), sink, ManualClock::new(1e-3, 1)).unwrap();
            for _ in 0..10_000 {
                logger.push_record(g_eng, black_box(&sample)).unwrap();
            }
            black_box(logger.close().unwrap());
        })
    });

    // 4. 10k interleaved frames across 2 groups (Unsorted, DeflateTranspose)
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

    // 5. 10k records via dynamic RecordWriter
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

    // 6. SpscRing ISR push + drain_into logger (10k records)
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

criterion_group!(benches, bench_logger_throughput);
criterion_main!(benches);
