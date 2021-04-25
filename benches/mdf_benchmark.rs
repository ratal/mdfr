use criterion::{criterion_group, criterion_main, Criterion};
use mdfr::mdfinfo::mdfinfo;

pub fn criterion_benchmark(c: &mut Criterion) {
    let file_name = "/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/Simple/Measure.mf4";
    c.bench_function("mdf4", |b| b.iter(|| mdfinfo(&file_name)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);