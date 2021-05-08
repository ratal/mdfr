use criterion::{criterion_group, criterion_main, Criterion};
use mdfr::mdfinfo::mdfinfo;
use std::process::Command;

fn python_launch() {
    Command::new("python3")
        .arg("-m")
        .arg("timeit")
        .arg("import mdfreader; yop=mdfreader.MdfInfo('/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/Simple/test.mf4')")
        .spawn()
        .expect("mdfinfo command failed to start");
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ExtremeInfo");
    let file_name = "/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/Simple/test.mf4";
    group.sample_size(20);
    group.bench_function("mdfr_with_mdf4", |b| b.iter(|| mdfinfo(&file_name)));
    group.finish();
    println!("mdfreader.mdfinfo\n");
    python_launch();
    println!("\n");
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);