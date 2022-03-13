use criterion::{criterion_group, criterion_main, Criterion};
use mdfr::mdfreader::mdfreader;
use std::process::Command;

fn python_launch() {
    Command::new("python3")
        .arg("-m")
        .arg("timeit")
        .arg("import mdfreader; yop=mdfreader.Mdf('/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/DataList/Vector_SD_List.MF4')")
        .spawn()
        .expect("mdfinfo command failed to start");
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ExtremeInfo");
    let file_name_sorted = "/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/Simple/error.mf4";
    group.sample_size(10);
    group.bench_function("mdfr_with_mdf4_sorted", |b| {
        b.iter(|| {
            let mut mdf = mdfreader(&file_name_sorted);
            mdf.write(&"/home/ratal/workspace/mdfr/test.mf4", false);
        })
    });
    group.finish();
    println!("mdfreader.mdfreader sorted \n");
    python_launch();
    println!("\n");
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
