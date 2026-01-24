use anyhow::Result;
use core::time::Duration;
use criterion::{criterion_group, criterion_main, Criterion};
use mdfr::mdfreader::Mdf;
use std::process::Command;
use std::sync::LazyLock;

static MDFREADER_TESTS_PATH: &str = "/home/ratal/workspace/mdfreader/mdfreader/tests/";
static MDFR_PATH: &str = "/home/ratal/workspace/mdfr/";

static BASE_PATH_MDF4: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/",
        MDFREADER_TESTS_PATH
    )
});
#[allow(dead_code)]
static BASE_PATH_MDF3: LazyLock<String> =
    LazyLock::new(|| format!("{}mdf3/", MDFREADER_TESTS_PATH));
#[allow(dead_code)]
static WRITING_FILE4: LazyLock<String> = LazyLock::new(|| format!("{}test.mf4", MDFR_PATH));
#[allow(dead_code)]
static WRITING_FILE3: LazyLock<String> = LazyLock::new(|| format!("{}test.dat", MDFR_PATH));
#[allow(dead_code)]
static PARQUET_FILE: LazyLock<String> = LazyLock::new(|| format!("{}test.parquet", MDFR_PATH));

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
    let file = format!(
        "{}{}",
        // BASE_PATH_MDF3.as_str(), &"RJ_N16-12-363_BM-15C-0024_228_2_20170116094355_CAN.dat"
        BASE_PATH_MDF4.as_str(),
        "Simple/error.mf4"
    );
    group.sample_size(15);
    group.measurement_time(Duration::new(10, 0));
    group.bench_function("mdfr_with_mdf4_sorted", |b| {
        b.iter(|| -> Result<()> {
            let mut mdf = Mdf::new(&file)?;
            mdf.load_all_channels_data_in_memory()?;
            // mdf.export_to_parquet(&PARQUET_FILE, "snappy");
            Ok(())
        })
    });
    group.finish();
    println!("mdfreader.mdfreader sorted \n");
    python_launch();
    println!("\n");
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
