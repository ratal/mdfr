/// Integration tests exercising Int8/Int16/Float32 conversion arms in conversions4.rs.
///
/// The existing sample files all use Float64 as raw channel type, so those arms
/// were never hit. These tests load a synthetic MDF4 fixture that has Int8/Int16/Float32
/// channels with a linear CC block.
use anyhow::Result;
use mdfr::data_holder::channel_data::ChannelData;
use mdfr::mdfreader::Mdf;
use std::sync::LazyLock;

const FIXTURE_PATH: &str = "test_files/synthetic/int_linear_cc.mf4";

/// Ensure the fixture file exists before tests run.
static FIXTURE: LazyLock<()> = LazyLock::new(|| {
    create_fixture().expect("failed to create int_linear_cc fixture");
});

// ─── Minimal MDF4 binary builder ─────────────────────────────────────────────

fn pu8(b: &mut Vec<u8>, v: u8) {
    b.push(v);
}
fn pu16(b: &mut Vec<u8>, v: u16) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn pi16(b: &mut Vec<u8>, v: i16) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn pu32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn pu64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn pi64(b: &mut Vec<u8>, v: i64) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn pf32(b: &mut Vec<u8>, v: f32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn pf64(b: &mut Vec<u8>, v: f64) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn zeros(b: &mut Vec<u8>, n: usize) {
    b.extend(std::iter::repeat_n(0u8, n));
}

fn id_block(b: &mut Vec<u8>) {
    b.extend_from_slice(b"MDF     ");
    b.extend_from_slice(b"4.30    ");
    b.extend_from_slice(b"mdfr    ");
    pu16(b, 0);
    pu16(b, 0);
    pu16(b, 430);
    pu16(b, 0);
    zeros(b, 2);
    zeros(b, 26);
    pu16(b, 0);
    pu16(b, 0);
}
fn hd4(b: &mut Vec<u8>, dg: i64, fh: i64) {
    b.extend_from_slice(b"##HD");
    zeros(b, 4);
    pu64(b, 104);
    pu64(b, 6);
    pi64(b, dg);
    pi64(b, fh);
    pi64(b, 0);
    pi64(b, 0);
    pi64(b, 0);
    pi64(b, 0);
    pu64(b, 0);
    pi16(b, 0);
    pi16(b, 0);
    pu8(b, 0);
    pu8(b, 0);
    pu8(b, 0);
    pu8(b, 0);
    pf64(b, 0.0);
    pf64(b, 0.0);
}
fn fh(b: &mut Vec<u8>) {
    b.extend_from_slice(b"##FH");
    zeros(b, 4);
    pu64(b, 56);
    pu64(b, 2);
    pi64(b, 0);
    pi64(b, 0);
    pu64(b, 0);
    pi16(b, 0);
    pi16(b, 0);
    pu8(b, 0);
    zeros(b, 3);
}
fn dg4(b: &mut Vec<u8>, cg: i64, data: i64) {
    b.extend_from_slice(b"##DG");
    zeros(b, 4);
    pu64(b, 64);
    pu64(b, 4);
    pi64(b, 0);
    pi64(b, cg);
    pi64(b, data);
    pi64(b, 0);
    pu8(b, 0);
    zeros(b, 7);
}
fn cg4(b: &mut Vec<u8>, cn: i64, cycles: u64, data_bytes: u32) {
    b.extend_from_slice(b"##CG");
    zeros(b, 4);
    pu64(b, 104);
    pu64(b, 6);
    pi64(b, 0);
    pi64(b, cn);
    pi64(b, 0);
    pi64(b, 0);
    pi64(b, 0);
    pi64(b, 0);
    pu64(b, 0);
    pu64(b, cycles);
    pu16(b, 0);
    pu16(b, 0);
    zeros(b, 4);
    pu32(b, data_bytes);
    pu32(b, 0);
}
#[allow(clippy::too_many_arguments)]
fn cn4(
    b: &mut Vec<u8>,
    cn_type: u8,
    sync: u8,
    dtype: u8,
    byte_off: u32,
    bits: u32,
    next: i64,
    tx: i64,
    cc: i64,
) {
    b.extend_from_slice(b"##CN");
    zeros(b, 4);
    pu64(b, 160);
    pu64(b, 8);
    pi64(b, next);
    pi64(b, 0);
    pi64(b, tx);
    pi64(b, 0);
    pi64(b, cc);
    pi64(b, 0);
    pi64(b, 0);
    pi64(b, 0);
    pu8(b, cn_type);
    pu8(b, sync);
    pu8(b, dtype);
    pu8(b, 0);
    pu32(b, byte_off);
    pu32(b, bits);
    pu32(b, 0);
    pu32(b, 0);
    pu8(b, 0xff);
    pu8(b, 0);
    pu16(b, 0);
    pf64(b, 0.0);
    pf64(b, 0.0);
    pf64(b, 0.0);
    pf64(b, 0.0);
    pf64(b, 0.0);
    pf64(b, 0.0);
}
fn tx(b: &mut Vec<u8>, text: &str) {
    let t = text.as_bytes();
    let len = 24u64 + t.len() as u64 + 1;
    b.extend_from_slice(b"##TX");
    zeros(b, 4);
    pu64(b, len);
    pu64(b, 0);
    b.extend_from_slice(t);
    b.push(0);
}
fn cc_linear(b: &mut Vec<u8>, a0: f64, a1: f64) {
    b.extend_from_slice(b"##CC");
    zeros(b, 4);
    pu64(b, 96);
    pu64(b, 4);
    pi64(b, 0);
    pi64(b, 0);
    pi64(b, 0);
    pi64(b, 0);
    pu8(b, 1);
    pu8(b, 0);
    pu16(b, 0);
    pu16(b, 0);
    pu16(b, 2);
    pf64(b, 0.0);
    pf64(b, 0.0);
    pf64(b, a0);
    pf64(b, a1);
}
fn dt(b: &mut Vec<u8>, records: &[u8]) {
    let len = 24u64 + records.len() as u64;
    b.extend_from_slice(b"##DT");
    zeros(b, 4);
    pu64(b, len);
    pu64(b, 0);
    b.extend_from_slice(records);
}

/// Creates `test_files/synthetic/int_linear_cc.mf4` if it doesn't already exist.
///
/// Layout (offsets):
/// [0]    IdBlock   64 b
/// [64]   Hd4       104 b  (hd_dg_first=224, hd_fh_first=168)
/// [168]  FhBlock   56 b
/// [224]  Dg4       64 b   (cg_first=288, data=1260)
/// [288]  CG        104 b  (cn_first=392, cycles=4, data_bytes=15)
/// [392]  CN_master 160 b  (FloatLE/64bit, byte_off=0,  tx=1032, cc=0)
/// [552]  CN_int8   160 b  (IntLE/8bit,   byte_off=8,  tx=1064, cc=1164)
/// [712]  CN_int16  160 b  (IntLE/16bit,  byte_off=9,  tx=1096, cc=1164)
/// [872]  CN_float32 160 b (FloatLE/32bit, byte_off=11, tx=1129, cc=1164)
/// [1032] TX "time_ch\0"     32 b
/// [1064] TX "int8_ch\0"     32 b
/// [1096] TX "int16_ch\0"    33 b
/// [1129] TX "float32_ch\0"  35 b
/// [1164] CC linear(0.5,2.0) 96 b
/// [1260] DT 4×15=60b + 24 hdr = 84 b
/// Total: 1344 b
fn create_fixture() -> Result<()> {
    if std::path::Path::new(FIXTURE_PATH).exists() {
        return Ok(());
    }
    std::fs::create_dir_all("test_files/synthetic")?;
    let mut b: Vec<u8> = Vec::with_capacity(1344);

    id_block(&mut b);
    debug_assert_eq!(b.len(), 64);
    hd4(&mut b, 224, 168);
    debug_assert_eq!(b.len(), 168);
    fh(&mut b);
    debug_assert_eq!(b.len(), 224);
    dg4(&mut b, 288, 1260);
    debug_assert_eq!(b.len(), 288);
    cg4(&mut b, 392, 4, 15);
    debug_assert_eq!(b.len(), 392);
    cn4(&mut b, 2, 1, 4, 0, 64, 552, 1032, 0);
    debug_assert_eq!(b.len(), 552);
    cn4(&mut b, 0, 0, 2, 8, 8, 712, 1064, 1164);
    debug_assert_eq!(b.len(), 712);
    cn4(&mut b, 0, 0, 2, 9, 16, 872, 1096, 1164);
    debug_assert_eq!(b.len(), 872);
    cn4(&mut b, 0, 0, 4, 11, 32, 0, 1129, 1164);
    debug_assert_eq!(b.len(), 1032);
    tx(&mut b, "time_ch");
    debug_assert_eq!(b.len(), 1064);
    tx(&mut b, "int8_ch");
    debug_assert_eq!(b.len(), 1096);
    tx(&mut b, "int16_ch");
    debug_assert_eq!(b.len(), 1129);
    tx(&mut b, "float32_ch");
    debug_assert_eq!(b.len(), 1164);
    cc_linear(&mut b, 0.5, 2.0);
    debug_assert_eq!(b.len(), 1260);

    let raw_i8: [i8; 4] = [-5, 0, 5, 10];
    let raw_i16: [i16; 4] = [-100, 0, 100, 200];
    let raw_f32: [f32; 4] = [1.5, 2.5, 3.5, 4.5];
    let mut recs: Vec<u8> = Vec::with_capacity(60);
    for i in 0..4 {
        pf64(&mut recs, i as f64);
        recs.push(raw_i8[i] as u8);
        recs.extend_from_slice(&raw_i16[i].to_le_bytes());
        pf32(&mut recs, raw_f32[i]);
    }
    dt(&mut b, &recs);
    debug_assert_eq!(b.len(), 1344);

    std::fs::write(FIXTURE_PATH, &b)?;
    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

fn load_fixture() -> Result<Mdf> {
    LazyLock::force(&FIXTURE);
    let mut mdf = Mdf::new(FIXTURE_PATH)?;
    mdf.load_all_channels_data_in_memory()?;
    Ok(mdf)
}

#[test]
fn int8_linear_conversion() -> Result<()> {
    let mdf = load_fixture()?;
    // raw: -5, 0, 5, 10 → phys = raw * 2.0 + 0.5 → -9.5, 0.5, 10.5, 20.5
    let data = mdf.get_channel_data("int8_ch").expect("int8_ch not found");
    assert!(
        matches!(data, ChannelData::Float64(_)),
        "expected Float64 after linear CC, got {}",
        data.data_type(false)
    );
    if let ChannelData::Float64(arr) = data {
        let expected = [-9.5f64, 0.5, 10.5, 20.5];
        for (i, (&got, &exp)) in arr.values_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-9,
                "int8_ch[{i}]: expected {exp}, got {got}"
            );
        }
    }
    Ok(())
}

#[test]
fn int16_linear_conversion() -> Result<()> {
    let mdf = load_fixture()?;
    // raw: -100, 0, 100, 200 → -199.5, 0.5, 200.5, 400.5
    let data = mdf
        .get_channel_data("int16_ch")
        .expect("int16_ch not found");
    assert!(
        matches!(data, ChannelData::Float64(_)),
        "expected Float64 after linear CC"
    );
    if let ChannelData::Float64(arr) = data {
        let expected = [-199.5f64, 0.5, 200.5, 400.5];
        for (i, (&got, &exp)) in arr.values_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-9,
                "int16_ch[{i}]: expected {exp}, got {got}"
            );
        }
    }
    Ok(())
}

#[test]
fn float32_linear_conversion() -> Result<()> {
    let mdf = load_fixture()?;
    // raw: 1.5, 2.5, 3.5, 4.5 → 3.5, 5.5, 7.5, 9.5
    let data = mdf
        .get_channel_data("float32_ch")
        .expect("float32_ch not found");
    assert!(
        matches!(data, ChannelData::Float64(_)),
        "expected Float64 after linear CC"
    );
    if let ChannelData::Float64(arr) = data {
        let expected = [3.5f64, 5.5, 7.5, 9.5];
        for (i, (&got, &exp)) in arr.values_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "float32_ch[{i}]: expected {exp}, got {got}"
            );
        }
    }
    Ok(())
}

#[test]
fn master_channel_loaded() -> Result<()> {
    let mdf = load_fixture()?;
    let data = mdf.get_channel_data("time_ch").expect("time_ch not found");
    assert!(
        matches!(data, ChannelData::Float64(_)),
        "master should be Float64"
    );
    assert_eq!(data.len(), 4);
    Ok(())
}
