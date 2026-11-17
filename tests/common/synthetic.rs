//! Thread-safe and process-safe synthetic MDF4 fixture builders.
//!
//! Shared `test_files/synthetic/*.mf4` files are initialized at most once per process
//! via `LazyLock` and written atomically via `NamedTempFile::persist_noclobber` so
//! concurrent test threads and concurrent test binaries never observe partial writes.
#![allow(dead_code)]

use anyhow::Result;
use std::io::Write;
use std::path::Path;
use std::sync::LazyLock;

pub const INT_LINEAR_CC_PATH: &str = "test_files/synthetic/int_linear_cc.mf4";
pub const UNSORTED_MULTI_CG_PATH: &str = "test_files/synthetic/unsorted_multi_cg.mf4";
pub const UNSORTED_VLSD_VLSC_PATH: &str = "test_files/synthetic/unsorted_vlsd_vlsc.mf4";
pub const SYNTHETIC_ARRAY_PATH: &str = "test_files/synthetic/synthetic_array.mf4";
pub const SYNTHETIC_DZ_DEFLATE_PATH: &str = "test_files/synthetic/synthetic_dz_deflate.mf4";
pub const BE_SCALARS_PATH: &str = "test_files/synthetic/be_scalars.mf4";
pub const COMPLEX_F32_LE_PATH: &str = "test_files/synthetic/complex_f32_le.mf4";
pub const ARRAY_F64_LE_PATH: &str = "test_files/synthetic/array_f64_le.mf4";
pub const LD_BE_CHANNELS_PATH: &str = "test_files/synthetic/ld_be_channels.mf4";

static INT_LINEAR_CC_ONCE: LazyLock<()> = LazyLock::new(|| {
    ensure_fixture(INT_LINEAR_CC_PATH, build_int_linear_cc)
        .expect("failed to create int_linear_cc fixture");
});
static UNSORTED_MULTI_CG_ONCE: LazyLock<()> = LazyLock::new(|| {
    ensure_fixture(UNSORTED_MULTI_CG_PATH, build_unsorted_multi_cg)
        .expect("failed to create unsorted_multi_cg fixture");
});
static UNSORTED_VLSD_VLSC_ONCE: LazyLock<()> = LazyLock::new(|| {
    ensure_fixture(UNSORTED_VLSD_VLSC_PATH, build_unsorted_vlsd_vlsc)
        .expect("failed to create unsorted_vlsd_vlsc fixture");
});
static SYNTHETIC_ARRAY_ONCE: LazyLock<()> = LazyLock::new(|| {
    ensure_fixture(SYNTHETIC_ARRAY_PATH, build_synthetic_array)
        .expect("failed to create synthetic_array fixture");
});
static SYNTHETIC_DZ_DEFLATE_ONCE: LazyLock<()> = LazyLock::new(|| {
    ensure_fixture(SYNTHETIC_DZ_DEFLATE_PATH, build_synthetic_dz_deflate)
        .expect("failed to create synthetic_dz_deflate fixture");
});
static BE_SCALARS_ONCE: LazyLock<()> = LazyLock::new(|| {
    ensure_fixture(BE_SCALARS_PATH, build_be_scalars).expect("failed to create be_scalars fixture");
});
static COMPLEX_F32_LE_ONCE: LazyLock<()> = LazyLock::new(|| {
    ensure_fixture(COMPLEX_F32_LE_PATH, build_complex_f32_le)
        .expect("failed to create complex_f32_le fixture");
});
static ARRAY_F64_LE_ONCE: LazyLock<()> = LazyLock::new(|| {
    ensure_fixture(ARRAY_F64_LE_PATH, build_array_f64_le)
        .expect("failed to create array_f64_le fixture");
});
static LD_BE_CHANNELS_ONCE: LazyLock<()> = LazyLock::new(|| {
    ensure_fixture(LD_BE_CHANNELS_PATH, build_ld_be_channels)
        .expect("failed to create ld_be_channels fixture");
});

pub fn ensure_int_linear_cc() -> &'static str {
    LazyLock::force(&INT_LINEAR_CC_ONCE);
    INT_LINEAR_CC_PATH
}

pub fn ensure_unsorted_multi_cg() -> &'static str {
    LazyLock::force(&UNSORTED_MULTI_CG_ONCE);
    UNSORTED_MULTI_CG_PATH
}

pub fn ensure_unsorted_vlsd_vlsc() -> &'static str {
    LazyLock::force(&UNSORTED_VLSD_VLSC_ONCE);
    UNSORTED_VLSD_VLSC_PATH
}

pub fn ensure_synthetic_array() -> &'static str {
    LazyLock::force(&SYNTHETIC_ARRAY_ONCE);
    SYNTHETIC_ARRAY_PATH
}

pub fn ensure_synthetic_dz_deflate() -> &'static str {
    LazyLock::force(&SYNTHETIC_DZ_DEFLATE_ONCE);
    SYNTHETIC_DZ_DEFLATE_PATH
}

pub fn ensure_be_scalars() -> &'static str {
    LazyLock::force(&BE_SCALARS_ONCE);
    BE_SCALARS_PATH
}

pub fn ensure_complex_f32_le() -> &'static str {
    LazyLock::force(&COMPLEX_F32_LE_ONCE);
    COMPLEX_F32_LE_PATH
}

pub fn ensure_array_f64_le() -> &'static str {
    LazyLock::force(&ARRAY_F64_LE_ONCE);
    ARRAY_F64_LE_PATH
}

pub fn ensure_ld_be_channels() -> &'static str {
    LazyLock::force(&LD_BE_CHANNELS_ONCE);
    LD_BE_CHANNELS_PATH
}

fn ensure_fixture(path: &str, build: fn() -> Result<Vec<u8>>) -> Result<()> {
    let target = Path::new(path);
    if target.exists() {
        return Ok(());
    }
    let parent = target.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let bytes = build()?;
    let mut tmp = tempfile::NamedTempFile::new_in(parent)?;
    tmp.write_all(&bytes)?;
    tmp.flush()?;
    match tmp.persist_noclobber(target) {
        Ok(_) => Ok(()),
        Err(e) if e.error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(e) => Err(e.error.into()),
    }
}

// ─── Low-level byte helpers ──────────────────────────────────────────────────

fn pu8(b: &mut Vec<u8>, v: u8) {
    b.push(v);
}
fn pu16(b: &mut Vec<u8>, v: u16) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn pu32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn pi32(b: &mut Vec<u8>, v: i32) {
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
fn pi16_be(b: &mut Vec<u8>, v: i16) {
    b.extend_from_slice(&v.to_be_bytes());
}
fn pu16_be(b: &mut Vec<u8>, v: u16) {
    b.extend_from_slice(&v.to_be_bytes());
}
fn pf32_be(b: &mut Vec<u8>, v: f32) {
    b.extend_from_slice(&v.to_be_bytes());
}
fn pf64_be(b: &mut Vec<u8>, v: f64) {
    b.extend_from_slice(&v.to_be_bytes());
}
fn zeros(b: &mut Vec<u8>, n: usize) {
    b.extend(std::iter::repeat_n(0u8, n));
}

// ─── Shared block writers ────────────────────────────────────────────────────

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
    b.extend_from_slice(&0i16.to_le_bytes());
    b.extend_from_slice(&0i16.to_le_bytes());
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
    b.extend_from_slice(&0i16.to_le_bytes());
    b.extend_from_slice(&0i16.to_le_bytes());
    pu8(b, 0);
    zeros(b, 3);
}

fn dg4(b: &mut Vec<u8>, cg: i64, data: i64) {
    dg4_full(b, 0, cg, data, 0);
}

fn dg4_full(b: &mut Vec<u8>, next: i64, cg: i64, data: i64, rec_id_size: u8) {
    b.extend_from_slice(b"##DG");
    zeros(b, 4);
    pu64(b, 64);
    pu64(b, 4);
    pi64(b, next);
    pi64(b, cg);
    pi64(b, data);
    pi64(b, 0);
    pu8(b, rec_id_size);
    zeros(b, 7);
}

fn cg4(b: &mut Vec<u8>, cn: i64, cycles: u64, data_bytes: u32) {
    cg4_full(b, 0, cn, 0, cycles, 0, data_bytes);
}

fn cg4_full(
    b: &mut Vec<u8>,
    cg_next: i64,
    cn_first: i64,
    rec_id: u64,
    cycles: u64,
    flags: u16,
    data_bytes: u32,
) {
    b.extend_from_slice(b"##CG");
    zeros(b, 4);
    pu64(b, 104);
    pu64(b, 6);
    pi64(b, cg_next);
    pi64(b, cn_first);
    pi64(b, 0);
    pi64(b, 0);
    pi64(b, 0);
    pi64(b, 0);
    pu64(b, rec_id);
    pu64(b, cycles);
    pu16(b, flags);
    pu16(b, 0);
    zeros(b, 4);
    pu32(b, data_bytes);
    pu32(b, 0);
}

type CnDesc = (u8, u8, u8);
type CnSpan = (u32, u32);
type CnRefs = (i64, i64, i64);

fn cn4(b: &mut Vec<u8>, desc: CnDesc, span: CnSpan, refs: CnRefs) {
    cn4_full(b, desc, span, refs, 0, 0);
}

fn cn4_full(
    b: &mut Vec<u8>,
    desc: CnDesc,
    span: CnSpan,
    refs: CnRefs,
    composition: i64,
    data: i64,
) {
    let (cn_type, sync, dtype) = desc;
    let (byte_off, bits) = span;
    let (next, tx_ref, cc) = refs;
    b.extend_from_slice(b"##CN");
    zeros(b, 4);
    pu64(b, 160);
    pu64(b, 8);
    pi64(b, next);
    pi64(b, composition);
    pi64(b, tx_ref);
    pi64(b, 0);
    pi64(b, cc);
    pi64(b, data);
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

fn ca(b: &mut Vec<u8>, dims: &[u64]) {
    let len = 24 + 8 + 1 + 1 + 2 + 4 + 4 + 4 + (dims.len() * 8);
    b.extend_from_slice(b"##CA");
    zeros(b, 4);
    pu64(b, len as u64);
    pu64(b, 1);
    pi64(b, 0);
    pu8(b, 0);
    pu8(b, 0);
    pu16(b, dims.len() as u16);
    pu32(b, 0);
    pi32(b, 0);
    pu32(b, 0);
    for &d in dims {
        pu64(b, d);
    }
}

fn ld1_block(b: &mut Vec<u8>, data_ptr: i64, cycle_count: u64) {
    b.extend_from_slice(b"##LD");
    zeros(b, 4);
    pu64(b, 56);
    pu64(b, 2);
    pi64(b, 0);
    pi64(b, data_ptr);
    pu8(b, 1);
    pu8(b, 0);
    pu8(b, 0);
    pu8(b, 0);
    pu32(b, 1);
    pu64(b, cycle_count);
}

fn dt(b: &mut Vec<u8>, records: &[u8]) {
    let len = 24u64 + records.len() as u64;
    b.extend_from_slice(b"##DT");
    zeros(b, 4);
    pu64(b, len);
    pu64(b, 0);
    b.extend_from_slice(records);
}

fn dz(
    b: &mut Vec<u8>,
    org_block_type: &[u8; 2],
    zip_type: u8,
    zip_parameter: u32,
    org_data_length: u64,
    compressed: &[u8],
) {
    let total_len = 48u64 + compressed.len() as u64;
    b.extend_from_slice(b"##DZ");
    zeros(b, 4);
    pu64(b, total_len);
    pu64(b, 0);
    b.extend_from_slice(org_block_type);
    pu8(b, zip_type);
    pu8(b, 0);
    pu32(b, zip_parameter);
    pu64(b, org_data_length);
    pu64(b, compressed.len() as u64);
    b.extend_from_slice(compressed);
}

// ─── Fixture Builders ────────────────────────────────────────────────────────

pub fn build_int_linear_cc() -> Result<Vec<u8>> {
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
    cn4(&mut b, (2, 1, 4), (0, 64), (552, 1032, 0));
    debug_assert_eq!(b.len(), 552);
    cn4(&mut b, (0, 0, 2), (8, 8), (712, 1064, 1164));
    debug_assert_eq!(b.len(), 712);
    cn4(&mut b, (0, 0, 2), (9, 16), (872, 1096, 1164));
    debug_assert_eq!(b.len(), 872);
    cn4(&mut b, (0, 0, 4), (11, 32), (0, 1129, 1164));
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
    let mut records: Vec<u8> = Vec::with_capacity(60);
    for i in 0..4usize {
        pf64(&mut records, i as f64);
        records.push(raw_i8[i] as u8);
        records.extend_from_slice(&raw_i16[i].to_le_bytes());
        pf32(&mut records, raw_f32[i]);
    }
    debug_assert_eq!(records.len(), 60);
    dt(&mut b, &records);
    debug_assert_eq!(b.len(), 1344);
    Ok(b)
}

pub fn build_unsorted_multi_cg() -> Result<Vec<u8>> {
    const N_RECORDS_PER_CG: u64 = 1_000_000;
    const CG1_REC_ID: u64 = 1;
    const CG2_REC_ID: u64 = 2;
    const CG3_REC_ID: u64 = 3;
    const CG4_REC_ID: u64 = 4;

    const CG1_DATA_BYTES: u32 = 8;
    const CG2_DATA_BYTES: u32 = 8;
    const CG3_DATA_BYTES: u32 = 16;
    const CG4_DATA_BYTES: u32 = 20;

    const DG_OFFSET: i64 = 224;
    const CG1_OFFSET: i64 = 288;
    const CG2_OFFSET: i64 = 392;
    const CG3_OFFSET: i64 = 496;
    const CG4_OFFSET: i64 = 600;
    const CN1_F64_OFFSET: i64 = 704;
    const CN2_F32_OFFSET: i64 = 864;
    const CN2_I32_OFFSET: i64 = 1024;
    const CN3_F64_OFFSET: i64 = 1184;
    const CN3_F32_OFFSET: i64 = 1344;
    const CN3_I32_OFFSET: i64 = 1504;
    const CN4_F64_1_OFFSET: i64 = 1664;
    const CN4_F64_2_OFFSET: i64 = 1824;
    const CN4_F32_OFFSET: i64 = 1984;
    const TX_CG1_CH1_OFFSET: i64 = 2144;
    const TX_CG2_CH1_OFFSET: i64 = 2176;
    const TX_CG2_CH2_OFFSET: i64 = 2208;
    const TX_CG3_CH1_OFFSET: i64 = 2240;
    const TX_CG3_CH2_OFFSET: i64 = 2272;
    const TX_CG3_CH3_OFFSET: i64 = 2304;
    const TX_CG4_CH1_OFFSET: i64 = 2336;
    const TX_CG4_CH2_OFFSET: i64 = 2368;
    const TX_CG4_CH3_OFFSET: i64 = 2400;
    const DT_OFFSET: i64 = 2432;

    let mut b: Vec<u8> = Vec::with_capacity(64_000_000);
    id_block(&mut b);
    debug_assert_eq!(b.len(), 64);
    hd4(&mut b, DG_OFFSET, 168);
    debug_assert_eq!(b.len(), 168);
    fh(&mut b);
    debug_assert_eq!(b.len(), 224);
    dg4_full(&mut b, 0, CG1_OFFSET, DT_OFFSET, 2);
    debug_assert_eq!(b.len(), 288);

    cg4_full(
        &mut b,
        CG2_OFFSET,
        CN1_F64_OFFSET,
        CG1_REC_ID,
        N_RECORDS_PER_CG,
        0,
        CG1_DATA_BYTES,
    );
    debug_assert_eq!(b.len(), 392);
    cg4_full(
        &mut b,
        CG3_OFFSET,
        CN2_F32_OFFSET,
        CG2_REC_ID,
        N_RECORDS_PER_CG,
        0,
        CG2_DATA_BYTES,
    );
    debug_assert_eq!(b.len(), 496);
    cg4_full(
        &mut b,
        CG4_OFFSET,
        CN3_F64_OFFSET,
        CG3_REC_ID,
        N_RECORDS_PER_CG,
        0,
        CG3_DATA_BYTES,
    );
    debug_assert_eq!(b.len(), 600);
    cg4_full(
        &mut b,
        0,
        CN4_F64_1_OFFSET,
        CG4_REC_ID,
        N_RECORDS_PER_CG,
        0,
        CG4_DATA_BYTES,
    );
    debug_assert_eq!(b.len(), 704);

    cn4(&mut b, (0, 0, 4), (0, 64), (0, TX_CG1_CH1_OFFSET, 0));
    debug_assert_eq!(b.len(), 864);
    cn4(
        &mut b,
        (0, 0, 4),
        (0, 32),
        (CN2_I32_OFFSET, TX_CG2_CH1_OFFSET, 0),
    );
    debug_assert_eq!(b.len(), 1024);
    cn4(&mut b, (0, 0, 2), (4, 32), (0, TX_CG2_CH2_OFFSET, 0));
    debug_assert_eq!(b.len(), 1184);
    cn4(
        &mut b,
        (0, 0, 4),
        (0, 64),
        (CN3_F32_OFFSET, TX_CG3_CH1_OFFSET, 0),
    );
    debug_assert_eq!(b.len(), 1344);
    cn4(
        &mut b,
        (0, 0, 4),
        (8, 32),
        (CN3_I32_OFFSET, TX_CG3_CH2_OFFSET, 0),
    );
    debug_assert_eq!(b.len(), 1504);
    cn4(&mut b, (0, 0, 2), (12, 32), (0, TX_CG3_CH3_OFFSET, 0));
    debug_assert_eq!(b.len(), 1664);
    cn4(
        &mut b,
        (0, 0, 4),
        (0, 64),
        (CN4_F64_2_OFFSET, TX_CG4_CH1_OFFSET, 0),
    );
    debug_assert_eq!(b.len(), 1824);
    cn4(
        &mut b,
        (0, 0, 4),
        (8, 64),
        (CN4_F32_OFFSET, TX_CG4_CH2_OFFSET, 0),
    );
    debug_assert_eq!(b.len(), 1984);
    cn4(&mut b, (0, 0, 4), (16, 32), (0, TX_CG4_CH3_OFFSET, 0));
    debug_assert_eq!(b.len(), 2144);

    for name in [
        "cg1_ch1", "cg2_ch1", "cg2_ch2", "cg3_ch1", "cg3_ch2", "cg3_ch3", "cg4_ch1", "cg4_ch2",
        "cg4_ch3",
    ] {
        tx(&mut b, name);
    }
    debug_assert_eq!(b.len(), DT_OFFSET as usize);

    let mut dt_data: Vec<u8> = Vec::with_capacity(
        N_RECORDS_PER_CG as usize
            * (CG1_DATA_BYTES + CG2_DATA_BYTES + CG3_DATA_BYTES + CG4_DATA_BYTES) as usize,
    );
    for i in 0..N_RECORDS_PER_CG {
        dt_data.extend_from_slice(&1u16.to_le_bytes());
        dt_data.extend_from_slice(&(i as f64).to_le_bytes());

        dt_data.extend_from_slice(&2u16.to_le_bytes());
        dt_data.extend_from_slice(&(i as f32).to_le_bytes());
        dt_data.extend_from_slice(&(i as i32).to_le_bytes());

        dt_data.extend_from_slice(&3u16.to_le_bytes());
        dt_data.extend_from_slice(&(i as f64).to_le_bytes());
        dt_data.extend_from_slice(&(i as f32).to_le_bytes());
        dt_data.extend_from_slice(&(i as i32).to_le_bytes());

        dt_data.extend_from_slice(&4u16.to_le_bytes());
        dt_data.extend_from_slice(&(i as f64).to_le_bytes());
        dt_data.extend_from_slice(&((i + 1) as f64).to_le_bytes());
        dt_data.extend_from_slice(&(i as f32).to_le_bytes());
    }
    dt(&mut b, &dt_data);
    Ok(b)
}

pub fn build_unsorted_vlsd_vlsc() -> Result<Vec<u8>> {
    const N_RECORDS_PER_CG: u64 = 100_000;
    const CG1_REC_ID: u64 = 1;
    const CG2_REC_ID: u64 = 2;
    const CG_F_VLSD: u16 = 1 << 0;

    const DG_OFFSET: i64 = 224;
    const CG1_OFFSET: i64 = 288;
    const CG2_OFFSET: i64 = 392;
    const CN2_OFFSET: i64 = 496;
    const TX_CG2_CH1_OFFSET: i64 = 656;
    const DT_OFFSET: i64 = 695;

    let mut b: Vec<u8> = Vec::with_capacity(6_000_000);
    id_block(&mut b);
    debug_assert_eq!(b.len(), 64);
    hd4(&mut b, DG_OFFSET, 168);
    debug_assert_eq!(b.len(), 168);
    fh(&mut b);
    debug_assert_eq!(b.len(), 224);
    dg4_full(&mut b, 0, CG1_OFFSET, DT_OFFSET, 2);
    debug_assert_eq!(b.len(), 288);

    cg4_full(
        &mut b,
        CG2_OFFSET,
        0,
        CG1_REC_ID,
        N_RECORDS_PER_CG,
        CG_F_VLSD,
        0,
    );
    debug_assert_eq!(b.len(), 392);
    cg4_full(&mut b, 0, CN2_OFFSET, CG2_REC_ID, N_RECORDS_PER_CG, 0, 0);
    debug_assert_eq!(b.len(), 496);
    cn4_full(
        &mut b,
        (0, 0, 7),
        (0, 0),
        (0, TX_CG2_CH1_OFFSET, 0),
        0,
        CG1_OFFSET,
    );
    debug_assert_eq!(b.len(), 656);
    tx(&mut b, "vlsd_target_ch");
    debug_assert_eq!(b.len(), DT_OFFSET as usize);

    let mut dt_data: Vec<u8> = Vec::with_capacity(6_000_000);
    for i in 0..N_RECORDS_PER_CG {
        let utf8_data = format!("record_{}_data_with_variable_length_content", i);
        let utf8_bytes = utf8_data.as_bytes();
        dt_data.extend_from_slice(&1u16.to_le_bytes());
        dt_data.extend_from_slice(&(utf8_bytes.len() as u32).to_le_bytes());
        dt_data.extend_from_slice(utf8_bytes);
    }
    dt(&mut b, &dt_data);
    Ok(b)
}

pub fn build_synthetic_array() -> Result<Vec<u8>> {
    let mut b = Vec::with_capacity(2048);
    id_block(&mut b);
    debug_assert_eq!(b.len(), 64);
    hd4(&mut b, 224, 168);
    debug_assert_eq!(b.len(), 168);
    fh(&mut b);
    debug_assert_eq!(b.len(), 224);
    dg4(&mut b, 288, 840);
    debug_assert_eq!(b.len(), 288);
    cg4(&mut b, 392, 3, 40);
    debug_assert_eq!(b.len(), 392);
    cn4(&mut b, (2, 1, 4), (0, 64), (552, 776, 0));
    debug_assert_eq!(b.len(), 552);
    cn4_full(&mut b, (0, 0, 4), (8, 64), (0, 808, 0), 712, 0);
    debug_assert_eq!(b.len(), 712);
    ca(&mut b, &[2, 2]);
    debug_assert_eq!(b.len(), 776);
    tx(&mut b, "time_ch");
    debug_assert_eq!(b.len(), 808);
    tx(&mut b, "array2d");
    debug_assert_eq!(b.len(), 840);

    let mut dt_data = Vec::with_capacity(120);
    for i in 0..3 {
        dt_data.extend_from_slice(&(i as f64).to_le_bytes());
        for j in 0..4 {
            let val = (i * 4 + j + 1) as f64;
            dt_data.extend_from_slice(&val.to_le_bytes());
        }
    }
    dt(&mut b, &dt_data);
    Ok(b)
}

pub fn build_synthetic_dz_deflate() -> Result<Vec<u8>> {
    let mut b = Vec::with_capacity(2048);
    id_block(&mut b);
    debug_assert_eq!(b.len(), 64);
    hd4(&mut b, 224, 168);
    debug_assert_eq!(b.len(), 168);
    fh(&mut b);
    debug_assert_eq!(b.len(), 224);
    dg4(&mut b, 288, 584);
    debug_assert_eq!(b.len(), 288);
    cg4(&mut b, 392, 10, 8);
    debug_assert_eq!(b.len(), 392);
    cn4(&mut b, (2, 1, 4), (0, 64), (0, 552, 0));
    debug_assert_eq!(b.len(), 552);
    tx(&mut b, "time_ch");
    debug_assert_eq!(b.len(), 584);

    let mut raw_data = Vec::with_capacity(80);
    for i in 0..10 {
        raw_data.extend_from_slice(&(i as f64).to_le_bytes());
    }
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(&raw_data)?;
    let compressed = encoder.finish()?;
    dz(&mut b, b"DT", 0, 0, raw_data.len() as u64, &compressed);
    Ok(b)
}

pub fn build_be_scalars() -> Result<Vec<u8>> {
    let mut b: Vec<u8> = Vec::with_capacity(1061);
    id_block(&mut b);
    debug_assert_eq!(b.len(), 64);
    hd4(&mut b, 224, 168);
    debug_assert_eq!(b.len(), 168);
    fh(&mut b);
    debug_assert_eq!(b.len(), 224);
    dg4(&mut b, 288, 965);
    debug_assert_eq!(b.len(), 288);
    cg4(&mut b, 392, 4, 18);
    debug_assert_eq!(b.len(), 392);
    cn4(&mut b, (2, 1, 4), (0, 64), (552, 872, 0));
    debug_assert_eq!(b.len(), 552);
    cn4(&mut b, (0, 0, 3), (8, 16), (712, 903, 0));
    debug_assert_eq!(b.len(), 712);
    cn4(&mut b, (0, 0, 5), (10, 64), (0, 934, 0));
    debug_assert_eq!(b.len(), 872);
    tx(&mut b, "master");
    debug_assert_eq!(b.len(), 903);
    tx(&mut b, "be_i16");
    debug_assert_eq!(b.len(), 934);
    tx(&mut b, "be_f64");
    debug_assert_eq!(b.len(), 965);

    let raw_i16_be: [i16; 4] = [-100, 0, 100, 200];
    let raw_f64_be: [f64; 4] = [1.5, 2.5, 3.5, 4.5];
    let mut recs: Vec<u8> = Vec::with_capacity(72);
    for i in 0..4 {
        pf64(&mut recs, i as f64);
        pi16_be(&mut recs, raw_i16_be[i]);
        pf64_be(&mut recs, raw_f64_be[i]);
    }
    dt(&mut b, &recs);
    debug_assert_eq!(b.len(), 1061);
    Ok(b)
}

pub fn build_complex_f32_le() -> Result<Vec<u8>> {
    let mut b: Vec<u8> = Vec::with_capacity(863);
    id_block(&mut b);
    debug_assert_eq!(b.len(), 64);
    hd4(&mut b, 224, 168);
    debug_assert_eq!(b.len(), 168);
    fh(&mut b);
    debug_assert_eq!(b.len(), 224);
    dg4(&mut b, 288, 775);
    debug_assert_eq!(b.len(), 288);
    cg4(&mut b, 392, 4, 16);
    debug_assert_eq!(b.len(), 392);
    cn4(&mut b, (2, 1, 4), (0, 64), (552, 712, 0));
    debug_assert_eq!(b.len(), 552);
    cn4(&mut b, (0, 0, 15), (8, 64), (0, 743, 0));
    debug_assert_eq!(b.len(), 712);
    tx(&mut b, "master");
    debug_assert_eq!(b.len(), 743);
    tx(&mut b, "cx32_ch");
    debug_assert_eq!(b.len(), 775);

    let samples: [(f32, f32); 4] = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)];
    let mut recs: Vec<u8> = Vec::with_capacity(64);
    for (i, (re, im)) in samples.iter().enumerate() {
        pf64(&mut recs, i as f64);
        pf32(&mut recs, *re);
        pf32(&mut recs, *im);
    }
    dt(&mut b, &recs);
    debug_assert_eq!(b.len(), 863);
    Ok(b)
}

pub fn build_array_f64_le() -> Result<Vec<u8>> {
    let mut b: Vec<u8> = Vec::with_capacity(983);
    id_block(&mut b);
    debug_assert_eq!(b.len(), 64);
    hd4(&mut b, 224, 168);
    debug_assert_eq!(b.len(), 168);
    fh(&mut b);
    debug_assert_eq!(b.len(), 224);
    dg4(&mut b, 288, 831);
    debug_assert_eq!(b.len(), 288);
    cg4(&mut b, 392, 4, 32);
    debug_assert_eq!(b.len(), 392);
    cn4(&mut b, (2, 1, 4), (0, 64), (552, 712, 0));
    debug_assert_eq!(b.len(), 552);
    cn4_full(&mut b, (0, 0, 4), (8, 64), (0, 743, 0), 775, 0);
    debug_assert_eq!(b.len(), 712);
    tx(&mut b, "master");
    debug_assert_eq!(b.len(), 743);
    tx(&mut b, "arr_f64");
    debug_assert_eq!(b.len(), 775);
    ca(&mut b, &[3]);
    debug_assert_eq!(b.len(), 831);

    let mut recs: Vec<u8> = Vec::with_capacity(128);
    for cycle in 0..4u64 {
        pf64(&mut recs, cycle as f64);
        for elem in 1..=3u64 {
            pf64(&mut recs, (cycle * 3 + elem) as f64);
        }
    }
    dt(&mut b, &recs);
    debug_assert_eq!(b.len(), 983);
    Ok(b)
}

pub fn build_ld_be_channels() -> Result<Vec<u8>> {
    let mut b: Vec<u8> = Vec::with_capacity(2040);
    id_block(&mut b);
    debug_assert_eq!(b.len(), 64);
    hd4(&mut b, 224, 168);
    debug_assert_eq!(b.len(), 168);
    fh(&mut b);
    debug_assert_eq!(b.len(), 224);
    dg4_full(&mut b, 288, 480, 1656, 0);
    debug_assert_eq!(b.len(), 288);
    dg4_full(&mut b, 352, 584, 1712, 0);
    debug_assert_eq!(b.len(), 352);
    dg4_full(&mut b, 416, 688, 1768, 0);
    debug_assert_eq!(b.len(), 416);
    dg4_full(&mut b, 0, 792, 1824, 0);
    debug_assert_eq!(b.len(), 480);
    cg4(&mut b, 896, 4, 2);
    debug_assert_eq!(b.len(), 584);
    cg4(&mut b, 1056, 4, 8);
    debug_assert_eq!(b.len(), 688);
    cg4(&mut b, 1216, 4, 2);
    debug_assert_eq!(b.len(), 792);
    cg4(&mut b, 1376, 4, 4);
    debug_assert_eq!(b.len(), 896);
    cn4(&mut b, (0, 0, 3), (0, 16), (0, 1536, 0));
    debug_assert_eq!(b.len(), 1056);
    cn4(&mut b, (0, 0, 5), (0, 64), (0, 1566, 0));
    debug_assert_eq!(b.len(), 1216);
    cn4(&mut b, (0, 0, 1), (0, 16), (0, 1596, 0));
    debug_assert_eq!(b.len(), 1376);
    cn4(&mut b, (0, 0, 5), (0, 32), (0, 1626, 0));
    debug_assert_eq!(b.len(), 1536);
    tx(&mut b, "bei16");
    debug_assert_eq!(b.len(), 1566);
    tx(&mut b, "bef64");
    debug_assert_eq!(b.len(), 1596);
    tx(&mut b, "beu16");
    debug_assert_eq!(b.len(), 1626);
    tx(&mut b, "bef32");
    debug_assert_eq!(b.len(), 1656);
    ld1_block(&mut b, 1880, 4);
    debug_assert_eq!(b.len(), 1712);
    ld1_block(&mut b, 1912, 4);
    debug_assert_eq!(b.len(), 1768);
    ld1_block(&mut b, 1968, 4);
    debug_assert_eq!(b.len(), 1824);
    ld1_block(&mut b, 2000, 4);
    debug_assert_eq!(b.len(), 1880);

    {
        let mut recs: Vec<u8> = Vec::with_capacity(8);
        for v in [-100i16, 0, 100, 200] {
            pi16_be(&mut recs, v);
        }
        dt(&mut b, &recs);
        debug_assert_eq!(b.len(), 1912);
    }
    {
        let mut recs: Vec<u8> = Vec::with_capacity(32);
        for v in [1.5f64, 2.5, 3.5, 4.5] {
            pf64_be(&mut recs, v);
        }
        dt(&mut b, &recs);
        debug_assert_eq!(b.len(), 1968);
    }
    {
        let mut recs: Vec<u8> = Vec::with_capacity(8);
        for v in [100u16, 200, 300, 400] {
            pu16_be(&mut recs, v);
        }
        dt(&mut b, &recs);
        debug_assert_eq!(b.len(), 2000);
    }
    {
        let mut recs: Vec<u8> = Vec::with_capacity(16);
        for v in [1.5f32, 2.5, 3.5, 4.5] {
            pf32_be(&mut recs, v);
        }
        dt(&mut b, &recs);
        debug_assert_eq!(b.len(), 2040);
    }
    Ok(b)
}
