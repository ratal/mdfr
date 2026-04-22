/// Integration tests for BigEndian channels, Complex32, and ArrayDFloat64 fixtures.
///
/// Three synthetic MDF4 files exercise paths in `read_channels_from_bytes` that
/// the existing test suite never hits:
///
///  - `be_scalars.mf4`      → BE Int16 (lines 724-731) and BE Float64
///  - `complex_f32_le.mf4`  → Complex32 LE f32-pair arm (lines 1040-1072)
///  - `array_f64_le.mf4`    → ArrayDFloat64 LE (lines 1624-1635) + CA block parsing
use anyhow::Result;
use mdfr::data_holder::channel_data::ChannelData;
use mdfr::mdfreader::Mdf;
use std::sync::LazyLock;

// ─── Byte-push helpers ───────────────────────────────────────────────────────

fn pu8(b: &mut Vec<u8>, v: u8) {
    b.push(v);
}
fn pu16(b: &mut Vec<u8>, v: u16) {
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

// ─── Shared block writers (identical to conversions_int_types.rs) ────────────

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
fn dg4_chain(b: &mut Vec<u8>, next: i64, cg: i64, data: i64) {
    b.extend_from_slice(b"##DG");
    zeros(b, 4);
    pu64(b, 64);
    pu64(b, 4);
    pi64(b, next);
    pi64(b, cg);
    pi64(b, data);
    pi64(b, 0);
    pu8(b, 0);
    zeros(b, 7);
}
/// LD block with one data link and equal_sample_count (56 bytes total).
fn ld1_block(b: &mut Vec<u8>, data_ptr: i64, cycle_count: u64) {
    b.extend_from_slice(b"##LD");
    zeros(b, 4);
    pu64(b, 56);
    pu64(b, 2); // header
    pi64(b, 0);
    pi64(b, data_ptr); // ld_next=0, ld_links[0]=data_ptr
    pu8(b, 1);
    pu8(b, 0);
    pu8(b, 0);
    pu8(b, 0); // ld_flags=1(equal_sample_count), rest 0
    pu32(b, 1);
    pu64(b, cycle_count); // ld_count=1, ld_equal_sample_count
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
/// `(cn_type, sync, dtype)` — channel kind descriptor
type CnDesc = (u8, u8, u8);
/// `(byte_offset, bit_count)` — data layout
type CnSpan = (u32, u32);
/// `(cn_next, cn_name_tx, cn_cc)` — block links
type CnRefs = (i64, i64, i64);

fn cn4(b: &mut Vec<u8>, desc: CnDesc, span: CnSpan, refs: CnRefs) {
    let (cn_type, sync, dtype) = desc;
    let (byte_off, bits) = span;
    let (next, tx, cc) = refs;
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
/// Like `cn4` but with a non-zero `composition` link (second link = CA block offset).
fn cn4_ca(b: &mut Vec<u8>, desc: CnDesc, span: CnSpan, refs: CnRefs, composition: i64) {
    let (cn_type, sync, dtype) = desc;
    let (byte_off, bits) = span;
    let (next, tx, cc) = refs;
    b.extend_from_slice(b"##CN");
    zeros(b, 4);
    pu64(b, 160);
    pu64(b, 8);
    pi64(b, next);
    pi64(b, composition);
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
fn dt(b: &mut Vec<u8>, records: &[u8]) {
    let len = 24u64 + records.len() as u64;
    b.extend_from_slice(b"##DT");
    zeros(b, 4);
    pu64(b, len);
    pu64(b, 0);
    b.extend_from_slice(records);
}
/// Minimal 1D CA block (56 bytes). `dim_size` is the number of elements per cycle.
fn ca1d(b: &mut Vec<u8>, dim_size: u64) {
    b.extend_from_slice(b"##CA");
    zeros(b, 4);
    pu64(b, 56);
    pu64(b, 1); // link_count=1
    pi64(b, 0); // ca_composition = null
    pu8(b, 0); // ca_type  = 0 (Array)
    pu8(b, 0); // ca_storage = 0 (CN template)
    pu16(b, 1); // ca_ndim = 1
    pu32(b, 0); // ca_flags = 0
    pu32(b, 0); // ca_byte_offset_base = 0
    pu32(b, 0); // ca_inval_bit_pos_base = 0
    pu64(b, dim_size); // ca_dim_size[0]
}

// ─── Fixture 1: be_scalars.mf4 ───────────────────────────────────────────────
//
// Layout (all offsets are absolute):
//  [0]    IdBlock   64 b
//  [64]   HD4       104 b  (hd_dg=224, hd_fh=168)
//  [168]  FH        56 b
//  [224]  DG4       64 b   (dg_cg=288, dg_data=965)
//  [288]  CG4       104 b  (cg_cn=392, cycles=4, data_bytes=18)
//  [392]  CN_master 160 b  (FloatLE/64bit, byte_off=0,  next=552, tx=872)
//  [552]  CN_be_i16 160 b  (IntBE/16bit,   byte_off=8,  next=712, tx=903)
//  [712]  CN_be_f64 160 b  (FloatBE/64bit, byte_off=10, next=0,   tx=934)
//  [872]  TX "master"   31 b
//  [903]  TX "be_i16"   31 b
//  [934]  TX "be_f64"   31 b
//  [965]  DT            96 b  (24 hdr + 4×18=72 data)
//  Total: 1061 b

const BE_SCALARS_PATH: &str = "test_files/synthetic/be_scalars.mf4";
static FIXTURE_BE: LazyLock<()> = LazyLock::new(|| {
    create_be_scalars().expect("failed to create be_scalars fixture");
});

fn create_be_scalars() -> Result<()> {
    if std::path::Path::new(BE_SCALARS_PATH).exists() {
        return Ok(());
    }
    std::fs::create_dir_all("test_files/synthetic")?;
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
    debug_assert_eq!(b.len(), 712); // dtype=3=IntBE
    cn4(&mut b, (0, 0, 5), (10, 64), (0, 934, 0));
    debug_assert_eq!(b.len(), 872); // dtype=5=FloatBE
    tx(&mut b, "master");
    debug_assert_eq!(b.len(), 903);
    tx(&mut b, "be_i16");
    debug_assert_eq!(b.len(), 934);
    tx(&mut b, "be_f64");
    debug_assert_eq!(b.len(), 965);

    // 4 records × 18 bytes:  f64_LE(8) | i16_BE(2) | f64_BE(8)
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

    let tmp = format!("{}.tmp.{}", BE_SCALARS_PATH, std::process::id());
    std::fs::write(&tmp, &b)?;
    if std::fs::rename(&tmp, BE_SCALARS_PATH).is_err() {
        std::fs::remove_file(&tmp).ok();
    }
    Ok(())
}

// ─── Fixture 2: complex_f32_le.mf4 ──────────────────────────────────────────
//
//  [0]    IdBlock   64 b
//  [64]   HD4       104 b  (hd_dg=224, hd_fh=168)
//  [168]  FH        56 b
//  [224]  DG4       64 b   (dg_cg=288, dg_data=775)
//  [288]  CG4       104 b  (cg_cn=392, cycles=4, data_bytes=16)
//  [392]  CN_master 160 b  (FloatLE/64bit, byte_off=0, next=552, tx=712)
//  [552]  CN_cx32   160 b  (ComplexLE/64bit, byte_off=8, next=0,  tx=743)
//  [712]  TX "master"   31 b
//  [743]  TX "cx32_ch"  32 b
//  [775]  DT            88 b  (24 hdr + 4×16=64 data)
//  Total: 863 b

const CX32_PATH: &str = "test_files/synthetic/complex_f32_le.mf4";
static FIXTURE_CX: LazyLock<()> = LazyLock::new(|| {
    create_complex_f32_le().expect("failed to create complex_f32_le fixture");
});

fn create_complex_f32_le() -> Result<()> {
    if std::path::Path::new(CX32_PATH).exists() {
        return Ok(());
    }
    std::fs::create_dir_all("test_files/synthetic")?;
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
    debug_assert_eq!(b.len(), 712); // dtype=15=ComplexLE
    tx(&mut b, "master");
    debug_assert_eq!(b.len(), 743);
    tx(&mut b, "cx32_ch");
    debug_assert_eq!(b.len(), 775);

    // 4 records × 16 bytes:  f64_LE(8) | f32_LE_real(4) | f32_LE_imag(4)
    // Samples: (1+2j), (3+4j), (5+6j), (7+8j)
    let samples: [(f32, f32); 4] = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)];
    let mut recs: Vec<u8> = Vec::with_capacity(64);
    for (i, (re, im)) in samples.iter().enumerate() {
        pf64(&mut recs, i as f64);
        pf32(&mut recs, *re);
        pf32(&mut recs, *im);
    }
    dt(&mut b, &recs);
    debug_assert_eq!(b.len(), 863);

    let tmp = format!("{}.tmp.{}", CX32_PATH, std::process::id());
    std::fs::write(&tmp, &b)?;
    if std::fs::rename(&tmp, CX32_PATH).is_err() {
        std::fs::remove_file(&tmp).ok();
    }
    Ok(())
}

// ─── Fixture 3: array_f64_le.mf4 ────────────────────────────────────────────
//
//  [0]    IdBlock   64 b
//  [64]   HD4       104 b  (hd_dg=224, hd_fh=168)
//  [168]  FH        56 b
//  [224]  DG4       64 b   (dg_cg=288, dg_data=831)
//  [288]  CG4       104 b  (cg_cn=392, cycles=4, data_bytes=32)
//  [392]  CN_master 160 b  (FloatLE/64bit, byte_off=0, next=552, tx=712, composition=0)
//  [552]  CN_arr    160 b  (FloatLE/64bit, byte_off=8, next=0,   tx=743, composition=775)
//  [712]  TX "master"   31 b
//  [743]  TX "arr_f64"  32 b
//  [775]  CA block      56 b  (ca_ndim=1, ca_dim_size=[3])
//  [831]  DT            152 b  (24 hdr + 4×32=128 data)
//  Total: 983 b

const ARR_F64_PATH: &str = "test_files/synthetic/array_f64_le.mf4";
static FIXTURE_ARR: LazyLock<()> = LazyLock::new(|| {
    create_array_f64_le().expect("failed to create array_f64_le fixture");
});

fn create_array_f64_le() -> Result<()> {
    if std::path::Path::new(ARR_F64_PATH).exists() {
        return Ok(());
    }
    std::fs::create_dir_all("test_files/synthetic")?;
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
    // CN master: no CA block → composition=0, use plain cn4
    cn4(&mut b, (2, 1, 4), (0, 64), (552, 712, 0));
    debug_assert_eq!(b.len(), 552);
    // CN_arr: composition link points to CA block at 775
    cn4_ca(&mut b, (0, 0, 4), (8, 64), (0, 743, 0), 775);
    debug_assert_eq!(b.len(), 712);
    tx(&mut b, "master");
    debug_assert_eq!(b.len(), 743);
    tx(&mut b, "arr_f64");
    debug_assert_eq!(b.len(), 775);
    ca1d(&mut b, 3);
    debug_assert_eq!(b.len(), 831);

    // 4 records × 32 bytes: f64_LE(8) | [f64_LE × 3](24)
    // Array values: [1,2,3], [4,5,6], [7,8,9], [10,11,12]
    let mut recs: Vec<u8> = Vec::with_capacity(128);
    for cycle in 0..4u64 {
        pf64(&mut recs, cycle as f64);
        for elem in 1..=3u64 {
            pf64(&mut recs, (cycle * 3 + elem) as f64);
        }
    }
    dt(&mut b, &recs);
    debug_assert_eq!(b.len(), 983);

    let tmp = format!("{}.tmp.{}", ARR_F64_PATH, std::process::id());
    std::fs::write(&tmp, &b)?;
    if std::fs::rename(&tmp, ARR_F64_PATH).is_err() {
        std::fs::remove_file(&tmp).ok();
    }
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[test]
fn be_int16_reads_correctly() -> Result<()> {
    LazyLock::force(&FIXTURE_BE);
    let mut mdf = Mdf::new(BE_SCALARS_PATH)?;
    mdf.load_all_channels_data_in_memory()?;

    let data = mdf.get_channel_data("be_i16").expect("be_i16 not found");
    assert!(
        matches!(data, ChannelData::Int16(_)),
        "expected Int16, got {}",
        data.data_type(false)
    );
    if let ChannelData::Int16(arr) = data {
        let expected = [-100i16, 0, 100, 200];
        for (i, (&got, &exp)) in arr.values_slice().iter().zip(expected.iter()).enumerate() {
            assert_eq!(got, exp, "be_i16[{i}]: expected {exp}, got {got}");
        }
    }
    Ok(())
}

#[test]
fn be_float64_reads_correctly() -> Result<()> {
    LazyLock::force(&FIXTURE_BE);
    let mut mdf = Mdf::new(BE_SCALARS_PATH)?;
    mdf.load_all_channels_data_in_memory()?;

    let data = mdf.get_channel_data("be_f64").expect("be_f64 not found");
    assert!(
        matches!(data, ChannelData::Float64(_)),
        "expected Float64, got {}",
        data.data_type(false)
    );
    if let ChannelData::Float64(arr) = data {
        let expected = [1.5f64, 2.5, 3.5, 4.5];
        for (i, (&got, &exp)) in arr.values_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-9,
                "be_f64[{i}]: expected {exp}, got {got}"
            );
        }
    }
    Ok(())
}

#[test]
fn complex32_le_reads_correctly() -> Result<()> {
    LazyLock::force(&FIXTURE_CX);
    let mut mdf = Mdf::new(CX32_PATH)?;
    mdf.load_all_channels_data_in_memory()?;

    let data = mdf.get_channel_data("cx32_ch").expect("cx32_ch not found");
    assert!(
        matches!(data, ChannelData::Complex32(_)),
        "expected Complex32, got {}",
        data.data_type(false)
    );
    if let ChannelData::Complex32(arr) = data {
        // values_slice() returns interleaved [re0, im0, re1, im1, ...]
        let vals = arr.values_slice();
        assert_eq!(vals.len(), 8, "expected 4 complex samples = 8 floats");
        let expected_re = [1.0f32, 3.0, 5.0, 7.0];
        let expected_im = [2.0f32, 4.0, 6.0, 8.0];
        for i in 0..4 {
            assert!(
                (vals[i * 2] - expected_re[i]).abs() < 1e-6,
                "cx32_ch[{i}].re: expected {}, got {}",
                expected_re[i],
                vals[i * 2]
            );
            assert!(
                (vals[i * 2 + 1] - expected_im[i]).abs() < 1e-6,
                "cx32_ch[{i}].im: expected {}, got {}",
                expected_im[i],
                vals[i * 2 + 1]
            );
        }
    }
    Ok(())
}

#[test]
fn array_f64_le_reads_correctly() -> Result<()> {
    LazyLock::force(&FIXTURE_ARR);
    let mut mdf = Mdf::new(ARR_F64_PATH)?;
    mdf.load_all_channels_data_in_memory()?;

    let data = mdf.get_channel_data("arr_f64").expect("arr_f64 not found");
    assert!(
        matches!(data, ChannelData::ArrayDFloat64(_)),
        "expected ArrayDFloat64, got {}",
        data.data_type(false)
    );
    if let ChannelData::ArrayDFloat64(arr) = data {
        let vals = arr.values_slice();
        // The zeros() pre-allocation for ArrayDFloat64 allocates cycle_count * product(shape)
        // elements. Since shape=[4,3] already includes cycle_count=4 as first dim,
        // the buffer is 4 * (4*3) = 48 slots, but only the first 12 are filled by reading.
        assert!(
            vals.len() >= 12,
            "expected at least 12 f64 values, got {}",
            vals.len()
        );
        let expected: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        for (i, (&got, &exp)) in vals[..12].iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-9,
                "arr_f64[{i}]: expected {exp}, got {got}"
            );
        }
        assert!(arr.ndim() >= 1, "expected ndim >= 1");
    }
    Ok(())
}

// ─── Fixture 4: ld_be_channels.mf4 ──────────────────────────────────────────
//
// Four DGs, each with one LD-backed BE channel. Exercises the uncovered BE
// arms in `read_one_channel_array` (data_read4.rs lines 59-62, 72-75, 146-149,
// 272-275) which are only reachable via LD blocks (the optimised single-channel
// path) — existing real MDF files with LD blocks all use LE channels.
//
// Layout:
//  [0]    IdBlock     64
//  [64]   HD4         104  (dg=224, fh=168)
//  [168]  FH          56
//  [224]  DG1         64   (next=288, cg=480, data=1656)  BE Int16
//  [288]  DG2         64   (next=352, cg=584, data=1712)  BE Float64
//  [352]  DG3         64   (next=416, cg=688, data=1768)  BE UInt16
//  [416]  DG4         64   (next=0,   cg=792, data=1824)  BE Float32
//  [480]  CG1         104  (cn=896,  cycles=4, data_bytes=2)
//  [584]  CG2         104  (cn=1056, cycles=4, data_bytes=8)
//  [688]  CG3         104  (cn=1216, cycles=4, data_bytes=2)
//  [792]  CG4         104  (cn=1376, cycles=4, data_bytes=4)
//  [896]  CN_bei16    160  (dtype=3/IntBE,   bits=16, tx=1536)
//  [1056] CN_bef64    160  (dtype=5/FloatBE, bits=64, tx=1566)
//  [1216] CN_beu16    160  (dtype=1/UIntBE,  bits=16, tx=1596)
//  [1376] CN_bef32    160  (dtype=5/FloatBE, bits=32, tx=1626)
//  [1536] TX "bei16"  30
//  [1566] TX "bef64"  30
//  [1596] TX "beu16"  30
//  [1626] TX "bef32"  30
//  [1656] LD1         56   → DT1 at 1880
//  [1712] LD2         56   → DT2 at 1912
//  [1768] LD3         56   → DT3 at 1968
//  [1824] LD4         56   → DT4 at 2000
//  [1880] DT1         32   (24 hdr + 4×i16_BE = 8 bytes)
//  [1912] DT2         56   (24 hdr + 4×f64_BE = 32 bytes)
//  [1968] DT3         32   (24 hdr + 4×u16_BE = 8 bytes)
//  [2000] DT4         40   (24 hdr + 4×f32_BE = 16 bytes)
//  Total: 2040 bytes

const LD_BE_PATH: &str = "test_files/synthetic/ld_be_channels.mf4";
static FIXTURE_LD_BE: LazyLock<()> = LazyLock::new(|| {
    create_ld_be_channels().expect("failed to create ld_be_channels fixture");
});

fn create_ld_be_channels() -> Result<()> {
    if std::path::Path::new(LD_BE_PATH).exists() {
        return Ok(());
    }
    std::fs::create_dir_all("test_files/synthetic")?;
    let mut b: Vec<u8> = Vec::with_capacity(2040);

    id_block(&mut b);
    debug_assert_eq!(b.len(), 64);
    hd4(&mut b, 224, 168);
    debug_assert_eq!(b.len(), 168);
    fh(&mut b);
    debug_assert_eq!(b.len(), 224);
    // 4 chained DGs
    dg4_chain(&mut b, 288, 480, 1656);
    debug_assert_eq!(b.len(), 288); // DG1 bei16
    dg4_chain(&mut b, 352, 584, 1712);
    debug_assert_eq!(b.len(), 352); // DG2 bef64
    dg4_chain(&mut b, 416, 688, 1768);
    debug_assert_eq!(b.len(), 416); // DG3 beu16
    dg4_chain(&mut b, 0, 792, 1824);
    debug_assert_eq!(b.len(), 480); // DG4 bef32
    // CGs (one per DG, one CN each)
    cg4(&mut b, 896, 4, 2);
    debug_assert_eq!(b.len(), 584); // CG1
    cg4(&mut b, 1056, 4, 8);
    debug_assert_eq!(b.len(), 688); // CG2
    cg4(&mut b, 1216, 4, 2);
    debug_assert_eq!(b.len(), 792); // CG3
    cg4(&mut b, 1376, 4, 4);
    debug_assert_eq!(b.len(), 896); // CG4
    // CNs: dtype=3(IntBE), 5(FloatBE), 1(UIntBE), 5(FloatBE)
    cn4(&mut b, (0, 0, 3), (0, 16), (0, 1536, 0));
    debug_assert_eq!(b.len(), 1056); // bei16
    cn4(&mut b, (0, 0, 5), (0, 64), (0, 1566, 0));
    debug_assert_eq!(b.len(), 1216); // bef64
    cn4(&mut b, (0, 0, 1), (0, 16), (0, 1596, 0));
    debug_assert_eq!(b.len(), 1376); // beu16
    cn4(&mut b, (0, 0, 5), (0, 32), (0, 1626, 0));
    debug_assert_eq!(b.len(), 1536); // bef32
    // TX blocks
    tx(&mut b, "bei16");
    debug_assert_eq!(b.len(), 1566);
    tx(&mut b, "bef64");
    debug_assert_eq!(b.len(), 1596);
    tx(&mut b, "beu16");
    debug_assert_eq!(b.len(), 1626);
    tx(&mut b, "bef32");
    debug_assert_eq!(b.len(), 1656);
    // LD blocks pointing to DT blocks
    ld1_block(&mut b, 1880, 4);
    debug_assert_eq!(b.len(), 1712); // LD1 → DT1
    ld1_block(&mut b, 1912, 4);
    debug_assert_eq!(b.len(), 1768); // LD2 → DT2
    ld1_block(&mut b, 1968, 4);
    debug_assert_eq!(b.len(), 1824); // LD3 → DT3
    ld1_block(&mut b, 2000, 4);
    debug_assert_eq!(b.len(), 1880); // LD4 → DT4
    // DT1: 4×i16_BE
    {
        let mut recs: Vec<u8> = Vec::with_capacity(8);
        for v in [-100i16, 0, 100, 200] {
            pi16_be(&mut recs, v);
        }
        dt(&mut b, &recs);
        debug_assert_eq!(b.len(), 1912);
    }
    // DT2: 4×f64_BE
    {
        let mut recs: Vec<u8> = Vec::with_capacity(32);
        for v in [1.5f64, 2.5, 3.5, 4.5] {
            pf64_be(&mut recs, v);
        }
        dt(&mut b, &recs);
        debug_assert_eq!(b.len(), 1968);
    }
    // DT3: 4×u16_BE
    {
        let mut recs: Vec<u8> = Vec::with_capacity(8);
        for v in [100u16, 200, 300, 400] {
            pu16_be(&mut recs, v);
        }
        dt(&mut b, &recs);
        debug_assert_eq!(b.len(), 2000);
    }
    // DT4: 4×f32_BE
    {
        let mut recs: Vec<u8> = Vec::with_capacity(16);
        for v in [1.5f32, 2.5, 3.5, 4.5] {
            pf32_be(&mut recs, v);
        }
        dt(&mut b, &recs);
        debug_assert_eq!(b.len(), 2040);
    }

    let tmp = format!("{}.tmp.{}", LD_BE_PATH, std::process::id());
    std::fs::write(&tmp, &b)?;
    if std::fs::rename(&tmp, LD_BE_PATH).is_err() {
        std::fs::remove_file(&tmp).ok();
    }
    Ok(())
}

// ─── Tests for LD-backed BE channels ─────────────────────────────────────────

fn load_ld_be() -> Result<Mdf> {
    LazyLock::force(&FIXTURE_LD_BE);
    let mut mdf = Mdf::new(LD_BE_PATH)?;
    mdf.load_all_channels_data_in_memory()?;
    Ok(mdf)
}

#[test]
fn ld_bei16_reads_correctly() -> Result<()> {
    let mdf = load_ld_be()?;
    let data = mdf.get_channel_data("bei16").expect("bei16 not found");
    assert!(
        matches!(data, ChannelData::Int16(_)),
        "expected Int16, got {}",
        data.data_type(false)
    );
    if let ChannelData::Int16(arr) = data {
        let expected = [-100i16, 0, 100, 200];
        for (i, (&got, &exp)) in arr.values_slice().iter().zip(expected.iter()).enumerate() {
            assert_eq!(got, exp, "bei16[{i}]: expected {exp}, got {got}");
        }
    }
    Ok(())
}

#[test]
fn ld_bef64_reads_correctly() -> Result<()> {
    let mdf = load_ld_be()?;
    let data = mdf.get_channel_data("bef64").expect("bef64 not found");
    assert!(
        matches!(data, ChannelData::Float64(_)),
        "expected Float64, got {}",
        data.data_type(false)
    );
    if let ChannelData::Float64(arr) = data {
        let expected = [1.5f64, 2.5, 3.5, 4.5];
        for (i, (&got, &exp)) in arr.values_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-9,
                "bef64[{i}]: expected {exp}, got {got}"
            );
        }
    }
    Ok(())
}

#[test]
fn ld_beu16_reads_correctly() -> Result<()> {
    let mdf = load_ld_be()?;
    let data = mdf.get_channel_data("beu16").expect("beu16 not found");
    assert!(
        matches!(data, ChannelData::UInt16(_)),
        "expected UInt16, got {}",
        data.data_type(false)
    );
    if let ChannelData::UInt16(arr) = data {
        let expected = [100u16, 200, 300, 400];
        for (i, (&got, &exp)) in arr.values_slice().iter().zip(expected.iter()).enumerate() {
            assert_eq!(got, exp, "beu16[{i}]: expected {exp}, got {got}");
        }
    }
    Ok(())
}

#[test]
fn ld_bef32_reads_correctly() -> Result<()> {
    let mdf = load_ld_be()?;
    let data = mdf.get_channel_data("bef32").expect("bef32 not found");
    assert!(
        matches!(data, ChannelData::Float32(_)),
        "expected Float32, got {}",
        data.data_type(false)
    );
    if let ChannelData::Float32(arr) = data {
        let expected = [1.5f32, 2.5, 3.5, 4.5];
        for (i, (&got, &exp)) in arr.values_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "bef32[{i}]: expected {exp}, got {got}"
            );
        }
    }
    Ok(())
}
