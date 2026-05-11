/// Synthetic MDF4 fixture builder.
///
/// Creates minimal binary MDF4 files for coverage testing of conversion code paths
/// that are never hit by the existing sample files (which all use Float64 raw channels).
///
/// Run once with `cargo test --test fixtures` to generate the files.
/// The files are checked in to test_files/synthetic/ so regular tests can load them.
use anyhow::Result;

// ─── Low-level byte helpers ──────────────────────────────────────────────────

fn push_u8(buf: &mut Vec<u8>, v: u8) {
    buf.push(v);
}
fn push_u16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn push_i16(buf: &mut Vec<u8>, v: i16) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn push_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn push_i64(buf: &mut Vec<u8>, v: i64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn push_f32(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn push_f64(buf: &mut Vec<u8>, v: f64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn push_zeros(buf: &mut Vec<u8>, n: usize) {
    buf.extend(std::iter::repeat_n(0u8, n));
}

// ─── Block writers ───────────────────────────────────────────────────────────

/// IdBlock (64 bytes, fixed MDF4 header)
fn write_id_block(buf: &mut Vec<u8>) {
    buf.extend_from_slice(b"MDF     "); // id_file_id (8)
    buf.extend_from_slice(b"4.30    "); // id_vers    (8)
    buf.extend_from_slice(b"mdfr    "); // id_prog    (8)
    push_u16(buf, 0); // id_default_byteorder
    push_u16(buf, 0); // id_floatingpointformat
    push_u16(buf, 430); // id_ver
    push_u16(buf, 0); // id_codepage
    push_zeros(buf, 2); // id_check
    push_zeros(buf, 26); // id_fill
    push_u16(buf, 0); // id_unfin_flags
    push_u16(buf, 0); // id_custom_unfin_flags
    // Total: 8+8+8+2+2+2+2+2+26+2+2 = 64 bytes
}

/// Hd4 block (104 bytes, self-contained including ##HD header)
fn write_hd4(buf: &mut Vec<u8>, dg_first: i64, fh_first: i64) {
    buf.extend_from_slice(b"##HD"); // hd_id
    push_zeros(buf, 4); // hd_reserved
    push_u64(buf, 104); // hd_len
    push_u64(buf, 6); // hd_link_counts
    push_i64(buf, dg_first); // hd_dg_first
    push_i64(buf, fh_first); // hd_fh_first
    push_i64(buf, 0); // hd_ch_first
    push_i64(buf, 0); // hd_at_first
    push_i64(buf, 0); // hd_ev_first
    push_i64(buf, 0); // hd_md_comment
    push_u64(buf, 0); // hd_start_time_ns
    push_i16(buf, 0); // hd_tz_offset_min
    push_i16(buf, 0); // hd_dst_offset_min
    push_u8(buf, 0); // hd_time_flags
    push_u8(buf, 0); // hd_time_class
    push_u8(buf, 0); // hd_flags
    push_u8(buf, 0); // hd_reserved2
    push_f64(buf, 0.0); // hd_start_angle_rad
    push_f64(buf, 0.0); // hd_start_distance_m
    // Total: 4+4+8+8 + 6×8 + 8+2+2+1+1+1+1+8+8 = 24+48+32 = 104 bytes
}

/// FhBlock (56 bytes, self-contained including ##FH header)
fn write_fh(buf: &mut Vec<u8>) {
    buf.extend_from_slice(b"##FH"); // fh_id
    push_zeros(buf, 4); // fh_gap
    push_u64(buf, 56); // fh_len
    push_u64(buf, 2); // fh_links (fh_fh_next + fh_md_comment)
    push_i64(buf, 0); // fh_fh_next (end of list)
    push_i64(buf, 0); // fh_md_comment (none)
    push_u64(buf, 0); // fh_time_ns
    push_i16(buf, 0); // fh_tz_offset_min
    push_i16(buf, 0); // fh_dst_offset_min
    push_u8(buf, 0); // fh_time_flags
    push_zeros(buf, 3); // fh_reserved
    // Total: 4+4+8+8+8+8+8+2+2+1+3 = 56 bytes
}

/// Dg4Block (64 bytes, self-contained including ##DG header)
fn write_dg4(buf: &mut Vec<u8>, cg_first: i64, data: i64) {
    buf.extend_from_slice(b"##DG"); // dg_id
    push_zeros(buf, 4); // reserved
    push_u64(buf, 64); // dg_len
    push_u64(buf, 4); // dg_links
    push_i64(buf, 0); // dg_dg_next
    push_i64(buf, cg_first); // dg_cg_first
    push_i64(buf, data); // dg_data
    push_i64(buf, 0); // dg_md_comment
    push_u8(buf, 0); // dg_rec_id_size
    push_zeros(buf, 7); // reserved_2
    // Total: 4+4+8+8+4×8+1+7 = 64 bytes
}

/// CG block: Blockheader4Short(16) + Cg4Block body(88) = 104 bytes total.
/// Uses 6 standard links (no cg_cg_master).
fn write_cg4(buf: &mut Vec<u8>, cn_first: i64, cycle_count: u64, data_bytes: u32) {
    // Blockheader4Short (16 bytes)
    buf.extend_from_slice(b"##CG"); // hdr_id
    push_zeros(buf, 4); // hdr_gap
    push_u64(buf, 104); // hdr_len (= 16 + 88)
    // Cg4Block body (88 bytes):
    push_u64(buf, 6); // cg_links (6 → no cg_cg_master)
    push_i64(buf, 0); // cg_cg_next
    push_i64(buf, cn_first); // cg_cn_first
    push_i64(buf, 0); // cg_tx_acq_name
    push_i64(buf, 0); // cg_si_acq_source
    push_i64(buf, 0); // cg_sr_first
    push_i64(buf, 0); // cg_md_comment
    push_u64(buf, 0); // cg_record_id
    push_u64(buf, cycle_count); // cg_cycle_count
    push_u16(buf, 0); // cg_flags
    push_u16(buf, 0); // cg_path_separator
    push_zeros(buf, 4); // cg_reserved
    push_u32(buf, data_bytes); // cg_data_bytes
    push_u32(buf, 0); // cg_inval_bytes
    // Body: 8+48+8+8+2+2+4+4+4 = 88 bytes → total 104
}

/// CN block: Blockheader4Short(16) + Cn4Block body(144) = 160 bytes total.
/// Uses 8 standard links (no extra CA/event links).
///
/// - cn_type: 0=fixed, 2=master
/// - cn_sync_type: 0=none, 1=time
/// - cn_data_type: 2=IntLE, 4=FloatLE
#[allow(clippy::too_many_arguments)]
fn write_cn4(
    buf: &mut Vec<u8>,
    cn_type: u8,
    cn_sync_type: u8,
    cn_data_type: u8,
    cn_byte_offset: u32,
    cn_bit_count: u32,
    cn_cn_next: i64,
    cn_tx_name: i64,
    cn_cc_conversion: i64,
) {
    // Blockheader4Short (16 bytes)
    buf.extend_from_slice(b"##CN"); // hdr_id
    push_zeros(buf, 4); // hdr_gap
    push_u64(buf, 160); // hdr_len (= 16 + 144)
    // Cn4Block body (8 + 64 + 72 = 144 bytes):
    push_u64(buf, 8); // cn_links (8 standard links)
    push_i64(buf, cn_cn_next); // cn_cn_next
    push_i64(buf, 0); // cn_composition
    push_i64(buf, cn_tx_name); // cn_tx_name
    push_i64(buf, 0); // cn_si_source
    push_i64(buf, cn_cc_conversion); // cn_cc_conversion
    push_i64(buf, 0); // cn_data
    push_i64(buf, 0); // cn_md_unit
    push_i64(buf, 0); // cn_md_comment
    // Data members (72 bytes):
    push_u8(buf, cn_type); // cn_type
    push_u8(buf, cn_sync_type); // cn_sync_type
    push_u8(buf, cn_data_type); // cn_data_type
    push_u8(buf, 0); // cn_bit_offset
    push_u32(buf, cn_byte_offset); // cn_byte_offset
    push_u32(buf, cn_bit_count); // cn_bit_count
    push_u32(buf, 0); // cn_flags
    push_u32(buf, 0); // cn_inval_bit_pos
    push_u8(buf, 0xff); // cn_precision (unrestricted)
    push_u8(buf, 0); // cn_alignment
    push_u16(buf, 0); // cn_attachment_count
    push_f64(buf, 0.0); // cn_val_range_min
    push_f64(buf, 0.0); // cn_val_range_max
    push_f64(buf, 0.0); // cn_limit_min
    push_f64(buf, 0.0); // cn_limit_max
    push_f64(buf, 0.0); // cn_limit_ext_min
    push_f64(buf, 0.0); // cn_limit_ext_max
    // Data members total: 1+1+1+1+4+4+4+4+1+1+2+8+8+8+8+8+8 = 72 bytes → total 160
}

/// TX block: Blockheader4(24) + null-terminated text.
/// `text` should NOT include the null terminator.
fn write_tx(buf: &mut Vec<u8>, text: &str) {
    let text_bytes = text.as_bytes();
    let total_len = 24u64 + text_bytes.len() as u64 + 1; // +1 for null
    buf.extend_from_slice(b"##TX"); // hdr_id
    push_zeros(buf, 4); // hdr_gap
    push_u64(buf, total_len); // hdr_len
    push_u64(buf, 0); // hdr_links
    buf.extend_from_slice(text_bytes); // text
    buf.push(0); // null terminator
}

/// CC block (linear, cc_type=1): Blockheader4Short(16) + Cc4Block body(80) = 96 bytes.
/// Formula: phys = a1 * raw + a0
fn write_cc_linear(buf: &mut Vec<u8>, a0: f64, a1: f64) {
    // Blockheader4Short (16 bytes)
    buf.extend_from_slice(b"##CC"); // hdr_id
    push_zeros(buf, 4); // hdr_gap
    push_u64(buf, 96); // hdr_len (= 16 + 80)
    // Cc4Block body (80 bytes):
    push_u64(buf, 4); // cc_links (4 standard links, no cc_ref)
    push_i64(buf, 0); // cc_tx_name
    push_i64(buf, 0); // cc_md_unit
    push_i64(buf, 0); // cc_md_comment
    push_i64(buf, 0); // cc_cc_inverse
    // cc_ref: empty (cc_links == 4)
    push_u8(buf, 1); // cc_type = 1 (Linear)
    push_u8(buf, 0); // cc_precision
    push_u16(buf, 0); // cc_flags
    push_u16(buf, 0); // cc_ref_count
    push_u16(buf, 2); // cc_val_count (a0, a1)
    push_f64(buf, 0.0); // cc_phy_range_min
    push_f64(buf, 0.0); // cc_phy_range_max
    // cc_val: Real([a0, a1])
    push_f64(buf, a0); // a0 (offset)
    push_f64(buf, a1); // a1 (factor)
    // Body: 8+32+1+1+2+2+2+8+8+16 = 80 bytes → total 96
}

/// DT block: 4-byte id + Dt4Block(20) + raw records.
/// `records` is a flat byte slice of all records concatenated.
fn write_dt(buf: &mut Vec<u8>, records: &[u8]) {
    let total_len = 24u64 + records.len() as u64;
    buf.extend_from_slice(b"##DT"); // id (read separately by reader before Dt4Block)
    push_zeros(buf, 4); // reserved
    push_u64(buf, total_len); // len (total block size)
    push_u64(buf, 0); // links = 0
    buf.extend_from_slice(records); // raw data
}

// ─── Fixture: int_linear_cc ──────────────────────────────────────────────────

/// Builds and writes `test_files/synthetic/int_linear_cc.mf4`.
///
/// Contains 4 channels in a single channel group with 4 samples each:
/// - `time_ch`    : Float64 LE master (sync_type=time), values 0.0..3.0
/// - `int8_ch`    : Int8 raw  + linear CC (a0=0.5, a1=2.0)
/// - `int16_ch`   : Int16 raw + linear CC (a0=0.5, a1=2.0)
/// - `float32_ch` : Float32 raw + linear CC (a0=0.5, a1=2.0)
///
/// File layout (exact byte offsets):
/// ```text
/// [0]    IdBlock  64 b
/// [64]   Hd4      104 b
/// [168]  FhBlock  56 b
/// [224]  Dg4      64 b
/// [288]  CG       104 b  (cg_cn_first=392, cg_data_bytes=15, cg_cycle_count=4)
/// [392]  CN_master  160 b  (data_type=4/FloatLE, bit_count=64, byte_offset=0)
/// [552]  CN_int8    160 b  (data_type=2/IntLE, bit_count=8, byte_offset=8, cc=1164)
/// [712]  CN_int16   160 b  (data_type=2/IntLE, bit_count=16, byte_offset=9, cc=1164)
/// [872]  CN_float32 160 b  (data_type=4/FloatLE, bit_count=32, byte_offset=11, cc=1164)
/// [1032] TX "time_ch\0"     32 b
/// [1064] TX "int8_ch\0"     32 b
/// [1096] TX "int16_ch\0"    33 b
/// [1129] TX "float32_ch\0"  35 b
/// [1164] CC linear(a0=0.5, a1=2.0)  96 b
/// [1260] DT  84 b  (4 records × 15 bytes)
/// Total: 1344 bytes
/// ```
pub fn create_int_linear_cc_fixture() -> Result<()> {
    const PATH: &str = "test_files/synthetic/int_linear_cc.mf4";
    std::fs::create_dir_all("test_files/synthetic")?;

    let mut buf: Vec<u8> = Vec::with_capacity(1344);

    write_id_block(&mut buf);
    debug_assert_eq!(buf.len(), 64, "IdBlock size mismatch");

    write_hd4(&mut buf, 224, 168); // dg_first=224, fh_first=168
    debug_assert_eq!(buf.len(), 168, "Hd4 size mismatch");

    write_fh(&mut buf);
    debug_assert_eq!(buf.len(), 224, "FhBlock size mismatch");

    write_dg4(&mut buf, 288, 1260); // cg_first=288, data=1260
    debug_assert_eq!(buf.len(), 288, "Dg4Block size mismatch");

    // CG: cn_first=392, cycle_count=4, data_bytes=15 (8+1+2+4)
    write_cg4(&mut buf, 392, 4, 15);
    debug_assert_eq!(buf.len(), 392, "CG size mismatch");

    // CN_master: type=2, sync=1(time), data_type=4(FloatLE), byte_offset=0, bit_count=64
    // cn_cn_next=552, tx_name=1032, cc=0
    write_cn4(&mut buf, 2, 1, 4, 0, 64, 552, 1032, 0);
    debug_assert_eq!(buf.len(), 552, "CN_master size mismatch");

    // CN_int8: type=0, sync=0, data_type=2(IntLE), byte_offset=8, bit_count=8
    // cn_cn_next=712, tx_name=1064, cc=1164
    write_cn4(&mut buf, 0, 0, 2, 8, 8, 712, 1064, 1164);
    debug_assert_eq!(buf.len(), 712, "CN_int8 size mismatch");

    // CN_int16: type=0, sync=0, data_type=2(IntLE), byte_offset=9, bit_count=16
    // cn_cn_next=872, tx_name=1096, cc=1164
    write_cn4(&mut buf, 0, 0, 2, 9, 16, 872, 1096, 1164);
    debug_assert_eq!(buf.len(), 872, "CN_int16 size mismatch");

    // CN_float32: type=0, sync=0, data_type=4(FloatLE), byte_offset=11, bit_count=32
    // cn_cn_next=0, tx_name=1129, cc=1164
    write_cn4(&mut buf, 0, 0, 4, 11, 32, 0, 1129, 1164);
    debug_assert_eq!(buf.len(), 1032, "CN_float32 size mismatch");

    write_tx(&mut buf, "time_ch"); // 24 + 7 + 1 = 32 bytes → [1032..1064)
    debug_assert_eq!(buf.len(), 1064, "TX time_ch size mismatch");

    write_tx(&mut buf, "int8_ch"); // 32 bytes → [1064..1096)
    debug_assert_eq!(buf.len(), 1096, "TX int8_ch size mismatch");

    write_tx(&mut buf, "int16_ch"); // 24 + 8 + 1 = 33 bytes → [1096..1129)
    debug_assert_eq!(buf.len(), 1129, "TX int16_ch size mismatch");

    write_tx(&mut buf, "float32_ch"); // 24 + 10 + 1 = 35 bytes → [1129..1164)
    debug_assert_eq!(buf.len(), 1164, "TX float32_ch size mismatch");

    write_cc_linear(&mut buf, 0.5, 2.0); // 96 bytes → [1164..1260)
    debug_assert_eq!(buf.len(), 1260, "CC linear size mismatch");

    // DT: 4 records × 15 bytes = 60 bytes + 24-byte header = 84 bytes → [1260..1344)
    // Record layout per row:
    //   [0..8)  : f64 LE  master time (0.0, 1.0, 2.0, 3.0)
    //   [8]     : i8      raw int8   (-5, 0, 5, 10)
    //   [9..11) : i16 LE  raw int16  (-100, 0, 100, 200)
    //   [11..15): f32 LE  raw float32 (1.5, 2.5, 3.5, 4.5)
    let raw_i8: [i8; 4] = [-5, 0, 5, 10];
    let raw_i16: [i16; 4] = [-100, 0, 100, 200];
    let raw_f32: [f32; 4] = [1.5, 2.5, 3.5, 4.5];
    let mut records: Vec<u8> = Vec::with_capacity(60);
    for i in 0..4usize {
        push_f64(&mut records, i as f64); // master time
        records.push(raw_i8[i] as u8); // i8 raw (bit-cast)
        records.extend_from_slice(&raw_i16[i].to_le_bytes()); // i16 raw
        push_f32(&mut records, raw_f32[i]); // f32 raw
    }
    assert_eq!(records.len(), 60, "record data size mismatch");
    write_dt(&mut buf, &records);
    debug_assert_eq!(buf.len(), 1344, "DT size mismatch");

    std::fs::write(PATH, &buf)?;
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[test]
fn create_fixtures() {
    create_int_linear_cc_fixture().expect("failed to create int_linear_cc fixture");
    assert!(
        std::path::Path::new("test_files/synthetic/int_linear_cc.mf4").exists(),
        "fixture file not created"
    );
}
