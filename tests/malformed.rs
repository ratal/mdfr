/// Malformed MDF fixture builder and regression tests.
///
/// Every file here is built in memory, written to a temporary directory and read
/// back through the public `MdfInfo::new` entry point. The fixtures exercise two
/// classes of corrupted structural fields:
///  - block chain links that loop back on themselves,
///  - size fields that claim more bytes than the file holds.
use mdfr::mdfinfo::MdfInfo;
use tempfile::TempDir;

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
fn push_f64(buf: &mut Vec<u8>, v: f64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn push_zeros(buf: &mut Vec<u8>, n: usize) {
    buf.extend(std::iter::repeat_n(0u8, n));
}

/// Writes `bytes` into a temporary directory and returns the directory plus path.
fn write_tmp(name: &str, bytes: &[u8]) -> (TempDir, String) {
    let dir = TempDir::new().expect("could not create temp dir");
    let path = dir.path().join(name);
    std::fs::write(&path, bytes).expect("could not write fixture");
    let path = path.to_string_lossy().into_owned();
    (dir, path)
}

// ─── MDF4 block writers ──────────────────────────────────────────────────────

const ID: usize = 64;
const HD: usize = 104;
/// Offset of the first block after the ID and HD blocks.
const B0: i64 = (ID + HD) as i64;
/// Offset of the attachment in the valid MDF4 fixture, right after the FH block.
const B0_AT: i64 = B0 + 56;

fn write_id4(buf: &mut Vec<u8>) {
    buf.extend_from_slice(b"MDF     ");
    buf.extend_from_slice(b"4.10    ");
    buf.extend_from_slice(b"mdfr    ");
    push_u16(buf, 0); // id_default_byteorder
    push_u16(buf, 0); // id_floatingpointformat
    push_u16(buf, 410); // id_ver
    push_u16(buf, 0); // id_codepage
    push_zeros(buf, 2); // id_check
    push_zeros(buf, 26); // id_fill
    push_u16(buf, 0); // id_unfin_flags
    push_u16(buf, 0); // id_custom_unfin_flags
}

#[derive(Default)]
struct Hd4Links {
    dg_first: i64,
    fh_first: i64,
    ch_first: i64,
    at_first: i64,
    ev_first: i64,
}

fn write_hd4(buf: &mut Vec<u8>, links: &Hd4Links) {
    buf.extend_from_slice(b"##HD");
    push_zeros(buf, 4);
    push_u64(buf, HD as u64);
    push_u64(buf, 6);
    push_i64(buf, links.dg_first);
    push_i64(buf, links.fh_first);
    push_i64(buf, links.ch_first);
    push_i64(buf, links.at_first);
    push_i64(buf, links.ev_first);
    push_i64(buf, 0); // hd_md_comment
    push_u64(buf, 0); // hd_start_time_ns
    push_i16(buf, 0);
    push_i16(buf, 0);
    push_u8(buf, 0);
    push_u8(buf, 0);
    push_u8(buf, 0);
    push_u8(buf, 0);
    push_f64(buf, 0.0);
    push_f64(buf, 0.0);
}

/// FHBLOCK, 56 bytes.
fn write_fh(buf: &mut Vec<u8>, fh_next: i64) {
    buf.extend_from_slice(b"##FH");
    push_zeros(buf, 4);
    push_u64(buf, 56);
    push_u64(buf, 2);
    push_i64(buf, fh_next);
    push_i64(buf, 0); // fh_md_comment
    push_u64(buf, 0); // fh_time_ns
    push_i16(buf, 0);
    push_i16(buf, 0);
    push_u8(buf, 0);
    push_zeros(buf, 3);
}

/// DGBLOCK, 64 bytes.
fn write_dg4(buf: &mut Vec<u8>, dg_next: i64, cg_first: i64) {
    buf.extend_from_slice(b"##DG");
    push_zeros(buf, 4);
    push_u64(buf, 64);
    push_u64(buf, 4);
    push_i64(buf, dg_next);
    push_i64(buf, cg_first);
    push_i64(buf, 0); // dg_data
    push_i64(buf, 0); // dg_md_comment
    push_u8(buf, 0); // dg_rec_id_size
    push_zeros(buf, 7);
}

/// CGBLOCK, 104 bytes (6 links, no cg_cg_master).
fn write_cg4(buf: &mut Vec<u8>, cg_next: i64, cn_first: i64, sr_first: i64) {
    buf.extend_from_slice(b"##CG");
    push_zeros(buf, 4);
    push_u64(buf, 104);
    push_u64(buf, 6);
    push_i64(buf, cg_next);
    push_i64(buf, cn_first);
    push_i64(buf, 0); // cg_tx_acq_name
    push_i64(buf, 0); // cg_si_acq_source
    push_i64(buf, sr_first);
    push_i64(buf, 0); // cg_md_comment
    push_u64(buf, 0); // cg_record_id
    push_u64(buf, 0); // cg_cycle_count
    push_u16(buf, 0); // cg_flags
    push_u16(buf, 0); // cg_path_separator
    push_zeros(buf, 4);
    push_u32(buf, 0); // cg_data_bytes
    push_u32(buf, 0); // cg_inval_bytes
}

/// CNBLOCK, 160 bytes (8 links).
fn write_cn4(buf: &mut Vec<u8>, cn_next: i64) {
    buf.extend_from_slice(b"##CN");
    push_zeros(buf, 4);
    push_u64(buf, 160);
    push_u64(buf, 8);
    push_i64(buf, cn_next);
    push_zeros(buf, 7 * 8); // composition, tx_name, si_source, cc_conversion, data, md_unit, md_comment
    push_u8(buf, 2); // cn_type = master
    push_u8(buf, 1); // cn_sync_type = time
    push_u8(buf, 4); // cn_data_type = FloatLE
    push_u8(buf, 0); // cn_bit_offset
    push_u32(buf, 0); // cn_byte_offset
    push_u32(buf, 64); // cn_bit_count
    push_u32(buf, 0); // cn_flags
    push_u32(buf, 0); // cn_inval_bit_pos
    push_u8(buf, 0xff); // cn_precision
    push_u8(buf, 0); // cn_alignment
    push_u16(buf, 0); // cn_attachment_count
    push_zeros(buf, 6 * 8); // val ranges and limits
}

/// ATBLOCK, 96 bytes plus `trailing` embedded bytes.
fn write_at4(buf: &mut Vec<u8>, at_next: i64, flags: u16, embedded_size: u64, trailing: &[u8]) {
    buf.extend_from_slice(b"##AT");
    push_zeros(buf, 4);
    push_u64(buf, 96);
    push_u64(buf, 4);
    push_i64(buf, at_next);
    push_i64(buf, 0); // at_tx_filename
    push_i64(buf, 0); // at_tx_mimetype
    push_i64(buf, 0); // at_md_comment
    push_u16(buf, flags);
    push_u16(buf, 0); // at_creator_index
    push_u8(buf, 0); // at_zip_type
    push_u8(buf, 0); // at_path_syntax
    push_zeros(buf, 2);
    push_zeros(buf, 16); // at_md5_checksum
    push_u64(buf, embedded_size); // at_original_size
    push_u64(buf, embedded_size); // at_embedded_size
    buf.extend_from_slice(trailing);
}

/// EVBLOCK, 96 bytes (5 links).
fn write_ev4(buf: &mut Vec<u8>, ev_next: i64) {
    buf.extend_from_slice(b"##EV");
    push_zeros(buf, 4);
    push_u64(buf, 96);
    push_u64(buf, 5); // ev_links
    push_i64(buf, ev_next);
    push_i64(buf, 0); // ev_ev_parent
    push_i64(buf, 0); // ev_ev_range
    push_i64(buf, 0); // ev_tx_name
    push_i64(buf, 0); // ev_md_comment
    push_u8(buf, 4); // ev_type = marker
    push_u8(buf, 1); // ev_sync_type = time
    push_u8(buf, 0); // ev_range_type
    push_u8(buf, 0); // ev_cause
    push_u8(buf, 0); // ev_flags
    push_zeros(buf, 3);
    push_u32(buf, 0); // ev_scope_count
    push_u16(buf, 0); // ev_attachment_count
    push_u16(buf, 0); // ev_creator_index
    push_i64(buf, 0); // ev_sync_base_value
    push_f64(buf, 0.0); // ev_sync_factor
}

/// CHBLOCK, 64 bytes (4 links, no elements).
fn write_ch4(buf: &mut Vec<u8>, ch_next: i64, ch_first: i64) {
    buf.extend_from_slice(b"##CH");
    push_zeros(buf, 4);
    push_u64(buf, 64);
    push_u64(buf, 4); // ch_links
    push_i64(buf, ch_next);
    push_i64(buf, ch_first);
    push_i64(buf, 0); // ch_tx_name
    push_i64(buf, 0); // ch_md_comment
    push_u32(buf, 0); // ch_element_count
    push_u8(buf, 0); // ch_type
    push_zeros(buf, 3);
}

/// SRBLOCK, 56 bytes. `len` overrides the declared block length.
fn write_sr4(buf: &mut Vec<u8>, sr_next: i64, len: u64) {
    buf.extend_from_slice(b"##SR");
    push_zeros(buf, 4);
    push_u64(buf, len);
    push_i64(buf, sr_next);
    push_i64(buf, 0); // sr_data
    push_u64(buf, 0); // sr_cycle_count
    push_f64(buf, 0.0); // sr_interval
    push_u8(buf, 1); // sr_sync_type
    push_u8(buf, 0); // sr_flags
    push_zeros(buf, 6);
}

// ─── MDF3 block writers ──────────────────────────────────────────────────────

fn write_id3(buf: &mut Vec<u8>) {
    buf.extend_from_slice(b"MDF     ");
    buf.extend_from_slice(b"3.30    ");
    buf.extend_from_slice(b"mdfr    ");
    push_u16(buf, 0); // id_default_byteorder
    push_u16(buf, 0); // id_floatingpointformat
    push_u16(buf, 330); // id_ver
    push_u16(buf, 0); // id_codepage
    push_zeros(buf, 2);
    push_zeros(buf, 26);
    push_u16(buf, 0);
    push_u16(buf, 0);
}

/// TXBLOCK of MDF3, 8 bytes (4 byte header plus "ok\0\0").
fn write_tx3(buf: &mut Vec<u8>) {
    buf.extend_from_slice(b"TX");
    push_u16(buf, 8);
    buf.extend_from_slice(b"ok\0\0");
}

/// HDBLOCK of MDF3, 208 bytes including the 3.2 extension.
fn write_hd3(buf: &mut Vec<u8>, dg_first: u32, md_comment: u32) {
    buf.extend_from_slice(b"HD");
    push_u16(buf, 208); // hd_len
    push_u32(buf, dg_first);
    push_u32(buf, md_comment); // hd_md_comment
    push_u32(buf, 0); // hd_pr
    push_u16(buf, 1); // hd_n_datagroups
    buf.extend_from_slice(b"01:01:2024"); // hd_date
    buf.extend_from_slice(b"00:00:00"); // hd_time
    push_zeros(buf, 32); // hd_author
    push_zeros(buf, 32); // hd_organization
    push_zeros(buf, 32); // hd_project
    push_zeros(buf, 32); // hd_subject
    // 3.2 extension
    push_u64(buf, 0); // hd_start_time_ns
    push_i16(buf, 0); // hd_time_offset
    push_u16(buf, 0); // hd_time_quality
    push_zeros(buf, 32); // hd_time_identifier
}

/// Offset of the first block after the MDF3 ID and HD blocks.
const B0_MDF3: u32 = (ID + 208) as u32;

/// DGBLOCK of MDF3, 24 bytes.
fn write_dg3(buf: &mut Vec<u8>, dg_next: u32, cg_first: u32, n_cg: u16) {
    buf.extend_from_slice(b"DG");
    push_u16(buf, 24); // dg_len
    push_u32(buf, dg_next);
    push_u32(buf, cg_first);
    push_u32(buf, 0); // dg_tr
    push_u32(buf, 0); // dg_data
    push_u16(buf, n_cg); // dg_n_cg
    push_u16(buf, 0); // dg_n_record_ids
}

// ─── Fixtures ────────────────────────────────────────────────────────────────

/// Valid MDF4 file: one FH block terminating the chain, one embedded attachment
/// whose declared size matches the bytes that follow it.
fn good_mdf4() -> Vec<u8> {
    let mut buf = Vec::new();
    write_id4(&mut buf);
    write_hd4(
        &mut buf,
        &Hd4Links {
            fh_first: B0,
            at_first: B0 + 56,
            ..Default::default()
        },
    );
    write_fh(&mut buf, 0);
    write_at4(&mut buf, 0, 0b1, 8, b"ABCDEFGH");
    buf
}

/// One MDF4 file per chain type, with the head block's `*_next` link pointing at
/// itself. Returns (name, bytes) pairs.
fn self_referencing_mdf4_chains() -> Vec<(&'static str, Vec<u8>)> {
    let mut out: Vec<(&'static str, Vec<u8>)> = Vec::new();

    // FH chain
    let mut buf = Vec::new();
    write_id4(&mut buf);
    write_hd4(
        &mut buf,
        &Hd4Links {
            fh_first: B0,
            ..Default::default()
        },
    );
    write_fh(&mut buf, B0);
    out.push(("fh_self_cycle.mf4", buf));

    // DG chain
    let mut buf = Vec::new();
    write_id4(&mut buf);
    write_hd4(
        &mut buf,
        &Hd4Links {
            dg_first: B0,
            ..Default::default()
        },
    );
    write_dg4(&mut buf, B0, 0);
    out.push(("dg_self_cycle.mf4", buf));

    // AT chain
    let mut buf = Vec::new();
    write_id4(&mut buf);
    write_hd4(
        &mut buf,
        &Hd4Links {
            at_first: B0,
            ..Default::default()
        },
    );
    write_at4(&mut buf, B0, 0, 0, &[]);
    out.push(("at_self_cycle.mf4", buf));

    // EV chain
    let mut buf = Vec::new();
    write_id4(&mut buf);
    write_hd4(
        &mut buf,
        &Hd4Links {
            ev_first: B0,
            ..Default::default()
        },
    );
    write_ev4(&mut buf, B0);
    out.push(("ev_self_cycle.mf4", buf));

    // CH chain, sibling link
    let mut buf = Vec::new();
    write_id4(&mut buf);
    write_hd4(
        &mut buf,
        &Hd4Links {
            ch_first: B0,
            ..Default::default()
        },
    );
    write_ch4(&mut buf, B0, 0);
    out.push(("ch_self_cycle.mf4", buf));

    // CG chain, reached through a DG block
    let mut buf = Vec::new();
    write_id4(&mut buf);
    write_hd4(
        &mut buf,
        &Hd4Links {
            dg_first: B0,
            ..Default::default()
        },
    );
    write_dg4(&mut buf, 0, B0 + 64);
    write_cg4(&mut buf, B0 + 64, 0, 0);
    out.push(("cg_self_cycle.mf4", buf));

    // CN chain, reached through DG → CG
    let mut buf = Vec::new();
    write_id4(&mut buf);
    write_hd4(
        &mut buf,
        &Hd4Links {
            dg_first: B0,
            ..Default::default()
        },
    );
    write_dg4(&mut buf, 0, B0 + 64);
    write_cg4(&mut buf, 0, B0 + 64 + 104, 0);
    write_cn4(&mut buf, B0 + 64 + 104);
    out.push(("cn_self_cycle.mf4", buf));

    // SR chain, reached through DG → CG
    let mut buf = Vec::new();
    write_id4(&mut buf);
    write_hd4(
        &mut buf,
        &Hd4Links {
            dg_first: B0,
            ..Default::default()
        },
    );
    write_dg4(&mut buf, 0, B0 + 64);
    write_cg4(&mut buf, 0, 0, B0 + 64 + 104);
    write_sr4(&mut buf, B0 + 64 + 104, 56);
    out.push(("sr_self_cycle.mf4", buf));

    out
}

/// MDF4 file with a CH block whose child link points back at itself, so the
/// recursive descent never bottoms out.
fn ch_self_child() -> Vec<u8> {
    let mut buf = Vec::new();
    write_id4(&mut buf);
    write_hd4(
        &mut buf,
        &Hd4Links {
            ch_first: B0,
            ..Default::default()
        },
    );
    write_ch4(&mut buf, 0, B0);
    buf
}

/// MDF3 file whose DG block links to itself.
fn mdf3_dg_self_cycle() -> Vec<u8> {
    let mut buf = Vec::new();
    write_id3(&mut buf);
    write_hd3(&mut buf, B0_MDF3 + 8, B0_MDF3);
    write_tx3(&mut buf);
    write_dg3(&mut buf, B0_MDF3 + 8, 0, 0);
    buf
}

/// Valid MDF3 file: DG chain terminates.
fn good_mdf3() -> Vec<u8> {
    let mut buf = Vec::new();
    write_id3(&mut buf);
    write_hd3(&mut buf, B0_MDF3 + 8, B0_MDF3);
    write_tx3(&mut buf);
    write_dg3(&mut buf, 0, 0, 1);
    buf
}

/// MDF4 file whose attachment declares an embedded size of u64::MAX.
fn at_embedded_size_overflow() -> Vec<u8> {
    let mut buf = Vec::new();
    write_id4(&mut buf);
    write_hd4(
        &mut buf,
        &Hd4Links {
            at_first: B0,
            ..Default::default()
        },
    );
    write_at4(&mut buf, 0, 0b1, u64::MAX, &[]);
    buf
}

/// MDF4 file whose attachment declares an embedded size far beyond the file end
/// but small enough to allocate.
fn at_embedded_size_beyond_eof() -> Vec<u8> {
    let mut buf = Vec::new();
    write_id4(&mut buf);
    write_hd4(
        &mut buf,
        &Hd4Links {
            at_first: B0,
            ..Default::default()
        },
    );
    write_at4(&mut buf, 0, 0b1, 1 << 34, &[]);
    buf
}

/// MDF4 file whose SR block declares a length below the 16 byte short header.
fn sr_undersized_len() -> Vec<u8> {
    let mut buf = Vec::new();
    write_id4(&mut buf);
    write_hd4(
        &mut buf,
        &Hd4Links {
            dg_first: B0,
            ..Default::default()
        },
    );
    write_dg4(&mut buf, 0, B0 + 64);
    write_cg4(&mut buf, 0, 0, B0 + 64 + 104);
    write_sr4(&mut buf, 0, 0);
    buf
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[test]
fn valid_files_still_parse() {
    let (_dir, path) = write_tmp("good.mf4", &good_mdf4());
    let info = MdfInfo::new(&path).expect("valid MDF4 file must parse");
    assert_eq!(info.get_version(), 410);
    match info {
        MdfInfo::V4(info) => {
            assert_eq!(info.fh.len(), 1);
            let (block, data) = info.at.get(&B0_AT).expect("attachment must be read");
            assert_eq!(block.at_embedded_size, 8);
            // the size check must not disturb the single exact-size allocation
            let data = data.as_ref().expect("embedded data must be read");
            assert_eq!(data.as_slice(), b"ABCDEFGH");
            assert_eq!(data.capacity(), 8);
        }
        MdfInfo::V3(_) => panic!("expected an MDF4 file"),
    }

    let (_dir, path) = write_tmp("good.mdf", &good_mdf3());
    let info = MdfInfo::new(&path).expect("valid MDF3 file must parse");
    assert_eq!(info.get_version(), 330);
    match info {
        MdfInfo::V3(info) => assert_eq!(info.dg.len(), 1),
        MdfInfo::V4(_) => panic!("expected an MDF3 file"),
    }
}

#[test]
fn self_referencing_chains_terminate() {
    // Each of these hangs the parser without the cycle guard.
    for (name, bytes) in self_referencing_mdf4_chains() {
        let (_dir, path) = write_tmp(name, &bytes);
        let info = MdfInfo::new(&path)
            .unwrap_or_else(|e| panic!("{name} should read as a truncated chain, got {e:#}"));
        match info {
            MdfInfo::V4(info) => {
                // the head block is kept, the repeated link is not followed
                assert!(info.fh.len() <= 1, "{name}: FH chain not truncated");
                assert!(info.at.len() <= 1, "{name}: AT chain not truncated");
                assert!(info.ev.len() <= 1, "{name}: EV chain not truncated");
                assert!(info.ch.len() <= 1, "{name}: CH chain not truncated");
                assert!(info.dg.len() <= 1, "{name}: DG chain not truncated");
                for dg in info.dg.values() {
                    assert!(dg.cg.len() <= 1, "{name}: CG chain not truncated");
                    for cg in dg.cg.values() {
                        assert!(cg.cn.len() <= 1, "{name}: CN chain not truncated");
                        assert!(cg.sr.len() <= 1, "{name}: SR chain not truncated");
                    }
                }
            }
            MdfInfo::V3(_) => panic!("{name}: expected an MDF4 file"),
        }
    }

    let (_dir, path) = write_tmp("dg_self_cycle.mdf", &mdf3_dg_self_cycle());
    let info = MdfInfo::new(&path).expect("MDF3 DG cycle should read as a truncated chain");
    match info {
        MdfInfo::V3(info) => assert_eq!(info.dg.len(), 1),
        MdfInfo::V4(_) => panic!("expected an MDF3 file"),
    }
}

#[test]
fn ch_child_cycle_terminates() {
    // Without the guard the recursive descent overflows the stack.
    let (_dir, path) = write_tmp("ch_self_child.mf4", &ch_self_child());
    let info = MdfInfo::new(&path).expect("CH child cycle should read as a truncated tree");
    match info {
        MdfInfo::V4(info) => assert_eq!(info.ch.len(), 1),
        MdfInfo::V3(_) => panic!("expected an MDF4 file"),
    }
}

#[test]
fn oversized_attachment_is_rejected() {
    let (_dir, path) = write_tmp("at_overflow.mf4", &at_embedded_size_overflow());
    assert!(
        MdfInfo::new(&path).is_err(),
        "attachment claiming u64::MAX embedded bytes must be rejected"
    );

    let (_dir, path) = write_tmp("at_beyond_eof.mf4", &at_embedded_size_beyond_eof());
    assert!(
        MdfInfo::new(&path).is_err(),
        "attachment claiming more bytes than the file holds must be rejected"
    );
}

#[test]
fn undersized_sr_block_is_rejected() {
    let (_dir, path) = write_tmp("sr_short.mf4", &sr_undersized_len());
    assert!(
        MdfInfo::new(&path).is_err(),
        "SR block shorter than its own header must be rejected"
    );
}

/// TXBLOCK of MDF3 with an overridden block length.
fn write_tx3_len(buf: &mut Vec<u8>, len: u16) {
    buf.extend_from_slice(b"TX");
    push_u16(buf, len);
    buf.extend_from_slice(b"ok\0\0");
}

/// MDF3 file whose HD comment points at a TX block shorter than its own header.
fn mdf3_tx_undersized() -> Vec<u8> {
    let mut buf = Vec::new();
    write_id3(&mut buf);
    write_hd3(&mut buf, B0_MDF3 + 8, B0_MDF3);
    write_tx3_len(&mut buf, 0);
    write_dg3(&mut buf, 0, 0, 1);
    buf
}

/// CHBLOCK with an overridden link count.
fn write_ch4_links(buf: &mut Vec<u8>, ch_links: u64) {
    buf.extend_from_slice(b"##CH");
    push_zeros(buf, 4);
    push_u64(buf, 64);
    push_u64(buf, ch_links);
    push_i64(buf, 0); // ch_ch_next
    push_i64(buf, 0); // ch_ch_first
    push_i64(buf, 0); // ch_tx_name
    push_i64(buf, 0); // ch_md_comment
    push_u32(buf, 0); // ch_element_count
    push_u8(buf, 0); // ch_type
    push_zeros(buf, 3);
}

/// MDF4 file whose CH block declares `ch_links` links in a 64 byte block.
fn ch_bad_link_count(ch_links: u64) -> Vec<u8> {
    let mut buf = Vec::new();
    write_id4(&mut buf);
    write_hd4(
        &mut buf,
        &Hd4Links {
            ch_first: B0,
            ..Default::default()
        },
    );
    write_ch4_links(&mut buf, ch_links);
    buf
}

#[test]
fn undersized_mdf3_tx_block_is_rejected() {
    let (_dir, path) = write_tmp("tx_short.mdf", &mdf3_tx_undersized());
    assert!(
        MdfInfo::new(&path).is_err(),
        "TX block shorter than its own header must be rejected"
    );
}

#[test]
fn implausible_ch_link_count_is_rejected() {
    // 0 and 3 underflow the ch_element count, the large values would size the
    // vector far beyond the 64 byte block
    for ch_links in [0u64, 3, 1 << 40, u64::MAX] {
        let (_dir, path) = write_tmp("ch_links.mf4", &ch_bad_link_count(ch_links));
        assert!(
            MdfInfo::new(&path).is_err(),
            "CH block declaring {ch_links} links in a 64 byte block must be rejected"
        );
    }
}
