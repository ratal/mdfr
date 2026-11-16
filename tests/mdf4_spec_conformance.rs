//! Tests for MDF 4.3.0 conformance features:
//! 1. Independent column-oriented channels (`cn_data` -> LD4 / DV4 / DZ4 with `dg_data == 0`).
//! 2. Virtual data channels (`cn_type == 6` with conversion).
//! 3. Array storage types: DG template (`ca_storage == 2`) and CG template (`ca_storage == 1`).

use anyhow::Result;
use mdfr::data_holder::channel_data::{AsValuesSlice, ChannelData};
use mdfr::mdfinfo::mdfinfo4::Ca4Block;
use mdfr::mdfreader::Mdf;
use std::io::Write;
use tempfile::NamedTempFile;

// ── Low-level byte helpers ───────────────────────────────────────────────────

fn push_u8(buf: &mut Vec<u8>, v: u8) {
    buf.push(v);
}
fn push_u16(buf: &mut Vec<u8>, v: u16) {
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

fn write_id_block(buf: &mut Vec<u8>) {
    buf.extend_from_slice(b"MDF     ");
    buf.extend_from_slice(b"4.30    ");
    buf.extend_from_slice(b"mdfr    ");
    push_u16(buf, 0);
    push_u16(buf, 0);
    push_u16(buf, 430);
    push_u16(buf, 0);
    push_zeros(buf, 28);
    push_u16(buf, 0);
    push_u16(buf, 0);
}

fn write_hd4(buf: &mut Vec<u8>, dg_first: i64) {
    buf.extend_from_slice(b"##HD");
    push_zeros(buf, 4);
    push_u64(buf, 104);
    push_u64(buf, 6);
    push_i64(buf, dg_first);
    push_i64(buf, 0);
    push_i64(buf, 0);
    push_i64(buf, 0);
    push_i64(buf, 0);
    push_i64(buf, 0);
    push_u64(buf, 0);
    push_u8(buf, 0);
    push_u8(buf, 0);
    push_u8(buf, 0);
    push_u8(buf, 0);
    push_f64(buf, 0.0);
    push_zeros(buf, 20);
}

fn write_dg4(buf: &mut Vec<u8>, cg_first: i64, data: i64) {
    buf.extend_from_slice(b"##DG");
    push_zeros(buf, 4);
    push_u64(buf, 64);
    push_u64(buf, 4);
    push_i64(buf, 0);
    push_i64(buf, cg_first);
    push_i64(buf, data);
    push_i64(buf, 0);
    push_u8(buf, 0); // rec_id_size = 0 (sorted)
    push_zeros(buf, 7);
}

fn write_cg4(buf: &mut Vec<u8>, cn_first: i64, cycle_count: u64) {
    write_cg4_with_data_bytes(buf, cn_first, cycle_count, 8);
}

fn write_cg4_with_data_bytes(buf: &mut Vec<u8>, cn_first: i64, cycle_count: u64, data_bytes: u32) {
    buf.extend_from_slice(b"##CG");
    push_zeros(buf, 4);
    push_u64(buf, 104);
    push_u64(buf, 6);
    push_i64(buf, 0);
    push_i64(buf, cn_first);
    push_i64(buf, 0);
    push_i64(buf, 0);
    push_i64(buf, 0);
    push_i64(buf, 0);
    push_u64(buf, 0); // record_id = 0
    push_u64(buf, cycle_count);
    push_u16(buf, 0); // cg_flags
    push_u16(buf, 0); // cg_path_separator
    push_zeros(buf, 4); // cg_reserved
    push_u32(buf, data_bytes); // data_bytes
    push_u32(buf, 0); // inval_bytes
}

#[allow(clippy::too_many_arguments)]
fn write_cn4(
    buf: &mut Vec<u8>,
    cn_type: u8,
    cn_data_type: u8,
    cn_bit_count: u32,
    cn_tx_name: i64,
    cn_cc_conversion: i64,
    cn_data: i64,
    composition: i64,
) {
    write_cn4_with_offset(
        buf,
        cn_type,
        cn_data_type,
        0,
        0,
        cn_bit_count,
        cn_tx_name,
        cn_cc_conversion,
        cn_data,
        composition,
    );
}

#[allow(clippy::too_many_arguments)]
fn write_cn4_with_offset(
    buf: &mut Vec<u8>,
    cn_type: u8,
    cn_data_type: u8,
    bit_offset: u8,
    byte_offset: u32,
    cn_bit_count: u32,
    cn_tx_name: i64,
    cn_cc_conversion: i64,
    cn_data: i64,
    composition: i64,
) {
    buf.extend_from_slice(b"##CN");
    push_zeros(buf, 4);
    push_u64(buf, 160);
    push_u64(buf, 8);
    push_i64(buf, 0);
    push_i64(buf, composition);
    push_i64(buf, cn_tx_name);
    push_i64(buf, 0);
    push_i64(buf, cn_cc_conversion);
    push_i64(buf, cn_data);
    push_i64(buf, 0);
    push_i64(buf, 0);
    push_u8(buf, cn_type);
    push_u8(buf, 0); // cn_sync_type
    push_u8(buf, cn_data_type);
    push_u8(buf, bit_offset);
    push_u32(buf, byte_offset);
    push_u32(buf, cn_bit_count);
    push_u32(buf, 0); // flags
    push_u32(buf, 0); // inval_bit_pos
    push_u8(buf, 0xff); // precision
    push_u8(buf, 0); // alignment
    push_u16(buf, 0); // attachment_count
    push_f64(buf, 0.0);
    push_f64(buf, 0.0);
    push_f64(buf, 0.0);
    push_f64(buf, 0.0);
    push_f64(buf, 0.0);
    push_f64(buf, 0.0);
}

fn write_tx(buf: &mut Vec<u8>, text: &str) {
    buf.extend_from_slice(b"##TX");
    push_zeros(buf, 4);
    let bytes = text.as_bytes();
    let text_len = bytes.len() + 1;
    let block_len = 24 + text_len;
    push_u64(buf, block_len as u64);
    push_u64(buf, 0);
    buf.extend_from_slice(bytes);
    buf.push(0);
}

fn write_cc_linear(buf: &mut Vec<u8>, factor: f64, offset: f64) {
    buf.extend_from_slice(b"##CC");
    push_zeros(buf, 4);
    push_u64(buf, 96); // 16 header + 80 body
    push_u64(buf, 4); // cc_links
    push_i64(buf, 0);
    push_i64(buf, 0);
    push_i64(buf, 0);
    push_i64(buf, 0);
    push_u8(buf, 1); // cc_type = 1 (linear)
    push_u8(buf, 0xff);
    push_u16(buf, 0);
    push_u16(buf, 0); // cc_ref_count
    push_u16(buf, 2); // cc_val_count
    push_f64(buf, 0.0); // phy_range_min
    push_f64(buf, 0.0); // phy_range_max
    push_f64(buf, offset);
    push_f64(buf, factor);
}

fn write_dv(buf: &mut Vec<u8>, data: &[u8]) {
    buf.extend_from_slice(b"##DV");
    push_zeros(buf, 4);
    push_u64(buf, 24 + data.len() as u64);
    push_u64(buf, 0);
    buf.extend_from_slice(data);
}

fn write_ld(buf: &mut Vec<u8>, dv_ptr: i64) {
    buf.extend_from_slice(b"##LD");
    push_zeros(buf, 4);
    push_u64(buf, 56); // 16 header + 16 links + 24 data = 56 bytes
    push_u64(buf, 2); // 1 next + 1 data
    push_i64(buf, 0); // ld_ld_next = 0
    push_i64(buf, dv_ptr); // ld_data[0]
    push_u8(buf, 1); // ld_flags: bit 0 set (equal sample count)
    push_u8(buf, 255); // ld_zip_info
    push_u8(buf, 255);
    push_u8(buf, 0); // ld_flags_ext
    push_u32(buf, 1); // ld_count = 1
    push_u64(buf, 3); // ld_equal_sample_count = 3
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_virtual_data_channel_with_conversion() -> Result<()> {
    // Priority 4: Virtual data channel (cn_type == 6)
    // Physical value = 100.0 + k * 2.5
    let mut buf = Vec::new();
    write_id_block(&mut buf); // 0..64

    let hd_pos = buf.len() as i64;
    write_hd4(&mut buf, 0); // placeholder for dg_first

    let dg_pos = buf.len() as i64;
    write_dg4(&mut buf, 0, 0); // dg_data == 0

    let cg_pos = buf.len() as i64;
    write_cg4(&mut buf, 0, 5); // 5 cycles

    let cn_pos = buf.len() as i64;
    write_cn4(&mut buf, 6, 0, 0, 0, 0, 0, 0); // cn_type = 6 (virtual data), cn_bit_count = 0

    let tx_pos = buf.len() as i64;
    write_tx(&mut buf, "VirtualRamp");

    let cc_pos = buf.len() as i64;
    write_cc_linear(&mut buf, 2.5, 100.0); // offset = 100.0, factor = 2.5

    // Patch links
    let hd_dg_first = (hd_pos + 24) as usize;
    buf[hd_dg_first..hd_dg_first + 8].copy_from_slice(&dg_pos.to_le_bytes());

    let dg_cg_first = (dg_pos + 32) as usize;
    buf[dg_cg_first..dg_cg_first + 8].copy_from_slice(&cg_pos.to_le_bytes());

    let cg_cn_first = (cg_pos + 32) as usize;
    buf[cg_cn_first..cg_cn_first + 8].copy_from_slice(&cn_pos.to_le_bytes());

    let cn_tx = (cn_pos + 40) as usize;
    buf[cn_tx..cn_tx + 8].copy_from_slice(&tx_pos.to_le_bytes());

    let cn_cc = (cn_pos + 56) as usize;
    buf[cn_cc..cn_cc + 8].copy_from_slice(&cc_pos.to_le_bytes());

    let mut temp = NamedTempFile::new()?;
    temp.write_all(&buf)?;
    temp.flush()?;

    let mut mdf = Mdf::new(temp.path().to_str().unwrap())?;
    mdf.load_all_channels_data_in_memory()?;

    let cd = mdf.get_channel_data("VirtualRamp");
    if let Some(ChannelData::Float64(arr)) = cd {
        let values: Vec<f64> = arr.values().to_vec();
        assert_eq!(values, vec![100.0, 102.5, 105.0, 107.5, 110.0]);
    } else {
        panic!(
            "Expected Float64 converted data for VirtualRamp, got {:?}",
            cd
        );
    }

    Ok(())
}

#[test]
fn test_independent_column_oriented_dvblock() -> Result<()> {
    // Priority 1: Independent column-oriented channel with dg_data == 0 and cn_data -> DVBLOCK
    let mut buf = Vec::new();
    write_id_block(&mut buf);

    let hd_pos = buf.len() as i64;
    write_hd4(&mut buf, 0);

    let dg_pos = buf.len() as i64;
    write_dg4(&mut buf, 0, 0); // dg_data == 0!

    let cg_pos = buf.len() as i64;
    write_cg4(&mut buf, 0, 4); // 4 cycles

    let cn_pos = buf.len() as i64;
    write_cn4(&mut buf, 0, 4, 32, 0, 0, 0, 0); // cn_type = 0, cn_data_type = 4 (Float32 LE)

    let tx_pos = buf.len() as i64;
    write_tx(&mut buf, "ColFloat32");

    let dv_pos = buf.len() as i64;
    let floats = [1.25f32, 2.5f32, 5.0f32, 10.0f32];
    let mut dv_data = Vec::new();
    for f in floats {
        dv_data.extend_from_slice(&f.to_le_bytes());
    }
    write_dv(&mut buf, &dv_data);

    // Patch links
    let hd_dg = (hd_pos + 24) as usize;
    buf[hd_dg..hd_dg + 8].copy_from_slice(&dg_pos.to_le_bytes());

    let dg_cg = (dg_pos + 32) as usize;
    buf[dg_cg..dg_cg + 8].copy_from_slice(&cg_pos.to_le_bytes());

    let cg_cn = (cg_pos + 32) as usize;
    buf[cg_cn..cg_cn + 8].copy_from_slice(&cn_pos.to_le_bytes());

    let cn_tx = (cn_pos + 40) as usize;
    buf[cn_tx..cn_tx + 8].copy_from_slice(&tx_pos.to_le_bytes());

    let cn_data_ptr = (cn_pos + 64) as usize;
    buf[cn_data_ptr..cn_data_ptr + 8].copy_from_slice(&dv_pos.to_le_bytes());

    let mut temp = NamedTempFile::new()?;
    temp.write_all(&buf)?;
    temp.flush()?;

    let mut mdf = Mdf::new(temp.path().to_str().unwrap())?;
    mdf.load_all_channels_data_in_memory()?;

    if let Some(ChannelData::Float32(arr)) = mdf.get_channel_data("ColFloat32") {
        assert_eq!(arr.values(), &[1.25, 2.5, 5.0, 10.0]);
    } else {
        panic!("Expected Float32 for ColFloat32");
    }

    Ok(())
}

#[test]
fn test_independent_column_oriented_ldblock() -> Result<()> {
    // Priority 1: Independent column-oriented channel with dg_data == 0 and cn_data -> LDBLOCK -> DVBLOCK
    let mut buf = Vec::new();
    write_id_block(&mut buf);

    let hd_pos = buf.len() as i64;
    write_hd4(&mut buf, 0);

    let dg_pos = buf.len() as i64;
    write_dg4(&mut buf, 0, 0); // dg_data == 0!

    let cg_pos = buf.len() as i64;
    write_cg4(&mut buf, 0, 3); // 3 cycles

    let cn_pos = buf.len() as i64;
    write_cn4(&mut buf, 0, 5, 64, 0, 0, 0, 0); // cn_type = 0, cn_data_type = 5 (Float64 BE)

    let tx_pos = buf.len() as i64;
    write_tx(&mut buf, "ColFloat64BE");

    let ld_pos = buf.len() as i64;
    write_ld(&mut buf, 0); // placeholder for dv_ptr

    let dv_pos = buf.len() as i64;
    let doubles = [11.0f64, 22.0f64, 33.0f64];
    let mut dv_data = Vec::new();
    for d in doubles {
        dv_data.extend_from_slice(&d.to_be_bytes()); // BE!
    }
    write_dv(&mut buf, &dv_data);

    // Patch links
    let hd_dg = (hd_pos + 24) as usize;
    buf[hd_dg..hd_dg + 8].copy_from_slice(&dg_pos.to_le_bytes());

    let dg_cg = (dg_pos + 32) as usize;
    buf[dg_cg..dg_cg + 8].copy_from_slice(&cg_pos.to_le_bytes());

    let cg_cn = (cg_pos + 32) as usize;
    buf[cg_cn..cg_cn + 8].copy_from_slice(&cn_pos.to_le_bytes());

    let cn_tx = (cn_pos + 40) as usize;
    buf[cn_tx..cn_tx + 8].copy_from_slice(&tx_pos.to_le_bytes());

    let cn_data_ptr = (cn_pos + 64) as usize;
    buf[cn_data_ptr..cn_data_ptr + 8].copy_from_slice(&ld_pos.to_le_bytes());

    let ld_dv_ptr = (ld_pos + 32) as usize;
    buf[ld_dv_ptr..ld_dv_ptr + 8].copy_from_slice(&dv_pos.to_le_bytes());

    let mut temp = NamedTempFile::new()?;
    temp.write_all(&buf)?;
    temp.flush()?;

    let mut mdf = Mdf::new(temp.path().to_str().unwrap())?;
    mdf.load_all_channels_data_in_memory()?;

    if let Some(ChannelData::Float64(arr)) = mdf.get_channel_data("ColFloat64BE") {
        assert_eq!(arr.values(), &[11.0, 22.0, 33.0]);
    } else {
        panic!("Expected Float64 for ColFloat64BE");
    }

    Ok(())
}

#[test]
fn test_dg_template_array() -> Result<()> {
    // Priority 3: DG template array storage (ca_storage == 2)
    // 2x3 tensor across 4 cycles. Each element k has its samples in ca_data[k].
    let mut buf = Vec::new();
    write_id_block(&mut buf);

    let hd_pos = buf.len() as i64;
    write_hd4(&mut buf, 0);

    let dg_pos = buf.len() as i64;
    write_dg4(&mut buf, 0, 0); // dg_data == 0

    let cg_pos = buf.len() as i64;
    write_cg4(&mut buf, 0, 4); // 4 cycles

    let cn_pos = buf.len() as i64;
    // cn_type = 0, cn_data_type = 4 (Float32 LE), cn_bit_count = 32
    write_cn4(&mut buf, 0, 4, 32, 0, 0, 0, 0);

    let tx_pos = buf.len() as i64;
    write_tx(&mut buf, "DgTensorArray");

    // Write 6 DVBLOCKs (one per element of 2x3 array)
    // Element k at cycle t has value: (k * 10 + t) as f32
    let mut dv_ptrs = Vec::new();
    for k in 0..6 {
        let dv_pos = buf.len() as i64;
        dv_ptrs.push(dv_pos);
        let mut elem_bytes = Vec::new();
        for t in 0..4 {
            let val = (k * 10 + t) as f32;
            elem_bytes.extend_from_slice(&val.to_le_bytes());
        }
        write_dv(&mut buf, &elem_bytes);
    }

    // Write CABLOCK
    let ca_pos = buf.len() as i64;
    let mut ca = Ca4Block {
        ca_type: 0,    // Array
        ca_storage: 2, // DG template
        ca_ndim: 2,
        ca_dim_size: vec![2, 3], // 2 rows, 3 cols
        ca_data: Some(dv_ptrs.clone()),
        ca_cycle_count: Some(vec![4; 6]),
        ..Default::default()
    };
    ca.prepare_for_write();
    ca.ca_data = Some(dv_ptrs); // Preserve data pointers
    let mut ca_bytes = Vec::new();
    ca.write_to(&mut std::io::Cursor::new(&mut ca_bytes))?;
    buf.extend_from_slice(&ca_bytes);

    // Patch links
    let hd_dg = (hd_pos + 24) as usize;
    buf[hd_dg..hd_dg + 8].copy_from_slice(&dg_pos.to_le_bytes());

    let dg_cg = (dg_pos + 32) as usize;
    buf[dg_cg..dg_cg + 8].copy_from_slice(&cg_pos.to_le_bytes());

    let cg_cn = (cg_pos + 32) as usize;
    buf[cg_cn..cg_cn + 8].copy_from_slice(&cn_pos.to_le_bytes());

    let cn_tx = (cn_pos + 40) as usize;
    buf[cn_tx..cn_tx + 8].copy_from_slice(&tx_pos.to_le_bytes());

    let cn_compo = (cn_pos + 32) as usize;
    buf[cn_compo..cn_compo + 8].copy_from_slice(&ca_pos.to_le_bytes());

    let mut temp = NamedTempFile::new()?;
    temp.write_all(&buf)?;
    temp.flush()?;

    let mut mdf = Mdf::new(temp.path().to_str().unwrap())?;
    mdf.load_all_channels_data_in_memory()?;

    if let Some(ChannelData::ArrayDFloat32(tensor)) = mdf.get_channel_data("DgTensorArray") {
        let values = tensor.values_slice();
        assert_eq!(values.len(), 24); // 4 cycles * 6 elements
        // Check cycle 0: elements k=0..6 should be [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        assert_eq!(&values[0..6], &[0.0, 10.0, 20.0, 30.0, 40.0, 50.0]);
        // Check cycle 1: elements k=0..6 should be [1.0, 11.0, 21.0, 31.0, 41.0, 51.0]
        assert_eq!(&values[6..12], &[1.0, 11.0, 21.0, 31.0, 41.0, 51.0]);
    } else {
        panic!("Expected ArrayDFloat32 for DgTensorArray");
    }

    Ok(())
}

fn write_dt(buf: &mut Vec<u8>, data: &[u8]) {
    buf.extend_from_slice(b"##DT");
    push_zeros(buf, 4);
    push_u64(buf, 24 + data.len() as u64);
    push_u64(buf, 0);
    buf.extend_from_slice(data);
}

#[test]
fn test_cg_template_array() -> Result<()> {
    // Priority 3: CG template array storage (ca_storage == 1)
    // 1D tensor of 2 elements across 3 cycles in unsorted DT block.
    // Record ID for parent CG = 1. Auto-incremented: elem 0 = 1, elem 1 = 2.
    let mut buf = Vec::new();
    write_id_block(&mut buf);

    let hd_pos = buf.len() as i64;
    write_hd4(&mut buf, 0);

    let dg_pos = buf.len() as i64;
    // Unsorted DG: rec_id_size = 1
    buf.extend_from_slice(b"##DG");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 64);
    push_u64(&mut buf, 4);
    push_i64(&mut buf, 0); // next
    push_i64(&mut buf, 0); // cg_first placeholder
    push_i64(&mut buf, 0); // data placeholder
    push_i64(&mut buf, 0); // comment
    push_u8(&mut buf, 1); // rec_id_size = 1 (unsorted)
    push_zeros(&mut buf, 7);

    let cg_pos = buf.len() as i64;
    buf.extend_from_slice(b"##CG");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 104);
    push_u64(&mut buf, 6);
    push_i64(&mut buf, 0);
    push_i64(&mut buf, 0); // cn_first placeholder
    push_i64(&mut buf, 0);
    push_i64(&mut buf, 0);
    push_i64(&mut buf, 0);
    push_i64(&mut buf, 0);
    push_u64(&mut buf, 1); // record_id = 1
    push_u64(&mut buf, 3); // cycle_count = 3
    push_u16(&mut buf, 0); // cg_flags
    push_u16(&mut buf, 0); // cg_path_separator
    push_zeros(&mut buf, 4); // cg_reserved
    push_u32(&mut buf, 4); // data_bytes = 4 (f32)
    push_u32(&mut buf, 0); // inval_bytes

    let cn_pos = buf.len() as i64;
    write_cn4(&mut buf, 0, 4, 32, 0, 0, 0, 0); // f32 LE

    let tx_pos = buf.len() as i64;
    write_tx(&mut buf, "CgTensorArray");

    // Write CABLOCK
    let ca_pos = buf.len() as i64;
    let mut ca = Ca4Block {
        ca_type: 0,    // Array
        ca_storage: 1, // CG template
        ca_ndim: 1,
        ca_dim_size: vec![2], // 2 elements
        ca_cycle_count: Some(vec![3, 3]),
        ..Default::default()
    };
    ca.prepare_for_write();
    let mut ca_bytes = Vec::new();
    ca.write_to(&mut std::io::Cursor::new(&mut ca_bytes))?;
    buf.extend_from_slice(&ca_bytes);

    // Write DT block with interleaved records for elem 0 (rec_id=1) and elem 1 (rec_id=2)
    let mut dt_data = Vec::new();
    for t in 0..3 {
        // Elem 0
        dt_data.push(1u8); // rec_id = 1
        dt_data.extend_from_slice(&(10.0 + t as f32).to_le_bytes());
        // Elem 1
        dt_data.push(2u8); // rec_id = 2
        dt_data.extend_from_slice(&(20.0 + t as f32).to_le_bytes());
    }
    let dt_pos = buf.len() as i64;
    write_dt(&mut buf, &dt_data);

    // Patch links
    let hd_dg = (hd_pos + 24) as usize;
    buf[hd_dg..hd_dg + 8].copy_from_slice(&dg_pos.to_le_bytes());

    let dg_cg = (dg_pos + 32) as usize;
    buf[dg_cg..dg_cg + 8].copy_from_slice(&cg_pos.to_le_bytes());

    let dg_data = (dg_pos + 40) as usize;
    buf[dg_data..dg_data + 8].copy_from_slice(&dt_pos.to_le_bytes());

    let cg_cn = (cg_pos + 32) as usize;
    buf[cg_cn..cg_cn + 8].copy_from_slice(&cn_pos.to_le_bytes());

    let cn_tx = (cn_pos + 40) as usize;
    buf[cn_tx..cn_tx + 8].copy_from_slice(&tx_pos.to_le_bytes());

    let cn_compo = (cn_pos + 32) as usize;
    buf[cn_compo..cn_compo + 8].copy_from_slice(&ca_pos.to_le_bytes());

    let mut temp = NamedTempFile::new()?;
    temp.write_all(&buf)?;
    temp.flush()?;

    let mut mdf = Mdf::new(temp.path().to_str().unwrap())?;
    mdf.load_all_channels_data_in_memory()?;

    if let Some(ChannelData::ArrayDFloat32(tensor)) = mdf.get_channel_data("CgTensorArray") {
        let values = tensor.values_slice();
        assert_eq!(values.len(), 6); // 3 cycles * 2 elements
        // Check cycle 0: [10.0, 20.0]
        assert_eq!(&values[0..2], &[10.0, 20.0]);
        // Check cycle 1: [11.0, 21.0]
        assert_eq!(&values[2..4], &[11.0, 21.0]);
        // Check cycle 2: [12.0, 22.0]
        assert_eq!(&values[4..6], &[12.0, 22.0]);
    } else {
        panic!("Expected ArrayDFloat32 for CgTensorArray");
    }

    Ok(())
}

// ── Additional Conformance Tests ─────────────────────────────────────────────

fn write_dz(buf: &mut Vec<u8>, data: &[u8]) {
    use flate2::Compression;
    use flate2::write::DeflateEncoder;
    let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data).unwrap();
    let compressed = encoder.finish().unwrap();

    let block_len = 24 + 24 + compressed.len() as u64;
    buf.extend_from_slice(b"##DZ");
    push_zeros(buf, 4); // reserved
    push_u64(buf, block_len);
    push_u64(buf, 0); // links
    // DZ data members
    buf.extend_from_slice(b"DV"); // dz_org_block_type
    push_u8(buf, 0); // dz_zip_type: 0 = Deflate
    push_u8(buf, 0); // dz_reserved
    push_u32(buf, 0); // dz_zip_parameter
    push_u64(buf, data.len() as u64); // uncompressed length
    push_u64(buf, compressed.len() as u64); // compressed length
    buf.extend_from_slice(&compressed);
}

#[test]
fn test_string_null_termination_truncation() -> Result<()> {
    use arrow::array::Array;

    let mut buf = Vec::new();
    write_id_block(&mut buf);

    let hd_pos = buf.len() as i64;
    write_hd4(&mut buf, 0);

    let dg_pos = buf.len() as i64;
    write_dg4(&mut buf, 0, 0);

    let cg_pos = buf.len() as i64;
    write_cg4_with_data_bytes(&mut buf, 0, 3, 16); // 3 cycles, 16 bytes per record

    let cn_pos = buf.len() as i64;
    // Fixed string UTF-8: cn_type = 0, cn_data_type = 7, bit_count = 128 (16 bytes)
    write_cn4(&mut buf, 0, 7, 128, 0, 0, 0, 0);

    let tx_pos = buf.len() as i64;
    write_tx(&mut buf, "SensorName");

    // DT block with embedded '\0' junk after the actual string
    let mut dt_data = Vec::new();
    let records: [&[u8; 16]; 3] = [
        b"Speed_kmh\0XYZ123",
        b"EngineTemp\0\0\0\0\0\0",
        b"RPM\0JUNKJUNKJUNK",
    ];
    for r in records {
        dt_data.extend_from_slice(r);
    }
    let dt_pos = buf.len() as i64;
    write_dt(&mut buf, &dt_data);

    // Patch links
    let hd_dg = (hd_pos + 24) as usize;
    buf[hd_dg..hd_dg + 8].copy_from_slice(&dg_pos.to_le_bytes());

    let dg_cg = (dg_pos + 32) as usize;
    buf[dg_cg..dg_cg + 8].copy_from_slice(&cg_pos.to_le_bytes());

    let dg_data = (dg_pos + 40) as usize;
    buf[dg_data..dg_data + 8].copy_from_slice(&dt_pos.to_le_bytes());

    let cg_cn = (cg_pos + 32) as usize;
    buf[cg_cn..cg_cn + 8].copy_from_slice(&cn_pos.to_le_bytes());

    let cn_tx = (cn_pos + 40) as usize;
    buf[cn_tx..cn_tx + 8].copy_from_slice(&tx_pos.to_le_bytes());

    let mut temp = NamedTempFile::new()?;
    temp.write_all(&buf)?;
    temp.flush()?;

    let mut mdf = Mdf::new(temp.path().to_str().unwrap())?;
    mdf.load_all_channels_data_in_memory()?;

    if let Some(ChannelData::Utf8(array)) = mdf.get_channel_data("SensorName") {
        assert_eq!(array.len(), 3);
        // MDF 4.3 §6.16.4 requires truncating at the FIRST null character
        assert_eq!(array.value(0), "Speed_kmh");
        assert_eq!(array.value(1), "EngineTemp");
        assert_eq!(array.value(2), "RPM");
    } else {
        panic!("Expected Utf8 channel data for SensorName");
    }

    Ok(())
}

#[test]
fn test_virtual_data_channel_raw_ramp() -> Result<()> {
    let mut buf = Vec::new();
    write_id_block(&mut buf);

    let hd_pos = buf.len() as i64;
    write_hd4(&mut buf, 0);

    let dg_pos = buf.len() as i64;
    write_dg4(&mut buf, 0, 0); // dg_data = 0

    let cg_pos = buf.len() as i64;
    write_cg4_with_data_bytes(&mut buf, 0, 5, 0); // 5 cycles, 0 record bytes

    let cn_pos = buf.len() as i64;
    // Virtual data channel: cn_type = 6, cn_data_type = 0 (UInt64), bit_count = 0, cc_conversion = 0
    write_cn4(&mut buf, 6, 0, 0, 0, 0, 0, 0);

    let tx_pos = buf.len() as i64;
    write_tx(&mut buf, "RawVirtualRamp");

    // Patch links
    let hd_dg = (hd_pos + 24) as usize;
    buf[hd_dg..hd_dg + 8].copy_from_slice(&dg_pos.to_le_bytes());

    let dg_cg = (dg_pos + 32) as usize;
    buf[dg_cg..dg_cg + 8].copy_from_slice(&cg_pos.to_le_bytes());

    let cg_cn = (cg_pos + 32) as usize;
    buf[cg_cn..cg_cn + 8].copy_from_slice(&cn_pos.to_le_bytes());

    let cn_tx = (cn_pos + 40) as usize;
    buf[cn_tx..cn_tx + 8].copy_from_slice(&tx_pos.to_le_bytes());

    let mut temp = NamedTempFile::new()?;
    temp.write_all(&buf)?;
    temp.flush()?;

    let mut mdf = Mdf::new(temp.path().to_str().unwrap())?;
    mdf.load_all_channels_data_in_memory()?;

    // Without conversion, virtual data channel holds the raw ramp 0..N-1
    if let Some(ChannelData::UInt64(builder)) = mdf.get_channel_data("RawVirtualRamp") {
        let vals = builder.values_slice();
        assert_eq!(vals, &[0, 1, 2, 3, 4]);
    } else {
        panic!("Expected UInt64 raw ramp for RawVirtualRamp");
    }

    Ok(())
}

#[test]
fn test_independent_column_oriented_dzblock() -> Result<()> {
    let mut buf = Vec::new();
    write_id_block(&mut buf);

    let hd_pos = buf.len() as i64;
    write_hd4(&mut buf, 0);

    let dg_pos = buf.len() as i64;
    write_dg4(&mut buf, 0, 0); // dg_data = 0

    let cg_pos = buf.len() as i64;
    write_cg4_with_data_bytes(&mut buf, 0, 4, 0); // 4 cycles

    let cn_pos = buf.len() as i64;
    write_cn4(&mut buf, 0, 4, 32, 0, 0, 0, 0); // f32 LE

    let tx_pos = buf.len() as i64;
    write_tx(&mut buf, "CompressedColumnSignal");

    // Write compressed DZ block
    let mut raw_data = Vec::new();
    for v in [100.5f32, 200.5, 300.5, 400.5] {
        raw_data.extend_from_slice(&v.to_le_bytes());
    }
    let dz_pos = buf.len() as i64;
    write_dz(&mut buf, &raw_data);

    // Patch links
    let hd_dg = (hd_pos + 24) as usize;
    buf[hd_dg..hd_dg + 8].copy_from_slice(&dg_pos.to_le_bytes());

    let dg_cg = (dg_pos + 32) as usize;
    buf[dg_cg..dg_cg + 8].copy_from_slice(&cg_pos.to_le_bytes());

    let cg_cn = (cg_pos + 32) as usize;
    buf[cg_cn..cg_cn + 8].copy_from_slice(&cn_pos.to_le_bytes());

    let cn_tx = (cn_pos + 40) as usize;
    buf[cn_tx..cn_tx + 8].copy_from_slice(&tx_pos.to_le_bytes());

    // cn_data -> DZ block
    let cn_data = (cn_pos + 64) as usize;
    buf[cn_data..cn_data + 8].copy_from_slice(&dz_pos.to_le_bytes());

    let mut temp = NamedTempFile::new()?;
    temp.write_all(&buf)?;
    temp.flush()?;

    let mut mdf = Mdf::new(temp.path().to_str().unwrap())?;
    mdf.load_all_channels_data_in_memory()?;

    if let Some(ChannelData::Float32(builder)) = mdf.get_channel_data("CompressedColumnSignal") {
        let values = builder.values_slice();
        assert_eq!(values, &[100.5, 200.5, 300.5, 400.5]);
    } else {
        panic!("Expected Float32 for CompressedColumnSignal");
    }

    Ok(())
}

#[test]
fn test_bitfield_rounding_edge_cases() -> Result<()> {
    use arrow::array::Float64Builder;
    use mdfr::data_holder::channel_data::ChannelData;
    use mdfr::mdfinfo::mdfinfo4::{Cc4Block, CcVal, Cn4, SharableBlocks};

    let mut sharable = SharableBlocks::default();
    let cc_ptr = 5000i64;
    // CCBLOCK type 7: Value to Text with keys 0, 1, 2, -1
    // cc_val: [0.0, 1.0, 2.0, -1.0], cc_ref: [txt0, txt1, txt2, txt_neg1, default]
    let cc = Cc4Block::new_with_val(
        7,
        CcVal::Real(vec![0.0, 1.0, 2.0, -1.0]),
        vec![6000, 6001, 6002, 6003, 6004],
    );
    sharable.cc.insert(cc_ptr, cc);
    sharable.create_tx(6000, "VAL_0".to_string());
    sharable.create_tx(6001, "VAL_1".to_string());
    sharable.create_tx(6002, "VAL_2".to_string());
    sharable.create_tx(6003, "VAL_NEG_1".to_string());
    sharable.create_tx(6004, "VAL_DEFAULT".to_string());

    let cc_val = vec![0xFFFF_FFFF_FFFF_FFFFu64]; // mask covering all bits
    let cc_ref = vec![cc_ptr];

    // Raw float values:
    // -0.9 -> rounds to -1 => VAL_NEG_1
    // -0.4 -> rounds to 0  => VAL_0
    // 0.5  -> rounds to 1  => VAL_1
    // 1.4  -> rounds to 1  => VAL_1
    // 1.5  -> rounds to 2  => VAL_2
    let mut builder = Float64Builder::new();
    builder.append_value(-0.9);
    builder.append_value(-0.4);
    builder.append_value(0.5);
    builder.append_value(1.4);
    builder.append_value(1.5);

    let mut cn = Cn4 {
        data: ChannelData::Float64(builder.finish()),
        ..Default::default()
    };

    mdfr::mdfreader::conversions4::bitfield_text_table(&mut cn, &cc_val, &cc_ref, &sharable)?;

    if let ChannelData::Utf8(ref s) = cn.data {
        assert_eq!(s.value(0), "VAL_NEG_1");
        assert_eq!(s.value(1), "VAL_0");
        assert_eq!(s.value(2), "VAL_1");
        assert_eq!(s.value(3), "VAL_1");
        assert_eq!(s.value(4), "VAL_2");
    } else {
        panic!("Expected Utf8 channel data");
    }

    Ok(())
}

#[test]
fn test_unaligned_integer_n_bytes_5() -> Result<()> {
    // According to §6.6.3 / §6.26.5.2:
    // A 28-bit integer with bit_offset = 6 spans cn_bit_count + cn_bit_offset = 34 bits -> n_bytes = 5.
    // data_type_init routes n_bytes == 5 to UInt64.
    let mut buf = Vec::new();
    write_id_block(&mut buf);

    let hd_pos = buf.len() as i64;
    write_hd4(&mut buf, 0);

    let dg_pos = buf.len() as i64;
    write_dg4(&mut buf, 0, 0);

    let cg_pos = buf.len() as i64;
    // 2 cycles, 5 bytes per record
    write_cg4_with_data_bytes(&mut buf, 0, 2, 5);

    let cn_pos = buf.len() as i64;
    // cn_type = 0, cn_data_type = 0 (UInt LE), bit_offset = 6, byte_offset = 0, bit_count = 28
    write_cn4_with_offset(&mut buf, 0, 0, 6, 0, 28, 0, 0, 0, 0);

    let tx_pos = buf.len() as i64;
    write_tx(&mut buf, "UnalignedInt28Bit");

    let dt_pos = buf.len() as i64;
    // Value 0: 12345u64 (0x3039). Shifted by 6: 790080 (0xC0E40). Bits 0..5 filled with 0b111001 (57).
    let raw0 = (12345u64 << 6) | 57;
    // Value 1: 0x0123_4567u64 (28-bit max). Shifted by 6. Bits 0..5 filled with 42.
    let raw1 = (0x0123_4567u64 << 6) | 42;
    let mut dt_data = Vec::new();
    dt_data.extend_from_slice(&raw0.to_le_bytes()[..5]);
    dt_data.extend_from_slice(&raw1.to_le_bytes()[..5]);
    write_dt(&mut buf, &dt_data);

    // Backpatch pointers
    let hd_dg = (hd_pos + 24) as usize;
    buf[hd_dg..hd_dg + 8].copy_from_slice(&dg_pos.to_le_bytes());

    let dg_cg = (dg_pos + 32) as usize;
    buf[dg_cg..dg_cg + 8].copy_from_slice(&cg_pos.to_le_bytes());

    let dg_data = (dg_pos + 40) as usize;
    buf[dg_data..dg_data + 8].copy_from_slice(&dt_pos.to_le_bytes());

    let cg_cn = (cg_pos + 32) as usize;
    buf[cg_cn..cg_cn + 8].copy_from_slice(&cn_pos.to_le_bytes());

    let cn_tx = (cn_pos + 40) as usize;
    buf[cn_tx..cn_tx + 8].copy_from_slice(&tx_pos.to_le_bytes());

    let mut temp = NamedTempFile::new()?;
    temp.write_all(&buf)?;
    temp.flush()?;

    let mut mdf = Mdf::new(temp.path().to_str().unwrap())?;
    mdf.load_all_channels_data_in_memory()?;

    if let Some(ChannelData::UInt64(builder)) = mdf.get_channel_data("UnalignedInt28Bit") {
        let vals = builder.values_slice();
        assert_eq!(vals, &[12345, 0x0123_4567]);
    } else {
        panic!("Expected UInt64 channel data for 5-byte integer");
    }

    Ok(())
}

#[test]
fn test_integer_over_64_bits_spec_limit() -> Result<()> {
    // According to §6.6.3:
    // cn_bit_count + cn_bit_offset <= 64 (unless cn_bit_count == 64 and cn_bit_offset == 0).
    // An integer with cn_bit_count = 60 and cn_bit_offset = 6 spans 66 bits -> n_bytes = 9.
    // data_type_init must reject this per specification with a descriptive error.
    let mut buf = Vec::new();
    write_id_block(&mut buf);

    let hd_pos = buf.len() as i64;
    write_hd4(&mut buf, 0);

    let dg_pos = buf.len() as i64;
    write_dg4(&mut buf, 0, 0);

    let cg_pos = buf.len() as i64;
    write_cg4_with_data_bytes(&mut buf, 0, 1, 9);

    let cn_pos = buf.len() as i64;
    // cn_type = 0, cn_data_type = 0, bit_offset = 6, byte_offset = 0, bit_count = 60 -> 66 bits -> n_bytes = 9
    write_cn4_with_offset(&mut buf, 0, 0, 6, 0, 60, 0, 0, 0, 0);

    let tx_pos = buf.len() as i64;
    write_tx(&mut buf, "OversizedInt60Bit");

    let dt_pos = buf.len() as i64;
    write_dt(&mut buf, &[0u8; 9]);

    // Backpatch pointers
    let hd_dg = (hd_pos + 24) as usize;
    buf[hd_dg..hd_dg + 8].copy_from_slice(&dg_pos.to_le_bytes());

    let dg_cg = (dg_pos + 32) as usize;
    buf[dg_cg..dg_cg + 8].copy_from_slice(&cg_pos.to_le_bytes());

    let dg_data = (dg_pos + 40) as usize;
    buf[dg_data..dg_data + 8].copy_from_slice(&dt_pos.to_le_bytes());

    let cg_cn = (cg_pos + 32) as usize;
    buf[cg_cn..cg_cn + 8].copy_from_slice(&cn_pos.to_le_bytes());

    let cn_tx = (cn_pos + 40) as usize;
    buf[cn_tx..cn_tx + 8].copy_from_slice(&tx_pos.to_le_bytes());

    let mut temp = NamedTempFile::new()?;
    temp.write_all(&buf)?;
    temp.flush()?;

    let mdf = Mdf::new(temp.path().to_str().unwrap());
    assert!(
        mdf.is_err(),
        "Expected parsing to fail for integer exceeding 64 bits"
    );
    let err_msg = format!("{:#}", mdf.unwrap_err());
    assert!(
        err_msg.contains("exceeding 64 bits limit"),
        "Unexpected error message: {err_msg}"
    );

    Ok(())
}

#[test]
fn test_be_7byte_uint64_and_array_uint64() -> Result<()> {
    use mdfr::data_holder::channel_data::data_type_init;
    use mdfr::data_holder::tensor_arrow::Order;
    use mdfr::mdfinfo::mdfinfo4::{Cn4, CnType, Endianness};
    use mdfr::mdfreader::data_read4::{read_channels_from_bytes, read_one_channel_array};

    // 1) Scalar 7-byte BE UInt64: 0x01_02_03_04_05_06_07
    let mut cn = Cn4 {
        n_bytes: 7,
        endian: Endianness::Big,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        pos_byte_beg: 0,
        should_read: true,
        data: data_type_init(0, 1, 7, 1, 1)?,
        ..Default::default()
    };
    cn.block.cn_data_type = 1; // UInt BE
    cn.block.cn_bit_count = 56;
    cn.data = cn.data.zeros(0, 2, 7, (vec![1], Order::RowMajor))?;
    let raw_bytes = [
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, // rec 0
        0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, // rec 1 -> 256
    ];
    read_one_channel_array(&raw_bytes, &mut cn, 2)?;
    if let ChannelData::UInt64(arr) = &cn.data {
        assert_eq!(arr.values_slice(), &[0x0001_0203_0405_0607u64, 256u64]);
    } else {
        panic!("expected UInt64");
    }

    // Also test via read_channels_from_bytes
    cn.init_builder(2)?;
    let mut channels = CnType::default();
    channels.insert(0, cn);
    read_channels_from_bytes(&raw_bytes, &mut channels, 7, 0, false)?;
    if let Some(b) = channels.get_mut(&0).unwrap().builder.take() {
        channels.get_mut(&0).unwrap().data = b.finish();
    }
    if let ChannelData::UInt64(arr) = &channels[&0].data {
        assert_eq!(arr.values_slice(), &[0x0001_0203_0405_0607u64, 256u64]);
    } else {
        panic!("expected UInt64");
    }

    // 2) ArrayDUInt64 8-byte BE
    let shape = (vec![2, 2], Order::RowMajor);
    let mut cn_arr = Cn4 {
        n_bytes: 8,
        endian: Endianness::Big,
        list_size: 2,
        shape: shape.clone(),
        pos_byte_beg: 0,
        should_read: true,
        data: data_type_init(0, 1, 8, 2, 1)?,
        ..Default::default()
    };
    cn_arr.block.cn_data_type = 1; // UInt BE
    cn_arr.block.cn_bit_count = 64;
    cn_arr.init_builder(2)?;
    let mut raw_arr = Vec::new();
    for v in [10u64, 20, 30, 40] {
        raw_arr.extend_from_slice(&v.to_be_bytes());
    }
    let mut channels_arr = CnType::default();
    channels_arr.insert(0, cn_arr);
    read_channels_from_bytes(&raw_arr, &mut channels_arr, 16, 0, false)?;
    if let Some(b) = channels_arr.get_mut(&0).unwrap().builder.take() {
        channels_arr.get_mut(&0).unwrap().data = b.finish();
    }
    if let ChannelData::ArrayDUInt64(arr) = &channels_arr[&0].data {
        assert_eq!(arr.values_slice(), &[10u64, 20, 30, 40]);
    } else {
        panic!("expected ArrayDUInt64");
    }
    Ok(())
}

#[test]
fn test_remainder_and_truncated_chunks_rejected() -> Result<()> {
    use mdfr::data_holder::channel_data::data_type_init;
    use mdfr::data_holder::tensor_arrow::Order;
    use mdfr::mdfinfo::mdfinfo4::{Cn4, CnType, Endianness};
    use mdfr::mdfreader::data_read4::{read_channels_from_bytes, read_one_channel_array};

    let mut cn = Cn4 {
        n_bytes: 8,
        endian: Endianness::Little,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        pos_byte_beg: 0,
        should_read: true,
        data: data_type_init(0, 4, 8, 1, 1)?,
        ..Default::default()
    };
    cn.block.cn_data_type = 4; // Float LE
    cn.block.cn_bit_count = 64;
    cn.data = cn.data.zeros(0, 2, 8, (vec![1], Order::RowMajor))?;

    let bytes_with_remainder = [0u8; 17];
    assert!(read_one_channel_array(&bytes_with_remainder, &mut cn, 2).is_err());

    let truncated_bytes = [0u8; 8];
    assert!(read_one_channel_array(&truncated_bytes, &mut cn, 2).is_err());

    let mut channels = CnType::default();
    channels.insert(0, cn);
    assert!(read_channels_from_bytes(&bytes_with_remainder, &mut channels, 8, 0, false).is_err());
    Ok(())
}

#[test]
fn test_mdf4_spec_fig45_le_and_fig46_be_bit_offset_mask() -> Result<()> {
    use mdfr::data_holder::channel_data::data_type_init;
    use mdfr::data_holder::tensor_arrow::Order;
    use mdfr::mdfinfo::mdfinfo4::{
        Blockheader4Short, Cg4, Cg4Block, Cn4, CnType, Dg4, Dg4Block, Endianness,
    };
    use mdfr::mdfreader::data_read4::read_channels_from_bytes;
    use mdfr::mdfreader::mdfreader4::apply_bit_mask_offset;
    use std::collections::HashSet;

    // MDF 4.3.0 §6.26.5.2 Figure 45:
    // Bytes on disk: [0xAB, 0xCD, 0xEF]
    // Little Endian unsigned (cn_data_type = 0), cn_byte_offset = 0, cn_bit_offset = 6, cn_bit_count = 14
    // -> n_bytes = 3 -> UInt32
    // Spec value: 16182 (0b11_1111_0011_0110 = 0x3F36)
    let bytes = [0xABu8, 0xCD, 0xEF];

    let mut cn_le = Cn4 {
        n_bytes: 3,
        endian: Endianness::Little,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        pos_byte_beg: 0,
        should_read: true,
        data: data_type_init(0, 0, 3, 1, 1)?,
        ..Default::default()
    };
    cn_le.block.cn_data_type = 0; // Unsigned LE
    cn_le.block.cn_bit_offset = 6;
    cn_le.block.cn_bit_count = 14;
    cn_le.init_builder(1)?;

    // MDF 4.3.0 §6.26.5.2 Figure 46:
    // Bytes on disk: [0xAB, 0xCD, 0xEF]
    // Big Endian unsigned (cn_data_type = 1), cn_bit_offset = 6, cn_bit_count = 14 -> n_bytes = 3 -> UInt32
    // Spec value: 12087 (0b10_1111_0011_0111 = 0x2F37)
    let mut cn_be = Cn4 {
        n_bytes: 3,
        endian: Endianness::Big,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        pos_byte_beg: 0,
        should_read: true,
        data: data_type_init(0, 1, 3, 1, 1)?,
        ..Default::default()
    };
    cn_be.block.cn_data_type = 1; // Unsigned BE
    cn_be.block.cn_bit_offset = 6;
    cn_be.block.cn_bit_count = 14;
    cn_be.init_builder(1)?;

    // Signed LE 14-bit with MSB=1 (0x3F36 has bit 13 set -> negative in 14-bit two's complement:
    // 16182 - 16384 = -202)
    let mut cn_signed_le = Cn4 {
        n_bytes: 3,
        endian: Endianness::Little,
        list_size: 1,
        shape: (vec![1], Order::RowMajor),
        pos_byte_beg: 0,
        should_read: true,
        data: data_type_init(0, 2, 3, 1, 1)?,
        ..Default::default()
    };
    cn_signed_le.block.cn_data_type = 2; // Signed LE
    cn_signed_le.block.cn_bit_offset = 6;
    cn_signed_le.block.cn_bit_count = 14;
    cn_signed_le.init_builder(1)?;

    let mut cn_map = CnType::default();
    cn_map.insert(0, cn_le);
    cn_map.insert(1, cn_be);
    cn_map.insert(2, cn_signed_le);

    read_channels_from_bytes(&bytes, &mut cn_map, 3, 0, false)?;

    let cg_block = Cg4Block {
        cg_cycle_count: 1,
        cg_data_bytes: 3,
        ..Default::default()
    };

    let mut dg = Dg4 {
        block: Dg4Block::default(),
        cg: Default::default(),
        recovered_dl_data: Default::default(),
    };
    dg.cg.insert(
        0,
        Cg4 {
            header: Blockheader4Short::default(),
            block: cg_block,
            cn: cn_map,
            master_channel_name: None,
            channel_names: HashSet::new(),
            block_position: 0,
            record_length: 3,
            vlsd_cg: None,
            invalid_bytes: None,
            sr: Vec::new(),
        },
    );

    apply_bit_mask_offset(&mut dg).map_err(|e| anyhow::anyhow!("{e}"))?;

    let cg = &dg.cg[&0];
    if let ChannelData::UInt32(arr) = &cg.cn[&0].data {
        assert_eq!(arr.values_slice()[0], 16182);
    } else {
        panic!("expected UInt32 for Fig 45 LE");
    }
    if let ChannelData::UInt32(arr) = &cg.cn[&1].data {
        assert_eq!(arr.values_slice()[0], 12087);
    } else {
        panic!("expected UInt32 for Fig 46 BE");
    }
    if let ChannelData::Int32(arr) = &cg.cn[&2].data {
        assert_eq!(arr.values_slice()[0], -202);
    } else {
        panic!("expected Int32 for signed 14-bit LE");
    }
    Ok(())
}

fn open_mdf_from_bytes(bytes: &[u8]) -> Result<(NamedTempFile, Mdf)> {
    let mut tmp = NamedTempFile::new()?;
    tmp.write_all(bytes)?;
    tmp.flush()?;
    let mdf = Mdf::new(tmp.path().to_str().unwrap())?;
    Ok((tmp, mdf))
}

#[test]
fn test_sr4_block_links_and_64_byte_layout() -> Result<()> {
    use binrw::BinWrite;
    use mdfr::mdfinfo::MdfInfo;
    use mdfr::mdfinfo::mdfinfo4::Sr4Block;

    let sr_def = Sr4Block::default();
    assert_eq!(sr_def.sr_links, 2);
    let mut raw_body = std::io::Cursor::new(Vec::new());
    sr_def.write_le(&mut raw_body)?;
    assert_eq!(raw_body.into_inner().len(), 48); // 16B header + 48B body = 64B block

    // Build an MDF4 file containing ID + HD + DG + CG + 64-byte ##SR
    let mut buf = Vec::new();
    write_id_block(&mut buf);
    let hd_pos = buf.len() as i64;
    write_hd4(&mut buf, 0);
    let dg_pos = buf.len() as i64;
    write_dg4(&mut buf, 0, 0);
    let cg_pos = buf.len() as i64;
    write_cg4(&mut buf, 0, 0);
    let sr_pos = buf.len() as i64;
    buf.extend_from_slice(b"##SR");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 64);
    push_u64(&mut buf, 2); // sr_links
    push_i64(&mut buf, 0); // sr_sr_next
    push_i64(&mut buf, 0); // sr_data
    push_u64(&mut buf, 42); // sr_cycle_count
    push_f64(&mut buf, 0.25); // sr_interval
    push_u8(&mut buf, 1); // sr_sync_type = time
    push_u8(&mut buf, 0); // sr_flags
    push_zeros(&mut buf, 6);

    buf[(hd_pos + 24) as usize..(hd_pos + 32) as usize].copy_from_slice(&dg_pos.to_le_bytes());
    buf[(dg_pos + 32) as usize..(dg_pos + 40) as usize].copy_from_slice(&cg_pos.to_le_bytes());
    buf[(cg_pos + 56) as usize..(cg_pos + 64) as usize].copy_from_slice(&sr_pos.to_le_bytes());

    let (_tmp, mdf) = open_mdf_from_bytes(&buf)?;
    let MdfInfo::V4(info) = &mdf.mdf_info else {
        panic!("expected MDF4");
    };
    let cg = &info.dg[&dg_pos].cg[&0];
    assert_eq!(cg.sr.len(), 1);
    assert_eq!(cg.sr[0].sr_links, 2);
    assert_eq!(cg.sr[0].sr_cycle_count, 42);
    assert!((cg.sr[0].sr_interval - 0.25).abs() < 1e-12);
    assert_eq!(cg.sr[0].sr_sync_type, 1);
    Ok(())
}

#[test]
fn test_structure_composition_preserves_member_channel() -> Result<()> {
    // MDF4-002: parent structure CN with Compo::CN must not overwrite flattened child M1 at offset 0.
    let mut buf = Vec::new();
    write_id_block(&mut buf);
    let hd_pos = buf.len() as i64;
    write_hd4(&mut buf, 0);
    let dg_pos = buf.len() as i64;
    write_dg4(&mut buf, 0, 0);
    let cg_pos = buf.len() as i64;
    write_cg4_with_data_bytes(&mut buf, 0, 2, 2); // 2 cycles x 2 bytes (u16)

    let parent_cn_pos = buf.len() as i64;
    write_cn4(&mut buf, 0, 0, 16, 0, 0, 0, 0);
    let parent_tx_pos = buf.len() as i64;
    write_tx(&mut buf, "StructParent");

    let member_cn_pos = buf.len() as i64;
    write_cn4(&mut buf, 0, 0, 16, 0, 0, 0, 0);
    let member_tx_pos = buf.len() as i64;
    write_tx(&mut buf, "StructLeaf");

    let dt_pos = buf.len() as i64;
    buf.extend_from_slice(b"##DT");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 28);
    push_u64(&mut buf, 0);
    push_u16(&mut buf, 111);
    push_u16(&mut buf, 222);

    buf[(hd_pos + 24) as usize..(hd_pos + 32) as usize].copy_from_slice(&dg_pos.to_le_bytes());
    buf[(dg_pos + 32) as usize..(dg_pos + 40) as usize].copy_from_slice(&cg_pos.to_le_bytes());
    buf[(dg_pos + 40) as usize..(dg_pos + 48) as usize].copy_from_slice(&dt_pos.to_le_bytes());
    buf[(cg_pos + 32) as usize..(cg_pos + 40) as usize]
        .copy_from_slice(&parent_cn_pos.to_le_bytes());

    // Parent CN: cn_composition -> member_cn_pos, cn_tx_name -> parent_tx_pos
    buf[(parent_cn_pos + 32) as usize..(parent_cn_pos + 40) as usize]
        .copy_from_slice(&member_cn_pos.to_le_bytes());
    buf[(parent_cn_pos + 40) as usize..(parent_cn_pos + 48) as usize]
        .copy_from_slice(&parent_tx_pos.to_le_bytes());
    // Member CN: cn_tx_name -> member_tx_pos
    buf[(member_cn_pos + 40) as usize..(member_cn_pos + 48) as usize]
        .copy_from_slice(&member_tx_pos.to_le_bytes());

    let (_tmp, mut mdf) = open_mdf_from_bytes(&buf)?;
    mdf.load_all_channels_data_in_memory()?;
    assert!(mdf.get_channel_data("StructParent").is_none());
    let ChannelData::UInt16(arr) = mdf
        .get_channel_data("StructLeaf")
        .expect("StructLeaf should remain accessible")
    else {
        panic!("expected UInt16 for StructLeaf");
    };
    assert_eq!(arr.values_slice(), &[111, 222]);
    Ok(())
}

#[test]
fn test_canopen_date_and_time_prefixed_subchannels() -> Result<()> {
    // MDF4-018: CANopenDate (11, 7B) and CANopenTime (12, 6B) create prefixed synthetic sub-channels.
    let mut buf = Vec::new();
    write_id_block(&mut buf);
    let hd_pos = buf.len() as i64;
    write_hd4(&mut buf, 0);
    let dg_pos = buf.len() as i64;
    write_dg4(&mut buf, 0, 0);
    let cg_pos = buf.len() as i64;
    write_cg4_with_data_bytes(&mut buf, 0, 1, 13);

    let cn_date_pos = buf.len() as i64;
    write_cn4_with_offset(&mut buf, 0, 13, 0, 0, 56, 0, 0, 0, 0);
    let tx_date_pos = buf.len() as i64;
    write_tx(&mut buf, "EvtDate");

    let cn_time_pos = buf.len() as i64;
    write_cn4_with_offset(&mut buf, 0, 14, 0, 7, 48, 0, 0, 0, 0);
    let tx_time_pos = buf.len() as i64;
    write_tx(&mut buf, "EvtTime");

    // 7 bytes CANopenDate (ms=1500, min=25, hour=14, day=7, month=11, year=26)
    // + 6 bytes CANopenTime (ms=60000, days=500)
    let dt_pos = buf.len() as i64;
    buf.extend_from_slice(b"##DT");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 24 + 13);
    push_u64(&mut buf, 0);
    push_u16(&mut buf, 1500); // ms
    push_u8(&mut buf, 25); // min
    push_u8(&mut buf, 14); // hour
    push_u8(&mut buf, 7); // day + dow=0
    push_u8(&mut buf, 11); // month
    push_u8(&mut buf, 26); // year
    push_u32(&mut buf, 60_000); // CANopenTime ms
    push_u16(&mut buf, 500); // CANopenTime days

    buf[(hd_pos + 24) as usize..(hd_pos + 32) as usize].copy_from_slice(&dg_pos.to_le_bytes());
    buf[(dg_pos + 32) as usize..(dg_pos + 40) as usize].copy_from_slice(&cg_pos.to_le_bytes());
    buf[(dg_pos + 40) as usize..(dg_pos + 48) as usize].copy_from_slice(&dt_pos.to_le_bytes());
    buf[(cg_pos + 32) as usize..(cg_pos + 40) as usize].copy_from_slice(&cn_date_pos.to_le_bytes());

    buf[(cn_date_pos + 24) as usize..(cn_date_pos + 32) as usize]
        .copy_from_slice(&cn_time_pos.to_le_bytes());
    buf[(cn_date_pos + 40) as usize..(cn_date_pos + 48) as usize]
        .copy_from_slice(&tx_date_pos.to_le_bytes());
    buf[(cn_time_pos + 40) as usize..(cn_time_pos + 48) as usize]
        .copy_from_slice(&tx_time_pos.to_le_bytes());

    let (_tmp, mut mdf) = open_mdf_from_bytes(&buf)?;
    mdf.load_all_channels_data_in_memory()?;

    let ChannelData::UInt16(ms) = mdf.get_channel_data("EvtDate_ms").unwrap() else {
        panic!("expected EvtDate_ms UInt16");
    };
    assert_eq!(ms.values_slice(), &[1500]);
    let ChannelData::UInt8(min) = mdf.get_channel_data("EvtDate_min").unwrap() else {
        panic!("expected EvtDate_min UInt8");
    };
    assert_eq!(min.values_slice(), &[25]);
    let ChannelData::UInt8(hour) = mdf.get_channel_data("EvtDate_hour").unwrap() else {
        panic!("expected EvtDate_hour UInt8");
    };
    assert_eq!(hour.values_slice(), &[14]);
    let ChannelData::UInt8(day) = mdf.get_channel_data("EvtDate_day").unwrap() else {
        panic!("expected EvtDate_day UInt8");
    };
    assert_eq!(day.values_slice(), &[7]);
    let ChannelData::UInt8(month) = mdf.get_channel_data("EvtDate_month").unwrap() else {
        panic!("expected EvtDate_month UInt8");
    };
    assert_eq!(month.values_slice(), &[11]);
    let ChannelData::UInt8(year) = mdf.get_channel_data("EvtDate_year").unwrap() else {
        panic!("expected EvtDate_year UInt8");
    };
    assert_eq!(year.values_slice(), &[26]);

    let ChannelData::UInt32(tms) = mdf.get_channel_data("EvtTime_ms").unwrap() else {
        panic!("expected EvtTime_ms UInt32");
    };
    assert_eq!(tms.values_slice(), &[60_000]);
    let ChannelData::UInt16(tdays) = mdf.get_channel_data("EvtTime_day").unwrap() else {
        panic!("expected EvtTime_day UInt16");
    };
    assert_eq!(tdays.values_slice(), &[500]);
    Ok(())
}

#[test]
fn test_half_float_and_complex_half_float_decode() -> Result<()> {
    // Gap 5: cn_data_type = 4 (Float LE, 16-bit f16) and 15 (Complex LE, 32-bit 2xf16).
    let mut buf = Vec::new();
    write_id_block(&mut buf);
    let hd_pos = buf.len() as i64;
    write_hd4(&mut buf, 0);
    let dg_pos = buf.len() as i64;
    write_dg4(&mut buf, 0, 0);
    let cg_pos = buf.len() as i64;
    write_cg4_with_data_bytes(&mut buf, 0, 2, 6); // 2B f16 + 4B complex f16 per cycle

    let cn_f16_pos = buf.len() as i64;
    write_cn4_with_offset(&mut buf, 0, 4, 0, 0, 16, 0, 0, 0, 0);
    let tx_f16_pos = buf.len() as i64;
    write_tx(&mut buf, "HalfFloat");

    let cn_cf16_pos = buf.len() as i64;
    write_cn4_with_offset(&mut buf, 0, 15, 0, 2, 32, 0, 0, 0, 0);
    let tx_cf16_pos = buf.len() as i64;
    write_tx(&mut buf, "ComplexHalf");

    // f16: 1.0 = 0x3C00, -2.5 = 0xC100; complex f16: (1.5=0x3E00, -0.5=0xB800), (2.0=0x4000, 4.0=0x4400)
    let dt_pos = buf.len() as i64;
    buf.extend_from_slice(b"##DT");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 24 + 12);
    push_u64(&mut buf, 0);
    push_u16(&mut buf, 0x3C00);
    push_u16(&mut buf, 0x3E00);
    push_u16(&mut buf, 0xB800);
    push_u16(&mut buf, 0xC100);
    push_u16(&mut buf, 0x4000);
    push_u16(&mut buf, 0x4400);

    buf[(hd_pos + 24) as usize..(hd_pos + 32) as usize].copy_from_slice(&dg_pos.to_le_bytes());
    buf[(dg_pos + 32) as usize..(dg_pos + 40) as usize].copy_from_slice(&cg_pos.to_le_bytes());
    buf[(dg_pos + 40) as usize..(dg_pos + 48) as usize].copy_from_slice(&dt_pos.to_le_bytes());
    buf[(cg_pos + 32) as usize..(cg_pos + 40) as usize].copy_from_slice(&cn_f16_pos.to_le_bytes());
    buf[(cn_f16_pos + 24) as usize..(cn_f16_pos + 32) as usize]
        .copy_from_slice(&cn_cf16_pos.to_le_bytes());
    buf[(cn_f16_pos + 40) as usize..(cn_f16_pos + 48) as usize]
        .copy_from_slice(&tx_f16_pos.to_le_bytes());
    buf[(cn_cf16_pos + 40) as usize..(cn_cf16_pos + 48) as usize]
        .copy_from_slice(&tx_cf16_pos.to_le_bytes());

    let (_tmp, mut mdf) = open_mdf_from_bytes(&buf)?;
    mdf.load_all_channels_data_in_memory()?;

    let ChannelData::Float32(hf) = mdf.get_channel_data("HalfFloat").unwrap() else {
        panic!("expected Float32 for decoded f16");
    };
    assert_eq!(hf.values_slice(), &[1.0f32, -2.5f32]);

    let ChannelData::Complex32(cf) = mdf.get_channel_data("ComplexHalf").unwrap() else {
        panic!("expected Complex32 for decoded Complex f16");
    };
    let vals = cf.values_slice();
    assert_eq!(vals.len(), 4);
    assert_eq!(vals, &[1.5f32, -0.5f32, 2.0f32, 4.0f32]);
    Ok(())
}

#[test]
fn test_dl_zero_pointer_and_gd4_36_byte_offset() -> Result<()> {
    // MDF4-005 (GD4_STRUCT_SIZE == 36) & MDF4-013 (skip NIL pointers in dl_data).
    let mut buf = Vec::new();
    write_id_block(&mut buf);
    let hd_pos = buf.len() as i64;
    write_hd4(&mut buf, 0);
    let dg_pos = buf.len() as i64;
    write_dg4(&mut buf, 0, 0);
    let cg_pos = buf.len() as i64;
    write_cg4_with_data_bytes(&mut buf, 0, 2, 4);
    let cn_pos = buf.len() as i64;
    write_cn4(&mut buf, 0, 0, 32, 0, 0, 0, 0);
    let tx_pos = buf.len() as i64;
    write_tx(&mut buf, "GuardedVal");

    let dt_pos = buf.len() as i64;
    buf.extend_from_slice(b"##DT");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 32);
    push_u64(&mut buf, 0);
    push_u32(&mut buf, 100);
    push_u32(&mut buf, 200);

    // ##DL with 3 data links: [0, dt_pos, 0]
    let dl_pos = buf.len() as i64;
    buf.extend_from_slice(b"##DL");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 16 + 40 + 16); // header(16) + 5*8(links: dl_len_only? wait: block header 24 + 4*8 + 16)
    buf.truncate(dl_pos as usize + 8);
    push_u64(&mut buf, 24 + 4 * 8 + 8 + 4 + 3 * 8); // dl_len = 92 wait: fixed fields + offset list
    // Write standard unfixed ##DL (dl_flags=0): 16(hdr)+8(links=4)+4*8(links)+1(flags)+3(res)+4(count)+3*8(offsets) = 88B
    buf.truncate(dl_pos as usize);
    buf.extend_from_slice(b"##DL");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 88);
    push_u64(&mut buf, 4); // dl_next + 3 data slots
    push_i64(&mut buf, 0); // dl_dl_next
    push_i64(&mut buf, 0); // dl_data[0] = NIL
    push_i64(&mut buf, dt_pos); // dl_data[1]
    push_i64(&mut buf, 0); // dl_data[2] = NIL
    push_u8(&mut buf, 0); // dl_flags (variable offsets)
    push_zeros(&mut buf, 3);
    push_u32(&mut buf, 3); // dl_count
    push_u64(&mut buf, 0);
    push_u64(&mut buf, 0);
    push_u64(&mut buf, 8);

    // ##GD (40 bytes): links=1 -> dl_pos, gd_version=430
    let gd_pos = buf.len() as i64;
    buf.extend_from_slice(b"##GD");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 40);
    push_u64(&mut buf, 1);
    push_i64(&mut buf, dl_pos);
    push_u16(&mut buf, 430);
    push_zeros(&mut buf, 6);

    buf[(hd_pos + 24) as usize..(hd_pos + 32) as usize].copy_from_slice(&dg_pos.to_le_bytes());
    buf[(dg_pos + 32) as usize..(dg_pos + 40) as usize].copy_from_slice(&cg_pos.to_le_bytes());
    buf[(dg_pos + 40) as usize..(dg_pos + 48) as usize].copy_from_slice(&gd_pos.to_le_bytes());
    buf[(cg_pos + 32) as usize..(cg_pos + 40) as usize].copy_from_slice(&cn_pos.to_le_bytes());
    buf[(cn_pos + 40) as usize..(cn_pos + 48) as usize].copy_from_slice(&tx_pos.to_le_bytes());

    let (_tmp, mut mdf) = open_mdf_from_bytes(&buf)?;
    mdf.load_all_channels_data_in_memory()?;
    let ChannelData::UInt32(arr) = mdf.get_channel_data("GuardedVal").unwrap() else {
        panic!("expected UInt32 for GuardedVal");
    };
    assert_eq!(arr.values_slice(), &[100, 200]);
    Ok(())
}

#[test]
fn test_unfinalized_cycle_count_rebuilds_invalid_mask() -> Result<()> {
    // MDF4-007: unfinalized CG (cg_cycle_count=0 initially) gets finalized in memory and rebuilds cn.invalid_mask.
    use mdfr::mdfinfo::MdfInfo;

    let mut buf = Vec::new();
    write_id_block(&mut buf);
    // Mark unfinalized cycle count flag (bit 0) at offset 60 in IDBLOCK
    buf[60..62].copy_from_slice(&1u16.to_le_bytes());

    let hd_pos = buf.len() as i64;
    write_hd4(&mut buf, 0);
    let dg_pos = buf.len() as i64;
    write_dg4(&mut buf, 0, 0);
    let cg_pos = buf.len() as i64;
    buf.extend_from_slice(b"##CG");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 104);
    push_u64(&mut buf, 6);
    push_i64(&mut buf, 0);
    push_i64(&mut buf, 0); // cn_first
    push_i64(&mut buf, 0);
    push_i64(&mut buf, 0);
    push_i64(&mut buf, 0);
    push_i64(&mut buf, 0);
    push_u64(&mut buf, 0);
    push_u64(&mut buf, 0); // cg_cycle_count = 0 (unfinalized!)
    push_u16(&mut buf, 0);
    push_u16(&mut buf, 0);
    push_zeros(&mut buf, 4);
    push_u32(&mut buf, 4); // cg_data_bytes
    push_u32(&mut buf, 1); // cg_inval_bytes

    let cn_pos = buf.len() as i64;
    write_cn4(&mut buf, 0, 0, 32, 0, 0, 0, 0);
    // Set CN_F_INVALID (bit 1 = 2) at offset +100 (cn_flags) in CNBLOCK
    buf[(cn_pos + 100) as usize..(cn_pos + 104) as usize].copy_from_slice(&2u32.to_le_bytes());
    let tx_pos = buf.len() as i64;
    write_tx(&mut buf, "InvalSig");

    // 2 cycles x (4 data bytes + 1 inval byte) = 10 bytes: cycle 0 valid (inval=0), cycle 1 invalid (inval=1)
    let dt_pos = buf.len() as i64;
    buf.extend_from_slice(b"##DT");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 34);
    push_u64(&mut buf, 0);
    push_u32(&mut buf, 10);
    push_u8(&mut buf, 0);
    push_u32(&mut buf, 20);
    push_u8(&mut buf, 1);

    buf[(hd_pos + 24) as usize..(hd_pos + 32) as usize].copy_from_slice(&dg_pos.to_le_bytes());
    buf[(dg_pos + 32) as usize..(dg_pos + 40) as usize].copy_from_slice(&cg_pos.to_le_bytes());
    buf[(dg_pos + 40) as usize..(dg_pos + 48) as usize].copy_from_slice(&dt_pos.to_le_bytes());
    buf[(cg_pos + 32) as usize..(cg_pos + 40) as usize].copy_from_slice(&cn_pos.to_le_bytes());
    buf[(cn_pos + 40) as usize..(cn_pos + 48) as usize].copy_from_slice(&tx_pos.to_le_bytes());

    let (_tmp, mut mdf) = open_mdf_from_bytes(&buf)?;
    if let MdfInfo::V4(info) = &mdf.mdf_info {
        let cn = &info.dg[&dg_pos].cg[&0].cn[&0];
        assert!(
            matches!(cn.invalid_mask, Some((Some(_), 0, 1))),
            "expected rebuilt BooleanBufferBuilder before loading data, got {:?}",
            cn.invalid_mask.as_ref().map(|(_, b, m)| (*b, *m))
        );
    }
    mdf.load_all_channels_data_in_memory()?;
    {
        use arrow::array::Array;
        let ChannelData::UInt32(arr) = mdf.get_channel_data("InvalSig").unwrap() else {
            panic!("expected UInt32 InvalSig");
        };
        assert_eq!(arr.len(), 2);
        assert!(!arr.is_null(0));
        assert!(arr.is_null(1));
    }
    Ok(())
}

#[test]
fn test_rename_channel_preserves_master_and_positions() -> Result<()> {
    // MDF4-008: rename_channel updates channel and master names without clobbering DG/CG/CN positions.
    let mut buf = Vec::new();
    write_id_block(&mut buf);
    let hd_pos = buf.len() as i64;
    write_hd4(&mut buf, 0);
    let dg_pos = buf.len() as i64;
    write_dg4(&mut buf, 0, 0);
    let cg_pos = buf.len() as i64;
    write_cg4_with_data_bytes(&mut buf, 0, 2, 16);

    let cn_master_pos = buf.len() as i64;
    write_cn4_with_offset(&mut buf, 2, 4, 0, 0, 64, 0, 0, 0, 0);
    let tx_master_pos = buf.len() as i64;
    write_tx(&mut buf, "TimeOld");

    let cn_sig_pos = buf.len() as i64;
    write_cn4_with_offset(&mut buf, 0, 4, 0, 8, 64, 0, 0, 0, 0);
    let tx_sig_pos = buf.len() as i64;
    write_tx(&mut buf, "SigOld");

    let dt_pos = buf.len() as i64;
    buf.extend_from_slice(b"##DT");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 24 + 32);
    push_u64(&mut buf, 0);
    push_f64(&mut buf, 0.0);
    push_f64(&mut buf, 1.5);
    push_f64(&mut buf, 1.0);
    push_f64(&mut buf, 2.5);

    buf[(hd_pos + 24) as usize..(hd_pos + 32) as usize].copy_from_slice(&dg_pos.to_le_bytes());
    buf[(dg_pos + 32) as usize..(dg_pos + 40) as usize].copy_from_slice(&cg_pos.to_le_bytes());
    buf[(dg_pos + 40) as usize..(dg_pos + 48) as usize].copy_from_slice(&dt_pos.to_le_bytes());
    buf[(cg_pos + 32) as usize..(cg_pos + 40) as usize]
        .copy_from_slice(&cn_master_pos.to_le_bytes());
    buf[(cn_master_pos + 24) as usize..(cn_master_pos + 32) as usize]
        .copy_from_slice(&cn_sig_pos.to_le_bytes());
    buf[(cn_master_pos + 40) as usize..(cn_master_pos + 48) as usize]
        .copy_from_slice(&tx_master_pos.to_le_bytes());
    buf[(cn_sig_pos + 40) as usize..(cn_sig_pos + 48) as usize]
        .copy_from_slice(&tx_sig_pos.to_le_bytes());

    let (_tmp, mut mdf) = open_mdf_from_bytes(&buf)?;
    mdf.rename_channel("SigOld", "SigNew");
    mdf.rename_channel("TimeOld", "TimeNew");
    mdf.load_all_channels_data_in_memory()?;

    assert_eq!(
        mdf.get_channel_master("SigNew"),
        Some("TimeNew".to_string())
    );
    assert!(mdf.get_channel_data("SigOld").is_none());
    let ChannelData::Float64(sig) = mdf.get_channel_data("SigNew").unwrap() else {
        panic!("expected Float64 SigNew");
    };
    assert_eq!(sig.values_slice(), &[1.5, 2.5]);
    Ok(())
}

#[test]
fn test_logger_vlsc_flag_and_unsorted_vlsd_offsets() -> Result<()> {
    // MDF4-009: companion VLSC cg_flags must only set CG_F_VLSC (not CG_F_VLSD),
    // and unsorted VLSD offsets increment by 4 + payload_len.
    use mdfr::mdfinfo::MdfInfo;
    use mdfr_core::blocks::cg_block::{CG_F_VLSC, CG_F_VLSD};
    use mdfr_core::blocks::cn::CnDataType;
    use mdfr_core::compress::CompressionAlgorithm;
    use mdfr_core::logger::{DataLayout, Logger, LoggerBuilder};
    use mdfr_core::time::ManualClock;

    let tmp = NamedTempFile::new()?;
    let file = std::fs::File::create(tmp.path())?;
    let mut b = LoggerBuilder::new(DataLayout::Unsorted, CompressionAlgorithm::NoCompression);
    let g = b.add_group("Mixed");
    let vlsd = b.add_vlsd_channel(g, "DynStr", CnDataType::StringUtf8)?;
    let vlsc = b.add_vlsc_channel(g, "DynBytes", CnDataType::ByteArray)?;
    let schema = b.build()?;
    let mut logger = Logger::start(schema, file, ManualClock::new(0.001, 0))?;

    logger
        .begin_record(g)?
        .set_bytes(vlsd, b"hi")? // 2B -> next offset 4+2 = 6
        .set_bytes(vlsc, b"ab")?
        .commit()?;

    logger
        .begin_record(g)?
        .set_bytes(vlsd, b"world")?
        .set_bytes(vlsc, b"cde")?
        .commit()?;
    logger.close()?;

    let mut mdf = Mdf::new(tmp.path().to_str().unwrap())?;
    mdf.load_all_channels_data_in_memory()?;
    let MdfInfo::V4(info) = &mdf.mdf_info else {
        panic!("expected MDF4");
    };
    let mut found_vlsc_cg = false;
    for dg in info.dg.values() {
        for cg in dg.cg.values() {
            if (cg.block.cg_flags & CG_F_VLSC) != 0 {
                found_vlsc_cg = true;
                assert_eq!(cg.block.cg_flags & CG_F_VLSD, 0);
            }
        }
    }
    assert!(found_vlsc_cg);
    Ok(())
}

#[test]
fn test_conversions_nil_precision_bit63_text_to_text_and_cc_precision() -> Result<()> {
    // MDF4-010 (NIL->null + exact float keys), MDF4-011 (bit 63 u64 mask), MDF4-012 (cc_precision), MDF4-017 (TextToText even/empty).
    use arrow::array::Array;
    let mut buf = Vec::new();
    write_id_block(&mut buf);
    let hd_pos = buf.len() as i64;
    write_hd4(&mut buf, 0);
    let dg_pos = buf.len() as i64;
    write_dg4(&mut buf, 0, 0);
    let cg_pos = buf.len() as i64;
    write_cg4_with_data_bytes(&mut buf, 0, 2, 24); // 8B u64 bitfield + 8B f64 lookup + 8B string_id/u32... wait: 4+8+8+4 = 24B

    // Ch0: u64 bitfield (bit 63) at offset 0
    let cn_bf_pos = buf.len() as i64;
    write_cn4_with_offset(&mut buf, 0, 0, 0, 0, 64, 0, 0, 0, 0);
    let tx_bf_name = buf.len() as i64;
    write_tx(&mut buf, "Bit63");
    let tx_bit63_lbl = buf.len() as i64;
    write_tx(&mut buf, "TopBit");

    // BitfieldTextTable CC (type 11) per Table 38: cc_ref[0] -> sub ValueToText CC (cc_type 7)
    let cc_bf_sub_pos = buf.len() as i64;
    buf.extend_from_slice(b"##CC");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 48 + 6 * 8 + 8); // 4 base + 2 refs (match, default=NIL) + 1 key
    push_u64(&mut buf, 6);
    push_i64(&mut buf, 0);
    push_i64(&mut buf, 0);
    push_i64(&mut buf, 0);
    push_i64(&mut buf, 0);
    push_i64(&mut buf, tx_bit63_lbl); // masked == (1<<63) -> "TopBit"
    push_i64(&mut buf, 0); // default -> ""
    push_u8(&mut buf, 7); // ValueToText
    push_u8(&mut buf, 0xff);
    push_u16(&mut buf, 0);
    push_u16(&mut buf, 2); // ref_count
    push_u16(&mut buf, 1); // val_count
    push_f64(&mut buf, 0.0);
    push_f64(&mut buf, 0.0);
    push_f64(&mut buf, (1u64 << 63) as f64);

    // BitfieldTextTable CC (type 11): 1 mask = (1<<63), len = 48 + 5*8 + 1*8 = 96
    let cc_bf_pos = buf.len() as i64;
    buf.extend_from_slice(b"##CC");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 96);
    push_u64(&mut buf, 5); // 4 base links + 1 ref
    push_i64(&mut buf, 0);
    push_i64(&mut buf, 0);
    push_i64(&mut buf, 0);
    push_i64(&mut buf, 0);
    push_i64(&mut buf, cc_bf_sub_pos); // cc_ref[0] -> sub ValueToText CC
    push_u8(&mut buf, 11); // BitfieldTextTable (cc_type = 11)
    push_u8(&mut buf, 3); // cc_precision
    push_u16(&mut buf, 1); // cc_flags: precision valid
    push_u16(&mut buf, 1); // ref_count
    push_u16(&mut buf, 1); // val_count
    push_f64(&mut buf, 0.0);
    push_f64(&mut buf, 0.0);
    push_u64(&mut buf, 1u64 << 63); // cc_val[0] as u64 mask

    // Ch1: u32 ValueToText with NIL default and NIL ref[0] -> produces Arrow nulls, plus cc_precision=4
    let cn_v2t_pos = buf.len() as i64;
    write_cn4_with_offset(&mut buf, 0, 0, 0, 8, 32, 0, 0, 0, 0);
    let tx_v2t_name = buf.len() as i64;
    write_tx(&mut buf, "V2TNil");
    let tx_v2t_one = buf.len() as i64;
    write_tx(&mut buf, "One");

    let cc_v2t_pos = buf.len() as i64;
    buf.extend_from_slice(b"##CC");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 120); // 48 + 7*8 + 2*8
    push_u64(&mut buf, 7); // 4 base + 3 refs (key0, key1, default=NIL)
    push_i64(&mut buf, 0);
    push_i64(&mut buf, 0);
    push_i64(&mut buf, 0);
    push_i64(&mut buf, 0);
    push_i64(&mut buf, tx_v2t_one); // val 1 -> "One"
    push_i64(&mut buf, 0); // val 2 -> NIL -> null
    push_i64(&mut buf, 0); // default -> NIL -> null
    push_u8(&mut buf, 7); // ValueToText
    push_u8(&mut buf, 4); // cc_precision
    push_u16(&mut buf, 1); // cc_flags: precision valid
    push_u16(&mut buf, 3); // ref_count
    push_u16(&mut buf, 2); // val_count
    push_f64(&mut buf, 0.0);
    push_f64(&mut buf, 0.0);
    push_f64(&mut buf, 1.0);
    push_f64(&mut buf, 2.0);

    // Ch2: f64 close-key ValueToValue (1.00000001 vs 1.00000002) at offset 12
    let cn_fval_pos = buf.len() as i64;
    write_cn4_with_offset(&mut buf, 0, 4, 0, 12, 64, 0, 0, 0, 0);
    let tx_fval_name = buf.len() as i64;
    write_tx(&mut buf, "FineLookup");

    let cc_fval_pos = buf.len() as i64;
    buf.extend_from_slice(b"##CC");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 112); // 48 + 4*8 + 4*8
    push_u64(&mut buf, 4);
    push_i64(&mut buf, 0);
    push_i64(&mut buf, 0);
    push_i64(&mut buf, 0);
    push_i64(&mut buf, 0);
    push_u8(&mut buf, 5); // ValueToValue (non-interpolated)
    push_u8(&mut buf, 0xff);
    push_u16(&mut buf, 0);
    push_u16(&mut buf, 0);
    push_u16(&mut buf, 4);
    push_f64(&mut buf, 0.0);
    push_f64(&mut buf, 0.0);
    push_f64(&mut buf, 1.00000001);
    push_f64(&mut buf, 100.0);
    push_f64(&mut buf, 1.00000002);
    push_f64(&mut buf, 200.0);

    // Ch3: fixed UTF-8 string 4B with TextToText even cc_ref (2 links, no default -> input passthrough) at offset 20
    let cn_t2t_pos = buf.len() as i64;
    write_cn4_with_offset(&mut buf, 0, 7, 0, 20, 32, 0, 0, 0, 0);
    let tx_t2t_name = buf.len() as i64;
    write_tx(&mut buf, "T2TEven");
    let tx_src = buf.len() as i64;
    write_tx(&mut buf, "AA");
    let tx_dst = buf.len() as i64;
    write_tx(&mut buf, "MappedAA");

    let cc_t2t_pos = buf.len() as i64;
    buf.extend_from_slice(b"##CC");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 96); // 48 + 6*8 + 0*8
    push_u64(&mut buf, 6);
    push_i64(&mut buf, 0);
    push_i64(&mut buf, 0);
    push_i64(&mut buf, 0);
    push_i64(&mut buf, 0);
    push_i64(&mut buf, tx_src);
    push_i64(&mut buf, tx_dst);
    push_u8(&mut buf, 10); // TextToText
    push_u8(&mut buf, 0xff);
    push_u16(&mut buf, 0);
    push_u16(&mut buf, 2);
    push_u16(&mut buf, 0);
    push_f64(&mut buf, 0.0);
    push_f64(&mut buf, 0.0);

    // 2 cycles x 24 bytes
    let dt_pos = buf.len() as i64;
    buf.extend_from_slice(b"##DT");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 24 + 48);
    push_u64(&mut buf, 0);
    // Cycle 0
    push_u64(&mut buf, 1u64 << 63);
    push_u32(&mut buf, 1);
    push_f64(&mut buf, 1.00000001);
    buf.extend_from_slice(b"AA\0\0");
    // Cycle 1
    push_u64(&mut buf, 0);
    push_u32(&mut buf, 99); // unmapped -> NIL default -> null
    push_f64(&mut buf, 1.00000002);
    buf.extend_from_slice(b"ZZ\0\0"); // unmatched -> even cc_ref -> null

    buf[(hd_pos + 24) as usize..(hd_pos + 32) as usize].copy_from_slice(&dg_pos.to_le_bytes());
    buf[(dg_pos + 32) as usize..(dg_pos + 40) as usize].copy_from_slice(&cg_pos.to_le_bytes());
    buf[(dg_pos + 40) as usize..(dg_pos + 48) as usize].copy_from_slice(&dt_pos.to_le_bytes());
    buf[(cg_pos + 32) as usize..(cg_pos + 40) as usize].copy_from_slice(&cn_bf_pos.to_le_bytes());

    for (cn, next, tx, cc) in [
        (cn_bf_pos, cn_v2t_pos, tx_bf_name, cc_bf_pos),
        (cn_v2t_pos, cn_fval_pos, tx_v2t_name, cc_v2t_pos),
        (cn_fval_pos, cn_t2t_pos, tx_fval_name, cc_fval_pos),
        (cn_t2t_pos, 0, tx_t2t_name, cc_t2t_pos),
    ] {
        buf[(cn + 24) as usize..(cn + 32) as usize].copy_from_slice(&next.to_le_bytes());
        buf[(cn + 40) as usize..(cn + 48) as usize].copy_from_slice(&tx.to_le_bytes());
        buf[(cn + 56) as usize..(cn + 64) as usize].copy_from_slice(&cc.to_le_bytes());
    }

    let (_tmp, mut mdf) = open_mdf_from_bytes(&buf)?;
    assert_eq!(mdf.get_channel_precision("V2TNil"), Some(4)); // MDF4-012 fallback from CCBLOCK

    mdf.load_all_channels_data_in_memory()?;

    let ChannelData::Utf8(bf) = mdf.get_channel_converted_data("Bit63").unwrap() else {
        panic!("expected Utf8 Bit63");
    };
    assert_eq!(bf.value(0), "TopBit");
    assert_eq!(bf.value(1), "");

    let ChannelData::Utf8(v2t) = mdf.get_channel_converted_data("V2TNil").unwrap() else {
        panic!("expected Utf8 V2TNil");
    };
    assert!(!v2t.is_null(0));
    assert_eq!(v2t.value(0), "One");
    assert!(v2t.is_null(1));

    let ChannelData::Float64(fv) = mdf.get_channel_converted_data("FineLookup").unwrap() else {
        panic!("expected Float64 FineLookup");
    };
    assert_eq!(fv.values_slice(), &[100.0, 200.0]);

    let ChannelData::Utf8(t2t) = mdf.get_channel_converted_data("T2TEven").unwrap() else {
        panic!("expected Utf8 T2TEven");
    };
    assert!(!t2t.is_null(0));
    assert_eq!(t2t.value(0), "MappedAA");
    assert!(!t2t.is_null(1));
    assert_eq!(t2t.value(1), "ZZ");
    Ok(())
}

#[test]
fn test_writer_deterministic_ld_zip_info_dz_param_and_ev_remapping() -> Result<()> {
    // MDF4-003, MDF4-004, MDF4-014
    use mdfr::mdfinfo::MdfInfo;
    use mdfr::mdfinfo::mdfinfo4::Ev4Block;
    use mdfr_core::blocks::cn::CnDataType;
    use mdfr_core::compress::CompressionAlgorithm;
    use mdfr_core::logger::{DataLayout, Logger, LoggerBuilder};
    use mdfr_core::time::ManualClock;

    let src_tmp = NamedTempFile::new()?;
    {
        let file = std::fs::File::create(src_tmp.path())?;
        let mut b = LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::NoCompression);
        let g = b.add_group("G");
        let ch = b.add_channel(g, "ByteSig", CnDataType::UIntLE, 8)?;
        let schema = b.build()?;
        let mut logger = Logger::start(schema, file, ManualClock::new(0.001, 0))?;
        for v in 0u64..512 {
            logger.begin_record(g)?.set_u64(ch, v % 4)?.commit()?;
        }
        logger.close()?;
    }

    let mut mdf = Mdf::new(src_tmp.path().to_str().unwrap())?;
    mdf.load_all_channels_data_in_memory()?;
    if let MdfInfo::V4(info) = &mut mdf.mdf_info {
        info.ev.insert(
            10,
            Ev4Block {
                ev_ev_parent: 0,
                ev_ev_range: 0,
                ev_sync_base_value: 111,
                ..Default::default()
            },
        );
        info.ev.insert(
            20,
            Ev4Block {
                ev_ev_parent: 10,
                ev_ev_range: 10,
                ev_sync_base_value: 222,
                ..Default::default()
            },
        );
    }

    let out1 = NamedTempFile::new()?;
    let out2 = NamedTempFile::new()?;
    mdf.write(
        out1.path().to_str().unwrap(),
        CompressionAlgorithm::DeflateTranspose,
    )?;
    mdf.write(
        out2.path().to_str().unwrap(),
        CompressionAlgorithm::DeflateTranspose,
    )?;

    let bytes1 = std::fs::read(out1.path())?;
    let bytes2 = std::fs::read(out2.path())?;
    assert_eq!(bytes1.len(), bytes2.len());
    // Apart from wall-clock timestamps inserted at write time (HD start_time_ns + new FH fh_time_ns),
    // the entire serialized file layout is deterministic.
    let diff_positions: Vec<usize> = bytes1
        .iter()
        .zip(bytes2.iter())
        .enumerate()
        .filter_map(|(i, (a, b))| (a != b).then_some(i))
        .collect();
    let second_fh = bytes1.windows(4).rposition(|w| w == b"##FH").unwrap();
    let non_ts_diffs: Vec<usize> = diff_positions
        .into_iter()
        .filter(|&p| !(136..144).contains(&p) && !(second_fh + 40..second_fh + 48).contains(&p))
        .collect();
    assert!(
        non_ts_diffs.is_empty(),
        "unexpected non-timestamp diffs at byte offsets: {non_ts_diffs:?}"
    );

    // Check 1-byte ByteSig DZ block downgraded DeflateTranspose (1) -> Deflate (0) with zip_param = 0,
    // and check 8-byte master time DZ block kept DeflateTranspose (1) with zip_param = 8.
    let dz_params: Vec<(u8, u32)> = bytes1
        .windows(32)
        .filter(|w| &w[0..4] == b"##DZ")
        .map(|w| (w[26], u32::from_le_bytes(w[28..32].try_into().unwrap())))
        .collect();
    assert!(
        dz_params.contains(&(0, 0)),
        "expected downgraded 1-byte DZ (0, 0), got {dz_params:?}"
    );
    assert!(
        dz_params.contains(&(1, 8)),
        "expected 8-byte transposed DZ (1, 8), got {dz_params:?}"
    );

    // Check LD blocks set ld_flags bit 7 (128) for valid ld_zip_info
    let ld_flags_list: Vec<u8> = bytes1
        .windows(48)
        .filter(|w| &w[0..4] == b"##LD")
        .map(|w| w[40])
        .collect();
    assert!(!ld_flags_list.is_empty());
    assert!(ld_flags_list.iter().all(|&f| (f & (1 << 7)) != 0));

    // Check EV parent/range remapped on reload
    let mut reloaded = Mdf::new(out1.path().to_str().unwrap())?;
    reloaded.load_all_channels_data_in_memory()?;
    let MdfInfo::V4(rinfo) = &reloaded.mdf_info else {
        panic!("expected MDF4");
    };
    let parent_pos = rinfo
        .ev
        .iter()
        .find(|(_, ev)| ev.ev_sync_base_value == 111)
        .map(|(p, _)| *p)
        .expect("ParentEv present");
    let child_ev = rinfo
        .ev
        .values()
        .find(|ev| ev.ev_sync_base_value == 222)
        .expect("ChildEv present");
    assert_eq!(child_ev.ev_ev_parent, parent_pos);
    assert_eq!(child_ev.ev_ev_range, parent_pos);
    Ok(())
}

#[test]
fn test_datastream_decoder_small_alignment_and_subbyte_offset() -> Result<()> {
    // MDF4-016: calculate_aligned_offset handles current_byte_pos < alignment_offset & intra-byte cn_bit_offset.
    use mdfr::mdfinfo::mdfinfo4::{Cn4, Cn4Block, CnType, Compo, Composition, Ds4Block};
    use mdfr::mdfreader::datastream_decoder::{calculate_aligned_offset, decode_datastream_blob};

    assert_eq!(calculate_aligned_offset(16, 4, 8), 4); // current_byte_pos=2 < alignment_offset=4 -> aligns up to byte 4

    // ds_cn_alignment_start points to block_position 200 (UInt8 [2]); ds_cn_composition points to 100.
    let ds_block = Ds4Block {
        ds_links: 2,
        links: vec![100, 200],
        ..Default::default()
    };
    let member_cn = Cn4 {
        block_position: 100,
        block: Cn4Block {
            cn_data_type: 0,
            cn_byte_offset: 2,
            cn_bit_offset: 2,
            cn_bit_count: 4,
            cn_alignment: 0, // 1-byte align
            ..Default::default()
        },
        ..Default::default()
    };
    let align_cn = Cn4 {
        block_position: 200,
        block: Cn4Block {
            cn_data_type: 0,
            cn_bit_count: 8,
            ..Default::default()
        },
        data: ChannelData::UInt8(arrow::array::UInt8Array::from(vec![2])),
        ..Default::default()
    };
    let mut cn_map = CnType::default();
    cn_map.insert(0, member_cn.clone());
    cn_map.insert(1, align_cn);
    let comp = Composition {
        block: Compo::CN(Box::new(member_cn)),
        compo: None,
    };
    let blob = [0xAAu8, 0xBB, 0b0011_1100];
    let decoded = decode_datastream_blob(&blob, &ds_block, &comp, &cn_map, &[0], &[3])?;
    assert_eq!(decoded[&0][0], vec![15]);
    Ok(())
}

#[test]
fn test_ca_axis_conversion_lookup_and_classification() -> Result<()> {
    // Gap 6: ca_cc_axis_conversion overrules axis cn_cc_conversion, 2D LookUp interpolation,
    // and IntervalAxis/ClassificationResult left-open vs left-closed interval bounds.
    let mut buf = Vec::new();
    write_id_block(&mut buf);
    let hd_pos = buf.len() as i64;
    write_hd4(&mut buf, 0);
    let dg_pos = buf.len() as i64;
    write_dg4(&mut buf, 0, 0);
    // 1 cycle:
    // offset 0..4: AxisX u16[2] ([1, 3])
    // offset 4..36: Map2D f64[4] ([10.0, 20.0, 30.0, 50.0]) (2x2 row-major)
    // offset 36..76: IntAxis f64[5] ([0.0, 10.0, 20.0, 30.0, 40.0])
    // offset 76..92: ClassCounts u32[4] ([100, 200, 300, 400])
    let cg_pos = buf.len() as i64;
    write_cg4_with_data_bytes(&mut buf, 0, 1, 92);

    // AxisX CN (type 0, u16, 16-bit) with own cn_cc_conversion (*10), overruled by Map2D's ca_cc_axis_conversion[0] (*2 -> [2, 6])
    let cn_ax_x = buf.len() as i64;
    write_cn4_with_offset(&mut buf, 0, 0, 0, 0, 16, 0, 0, 0, 0);
    let tx_ax_x = buf.len() as i64;
    write_tx(&mut buf, "AxisX");
    let cc_times_10 = buf.len() as i64;
    write_cc_linear(&mut buf, 10.0, 0.0);
    let cc_times_2 = buf.len() as i64;
    write_cc_linear(&mut buf, 2.0, 0.0);

    // AxisX CABLOCK: ca_type=1 (ScalingAxis), ca_ndim=1, dim=[2], simple array [2]
    let ca_ax_x = buf.len() as i64;
    buf.extend_from_slice(b"##CA");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 56);
    push_u64(&mut buf, 1); // ca_links = 1 (ca_composition=0)
    push_i64(&mut buf, 0); // ca_composition
    push_u8(&mut buf, 1); // ca_type = ScalingAxis
    push_u8(&mut buf, 0); // ca_storage = CN template
    push_u16(&mut buf, 1); // ca_ndim = 1
    push_u32(&mut buf, 0); // ca_flags = 0
    push_u32(&mut buf, 2); // ca_byte_offset_base
    push_u32(&mut buf, 0);
    push_u64(&mut buf, 2); // ca_dim_size[0] = 2

    // Map2D LookUp CN (ca_type = 2, 2x2 f64) at byte_offset = 4
    let cn_map = buf.len() as i64;
    write_cn4_with_offset(&mut buf, 0, 4, 0, 4, 64, 0, 0, 0, 0);
    let tx_map = buf.len() as i64;
    write_tx(&mut buf, "Map2D");
    // Map2D CABLOCK: has axis flag (bit 4 = 16): ca_links = 1 + 2 (ca_cc_axis_conversion) + 6 (ca_axis) = 9 links
    // len = 24 + 9*8 + 16 + 2*8 = 128 bytes
    let ca_map = buf.len() as i64;
    buf.extend_from_slice(b"##CA");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 128);
    push_u64(&mut buf, 9);
    push_i64(&mut buf, 0); // ca_composition
    push_i64(&mut buf, cc_times_2); // ca_cc_axis_conversion[0] (overrules AxisX's *10)
    push_i64(&mut buf, 0); // ca_cc_axis_conversion[1] (NIL -> 1:1)
    push_i64(&mut buf, dg_pos);
    push_i64(&mut buf, cg_pos);
    push_i64(&mut buf, cn_ax_x);
    push_i64(&mut buf, 0); // axis 1 NIL -> [0.0, 1.0]
    push_i64(&mut buf, 0);
    push_i64(&mut buf, 0);
    push_u8(&mut buf, 2); // ca_type = LookUp
    push_u8(&mut buf, 0); // ca_storage = CN template
    push_u16(&mut buf, 2); // ca_ndim = 2
    push_u32(&mut buf, 1 << 4); // ca_flags: axis flag (row-major)
    push_u32(&mut buf, 8); // ca_byte_offset_base
    push_u32(&mut buf, 0);
    push_u64(&mut buf, 2); // dim 0 = 2
    push_u64(&mut buf, 2); // dim 1 = 2

    // IntAxis CN (ca_type = 3 IntervalAxis, f64[5]) at byte_offset = 36 with left-open bit 7
    let cn_int = buf.len() as i64;
    write_cn4_with_offset(&mut buf, 0, 4, 0, 36, 64, 0, 0, 0, 0);
    let tx_int = buf.len() as i64;
    write_tx(&mut buf, "IntAxis");
    let ca_int = buf.len() as i64;
    buf.extend_from_slice(b"##CA");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 56);
    push_u64(&mut buf, 1);
    push_i64(&mut buf, 0);
    push_u8(&mut buf, 3); // IntervalAxis
    push_u8(&mut buf, 0);
    push_u16(&mut buf, 1);
    push_u32(&mut buf, 1 << 7); // CA_F_LEFT_OPEN
    push_u32(&mut buf, 8);
    push_u32(&mut buf, 0);
    push_u64(&mut buf, 5);

    // ClassificationResult CN (ca_type = 4, 1x4 u32) at byte_offset = 76 referencing IntAxis
    let cn_cls = buf.len() as i64;
    write_cn4_with_offset(&mut buf, 0, 0, 0, 76, 32, 0, 0, 0, 0);
    let tx_cls = buf.len() as i64;
    write_tx(&mut buf, "ClsHist");
    // axis flag (bit 4 = 16): links = 1 + 1 (cc_axis_conv) + 3 (ca_axis) = 5 -> len = 24 + 5*8 + 16 + 8 = 88
    let ca_cls = buf.len() as i64;
    buf.extend_from_slice(b"##CA");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 88);
    push_u64(&mut buf, 5);
    push_i64(&mut buf, 0);
    push_i64(&mut buf, 0); // ca_cc_axis_conversion[0] = 0 (1:1)
    push_i64(&mut buf, dg_pos);
    push_i64(&mut buf, cg_pos);
    push_i64(&mut buf, cn_int);
    push_u8(&mut buf, 4); // ca_type = ClassificationResult
    push_u8(&mut buf, 0);
    push_u16(&mut buf, 1); // ca_ndim = 1
    push_u32(&mut buf, 1 << 4); // axis flag
    push_u32(&mut buf, 4);
    push_u32(&mut buf, 0);
    push_u64(&mut buf, 4); // 4 bins

    // Data record (92 bytes)
    let dt_pos = buf.len() as i64;
    buf.extend_from_slice(b"##DT");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 24 + 92);
    push_u64(&mut buf, 0);
    push_u16(&mut buf, 1); // AxisX[0] * 2 = 2.0
    push_u16(&mut buf, 3); // AxisX[1] * 2 = 6.0
    push_f64(&mut buf, 10.0); // Map2D[0,0]
    push_f64(&mut buf, 20.0); // Map2D[0,1]
    push_f64(&mut buf, 30.0); // Map2D[1,0]
    push_f64(&mut buf, 50.0); // Map2D[1,1]
    for &b in &[0.0f64, 10.0, 20.0, 30.0, 40.0] {
        push_f64(&mut buf, b);
    }
    for c in [100u32, 200, 300, 400] {
        push_u32(&mut buf, c);
    }

    buf[(hd_pos + 24) as usize..(hd_pos + 32) as usize].copy_from_slice(&dg_pos.to_le_bytes());
    buf[(dg_pos + 32) as usize..(dg_pos + 40) as usize].copy_from_slice(&cg_pos.to_le_bytes());
    buf[(dg_pos + 40) as usize..(dg_pos + 48) as usize].copy_from_slice(&dt_pos.to_le_bytes());
    buf[(cg_pos + 32) as usize..(cg_pos + 40) as usize].copy_from_slice(&cn_ax_x.to_le_bytes());

    buf[(cn_ax_x + 24) as usize..(cn_ax_x + 32) as usize].copy_from_slice(&cn_map.to_le_bytes());
    buf[(cn_ax_x + 32) as usize..(cn_ax_x + 40) as usize].copy_from_slice(&ca_ax_x.to_le_bytes());
    buf[(cn_ax_x + 40) as usize..(cn_ax_x + 48) as usize].copy_from_slice(&tx_ax_x.to_le_bytes());
    buf[(cn_ax_x + 56) as usize..(cn_ax_x + 64) as usize]
        .copy_from_slice(&cc_times_10.to_le_bytes());

    buf[(cn_map + 24) as usize..(cn_map + 32) as usize].copy_from_slice(&cn_int.to_le_bytes());
    buf[(cn_map + 32) as usize..(cn_map + 40) as usize].copy_from_slice(&ca_map.to_le_bytes());
    buf[(cn_map + 40) as usize..(cn_map + 48) as usize].copy_from_slice(&tx_map.to_le_bytes());

    buf[(cn_int + 24) as usize..(cn_int + 32) as usize].copy_from_slice(&cn_cls.to_le_bytes());
    buf[(cn_int + 32) as usize..(cn_int + 40) as usize].copy_from_slice(&ca_int.to_le_bytes());
    buf[(cn_int + 40) as usize..(cn_int + 48) as usize].copy_from_slice(&tx_int.to_le_bytes());

    buf[(cn_cls + 32) as usize..(cn_cls + 40) as usize].copy_from_slice(&ca_cls.to_le_bytes());
    buf[(cn_cls + 40) as usize..(cn_cls + 48) as usize].copy_from_slice(&tx_cls.to_le_bytes());

    let (_tmp, mut mdf) = open_mdf_from_bytes(&buf)?;
    mdf.load_all_channels_data_in_memory()?;

    let axes = mdf.get_channel_ca_axes("Map2D")?.expect("axes present");
    // AxisX raw [1, 3] converted with ca_cc_axis_conversion (*2 -> [2, 6]), NOT *10 ([10, 30])
    assert_eq!(axes, vec![vec![2.0, 6.0], vec![0.0, 1.0]]);
    // Midpoint (x=4.0, y=0.5): avg of [10, 20, 30, 50] = 27.5
    let interp = mdf.eval_ca_lookup("Map2D", 0, &[4.0, 0.5])?.unwrap();
    assert!((interp - 27.5).abs() < 1e-12);
    // Asymmetric point (x=4.0, y=0.25): 0.5*(0.75*10+0.25*20) + 0.5*(0.75*30+0.25*50) = 0.5*12.5 + 0.5*35.0 = 23.75
    let interp2 = mdf.eval_ca_lookup("Map2D", 0, &[4.0, 0.25])?.unwrap();
    assert!((interp2 - 23.75).abs() < 1e-12);

    // Column-major lookup sanity check directly via eval_ca_lookup
    let col_major_val = mdfr::mdfreader::conversions4::eval_ca_lookup(
        &axes,
        &[10.0, 30.0, 20.0, 50.0],
        &[4.0, 0.25],
        false,
    )?;
    assert!((col_major_val - 23.75).abs() < 1e-12);

    // Left-open (CA_F_LEFT_OPEN) classification on file: (0,10], (10,20], (20,30], (30,40]
    assert_eq!(mdf.eval_ca_classification("ClsHist", 0, 0.0)?, None);
    assert_eq!(mdf.eval_ca_classification("ClsHist", 0, 10.0)?, Some(100.0));
    assert_eq!(mdf.eval_ca_classification("ClsHist", 0, 20.0)?, Some(200.0));
    // Left-closed direct check: [0, 10), [10, 20)
    assert_eq!(
        mdfr::mdfreader::conversions4::eval_ca_classification(
            &[0.0, 10.0, 20.0],
            &[100.0, 200.0],
            10.0,
            false
        ),
        Some(200.0)
    );
    Ok(())
}

#[test]
fn test_synchronization_channel_cn_type_4() -> Result<()> {
    // MDF4 spec §6.6: synchronization channel (cn_type = 4) stores sync values in the record
    // and links cn_data to an ATBLOCK in the global hd_at_first attachment list.
    let mut buf = Vec::new();
    write_id_block(&mut buf);
    let hd_pos = buf.len() as i64;
    write_hd4(&mut buf, 0);

    // External AVI ATBLOCK (96 bytes)
    let tx_at_file = buf.len() as i64;
    write_tx(&mut buf, "video.avi");
    let tx_at_mime = buf.len() as i64;
    write_tx(&mut buf, "video/x-msvideo");
    let at_pos = buf.len() as i64;
    buf.extend_from_slice(b"##AT");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 96);
    push_u64(&mut buf, 4);
    push_i64(&mut buf, 0); // at_at_next
    push_i64(&mut buf, tx_at_file); // at_tx_filename
    push_i64(&mut buf, tx_at_mime); // at_tx_mimetype
    push_i64(&mut buf, 0); // at_md_comment
    push_u16(&mut buf, 0); // at_flags = 0 (external)
    push_u16(&mut buf, 0); // at_creator_index
    push_u8(&mut buf, 0); // at_zip_type
    push_u8(&mut buf, 0); // at_path_syntax
    push_zeros(&mut buf, 2); // at_reserved
    push_zeros(&mut buf, 16); // at_md5_checksum
    push_u64(&mut buf, 0); // at_original_size
    push_u64(&mut buf, 0); // at_embedded_size

    // DG0: sorted DT block with 2 channels: time (cn_type=2, <f8) and video_signal (cn_type=4, <f8)
    let dg0_pos = buf.len() as i64;
    write_dg4(&mut buf, 0, 0);
    let cg0_pos = buf.len() as i64;
    write_cg4_with_data_bytes(&mut buf, 0, 5, 16);

    let tx_time = buf.len() as i64;
    write_tx(&mut buf, "time");
    let cn_time_pos = buf.len() as i64;
    write_cn4_with_offset(&mut buf, 2, 4, 0, 0, 64, tx_time, 0, 0, 0);
    // Set cn_sync_type = 1 (Time) at offset 73 of CN block (24 + 64 + 1? No: 24 + 8*8 = 88 -> cn_type at +88, cn_sync_type at +89)
    buf[(cn_time_pos + 89) as usize] = 1;

    let tx_video = buf.len() as i64;
    write_tx(&mut buf, "video_signal");
    let cn_video_pos = buf.len() as i64;
    write_cn4_with_offset(&mut buf, 4, 4, 0, 8, 64, tx_video, 0, at_pos, 0);
    buf[(cn_video_pos + 89) as usize] = 1; // cn_sync_type = 1 (Time)

    let dt0_pos = buf.len() as i64;
    let records = [
        (9167037.0_f64, 0.0_f64),
        (9167038.0_f64, 0.1307268_f64),
        (14466255.0_f64, 0.1307268_f64),
        (139054816.0_f64, 0.19609020000000002_f64),
        (206070512.0_f64, 0.2614536_f64),
    ];
    buf.extend_from_slice(b"##DT");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 24 + (records.len() * 16) as u64);
    push_u64(&mut buf, 0);
    for &(t, v) in &records {
        push_f64(&mut buf, t);
        push_f64(&mut buf, v);
    }

    // DG1: single-channel LD+DV with cn_type = 4 + linear conversion (exercises read_one_channel_array)
    let dg1_pos = buf.len() as i64;
    write_dg4(&mut buf, 0, 0);
    let cg1_pos = buf.len() as i64;
    write_cg4_with_data_bytes(&mut buf, 0, 3, 4);
    let tx_frame = buf.len() as i64;
    write_tx(&mut buf, "frame_sync");
    let cc_frame = buf.len() as i64;
    write_cc_linear(&mut buf, 0.5, 1.0); // phys = 1.0 + 0.5 * raw
    let cn_frame_pos = buf.len() as i64;
    write_cn4_with_offset(&mut buf, 4, 0, 0, 0, 32, tx_frame, cc_frame, at_pos, 0);
    buf[(cn_frame_pos + 89) as usize] = 4; // cn_sync_type = 4 (Index)

    let dv1_pos = buf.len() as i64;
    buf.extend_from_slice(b"##DV");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 24 + 12);
    push_u64(&mut buf, 0);
    push_u32(&mut buf, 2);
    push_u32(&mut buf, 4);
    push_u32(&mut buf, 6);

    let ld1_pos = buf.len() as i64;
    buf.extend_from_slice(b"##LD");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 56);
    push_u64(&mut buf, 2); // ld_links: next + dv1
    push_i64(&mut buf, 0);
    push_i64(&mut buf, dv1_pos);
    push_u32(&mut buf, 0); // ld_flags..ld_flags_ext
    push_u32(&mut buf, 1); // ld_count
    push_u64(&mut buf, 0); // ld_sample_offset[0]

    // Patch links
    buf[(hd_pos + 24) as usize..(hd_pos + 32) as usize].copy_from_slice(&dg0_pos.to_le_bytes());
    buf[(hd_pos + 48) as usize..(hd_pos + 56) as usize].copy_from_slice(&at_pos.to_le_bytes());

    buf[(dg0_pos + 24) as usize..(dg0_pos + 32) as usize].copy_from_slice(&dg1_pos.to_le_bytes());
    buf[(dg0_pos + 32) as usize..(dg0_pos + 40) as usize].copy_from_slice(&cg0_pos.to_le_bytes());
    buf[(dg0_pos + 40) as usize..(dg0_pos + 48) as usize].copy_from_slice(&dt0_pos.to_le_bytes());
    buf[(cg0_pos + 32) as usize..(cg0_pos + 40) as usize]
        .copy_from_slice(&cn_time_pos.to_le_bytes());
    buf[(cn_time_pos + 24) as usize..(cn_time_pos + 32) as usize]
        .copy_from_slice(&cn_video_pos.to_le_bytes());

    buf[(dg1_pos + 32) as usize..(dg1_pos + 40) as usize].copy_from_slice(&cg1_pos.to_le_bytes());
    buf[(dg1_pos + 40) as usize..(dg1_pos + 48) as usize].copy_from_slice(&ld1_pos.to_le_bytes());
    buf[(cg1_pos + 32) as usize..(cg1_pos + 40) as usize]
        .copy_from_slice(&cn_frame_pos.to_le_bytes());

    let (_tmp, mut mdf) = open_mdf_from_bytes(&buf)?;
    mdf.load_all_channels_data_in_memory()?;

    assert!(mdf.is_sync_channel("video_signal"));
    assert!(mdf.is_sync_channel("frame_sync"));
    assert!(!mdf.is_sync_channel("time"));

    let video = mdf.get_channel_data("video_signal").expect("video_signal");
    let ChannelData::Float64(video_arr) = video else {
        panic!("expected Float64 for video_signal, got {:?}", video);
    };
    let expected_video: Vec<f64> = records.iter().map(|&(_, v)| v).collect();
    assert_eq!(video_arr.values(), expected_video.as_slice());

    let time = mdf.get_channel_data("time").expect("time");
    let ChannelData::Float64(time_arr) = time else {
        panic!("expected Float64 for time, got {:?}", time);
    };
    let expected_time: Vec<f64> = records.iter().map(|&(t, _)| t).collect();
    assert_eq!(time_arr.values(), expected_time.as_slice());

    let frame = mdf.get_channel_data("frame_sync").expect("frame_sync");
    let ChannelData::Float64(frame_arr) = frame else {
        panic!(
            "expected Float64 after conversion for frame_sync, got {:?}",
            frame
        );
    };
    assert_eq!(frame_arr.values(), &[2.0, 3.0, 4.0]);

    let at = mdf
        .get_attachment_block(at_pos)
        .expect("ATBLOCK should be registered at at_pos");
    assert_eq!(at.at_tx_filename, tx_at_file);
    Ok(())
}

#[test]
fn test_invalidation_multi_byte_dl_and_cn_flags() -> Result<()> {
    // Covers multi-byte invalidation offsets across split ##DL chunks, cn_flags=0 (invalid bit unused),
    // and cn_flags=3 (CN_F_ALL_VALUES_INVALID | CN_F_INVALID).
    let mut buf = Vec::new();
    write_id_block(&mut buf);
    let hd_pos = buf.len() as i64;
    write_hd4(&mut buf, 0);
    let dg_pos = buf.len() as i64;
    write_dg4(&mut buf, 0, 0);

    const DATA_B: usize = 32; // 4 x f64
    const INV_B: usize = 20; // bits 0..159
    const REC_LEN: usize = DATA_B + INV_B;
    const CYCLES: usize = 5;

    let cg_pos = buf.len() as i64;
    write_cg4_with_data_bytes(&mut buf, 0, CYCLES as u64, DATA_B as u32);
    // Patch cg_inval_bytes at cg_pos + 100
    buf[(cg_pos + 100) as usize..(cg_pos + 104) as usize]
        .copy_from_slice(&(INV_B as u32).to_le_bytes());

    let mut make_cn = |name: &str, byte_off: u32, flags: u32, inval_bit_pos: u32| -> i64 {
        let tx_pos = buf.len() as i64;
        write_tx(&mut buf, name);
        let cn_pos = buf.len() as i64;
        write_cn4_with_offset(&mut buf, 0, 4, 0, byte_off, 64, tx_pos, 0, 0, 0);
        // cn_flags at +100, cn_inval_bit_pos at +104 in CNBLOCK
        buf[(cn_pos + 100) as usize..(cn_pos + 104) as usize].copy_from_slice(&flags.to_le_bytes());
        buf[(cn_pos + 104) as usize..(cn_pos + 108) as usize]
            .copy_from_slice(&inval_bit_pos.to_le_bytes());
        cn_pos
    };

    // ch_disabled: cn_flags=0, bit=0 (even if byte 0 bit 0 is set in file, must NOT mask)
    let cn0 = make_cn("ch_disabled", 0, 0, 0);
    // ch_all_invalid: cn_flags=3 (CN_F_ALL_VALUES_INVALID | CN_F_INVALID), bit=0 (all masked even when bit=0)
    let cn1 = make_cn("ch_all_invalid", 8, 3, 0);
    // ch_bit10: cn_flags=2, bit=10 -> inv byte 1, mask 1<<2 = 4
    let cn2 = make_cn("ch_bit10", 16, 2, 10);
    // ch_bit129: cn_flags=2, bit=129 -> inv byte 16, mask 1<<1 = 2
    let cn3 = make_cn("ch_bit129", 24, 2, 129);

    buf[(cn0 + 24) as usize..(cn0 + 32) as usize].copy_from_slice(&cn1.to_le_bytes());
    buf[(cn1 + 24) as usize..(cn1 + 32) as usize].copy_from_slice(&cn2.to_le_bytes());
    buf[(cn2 + 24) as usize..(cn2 + 32) as usize].copy_from_slice(&cn3.to_le_bytes());

    // Build 5 x 52 = 260 raw record bytes with crafted invalidation bytes
    let mut raw_records = Vec::with_capacity(CYCLES * REC_LEN);
    for r in 0..CYCLES {
        for c in 0..4 {
            push_f64(&mut raw_records, (r * 10 + c) as f64);
        }
        let mut inv = [0u8; INV_B];
        // Set bit 0 on all rows to verify cn_flags=0 ignores it;
        // set bit 10 (byte 1, bit 2) on rows 0 and 3;
        // set bit 129 (byte 16, bit 1) on rows 1, 2, and 4.
        inv[0] = 0x01;
        if r == 0 || r == 3 {
            inv[1] |= 1 << 2; // bit 10
        }
        if r == 1 || r == 2 || r == 4 {
            inv[16] |= 1 << 1; // bit 129
        }
        raw_records.extend_from_slice(&inv);
    }

    // Split across 3 unaligned ##DT blocks inside ##DL to test remainder-record reassembly
    let split1 = 111;
    let split2 = 191;
    let chunks = [
        &raw_records[..split1],
        &raw_records[split1..split2],
        &raw_records[split2..],
    ];
    let mut dt_ptrs = Vec::new();
    for chunk in &chunks {
        let pos = buf.len() as i64;
        buf.extend_from_slice(b"##DT");
        push_zeros(&mut buf, 4);
        push_u64(&mut buf, 24 + chunk.len() as u64);
        push_u64(&mut buf, 0);
        buf.extend_from_slice(chunk);
        dt_ptrs.push(pos);
    }

    let dl_pos = buf.len() as i64;
    buf.extend_from_slice(b"##DL");
    push_zeros(&mut buf, 4);
    let dl_links = 1 + dt_ptrs.len() as u64;
    let dl_len = 24 + dl_links * 8 + 8 + dt_ptrs.len() as u64 * 8;
    push_u64(&mut buf, dl_len);
    push_u64(&mut buf, dl_links);
    push_i64(&mut buf, 0); // dl_dl_next
    for &p in &dt_ptrs {
        push_i64(&mut buf, p);
    }
    push_u8(&mut buf, 0); // dl_flags
    push_zeros(&mut buf, 3);
    push_u32(&mut buf, dt_ptrs.len() as u32);
    let mut off = 0u64;
    for chunk in &chunks {
        push_u64(&mut buf, off);
        off += chunk.len() as u64;
    }

    buf[(hd_pos + 24) as usize..(hd_pos + 32) as usize].copy_from_slice(&dg_pos.to_le_bytes());
    buf[(dg_pos + 32) as usize..(dg_pos + 40) as usize].copy_from_slice(&cg_pos.to_le_bytes());
    buf[(dg_pos + 40) as usize..(dg_pos + 48) as usize].copy_from_slice(&dl_pos.to_le_bytes());
    buf[(cg_pos + 32) as usize..(cg_pos + 40) as usize].copy_from_slice(&cn0.to_le_bytes());

    let (tmp, mut mdf) = open_mdf_from_bytes(&buf)?;
    mdf.load_all_channels_data_in_memory()?;

    let d0 = mdf.get_channel_data("ch_disabled").unwrap();
    assert!(
        d0.validity().is_none(),
        "cn_flags=0 channel must have no validity mask"
    );

    let d1 = mdf.get_channel_data("ch_all_invalid").unwrap();
    let v1 = d1
        .validity()
        .expect("cn_flags=3 channel must have validity mask");
    assert_eq!(v1.null_count(), CYCLES, "all samples must be invalid");

    let d2 = mdf.get_channel_data("ch_bit10").unwrap();
    let v2 = d2.validity().expect("ch_bit10 must have validity mask");
    assert_eq!(
        (0..CYCLES).map(|i| v2.is_valid(i)).collect::<Vec<_>>(),
        vec![false, true, true, false, true]
    );

    let d3 = mdf.get_channel_data("ch_bit129").unwrap();
    let v3 = d3.validity().expect("ch_bit129 must have validity mask");
    assert_eq!(
        (0..CYCLES).map(|i| v3.is_valid(i)).collect::<Vec<_>>(),
        vec![true, false, false, true, false]
    );

    #[cfg(feature = "numpy")]
    {
        use pyo3::prelude::*;
        Python::initialize();
        Python::attach(|py| -> Result<()> {
            let np_ma = py.import("numpy.ma")?;
            let mdfr_cls = py.get_type::<mdfr::mdfr::Mdfr>();
            let py_mdf = mdfr_cls.call1((tmp.path().to_str().unwrap(),))?;
            py_mdf.call_method0("load_all_channels_data_in_memory")?;

            let arr0 = py_mdf.call_method1("get_channel_data", ("ch_disabled",))?;
            let masked0: usize = np_ma.call_method1("count_masked", (&arr0,))?.extract()?;
            assert_eq!(masked0, 0);

            let arr1 = py_mdf.call_method1("get_channel_data", ("ch_all_invalid",))?;
            let masked1: usize = np_ma.call_method1("count_masked", (&arr1,))?.extract()?;
            assert_eq!(masked1, CYCLES);

            let arr2 = py_mdf.call_method1("get_channel_data", ("ch_bit10",))?;
            let masked2: usize = np_ma.call_method1("count_masked", (&arr2,))?.extract()?;
            assert_eq!(masked2, 2);

            let arr3 = py_mdf.call_method1("get_channel_data", ("ch_bit129",))?;
            let masked3: usize = np_ma.call_method1("count_masked", (&arr3,))?.extract()?;
            assert_eq!(masked3, 3);
            Ok(())
        })?;
    }
    Ok(())
}

#[test]
fn test_invalidation_hl_transposed_dz_and_ld_lazy_load() -> Result<()> {
    // 1) ##HL -> ##DL -> transposed Deflate ##DZ with multi-byte invalidation (inv_B=7, bit=18).
    // 2) ##LD -> ##DV + ##DI lazy single-channel loading must preserve masks across loads.
    use mdfr_core::compress::{CompressionAlgorithm, compress_dz_data};

    let mut buf = Vec::new();
    write_id_block(&mut buf);
    let hd_pos = buf.len() as i64;
    write_hd4(&mut buf, 0);

    // --- DG0: ##HL -> ##DL -> ##DZ (transposed Deflate) ---
    let dg0_pos = buf.len() as i64;
    write_dg4(&mut buf, 0, 0);

    const CYCLES0: usize = 64;
    const DATA_B0: usize = 16; // 2 x f64
    const INV_B0: usize = 7;
    const REC_LEN0: usize = DATA_B0 + INV_B0; // 23 bytes

    let cg0_pos = buf.len() as i64;
    write_cg4_with_data_bytes(&mut buf, 0, CYCLES0 as u64, DATA_B0 as u32);
    buf[(cg0_pos + 100) as usize..(cg0_pos + 104) as usize]
        .copy_from_slice(&(INV_B0 as u32).to_le_bytes());

    let tx_ts = buf.len() as i64;
    write_tx(&mut buf, "time0");
    let cn_ts = buf.len() as i64;
    write_cn4_with_offset(&mut buf, 2, 4, 0, 0, 64, tx_ts, 0, 0, 0);

    let tx_sig = buf.len() as i64;
    write_tx(&mut buf, "flex_sig");
    let cn_sig = buf.len() as i64;
    write_cn4_with_offset(&mut buf, 0, 4, 0, 8, 64, tx_sig, 0, 0, 0);
    buf[(cn_sig + 100) as usize..(cn_sig + 104) as usize].copy_from_slice(&2u32.to_le_bytes()); // CN_F_INVALID
    buf[(cn_sig + 104) as usize..(cn_sig + 108) as usize].copy_from_slice(&18u32.to_le_bytes()); // bit 18 -> byte 2 bit 2

    let mut raw0 = Vec::with_capacity(CYCLES0 * REC_LEN0);
    for r in 0..CYCLES0 {
        push_f64(&mut raw0, r as f64 * 0.01);
        push_f64(&mut raw0, r as f64 * 1.5);
        let mut inv = [0u8; INV_B0];
        if r == 5 || r == 31 || r == 60 {
            inv[2] |= 1 << 2; // bit 18
        }
        raw0.extend_from_slice(&inv);
    }
    let compressed0 = compress_dz_data(&raw0, CompressionAlgorithm::DeflateTranspose, REC_LEN0)?;
    let dz0_pos = buf.len() as i64;
    buf.extend_from_slice(b"##DZ");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 48 + compressed0.len() as u64);
    push_u64(&mut buf, 0);
    buf.extend_from_slice(b"DT");
    push_u8(&mut buf, 1); // DeflateTranspose
    push_u8(&mut buf, 0);
    push_u32(&mut buf, REC_LEN0 as u32);
    push_u64(&mut buf, raw0.len() as u64);
    push_u64(&mut buf, compressed0.len() as u64);
    buf.extend_from_slice(&compressed0);

    let dl0_pos = buf.len() as i64;
    buf.extend_from_slice(b"##DL");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 56);
    push_u64(&mut buf, 2);
    push_i64(&mut buf, 0);
    push_i64(&mut buf, dz0_pos);
    push_u32(&mut buf, 0);
    push_u32(&mut buf, 1);
    push_u64(&mut buf, 0);

    let hl0_pos = buf.len() as i64;
    buf.extend_from_slice(b"##HL");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 40);
    push_u64(&mut buf, 1);
    push_i64(&mut buf, dl0_pos);
    push_u16(&mut buf, 0); // hl_flags
    push_u8(&mut buf, 1); // DeflateTranspose
    push_zeros(&mut buf, 5);

    // --- DG1: ##LD with DV + DI (testing lazy channel loading) ---
    let dg1_pos = buf.len() as i64;
    write_dg4(&mut buf, 0, 0);
    const CYCLES1: usize = 4;
    const INV_B1: usize = 2;

    let cg1_pos = buf.len() as i64;
    write_cg4_with_data_bytes(&mut buf, 0, CYCLES1 as u64, 16);
    buf[(cg1_pos + 100) as usize..(cg1_pos + 104) as usize]
        .copy_from_slice(&(INV_B1 as u32).to_le_bytes());

    let tx_a = buf.len() as i64;
    write_tx(&mut buf, "ld_a");
    let cn_a = buf.len() as i64;
    write_cn4_with_offset(&mut buf, 0, 4, 0, 0, 64, tx_a, 0, 0, 0);
    buf[(cn_a + 100) as usize..(cn_a + 104) as usize].copy_from_slice(&2u32.to_le_bytes());
    buf[(cn_a + 104) as usize..(cn_a + 108) as usize].copy_from_slice(&9u32.to_le_bytes()); // byte 1 bit 1

    let tx_b = buf.len() as i64;
    write_tx(&mut buf, "ld_b");
    let cn_b = buf.len() as i64;
    write_cn4_with_offset(&mut buf, 0, 4, 0, 8, 64, tx_b, 0, 0, 0);
    buf[(cn_b + 100) as usize..(cn_b + 104) as usize].copy_from_slice(&2u32.to_le_bytes());
    buf[(cn_b + 104) as usize..(cn_b + 108) as usize].copy_from_slice(&12u32.to_le_bytes()); // byte 1 bit 4

    let dv1_pos = buf.len() as i64;
    buf.extend_from_slice(b"##DV");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 24 + (CYCLES1 * 16) as u64);
    push_u64(&mut buf, 0);
    for r in 0..CYCLES1 {
        push_f64(&mut buf, r as f64);
        push_f64(&mut buf, r as f64 + 10.0);
    }

    let di1_pos = buf.len() as i64;
    buf.extend_from_slice(b"##DI");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 24 + (CYCLES1 * INV_B1) as u64);
    push_u64(&mut buf, 0);
    // Row 1 invalid for ld_a (bit 9 = byte 1 bit 1), Row 2 invalid for ld_b (bit 12 = byte 1 bit 4)
    buf.extend_from_slice(&[0, 0, 0, 1 << 1, 0, 1 << 4, 0, 0]);

    let ld1_pos = buf.len() as i64;
    buf.extend_from_slice(b"##LD");
    push_zeros(&mut buf, 4);
    push_u64(&mut buf, 64);
    push_u64(&mut buf, 3); // ld_next, dv1, di1
    push_i64(&mut buf, 0);
    push_i64(&mut buf, dv1_pos);
    push_i64(&mut buf, di1_pos);
    push_u32(&mut buf, 1 << 31); // ld_flags bit 31 = invalidation data present
    push_u32(&mut buf, 1); // ld_count = 1
    push_u64(&mut buf, 0);

    // Patch links
    buf[(hd_pos + 24) as usize..(hd_pos + 32) as usize].copy_from_slice(&dg0_pos.to_le_bytes());
    buf[(dg0_pos + 24) as usize..(dg0_pos + 32) as usize].copy_from_slice(&dg1_pos.to_le_bytes());
    buf[(dg0_pos + 32) as usize..(dg0_pos + 40) as usize].copy_from_slice(&cg0_pos.to_le_bytes());
    buf[(dg0_pos + 40) as usize..(dg0_pos + 48) as usize].copy_from_slice(&hl0_pos.to_le_bytes());
    buf[(cg0_pos + 32) as usize..(cg0_pos + 40) as usize].copy_from_slice(&cn_ts.to_le_bytes());
    buf[(cn_ts + 24) as usize..(cn_ts + 32) as usize].copy_from_slice(&cn_sig.to_le_bytes());

    buf[(dg1_pos + 32) as usize..(dg1_pos + 40) as usize].copy_from_slice(&cg1_pos.to_le_bytes());
    buf[(dg1_pos + 40) as usize..(dg1_pos + 48) as usize].copy_from_slice(&ld1_pos.to_le_bytes());
    buf[(cg1_pos + 32) as usize..(cg1_pos + 40) as usize].copy_from_slice(&cn_a.to_le_bytes());
    buf[(cn_a + 24) as usize..(cn_a + 32) as usize].copy_from_slice(&cn_b.to_le_bytes());

    let (_tmp, mut mdf) = open_mdf_from_bytes(&buf)?;
    // Lazy-load flex_sig and ld_a first, then ld_b separately
    mdf.load_channels_data_in_memory(["flex_sig", "ld_a"].into_iter().map(String::from).collect())?;
    mdf.load_channels_data_in_memory(["ld_b"].into_iter().map(String::from).collect())?;

    let sig = mdf.get_channel_data("flex_sig").unwrap();
    let v_sig = sig.validity().expect("flex_sig validity");
    assert_eq!(v_sig.null_count(), 3);
    assert!(!v_sig.is_valid(5));
    assert!(!v_sig.is_valid(31));
    assert!(!v_sig.is_valid(60));
    assert!(v_sig.is_valid(0) && v_sig.is_valid(6));

    let da = mdf.get_channel_data("ld_a").unwrap();
    let va = da.validity().expect("ld_a validity must survive lazy load");
    assert_eq!(
        (0..CYCLES1).map(|i| va.is_valid(i)).collect::<Vec<_>>(),
        vec![true, false, true, true]
    );

    let db = mdf.get_channel_data("ld_b").unwrap();
    let vb = db.validity().expect("ld_b validity");
    assert_eq!(
        (0..CYCLES1).map(|i| vb.is_valid(i)).collect::<Vec<_>>(),
        vec![true, true, false, true]
    );

    Ok(())
}
