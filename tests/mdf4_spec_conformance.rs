//! Tests for MDF 4.3.0 conformance features:
//! 1. Independent column-oriented channels (`cn_data` -> LD4 / DV4 / DZ4 with `dg_data == 0`).
//! 2. Virtual data channels (`cn_type == 6` with conversion).
//! 3. Array storage types: DG template (`ca_storage == 2`) and CG template (`ca_storage == 1`).

use anyhow::Result;
use mdfr::data_holder::channel_data::ChannelData;
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
    if let Some(ChannelData::Float64(builder)) = cd {
        let arr = builder.finish_cloned();
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

    if let Some(ChannelData::Float32(builder)) = mdf.get_channel_data("ColFloat32") {
        let arr = builder.finish_cloned();
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

    if let Some(ChannelData::Float64(builder)) = mdf.get_channel_data("ColFloat64BE") {
        let arr = builder.finish_cloned();
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
    let mut ca = Ca4Block::default();
    ca.ca_type = 0; // Array
    ca.ca_storage = 2; // DG template
    ca.ca_ndim = 2;
    ca.ca_dim_size = vec![2, 3]; // 2 rows, 3 cols
    ca.ca_data = Some(dv_ptrs.clone());
    ca.ca_cycle_count = Some(vec![4; 6]);
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
    let mut ca = Ca4Block::default();
    ca.ca_type = 0; // Array
    ca.ca_storage = 1; // CG template
    ca.ca_ndim = 1;
    ca.ca_dim_size = vec![2]; // 2 elements
    ca.ca_cycle_count = Some(vec![3, 3]);
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

    if let Some(ChannelData::Utf8(builder)) = mdf.get_channel_data("SensorName") {
        let array = builder.finish_cloned();
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
        data: ChannelData::Float64(builder),
        ..Default::default()
    };

    mdfr::mdfreader::conversions4::bitfield_text_table(&mut cn, &cc_val, &cc_ref, &sharable)?;

    if let ChannelData::Utf8(ref s) = cn.data {
        let res = s.finish_cloned();
        assert_eq!(res.value(0), " | VAL_NEG_1");
        assert_eq!(res.value(1), " | VAL_0");
        assert_eq!(res.value(2), " | VAL_1");
        assert_eq!(res.value(3), " | VAL_1");
        assert_eq!(res.value(4), " | VAL_2");
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
    cn.data = cn.data.zeros(0, 2, 7, (vec![1], Order::RowMajor))?;
    let mut channels = CnType::default();
    channels.insert(0, cn);
    read_channels_from_bytes(&raw_bytes, &mut channels, 7, 0, false)?;
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
    cn_arr.data = cn_arr.data.zeros(0, 2, 8, shape)?;
    let mut raw_arr = Vec::new();
    for v in [10u64, 20, 30, 40] {
        raw_arr.extend_from_slice(&v.to_be_bytes());
    }
    let mut channels_arr = CnType::default();
    channels_arr.insert(0, cn_arr);
    read_channels_from_bytes(&raw_arr, &mut channels_arr, 16, 0, false)?;
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
    assert!(
        read_channels_from_bytes(&bytes_with_remainder, &mut channels, 8, 0, false).is_err()
    );
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
    cn_le.data = cn_le.data.zeros(0, 1, 3, (vec![1], Order::RowMajor))?;

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
    cn_be.data = cn_be.data.zeros(0, 1, 3, (vec![1], Order::RowMajor))?;

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
    cn_signed_le.data = cn_signed_le
        .data
        .zeros(0, 1, 3, (vec![1], Order::RowMajor))?;

    let mut cn_map = CnType::default();
    cn_map.insert(0, cn_le);
    cn_map.insert(1, cn_be);
    cn_map.insert(2, cn_signed_le);

    read_channels_from_bytes(&bytes, &mut cn_map, 3, 0, false)?;

    let mut cg_block = Cg4Block::default();
    cg_block.cg_cycle_count = 1;
    cg_block.cg_data_bytes = 3;

    let mut dg = Dg4 {
        block: Dg4Block::default(),
        cg: Default::default(),
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
