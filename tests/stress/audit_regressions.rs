use anyhow::Result;
use arrow::array::{Array, BooleanBufferBuilder, Float64Array, LargeStringArray, UInt32Array};
use binrw::{BinReaderExt, BinWrite};
use mdfr::data_holder::channel_data::ChannelData;
use mdfr::mdfinfo::mdfinfo4::{
    Blockheader4Short, CN_F_INVALID, Cg4, Cg4Block, Cn4, Cn4Block, Dg4Block, Dl4Block, Dt4Block,
};
use mdfr::mdfinfo::{IdBlock, MdfInfo};
use mdfr::mdfreader::Mdf;
use mdfr::mdfreader::data_read4::decode_u64_slice;
use mdfr_core::blocks::{fh_block::FhBlock, hd_block::Hd4};
use mdfr_core::compress::CompressionAlgorithm;
use mdfr_core::logger::recover::recover_in_place;
use rustc_hash::FxHashMap;
use std::collections::{HashMap, HashSet};
use std::io::Cursor;
use tempfile::NamedTempFile;

/// Regression 1: MDF4 invalidation bits must not produce false nulls on channels
/// without `CN_F_INVALID` (cn_flags & 2 == 0) or on VLSD channels (cn_type == 1).
#[test]
fn test_mdf4_invalidation_flag_and_vlsd_no_false_nulls() -> Result<()> {
    let mut cg = Cg4 {
        header: Blockheader4Short::default(),
        block: Cg4Block {
            cg_cycle_count: 3,
            cg_inval_bytes: 1,
            ..Default::default()
        },
        cn: HashMap::with_hasher(Default::default()),
        master_channel_name: None,
        channel_names: HashSet::new(),
        block_position: 0,
        record_length: 13,
        vlsd_cg: None,
        invalid_bytes: Some(vec![0x01, 0x01, 0x01]),
        sr: Vec::new(),
    };

    // Channel 1: has CN_F_INVALID, bit 0 -> should get all nulls (invalid bit = 1)
    let mut mask_builder = BooleanBufferBuilder::new(3);
    mask_builder.append_n(3, true);
    let cn_invalid = Cn4 {
        unique_name: "ch_with_inval".to_string(),
        block: Cn4Block {
            cn_type: 0,
            cn_flags: CN_F_INVALID,
            cn_inval_bit_pos: 0,
            ..Default::default()
        },
        data: ChannelData::UInt32(UInt32Array::from(vec![10, 20, 30])),
        invalid_mask: Some((Some(mask_builder), 0, 0x01)),
        ..Default::default()
    };

    // Channel 2: does NOT have CN_F_INVALID -> must NOT get nulls even though invalid_bytes are 0x01
    let cn_no_inval = Cn4 {
        unique_name: "ch_no_inval".to_string(),
        block: Cn4Block {
            cn_type: 0,
            cn_flags: 0,
            cn_inval_bit_pos: 0,
            ..Default::default()
        },
        data: ChannelData::UInt32(UInt32Array::from(vec![100, 200, 300])),
        invalid_mask: None,
        ..Default::default()
    };

    // Channel 3: VLSD (cn_type == 1) without CN_F_INVALID -> must NOT get nulls
    let cn_vlsd = Cn4 {
        unique_name: "ch_vlsd".to_string(),
        block: Cn4Block {
            cn_type: 1,
            cn_flags: 0,
            ..Default::default()
        },
        data: ChannelData::Utf8(LargeStringArray::from(vec!["a", "b", "c"])),
        invalid_mask: None,
        ..Default::default()
    };

    let mut cn_map: FxHashMap<i64, Cn4> = HashMap::with_hasher(Default::default());
    cn_map.insert(0, cn_invalid);
    cn_map.insert(4, cn_no_inval);
    cn_map.insert(8, cn_vlsd);
    cg.cn = cn_map;

    cg.process_all_channel_invalid_bits()?;

    assert_eq!(cg.cn[&0].data.as_arrow().null_count(), 3);
    assert_eq!(cg.cn[&4].data.as_arrow().null_count(), 0);
    assert_eq!(cg.cn[&8].data.as_arrow().null_count(), 0);
    Ok(())
}

/// Regression 2: 9-byte unaligned integer with bit_offset > 0 in decode_u64_slice.
#[test]
fn test_9byte_unaligned_integer_with_nonzero_bit_offset() -> Result<()> {
    // 64-bit value 0x0123_4567_89AB_CDEF shifted left by 5 bits across 9 bytes (LE)
    let expected: u64 = 0x0123_4567_89AB_CDEF;
    let shifted: u128 = (expected as u128) << 5;
    let le_bytes = shifted.to_le_bytes();
    let sample = &le_bytes[..9];
    let mut dst = [0u64; 1];
    decode_u64_slice(sample, &mut dst, 9, false, 5, 64)?;
    assert_eq!(dst[0], expected);
    Ok(())
}

/// Regression 3: MDF4 writer 8-byte alignment, sorted VLSD writing both `dg_data`
/// 64-bit offsets and `cn_data` `SDBLOCK`, and `CCBLOCK` physical units preserved.
#[test]
fn test_mdfwriter4_alignment_sorted_vlsd_and_cc_units() -> Result<()> {
    use polars::df;

    let df = df!(
        "time" => [0.0f64, 0.1, 0.2],
        "label" => ["x", "yy", "zzz"],
    )?;
    let tmp1 = NamedTempFile::new()?;
    let p1 = tmp1.path().to_str().unwrap();
    let _written = Mdf::write_polars(&df, p1, Some("time"), CompressionAlgorithm::NoCompression)?;
    let mut mdf = Mdf::new(p1)?;
    mdf.load_all_channels_data_in_memory()?;

    // Set unit on "time"
    mdf.set_channel_unit("time", "s");

    let tmp2 = NamedTempFile::new()?;
    let p2 = tmp2.path().to_str().unwrap();
    mdf.write(p2, CompressionAlgorithm::Deflate)?;

    // Verify 8-byte alignment and sorted VLSD layout
    let info = MdfInfo::new(p2)?;
    if let MdfInfo::V4(v4) = &info {
        for &at_pos in v4.at.keys() {
            assert_eq!(at_pos % 8, 0, "ATBLOCK at {at_pos} must be 8-byte aligned");
        }
        let mut found_vlsd = false;
        let mut found_time_unit = false;
        for dg in v4.dg.values() {
            for cg in dg.cg.values() {
                for cn in cg.cn.values() {
                    if cn.unique_name == "label" {
                        found_vlsd = true;
                        assert_eq!(cn.block.cn_type, 1, "VLSD channel must have cn_type == 1");
                        assert_ne!(
                            cn.block.cn_data, 0,
                            "Sorted VLSD must write cn_data SDBLOCK"
                        );
                        assert_ne!(
                            dg.block.dg_data, 0,
                            "Sorted VLSD must write dg_data offsets"
                        );
                    }
                    if cn.unique_name == "time"
                        && info.get_channel_unit("time")?.as_deref() == Some("s")
                    {
                        found_time_unit = true;
                    }
                }
            }
        }
        assert!(found_vlsd, "label VLSD channel not found");
        assert!(found_time_unit, "time unit 's' was not preserved");
    } else {
        panic!("Expected MDF4");
    }

    // Read back data to confirm full roundtrip
    let mut r = Mdf::new(p2)?;
    r.load_all_channels_data_in_memory()?;
    let labels = r.get_channel_data("label").expect("label data");
    if let ChannelData::Utf8(arr) = labels {
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.value(0), "x");
        assert_eq!(arr.value(1), "yy");
        assert_eq!(arr.value(2), "zzz");
    } else {
        panic!("Expected Utf8 for label");
    }
    Ok(())
}

/// Regression 4: Unified recovery (`recover_in_place` + read-only `MdfInfo::new`)
/// on unpatched `DLBLOCK` (`dl_count == 0`, `dl_links == 1`).
#[test]
fn test_unified_recovery_and_readonly_unpatched_dl() -> Result<()> {
    let mut buf = Vec::<u8>::new();
    let id = IdBlock {
        id_unfin_flags: 0x01 | 0x20,
        ..Default::default()
    };
    let mut id_bytes = Vec::new();
    id.write_le(&mut Cursor::new(&mut id_bytes))?;
    id_bytes[..8].copy_from_slice(b"UnFinMF ");
    buf.extend_from_slice(&id_bytes);

    let dl_slot_len = 72u64;
    let dt1_len = 24u64 + 16u64; // 2 cycles of 8 bytes
    let dt2_len = 24u64 + 24u64; // 3 cycles of 8 bytes

    let mut cur = Cursor::new(buf);
    cur.set_position(64);
    let fh_pos = 64u64 + 104u64;
    let mut hd = Hd4 {
        hd_fh_first: fh_pos as i64,
        ..Default::default()
    };
    hd.write_le(&mut cur)?;

    let fh = FhBlock::default();
    fh.write_le(&mut cur)?;

    let dg_pos = cur.position();
    let mut dg = Dg4Block::default();
    dg.write_le(&mut cur)?;

    let cg_pos = cur.position();
    let mut cg = Cg4Block {
        cg_data_bytes: 8,
        cg_cycle_count: 0, // unfinalized
        ..Default::default()
    };
    std::io::Write::write_all(&mut cur, b"##CG")?;
    std::io::Write::write_all(&mut cur, &[0u8; 4])?;
    std::io::Write::write_all(&mut cur, &104u64.to_le_bytes())?;
    cg.write_le(&mut cur)?;

    let cn_pos = cur.position();
    let cn = Cn4Block {
        cn_data_type: 4, // float64 LE
        cn_bit_count: 64,
        ..Default::default()
    };
    std::io::Write::write_all(&mut cur, b"##CN")?;
    std::io::Write::write_all(&mut cur, &[0u8; 4])?;
    std::io::Write::write_all(&mut cur, &160u64.to_le_bytes())?;
    cn.write_le(&mut cur)?;

    // Unpatched DLBLOCK (dl_count = 0, dl_links = 1, reserved 64-byte slot)
    let dl_pos = cur.position();
    let dl = Dl4Block {
        reserved: [0; 4],
        dl_len: 40,
        dl_links: 1,
        dl_dl_next: 0,
        dl_data: vec![],
        dl_flags: 0,
        dl_reserved: [0; 3],
        dl_count: 0,
        dl_equal_length: None,
        dl_offset: vec![],
        dl_time_values: vec![],
        dl_angle_values: vec![],
        dl_distance_values: vec![],
    };
    std::io::Write::write_all(&mut cur, b"##DL")?;
    dl.write_le(&mut cur)?;
    let dt1_pos = dl_pos + dl_slot_len;
    let mut buf = cur.into_inner();
    buf.resize(dt1_pos as usize, 0);
    let mut cur = Cursor::new(buf);
    cur.set_position(dt1_pos);

    // DT1 (2 float64s)
    let dt1 = Dt4Block {
        reserved: [0; 4],
        len: dt1_len,
        links: 0,
    };
    std::io::Write::write_all(&mut cur, b"##DT")?;
    dt1.write_le(&mut cur)?;
    for v in [1.0f64, 2.0f64] {
        std::io::Write::write_all(&mut cur, &v.to_le_bytes())?;
    }

    // DT2 (3 float64s)
    let dt2_pos = cur.position();
    let dt2 = Dt4Block {
        reserved: [0; 4],
        len: dt2_len,
        links: 0,
    };
    std::io::Write::write_all(&mut cur, b"##DT")?;
    dt2.write_le(&mut cur)?;
    for v in [3.0f64, 4.0f64, 5.0f64] {
        std::io::Write::write_all(&mut cur, &v.to_le_bytes())?;
    }

    // Patch links now that exact positions are known
    hd.hd_dg_first = dg_pos as i64;
    cur.set_position(64);
    hd.write_le(&mut cur)?;
    dg.dg_cg_first = cg_pos as i64;
    dg.dg_data = dl_pos as i64;
    cur.set_position(dg_pos);
    dg.write_le(&mut cur)?;
    cg.cg_cn_first = cn_pos as i64;
    cur.set_position(cg_pos + 16);
    cg.write_le(&mut cur)?;

    let buf = cur.into_inner();

    // 1. Read-only MdfInfo::new on unpatched DLBLOCK (does not modify file on disk)
    let tmp = NamedTempFile::new()?;
    std::fs::write(tmp.path(), &buf)?;
    let path_str = tmp.path().to_str().unwrap();

    let info_ro = MdfInfo::new(path_str)?;
    if let MdfInfo::V4(v4) = &info_ro {
        let dg0 = &v4.dg[&(dg_pos as i64)];
        assert_eq!(dg0.cg[&0].block.cg_cycle_count, 5);
        assert_eq!(
            dg0.recovered_dl_data.get(&(dl_pos as i64)),
            Some(&vec![dt1_pos as i64, dt2_pos as i64])
        );
    }

    // 2. In-place recovery via mdfr_core::logger::recover on the temp file
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(tmp.path())?;
    let patched = recover_in_place(&mut file)?;
    assert_eq!(patched, 1);

    let recovered = std::fs::read(tmp.path())?;
    assert_eq!(&recovered[0..8], b"MDF     ");
    let mut dl_cur = Cursor::new(&recovered[(dl_pos as usize) + 4..]);
    let rec_dl: Dl4Block = dl_cur.read_le()?;
    assert_eq!(rec_dl.dl_count, 2);
    assert_eq!(rec_dl.dl_offset, vec![0, 16]);
    let _ = Float64Array::from(vec![1.0]);
    Ok(())
}
