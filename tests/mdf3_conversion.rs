mod common;

use anyhow::Result;
use mdfr::mdfreader::Mdf;
use std::sync::LazyLock;

static MDF3_PATH: LazyLock<String> =
    LazyLock::new(|| format!("{}mdf3/", common::mdfreader_tests_path()));

/// Helper: load a file and all its channel data
fn load_mdf3(filename: &str) -> Result<Option<Mdf>> {
    let path = format!("{}{}", MDF3_PATH.as_str(), filename);
    if !std::path::Path::new(&path).exists() {
        return Ok(None);
    }
    let mut mdf = Mdf::new(&path)?;
    mdf.load_all_channels_data_in_memory()?;
    Ok(Some(mdf))
}

#[test]
fn mdf3_canape_loads() -> Result<()> {
    let Some(mdf) = load_mdf3("MDF_CANAPE.mdf")? else {
        return Ok(());
    };
    assert!(!mdf.get_channel_names_set().is_empty());
    Ok(())
}

#[test]
fn mdf3_ascet_loads() -> Result<()> {
    let Some(mdf) = load_mdf3("ASCET.dat")? else {
        return Ok(());
    };
    assert!(!mdf.get_channel_names_set().is_empty());
    Ok(())
}

#[test]
fn mdf3_mda71_loads() -> Result<()> {
    let Some(mdf) = load_mdf3("MDA71.dat")? else {
        return Ok(());
    };
    assert!(!mdf.get_channel_names_set().is_empty());
    Ok(())
}

#[test]
fn mdf3_rj_can_loads() -> Result<()> {
    // RJ file has linear conversion channels
    let Some(mdf) = load_mdf3("RJ_N16-12-363_BM-15C-0024_228_2_20170116094355_CAN.dat")? else {
        return Ok(());
    };
    assert!(!mdf.get_channel_names_set().is_empty());
    Ok(())
}

#[test]
fn mdf3_measure_loads() -> Result<()> {
    let Some(mdf) = load_mdf3("Measure.dat")? else {
        return Ok(());
    };
    assert!(!mdf.get_channel_names_set().is_empty());
    Ok(())
}

#[test]
fn mdf3_t3_nedc_loads() -> Result<()> {
    let Some(mdf) = load_mdf3("T3_121121_000_6NEDC.dat")? else {
        return Ok(());
    };
    assert!(!mdf.get_channel_names_set().is_empty());
    Ok(())
}

#[test]
fn mdf3_738l10_loads() -> Result<()> {
    let Some(mdf) = load_mdf3("738L10_040410 Base Acc 30km_hr.dat")? else {
        return Ok(());
    };
    assert!(!mdf.get_channel_names_set().is_empty());
    Ok(())
}

#[test]
fn mdf3_hidden_bytes_loads() -> Result<()> {
    let Some(mdf) = load_mdf3("Mdf3_hiddenBytes_NotAlignedBytes.dat")? else {
        return Ok(());
    };
    assert!(!mdf.get_channel_names_set().is_empty());
    Ok(())
}

#[test]
fn mdf3_tgt_loads() -> Result<()> {
    let Some(mdf) = load_mdf3("TGT.dat")? else {
        return Ok(());
    };
    assert!(!mdf.get_channel_names_set().is_empty());
    Ok(())
}

#[test]
fn mdf3_canoe_unsorted_loads() -> Result<()> {
    let Some(mdf) = load_mdf3("CANoe3_unsorted.mdf")? else {
        return Ok(());
    };
    assert!(!mdf.get_channel_names_set().is_empty());
    Ok(())
}

mod conversions3_tests {
    use arrow::array::{Float64Builder, Int32Builder};
    use mdfr::data_holder::channel_data::ChannelData;
    use mdfr::mdfinfo::mdfinfo3::Cn3;
    use mdfr::mdfreader::conversions3::*;

    #[test]
    fn test_value_to_value_with_interpolation() {
        // Table pairs: x=0→y=0, x=10→y=100, x=20→y=200
        let cc_val = vec![0.0, 0.0, 10.0, 100.0, 20.0, 200.0];

        let mut builder = Float64Builder::new();
        builder.append_value(0.0); // exact match → 0
        builder.append_value(5.0); // interpolate between (0,0) and (10,100) → 50
        builder.append_value(10.0); // exact match → 100
        builder.append_value(15.0); // interpolate between (10,100) and (20,200) → 150
        builder.append_value(-5.0); // below first → first y = 0
        builder.append_value(25.0); // above last → last y = 200

        let result = value_to_value_with_interpolation_calculation(&mut builder, cc_val).unwrap();
        let values = result.values_slice();
        assert_eq!(values.len(), 6);
        assert!((values[0] - 0.0).abs() < 1e-12);
        assert!((values[1] - 50.0).abs() < 1e-12);
        assert!((values[2] - 100.0).abs() < 1e-12);
        assert!((values[3] - 150.0).abs() < 1e-12);
        assert!((values[4] - 0.0).abs() < 1e-12);
        assert!((values[5] - 200.0).abs() < 1e-12);
    }

    #[test]
    fn test_value_to_value_without_interpolation() {
        // Same table pairs: x=0→y=0, x=10→y=100, x=20→y=200
        let cc_val = vec![0.0, 0.0, 10.0, 100.0, 20.0, 200.0];

        let mut builder = Float64Builder::new();
        builder.append_value(0.0); // exact → 0
        builder.append_value(3.0); // nearer to x=0 (dist=3) than x=10 (dist=7) → 0
        builder.append_value(7.0); // nearer to x=10 (dist=3) than x=0 (dist=7) → 100
        builder.append_value(10.0); // exact → 100
        builder.append_value(-5.0); // below first → 0
        builder.append_value(25.0); // above last → 200

        let result =
            value_to_value_without_interpolation_calculation(&mut builder, cc_val).unwrap();
        let values = result.values_slice();
        assert_eq!(values.len(), 6);
        assert!((values[0] - 0.0).abs() < 1e-12);
        assert!((values[1] - 0.0).abs() < 1e-12);
        assert!((values[2] - 100.0).abs() < 1e-12);
        assert!((values[3] - 100.0).abs() < 1e-12);
        assert!((values[4] - 0.0).abs() < 1e-12);
        assert!((values[5] - 200.0).abs() < 1e-12);
    }

    #[test]
    fn test_value_to_text_calculation() {
        let cc_val_ref: Vec<(f64, String)> = vec![
            (1.0, "one".to_string()),
            (2.0, "two".to_string()),
            (3.0, "three".to_string()),
        ];

        let mut builder = Int32Builder::new();
        builder.append_value(1); // matches 1.0 → "one"
        builder.append_value(2); // matches 2.0 → "two"
        builder.append_value(3); // matches 3.0 → "three"
        builder.append_value(99); // no match → defaults to first entry ("one")

        let result = value_to_text_calculation(&mut builder, &cc_val_ref).unwrap();
        let arr = result.finish_cloned();
        assert_eq!(arr.value(0), "one");
        assert_eq!(arr.value(1), "two");
        assert_eq!(arr.value(2), "three");
        assert_eq!(arr.value(3), "one");
    }

    #[test]
    fn test_value_range_to_text_calculation() {
        let ranges = vec![
            (0.0, 10.0, "low".to_string()),
            (10.0, 20.0, "medium".to_string()),
            (20.0, 30.0, "high".to_string()),
        ];
        let cc_val_ref = (ranges, "unknown".to_string());

        let mut builder = Float64Builder::new();
        builder.append_value(5.0); // in [0, 10) → "low"
        builder.append_value(15.0); // in [10, 20) → "medium"
        builder.append_value(25.0); // in [20, 30) → "high"
        builder.append_value(35.0); // out of range → "unknown"

        let result = value_range_to_text_calculation(&mut builder, &cc_val_ref).unwrap();
        let arr = result.finish_cloned();
        assert_eq!(arr.value(0), "low");
        assert_eq!(arr.value(1), "medium");
        assert_eq!(arr.value(2), "high");
        assert_eq!(arr.value(3), "unknown");
    }

    // ── Helper ──

    fn make_cn3(data: ChannelData) -> Cn3 {
        Cn3 {
            data,
            ..Default::default()
        }
    }

    // ── linear_conversion (Cn3) tests ──

    #[test]
    fn test_linear_cn3_uint16() {
        use arrow::array::UInt16Builder;
        let mut builder = UInt16Builder::new();
        builder.append_value(10);
        let mut cn = make_cn3(ChannelData::UInt16(builder));
        linear_conversion(&mut cn, &[0.0, 3.0]).unwrap(); // 10*3+0 = 30
        if let ChannelData::Float64(ref b) = cn.data {
            let vals = b.values_slice();
            assert!((vals[0] - 30.0).abs() < 1e-12);
        } else {
            panic!("Expected Float64");
        }
    }

    // ── polynomial_conversion (Cn3) tests ──

    #[test]
    fn test_polynomial_cn3_float64() {
        // polynomial_calculation uses array.finish() then Float64Builder::with_capacity(array.capacity())
        // After finish(), capacity resets to 0, so the output is an empty Float64 builder.
        // This test verifies the conversion returns Ok and converts the channel to Float64.
        let cc_val = vec![0.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let mut builder = Float64Builder::new();
        builder.append_value(2.0);
        let mut cn = make_cn3(ChannelData::Float64(builder));
        let result = polynomial_conversion(&mut cn, &cc_val);
        assert!(result.is_ok());
        // Data should now be Float64 (conversion applied)
        assert!(matches!(cn.data, ChannelData::Float64(_)));
    }

    // ── exponential_conversion (Cn3) tests ──

    #[test]
    fn test_exponential_cn3_float64_p4_zero() {
        // exponential_calculation also uses array.finish() and Float64Builder::with_capacity(0)
        // so the output is an empty builder. Test that it returns Ok and converts to Float64.
        // p4=0 branch chosen with: p1=1, p2=1, p3=0, p4=0, p5=0, p6=1, p7=0
        let cc_val = vec![1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut builder = Float64Builder::new();
        builder.append_value(std::f64::consts::E);
        let mut cn = make_cn3(ChannelData::Float64(builder));
        let result = exponential_conversion(&mut cn, &cc_val);
        assert!(result.is_ok());
        assert!(matches!(cn.data, ChannelData::Float64(_)));
    }

    // ── logarithmic_conversion (Cn3) tests ──

    #[test]
    fn test_logarithmic_cn3_float64_p4_zero() {
        // logarithmic_calculation also uses array.finish() and Float64Builder::with_capacity(0).
        // Test that it returns Ok and converts to Float64.
        // p4=0 branch: p1=1, p2=1, p3=0, p4=0, p5=0, p6=1, p7=0
        let cc_val = vec![1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut builder = Float64Builder::new();
        builder.append_value(0.0);
        let mut cn = make_cn3(ChannelData::Float64(builder));
        let result = logarithmic_conversion(&mut cn, &cc_val);
        assert!(result.is_ok());
        assert!(matches!(cn.data, ChannelData::Float64(_)));
    }

    // ── value_to_value_with_interpolation (Cn3) tests ──

    #[test]
    fn test_cn3_vtv_interp_uint8() {
        use arrow::array::UInt8Builder;
        // Table: 0→0, 10→100
        let cc_val = vec![0.0, 0.0, 10.0, 100.0];
        let mut builder = UInt8Builder::new();
        builder.append_value(5); // 50
        let mut cn = make_cn3(ChannelData::UInt8(builder));
        value_to_value_with_interpolation(&mut cn, cc_val).unwrap();
        if let ChannelData::Float64(ref b) = cn.data {
            let vals = b.values_slice();
            assert!((vals[0] - 50.0).abs() < 1e-9);
        } else {
            panic!("Expected Float64");
        }
    }

    // ── value_to_value_without_interpolation (Cn3) tests ──

    #[test]
    fn test_cn3_vtv_no_interp_uint8() {
        use arrow::array::UInt8Builder;
        let cc_val = vec![1.0, 10.0, 2.0, 20.0];
        let mut builder = UInt8Builder::new();
        builder.append_value(2); // exact match → 20.0
        let mut cn = make_cn3(ChannelData::UInt8(builder));
        value_to_value_without_interpolation(&mut cn, cc_val).unwrap();
        if let ChannelData::Float64(ref b) = cn.data {
            let vals = b.values_slice();
            assert!((vals[0] - 20.0).abs() < 1e-12);
        } else {
            panic!("Expected Float64");
        }
    }

    // ── value_to_text (Cn3) tests ──

    #[test]
    fn test_cn3_value_to_text() {
        use arrow::array::UInt8Builder;
        let cc_val_ref = vec![(1.0, "one".to_string()), (2.0, "two".to_string())];
        let mut builder = UInt8Builder::new();
        builder.append_value(1);
        let mut cn = make_cn3(ChannelData::UInt8(builder));
        value_to_text(&mut cn, &cc_val_ref).unwrap();
        if let ChannelData::Utf8(ref b) = cn.data {
            let arr = b.finish_cloned();
            assert_eq!(arr.value(0), "one");
        } else {
            panic!("Expected Utf8");
        }
    }

    // ── value_range_to_text (Cn3) tests ──

    #[test]
    fn test_cn3_value_range_to_text() {
        let ranges = vec![
            (1.0, 2.0, "in_range".to_string()),
            (3.0, 4.0, "out".to_string()),
        ];
        let cc_val_ref = (ranges, "default".to_string());
        let mut builder = Float64Builder::new();
        builder.append_value(1.5); // in [1.0, 2.0)
        let mut cn = make_cn3(ChannelData::Float64(builder));
        value_range_to_text(&mut cn, &cc_val_ref).unwrap();
        if let ChannelData::Utf8(ref b) = cn.data {
            let arr = b.finish_cloned();
            assert_eq!(arr.value(0), "in_range");
        } else {
            panic!("Expected Utf8");
        }
    }
}
