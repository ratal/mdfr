use anyhow::Result;
use arrow::array::{
    Array, AsArray, Float64Builder, Int32Builder, LargeStringBuilder, UInt16Builder, UInt64Builder,
};
use arrow::datatypes::Float64Type;
use mdfr::data_holder::channel_data::ChannelData;
use mdfr::mdfreader::Mdf;
use std::path::Path;
use std::sync::LazyLock;

static BASE_PATH_MDF4: LazyLock<String> = LazyLock::new(|| {
    "/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/MDF4.3/Base_Standard/Examples/"
        .to_string()
});

#[test]
fn master_channels() -> Result<()> {
    let list_of_paths = ["ChannelTypes/MasterChannels/".to_string()];

    // MasterTypes testing
    let mut vect: Vec<f64> = vec![0.; 101];
    let mut counter: f64 = 0.;
    vect.iter_mut().for_each(|v| {
        *v = counter * 0.03;
        counter += 1.
    });
    let expected_master = ChannelData::Float64(Float64Builder::new_from_buffer(vect.into(), None));

    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_VirtualTimeMasterChannel.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Time channel") {
        assert_eq!(expected_master, data.clone());
    }

    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_DifferentMasterChannels.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Time channel") {
        assert_eq!(expected_master, data.clone());
    }

    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_NoMasterChannel.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Time channel") {
        assert_eq!(expected_master, data.clone());
    }
    Ok(())
}

#[test]
fn mlsd_channels() -> Result<()> {
    let list_of_paths = ["ChannelTypes/MLSD/".to_string()];

    // MLSD testing
    let mut expected_string_result = LargeStringBuilder::with_capacity(10, 6);
    expected_string_result.append_value("zero");
    expected_string_result.append_value("one");
    expected_string_result.append_value("two");
    expected_string_result.append_value("three");
    expected_string_result.append_value("four");
    expected_string_result.append_value("five");
    expected_string_result.append_value("six");
    expected_string_result.append_value("seven");
    expected_string_result.append_value("eight");
    expected_string_result.append_value("nine");
    let expected_string_result = ChannelData::Utf8(expected_string_result);

    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_MLSD_String_UTF8.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Data channel") {
        assert_eq!(expected_string_result, data.clone());
    }

    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_MLSD_String_SBC.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Data channel") {
        assert_eq!(expected_string_result, data.clone());
    }

    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_MLSD_String_UTF16_BE.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Data channel") {
        assert_eq!(expected_string_result, data.clone());
    }

    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_MLSD_String_UTF16_LE.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Data channel") {
        assert_eq!(expected_string_result, data.clone());
    }
    Ok(())
}

#[test]
fn virtual_data_channels() -> Result<()> {
    let list_of_paths = ["ChannelTypes/VirtualData/".to_string()];

    // Virtual data testing
    let mut vect: Vec<u64> = vec![0; 200];
    let mut counter: u64 = 0;
    vect.iter_mut().for_each(|v| {
        *v += counter;
        counter += 1
    });
    let virtal_vect = UInt64Builder::new_from_buffer(vect.into(), None);

    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_VirtualDataChannelNoConversion.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Data channel") {
        assert_eq!(ChannelData::UInt64(virtal_vect), *data);
    }

    let mut vect: Vec<f64> = vec![100.0f64; 200];
    let mut counter: f64 = 0.0;
    vect.iter_mut().for_each(|v| {
        *v += counter;
        counter -= 2.0
    });
    let virtal_linear_vect = Float64Builder::new_from_buffer(vect.into(), None);

    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_VirtualDataChannelLinearConversion.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Data channel") {
        assert_eq!(ChannelData::Float64(virtal_linear_vect), *data);
    }

    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_VirtualDataChannelConstantConversion.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Data channel") {
        assert_eq!(
            ChannelData::Float64(Float64Builder::new_from_buffer(
                vec![42f64; 200].into(),
                None
            )),
            *data
        );
    }
    Ok(())
}

#[test]
fn vlsd_channels() -> Result<()> {
    let list_of_paths = ["ChannelTypes/VLSD/".to_string()];

    let mut expected_string_result = LargeStringBuilder::with_capacity(10, 6);
    expected_string_result.append_value("zero");
    expected_string_result.append_value("one");
    expected_string_result.append_value("two");
    expected_string_result.append_value("three");
    expected_string_result.append_value("four");
    expected_string_result.append_value("five");
    expected_string_result.append_value("six");
    expected_string_result.append_value("seven");
    expected_string_result.append_value("eight");
    expected_string_result.append_value("nine");
    let expected_string_result = ChannelData::Utf8(expected_string_result);

    // VLSD testing
    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_VLSD_String_UTF8.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Data channel") {
        assert_eq!(expected_string_result, data.clone());
    }

    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_VLSD_String_UTF16_LE.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Data channel") {
        assert_eq!(expected_string_result, data.clone());
    }

    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_VLSD_String_UTF16_BE.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Data channel") {
        assert_eq!(expected_string_result, data.clone());
    }
    Ok(())
}

#[test]
fn vlsc_channels() -> Result<()> {
    let list_of_paths = ["ChannelTypes/VLSC/".to_string()];

    let mut expected_string_result = LargeStringBuilder::with_capacity(10, 6);
    expected_string_result.append_value("zero");
    expected_string_result.append_value("one");
    expected_string_result.append_value("two");
    expected_string_result.append_value("three");
    expected_string_result.append_value("four");
    expected_string_result.append_value("five");
    expected_string_result.append_value("six");
    expected_string_result.append_value("seven");
    expected_string_result.append_value("eight");
    expected_string_result.append_value("nine");
    let expected_string_result = ChannelData::Utf8(expected_string_result);

    // VLSC
    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Vector_VLSC_String_UTF8.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Data channel") {
        assert!(!data.is_empty(), "VLSC channel data should not be empty");
        assert_eq!(data.len(), 10, "VLSC channel should have 10 samples");
        assert_eq!(expected_string_result, data.clone());
    } else {
        panic!("VLSC Data channel not found");
    }

    // VLSC with different string encodings
    for file in [
        "Vector_VLSC_String_SBC.mf4",
        "Vector_VLSC_String_UTF16_LE.mf4",
        "Vector_VLSC_String_UTF16_BE.mf4",
    ] {
        let file_name = format!("{}{}{}", BASE_PATH_MDF4.as_str(), list_of_paths[0], file);
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("Data channel") {
            assert_eq!(expected_string_result, data.clone(), "Failed for {}", file);
        } else {
            panic!("VLSC Data channel not found in {}", file);
        }
    }

    // VLSC with single VD block (uncompressed and compressed)
    for file in [
        "Vector_VLSC_Single_VD.mf4",
        "Vector_VLSC_Single_VD_Compressed.mf4",
    ] {
        let file_name = format!("{}{}{}", BASE_PATH_MDF4.as_str(), list_of_paths[0], file);
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("data") {
            assert!(!data.is_empty(), "VLSC data should not be empty in {}", file);
        } else {
            panic!("VLSC data channel not found in {}", file);
        }
    }

    // VLSC with Data List (DL -> VD blocks), uncompressed and compressed (DL -> DZ)
    for file in [
        "Vector_VLSC_DataList_VD.mf4",
        "Vector_VLSC_DataList_VD_Compressed.mf4",
    ] {
        let file_name = format!("{}{}{}", BASE_PATH_MDF4.as_str(), list_of_paths[0], file);
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("data") {
            assert!(!data.is_empty(), "VLSC data should not be empty in {}", file);
        } else {
            panic!("VLSC data channel not found in {}", file);
        }
    }

    // VLSC Etas with BOM - this file has channels: time, size, comment (VLSC)
    // Note: This file has mixed BOM encodings (UTF-8 and UTF-16 LE)
    let file_name = format!(
        "{}{}{}",
        BASE_PATH_MDF4.as_str(),
        list_of_paths[0],
        "Etas_VLSC_String_UTF_with_BOM.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("comment") {
        assert!(
            !data.is_empty(),
            "VLSC Etas 'comment' channel data should not be empty"
        );
        // Verify it's actually string data (Utf8 type)
        assert!(
            matches!(data, ChannelData::Utf8(_)),
            "VLSC Etas 'comment' channel should be Utf8 type, got {:?}",
            data
        );
    } else {
        panic!("VLSC Etas 'comment' channel not found");
    }
    Ok(())
}

#[test]
fn synchronization_channels() -> Result<()> {
    // Synchronization
    let file_name = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "ChannelTypes/Synchronization/Vector_SyncStreamChannel.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    Ok(())
}

#[test]
fn channel_list() -> Result<()> {
    // Channel List (CL) + Data Stream (DS) test
    // File: simple_list.mf4 contains a dynamic list structure with:
    // - time: master channel (2 samples) [0.0, 1.0]
    // - size: indicates list size (2 samples) [0, 202]
    // - x: parent structure (FixedSizeByteArray with CL block)
    // - x.a: first member (Int32) [0, 1000000000]
    // - x.b: second member (Float64) [0.0, 2.121995791e-314]
    let file_name = "/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/MDF4.3/Base_Standard/Examples/DynamicData/ChannelList/simple_list.mf4";
    let mut mdf = Mdf::new(file_name)?;
    mdf.load_all_channels_data_in_memory()?;

    // Verify all channels are present
    assert!(
        mdf.get_channel_data("size").is_some(),
        "size channel should exist"
    );
    assert!(
        mdf.get_channel_data("x").is_some(),
        "x channel should exist"
    );
    assert!(
        mdf.get_channel_data("x.a").is_some(),
        "x.a channel should exist"
    );
    assert!(
        mdf.get_channel_data("x.b").is_some(),
        "x.b channel should exist"
    );
    assert!(
        mdf.get_channel_data("time").is_some(),
        "time channel should exist"
    );

    // Verify time channel (master)
    if let Some(time_data) = mdf.get_channel_data("time") {
        assert_eq!(time_data.len(), 2, "time should have 2 samples");
        assert_eq!(
            mdf.get_channel_master_type("time"),
            1,
            "time should be master type 1 (Time)"
        );
    }

    // Verify size channel values
    if let Some(size_data) = mdf.get_channel_data("size") {
        assert_eq!(size_data.len(), 2, "size should have 2 samples");
        // Size values: [0, 202]
        let expected_size = ChannelData::UInt16(UInt16Builder::new_from_buffer(
            vec![0u16, 202u16].into(),
            None,
        ));
        assert_eq!(
            &expected_size, size_data,
            "size channel values should match"
        );
    }

    // Verify x.a channel values (Int32)
    if let Some(xa_data) = mdf.get_channel_data("x.a") {
        assert_eq!(xa_data.len(), 2, "x.a should have 2 samples");
        let expected_xa = ChannelData::Int32(Int32Builder::new_from_buffer(
            vec![0i32, 1000000000i32].into(),
            None,
        ));
        assert_eq!(&expected_xa, xa_data, "x.a channel values should match");
    }

    // Verify x.b channel values (Float64)
    if let Some(xb_data) = mdf.get_channel_data("x.b") {
        assert_eq!(xb_data.len(), 2, "x.b should have 2 samples");
        // First value should be 0.0
        let values = xb_data.finish_cloned();
        let float_values = values.as_primitive::<Float64Type>();
        assert_eq!(float_values.value(0), 0.0, "x.b first value should be 0.0");
    }
    Ok(())
}

#[test]
fn channel_variant() -> Result<()> {
    // Channel Variant (CV) test
    // File: Etas_cv_storage_with_fixed_length.mf4 contains:
    // - time: master channel (3 samples) [0.0, 1.0, 2.0]
    // - discriminator: variant selector (3 samples) [0, 1, 2]
    // - variant: merged variant data based on discriminator value
    let file_name = "/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/MDF4.3/Base_Standard/Examples/Variant/Etas_cv_storage_with_fixed_length.mf4";
    let mut mdf = Mdf::new(file_name)?;
    mdf.load_all_channels_data_in_memory()?;

    // Verify all 3 channels are present
    assert!(
        mdf.get_channel_data("time").is_some(),
        "time channel should exist"
    );
    assert!(
        mdf.get_channel_data("discriminator").is_some(),
        "discriminator channel should exist"
    );
    assert!(
        mdf.get_channel_data("variant").is_some(),
        "variant channel should exist"
    );

    // Verify time channel (master)
    if let Some(time_data) = mdf.get_channel_data("time") {
        assert_eq!(time_data.len(), 3, "time should have 3 samples");
        let expected_time = ChannelData::Float64(Float64Builder::new_from_buffer(
            vec![0.0f64, 1.0f64, 2.0f64].into(),
            None,
        ));
        assert_eq!(
            &expected_time, time_data,
            "time channel values should match [0.0, 1.0, 2.0]"
        );
        assert_eq!(
            mdf.get_channel_master_type("time"),
            1,
            "time should be master type 1 (Time)"
        );
    }

    // Verify discriminator channel
    if let Some(disc_data) = mdf.get_channel_data("discriminator") {
        assert_eq!(disc_data.len(), 3, "discriminator should have 3 samples");
        let expected_disc = ChannelData::UInt16(UInt16Builder::new_from_buffer(
            vec![0u16, 1u16, 2u16].into(),
            None,
        ));
        assert_eq!(
            &expected_disc, disc_data,
            "discriminator values should match [0, 1, 2]"
        );
    }

    // Verify variant channel is a dense UnionArray with 3 mixed-type options
    if let Some(variant_data) = mdf.get_channel_data("variant") {
        assert_eq!(variant_data.len(), 3, "variant should have 3 samples");
        if let ChannelData::Union(arr) = variant_data {
            assert_eq!(arr.len(), 3, "UnionArray should have 3 samples");
            let data_type = arr.data_type();
            if let arrow::datatypes::DataType::Union(fields, arrow::datatypes::UnionMode::Dense) =
                data_type
            {
                assert_eq!(fields.len(), 3, "Union should have 3 option fields");
            } else {
                panic!(
                    "variant channel should be a Dense Union, got {:?}",
                    data_type
                );
            }
            // Verify each sample selects the correct option via type_ids [0, 1, 2]
            assert_eq!(arr.type_id(0), 0, "sample 0 should select option 0");
            assert_eq!(arr.type_id(1), 1, "sample 1 should select option 1");
            assert_eq!(arr.type_id(2), 2, "sample 2 should select option 2");
        } else {
            panic!(
                "variant channel should be ChannelData::Union for mixed types, got {:?}",
                std::mem::discriminant(variant_data)
            );
        }
    }
    Ok(())
}

#[test]
fn data_stream_block() -> Result<()> {
    // DS Block (Data Stream) test
    // Note: DS blocks are implicitly tested via ChannelList (simple_list.mf4)
    // where the "x" channel uses DSBLOCK for data stream mode
    let file_name = "/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/MDF4.3/Base_Standard/Examples/DynamicData/ChannelList/simple_list.mf4";
    let mut mdf = Mdf::new(file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    // "x" channel uses DS block - verify it's readable
    assert!(
        mdf.get_channel_data("x").is_some(),
        "DS-based channel 'x' should be readable"
    );
    Ok(())
}

#[test]
fn channel_union() -> Result<()> {
    // CU Block (Channel Union) test
    // File: Etas_cu_storage_with_fixed_length.mf4 contains:
    // - time: master channel (3 samples) [0.0, 1.0, 2.0]
    // - union: union data storing different member types in same space
    let file_name = "/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/MDF4.3/Base_Standard/Examples/Union/Etas_cu_storage_with_fixed_length.mf4";
    if Path::new(file_name).exists() {
        let mut mdf = Mdf::new(file_name)?;
        mdf.load_all_channels_data_in_memory()?;

        // Verify both channels present
        assert!(
            mdf.get_channel_data("time").is_some(),
            "time channel should exist"
        );
        assert!(
            mdf.get_channel_data("union").is_some(),
            "union channel should exist"
        );

        // Verify time channel
        if let Some(time_data) = mdf.get_channel_data("time") {
            assert_eq!(time_data.len(), 3, "time should have 3 samples");
            let expected_time = ChannelData::Float64(Float64Builder::new_from_buffer(
                vec![0.0f64, 1.0f64, 2.0f64].into(),
                None,
            ));
            assert_eq!(
                &expected_time, time_data,
                "time channel values should match [0.0, 1.0, 2.0]"
            );
        }

        // Verify union channel exists, has correct length, and is Union type
        if let Some(union_data) = mdf.get_channel_data("union") {
            assert_eq!(union_data.len(), 3, "union should have 3 samples");
            // Verify it's now a Union type (not FixedSizeByteArray)
            if let ChannelData::Union(arr) = union_data {
                // UnionArray should have the same length
                assert_eq!(arr.len(), 3, "UnionArray should have 3 samples");
                // Check that we have member fields
                let data_type = arr.data_type();
                if let arrow::datatypes::DataType::Union(fields, _mode) = data_type {
                    assert!(
                        !fields.is_empty(),
                        "Union should have at least one member field"
                    );
                }
            } else {
                panic!(
                    "union channel should be ChannelData::Union type, got {:?}",
                    std::mem::discriminant(union_data)
                );
            }
        }
    }
    Ok(())
}

#[test]
fn vd_block() -> Result<()> {
    // VD Block (Virtual Data)
    let file_name = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "ChannelTypes/VirtualData/Vector_VirtualDataChannelLinearConversion.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    if let Some(data) = mdf.get_channel_data("Data channel") {
        // Virtual data should be generated correctly
        assert_eq!(data.len(), 200);
    }
    Ok(())
}
