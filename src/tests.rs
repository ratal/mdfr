#[cfg(test)]
mod tests {
    use anyhow::Result;
    use arrow2::array::BinaryArray;
    use arrow2::array::MutableArray;
    use arrow2::array::MutablePrimitiveArray;
    use arrow2::array::MutableUtf8Array;
    use arrow2::array::PrimitiveArray;
    use arrow2::array::Utf8Array;
    use arrow2::buffer::Buffer;
    use arrow2::compute::aggregate::min_primitive;
    use arrow2::datatypes::DataType;

    use crate::mdfreader::Mdf;
    use std::fs;
    use std::io;
    use std::path::Path;
    use std::vec::Vec;
    use test_log::test;

    static BASE_PATH_MDF4: &str = "/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/";
    static BASE_PATH_MDF3: &str = "/home/ratal/workspace/mdfreader/mdfreader/tests/mdf3/";
    static WRITING_MDF_FILE: &str = "/home/ratal/workspace/mdfr/test_files/test.mf4";
    static WRITING_PARQUET_FILE: &str = "/home/ratal/workspace/mdfr/test_files/test.parquet";

    #[test]
    fn info_test() -> Result<()> {
        let mut file_name = "test_files/test_basic.mf4";
        println!("reading {}", file_name);
        let mdf = Mdf::new(file_name)?;
        println!("{:#?}", mdf);
        assert_eq!(mdf.get_version(), 410);
        file_name = "test_files/test_mdf3.mdf";
        println!("reading {}", file_name);
        let mdf = Mdf::new(file_name)?;
        println!("{:#?}", &mdf);
        assert_eq!(mdf.get_version(), 310);
        file_name = "test_files/test_mdf4.mf4";
        println!("reading {}", file_name);
        let mdf = Mdf::new(file_name)?;
        println!("{:#?}", &mdf);
        assert_eq!(mdf.get_version(), 400);
        Ok(())
    }

    fn parse_info_folder(folder: &String) -> Result<()> {
        let path = Path::new(folder);
        let valid_ext: Vec<String> = vec![
            "mf4".to_string(),
            "MF4".to_string(),
            "DAT".to_string(),
            "dat".to_string(),
            "MDF".to_string(),
            "mdf".to_string(),
        ];
        if path.is_dir() {
            for entry in fs::read_dir(path)? {
                let entry = entry?;
                if let Ok(metadata) = entry.metadata() {
                    if metadata.is_file() {
                        if let Ok(ext) = entry
                            .path()
                            .extension()
                            .unwrap()
                            .to_os_string()
                            .into_string()
                        {
                            if valid_ext.contains(&ext) {
                                if let Some(file_name) = entry.path().to_str() {
                                    println!(" Reading file : {}", file_name);
                                    let mut mdf = Mdf::new(file_name)?;
                                    mdf.load_all_channels_data_in_memory()?;
                                }
                            }
                        }
                    } else if metadata.is_dir() {
                        if let Some(path) = entry.path().to_str() {
                            let path_str = path.to_owned();
                            let _ = match parse_info_folder(&path_str) {
                                Ok(v) => v,
                                Err(e) => {
                                    println!("Error parsing the folder {} \n {}", path_str, e)
                                }
                            };
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn parse_all_folders4() -> io::Result<()> {
        let list_of_paths = [
            "Simple".to_string(),
            "ChannelInfo".to_string(),
            "ChannelTypes/MasterChannels".to_string(),
            "ChannelTypes/MLSD".to_string(),
            "ChannelTypes/VLSD".to_string(),
            "ChannelTypes/VirtualData".to_string(),
            "ChannelTypes/Synchronization".to_string(),
            "DataTypes/ByteArray".to_string(),
            "DataTypes/CANopenTypes".to_string(),
            "DataTypes/IntegerTypes".to_string(),
            "DataTypes/RealTypes".to_string(),
            "DataTypes/StringTypes".to_string(),
            "MetaData".to_string(),
            "RecordLayout".to_string(),
            "Events".to_string(),
            "SampleReduction".to_string(),
            "BusLogging".to_string(),
            "Conversion/LinearConversion".to_string(),
            "Conversion/LookUpConversion".to_string(),
            "Conversion/PartialConversion".to_string(),
            "Conversion/RationalConversion".to_string(),
            "Conversion/StringConversion".to_string(),
            "Conversion/TextConversion".to_string(),
            "Attachments/Embedded".to_string(),
            "Attachments/EmbeddedCompressed".to_string(),
            "Attachments/External".to_string(),
            "CompressedData/DataList".to_string(),
            "CompressedData/Simple".to_string(),
            "CompressedData/Unsorted".to_string(),
            "DataList".to_string(),
            "UnsortedData".to_string(),
            "Arrays".to_string(),
            "ClassificationResults".to_string(),
        ];
        for path in list_of_paths.iter() {
            println!("reading folder : {}", path);
            parse_info_folder(&format!("{}{}", BASE_PATH_MDF4, &path)).unwrap();
        }
        Ok(())
    }

    #[test]
    fn parse_all_folders3() -> io::Result<()> {
        let base_path = String::from("/home/ratal/workspace/mdfreader/mdfreader/tests/mdf3/");
        parse_info_folder(&base_path).unwrap();
        Ok(())
    }

    #[test]
    fn basic_test() -> Result<()> {
        let file = "test_files/test_basic.mf4";
        let mut mdf = Mdf::new(&file)?;
        mdf.load_all_channels_data_in_memory()?;
        mdf.write("test_files/test.mf4", true)?;
        Ok(())
    }

    #[test]
    fn data_types() -> Result<()> {
        let list_of_paths = [
            "DataTypes/ByteArray/".to_string(),
            "DataTypes/CANopenTypes/".to_string(),
            "DataTypes/IntegerTypes/".to_string(),
            "DataTypes/RealTypes/".to_string(),
            "DataTypes/StringTypes/".to_string(),
        ];
        let writing_mdf_file = format!("{}{}", WRITING_MDF_FILE, "_test".to_owned()).to_owned();

        // StringTypes testing
        // UTF8
        let expected_string_result = Utf8Array::<i64>::from([
            Some("zero"),
            Some("one"),
            Some("two"),
            Some("three"),
            Some("four"),
            Some("five"),
            Some("six"),
            Some("seven"),
            Some("eight"),
            Some("nine"),
        ])
        .boxed();
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[4], "Vector_FixedLengthStringUTF8.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(&"Time channel".to_string()) {
            assert_eq!(
                PrimitiveArray::new(
                    DataType::Float64,
                    Buffer::from([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.].to_vec()),
                    None
                )
                .boxed(),
                data
            );
        }
        if let Some(data) = mdf.get_channel_data(&"Data channel".to_string()) {
            assert_eq!(expected_string_result, data);
        }
        let mut mdf2 = mdf.write(&writing_mdf_file, false)?;
        mdf2.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf2.get_channel_data(&"Data channel".to_string()) {
            assert_eq!(expected_string_result, data);
        }
        //UTF16
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[4], "Vector_FixedLengthStringUTF16_BE.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(&"Data channel".to_string()) {
            assert_eq!(expected_string_result, data);
        }
        let mut mdf2 = mdf.write(&writing_mdf_file, false)?;
        mdf2.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf2.get_channel_data(&"Data channel".to_string()) {
            assert_eq!(expected_string_result, data);
        }
        //SBC
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[4], "Vector_FixedLengthStringSBC.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(&"Data channel".to_string()) {
            assert_eq!(expected_string_result, data);
        }
        let mut mdf2 = mdf.write(&writing_mdf_file, false)?;
        mdf2.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf2.get_channel_data(&"Data channel".to_string()) {
            assert_eq!(expected_string_result, data);
        }
        // byteArray testing
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[0], "Vector_ByteArrayFixedLength.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        let byte_array = BinaryArray::<i64>::from([
            Some([255, 255, 255, 255, 255].as_ref()),
            Some([18, 35, 52, 69, 86].as_ref()),
            Some([0, 1, 2, 3, 4].as_ref()),
            Some([4, 3, 2, 1, 0].as_ref()),
            Some([255, 254, 253, 252, 251].as_ref()),
            Some([250, 249, 248, 247, 246].as_ref()),
            Some([245, 244, 243, 242, 241].as_ref()),
            Some([240, 239, 238, 237, 236].as_ref()),
            Some([235, 234, 233, 232, 231].as_ref()),
            Some([255, 255, 255, 255, 255].as_ref()),
        ])
        .boxed();
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(&"Time channel".to_string()) {
            assert_eq!(
                PrimitiveArray::new(
                    DataType::Float64,
                    Buffer::from([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.].to_vec()),
                    None
                )
                .boxed(),
                data
            );
        }
        if let Some(data) = mdf.get_channel_data(&"Data channel".to_string()) {
            assert_eq!(byte_array, *data);
        }
        let mut mdf2 = mdf.write(&writing_mdf_file, false)?;
        mdf2.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf2.get_channel_data(&"Data channel".to_string()) {
            assert_eq!(byte_array, *data);
        }

        // Integer testing
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[2], "Vector_IntegerTypes.MF4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        let mut vect: Vec<i64> = vec![100; 201];
        let mut counter: i64 = 0;
        vect.iter_mut().for_each(|v| {
            *v -= counter;
            counter += 1
        });
        if let Some(data) = mdf.get_channel_data(&"Counter_INT64_BE".to_string()) {
            assert_eq!(
                PrimitiveArray::new(DataType::Int64, Buffer::from(vect.clone()), None).boxed(),
                data
            );
        }
        let mut mdf2 = mdf.write(&writing_mdf_file, false)?;
        mdf2.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf2.get_channel_data(&"Counter_INT64_BE".to_string()) {
            assert_eq!(
                PrimitiveArray::new(DataType::Int64, Buffer::from(vect), None).boxed(),
                data
            );
        }
        let mut vect: Vec<i32> = vec![100; 201];
        let mut counter: i32 = 0;
        vect.iter_mut().for_each(|v| {
            *v -= counter;
            counter += 1
        });
        if let Some(data) = mdf.get_channel_data(&"Counter_INT32_LE".to_string()) {
            assert_eq!(
                PrimitiveArray::new(DataType::Int32, Buffer::from(vect.clone()), None).boxed(),
                data
            );
        }
        if let Some(data) = mdf2.get_channel_data(&"Counter_INT32_LE".to_string()) {
            assert_eq!(
                PrimitiveArray::new(DataType::Int32, Buffer::from(vect), None).boxed(),
                data
            );
        }
        let mut vect: Vec<i16> = vec![100; 201];
        let mut counter: i16 = 0;
        vect.iter_mut().for_each(|v| {
            *v -= counter;
            counter += 1
        });
        if let Some(data) = mdf.get_channel_data(&"Counter_INT16_LE".to_string()) {
            assert_eq!(
                PrimitiveArray::new(DataType::Int16, Buffer::from(vect.clone()), None).boxed(),
                data
            );
        }
        if let Some(data) = mdf2.get_channel_data(&"Counter_INT16_LE".to_string()) {
            assert_eq!(
                PrimitiveArray::new(DataType::Int16, Buffer::from(vect), None).boxed(),
                data
            );
        }
        Ok(())
    }

    #[test]
    fn channel_types() -> Result<()> {
        let list_of_paths = [
            "ChannelTypes/MasterChannels/".to_string(),
            "ChannelTypes/MLSD/".to_string(),
            "ChannelTypes/VLSD/".to_string(),
            "ChannelTypes/VirtualData/".to_string(),
            "ChannelTypes/Synchronization/".to_string(),
        ];

        // MasterTypes testing
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[0], "Vector_VirtualTimeMasterChannel.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(&"Time channel".to_string()) {
            let mut vect: Vec<f64> = vec![0.; 101];
            let mut counter: f64 = 0.;
            vect.iter_mut().for_each(|v| {
                *v = counter.clone() * 0.03;
                counter += 1.
            });
            assert_eq!(
                PrimitiveArray::new(DataType::Float64, Buffer::from(vect), None).boxed(),
                data
            );
        }
        // MLSD testing
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[1], "Vector_MLSDStringUTF8.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        let expected_string_result = Utf8Array::<i64>::from([
            Some("zero"),
            Some("one"),
            Some("two"),
            Some("three"),
            Some("four"),
            Some("five"),
            Some("six"),
            Some("seven"),
            Some("eight"),
            Some("nine"),
        ])
        .boxed();
        if let Some(data) = mdf.get_channel_data(&"Data channel".to_string()) {
            assert_eq!(expected_string_result, data);
        }
        // Virtual data testing
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[3], "Vector_VirtualDataChannelNoConversion.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        let mut vect: Vec<u64> = vec![0; 200];
        let mut counter: u64 = 0;
        vect.iter_mut().for_each(|v| {
            *v += counter;
            counter += 1
        });
        //if let Some(data) = mdf.get_channel_data(&"Data channel".to_string()) {
        //    assert_eq!(ChannelData::UInt64(vect), *data);
        //}
        // VLSD testing
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[2], "Vector_VLSDStringUTF8.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(&"Data channel".to_string()) {
            assert_eq!(expected_string_result, data);
        }
        // Synchronization
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[4], "Vector_SyncStreamChannel.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        Ok(())
    }
    #[test]
    fn record_layout() -> Result<()> {
        // Overlapping signals
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4, "RecordLayout/Vector_NotByteAligned.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(&"Channel B".to_string()) {
            let mut vect: Vec<u64> = vec![0; 30];
            let mut counter: u64 = 0;
            vect.iter_mut().for_each(|v| {
                *v += counter;
                counter += 1
            });
            assert_eq!(
                PrimitiveArray::new(DataType::UInt64, Buffer::from(vect), None).boxed(),
                data
            );
        }

        // Overlapping signals
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4, "RecordLayout/Vector_OverlappingSignals.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        Ok(())
    }
    #[test]
    fn data_list() -> Result<()> {
        // Equal length
        let file_name = format!("{}{}", BASE_PATH_MDF4, "DataList/Vector_DT_EqualLen.MF4");
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(&"channel1".to_string()) {
            assert_eq!(data.len(), 254552);
        }
        // Equal length
        let file_name = format!("{}{}", BASE_PATH_MDF4, "DataList/Vector_DL_Linked_List.MF4");
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(&"channel1".to_string()) {
            assert_eq!(data.len(), 254552);
        }

        // Empty data
        let file_name = format!("{}{}", BASE_PATH_MDF4, "DataList/ETAS_EmptyDL.mf4");
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;

        // SD List
        let file_name = format!("{}{}", BASE_PATH_MDF4, "DataList/Vector_SD_List.MF4");
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        Ok(())
    }
    #[test]
    fn compressed_data() -> Result<()> {
        // Single DZ deflate
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4, "CompressedData/Simple/Vector_SingleDZ_Deflate.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;

        // Single DZ transpose deflate
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4, "CompressedData/Simple/Vector_SingleDZ_TransposeDeflate.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;

        // deflate data list
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4, "CompressedData/DataList/Vector_DataList_Deflate.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(&"Time channel".to_string()) {
            let mut vect: Vec<f64> = vec![0.; 10000];
            let mut counter: u64 = 0;
            vect.iter_mut().for_each(|v| {
                *v = (counter.clone() as f64) / 10.0;
                counter += 1;
            });
            assert_eq!(
                PrimitiveArray::new(DataType::Float64, Buffer::from(vect), None).boxed(),
                data
            );
        }

        // transpose deflate data list
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4, "CompressedData/DataList/Vector_DataList_TransposeDeflate.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(&"Time channel".to_string()) {
            let mut vect: Vec<f64> = vec![0.; 10000];
            let mut counter: u64 = 0;
            vect.iter_mut().for_each(|v| {
                *v = (counter.clone() as f64) / 10.0;
                counter += 1;
            });
            assert_eq!(
                PrimitiveArray::new(DataType::Float64, Buffer::from(vect), None).boxed(),
                data
            );
        }

        // Unsorted
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4, "CompressedData/Unsorted/Vector_SingleDZ_Unsorted.MF4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        Ok(())
    }

    #[test]
    fn unsorted_data() -> Result<()> {
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4, "UnsortedData/Vector_Unsorted_VLSD.MF4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        Ok(())
    }
    #[test]
    fn conversion() -> Result<()> {
        let list_of_paths = [
            "Conversion/LinearConversion/".to_string(),
            "Conversion/LookUpConversion/".to_string(),
            "Conversion/PartialConversion/".to_string(),
            "Conversion/RationalConversion/".to_string(),
            "Conversion/StringConversion/".to_string(),
            "Conversion/TextConversion/".to_string(),
        ];

        // Lindear conversion testing
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[0], "Vector_LinearConversion.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(&"Data channel".to_string()) {
            let mut vect: Vec<f64> = vec![0.; 10];
            let mut counter: f64 = 0.;
            vect.iter_mut().for_each(|v| {
                *v = counter.clone() * -3.2 - 4.8;
                counter += 1.
            });
            assert_eq!(
                PrimitiveArray::new(DataType::Float64, Buffer::from(vect), None).boxed(),
                data
            );
        }
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[0], "Vector_LinearConversionFactor0.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(&"Data channel".to_string()) {
            let vect: Vec<f64> = vec![3.; 10];
            assert_eq!(
                PrimitiveArray::new(DataType::Float64, Buffer::from(vect), None).boxed(),
                data
            );
        }
        // Rational conversion
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[3], "Vector_RationalConversionIntParams.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;

        // Text conversion
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[5], "Vector_AlgebraicConversionQuadratic.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(&"Data channel".to_string()) {
            let vect = Vec::from([1., 2., 5., 10., 17., 26., 37., 50., 65., 82.]);
            assert_eq!(
                PrimitiveArray::new(DataType::Float64, Buffer::from(vect), None).boxed(),
                data
            );
        }

        // Lookup conversion : Value to Value Table With Interpolation
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[1], "Vector_Value2ValueConversionInterpolation.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(&"Data channel".to_string()) {
            let vect = Vec::from([
                -5.,
                -5.,
                -5.,
                -5.,
                -4.5,
                -4.,
                -3.5,
                -3.,
                -2.5,
                -2.,
                -4. / 3.,
                -2. / 3.,
                0.,
                1. / 3.,
                2. / 3.,
                1.,
                1.5,
                2.,
                1.,
                0.,
                1.5,
                3.,
                4.5,
                6.,
                4.5,
                3.,
                1.5,
                0.,
                0.,
                0.,
            ]);
            assert_eq!(
                PrimitiveArray::new(DataType::Float64, Buffer::from(vect), None).boxed(),
                data
            );
        }

        // Lookup conversion : Value to Value Table Without Interpolation
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[1], "Vector_Value2ValueConversionNoInterpolation.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(&"Data channel".to_string()) {
            let vect = Vec::from([
                -5., -5., -5., -5., -5., -5., -5., -2., -2., -2., -2., 0., 0., 0., 1., 1., 1., 2.,
                2., 0., 0., 3., 3., 6., 6., 3., 3., 0., 0., 0.,
            ]);
            assert_eq!(
                PrimitiveArray::new(DataType::Float64, Buffer::from(vect), None).boxed(),
                data
            );
        }

        // Lookup conversion : Value Range to Value
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[1], "Vector_ValueRange2ValueConversion.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(&"Data channel".to_string()) {
            let vect = Vec::from([
                -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0,
                2.0, 3.0, 3.0, 5.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 9.0, 9.0, 9.0,
            ]);
            assert_eq!(
                PrimitiveArray::new(DataType::Float64, Buffer::from(vect), None).boxed(),
                data
            );
        }

        // Lookup conversion : Value to Text
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[1], "Vector_Value2TextConversion.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(&"Data channel".to_string()) {
            let target = Utf8Array::<i64>::from([
                Some("No match"),
                Some("first gear"),
                Some("second gear"),
                Some("third gear"),
                Some("fourth gear"),
                Some("fifth gear"),
                Some("No match"),
                Some("No match"),
                Some("No match"),
                Some("No match"),
            ])
            .boxed();
            assert_eq!(target, data);
        }

        // Lookup conversion : Value range to Text
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[1], "Vector_ValueRange2TextConversion.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(&"Data channel".to_string()) {
            let target = Utf8Array::<i64>::from([
                Some("Out of range"),
                Some("very low"),
                Some("very low"),
                Some("very low"),
                Some("low"),
                Some("low"),
                Some("medium"),
                Some("medium"),
                Some("high"),
                Some("high"),
            ])
            .boxed();
            assert_eq!(target, data);
        }

        // Lookup conversion : Value range to Text,
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[2], "Vector_StatusStringTableConversionAlgebraic.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(&"Data channel".to_string()) {
            let data = data.as_any().downcast_ref::<Utf8Array<i64>>().expect("");
            let mut vect: Vec<f64> = vec![0.; 300];
            let mut counter: f64 = 0.;
            vect.iter_mut().for_each(|v| {
                *v = counter.clone();
                counter += 0.1
            });
            let mut target = MutableUtf8Array::<i64>::with_capacity(vect.len());
            vect.iter().for_each(|v| {
                if 9.9999 <= *v && *v <= 10.1001 {
                    target.push(Some("Illegal value".to_string()))
                } else if 20.0 <= *v && *v <= 30.0 {
                    target.push(Some("Out of range".to_string()))
                } else {
                    target.push(Some((10.0 / (v - 10.0)).to_string()))
                }
            });
            let target: Utf8Array<i64> = target.into();
            assert_eq!(target.value(0), data.value(0));
            assert_eq!(target.value(299), data.value(299));
            assert_eq!(target.value(101), data.value(101));
        }

        // Text conversion : Text to Value
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[4], "Vector_Text2ValueConversion.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(&"Data channel".to_string()) {
            let vect = Vec::from([-50., 1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            assert_eq!(
                PrimitiveArray::new(DataType::Float64, Buffer::from(vect), None).boxed(),
                data
            );
        }

        // Text conversion : Text to Text
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[4], "Vector_Text2TextConversion.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(&"Data channel".to_string()) {
            let target = Utf8Array::<i64>::from([
                Some("No translation"),
                Some("Eins"),
                Some("Zwei"),
                Some("Drei"),
                Some("Vier"),
                Some("Fünf"),
                Some("Sechs"),
                Some("Sieben"),
                Some("Acht"),
                Some("Neun"),
            ])
            .boxed();
            assert_eq!(target, data);
        }
        Ok(())
    }

    #[test]
    fn bus_logging() -> Result<()> {
        // sort bus
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4, "BusLogging/Vector_CAN_DataFrame_Sort_ID.MF4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(
            &"CAN_DataFrame.ID CAN_DataFrame_101 CANReplay_7_5 Message".to_string(),
        ) {
            let vect: Vec<f64> = vec![101.; 79];
            // assert!(ChannelData::Float64(target).compare_f64(data, 1e-9f64));
            assert_eq!(
                PrimitiveArray::new(DataType::Float64, Buffer::from(vect), None).boxed(),
                *data
            );
        }
        Ok(())
    }
    #[test]
    fn writing_mdf4() -> Result<()> {
        // write file with invalid channels
        let file = format!(
            "{}{}",
            BASE_PATH_MDF4, &"Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4"
        );
        let ref_channel = r"NO";
        let mut mdf = Mdf::new(&file)?;
        mdf.load_all_channels_data_in_memory()?;
        // with compression
        let mut info2 = mdf.write(WRITING_MDF_FILE, true)?;
        info2.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(&ref_channel.to_string()) {
            if let Some(data2) = info2.get_channel_data(&ref_channel.to_string()) {
                assert_eq!(*data2, *data);
            } else {
                panic!("Channel not found");
            }
        } else {
            panic!("Channel not found");
        }
        // without compression
        let mut info2 = mdf.write(WRITING_MDF_FILE, false)?;
        info2.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(&ref_channel.to_string()) {
            if let Some(data2) = info2.get_channel_data(&ref_channel.to_string()) {
                assert_eq!(*data2, *data);
            } else {
                panic!("Channel not found");
            }
        } else {
            panic!("Channel not found");
        }

        // write file with many channels
        let file = format!("{}{}", BASE_PATH_MDF4, &"Simple/test.mf4");
        let ref_channel = r"C90 CG21 in error.mdf";
        let mut mdf = Mdf::new(&file)?;
        mdf.load_all_channels_data_in_memory()?;
        // with compression
        let mut info2 = mdf.write(WRITING_MDF_FILE, true)?;
        info2.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(&ref_channel.to_string()) {
            if let Some(data2) = info2.get_channel_data(&ref_channel.to_string()) {
                assert_eq!(*data2, *data);
            } else {
                panic!("Channel not found");
            }
        } else {
            panic!("Channel not found");
        }
        // without compression
        let mut info2 = mdf.write(WRITING_MDF_FILE, false)?;
        info2.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(&ref_channel.to_string()) {
            if let Some(data2) = info2.get_channel_data(&ref_channel.to_string()) {
                assert_eq!(*data2, *data);
            } else {
                panic!("Channel not found");
            }
        } else {
            panic!("Channel not found");
        }
        // TODO test write file with arrays

        //mdf3 conversion
        drop(mdf);
        let file = format!(
            "{}{}",
            BASE_PATH_MDF3, &"RJ_N16-12-363_BM-15C-0024_228_2_20170116094355_CAN.dat"
        );
        let mut mdf = Mdf::new(&file)?;
        mdf.load_all_channels_data_in_memory()?;
        let channel_name3 = r"TEMP_FUEL";
        let mut mdf4 = mdf.write(WRITING_MDF_FILE, true)?;
        mdf4.load_all_channels_data_in_memory()?;
        let mdf3_data = mdf.get_channel_data(&channel_name3);
        let mdf4_data = mdf4.get_channel_data(&channel_name3);
        assert_eq!(mdf3_data, mdf4_data);
        Ok(())
    }
    #[test]
    fn mdf_modifications() -> Result<()> {
        // write file with invalid channels
        let file = format!(
            "{}{}",
            BASE_PATH_MDF4, &"Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4"
        );
        let ref_channel = r"PANS";
        let ref_desc = r"tralala";
        let ref_unit = r"Bar";
        let mut mdf = Mdf::new(&file)?;
        mdf.load_all_channels_data_in_memory()?;
        // modify data
        if let Some(data) = mdf.get_channel_data(&ref_channel.to_string()) {
            let array = data
                .as_any()
                .downcast_ref::<PrimitiveArray<f32>>()
                .expect("could not downcast to f32 array");
            let mut new_data = MutablePrimitiveArray::<f32>::from_trusted_len_values_iter(
                array.values_iter().copied(),
            );
            new_data.set(0, Some(0.0f32));
            mdf.set_channel_data(&ref_channel.to_string(), new_data.as_box());
            mdf.set_channel_desc(&ref_channel.to_string(), ref_desc);
            mdf.set_channel_unit(&ref_channel.to_string(), ref_unit);
            mdf.set_channel_master_type(&ref_channel.to_string(), 1);
        } else {
            panic!("channel not found");
        }
        if let Some(data) = mdf.get_channel_data(&ref_channel.to_string()) {
            let array = data
                .as_any()
                .downcast_ref::<PrimitiveArray<f32>>()
                .expect("could not downcast to f32 array");
            let minimum = min_primitive(&array);
            if let Some(min) = minimum {
                assert!(min < 1000.0f32);
            }
        } else {
            panic!("channel not found");
        }
        if let Ok(Some(desc)) = mdf.get_channel_desc(&ref_channel.to_string()) {
            assert_eq!(desc, ref_desc);
        } else {
            panic!("channel not found");
        }
        if let Ok(Some(unit)) = mdf.get_channel_unit(&ref_channel.to_string()) {
            assert_eq!(unit, ref_unit);
        } else {
            panic!("channel not found");
        }
        assert_eq!(mdf.get_channel_master_type(&ref_channel.to_string()), 1);

        // add new channel
        drop(mdf);
        let mut mdf = Mdf::new(&file)?;
        mdf.load_all_channels_data_in_memory()?;
        let channel_name = r"Fake_name".to_string();
        let new_channel_name = r"New fake_name".to_string();
        let new_data =
            PrimitiveArray::new(DataType::Float64, Buffer::from(vec![0f64; 3300]), None).boxed();
        let master_channel = mdf.get_channel_master(&ref_channel.to_string());
        let master_type = Some(0);
        let master_flag = false;
        let unit = Some(ref_unit.to_string());
        let desc = Some(ref_desc.to_string());
        mdf.add_channel(
            channel_name.clone(),
            new_data,
            master_channel,
            master_type,
            master_flag,
            unit,
            desc,
        );

        if let Some(data) = mdf.get_channel_data(&channel_name.to_string()) {
            let array = data
                .as_any()
                .downcast_ref::<PrimitiveArray<f64>>()
                .expect("could not downcast to f64 array");
            let minimum = min_primitive(&array);
            if let Some(min) = minimum {
                assert!(min == 0.0f64);
            }
        } else {
            panic!("channel not found");
        }
        if let Ok(Some(desc)) = mdf.get_channel_desc(&channel_name.to_string()) {
            assert_eq!(desc, ref_desc.to_string());
        } else {
            panic!("channel not found");
        }
        if let Ok(Some(unit)) = mdf.get_channel_unit(&channel_name.to_string()) {
            assert_eq!(unit, ref_unit);
        } else {
            panic!("channel not found");
        }
        assert_eq!(mdf.get_channel_master_type(&channel_name.to_string()), 0);

        //rename
        assert!(mdf.get_channel_data(&channel_name.to_string()).is_some());
        mdf.rename_channel(&channel_name.to_string(), &new_channel_name);
        assert!(mdf.get_channel_data(&channel_name.to_string()).is_none());

        //remove
        assert!(mdf
            .get_channel_data(&new_channel_name.to_string())
            .is_some());
        mdf.remove_channel(&new_channel_name);
        assert!(mdf
            .get_channel_data(&new_channel_name.to_string())
            .is_none());
        Ok(())
    }
    #[test]
    fn export_to_parquet() -> Result<()> {
        // Export to Parquet file
        let file = format!(
            "{}{}",
            BASE_PATH_MDF3, &"RJ_N16-12-363_BM-15C-0024_228_2_20170116094355_CAN.dat"
        );
        let mut mdf = Mdf::new(&file)?;
        mdf.load_all_channels_data_in_memory()?;
        mdf.export_to_parquet(&WRITING_PARQUET_FILE, Some("snappy"))
            .expect("failed writing parquet file");
        Ok(())
    }
}
