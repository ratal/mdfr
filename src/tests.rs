#[cfg(test)]
mod tests {
    use crate::mdfinfo::MdfInfo;
    use crate::mdfreader::channel_data::ChannelData;
    use ndarray::array;
    use ndarray::Array1;
    use ndarray_stats::QuantileExt;
    use std::fs;
    use std::io;
    use std::path::Path;
    use std::vec::Vec;

    static BASE_PATH_MDF4: &str = "/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/";
    static BASE_PATH_MDF3: &str = "/home/ratal/workspace/mdfreader/mdfreader/tests/mdf3/";
    static WRITING_FILE: &str = "/home/ratal/workspace/mdfr/test.mf4";

    #[test]
    fn info_test() -> io::Result<()> {
        let mut file_name = "test_files/test_basic.mf4";
        println!("reading {}", file_name);
        let mut info = MdfInfo::new(file_name);
        println!("{:#?}", info);
        assert_eq!(info.get_version(), 410);
        file_name = "test_files/test_mdf3.mdf";
        println!("reading {}", file_name);
        let mut info = MdfInfo::new(file_name);
        println!("{:#?}", &info);
        assert_eq!(info.get_version(), 310);
        file_name = "test_files/test_mdf4.mf4";
        println!("reading {}", file_name);
        let mut info = MdfInfo::new(file_name);
        println!("{:#?}", &info);
        assert_eq!(info.get_version(), 400);
        Ok(())
    }

    fn parse_info_folder(folder: &String) -> io::Result<()> {
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
                                    let mut info = MdfInfo::new(file_name);
                                    info.load_all_channels_data_in_memory();
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
    fn parse_file() {
        let file = format!(
            "{}{}",
            BASE_PATH_MDF4, &"Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4"
        );
        let mut mdf = MdfInfo::new(&file);
        mdf.load_all_channels_data_in_memory();
        mdf.write(WRITING_FILE, true);
    }

    #[test]
    fn data_types() {
        let list_of_paths = [
            "DataTypes/ByteArray/".to_string(),
            "DataTypes/CANopenTypes/".to_string(),
            "DataTypes/IntegerTypes/".to_string(),
            "DataTypes/RealTypes/".to_string(),
            "DataTypes/StringTypes/".to_string(),
        ];

        // StringTypes testing
        // UTF8
        let expected_string_result: Vec<String> = vec![
            "zero".to_string(),
            "one".to_string(),
            "two".to_string(),
            "three".to_string(),
            "four".to_string(),
            "five".to_string(),
            "six".to_string(),
            "seven".to_string(),
            "eight".to_string(),
            "nine".to_string(),
        ];
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[4], "Vector_FixedLengthStringUTF8.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info.get_channel_data(&"Time channel".to_string()) {
            assert_eq!(
                ChannelData::Float64(array![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                *data
            );
        }
        if let (Some(data), Some(_mask)) = info.get_channel_data(&"Data channel".to_string()) {
            assert_eq!(
                ChannelData::StringUTF8(expected_string_result.clone()),
                *data
            );
        }
        let mut info2 = info.write(WRITING_FILE, false);
        info2.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info2.get_channel_data(&"Data channel".to_string()) {
            assert_eq!(
                ChannelData::StringUTF8(expected_string_result.clone()),
                *data
            );
        }
        //UTF16
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[4], "Vector_FixedLengthStringUTF16_BE.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info.get_channel_data(&"Data channel".to_string()) {
            assert_eq!(
                ChannelData::StringUTF16(expected_string_result.clone()),
                *data
            );
        }
        let mut info2 = info.write(WRITING_FILE, false);
        info2.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info2.get_channel_data(&"Data channel".to_string()) {
            assert_eq!(
                ChannelData::StringUTF8(expected_string_result.clone()),
                *data
            );
        }
        //SBC
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[4], "Vector_FixedLengthStringSBC.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info.get_channel_data(&"Data channel".to_string()) {
            assert_eq!(
                ChannelData::StringSBC(expected_string_result.clone()),
                *data
            );
        }
        let mut info2 = info.write(WRITING_FILE, false);
        info2.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info2.get_channel_data(&"Data channel".to_string()) {
            assert_eq!(
                ChannelData::StringUTF8(expected_string_result.clone()),
                *data
            );
        }
        // byteArray testing
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[0], "Vector_ByteArrayFixedLength.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        let byte_array = ChannelData::ByteArray(vec![
            vec![255, 255, 255, 255, 255],
            vec![18, 35, 52, 69, 86],
            vec![0, 1, 2, 3, 4],
            vec![4, 3, 2, 1, 0],
            vec![255, 254, 253, 252, 251],
            vec![250, 249, 248, 247, 246],
            vec![245, 244, 243, 242, 241],
            vec![240, 239, 238, 237, 236],
            vec![235, 234, 233, 232, 231],
            vec![255, 255, 255, 255, 255],
        ]);
        info.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info.get_channel_data(&"Time channel".to_string()) {
            assert_eq!(
                ChannelData::Float64(array![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                *data
            );
        }
        if let (Some(data), Some(_mask)) = info.get_channel_data(&"Data channel".to_string()) {
            assert_eq!(byte_array, *data);
        }
        let mut info2 = info.write(WRITING_FILE, false);
        info2.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info2.get_channel_data(&"Data channel".to_string()) {
            assert_eq!(byte_array, *data);
        }

        // Integer testing
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[2], "Vector_IntegerTypes.MF4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        let mut vect: Vec<i64> = vec![100; 201];
        let mut counter: i64 = 0;
        vect.iter_mut().for_each(|v| {
            *v -= counter;
            counter += 1
        });
        if let (Some(data), Some(_mask)) = info.get_channel_data(&"Counter_INT64_BE".to_string()) {
            assert_eq!(
                ChannelData::Int64(Array1::<i64>::from_vec(vect.clone())),
                *data
            );
        }
        let mut info2 = info.write(WRITING_FILE, false);
        info2.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info2.get_channel_data(&"Counter_INT64_BE".to_string()) {
            assert_eq!(ChannelData::Int64(Array1::<i64>::from_vec(vect)), *data);
        }
        let mut vect: Vec<i32> = vec![100; 201];
        let mut counter: i32 = 0;
        vect.iter_mut().for_each(|v| {
            *v -= counter;
            counter += 1
        });
        if let (Some(data), Some(_mask)) = info.get_channel_data(&"Counter_INT32_LE".to_string()) {
            assert_eq!(
                ChannelData::Int32(Array1::<i32>::from_vec(vect.clone())),
                *data
            );
        }
        if let (Some(data), Some(_mask)) = info2.get_channel_data(&"Counter_INT32_LE".to_string()) {
            assert_eq!(ChannelData::Int32(Array1::<i32>::from_vec(vect)), *data);
        }
        let mut vect: Vec<i16> = vec![100; 201];
        let mut counter: i16 = 0;
        vect.iter_mut().for_each(|v| {
            *v -= counter;
            counter += 1
        });
        if let (Some(data), Some(_mask)) = info.get_channel_data(&"Counter_INT16_LE".to_string()) {
            assert_eq!(
                ChannelData::Int16(Array1::<i16>::from_vec(vect.clone())),
                *data
            );
        }
        if let (Some(data), Some(_mask)) = info2.get_channel_data(&"Counter_INT16_LE".to_string()) {
            assert_eq!(ChannelData::Int16(Array1::<i16>::from_vec(vect)), *data);
        }
    }

    #[test]
    fn channel_types() {
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
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info.get_channel_data(&"Time channel".to_string()) {
            let mut vect: Vec<f64> = vec![0.; 101];
            let mut counter: f64 = 0.;
            vect.iter_mut().for_each(|v| {
                *v = counter.clone() * 0.03;
                counter += 1.
            });
            let target = Array1::<f64>::from_vec(vect);
            assert_eq!(ChannelData::Float64(target), *data);
        }
        // MLSD testing
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[1], "Vector_MLSDStringUTF8.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info.get_channel_data(&"Data channel".to_string()) {
            let expected_string_result: Vec<String> = vec![
                "zero".to_string(),
                "one".to_string(),
                "two".to_string(),
                "three".to_string(),
                "four".to_string(),
                "five".to_string(),
                "six".to_string(),
                "seven".to_string(),
                "eight".to_string(),
                "nine".to_string(),
            ];
            assert_eq!(ChannelData::StringUTF8(expected_string_result), *data);
        }
        // Virtual data testing
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[3], "Vector_VirtualDataChannelNoConversion.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        let mut vect: Vec<u64> = vec![0; 200];
        let mut counter: u64 = 0;
        vect.iter_mut().for_each(|v| {
            *v += counter;
            counter += 1
        });
        if let (Some(data), Some(_mask)) = info.get_channel_data(&"Data channel".to_string()) {
            assert_eq!(ChannelData::UInt64(Array1::<u64>::from_vec(vect)), *data);
        }
        // VLSD testing
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[2], "Vector_VLSDStringUTF8.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info.get_channel_data(&"Data channel".to_string()) {
            let expected_string_result: Vec<String> = vec![
                "zero".to_string(),
                "one".to_string(),
                "two".to_string(),
                "three".to_string(),
                "four".to_string(),
                "five".to_string(),
                "six".to_string(),
                "seven".to_string(),
                "eight".to_string(),
                "nine".to_string(),
            ];
            assert_eq!(ChannelData::StringUTF8(expected_string_result), *data);
        }
        // Synchronization
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[4], "Vector_SyncStreamChannel.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
    }
    #[test]
    fn record_layout() {
        // Overlapping signals
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4, "RecordLayout/Vector_NotByteAligned.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info.get_channel_data(&"Channel B".to_string()) {
            let mut vect: Vec<u64> = vec![0; 30];
            let mut counter: u64 = 0;
            vect.iter_mut().for_each(|v| {
                *v += counter;
                counter += 1
            });
            assert_eq!(ChannelData::UInt48(Array1::<u64>::from_vec(vect)), *data);
        }

        // Overlapping signals
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4, "RecordLayout/Vector_OverlappingSignals.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
    }
    #[test]
    fn data_list() {
        // Equal length
        let file_name = format!("{}{}", BASE_PATH_MDF4, "DataList/Vector_DT_EqualLen.MF4");
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info.get_channel_data(&"channel1".to_string()) {
            assert_eq!(data.len(), 254552);
        }
        // Equal length
        let file_name = format!("{}{}", BASE_PATH_MDF4, "DataList/Vector_DL_Linked_List.MF4");
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info.get_channel_data(&"channel1".to_string()) {
            assert_eq!(data.len(), 254552);
        }

        // Empty data
        let file_name = format!("{}{}", BASE_PATH_MDF4, "DataList/ETAS_EmptyDL.mf4");
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();

        // SD List
        let file_name = format!("{}{}", BASE_PATH_MDF4, "DataList/Vector_SD_List.MF4");
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
    }
    #[test]
    fn compressed_data() {
        // Single DZ deflate
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4, "CompressedData/Simple/Vector_SingleDZ_Deflate.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();

        // Single DZ transpose deflate
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4, "CompressedData/Simple/Vector_SingleDZ_TransposeDeflate.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();

        // deflate data list
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4, "CompressedData/DataList/Vector_DataList_Deflate.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        println!("{}", info);
        if let (Some(data), Some(_mask)) = info.get_channel_data(&"Time channel".to_string()) {
            let mut vect: Vec<f64> = vec![0.; 10000];
            let mut counter: f64 = 0.;
            vect.iter_mut().for_each(|v| {
                *v = counter.clone();
                counter += 0.1
            });
            let target = Array1::<f64>::from_vec(vect);
            println!("{:?}\n {:?}", target, data);
            assert!(ChannelData::Float64(target).compare_f64(data, 1e-9f64));
        }

        // transpose deflate data list
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4, "CompressedData/DataList/Vector_DataList_TransposeDeflate.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info.get_channel_data(&"Time channel".to_string()) {
            let mut vect: Vec<f64> = vec![0.; 10000];
            let mut counter: f64 = 0.;
            vect.iter_mut().for_each(|v| {
                *v = counter.clone();
                counter += 0.1
            });
            let target = Array1::<f64>::from_vec(vect);
            assert!(ChannelData::Float64(target).compare_f64(data, 1e-9f64));
        }

        // Unsorted
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4, "CompressedData/Unsorted/Vector_SingleDZ_Unsorted.MF4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
    }

    #[test]
    fn unsorted_data() {
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4, "UnsortedData/Vector_Unsorted_VLSD.MF4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
    }
    #[test]
    fn conversion() {
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
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info.get_channel_data(&"Data channel".to_string()) {
            let mut vect: Vec<f64> = vec![0.; 10];
            let mut counter: f64 = 0.;
            vect.iter_mut().for_each(|v| {
                *v = counter.clone() * -3.2 - 4.8;
                counter += 1.
            });
            let target = Array1::<f64>::from_vec(vect);
            assert!(ChannelData::Float64(target).compare_f64(data, f64::EPSILON));
        }
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[0], "Vector_LinearConversionFactor0.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info.get_channel_data(&"Data channel".to_string()) {
            let vect: Vec<f64> = vec![3.; 10];
            let target = Array1::<f64>::from_vec(vect);
            assert!(ChannelData::Float64(target).compare_f64(data, f64::EPSILON));
        }
        // Rational conversion
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[3], "Vector_RationalConversionIntParams.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();

        // Text conversion
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[5], "Vector_AlgebraicConversionQuadratic.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info.get_channel_data(&"Data channel".to_string()) {
            let vect: [f64; 10] = [1., 2., 5., 10., 17., 26., 37., 50., 65., 82.];
            let target = Array1::<f64>::from_vec(vect.to_vec());
            assert!(ChannelData::Float64(target).compare_f64(data, f64::EPSILON));
        }

        // Lookup conversion : Value to Value Table With Interpolation
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[1], "Vector_Value2ValueConversionInterpolation.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info.get_channel_data(&"Data channel".to_string()) {
            let vect: [f64; 30] = [
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
            ];
            let target = Array1::<f64>::from_vec(vect.to_vec());
            assert!(ChannelData::Float64(target).compare_f64(data, f64::EPSILON));
        }

        // Lookup conversion : Value to Value Table Without Interpolation
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[1], "Vector_Value2ValueConversionNoInterpolation.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info.get_channel_data(&"Data channel".to_string()) {
            let vect: [f64; 30] = [
                -5., -5., -5., -5., -5., -5., -5., -2., -2., -2., -2., 0., 0., 0., 1., 1., 1., 2.,
                2., 0., 0., 3., 3., 6., 6., 3., 3., 0., 0., 0.,
            ];
            let target = Array1::<f64>::from_vec(vect.to_vec());
            assert!(ChannelData::Float64(target).compare_f64(data, f64::EPSILON));
        }

        // Lookup conversion : Value Range to Value
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[1], "Vector_ValueRange2ValueConversion.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info.get_channel_data(&"Data channel".to_string()) {
            let vect: [f64; 30] = [
                -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0,
                2.0, 3.0, 3.0, 5.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 9.0, 9.0, 9.0,
            ];
            let target = Array1::<f64>::from_vec(vect.to_vec());
            assert!(ChannelData::Float64(target).compare_f64(data, f64::EPSILON));
        }

        // Lookup conversion : Value to Text
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[1], "Vector_Value2TextConversion.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info.get_channel_data(&"Data channel".to_string()) {
            let target: Vec<String> = [
                "No match".to_string(),
                "first gear".to_string(),
                "second gear".to_string(),
                "third gear".to_string(),
                "fourth gear".to_string(),
                "fifth gear".to_string(),
                "No match".to_string(),
                "No match".to_string(),
                "No match".to_string(),
                "No match".to_string(),
            ]
            .to_vec();
            assert_eq!(ChannelData::StringUTF8(target), *data);
        }

        // Lookup conversion : Value range to Text
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[1], "Vector_ValueRange2TextConversion.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info.get_channel_data(&"Data channel".to_string()) {
            let target: Vec<String> = [
                "Out of range".to_string(),
                "very low".to_string(),
                "very low".to_string(),
                "very low".to_string(),
                "low".to_string(),
                "low".to_string(),
                "medium".to_string(),
                "medium".to_string(),
                "high".to_string(),
                "high".to_string(),
            ]
            .to_vec();
            assert_eq!(ChannelData::StringUTF8(target), *data);
        }

        // Lookup conversion : Value range to Text,
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[2], "Vector_StatusStringTableConversionAlgebraic.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info.get_channel_data(&"Data channel".to_string()) {
            let mut vect: Vec<f64> = vec![0.; 300];
            let mut counter: f64 = 0.;
            vect.iter_mut().for_each(|v| {
                *v = counter.clone();
                counter += 0.1
            });
            let target = vect
                .iter()
                .map(|v| {
                    if 9.9999 <= *v && *v <= 10.1001 {
                        "Illegal value".to_string()
                    } else if 20.0 <= *v && *v <= 30.0 {
                        "Out of range".to_string()
                    } else {
                        (10.0 / (v - 10.0)).to_string()
                    }
                })
                .collect::<Vec<String>>();
            // println!("{:?} {}", target, target.len());
            // println!("{} {}", data, data.len());
            // assert!(ChannelData::StringUTF8(target.clone()).compare_f64(data, 1e-6f64));
        }

        // Text conversion : Text to Value
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[4], "Vector_Text2ValueConversion.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info.get_channel_data(&"Data channel".to_string()) {
            let vect: [f64; 10] = [-50., 1., 2., 3., 4., 5., 6., 7., 8., 9.];
            let target = Array1::<f64>::from_vec(vect.to_vec());
            assert_eq!(ChannelData::Float64(target), *data);
        }

        // Text conversion : Text to Text
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[4], "Vector_Text2TextConversion.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info.get_channel_data(&"Data channel".to_string()) {
            let target: Vec<String> = [
                "No translation".to_string(),
                "Eins".to_string(),
                "Zwei".to_string(),
                "Drei".to_string(),
                "Vier".to_string(),
                "Fünf".to_string(),
                "Sechs".to_string(),
                "Sieben".to_string(),
                "Acht".to_string(),
                "Neun".to_string(),
            ]
            .to_vec();
            assert_eq!(ChannelData::StringUTF8(target), *data);
        }
    }

    #[test]
    fn bus_logging() {
        // sort bus
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4, "BusLogging/Vector_CAN_DataFrame_Sort_ID.MF4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info.get_channel_data(
            &"CAN_DataFrame.ID CAN_DataFrame_101 CANReplay_7_5 Message".to_string(),
        ) {
            let vect: Vec<f64> = vec![101.; 79];
            let target = Array1::<f64>::from_vec(vect);
            // assert!(ChannelData::Float64(target).compare_f64(data, 1e-9f64));
            assert_eq!(ChannelData::Float64(target), *data);
        }
    }
    #[test]
    fn writing_mdf4() {
        // write file with invalid channels
        let file = format!(
            "{}{}",
            BASE_PATH_MDF4, &"Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4"
        );
        let ref_channel = r"recorder_time !P";
        let mut info = MdfInfo::new(&file);
        info.load_all_channels_data_in_memory();
        // with compression
        let mut info2 = info.write(WRITING_FILE, true);
        info2.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info.get_channel_data(&ref_channel.to_string()) {
            if let (Some(data2), Some(_mask)) = info2.get_channel_data(&ref_channel.to_string()) {
                assert_eq!(*data2, *data);
            }
        }
        // without compression
        let mut info2 = info.write(WRITING_FILE, false);
        info2.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info.get_channel_data(&ref_channel.to_string()) {
            if let (Some(data2), Some(_mask)) = info2.get_channel_data(&ref_channel.to_string()) {
                assert_eq!(*data2, *data);
            }
        }

        // write file with many channels
        let file = format!("{}{}", BASE_PATH_MDF4, &"Simple/test.mf4");
        let ref_channel = r"C90 CG21 in error.mdf";
        let mut info = MdfInfo::new(&file);
        info.load_all_channels_data_in_memory();
        // with compression
        let mut info2 = info.write(WRITING_FILE, true);
        info2.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info.get_channel_data(&ref_channel.to_string()) {
            if let (Some(data2), Some(_mask)) = info2.get_channel_data(&ref_channel.to_string()) {
                assert_eq!(*data2, *data);
            }
        }
        // without compression
        let mut info2 = info.write(WRITING_FILE, false);
        info2.load_all_channels_data_in_memory();
        if let (Some(data), Some(_mask)) = info.get_channel_data(&ref_channel.to_string()) {
            if let (Some(data2), Some(_mask)) = info2.get_channel_data(&ref_channel.to_string()) {
                assert_eq!(*data2, *data);
            }
        }
    }
    #[test]
    fn mdf_modifications() {
        // write file with invalid channels
        let file = format!(
            "{}{}",
            BASE_PATH_MDF4, &"Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4"
        );
        let ref_channel = r"PANS";
        let ref_desc = r"tralala";
        let ref_unit = r"Bar";
        let mut info = MdfInfo::new(&file);
        info.load_all_channels_data_in_memory();
        // modify data
        if let (Some(data), Some(_mask)) = info.get_channel_data(&ref_channel.to_string()) {
            if let ChannelData::Float32(mut new_data) = data.clone() {
                new_data[0] = 0.0;
                info.set_channel_data(&ref_channel.to_string(), &ChannelData::Float32(new_data));
                info.set_channel_desc(&ref_channel.to_string(), ref_desc);
                info.set_channel_unit(&ref_channel.to_string(), ref_unit);
                info.set_channel_master_type(&ref_channel.to_string(), 1);
            } else {
                panic!("not correct data type");
            }
        } else {
            panic!("channel not found");
        }
        if let (Some(data), Some(_mask)) = info.get_channel_data(&ref_channel.to_string()) {
            if let ChannelData::Float32(data) = data.clone() {
                assert!(data.min_skipnan() < &1000.0f32); // data was modified
            } else {
                panic!("not correct data type");
            }
        } else {
            panic!("channel not found");
        }
        if let Some(desc) = info.get_channel_desc(&ref_channel.to_string()) {
            assert_eq!(desc, ref_desc);
        } else {
            panic!("channel not found");
        }
        if let Some(unit) = info.get_channel_unit(&ref_channel.to_string()) {
            assert_eq!(unit, ref_unit);
        } else {
            panic!("channel not found");
        }
        assert_eq!(info.get_channel_master_type(&ref_channel.to_string()), 1);

        // add new channel
        drop(info);
        let mut info = MdfInfo::new(&file);
        info.load_all_channels_data_in_memory();
        let channel_name = r"Fake_name".to_string();
        let new_channel_name = r"New fake_name".to_string();
        let new_data = ChannelData::Float64(Array1::<f64>::zeros((3300,)));
        let master_channel = info.get_channel_master(&ref_channel.to_string());
        let master_type = Some(0);
        let master_flag = false;
        let unit = Some(ref_unit.to_string());
        let desc = Some(ref_desc.to_string());
        info.add_channel(
            channel_name.clone(),
            new_data,
            master_channel,
            master_type,
            master_flag,
            unit,
            desc,
        );

        if let (Some(data), _mask) = info.get_channel_data(&channel_name.to_string()) {
            if let ChannelData::Float64(data) = data.clone() {
                assert!(data.min_skipnan() == &0.0f64); // data was modified
            } else {
                panic!("not correct data type");
            }
        } else {
            panic!("channel not found");
        }
        if let Some(desc) = info.get_channel_desc(&channel_name.to_string()) {
            assert_eq!(desc, ref_desc.to_string());
        } else {
            panic!("channel not found");
        }
        if let Some(unit) = info.get_channel_unit(&channel_name.to_string()) {
            assert_eq!(unit, ref_unit);
        } else {
            panic!("channel not found");
        }
        assert_eq!(info.get_channel_master_type(&channel_name.to_string()), 0);

        //rename
        assert!(info.get_channel_data(&channel_name.to_string()).0.is_some());
        info.rename_channel(&channel_name.to_string(), &new_channel_name);
        assert!(info.get_channel_data(&channel_name.to_string()).0.is_none());

        //remove
        assert!(info
            .get_channel_data(&new_channel_name.to_string())
            .0
            .is_some());
        info.remove_channel(&new_channel_name);
        assert!(info
            .get_channel_data(&new_channel_name.to_string())
            .0
            .is_none());

        //mdf3 conversion
        drop(info);
        let file = format!(
            "{}{}",
            BASE_PATH_MDF3, &"RJ_N16-12-363_BM-15C-0024_228_2_20170116094355_CAN.dat"
        );
        let mut info = MdfInfo::new(&file);
        info.load_all_channels_data_in_memory();
        let channel_name3 = r"TEMP_FUEL";
        let mut info2 = info.convert3to4(WRITING_FILE);
        assert_eq!(info2.get_channel_data(&channel_name3), info.get_channel_data(&channel_name3));
    }
}
