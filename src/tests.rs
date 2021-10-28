#[cfg(test)]
mod tests {
    use crate::mdfinfo::MdfInfo;
    use crate::mdfreader::channel_data::ChannelData;
    use ndarray::array;
    use ndarray::Array1;
    use std::fs;
    use std::io;
    use std::path::Path;
    use std::vec::Vec;

    #[test]
    fn info_test() -> io::Result<()> {
        let mut file_name = "/home/ratal/workspace/mdfr/test_files/Test.mf4";
        println!("reading {}", file_name);
        let mut info = MdfInfo::new(file_name);
        println!("{:#?}", info);
        assert_eq!(info.get_version(), 410);
        file_name = "/home/ratal/workspace/mdfr/test_files/Mdf3_hiddenBytes_NotAlignedBytes.dat";
        println!("reading {}", file_name);
        let mut info = MdfInfo::new(file_name);
        println!("{:#?}", &info);
        assert_eq!(info.get_version(), 320);
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
        let base_path = String::from("/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/");
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
            parse_info_folder(&format!("{}{}", &base_path, &path)).unwrap();
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
        // let file = String::from("/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/Simple/error.mf4"); // DT, big many channels
        // let file = String::from("/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/Simple/Measure.mf4"); // DataList, big many channels
        // let file = String::from("/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/Simple/measure2.mf4");  // many cc_ref with value to text
        // let file = String::from("/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/Simple/ZED_1hz_7197.mf4"); // invalid bytes
        // let file = String::from("/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/Simple/20161129_IN-x1234_Erprobungsort_0000032.mf4");
        // let file = String::from("/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/Simple/isse_107.mf4"); // DZ
        // let file = String::from("/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/Simple/Vector_MinimumFile.MF4");  // DT
        // let file = String::from("/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4");  // DT
        // let file = String::from("/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/UnsortedData/Vector_Unsorted_VLSD.MF4"); // unsorted with VLSD
        let file = String::from("/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/Simple/test.mf4");
        // let file = String::from("/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/Simple/ASAP2_Demo_V171.mf4");
        // let file = String::from("/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/Simple/V447_20190514_164254_Gearshift_0m_20Grad.mf4"); // HL DL DZ
        // let file = String::from("/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/ChannelTypes/MLSD/Vector_MLSDStringUTF8.mf4");
        // let file = String::from("/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/RecordLayout/Vector_NotByteAligned.mf4");
        // let file = String::from("/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/DataList/Vector_DT_EqualLen.MF4");
        // let file = String::from("/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/ChannelTypes/VLSD/Vector_VLSDStringUTF16_LE.mf4");
        // let file = String::from("/home/ratal/workspace/mdfreader/T3_121121_000_6NEDC_col.mf4");
        // let file = String::from("/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/Simple/ZED_1hz_7197_col.mf4"); // invalid bytes
        let mut mdf = MdfInfo::new(&file);
        mdf.load_all_channels_data_in_memory();
    }

    #[test]
    fn data_types() {
        let base_path = String::from("/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/");
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
            base_path, list_of_paths[4], "Vector_FixedLengthStringUTF8.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let Some(data) = info.get_channel_data(&"Time channel".to_string()) {
            assert_eq!(
                ChannelData::Float64(array![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                *data
            );
        }
        if let Some(data) = info.get_channel_data(&"Data channel".to_string()) {
            assert_eq!(
                ChannelData::StringUTF8(expected_string_result.clone()),
                *data
            );
        }
        //UTF16
        let file_name = format!(
            "{}{}{}",
            base_path, list_of_paths[4], "Vector_FixedLengthStringUTF16_BE.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let Some(data) = info.get_channel_data(&"Data channel".to_string()) {
            assert_eq!(
                ChannelData::StringUTF16(expected_string_result.clone()),
                *data
            );
        }
        //SBC
        let file_name = format!(
            "{}{}{}",
            base_path, list_of_paths[4], "Vector_FixedLengthStringSBC.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let Some(data) = info.get_channel_data(&"Data channel".to_string()) {
            assert_eq!(ChannelData::StringSBC(expected_string_result), *data);
        }

        // byteArray testing
        let file_name = format!(
            "{}{}{}",
            base_path, list_of_paths[0], "Vector_ByteArrayFixedLength.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let Some(data) = info.get_channel_data(&"Time channel".to_string()) {
            assert_eq!(
                ChannelData::Float64(array![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                *data
            );
        }

        // Integer testing
        let file_name = format!(
            "{}{}{}",
            base_path, list_of_paths[2], "Vector_IntegerTypes.MF4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        let mut vect: Vec<i64> = vec![100; 201];
        let mut counter: i64 = 0;
        vect.iter_mut().for_each(|v| {
            *v -= counter;
            counter += 1
        });
        if let Some(data) = info.get_channel_data(&"Counter_INT64_BE".to_string()) {
            assert_eq!(ChannelData::Int64(Array1::<i64>::from_vec(vect)), *data);
        }
        let mut vect: Vec<i32> = vec![100; 201];
        let mut counter: i32 = 0;
        vect.iter_mut().for_each(|v| {
            *v -= counter;
            counter += 1
        });
        if let Some(data) = info.get_channel_data(&"Counter_INT32_LE".to_string()) {
            assert_eq!(ChannelData::Int32(Array1::<i32>::from_vec(vect)), *data);
        }
        let mut vect: Vec<i16> = vec![100; 201];
        let mut counter: i16 = 0;
        vect.iter_mut().for_each(|v| {
            *v -= counter;
            counter += 1
        });
        if let Some(data) = info.get_channel_data(&"Counter_INT16_LE".to_string()) {
            assert_eq!(ChannelData::Int16(Array1::<i16>::from_vec(vect)), *data);
        }
    }

    #[test]
    fn channel_types() {
        let base_path = String::from("/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/");
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
            base_path, list_of_paths[0], "Vector_VirtualTimeMasterChannel.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let Some(data) = info.get_channel_data(&"Time channel".to_string()) {
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
            base_path, list_of_paths[1], "Vector_MLSDStringUTF8.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let Some(data) = info.get_channel_data(&"Data channel".to_string()) {
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
            base_path, list_of_paths[3], "Vector_VirtualDataChannelNoConversion.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        let mut vect: Vec<u64> = vec![0; 200];
        let mut counter: u64 = 0;
        vect.iter_mut().for_each(|v| {
            *v += counter;
            counter += 1
        });
        if let Some(data) = info.get_channel_data(&"Data channel".to_string()) {
            assert_eq!(ChannelData::UInt64(Array1::<u64>::from_vec(vect)), *data);
        }
        // VLSD testing
        let file_name = format!(
            "{}{}{}",
            base_path, list_of_paths[2], "Vector_VLSDStringUTF8.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let Some(data) = info.get_channel_data(&"Data channel".to_string()) {
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
            base_path, list_of_paths[4], "Vector_SyncStreamChannel.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        // TODO how to check ?
    }
    #[test]
    fn record_layout() {
        let base_path = String::from("/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/RecordLayout/");
        // Overlapping signals
        let file_name = format!("{}{}", base_path, "Vector_NotByteAligned.mf4");
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let Some(data) = info.get_channel_data(&"Channel B".to_string()) {
            let mut vect: Vec<u64> = vec![0; 30];
            let mut counter: u64 = 0;
            vect.iter_mut().for_each(|v| {
                *v += counter;
                counter += 1
            });
            assert_eq!(ChannelData::UInt48(Array1::<u64>::from_vec(vect)), *data);
        }

        // Overlapping signals
        let file_name = format!("{}{}", base_path, "Vector_OverlappingSignals.mf4");
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        // TODO How to check ?
    }
    #[test]
    fn data_list() {
        let base_path = String::from("/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/DataList/");
        // Equal length
        let file_name = format!("{}{}", base_path, "Vector_DT_EqualLen.MF4");
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let Some(data) = info.get_channel_data(&"channel1".to_string()) {
            assert_eq!(data.len(), 254552);
        }
        // Equal length
        let file_name = format!("{}{}", base_path, "Vector_DL_Linked_List.MF4");
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let Some(data) = info.get_channel_data(&"channel1".to_string()) {
            assert_eq!(data.len(), 254552);
        }
        // TODO conversions missing to be tested for Write and WriteTextToWriteWindow channels

        // Empty data
        let file_name = format!("{}{}", base_path, "ETAS_EmptyDL.mf4");
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();

        // SD List
        let file_name = format!("{}{}", base_path, "Vector_SD_List.MF4");
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
    }
    #[test]
    fn compressed_data() {
        let base_path = String::from("/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/CompressedData/");
        // Single DZ deflate
        let file_name = format!("{}{}", base_path, "Simple/Vector_SingleDZ_Deflate.mf4");
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();

        // Single DZ transpose deflate
        let file_name = format!(
            "{}{}",
            base_path, "Simple/Vector_SingleDZ_TransposeDeflate.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();

        // deflate data list
        let file_name = format!("{}{}", base_path, "DataList/Vector_DataList_Deflate.mf4");
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let Some(data) = info.get_channel_data(&"Time channel".to_string()) {
            let mut vect: Vec<f64> = vec![0.; 10000];
            let mut counter: f64 = 0.;
            vect.iter_mut().for_each(|v| {
                *v = counter.clone();
                counter += 0.1
            });
            let target = Array1::<f64>::from_vec(vect);
            assert_eq!(ChannelData::Float64(target), *data);
        }

        // transpose deflate data list
        let file_name = format!(
            "{}{}",
            base_path, "DataList/Vector_DataList_TransposeDeflate.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let Some(data) = info.get_channel_data(&"Time channel".to_string()) {
            let mut vect: Vec<f64> = vec![0.; 10000];
            let mut counter: f64 = 0.;
            vect.iter_mut().for_each(|v| {
                *v = counter.clone();
                counter += 0.1
            });
            let target = Array1::<f64>::from_vec(vect);
            assert_eq!(ChannelData::Float64(target), *data);
        }

        // Unsorted
        let file_name = format!("{}{}", base_path, "Unsorted/Vector_SingleDZ_Unsorted.MF4");
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
    }

    #[test]
    fn unsorted_data() {
        let base_path = String::from("/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/UnsortedData/");
        let file_name = format!("{}{}", base_path, "Vector_Unsorted_VLSD.MF4");
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
    }
    #[test]
    fn conversion() {
        let base_path = String::from("/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/");
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
            base_path, list_of_paths[0], "Vector_LinearConversion.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let Some(data) = info.get_channel_data(&"Data channel".to_string()) {
            let mut vect: Vec<f64> = vec![0.; 10];
            let mut counter: f64 = 0.;
            vect.iter_mut().for_each(|v| {
                *v = counter.clone() * -3.2 - 4.8;
                counter += 1.
            });
            let target = Array1::<f64>::from_vec(vect);
            assert_eq!(ChannelData::Float64(target), *data);
        }
        let file_name = format!(
            "{}{}{}",
            base_path, list_of_paths[0], "Vector_LinearConversionFactor0.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let Some(data) = info.get_channel_data(&"Data channel".to_string()) {
            let vect: Vec<f64> = vec![3.; 10];
            let target = Array1::<f64>::from_vec(vect);
            assert_eq!(ChannelData::Float64(target), *data);
        }
        // Rational conversion
        let file_name = format!(
            "{}{}{}",
            base_path, list_of_paths[3], "Vector_RationalConversionIntParams.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();

        // Text conversion
        let file_name = format!(
            "{}{}{}",
            base_path, list_of_paths[5], "Vector_AlgebraicConversionQuadratic.mf4"
        );
        let mut info = MdfInfo::new(&file_name);
        info.load_all_channels_data_in_memory();
        if let Some(data) = info.get_channel_data(&"Data channel".to_string()) {
            let mut vect: Vec<f64> = vec![0.; 10];
            let mut counter: f64 = 0.;
            vect.iter_mut().for_each(|v| {
                *v = counter.clone() * 2. + 1.;
                counter += 1.
            });
            let target = Array1::<f64>::from_vec(vect);
            assert_eq!(ChannelData::Float64(target), *data);
        }
    }
}
