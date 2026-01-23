#[cfg(test)]
mod tests {
    use anyhow::Result;
    use arrow::array::{
        AsArray, FixedSizeBinaryBuilder, Float64Array, Float64Builder, Int16Builder, Int32Builder,
        Int64Builder, LargeStringBuilder, PrimitiveBuilder, UInt16Builder, UInt64Builder,
    };

    use arrow::datatypes::{Float32Type, Float64Type};

    use crate::data_holder::channel_data::ChannelData;
    use crate::mdfreader::Mdf;
    use glob::glob;
    use std::fs;
    use std::io;
    use std::path::Path;
    use std::sync::Arc;
    use test_log::test;

    static BASE_PATH_MDF4: &str =
        "/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/MDF4.3/Base_Standard/Examples/";
    static BASE_PATH_MDF3: &str = "/home/ratal/workspace/mdfreader/mdfreader/tests/mdf3/";
    static BASE_TEST_PATH: &str = "/home/ratal/workspace/mdfr/test_files";
    static WRITING_MDF_FILE: &str = "/home/ratal/workspace/mdfr/test_files/test.mf4";
    #[cfg(feature = "parquet")]
    static WRITING_PARQUET_FILE: &str = "/home/ratal/workspace/mdfr/test_files/test_parquet";
    #[cfg(feature = "hdf5")]
    static WRITING_HDF5_FILE: &str = "/home/ratal/workspace/mdfr/test_files/test_hdf5.hdf5";

    #[test]
    fn info_test() -> Result<()> {
        let mut file_name = "test_files/test_basic.mf4";
        println!("reading {}", file_name);
        let mdf = Mdf::new(file_name)?;
        // println!("{:#?}", mdf);
        assert_eq!(mdf.get_version(), 410);
        file_name = "test_files/test_mdf3.mdf";
        // println!("reading {}", file_name);
        let mdf = Mdf::new(file_name)?;
        // println!("{:#?}", &mdf);
        assert_eq!(mdf.get_version(), 310);
        file_name = "test_files/test_mdf4.mf4";
        // println!("reading {}", file_name);
        let mdf = Mdf::new(file_name)?;
        // println!("{:#?}", &mdf);
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
                            #[allow(clippy::collapsible_if)]
                            if valid_ext.contains(&ext) {
                                if let Some(file_name) = entry.path().to_str() {
                                    println!(" Reading file : {}", file_name);
                                    let mut mdf = Mdf::new(file_name)?;
                                    mdf.load_all_channels_data_in_memory()?;
                                }
                            }
                        }
                    } else if metadata.is_dir() {
                        #[allow(clippy::collapsible_if)]
                        if let Some(path) = entry.path().to_str() {
                            let path_str = path.to_owned();
                            match parse_info_folder(&path_str) {
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
            "Arrays/Classification".to_string(),
            "Arrays/Simple".to_string(),
            "Attachments/Embedded".to_string(),
            "Attachments/EmbeddedCompressed".to_string(),
            "Attachments/External".to_string(),
            "BusLogging/CAN".to_string(),
            "ChannelInfo/AttachmentRef".to_string(),
            "ChannelInfo/DefaultX".to_string(),
            "ChannelTypes/MasterChannels".to_string(),
            "ChannelTypes/MLSD".to_string(),
            "ChannelTypes/Monotonicity".to_string(),
            "ChannelTypes/Synchronization".to_string(),
            "ChannelTypes/VirtualData".to_string(),
            "ChannelTypes/VLSC".to_string(),
            "ChannelTypes/VLSD".to_string(),
            "CompressedData/DataList".to_string(),
            "CompressedData/MDF430_Algorithms".to_string(),
            "CompressedData/Simple".to_string(),
            "CompressedData/Unsorted".to_string(),
            "Conversion/BitfieldConversion".to_string(),
            "Conversion/LinearConversion".to_string(),
            "Conversion/LookUpConversion".to_string(),
            "Conversion/PartialConversion".to_string(),
            "Conversion/RationalConversion".to_string(),
            "Conversion/StringConversion".to_string(),
            "Conversion/TextConversion".to_string(),
            "DataList".to_string(),
            "DataTypes/ByteArray".to_string(),
            "DataTypes/CANopenTypes".to_string(),
            "DataTypes/Complex".to_string(),
            "DataTypes/IntegerTypes".to_string(),
            "DataTypes/RealTypes".to_string(),
            "DataTypes/StringTypes".to_string(),
            "DynamicData/ChannelList".to_string(),
            "Events/EventSignals".to_string(),
            "Events/Marker".to_string(),
            "Events/Recording".to_string(),
            "Events/Trigger".to_string(),
            "GnssDataStorage".to_string(),
            "Halffloat".to_string(),
            "MetaData".to_string(),
            "RawSensorLogging/Lidar".to_string(),
            "RawSensorLogging/Video".to_string(),
            "RecordLayout".to_string(),
            "RemoteMaster".to_string(),
            "SampleReduction".to_string(),
            "Simple".to_string(),
            "Union".to_string(),
            "UnsortedData/VLSC".to_string(),
            "UnsortedData/VLSD".to_string(),
            "Variant".to_string(),
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
        let mut mdf = Mdf::new(file)?;
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
            "DataTypes/Complex/".to_string(),
        ];
        let writing_mdf_file = format!("{}/data_types_test.mf4", BASE_TEST_PATH);

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
        if let Some(data) = mdf.get_channel_data("Counter_INT64_BE") {
            assert_eq!(
                ChannelData::Int64(Int64Builder::new_from_buffer(vect.clone().into(), None)),
                data.clone()
            );
        }
        let mut mdf2 = mdf.write(&writing_mdf_file, false)?;
        mdf2.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf2.get_channel_data("Counter_INT64_LE") {
            assert_eq!(
                ChannelData::Int64(Int64Builder::new_from_buffer(vect.into(), None)),
                data.clone()
            );
        }
        let mut vect: Vec<i32> = vec![100; 201];
        let mut counter: i32 = 0;
        vect.iter_mut().for_each(|v| {
            *v -= counter;
            counter += 1
        });
        // error in file, cn data type is 1 but should be 3
        // if let Some(data) = mdf.get_channel_data(&"Counter_INT32_BE".to_string()) {
        //     assert_eq!(
        //         PrimitiveArray::new(DataType::Int32, Buffer::from(vect.clone()), None).boxed(),
        //         data
        //     );
        // }
        if let Some(data) = mdf2.get_channel_data("Counter_INT32_LE") {
            assert_eq!(
                ChannelData::Int32(Int32Builder::new_from_buffer(vect.into(), None)),
                data.clone()
            );
        }
        let mut vect: Vec<i16> = vec![100; 201];
        let mut counter: i16 = 0;
        vect.iter_mut().for_each(|v| {
            *v -= counter;
            counter += 1
        });
        if let Some(data) = mdf.get_channel_data("Counter_INT16_BE") {
            assert_eq!(
                ChannelData::Int16(Int16Builder::new_from_buffer(vect.clone().into(), None)),
                data.clone()
            );
        }
        if let Some(data) = mdf2.get_channel_data("Counter_INT16_LE") {
            assert_eq!(
                ChannelData::Int16(Int16Builder::new_from_buffer(vect.into(), None)),
                data.clone()
            );
        }

        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[2], "ETAS_IntegerTypes.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;

        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[2], "dSPACE_IntegerTypes.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;

        // Real types
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[3], "Vector_RealTypes.MF4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        let file_name = format!("{}{}", BASE_PATH_MDF4, "Halffloat/halffloat_sinus.mf4");
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[3], "dSPACE_RealTypes.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;

        // StringTypes testing
        // UTF8
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
            BASE_PATH_MDF4, list_of_paths[4], "Vector_FixedLengthStringUTF8.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("Time channel") {
            assert_eq!(
                ChannelData::Float64(Float64Builder::new_from_buffer(
                    vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.].into(),
                    None
                )),
                data.clone()
            );
        }
        if let Some(data) = mdf.get_channel_data("Data channel") {
            assert_eq!(expected_string_result, data.clone());
        }
        let mut mdf2 = mdf.write(&writing_mdf_file, false)?;
        mdf2.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf2.get_channel_data("Data channel") {
            assert_eq!(expected_string_result, data.clone());
        }
        //UTF16
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[4], "Vector_FixedLengthStringUTF16_BE.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("Data channel") {
            assert_eq!(expected_string_result, data.clone());
        }
        let mut mdf2 = mdf.write(&writing_mdf_file, false)?;
        mdf2.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf2.get_channel_data("Data channel") {
            assert_eq!(expected_string_result, data.clone());
        }
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[4], "Vector_FixedLengthStringUTF16_LE.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("Data channel") {
            assert_eq!(expected_string_result, data.clone());
        }
        //SBC
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[4], "Vector_FixedLengthStringSBC.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("Data channel") {
            assert_eq!(expected_string_result, data.clone());
        }
        let mut mdf2 = mdf.write(&writing_mdf_file, false)?;
        mdf2.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf2.get_channel_data("Data channel") {
            assert_eq!(expected_string_result, data.clone());
        }

        // byteArray testing
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[0], "Vector_ByteArrayFixedLength.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        let mut byte_array = FixedSizeBinaryBuilder::with_capacity(10, 5);
        byte_array.append_value(vec![255, 255, 255, 255, 255])?;
        byte_array.append_value(vec![18, 35, 52, 69, 86])?;
        byte_array.append_value(vec![0, 1, 2, 3, 4])?;
        byte_array.append_value(vec![4, 3, 2, 1, 0])?;
        byte_array.append_value(vec![255, 254, 253, 252, 251])?;
        byte_array.append_value(vec![250, 249, 248, 247, 246])?;
        byte_array.append_value(vec![245, 244, 243, 242, 241])?;
        byte_array.append_value(vec![240, 239, 238, 237, 236])?;
        byte_array.append_value(vec![235, 234, 233, 232, 231])?;
        byte_array.append_value(vec![255, 255, 255, 255, 255])?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("Time channel") {
            assert_eq!(
                ChannelData::Float64(Float64Builder::new_from_buffer(
                    vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.].into(),
                    None
                )),
                data.clone()
            );
        }
        if let Some(data) = mdf.get_channel_data("Data channel") {
            assert_eq!(&ChannelData::FixedSizeByteArray(byte_array), data);
        }
        let mut mdf2 = mdf.write(&writing_mdf_file, false)?;
        mdf2.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf2.get_channel_data("Data channel") {
            let mut byte_array = FixedSizeBinaryBuilder::with_capacity(10, 5);
            byte_array.append_value(vec![255, 255, 255, 255, 255])?;
            byte_array.append_value(vec![18, 35, 52, 69, 86])?;
            byte_array.append_value(vec![0, 1, 2, 3, 4])?;
            byte_array.append_value(vec![4, 3, 2, 1, 0])?;
            byte_array.append_value(vec![255, 254, 253, 252, 251])?;
            byte_array.append_value(vec![250, 249, 248, 247, 246])?;
            byte_array.append_value(vec![245, 244, 243, 242, 241])?;
            byte_array.append_value(vec![240, 239, 238, 237, 236])?;
            byte_array.append_value(vec![235, 234, 233, 232, 231])?;
            byte_array.append_value(vec![255, 255, 255, 255, 255])?;
            assert_eq!(&ChannelData::FixedSizeByteArray(byte_array), data);
        }
        //Complex testing
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[5], "Vector_ComplexNumbers.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;

        // Cleanup temporary file
        fs::remove_file(&writing_mdf_file).ok();
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
            "ChannelTypes/Monotonicity/".to_string(),
            "ChannelTypes/VLSC/".to_string(),
        ];

        // MasterTypes testing
        let mut vect: Vec<f64> = vec![0.; 101];
        let mut counter: f64 = 0.;
        vect.iter_mut().for_each(|v| {
            *v = counter * 0.03;
            counter += 1.
        });
        let expected_master =
            ChannelData::Float64(Float64Builder::new_from_buffer(vect.into(), None));
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[0], "Vector_VirtualTimeMasterChannel.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("Time channel") {
            assert_eq!(expected_master, data.clone());
        }
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[0], "Vector_DifferentMasterChannels.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("Time channel") {
            assert_eq!(expected_master, data.clone());
        }
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[0], "Vector_NoMasterChannel.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("Time channel") {
            assert_eq!(expected_master, data.clone());
        }

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
            BASE_PATH_MDF4, list_of_paths[1], "Vector_MLSD_String_UTF8.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("Data channel") {
            assert_eq!(expected_string_result, data.clone());
        }
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[1], "Vector_MLSD_String_SBC.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("Data channel") {
            assert_eq!(expected_string_result, data.clone());
        }
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[1], "Vector_MLSD_String_UTF16_BE.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("Data channel") {
            assert_eq!(expected_string_result, data.clone());
        }
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[1], "Vector_MLSD_String_UTF16_LE.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("Data channel") {
            assert_eq!(expected_string_result, data.clone());
        }

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
            BASE_PATH_MDF4, list_of_paths[3], "Vector_VirtualDataChannelNoConversion.mf4"
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
            BASE_PATH_MDF4, list_of_paths[3], "Vector_VirtualDataChannelLinearConversion.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("Data channel") {
            assert_eq!(ChannelData::Float64(virtal_linear_vect), *data);
        }
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[3], "Vector_VirtualDataChannelConstantConversion.mf4"
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
        // VLSD testing
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[2], "Vector_VLSD_String_UTF8.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("Data channel") {
            assert_eq!(expected_string_result, data.clone());
        }
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[2], "Vector_VLSD_String_UTF16_LE.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("Data channel") {
            assert_eq!(expected_string_result, data.clone());
        }
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[2], "Vector_VLSD_String_UTF16_BE.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("Data channel") {
            assert_eq!(expected_string_result, data.clone());
        }

        // Synchronization
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[4], "Vector_SyncStreamChannel.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;

        // VLSC
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[6], "Vector_VLSC_String_UTF8.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("Data channel") {
            assert_eq!(expected_string_result, data.clone());
        }

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

        // Verify variant channel exists and has correct length
        if let Some(variant_data) = mdf.get_channel_data("variant") {
            assert_eq!(variant_data.len(), 3, "variant should have 3 samples");
            // Variant data is FixedSizeByteArray containing merged data from different options
        }

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

            // Verify union channel exists and has correct length
            if let Some(union_data) = mdf.get_channel_data("union") {
                assert_eq!(union_data.len(), 3, "union should have 3 samples");
                // Union data is FixedSizeByteArray containing overlapping member data
            }
        }

        // VD Block (Virtual Data)
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4,
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

    #[test]
    fn record_layout() -> Result<()> {
        // Overlapping signals
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4, "RecordLayout/NotByteAligned/Vector_NotByteAligned.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("Channel B") {
            let mut vect: Vec<u64> = vec![0; 30];
            let mut counter: u64 = 0;
            vect.iter_mut().for_each(|v| {
                *v += counter;
                counter += 1
            });
            assert_eq!(
                ChannelData::UInt64(UInt64Builder::new_from_buffer(vect.into(), None)),
                data.clone()
            );
        }

        // Overlapping signals
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4, "RecordLayout/OverlappingSignals/Vector_OverlappingSignals.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        Ok(())
    }
    #[test]
    fn data_list() -> Result<()> {
        // Equal length
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4, "DataList/DT_EqualLength/Vector_DT_EqualLen.MF4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("channel1") {
            assert_eq!(data.len(), 254552);
        }
        // Equal length
        let file_name = format!("{}{}", BASE_PATH_MDF4, "DataList/Vector_DL_Linked_List.MF4");
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("channel1") {
            assert_eq!(data.len(), 254552);
        }

        // Empty data
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4, "DataList/EmptyList/ETAS_EmptyDL.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;

        // SD List
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4, "DataList/SD_List/Vector_SD_List.MF4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;

        // Separate invalidation bits
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4, "DataList/RAC_MDF420_ListData_SeparateInvalidationBits.mf4"
        );
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
        if let Some(data) = mdf.get_channel_data("Time channel") {
            let mut vect: Vec<f64> = vec![0.; 10000];
            let mut counter: u64 = 0;
            vect.iter_mut().for_each(|v| {
                *v = (counter as f64) / 10.0;
                counter += 1;
            });
            assert_eq!(
                &ChannelData::Float64(Float64Builder::new_from_buffer(vect.into(), None)),
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
        if let Some(data) = mdf.get_channel_data("Time channel") {
            let mut vect: Vec<f64> = vec![0.; 10000];
            let mut counter: u64 = 0;
            vect.iter_mut().for_each(|v| {
                *v = (counter as f64) / 10.0;
                counter += 1;
            });
            assert_eq!(
                &ChannelData::Float64(Float64Builder::new_from_buffer(vect.into(), None)),
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
    fn compressed_data_mdf43_algo() -> Result<()> {
        // read all the file in the folder
        let path = format!("{}{}", BASE_PATH_MDF4, "CompressedData/MDF430_Algorithms");
        parse_info_folder(&path).unwrap();
        Ok(())
    }

    #[test]
    fn unsorted_data() -> Result<()> {
        // VLSD
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4, "UnsortedData/VLSD/Vector_Unsorted_VLSD.MF4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4, "UnsortedData/VLSD/RAC_MDF430_Unsorted_VLSD_Compact_Structure.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;

        //VLSC
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4, "UnsortedData/VLSC/RAC_MDF430_Unsorted_VLSC_Compact_Structure.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;

        // Construct expected data (standard pattern for these tests)
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

        if let Some(data) = mdf.get_channel_data("Data channel") {
            assert_eq!(expected_string_result, data.clone());
        }
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
            "Conversion/BitfieldConversion/".to_string(),
        ];

        // Linear conversion testing
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[0], "Vector_LinearConversion.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("Data channel") {
            let mut vect: Vec<f64> = vec![0.; 10];
            let mut counter: f64 = 0.;
            vect.iter_mut().for_each(|v| {
                *v = counter * -3.2 - 4.8;
                counter += 1.
            });
            assert_eq!(
                &ChannelData::Float64(Float64Builder::new_from_buffer(vect.into(), None)),
                data
            );
        }
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[0], "Vector_LinearConversionFactor0.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("Data channel") {
            let vect: Vec<f64> = vec![3.; 10];
            assert_eq!(
                &ChannelData::Float64(Float64Builder::new_from_buffer(vect.into(), None)),
                data
            );
        }
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[0], "dSPACE_LinearConversion.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;

        // Rational conversion
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[3], "Vector_RationalConversionIntParams.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[3], "Vector_RationalConversionRealParams.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[3], "Vector_RationalConversionZeroedParams.mf4"
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
        if let Some(data) = mdf.get_channel_data("Data channel") {
            let vect = Vec::from([1., 2., 5., 10., 17., 26., 37., 50., 65., 82.]);
            assert_eq!(
                &ChannelData::Float64(Float64Builder::new_from_buffer(vect.into(), None)),
                data
            );
        }
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[5], "Vector_AlgebraicConversionRational.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[5], "Vector_AlgebraicConversionSinus.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[5], "dSPACE_AlgebraicConversion.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;

        // Lookup conversion : Value to Value Table With Interpolation
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[1], "Vector_Value2ValueConversionInterpolation.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("Data channel") {
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
                &ChannelData::Float64(Float64Builder::new_from_buffer(vect.into(), None)),
                data
            );
        }
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[1], "dSPACE_Value2ValueConversionInterpolation.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;

        // Lookup conversion : Value to Value Table Without Interpolation
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[1], "Vector_Value2ValueConversionNoInterpolation.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("Data channel") {
            let vect = Vec::from([
                -5., -5., -5., -5., -5., -5., -5., -2., -2., -2., -2., 0., 0., 0., 1., 1., 1., 2.,
                2., 0., 0., 3., 3., 6., 6., 3., 3., 0., 0., 0.,
            ]);
            assert_eq!(
                &ChannelData::Float64(Float64Builder::new_from_buffer(vect.into(), None)),
                data
            );
        }
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[1], "dSPACE_Value2ValueConversionNoInterpolation.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;

        // Lookup conversion : Value Range to Value
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[1], "Vector_ValueRange2ValueConversion.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("Data channel") {
            let vect = Vec::from([
                -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0,
                2.0, 3.0, 3.0, 5.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 9.0, 9.0, 9.0,
            ]);
            assert_eq!(
                &ChannelData::Float64(Float64Builder::new_from_buffer(vect.into(), None)),
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
        if let Some(data) = mdf.get_channel_data("Data channel") {
            let mut target = LargeStringBuilder::with_capacity(10, 20);
            target.append_value("No match");
            target.append_value("first gear");
            target.append_value("second gear");
            target.append_value("third gear");
            target.append_value("fourth gear");
            target.append_value("fifth gear");
            target.append_value("No match");
            target.append_value("No match");
            target.append_value("No match");
            target.append_value("No match");
            assert_eq!(&ChannelData::Utf8(target), data);
        }
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[1], "dSPACE_Value2TextConversion.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;

        // Lookup conversion : Value range to Text
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[1], "Vector_ValueRange2TextConversion.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("Data channel") {
            let mut target = LargeStringBuilder::with_capacity(10, 20);
            target.append_value("Out of range");
            target.append_value("very low");
            target.append_value("very low");
            target.append_value("very low");
            target.append_value("low");
            target.append_value("low");
            target.append_value("medium");
            target.append_value("medium");
            target.append_value("high");
            target.append_value("high");
            assert_eq!(&ChannelData::Utf8(target), data);
        }
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[1], "dSPACE_ValueRange2TextConversion.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;

        // Lookup conversion : Value range to Text,
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[2], "Vector_StatusStringTableConversionAlgebraic.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("Data channel") {
            let mut vect: Vec<f64> = vec![0.; 300];
            let mut counter: f64 = 0.;
            vect.iter_mut().for_each(|v| {
                *v = counter;
                counter += 0.1
            });
            let mut target = LargeStringBuilder::with_capacity(vect.len(), 32);
            vect.iter().for_each(|v| {
                if 9.9999 <= *v && *v <= 10.1001 {
                    target.append_value("Illegal value")
                } else if 20.0 <= *v && *v <= 30.0 {
                    target.append_value("Out of range")
                } else {
                    target.append_value((10.0 / (v - 10.0)).to_string())
                }
            });
            let data_values = data.finish_cloned();
            let data_values = data_values
                .as_string::<i64>()
                .iter()
                .collect::<Vec<Option<&str>>>();
            let target_values = target.finish_cloned();
            let target_values = target_values.iter().collect::<Vec<Option<&str>>>();
            assert_eq!(target_values[0], data_values[0]);
            assert_eq!(target_values[299], data_values[299]);
            assert_eq!(target_values[101], data_values[101]);
        }

        // Text conversion : Text to Value
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[4], "Vector_Text2ValueConversion.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data("Data channel") {
            let vect = Vec::from([-50., 1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            assert_eq!(
                &ChannelData::Float64(Float64Builder::new_from_buffer(vect.into(), None)),
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
        if let Some(data) = mdf.get_channel_data("Data channel") {
            let mut target = LargeStringBuilder::with_capacity(10, 20);
            target.append_value("No translation");
            target.append_value("Eins");
            target.append_value("Zwei");
            target.append_value("Drei");
            target.append_value("Vier");
            target.append_value("Fünf");
            target.append_value("Sechs");
            target.append_value("Sieben");
            target.append_value("Acht");
            target.append_value("Neun");
            assert_eq!(&ChannelData::Utf8(target), data);
        }

        // Partial conversion
        // TODO
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[2], "Vector_PartialConversionLinearIdentityAlgebraic.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[2], "Vector_PartialConversionValueRange2TextRational.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[2], "Vector_StatusStringTableConversionAlgebraic.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;

        // Bitfield conversion
        // TODO
        let file_name = format!(
            "{}{}{}",
            BASE_PATH_MDF4, list_of_paths[6], "RAC_MDF420_BitfieldTextTable.mf4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        Ok(())
    }

    #[test]
    fn bus_logging() -> Result<()> {
        // sort bus
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4, "BusLogging/CAN/Vector_CAN_DataFrame_Sort_ID.MF4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        if let Some(data) =
            mdf.get_channel_data("CAN_DataFrame.ID CAN_DataFrame_101 CANReplay_7_5 Message")
        {
            let vect: Vec<f64> = vec![101.; 79];
            assert_eq!(
                ChannelData::Float64(Float64Builder::new_from_buffer(vect.into(), None)),
                *data
            );
        }
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4, "BusLogging/CAN/Vector_CAN_DataFrame_Sort_Bus.MF4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
        let file_name = format!(
            "{}{}",
            BASE_PATH_MDF4, "BusLogging/CAN/Vector_CAN_DataFrame_Sort_ID_SignalDesc.MF4"
        );
        let mut mdf = Mdf::new(&file_name)?;
        mdf.load_all_channels_data_in_memory()?;
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
        // without compression
        let writing_mdf_file = format!("{}/writing_mdf4_test.mf4", BASE_TEST_PATH);
        let mut info2 = mdf.write(&writing_mdf_file, false)?;
        info2.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(ref_channel) {
            if let Some(data2) = info2.get_channel_data(ref_channel) {
                assert_eq!(*data2, *data);
            } else {
                panic!("Channel not found");
            }
        } else {
            panic!("Channel not found");
        }
        // with compression
        let mut info2 = mdf.write(&writing_mdf_file, true)?;
        info2.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(ref_channel) {
            if let Some(data2) = info2.get_channel_data(ref_channel) {
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
        let mut info2 = mdf.write(&writing_mdf_file, true)?;
        info2.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(ref_channel) {
            if let Some(data2) = info2.get_channel_data(ref_channel) {
                assert_eq!(*data2, *data);
            } else {
                panic!("Channel not found");
            }
        } else {
            panic!("Channel not found");
        }
        // without compression
        let mut info2 = mdf.write(&writing_mdf_file, false)?;
        info2.load_all_channels_data_in_memory()?;
        if let Some(data) = mdf.get_channel_data(ref_channel) {
            if let Some(data2) = info2.get_channel_data(ref_channel) {
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
        let mut mdf4 = mdf.write(&writing_mdf_file, true)?;
        mdf4.load_all_channels_data_in_memory()?;
        let mdf3_data = mdf.get_channel_data(channel_name3);
        let mdf4_data = mdf4.get_channel_data(channel_name3);
        assert_eq!(mdf3_data, mdf4_data);

        // Cleanup temporary file
        fs::remove_file(&writing_mdf_file).ok();
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
        if let Some(data) = mdf.get_channel_data(ref_channel) {
            let mut new_data = PrimitiveBuilder::with_capacity(data.len());
            data.finish_cloned()
                .as_primitive::<Float32Type>()
                .iter()
                .for_each(|v| new_data.append_option(v));
            new_data.values_slice_mut()[0] = 0.0f32;
            mdf.set_channel_data(ref_channel, ChannelData::Float32(new_data).as_ref())?;
            mdf.set_channel_desc(ref_channel, ref_desc);
            mdf.set_channel_unit(ref_channel, ref_unit);
            mdf.set_channel_master_type(ref_channel, 1)?;
        } else {
            panic!("channel not found");
        }
        if let Some(data) = mdf.get_channel_data(ref_channel) {
            let (minimum, _) = data.min_max();
            if let Some(min) = minimum {
                assert!(min < 1000.0f64);
            }
        } else {
            panic!("channel not found");
        }
        match mdf.get_channel_desc(ref_channel) {
            Ok(Some(desc)) => {
                assert_eq!(desc, ref_desc);
            }
            _ => {
                panic!("channel not found");
            }
        }
        match mdf.get_channel_unit(ref_channel) {
            Ok(Some(unit)) => {
                assert_eq!(unit, ref_unit);
            }
            _ => {
                panic!("channel not found");
            }
        }
        assert_eq!(mdf.get_channel_master_type(ref_channel), 1);

        // add new channel
        drop(mdf);
        let mut mdf = Mdf::new(&file)?;
        mdf.load_all_channels_data_in_memory()?;
        let channel_name = r"Fake_name".to_string();
        let new_channel_name = r"New fake_name".to_string();
        let new_data = Arc::new(Float64Array::try_new(vec![0f64; 3300].into(), None)?);
        let master_channel = mdf.get_channel_master(ref_channel);
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
        )?;

        if let Some(data) = mdf.get_channel_data(&channel_name) {
            let (minimum, _) = data.min_max();
            if let Some(min) = minimum {
                assert!(min == 0.0f64);
            }
        } else {
            panic!("channel not found");
        }
        match mdf.get_channel_desc(&channel_name) {
            Ok(Some(desc)) => {
                assert_eq!(desc, ref_desc.to_string());
            }
            _ => {
                panic!("channel not found");
            }
        }
        match mdf.get_channel_unit(&channel_name) {
            Ok(Some(unit)) => {
                assert_eq!(unit, ref_unit);
            }
            _ => {
                panic!("channel not found");
            }
        }
        assert_eq!(mdf.get_channel_master_type(&channel_name), 0);

        //rename
        assert!(mdf.get_channel_data(&channel_name).is_some());
        mdf.rename_channel(&channel_name, &new_channel_name);
        assert!(mdf.get_channel_data(&channel_name).is_none());

        //remove
        assert!(mdf.get_channel_data(&new_channel_name).is_some());
        mdf.remove_channel(&new_channel_name);
        assert!(mdf.get_channel_data(&new_channel_name).is_none());
        Ok(())
    }
    #[cfg(feature = "parquet")]
    #[test]
    fn export_to_parquet() -> Result<()> {
        // Export mdf4 to Parquet file
        let file = format!(
            "{}{}",
            BASE_PATH_MDF4, &"Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4"
        );
        let extension = "*.parquet";
        let mut mdf = Mdf::new(&file)?;
        mdf.load_all_channels_data_in_memory()?;
        mdf.export_to_parquet(WRITING_PARQUET_FILE, Some("zstd"))
            .expect("failed writing mdf4 parquet file");
        // Export mdf3 to Parquet file
        let file = format!(
            "{}{}",
            BASE_PATH_MDF3, &"RJ_N16-12-363_BM-15C-0024_228_2_20170116094355_CAN.dat"
        );
        let mut mdf = Mdf::new(&file)?;
        mdf.load_all_channels_data_in_memory()?;
        mdf.export_to_parquet(WRITING_PARQUET_FILE, Some("snappy"))
            .expect("failed writing mdf3 parquet file");
        // remove all generated parquet files
        let pattern = format!("{}/{}", BASE_TEST_PATH, extension);
        for path in glob(&pattern).unwrap().filter_map(Result::ok) {
            fs::remove_file(path)?;
        }
        Ok(())
    }
    #[cfg(feature = "hdf5")]
    #[test]
    fn export_to_hdf5() -> Result<()> {
        // Export mdf4 to Parquet file
        let file = format!(
            "{}{}",
            BASE_PATH_MDF4, &"Simple/PCV_iO_Gen3_LK1__3l_TDI.mf4"
        );
        let extension = "*.hdf5";
        let mut mdf = Mdf::new(&file)?;
        mdf.load_all_channels_data_in_memory()?;
        mdf.export_to_hdf5(&WRITING_HDF5_FILE, Some(&"lzf"))
            .expect("failed writing mdf4 hdf5 file");
        // Export mdf3 to Parquet file
        let file = format!(
            "{}{}",
            BASE_PATH_MDF3, &"RJ_N16-12-363_BM-15C-0024_228_2_20170116094355_CAN.dat"
        );
        let mut mdf = Mdf::new(&file)?;
        mdf.load_all_channels_data_in_memory()?;
        mdf.export_to_hdf5(&WRITING_HDF5_FILE, Some(&"deflate"))
            .expect("failed writing mdf3 hdf5 file");
        // remove all generated hdf5 files
        let pattern = format!("{}/{}", BASE_TEST_PATH, extension);
        for path in glob(&pattern).unwrap().filter_map(Result::ok) {
            fs::remove_file(path)?;
        }
        Ok(())
    }
}
