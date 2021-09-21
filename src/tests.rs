#[cfg(test)]
mod tests {
    use crate::mdfinfo::MdfInfo;
    use crate::mdfreader;
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
        let file = String::from("/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/Simple/error.mf4"); // DT, big many channels
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
}
