
#[cfg(test)]
mod tests {
    use std::io;
    use std::fs::{self, DirEntry};
    use std::path::Path;
    use std::vec::Vec;
    use crate::mdfinfo;

    #[test]
    fn info_test() -> io::Result<()>{
        let mut file_name ="/home/ratal/workspace/mdfr/test_files/Test.mf4";
        println!("reading {}", file_name);
        let mut info = mdfinfo::mdfinfo(file_name);
        println!("{:#?}", info);
        assert_eq!(info.get_version(), 410);
        file_name ="/home/ratal/workspace/mdfr/test_files/Mdf3_hiddenBytes_NotAlignedBytes.dat";
        println!("reading {}", file_name);
        let mut info = mdfinfo::mdfinfo(file_name);
        println!("{:#?}", &info);
        assert_eq!(info.get_version(), 320);
        Ok(())
    }

    
    fn parse_info_folder(folder: &String) -> io::Result<()> {
        let path = Path::new(folder);
        let mut valid_ext:Vec<String>  = vec!["mf4".to_string(), "DAT".to_string(), "dat".to_string(), "MDF".to_string(), "mdf".to_string()];
        if path.is_dir() {
            for entry in fs::read_dir(path)? {
                let entry = entry?;
                if let Ok(metadata) = entry.metadata() {
                    if metadata.is_file() {
                        if let Ok(ext) = entry.path().extension().unwrap().to_os_string().into_string() {
                            if valid_ext.contains(&ext) {
                                if let Some(file_name) = entry.path().to_str() {
                                    println!(" Reading file : {}",file_name);
                                    let info = mdfinfo::mdfinfo(file_name);
                                }
                            }
                        }
                    } else if metadata.is_dir() {
                        if let Some(path) = entry.path().to_str() {
                            let path_str = path.to_owned();
                            let _ = match parse_info_folder(&path_str) {
                                Ok(v) => v,
                                Err(e) => println!("Error parsing the folder {} \n {}", path_str, e),
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
        let list_of_paths = ["Simple".to_string(), "ChannelInfo".to_string(), "ChannelTypes".to_string(),
             "DataTypes".to_string(), "MetaData".to_string(), "RecordLayout".to_string(), 
             "Events".to_string(), "SampleReduction".to_string(), "Conversion".to_string(),
             "BusLogging".to_string(), "Attachments".to_string(), "ClassificationResults".to_string(),
             "CompressedData".to_string(), "DataList".to_string(), "UnsortedData".to_string(),
             "Arrays".to_string()];
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
        let file = String::from("/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/Simple/test.mf4");
        //let file = String::from("/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/ASAM_COMMON_MDF_V4-1-0/Base_Standard/Examples/Simple/ZED_1hz_7197.mf4");
        let info = mdfinfo::mdfinfo(&file);
    }
}