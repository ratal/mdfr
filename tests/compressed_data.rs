use anyhow::Result;
use arrow::array::Float64Builder;
use mdfr::data_holder::channel_data::ChannelData;
use mdfr::mdfreader::Mdf;
use std::fs;
use std::path::Path;
use std::sync::LazyLock;

static BASE_PATH_MDF4: LazyLock<String> = LazyLock::new(|| {
    "/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/MDF4.3/Base_Standard/Examples/".to_string()
});

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
fn single_dz_deflate() -> Result<()> {
    // Single DZ deflate
    let file_name = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "CompressedData/Simple/Vector_SingleDZ_Deflate.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    Ok(())
}

#[test]
fn single_dz_transpose_deflate() -> Result<()> {
    // Single DZ transpose deflate
    let file_name = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "CompressedData/Simple/Vector_SingleDZ_TransposeDeflate.mf4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    Ok(())
}

#[test]
fn deflate_data_list() -> Result<()> {
    // deflate data list
    let file_name = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "CompressedData/DataList/Vector_DataList_Deflate.mf4"
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
    Ok(())
}

#[test]
fn transpose_deflate_data_list() -> Result<()> {
    // transpose deflate data list
    let file_name = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "CompressedData/DataList/Vector_DataList_TransposeDeflate.mf4"
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
    Ok(())
}

#[test]
fn unsorted_compressed() -> Result<()> {
    // Unsorted
    let file_name = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "CompressedData/Unsorted/Vector_SingleDZ_Unsorted.MF4"
    );
    let mut mdf = Mdf::new(&file_name)?;
    mdf.load_all_channels_data_in_memory()?;
    Ok(())
}

#[test]
fn compressed_data_mdf43_algo() -> Result<()> {
    // read all the file in the folder
    let path = format!(
        "{}{}",
        BASE_PATH_MDF4.as_str(),
        "CompressedData/MDF430_Algorithms"
    );
    parse_info_folder(&path).unwrap();
    Ok(())
}
