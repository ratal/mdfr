//! Parallel per-folder ASAM MDF4 and MDF3 reference file parsing tests.
//!
//! Each MDF4 example folder runs as an independent `#[test]` so Cargo schedules
//! folders concurrently across threads while keeping file iteration sequential
//! inside each folder to bound peak memory on the 1.74 GB `Simple/` and 1.11 GB
//! `mdf3/` files.

use crate::common;

use anyhow::Result;
use mdfr::mdfreader::Mdf;
use std::fs;
use std::path::Path;
use std::sync::LazyLock;

static BASE_PATH_MDF4: LazyLock<String> = LazyLock::new(common::base_path_mdf4);

fn parse_info_folder(folder: &str) -> Result<()> {
    let path = Path::new(folder);
    let valid_ext = ["mf4", "MF4", "DAT", "dat", "MDF", "mdf"];
    if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            if let Ok(metadata) = entry.metadata() {
                if metadata.is_file() {
                    if let Some(ext) = entry.path().extension().and_then(|s| s.to_str())
                        && valid_ext.contains(&ext)
                        && let Some(file_name) = entry.path().to_str()
                    {
                        println!(" Reading file : {file_name}");
                        let mut mdf = Mdf::new(file_name)?;
                        mdf.load_all_channels_data_in_memory()?;
                    }
                } else if metadata.is_dir()
                    && let Some(sub_path) = entry.path().to_str()
                    && let Err(e) = parse_info_folder(sub_path)
                {
                    println!("Error parsing the folder {sub_path} \n {e}");
                }
            }
        }
    }
    Ok(())
}

macro_rules! mdf4_folder_test {
    ($fn_name:ident, $rel_path:literal) => {
        #[test]
        fn $fn_name() -> Result<()> {
            if !Path::new(BASE_PATH_MDF4.as_str()).is_dir() {
                return Ok(());
            }
            let folder = format!("{}{}", BASE_PATH_MDF4.as_str(), $rel_path);
            println!("reading folder : {}", $rel_path);
            parse_info_folder(&folder)
        }
    };
}

mdf4_folder_test!(folder_arrays_classification, "Arrays/Classification");
mdf4_folder_test!(folder_arrays_simple, "Arrays/Simple");
mdf4_folder_test!(folder_attachments_embedded, "Attachments/Embedded");
mdf4_folder_test!(
    folder_attachments_embedded_compressed,
    "Attachments/EmbeddedCompressed"
);
mdf4_folder_test!(folder_attachments_external, "Attachments/External");
mdf4_folder_test!(folder_bus_logging_can, "BusLogging/CAN");
mdf4_folder_test!(
    folder_channel_info_attachment_ref,
    "ChannelInfo/AttachmentRef"
);
mdf4_folder_test!(folder_channel_info_default_x, "ChannelInfo/DefaultX");
mdf4_folder_test!(
    folder_channel_types_master_channels,
    "ChannelTypes/MasterChannels"
);
mdf4_folder_test!(folder_channel_types_mlsd, "ChannelTypes/MLSD");
mdf4_folder_test!(
    folder_channel_types_monotonicity,
    "ChannelTypes/Monotonicity"
);
mdf4_folder_test!(
    folder_channel_types_synchronization,
    "ChannelTypes/Synchronization"
);
mdf4_folder_test!(
    folder_channel_types_virtual_data,
    "ChannelTypes/VirtualData"
);
mdf4_folder_test!(folder_channel_types_vlsc, "ChannelTypes/VLSC");
mdf4_folder_test!(folder_channel_types_vlsd, "ChannelTypes/VLSD");
mdf4_folder_test!(folder_compressed_data_datalist, "CompressedData/DataList");
mdf4_folder_test!(
    folder_compressed_data_mdf430_algorithms,
    "CompressedData/MDF430_Algorithms"
);
mdf4_folder_test!(folder_compressed_data_simple, "CompressedData/Simple");
mdf4_folder_test!(folder_compressed_data_unsorted, "CompressedData/Unsorted");
mdf4_folder_test!(folder_conversion_bitfield, "Conversion/BitfieldConversion");
mdf4_folder_test!(folder_conversion_linear, "Conversion/LinearConversion");
mdf4_folder_test!(folder_conversion_lookup, "Conversion/LookUpConversion");
mdf4_folder_test!(folder_conversion_partial, "Conversion/PartialConversion");
mdf4_folder_test!(folder_conversion_rational, "Conversion/RationalConversion");
mdf4_folder_test!(folder_conversion_string, "Conversion/StringConversion");
mdf4_folder_test!(folder_conversion_text, "Conversion/TextConversion");
mdf4_folder_test!(folder_datalist, "DataList");
mdf4_folder_test!(folder_data_types_byte_array, "DataTypes/ByteArray");
mdf4_folder_test!(folder_data_types_canopen, "DataTypes/CANopenTypes");
mdf4_folder_test!(folder_data_types_complex, "DataTypes/Complex");
mdf4_folder_test!(folder_data_types_integer, "DataTypes/IntegerTypes");
mdf4_folder_test!(folder_data_types_real, "DataTypes/RealTypes");
mdf4_folder_test!(folder_data_types_string, "DataTypes/StringTypes");
mdf4_folder_test!(folder_dynamic_data_channel_list, "DynamicData/ChannelList");
mdf4_folder_test!(folder_events_signal, "Events/EventSignals");
mdf4_folder_test!(folder_events_marker, "Events/Marker");
mdf4_folder_test!(folder_events_recording, "Events/Recording");
mdf4_folder_test!(folder_events_trigger, "Events/Trigger");
mdf4_folder_test!(folder_gnss_data_storage, "GnssDataStorage");
mdf4_folder_test!(folder_halffloat, "Halffloat");
mdf4_folder_test!(
    folder_metadata_custom_extensions,
    "MetaData/CustomExtensions"
);
mdf4_folder_test!(folder_metadata_hdo, "MetaData/HDO");
mdf4_folder_test!(folder_raw_sensor_lidar, "RawSensorLogging/Lidar");
mdf4_folder_test!(folder_raw_sensor_video, "RawSensorLogging/Video");
mdf4_folder_test!(
    folder_record_layout_not_byte_aligned,
    "RecordLayout/NotByteAligned"
);
mdf4_folder_test!(
    folder_record_layout_overlapping_signals,
    "RecordLayout/OverlappingSignals"
);
mdf4_folder_test!(folder_remote_master, "RemoteMaster");
mdf4_folder_test!(folder_sample_reduction_simple, "SampleReduction/Simple");
mdf4_folder_test!(folder_simple, "Simple");
mdf4_folder_test!(folder_union, "Union");
mdf4_folder_test!(folder_unsorted_data_vlsc, "UnsortedData/VLSC");
mdf4_folder_test!(folder_unsorted_data_vlsd, "UnsortedData/VLSD");
mdf4_folder_test!(folder_variant, "Variant");

#[test]
fn parse_all_folders3() -> Result<()> {
    let base_path = common::base_path_mdf3();
    if !Path::new(&base_path).is_dir() {
        return Ok(());
    }
    parse_info_folder(&base_path)
}
