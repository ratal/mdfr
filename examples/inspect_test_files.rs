// Test file inspector to understand MDF 4.3 file contents for test refinement
use anyhow::Result;
use mdfr::mdfreader::Mdf;

fn main() -> Result<()> {
    println!("=== MDF 4.3 Test File Inspector ===\n");
    
    // 1. Channel List (CL) + Data Stream (DS) test file
    println!("1. ChannelList test (simple_list.mf4):");
    println!("=====================================");
    let file = "/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/MDF4.3/Base_Standard/Examples/DynamicData/ChannelList/simple_list.mf4";
    inspect_file(file)?;
    
    // 2. Channel Variant (CV) test file
    println!("\n2. Channel Variant test (Etas_cv_storage_with_fixed_length.mf4):");
    println!("==================================================================");
    let file = "/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/MDF4.3/Base_Standard/Examples/Variant/Etas_cv_storage_with_fixed_length.mf4";
    inspect_file(file)?;
    
    // 3. Channel Union (CU) test file
    println!("\n3. Channel Union test (Etas_cu_storage_with_fixed_length.mf4):");
    println!("============================================================");
    let file = "/home/ratal/workspace/mdfreader/mdfreader/tests/MDF4/MDF4.3/Base_Standard/Examples/Union/Etas_cu_storage_with_fixed_length.mf4";
    inspect_file(file)?;
    
    Ok(())
}

fn inspect_file(file_path: &str) -> Result<()> {
    let mut mdf = Mdf::new(file_path)?;
    mdf.load_all_channels_data_in_memory()?;
    
    // Get all channels
    let channels = mdf.get_channel_names_set();
    println!("Total channels: {}", channels.len());
    println!("Channel names: {:?}", channels.iter().collect::<Vec<_>>());
    
    // Inspect each channel
    for channel_name in channels.iter() {
        println!("\nChannel: {}", channel_name);
        
        // Get channel data
        if let Some(data) = mdf.get_channel_data(channel_name) {
            println!("  Length: {}", data.len());
            println!("  Type: {:?}", data.data_type(false));
            
            // Show first few values
            let formatted = format!("{:?}", data);
            let preview = if formatted.len() > 200 {
                format!("{}...", &formatted[0..200])
            } else {
                formatted
            };
            println!("  Preview: {}", preview);
        }
        
        // Get metadata
        if let Ok(Some(unit)) = mdf.get_channel_unit(channel_name) {
            println!("  Unit: {}", unit);
        }
        if let Ok(Some(desc)) = mdf.get_channel_desc(channel_name) {
            println!("  Description: {}", desc);
        }
        let master_type = mdf.get_channel_master_type(channel_name);
        if master_type != 0 {
            println!("  Master type: {}", master_type);
        }
        if let Some(master) = mdf.get_channel_master(channel_name) {
            println!("  Master channel: {}", master);
        }
    }
    
    Ok(())
}
