//! Composition block structs (`Ds4Block`, `Cl4Block`, `Cv4Block`, `Cu4Block`) for MDF 4.3 — spec sections 6.24-6.27, Tables 67-70.
use alloc::vec::Vec;
use binrw::binrw;
use core::fmt::{self, Display};

/// DS4 Data Stream block struct (Table 67).
#[derive(Debug, PartialEq, Eq, Default, Clone)]
#[binrw]
#[br(little)]
#[bw(little)]
#[repr(C)]
pub struct Ds4Block {
    /// # of links
    pub ds_links: u64,
    /// links
    #[br(count = ds_links)]
    pub links: Vec<i64>,
    /// Minimum version of the reader to read the data
    pub ds_version: u16,
    /// DSBlock mode, 0 data stream, 1 data description
    pub ds_mode: u8,
    /// Reserved
    pub ds_reserved: [u8; 5],
}

/// Data stream mode constant - data layout described by CNBLOCK composition
pub const DS_MODE_DATA_STREAM: u8 = 0;
/// Data description mode constant - data layout described by external attachment (FIBEX, DBC, ARXML)
pub const DS_MODE_DATA_DESCRIPTION: u8 = 1;

impl Ds4Block {
    pub fn ds_cn_composition(&self) -> i64 {
        self.links.first().copied().unwrap_or(0)
    }
    pub fn ds_cn_alignment_start(&self) -> i64 {
        self.links.get(1).copied().unwrap_or(0)
    }
    pub fn ds_data(&self) -> i64 {
        self.links.get(2).copied().unwrap_or(0)
    }
    pub fn ds_md_comment(&self) -> i64 {
        self.links.get(3).copied().unwrap_or(0)
    }

    /// Returns true if this is data stream mode (`ds_mode = 0`).
    pub fn is_data_stream_mode(&self) -> bool {
        self.ds_mode == DS_MODE_DATA_STREAM
    }

    /// Returns true if this is data description mode (`ds_mode = 1`).
    pub fn is_data_description_mode(&self) -> bool {
        self.ds_mode == DS_MODE_DATA_DESCRIPTION
    }

    /// Returns a string description of the data stream mode.
    pub fn get_mode_str(&self) -> &'static str {
        match self.ds_mode {
            DS_MODE_DATA_STREAM => "Data Stream Mode",
            DS_MODE_DATA_DESCRIPTION => "Data Description Mode",
            _ => "Unknown Mode",
        }
    }
}

impl Display for Ds4Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DS: mode={} ({}) version={} links={}",
            self.get_mode_str(),
            self.ds_mode,
            self.ds_version,
            self.ds_links
        )
    }
}

/// CL4 Channel List block struct (Table 68).
#[derive(Debug, PartialEq, Eq, Default, Clone)]
#[binrw]
#[br(little)]
#[bw(little)]
#[repr(C)]
pub struct Cl4Block {
    /// # of links
    pub cl_links: u64,
    /// link to CNBlock describing dynamic data
    pub cl_composition: i64,
    /// link to CNBlock for the alignment start with data stream mode
    pub cl_cn_size: i64,
    /// Flags
    pub cl_flags: u16,
    /// Bytes alignment
    pub cl_alignment: u8,
    /// Bit Offset
    pub cl_bit_offset: u8,
    /// Byte Offset
    pub cl_byte_offset: u32,
}

impl Display for Cl4Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CL: flags=0x{:04X} alignment={} bit_offset={} byte_offset={}",
            self.cl_flags, self.cl_alignment, self.cl_bit_offset, self.cl_byte_offset
        )
    }
}

/// CV4 Channel Variant block struct (Table 69).
#[derive(Debug, PartialEq, Eq, Default, Clone)]
#[binrw]
#[br(little)]
#[bw(little)]
#[repr(C)]
pub struct Cv4Block {
    /// # of links
    pub cv_n_links: u64,
    /// link to CNBlock for discriminator channel
    pub cv_cn_discriminator: i64,
    /// list of option channel
    #[br(if(cv_n_links > 1), little, count = cv_n_links - 1)]
    pub cv_cn_option: Vec<i64>,
    /// number of option channels
    pub cv_option_count: u32,
    /// reserved
    pub cv_reserved: [u8; 4],
    /// list of discriminator values for the options
    #[br(if(cv_option_count > 1), little, count = cv_option_count)]
    pub cv_option_val: Vec<u64>,
}

impl Display for Cv4Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CV: {} options", self.cv_option_count)
    }
}

/// CU4 Channel Union block struct (Table 70).
#[derive(Debug, PartialEq, Eq, Default, Clone)]
#[binrw]
#[br(little)]
#[bw(little)]
#[repr(C)]
pub struct Cu4Block {
    /// # of links
    pub cu_n_links: u64,
    /// list of member channel
    #[br(if(cu_n_links > 1), little, count = cu_n_links)]
    pub cu_cn_member: Vec<i64>,
    /// number of member channels
    pub cu_member_count: u32,
    /// reserved
    pub cu_reserved: [u8; 4],
}

impl Display for Cu4Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CU: {} members", self.cu_member_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{format, vec};

    #[test]
    fn test_ds_cl_cv_cu() {
        let ds = Ds4Block {
            ds_links: 4,
            links: vec![100, 200, 300, 400],
            ds_version: 430,
            ds_mode: DS_MODE_DATA_STREAM,
            ds_reserved: [0; 5],
        };
        assert_eq!(ds.ds_cn_composition(), 100);
        assert!(ds.is_data_stream_mode());
        assert!(format!("{ds}").contains("Data Stream Mode"));

        let cl = Cl4Block {
            cl_flags: 3,
            cl_alignment: 4,
            ..Default::default()
        };
        assert!(format!("{cl}").contains("alignment=4"));

        let cv = Cv4Block {
            cv_option_count: 3,
            ..Default::default()
        };
        assert!(format!("{cv}").contains("3 options"));

        let cu = Cu4Block {
            cu_member_count: 5,
            ..Default::default()
        };
        assert!(format!("{cu}").contains("5 members"));
    }
}

