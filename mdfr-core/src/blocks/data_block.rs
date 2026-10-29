//! Data blocks (DT, DL, DZ, LD, HL, GD) for MDF4 — spec sections 6.12-6.17, Tables 55-63.
//!
//! Pure block definitions (no file I/O, no decompression).
use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use binrw::binrw;
use core::fmt::{self, Display};
use core::str;

/// DTBLOCK structure (MDF 4.2 spec, Table 55) — plain data storage (after 4-byte `##DT` ID).
#[derive(Debug, PartialEq, Eq, Default, Clone)]
#[binrw]
#[br(little)]
#[bw(little)]
#[repr(C)]
pub struct Dt4Block {
    /// reserved
    pub reserved: [u8; 4],
    /// Length of block in bytes
    pub len: u64,
    /// # of links
    pub links: u64,
}

impl Display for Dt4Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DT: len={}", self.len)
    }
}

/// DLBLOCK structure (MDF 4.2 spec, Table 56) — chains data blocks (after 4-byte `##DL` ID).
#[derive(Debug, PartialEq, Eq, Default, Clone)]
#[binrw]
#[br(little)]
#[bw(little)]
#[repr(C)]
pub struct Dl4Block {
    /// reserved
    pub reserved: [u8; 4],
    /// Length of block in bytes
    pub dl_len: u64,
    /// # of links
    pub dl_links: u64,
    // links
    /// next DL
    pub dl_dl_next: i64,
    #[br(if(dl_links > 1), little, count = dl_links - 1)]
    pub dl_data: Vec<i64>,
    // members
    /// Flags
    pub dl_flags: u8,
    pub dl_reserved: [u8; 3],
    /// Number of data blocks
    pub dl_count: u32,
    #[br(if((dl_flags & 0b1)>0), little)]
    pub dl_equal_length: Option<u64>,
    #[br(if((dl_flags & 0b1)==0), little, count = dl_count)]
    pub dl_offset: Vec<u64>,
    #[br(if((dl_flags & 0b10)>0), little, count = dl_count)]
    pub dl_time_values: Vec<i64>,
    #[br(if((dl_flags & 0b100)>0), little, count = dl_count)]
    pub dl_angle_values: Vec<i64>,
    #[br(if((dl_flags & 0b1000)>0), little, count = dl_count)]
    pub dl_distance_values: Vec<i64>,
}

impl Display for Dl4Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DL: {} data blocks flags=0x{:02X}",
            self.dl_count, self.dl_flags
        )
    }
}

/// DZBLOCK structure (MDF 4.2 spec, Table 57) — compressed data block (after 4-byte `##DZ` ID).
#[derive(Debug, PartialEq, Eq, Clone)]
#[binrw]
#[br(little)]
#[bw(little)]
#[repr(C)]
pub struct Dz4Block {
    /// reserved
    pub reserved: [u8; 4],
    /// Length of block in bytes
    pub len: u64,
    /// # of links
    pub dz_links: u64,
    // members
    /// "DT", "SD", "RD" or "DV", "DI", "RV", "RI"
    pub dz_org_block_type: [u8; 2],
    /// Zip algorithm, 0 deflate, 1 transpose + deflate, 2 zstd, 3 transpose+zstd, 4 lz4, 5 transpose+lz4
    pub dz_zip_type: u8,
    /// reserved
    pub dz_reserved: u8,
    /// Zip algorithm parameter
    pub dz_zip_parameter: u32,
    /// length of uncompressed data
    pub dz_org_data_length: u64,
    /// length of compressed data
    pub dz_data_length: u64,
}

impl Default for Dz4Block {
    fn default() -> Self {
        Dz4Block {
            reserved: [0; 4],
            len: 0,
            dz_links: 0,
            dz_org_block_type: [68, 86], // DV
            dz_zip_type: 0,
            dz_reserved: 0,
            dz_zip_parameter: 0,
            dz_org_data_length: 0,
            dz_data_length: 0,
        }
    }
}

impl Dz4Block {
    /// Returns a string representation of the compression algorithm
    pub fn get_compression_str(&self) -> &'static str {
        match self.dz_zip_type {
            0 => "Deflate",
            1 => "Deflate+Transpose",
            2 => "Zstd",
            3 => "Zstd+Transpose",
            4 => "LZ4",
            5 => "LZ4+Transpose",
            254 => "Custom",
            _ => "Unknown",
        }
    }
}

impl Display for Dz4Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let org_type = str::from_utf8(&self.dz_org_block_type).unwrap_or("??");
        let ratio = if self.dz_org_data_length > 0 {
            (self.dz_data_length as f64 / self.dz_org_data_length as f64) * 100.0
        } else {
            0.0
        };
        write!(
            f,
            "DZ: {} org_type={} compressed={} original={} ratio={:.1}%",
            self.get_compression_str(),
            org_type,
            self.dz_data_length,
            self.dz_org_data_length,
            ratio
        )
    }
}

/// LDBLOCK structure (MDF 4.2 spec, Table 61) — list data block for sorted data (after 4-byte `##LD` ID).
#[derive(Debug, PartialEq, Eq, Clone)]
#[binrw]
#[br(little)]
#[bw(little)]
#[repr(C)]
pub struct Ld4Block {
    /// reserved
    pub reserved: [u8; 4],
    /// Length of block in bytes
    pub ld_len: u64,
    /// # of links
    pub ld_n_links: u64,
    // links
    /// next ld block
    pub ld_next: i64,
    /// number of links
    #[br(if(ld_n_links > 1), little, count = ld_n_links - 1)]
    pub ld_links: Vec<i64>,
    // members
    /// Flags
    pub ld_flags: u8,
    /// Zip info in valid data
    pub ld_zip_info: u8,
    /// Zip info in invalid data
    pub ld_zip_info_inval: u8,
    /// Extended flags
    pub ld_flags_ext: u8,
    /// Number of data blocks
    pub ld_count: u32,
    #[br(if((ld_flags & 0b1)!=0), little)]
    pub ld_equal_sample_count: Option<u64>,
    #[br(if((ld_flags & 0b1)==0), little, count = ld_count)]
    pub ld_sample_offset: Vec<u64>,
    #[br(if((ld_flags & 0b10)>0), little, count = ld_count)]
    pub dl_time_values: Vec<i64>,
    #[br(if((ld_flags & 0b100)>0), little, count = ld_count)]
    pub dl_angle_values: Vec<i64>,
    #[br(if((ld_flags & 0b1000)>0), little, count = ld_count)]
    pub dl_distance_values: Vec<i64>,
}

impl Default for Ld4Block {
    fn default() -> Self {
        Ld4Block {
            reserved: [0; 4],
            ld_len: 56,
            ld_n_links: 2,
            ld_next: 0,
            ld_links: vec![],
            ld_flags: 0,
            ld_zip_info: 0,
            ld_zip_info_inval: 0,
            ld_flags_ext: 0,
            ld_count: 1,
            ld_equal_sample_count: None,
            ld_sample_offset: vec![],
            dl_time_values: vec![],
            dl_angle_values: vec![],
            dl_distance_values: vec![],
        }
    }
}

impl Ld4Block {
    /// Next LD block link.
    pub fn ld_ld_next(&self) -> i64 {
        self.ld_next
    }
    /// Data block positions
    pub fn ld_data(&self) -> Vec<i64> {
        let n = self.ld_count as usize;
        if ((1u8 << 7) & self.ld_flags_ext > 0 || self.ld_links.len() == n * 2)
            && self.ld_links.len() >= n
        {
            self.ld_links[..n].to_vec()
        } else {
            self.ld_links.clone()
        }
    }
    /// Invalid data block positions
    pub fn ld_invalid_data(&self) -> Vec<i64> {
        let n = self.ld_count as usize;
        if ((1u8 << 7) & self.ld_flags_ext > 0 || self.ld_links.len() == n * 2)
            && self.ld_links.len() >= n * 2
        {
            self.ld_links[n..n * 2].to_vec()
        } else {
            Vec::<i64>::new()
        }
    }
}

impl Display for Ld4Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sample_count = self
            .ld_equal_sample_count
            .map(|c| format!("{c}"))
            .unwrap_or_else(|| format!("{} offsets", self.ld_sample_offset.len()));
        write!(
            f,
            "LD: {} blocks sample_count={} flags=0x{:02X}",
            self.ld_count, sample_count, self.ld_flags
        )
    }
}

/// HLBLOCK structure (MDF 4.2 spec, Table 58) — header list for DZBLOCK chains (after 4-byte `##HL` ID).
#[derive(Debug, PartialEq, Eq, Default, Clone)]
#[binrw]
#[br(little)]
#[bw(little)]
#[repr(C)]
pub struct Hl4Block {
    /// reserved
    pub reserved: [u8; 4],
    /// Length of block in bytes
    pub hl_len: u64,
    /// # of links
    pub hl_links: u64,
    /// first DL block
    pub hl_dl_first: i64,
    // members
    /// flags
    pub hl_flags: u16,
    /// Zip algorithm
    pub hl_zip_type: u8,
    /// reserved
    pub hl_reserved: [u8; 5],
}

impl Hl4Block {
    /// Returns a string representation of the compression algorithm
    pub fn get_zip_type_str(&self) -> &'static str {
        match self.hl_zip_type {
            0 => "Deflate",
            1 => "Deflate+Transpose",
            2 => "Zstd",
            3 => "Zstd+Transpose",
            4 => "LZ4",
            5 => "LZ4+Transpose",
            _ => "Unknown",
        }
    }
}

impl Display for Hl4Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HL: flags=0x{:04X} zip_type={}",
            self.hl_flags,
            self.get_zip_type_str()
        )
    }
}

/// GDBLOCK structure (MDF 4.3 spec, Table 63) — guards 4.3 features in unsorted data (after 4-byte `##GD` ID).
#[derive(Debug, PartialEq, Eq, Default, Clone)]
#[binrw]
#[br(little)]
#[bw(little)]
#[repr(C)]
pub struct Gd4Block {
    /// reserved
    pub reserved: [u8; 4],
    /// Length of block in bytes
    pub gd_len: u64,
    /// # of links (always 1)
    pub gd_links: u64,
    // link section
    /// Pointer to the guarded block (shall not be NIL)
    pub gd_link: i64,
    // data section
    /// Minimum version number of the MDF format the reader shall support
    pub gd_version: u16,
}

impl Display for Gd4Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let major = self.gd_version / 100;
        let minor = (self.gd_version % 100) / 10;
        let patch = self.gd_version % 10;
        write!(f, "GD: min_version={major}.{minor}.{patch}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use binrw::{BinReaderExt, BinWrite, io::Cursor};

    #[test]
    fn roundtrip_dl() {
        let dl = Dl4Block {
            dl_len: 56,
            dl_links: 2,
            dl_dl_next: 0,
            dl_data: vec![0x200],
            dl_flags: 0b1,
            dl_reserved: [0; 3],
            dl_count: 1,
            dl_equal_length: Some(1024),
            dl_offset: vec![],
            dl_time_values: vec![],
            dl_angle_values: vec![],
            dl_distance_values: vec![],
            reserved: [0; 4],
        };
        let mut c = Cursor::new(Vec::new());
        dl.write_le(&mut c).unwrap();
        let buf = c.into_inner();
        assert_eq!(4 + buf.len(), 56);
        let back: Dl4Block = Cursor::new(buf).read_le().unwrap();
        assert_eq!(back, dl);
    }

    #[test]
    fn roundtrip_dz_hl() {
        let dz = Dz4Block {
            len: 48,
            dz_org_data_length: 1000,
            dz_data_length: 400,
            ..Default::default()
        };
        let mut c = Cursor::new(Vec::new());
        dz.write_le(&mut c).unwrap();
        let buf = c.into_inner();
        assert_eq!(4 + buf.len(), 48);
        let back: Dz4Block = Cursor::new(buf).read_le().unwrap();
        assert_eq!(back, dz);

        let hl = Hl4Block {
            hl_len: 40,
            hl_links: 1,
            hl_dl_first: 0x100,
            hl_flags: 1,
            hl_zip_type: 1,
            ..Default::default()
        };
        let mut c = Cursor::new(Vec::new());
        hl.write_le(&mut c).unwrap();
        let buf = c.into_inner();
        assert_eq!(4 + buf.len(), 40);
        let back: Hl4Block = Cursor::new(buf).read_le().unwrap();
        assert_eq!(back, hl);
    }

    #[test]
    fn test_dz_get_compression_str() {
        let mut dz = Dz4Block::default();
        let expected = [
            (0, "Deflate"),
            (1, "Deflate+Transpose"),
            (2, "Zstd"),
            (3, "Zstd+Transpose"),
            (4, "LZ4"),
            (5, "LZ4+Transpose"),
            (254, "Custom"),
            (255, "Unknown"),
        ];
        for (val, name) in expected {
            dz.dz_zip_type = val;
            assert_eq!(dz.get_compression_str(), name, "dz_zip_type={val}");
        }
    }

    #[test]
    fn test_ld_data_and_invalidation() {
        let ld = Ld4Block {
            ld_count: 2,
            ld_flags_ext: 1u8 << 7,
            ld_links: vec![100, 200, 101, 201],
            ..Default::default()
        };
        assert_eq!(ld.ld_data(), vec![100, 200]);
        assert_eq!(ld.ld_invalid_data(), vec![101, 201]);
    }

    #[test]
    fn test_hl_get_zip_type_str() {
        let expected = [
            (0, "Deflate"),
            (1, "Deflate+Transpose"),
            (2, "Zstd"),
            (3, "Zstd+Transpose"),
            (4, "LZ4"),
            (5, "LZ4+Transpose"),
            (255, "Unknown"),
        ];
        for (val, name) in expected {
            let hl = Hl4Block {
                hl_zip_type: val,
                ..Default::default()
            };
            assert_eq!(hl.get_zip_type_str(), name, "hl_zip_type={val}");
        }
    }
}
