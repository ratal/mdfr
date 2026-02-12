//! Data blocks (DT, DL, DZ, LD, HL, GD) for MDF4 — spec sections 6.12-6.17, Tables 55-63
//!
//! These blocks store raw channel data. DTBLOCK holds plain data, DZBLOCK holds
//! compressed data (zlib/LZ4/zstd, transposition), DLBLOCK chains data blocks,
//! HLBLOCK is a header list for DZBLOCK chains, LDBLOCK is a list for sorted data,
//! and GDBLOCK guards MDF 4.3 features in unsorted data.
use anyhow::{Context, Result, bail};
use binrw::{BinReaderExt, binrw};
use flate2::read::ZlibDecoder;
use log::warn;
use lz4::Decoder as Lz4Decoder;
use std::fmt::{self, Display};
use std::fs::File;
use std::io::{BufReader, Cursor, Read};
use std::str;
use zstd::Decoder as ZstdDecoder;

/// DTBLOCK structure (MDF 4.2 spec, Table 55) — plain data storage
#[derive(Debug, PartialEq, Eq, Default, Clone)]
#[binrw]
#[br(little)]
#[repr(C)]
pub struct Dt4Block {
    //header
    // dl_id: [u8; 4],  // ##DL
    /// reserved
    reserved: [u8; 4],
    /// Length of block in bytes
    pub len: u64,
    /// # of links
    links: u64,
}

impl Display for Dt4Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DT: len={}", self.len)
    }
}

/// DLBLOCK structure (MDF 4.2 spec, Table 56) — chains data blocks
#[derive(Debug, PartialEq, Eq, Default, Clone)]
#[binrw]
#[br(little)]
#[repr(C)]
pub struct Dl4Block {
    //header
    // dl_id: [u8; 4],  // ##DL
    /// reserved
    reserved: [u8; 4],
    /// Length of block in bytes
    dl_len: u64,
    /// # of links
    dl_links: u64,
    // links
    /// next DL
    pub dl_dl_next: i64,
    #[br(if(dl_links > 1), little, count = dl_links - 1)]
    pub dl_data: Vec<i64>,
    // members
    /// Flags
    dl_flags: u8,
    dl_reserved: [u8; 3],
    /// Number of data blocks
    dl_count: u32,
    #[br(if((dl_flags & 0b1)>0), little)]
    dl_equal_length: Option<u64>,
    #[br(if((dl_flags & 0b1)==0), little, count = dl_count)]
    dl_offset: Vec<u64>,
    #[br(if((dl_flags & 0b10)>0), little, count = dl_count)]
    dl_time_values: Vec<i64>,
    #[br(if((dl_flags & 0b100)>0), little, count = dl_count)]
    dl_angle_values: Vec<i64>,
    #[br(if((dl_flags & 0b1000)>0), little, count = dl_count)]
    dl_distance_values: Vec<i64>,
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

/// parses Data List block
/// pointing to DT, SD, RD or DZ blocks
pub fn parser_dl4_block(
    rdr: &mut BufReader<&File>,
    target: i64,
    mut position: i64,
) -> Result<(Dl4Block, i64)> {
    rdr.seek_relative(target - position)
        .context("Could not reach position to read Dl4Block")?;
    let block: Dl4Block = rdr
        .read_le()
        .context("Could not read into Dl4Block struct")?;
    position = target + block.dl_len as i64;
    Ok((block, position))
}

/// Helper function to decompress data using various algorithms
pub fn decompress_data(
    zip_type: u8,
    zip_parameter: u32,
    buf: Vec<u8>,
    org_data_length: u64,
) -> Result<Vec<u8>> {
    let mut data = Vec::<u8>::new();
    match zip_type {
        0 | 1 => {
            // deflate algorithm (zlib format)
            let reader = Cursor::new(buf);
            let mut decoder = ZlibDecoder::new(reader);
            decoder
                .read_to_end(&mut data)
                .context("Error decompressing Deflate data")?;
        }
        2 | 3 => {
            // zstd algorithm
            let reader = Cursor::new(buf);
            let mut decoder =
                ZstdDecoder::new(reader).context("Error creating Zstd decoder from read vector")?;
            let _nbbytesread = decoder
                .read_to_end(&mut data)
                .context("error reading the compressed bytes")?;
        }
        4 | 5 => {
            // lz4 algorithm
            let reader = Cursor::new(buf);
            let mut decoder =
                Lz4Decoder::new(reader).context("Error creating Lz4 decoder from read vector")?;
            let _nbbytesread = decoder
                .read_to_end(&mut data)
                .context("error reading the compressed bytes")?;
        }
        254 => {
            // MDF 4.3 custom/vendor-specific compression
            warn!("Custom compression (zip_type=254) not supported - data will be empty");
            return Ok(data);
        }
        _ => {
            bail!("not implemented compression algorithm: {}", zip_type)
        }
    };
    // transpose data
    if matches!(zip_type, 1 | 3 | 5) && zip_parameter > 0 {
        // transpose
        let m = org_data_length / zip_parameter as u64;
        let tail: Vec<u8> = data.split_off((m * zip_parameter as u64) as usize);
        let mut output = vec![0u8; (m * zip_parameter as u64) as usize];
        transpose::transpose(&data, &mut output, m as usize, zip_parameter as usize);
        data = output;
        if !tail.is_empty() {
            data.extend(tail);
        }
    }
    Ok(data)
}

/// parses DZBlock
pub fn parse_dz(rdr: &mut BufReader<&File>) -> Result<(Vec<u8>, Dz4Block)> {
    let mut block: Dz4Block = rdr
        .read_le()
        .context("Could not read into Dz4Block struct")?;
    let mut buf = vec![0u8; block.dz_data_length as usize];
    rdr.read_exact(&mut buf).context("Could not read Dz data")?;
    // decompress data
    let data = decompress_data(
        block.dz_zip_type,
        block.dz_zip_parameter,
        buf,
        block.dz_org_data_length,
    )?;
    block.dz_org_data_length = data.len() as u64;
    Ok((data, block))
}

/// DZBLOCK structure (MDF 4.2 spec, Table 57) — compressed data block
#[derive(Debug, PartialEq, Eq, Clone)]
#[binrw]
#[br(little)]
#[repr(C)]
pub struct Dz4Block {
    //header
    // dz_id: [u8; 4],  // ##DZ
    reserved: [u8; 4], // reserved
    /// Length of block in bytes
    pub len: u64,
    dz_links: u64, // # of links
    // links
    // members
    /// "DT", "SD", "RD" or "DV", "DI", "RV", "RI"
    pub dz_org_block_type: [u8; 2],
    /// Zip algorithm, 0 deflate, 1 transpose + deflate
    dz_zip_type: u8,
    /// reserved
    dz_reserved: u8,
    /// Zip algorithm parameter
    dz_zip_parameter: u32, //
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
            dz_zip_type: 0,              // No transposition for a single channel
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
        let org_type =
            str::from_utf8(&self.dz_org_block_type).unwrap_or("??");
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

/// LDBLOCK structure (MDF 4.2 spec, Table 61) — list data block for sorted data
#[derive(Debug, PartialEq, Eq, Clone)]
#[binrw]
#[br(little)]
#[repr(C)]
pub struct Ld4Block {
    // header
    // ld_id: [u8; 4],  // ##LD
    reserved: [u8; 4], // reserved
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
    /// Zip info in ivalid data
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
    dl_time_values: Vec<i64>,
    #[br(if((ld_flags & 0b100)>0), little, count = ld_count)]
    dl_angle_values: Vec<i64>,
    #[br(if((ld_flags & 0b1000)>0), little, count = ld_count)]
    dl_distance_values: Vec<i64>,
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
    pub fn ld_ld_next(&self) -> i64 {
        self.ld_next
    }
    /// Data block positions
    pub fn ld_data(&self) -> Vec<i64> {
        // In MDF 4.3, bit 7 of ld_flags_ext indicates invalid data present.
        // If present, links are interleaved: Data 1, Inval 1, Data 2, Inval 2, ...
        // We can also check if the number of links matches 2 * ld_count.
        if (1u8 << 7) & self.ld_flags_ext > 0 || self.ld_links.len() as u32 == self.ld_count * 2 {
            self.ld_links.iter().step_by(2).copied().collect()
        } else {
            self.ld_links.clone()
        }
    }
    /// Invalid data block positions
    pub fn ld_invalid_data(&self) -> Vec<i64> {
        if (1u8 << 7) & self.ld_flags_ext > 0 || self.ld_links.len() as u32 == self.ld_count * 2 {
            self.ld_links.iter().skip(1).step_by(2).copied().collect()
        } else {
            Vec::<i64>::new()
        }
    }
}

impl Display for Ld4Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sample_count = self
            .ld_equal_sample_count
            .map(|c| format!("{}", c))
            .unwrap_or_else(|| format!("{} offsets", self.ld_sample_offset.len()));
        write!(
            f,
            "LD: {} blocks sample_count={} flags=0x{:02X}",
            self.ld_count, sample_count, self.ld_flags
        )
    }
}

/// parse List Data block
/// equivalent ot DLBlock but unsorted data is not allowed
/// pointing to DV/DI and RV/RI blocks
pub fn parser_ld4_block(
    rdr: &mut BufReader<&File>,
    target: i64,
    mut position: i64,
) -> Result<(Ld4Block, i64)> {
    rdr.seek_relative(target - position)
        .context("Could not reach Ld4Block position")?;
    let block: Ld4Block = rdr
        .read_le()
        .context("Could not read buffer into Ld4Block struct")?;
    position = target + block.ld_len as i64;
    Ok((block, position))
}

/// HLBLOCK structure (MDF 4.2 spec, Table 58) — header list for DZBLOCK chains
#[derive(Debug, PartialEq, Eq, Default, Clone)]
#[binrw]
#[br(little)]
#[repr(C)]
pub struct Hl4Block {
    //header
    // ##HL
    // hl_id: [u8; 4],
    /// reserved
    reserved: [u8; 4],
    /// Length of block in bytes
    pub hl_len: u64,
    /// # of links
    hl_links: u64,
    /// links
    pub hl_dl_first: i64, // first LD block
    // members
    /// flags
    hl_flags: u16,
    /// Zip algorithn
    hl_zip_type: u8,
    /// reserved
    hl_reserved: [u8; 5],
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

/// GDBLOCK structure (MDF 4.3 spec, Table 63) — guards 4.3 features in unsorted data
#[derive(Debug, PartialEq, Eq, Default, Clone)]
#[binrw]
#[br(little)]
#[repr(C)]
pub struct Gd4Block {
    // header
    // ##GD
    // gd_id: [u8; 4],
    /// reserved
    reserved: [u8; 4],
    /// Length of block in bytes
    pub gd_len: u64,
    /// # of links (always 1)
    gd_links: u64,
    // link section
    /// Pointer to the guarded block (shall not be NIL)
    pub gd_link: i64,
    // data section
    /// Minimum version number of the MDF format the reader shall support
    /// Same format as id_ver in IDBLOCK, i.e. 430 for MDF 4.3.0
    pub gd_version: u16,
    // gd_reserved is not included - size varies, position handled manually
}

impl Display for Gd4Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let major = self.gd_version / 100;
        let minor = (self.gd_version % 100) / 10;
        let patch = self.gd_version % 10;
        write!(f, "GD: min_version={}.{}.{}", major, minor, patch)
    }
}
