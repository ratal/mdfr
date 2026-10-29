//! Schema builder and frozen layout computation for the MDF 4.3 logger.
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use crate::blocks::cn::{CnChannelType, CnDataType, CnSyncType};
use crate::compress::CompressionAlgorithm;
use crate::error::{Error, Result};

/// On-disk data layout strategy for the logger.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub enum DataLayout {
    /// Sorted: one DGBLOCK per channel group, each with its own DT/DZ block chain (via DL/HL).
    /// Highest read speed in `mdfr` (no sorting pass required).
    #[default]
    Sorted,
    /// Unsorted: single DGBLOCK containing all channel groups, distinguished by a 1- or 2-byte
    /// record ID prefix in a single shared DT/DZ block chain. Lowest RAM footprint on MCU.
    Unsorted,
}

/// Opaque handle to a channel group in a [`Schema`] / [`Logger`](super::Logger).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct GroupHandle(pub u16);

/// Opaque handle to a channel within a group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct ChannelHandle {
    /// Group index (`0..groups.len()`).
    pub group: u16,
    /// Channel index within the group (`0..group.channels.len()`, where `0` is always the master time channel).
    pub channel: u16,
}

/// Conversion rule attached to a channel.
#[derive(Debug, Clone, PartialEq)]
pub enum ConversionDef {
    /// 1:1 identity (`cc_type = 0`).
    Identity,
    /// Linear `phys = p1 + p2 * int` (`cc_type = 1`).
    Linear {
        /// Offset `P1`.
        p1: f64,
        /// Factor `P2`.
        p2: f64,
    },
    /// Rational `phys = (p1*x^2 + p2*x + p3) / (p4*x^2 + p5*x + p6)` (`cc_type = 2`).
    Rational([f64; 6]),
    /// Value-to-text lookup table (`cc_type = 7`): pairs of `(key, text)` plus a default text.
    ValueToText {
        /// `(raw_value, label)` pairs.
        entries: Vec<(f64, String)>,
        /// Fallback label when no key matches.
        default_text: String,
    },
}

/// Source information (`SIBLOCK`) description.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceDef {
    /// Source name (`si_tx_name`).
    pub name: String,
    /// Optional source path (`si_tx_path`).
    pub path: Option<String>,
    /// `si_type`: 0=Other, 1=ECU, 2=Bus, 3=I/O, 4=Tool, 5=User.
    pub si_type: u8,
    /// `si_bus_type`: 0=None, 1=Other, 2=CAN, 3=LIN, 4=MOST, 5=FlexRay, 6=K-Line, 7=Ethernet, 8=USB.
    pub si_bus_type: u8,
    /// `si_flags`.
    pub si_flags: u8,
}

/// Channel description inside a [`GroupDef`].
#[derive(Debug, Clone)]
pub struct ChannelDef {
    /// Channel name (`cn_tx_name`).
    pub name: String,
    /// Physical unit (`cn_md_unit`), if any.
    pub unit: Option<String>,
    /// Optional description (`cn_md_comment`).
    pub comment: Option<String>,
    /// MDF4 channel type (`FixedLength`, `Master`, `Vlsd`, `Vlsc`, ...).
    pub cn_type: CnChannelType,
    /// Sync type (`Time` for master time channel, `None` for normal signals).
    pub sync_type: CnSyncType,
    /// Raw signal data type.
    pub data_type: CnDataType,
    /// Bit offset (`0..=7`) from `byte_offset`.
    pub bit_offset: u8,
    /// Byte offset from the start of the record (excluding any unsorted record ID).
    pub byte_offset: u32,
    /// Signal width in bits (`cn_bit_count`).
    pub bit_count: u32,
    /// Optional `CN_F_*` flags.
    pub flags: u32,
    /// Optional conversion rule.
    pub conversion: Option<ConversionDef>,
    /// Optional source info.
    pub source: Option<SourceDef>,
    /// Optional 1D/ND array dimensions (`CABLOCK` CN-template storage).
    pub array_dims: Option<Vec<u64>>,
    /// For VLSD channels in sorted mode, accumulated SD bytes are stored per channel;
    /// in unsorted mode or VLSC mode, links to companion VLSD CG / size channel.
    pub is_vlsd: bool,
    /// For MDF 4.3 `VLSC` channels (`CnChannelType::Vlsc`), holds the channel index within this group
    /// of the companion size channel referenced by the 9th link (`cn_cn_size`).
    pub vlsc_size_channel: Option<u16>,
}

/// Channel group definition.
#[derive(Debug, Clone)]
pub struct GroupDef {
    /// Group acquisition name (`cg_tx_acq_name`).
    pub name: String,
    /// Optional comment (`cg_md_comment`).
    pub comment: Option<String>,
    /// Optional group source (`cg_si_acq_source`).
    pub source: Option<SourceDef>,
    /// `CG_F_*` flags.
    pub flags: u16,
    /// Path separator character (e.g. `b'.' as u16`), or 0.
    pub path_separator: u16,
    /// Channels in this group. Channel 0 is the master time channel (64-bit unsigned LE ticks).
    pub channels: Vec<ChannelDef>,
    /// Total plain data bytes per record (`cg_data_bytes`).
    pub record_data_bytes: u32,
    /// Buffer flush threshold in bytes for this group (defaults to 32 KiB).
    pub flush_threshold_bytes: usize,
}

/// Complete frozen logger schema + configuration.
#[derive(Debug, Clone)]
pub struct Schema {
    /// Sorted or Unsorted layout.
    pub layout: DataLayout,
    /// Compression algorithm for data blocks (defaults to [`CompressionAlgorithm::DeflateTranspose`] when `deflate` is enabled, else `NoCompression`).
    pub compression: CompressionAlgorithm,
    /// Maximum number of data blocks referenced per `DLBLOCK` before chaining a new `DLBLOCK`.
    pub max_blocks_per_dl: usize,
    /// Automatic checkpoint interval in flushed data blocks (`0` = only on explicit `checkpoint()` / `close()`).
    pub checkpoint_every_n_blocks: usize,
    /// Defined channel groups.
    pub groups: Vec<GroupDef>,
}

/// Fluent builder for constructing a [`Schema`].
#[derive(Debug, Clone)]
pub struct LoggerBuilder {
    layout: DataLayout,
    compression: CompressionAlgorithm,
    max_blocks_per_dl: usize,
    checkpoint_every_n_blocks: usize,
    groups: Vec<GroupDef>,
}

impl LoggerBuilder {
    /// Creates a new builder with the given layout and compression algorithm.
    pub fn new(layout: DataLayout, compression: CompressionAlgorithm) -> Self {
        Self {
            layout,
            compression,
            max_blocks_per_dl: 64,
            checkpoint_every_n_blocks: 0,
            groups: Vec::new(),
        }
    }

    /// Sets the flush buffer threshold (in bytes) for newly added groups or overrides a specific group.
    pub fn set_group_flush_threshold(
        &mut self,
        group: GroupHandle,
        threshold_bytes: usize,
    ) -> Result<&mut Self> {
        let g = self
            .groups
            .get_mut(group.0 as usize)
            .ok_or_else(|| Error::InvalidSchema(format!("unknown group {}", group.0)))?;
        g.flush_threshold_bytes = threshold_bytes.max(64);
        Ok(self)
    }

    /// Sets the maximum number of data blocks per `DLBLOCK` before chaining a new `DLBLOCK`.
    pub fn max_blocks_per_dl(&mut self, max_blocks: usize) -> &mut Self {
        self.max_blocks_per_dl = max_blocks.max(1);
        self
    }

    /// Enables automatic `checkpoint()` every `n` flushed data blocks (`0` disables auto-checkpoint).
    pub fn checkpoint_every_n_blocks(&mut self, n: usize) -> &mut Self {
        self.checkpoint_every_n_blocks = n;
        self
    }

    /// Adds a new channel group with an automatic 64-bit unsigned tick master time channel (`"time"`, byte offset `0..8`).
    pub fn add_group(&mut self, name: &str) -> GroupHandle {
        self.add_group_with_master(name, "time")
    }

    /// Adds a new channel group with a custom name for its 64-bit tick master time channel.
    pub fn add_group_with_master(&mut self, name: &str, master_name: &str) -> GroupHandle {
        let idx = self.groups.len() as u16;
        let master = ChannelDef {
            name: master_name.to_string(),
            unit: Some("s".to_string()),
            comment: None,
            cn_type: CnChannelType::Master,
            sync_type: CnSyncType::Time,
            data_type: CnDataType::UIntLE,
            bit_offset: 0,
            byte_offset: 0,
            bit_count: 64,
            flags: 0,
            // Linear conversion P1=0.0, P2=seconds_per_tick is filled in from TimeSource at start()
            conversion: None,
            source: None,
            array_dims: None,
            is_vlsd: false,
            vlsc_size_channel: None,
        };
        self.groups.push(GroupDef {
            name: name.to_string(),
            comment: None,
            source: None,
            flags: 0,
            path_separator: 0,
            channels: vec![master],
            record_data_bytes: 8,
            flush_threshold_bytes: 32 * 1024,
        });
        GroupHandle(idx)
    }

    /// Registers a typed `#[derive(MdfRecord)]` struct as a new channel group using `R::GROUP_NAME`.
    pub fn add_record_group<R: super::MdfRecord>(&mut self) -> Result<GroupHandle> {
        self.add_record_group_named::<R>(R::GROUP_NAME)
    }

    /// Registers a typed `#[derive(MdfRecord)]` struct as a new channel group with a custom group name.
    pub fn add_record_group_named<R: super::MdfRecord>(
        &mut self,
        name: &str,
    ) -> Result<GroupHandle> {
        let g = self.add_group(name);
        R::register_channels(self, g)?;
        Ok(g)
    }

    /// Sets the source info (`SIBLOCK`) for a channel group.
    pub fn set_group_source(&mut self, group: GroupHandle, source: SourceDef) -> Result<&mut Self> {
        let g = self
            .groups
            .get_mut(group.0 as usize)
            .ok_or_else(|| Error::InvalidSchema(format!("unknown group {}", group.0)))?;
        g.source = Some(source);
        Ok(self)
    }

    /// Appends a byte-aligned scalar channel to `group`, automatically advancing the group's record byte length.
    pub fn add_channel(
        &mut self,
        group: GroupHandle,
        name: &str,
        data_type: CnDataType,
        bit_count: u32,
    ) -> Result<ChannelHandle> {
        let g = self
            .groups
            .get_mut(group.0 as usize)
            .ok_or_else(|| Error::InvalidSchema(format!("unknown group {}", group.0)))?;
        if bit_count == 0 {
            return Err(Error::InvalidSchema(format!(
                "channel '{name}' has bit_count=0"
            )));
        }
        let byte_offset = g.record_data_bytes;
        let n_bytes = bit_count.div_ceil(8);
        g.record_data_bytes = byte_offset + n_bytes;
        let ch_idx = g.channels.len() as u16;
        g.channels.push(ChannelDef {
            name: name.to_string(),
            unit: None,
            comment: None,
            cn_type: CnChannelType::FixedLength,
            sync_type: CnSyncType::None,
            data_type,
            bit_offset: 0,
            byte_offset,
            bit_count,
            flags: 0,
            conversion: None,
            source: None,
            array_dims: None,
            is_vlsd: false,
            vlsc_size_channel: None,
        });
        Ok(ChannelHandle {
            group: group.0,
            channel: ch_idx,
        })
    }

    /// Adds a bit-packed channel at an explicit `(byte_offset, bit_offset, bit_count)` inside `group`'s record.
    /// Automatically expands `record_data_bytes` if the bit range extends past the current record end.
    pub fn add_bit_channel(
        &mut self,
        group: GroupHandle,
        name: &str,
        data_type: CnDataType,
        byte_offset: u32,
        bit_offset: u8,
        bit_count: u32,
    ) -> Result<ChannelHandle> {
        if bit_offset > 7 || bit_count == 0 || (bit_offset as u32 + bit_count) > 64 {
            return Err(Error::InvalidSchema(format!(
                "invalid bit channel '{name}': byte_offset={byte_offset} bit_offset={bit_offset} bit_count={bit_count}"
            )));
        }
        if byte_offset < 8 {
            return Err(Error::InvalidSchema(format!(
                "bit channel '{name}' overlaps master time channel (bytes 0..8)"
            )));
        }
        let g = self
            .groups
            .get_mut(group.0 as usize)
            .ok_or_else(|| Error::InvalidSchema(format!("unknown group {}", group.0)))?;
        let end_byte = byte_offset + (bit_offset as u32 + bit_count).div_ceil(8);
        if end_byte > g.record_data_bytes {
            g.record_data_bytes = end_byte;
        }
        let ch_idx = g.channels.len() as u16;
        g.channels.push(ChannelDef {
            name: name.to_string(),
            unit: None,
            comment: None,
            cn_type: CnChannelType::FixedLength,
            sync_type: CnSyncType::None,
            data_type,
            bit_offset,
            byte_offset,
            bit_count,
            flags: 0,
            conversion: None,
            source: None,
            array_dims: None,
            is_vlsd: false,
            vlsc_size_channel: None,
        });
        Ok(ChannelHandle {
            group: group.0,
            channel: ch_idx,
        })
    }

    /// Adds a fixed-size N-dimensional array channel (`CABLOCK` with `CN Template` storage).
    /// Total array bytes in each record = ` (elem_bit_count / 8) * product(dims)`.
    pub fn add_array_channel(
        &mut self,
        group: GroupHandle,
        name: &str,
        elem_data_type: CnDataType,
        elem_bit_count: u32,
        dims: Vec<u64>,
    ) -> Result<ChannelHandle> {
        if dims.is_empty()
            || dims.contains(&0)
            || elem_bit_count == 0
            || !elem_bit_count.is_multiple_of(8)
        {
            return Err(Error::InvalidSchema(format!(
                "invalid array channel '{name}': elem_bit_count={elem_bit_count} dims={dims:?}"
            )));
        }
        let g = self
            .groups
            .get_mut(group.0 as usize)
            .ok_or_else(|| Error::InvalidSchema(format!("unknown group {}", group.0)))?;
        let elem_bytes = elem_bit_count / 8;
        let total_elems: u64 = dims.iter().product();
        let total_bytes = elem_bytes as u64 * total_elems;
        let byte_offset = g.record_data_bytes;
        g.record_data_bytes = byte_offset + total_bytes as u32;
        let ch_idx = g.channels.len() as u16;
        g.channels.push(ChannelDef {
            name: name.to_string(),
            unit: None,
            comment: None,
            cn_type: CnChannelType::FixedLength,
            sync_type: CnSyncType::None,
            data_type: elem_data_type,
            bit_offset: 0,
            byte_offset,
            bit_count: elem_bit_count,
            flags: 0,
            conversion: None,
            source: None,
            array_dims: Some(dims),
            is_vlsd: false,
            vlsc_size_channel: None,
        });
        Ok(ChannelHandle {
            group: group.0,
            channel: ch_idx,
        })
    }

    /// Adds a Variable-Length Signal Data (`VLSD`, `cn_type = 1`) channel (e.g. UTF-8 string or byte array).
    /// Stores a 64-bit unsigned byte offset into the channel's `SDBLOCK` stream (or VLSD CG in unsorted mode).
    pub fn add_vlsd_channel(
        &mut self,
        group: GroupHandle,
        name: &str,
        data_type: CnDataType,
    ) -> Result<ChannelHandle> {
        let g = self
            .groups
            .get_mut(group.0 as usize)
            .ok_or_else(|| Error::InvalidSchema(format!("unknown group {}", group.0)))?;
        let byte_offset = g.record_data_bytes;
        g.record_data_bytes = byte_offset + 8;
        let ch_idx = g.channels.len() as u16;
        g.channels.push(ChannelDef {
            name: name.to_string(),
            unit: None,
            comment: None,
            cn_type: CnChannelType::Vlsd,
            sync_type: CnSyncType::None,
            data_type,
            bit_offset: 0,
            byte_offset,
            bit_count: 64,
            flags: 0,
            conversion: None,
            source: None,
            array_dims: None,
            is_vlsd: true,
            vlsc_size_channel: None,
        });
        Ok(ChannelHandle {
            group: group.0,
            channel: ch_idx,
        })
    }

    /// Adds an MDF 4.3 Variable-Length Size Channel (`VLSC`, `cn_type = 7`) and its companion
    /// 32-bit size channel (`"{name}_size"`), linked via the 9th `CNBLOCK` link (`cn_cn_size`).
    ///
    /// In sorted mode, raw variable-length values (without a 4-byte length prefix) are stored in
    /// a `VDBLOCK` (`##VD` / `##DZ`) chain while offsets and sizes live in the fixed record; in
    /// unsorted mode, values are emitted to a companion `CG_F_VLSC` channel group.
    pub fn add_vlsc_channel(
        &mut self,
        group: GroupHandle,
        name: &str,
        data_type: CnDataType,
    ) -> Result<ChannelHandle> {
        let size_name = format!("{name}_size");
        let size_ch = self.add_channel(group, &size_name, CnDataType::UIntLE, 32)?;
        let g = self
            .groups
            .get_mut(group.0 as usize)
            .ok_or_else(|| Error::InvalidSchema(format!("unknown group {}", group.0)))?;
        let byte_offset = g.record_data_bytes;
        g.record_data_bytes = byte_offset + 8;
        let ch_idx = g.channels.len() as u16;
        g.channels.push(ChannelDef {
            name: name.to_string(),
            unit: None,
            comment: None,
            cn_type: CnChannelType::Vlsc,
            sync_type: CnSyncType::None,
            data_type,
            bit_offset: 0,
            byte_offset,
            bit_count: 64,
            flags: 0,
            conversion: None,
            source: None,
            array_dims: None,
            is_vlsd: true,
            vlsc_size_channel: Some(size_ch.channel),
        });
        Ok(ChannelHandle {
            group: group.0,
            channel: ch_idx,
        })
    }

    /// Sets the physical unit string (`cn_md_unit`) for `ch`.
    pub fn set_channel_unit(&mut self, ch: ChannelHandle, unit: &str) -> Result<&mut Self> {
        self.channel_mut(ch)?.unit = Some(unit.to_string());
        Ok(self)
    }

    /// Sets the description comment (`cn_md_comment`) for `ch`.
    pub fn set_channel_comment(&mut self, ch: ChannelHandle, comment: &str) -> Result<&mut Self> {
        self.channel_mut(ch)?.comment = Some(comment.to_string());
        Ok(self)
    }

    /// Sets a conversion formula (`CCBLOCK`) for `ch`.
    pub fn set_channel_conversion(
        &mut self,
        ch: ChannelHandle,
        conversion: ConversionDef,
    ) -> Result<&mut Self> {
        self.channel_mut(ch)?.conversion = Some(conversion);
        Ok(self)
    }

    /// Sets the source info (`SIBLOCK`) for `ch`.
    pub fn set_channel_source(
        &mut self,
        ch: ChannelHandle,
        source: SourceDef,
    ) -> Result<&mut Self> {
        self.channel_mut(ch)?.source = Some(source);
        Ok(self)
    }

    fn channel_mut(&mut self, ch: ChannelHandle) -> Result<&mut ChannelDef> {
        let g = self
            .groups
            .get_mut(ch.group as usize)
            .ok_or_else(|| Error::InvalidSchema(format!("unknown group {}", ch.group)))?;
        g.channels
            .get_mut(ch.channel as usize)
            .ok_or_else(|| Error::InvalidSchema(format!("unknown channel {}", ch.channel)))
    }

    /// Returns the estimated RAM buffer budget in bytes across all groups.
    pub fn estimated_ram_budget(&self) -> usize {
        match self.layout {
            DataLayout::Sorted => self.groups.iter().map(|g| g.flush_threshold_bytes).sum(),
            DataLayout::Unsorted => self
                .groups
                .first()
                .map(|g| g.flush_threshold_bytes)
                .unwrap_or(32 * 1024),
        }
    }

    /// Validates and freezes the schema.
    pub fn build(self) -> Result<Schema> {
        if self.groups.is_empty() {
            return Err(Error::InvalidSchema(
                "schema must contain at least one channel group".to_string(),
            ));
        }
        Ok(Schema {
            layout: self.layout,
            compression: self.compression,
            max_blocks_per_dl: self.max_blocks_per_dl,
            checkpoint_every_n_blocks: self.checkpoint_every_n_blocks,
            groups: self.groups,
        })
    }
}
