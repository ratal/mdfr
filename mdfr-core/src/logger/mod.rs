//! Embedded MDF 4.3 streaming logger (`no_std + alloc`).
pub mod engine;
pub mod presets;
pub mod recover;
pub mod schema;
pub mod spsc;

pub use engine::{Logger, RecordWriter};
#[cfg(feature = "derive")]
pub use mdfr_derive::MdfRecord;
pub use presets::{BusPreset, CanGroupHandles};
pub use recover::{ReadWriteSeek, recover_in_place};
pub use schema::{
    ChannelDef, ChannelHandle, ConversionDef, DataLayout, GroupDef, GroupHandle, LoggerBuilder,
    Schema, SourceDef,
};
pub use spsc::SpscRing;

/// Trait implemented by `#[derive(MdfRecord)]` for zero-boilerplate typed channel groups.
pub trait MdfRecord {
    /// Default acquisition name of the channel group.
    const GROUP_NAME: &'static str;
    /// Total fixed plain record byte length (including the 8-byte master timestamp prefix at `0..8`).
    const RECORD_BYTES: usize;

    /// Registers this record's channels onto `group` in `builder`.
    fn register_channels(
        builder: &mut LoggerBuilder,
        group: GroupHandle,
    ) -> crate::error::Result<()>;

    /// Encodes this record's fields into `buf` (`buf.len() == Self::RECORD_BYTES`; bytes `0..8` are reserved for the master timestamp).
    fn encode_record(&self, buf: &mut [u8]);
}
