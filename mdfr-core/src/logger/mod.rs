//! Embedded MDF 4.3 streaming logger (`no_std + alloc`).
pub mod engine;
pub mod presets;
pub mod recover;
pub mod schema;
pub mod spsc;

pub use engine::{Logger, RecordWriter};
pub use presets::{BusPreset, CanGroupHandles};
pub use recover::{ReadWriteSeek, recover_in_place};
pub use schema::{
    ChannelDef, ChannelHandle, ConversionDef, DataLayout, GroupDef, GroupHandle, LoggerBuilder,
    Schema, SourceDef,
};
pub use spsc::SpscRing;
