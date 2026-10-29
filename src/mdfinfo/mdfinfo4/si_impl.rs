//! Source Information block (SIBLOCK) for MDF4 — spec section 6.9, Tables 44-46
//!
//! Describes the source of a channel or channel group (ECU, bus, tool, etc.).
//! SIBLOCKs are sharable and stored in the SharableBlocks hashmap.
use anyhow::Result;

use super::block_header_impl::SharableBlocks;

pub use mdfr_core::blocks::si::Si4Block;

/// Extension methods on [`Si4Block`] that resolve links through [`SharableBlocks`].
pub trait Si4BlockExt {
    /// returns the source name
    fn get_si_source_name(&self, sharable: &SharableBlocks) -> Result<Option<String>>;
    /// returns the source path
    fn get_si_path_name(&self, sharable: &SharableBlocks) -> Result<Option<String>>;
}

impl Si4BlockExt for Si4Block {
    fn get_si_source_name(&self, sharable: &SharableBlocks) -> Result<Option<String>> {
        sharable.get_tx(self.si_tx_name)
    }
    fn get_si_path_name(&self, sharable: &SharableBlocks) -> Result<Option<String>> {
        sharable.get_tx(self.si_tx_path)
    }
}
