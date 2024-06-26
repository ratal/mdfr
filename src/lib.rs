//#![forbid(unsafe_code)]
mod c_api;
pub mod data_holder;
pub mod export;
pub mod mdfinfo;
#[cfg(feature = "numpy")]
pub mod mdfr;

pub mod mdfreader;
pub mod mdfwriter;
mod tests;
