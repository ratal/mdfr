//! Typed errors for `no_std` code, mapped 1:1 to C error codes.
use alloc::string::String;

/// Errors of the core and logger.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Sink I/O failure (write, seek or flush).
    #[error("sink I/O error")]
    Io,
    /// Block could not be serialized or parsed.
    #[error("invalid block: {0}")]
    InvalidBlock(String),
    /// Schema is inconsistent (overlapping channels, unknown group, ...).
    #[error("invalid schema: {0}")]
    InvalidSchema(String),
    /// Record length does not match the group record size.
    #[error("record size mismatch: expected {expected}, got {got}")]
    RecordSize {
        /// Expected size in bytes.
        expected: usize,
        /// Provided size in bytes.
        got: usize,
    },
    /// Feature or data type not supported.
    #[error("unsupported: {0}")]
    Unsupported(&'static str),
}

/// Result alias for the core.
pub type Result<T> = core::result::Result<T, Error>;

impl Error {
    /// Stable integer code for the C API (0 is success).
    pub fn code(&self) -> i32 {
        match self {
            Error::Io => -1,
            Error::InvalidBlock(_) => -2,
            Error::InvalidSchema(_) => -3,
            Error::RecordSize { .. } => -4,
            Error::Unsupported(_) => -5,
        }
    }
}
