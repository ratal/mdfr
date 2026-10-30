//! Typed errors for `no_std` code, mapped 1:1 to C error codes.
use alloc::string::String;

/// Errors of the core and logger.
#[derive(Debug, thiserror::Error)]
#[cfg_attr(all(feature = "defmt", not(feature = "std")), derive(defmt::Format))]
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

impl From<binrw::Error> for Error {
    fn from(e: binrw::Error) -> Self {
        Error::InvalidBlock(alloc::format!("{e}"))
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;

    #[test]
    fn test_error_display_and_codes() {
        assert_eq!(Error::Io.code(), -1);
        assert_eq!(alloc::format!("{}", Error::Io), "sink I/O error");

        let err_block = Error::InvalidBlock("bad header".into());
        assert_eq!(err_block.code(), -2);
        assert_eq!(alloc::format!("{err_block}"), "invalid block: bad header");

        let err_schema = Error::InvalidSchema("dup name".into());
        assert_eq!(err_schema.code(), -3);
        assert_eq!(alloc::format!("{err_schema}"), "invalid schema: dup name");

        let err_size = Error::RecordSize {
            expected: 16,
            got: 8,
        };
        assert_eq!(err_size.code(), -4);
        assert_eq!(
            alloc::format!("{err_size}"),
            "record size mismatch: expected 16, got 8"
        );

        let err_unsupported = Error::Unsupported("lzma");
        assert_eq!(err_unsupported.code(), -5);
        assert_eq!(alloc::format!("{err_unsupported}"), "unsupported: lzma");

        let binrw_err = binrw::Error::Custom {
            pos: 10,
            err: Box::new("custom"),
        };
        let err_from_binrw: Error = binrw_err.into();
        assert_eq!(err_from_binrw.code(), -2);
    }
}
