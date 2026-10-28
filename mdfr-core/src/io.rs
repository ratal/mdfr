//! Storage abstraction. Mirrors the `embedded-io` `Write + Seek` shape so it can be
//! swapped for those traits; seek is only used for sparse back-patching.
use crate::error::{Error, Result};

/// Position for [`Sink::seek`].
#[derive(Debug, Clone, Copy)]
pub enum SeekFrom {
    /// Absolute offset from file start.
    Start(u64),
    /// Offset from the end.
    End(i64),
}

/// Output of the logger (SD card, flash, file, C callbacks...).
pub trait Sink {
    /// Writes the whole buffer.
    fn write_all(&mut self, buf: &[u8]) -> Result<()>;
    /// Moves the write position, returning the new absolute position.
    fn seek(&mut self, pos: SeekFrom) -> Result<u64>;
    /// Flushes buffered data to the medium.
    fn flush(&mut self) -> Result<()>;
}

/// In-RAM sink, useful for tests and QEMU smoke tests.
#[derive(Debug, Default)]
pub struct VecSink {
    /// File content.
    pub data: alloc::vec::Vec<u8>,
    pos: usize,
}

impl Sink for VecSink {
    fn write_all(&mut self, buf: &[u8]) -> Result<()> {
        let end = self.pos + buf.len();
        if end > self.data.len() {
            self.data.resize(end, 0);
        }
        self.data[self.pos..end].copy_from_slice(buf);
        self.pos = end;
        Ok(())
    }
    fn seek(&mut self, pos: SeekFrom) -> Result<u64> {
        let new = match pos {
            SeekFrom::Start(p) => p as i64,
            SeekFrom::End(o) => self.data.len() as i64 + o,
        };
        if new < 0 {
            return Err(Error::Io);
        }
        self.pos = new as usize;
        Ok(new as u64)
    }
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

#[cfg(feature = "std")]
impl Sink for std::fs::File {
    fn write_all(&mut self, buf: &[u8]) -> Result<()> {
        std::io::Write::write_all(self, buf).map_err(|_| Error::Io)
    }
    fn seek(&mut self, pos: SeekFrom) -> Result<u64> {
        let p = match pos {
            SeekFrom::Start(p) => std::io::SeekFrom::Start(p),
            SeekFrom::End(o) => std::io::SeekFrom::End(o),
        };
        std::io::Seek::seek(self, p).map_err(|_| Error::Io)
    }
    fn flush(&mut self) -> Result<()> {
        std::io::Write::flush(self).map_err(|_| Error::Io)
    }
}
