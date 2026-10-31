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

impl VecSink {
    /// Creates an empty [`VecSink`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty [`VecSink`] with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: alloc::vec::Vec::with_capacity(capacity),
            pos: 0,
        }
    }

    /// Returns the current cursor byte position.
    pub fn position(&self) -> u64 {
        self.pos as u64
    }

    /// Returns a slice of the written bytes.
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }
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

/// Sink adapter wrapping any type implementing `embedded_io::Write + embedded_io::Seek`.
#[cfg(feature = "embedded-io")]
#[derive(Debug)]
pub struct EmbeddedIoSink<T> {
    inner: T,
}

#[cfg(feature = "embedded-io")]
impl<T> EmbeddedIoSink<T> {
    /// Creates a new [`EmbeddedIoSink`] wrapping `inner`.
    pub fn new(inner: T) -> Self {
        Self { inner }
    }

    /// Consumes the adapter and returns the underlying writer.
    pub fn into_inner(self) -> T {
        self.inner
    }

    /// Returns a shared reference to the inner writer.
    pub fn get_ref(&self) -> &T {
        &self.inner
    }

    /// Returns a mutable reference to the inner writer.
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.inner
    }
}

#[cfg(feature = "embedded-io")]
impl<T> From<T> for EmbeddedIoSink<T> {
    fn from(inner: T) -> Self {
        Self::new(inner)
    }
}

#[cfg(feature = "embedded-io")]
impl<T: embedded_io::Write + embedded_io::Seek> Sink for EmbeddedIoSink<T> {
    fn write_all(&mut self, buf: &[u8]) -> Result<()> {
        embedded_io::Write::write_all(&mut self.inner, buf).map_err(|_| Error::Io)
    }

    fn seek(&mut self, pos: SeekFrom) -> Result<u64> {
        let p = match pos {
            SeekFrom::Start(p) => embedded_io::SeekFrom::Start(p),
            SeekFrom::End(o) => embedded_io::SeekFrom::End(o),
        };
        embedded_io::Seek::seek(&mut self.inner, p).map_err(|_| Error::Io)
    }

    fn flush(&mut self) -> Result<()> {
        embedded_io::Write::flush(&mut self.inner).map_err(|_| Error::Io)
    }
}

/// Async output of the logger (async SD card, flash, network sink...).
pub trait AsyncSink {
    /// Writes the whole buffer asynchronously.
    fn write_all(&mut self, buf: &[u8]) -> impl core::future::Future<Output = Result<()>>;
    /// Moves the write position asynchronously, returning the new absolute position.
    fn seek(&mut self, pos: SeekFrom) -> impl core::future::Future<Output = Result<u64>>;
    /// Flushes buffered data to the medium asynchronously.
    fn flush(&mut self) -> impl core::future::Future<Output = Result<()>>;
}

/// Sink adapter wrapping any type implementing `embedded_io_async::Write + embedded_io_async::Seek`.
#[cfg(feature = "embedded-io-async")]
#[derive(Debug)]
pub struct EmbeddedIoAsyncSink<T> {
    inner: T,
}

#[cfg(feature = "embedded-io-async")]
impl<T> EmbeddedIoAsyncSink<T> {
    /// Creates a new [`EmbeddedIoAsyncSink`] wrapping `inner`.
    pub fn new(inner: T) -> Self {
        Self { inner }
    }

    /// Consumes the adapter and returns the underlying writer.
    pub fn into_inner(self) -> T {
        self.inner
    }

    /// Returns a shared reference to the inner writer.
    pub fn get_ref(&self) -> &T {
        &self.inner
    }

    /// Returns a mutable reference to the inner writer.
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.inner
    }
}

#[cfg(feature = "embedded-io-async")]
impl<T> From<T> for EmbeddedIoAsyncSink<T> {
    fn from(inner: T) -> Self {
        Self::new(inner)
    }
}

#[cfg(feature = "embedded-io-async")]
impl<T: embedded_io_async::Write + embedded_io_async::Seek> AsyncSink for EmbeddedIoAsyncSink<T> {
    async fn write_all(&mut self, buf: &[u8]) -> Result<()> {
        embedded_io_async::Write::write_all(&mut self.inner, buf)
            .await
            .map_err(|_| Error::Io)
    }

    async fn seek(&mut self, pos: SeekFrom) -> Result<u64> {
        let p = match pos {
            SeekFrom::Start(p) => embedded_io_async::SeekFrom::Start(p),
            SeekFrom::End(o) => embedded_io_async::SeekFrom::End(o),
        };
        embedded_io_async::Seek::seek(&mut self.inner, p)
            .await
            .map_err(|_| Error::Io)
    }

    async fn flush(&mut self) -> Result<()> {
        embedded_io_async::Write::flush(&mut self.inner)
            .await
            .map_err(|_| Error::Io)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "embedded-io")]
    use super::*;

    #[cfg(feature = "embedded-io")]
    struct MemWriter {
        buf: alloc::vec::Vec<u8>,
        pos: usize,
    }

    #[cfg(feature = "embedded-io")]
    impl embedded_io::ErrorType for MemWriter {
        type Error = embedded_io::ErrorKind;
    }

    #[cfg(feature = "embedded-io")]
    impl embedded_io::Write for MemWriter {
        fn write(&mut self, buf: &[u8]) -> core::result::Result<usize, Self::Error> {
            let end = self.pos + buf.len();
            if end > self.buf.len() {
                self.buf.resize(end, 0);
            }
            self.buf[self.pos..end].copy_from_slice(buf);
            self.pos = end;
            Ok(buf.len())
        }
        fn flush(&mut self) -> core::result::Result<(), Self::Error> {
            Ok(())
        }
    }

    #[cfg(feature = "embedded-io")]
    impl embedded_io::Seek for MemWriter {
        fn seek(&mut self, pos: embedded_io::SeekFrom) -> core::result::Result<u64, Self::Error> {
            let new_pos = match pos {
                embedded_io::SeekFrom::Start(p) => p as i64,
                embedded_io::SeekFrom::End(o) => self.buf.len() as i64 + o,
                embedded_io::SeekFrom::Current(o) => self.pos as i64 + o,
            };
            if new_pos < 0 {
                return Err(embedded_io::ErrorKind::InvalidInput);
            }
            self.pos = new_pos as usize;
            Ok(self.pos as u64)
        }
    }

    #[cfg(feature = "embedded-io")]
    #[test]
    fn test_embedded_io_sink() {
        let mem = MemWriter {
            buf: alloc::vec::Vec::new(),
            pos: 0,
        };
        let mut sink = EmbeddedIoSink::new(mem);
        sink.write_all(b"Hello ").unwrap();
        sink.write_all(b"World").unwrap();
        assert_eq!(sink.seek(SeekFrom::Start(0)).unwrap(), 0);
        sink.write_all(b"MDF4!").unwrap();
        assert_eq!(sink.into_inner().buf, b"MDF4! World");
    }

    #[cfg(feature = "embedded-io-async")]
    struct AsyncMemWriter {
        buf: alloc::vec::Vec<u8>,
        pos: usize,
    }

    #[cfg(feature = "embedded-io-async")]
    impl embedded_io_async::ErrorType for AsyncMemWriter {
        type Error = embedded_io_async::ErrorKind;
    }

    #[cfg(feature = "embedded-io-async")]
    impl embedded_io_async::Write for AsyncMemWriter {
        async fn write(&mut self, buf: &[u8]) -> core::result::Result<usize, Self::Error> {
            let end = self.pos + buf.len();
            if end > self.buf.len() {
                self.buf.resize(end, 0);
            }
            self.buf[self.pos..end].copy_from_slice(buf);
            self.pos = end;
            Ok(buf.len())
        }
        async fn flush(&mut self) -> core::result::Result<(), Self::Error> {
            Ok(())
        }
    }

    #[cfg(feature = "embedded-io-async")]
    impl embedded_io_async::Seek for AsyncMemWriter {
        async fn seek(&mut self, pos: embedded_io_async::SeekFrom) -> core::result::Result<u64, Self::Error> {
            let new_pos = match pos {
                embedded_io_async::SeekFrom::Start(p) => p as i64,
                embedded_io_async::SeekFrom::End(o) => self.buf.len() as i64 + o,
                embedded_io_async::SeekFrom::Current(o) => self.pos as i64 + o,
            };
            if new_pos < 0 {
                return Err(embedded_io_async::ErrorKind::InvalidInput);
            }
            self.pos = new_pos as usize;
            Ok(self.pos as u64)
        }
    }

    #[cfg(feature = "embedded-io-async")]
    #[test]
    fn test_embedded_io_async_sink() {
        let mem = AsyncMemWriter {
            buf: alloc::vec::Vec::new(),
            pos: 0,
        };
        let mut sink = EmbeddedIoAsyncSink::new(mem);

        fn dummy_raw_waker() -> core::task::RawWaker {
            fn noop(_: *const ()) {}
            fn clone(p: *const ()) -> core::task::RawWaker {
                core::task::RawWaker::new(p, &VTABLE)
            }
            static VTABLE: core::task::RawWakerVTable =
                core::task::RawWakerVTable::new(clone, noop, noop, noop);
            core::task::RawWaker::new(core::ptr::null(), &VTABLE)
        }
        let waker = unsafe { core::task::Waker::from_raw(dummy_raw_waker()) };
        let mut cx = core::task::Context::from_waker(&waker);

        {
            let mut fut_write = core::pin::pin!(sink.write_all(b"Async MDF4"));
            assert!(core::future::Future::poll(fut_write.as_mut(), &mut cx).is_ready());
        }
        {
            let mut fut_seek = core::pin::pin!(sink.seek(SeekFrom::Start(6)));
            assert!(core::future::Future::poll(fut_seek.as_mut(), &mut cx).is_ready());
        }
        {
            let mut fut_write2 = core::pin::pin!(sink.write_all(b"Core!"));
            assert!(core::future::Future::poll(fut_write2.as_mut(), &mut cx).is_ready());
        }
        {
            let mut fut_flush = core::pin::pin!(sink.flush());
            assert!(core::future::Future::poll(fut_flush.as_mut(), &mut cx).is_ready());
        }

        assert_eq!(sink.into_inner().buf, b"Async Core!");
    }
}
