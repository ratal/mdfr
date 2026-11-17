//! buffered reader taken from std but with filling buffer starting position
//! at the middle of buffer to optimise forward and especially backward reading,
//! plus zero-copy `MappedFile` and `SliceCursor` for memory-mapped file access.
use std::fs::File;
use std::io::{Error, ErrorKind, Read, Result, Seek, SeekFrom};
use std::ops::Deref;
use std::{cmp, io};

/// Memory-mapped read-only file with automatic heap-buffer fallback for
/// empty or non-mappable files (e.g., virtual filesystems, FIFOs).
pub enum MappedFile {
    Mmap(memmap2::Mmap),
    Bytes(Vec<u8>),
}

impl MappedFile {
    /// Opens `path` read-only and maps it into memory, falling back to reading
    /// into a `Vec<u8>` if the file is empty or cannot be memory-mapped.
    pub fn open(path: &str) -> Result<Self> {
        let mut file = File::open(path)?;
        let len = file.metadata()?.len();
        if len == 0 {
            return Ok(Self::Bytes(Vec::new()));
        }
        // SAFETY: The mapping is read-only and transient (dropped as soon as metadata
        // parsing or data loading finishes); all outputs are copied into owned buffers.
        match unsafe { memmap2::MmapOptions::new().map(&file) } {
            Ok(mmap) => Ok(Self::Mmap(mmap)),
            Err(_) => {
                let mut bytes = Vec::with_capacity(usize::try_from(len).unwrap_or(0));
                file.read_to_end(&mut bytes)?;
                Ok(Self::Bytes(bytes))
            }
        }
    }
}

impl Deref for MappedFile {
    type Target = [u8];

    #[inline(always)]
    fn deref(&self) -> &[u8] {
        match self {
            Self::Mmap(m) => m.as_ref(),
            Self::Bytes(v) => v.as_slice(),
        }
    }
}

/// Zero-copy cursor over an in-memory or memory-mapped byte slice (`&'a [u8]`).
///
/// Implements [`Read`] and [`Seek`] for compatibility with `binrw` and `byteorder`,
/// while also exposing [`read_slice`](Self::read_slice) and [`as_slice`](Self::as_slice)
/// to borrow sub-slices directly without copying.
pub struct SliceCursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> SliceCursor<'a> {
    /// Creates a new cursor positioned at byte 0 of `data`.
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Returns the entire underlying byte slice.
    #[inline(always)]
    pub fn as_slice(&self) -> &'a [u8] {
        self.data
    }

    /// Returns the total byte length of the underlying slice.
    #[inline(always)]
    pub fn file_size(&self) -> Result<u64> {
        Ok(self.data.len() as u64)
    }

    /// Seeks relative to the current position without syscalls.
    #[inline(always)]
    pub fn seek_relative(&mut self, offset: i64) -> Result<()> {
        let new_pos = (self.pos as i128) + (offset as i128);
        if new_pos < 0 || new_pos > usize::MAX as i128 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "invalid seek to a negative or overflowing position",
            ));
        }
        self.pos = new_pos as usize;
        Ok(())
    }

    /// Borrows `len` bytes starting at the current position and advances the cursor.
    ///
    /// Returns `UnexpectedEof` if fewer than `len` bytes remain or `pos + len` overflows.
    #[inline(always)]
    pub fn read_slice(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| Error::new(ErrorKind::UnexpectedEof, "slice read length overflow"))?;
        let slice = self
            .data
            .get(self.pos..end)
            .ok_or_else(|| Error::new(ErrorKind::UnexpectedEof, "failed to fill whole buffer"))?;
        self.pos = end;
        Ok(slice)
    }
}

impl Read for SliceCursor<'_> {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        let available = self.data.len().saturating_sub(self.pos);
        let n = buf.len().min(available);
        if n > 0 {
            buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
            self.pos += n;
        }
        Ok(n)
    }

    #[inline]
    fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
        let slice = self.read_slice(buf.len())?;
        buf.copy_from_slice(slice);
        Ok(())
    }

    #[inline]
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> Result<usize> {
        let rem = self.data.get(self.pos..).unwrap_or(&[]);
        buf.extend_from_slice(rem);
        self.pos = self.data.len();
        Ok(rem.len())
    }
}

impl Seek for SliceCursor<'_> {
    #[inline]
    fn seek(&mut self, pos: SeekFrom) -> Result<u64> {
        let new_pos: i128 = match pos {
            SeekFrom::Start(n) => n as i128,
            SeekFrom::End(n) => (self.data.len() as i128) + (n as i128),
            SeekFrom::Current(n) => (self.pos as i128) + (n as i128),
        };
        if new_pos < 0 || new_pos > usize::MAX as i128 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "invalid seek to a negative or overflowing position",
            ));
        }
        self.pos = new_pos as usize;
        Ok(self.pos as u64)
    }

    #[inline]
    fn stream_position(&mut self) -> Result<u64> {
        Ok(self.pos as u64)
    }
}

/// reader buffer size, by default same as Rust BufReader
pub const DEFAULT_BUF_SIZE: usize = 8192;
#[allow(dead_code)]
pub struct SymBufReader<R>
where
    R: Read,
{
    reader: R,
    pos: usize,
    cap: usize,
    buf: Vec<u8>,
    file_size_cache: Option<u64>,
    last_seek_backward: bool,
}

#[allow(dead_code)]
impl<R> SymBufReader<R>
where
    R: Read,
{
    /// Creates a new SymBufReader with a specified `buffer_size`.
    /// This newly created object wraps another object which is Read (std::io::Read).
    pub fn new(reader: R) -> Self {
        let buffer = vec![0; DEFAULT_BUF_SIZE];
        Self {
            reader,
            buf: buffer,
            cap: 0,
            pos: 0,
            file_size_cache: None,
            last_seek_backward: false,
        }
    }

    /// Invalidates all data in the internal buffer.
    #[inline]
    fn discard_buffer(&mut self) {
        self.pos = 0;
        self.cap = 0;
    }

    pub fn buffer(&self) -> &[u8] {
        if self.pos >= self.cap {
            &[]
        } else {
            &self.buf[self.pos..self.cap]
        }
    }

    /// Gets a reference to the underlying reader.
    #[allow(dead_code)]
    pub fn get_ref(&self) -> &R {
        &self.reader
    }

    /// Unwraps this `SymBufReader<R>`, returning the underlying reader.
    #[allow(dead_code)]
    pub fn into_inner(self) -> R {
        self.reader
    }
}

impl<R> Read for SymBufReader<R>
where
    R: Read + Seek,
{
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        // If we don't have any buffered data and we're doing a massive read
        // (larger than our internal buffer), bypass our internal buffer
        // entirely.
        if self.pos == self.cap && buf.len() >= self.buf.len() {
            self.discard_buffer();
            return self.reader.read(buf);
        }
        let nread = {
            let mut rem = self.fill_buf()?;
            rem.read(buf)?
        };
        self.consume(nread);
        Ok(nread)
    }

    // Small read_exacts from a BufReader are extremely common when used with a deserializer.
    // The default implementation calls read in a loop, which results in surprisingly poor code
    // generation for the common path where the buffer has enough bytes to fill the passed-in
    // buffer.
    fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
        if self.buffer().len() >= buf.len() {
            buf.copy_from_slice(&self.buffer()[..buf.len()]);
            self.consume(buf.len());
            return Ok(());
        }

        default_read_exact(self, buf)
    }
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        let nread = self.cap - self.pos;
        buf.extend_from_slice(self.buffer());
        self.discard_buffer();
        Ok(nread + self.reader.read_to_end(buf)?)
    }
}

#[allow(dead_code)]
fn default_read_exact<R: Read + ?Sized>(this: &mut R, mut buf: &mut [u8]) -> Result<()> {
    while !buf.is_empty() {
        match this.read(buf) {
            Ok(0) => break,
            Ok(n) => {
                let tmp = buf;
                buf = &mut tmp[n..];
            }
            Err(ref e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    if !buf.is_empty() {
        Err(Error::new(
            ErrorKind::UnexpectedEof,
            "failed to fill whole buffer",
        ))
    } else {
        Ok(())
    }
}

impl<R: Seek> Seek for SymBufReader<R>
where
    R: Read,
{
    fn seek(&mut self, pos: SeekFrom) -> Result<u64> {
        let result: u64;
        match pos {
            SeekFrom::Current(n) => {
                self.last_seek_backward = n < 0;
                let remainder = (self.cap - self.pos) as i64;
                // it should be safe to assume that remainder fits within an i64 as the alternative
                // means we managed to allocate 8 exbibytes and that's absurd.
                // But it's not out of the realm of possibility for some weird underlying reader to
                // support seeking by i64::MIN so we need to handle underflow when subtracting
                // remainder.
                if let Some(offset) = n.checked_sub(remainder) {
                    result = self.reader.seek(SeekFrom::Current(offset))?;
                } else {
                    // seek backwards by our remainder, and then by the offset
                    self.reader.seek(SeekFrom::Current(-remainder))?;
                    self.discard_buffer();
                    result = self.reader.seek(SeekFrom::Current(n))?;
                }
            }
            _ => {
                self.last_seek_backward = false;
                // Seeking with Start/End doesn't care about our buffer length.
                result = self.reader.seek(pos)?;
            }
        };
        self.discard_buffer();
        Ok(result)
    }
    fn stream_position(&mut self) -> io::Result<u64> {
        let remainder = (self.cap - self.pos) as u64;
        self.reader.stream_position().map(|pos| {
            pos.checked_sub(remainder).expect(
                "overflow when subtracting remaining buffer size from inner stream position",
            )
        })
    }
}

#[allow(dead_code)]
impl<R: Read + Seek> SymBufReader<R> {
    /// Returns the total length of the underlying stream without changing the
    /// logical read position or discarding the internal buffer. Result is cached after the first call.
    pub fn file_size(&mut self) -> Result<u64> {
        if let Some(size) = self.file_size_cache {
            return Ok(size);
        }
        let underlying_pos = self.reader.stream_position()?;
        let size = self.reader.seek(SeekFrom::End(0))?;
        self.reader.seek(SeekFrom::Start(underlying_pos))?;
        self.file_size_cache = Some(size);
        Ok(size)
    }

    /// Seeks relative to the current position. If the new position lies within the buffer,
    /// the buffer will not be flushed, allowing for more efficient seeks.
    /// This method does not return the location of the underlying reader, so the caller
    /// must track this information themselves if it is required.
    pub fn seek_relative(&mut self, offset: i64) -> Result<()> {
        let pos = self.pos as u64;
        if offset < 0 {
            self.last_seek_backward = true;
            if let Some(new_pos) = pos.checked_sub((-offset) as u64) {
                self.pos = new_pos as usize;
                return Ok(());
            }
        } else {
            self.last_seek_backward = false;
            if let Some(new_pos) = pos.checked_add(offset as u64)
                && new_pos <= self.cap as u64
            {
                self.pos = new_pos as usize;
                return Ok(());
            }
        }

        // Flushes the buffer
        self.seek(SeekFrom::Current(offset)).map(drop)
    }

    fn fill_buf(&mut self) -> Result<&[u8]> {
        // If we've reached the end of our internal buffer then we need to fetch
        // some more data from the underlying reader.
        // Branch using `>=` instead of the more correct `==`
        // to tell the compiler that the pos..cap slice is always valid.
        if self.pos >= self.cap {
            debug_assert!(self.pos == self.cap);

            if !self.last_seek_backward {
                // Forward sequential read: reader is ALREADY at stream position.
                // Do NOT call self.stream_position()? or self.reader.seek(...)!
                let mut total_read = 0;
                while total_read < self.buf.len() {
                    let n = self.reader.read(&mut self.buf[total_read..])?;
                    if n == 0 {
                        break;
                    }
                    total_read += n;
                }
                self.cap = total_read;
                self.pos = 0;
                return Ok(self.buffer());
            }

            let stream_position = self.stream_position()? as i64;
            let middle_of_buffer = (DEFAULT_BUF_SIZE as i64) / 2;
            // checks if close to stream start
            if let Some(remaining) = stream_position.checked_sub(middle_of_buffer)
                && remaining <= 0
            {
                self.rewind()?;
                let mut total_read = 0;
                while total_read < self.buf.len() {
                    let n = self.reader.read(&mut self.buf[total_read..])?;
                    if n == 0 {
                        break;
                    }
                    total_read += n;
                }
                self.cap = total_read;
                self.pos = (stream_position as usize).min(self.cap);
                self.last_seek_backward = false; // Reset direction after refill
                return Ok(self.buffer());
            }

            // reposition half buffer size before
            let result = self.seek(SeekFrom::Current(-middle_of_buffer));
            match result {
                Ok(_) => {
                    let mut total_read = 0;
                    while total_read < self.buf.len() {
                        let n = self.reader.read(&mut self.buf[total_read..])?;
                        if n == 0 {
                            break;
                        }
                        total_read += n;
                    }
                    self.cap = total_read;
                    self.pos = (middle_of_buffer as usize).min(self.cap);
                    self.last_seek_backward = false; // Reset direction after refill
                }
                Err(e) => return Err(e),
            }
        }
        Ok(self.buffer())
    }

    fn consume(&mut self, amt: usize) {
        self.pos = cmp::min(self.pos + amt, self.cap);
    }
}

#[test]
fn sym_buf_reader_test() -> io::Result<()> {
    use std::io::Cursor;
    let mut data = [0u8; 25000];
    data.iter_mut().enumerate().for_each(|x| *x.1 = x.0 as u8);
    let inner = Cursor::new(&data);
    let mut reader = SymBufReader::new(inner);

    let mut buffer = [0, 0, 0];
    assert_eq!(reader.read(&mut buffer).ok(), Some(3));
    assert_eq!(buffer, [0, 1, 2]);

    let mut buffer = [0, 0, 0, 0, 0];
    reader.read_exact(&mut buffer)?;
    assert_eq!(buffer, [3, 4, 5, 6, 7]);

    reader.seek_relative(-5)?;
    let mut buffer = [0, 0, 0, 0, 0];
    assert_eq!(reader.read(&mut buffer).ok(), Some(5));
    assert_eq!(buffer, [3, 4, 5, 6, 7]);

    reader.seek_relative(-8)?; // remain within buffer
    let mut buffer = [0, 0];
    assert_eq!(reader.read(&mut buffer).ok(), Some(2));
    assert_eq!(buffer, [0, 1]);

    reader.seek_relative(8192)?; // clears buffer
    let mut buffer = [0, 0];
    reader.read_exact(&mut buffer).ok();
    assert_eq!(buffer, [2, 3]);

    reader.seek_relative(9000)?; // clears buffer
    let mut buffer = [0, 0];
    reader.read_exact(&mut buffer).ok();
    assert_eq!(buffer, [44, 45]);

    Ok(())
}

#[test]
fn slice_cursor_and_mapped_file_test() -> io::Result<()> {
    use std::io::Write;
    let payload: Vec<u8> = (0..128u8).collect();
    let mut cursor = SliceCursor::new(&payload);
    assert_eq!(cursor.file_size()?, 128);
    assert_eq!(cursor.read_slice(4)?, &[0, 1, 2, 3]);
    cursor.seek_relative(6)?;
    assert_eq!(cursor.read_slice(3)?, &[10, 11, 12]);
    cursor.seek_relative(-3)?;
    assert_eq!(cursor.read_slice(3)?, &[10, 11, 12]);
    assert!(cursor.seek_relative(-100).is_err());
    assert!(cursor.read_slice(200).is_err());
    assert!(cursor.read_slice(usize::MAX).is_err());

    // Empty file maps safely via fallback
    let empty_tmp = tempfile::NamedTempFile::new()?;
    let empty_mapped = MappedFile::open(empty_tmp.path().to_str().unwrap())?;
    assert!(empty_mapped.is_empty());

    // Non-empty file maps via mmap and matches bytes
    let mut tmp = tempfile::NamedTempFile::new()?;
    tmp.write_all(&payload)?;
    tmp.flush()?;
    let mapped = MappedFile::open(tmp.path().to_str().unwrap())?;
    assert_eq!(&*mapped, &payload[..]);
    Ok(())
}
