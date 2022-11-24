//! buffered reader taken from std but with filling buffer starting position
//!  at the middle of buffer to optimise forward and especially backward reading
use std::io::{Error, ErrorKind, Read, Result, Seek, SeekFrom};
use std::{cmp, io};

/// reader buffer size, by default same as Rust BufReader
pub const DEFAULT_BUF_SIZE: usize = 8192;
pub struct SymBufReader<R>
where
    R: Read,
{
    reader: R,
    pos: usize,
    cap: usize,
    buf: Vec<u8>,
}

impl<R> SymBufReader<R>
where
    R: Read,
{
    /// Creates a new SymBufReader with a specified `buffer_size`.
    /// This newly created object wraps another object which is [Read](std::io::Read).
    pub fn new(reader: R) -> Self {
        let buffer = vec![0; DEFAULT_BUF_SIZE];
        Self {
            reader,
            buf: buffer,
            cap: 0,
            pos: 0,
        }
    }

    /// Invalidates all data in the internal buffer.
    #[inline]
    fn discard_buffer(&mut self) {
        self.pos = 0;
        self.cap = 0;
    }

    pub fn buffer(&self) -> &[u8] {
        &self.buf[self.pos..self.cap]
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
            "failed to write whole buffer",
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

impl<R: Seek> SymBufReader<R>
where
    R: Read,
{
    /// Seeks relative to the current position. If the new position lies within the buffer,
    /// the buffer will not be flushed, allowing for more efficient seeks.
    /// This method does not return the location of the underlying reader, so the caller
    /// must track this information themselves if it is required.
    pub fn seek_relative(&mut self, offset: i64) -> Result<()> {
        let pos = self.pos as u64;
        if offset < 0 {
            if let Some(new_pos) = pos.checked_sub((-offset) as u64) {
                self.pos = new_pos as usize;
                return Ok(());
            }
        } else if let Some(new_pos) = pos.checked_add(offset as u64) {
            if new_pos <= self.cap as u64 {
                self.pos = new_pos as usize;
                return Ok(());
            }
        }

        // Flushes the buffer
        self.seek(SeekFrom::Current(offset)).map(drop)
    }
}

impl<R> SymBufReader<R>
where
    R: Read + Seek,
{
    fn fill_buf(&mut self) -> Result<&[u8]> {
        // If we've reached the end of our internal buffer then we need to fetch
        // some more data from the underlying reader.
        // Branch using `>=` instead of the more correct `==`
        // to tell the compiler that the pos..cap slice is always valid.
        if self.pos >= self.cap {
            debug_assert!(self.pos == self.cap);

            let middle_of_buffer = (DEFAULT_BUF_SIZE as i64) / 2;
            // checks if close to stream start
            let stream_position = self.stream_position()? as i64;
            if let Some(remaining) = stream_position.checked_sub(middle_of_buffer) {
                if remaining <= 0 {
                    self.seek(SeekFrom::Start(0))?;
                    let n_read = self.reader.read(&mut self.buf)?;
                    self.cap = n_read;
                    self.pos = stream_position as usize;
                    return Ok(self.buffer());
                }
            }

            // reposition half buffer size before
            let result = self.seek(SeekFrom::Current(-middle_of_buffer));
            match result {
                Ok(_) => {
                    let n_read = self.reader.read(&mut self.buf)?;
                    self.cap = n_read;
                    self.pos = middle_of_buffer as usize;
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
