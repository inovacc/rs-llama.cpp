//! Ported from `gguf/reader.go`.
//!
//! Go's `bufferedReader` wraps a `bufio.Reader` over an `io.ReadSeeker` and
//! tracks the logical byte offset consumed so far. This is used purely for
//! sequential header/metadata parsing in `gguf.rs`; tensor data reads use a
//! fresh positional read against the file path instead (mirroring how Go's
//! `io.NewSectionReader` reads via `ReaderAt`, independent of whatever the
//! `bufio.Reader` had buffered ahead).

use std::io::{self, BufReader, Read};

/// A buffered reader that additionally tracks the total number of bytes
/// consumed via `Read` (Go's `bufferedReader.offset`).
pub struct BufferedReader<R: Read> {
    inner: BufReader<R>,
    pub offset: u64,
}

impl<R: Read> BufferedReader<R> {
    pub fn new(rs: R, size: usize) -> Self {
        BufferedReader {
            inner: BufReader::with_capacity(size, rs),
            offset: 0,
        }
    }
}

impl<R: Read> Read for BufferedReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.offset += n as u64;
        Ok(n)
    }
}
