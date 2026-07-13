//! Incremental FASTA, FASTQ, gzip, and raw-sequence input.

use flate2::read::MultiGzDecoder;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};

/// A normalized sequence record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SequenceRecord {
    pub id: String,
    pub sequence: Vec<u8>,
}

impl SequenceRecord {
    pub fn new(id: impl Into<String>, sequence: Vec<u8>) -> Self {
        Self {
            id: id.into(),
            sequence,
        }
    }
}

/// A parse failure scoped to one sequence record whenever boundaries are known.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequenceReadError {
    pub record_index: usize,
    pub line: usize,
    pub message: String,
    pub recoverable: bool,
}

impl std::fmt::Display for SequenceReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (line {})", self.message, self.line)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SequenceFormat {
    Fasta,
    Fastq,
    Raw,
}

/// Incremental sequence parser over any buffered reader.
///
/// The parser holds at most one complete record. Wrapped FASTA and multiline
/// FASTQ are supported; quality bytes are validated and discarded.
pub struct SequenceReader<R: BufRead> {
    reader: R,
    label: String,
    fallback_id: String,
    format: SequenceFormat,
    pending: Option<Vec<u8>>,
    line: usize,
    ordinal: usize,
    finished: bool,
}

impl<R: BufRead> SequenceReader<R> {
    pub fn new(
        mut reader: R,
        label: impl Into<String>,
        fallback_id: impl Into<String>,
    ) -> Result<Self, String> {
        let label = label.into();
        let mut line_number = 0usize;
        let first = loop {
            let mut line = Vec::new();
            let bytes = reader
                .read_until(b'\n', &mut line)
                .map_err(|err| format!("failed to read {label} input: {err}"))?;
            if bytes == 0 {
                return Err(format!("{label} input contains no records"));
            }
            line_number += 1;
            trim_line_ending(&mut line);
            if !line.iter().all(u8::is_ascii) {
                return Err(format!("{label} input contains non-ASCII data"));
            }
            if !line.iter().all(u8::is_ascii_whitespace) {
                break line;
            }
        };
        let format = match first.first() {
            Some(b'>') => SequenceFormat::Fasta,
            Some(b'@') => SequenceFormat::Fastq,
            _ => SequenceFormat::Raw,
        };
        Ok(Self {
            reader,
            label,
            fallback_id: fallback_id.into(),
            format,
            pending: Some(first),
            line: line_number,
            ordinal: 0,
            finished: false,
        })
    }

    fn read_line(&mut self) -> Result<Option<Vec<u8>>, SequenceReadError> {
        let mut line = Vec::new();
        match self.reader.read_until(b'\n', &mut line) {
            Ok(0) => Ok(None),
            Ok(_) => {
                self.line += 1;
                trim_line_ending(&mut line);
                if line.iter().all(u8::is_ascii) {
                    Ok(Some(line))
                } else {
                    Err(self.error("sequence input contains non-ASCII data", true))
                }
            }
            Err(err) => Err(self.error(format!("failed to read input stream: {err}"), false)),
        }
    }

    fn error(&self, message: impl Into<String>, recoverable: bool) -> SequenceReadError {
        SequenceReadError {
            record_index: self.ordinal,
            line: self.line,
            message: format!("{} {}", self.label, message.into()),
            recoverable,
        }
    }

    fn next_fasta(&mut self) -> Option<Result<SequenceRecord, SequenceReadError>> {
        let header = match self.pending.take() {
            Some(line) => line,
            None => match self.read_line() {
                Ok(Some(line)) => line,
                Ok(None) => {
                    self.finished = true;
                    return None;
                }
                Err(error) => {
                    self.finished = true;
                    return Some(Err(error));
                }
            },
        };
        if !header.starts_with(b">") {
            self.ordinal += 1;
            return Some(Err(self.error("FASTA record does not start with '>'", true)));
        }
        let id = parse_record_id(&header[1..], self.ordinal + 1);
        let mut sequence = Vec::new();
        loop {
            match self.read_line() {
                Ok(Some(line)) if line.starts_with(b">") => {
                    self.pending = Some(line);
                    break;
                }
                Ok(Some(line)) => sequence.extend_from_slice(&line),
                Ok(None) => {
                    self.finished = true;
                    break;
                }
                Err(error) => {
                    self.finished = true;
                    return Some(Err(error));
                }
            }
        }
        let index = self.ordinal;
        self.ordinal += 1;
        Some(
            normalize_sequence(&sequence, &self.label)
                .map(|sequence| SequenceRecord::new(id, sequence))
                .map_err(|message| SequenceReadError {
                    record_index: index,
                    line: self.line,
                    message,
                    recoverable: true,
                }),
        )
    }

    fn next_fastq(&mut self) -> Option<Result<SequenceRecord, SequenceReadError>> {
        let header = match self.pending.take() {
            Some(line) => line,
            None => match self.read_line() {
                Ok(Some(line)) => line,
                Ok(None) => {
                    self.finished = true;
                    return None;
                }
                Err(error) => {
                    self.finished = true;
                    return Some(Err(error));
                }
            },
        };
        if !header.starts_with(b"@") {
            // Resynchronize at the next plausible record header.
            loop {
                match self.read_line() {
                    Ok(Some(line)) if line.starts_with(b"@") => {
                        self.pending = Some(line);
                        break;
                    }
                    Ok(Some(_)) => continue,
                    Ok(None) => {
                        self.finished = true;
                        break;
                    }
                    Err(error) => {
                        self.finished = true;
                        return Some(Err(error));
                    }
                }
            }
            self.ordinal += 1;
            return Some(Err(self.error("FASTQ record does not start with '@'", true)));
        }
        let id = parse_record_id(&header[1..], self.ordinal + 1);
        let mut sequence = Vec::new();
        loop {
            match self.read_line() {
                Ok(Some(line)) if line.starts_with(b"+") => break,
                Ok(Some(line)) => sequence.extend_from_slice(&line),
                Ok(None) => {
                    self.finished = true;
                    self.ordinal += 1;
                    return Some(Err(self.error("FASTQ record is truncated before '+'", true)));
                }
                Err(error) => {
                    self.finished = true;
                    return Some(Err(error));
                }
            }
        }
        let normalized = normalize_sequence(&sequence, &self.label);
        let expected = sequence
            .iter()
            .filter(|byte| !byte.is_ascii_whitespace())
            .count();
        let mut quality = 0usize;
        while quality < expected {
            match self.read_line() {
                Ok(Some(line)) => quality = quality.saturating_add(line.len()),
                Ok(None) => {
                    self.finished = true;
                    self.ordinal += 1;
                    return Some(Err(self.error("FASTQ quality is truncated", true)));
                }
                Err(error) => {
                    self.finished = true;
                    return Some(Err(error));
                }
            }
        }
        let index = self.ordinal;
        self.ordinal += 1;
        if quality != expected {
            return Some(Err(SequenceReadError {
                record_index: index,
                line: self.line,
                message: format!(
                    "{} FASTQ quality length {quality} does not match sequence length {expected}",
                    self.label
                ),
                recoverable: true,
            }));
        }
        Some(
            normalized
                .map(|sequence| SequenceRecord::new(id, sequence))
                .map_err(|message| SequenceReadError {
                    record_index: index,
                    line: self.line,
                    message,
                    recoverable: true,
                }),
        )
    }

    fn next_raw(&mut self) -> Option<Result<SequenceRecord, SequenceReadError>> {
        let mut sequence = self.pending.take().unwrap_or_default();
        loop {
            match self.read_line() {
                Ok(Some(line)) => sequence.extend_from_slice(&line),
                Ok(None) => break,
                Err(error) => {
                    self.finished = true;
                    return Some(Err(error));
                }
            }
        }
        self.finished = true;
        self.ordinal = 1;
        Some(
            normalize_sequence(&sequence, &self.label)
                .map(|sequence| SequenceRecord::new(self.fallback_id.clone(), sequence))
                .map_err(|message| SequenceReadError {
                    record_index: 0,
                    line: self.line,
                    message,
                    recoverable: true,
                }),
        )
    }
}

impl<R: BufRead> Iterator for SequenceReader<R> {
    type Item = Result<SequenceRecord, SequenceReadError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        match self.format {
            SequenceFormat::Fasta => self.next_fasta(),
            SequenceFormat::Fastq => self.next_fastq(),
            SequenceFormat::Raw => self.next_raw(),
        }
    }
}

/// Type-erased incremental input used by the CLI and batch APIs.
pub enum SequenceSource {
    Inline(Option<Result<SequenceRecord, SequenceReadError>>),
    Reader(SequenceReader<Box<dyn BufRead + Send>>),
}

impl Iterator for SequenceSource {
    type Item = Result<SequenceRecord, SequenceReadError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Inline(record) => record.take(),
            Self::Reader(reader) => reader.next(),
        }
    }
}

/// Open an inline or file-backed source without reading the entire stream.
pub fn open_sequence_source(
    inline: &Option<String>,
    path: &Option<impl AsRef<Path>>,
    label: &str,
) -> Result<SequenceSource, String> {
    match (inline, path) {
        (Some(_), Some(_)) => Err(format!(
            "provide either --{label} or --{label}-file, not both"
        )),
        (None, None) => Err(format!("missing --{label} or --{label}-file")),
        (Some(sequence), None) => Ok(SequenceSource::Inline(Some(
            normalize_sequence(sequence.as_bytes(), label)
                .map(|sequence| SequenceRecord::new(label, sequence))
                .map_err(|message| SequenceReadError {
                    record_index: 0,
                    line: 1,
                    message,
                    recoverable: true,
                }),
        ))),
        (None, Some(path)) => {
            let path = path.as_ref();
            let fallback_id = raw_id(path, label);
            let reader: Box<dyn BufRead + Send> = if path == Path::new("-") {
                Box::new(BufReader::new(io::stdin()))
            } else {
                let file = File::open(path).map_err(|err| {
                    format!("failed to open {label} file {}: {err}", path.display())
                })?;
                if path.extension().and_then(|value| value.to_str()) == Some("gz") {
                    Box::new(BufReader::new(MultiGzDecoder::new(file)))
                } else {
                    Box::new(BufReader::new(file))
                }
            };
            SequenceReader::new(reader, label, fallback_id).map(SequenceSource::Reader)
        }
    }
}

/// Compatibility helper that explicitly collects all records.
pub fn read_sequence_records(
    inline: &Option<String>,
    path: &Option<impl AsRef<Path>>,
    label: &str,
) -> Result<Vec<SequenceRecord>, String> {
    open_sequence_source(inline, path, label)?
        .map(|record| record.map_err(|error| error.to_string()))
        .collect()
}

fn trim_line_ending(line: &mut Vec<u8>) {
    if line.last() == Some(&b'\n') {
        line.pop();
    }
    if line.last() == Some(&b'\r') {
        line.pop();
    }
}

fn parse_record_id(header: &[u8], ordinal: usize) -> String {
    header
        .split(|byte| byte.is_ascii_whitespace())
        .find(|part| !part.is_empty())
        .and_then(|part| std::str::from_utf8(part).ok())
        .map(str::to_string)
        .unwrap_or_else(|| format!("record_{ordinal}"))
}

fn raw_id(path: &Path, label: &str) -> String {
    let path = if path.extension().and_then(|value| value.to_str()) == Some("gz") {
        path.file_stem().map(PathBuf::from).unwrap_or_default()
    } else {
        path.to_path_buf()
    };
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or(label)
        .to_string()
}

fn normalize_sequence(bytes: &[u8], label: &str) -> Result<Vec<u8>, String> {
    let mut normalized = Vec::new();
    for &byte in bytes {
        if byte.is_ascii_whitespace() {
            continue;
        }
        if !byte.is_ascii() {
            return Err(format!("{label} sequence contains non-ASCII data"));
        }
        normalized.push(byte.to_ascii_uppercase());
    }
    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::{write::GzEncoder, Compression};
    use std::io::{Cursor, Write};

    fn parse(text: &str) -> Vec<Result<SequenceRecord, SequenceReadError>> {
        SequenceReader::new(Cursor::new(text.as_bytes()), "query", "raw")
            .unwrap()
            .collect()
    }

    #[test]
    fn parses_wrapped_fasta_records_incrementally() {
        let records = parse(">q1\nac\nGT\n>q2 description\nTT\n");
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].as_ref().unwrap().sequence, b"ACGT");
        assert_eq!(records[1].as_ref().unwrap().id, "q2");
    }

    #[test]
    fn parses_multiline_fastq_and_discards_quality() {
        let records = parse("@q1\nac\nGT\n+\n!!\n!!\n@q2\nTG\n+\n##\n");
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].as_ref().unwrap().sequence, b"ACGT");
        assert_eq!(records[1].as_ref().unwrap().sequence, b"TG");
    }

    #[test]
    fn malformed_fastq_quality_is_record_scoped() {
        let records = parse("@q1\nAC\n+\n!!!\n@q2\nTG\n+\n##\n");
        assert!(records[0].is_err());
        assert_eq!(records[1].as_ref().unwrap().id, "q2");
    }

    #[test]
    fn multi_member_gzip_is_decoded_incrementally() {
        let mut first = GzEncoder::new(Vec::new(), Compression::fast());
        first.write_all(b">q1\nAC\n").unwrap();
        let mut bytes = first.finish().unwrap();
        let mut second = GzEncoder::new(Vec::new(), Compression::fast());
        second.write_all(b">q2\nGT\n").unwrap();
        bytes.extend(second.finish().unwrap());
        let decoder = MultiGzDecoder::new(Cursor::new(bytes));
        let records: Vec<_> = SequenceReader::new(BufReader::new(decoder), "query", "raw")
            .unwrap()
            .collect();
        assert_eq!(records.len(), 2);
    }
}
