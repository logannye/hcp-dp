//! Sequence input parsing used by the CLI.

use std::{fs, path::Path};

/// A normalized sequence record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequenceRecord {
    pub id: String,
    pub sequence: Vec<u8>,
}

impl SequenceRecord {
    fn new(id: impl Into<String>, sequence: Vec<u8>) -> Self {
        Self {
            id: id.into(),
            sequence,
        }
    }
}

/// Read one or more sequence records from either an inline sequence or a file.
pub fn read_sequence_records(
    inline: &Option<String>,
    path: &Option<impl AsRef<Path>>,
    label: &str,
) -> Result<Vec<SequenceRecord>, String> {
    match (inline, path) {
        (Some(_), Some(_)) => Err(format!(
            "provide either --{label} or --{label}-file, not both"
        )),
        (None, None) => Err(format!("missing --{label} or --{label}-file")),
        (Some(sequence), None) => Ok(vec![SequenceRecord::new(
            label,
            normalize_sequence(sequence.as_bytes(), label)?,
        )]),
        (None, Some(path)) => {
            let path = path.as_ref();
            let bytes = fs::read(path)
                .map_err(|err| format!("failed to read {label} file {}: {err}", path.display()))?;
            parse_sequence_file(&bytes, label, path)
        }
    }
}

fn parse_sequence_file(
    bytes: &[u8],
    label: &str,
    path: &Path,
) -> Result<Vec<SequenceRecord>, String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| format!("{label} file must be valid UTF-8/ASCII sequence text"))?;
    let trimmed = text.trim_start();
    if trimmed.starts_with('>') {
        parse_fasta(trimmed, label)
    } else if trimmed.starts_with('@') {
        parse_fastq(trimmed, label)
    } else {
        let id = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .filter(|stem| !stem.is_empty())
            .unwrap_or(label);
        Ok(vec![SequenceRecord::new(
            id,
            normalize_sequence(text.as_bytes(), label)?,
        )])
    }
}

fn parse_fasta(text: &str, label: &str) -> Result<Vec<SequenceRecord>, String> {
    let mut records = Vec::new();
    let mut current_id: Option<String> = None;
    let mut current_seq = Vec::new();

    for line in text.lines() {
        if let Some(header) = line.strip_prefix('>') {
            if let Some(id) = current_id.take() {
                records.push(SequenceRecord::new(
                    id,
                    normalize_sequence(&current_seq, label)?,
                ));
                current_seq.clear();
            }
            current_id = Some(parse_record_id(header, records.len() + 1));
        } else if current_id.is_some() {
            current_seq.extend_from_slice(line.as_bytes());
        } else if !line.trim().is_empty() {
            return Err(format!(
                "{label} FASTA sequence appears before first header"
            ));
        }
    }

    if let Some(id) = current_id {
        records.push(SequenceRecord::new(
            id,
            normalize_sequence(&current_seq, label)?,
        ));
    }

    if records.is_empty() {
        return Err(format!("{label} FASTA input contains no records"));
    }
    Ok(records)
}

fn parse_fastq(text: &str, label: &str) -> Result<Vec<SequenceRecord>, String> {
    let lines: Vec<&str> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    if lines.is_empty() {
        return Err(format!("{label} FASTQ input contains no records"));
    }
    #[allow(clippy::manual_is_multiple_of)]
    if lines.len() % 4 != 0 {
        return Err(format!(
            "{label} FASTQ input must contain complete four-line records"
        ));
    }

    let mut records = Vec::with_capacity(lines.len() / 4);
    for (idx, chunk) in lines.chunks_exact(4).enumerate() {
        let header = chunk[0];
        let plus = chunk[2];
        if !header.starts_with('@') || !plus.starts_with('+') {
            return Err(format!(
                "{label} FASTQ record {} must have @ header and + separator",
                idx + 1
            ));
        }
        let id = parse_record_id(&header[1..], idx + 1);
        records.push(SequenceRecord::new(
            id,
            normalize_sequence(chunk[1].as_bytes(), label)?,
        ));
    }
    Ok(records)
}

fn parse_record_id(header: &str, ordinal: usize) -> String {
    header
        .split_whitespace()
        .next()
        .filter(|id| !id.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| format!("record_{ordinal}"))
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

    #[test]
    fn parses_wrapped_fasta_records() {
        let records = parse_fasta(">q1\nac\nGT\n>q2 description\nTT\n", "query").unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].id, "q1");
        assert_eq!(records[0].sequence, b"ACGT");
        assert_eq!(records[1].id, "q2");
        assert_eq!(records[1].sequence, b"TT");
    }

    #[test]
    fn parses_multiple_fastq_records() {
        let records = parse_fastq("@q1\nac\n+\n!!\n@q2\nTG\n+\n!!\n", "query").unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].sequence, b"AC");
        assert_eq!(records[1].sequence, b"TG");
    }
}
