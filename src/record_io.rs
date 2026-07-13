//! Streaming alignment-record readers and SAM spooling.

use crate::batch::{ErrorContext, ErrorRecord, ErrorStage};
use crate::oracle::{
    record_to_sam, records_from_paf, records_from_sam, AlignmentRecord, ProblemSpec,
};
use crate::sequence_store::{SequenceKind, SequenceStore};
use redb::{Database, ReadableTable, TableDefinition};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Seek, SeekFrom, Write};
use std::path::Path;
use tempfile::TempDir;

const SAM_REFERENCES: TableDefinition<&str, u64> = TableDefinition::new("sam_references");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlignmentInputFormat {
    Json,
    Jsonl,
    Paf,
    Sam,
}

#[derive(Clone, Debug)]
pub struct ResolvedAlignment {
    pub record: AlignmentRecord,
    pub query: Vec<u8>,
    pub target: Vec<u8>,
}

/// Incrementally parse alignment records and resolve their source sequences.
pub struct AlignmentRecordReader<'a> {
    reader: Box<dyn BufRead + Send>,
    format: AlignmentInputFormat,
    problem: Option<&'a ProblemSpec>,
    store: &'a SequenceStore,
    line: usize,
    current_source_line: usize,
    record_index: usize,
    finished: bool,
    json_array: Option<bool>,
    json_need_separator: bool,
}

impl<'a> AlignmentRecordReader<'a> {
    pub fn open(
        path: &Path,
        format: AlignmentInputFormat,
        problem: Option<&'a ProblemSpec>,
        store: &'a SequenceStore,
    ) -> Result<Self, String> {
        let reader: Box<dyn BufRead + Send> = if path == Path::new("-") {
            Box::new(BufReader::new(io::stdin()))
        } else {
            Box::new(BufReader::new(File::open(path).map_err(|err| {
                format!("failed to open alignment input {}: {err}", path.display())
            })?))
        };
        Ok(Self {
            reader,
            format,
            problem,
            store,
            line: 1,
            current_source_line: 1,
            record_index: 0,
            finished: false,
            json_array: None,
            json_need_separator: false,
        })
    }

    fn context(&self) -> ErrorContext {
        ErrorContext {
            pair_index: Some(self.record_index),
            source_line: Some(self.current_source_line),
            ..ErrorContext::default()
        }
    }

    fn error(
        &mut self,
        stage: ErrorStage,
        code: &str,
        message: impl Into<String>,
        recoverable: bool,
    ) -> ErrorRecord {
        if !recoverable {
            self.finished = true;
        }
        let mut error = ErrorRecord::new(stage, code, message, self.context());
        error.error.recoverable = recoverable;
        error
    }

    fn next_line(&mut self) -> Option<Result<String, ErrorRecord>> {
        loop {
            let mut bytes = Vec::new();
            match self.reader.read_until(b'\n', &mut bytes) {
                Ok(0) => {
                    self.finished = true;
                    return None;
                }
                Ok(_) => {
                    let source_line = self.line;
                    self.line += 1;
                    while matches!(bytes.last(), Some(b'\n' | b'\r')) {
                        bytes.pop();
                    }
                    if bytes.iter().all(u8::is_ascii) {
                        let text = String::from_utf8(bytes).expect("ASCII is UTF-8");
                        if text.trim().is_empty()
                            || (self.format == AlignmentInputFormat::Sam && text.starts_with('@'))
                        {
                            continue;
                        }
                        self.current_source_line = source_line;
                        return Some(Ok(text));
                    }
                    let mut error = ErrorRecord::new(
                        ErrorStage::RecordParse,
                        "non_ascii_record",
                        "alignment record contains non-ASCII data",
                        ErrorContext {
                            pair_index: Some(self.record_index),
                            source_line: Some(source_line),
                            ..ErrorContext::default()
                        },
                    );
                    error.error.recoverable = true;
                    return Some(Err(error));
                }
                Err(err) => {
                    return Some(Err(self.error(
                        ErrorStage::InputParse,
                        "input_read_failed",
                        format!("failed to read alignment input: {err}"),
                        false,
                    )));
                }
            }
        }
    }

    fn read_byte(&mut self) -> Result<Option<u8>, ErrorRecord> {
        let buffer = match self.reader.fill_buf() {
            Ok(buffer) => buffer,
            Err(err) => {
                return Err(self.error(
                    ErrorStage::InputParse,
                    "input_read_failed",
                    format!("failed to read JSON input: {err}"),
                    false,
                ));
            }
        };
        let Some(&byte) = buffer.first() else {
            return Ok(None);
        };
        self.reader.consume(1);
        if byte == b'\n' {
            self.line += 1;
        }
        Ok(Some(byte))
    }

    fn next_non_whitespace(&mut self) -> Result<Option<u8>, ErrorRecord> {
        loop {
            match self.read_byte()? {
                Some(byte) if byte.is_ascii_whitespace() => continue,
                value => return Ok(value),
            }
        }
    }

    fn json_object(&mut self, first: u8) -> Result<Vec<u8>, ErrorRecord> {
        if first != b'{' {
            return Err(self.error(
                ErrorStage::InputParse,
                "invalid_json_container",
                "JSON alignment input must contain objects",
                false,
            ));
        }
        let mut output = vec![first];
        let mut depth = 1usize;
        let mut in_string = false;
        let mut escaped = false;
        while depth > 0 {
            let byte = self.read_byte()?.ok_or_else(|| {
                self.error(
                    ErrorStage::InputParse,
                    "truncated_json",
                    "JSON input ended inside an alignment record",
                    false,
                )
            })?;
            output.push(byte);
            if in_string {
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == b'"' {
                    in_string = false;
                }
                continue;
            }
            match byte {
                b'"' => in_string = true,
                b'{' | b'[' => depth += 1,
                b'}' | b']' => depth = depth.saturating_sub(1),
                _ => {}
            }
        }
        Ok(output)
    }

    fn next_json(&mut self) -> Option<Result<AlignmentRecord, ErrorRecord>> {
        if self.finished {
            return None;
        }
        let mut first = match self.next_non_whitespace() {
            Ok(Some(byte)) => byte,
            Ok(None) if self.json_array.is_none() => {
                self.finished = true;
                return Some(Err(self.error(
                    ErrorStage::InputParse,
                    "empty_json",
                    "JSON input contains no alignment records",
                    false,
                )));
            }
            Ok(None) => {
                self.finished = true;
                return None;
            }
            Err(error) => return Some(Err(error)),
        };
        if self.json_array.is_none() {
            if first == b'[' {
                self.json_array = Some(true);
                first = match self.next_non_whitespace() {
                    Ok(Some(b']')) => {
                        self.finished = true;
                        return None;
                    }
                    Ok(Some(byte)) => byte,
                    Ok(None) => {
                        return Some(Err(self.error(
                            ErrorStage::InputParse,
                            "truncated_json",
                            "JSON array is not closed",
                            false,
                        )));
                    }
                    Err(error) => return Some(Err(error)),
                };
            } else {
                self.json_array = Some(false);
            }
        } else if self.json_need_separator {
            if self.json_array == Some(true) {
                match first {
                    b']' => {
                        self.finished = true;
                        return None;
                    }
                    b',' => {
                        first = match self.next_non_whitespace() {
                            Ok(Some(byte)) => byte,
                            Ok(None) => {
                                return Some(Err(self.error(
                                    ErrorStage::InputParse,
                                    "truncated_json",
                                    "JSON array ends after a comma",
                                    false,
                                )));
                            }
                            Err(error) => return Some(Err(error)),
                        };
                    }
                    _ => {
                        return Some(Err(self.error(
                            ErrorStage::InputParse,
                            "invalid_json_separator",
                            "expected ',' or ']' after JSON alignment record",
                            false,
                        )));
                    }
                }
            } else {
                self.finished = true;
                return Some(Err(self.error(
                    ErrorStage::InputParse,
                    "trailing_json",
                    "unexpected data after JSON alignment record",
                    false,
                )));
            }
        }
        self.current_source_line = self.line;
        let bytes = match self.json_object(first) {
            Ok(bytes) => bytes,
            Err(error) => return Some(Err(error)),
        };
        self.json_need_separator = true;
        if self.json_array == Some(false) {
            // A following call validates that only whitespace remains.
        }
        Some(serde_json::from_slice(&bytes).map_err(|err| {
            self.error(
                ErrorStage::RecordParse,
                "invalid_alignment_json",
                err.to_string(),
                false,
            )
        }))
    }

    fn parse_line_record(&mut self, line: &str) -> Result<AlignmentRecord, ErrorRecord> {
        match self.format {
            AlignmentInputFormat::Jsonl => serde_json::from_str(line).map_err(|err| {
                self.error(
                    ErrorStage::RecordParse,
                    "invalid_alignment_json",
                    err.to_string(),
                    true,
                )
            }),
            AlignmentInputFormat::Paf | AlignmentInputFormat::Sam => {
                let fields: Vec<&str> = line.split('\t').collect();
                let (query_id, target_id, minimum) = if self.format == AlignmentInputFormat::Paf {
                    (fields.first().copied(), fields.get(5).copied(), 12usize)
                } else {
                    (fields.first().copied(), fields.get(2).copied(), 11usize)
                };
                if fields.len() < minimum {
                    return Err(self.error(
                        ErrorStage::RecordParse,
                        "too_few_fields",
                        format!("record has fewer than {minimum} tab-separated fields"),
                        true,
                    ));
                }
                let query_id = query_id.unwrap_or_default();
                let target_id = target_id.unwrap_or_default();
                let query = self
                    .store
                    .get(SequenceKind::Query, query_id)
                    .map_err(|err| {
                        self.error(ErrorStage::Resolve, "source_read_failed", err, false)
                    })?
                    .ok_or_else(|| {
                        self.error(
                            ErrorStage::Resolve,
                            "missing_query_id",
                            format!("query id '{query_id}' is missing from query sources"),
                            true,
                        )
                    })?;
                let target = self
                    .store
                    .get(SequenceKind::Target, target_id)
                    .map_err(|err| {
                        self.error(ErrorStage::Resolve, "source_read_failed", err, false)
                    })?
                    .ok_or_else(|| {
                        self.error(
                            ErrorStage::Resolve,
                            "missing_target_id",
                            format!("target id '{target_id}' is missing from target sources"),
                            true,
                        )
                    })?;
                let mut queries = HashMap::new();
                queries.insert(query_id.to_string(), query);
                let mut targets = HashMap::new();
                targets.insert(target_id.to_string(), target);
                let problem = self.problem.ok_or_else(|| {
                    self.error(
                        ErrorStage::RecordParse,
                        "missing_problem",
                        "--mode and scoring are required for PAF/SAM",
                        false,
                    )
                })?;
                let parsed = if self.format == AlignmentInputFormat::Paf {
                    records_from_paf(line, problem, &queries, &targets)
                } else {
                    records_from_sam(line, problem, &queries, &targets)
                };
                parsed
                    .and_then(|mut records| {
                        records
                            .pop()
                            .ok_or_else(|| "record did not contain an alignment".to_string())
                    })
                    .map_err(|err| {
                        self.error(
                            ErrorStage::RecordParse,
                            "invalid_alignment_record",
                            err,
                            true,
                        )
                    })
            }
            AlignmentInputFormat::Json => unreachable!("JSON uses object framing"),
        }
    }

    fn resolve(&mut self, mut record: AlignmentRecord) -> Result<ResolvedAlignment, ErrorRecord> {
        if matches!(
            self.format,
            AlignmentInputFormat::Paf | AlignmentInputFormat::Sam
        ) {
            record.pair.index = self.record_index;
        }
        let query_id = record.pair.query.id.clone();
        let target_id = record.pair.target.id.clone();
        let context = ErrorContext {
            pair_index: Some(record.pair.index),
            query_id: Some(query_id.clone()),
            target_id: Some(target_id.clone()),
            source_line: Some(self.current_source_line),
        };
        let query = self
            .store
            .get(SequenceKind::Query, &query_id)
            .map_err(|err| {
                let mut error = ErrorRecord::new(
                    ErrorStage::Resolve,
                    "source_read_failed",
                    err,
                    context.clone(),
                );
                error.error.recoverable = false;
                self.finished = true;
                error
            })?
            .ok_or_else(|| {
                ErrorRecord::new(
                    ErrorStage::Resolve,
                    "missing_query_id",
                    format!("query id '{query_id}' is missing from query sources"),
                    context.clone(),
                )
            })?;
        let target = self
            .store
            .get(SequenceKind::Target, &target_id)
            .map_err(|err| {
                let mut error = ErrorRecord::new(
                    ErrorStage::Resolve,
                    "source_read_failed",
                    err,
                    context.clone(),
                );
                error.error.recoverable = false;
                self.finished = true;
                error
            })?
            .ok_or_else(|| {
                ErrorRecord::new(
                    ErrorStage::Resolve,
                    "missing_target_id",
                    format!("target id '{target_id}' is missing from target sources"),
                    context,
                )
            })?;
        Ok(ResolvedAlignment {
            record,
            query,
            target,
        })
    }
}

impl Iterator for AlignmentRecordReader<'_> {
    type Item = Result<ResolvedAlignment, ErrorRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        let parsed = if self.format == AlignmentInputFormat::Json {
            self.next_json()?
        } else {
            match self.next_line()? {
                Ok(line) => self.parse_line_record(&line),
                Err(error) => Err(error),
            }
        };
        let result = parsed.and_then(|record| self.resolve(record));
        self.record_index += 1;
        Some(result)
    }
}

/// Disk-backed SAM body and reference-header spool.
pub struct SamSpool {
    references: Database,
    body: File,
    _directory: TempDir,
}

impl SamSpool {
    pub fn create(temp_dir: Option<&Path>) -> Result<Self, String> {
        let mut builder = tempfile::Builder::new();
        builder.prefix("hcp-align-sam-");
        let directory = match temp_dir {
            Some(parent) => builder.tempdir_in(parent),
            None => builder.tempdir(),
        }
        .map_err(|err| format!("failed to create SAM spool: {err}"))?;
        let mut database_builder = Database::builder();
        database_builder.set_cache_size(2 * 1024 * 1024);
        let references = database_builder
            .create(directory.path().join("references.redb"))
            .map_err(|err| format!("failed to create SAM reference spool: {err}"))?;
        let body = File::options()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(directory.path().join("body.sam"))
            .map_err(|err| format!("failed to create SAM body spool: {err}"))?;
        Ok(Self {
            references,
            body,
            _directory: directory,
        })
    }

    pub fn push(&mut self, record: &AlignmentRecord, query: &[u8]) -> Result<(), String> {
        let transaction = self
            .references
            .begin_write()
            .map_err(|err| format!("failed to begin SAM reference write: {err}"))?;
        {
            let mut table = transaction
                .open_table(SAM_REFERENCES)
                .map_err(|err| format!("failed to open SAM reference spool: {err}"))?;
            let existing = table
                .get(record.pair.target.id.as_str())
                .map_err(|err| format!("failed to read SAM reference spool: {err}"))?
                .map(|guard| guard.value());
            if let Some(existing) = existing {
                if existing != record.pair.target.length as u64 {
                    return Err(format!(
                        "target '{}' has inconsistent lengths in SAM records",
                        record.pair.target.id
                    ));
                }
            } else {
                table
                    .insert(
                        record.pair.target.id.as_str(),
                        record.pair.target.length as u64,
                    )
                    .map_err(|err| format!("failed to update SAM reference spool: {err}"))?;
            }
        }
        transaction
            .commit()
            .map_err(|err| format!("failed to commit SAM reference spool: {err}"))?;
        self.body
            .write_all(record_to_sam(record, query)?.as_bytes())
            .map_err(|err| format!("failed to write SAM body spool: {err}"))
    }

    pub fn finish(mut self, output: &mut dyn Write) -> Result<(), String> {
        output
            .write_all(b"@HD\tVN:1.6\tSO:unknown\n")
            .map_err(|err| format!("failed to write SAM header: {err}"))?;
        let transaction = self
            .references
            .begin_read()
            .map_err(|err| format!("failed to read SAM references: {err}"))?;
        let table = transaction
            .open_table(SAM_REFERENCES)
            .map_err(|err| format!("failed to open SAM references: {err}"))?;
        for entry in table
            .iter()
            .map_err(|err| format!("failed to iterate SAM references: {err}"))?
        {
            let (id, len) = entry.map_err(|err| format!("failed to read SAM reference: {err}"))?;
            writeln!(output, "@SQ\tSN:{}\tLN:{}", id.value(), len.value())
                .map_err(|err| format!("failed to write SAM reference: {err}"))?;
        }
        self.body
            .seek(SeekFrom::Start(0))
            .map_err(|err| format!("failed to rewind SAM body spool: {err}"))?;
        io::copy(&mut self.body, output)
            .map_err(|err| format!("failed to copy SAM body: {err}"))?;
        Ok(())
    }
}
