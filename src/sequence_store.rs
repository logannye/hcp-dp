//! Opaque transient sequence storage for bounded-RSS ID resolution.

use crate::sequence_io::SequenceSource;
use redb::{Database, ReadableTable, TableDefinition};
use std::path::Path;
use tempfile::TempDir;

const QUERY_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("queries");
const TARGET_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("targets");
const CACHE_BYTES: usize = 16 * 1024 * 1024;
const INGEST_BATCH_RECORDS: usize = 4_096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SequenceKind {
    Query,
    Target,
}

impl SequenceKind {
    fn table(self) -> TableDefinition<'static, &'static str, &'static [u8]> {
        match self {
            Self::Query => QUERY_TABLE,
            Self::Target => TARGET_TABLE,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::Target => "target",
        }
    }
}

/// A temporary redb database deleted automatically when dropped.
pub(crate) struct SequenceStore {
    database: Database,
    _directory: TempDir,
}

impl SequenceStore {
    pub(crate) fn create(temp_dir: Option<&Path>) -> Result<Self, String> {
        let mut builder = tempfile::Builder::new();
        builder.prefix("hcp-align-sequences-");
        let directory = match temp_dir {
            Some(parent) => builder.tempdir_in(parent).map_err(|err| {
                format!(
                    "failed to create temporary sequence store in {}: {err}",
                    parent.display()
                )
            })?,
            None => builder
                .tempdir()
                .map_err(|err| format!("failed to create temporary sequence store: {err}"))?,
        };
        let path = directory.path().join("sequences.redb");
        let mut database_builder = Database::builder();
        database_builder.set_cache_size(CACHE_BYTES);
        let database = database_builder
            .create(&path)
            .map_err(|err| format!("failed to create temporary sequence database: {err}"))?;
        Ok(Self {
            database,
            _directory: directory,
        })
    }

    pub(crate) fn ingest(
        &self,
        kind: SequenceKind,
        mut source: SequenceSource,
    ) -> Result<u64, String> {
        let mut count = 0u64;
        let mut finished = false;
        while !finished {
            let transaction = self
                .database
                .begin_write()
                .map_err(|err| format!("failed to begin sequence-store write: {err}"))?;
            let mut batch_count = 0usize;
            {
                let mut table = transaction
                    .open_table(kind.table())
                    .map_err(|err| format!("failed to open sequence-store table: {err}"))?;
                while batch_count < INGEST_BATCH_RECORDS {
                    let Some(record) = source.next() else {
                        finished = true;
                        break;
                    };
                    let record = record.map_err(|error| error.to_string())?;
                    if table
                        .get(record.id.as_str())
                        .map_err(|err| format!("failed to query sequence store: {err}"))?
                        .is_some()
                    {
                        return Err(format!("duplicate {} id '{}'", kind.label(), record.id));
                    }
                    table
                        .insert(record.id.as_str(), record.sequence.as_slice())
                        .map_err(|err| {
                            format!("failed to write temporary sequence store: {err}")
                        })?;
                    batch_count += 1;
                    count += 1;
                }
            }
            transaction
                .commit()
                .map_err(|err| format!("failed to commit temporary sequence store: {err}"))?;
        }
        Ok(count)
    }

    pub(crate) fn get(&self, kind: SequenceKind, id: &str) -> Result<Option<Vec<u8>>, String> {
        let transaction = self
            .database
            .begin_read()
            .map_err(|err| format!("failed to begin sequence-store read: {err}"))?;
        let table = transaction
            .open_table(kind.table())
            .map_err(|err| format!("failed to open sequence-store table: {err}"))?;
        table
            .get(id)
            .map_err(|err| format!("failed to read temporary sequence store: {err}"))
            .map(|value| value.map(|guard| guard.value().to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence_io::{SequenceReadError, SequenceRecord};

    #[test]
    fn resolves_arbitrary_ids_and_rejects_duplicates() {
        let store = SequenceStore::create(None).unwrap();
        let source = SequenceSource::Inline(Some(Ok(SequenceRecord::new("q", b"AC".to_vec()))));
        store.ingest(SequenceKind::Query, source).unwrap();
        assert_eq!(
            store.get(SequenceKind::Query, "q").unwrap(),
            Some(b"AC".to_vec())
        );

        let duplicate = SequenceSource::Inline(Some(Ok(SequenceRecord::new("q", b"GT".to_vec()))));
        assert!(store
            .ingest(SequenceKind::Query, duplicate)
            .unwrap_err()
            .contains("duplicate"));
        let _type_anchor: Option<SequenceReadError> = None;
    }
}
