//! Shared alignment trace formatting.

use serde::Serialize;

/// One alignment operation between query (`s`) and target (`t`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AlignmentOpKind {
    Match,
    Mismatch,
    GapInTarget,
    GapInQuery,
}

impl AlignmentOpKind {
    fn symbol(self) -> char {
        match self {
            Self::Match => '=',
            Self::Mismatch => 'X',
            Self::GapInTarget => 'D',
            Self::GapInQuery => 'I',
        }
    }
}

/// A single consumed step in an alignment path.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct AlignmentStep {
    pub op: AlignmentOpKind,
    pub query_pos: Option<usize>,
    pub target_pos: Option<usize>,
    pub query_base: Option<char>,
    pub target_base: Option<char>,
}

/// User-facing trace derived from a returned DP path.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct AlignmentTrace {
    pub query_start: usize,
    pub query_end: usize,
    pub target_start: usize,
    pub target_end: usize,
    pub cigar: String,
    pub operations: Vec<AlignmentStep>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aligned_query: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aligned_target: Option<String>,
}

impl AlignmentTrace {
    /// Build an alignment trace from a monotone cell path.
    pub fn from_cells(
        query: &[u8],
        target: &[u8],
        path: &[(usize, usize)],
        show_alignment: bool,
    ) -> Self {
        if path.is_empty() {
            return Self {
                query_start: 0,
                query_end: 0,
                target_start: 0,
                target_end: 0,
                cigar: String::new(),
                operations: Vec::new(),
                aligned_query: show_alignment.then(String::new),
                aligned_target: show_alignment.then(String::new),
            };
        }

        let (query_start, target_start) = path[0];
        let (query_end, target_end) = *path.last().expect("non-empty path has a last cell");
        let mut operations = Vec::with_capacity(path.len().saturating_sub(1));
        let mut aligned_query = show_alignment.then(String::new);
        let mut aligned_target = show_alignment.then(String::new);

        for window in path.windows(2) {
            let (a_row, a_col) = window[0];
            let (b_row, b_col) = window[1];
            let di = b_row
                .checked_sub(a_row)
                .expect("alignment path rows must be monotone");
            let dj = b_col
                .checked_sub(a_col)
                .expect("alignment path columns must be monotone");

            let step = match (di, dj) {
                (1, 1) => {
                    let query_base = byte_to_char(
                        *query
                            .get(a_row)
                            .expect("query path coordinate out of bounds"),
                    );
                    let target_base = byte_to_char(
                        *target
                            .get(a_col)
                            .expect("target path coordinate out of bounds"),
                    );
                    let op = if query_base == target_base {
                        AlignmentOpKind::Match
                    } else {
                        AlignmentOpKind::Mismatch
                    };
                    if let Some(buf) = aligned_query.as_mut() {
                        buf.push(query_base);
                    }
                    if let Some(buf) = aligned_target.as_mut() {
                        buf.push(target_base);
                    }
                    AlignmentStep {
                        op,
                        query_pos: Some(a_row),
                        target_pos: Some(a_col),
                        query_base: Some(query_base),
                        target_base: Some(target_base),
                    }
                }
                (1, 0) => {
                    let query_base = byte_to_char(
                        *query
                            .get(a_row)
                            .expect("query path coordinate out of bounds"),
                    );
                    if let Some(buf) = aligned_query.as_mut() {
                        buf.push(query_base);
                    }
                    if let Some(buf) = aligned_target.as_mut() {
                        buf.push('-');
                    }
                    AlignmentStep {
                        op: AlignmentOpKind::GapInTarget,
                        query_pos: Some(a_row),
                        target_pos: None,
                        query_base: Some(query_base),
                        target_base: None,
                    }
                }
                (0, 1) => {
                    let target_base = byte_to_char(
                        *target
                            .get(a_col)
                            .expect("target path coordinate out of bounds"),
                    );
                    if let Some(buf) = aligned_query.as_mut() {
                        buf.push('-');
                    }
                    if let Some(buf) = aligned_target.as_mut() {
                        buf.push(target_base);
                    }
                    AlignmentStep {
                        op: AlignmentOpKind::GapInQuery,
                        query_pos: None,
                        target_pos: Some(a_col),
                        query_base: None,
                        target_base: Some(target_base),
                    }
                }
                _ => panic!("alignment path contains a non-unit move"),
            };
            operations.push(step);
        }

        let cigar = cigar_from_ops(&operations);
        Self {
            query_start,
            query_end,
            target_start,
            target_end,
            cigar,
            operations,
            aligned_query,
            aligned_target,
        }
    }
}

fn cigar_from_ops(operations: &[AlignmentStep]) -> String {
    let mut cigar = String::new();
    let mut iter = operations.iter();
    let Some(first) = iter.next() else {
        return cigar;
    };

    let mut current = first.op;
    let mut count = 1usize;
    for step in iter {
        if step.op == current {
            count += 1;
        } else {
            push_cigar_run(&mut cigar, count, current);
            current = step.op;
            count = 1;
        }
    }
    push_cigar_run(&mut cigar, count, current);
    cigar
}

fn push_cigar_run(cigar: &mut String, count: usize, op: AlignmentOpKind) {
    cigar.push_str(&count.to_string());
    cigar.push(op.symbol());
}

fn byte_to_char(byte: u8) -> char {
    if byte.is_ascii() {
        byte as char
    } else {
        char::REPLACEMENT_CHARACTER
    }
}
