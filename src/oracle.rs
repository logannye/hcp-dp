//! Stable v0.2 alignment-oracle API.

use crate::alignment::{AlignmentOpKind, AlignmentStep, AlignmentTrace};
use crate::engine::{HcpEngine, HcpRunStats};
use crate::problems::{
    edit_distance::{distance_myers, trace_adaptive_banded, EditDistanceProblem},
    nw_affine::{trace_banded_affine, NwAffineProblem, NwAffineState},
    nw_align::NwProblem,
    semiglobal::{SemiGlobalCell, SemiGlobalProblem},
    smith_waterman::{SmithWatermanProblem, SwCell},
};
use crate::scoring::{SubstitutionMatrix, SubstitutionScoring};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::str::FromStr;
use std::time::Instant;

/// Structured record schema emitted by the v0.2 oracle.
pub const SCHEMA_VERSION: &str = "hcp-align.v2";
/// Default maximum number of reference DP cells for exact verification.
pub const DEFAULT_MAX_CELLS: u64 = 25_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AlignmentMode {
    EditDistance,
    GlobalLinear,
    GlobalAffine,
    LocalLinear,
    SemiglobalLinear,
}

impl Display for AlignmentMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let value = match self {
            Self::EditDistance => "edit-distance",
            Self::GlobalLinear => "global-linear",
            Self::GlobalAffine => "global-affine",
            Self::LocalLinear => "local-linear",
            Self::SemiglobalLinear => "semiglobal-linear",
        };
        f.write_str(value)
    }
}

impl FromStr for AlignmentMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "edit-distance" => Ok(Self::EditDistance),
            "global-linear" => Ok(Self::GlobalLinear),
            "global-affine" => Ok(Self::GlobalAffine),
            "local-linear" => Ok(Self::LocalLinear),
            "semiglobal-linear" => Ok(Self::SemiglobalLinear),
            _ => Err(format!("unsupported alignment mode '{value}'")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GapModel {
    None,
    Linear,
    Affine,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum SubstitutionScheme {
    MatchMismatch {
        match_score: i32,
        mismatch_penalty: i32,
    },
    Blosum62,
    Matrix {
        name: String,
        alphabet: Vec<String>,
        scores: Vec<Vec<i32>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        source_sha256: Option<String>,
    },
}

impl Default for SubstitutionScheme {
    fn default() -> Self {
        Self::MatchMismatch {
            match_score: 2,
            mismatch_penalty: 1,
        }
    }
}

impl SubstitutionScheme {
    fn internal(&self) -> Result<SubstitutionScoring, String> {
        match self {
            Self::MatchMismatch {
                match_score,
                mismatch_penalty,
            } => Ok(SubstitutionScoring::match_mismatch(
                *match_score,
                *mismatch_penalty,
            )),
            Self::Blosum62 => Ok(SubstitutionScoring::blosum62()),
            Self::Matrix {
                name,
                alphabet,
                scores,
                ..
            } => {
                let alphabet = alphabet
                    .iter()
                    .map(|symbol| {
                        let bytes = symbol.as_bytes();
                        if bytes.len() == 1 && bytes[0].is_ascii() {
                            Ok(bytes[0].to_ascii_uppercase())
                        } else {
                            Err(format!(
                                "matrix alphabet symbol '{symbol}' must be one ASCII character"
                            ))
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                SubstitutionMatrix::new(name.clone(), alphabet, scores.clone())
                    .map(SubstitutionScoring::matrix)
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum ScoringScheme {
    EditDistance,
    Linear {
        substitution: SubstitutionScheme,
        gap: i32,
    },
    Affine {
        substitution: SubstitutionScheme,
        gap_open: i32,
        gap_extend: i32,
    },
}

impl ScoringScheme {
    pub fn default_for(mode: AlignmentMode) -> Self {
        match mode {
            AlignmentMode::EditDistance => Self::EditDistance,
            AlignmentMode::GlobalAffine => Self::Affine {
                substitution: SubstitutionScheme::default(),
                gap_open: -3,
                gap_extend: -1,
            },
            _ => Self::Linear {
                substitution: SubstitutionScheme::default(),
                gap: -2,
            },
        }
    }

    pub fn gap_model(&self) -> GapModel {
        match self {
            Self::EditDistance => GapModel::None,
            Self::Linear { .. } => GapModel::Linear,
            Self::Affine { .. } => GapModel::Affine,
        }
    }

    fn validate_for(&self, mode: AlignmentMode) -> Result<(), String> {
        match (mode, self) {
            (AlignmentMode::EditDistance, Self::EditDistance)
            | (AlignmentMode::GlobalAffine, Self::Affine { .. })
            | (
                AlignmentMode::GlobalLinear
                | AlignmentMode::LocalLinear
                | AlignmentMode::SemiglobalLinear,
                Self::Linear { .. },
            ) => Ok(()),
            _ => Err(format!(
                "scoring model '{}' is incompatible with mode {mode}",
                match self {
                    Self::EditDistance => "edit-distance",
                    Self::Linear { .. } => "linear",
                    Self::Affine { .. } => "affine",
                }
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CigarOp {
    Match,
    Mismatch,
    Insertion,
    Deletion,
}

impl CigarOp {
    pub const fn symbol(self) -> char {
        match self {
            Self::Match => '=',
            Self::Mismatch => 'X',
            Self::Insertion => 'I',
            Self::Deletion => 'D',
        }
    }

    fn from_symbol(symbol: char) -> Result<Self, String> {
        match symbol {
            '=' => Ok(Self::Match),
            'X' => Ok(Self::Mismatch),
            'I' => Ok(Self::Insertion),
            'D' => Ok(Self::Deletion),
            _ => Err(format!("unsupported canonical CIGAR operation '{symbol}'")),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CigarRun {
    pub len: usize,
    pub op: CigarOp,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cigar {
    runs: Vec<CigarRun>,
}

impl Cigar {
    pub fn new(runs: Vec<CigarRun>) -> Result<Self, String> {
        if runs.iter().any(|run| run.len == 0) {
            return Err("CIGAR runs must be positive".to_string());
        }
        let mut normalized: Vec<CigarRun> = Vec::with_capacity(runs.len());
        for run in runs {
            if let Some(last) = normalized.last_mut().filter(|last| last.op == run.op) {
                last.len = last
                    .len
                    .checked_add(run.len)
                    .ok_or_else(|| "CIGAR run length overflowed usize".to_string())?;
            } else {
                normalized.push(run);
            }
        }
        Ok(Self { runs: normalized })
    }

    pub fn runs(&self) -> &[CigarRun] {
        &self.runs
    }

    pub fn query_consumed(&self) -> usize {
        self.runs
            .iter()
            .filter(|run| run.op != CigarOp::Deletion)
            .map(|run| run.len)
            .sum()
    }

    pub fn target_consumed(&self) -> usize {
        self.runs
            .iter()
            .filter(|run| run.op != CigarOp::Insertion)
            .map(|run| run.len)
            .sum()
    }

    fn from_trace(trace: &AlignmentTrace) -> Result<Self, String> {
        trace.cigar.parse()
    }
}

impl Display for Cigar {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for run in &self.runs {
            write!(f, "{}{}", run.len, run.op.symbol())?;
        }
        Ok(())
    }
}

impl FromStr for Cigar {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.is_empty() || value == "*" {
            return Ok(Self::default());
        }
        let mut count = 0usize;
        let mut has_count = false;
        let mut runs = Vec::new();
        for ch in value.chars() {
            if let Some(digit) = ch.to_digit(10) {
                has_count = true;
                count = count
                    .checked_mul(10)
                    .and_then(|v| v.checked_add(digit as usize))
                    .ok_or_else(|| "CIGAR run length overflowed usize".to_string())?;
                continue;
            }
            if !has_count || count == 0 {
                return Err(format!("invalid CIGAR run before '{ch}'"));
            }
            runs.push(CigarRun {
                len: count,
                op: CigarOp::from_symbol(ch)?,
            });
            count = 0;
            has_count = false;
        }
        if has_count {
            return Err("CIGAR ends with a length and no operation".to_string());
        }
        Self::new(runs)
    }
}

impl Serialize for Cigar {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for Cigar {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;
        text.parse().map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AlignmentSpan {
    pub start: usize,
    pub end: usize,
}

impl AlignmentSpan {
    pub fn len(self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(self) -> bool {
        self.start == self.end
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Producer {
    pub name: String,
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
    pub backend: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequenceIdentity {
    pub id: String,
    pub length: usize,
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairIdentity {
    pub index: usize,
    pub query: SequenceIdentity,
    pub target: SequenceIdentity,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProblemSpec {
    pub mode: AlignmentMode,
    pub scoring: ScoringScheme,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveKind {
    Score,
    Distance,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Objective {
    pub kind: ObjectiveKind,
    pub value: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceRecord {
    pub query_span: AlignmentSpan,
    pub target_span: AlignmentSpan,
    pub cigar: Cigar,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operations: Option<Vec<AlignmentStep>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AlignmentResult {
    pub objective: Objective,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace: Option<TraceRecord>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationLevel {
    Structure,
    Score,
    Optimal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationStatus {
    Verified,
    Invalid,
    Inconclusive,
    Error,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckResult {
    pub name: String,
    pub passed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerificationSummary {
    pub level: VerificationLevel,
    pub status: VerificationStatus,
    pub checks: Vec<CheckResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metrics {
    pub elapsed_ms: f64,
    pub work_cells: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntegrityFingerprint {
    pub version: String,
    pub hash_algorithm: String,
    pub deterministic_fields_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AlignmentRecord {
    pub schema_version: String,
    pub record_type: String,
    pub producer: Producer,
    pub pair: PairIdentity,
    pub problem: ProblemSpec,
    pub alignment: AlignmentResult,
    pub verification: VerificationSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<Metrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub integrity: Option<IntegrityFingerprint>,
}

#[derive(Clone, Debug)]
pub struct SolveRequest {
    pub pair_index: usize,
    pub query_id: String,
    pub target_id: String,
    pub query: Vec<u8>,
    pub target: Vec<u8>,
    pub problem: ProblemSpec,
    pub options: SolveOptions,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolveScope {
    Alignment,
    Score,
}

#[derive(Clone, Debug)]
pub struct SolveOptions {
    pub scope: SolveScope,
    pub backend: String,
    pub block_size: Option<usize>,
    pub band: usize,
    pub cross_check: bool,
    pub include_operations: bool,
    pub integrity: bool,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            scope: SolveScope::Alignment,
            backend: "auto".to_string(),
            block_size: None,
            band: 64,
            cross_check: false,
            include_operations: false,
            integrity: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct VerificationRequest {
    pub record: AlignmentRecord,
    pub query: Vec<u8>,
    pub target: Vec<u8>,
    pub level: VerificationLevel,
    pub max_cells: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerificationReport {
    pub schema_version: String,
    pub record_type: String,
    pub pair: PairIdentity,
    pub problem: ProblemSpec,
    pub submitted: AlignmentResult,
    pub verification: VerificationSummary,
    pub metrics: Metrics,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComparisonInputs {
    pub query: String,
    pub target: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComparisonReport {
    pub schema_version: String,
    pub record_type: String,
    pub inputs: ComparisonInputs,
    pub pair: PairIdentity,
    pub problem: ProblemSpec,
    pub left: AlignmentRecord,
    pub right: AlignmentRecord,
    pub objectives_equal: bool,
    pub traces_equal: bool,
    pub equivalent: bool,
    pub differences: Vec<String>,
}

/// Generate an exact reference score or alignment.
pub fn solve(request: SolveRequest) -> Result<AlignmentRecord, String> {
    request.problem.scoring.validate_for(request.problem.mode)?;
    validate_sequences(&request.problem.scoring, &request.query, &request.target)?;
    validate_solve_backend(
        request.problem.mode,
        request.options.scope,
        &request.options.backend,
    )?;
    let start = Instant::now();
    let work_cells = cell_count(request.query.len(), request.target.len());
    let (objective, trace, backend, path_check, primary_check) = match request.options.scope {
        SolveScope::Score if request.problem.mode == AlignmentMode::EditDistance => (
            Objective {
                kind: ObjectiveKind::Distance,
                value: i64::from(distance_myers(&request.query, &request.target)),
            },
            None,
            "myers".to_string(),
            true,
            "objective_computed",
        ),
        SolveScope::Score => {
            let (objective, _trace, backend, path_check) = solve_trace(&request)?;
            (
                objective,
                None,
                backend,
                path_check,
                "backend_result_rescore",
            )
        }
        SolveScope::Alignment => {
            let (objective, trace, backend, path_check) = solve_trace(&request)?;
            (objective, trace, backend, path_check, "path_rescore")
        }
    };

    let mut checks = vec![CheckResult {
        name: primary_check.to_string(),
        passed: path_check,
        detail: (!path_check)
            .then(|| "backend result does not realize the reported objective".to_string()),
    }];
    if request.options.cross_check {
        let reference = reference_objective(&request.problem, &request.query, &request.target)?;
        let passed = reference == objective;
        checks.push(CheckResult {
            name: "reference_optimum".to_string(),
            passed,
            detail: (!passed).then(|| {
                format!(
                    "reported objective {} differs from reference {}",
                    objective.value, reference.value
                )
            }),
        });
    }
    let status = if checks.iter().all(|check| check.passed) {
        VerificationStatus::Verified
    } else {
        VerificationStatus::Invalid
    };
    let mut record = AlignmentRecord {
        schema_version: SCHEMA_VERSION.to_string(),
        record_type: "alignment".to_string(),
        producer: Producer {
            name: "hcp-align".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            commit: option_env!("HCP_DP_GIT_COMMIT").map(str::to_string),
            backend,
        },
        pair: pair_identity(
            request.pair_index,
            &request.query_id,
            &request.target_id,
            &request.query,
            &request.target,
        ),
        problem: request.problem,
        alignment: AlignmentResult { objective, trace },
        verification: VerificationSummary {
            level: if request.options.cross_check {
                VerificationLevel::Optimal
            } else if request.options.scope == SolveScope::Alignment {
                VerificationLevel::Score
            } else {
                VerificationLevel::Structure
            },
            status,
            checks,
            reference_backend: request
                .options
                .cross_check
                .then(|| "baseline-rolling-v1".to_string()),
            reason: None,
        },
        metrics: Some(Metrics {
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            work_cells,
        }),
        integrity: None,
    };
    if request.options.integrity {
        record.integrity = Some(integrity_for(&record)?);
    }
    Ok(record)
}

fn validate_solve_backend(
    mode: AlignmentMode,
    scope: SolveScope,
    backend: &str,
) -> Result<(), String> {
    let allowed: &[&str] = if scope == SolveScope::Score {
        &["auto"]
    } else {
        match mode {
            AlignmentMode::EditDistance => &["auto", "adaptive-banded"],
            AlignmentMode::GlobalAffine => &["auto", "hcp", "banded-affine"],
            AlignmentMode::GlobalLinear
            | AlignmentMode::LocalLinear
            | AlignmentMode::SemiglobalLinear => &["auto", "hcp"],
        }
    };
    if allowed.contains(&backend) {
        Ok(())
    } else {
        Err(format!(
            "backend '{backend}' is not supported for {mode} with scope {scope:?}; choose {}",
            allowed.join(", ")
        ))
    }
}

fn solve_trace(
    request: &SolveRequest,
) -> Result<(Objective, Option<TraceRecord>, String, bool), String> {
    let query = &request.query;
    let target = &request.target;
    match (&request.problem.mode, &request.problem.scoring) {
        (AlignmentMode::EditDistance, ScoringScheme::EditDistance) => {
            let trace = trace_adaptive_banded(query, target);
            let problem = EditDistanceProblem::new(query, target);
            let path_score = problem.score_path(&trace.path);
            let alignment = AlignmentTrace::from_cells(query, target, &trace.path, false);
            Ok((
                Objective {
                    kind: ObjectiveKind::Distance,
                    value: i64::from(trace.distance),
                },
                Some(trace_record(alignment, request.options.include_operations)?),
                "adaptive-banded".to_string(),
                path_score == Some(trace.distance),
            ))
        }
        (AlignmentMode::GlobalLinear, ScoringScheme::Linear { substitution, gap }) => {
            let scoring = substitution.internal()?;
            let problem = NwProblem::with_scoring(query, target, scoring, *gap);
            let (score, path, _) = run_hcp(problem.clone(), request.options.block_size);
            let path_score = problem.score_path(&path);
            let alignment = AlignmentTrace::from_cells(query, target, &path, false);
            Ok((
                Objective {
                    kind: ObjectiveKind::Score,
                    value: i64::from(score),
                },
                Some(trace_record(alignment, request.options.include_operations)?),
                "hcp".to_string(),
                path_score == Some(score),
            ))
        }
        (
            AlignmentMode::GlobalAffine,
            ScoringScheme::Affine {
                substitution,
                gap_open,
                gap_extend,
            },
        ) => {
            let scoring = substitution.internal()?;
            let problem =
                NwAffineProblem::with_scoring(query, target, scoring, *gap_open, *gap_extend);
            let requested = request.options.backend.as_str();
            let (score, path, backend) = match requested {
                "banded-affine" => {
                    let trace = trace_banded_affine(&problem, request.options.band).ok_or_else(|| {
                        format!(
                            "banded-affine did not find an optimum inside band {}; use --backend auto or increase --band",
                            request.options.band
                        )
                    })?;
                    (trace.score, trace.path, "banded-affine")
                }
                "auto" => match trace_banded_affine(&problem, request.options.band) {
                    Some(trace) => (trace.score, trace.path, "banded-affine"),
                    None => {
                        let (score, path, _) =
                            HcpEngine::linear_space(problem.clone()).run_with_stats();
                        (score, path, "hcp-linear")
                    }
                },
                "hcp" => {
                    let (score, path, _) = run_hcp(problem.clone(), request.options.block_size);
                    (score, path, "hcp")
                }
                other => return Err(format!("unsupported affine backend '{other}'")),
            };
            let cells: Vec<(usize, usize)> = path
                .iter()
                .map(|state: &NwAffineState| (state.row, state.col))
                .collect();
            let alignment = AlignmentTrace::from_cells(query, target, &cells, false);
            Ok((
                Objective {
                    kind: ObjectiveKind::Score,
                    value: i64::from(score),
                },
                Some(trace_record(alignment, request.options.include_operations)?),
                backend.to_string(),
                problem.score_path(&path) == Some(score),
            ))
        }
        (AlignmentMode::LocalLinear, ScoringScheme::Linear { substitution, gap }) => {
            let scoring = substitution.internal()?;
            let problem = SmithWatermanProblem::with_scoring(query, target, scoring, *gap);
            let (score, path, _) = run_hcp(problem.clone(), request.options.block_size);
            let cells: Vec<(usize, usize)> = path
                .iter()
                .map(|cell: &SwCell| (cell.row, cell.col))
                .collect();
            let alignment = AlignmentTrace::from_cells(query, target, &cells, false);
            Ok((
                Objective {
                    kind: ObjectiveKind::Score,
                    value: i64::from(score),
                },
                Some(trace_record(alignment, request.options.include_operations)?),
                "hcp".to_string(),
                problem.score_path(&path) == Some(score),
            ))
        }
        (AlignmentMode::SemiglobalLinear, ScoringScheme::Linear { substitution, gap }) => {
            let scoring = substitution.internal()?;
            let problem = SemiGlobalProblem::with_scoring(query, target, scoring, *gap);
            let (score, path, _) = run_hcp(problem.clone(), request.options.block_size);
            let cells: Vec<(usize, usize)> = path
                .iter()
                .map(|cell: &SemiGlobalCell| (cell.row, cell.col))
                .collect();
            let alignment = AlignmentTrace::from_cells(query, target, &cells, false);
            Ok((
                Objective {
                    kind: ObjectiveKind::Score,
                    value: i64::from(score),
                },
                Some(trace_record(alignment, request.options.include_operations)?),
                "hcp".to_string(),
                problem.score_path(&path) == Some(score),
            ))
        }
        _ => Err("problem mode and scoring model are incompatible".to_string()),
    }
}

fn run_hcp<P>(problem: P, block_size: Option<usize>) -> (P::Cost, Vec<P::State>, HcpRunStats)
where
    P: crate::traits::HcpProblem,
{
    match block_size {
        Some(0) => panic!("block size must be positive"),
        Some(size) => HcpEngine::with_block_size(problem, size).run_with_stats(),
        None => HcpEngine::new(problem).run_with_stats(),
    }
}

fn trace_record(trace: AlignmentTrace, include_operations: bool) -> Result<TraceRecord, String> {
    Ok(TraceRecord {
        query_span: AlignmentSpan {
            start: trace.query_start,
            end: trace.query_end,
        },
        target_span: AlignmentSpan {
            start: trace.target_start,
            end: trace.target_end,
        },
        cigar: Cigar::from_trace(&trace)?,
        operations: include_operations.then_some(trace.operations),
    })
}

/// Verify a submitted alignment at the requested assurance level.
pub fn verify(request: VerificationRequest) -> VerificationReport {
    let start = Instant::now();
    let work_cells = cell_count(request.query.len(), request.target.len());
    let mut checks = Vec::new();
    let record = &request.record;
    add_check(
        &mut checks,
        "schema_version",
        record.schema_version == SCHEMA_VERSION,
        format!("expected {SCHEMA_VERSION}, found {}", record.schema_version),
    );
    add_check(
        &mut checks,
        "record_type",
        record.record_type == "alignment",
        format!("expected alignment, found {}", record.record_type),
    );
    let problem_semantics = record
        .problem
        .scoring
        .validate_for(record.problem.mode)
        .and_then(|()| {
            validate_sequences(&record.problem.scoring, &request.query, &request.target)
        });
    add_check(
        &mut checks,
        "problem_semantics",
        problem_semantics.is_ok(),
        problem_semantics
            .err()
            .unwrap_or_else(|| "problem semantics are invalid".to_string()),
    );
    let expected_kind = if record.problem.mode == AlignmentMode::EditDistance {
        ObjectiveKind::Distance
    } else {
        ObjectiveKind::Score
    };
    add_check(
        &mut checks,
        "objective_kind",
        record.alignment.objective.kind == expected_kind,
        format!(
            "mode {} requires objective kind {:?}",
            record.problem.mode, expected_kind
        ),
    );
    add_check(
        &mut checks,
        "query_length",
        record.pair.query.length == request.query.len(),
        format!(
            "record has {}, source has {}",
            record.pair.query.length,
            request.query.len()
        ),
    );
    add_check(
        &mut checks,
        "target_length",
        record.pair.target.length == request.target.len(),
        format!(
            "record has {}, source has {}",
            record.pair.target.length,
            request.target.len()
        ),
    );
    add_check(
        &mut checks,
        "query_sha256",
        record.pair.query.sha256 == sha256_hex(&request.query),
        "query digest does not match source".to_string(),
    );
    add_check(
        &mut checks,
        "target_sha256",
        record.pair.target.sha256 == sha256_hex(&request.target),
        "target digest does not match source".to_string(),
    );
    if record.integrity.is_some() {
        match verify_integrity(record) {
            Ok(passed) => add_check(
                &mut checks,
                "integrity_fingerprint",
                passed,
                "deterministic record fields do not match the integrity fingerprint".to_string(),
            ),
            Err(error) => add_check(&mut checks, "integrity_fingerprint", false, error),
        }
    }
    let structural = match record.alignment.trace.as_ref() {
        None => {
            add_check(
                &mut checks,
                "trace_present",
                false,
                "verification requires a traceback".to_string(),
            );
            false
        }
        Some(trace) => validate_structure(
            record.problem.mode,
            trace,
            &request.query,
            &request.target,
            &mut checks,
        ),
    };

    let mut realized = None;
    if structural && request.level != VerificationLevel::Structure {
        if let Some(trace) = record.alignment.trace.as_ref() {
            if let Some(operations) = trace.operations.as_ref() {
                match expected_operations(trace, &request.query, &request.target) {
                    Ok(expected) => add_check(
                        &mut checks,
                        "operations_match_cigar",
                        *operations == expected,
                        "expanded operations disagree with CIGAR, coordinates, or sequence bases"
                            .to_string(),
                    ),
                    Err(error) => add_check(&mut checks, "operations_match_cigar", false, error),
                }
            }
        }
        match realized_objective(record, &request.query, &request.target) {
            Ok(value) => {
                let passed = value == record.alignment.objective;
                add_check(
                    &mut checks,
                    "realized_objective",
                    passed,
                    format!(
                        "trace realizes {}, record claims {}",
                        value.value, record.alignment.objective.value
                    ),
                );
                realized = Some(value);
            }
            Err(err) => add_check(&mut checks, "realized_objective", false, err),
        }
    }

    let failed = checks.iter().any(|check| !check.passed);
    let mut reason = None;
    let mut reference_backend = None;
    let status = if failed {
        VerificationStatus::Invalid
    } else if request.level == VerificationLevel::Optimal {
        if request.max_cells != 0 && work_cells > request.max_cells {
            reason = Some(format!(
                "exact optimality requires {work_cells} cells, exceeding --max-cells {}",
                request.max_cells
            ));
            VerificationStatus::Inconclusive
        } else {
            reference_backend = Some("baseline-rolling-v1".to_string());
            match reference_objective(&record.problem, &request.query, &request.target) {
                Ok(reference) => {
                    let passed = reference == record.alignment.objective;
                    add_check(
                        &mut checks,
                        "reference_optimum",
                        passed,
                        format!(
                            "record objective {}, exact reference {}",
                            record.alignment.objective.value, reference.value
                        ),
                    );
                    if passed {
                        VerificationStatus::Verified
                    } else {
                        VerificationStatus::Invalid
                    }
                }
                Err(err) => {
                    reason = Some(err);
                    VerificationStatus::Error
                }
            }
        }
    } else {
        let _ = realized;
        VerificationStatus::Verified
    };

    VerificationReport {
        schema_version: SCHEMA_VERSION.to_string(),
        record_type: "verification".to_string(),
        pair: record.pair.clone(),
        problem: record.problem.clone(),
        submitted: record.alignment.clone(),
        verification: VerificationSummary {
            level: request.level,
            status,
            checks,
            reference_backend,
            reason,
        },
        metrics: Metrics {
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            work_cells,
        },
    }
}

fn expected_operations(
    trace: &TraceRecord,
    query: &[u8],
    target: &[u8],
) -> Result<Vec<AlignmentStep>, String> {
    let mut query_pos = trace.query_span.start;
    let mut target_pos = trace.target_span.start;
    let mut operations = Vec::new();
    for run in trace.cigar.runs() {
        for _ in 0..run.len {
            let step =
                match run.op {
                    CigarOp::Match | CigarOp::Mismatch => {
                        let query_base = *query.get(query_pos).ok_or_else(|| {
                            "operation query coordinate is out of bounds".to_string()
                        })? as char;
                        let target_base = *target.get(target_pos).ok_or_else(|| {
                            "operation target coordinate is out of bounds".to_string()
                        })? as char;
                        let op = if run.op == CigarOp::Match {
                            AlignmentOpKind::Match
                        } else {
                            AlignmentOpKind::Mismatch
                        };
                        let step = AlignmentStep {
                            op,
                            query_pos: Some(query_pos),
                            target_pos: Some(target_pos),
                            query_base: Some(query_base),
                            target_base: Some(target_base),
                        };
                        query_pos += 1;
                        target_pos += 1;
                        step
                    }
                    CigarOp::Insertion => {
                        let query_base = *query.get(query_pos).ok_or_else(|| {
                            "operation query coordinate is out of bounds".to_string()
                        })? as char;
                        let step = AlignmentStep {
                            op: AlignmentOpKind::Insertion,
                            query_pos: Some(query_pos),
                            target_pos: None,
                            query_base: Some(query_base),
                            target_base: None,
                        };
                        query_pos += 1;
                        step
                    }
                    CigarOp::Deletion => {
                        let target_base = *target.get(target_pos).ok_or_else(|| {
                            "operation target coordinate is out of bounds".to_string()
                        })? as char;
                        let step = AlignmentStep {
                            op: AlignmentOpKind::Deletion,
                            query_pos: None,
                            target_pos: Some(target_pos),
                            query_base: None,
                            target_base: Some(target_base),
                        };
                        target_pos += 1;
                        step
                    }
                };
            operations.push(step);
        }
    }
    Ok(operations)
}

fn validate_structure(
    mode: AlignmentMode,
    trace: &TraceRecord,
    query: &[u8],
    target: &[u8],
    checks: &mut Vec<CheckResult>,
) -> bool {
    let first_check = checks.len();
    add_check(
        checks,
        "query_span_bounds",
        trace.query_span.start <= trace.query_span.end && trace.query_span.end <= query.len(),
        "query span is outside source sequence".to_string(),
    );
    add_check(
        checks,
        "target_span_bounds",
        trace.target_span.start <= trace.target_span.end && trace.target_span.end <= target.len(),
        "target span is outside source sequence".to_string(),
    );
    add_check(
        checks,
        "query_cigar_consumption",
        trace.cigar.query_consumed() == trace.query_span.len(),
        format!(
            "CIGAR consumes {}, span covers {}",
            trace.cigar.query_consumed(),
            trace.query_span.len()
        ),
    );
    add_check(
        checks,
        "target_cigar_consumption",
        trace.cigar.target_consumed() == trace.target_span.len(),
        format!(
            "CIGAR consumes {}, span covers {}",
            trace.cigar.target_consumed(),
            trace.target_span.len()
        ),
    );
    match mode {
        AlignmentMode::EditDistance | AlignmentMode::GlobalLinear | AlignmentMode::GlobalAffine => {
            add_check(
                checks,
                "global_query_span",
                trace.query_span
                    == (AlignmentSpan {
                        start: 0,
                        end: query.len(),
                    }),
                "global mode must consume the full query".to_string(),
            );
            add_check(
                checks,
                "global_target_span",
                trace.target_span
                    == (AlignmentSpan {
                        start: 0,
                        end: target.len(),
                    }),
                "global mode must consume the full target".to_string(),
            );
        }
        AlignmentMode::SemiglobalLinear => add_check(
            checks,
            "semiglobal_query_span",
            trace.query_span
                == (AlignmentSpan {
                    start: 0,
                    end: query.len(),
                }),
            "semi-global mode must consume the full query".to_string(),
        ),
        AlignmentMode::LocalLinear => {}
    }
    checks[first_check..].iter().all(|check| check.passed)
}

fn realized_objective(
    record: &AlignmentRecord,
    query: &[u8],
    target: &[u8],
) -> Result<Objective, String> {
    let trace = record
        .alignment
        .trace
        .as_ref()
        .ok_or_else(|| "traceback is missing".to_string())?;
    let mut query_pos = trace.query_span.start;
    let mut target_pos = trace.target_span.start;
    let scoring = &record.problem.scoring;
    let mut value = 0i64;
    let substitution = match scoring {
        ScoringScheme::Linear { substitution, .. } | ScoringScheme::Affine { substitution, .. } => {
            Some(substitution.internal()?)
        }
        ScoringScheme::EditDistance => None,
    };
    for run in trace.cigar.runs() {
        match run.op {
            CigarOp::Match | CigarOp::Mismatch => {
                for _ in 0..run.len {
                    let q = *query
                        .get(query_pos)
                        .ok_or_else(|| "CIGAR consumes query out of bounds".to_string())?;
                    let t = *target
                        .get(target_pos)
                        .ok_or_else(|| "CIGAR consumes target out of bounds".to_string())?;
                    let equal = q == t;
                    if (run.op == CigarOp::Match) != equal {
                        return Err(format!(
                            "CIGAR {} disagrees with bases at query {query_pos}, target {target_pos}",
                            run.op.symbol()
                        ));
                    }
                    value += match scoring {
                        ScoringScheme::EditDistance => i64::from(!equal),
                        _ => i64::from(
                            substitution
                                .as_ref()
                                .expect("score modes have substitution scoring")
                                .score(q, t),
                        ),
                    };
                    query_pos += 1;
                    target_pos += 1;
                }
            }
            CigarOp::Insertion => {
                query_pos = query_pos
                    .checked_add(run.len)
                    .ok_or_else(|| "query coordinate overflow".to_string())?;
                value += gap_value(scoring, run.len)?;
            }
            CigarOp::Deletion => {
                target_pos = target_pos
                    .checked_add(run.len)
                    .ok_or_else(|| "target coordinate overflow".to_string())?;
                value += gap_value(scoring, run.len)?;
            }
        }
    }
    if query_pos != trace.query_span.end || target_pos != trace.target_span.end {
        return Err("CIGAR endpoints do not match declared spans".to_string());
    }
    Ok(Objective {
        kind: if matches!(scoring, ScoringScheme::EditDistance) {
            ObjectiveKind::Distance
        } else {
            ObjectiveKind::Score
        },
        value,
    })
}

fn gap_value(scoring: &ScoringScheme, len: usize) -> Result<i64, String> {
    match scoring {
        ScoringScheme::EditDistance => Ok(len as i64),
        ScoringScheme::Linear { gap, .. } => Ok(i64::from(*gap) * len as i64),
        ScoringScheme::Affine {
            gap_open,
            gap_extend,
            ..
        } => Ok(i64::from(*gap_open) + i64::from(*gap_extend) * len as i64),
    }
}

fn reference_objective(
    problem: &ProblemSpec,
    query: &[u8],
    target: &[u8],
) -> Result<Objective, String> {
    let objective = match (&problem.mode, &problem.scoring) {
        (AlignmentMode::EditDistance, ScoringScheme::EditDistance) => Objective {
            kind: ObjectiveKind::Distance,
            value: i64::from(EditDistanceProblem::new(query, target).full_table_distance()),
        },
        (AlignmentMode::GlobalLinear, ScoringScheme::Linear { substitution, gap }) => Objective {
            kind: ObjectiveKind::Score,
            value: i64::from(
                NwProblem::with_scoring(query, target, substitution.internal()?, *gap)
                    .full_table_score(),
            ),
        },
        (
            AlignmentMode::GlobalAffine,
            ScoringScheme::Affine {
                substitution,
                gap_open,
                gap_extend,
            },
        ) => Objective {
            kind: ObjectiveKind::Score,
            value: i64::from(
                NwAffineProblem::with_scoring(
                    query,
                    target,
                    substitution.internal()?,
                    *gap_open,
                    *gap_extend,
                )
                .full_table_score(),
            ),
        },
        (AlignmentMode::LocalLinear, ScoringScheme::Linear { substitution, gap }) => Objective {
            kind: ObjectiveKind::Score,
            value: i64::from(
                SmithWatermanProblem::with_scoring(query, target, substitution.internal()?, *gap)
                    .full_table_score(),
            ),
        },
        (AlignmentMode::SemiglobalLinear, ScoringScheme::Linear { substitution, gap }) => {
            Objective {
                kind: ObjectiveKind::Score,
                value: i64::from(
                    SemiGlobalProblem::with_scoring(query, target, substitution.internal()?, *gap)
                        .full_table_score(),
                ),
            }
        }
        _ => return Err("problem mode and scoring model are incompatible".to_string()),
    };
    Ok(objective)
}

/// Compare two records by exact objective equivalence and validated traces.
pub fn compare(
    left: AlignmentRecord,
    right: AlignmentRecord,
    query: &[u8],
    target: &[u8],
) -> Result<ComparisonReport, String> {
    if left.problem != right.problem {
        return Err("records use different problem definitions".to_string());
    }
    if left.pair.query.sha256 != right.pair.query.sha256
        || left.pair.target.sha256 != right.pair.target.sha256
    {
        return Err("records refer to different sequence pairs".to_string());
    }
    let left_report = verify(VerificationRequest {
        record: left.clone(),
        query: query.to_vec(),
        target: target.to_vec(),
        level: VerificationLevel::Optimal,
        max_cells: 0,
    });
    let right_report = verify(VerificationRequest {
        record: right.clone(),
        query: query.to_vec(),
        target: target.to_vec(),
        level: VerificationLevel::Optimal,
        max_cells: 0,
    });
    let objectives_equal = left.alignment.objective == right.alignment.objective;
    let traces_equal = left.alignment.trace == right.alignment.trace;
    let equivalent = objectives_equal
        && left_report.verification.status == VerificationStatus::Verified
        && right_report.verification.status == VerificationStatus::Verified;
    let mut differences = Vec::new();
    if !objectives_equal {
        differences.push(format!(
            "objective: left={}, right={}",
            left.alignment.objective.value, right.alignment.objective.value
        ));
    }
    if objectives_equal && !traces_equal {
        if let (Some(left_trace), Some(right_trace)) =
            (&left.alignment.trace, &right.alignment.trace)
        {
            if left_trace.query_span != right_trace.query_span
                || left_trace.target_span != right_trace.target_span
            {
                differences.push(format!(
                    "coordinates: left=query {:?},target {:?}; right=query {:?},target {:?}",
                    left_trace.query_span,
                    left_trace.target_span,
                    right_trace.query_span,
                    right_trace.target_span
                ));
            }
            let left_count: usize = left_trace.cigar.runs().iter().map(|run| run.len).sum();
            let right_count: usize = right_trace.cigar.runs().iter().map(|run| run.len).sum();
            if left_count != right_count {
                differences.push(format!(
                    "operation_count: left={left_count}, right={right_count}"
                ));
            }
        }
        differences.push("tie_breaking: different traces realize the same optimum".to_string());
    }
    if left_report.verification.status != VerificationStatus::Verified {
        let failed = left_report
            .verification
            .checks
            .iter()
            .filter(|check| !check.passed)
            .map(|check| check.name.as_str())
            .collect::<Vec<_>>()
            .join(",");
        differences.push(format!("left_invalid: {failed}"));
    }
    if right_report.verification.status != VerificationStatus::Verified {
        let failed = right_report
            .verification
            .checks
            .iter()
            .filter(|check| !check.passed)
            .map(|check| check.name.as_str())
            .collect::<Vec<_>>()
            .join(",");
        differences.push(format!("right_invalid: {failed}"));
    }
    let mut left = left;
    let mut right = right;
    left.metrics = None;
    right.metrics = None;
    left.integrity = None;
    right.integrity = None;
    if let Some(trace) = left.alignment.trace.as_mut() {
        trace.operations = None;
    }
    if let Some(trace) = right.alignment.trace.as_mut() {
        trace.operations = None;
    }
    Ok(ComparisonReport {
        schema_version: SCHEMA_VERSION.to_string(),
        record_type: "comparison".to_string(),
        inputs: ComparisonInputs {
            query: std::str::from_utf8(query)
                .map_err(|_| "comparison artifacts require ASCII/UTF-8 query sequences")?
                .to_string(),
            target: std::str::from_utf8(target)
                .map_err(|_| "comparison artifacts require ASCII/UTF-8 target sequences")?
                .to_string(),
        },
        pair: left.pair.clone(),
        problem: left.problem.clone(),
        left,
        right,
        objectives_equal,
        traces_equal,
        equivalent,
        differences,
    })
}

pub fn records_from_json(text: &str) -> Result<Vec<AlignmentRecord>, String> {
    let value: serde_json::Value = serde_json::from_str(text).map_err(|err| err.to_string())?;
    if value.is_array() {
        serde_json::from_value(value).map_err(|err| err.to_string())
    } else {
        serde_json::from_value(value)
            .map(|record| vec![record])
            .map_err(|err| err.to_string())
    }
}

pub fn records_from_jsonl(text: &str) -> Result<Vec<AlignmentRecord>, String> {
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).map_err(|err| err.to_string()))
        .collect()
}

pub fn records_to_json(records: &[AlignmentRecord]) -> Result<String, String> {
    let mut output = if records.len() == 1 {
        serde_json::to_string_pretty(&records[0]).map_err(|err| err.to_string())?
    } else {
        serde_json::to_string_pretty(records).map_err(|err| err.to_string())?
    };
    output.push('\n');
    Ok(output)
}

pub fn records_to_jsonl(records: &[AlignmentRecord]) -> Result<String, String> {
    let mut output = String::new();
    for record in records {
        output.push_str(&serde_json::to_string(record).map_err(|err| err.to_string())?);
        output.push('\n');
    }
    Ok(output)
}

pub fn record_to_paf(record: &AlignmentRecord) -> Result<String, String> {
    let trace = record
        .alignment
        .trace
        .as_ref()
        .ok_or_else(|| "PAF requires a traceback".to_string())?;
    let matches: usize = trace
        .cigar
        .runs()
        .iter()
        .filter(|run| run.op == CigarOp::Match)
        .map(|run| run.len)
        .sum();
    let block_len: usize = trace.cigar.runs().iter().map(|run| run.len).sum();
    let nm: usize = trace
        .cigar
        .runs()
        .iter()
        .filter(|run| run.op != CigarOp::Match)
        .map(|run| run.len)
        .sum();
    let mut output = format!(
        "{}\t{}\t{}\t{}\t+\t{}\t{}\t{}\t{}\t{}\t{}\t255\tcg:Z:{}\tNM:i:{}",
        record.pair.query.id,
        record.pair.query.length,
        trace.query_span.start,
        trace.query_span.end,
        record.pair.target.id,
        record.pair.target.length,
        trace.target_span.start,
        trace.target_span.end,
        matches,
        block_len,
        trace.cigar,
        nm,
    );
    if record.alignment.objective.kind == ObjectiveKind::Score {
        output.push_str(&format!("\tAS:i:{}", record.alignment.objective.value));
    }
    output.push('\n');
    Ok(output)
}

pub fn records_to_sam(
    records: &[AlignmentRecord],
    queries: &HashMap<String, Vec<u8>>,
) -> Result<String, String> {
    let mut output = String::from("@HD\tVN:1.6\tSO:unknown\n");
    let mut references: HashMap<&str, usize> = HashMap::new();
    for record in records {
        references.insert(&record.pair.target.id, record.pair.target.length);
    }
    let mut references: Vec<_> = references.into_iter().collect();
    references.sort_by_key(|(id, _)| *id);
    for (id, len) in references {
        output.push_str(&format!("@SQ\tSN:{id}\tLN:{len}\n"));
    }
    for record in records {
        let query = queries
            .get(&record.pair.query.id)
            .ok_or_else(|| format!("missing query sequence '{}'", record.pair.query.id))?;
        output.push_str(&record_to_sam(record, query)?);
    }
    Ok(output)
}

/// Serialize one SAM body record without headers.
pub fn record_to_sam(record: &AlignmentRecord, query: &[u8]) -> Result<String, String> {
    let trace = record
        .alignment
        .trace
        .as_ref()
        .ok_or_else(|| "SAM requires a traceback".to_string())?;
    if query.len() != record.pair.query.length {
        return Err(format!(
            "query sequence '{}' has length {} but record declares {}",
            record.pair.query.id,
            query.len(),
            record.pair.query.length
        ));
    }
    let mut cigar = String::new();
    if trace.query_span.start > 0 {
        cigar.push_str(&format!("{}S", trace.query_span.start));
    }
    cigar.push_str(&trace.cigar.to_string());
    let suffix = record
        .pair
        .query
        .length
        .saturating_sub(trace.query_span.end);
    if suffix > 0 {
        cigar.push_str(&format!("{suffix}S"));
    }
    if cigar.is_empty() {
        cigar.push('*');
    }
    let sequence = String::from_utf8_lossy(query);
    let mut output = format!(
        "{}\t0\t{}\t{}\t255\t{}\t*\t0\t0\t{}\t*",
        record.pair.query.id,
        record.pair.target.id,
        trace.target_span.start + 1,
        cigar,
        sequence,
    );
    if record.alignment.objective.kind == ObjectiveKind::Score {
        output.push_str(&format!("\tAS:i:{}", record.alignment.objective.value));
    }
    let nm: usize = trace
        .cigar
        .runs()
        .iter()
        .filter(|run| run.op != CigarOp::Match)
        .map(|run| run.len)
        .sum();
    output.push_str(&format!("\tNM:i:{nm}\n"));
    Ok(output)
}

/// Parse forward-strand PAF records using explicit query/target sequence maps.
pub fn records_from_paf(
    text: &str,
    problem: &ProblemSpec,
    queries: &HashMap<String, Vec<u8>>,
    targets: &HashMap<String, Vec<u8>>,
) -> Result<Vec<AlignmentRecord>, String> {
    let mut records = Vec::new();
    for (index, line) in text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
    {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 12 {
            return Err(format!("PAF line {} has fewer than 12 fields", index + 1));
        }
        if fields[4] != "+" {
            return Err(format!(
                "PAF line {} uses unsupported reverse strand",
                index + 1
            ));
        }
        let query_id = fields[0];
        let target_id = fields[5];
        let query = queries
            .get(query_id)
            .ok_or_else(|| format!("PAF query id '{query_id}' is missing from query sources"))?;
        let target = targets
            .get(target_id)
            .ok_or_else(|| format!("PAF target id '{target_id}' is missing from target sources"))?;
        let query_len = parse_usize(fields[1], "PAF query length")?;
        let query_start = parse_usize(fields[2], "PAF query start")?;
        let query_end = parse_usize(fields[3], "PAF query end")?;
        let target_len = parse_usize(fields[6], "PAF target length")?;
        let target_start = parse_usize(fields[7], "PAF target start")?;
        let target_end = parse_usize(fields[8], "PAF target end")?;
        let tags = parse_tags(&fields[12..])?;
        let (cigar_type, raw_cigar) = tags
            .get("cg")
            .ok_or_else(|| format!("PAF line {} is missing cg:Z", index + 1))?;
        if *cigar_type != 'Z' {
            return Err(format!("PAF line {} cg tag must use type Z", index + 1));
        }
        let cigar =
            canonicalize_external_cigar(raw_cigar, query_start, target_start, query, target)?;
        let declared_matches = parse_usize(fields[9], "PAF matching-base count")?;
        let declared_block = parse_usize(fields[10], "PAF alignment block length")?;
        let actual_matches: usize = cigar
            .runs()
            .iter()
            .filter(|run| run.op == CigarOp::Match)
            .map(|run| run.len)
            .sum();
        let actual_block: usize = cigar.runs().iter().map(|run| run.len).sum();
        if declared_matches != actual_matches {
            return Err(format!(
                "PAF line {} reports {declared_matches} matching bases but cg:Z implies {actual_matches}",
                index + 1
            ));
        }
        if declared_block != actual_block {
            return Err(format!(
                "PAF line {} reports block length {declared_block} but cg:Z implies {actual_block}",
                index + 1
            ));
        }
        if query_end.saturating_sub(query_start) != cigar.query_consumed()
            || target_end.saturating_sub(target_start) != cigar.target_consumed()
        {
            return Err(format!(
                "PAF line {} coordinates disagree with cg:Z consumption",
                index + 1
            ));
        }
        validate_nm_tag(&tags, &cigar, index + 1, "PAF")?;
        let objective = external_objective(problem, &tags, index + 1, "PAF")?;
        records.push(external_record(ExternalRecordInput {
            index,
            query_id,
            target_id,
            query,
            target,
            declared_query_len: query_len,
            declared_target_len: target_len,
            problem: problem.clone(),
            objective,
            trace: TraceRecord {
                query_span: AlignmentSpan {
                    start: query_start,
                    end: query_end,
                },
                target_span: AlignmentSpan {
                    start: target_start,
                    end: target_end,
                },
                cigar,
                operations: None,
            },
            format: "paf",
        }));
    }
    Ok(records)
}

/// Parse primary forward-strand SAM records using explicit query/target maps.
pub fn records_from_sam(
    text: &str,
    problem: &ProblemSpec,
    queries: &HashMap<String, Vec<u8>>,
    targets: &HashMap<String, Vec<u8>>,
) -> Result<Vec<AlignmentRecord>, String> {
    let mut records = Vec::new();
    for (line_number, line) in text.lines().enumerate() {
        if line.is_empty() || line.starts_with('@') {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 11 {
            return Err(format!(
                "SAM line {} has fewer than 11 fields",
                line_number + 1
            ));
        }
        let flag = fields[1]
            .parse::<u16>()
            .map_err(|_| format!("SAM line {} has invalid FLAG", line_number + 1))?;
        if flag & 0x4 != 0 {
            return Err(format!("SAM line {} is unmapped", line_number + 1));
        }
        if flag & 0x10 != 0 {
            return Err(format!(
                "SAM line {} uses unsupported reverse strand",
                line_number + 1
            ));
        }
        let query_id = fields[0];
        let target_id = fields[2];
        let query = queries
            .get(query_id)
            .ok_or_else(|| format!("SAM query id '{query_id}' is missing from query sources"))?;
        let target = targets
            .get(target_id)
            .ok_or_else(|| format!("SAM target id '{target_id}' is missing from target sources"))?;
        let target_start = fields[3]
            .parse::<usize>()
            .map_err(|_| format!("SAM line {} has invalid POS", line_number + 1))?
            .checked_sub(1)
            .ok_or_else(|| format!("SAM line {} POS must be positive", line_number + 1))?;
        let raw_runs = raw_cigar_runs(fields[5])?;
        let leading_soft = raw_runs
            .first()
            .filter(|run| run.1 == 'S')
            .map_or(0, |run| run.0);
        let trailing_soft = raw_runs
            .last()
            .filter(|run| run.1 == 'S')
            .map_or(0, |run| run.0);
        if raw_runs
            .iter()
            .enumerate()
            .any(|(idx, run)| run.1 == 'S' && idx != 0 && idx + 1 != raw_runs.len())
        {
            return Err(format!(
                "SAM line {} has internal soft clipping",
                line_number + 1
            ));
        }
        let core: String = raw_runs
            .iter()
            .filter(|run| run.1 != 'S')
            .map(|(len, op)| format!("{len}{op}"))
            .collect();
        let query_start = leading_soft;
        let cigar = canonicalize_external_cigar(&core, query_start, target_start, query, target)?;
        let query_end = query
            .len()
            .checked_sub(trailing_soft)
            .ok_or_else(|| format!("SAM line {} clipping exceeds query length", line_number + 1))?;
        let target_end = target_start
            .checked_add(cigar.target_consumed())
            .ok_or_else(|| "SAM target coordinate overflow".to_string())?;
        if fields[9] != "*" && fields[9].as_bytes() != query {
            return Err(format!(
                "SAM line {} SEQ does not match the resolved query source",
                line_number + 1
            ));
        }
        let tags = parse_tags(&fields[11..])?;
        validate_nm_tag(&tags, &cigar, line_number + 1, "SAM")?;
        let objective = external_objective(problem, &tags, line_number + 1, "SAM")?;
        records.push(external_record(ExternalRecordInput {
            index: records.len(),
            query_id,
            target_id,
            query,
            target,
            declared_query_len: query.len(),
            declared_target_len: target.len(),
            problem: problem.clone(),
            objective,
            trace: TraceRecord {
                query_span: AlignmentSpan {
                    start: query_start,
                    end: query_end,
                },
                target_span: AlignmentSpan {
                    start: target_start,
                    end: target_end,
                },
                cigar,
                operations: None,
            },
            format: "sam",
        }));
    }
    Ok(records)
}

struct ExternalRecordInput<'a> {
    index: usize,
    query_id: &'a str,
    target_id: &'a str,
    query: &'a [u8],
    target: &'a [u8],
    declared_query_len: usize,
    declared_target_len: usize,
    problem: ProblemSpec,
    objective: Objective,
    trace: TraceRecord,
    format: &'a str,
}

fn external_record(input: ExternalRecordInput<'_>) -> AlignmentRecord {
    let ExternalRecordInput {
        index,
        query_id,
        target_id,
        query,
        target,
        declared_query_len,
        declared_target_len,
        problem,
        objective,
        trace,
        format,
    } = input;
    let mut pair = pair_identity(index, query_id, target_id, query, target);
    pair.query.length = declared_query_len;
    pair.target.length = declared_target_len;
    AlignmentRecord {
        schema_version: SCHEMA_VERSION.to_string(),
        record_type: "alignment".to_string(),
        producer: Producer {
            name: format!("external-{format}"),
            version: "unknown".to_string(),
            commit: None,
            backend: "external".to_string(),
        },
        pair,
        problem,
        alignment: AlignmentResult {
            objective,
            trace: Some(trace),
        },
        verification: VerificationSummary {
            level: VerificationLevel::Structure,
            status: VerificationStatus::Inconclusive,
            checks: Vec::new(),
            reference_backend: None,
            reason: Some("record has not yet been verified".to_string()),
        },
        metrics: None,
        integrity: None,
    }
}

fn external_objective(
    problem: &ProblemSpec,
    tags: &HashMap<String, (char, String)>,
    line_number: usize,
    format: &str,
) -> Result<Objective, String> {
    match problem.mode {
        AlignmentMode::EditDistance => integer_tag(tags, "NM", line_number, format)?
            .parse::<i64>()
            .map(|value| Objective {
                kind: ObjectiveKind::Distance,
                value,
            })
            .map_err(|_| format!("{format} line {line_number} has invalid NM:i")),
        _ => integer_tag(tags, "AS", line_number, format)?
            .parse::<i64>()
            .map(|value| Objective {
                kind: ObjectiveKind::Score,
                value,
            })
            .map_err(|_| format!("{format} line {line_number} has invalid AS:i")),
    }
}

fn parse_tags(fields: &[&str]) -> Result<HashMap<String, (char, String)>, String> {
    let mut tags = HashMap::new();
    for field in fields {
        let mut parts = field.splitn(3, ':');
        let name = parts
            .next()
            .filter(|name| name.len() == 2)
            .ok_or_else(|| format!("malformed optional tag '{field}'"))?;
        let kind = parts
            .next()
            .and_then(|kind| (kind.len() == 1).then(|| kind.chars().next()).flatten())
            .ok_or_else(|| format!("malformed optional tag '{field}'"))?;
        let value = parts
            .next()
            .ok_or_else(|| format!("malformed optional tag '{field}'"))?;
        if tags
            .insert(name.to_string(), (kind, value.to_string()))
            .is_some()
        {
            return Err(format!("duplicate optional tag '{name}'"));
        }
    }
    Ok(tags)
}

fn integer_tag<'a>(
    tags: &'a HashMap<String, (char, String)>,
    name: &str,
    line_number: usize,
    format: &str,
) -> Result<&'a str, String> {
    let (kind, value) = tags
        .get(name)
        .ok_or_else(|| format!("{format} line {line_number} is missing {name}:i"))?;
    if *kind != 'i' {
        return Err(format!(
            "{format} line {line_number} tag {name} must use integer type i"
        ));
    }
    Ok(value)
}

fn validate_nm_tag(
    tags: &HashMap<String, (char, String)>,
    cigar: &Cigar,
    line_number: usize,
    format: &str,
) -> Result<(), String> {
    let Some(_) = tags.get("NM") else {
        return Ok(());
    };
    let declared = integer_tag(tags, "NM", line_number, format)?
        .parse::<usize>()
        .map_err(|_| format!("{format} line {line_number} has invalid NM:i"))?;
    let actual: usize = cigar
        .runs()
        .iter()
        .filter(|run| run.op != CigarOp::Match)
        .map(|run| run.len)
        .sum();
    if declared != actual {
        return Err(format!(
            "{format} line {line_number} reports NM:i:{declared} but CIGAR and sources imply {actual}"
        ));
    }
    Ok(())
}

fn canonicalize_external_cigar(
    raw: &str,
    mut query_pos: usize,
    mut target_pos: usize,
    query: &[u8],
    target: &[u8],
) -> Result<Cigar, String> {
    let mut canonical = Vec::new();
    for (len, op) in raw_cigar_runs(raw)? {
        match op {
            '=' | 'X' | 'M' => {
                for _ in 0..len {
                    let q = *query
                        .get(query_pos)
                        .ok_or_else(|| "external CIGAR consumes query out of bounds".to_string())?;
                    let t = *target.get(target_pos).ok_or_else(|| {
                        "external CIGAR consumes target out of bounds".to_string()
                    })?;
                    let actual = if q == t {
                        CigarOp::Match
                    } else {
                        CigarOp::Mismatch
                    };
                    if op == '=' && actual != CigarOp::Match {
                        return Err("external CIGAR '=' disagrees with source bases".to_string());
                    }
                    if op == 'X' && actual != CigarOp::Mismatch {
                        return Err("external CIGAR 'X' disagrees with source bases".to_string());
                    }
                    canonical.push(CigarRun { len: 1, op: actual });
                    query_pos += 1;
                    target_pos += 1;
                }
            }
            'I' => {
                canonical.push(CigarRun {
                    len,
                    op: CigarOp::Insertion,
                });
                query_pos = query_pos
                    .checked_add(len)
                    .ok_or_else(|| "query coordinate overflow".to_string())?;
            }
            'D' => {
                canonical.push(CigarRun {
                    len,
                    op: CigarOp::Deletion,
                });
                target_pos = target_pos
                    .checked_add(len)
                    .ok_or_else(|| "target coordinate overflow".to_string())?;
            }
            other => return Err(format!("unsupported external CIGAR operation '{other}'")),
        }
    }
    Cigar::new(canonical)
}

fn raw_cigar_runs(raw: &str) -> Result<Vec<(usize, char)>, String> {
    if raw.is_empty() || raw == "*" {
        return Ok(Vec::new());
    }
    let mut count = 0usize;
    let mut has_count = false;
    let mut runs = Vec::new();
    for ch in raw.chars() {
        if let Some(digit) = ch.to_digit(10) {
            has_count = true;
            count = count
                .checked_mul(10)
                .and_then(|value| value.checked_add(digit as usize))
                .ok_or_else(|| "CIGAR run length overflowed usize".to_string())?;
        } else {
            if !has_count || count == 0 {
                return Err(format!("invalid CIGAR run before '{ch}'"));
            }
            runs.push((count, ch));
            count = 0;
            has_count = false;
        }
    }
    if has_count {
        return Err("CIGAR ends with a length and no operation".to_string());
    }
    Ok(runs)
}

fn parse_usize(value: &str, label: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|_| format!("{label} is not a non-negative integer"))
}

fn integrity_for(record: &AlignmentRecord) -> Result<IntegrityFingerprint, String> {
    #[derive(Serialize)]
    struct Deterministic<'a> {
        schema_version: &'a str,
        producer: &'a Producer,
        pair: &'a PairIdentity,
        problem: &'a ProblemSpec,
        alignment: &'a AlignmentResult,
        verification: &'a VerificationSummary,
    }
    let bytes = serde_json::to_vec(&Deterministic {
        schema_version: &record.schema_version,
        producer: &record.producer,
        pair: &record.pair,
        problem: &record.problem,
        alignment: &record.alignment,
        verification: &record.verification,
    })
    .map_err(|err| err.to_string())?;
    Ok(IntegrityFingerprint {
        version: "hcp-align.integrity.v1".to_string(),
        hash_algorithm: "sha256".to_string(),
        deterministic_fields_sha256: sha256_hex(&bytes),
    })
}

pub fn verify_integrity(record: &AlignmentRecord) -> Result<bool, String> {
    let Some(expected) = &record.integrity else {
        return Err("record has no integrity fingerprint".to_string());
    };
    if expected.version != "hcp-align.integrity.v1" {
        return Err(format!(
            "unsupported integrity fingerprint version '{}'",
            expected.version
        ));
    }
    if expected.hash_algorithm != "sha256" {
        return Err(format!(
            "unsupported integrity hash algorithm '{}'",
            expected.hash_algorithm
        ));
    }
    Ok(integrity_for(record)?.deterministic_fields_sha256 == expected.deterministic_fields_sha256)
}

fn validate_sequences(scoring: &ScoringScheme, query: &[u8], target: &[u8]) -> Result<(), String> {
    let substitution = match scoring {
        ScoringScheme::EditDistance => return Ok(()),
        ScoringScheme::Linear { substitution, .. } | ScoringScheme::Affine { substitution, .. } => {
            substitution.internal()?
        }
    };
    substitution
        .validate_sequence(query)
        .map_err(|err| format!("query: {err}"))?;
    substitution
        .validate_sequence(target)
        .map_err(|err| format!("target: {err}"))
}

fn pair_identity(
    index: usize,
    query_id: &str,
    target_id: &str,
    query: &[u8],
    target: &[u8],
) -> PairIdentity {
    PairIdentity {
        index,
        query: SequenceIdentity {
            id: query_id.to_string(),
            length: query.len(),
            sha256: sha256_hex(query),
        },
        target: SequenceIdentity {
            id: target_id.to_string(),
            length: target.len(),
            sha256: sha256_hex(target),
        },
    }
}

fn add_check(checks: &mut Vec<CheckResult>, name: &str, passed: bool, failed_detail: String) {
    checks.push(CheckResult {
        name: name.to_string(),
        passed,
        detail: (!passed).then_some(failed_detail),
    });
}

fn cell_count(query_len: usize, target_len: usize) -> u64 {
    (query_len as u64)
        .saturating_add(1)
        .saturating_mul((target_len as u64).saturating_add(1))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(mode: AlignmentMode, query: &[u8], target: &[u8]) -> SolveRequest {
        SolveRequest {
            pair_index: 0,
            query_id: "q".to_string(),
            target_id: "t".to_string(),
            query: query.to_vec(),
            target: target.to_vec(),
            problem: ProblemSpec {
                mode,
                scoring: ScoringScheme::default_for(mode),
            },
            options: SolveOptions {
                cross_check: true,
                ..SolveOptions::default()
            },
        }
    }

    #[test]
    fn canonical_cigar_uses_sam_orientation() {
        let cigar: Cigar = "1=2I3D1X".parse().unwrap();
        assert_eq!(cigar.query_consumed(), 4);
        assert_eq!(cigar.target_consumed(), 5);
        assert_eq!(cigar.to_string(), "1=2I3D1X");
    }

    #[test]
    fn solve_and_verify_edit_distance() {
        let record = solve(request(AlignmentMode::EditDistance, b"kitten", b"sitting")).unwrap();
        assert_eq!(record.alignment.objective.value, 3);
        let report = verify(VerificationRequest {
            record,
            query: b"kitten".to_vec(),
            target: b"sitting".to_vec(),
            level: VerificationLevel::Optimal,
            max_cells: DEFAULT_MAX_CELLS,
        });
        assert_eq!(report.verification.status, VerificationStatus::Verified);
    }

    #[test]
    fn optimality_limit_is_explicitly_inconclusive() {
        let record = solve(request(AlignmentMode::EditDistance, b"kitten", b"sitting")).unwrap();
        let report = verify(VerificationRequest {
            record,
            query: b"kitten".to_vec(),
            target: b"sitting".to_vec(),
            level: VerificationLevel::Optimal,
            max_cells: 1,
        });
        assert_eq!(report.verification.status, VerificationStatus::Inconclusive);
    }

    #[test]
    fn comparison_accepts_distinct_optimal_records() {
        let left = solve(request(AlignmentMode::EditDistance, b"AA", b"A")).unwrap();
        let mut right = left.clone();
        right.alignment.trace.as_mut().unwrap().cigar = "1=1I".parse().unwrap();
        let report = compare(left, right, b"AA", b"A").unwrap();
        assert!(report.equivalent);
        assert!(!report.traces_equal);
        assert!(report
            .differences
            .iter()
            .any(|difference| difference.contains("tie_breaking")));
    }

    #[test]
    fn mutation_failures_name_the_broken_contract() {
        let mut solve_request = request(AlignmentMode::EditDistance, b"ACG", b"ACG");
        solve_request.options.include_operations = true;
        solve_request.options.integrity = true;
        let original = solve(solve_request).unwrap();

        let failed_checks = |record: AlignmentRecord, query: &[u8], target: &[u8]| {
            verify(VerificationRequest {
                record,
                query: query.to_vec(),
                target: target.to_vec(),
                level: VerificationLevel::Optimal,
                max_cells: DEFAULT_MAX_CELLS,
            })
            .verification
            .checks
            .into_iter()
            .filter(|check| !check.passed)
            .map(|check| check.name)
            .collect::<Vec<_>>()
        };

        let mut coordinate = original.clone();
        coordinate.alignment.trace.as_mut().unwrap().query_span.end = 2;
        assert!(failed_checks(coordinate, b"ACG", b"ACG")
            .contains(&"query_cigar_consumption".to_string()));

        let mut cigar = original.clone();
        cigar.alignment.trace.as_mut().unwrap().cigar = "2=1I".parse().unwrap();
        assert!(
            failed_checks(cigar, b"ACG", b"ACG").contains(&"target_cigar_consumption".to_string())
        );

        let mut operation = original.clone();
        operation
            .alignment
            .trace
            .as_mut()
            .unwrap()
            .operations
            .as_mut()
            .unwrap()[0]
            .query_base = Some('T');
        assert!(failed_checks(operation, b"ACG", b"ACG")
            .contains(&"operations_match_cigar".to_string()));

        let mut objective = original.clone();
        objective.alignment.objective.value = 1;
        assert!(
            failed_checks(objective, b"ACG", b"ACG").contains(&"realized_objective".to_string())
        );

        let mut digest = original.clone();
        digest.pair.query.sha256 = "0".repeat(64);
        assert!(failed_checks(digest, b"ACG", b"ACG").contains(&"query_sha256".to_string()));

        let mut problem = original.clone();
        problem.problem.mode = AlignmentMode::GlobalLinear;
        assert!(failed_checks(problem, b"ACG", b"ACG").contains(&"problem_semantics".to_string()));

        let mut producer = original.clone();
        producer.producer.backend = "mutated".to_string();
        assert!(
            failed_checks(producer, b"ACG", b"ACG").contains(&"integrity_fingerprint".to_string())
        );

        assert!(failed_checks(original, b"ATG", b"ACG").contains(&"query_sha256".to_string()));
    }

    #[test]
    fn all_modes_cross_check_exhaustive_small_binary_pairs() {
        fn strings(max_len: usize) -> Vec<Vec<u8>> {
            let mut values = Vec::new();
            for len in 0..=max_len {
                for bits in 0..(1usize << len) {
                    values.push(
                        (0..len)
                            .map(|index| if bits & (1 << index) == 0 { b'A' } else { b'B' })
                            .collect(),
                    );
                }
            }
            values
        }
        let values = strings(3);
        for mode in [
            AlignmentMode::EditDistance,
            AlignmentMode::GlobalLinear,
            AlignmentMode::GlobalAffine,
            AlignmentMode::LocalLinear,
            AlignmentMode::SemiglobalLinear,
        ] {
            for query in &values {
                for target in &values {
                    let record = solve(request(mode, query, target)).unwrap_or_else(|error| {
                        panic!("{mode} failed for {query:?}/{target:?}: {error}")
                    });
                    assert_eq!(
                        record.verification.status,
                        VerificationStatus::Verified,
                        "{mode} cross-check failed for {query:?}/{target:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn cigar_parser_rejects_malformed_runs() {
        for malformed in ["0=", "1M", "=", "1", "1=0X"] {
            assert!(malformed.parse::<Cigar>().is_err(), "accepted {malformed}");
        }
    }

    #[test]
    fn every_mode_round_trips_json_paf_and_sam() {
        let cases: &[(AlignmentMode, &[u8], &[u8])] = &[
            (AlignmentMode::EditDistance, b"GATTACA", b"GCATGCU"),
            (AlignmentMode::GlobalLinear, b"GATTACA", b"GCATGCU"),
            (AlignmentMode::GlobalAffine, b"ACB", b"A"),
            (AlignmentMode::LocalLinear, b"TTTACGTAAA", b"GGACGTCC"),
            (AlignmentMode::SemiglobalLinear, b"ACGT", b"TTACGTTT"),
        ];
        for (mode, query, target) in cases {
            let original = solve(request(*mode, query, target)).unwrap();
            let query_map = HashMap::from([("q".to_string(), query.to_vec())]);
            let target_map = HashMap::from([("t".to_string(), target.to_vec())]);
            let json = records_to_json(std::slice::from_ref(&original)).unwrap();
            let paf = record_to_paf(&original).unwrap();
            let sam = records_to_sam(std::slice::from_ref(&original), &query_map).unwrap();
            let variants = [
                records_from_json(&json).unwrap().remove(0),
                records_from_paf(&paf, &original.problem, &query_map, &target_map)
                    .unwrap()
                    .remove(0),
                records_from_sam(&sam, &original.problem, &query_map, &target_map)
                    .unwrap()
                    .remove(0),
            ];
            for parsed in variants {
                assert_eq!(parsed.alignment, original.alignment, "{mode} round trip");
                let report = verify(VerificationRequest {
                    record: parsed,
                    query: query.to_vec(),
                    target: target.to_vec(),
                    level: VerificationLevel::Optimal,
                    max_cells: DEFAULT_MAX_CELLS,
                });
                assert_eq!(
                    report.verification.status,
                    VerificationStatus::Verified,
                    "{mode} round-trip verification"
                );
            }
        }
    }
}
