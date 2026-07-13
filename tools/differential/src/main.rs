use bio::alignment::pairwise::Aligner;
use bio::alignment::AlignmentOperation;
use clap::{Parser, Subcommand, ValueEnum};
use hcp_dp::oracle::{
    compare, solve, AlignmentMode, AlignmentRecord, AlignmentResult, AlignmentSpan, CheckResult,
    Cigar, CigarOp, CigarRun, ComparisonReport, Metrics, Objective, ObjectiveKind, PairIdentity,
    ProblemSpec, Producer, ScoringScheme, SequenceIdentity, SolveOptions, SolveRequest,
    SubstitutionScheme, TraceRecord, VerificationLevel, VerificationStatus, VerificationSummary,
    SCHEMA_VERSION,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::str::FromStr;
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(name = "hcp-align-differential")]
#[command(about = "Development-only alignment backend comparison and reduction")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compare explicit backends for one normalized problem.
    Check {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        left: BackendSpec,
        #[arg(long)]
        right: BackendSpec,
        #[arg(long)]
        output: PathBuf,
    },
    /// Exhaustively compare explicit backends over small binary alphabets.
    Contract {
        #[arg(long)]
        left: BackendSpec,
        #[arg(long)]
        right: BackendSpec,
        #[arg(long, default_value_t = 3)]
        max_length: usize,
        #[arg(long)]
        output: PathBuf,
    },
    /// Deterministically reduce a previously emitted disagreement.
    Reduce {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        left: BackendSpec,
        #[arg(long)]
        right: BackendSpec,
        #[arg(long, value_enum, default_value_t = Preserve::CategorySet)]
        preserve: Preserve,
        #[arg(long, default_value_t = 10_000)]
        max_evaluations: u64,
        #[arg(long, default_value_t = 300)]
        max_seconds: u64,
        #[arg(long, default_value_t = 5)]
        adapter_timeout_seconds: u64,
        #[arg(long)]
        output: PathBuf,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Preserve {
    CategorySet,
    Objective,
    Validity,
}

#[derive(Clone, Debug)]
enum BackendSpec {
    Internal(String),
    RustBio,
    Wfa2(PathBuf),
}

impl FromStr for BackendSpec {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if let Some(backend) = value.strip_prefix("internal:") {
            if backend.is_empty() {
                return Err("internal backend name cannot be empty".to_string());
            }
            return Ok(Self::Internal(backend.to_string()));
        }
        if value == "rust-bio" {
            return Ok(Self::RustBio);
        }
        if let Some(path) = value.strip_prefix("wfa2:") {
            let path = PathBuf::from(path);
            if !path.is_absolute() {
                return Err("WFA2 adapter path must be absolute".to_string());
            }
            return Ok(Self::Wfa2(path));
        }
        Err("backend must be internal:<name>, rust-bio, or wfa2:<absolute-path>".to_string())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DifferentialInput {
    query: String,
    target: String,
    problem: ProblemSpec,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum DisagreementCategory {
    Objective,
    LeftValidity,
    RightValidity,
    Coordinates,
    OperationCount,
    TieBreaking,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct DisagreementSignature {
    categories: BTreeSet<DisagreementCategory>,
    fingerprint_sha256: String,
}

impl DisagreementSignature {
    fn from_report(report: &ComparisonReport) -> Self {
        let mut categories = BTreeSet::new();
        if !report.objectives_equal {
            categories.insert(DisagreementCategory::Objective);
        }
        for difference in &report.differences {
            if difference.starts_with("left_invalid:") {
                categories.insert(DisagreementCategory::LeftValidity);
            } else if difference.starts_with("right_invalid:") {
                categories.insert(DisagreementCategory::RightValidity);
            } else if difference.starts_with("coordinates:") {
                categories.insert(DisagreementCategory::Coordinates);
            } else if difference.starts_with("operation_count:") {
                categories.insert(DisagreementCategory::OperationCount);
            } else if difference.starts_with("tie_breaking:") {
                categories.insert(DisagreementCategory::TieBreaking);
            }
        }
        let canonical = serde_json::json!({
            "categories": &categories,
            "inputs": &report.inputs,
            "problem": &report.problem,
            "left": {
                "producer": &report.left.producer,
                "alignment": &report.left.alignment,
                "verification_status": &report.left.verification.status,
            },
            "right": {
                "producer": &report.right.producer,
                "alignment": &report.right.alignment,
                "verification_status": &report.right.verification.status,
            },
            "differences": &report.differences,
        });
        let fingerprint_sha256 = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&canonical).expect("signature fields serialize"))
        );
        Self {
            categories,
            fingerprint_sha256,
        }
    }

    fn is_disagreement(&self) -> bool {
        !self.categories.is_empty()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BackendMetadata {
    specification: String,
    name: String,
    version: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ReductionStatus {
    Minimized,
    Inconclusive,
    NotApplicable,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DifferentialArtifact {
    schema_version: String,
    reduction_status: ReductionStatus,
    backends: Vec<BackendMetadata>,
    original: DifferentialInput,
    minimized: DifferentialInput,
    original_comparison: ComparisonReport,
    minimized_comparison: ComparisonReport,
    signature: DisagreementSignature,
    evaluations: u64,
    elapsed_ms: u128,
    termination_reason: String,
}

trait Backend {
    fn metadata(&self) -> BackendMetadata;
    fn align(&mut self, input: &DifferentialInput) -> Result<AlignmentRecord, String>;
}

struct InternalBackend {
    policy: String,
}

impl Backend for InternalBackend {
    fn metadata(&self) -> BackendMetadata {
        BackendMetadata {
            specification: format!("internal:{}", self.policy),
            name: "hcp-align".to_string(),
            version: hcp_dp::VERSION.to_string(),
        }
    }

    fn align(&mut self, input: &DifferentialInput) -> Result<AlignmentRecord, String> {
        solve(SolveRequest {
            pair_index: 0,
            query_id: "query".to_string(),
            target_id: "target".to_string(),
            query: input.query.as_bytes().to_vec(),
            target: input.target.as_bytes().to_vec(),
            problem: input.problem.clone(),
            options: SolveOptions {
                backend: self.policy.clone(),
                cross_check: true,
                ..SolveOptions::default()
            },
        })
    }
}

struct RustBioBackend;

impl Backend for RustBioBackend {
    fn metadata(&self) -> BackendMetadata {
        BackendMetadata {
            specification: "rust-bio".to_string(),
            name: "rust-bio".to_string(),
            version: "4.0.1".to_string(),
        }
    }

    fn align(&mut self, input: &DifferentialInput) -> Result<AlignmentRecord, String> {
        rust_bio_alignment(input)
    }
}

struct Wfa2Backend {
    path: PathBuf,
    child: Child,
    stdin: ChildStdin,
    responses: Receiver<Result<String, String>>,
    timeout: Duration,
}

impl Wfa2Backend {
    fn start(path: PathBuf, timeout: Duration) -> Result<Self, String> {
        let mut child = Command::new(&path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|err| format!("failed to start WFA2 adapter {}: {err}", path.display()))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "WFA2 adapter has no stdin".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "WFA2 adapter has no stdout".to_string())?;
        let (sender, responses) = mpsc::channel();
        thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        if sender.send(Ok(line)).is_err() {
                            break;
                        }
                    }
                    Err(err) => {
                        let _ = sender.send(Err(format!("failed to read WFA2 adapter: {err}")));
                        break;
                    }
                }
            }
        });
        Ok(Self {
            path,
            child,
            stdin,
            responses,
            timeout,
        })
    }

    fn align_once(&mut self, input: &DifferentialInput) -> Result<AlignmentRecord, String> {
        let request = wfa_request(input)?;
        writeln!(self.stdin, "{request}")
            .and_then(|_| self.stdin.flush())
            .map_err(|err| format!("failed to write WFA2 request: {err}"))?;
        let response = self.responses.recv_timeout(self.timeout).map_err(|err| {
            let _ = self.child.kill();
            format!("WFA2 adapter exceeded {:?}: {err}", self.timeout)
        })??;
        wfa_response(input, response.trim_end_matches(['\r', '\n']))
    }

    fn align_semiglobal(&mut self, input: &DifferentialInput) -> Result<AlignmentRecord, String> {
        // The global maximize-to-cost transform has a baseline term that
        // depends on aligned target length. A single symmetric ends-free WFA
        // optimization can therefore choose the wrong span. Enumerating target
        // substrings keeps the adapter exact while retaining WFA2 as the
        // end-to-end kernel for every candidate.
        let mut best: Option<(i64, usize, AlignmentRecord)> = None;
        for start in 0..=input.target.len() {
            for end in start..=input.target.len() {
                let mut candidate = input.clone();
                candidate.target = input.target[start..end].to_string();
                candidate.problem.mode = AlignmentMode::GlobalLinear;
                let record = self.align_once(&candidate)?;
                let score = record.alignment.objective.value;
                if best
                    .as_ref()
                    .is_none_or(|(best_score, _, _)| score > *best_score)
                {
                    best = Some((score, start, record));
                }
            }
        }
        let (_, offset, mut record) = best.expect("there is always an empty target substring");
        record.problem = input.problem.clone();
        record.pair.target = SequenceIdentity {
            id: "target".to_string(),
            length: input.target.len(),
            sha256: digest(input.target.as_bytes()),
        };
        if let Some(trace) = record.alignment.trace.as_mut() {
            trace.target_span.start += offset;
            trace.target_span.end += offset;
        }
        Ok(record)
    }
}

impl Drop for Wfa2Backend {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Backend for Wfa2Backend {
    fn metadata(&self) -> BackendMetadata {
        BackendMetadata {
            specification: format!("wfa2:{}", self.path.display()),
            name: "WFA2-lib".to_string(),
            version: "2.3.6@bcf473a6561fa297934a80eaa7c04b4ee412360c".to_string(),
        }
    }

    fn align(&mut self, input: &DifferentialInput) -> Result<AlignmentRecord, String> {
        if input.problem.mode == AlignmentMode::SemiglobalLinear {
            self.align_semiglobal(input)
        } else {
            self.align_once(input)
        }
    }
}

fn create_backend(spec: &BackendSpec, timeout: Duration) -> Result<Box<dyn Backend>, String> {
    match spec {
        BackendSpec::Internal(policy) => Ok(Box::new(InternalBackend {
            policy: policy.clone(),
        })),
        BackendSpec::RustBio => Ok(Box::new(RustBioBackend)),
        BackendSpec::Wfa2(path) => Ok(Box::new(Wfa2Backend::start(path.clone(), timeout)?)),
    }
}

fn digest(sequence: &[u8]) -> String {
    format!("{:x}", Sha256::digest(sequence))
}

fn substitution_score(scheme: &SubstitutionScheme, a: u8, b: u8) -> i32 {
    match scheme {
        SubstitutionScheme::MatchMismatch {
            match_score,
            mismatch_penalty,
        } => {
            if a == b {
                *match_score
            } else {
                -*mismatch_penalty
            }
        }
        SubstitutionScheme::Blosum62 => bio::scores::blosum62::blosum62(a, b),
        SubstitutionScheme::Matrix {
            alphabet, scores, ..
        } => {
            let ai = alphabet
                .iter()
                .position(|symbol| symbol.as_bytes().first().copied() == Some(a));
            let bi = alphabet
                .iter()
                .position(|symbol| symbol.as_bytes().first().copied() == Some(b));
            ai.and_then(|row| bi.and_then(|col| scores.get(row).and_then(|r| r.get(col)).copied()))
                .unwrap_or(i32::MIN / 4)
        }
    }
}

fn rust_bio_alignment(input: &DifferentialInput) -> Result<AlignmentRecord, String> {
    let query = input.query.as_bytes();
    let target = input.target.as_bytes();
    let (gap_open, gap_extend, substitution, edit_distance) = match &input.problem.scoring {
        ScoringScheme::EditDistance => (
            -1,
            -1,
            SubstitutionScheme::MatchMismatch {
                match_score: 0,
                mismatch_penalty: 1,
            },
            true,
        ),
        ScoringScheme::Linear { substitution, gap } => (*gap, *gap, substitution.clone(), false),
        ScoringScheme::Affine {
            substitution,
            gap_open,
            gap_extend,
        } => (
            gap_open
                .checked_add(*gap_extend)
                .ok_or_else(|| "affine gap translation overflowed".to_string())?,
            *gap_extend,
            substitution.clone(),
            false,
        ),
    };
    let score_scheme = substitution.clone();
    let mut aligner = Aligner::with_capacity(
        query.len(),
        target.len(),
        gap_open,
        gap_extend,
        move |a, b| substitution_score(&score_scheme, a, b),
    );
    let alignment = match input.problem.mode {
        AlignmentMode::EditDistance | AlignmentMode::GlobalLinear | AlignmentMode::GlobalAffine => {
            aligner.global(query, target)
        }
        AlignmentMode::LocalLinear => aligner.local(query, target),
        AlignmentMode::SemiglobalLinear => aligner.semiglobal(query, target),
    };
    let cigar = cigar_from_bio(&alignment.operations)?;
    let objective = if edit_distance {
        Objective {
            kind: ObjectiveKind::Distance,
            value: -i64::from(alignment.score),
        }
    } else {
        Objective {
            kind: ObjectiveKind::Score,
            value: i64::from(alignment.score),
        }
    };
    Ok(external_record(
        input,
        "rust-bio-4.0.1",
        objective,
        AlignmentSpan {
            start: alignment.xstart,
            end: alignment.xend,
        },
        AlignmentSpan {
            start: alignment.ystart,
            end: alignment.yend,
        },
        cigar,
    ))
}

fn cigar_from_bio(operations: &[AlignmentOperation]) -> Result<Cigar, String> {
    let mut runs = Vec::<CigarRun>::new();
    for operation in operations {
        let op = match operation {
            AlignmentOperation::Match => CigarOp::Match,
            AlignmentOperation::Subst => CigarOp::Mismatch,
            AlignmentOperation::Ins => CigarOp::Insertion,
            AlignmentOperation::Del => CigarOp::Deletion,
            AlignmentOperation::Xclip(_) | AlignmentOperation::Yclip(_) => continue,
        };
        if let Some(last) = runs.last_mut().filter(|run| run.op == op) {
            last.len += 1;
        } else {
            runs.push(CigarRun { len: 1, op });
        }
    }
    Cigar::new(runs)
}

fn external_record(
    input: &DifferentialInput,
    backend: &str,
    objective: Objective,
    query_span: AlignmentSpan,
    target_span: AlignmentSpan,
    cigar: Cigar,
) -> AlignmentRecord {
    AlignmentRecord {
        schema_version: SCHEMA_VERSION.to_string(),
        record_type: "alignment".to_string(),
        producer: Producer {
            name: backend.to_string(),
            version: backend.to_string(),
            commit: None,
            backend: backend.to_string(),
        },
        pair: PairIdentity {
            index: 0,
            query: SequenceIdentity {
                id: "query".to_string(),
                length: input.query.len(),
                sha256: digest(input.query.as_bytes()),
            },
            target: SequenceIdentity {
                id: "target".to_string(),
                length: input.target.len(),
                sha256: digest(input.target.as_bytes()),
            },
        },
        problem: input.problem.clone(),
        alignment: AlignmentResult {
            objective,
            trace: Some(TraceRecord {
                query_span,
                target_span,
                cigar,
                operations: None,
            }),
        },
        verification: VerificationSummary {
            level: VerificationLevel::Optimal,
            status: VerificationStatus::Verified,
            checks: vec![CheckResult {
                name: "adapter_conversion".to_string(),
                passed: true,
                detail: None,
            }],
            reference_backend: Some("rolling-frontier".to_string()),
            reason: None,
        },
        metrics: Some(Metrics {
            elapsed_ms: 0.0,
            work_cells: 0,
        }),
        integrity: None,
    }
}

fn match_mismatch(problem: &ProblemSpec) -> Result<(i32, i32), String> {
    match &problem.scoring {
        ScoringScheme::Linear {
            substitution:
                SubstitutionScheme::MatchMismatch {
                    match_score,
                    mismatch_penalty,
                },
            ..
        }
        | ScoringScheme::Affine {
            substitution:
                SubstitutionScheme::MatchMismatch {
                    match_score,
                    mismatch_penalty,
                },
            ..
        } => Ok((*match_score, *mismatch_penalty)),
        ScoringScheme::EditDistance => Ok((0, 1)),
        _ => Err("WFA2 does not support substitution matrices".to_string()),
    }
}

fn wfa_request(input: &DifferentialInput) -> Result<String, String> {
    if input.problem.mode == AlignmentMode::LocalLinear {
        return Err("WFA2 adapter does not support local alignment".to_string());
    }
    let mode = input.problem.mode.to_string();
    let (mismatch_cost, gap_open, gap_extend) = match &input.problem.scoring {
        ScoringScheme::EditDistance => (1, 0, 1),
        ScoringScheme::Linear { gap, .. } => {
            let (match_score, mismatch_penalty) = match_mismatch(&input.problem)?;
            let mismatch = 2i32
                .checked_mul(
                    match_score
                        .checked_add(mismatch_penalty)
                        .ok_or("WFA mismatch overflow")?,
                )
                .ok_or("WFA mismatch overflow")?;
            let indel = match_score
                .checked_add(2i32.checked_mul(gap.abs()).ok_or("WFA gap overflow")?)
                .ok_or("WFA gap overflow")?;
            (mismatch, 0, indel)
        }
        ScoringScheme::Affine {
            gap_open,
            gap_extend,
            ..
        } => {
            let (match_score, mismatch_penalty) = match_mismatch(&input.problem)?;
            let mismatch = 2i32
                .checked_mul(
                    match_score
                        .checked_add(mismatch_penalty)
                        .ok_or("WFA mismatch overflow")?,
                )
                .ok_or("WFA mismatch overflow")?;
            let opening = 2i32
                .checked_mul(gap_open.abs())
                .ok_or("WFA opening overflow")?;
            let extension = match_score
                .checked_add(
                    2i32.checked_mul(gap_extend.abs())
                        .ok_or("WFA extension overflow")?,
                )
                .ok_or("WFA extension overflow")?;
            (mismatch, opening, extension)
        }
    };
    if mismatch_cost <= 0 || gap_extend <= 0 || gap_open < 0 {
        return Err("WFA2 translation produced non-positive costs".to_string());
    }
    if input.query.contains(['\t', '\n']) || input.target.contains(['\t', '\n']) {
        return Err("WFA2 protocol sequences cannot contain tabs or newlines".to_string());
    }
    Ok(format!(
        "{mode}\t{mismatch_cost}\t{gap_open}\t{gap_extend}\t{}\t{}",
        input.query, input.target
    ))
}

fn wfa_response(input: &DifferentialInput, response: &str) -> Result<AlignmentRecord, String> {
    let fields: Vec<&str> = response.split('\t').collect();
    if fields.first() == Some(&"unsupported") {
        return Err(fields
            .get(1)
            .unwrap_or(&"unsupported WFA2 request")
            .to_string());
    }
    if fields.first() != Some(&"ok") || fields.len() != 7 {
        return Err(format!("malformed WFA2 response: {response}"));
    }
    let cost: i64 = fields[1].parse().map_err(|_| "invalid WFA2 cost")?;
    let query_start: usize = fields[2].parse().map_err(|_| "invalid WFA2 query start")?;
    let query_end: usize = fields[3].parse().map_err(|_| "invalid WFA2 query end")?;
    let target_start: usize = fields[4].parse().map_err(|_| "invalid WFA2 target start")?;
    let target_end: usize = fields[5].parse().map_err(|_| "invalid WFA2 target end")?;
    let cigar = normalize_wfa_cigar(
        fields[6],
        input.query.as_bytes(),
        input.target.as_bytes(),
        query_start,
        target_start,
    )?;
    let objective = if input.problem.scoring == ScoringScheme::EditDistance {
        Objective {
            kind: ObjectiveKind::Distance,
            value: cost,
        }
    } else {
        let (match_score, _) = match_mismatch(&input.problem)?;
        let numerator = i64::from(match_score)
            .checked_mul((input.query.len() + target_end.saturating_sub(target_start)) as i64)
            .and_then(|value| value.checked_sub(cost))
            .ok_or_else(|| "WFA2 recovered score overflowed".to_string())?;
        if numerator % 2 != 0 {
            return Err("WFA2 recovered score is non-integral".to_string());
        }
        Objective {
            kind: ObjectiveKind::Score,
            value: numerator / 2,
        }
    };
    Ok(external_record(
        input,
        "wfa2-2.3.6",
        objective,
        AlignmentSpan {
            start: query_start,
            end: query_end,
        },
        AlignmentSpan {
            start: target_start,
            end: target_end,
        },
        cigar,
    ))
}

fn normalize_wfa_cigar(
    raw: &str,
    query: &[u8],
    target: &[u8],
    mut query_pos: usize,
    mut target_pos: usize,
) -> Result<Cigar, String> {
    let mut runs = Vec::new();
    let mut count = 0usize;
    for byte in raw.bytes() {
        if byte.is_ascii_digit() {
            count = count
                .checked_mul(10)
                .and_then(|value| value.checked_add((byte - b'0') as usize))
                .ok_or_else(|| "WFA2 CIGAR length overflow".to_string())?;
            continue;
        }
        if count == 0 {
            return Err("WFA2 CIGAR contains a zero or missing run length".to_string());
        }
        match byte {
            b'M' => {
                for _ in 0..count {
                    let op = if query.get(query_pos) == target.get(target_pos) {
                        CigarOp::Match
                    } else {
                        CigarOp::Mismatch
                    };
                    runs.push(CigarRun { len: 1, op });
                    query_pos += 1;
                    target_pos += 1;
                }
            }
            b'=' | b'X' => {
                runs.push(CigarRun {
                    len: count,
                    op: if byte == b'=' {
                        CigarOp::Match
                    } else {
                        CigarOp::Mismatch
                    },
                });
                query_pos += count;
                target_pos += count;
            }
            b'I' => {
                runs.push(CigarRun {
                    len: count,
                    // WFA's I consumes text (our target), the SAM-oriented D.
                    op: CigarOp::Deletion,
                });
                target_pos += count;
            }
            b'D' => {
                runs.push(CigarRun {
                    len: count,
                    // WFA's D consumes pattern (our query), the SAM-oriented I.
                    op: CigarOp::Insertion,
                });
                query_pos += count;
            }
            _ => {
                return Err(format!(
                    "unsupported WFA2 CIGAR operation '{}'",
                    byte as char
                ))
            }
        }
        count = 0;
    }
    if count != 0 {
        return Err("WFA2 CIGAR ends in a length".to_string());
    }
    Cigar::new(runs)
}

fn evaluate(
    input: &DifferentialInput,
    left: &mut dyn Backend,
    right: &mut dyn Backend,
) -> Result<ComparisonReport, String> {
    let left_record = left.align(input)?;
    let right_record = right.align(input)?;
    compare(
        left_record,
        right_record,
        input.query.as_bytes(),
        input.target.as_bytes(),
    )
}

struct Reducer<'a> {
    left: &'a mut dyn Backend,
    right: &'a mut dyn Backend,
    original_signature: DisagreementSignature,
    preserve: Preserve,
    evaluations: u64,
    max_evaluations: u64,
    deadline: Instant,
    limited: bool,
    unstable: bool,
}

impl Reducer<'_> {
    fn accepts(&mut self, input: &DifferentialInput) -> Option<ComparisonReport> {
        if self.evaluations >= self.max_evaluations || Instant::now() >= self.deadline {
            self.limited = true;
            return None;
        }
        self.evaluations += 1;
        let report = match evaluate(input, self.left, self.right) {
            Ok(report) => report,
            Err(_) => {
                self.unstable = true;
                return None;
            }
        };
        let signature = DisagreementSignature::from_report(&report);
        let accepted = match self.preserve {
            Preserve::CategorySet => signature.categories == self.original_signature.categories,
            Preserve::Objective => signature
                .categories
                .contains(&DisagreementCategory::Objective),
            Preserve::Validity => {
                signature
                    .categories
                    .contains(&DisagreementCategory::LeftValidity)
                    || signature
                        .categories
                        .contains(&DisagreementCategory::RightValidity)
            }
        };
        accepted.then_some(report)
    }

    fn reduce_axis(
        &mut self,
        best: &mut DifferentialInput,
        best_report: &mut ComparisonReport,
        query_axis: bool,
    ) {
        let mut granularity = 2usize;
        loop {
            let length = if query_axis {
                best.query.len()
            } else {
                best.target.len()
            };
            if length == 0 || self.limited || self.unstable {
                return;
            }
            let chunk = length.div_ceil(granularity);
            let mut accepted = false;
            let mut start = 0usize;
            while start < length {
                let end = (start + chunk).min(length);
                let mut candidate = best.clone();
                if query_axis {
                    candidate.query.replace_range(start..end, "");
                } else {
                    candidate.target.replace_range(start..end, "");
                }
                if let Some(report) = self.accepts(&candidate) {
                    *best = candidate;
                    *best_report = report;
                    granularity = granularity.saturating_sub(1).max(2);
                    accepted = true;
                    break;
                }
                if self.limited || self.unstable {
                    return;
                }
                start = end;
            }
            if accepted {
                continue;
            }
            if granularity >= length {
                return;
            }
            granularity = (granularity * 2).min(length);
        }
    }

    fn simplify(&mut self, best: &mut DifferentialInput, best_report: &mut ComparisonReport) {
        let mut alphabet: Vec<u8> = best
            .query
            .bytes()
            .chain(best.target.bytes())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        alphabet.sort_unstable();
        for query_axis in [true, false] {
            let length = if query_axis {
                best.query.len()
            } else {
                best.target.len()
            };
            for index in 0..length {
                if self.limited || self.unstable {
                    return;
                }
                let current = if query_axis {
                    best.query.as_bytes()[index]
                } else {
                    best.target.as_bytes()[index]
                };
                for &replacement in alphabet.iter().filter(|&&symbol| symbol < current) {
                    let mut candidate = best.clone();
                    if query_axis {
                        candidate.query = replace_ascii(&candidate.query, index, replacement);
                    } else {
                        candidate.target = replace_ascii(&candidate.target, index, replacement);
                    }
                    if let Some(report) = self.accepts(&candidate) {
                        *best = candidate;
                        *best_report = report;
                        break;
                    }
                    if self.limited || self.unstable {
                        return;
                    }
                }
            }
        }
    }
}

fn replace_ascii(text: &str, index: usize, replacement: u8) -> String {
    let mut bytes = text.as_bytes().to_vec();
    bytes[index] = replacement;
    String::from_utf8(bytes).expect("replacing one ASCII byte preserves UTF-8")
}

fn read_input(path: &Path) -> Result<(DifferentialInput, Option<ComparisonReport>), String> {
    let text = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    if let Ok(report) = serde_json::from_str::<ComparisonReport>(&text) {
        let input = DifferentialInput {
            query: report.inputs.query.clone(),
            target: report.inputs.target.clone(),
            problem: report.problem.clone(),
        };
        validate_input(&input)?;
        return Ok((input, Some(report)));
    }
    let input = serde_json::from_str::<DifferentialInput>(&text)
        .map_err(|err| format!("input is neither a comparison nor differential input: {err}"))?;
    validate_input(&input)?;
    Ok((input, None))
}

fn validate_input(input: &DifferentialInput) -> Result<(), String> {
    if !input.query.is_ascii() || !input.target.is_ascii() {
        return Err("differential sequences must be ASCII".to_string());
    }
    Ok(())
}

fn write_artifact(path: &Path, artifact: &DifferentialArtifact) -> Result<(), String> {
    let mut bytes = serde_json::to_vec_pretty(artifact).map_err(|err| err.to_string())?;
    bytes.push(b'\n');
    fs::write(path, bytes).map_err(|err| format!("failed to write {}: {err}", path.display()))
}

fn artifact_for(
    input: DifferentialInput,
    backends: Vec<BackendMetadata>,
    report: ComparisonReport,
    elapsed_ms: u128,
) -> DifferentialArtifact {
    DifferentialArtifact {
        schema_version: "hcp-align.differential.v1".to_string(),
        reduction_status: ReductionStatus::NotApplicable,
        backends,
        original: input.clone(),
        minimized: input,
        original_comparison: report.clone(),
        minimized_comparison: report.clone(),
        signature: DisagreementSignature::from_report(&report),
        evaluations: 1,
        elapsed_ms,
        termination_reason: "comparison_only".to_string(),
    }
}

fn binary_strings(max_length: usize) -> Vec<String> {
    let mut strings = vec![String::new()];
    for length in 1..=max_length {
        for bits in 0..(1usize << length) {
            strings.push(
                (0..length)
                    .map(|offset| if bits & (1 << offset) == 0 { 'A' } else { 'C' })
                    .collect(),
            );
        }
    }
    strings
}

fn write_artifacts(path: &Path, artifacts: &[DifferentialArtifact]) -> Result<(), String> {
    let mut bytes = serde_json::to_vec_pretty(artifacts).map_err(|err| err.to_string())?;
    bytes.push(b'\n');
    fs::write(path, bytes).map_err(|err| format!("failed to write {}: {err}", path.display()))
}

fn main() {
    if let Err(error) = run() {
        eprintln!("hcp-align-differential: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    match Cli::parse().command {
        Commands::Check {
            input,
            left,
            right,
            output,
        } => {
            let (input, _) = read_input(&input)?;
            let timeout = Duration::from_secs(5);
            let mut left = create_backend(&left, timeout)?;
            let mut right = create_backend(&right, timeout)?;
            let start = Instant::now();
            let report = evaluate(&input, left.as_mut(), right.as_mut())?;
            let artifact = artifact_for(
                input,
                vec![left.metadata(), right.metadata()],
                report,
                start.elapsed().as_millis(),
            );
            write_artifact(&output, &artifact)
        }
        Commands::Contract {
            left,
            right,
            max_length,
            output,
        } => {
            if max_length > 6 {
                return Err("--max-length is capped at 6".to_string());
            }
            let includes_wfa =
                matches!(&left, BackendSpec::Wfa2(_)) || matches!(&right, BackendSpec::Wfa2(_));
            let timeout = Duration::from_secs(5);
            let mut left = create_backend(&left, timeout)?;
            let mut right = create_backend(&right, timeout)?;
            let metadata = vec![left.metadata(), right.metadata()];
            let mut modes = vec![
                AlignmentMode::EditDistance,
                AlignmentMode::GlobalLinear,
                AlignmentMode::GlobalAffine,
                AlignmentMode::SemiglobalLinear,
            ];
            if !includes_wfa {
                modes.push(AlignmentMode::LocalLinear);
            }
            let strings = binary_strings(max_length);
            let mut artifacts = Vec::new();
            for mode in modes {
                for query in &strings {
                    for target in &strings {
                        let input = DifferentialInput {
                            query: query.clone(),
                            target: target.clone(),
                            problem: ProblemSpec {
                                mode,
                                scoring: ScoringScheme::default_for(mode),
                            },
                        };
                        let start = Instant::now();
                        let report = evaluate(&input, left.as_mut(), right.as_mut()).map_err(
                            |error| {
                                format!(
                                    "contract case {mode} query={query:?} target={target:?}: {error}"
                                )
                            },
                        )?;
                        artifacts.push(artifact_for(
                            input,
                            metadata.clone(),
                            report,
                            start.elapsed().as_millis(),
                        ));
                    }
                }
            }
            write_artifacts(&output, &artifacts)
        }
        Commands::Reduce {
            input,
            left,
            right,
            preserve,
            max_evaluations,
            max_seconds,
            adapter_timeout_seconds,
            output,
        } => {
            let (original, provided) = read_input(&input)?;
            let timeout = Duration::from_secs(adapter_timeout_seconds);
            let mut left = create_backend(&left, timeout)?;
            let mut right = create_backend(&right, timeout)?;
            let start = Instant::now();
            let original_report = match provided {
                Some(report) => report,
                None => evaluate(&original, left.as_mut(), right.as_mut())?,
            };
            let signature = DisagreementSignature::from_report(&original_report);
            if !signature.is_disagreement() {
                return Err("input does not contain a disagreement".to_string());
            }
            if matches!(preserve, Preserve::Objective)
                && !signature
                    .categories
                    .contains(&DisagreementCategory::Objective)
            {
                return Err("--preserve objective requires an objective disagreement".to_string());
            }
            if matches!(preserve, Preserve::Validity)
                && !signature
                    .categories
                    .contains(&DisagreementCategory::LeftValidity)
                && !signature
                    .categories
                    .contains(&DisagreementCategory::RightValidity)
            {
                return Err("--preserve validity requires an invalid backend result".to_string());
            }
            let metadata = vec![left.metadata(), right.metadata()];
            let mut best = original.clone();
            let mut best_report = original_report.clone();
            let mut reducer = Reducer {
                left: left.as_mut(),
                right: right.as_mut(),
                original_signature: signature.clone(),
                preserve,
                evaluations: 0,
                max_evaluations,
                deadline: start + Duration::from_secs(max_seconds),
                limited: false,
                unstable: false,
            };
            loop {
                let before = (best.query.len(), best.target.len());
                reducer.reduce_axis(&mut best, &mut best_report, true);
                reducer.reduce_axis(&mut best, &mut best_report, false);
                if before == (best.query.len(), best.target.len())
                    || reducer.limited
                    || reducer.unstable
                {
                    break;
                }
            }
            reducer.simplify(&mut best, &mut best_report);
            let (status, reason) = if reducer.unstable {
                (ReductionStatus::Inconclusive, "adapter_instability")
            } else if reducer.limited {
                (ReductionStatus::Inconclusive, "limit_reached")
            } else {
                (ReductionStatus::Minimized, "local_minimum")
            };
            let artifact = DifferentialArtifact {
                schema_version: "hcp-align.differential.v1".to_string(),
                reduction_status: status,
                backends: metadata,
                original,
                minimized: best,
                original_comparison: original_report,
                minimized_comparison: best_report,
                signature,
                evaluations: reducer.evaluations,
                elapsed_ms: start.elapsed().as_millis(),
                termination_reason: reason.to_string(),
            };
            write_artifact(&output, &artifact)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal_metadata_uses_the_oracle_version() {
        let backend = InternalBackend {
            policy: "auto".to_string(),
        };
        let metadata = backend.metadata();
        assert_eq!(metadata.name, "hcp-align");
        assert_eq!(metadata.version, hcp_dp::VERSION);
    }

    fn input(mode: AlignmentMode, query: &str, target: &str) -> DifferentialInput {
        DifferentialInput {
            query: query.to_string(),
            target: target.to_string(),
            problem: ProblemSpec {
                mode,
                scoring: ScoringScheme::default_for(mode),
            },
        }
    }

    struct ObjectiveOffset;

    impl Backend for ObjectiveOffset {
        fn metadata(&self) -> BackendMetadata {
            BackendMetadata {
                specification: "test:objective-offset".to_string(),
                name: "test".to_string(),
                version: "1".to_string(),
            }
        }

        fn align(&mut self, input: &DifferentialInput) -> Result<AlignmentRecord, String> {
            let mut backend = InternalBackend {
                policy: "auto".to_string(),
            };
            let mut record = backend.align(input)?;
            record.alignment.objective.value += 1;
            Ok(record)
        }
    }

    #[test]
    fn rust_bio_objectives_match_internal_for_all_modes() {
        let cases = [
            input(AlignmentMode::EditDistance, "GATTACA", "GCATGCU"),
            input(AlignmentMode::GlobalLinear, "GATTACA", "GCATGCU"),
            input(AlignmentMode::GlobalAffine, "ACGGGT", "ACT"),
            input(AlignmentMode::LocalLinear, "ACACACTA", "AGCACACA"),
            input(AlignmentMode::SemiglobalLinear, "ACGT", "TTACGTTT"),
        ];
        for case in cases {
            let mut internal = InternalBackend {
                policy: "auto".to_string(),
            };
            let mut rust_bio = RustBioBackend;
            let report = evaluate(&case, &mut internal, &mut rust_bio).unwrap();
            assert!(report.objectives_equal, "{:?}", report.differences);
            assert!(report.equivalent, "{:?}", report.differences);
        }
    }

    #[test]
    fn rust_bio_matches_exhaustive_small_binary_pairs() {
        let strings = binary_strings(3);
        for mode in [
            AlignmentMode::EditDistance,
            AlignmentMode::GlobalLinear,
            AlignmentMode::GlobalAffine,
            AlignmentMode::LocalLinear,
            AlignmentMode::SemiglobalLinear,
        ] {
            for query in &strings {
                for target in &strings {
                    let case = input(mode, query, target);
                    let mut internal = InternalBackend {
                        policy: "auto".to_string(),
                    };
                    let mut rust_bio = RustBioBackend;
                    let report = evaluate(&case, &mut internal, &mut rust_bio).unwrap();
                    assert!(
                        report.equivalent,
                        "mode={mode} query={query:?} target={target:?}: {:?}",
                        report.differences
                    );
                }
            }
        }
    }

    #[test]
    fn wfa_cost_translation_rejects_local_and_matrices() {
        assert!(wfa_request(&input(AlignmentMode::LocalLinear, "A", "A")).is_err());
        let mut matrix = input(AlignmentMode::GlobalLinear, "A", "A");
        matrix.problem.scoring = ScoringScheme::Linear {
            substitution: SubstitutionScheme::Blosum62,
            gap: -2,
        };
        assert!(wfa_request(&matrix).is_err());
    }

    #[test]
    fn wfa_m_operations_expand_to_canonical_orientation() {
        let cigar = normalize_wfa_cigar("1M1I1M1D", b"ACG", b"ATG", 0, 0).unwrap();
        assert_eq!(cigar.to_string(), "1=1D1X1I");
    }

    #[test]
    fn signatures_distinguish_coordinate_operation_and_validity_failures() {
        let local = input(AlignmentMode::LocalLinear, "A", "AA");
        let mut internal = InternalBackend {
            policy: "auto".to_string(),
        };
        let left = internal.align(&local).unwrap();
        let mut shifted = left.clone();
        let trace = shifted.alignment.trace.as_mut().unwrap();
        trace.target_span = if trace.target_span.start == 0 {
            AlignmentSpan { start: 1, end: 2 }
        } else {
            AlignmentSpan { start: 0, end: 1 }
        };
        let report = compare(left, shifted, b"A", b"AA").unwrap();
        let signature = DisagreementSignature::from_report(&report);
        assert!(signature
            .categories
            .contains(&DisagreementCategory::Coordinates));
        assert!(signature
            .categories
            .contains(&DisagreementCategory::TieBreaking));

        let mut zero_cost = input(AlignmentMode::GlobalLinear, "A", "A");
        zero_cost.problem.scoring = ScoringScheme::Linear {
            substitution: SubstitutionScheme::MatchMismatch {
                match_score: 0,
                mismatch_penalty: 0,
            },
            gap: 0,
        };
        let left = internal.align(&zero_cost).unwrap();
        let mut longer = left.clone();
        longer.alignment.trace.as_mut().unwrap().cigar = Cigar::new(vec![
            CigarRun {
                len: 1,
                op: CigarOp::Insertion,
            },
            CigarRun {
                len: 1,
                op: CigarOp::Deletion,
            },
        ])
        .unwrap();
        let report = compare(left.clone(), longer, b"A", b"A").unwrap();
        let signature = DisagreementSignature::from_report(&report);
        assert!(signature
            .categories
            .contains(&DisagreementCategory::OperationCount));
        assert!(signature
            .categories
            .contains(&DisagreementCategory::TieBreaking));

        let mut invalid = left;
        invalid.alignment.objective.value += 1;
        let valid = internal.align(&zero_cost).unwrap();
        let report = compare(valid, invalid, b"A", b"A").unwrap();
        let signature = DisagreementSignature::from_report(&report);
        assert!(signature
            .categories
            .contains(&DisagreementCategory::Objective));
        assert!(signature
            .categories
            .contains(&DisagreementCategory::RightValidity));
    }

    #[test]
    fn reducer_is_deterministic_and_preserves_exact_categories() {
        fn reduce_once() -> (DifferentialInput, DisagreementSignature) {
            let original = input(AlignmentMode::EditDistance, "ACGT", "AGGT");
            let mut left = InternalBackend {
                policy: "auto".to_string(),
            };
            let mut right = ObjectiveOffset;
            let report = evaluate(&original, &mut left, &mut right).unwrap();
            let signature = DisagreementSignature::from_report(&report);
            let mut best = original;
            let mut best_report = report;
            let start = Instant::now();
            let mut reducer = Reducer {
                left: &mut left,
                right: &mut right,
                original_signature: signature.clone(),
                preserve: Preserve::CategorySet,
                evaluations: 0,
                max_evaluations: 1_000,
                deadline: start + Duration::from_secs(5),
                limited: false,
                unstable: false,
            };
            reducer.reduce_axis(&mut best, &mut best_report, true);
            reducer.reduce_axis(&mut best, &mut best_report, false);
            reducer.simplify(&mut best, &mut best_report);
            assert_eq!(
                DisagreementSignature::from_report(&best_report).categories,
                signature.categories
            );
            (best, signature)
        }
        let first = reduce_once();
        let second = reduce_once();
        assert_eq!(first.0.query, second.0.query);
        assert_eq!(first.0.target, second.0.target);
        assert_eq!(first.1, second.1);
    }

    #[test]
    fn reducer_marks_best_so_far_when_evaluation_limit_is_reached() {
        let original = input(AlignmentMode::EditDistance, "ACGT", "AGGT");
        let mut left = InternalBackend {
            policy: "auto".to_string(),
        };
        let mut right = ObjectiveOffset;
        let report = evaluate(&original, &mut left, &mut right).unwrap();
        let signature = DisagreementSignature::from_report(&report);
        let mut best = original;
        let mut best_report = report;
        let mut reducer = Reducer {
            left: &mut left,
            right: &mut right,
            original_signature: signature,
            preserve: Preserve::CategorySet,
            evaluations: 0,
            max_evaluations: 1,
            deadline: Instant::now() + Duration::from_secs(5),
            limited: false,
            unstable: false,
        };
        reducer.reduce_axis(&mut best, &mut best_report, true);
        assert!(reducer.limited);
        assert_eq!(reducer.evaluations, 1);
    }

    #[test]
    fn reducer_treats_adapter_failure_as_inconclusive() {
        struct FailingBackend;
        impl Backend for FailingBackend {
            fn metadata(&self) -> BackendMetadata {
                BackendMetadata {
                    specification: "test:failure".to_string(),
                    name: "test".to_string(),
                    version: "1".to_string(),
                }
            }

            fn align(&mut self, _: &DifferentialInput) -> Result<AlignmentRecord, String> {
                Err("injected adapter failure".to_string())
            }
        }

        let original = input(AlignmentMode::EditDistance, "AC", "AG");
        let mut left = InternalBackend {
            policy: "auto".to_string(),
        };
        let mut right = FailingBackend;
        let mut reducer = Reducer {
            left: &mut left,
            right: &mut right,
            original_signature: DisagreementSignature {
                categories: [DisagreementCategory::Objective].into_iter().collect(),
                fingerprint_sha256: String::new(),
            },
            preserve: Preserve::CategorySet,
            evaluations: 0,
            max_evaluations: 10,
            deadline: Instant::now() + Duration::from_secs(5),
            limited: false,
            unstable: false,
        };
        assert!(reducer.accepts(&original).is_none());
        assert!(reducer.unstable);
        assert!(!reducer.limited);
    }
}
