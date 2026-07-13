use clap::{Args, Parser, Subcommand, ValueEnum};
use hcp_dp::batch::{run_ordered, BatchItem, BatchOptions, ErrorContext, ErrorRecord, ErrorStage};
use hcp_dp::oracle::{
    record_to_paf, AlignmentMode, AlignmentRecord, ComparisonReport, ProblemSpec, ScoringScheme,
    SolveOptions, SolveRequest, SolveScope, SubstitutionScheme, VerificationLevel,
    VerificationReport, VerificationRequest, VerificationStatus, DEFAULT_MAX_CELLS,
};
use hcp_dp::record_io::{AlignmentInputFormat, AlignmentRecordReader, ResolvedAlignment, SamSpool};
use hcp_dp::sequence_io::{open_sequence_source, SequenceReadError, SequenceSource};
use hcp_dp::sequence_store::{SequenceKind, SequenceStore};
use hcp_dp::{compare, solve, verify};
use serde::Serialize;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    match run() {
        Ok(code) => ExitCode::from(code),
        Err(error) => {
            eprintln!("hcp-align: {error}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<u8, String> {
    match Cli::parse().command {
        Command::Solve(command) => run_solve(command),
        Command::Verify(command) => run_verify(command),
        Command::Compare(command) => run_compare(command),
    }
}

#[derive(Parser)]
#[command(name = "hcp-align", version)]
#[command(about = "Exact alignment reference generation, verification, and comparison")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate an exact reference score or alignment.
    Solve(SolveCommand),
    /// Verify JSON, JSONL, PAF, or SAM records against source sequences.
    Verify(VerifyCommand),
    /// Compare two alignment result sets by exact objective equivalence.
    Compare(CompareCommand),
}

#[derive(Args)]
struct SolveCommand {
    #[command(subcommand)]
    mode: SolveMode,
}

#[derive(Subcommand)]
enum SolveMode {
    EditDistance(EditSolve),
    GlobalLinear(LinearSolve),
    GlobalAffine(AffineSolve),
    LocalLinear(LinearSolve),
    SemiglobalLinear(LinearSolve),
}

#[derive(Args)]
struct EditSolve {
    #[command(flatten)]
    common: CommonSolve,
}

#[derive(Args)]
struct LinearSolve {
    #[command(flatten)]
    common: CommonSolve,
    #[command(flatten)]
    scoring: LinearScoringArgs,
}

#[derive(Args)]
struct AffineSolve {
    #[command(flatten)]
    common: CommonSolve,
    #[command(flatten)]
    scoring: AffineScoringArgs,
}

#[derive(Args)]
struct CommonSolve {
    #[command(flatten)]
    input: InputArgs,
    #[command(flatten)]
    output: OutputArgs,
    #[command(flatten)]
    batch: BatchArgs,
    /// Emit only the exact objective, without traceback.
    #[arg(long, value_enum, default_value_t = ScopeArg::Alignment)]
    scope: ScopeArg,
    /// Exact backend policy: auto, hcp, or banded-affine where applicable.
    #[arg(long, default_value = "auto")]
    backend: String,
    /// HCP summary-tree block size.
    #[arg(long)]
    block_size: Option<usize>,
    /// Diagonal half-band for banded-affine.
    #[arg(long, default_value_t = 64)]
    band: usize,
    /// Cross-check the result with an independent rolling-frontier optimum.
    #[arg(long)]
    cross_check: bool,
    /// Include per-operation JSON details.
    #[arg(long)]
    operations: bool,
    /// Add a SHA-256 integrity fingerprint over deterministic record fields.
    #[arg(long)]
    integrity: bool,
}

#[derive(Clone, Args)]
struct InputArgs {
    #[arg(long)]
    query: Option<String>,
    #[arg(long)]
    query_file: Option<PathBuf>,
    #[arg(long)]
    target: Option<String>,
    #[arg(long)]
    target_file: Option<PathBuf>,
}

#[derive(Clone, Args)]
struct OutputArgs {
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    format: OutputFormat,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Clone, Args)]
struct BatchArgs {
    /// Number of worker threads (default: min(available CPUs, 8)).
    #[arg(long)]
    threads: Option<usize>,
    /// Maximum submitted but not emitted records (default: 2 x threads).
    #[arg(long)]
    max_in_flight: Option<usize>,
    /// Maximum submitted but not emitted query+target bases.
    #[arg(long, default_value_t = 64_000_000)]
    max_in_flight_bases: usize,
    /// Stop after the first record-scoped error.
    #[arg(long, conflicts_with = "errors")]
    fail_fast: bool,
    /// Write recoverable errors as JSONL and continue non-JSONL output.
    #[arg(long)]
    errors: Option<PathBuf>,
    /// Parent directory for transient sequence and SAM stores.
    #[arg(long)]
    temp_dir: Option<PathBuf>,
}

impl BatchArgs {
    fn options(&self, jsonl: bool) -> Result<BatchOptions, String> {
        let defaults = BatchOptions::default();
        let threads = self.threads.unwrap_or(defaults.threads);
        let options = BatchOptions {
            threads,
            max_in_flight: self
                .max_in_flight
                .unwrap_or_else(|| threads.saturating_mul(2).max(1)),
            max_in_flight_bases: self.max_in_flight_bases,
            fail_fast: self.fail_fast || (!jsonl && self.errors.is_none()),
        };
        options.validate()?;
        if let Some(parent) = &self.temp_dir {
            if !parent.is_dir() {
                return Err(format!(
                    "--temp-dir {} is not a directory",
                    parent.display()
                ));
            }
        }
        Ok(options)
    }
}

#[derive(Clone, Args)]
struct LinearScoringArgs {
    #[arg(long = "match", default_value_t = 2)]
    match_score: i32,
    #[arg(long, default_value_t = 1)]
    mismatch_penalty: i32,
    #[arg(long, allow_hyphen_values = true, default_value_t = -2)]
    gap: i32,
    #[arg(long, value_enum)]
    matrix: Option<NamedMatrix>,
}

#[derive(Clone, Args)]
struct AffineScoringArgs {
    #[arg(long = "match", default_value_t = 2)]
    match_score: i32,
    #[arg(long, default_value_t = 1)]
    mismatch_penalty: i32,
    #[arg(long, allow_hyphen_values = true, default_value_t = -3)]
    gap_open: i32,
    #[arg(long, allow_hyphen_values = true, default_value_t = -1)]
    gap_extend: i32,
    #[arg(long, value_enum)]
    matrix: Option<NamedMatrix>,
}

#[derive(Args)]
struct VerifyCommand {
    /// Alignment records, or '-' for stdin.
    #[arg(long)]
    alignment: PathBuf,
    #[arg(long, value_enum)]
    format: InputFormat,
    #[command(flatten)]
    input: InputArgs,
    #[command(flatten)]
    batch: BatchArgs,
    /// Required for PAF/SAM; ignored when native JSON embeds a problem.
    #[arg(long, value_enum)]
    mode: Option<ModeArg>,
    #[command(flatten)]
    scoring: VerificationScoring,
    #[arg(long, value_enum, default_value_t = LevelArg::Optimal)]
    level: LevelArg,
    /// Maximum reference DP cells; 0 means unlimited.
    #[arg(long, default_value_t = DEFAULT_MAX_CELLS)]
    max_cells: u64,
    #[arg(long, value_enum, default_value_t = ReportFormat::Text)]
    output_format: ReportFormat,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Args)]
struct CompareCommand {
    #[arg(long)]
    left: PathBuf,
    #[arg(long, value_enum)]
    left_format: InputFormat,
    #[arg(long)]
    right: PathBuf,
    #[arg(long, value_enum)]
    right_format: InputFormat,
    #[command(flatten)]
    input: InputArgs,
    #[command(flatten)]
    batch: BatchArgs,
    /// Required when either input is PAF or SAM.
    #[arg(long, value_enum)]
    mode: Option<ModeArg>,
    #[command(flatten)]
    scoring: VerificationScoring,
    #[arg(long, value_enum, default_value_t = ReportFormat::Text)]
    output_format: ReportFormat,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Clone, Args)]
struct VerificationScoring {
    #[arg(long = "match")]
    match_score: Option<i32>,
    #[arg(long)]
    mismatch_penalty: Option<i32>,
    #[arg(long, allow_hyphen_values = true)]
    gap: Option<i32>,
    #[arg(long, allow_hyphen_values = true)]
    gap_open: Option<i32>,
    #[arg(long, allow_hyphen_values = true)]
    gap_extend: Option<i32>,
    #[arg(long, value_enum)]
    matrix: Option<NamedMatrix>,
}

#[derive(Clone, Copy, ValueEnum)]
enum NamedMatrix {
    Blosum62,
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
enum ScopeArg {
    Alignment,
    Score,
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
enum OutputFormat {
    Text,
    Json,
    Jsonl,
    Paf,
    Sam,
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
enum InputFormat {
    Json,
    Jsonl,
    Paf,
    Sam,
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
enum ReportFormat {
    Text,
    Json,
    Jsonl,
}

#[derive(Clone, Copy, ValueEnum)]
enum LevelArg {
    Structure,
    Score,
    Optimal,
}

#[derive(Clone, Copy, ValueEnum)]
enum ModeArg {
    EditDistance,
    GlobalLinear,
    GlobalAffine,
    LocalLinear,
    SemiglobalLinear,
}

#[derive(Clone)]
struct SolveJob {
    request: SolveRequest,
}

struct PairStream {
    queries: SequenceSource,
    targets: SequenceSource,
    problem: ProblemSpec,
    options: SolveOptions,
    index: usize,
    finished: bool,
}

impl Iterator for PairStream {
    type Item = Result<SolveJob, ErrorRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        let query = self.queries.next();
        let target = self.targets.next();
        match (query, target) {
            (None, None) => {
                self.finished = true;
                None
            }
            (Some(Ok(query)), Some(Ok(target))) => {
                let index = self.index;
                self.index += 1;
                Some(Ok(SolveJob {
                    request: SolveRequest {
                        pair_index: index,
                        query_id: query.id,
                        target_id: target.id,
                        query: query.sequence,
                        target: target.sequence,
                        problem: self.problem.clone(),
                        options: self.options.clone(),
                    },
                }))
            }
            (Some(Err(error)), Some(_)) => Some(Err(self.sequence_error(error, "query"))),
            (Some(_), Some(Err(error))) => Some(Err(self.sequence_error(error, "target"))),
            (Some(_), None) | (None, Some(_)) => {
                self.finished = true;
                let mut error = ErrorRecord::new(
                    ErrorStage::InputParse,
                    "pair_count_mismatch",
                    format!("query and target sources differ at pair {}", self.index),
                    ErrorContext {
                        pair_index: Some(self.index),
                        ..ErrorContext::default()
                    },
                );
                error.error.recoverable = false;
                Some(Err(error))
            }
        }
    }
}

impl PairStream {
    fn sequence_error(&mut self, error: SequenceReadError, label: &str) -> ErrorRecord {
        if !error.recoverable {
            self.finished = true;
        }
        let mut record = ErrorRecord::new(
            ErrorStage::InputParse,
            "invalid_sequence_record",
            format!("{label}: {}", error.message),
            ErrorContext {
                pair_index: Some(self.index),
                source_line: Some(error.line),
                ..ErrorContext::default()
            },
        );
        record.error.recoverable = error.recoverable;
        self.index += 1;
        record
    }
}

fn run_solve(command: SolveCommand) -> Result<u8, String> {
    let (mode, common, scoring) = match command.mode {
        SolveMode::EditDistance(args) => (
            AlignmentMode::EditDistance,
            args.common,
            ScoringScheme::EditDistance,
        ),
        SolveMode::GlobalLinear(args) => (
            AlignmentMode::GlobalLinear,
            args.common,
            linear_scoring(&args.scoring),
        ),
        SolveMode::GlobalAffine(args) => (
            AlignmentMode::GlobalAffine,
            args.common,
            affine_scoring(&args.scoring),
        ),
        SolveMode::LocalLinear(args) => (
            AlignmentMode::LocalLinear,
            args.common,
            linear_scoring(&args.scoring),
        ),
        SolveMode::SemiglobalLinear(args) => (
            AlignmentMode::SemiglobalLinear,
            args.common,
            linear_scoring(&args.scoring),
        ),
    };
    if common.scope == ScopeArg::Score
        && matches!(common.output.format, OutputFormat::Paf | OutputFormat::Sam)
    {
        return Err("PAF and SAM require --scope alignment".to_string());
    }
    validate_backend(mode, common.scope, &common.backend)?;
    let batch = common
        .batch
        .options(common.output.format == OutputFormat::Jsonl)?;
    let queries = open_sequence_source(&common.input.query, &common.input.query_file, "query")?;
    let targets = open_sequence_source(&common.input.target, &common.input.target_file, "target")?;
    let stream = PairStream {
        queries,
        targets,
        problem: ProblemSpec { mode, scoring },
        options: SolveOptions {
            scope: match common.scope {
                ScopeArg::Alignment => SolveScope::Alignment,
                ScopeArg::Score => SolveScope::Score,
            },
            backend: common.backend,
            block_size: common.block_size,
            band: common.band,
            cross_check: common.cross_check,
            include_operations: common.operations,
            integrity: common.integrity,
        },
        index: 0,
        finished: false,
    };
    let include_query = common.output.format == OutputFormat::Sam;
    let mut output = SolveOutput::open(
        common.output.format,
        common.output.output.as_deref(),
        common.batch.temp_dir.as_deref(),
    )?;
    let mut errors = ErrorSidecar::open(common.batch.errors.as_deref())?;
    let mut exit = ExitState::default();
    run_ordered(
        stream,
        &batch,
        |job| {
            job.request
                .query
                .len()
                .saturating_add(job.request.target.len())
        },
        move |job| {
            let context = ErrorContext {
                pair_index: Some(job.request.pair_index),
                query_id: Some(job.request.query_id.clone()),
                target_id: Some(job.request.target_id.clone()),
                source_line: None,
            };
            let query = include_query.then(|| job.request.query.clone());
            solve(job.request)
                .map(|record| (record, query))
                .map_err(|message| {
                    ErrorRecord::new(ErrorStage::Solve, "solve_failed", message, context)
                })
        },
        |item| match item {
            BatchItem::Record((record, query)) => {
                exit.observe_alignment(&record);
                output.push(&record, query.as_deref())
            }
            BatchItem::Error(error) => {
                exit.had_error = true;
                emit_error(
                    &error,
                    common.output.format == OutputFormat::Jsonl,
                    batch.fail_fast,
                    &mut output,
                    &mut errors,
                )
            }
        },
    )?;
    output.finish()?;
    errors.finish()?;
    Ok(exit.code())
}

fn validate_backend(mode: AlignmentMode, scope: ScopeArg, backend: &str) -> Result<(), String> {
    let allowed: &[&str] = match scope {
        ScopeArg::Score => &["auto"],
        ScopeArg::Alignment => match mode {
            AlignmentMode::EditDistance => &["auto", "adaptive-banded"],
            AlignmentMode::GlobalAffine => &["auto", "hcp", "banded-affine"],
            AlignmentMode::GlobalLinear
            | AlignmentMode::LocalLinear
            | AlignmentMode::SemiglobalLinear => &["auto", "hcp"],
        },
    };
    if allowed.contains(&backend) {
        Ok(())
    } else {
        Err(format!(
            "backend '{backend}' is not supported for {mode}; choose {}",
            allowed.join(", ")
        ))
    }
}

fn build_source_store(input: &InputArgs, temp_dir: Option<&Path>) -> Result<SequenceStore, String> {
    let store = SequenceStore::create(temp_dir)?;
    store.ingest(
        SequenceKind::Query,
        open_sequence_source(&input.query, &input.query_file, "query")?,
    )?;
    store.ingest(
        SequenceKind::Target,
        open_sequence_source(&input.target, &input.target_file, "target")?,
    )?;
    Ok(store)
}

fn run_verify(command: VerifyCommand) -> Result<u8, String> {
    let batch = command
        .batch
        .options(command.output_format == ReportFormat::Jsonl)?;
    let external_problem = if matches!(command.format, InputFormat::Paf | InputFormat::Sam) {
        problem_from_cli(command.mode, &command.scoring)?
    } else {
        None
    };
    let store = build_source_store(&command.input, command.batch.temp_dir.as_deref())?;
    let reader = AlignmentRecordReader::open(
        &command.alignment,
        command.format.into(),
        external_problem.as_ref(),
        &store,
    )?;
    let level: VerificationLevel = command.level.into();
    let max_cells = command.max_cells;
    let mut output = ReportOutput::open(command.output_format, command.output.as_deref())?;
    let mut errors = ErrorSidecar::open(command.batch.errors.as_deref())?;
    let mut exit = ExitState::default();
    run_ordered(
        reader,
        &batch,
        |job| job.query.len().saturating_add(job.target.len()),
        move |job: ResolvedAlignment| {
            Ok(verify(VerificationRequest {
                record: job.record,
                query: job.query,
                target: job.target,
                level,
                max_cells,
            }))
        },
        |item| match item {
            BatchItem::Record(report) => {
                exit.observe_verification(&report);
                output.push_verification(&report)
            }
            BatchItem::Error(error) => {
                exit.had_error = true;
                emit_report_error(
                    &error,
                    command.output_format == ReportFormat::Jsonl,
                    batch.fail_fast,
                    &mut output,
                    &mut errors,
                )
            }
        },
    )?;
    output.finish()?;
    errors.finish()?;
    Ok(exit.code())
}

#[derive(Clone)]
struct ComparisonJob {
    left: AlignmentRecord,
    right: AlignmentRecord,
    query: Vec<u8>,
    target: Vec<u8>,
}

fn run_compare(command: CompareCommand) -> Result<u8, String> {
    let batch = command
        .batch
        .options(command.output_format == ReportFormat::Jsonl)?;
    let needs_problem = matches!(command.left_format, InputFormat::Paf | InputFormat::Sam)
        || matches!(command.right_format, InputFormat::Paf | InputFormat::Sam);
    let problem = if needs_problem {
        problem_from_cli(command.mode, &command.scoring)?
    } else {
        None
    };
    let store = build_source_store(&command.input, command.batch.temp_dir.as_deref())?;
    let mut left = AlignmentRecordReader::open(
        &command.left,
        command.left_format.into(),
        problem.as_ref(),
        &store,
    )?;
    let mut right = AlignmentRecordReader::open(
        &command.right,
        command.right_format.into(),
        problem.as_ref(),
        &store,
    )?;
    let mut index = 0usize;
    let mut done = false;
    let pairs = std::iter::from_fn(move || {
        if done {
            return None;
        }
        let left_item = left.next();
        let right_item = right.next();
        let result = match (left_item, right_item) {
            (None, None) => {
                done = true;
                return None;
            }
            (Some(Ok(left)), Some(Ok(right))) => Ok(ComparisonJob {
                left: left.record,
                right: right.record,
                query: left.query,
                target: left.target,
            }),
            (Some(Err(error)), Some(_)) | (Some(_), Some(Err(error))) => Err(error),
            (Some(_), None) | (None, Some(_)) => {
                done = true;
                let mut error = ErrorRecord::new(
                    ErrorStage::RecordParse,
                    "comparison_count_mismatch",
                    "comparison inputs have different record counts",
                    ErrorContext {
                        pair_index: Some(index),
                        ..ErrorContext::default()
                    },
                );
                error.error.recoverable = false;
                Err(error)
            }
        };
        index += 1;
        Some(result)
    });
    let mut output = ReportOutput::open(command.output_format, command.output.as_deref())?;
    let mut errors = ErrorSidecar::open(command.batch.errors.as_deref())?;
    let mut exit = ExitState::default();
    run_ordered(
        pairs,
        &batch,
        |job| job.query.len().saturating_add(job.target.len()),
        |job: ComparisonJob| {
            let context = ErrorContext {
                pair_index: Some(job.left.pair.index),
                query_id: Some(job.left.pair.query.id.clone()),
                target_id: Some(job.left.pair.target.id.clone()),
                source_line: None,
            };
            compare(job.left, job.right, &job.query, &job.target).map_err(|message| {
                ErrorRecord::new(ErrorStage::Compare, "comparison_failed", message, context)
            })
        },
        |item| match item {
            BatchItem::Record(report) => {
                exit.observe_comparison(&report);
                output.push_comparison(&report)
            }
            BatchItem::Error(error) => {
                exit.had_error = true;
                emit_report_error(
                    &error,
                    command.output_format == ReportFormat::Jsonl,
                    batch.fail_fast,
                    &mut output,
                    &mut errors,
                )
            }
        },
    )?;
    output.finish()?;
    errors.finish()?;
    Ok(exit.code())
}

fn problem_from_cli(
    mode: Option<ModeArg>,
    args: &VerificationScoring,
) -> Result<Option<ProblemSpec>, String> {
    let Some(mode) = mode else {
        return Ok(None);
    };
    let mode: AlignmentMode = mode.into();
    let scoring = match mode {
        AlignmentMode::EditDistance => ScoringScheme::EditDistance,
        AlignmentMode::GlobalAffine => ScoringScheme::Affine {
            substitution: verification_substitution(args)?,
            gap_open: args
                .gap_open
                .ok_or_else(|| "--gap-open is required for global-affine PAF/SAM".to_string())?,
            gap_extend: args
                .gap_extend
                .ok_or_else(|| "--gap-extend is required for global-affine PAF/SAM".to_string())?,
        },
        _ => ScoringScheme::Linear {
            substitution: verification_substitution(args)?,
            gap: args
                .gap
                .ok_or_else(|| "--gap is required for linear-mode PAF/SAM".to_string())?,
        },
    };
    Ok(Some(ProblemSpec { mode, scoring }))
}

fn verification_substitution(args: &VerificationScoring) -> Result<SubstitutionScheme, String> {
    match args.matrix {
        Some(NamedMatrix::Blosum62) => Ok(SubstitutionScheme::Blosum62),
        None => Ok(SubstitutionScheme::MatchMismatch {
            match_score: args.match_score.ok_or_else(|| {
                "--match is required for match/mismatch PAF/SAM verification".to_string()
            })?,
            mismatch_penalty: args.mismatch_penalty.ok_or_else(|| {
                "--mismatch-penalty is required for match/mismatch PAF/SAM verification".to_string()
            })?,
        }),
    }
}

fn linear_scoring(args: &LinearScoringArgs) -> ScoringScheme {
    ScoringScheme::Linear {
        substitution: substitution(args.matrix, args.match_score, args.mismatch_penalty),
        gap: args.gap,
    }
}

fn affine_scoring(args: &AffineScoringArgs) -> ScoringScheme {
    ScoringScheme::Affine {
        substitution: substitution(args.matrix, args.match_score, args.mismatch_penalty),
        gap_open: args.gap_open,
        gap_extend: args.gap_extend,
    }
}

fn substitution(
    matrix: Option<NamedMatrix>,
    match_score: i32,
    mismatch_penalty: i32,
) -> SubstitutionScheme {
    match matrix {
        Some(NamedMatrix::Blosum62) => SubstitutionScheme::Blosum62,
        None => SubstitutionScheme::MatchMismatch {
            match_score,
            mismatch_penalty,
        },
    }
}

fn open_writer(path: Option<&Path>) -> Result<Box<dyn Write>, String> {
    match path {
        Some(path) => File::create(path)
            .map(|file| Box::new(BufWriter::new(file)) as Box<dyn Write>)
            .map_err(|err| format!("failed to create {}: {err}", path.display())),
        None => Ok(Box::new(BufWriter::new(io::stdout()))),
    }
}

struct JsonStream {
    writer: Box<dyn Write>,
    jsonl: bool,
    first: Option<Vec<u8>>,
    array_started: bool,
}

impl JsonStream {
    fn new(writer: Box<dyn Write>, jsonl: bool) -> Self {
        Self {
            writer,
            jsonl,
            first: None,
            array_started: false,
        }
    }

    fn push<T: Serialize>(&mut self, value: &T) -> Result<(), String> {
        let bytes = serde_json::to_vec(value).map_err(|err| err.to_string())?;
        if self.jsonl {
            self.writer
                .write_all(&bytes)
                .and_then(|_| self.writer.write_all(b"\n"))
                .map_err(|err| format!("failed to write JSONL output: {err}"))?;
            self.writer
                .flush()
                .map_err(|err| format!("failed to flush JSONL output: {err}"))?;
        } else if let Some(first) = self.first.take() {
            self.writer
                .write_all(b"[\n")
                .and_then(|_| self.writer.write_all(&first))
                .and_then(|_| self.writer.write_all(b",\n"))
                .and_then(|_| self.writer.write_all(&bytes))
                .map_err(|err| format!("failed to write JSON output: {err}"))?;
            self.array_started = true;
        } else if self.array_started {
            self.writer
                .write_all(b",\n")
                .and_then(|_| self.writer.write_all(&bytes))
                .map_err(|err| format!("failed to write JSON output: {err}"))?;
        } else {
            self.first = Some(bytes);
        }
        Ok(())
    }

    fn finish(mut self) -> Result<(), String> {
        if !self.jsonl {
            if let Some(first) = self.first.take() {
                self.writer
                    .write_all(&first)
                    .and_then(|_| self.writer.write_all(b"\n"))
                    .map_err(|err| format!("failed to finish JSON output: {err}"))?;
            } else if self.array_started {
                self.writer
                    .write_all(b"\n]\n")
                    .map_err(|err| format!("failed to finish JSON output: {err}"))?;
            } else {
                self.writer
                    .write_all(b"[]\n")
                    .map_err(|err| format!("failed to finish JSON output: {err}"))?;
            }
        }
        self.writer
            .flush()
            .map_err(|err| format!("failed to flush output: {err}"))
    }
}

enum SolveOutput {
    Json(JsonStream),
    Raw {
        format: OutputFormat,
        writer: Box<dyn Write>,
    },
    Sam {
        spool: SamSpool,
        writer: Box<dyn Write>,
    },
}

impl SolveOutput {
    fn open(
        format: OutputFormat,
        path: Option<&Path>,
        temp_dir: Option<&Path>,
    ) -> Result<Self, String> {
        match format {
            OutputFormat::Json => Ok(Self::Json(JsonStream::new(open_writer(path)?, false))),
            OutputFormat::Jsonl => Ok(Self::Json(JsonStream::new(open_writer(path)?, true))),
            OutputFormat::Sam => Ok(Self::Sam {
                spool: SamSpool::create(temp_dir)?,
                writer: open_writer(path)?,
            }),
            format => Ok(Self::Raw {
                format,
                writer: open_writer(path)?,
            }),
        }
    }

    fn push(&mut self, record: &AlignmentRecord, query: Option<&[u8]>) -> Result<(), String> {
        match self {
            Self::Json(writer) => writer.push(record),
            Self::Raw { format, writer } => {
                let text = match format {
                    OutputFormat::Paf => record_to_paf(record)?,
                    OutputFormat::Text => alignment_text(record),
                    _ => unreachable!(),
                };
                writer
                    .write_all(text.as_bytes())
                    .map_err(|err| format!("failed to write output: {err}"))
            }
            Self::Sam { spool, .. } => spool.push(
                record,
                query.ok_or_else(|| "SAM output requires the query sequence".to_string())?,
            ),
        }
    }

    fn push_error(&mut self, error: &ErrorRecord) -> Result<(), String> {
        match self {
            Self::Json(writer) if writer.jsonl => writer.push(error),
            _ => Ok(()),
        }
    }

    fn finish(self) -> Result<(), String> {
        match self {
            Self::Json(writer) => writer.finish(),
            Self::Raw { mut writer, .. } => writer
                .flush()
                .map_err(|err| format!("failed to flush output: {err}")),
            Self::Sam { spool, mut writer } => {
                spool.finish(writer.as_mut())?;
                writer
                    .flush()
                    .map_err(|err| format!("failed to flush SAM output: {err}"))
            }
        }
    }
}

fn alignment_text(record: &AlignmentRecord) -> String {
    let mut output = format!(
        "{} vs {}\nmode: {}\nbackend: {}\nobjective: {}\n",
        record.pair.query.id,
        record.pair.target.id,
        record.problem.mode,
        record.producer.backend,
        record.alignment.objective.value,
    );
    if let Some(trace) = &record.alignment.trace {
        output.push_str(&format!(
            "query: [{}..{}), target: [{}..{})\ncigar: {}\n",
            trace.query_span.start,
            trace.query_span.end,
            trace.target_span.start,
            trace.target_span.end,
            trace.cigar,
        ));
    }
    output.push_str(&format!("status: {:?}\n\n", record.verification.status));
    output
}

enum ReportOutput {
    Json(JsonStream),
    Text(Box<dyn Write>),
}

impl ReportOutput {
    fn open(format: ReportFormat, path: Option<&Path>) -> Result<Self, String> {
        match format {
            ReportFormat::Json => Ok(Self::Json(JsonStream::new(open_writer(path)?, false))),
            ReportFormat::Jsonl => Ok(Self::Json(JsonStream::new(open_writer(path)?, true))),
            ReportFormat::Text => Ok(Self::Text(open_writer(path)?)),
        }
    }

    fn push_verification(&mut self, report: &VerificationReport) -> Result<(), String> {
        match self {
            Self::Json(writer) => writer.push(report),
            Self::Text(writer) => writer
                .write_all(verification_text(report).as_bytes())
                .map_err(|err| format!("failed to write report: {err}")),
        }
    }

    fn push_comparison(&mut self, report: &ComparisonReport) -> Result<(), String> {
        match self {
            Self::Json(writer) => writer.push(report),
            Self::Text(writer) => writer
                .write_all(comparison_text(report).as_bytes())
                .map_err(|err| format!("failed to write report: {err}")),
        }
    }

    fn push_error(&mut self, error: &ErrorRecord) -> Result<(), String> {
        match self {
            Self::Json(writer) if writer.jsonl => writer.push(error),
            _ => Ok(()),
        }
    }

    fn finish(self) -> Result<(), String> {
        match self {
            Self::Json(writer) => writer.finish(),
            Self::Text(mut writer) => writer
                .flush()
                .map_err(|err| format!("failed to flush report: {err}")),
        }
    }
}

fn verification_text(report: &VerificationReport) -> String {
    let mut text = format!(
        "{} vs {}: {:?} ({:?})\n",
        report.pair.query.id,
        report.pair.target.id,
        report.verification.status,
        report.verification.level,
    );
    for check in &report.verification.checks {
        text.push_str(&format!(
            "  {}: {}{}\n",
            check.name,
            if check.passed { "passed" } else { "failed" },
            check
                .detail
                .as_ref()
                .map_or(String::new(), |detail| format!(" ({detail})")),
        ));
    }
    if let Some(reason) = &report.verification.reason {
        text.push_str(&format!("  reason: {reason}\n"));
    }
    text
}

fn comparison_text(report: &ComparisonReport) -> String {
    format!(
        "{} vs {}: equivalent={} objectives_equal={} traces_equal={}\n{}",
        report.pair.query.id,
        report.pair.target.id,
        report.equivalent,
        report.objectives_equal,
        report.traces_equal,
        report
            .differences
            .iter()
            .map(|difference| format!("  {difference}\n"))
            .collect::<String>(),
    )
}

struct ErrorSidecar {
    writer: Option<BufWriter<File>>,
}

impl ErrorSidecar {
    fn open(path: Option<&Path>) -> Result<Self, String> {
        let writer = path
            .map(|path| {
                File::create(path).map(BufWriter::new).map_err(|err| {
                    format!("failed to create error sidecar {}: {err}", path.display())
                })
            })
            .transpose()?;
        Ok(Self { writer })
    }

    fn push(&mut self, error: &ErrorRecord) -> Result<bool, String> {
        let Some(writer) = self.writer.as_mut() else {
            return Ok(false);
        };
        serde_json::to_writer(&mut *writer, error).map_err(|err| err.to_string())?;
        writer
            .write_all(b"\n")
            .and_then(|_| writer.flush())
            .map_err(|err| format!("failed to write error sidecar: {err}"))?;
        Ok(true)
    }

    fn finish(mut self) -> Result<(), String> {
        if let Some(writer) = self.writer.as_mut() {
            writer
                .flush()
                .map_err(|err| format!("failed to flush error sidecar: {err}"))?;
        }
        Ok(())
    }
}

fn emit_error(
    error: &ErrorRecord,
    inline: bool,
    fail_fast: bool,
    output: &mut SolveOutput,
    sidecar: &mut ErrorSidecar,
) -> Result<(), String> {
    let redirected = sidecar.push(error)?;
    if inline && !redirected {
        output.push_error(error)?;
    }
    if fail_fast {
        return Err(error.error.message.clone());
    }
    Ok(())
}

fn emit_report_error(
    error: &ErrorRecord,
    inline: bool,
    fail_fast: bool,
    output: &mut ReportOutput,
    sidecar: &mut ErrorSidecar,
) -> Result<(), String> {
    let redirected = sidecar.push(error)?;
    if inline && !redirected {
        output.push_error(error)?;
    }
    if fail_fast {
        return Err(error.error.message.clone());
    }
    Ok(())
}

#[derive(Default)]
struct ExitState {
    had_error: bool,
    invalid: bool,
    inconclusive: bool,
}

impl ExitState {
    fn observe_alignment(&mut self, record: &AlignmentRecord) {
        self.invalid |= record.verification.status == VerificationStatus::Invalid;
        self.inconclusive |= record.verification.status == VerificationStatus::Inconclusive;
    }

    fn observe_verification(&mut self, report: &VerificationReport) {
        self.invalid |= report.verification.status == VerificationStatus::Invalid;
        self.inconclusive |= matches!(
            report.verification.status,
            VerificationStatus::Inconclusive | VerificationStatus::Error
        );
    }

    fn observe_comparison(&mut self, report: &ComparisonReport) {
        self.invalid |= !report.equivalent;
    }

    fn code(&self) -> u8 {
        if self.had_error {
            2
        } else if self.invalid {
            1
        } else if self.inconclusive {
            3
        } else {
            0
        }
    }
}

impl From<InputFormat> for AlignmentInputFormat {
    fn from(value: InputFormat) -> Self {
        match value {
            InputFormat::Json => Self::Json,
            InputFormat::Jsonl => Self::Jsonl,
            InputFormat::Paf => Self::Paf,
            InputFormat::Sam => Self::Sam,
        }
    }
}

impl From<ModeArg> for AlignmentMode {
    fn from(value: ModeArg) -> Self {
        match value {
            ModeArg::EditDistance => Self::EditDistance,
            ModeArg::GlobalLinear => Self::GlobalLinear,
            ModeArg::GlobalAffine => Self::GlobalAffine,
            ModeArg::LocalLinear => Self::LocalLinear,
            ModeArg::SemiglobalLinear => Self::SemiglobalLinear,
        }
    }
}

impl From<LevelArg> for VerificationLevel {
    fn from(value: LevelArg) -> Self {
        match value {
            LevelArg::Structure => Self::Structure,
            LevelArg::Score => Self::Score,
            LevelArg::Optimal => Self::Optimal,
        }
    }
}
