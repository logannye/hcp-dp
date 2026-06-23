use std::{
    fs,
    io::{self, IsTerminal, Write},
    path::PathBuf,
    process::ExitCode,
    time::Instant,
};

use clap::{Args, Parser, Subcommand, ValueEnum};
use hcp_dp::{
    alignment::{AlignmentOpKind, AlignmentTrace},
    problems::{
        edit_distance::EditDistanceProblem,
        nw_affine::{NwAffineProblem, NwAffineState},
        nw_align::NwProblem,
        semiglobal::{SemiGlobalCell, SemiGlobalProblem},
        smith_waterman::{SmithWatermanProblem, SwCell},
    },
    HcpEngine, HcpProblem, HcpRunStats,
};
use serde::Serialize;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "parallel")]
use std::sync::atomic::{AtomicUsize, Ordering};

#[path = "../sequence_io.rs"]
mod sequence_io;

use sequence_io::{read_sequence_records, SequenceRecord};

const SCHEMA_VERSION: &str = "hcp-align.v1";
const ENGINE_NAME: &str = "hcp-dp";

fn main() -> ExitCode {
    match run() {
        Ok(success) => {
            if success {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        Err(err) => {
            eprintln!("hcp-align: {err}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<bool, String> {
    let cli = Cli::parse();
    let (reports, output) = match cli.command {
        Command::GlobalLinear(args) => {
            let output = args.output.clone();
            (
                run_linear_mode(args, "global-linear", align_global_linear)?,
                output,
            )
        }
        Command::GlobalAffine(args) => {
            let output = args.output.clone();
            (run_affine_mode(args)?, output)
        }
        Command::LocalLinear(args) => {
            let output = args.output.clone();
            (
                run_linear_mode(args, "local-linear", align_local_linear)?,
                output,
            )
        }
        Command::EditDistance(args) => {
            let output = args.output.clone();
            (run_edit_distance_mode(args)?, output)
        }
        Command::SemiglobalLinear(args) => {
            let output = args.output.clone();
            (
                run_linear_mode(args, "semiglobal-linear", align_semiglobal_linear)?,
                output,
            )
        }
    };

    write_reports(&reports, &output)?;
    Ok(reports
        .iter()
        .all(|report| report.verification_status != VerificationStatus::Failed))
}

#[derive(Parser)]
#[command(name = "hcp-align")]
#[command(about = "Exact HCP-DP sequence alignment CLI")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Global Needleman-Wunsch alignment with a linear gap penalty.
    GlobalLinear(LinearCommand),
    /// Global Gotoh alignment with affine gap penalties.
    GlobalAffine(AffineCommand),
    /// Smith-Waterman local alignment with a linear gap penalty.
    LocalLinear(LinearCommand),
    /// Levenshtein edit distance.
    EditDistance(EditDistanceCommand),
    /// Semi-global linear alignment: full query against any target interval.
    SemiglobalLinear(LinearCommand),
}

#[derive(Args)]
struct LinearCommand {
    #[command(flatten)]
    input: InputArgs,
    #[command(flatten)]
    output: OutputArgs,
    #[command(flatten)]
    block: BlockArgs,
    #[command(flatten)]
    scoring: LinearScoring,
}

#[derive(Args)]
struct AffineCommand {
    #[command(flatten)]
    input: InputArgs,
    #[command(flatten)]
    output: OutputArgs,
    #[command(flatten)]
    block: BlockArgs,
    #[command(flatten)]
    scoring: AffineScoring,
}

#[derive(Args)]
struct EditDistanceCommand {
    #[command(flatten)]
    input: InputArgs,
    #[command(flatten)]
    output: OutputArgs,
    #[command(flatten)]
    block: BlockArgs,
}

#[derive(Args)]
struct InputArgs {
    /// Raw query sequence.
    #[arg(long)]
    query: Option<String>,
    /// Query file containing raw sequence, FASTA, or FASTQ.
    #[arg(long)]
    query_file: Option<PathBuf>,
    /// Raw target sequence.
    #[arg(long)]
    target: Option<String>,
    /// Target file containing raw sequence, FASTA, or FASTQ.
    #[arg(long)]
    target_file: Option<PathBuf>,
}

#[derive(Clone, Args)]
struct OutputArgs {
    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    format: OutputFormat,
    /// Write output to this path instead of stdout.
    #[arg(long = "output")]
    output_path: Option<PathBuf>,
    /// Include aligned query and target strings.
    #[arg(long)]
    show_alignment: bool,
    /// Control operation detail in JSON/JSONL output.
    #[arg(long, value_enum, default_value_t = OperationDetail::Summary)]
    operation_detail: OperationDetail,
    /// Check the HCP result against a full-table baseline when within --verify-limit.
    #[arg(long)]
    verify: bool,
    /// Largest sequence length eligible for full-table --verify. Use 0 for no limit.
    #[arg(long, default_value_t = 2048)]
    verify_limit: usize,
    /// Pairwise batch threads. Values above 1 require the `parallel` feature.
    #[arg(long, default_value_t = 1)]
    threads: usize,
    /// Emit failed pair records and continue after per-pair errors.
    #[arg(long)]
    continue_on_error: bool,
    /// Progress reporting mode. Progress is always written to stderr.
    #[arg(long, value_enum, default_value_t = ProgressMode::Auto)]
    progress: ProgressMode,
}

#[derive(Clone, Copy, Args)]
struct BlockArgs {
    /// Override the HCP summary-tree block size.
    #[arg(long)]
    block_size: Option<usize>,
}

#[derive(Clone, Copy, Args)]
struct LinearScoring {
    #[arg(long = "match", default_value_t = 2)]
    match_score: i32,
    #[arg(long, default_value_t = 1)]
    mismatch_penalty: i32,
    #[arg(long, allow_hyphen_values = true, default_value_t = -2)]
    gap: i32,
}

#[derive(Clone, Copy, Args)]
struct AffineScoring {
    #[arg(long = "match", default_value_t = 2)]
    match_score: i32,
    #[arg(long, default_value_t = 1)]
    mismatch_penalty: i32,
    #[arg(long, allow_hyphen_values = true, default_value_t = -3)]
    gap_open: i32,
    #[arg(long, allow_hyphen_values = true, default_value_t = -1)]
    gap_extend: i32,
}

#[derive(Clone, Copy, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
    Jsonl,
    Tsv,
    Cigar,
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
enum OperationDetail {
    None,
    Summary,
    Full,
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
enum ProgressMode {
    Auto,
    Always,
    Never,
}

#[derive(Clone)]
struct PairInput {
    pair_index: usize,
    query: SequenceRecord,
    target: SequenceRecord,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum VerificationStatus {
    Full,
    PathOnly,
    Failed,
}

impl VerificationStatus {
    fn label(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::PathOnly => "path_only",
            Self::Failed => "failed",
        }
    }
}

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    engine: &'static str,
    pair_index: usize,
    query_id: String,
    target_id: String,
    mode: &'static str,
    score: Option<i32>,
    distance: Option<u32>,
    path_score: Option<i64>,
    verification_status: VerificationStatus,
    verified: Option<bool>,
    query_start: usize,
    query_end: usize,
    target_start: usize,
    target_end: usize,
    cigar: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    operation_counts: Option<Vec<OperationCount>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    operations: Option<Vec<hcp_dp::alignment::AlignmentStep>>,
    block_size: usize,
    path_length: usize,
    summary_build_ms: f64,
    reconstruction_ms: f64,
    verification_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    aligned_query: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    aligned_target: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    elapsed_ms: f64,
}

struct VerificationResult {
    path_score: Option<i64>,
    status: VerificationStatus,
    verified: Option<bool>,
    verification_ms: f64,
}

#[derive(Serialize)]
struct OperationCount {
    op: AlignmentOpKind,
    count: usize,
}

struct ReportObjective {
    score: Option<i32>,
    distance: Option<u32>,
}

struct ReportMetrics {
    objective: ReportObjective,
    verification: VerificationResult,
    block_size: usize,
    path_length: usize,
    stats: HcpRunStats,
    elapsed_ms: f64,
    operation_detail: OperationDetail,
}

impl ReportObjective {
    fn score(value: i32) -> Self {
        Self {
            score: Some(value),
            distance: None,
        }
    }

    fn distance(value: u32) -> Self {
        Self {
            score: None,
            distance: Some(value),
        }
    }
}

fn run_linear_mode<F>(
    args: LinearCommand,
    mode: &'static str,
    align: F,
) -> Result<Vec<Report>, String>
where
    F: Fn(
            &PairInput,
            LinearScoring,
            &OutputArgs,
            BlockArgs,
            &'static str,
        ) -> Result<Report, String>
        + Sync,
{
    let pairs = read_pairs(&args.input)?;
    run_pairs(&pairs, &args.output, mode, |pair| {
        align(pair, args.scoring, &args.output, args.block, mode)
    })
}

fn run_affine_mode(args: AffineCommand) -> Result<Vec<Report>, String> {
    let pairs = read_pairs(&args.input)?;
    run_pairs(&pairs, &args.output, "global-affine", |pair| {
        align_global_affine(pair, args.scoring, &args.output, args.block)
    })
}

fn run_edit_distance_mode(args: EditDistanceCommand) -> Result<Vec<Report>, String> {
    let pairs = read_pairs(&args.input)?;
    run_pairs(&pairs, &args.output, "edit-distance", |pair| {
        align_edit_distance(pair, &args.output, args.block)
    })
}

fn run_pairs<F>(
    pairs: &[PairInput],
    output: &OutputArgs,
    mode: &'static str,
    align: F,
) -> Result<Vec<Report>, String>
where
    F: Fn(&PairInput) -> Result<Report, String> + Sync,
{
    validate_execution_args(output)?;
    let progress = progress_enabled(output, pairs.len());

    #[cfg(feature = "parallel")]
    if output.threads > 1 && pairs.len() > 1 {
        let completed = AtomicUsize::new(0);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(output.threads)
            .build()
            .map_err(|err| err.to_string())?;
        let results: Vec<Result<Report, String>> = pool.install(|| {
            pairs
                .par_iter()
                .map(|pair| {
                    let result = align(pair);
                    if progress {
                        let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                        emit_progress(done, pairs.len(), pair, result.as_ref().ok());
                    }
                    match result {
                        Ok(report) => Ok(report),
                        Err(err) if output.continue_on_error => Ok(error_report(pair, mode, err)),
                        Err(err) => Err(err),
                    }
                })
                .collect()
        });
        return results.into_iter().collect();
    }

    let mut reports = Vec::with_capacity(pairs.len());
    for pair in pairs {
        match align(pair) {
            Ok(report) => {
                if progress {
                    emit_progress(reports.len() + 1, pairs.len(), pair, Some(&report));
                }
                reports.push(report);
            }
            Err(err) if output.continue_on_error => {
                let report = error_report(pair, mode, err);
                if progress {
                    emit_progress(reports.len() + 1, pairs.len(), pair, Some(&report));
                }
                reports.push(report);
            }
            Err(err) => return Err(err),
        }
    }
    Ok(reports)
}

fn validate_execution_args(output: &OutputArgs) -> Result<(), String> {
    if output.threads == 0 {
        return Err("--threads must be at least 1".to_string());
    }
    #[cfg(not(feature = "parallel"))]
    if output.threads != 1 {
        return Err(
            "--threads greater than 1 requires building with the `parallel` feature".to_string(),
        );
    }
    Ok(())
}

fn progress_enabled(output: &OutputArgs, pair_count: usize) -> bool {
    match output.progress {
        ProgressMode::Always => true,
        ProgressMode::Never => false,
        ProgressMode::Auto => pair_count > 1 && io::stderr().is_terminal(),
    }
}

fn emit_progress(done: usize, total: usize, pair: &PairInput, report: Option<&Report>) {
    let status = report
        .map(|report| report.verification_status.label())
        .unwrap_or("error");
    let _ = writeln!(
        io::stderr(),
        "hcp-align: {done}/{total} {} vs {} {status}",
        pair.query.id,
        pair.target.id
    );
}

fn read_pairs(args: &InputArgs) -> Result<Vec<PairInput>, String> {
    let query_records = read_sequence_records(&args.query, &args.query_file, "query")?;
    let target_records = read_sequence_records(&args.target, &args.target_file, "target")?;
    if query_records.len() != target_records.len() {
        return Err(format!(
            "pairwise batch mode requires equal record counts; query has {}, target has {}",
            query_records.len(),
            target_records.len()
        ));
    }

    Ok(query_records
        .into_iter()
        .zip(target_records)
        .enumerate()
        .map(|(idx, (query, target))| PairInput {
            pair_index: idx,
            query,
            target,
        })
        .collect())
}

fn align_global_linear(
    pair: &PairInput,
    scoring: LinearScoring,
    output: &OutputArgs,
    block: BlockArgs,
    mode: &'static str,
) -> Result<Report, String> {
    let start = Instant::now();
    let problem = NwProblem::new(
        &pair.query.sequence,
        &pair.target.sequence,
        scoring.match_score,
        scoring.mismatch_penalty,
        scoring.gap,
    );
    let block_size = block_size_for(&problem, block.block_size)?;
    let (score, path, stats) = run_engine(problem.clone(), block.block_size);
    let verify_start = Instant::now();
    let mut verification = verify_i32(output, pair, problem.score_path(&path), score, || {
        problem.full_table_score()
    });
    verification.verification_ms = verify_start.elapsed().as_secs_f64() * 1000.0;
    let trace = AlignmentTrace::from_cells(
        &pair.query.sequence,
        &pair.target.sequence,
        &path,
        output.show_alignment,
    );
    Ok(report_from_trace(
        pair,
        mode,
        trace,
        ReportMetrics {
            objective: ReportObjective::score(score),
            verification,
            block_size,
            path_length: path.len(),
            stats,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            operation_detail: output.operation_detail,
        },
    ))
}

fn align_global_affine(
    pair: &PairInput,
    scoring: AffineScoring,
    output: &OutputArgs,
    block: BlockArgs,
) -> Result<Report, String> {
    let start = Instant::now();
    let problem = NwAffineProblem::new(
        &pair.query.sequence,
        &pair.target.sequence,
        scoring.match_score,
        scoring.mismatch_penalty,
        scoring.gap_open,
        scoring.gap_extend,
    );
    let block_size = block_size_for(&problem, block.block_size)?;
    let (score, path, stats) = run_engine(problem.clone(), block.block_size);
    let verify_start = Instant::now();
    let mut verification = verify_i32(output, pair, problem.score_path(&path), score, || {
        problem.full_table_score()
    });
    verification.verification_ms = verify_start.elapsed().as_secs_f64() * 1000.0;
    let cells = affine_cells(&path);
    let trace = AlignmentTrace::from_cells(
        &pair.query.sequence,
        &pair.target.sequence,
        &cells,
        output.show_alignment,
    );
    Ok(report_from_trace(
        pair,
        "global-affine",
        trace,
        ReportMetrics {
            objective: ReportObjective::score(score),
            verification,
            block_size,
            path_length: path.len(),
            stats,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            operation_detail: output.operation_detail,
        },
    ))
}

fn align_local_linear(
    pair: &PairInput,
    scoring: LinearScoring,
    output: &OutputArgs,
    block: BlockArgs,
    mode: &'static str,
) -> Result<Report, String> {
    let start = Instant::now();
    let problem = SmithWatermanProblem::new(
        &pair.query.sequence,
        &pair.target.sequence,
        scoring.match_score,
        scoring.mismatch_penalty,
        scoring.gap,
    );
    let block_size = block_size_for(&problem, block.block_size)?;
    let (score, path, stats) = run_engine(problem.clone(), block.block_size);
    let verify_start = Instant::now();
    let mut verification = verify_i32(output, pair, problem.score_path(&path), score, || {
        problem.full_table_score()
    });
    verification.verification_ms = verify_start.elapsed().as_secs_f64() * 1000.0;
    let cells = sw_cells(&path);
    let trace = AlignmentTrace::from_cells(
        &pair.query.sequence,
        &pair.target.sequence,
        &cells,
        output.show_alignment,
    );
    Ok(report_from_trace(
        pair,
        mode,
        trace,
        ReportMetrics {
            objective: ReportObjective::score(score),
            verification,
            block_size,
            path_length: path.len(),
            stats,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            operation_detail: output.operation_detail,
        },
    ))
}

fn align_edit_distance(
    pair: &PairInput,
    output: &OutputArgs,
    block: BlockArgs,
) -> Result<Report, String> {
    let start = Instant::now();
    let problem = EditDistanceProblem::new(&pair.query.sequence, &pair.target.sequence);
    let block_size = block_size_for(&problem, block.block_size)?;
    let (distance, path, stats) = run_engine(problem.clone(), block.block_size);
    let verify_start = Instant::now();
    let mut verification = verify_u32(output, pair, problem.score_path(&path), distance, || {
        problem.full_table_distance()
    });
    verification.verification_ms = verify_start.elapsed().as_secs_f64() * 1000.0;
    let trace = AlignmentTrace::from_cells(
        &pair.query.sequence,
        &pair.target.sequence,
        &path,
        output.show_alignment,
    );
    Ok(report_from_trace(
        pair,
        "edit-distance",
        trace,
        ReportMetrics {
            objective: ReportObjective::distance(distance),
            verification,
            block_size,
            path_length: path.len(),
            stats,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            operation_detail: output.operation_detail,
        },
    ))
}

fn align_semiglobal_linear(
    pair: &PairInput,
    scoring: LinearScoring,
    output: &OutputArgs,
    block: BlockArgs,
    mode: &'static str,
) -> Result<Report, String> {
    let start = Instant::now();
    let problem = SemiGlobalProblem::new(
        &pair.query.sequence,
        &pair.target.sequence,
        scoring.match_score,
        scoring.mismatch_penalty,
        scoring.gap,
    );
    let block_size = block_size_for(&problem, block.block_size)?;
    let (score, path, stats) = run_engine(problem.clone(), block.block_size);
    let verify_start = Instant::now();
    let mut verification = verify_i32(output, pair, problem.score_path(&path), score, || {
        problem.full_table_score()
    });
    verification.verification_ms = verify_start.elapsed().as_secs_f64() * 1000.0;
    let cells = semiglobal_cells(&path);
    let trace = AlignmentTrace::from_cells(
        &pair.query.sequence,
        &pair.target.sequence,
        &cells,
        output.show_alignment,
    );
    Ok(report_from_trace(
        pair,
        mode,
        trace,
        ReportMetrics {
            objective: ReportObjective::score(score),
            verification,
            block_size,
            path_length: path.len(),
            stats,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            operation_detail: output.operation_detail,
        },
    ))
}

fn report_from_trace(
    pair: &PairInput,
    mode: &'static str,
    trace: AlignmentTrace,
    metrics: ReportMetrics,
) -> Report {
    let operation_counts = match metrics.operation_detail {
        OperationDetail::None => None,
        OperationDetail::Summary | OperationDetail::Full => {
            Some(operation_counts(&trace.operations))
        }
    };
    let operations = match metrics.operation_detail {
        OperationDetail::Full => Some(trace.operations),
        OperationDetail::None | OperationDetail::Summary => None,
    };
    Report {
        schema_version: SCHEMA_VERSION,
        engine: ENGINE_NAME,
        pair_index: pair.pair_index,
        query_id: pair.query.id.clone(),
        target_id: pair.target.id.clone(),
        mode,
        score: metrics.objective.score,
        distance: metrics.objective.distance,
        path_score: metrics.verification.path_score,
        verification_status: metrics.verification.status,
        verified: metrics.verification.verified,
        query_start: trace.query_start,
        query_end: trace.query_end,
        target_start: trace.target_start,
        target_end: trace.target_end,
        cigar: trace.cigar,
        operation_counts,
        operations,
        block_size: metrics.block_size,
        path_length: metrics.path_length,
        summary_build_ms: metrics.stats.summary_build_ms,
        reconstruction_ms: metrics.stats.reconstruction_ms,
        verification_ms: metrics.verification.verification_ms,
        aligned_query: trace.aligned_query,
        aligned_target: trace.aligned_target,
        error: None,
        elapsed_ms: metrics.elapsed_ms,
    }
}

fn operation_counts(operations: &[hcp_dp::alignment::AlignmentStep]) -> Vec<OperationCount> {
    let mut counts: Vec<OperationCount> = Vec::new();
    for operation in operations {
        if let Some(existing) = counts.iter_mut().find(|item| item.op == operation.op) {
            existing.count += 1;
        } else {
            counts.push(OperationCount {
                op: operation.op,
                count: 1,
            });
        }
    }
    counts
}

fn error_report(pair: &PairInput, mode: &'static str, error: String) -> Report {
    Report {
        schema_version: SCHEMA_VERSION,
        engine: ENGINE_NAME,
        pair_index: pair.pair_index,
        query_id: pair.query.id.clone(),
        target_id: pair.target.id.clone(),
        mode,
        score: None,
        distance: None,
        path_score: None,
        verification_status: VerificationStatus::Failed,
        verified: Some(false),
        query_start: 0,
        query_end: 0,
        target_start: 0,
        target_end: 0,
        cigar: String::new(),
        operation_counts: None,
        operations: None,
        block_size: 0,
        path_length: 0,
        summary_build_ms: 0.0,
        reconstruction_ms: 0.0,
        verification_ms: 0.0,
        aligned_query: None,
        aligned_target: None,
        error: Some(error),
        elapsed_ms: 0.0,
    }
}

fn verify_i32<F>(
    output: &OutputArgs,
    pair: &PairInput,
    path_score: Option<i32>,
    reported: i32,
    baseline: F,
) -> VerificationResult
where
    F: FnOnce() -> i32,
{
    let path_score_i64 = path_score.map(i64::from);
    if path_score != Some(reported) {
        return VerificationResult {
            path_score: path_score_i64,
            status: VerificationStatus::Failed,
            verified: full_verify_requested(output, pair).then_some(false),
            verification_ms: 0.0,
        };
    }
    if full_verify_requested(output, pair) {
        let passed = baseline() == reported;
        VerificationResult {
            path_score: path_score_i64,
            status: if passed {
                VerificationStatus::Full
            } else {
                VerificationStatus::Failed
            },
            verified: Some(passed),
            verification_ms: 0.0,
        }
    } else {
        VerificationResult {
            path_score: path_score_i64,
            status: VerificationStatus::PathOnly,
            verified: None,
            verification_ms: 0.0,
        }
    }
}

fn verify_u32<F>(
    output: &OutputArgs,
    pair: &PairInput,
    path_score: Option<u32>,
    reported: u32,
    baseline: F,
) -> VerificationResult
where
    F: FnOnce() -> u32,
{
    let path_score_i64 = path_score.map(i64::from);
    if path_score != Some(reported) {
        return VerificationResult {
            path_score: path_score_i64,
            status: VerificationStatus::Failed,
            verified: full_verify_requested(output, pair).then_some(false),
            verification_ms: 0.0,
        };
    }
    if full_verify_requested(output, pair) {
        let passed = baseline() == reported;
        VerificationResult {
            path_score: path_score_i64,
            status: if passed {
                VerificationStatus::Full
            } else {
                VerificationStatus::Failed
            },
            verified: Some(passed),
            verification_ms: 0.0,
        }
    } else {
        VerificationResult {
            path_score: path_score_i64,
            status: VerificationStatus::PathOnly,
            verified: None,
            verification_ms: 0.0,
        }
    }
}

fn full_verify_requested(output: &OutputArgs, pair: &PairInput) -> bool {
    if !output.verify {
        return false;
    }
    let max_len = pair.query.sequence.len().max(pair.target.sequence.len());
    output.verify_limit == 0 || max_len <= output.verify_limit
}

fn run_engine<P>(problem: P, block_size: Option<usize>) -> (P::Cost, Vec<P::State>, HcpRunStats)
where
    P: HcpProblem,
{
    if let Some(block_size) = block_size {
        HcpEngine::with_block_size(problem, block_size).run_with_stats()
    } else {
        HcpEngine::new(problem).run_with_stats()
    }
}

fn block_size_for<P: HcpProblem>(problem: &P, block_size: Option<usize>) -> Result<usize, String> {
    match block_size {
        Some(0) => Err("block size must be positive".to_string()),
        Some(value) => Ok(value),
        None => Ok(hcp_dp::utils::default_block_size(
            problem.num_layers().max(1),
        )),
    }
}

fn write_reports(reports: &[Report], output: &OutputArgs) -> Result<(), String> {
    let rendered = match output.format {
        OutputFormat::Json => json_reports(reports)?,
        OutputFormat::Jsonl => jsonl_reports(reports)?,
        OutputFormat::Text => text_reports(reports),
        OutputFormat::Tsv => tsv_reports(reports),
        OutputFormat::Cigar => cigar_reports(reports),
    };
    if let Some(path) = &output.output_path {
        fs::write(path, rendered)
            .map_err(|err| format!("failed to write {}: {err}", path.display()))
    } else {
        print!("{rendered}");
        Ok(())
    }
}

fn json_reports(reports: &[Report]) -> Result<String, String> {
    let mut json = if reports.len() == 1 {
        serde_json::to_string_pretty(&reports[0]).map_err(|err| err.to_string())?
    } else {
        serde_json::to_string_pretty(reports).map_err(|err| err.to_string())?
    };
    json.push('\n');
    Ok(json)
}

fn jsonl_reports(reports: &[Report]) -> Result<String, String> {
    let mut output = String::new();
    for report in reports {
        let json = serde_json::to_string(report).map_err(|err| err.to_string())?;
        output.push_str(&json);
        output.push('\n');
    }
    Ok(output)
}

fn text_reports(reports: &[Report]) -> String {
    let mut output = String::new();
    for report in reports {
        if reports.len() > 1 {
            output.push_str(&format!(
                "pair {}: {} vs {}",
                report.pair_index, report.query_id, report.target_id
            ));
            output.push('\n');
        }
        output.push_str(&format!("mode: {}\n", report.mode));
        if let Some(score) = report.score {
            output.push_str(&format!("score: {score}\n"));
        }
        if let Some(distance) = report.distance {
            output.push_str(&format!("distance: {distance}\n"));
        }
        if let Some(path_score) = report.path_score {
            output.push_str(&format!("path_score: {path_score}\n"));
        }
        output.push_str(&format!(
            "verification_status: {}\n",
            report.verification_status.label()
        ));
        match report.verified {
            Some(true) => output.push_str("verified: true\n"),
            Some(false) => output.push_str("verified: false\n"),
            None => output.push_str("verified: not_full\n"),
        }
        output.push_str(&format!(
            "query: {} [{}..{}), target: {} [{}..{})",
            report.query_id,
            report.query_start,
            report.query_end,
            report.target_id,
            report.target_start,
            report.target_end
        ));
        output.push('\n');
        output.push_str(&format!("cigar: {}\n", report.cigar));
        output.push_str(&format!("block_size: {}\n", report.block_size));
        output.push_str(&format!("path_length: {}\n", report.path_length));
        output.push_str(&format!(
            "summary_build_ms: {:.3}\n",
            report.summary_build_ms
        ));
        output.push_str(&format!(
            "reconstruction_ms: {:.3}\n",
            report.reconstruction_ms
        ));
        output.push_str(&format!("verification_ms: {:.3}\n", report.verification_ms));
        output.push_str(&format!("elapsed_ms: {:.3}\n", report.elapsed_ms));
        if let Some(error) = &report.error {
            output.push_str(&format!("error: {error}\n"));
        }
        if let Some(aligned_query) = &report.aligned_query {
            output.push_str(&format!("aligned_query: {aligned_query}\n"));
        }
        if let Some(aligned_target) = &report.aligned_target {
            output.push_str(&format!("aligned_target: {aligned_target}\n"));
        }
        if reports.len() > 1 {
            output.push('\n');
        }
    }
    output
}

fn tsv_reports(reports: &[Report]) -> String {
    let mut output = String::from(
        "pair_index\tquery_id\ttarget_id\tmode\tscore\tdistance\tpath_score\tverification_status\tquery_start\tquery_end\ttarget_start\ttarget_end\tcigar\tblock_size\tpath_length\tsummary_build_ms\treconstruction_ms\tverification_ms\telapsed_ms\n",
    );
    for report in reports {
        output.push_str(&tsv_row(report));
        output.push('\n');
    }
    output
}

fn cigar_reports(reports: &[Report]) -> String {
    let mut output = String::from(
        "pair_index\tquery_id\ttarget_id\tmode\tscore\tdistance\tpath_score\tverification_status\tquery_start\tquery_end\ttarget_start\ttarget_end\tcigar\n",
    );
    for report in reports {
        output.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            report.pair_index,
            escape_tab(&report.query_id),
            escape_tab(&report.target_id),
            report.mode,
            optional_i32(report.score),
            optional_u32(report.distance),
            optional_i64(report.path_score),
            report.verification_status.label(),
            report.query_start,
            report.query_end,
            report.target_start,
            report.target_end,
            escape_tab(&report.cigar)
        ));
        output.push('\n');
    }
    output
}

fn tsv_row(report: &Report) -> String {
    format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}",
        report.pair_index,
        escape_tab(&report.query_id),
        escape_tab(&report.target_id),
        report.mode,
        optional_i32(report.score),
        optional_u32(report.distance),
        optional_i64(report.path_score),
        report.verification_status.label(),
        report.query_start,
        report.query_end,
        report.target_start,
        report.target_end,
        escape_tab(&report.cigar),
        report.block_size,
        report.path_length,
        report.summary_build_ms,
        report.reconstruction_ms,
        report.verification_ms,
        report.elapsed_ms
    )
}

fn optional_i32(value: Option<i32>) -> String {
    value.map_or_else(String::new, |value| value.to_string())
}

fn optional_u32(value: Option<u32>) -> String {
    value.map_or_else(String::new, |value| value.to_string())
}

fn optional_i64(value: Option<i64>) -> String {
    value.map_or_else(String::new, |value| value.to_string())
}

fn escape_tab(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

fn affine_cells(path: &[NwAffineState]) -> Vec<(usize, usize)> {
    path.iter().map(|state| (state.row, state.col)).collect()
}

fn sw_cells(path: &[SwCell]) -> Vec<(usize, usize)> {
    path.iter().map(|cell| (cell.row, cell.col)).collect()
}

fn semiglobal_cells(path: &[SemiGlobalCell]) -> Vec<(usize, usize)> {
    path.iter().map(|cell| (cell.row, cell.col)).collect()
}
