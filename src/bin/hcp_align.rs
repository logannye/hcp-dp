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
        edit_distance::{
            distance_adaptive_banded, distance_myers, trace_adaptive_banded, trace_banded,
            EditDistanceProblem,
        },
        nw_affine::{trace_banded_affine, NwAffineProblem, NwAffineState},
        nw_align::NwProblem,
        semiglobal::{SemiGlobalCell, SemiGlobalProblem},
        smith_waterman::{SmithWatermanProblem, SwCell},
    },
    scoring::{SubstitutionMatrix, SubstitutionScoring},
    seeding::seeded_window,
    HcpEngine, HcpProblem, HcpRunStats,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "parallel")]
use std::sync::atomic::{AtomicUsize, Ordering};

#[path = "../sequence_io.rs"]
mod sequence_io;

use sequence_io::{read_sequence_records, SequenceRecord};

const SCHEMA_VERSION: &str = "hcp-align.v1";
const ENGINE_NAME: &str = "hcp-dp";
const AUTO_EDIT_BAND_MIN: usize = 8;
const AUTO_EDIT_BAND_MAX: usize = 512;
const AUTO_EDIT_BAND_DIVISOR: usize = 32;

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
        Command::SeededGlobalLinear(args) => {
            let output = args.output.clone();
            (run_seeded_linear_mode(args)?, output)
        }
    };

    write_reports(&reports, &output)?;
    Ok(reports
        .iter()
        .all(|report| report.verification_status != VerificationStatus::Failed))
}

#[derive(Parser)]
#[command(name = "hcp-align")]
#[command(version)]
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
    /// Minimizer-seeded exact global alignment inside a candidate window.
    SeededGlobalLinear(SeededLinearCommand),
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
struct SeededLinearCommand {
    #[command(flatten)]
    input: InputArgs,
    #[command(flatten)]
    output: OutputArgs,
    #[command(flatten)]
    block: BlockArgs,
    #[command(flatten)]
    scoring: LinearScoring,
    #[command(flatten)]
    seed: SeedArgs,
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
    /// Exact traceback engine to use.
    #[arg(long, value_enum, default_value_t = AffineEngine::Hcp)]
    engine: AffineEngine,
    /// Diagonal half-band used by --engine wavefront or --engine auto.
    #[arg(long, default_value_t = 64)]
    wavefront_band: usize,
}

#[derive(Args)]
struct EditDistanceCommand {
    #[command(flatten)]
    input: InputArgs,
    #[command(flatten)]
    output: OutputArgs,
    #[command(flatten)]
    block: BlockArgs,
    /// Exact traceback engine to use.
    #[arg(long, value_enum, default_value_t = EditDistanceEngine::Auto)]
    engine: EditDistanceEngine,
    /// Compute only the exact edit distance, without traceback or CIGAR output.
    #[arg(long)]
    score_only: bool,
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
    /// Emit a proof-carrying hash certificate for each structured result.
    #[arg(long)]
    certificate: bool,
}

#[derive(Clone, Copy, Args)]
struct BlockArgs {
    /// Override the HCP summary-tree block size.
    #[arg(long)]
    block_size: Option<usize>,
}

#[derive(Clone, Copy, Args)]
struct SeedArgs {
    /// K-mer length used for minimizer seeds.
    #[arg(long, default_value_t = 15)]
    seed_k: usize,
    /// Number of adjacent k-mers per minimizer window.
    #[arg(long, default_value_t = 10)]
    seed_window: usize,
    /// Bases to include on each side of the selected seed chain.
    #[arg(long, default_value_t = 128)]
    seed_flank: usize,
}

#[derive(Clone, Args)]
struct LinearScoring {
    #[arg(long = "match", default_value_t = 2)]
    match_score: i32,
    #[arg(long, default_value_t = 1)]
    mismatch_penalty: i32,
    #[arg(long, allow_hyphen_values = true, default_value_t = -2)]
    gap: i32,
    /// Named substitution matrix to use instead of match/mismatch scoring.
    #[arg(long, value_enum)]
    matrix: Option<NamedMatrix>,
    /// Custom whitespace-delimited substitution matrix file.
    #[arg(long)]
    matrix_file: Option<PathBuf>,
}

#[derive(Clone, Args)]
struct AffineScoring {
    #[arg(long = "match", default_value_t = 2)]
    match_score: i32,
    #[arg(long, default_value_t = 1)]
    mismatch_penalty: i32,
    #[arg(long, allow_hyphen_values = true, default_value_t = -3)]
    gap_open: i32,
    #[arg(long, allow_hyphen_values = true, default_value_t = -1)]
    gap_extend: i32,
    /// Named substitution matrix to use instead of match/mismatch scoring.
    #[arg(long, value_enum)]
    matrix: Option<NamedMatrix>,
    /// Custom whitespace-delimited substitution matrix file.
    #[arg(long)]
    matrix_file: Option<PathBuf>,
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
enum NamedMatrix {
    Blosum62,
}

impl NamedMatrix {
    fn label(self) -> &'static str {
        match self {
            Self::Blosum62 => "BLOSUM62",
        }
    }

    fn scoring(self) -> SubstitutionScoring {
        match self {
            Self::Blosum62 => SubstitutionScoring::blosum62(),
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
    Jsonl,
    Tsv,
    Cigar,
    Paf,
    Sam,
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
enum OperationDetail {
    None,
    Summary,
    Full,
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
enum EditDistanceEngine {
    /// Deterministically choose an exact backend.
    Auto,
    /// Generic HCP summary-tree traceback with square-root checkpointing.
    Hcp,
    /// Generic HCP traceback with one-layer blocks for minimal retained state.
    HcpLinear,
    /// Exact adaptive-banded traceback for low-edit-distance inputs.
    AdaptiveBanded,
    /// Exact Myers bit-vector distance without traceback.
    Myers,
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
enum AffineEngine {
    /// Generic HCP summary-tree traceback.
    Hcp,
    /// Try diagonal-band affine traceback, then fall back to HCP linear-space.
    Auto,
    /// Exact diagonal-band affine traceback.
    Wavefront,
}

impl AffineEngine {
    fn backend_label(self) -> &'static str {
        match self {
            Self::Hcp => "hcp",
            Self::Auto => "auto",
            Self::Wavefront => "wavefront-affine",
        }
    }
}

impl EditDistanceEngine {
    fn backend_label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Hcp => "hcp",
            Self::HcpLinear => "hcp-linear",
            Self::AdaptiveBanded => "adaptive-banded",
            Self::Myers => "myers",
        }
    }
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
    ScoreOnly,
    Failed,
}

impl VerificationStatus {
    fn label(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::PathOnly => "path_only",
            Self::ScoreOnly => "score_only",
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
    #[serde(skip)]
    query_len: usize,
    #[serde(skip)]
    target_len: usize,
    #[serde(skip)]
    query_sequence: String,
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
    backend: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    operation_counts: Option<Vec<OperationCount>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    operations: Option<Vec<hcp_dp::alignment::AlignmentStep>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    certificate: Option<AlignmentCertificate>,
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

#[derive(Clone, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
enum ScoringParams {
    Linear {
        substitution: SubstitutionParams,
        gap: i32,
    },
    SeededLinear {
        substitution: SubstitutionParams,
        gap: i32,
        seed_k: usize,
        seed_window: usize,
        seed_flank: usize,
    },
    Affine {
        substitution: SubstitutionParams,
        gap_open: i32,
        gap_extend: i32,
    },
    EditDistance {
        substitution: u32,
        insertion: u32,
        deletion: u32,
    },
}

#[derive(Clone, Serialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
enum SubstitutionParams {
    MatchMismatch {
        match_score: i32,
        mismatch_penalty: i32,
    },
    Matrix {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        source_sha256: Option<String>,
    },
}

#[derive(Clone, Serialize)]
struct AlignmentCertificate {
    version: &'static str,
    hash_algorithm: &'static str,
    query_sha256: String,
    target_sha256: String,
    parameters_sha256: String,
    trace_sha256: Option<String>,
    result_sha256: String,
    certificate_sha256: String,
}

#[derive(Clone, Copy)]
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
    backend: Option<&'static str>,
    scoring: ScoringParams,
    include_certificate: bool,
}

struct ResolvedSubstitutionScoring {
    scoring: SubstitutionScoring,
    params: SubstitutionParams,
}

struct ResultHashInput<'a> {
    mode: &'static str,
    backend: Option<&'static str>,
    objective: ReportObjective,
    verification: &'a VerificationResult,
    query_start: usize,
    query_end: usize,
    target_start: usize,
    target_end: usize,
    cigar: &'a str,
    block_size: usize,
    path_length: usize,
    trace_sha256: Option<&'a str>,
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
        align(pair, args.scoring.clone(), &args.output, args.block, mode)
    })
}

fn run_seeded_linear_mode(args: SeededLinearCommand) -> Result<Vec<Report>, String> {
    let pairs = read_pairs(&args.input)?;
    run_pairs(&pairs, &args.output, "seeded-global-linear", |pair| {
        align_seeded_global_linear(
            pair,
            args.scoring.clone(),
            &args.output,
            args.block,
            args.seed,
        )
    })
}

fn run_affine_mode(args: AffineCommand) -> Result<Vec<Report>, String> {
    let pairs = read_pairs(&args.input)?;
    run_pairs(&pairs, &args.output, "global-affine", |pair| {
        align_global_affine(
            pair,
            args.scoring.clone(),
            &args.output,
            args.block,
            args.engine,
            args.wavefront_band,
        )
    })
}

fn run_edit_distance_mode(args: EditDistanceCommand) -> Result<Vec<Report>, String> {
    let pairs = read_pairs(&args.input)?;
    run_pairs(&pairs, &args.output, "edit-distance", |pair| {
        align_edit_distance(pair, &args.output, args.block, args.engine, args.score_only)
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

fn resolve_substitution_scoring(
    matrix: Option<NamedMatrix>,
    matrix_file: &Option<PathBuf>,
    match_score: i32,
    mismatch_penalty: i32,
) -> Result<ResolvedSubstitutionScoring, String> {
    match (matrix, matrix_file) {
        (Some(_), Some(_)) => Err("provide either --matrix or --matrix-file, not both".to_string()),
        (Some(matrix), None) => Ok(ResolvedSubstitutionScoring {
            scoring: matrix.scoring(),
            params: SubstitutionParams::Matrix {
                name: matrix.label().to_string(),
                source_sha256: None,
            },
        }),
        (None, Some(path)) => {
            let text = fs::read_to_string(path)
                .map_err(|err| format!("failed to read matrix file {}: {err}", path.display()))?;
            let name = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .filter(|stem| !stem.is_empty())
                .unwrap_or("custom")
                .to_string();
            let source_sha256 = sha256_hex(text.as_bytes());
            let matrix = SubstitutionMatrix::from_text(name.clone(), &text)
                .map_err(|err| format!("failed to parse matrix file {}: {err}", path.display()))?;
            Ok(ResolvedSubstitutionScoring {
                scoring: SubstitutionScoring::matrix(matrix),
                params: SubstitutionParams::Matrix {
                    name,
                    source_sha256: Some(source_sha256),
                },
            })
        }
        (None, None) => Ok(ResolvedSubstitutionScoring {
            scoring: SubstitutionScoring::match_mismatch(match_score, mismatch_penalty),
            params: SubstitutionParams::MatchMismatch {
                match_score,
                mismatch_penalty,
            },
        }),
    }
}

fn validate_substitution_inputs(
    scoring: &SubstitutionScoring,
    pair: &PairInput,
) -> Result<(), String> {
    scoring
        .validate_sequence(&pair.query.sequence)
        .map_err(|err| format!("query {}: {err}", pair.query.id))?;
    scoring
        .validate_sequence(&pair.target.sequence)
        .map_err(|err| format!("target {}: {err}", pair.target.id))?;
    Ok(())
}

fn align_global_linear(
    pair: &PairInput,
    scoring: LinearScoring,
    output: &OutputArgs,
    block: BlockArgs,
    mode: &'static str,
) -> Result<Report, String> {
    let start = Instant::now();
    let ResolvedSubstitutionScoring {
        scoring: substitution_scoring,
        params: substitution,
    } = resolve_substitution_scoring(
        scoring.matrix,
        &scoring.matrix_file,
        scoring.match_score,
        scoring.mismatch_penalty,
    )?;
    validate_substitution_inputs(&substitution_scoring, pair)?;
    let problem = NwProblem::with_scoring(
        &pair.query.sequence,
        &pair.target.sequence,
        substitution_scoring,
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
            backend: None,
            scoring: ScoringParams::Linear {
                substitution,
                gap: scoring.gap,
            },
            include_certificate: output.certificate,
        },
    ))
}

fn align_seeded_global_linear(
    pair: &PairInput,
    scoring: LinearScoring,
    output: &OutputArgs,
    block: BlockArgs,
    seed: SeedArgs,
) -> Result<Report, String> {
    let start = Instant::now();
    let ResolvedSubstitutionScoring {
        scoring: substitution_scoring,
        params: substitution,
    } = resolve_substitution_scoring(
        scoring.matrix,
        &scoring.matrix_file,
        scoring.match_score,
        scoring.mismatch_penalty,
    )?;
    validate_substitution_inputs(&substitution_scoring, pair)?;
    let seed_window = seeded_window(
        &pair.query.sequence,
        &pair.target.sequence,
        seed.seed_k,
        seed.seed_window,
        seed.seed_flank,
    )?
    .ok_or_else(|| {
        format!(
            "no minimizer seed found with --seed-k {} and --seed-window {}; use smaller seed parameters or global-linear",
            seed.seed_k, seed.seed_window
        )
    })?;
    let query_slice = &pair.query.sequence[seed_window.query_start..seed_window.query_end];
    let target_slice = &pair.target.sequence[seed_window.target_start..seed_window.target_end];
    let problem =
        NwProblem::with_scoring(query_slice, target_slice, substitution_scoring, scoring.gap);
    let block_size = block_size_for(&problem, block.block_size)?;
    let (score, path, stats) = run_engine(problem.clone(), block.block_size);
    let verify_start = Instant::now();
    let mut verification = verify_i32(output, pair, problem.score_path(&path), score, || {
        problem.full_table_score()
    });
    verification.verification_ms = verify_start.elapsed().as_secs_f64() * 1000.0;
    let trace = offset_trace(
        AlignmentTrace::from_cells(query_slice, target_slice, &path, output.show_alignment),
        seed_window.query_start,
        seed_window.target_start,
    );
    Ok(report_from_trace(
        pair,
        "seeded-global-linear",
        trace,
        ReportMetrics {
            objective: ReportObjective::score(score),
            verification,
            block_size,
            path_length: path.len(),
            stats,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            operation_detail: output.operation_detail,
            backend: Some("minimizer-seeded"),
            scoring: ScoringParams::SeededLinear {
                substitution,
                gap: scoring.gap,
                seed_k: seed.seed_k,
                seed_window: seed.seed_window,
                seed_flank: seed.seed_flank,
            },
            include_certificate: output.certificate,
        },
    ))
}

fn align_global_affine(
    pair: &PairInput,
    scoring: AffineScoring,
    output: &OutputArgs,
    block: BlockArgs,
    engine: AffineEngine,
    wavefront_band: usize,
) -> Result<Report, String> {
    let start = Instant::now();
    let ResolvedSubstitutionScoring {
        scoring: substitution_scoring,
        params: substitution,
    } = resolve_substitution_scoring(
        scoring.matrix,
        &scoring.matrix_file,
        scoring.match_score,
        scoring.mismatch_penalty,
    )?;
    validate_substitution_inputs(&substitution_scoring, pair)?;
    let problem = NwAffineProblem::with_scoring(
        &pair.query.sequence,
        &pair.target.sequence,
        substitution_scoring,
        scoring.gap_open,
        scoring.gap_extend,
    );
    let (score, path, stats, block_size, backend) = match engine {
        AffineEngine::Hcp => {
            let block_size = block_size_for(&problem, block.block_size)?;
            let (score, path, stats) = run_engine(problem.clone(), block.block_size);
            (score, path, stats, block_size, None)
        }
        AffineEngine::Wavefront => {
            if block.block_size.is_some() {
                return Err("--block-size applies only to --engine hcp".to_string());
            }
            let trace_start = Instant::now();
            let trace = trace_banded_affine(&problem, wavefront_band).ok_or_else(|| {
                format!(
                    "--engine wavefront did not find a path inside --wavefront-band {wavefront_band}; increase the band or use --engine hcp"
                )
            })?;
            (
                trace.score,
                trace.path,
                HcpRunStats {
                    summary_build_ms: 0.0,
                    reconstruction_ms: trace_start.elapsed().as_secs_f64() * 1000.0,
                },
                0,
                Some(AffineEngine::Wavefront.backend_label()),
            )
        }
        AffineEngine::Auto => {
            if block.block_size.is_some() {
                return Err("--block-size applies only to --engine hcp".to_string());
            }
            let trace_start = Instant::now();
            if let Some(trace) = trace_banded_affine(&problem, wavefront_band) {
                (
                    trace.score,
                    trace.path,
                    HcpRunStats {
                        summary_build_ms: 0.0,
                        reconstruction_ms: trace_start.elapsed().as_secs_f64() * 1000.0,
                    },
                    0,
                    Some(AffineEngine::Wavefront.backend_label()),
                )
            } else {
                let (score, path, stats) =
                    HcpEngine::linear_space(problem.clone()).run_with_stats();
                (score, path, stats, 1, Some("hcp-linear"))
            }
        }
    };
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
            backend,
            scoring: ScoringParams::Affine {
                substitution,
                gap_open: scoring.gap_open,
                gap_extend: scoring.gap_extend,
            },
            include_certificate: output.certificate,
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
    let ResolvedSubstitutionScoring {
        scoring: substitution_scoring,
        params: substitution,
    } = resolve_substitution_scoring(
        scoring.matrix,
        &scoring.matrix_file,
        scoring.match_score,
        scoring.mismatch_penalty,
    )?;
    validate_substitution_inputs(&substitution_scoring, pair)?;
    let problem = SmithWatermanProblem::with_scoring(
        &pair.query.sequence,
        &pair.target.sequence,
        substitution_scoring,
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
            backend: None,
            scoring: ScoringParams::Linear {
                substitution,
                gap: scoring.gap,
            },
            include_certificate: output.certificate,
        },
    ))
}

fn align_edit_distance(
    pair: &PairInput,
    output: &OutputArgs,
    block: BlockArgs,
    engine: EditDistanceEngine,
    score_only: bool,
) -> Result<Report, String> {
    if score_only {
        return score_only_edit_distance(pair, output, block, engine);
    }
    let start = Instant::now();
    let problem = EditDistanceProblem::new(&pair.query.sequence, &pair.target.sequence);
    let (distance, path, stats, block_size, backend) = match engine {
        EditDistanceEngine::Auto => {
            if block.block_size.is_some() {
                return Err(
                    "--block-size requires an explicit HCP edit-distance engine; use --engine hcp or --engine hcp-linear"
                        .to_string(),
                );
            }
            let band_limit =
                auto_edit_band_limit(pair.query.sequence.len(), pair.target.sequence.len());
            let trace_start = Instant::now();
            if let Some(trace) =
                trace_banded(&pair.query.sequence, &pair.target.sequence, band_limit)
            {
                (
                    trace.distance,
                    trace.path,
                    HcpRunStats {
                        summary_build_ms: 0.0,
                        reconstruction_ms: trace_start.elapsed().as_secs_f64() * 1000.0,
                    },
                    0,
                    Some(EditDistanceEngine::AdaptiveBanded.backend_label()),
                )
            } else {
                let (distance, path, stats) =
                    HcpEngine::linear_space(problem.clone()).run_with_stats();
                (
                    distance,
                    path,
                    stats,
                    1,
                    Some(EditDistanceEngine::HcpLinear.backend_label()),
                )
            }
        }
        EditDistanceEngine::Hcp => {
            let block_size = block_size_for(&problem, block.block_size)?;
            let (distance, path, stats) = run_engine(problem.clone(), block.block_size);
            (distance, path, stats, block_size, None)
        }
        EditDistanceEngine::HcpLinear => {
            if let Some(value) = block.block_size {
                if value != 1 {
                    return Err(
                        "--engine hcp-linear requires --block-size 1 or no block override"
                            .to_string(),
                    );
                }
            }
            let (distance, path, stats) = HcpEngine::linear_space(problem.clone()).run_with_stats();
            (
                distance,
                path,
                stats,
                1,
                Some(EditDistanceEngine::HcpLinear.backend_label()),
            )
        }
        EditDistanceEngine::AdaptiveBanded => {
            if block.block_size.is_some() {
                return Err("--block-size applies only to HCP edit-distance engines".to_string());
            }
            let trace_start = Instant::now();
            let trace = trace_adaptive_banded(&pair.query.sequence, &pair.target.sequence);
            (
                trace.distance,
                trace.path,
                HcpRunStats {
                    summary_build_ms: 0.0,
                    reconstruction_ms: trace_start.elapsed().as_secs_f64() * 1000.0,
                },
                0,
                Some(EditDistanceEngine::AdaptiveBanded.backend_label()),
            )
        }
        EditDistanceEngine::Myers => {
            return Err(
                "--engine myers is score-only; add --score-only or choose a traceback engine"
                    .to_string(),
            );
        }
    };
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
            backend,
            scoring: ScoringParams::EditDistance {
                substitution: 1,
                insertion: 1,
                deletion: 1,
            },
            include_certificate: output.certificate,
        },
    ))
}

fn score_only_edit_distance(
    pair: &PairInput,
    output: &OutputArgs,
    block: BlockArgs,
    engine: EditDistanceEngine,
) -> Result<Report, String> {
    if block.block_size.is_some() {
        return Err("--block-size applies only to traceback-producing HCP engines".to_string());
    }
    if output.show_alignment {
        return Err("--score-only cannot be combined with --show-alignment".to_string());
    }

    let start = Instant::now();
    let (distance, backend) = match engine {
        EditDistanceEngine::Auto | EditDistanceEngine::Myers => (
            distance_myers(&pair.query.sequence, &pair.target.sequence),
            EditDistanceEngine::Myers.backend_label(),
        ),
        EditDistanceEngine::AdaptiveBanded => (
            distance_adaptive_banded(&pair.query.sequence, &pair.target.sequence),
            EditDistanceEngine::AdaptiveBanded.backend_label(),
        ),
        EditDistanceEngine::Hcp | EditDistanceEngine::HcpLinear => {
            return Err(
                "--score-only supports --engine auto, --engine myers, or --engine adaptive-banded"
                    .to_string(),
            );
        }
    };

    let verify_start = Instant::now();
    let mut verification = verify_score_only_u32(output, pair, distance, || {
        EditDistanceProblem::new(&pair.query.sequence, &pair.target.sequence).full_table_distance()
    });
    verification.verification_ms = verify_start.elapsed().as_secs_f64() * 1000.0;
    let scoring = ScoringParams::EditDistance {
        substitution: 1,
        insertion: 1,
        deletion: 1,
    };
    let certificate = output.certificate.then(|| {
        build_score_only_certificate(
            pair,
            "edit-distance",
            backend,
            &scoring,
            distance,
            &verification,
        )
    });

    Ok(Report {
        schema_version: SCHEMA_VERSION,
        engine: ENGINE_NAME,
        pair_index: pair.pair_index,
        query_id: pair.query.id.clone(),
        target_id: pair.target.id.clone(),
        query_len: pair.query.sequence.len(),
        target_len: pair.target.sequence.len(),
        query_sequence: sequence_string(&pair.query.sequence),
        mode: "edit-distance",
        score: None,
        distance: Some(distance),
        path_score: None,
        verification_status: verification.status,
        verified: verification.verified,
        query_start: 0,
        query_end: pair.query.sequence.len(),
        target_start: 0,
        target_end: pair.target.sequence.len(),
        cigar: String::new(),
        backend: Some(backend),
        operation_counts: None,
        operations: None,
        certificate,
        block_size: 0,
        path_length: 0,
        summary_build_ms: 0.0,
        reconstruction_ms: 0.0,
        verification_ms: verification.verification_ms,
        aligned_query: None,
        aligned_target: None,
        error: None,
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
    })
}

fn auto_edit_band_limit(query_len: usize, target_len: usize) -> usize {
    let max_len = query_len.max(target_len);
    if max_len == 0 {
        return 0;
    }
    let length_delta = query_len.abs_diff(target_len);
    let similarity_band = (max_len / AUTO_EDIT_BAND_DIVISOR).clamp(
        AUTO_EDIT_BAND_MIN.min(max_len),
        AUTO_EDIT_BAND_MAX.min(max_len),
    );
    length_delta.max(similarity_band).min(max_len)
}

fn align_semiglobal_linear(
    pair: &PairInput,
    scoring: LinearScoring,
    output: &OutputArgs,
    block: BlockArgs,
    mode: &'static str,
) -> Result<Report, String> {
    let start = Instant::now();
    let ResolvedSubstitutionScoring {
        scoring: substitution_scoring,
        params: substitution,
    } = resolve_substitution_scoring(
        scoring.matrix,
        &scoring.matrix_file,
        scoring.match_score,
        scoring.mismatch_penalty,
    )?;
    validate_substitution_inputs(&substitution_scoring, pair)?;
    let problem = SemiGlobalProblem::with_scoring(
        &pair.query.sequence,
        &pair.target.sequence,
        substitution_scoring,
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
            backend: None,
            scoring: ScoringParams::Linear {
                substitution,
                gap: scoring.gap,
            },
            include_certificate: output.certificate,
        },
    ))
}

fn report_from_trace(
    pair: &PairInput,
    mode: &'static str,
    trace: AlignmentTrace,
    metrics: ReportMetrics,
) -> Report {
    let certificate = metrics
        .include_certificate
        .then(|| build_alignment_certificate(pair, mode, &trace, &metrics));
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
        query_len: pair.query.sequence.len(),
        target_len: pair.target.sequence.len(),
        query_sequence: sequence_string(&pair.query.sequence),
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
        backend: metrics.backend,
        operation_counts,
        operations,
        certificate,
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

fn build_alignment_certificate(
    pair: &PairInput,
    mode: &'static str,
    trace: &AlignmentTrace,
    metrics: &ReportMetrics,
) -> AlignmentCertificate {
    let trace_sha256 = trace_operations_sha256(&trace.operations);
    let result_sha256 = result_sha256(ResultHashInput {
        mode,
        backend: metrics.backend,
        objective: metrics.objective,
        verification: &metrics.verification,
        query_start: trace.query_start,
        query_end: trace.query_end,
        target_start: trace.target_start,
        target_end: trace.target_end,
        cigar: &trace.cigar,
        block_size: metrics.block_size,
        path_length: metrics.path_length,
        trace_sha256: Some(&trace_sha256),
    });
    build_certificate(
        &pair.query.sequence,
        &pair.target.sequence,
        &metrics.scoring,
        Some(trace_sha256),
        result_sha256,
    )
}

fn build_score_only_certificate(
    pair: &PairInput,
    mode: &'static str,
    backend: &'static str,
    scoring: &ScoringParams,
    distance: u32,
    verification: &VerificationResult,
) -> AlignmentCertificate {
    let result_sha256 = result_sha256(ResultHashInput {
        mode,
        backend: Some(backend),
        objective: ReportObjective::distance(distance),
        verification,
        query_start: 0,
        query_end: pair.query.sequence.len(),
        target_start: 0,
        target_end: pair.target.sequence.len(),
        cigar: "",
        block_size: 0,
        path_length: 0,
        trace_sha256: None,
    });
    build_certificate(
        &pair.query.sequence,
        &pair.target.sequence,
        scoring,
        None,
        result_sha256,
    )
}

fn build_certificate(
    query: &[u8],
    target: &[u8],
    scoring: &ScoringParams,
    trace_sha256: Option<String>,
    result_sha256: String,
) -> AlignmentCertificate {
    let query_sha256 = sha256_hex(query);
    let target_sha256 = sha256_hex(target);
    let parameters_sha256 = sha256_json(scoring);
    let certificate_sha256 = certificate_payload_sha256(
        &query_sha256,
        &target_sha256,
        &parameters_sha256,
        trace_sha256.as_deref(),
        &result_sha256,
    );
    AlignmentCertificate {
        version: "hcp-align.certificate.v1",
        hash_algorithm: "sha256",
        query_sha256,
        target_sha256,
        parameters_sha256,
        trace_sha256,
        result_sha256,
        certificate_sha256,
    }
}

fn result_sha256(input: ResultHashInput<'_>) -> String {
    let mut hasher = Sha256::new();
    update_field(&mut hasher, "schema_version", SCHEMA_VERSION);
    update_field(&mut hasher, "engine", ENGINE_NAME);
    update_field(&mut hasher, "mode", input.mode);
    update_field(&mut hasher, "backend", input.backend.unwrap_or("hcp"));
    update_optional_field(
        &mut hasher,
        "score",
        input.objective.score.map(|value| value.to_string()),
    );
    update_optional_field(
        &mut hasher,
        "distance",
        input.objective.distance.map(|value| value.to_string()),
    );
    update_optional_field(
        &mut hasher,
        "path_score",
        input.verification.path_score.map(|value| value.to_string()),
    );
    update_field(
        &mut hasher,
        "verification_status",
        input.verification.status.label(),
    );
    update_optional_field(
        &mut hasher,
        "verified",
        input.verification.verified.map(|value| value.to_string()),
    );
    update_field(&mut hasher, "query_start", &input.query_start.to_string());
    update_field(&mut hasher, "query_end", &input.query_end.to_string());
    update_field(&mut hasher, "target_start", &input.target_start.to_string());
    update_field(&mut hasher, "target_end", &input.target_end.to_string());
    update_field(&mut hasher, "cigar", input.cigar);
    update_field(&mut hasher, "block_size", &input.block_size.to_string());
    update_field(&mut hasher, "path_length", &input.path_length.to_string());
    update_optional_field(
        &mut hasher,
        "trace_sha256",
        input.trace_sha256.map(str::to_string),
    );
    hex_lower(&hasher.finalize())
}

fn trace_operations_sha256(operations: &[hcp_dp::alignment::AlignmentStep]) -> String {
    let mut hasher = Sha256::new();
    update_field(
        &mut hasher,
        "operation_count",
        &operations.len().to_string(),
    );
    for (idx, operation) in operations.iter().enumerate() {
        update_field(&mut hasher, "index", &idx.to_string());
        update_field(&mut hasher, "op", alignment_op_label(operation.op));
        update_optional_field(
            &mut hasher,
            "query_pos",
            operation.query_pos.map(|value| value.to_string()),
        );
        update_optional_field(
            &mut hasher,
            "target_pos",
            operation.target_pos.map(|value| value.to_string()),
        );
        update_optional_field(
            &mut hasher,
            "query_base",
            operation.query_base.map(|value| value.to_string()),
        );
        update_optional_field(
            &mut hasher,
            "target_base",
            operation.target_base.map(|value| value.to_string()),
        );
    }
    hex_lower(&hasher.finalize())
}

fn certificate_payload_sha256(
    query_sha256: &str,
    target_sha256: &str,
    parameters_sha256: &str,
    trace_sha256: Option<&str>,
    result_sha256: &str,
) -> String {
    let mut hasher = Sha256::new();
    update_field(&mut hasher, "version", "hcp-align.certificate.v1");
    update_field(&mut hasher, "hash_algorithm", "sha256");
    update_field(&mut hasher, "query_sha256", query_sha256);
    update_field(&mut hasher, "target_sha256", target_sha256);
    update_field(&mut hasher, "parameters_sha256", parameters_sha256);
    update_optional_field(
        &mut hasher,
        "trace_sha256",
        trace_sha256.map(str::to_string),
    );
    update_field(&mut hasher, "result_sha256", result_sha256);
    hex_lower(&hasher.finalize())
}

fn sha256_json<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).expect("certificate parameters must serialize");
    sha256_hex(&bytes)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_lower(&hasher.finalize())
}

fn update_optional_field(hasher: &mut Sha256, key: &str, value: Option<String>) {
    match value {
        Some(value) => update_field(hasher, key, &value),
        None => update_field(hasher, key, "<none>"),
    }
}

fn update_field(hasher: &mut Sha256, key: &str, value: &str) {
    update_bytes(hasher, key.as_bytes());
    update_bytes(hasher, value.as_bytes());
}

fn update_bytes(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn alignment_op_label(op: AlignmentOpKind) -> &'static str {
    match op {
        AlignmentOpKind::Match => "match",
        AlignmentOpKind::Mismatch => "mismatch",
        AlignmentOpKind::GapInTarget => "gap_in_target",
        AlignmentOpKind::GapInQuery => "gap_in_query",
    }
}

fn error_report(pair: &PairInput, mode: &'static str, error: String) -> Report {
    Report {
        schema_version: SCHEMA_VERSION,
        engine: ENGINE_NAME,
        pair_index: pair.pair_index,
        query_id: pair.query.id.clone(),
        target_id: pair.target.id.clone(),
        query_len: pair.query.sequence.len(),
        target_len: pair.target.sequence.len(),
        query_sequence: sequence_string(&pair.query.sequence),
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
        backend: None,
        operation_counts: None,
        operations: None,
        certificate: None,
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

fn verify_score_only_u32<F>(
    output: &OutputArgs,
    pair: &PairInput,
    reported: u32,
    baseline: F,
) -> VerificationResult
where
    F: FnOnce() -> u32,
{
    if full_verify_requested(output, pair) {
        let passed = baseline() == reported;
        VerificationResult {
            path_score: None,
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
            path_score: None,
            status: VerificationStatus::ScoreOnly,
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
        OutputFormat::Paf => paf_reports(reports)?,
        OutputFormat::Sam => sam_reports(reports)?,
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
        if let Some(backend) = report.backend {
            output.push_str(&format!("backend: {backend}\n"));
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

fn paf_reports(reports: &[Report]) -> Result<String, String> {
    let mut output = String::new();
    for report in reports {
        ensure_traceback_for_interop("PAF", report)?;
        let stats = cigar_stats(&report.cigar)?;
        let paf_cigar = paf_cigar_from_hcp_cigar(&report.cigar)?;
        output.push_str(&format!(
            "{}\t{}\t{}\t{}\t+\t{}\t{}\t{}\t{}\t{}\t{}\t255",
            escape_tab(&report.query_id),
            report.query_len,
            report.query_start,
            report.query_end,
            escape_tab(&report.target_id),
            report.target_len,
            report.target_start,
            report.target_end,
            stats.matches,
            stats.block_len,
        ));
        if let Some(score) = report.score {
            output.push_str(&format!("\tAS:i:{score}"));
        }
        let nm = report
            .distance
            .map_or(stats.edit_ops, |distance| distance as usize);
        output.push_str(&format!(
            "\tNM:i:{nm}\tcg:Z:{paf_cigar}\tvs:Z:{}\tpi:i:{}",
            report.verification_status.label(),
            report.pair_index,
        ));
        output.push('\n');
    }
    Ok(output)
}

fn sam_reports(reports: &[Report]) -> Result<String, String> {
    let mut output = String::from("@HD\tVN:1.6\tSO:unknown\n");
    let mut references: Vec<(&str, usize)> = Vec::new();
    for report in reports {
        if let Some((_, length)) = references
            .iter()
            .find(|(name, _)| *name == report.target_id.as_str())
        {
            if *length != report.target_len {
                return Err("SAM output requires each target id to have one length".to_string());
            }
        } else {
            references.push((&report.target_id, report.target_len));
        }
    }
    for (name, length) in references {
        output.push_str(&format!("@SQ\tSN:{}\tLN:{length}\n", escape_tab(name)));
    }

    for report in reports {
        ensure_traceback_for_interop("SAM", report)?;
        let stats = cigar_stats(&report.cigar)?;
        let cigar = sam_cigar(report)?;
        let pos = report.target_start + 1;
        let seq = if report.query_sequence.is_empty() {
            "*"
        } else {
            &report.query_sequence
        };
        output.push_str(&format!(
            "{}\t0\t{}\t{}\t255\t{}\t*\t0\t0\t{}\t*",
            escape_tab(&report.query_id),
            escape_tab(&report.target_id),
            pos,
            cigar,
            seq,
        ));
        if let Some(score) = report.score {
            output.push_str(&format!("\tAS:i:{score}"));
        }
        let nm = report
            .distance
            .map_or(stats.edit_ops, |distance| distance as usize);
        output.push_str(&format!(
            "\tNM:i:{nm}\tVS:Z:{}\tPI:i:{}",
            report.verification_status.label(),
            report.pair_index,
        ));
        output.push('\n');
    }
    Ok(output)
}

fn ensure_traceback_for_interop(format: &str, report: &Report) -> Result<(), String> {
    if report.error.is_some() {
        return Err(format!("{format} output cannot represent failed records"));
    }
    if report.path_score.is_none() {
        return Err(format!(
            "{format} output requires traceback; disable --score-only"
        ));
    }
    Ok(())
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

fn offset_trace(
    mut trace: AlignmentTrace,
    query_offset: usize,
    target_offset: usize,
) -> AlignmentTrace {
    trace.query_start += query_offset;
    trace.query_end += query_offset;
    trace.target_start += target_offset;
    trace.target_end += target_offset;
    for operation in &mut trace.operations {
        if let Some(query_pos) = operation.query_pos.as_mut() {
            *query_pos += query_offset;
        }
        if let Some(target_pos) = operation.target_pos.as_mut() {
            *target_pos += target_offset;
        }
    }
    trace
}

fn sequence_string(sequence: &[u8]) -> String {
    String::from_utf8_lossy(sequence).into_owned()
}

#[derive(Default)]
struct CigarStats {
    matches: usize,
    block_len: usize,
    edit_ops: usize,
}

fn cigar_stats(cigar: &str) -> Result<CigarStats, String> {
    let mut stats = CigarStats::default();
    for (count, op) in cigar_runs(cigar)? {
        match op {
            '=' => stats.matches += count,
            'X' | 'D' | 'I' => stats.edit_ops += count,
            _ => return Err(format!("unsupported CIGAR operation '{op}'")),
        }
        stats.block_len += count;
    }
    Ok(stats)
}

fn paf_cigar_from_hcp_cigar(cigar: &str) -> Result<String, String> {
    let mut output = String::new();
    for (count, op) in cigar_runs(cigar)? {
        let paf_op = match op {
            '=' | 'X' => op,
            'D' => 'I',
            'I' => 'D',
            _ => return Err(format!("unsupported CIGAR operation '{op}'")),
        };
        output.push_str(&count.to_string());
        output.push(paf_op);
    }
    Ok(output)
}

fn sam_cigar(report: &Report) -> Result<String, String> {
    let mut cigar = String::new();
    if report.query_start > 0 {
        cigar.push_str(&report.query_start.to_string());
        cigar.push('S');
    }
    cigar.push_str(&paf_cigar_from_hcp_cigar(&report.cigar)?);
    let suffix = report.query_len.saturating_sub(report.query_end);
    if suffix > 0 {
        cigar.push_str(&suffix.to_string());
        cigar.push('S');
    }
    if cigar.is_empty() {
        cigar.push('*');
    }
    Ok(cigar)
}

fn cigar_runs(cigar: &str) -> Result<Vec<(usize, char)>, String> {
    let mut runs = Vec::new();
    let mut count = 0usize;
    let mut in_count = false;
    for ch in cigar.chars() {
        if let Some(digit) = ch.to_digit(10) {
            in_count = true;
            count = count
                .checked_mul(10)
                .and_then(|value| value.checked_add(digit as usize))
                .ok_or_else(|| "CIGAR run length overflowed usize".to_string())?;
            continue;
        }
        if !in_count || count == 0 {
            return Err(format!("invalid CIGAR run before operation '{ch}'"));
        }
        runs.push((count, ch));
        count = 0;
        in_count = false;
    }
    if in_count {
        return Err("CIGAR ended with a run length but no operation".to_string());
    }
    Ok(runs)
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
