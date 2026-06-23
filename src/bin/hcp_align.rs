use std::{path::PathBuf, process::ExitCode, time::Instant};

use clap::{Args, Parser, Subcommand, ValueEnum};
use hcp_dp::{
    alignment::AlignmentTrace,
    problems::{
        edit_distance::EditDistanceProblem,
        nw_affine::{NwAffineProblem, NwAffineState},
        nw_align::NwProblem,
        semiglobal::{SemiGlobalCell, SemiGlobalProblem},
        smith_waterman::{SmithWatermanProblem, SwCell},
    },
    HcpEngine, HcpProblem,
};
use serde::Serialize;

#[path = "../sequence_io.rs"]
mod sequence_io;

use sequence_io::{read_sequence_records, SequenceRecord};

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
    let (reports, format) = match cli.command {
        Command::GlobalLinear(args) => {
            let format = args.output.format;
            (
                run_linear_mode(args, "global-linear", align_global_linear)?,
                format,
            )
        }
        Command::GlobalAffine(args) => {
            let format = args.output.format;
            (run_affine_mode(args)?, format)
        }
        Command::LocalLinear(args) => {
            let format = args.output.format;
            (
                run_linear_mode(args, "local-linear", align_local_linear)?,
                format,
            )
        }
        Command::EditDistance(args) => {
            let format = args.output.format;
            (run_edit_distance_mode(args)?, format)
        }
        Command::SemiglobalLinear(args) => {
            let format = args.output.format;
            (
                run_linear_mode(args, "semiglobal-linear", align_semiglobal_linear)?,
                format,
            )
        }
    };

    write_reports(&reports, format)?;
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

#[derive(Clone, Copy, Args)]
struct OutputArgs {
    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    format: OutputFormat,
    /// Include aligned query and target strings.
    #[arg(long)]
    show_alignment: bool,
    /// Check the HCP result against a full-table baseline when within --verify-limit.
    #[arg(long)]
    verify: bool,
    /// Largest sequence length eligible for full-table --verify. Use 0 for no limit.
    #[arg(long, default_value_t = 2048)]
    verify_limit: usize,
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
    operations: Vec<hcp_dp::alignment::AlignmentStep>,
    block_size: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    aligned_query: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    aligned_target: Option<String>,
    elapsed_ms: f64,
}

struct VerificationResult {
    path_score: Option<i64>,
    status: VerificationStatus,
    verified: Option<bool>,
}

struct ReportObjective {
    score: Option<i32>,
    distance: Option<u32>,
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
    F: Fn(&PairInput, LinearScoring, OutputArgs, BlockArgs, &'static str) -> Result<Report, String>,
{
    let pairs = read_pairs(&args.input)?;
    pairs
        .iter()
        .map(|pair| align(pair, args.scoring, args.output, args.block, mode))
        .collect()
}

fn run_affine_mode(args: AffineCommand) -> Result<Vec<Report>, String> {
    let pairs = read_pairs(&args.input)?;
    pairs
        .iter()
        .map(|pair| align_global_affine(pair, args.scoring, args.output, args.block))
        .collect()
}

fn run_edit_distance_mode(args: EditDistanceCommand) -> Result<Vec<Report>, String> {
    let pairs = read_pairs(&args.input)?;
    pairs
        .iter()
        .map(|pair| align_edit_distance(pair, args.output, args.block))
        .collect()
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
    output: OutputArgs,
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
    let (score, path) = run_engine(problem.clone(), block.block_size);
    let verification = verify_i32(output, pair, problem.score_path(&path), score, || {
        problem.full_table_score()
    });
    let trace = AlignmentTrace::from_cells(
        &pair.query.sequence,
        &pair.target.sequence,
        &path,
        output.show_alignment,
    );
    Ok(report_from_trace(
        pair,
        mode,
        ReportObjective::score(score),
        verification,
        block_size,
        trace,
        start.elapsed().as_secs_f64() * 1000.0,
    ))
}

fn align_global_affine(
    pair: &PairInput,
    scoring: AffineScoring,
    output: OutputArgs,
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
    let (score, path) = run_engine(problem.clone(), block.block_size);
    let verification = verify_i32(output, pair, problem.score_path(&path), score, || {
        problem.full_table_score()
    });
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
        ReportObjective::score(score),
        verification,
        block_size,
        trace,
        start.elapsed().as_secs_f64() * 1000.0,
    ))
}

fn align_local_linear(
    pair: &PairInput,
    scoring: LinearScoring,
    output: OutputArgs,
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
    let (score, path) = run_engine(problem.clone(), block.block_size);
    let verification = verify_i32(output, pair, problem.score_path(&path), score, || {
        problem.full_table_score()
    });
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
        ReportObjective::score(score),
        verification,
        block_size,
        trace,
        start.elapsed().as_secs_f64() * 1000.0,
    ))
}

fn align_edit_distance(
    pair: &PairInput,
    output: OutputArgs,
    block: BlockArgs,
) -> Result<Report, String> {
    let start = Instant::now();
    let problem = EditDistanceProblem::new(&pair.query.sequence, &pair.target.sequence);
    let block_size = block_size_for(&problem, block.block_size)?;
    let (distance, path) = run_engine(problem.clone(), block.block_size);
    let verification = verify_u32(output, pair, problem.score_path(&path), distance, || {
        problem.full_table_distance()
    });
    let trace = AlignmentTrace::from_cells(
        &pair.query.sequence,
        &pair.target.sequence,
        &path,
        output.show_alignment,
    );
    Ok(report_from_trace(
        pair,
        "edit-distance",
        ReportObjective::distance(distance),
        verification,
        block_size,
        trace,
        start.elapsed().as_secs_f64() * 1000.0,
    ))
}

fn align_semiglobal_linear(
    pair: &PairInput,
    scoring: LinearScoring,
    output: OutputArgs,
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
    let (score, path) = run_engine(problem.clone(), block.block_size);
    let verification = verify_i32(output, pair, problem.score_path(&path), score, || {
        problem.full_table_score()
    });
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
        ReportObjective::score(score),
        verification,
        block_size,
        trace,
        start.elapsed().as_secs_f64() * 1000.0,
    ))
}

fn report_from_trace(
    pair: &PairInput,
    mode: &'static str,
    objective: ReportObjective,
    verification: VerificationResult,
    block_size: usize,
    trace: AlignmentTrace,
    elapsed_ms: f64,
) -> Report {
    Report {
        pair_index: pair.pair_index,
        query_id: pair.query.id.clone(),
        target_id: pair.target.id.clone(),
        mode,
        score: objective.score,
        distance: objective.distance,
        path_score: verification.path_score,
        verification_status: verification.status,
        verified: verification.verified,
        query_start: trace.query_start,
        query_end: trace.query_end,
        target_start: trace.target_start,
        target_end: trace.target_end,
        cigar: trace.cigar,
        operations: trace.operations,
        block_size,
        aligned_query: trace.aligned_query,
        aligned_target: trace.aligned_target,
        elapsed_ms,
    }
}

fn verify_i32<F>(
    output: OutputArgs,
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
        }
    } else {
        VerificationResult {
            path_score: path_score_i64,
            status: VerificationStatus::PathOnly,
            verified: None,
        }
    }
}

fn verify_u32<F>(
    output: OutputArgs,
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
        }
    } else {
        VerificationResult {
            path_score: path_score_i64,
            status: VerificationStatus::PathOnly,
            verified: None,
        }
    }
}

fn full_verify_requested(output: OutputArgs, pair: &PairInput) -> bool {
    if !output.verify {
        return false;
    }
    let max_len = pair.query.sequence.len().max(pair.target.sequence.len());
    output.verify_limit == 0 || max_len <= output.verify_limit
}

fn run_engine<P>(problem: P, block_size: Option<usize>) -> (P::Cost, Vec<P::State>)
where
    P: HcpProblem,
{
    if let Some(block_size) = block_size {
        HcpEngine::with_block_size(problem, block_size).run()
    } else {
        HcpEngine::new(problem).run()
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

fn write_reports(reports: &[Report], format: OutputFormat) -> Result<(), String> {
    match format {
        OutputFormat::Json => write_json_reports(reports),
        OutputFormat::Jsonl => write_jsonl_reports(reports),
        OutputFormat::Text => {
            write_text_reports(reports);
            Ok(())
        }
        OutputFormat::Tsv => {
            write_tsv_reports(reports);
            Ok(())
        }
        OutputFormat::Cigar => {
            write_cigar_reports(reports);
            Ok(())
        }
    }
}

fn write_json_reports(reports: &[Report]) -> Result<(), String> {
    let json = if reports.len() == 1 {
        serde_json::to_string_pretty(&reports[0]).map_err(|err| err.to_string())?
    } else {
        serde_json::to_string_pretty(reports).map_err(|err| err.to_string())?
    };
    println!("{json}");
    Ok(())
}

fn write_jsonl_reports(reports: &[Report]) -> Result<(), String> {
    for report in reports {
        let json = serde_json::to_string(report).map_err(|err| err.to_string())?;
        println!("{json}");
    }
    Ok(())
}

fn write_text_reports(reports: &[Report]) {
    for report in reports {
        if reports.len() > 1 {
            println!(
                "pair {}: {} vs {}",
                report.pair_index, report.query_id, report.target_id
            );
        }
        println!("mode: {}", report.mode);
        if let Some(score) = report.score {
            println!("score: {score}");
        }
        if let Some(distance) = report.distance {
            println!("distance: {distance}");
        }
        if let Some(path_score) = report.path_score {
            println!("path_score: {path_score}");
        }
        println!(
            "verification_status: {}",
            report.verification_status.label()
        );
        match report.verified {
            Some(true) => println!("verified: true"),
            Some(false) => println!("verified: false"),
            None => println!("verified: not_full"),
        }
        println!(
            "query: {} [{}..{}), target: {} [{}..{})",
            report.query_id,
            report.query_start,
            report.query_end,
            report.target_id,
            report.target_start,
            report.target_end
        );
        println!("cigar: {}", report.cigar);
        println!("block_size: {}", report.block_size);
        println!("elapsed_ms: {:.3}", report.elapsed_ms);
        if let Some(aligned_query) = &report.aligned_query {
            println!("aligned_query: {aligned_query}");
        }
        if let Some(aligned_target) = &report.aligned_target {
            println!("aligned_target: {aligned_target}");
        }
        if reports.len() > 1 {
            println!();
        }
    }
}

fn write_tsv_reports(reports: &[Report]) {
    println!(
        "pair_index\tquery_id\ttarget_id\tmode\tscore\tdistance\tpath_score\tverification_status\tquery_start\tquery_end\ttarget_start\ttarget_end\tcigar\tblock_size\telapsed_ms"
    );
    for report in reports {
        println!("{}", tsv_row(report));
    }
}

fn write_cigar_reports(reports: &[Report]) {
    println!(
        "pair_index\tquery_id\ttarget_id\tmode\tscore\tdistance\tpath_score\tverification_status\tquery_start\tquery_end\ttarget_start\ttarget_end\tcigar"
    );
    for report in reports {
        println!(
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
        );
    }
}

fn tsv_row(report: &Report) -> String {
    format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}",
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
