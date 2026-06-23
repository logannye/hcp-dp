use std::{fs, path::PathBuf, process::ExitCode};

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
    match cli.command {
        Command::GlobalLinear(args) => run_global_linear(args),
        Command::GlobalAffine(args) => run_global_affine(args),
        Command::LocalLinear(args) => run_local_linear(args),
        Command::EditDistance(args) => run_edit_distance(args),
        Command::SemiglobalLinear(args) => run_semiglobal_linear(args),
    }
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

#[derive(Args)]
struct OutputArgs {
    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    format: OutputFormat,
    /// Include aligned query and target strings.
    #[arg(long)]
    show_alignment: bool,
    /// Check the HCP result against a full-table baseline and path score.
    #[arg(long)]
    verify: bool,
}

#[derive(Args)]
struct BlockArgs {
    /// Override the HCP summary-tree block size.
    #[arg(long)]
    block_size: Option<usize>,
}

#[derive(Args)]
struct LinearScoring {
    #[arg(long = "match", default_value_t = 2)]
    match_score: i32,
    #[arg(long, default_value_t = 1)]
    mismatch_penalty: i32,
    #[arg(long, allow_hyphen_values = true, default_value_t = -2)]
    gap: i32,
}

#[derive(Args)]
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
}

#[derive(Serialize)]
struct Report {
    mode: &'static str,
    score: Option<i32>,
    distance: Option<u32>,
    path_score: Option<i64>,
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
}

impl Report {
    fn from_trace(
        mode: &'static str,
        score: Option<i32>,
        distance: Option<u32>,
        path_score: Option<i64>,
        verified: Option<bool>,
        block_size: usize,
        trace: AlignmentTrace,
    ) -> Self {
        Self {
            mode,
            score,
            distance,
            path_score,
            verified,
            query_start: trace.query_start,
            query_end: trace.query_end,
            target_start: trace.target_start,
            target_end: trace.target_end,
            cigar: trace.cigar,
            operations: trace.operations,
            block_size,
            aligned_query: trace.aligned_query,
            aligned_target: trace.aligned_target,
        }
    }
}

fn run_global_linear(args: LinearCommand) -> Result<bool, String> {
    let (query, target) = read_inputs(&args.input)?;
    let problem = NwProblem::new(
        &query,
        &target,
        args.scoring.match_score,
        args.scoring.mismatch_penalty,
        args.scoring.gap,
    );
    let block_size = block_size_for(&problem, args.block.block_size)?;
    let (score, path) = run_engine(problem.clone(), args.block.block_size);
    let path_score = problem.score_path(&path);
    let verified = args
        .output
        .verify
        .then(|| path_score == Some(score) && problem.full_table_score() == score);
    let trace = AlignmentTrace::from_cells(&query, &target, &path, args.output.show_alignment);
    let report = Report::from_trace(
        "global-linear",
        Some(score),
        None,
        path_score.map(i64::from),
        verified,
        block_size,
        trace,
    );
    write_report(&report, args.output.format)?;
    Ok(verified.unwrap_or(true))
}

fn run_global_affine(args: AffineCommand) -> Result<bool, String> {
    let (query, target) = read_inputs(&args.input)?;
    let problem = NwAffineProblem::new(
        &query,
        &target,
        args.scoring.match_score,
        args.scoring.mismatch_penalty,
        args.scoring.gap_open,
        args.scoring.gap_extend,
    );
    let block_size = block_size_for(&problem, args.block.block_size)?;
    let (score, path) = run_engine(problem.clone(), args.block.block_size);
    let path_score = problem.score_path(&path);
    let verified = args
        .output
        .verify
        .then(|| path_score == Some(score) && problem.full_table_score() == score);
    let cells = affine_cells(&path);
    let trace = AlignmentTrace::from_cells(&query, &target, &cells, args.output.show_alignment);
    let report = Report::from_trace(
        "global-affine",
        Some(score),
        None,
        path_score.map(i64::from),
        verified,
        block_size,
        trace,
    );
    write_report(&report, args.output.format)?;
    Ok(verified.unwrap_or(true))
}

fn run_local_linear(args: LinearCommand) -> Result<bool, String> {
    let (query, target) = read_inputs(&args.input)?;
    let problem = SmithWatermanProblem::new(
        &query,
        &target,
        args.scoring.match_score,
        args.scoring.mismatch_penalty,
        args.scoring.gap,
    );
    let block_size = block_size_for(&problem, args.block.block_size)?;
    let (score, path) = run_engine(problem.clone(), args.block.block_size);
    let path_score = problem.score_path(&path);
    let verified = args
        .output
        .verify
        .then(|| path_score == Some(score) && problem.full_table_score() == score);
    let cells = sw_cells(&path);
    let trace = AlignmentTrace::from_cells(&query, &target, &cells, args.output.show_alignment);
    let report = Report::from_trace(
        "local-linear",
        Some(score),
        None,
        path_score.map(i64::from),
        verified,
        block_size,
        trace,
    );
    write_report(&report, args.output.format)?;
    Ok(verified.unwrap_or(true))
}

fn run_edit_distance(args: EditDistanceCommand) -> Result<bool, String> {
    let (query, target) = read_inputs(&args.input)?;
    let problem = EditDistanceProblem::new(&query, &target);
    let block_size = block_size_for(&problem, args.block.block_size)?;
    let (distance, path) = run_engine(problem.clone(), args.block.block_size);
    let path_score = problem.score_path(&path);
    let verified = args
        .output
        .verify
        .then(|| path_score == Some(distance) && problem.full_table_distance() == distance);
    let trace = AlignmentTrace::from_cells(&query, &target, &path, args.output.show_alignment);
    let report = Report::from_trace(
        "edit-distance",
        None,
        Some(distance),
        path_score.map(i64::from),
        verified,
        block_size,
        trace,
    );
    write_report(&report, args.output.format)?;
    Ok(verified.unwrap_or(true))
}

fn run_semiglobal_linear(args: LinearCommand) -> Result<bool, String> {
    let (query, target) = read_inputs(&args.input)?;
    let problem = SemiGlobalProblem::new(
        &query,
        &target,
        args.scoring.match_score,
        args.scoring.mismatch_penalty,
        args.scoring.gap,
    );
    let block_size = block_size_for(&problem, args.block.block_size)?;
    let (score, path) = run_engine(problem.clone(), args.block.block_size);
    let path_score = problem.score_path(&path);
    let verified = args
        .output
        .verify
        .then(|| path_score == Some(score) && problem.full_table_score() == score);
    let cells = semiglobal_cells(&path);
    let trace = AlignmentTrace::from_cells(&query, &target, &cells, args.output.show_alignment);
    let report = Report::from_trace(
        "semiglobal-linear",
        Some(score),
        None,
        path_score.map(i64::from),
        verified,
        block_size,
        trace,
    );
    write_report(&report, args.output.format)?;
    Ok(verified.unwrap_or(true))
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

fn write_report(report: &Report, format: OutputFormat) -> Result<(), String> {
    match format {
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(report).map_err(|err| err.to_string())?;
            println!("{json}");
        }
        OutputFormat::Text => write_text_report(report),
    }
    Ok(())
}

fn write_text_report(report: &Report) {
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
    match report.verified {
        Some(true) => println!("verified: true"),
        Some(false) => println!("verified: false"),
        None => println!("verified: not requested"),
    }
    println!(
        "query: [{}..{}), target: [{}..{})",
        report.query_start, report.query_end, report.target_start, report.target_end
    );
    println!("cigar: {}", report.cigar);
    println!("block_size: {}", report.block_size);
    if let Some(aligned_query) = &report.aligned_query {
        println!("aligned_query: {aligned_query}");
    }
    if let Some(aligned_target) = &report.aligned_target {
        println!("aligned_target: {aligned_target}");
    }
}

fn read_inputs(args: &InputArgs) -> Result<(Vec<u8>, Vec<u8>), String> {
    let query = read_sequence_source(&args.query, &args.query_file, "query")?;
    let target = read_sequence_source(&args.target, &args.target_file, "target")?;
    Ok((query, target))
}

fn read_sequence_source(
    inline: &Option<String>,
    path: &Option<PathBuf>,
    label: &str,
) -> Result<Vec<u8>, String> {
    match (inline, path) {
        (Some(_), Some(_)) => Err(format!(
            "provide either --{label} or --{label}-file, not both"
        )),
        (None, None) => Err(format!("missing --{label} or --{label}-file")),
        (Some(sequence), None) => normalize_sequence(sequence.as_bytes(), label),
        (None, Some(path)) => {
            let bytes = fs::read(path)
                .map_err(|err| format!("failed to read {label} file {}: {err}", path.display()))?;
            parse_sequence_file(&bytes, label)
        }
    }
}

fn parse_sequence_file(bytes: &[u8], label: &str) -> Result<Vec<u8>, String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| format!("{label} file must be valid UTF-8/ASCII sequence text"))?;
    let trimmed = text.trim_start();
    if trimmed.starts_with('>') {
        parse_fasta(trimmed, label)
    } else if trimmed.starts_with('@') {
        parse_fastq(trimmed, label)
    } else {
        normalize_sequence(text.as_bytes(), label)
    }
}

fn parse_fasta(text: &str, label: &str) -> Result<Vec<u8>, String> {
    let mut records = Vec::new();
    let mut current = Vec::new();
    let mut saw_header = false;

    for line in text.lines() {
        if line.starts_with('>') {
            if saw_header {
                records.push(std::mem::take(&mut current));
            }
            saw_header = true;
        } else if saw_header {
            current.extend_from_slice(line.as_bytes());
        } else if !line.trim().is_empty() {
            return Err(format!(
                "{label} FASTA sequence appears before first header"
            ));
        }
    }

    if saw_header {
        records.push(current);
    }
    if records.len() != 1 {
        return Err(format!(
            "{label} FASTA input must contain exactly one record; found {}",
            records.len()
        ));
    }
    normalize_sequence(&records[0], label)
}

fn parse_fastq(text: &str, label: &str) -> Result<Vec<u8>, String> {
    let lines: Vec<&str> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    if lines.len() != 4 || !lines[0].starts_with('@') || !lines[2].starts_with('+') {
        return Err(format!(
            "{label} FASTQ input must contain exactly one four-line record"
        ));
    }
    normalize_sequence(lines[1].as_bytes(), label)
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

fn affine_cells(path: &[NwAffineState]) -> Vec<(usize, usize)> {
    path.iter().map(|state| (state.row, state.col)).collect()
}

fn sw_cells(path: &[SwCell]) -> Vec<(usize, usize)> {
    path.iter().map(|cell| (cell.row, cell.col)).collect()
}

fn semiglobal_cells(path: &[SemiGlobalCell]) -> Vec<(usize, usize)> {
    path.iter().map(|cell| (cell.row, cell.col)).collect()
}
