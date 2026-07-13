use clap::{Args, Parser, Subcommand, ValueEnum};
use hcp_dp::oracle::{
    record_to_paf, records_from_json, records_from_jsonl, records_from_paf, records_from_sam,
    records_to_json, records_to_jsonl, records_to_sam, AlignmentMode, AlignmentRecord,
    ComparisonReport, ProblemSpec, ScoringScheme, SolveOptions, SolveRequest, SolveScope,
    SubstitutionScheme, VerificationLevel, VerificationReport, VerificationRequest,
    VerificationStatus, DEFAULT_MAX_CELLS,
};
use hcp_dp::sequence_io::{read_sequence_records, SequenceRecord};
use hcp_dp::{compare, solve, verify};
use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

type SequenceMap = HashMap<String, Vec<u8>>;

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

#[derive(Clone, Copy, ValueEnum)]
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

#[derive(Clone, Copy, ValueEnum)]
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
    let pairs = read_pairs(&common.input)?;
    let mut records = Vec::with_capacity(pairs.len());
    for (index, (query, target)) in pairs.iter().enumerate() {
        records.push(solve(SolveRequest {
            pair_index: index,
            query_id: query.id.clone(),
            target_id: target.id.clone(),
            query: query.sequence.clone(),
            target: target.sequence.clone(),
            problem: ProblemSpec {
                mode,
                scoring: scoring.clone(),
            },
            options: SolveOptions {
                scope: match common.scope {
                    ScopeArg::Alignment => SolveScope::Alignment,
                    ScopeArg::Score => SolveScope::Score,
                },
                backend: common.backend.clone(),
                block_size: common.block_size,
                band: common.band,
                cross_check: common.cross_check,
                include_operations: common.operations,
                integrity: common.integrity,
            },
        })?);
    }
    write_alignment_records(&records, &pairs, &common.output)?;
    Ok(u8::from(records.iter().any(|record| {
        record.verification.status == VerificationStatus::Invalid
    })))
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

fn run_verify(command: VerifyCommand) -> Result<u8, String> {
    let (queries, targets) = read_source_maps(&command.input)?;
    let text = read_text(&command.alignment)?;
    let external_problem = if matches!(command.format, InputFormat::Paf | InputFormat::Sam) {
        problem_from_cli(command.mode, &command.scoring)?
    } else {
        None
    };
    let records = parse_records(
        &text,
        command.format,
        external_problem.as_ref(),
        &queries,
        &targets,
    )?;
    let mut reports = Vec::with_capacity(records.len());
    for record in records {
        let query = queries
            .get(&record.pair.query.id)
            .ok_or_else(|| format!("missing query source '{}'", record.pair.query.id))?;
        let target = targets
            .get(&record.pair.target.id)
            .ok_or_else(|| format!("missing target source '{}'", record.pair.target.id))?;
        reports.push(verify(VerificationRequest {
            record,
            query: query.clone(),
            target: target.clone(),
            level: command.level.into(),
            max_cells: command.max_cells,
        }));
    }
    let exit = verification_exit(&reports);
    write_reports(&reports, command.output_format, command.output.as_deref())?;
    Ok(exit)
}

fn run_compare(command: CompareCommand) -> Result<u8, String> {
    let (queries, targets) = read_source_maps(&command.input)?;
    let needs_problem = matches!(command.left_format, InputFormat::Paf | InputFormat::Sam)
        || matches!(command.right_format, InputFormat::Paf | InputFormat::Sam);
    let problem = if needs_problem {
        problem_from_cli(command.mode, &command.scoring)?
    } else {
        None
    };
    let left_text = read_text(&command.left)?;
    let right_text = read_text(&command.right)?;
    let left = parse_records(
        &left_text,
        command.left_format,
        problem.as_ref(),
        &queries,
        &targets,
    )?;
    let right = parse_records(
        &right_text,
        command.right_format,
        problem.as_ref(),
        &queries,
        &targets,
    )?;
    if left.len() != right.len() {
        return Err(format!(
            "comparison inputs have different record counts: {} vs {}",
            left.len(),
            right.len()
        ));
    }
    let mut reports = Vec::with_capacity(left.len());
    for (left, right) in left.into_iter().zip(right) {
        let query = queries
            .get(&left.pair.query.id)
            .ok_or_else(|| format!("missing query source '{}'", left.pair.query.id))?;
        let target = targets
            .get(&left.pair.target.id)
            .ok_or_else(|| format!("missing target source '{}'", left.pair.target.id))?;
        reports.push(compare(left, right, query, target)?);
    }
    let exit = u8::from(reports.iter().any(|report| !report.equivalent));
    write_comparisons(&reports, command.output_format, command.output.as_deref())?;
    Ok(exit)
}

fn parse_records(
    text: &str,
    format: InputFormat,
    problem: Option<&ProblemSpec>,
    queries: &HashMap<String, Vec<u8>>,
    targets: &HashMap<String, Vec<u8>>,
) -> Result<Vec<AlignmentRecord>, String> {
    match format {
        InputFormat::Json => records_from_json(text),
        InputFormat::Jsonl => records_from_jsonl(text),
        InputFormat::Paf => records_from_paf(
            text,
            problem.ok_or_else(|| "--mode and scoring are required for PAF".to_string())?,
            queries,
            targets,
        ),
        InputFormat::Sam => records_from_sam(
            text,
            problem.ok_or_else(|| "--mode and scoring are required for SAM".to_string())?,
            queries,
            targets,
        ),
    }
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

fn read_pairs(args: &InputArgs) -> Result<Vec<(SequenceRecord, SequenceRecord)>, String> {
    let queries = read_sequence_records(&args.query, &args.query_file, "query")?;
    let targets = read_sequence_records(&args.target, &args.target_file, "target")?;
    if queries.len() != targets.len() {
        return Err(format!(
            "pairwise mode requires equal record counts; query has {}, target has {}",
            queries.len(),
            targets.len()
        ));
    }
    Ok(queries.into_iter().zip(targets).collect())
}

fn read_source_maps(args: &InputArgs) -> Result<(SequenceMap, SequenceMap), String> {
    let query_records = read_sequence_records(&args.query, &args.query_file, "query")?;
    let target_records = read_sequence_records(&args.target, &args.target_file, "target")?;
    let mut queries = HashMap::new();
    let mut targets = HashMap::new();
    for query in query_records {
        if queries.insert(query.id.clone(), query.sequence).is_some() {
            return Err(format!("duplicate query id '{}'", query.id));
        }
    }
    for target in target_records {
        if targets.insert(target.id.clone(), target.sequence).is_some() {
            return Err(format!("duplicate target id '{}'", target.id));
        }
    }
    Ok((queries, targets))
}

fn write_alignment_records(
    records: &[AlignmentRecord],
    pairs: &[(SequenceRecord, SequenceRecord)],
    output: &OutputArgs,
) -> Result<(), String> {
    let rendered = match output.format {
        OutputFormat::Json => records_to_json(records)?,
        OutputFormat::Jsonl => records_to_jsonl(records)?,
        OutputFormat::Paf => {
            let mut text = String::new();
            for record in records {
                text.push_str(&record_to_paf(record)?);
            }
            text
        }
        OutputFormat::Sam => {
            let queries = pairs
                .iter()
                .map(|(query, _)| (query.id.clone(), query.sequence.clone()))
                .collect();
            records_to_sam(records, &queries)?
        }
        OutputFormat::Text => alignment_text(records),
    };
    write_text(&rendered, output.output.as_deref())
}

fn alignment_text(records: &[AlignmentRecord]) -> String {
    let mut output = String::new();
    for record in records {
        output.push_str(&format!(
            "{} vs {}\nmode: {}\nbackend: {}\nobjective: {}\n",
            record.pair.query.id,
            record.pair.target.id,
            record.problem.mode,
            record.producer.backend,
            record.alignment.objective.value,
        ));
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
    }
    output
}

fn write_reports(
    reports: &[VerificationReport],
    format: ReportFormat,
    path: Option<&Path>,
) -> Result<(), String> {
    let rendered = match format {
        ReportFormat::Json => json_collection(reports)?,
        ReportFormat::Jsonl => json_lines(reports)?,
        ReportFormat::Text => {
            let mut text = String::new();
            for report in reports {
                text.push_str(&format!(
                    "{} vs {}: {:?} ({:?})\n",
                    report.pair.query.id,
                    report.pair.target.id,
                    report.verification.status,
                    report.verification.level,
                ));
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
            }
            text
        }
    };
    write_text(&rendered, path)
}

fn write_comparisons(
    reports: &[ComparisonReport],
    format: ReportFormat,
    path: Option<&Path>,
) -> Result<(), String> {
    let rendered = match format {
        ReportFormat::Json => json_collection(reports)?,
        ReportFormat::Jsonl => json_lines(reports)?,
        ReportFormat::Text => reports
            .iter()
            .map(|report| {
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
            })
            .collect(),
    };
    write_text(&rendered, path)
}

fn verification_exit(reports: &[VerificationReport]) -> u8 {
    if reports
        .iter()
        .any(|report| report.verification.status == VerificationStatus::Invalid)
    {
        1
    } else if reports.iter().any(|report| {
        matches!(
            report.verification.status,
            VerificationStatus::Inconclusive | VerificationStatus::Error
        )
    }) {
        3
    } else {
        0
    }
}

fn json_collection<T: Serialize>(values: &[T]) -> Result<String, String> {
    let mut text = if values.len() == 1 {
        serde_json::to_string_pretty(&values[0]).map_err(|err| err.to_string())?
    } else {
        serde_json::to_string_pretty(values).map_err(|err| err.to_string())?
    };
    text.push('\n');
    Ok(text)
}

fn json_lines<T: Serialize>(values: &[T]) -> Result<String, String> {
    let mut text = String::new();
    for value in values {
        text.push_str(&serde_json::to_string(value).map_err(|err| err.to_string())?);
        text.push('\n');
    }
    Ok(text)
}

fn read_text(path: &Path) -> Result<String, String> {
    if path == Path::new("-") {
        let mut text = String::new();
        io::stdin()
            .read_to_string(&mut text)
            .map_err(|err| format!("failed to read stdin: {err}"))?;
        Ok(text)
    } else {
        fs::read_to_string(path).map_err(|err| format!("failed to read {}: {err}", path.display()))
    }
}

fn write_text(text: &str, path: Option<&Path>) -> Result<(), String> {
    match path {
        Some(path) => fs::write(path, text)
            .map_err(|err| format!("failed to write {}: {err}", path.display())),
        None => {
            print!("{text}");
            Ok(())
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
