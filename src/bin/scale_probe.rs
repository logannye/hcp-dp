use std::{
    env,
    process::Command,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use hcp_dp::{
    problems::{
        edit_distance::{
            distance_adaptive_banded, distance_linear_space, distance_myers_u64,
            trace_adaptive_banded, EditDistanceProblem,
        },
        lcs::LcsProblem,
        nw_affine::NwAffineProblem,
        nw_align::NwProblem,
        semiglobal::SemiGlobalProblem,
        smith_waterman::SmithWatermanProblem,
    },
    HcpEngine, HcpRunStats,
};
use serde::Serialize;
use sysinfo::{get_current_pid, ProcessRefreshKind, System};

const AFFINE_NEG_INF: i32 = i32::MIN / 4;

fn main() {
    let options = match Options::parse(env::args().skip(1)) {
        Ok(options) => options,
        Err(err) => {
            eprintln!("scale_probe: {err}");
            Options::print_help();
            std::process::exit(2);
        }
    };

    eprintln!("HCP-DP scale probe");
    eprintln!("Verified means: baseline objective matches and returned path realizes it.");
    eprintln!("Verification limit: {}", options.verify_limit);
    eprintln!("Mode: {}", options.mode.label());
    if let Some(scenario) = &options.scenario {
        eprintln!("Scenario filter: {scenario}");
    }

    let mut sys = System::new();
    let mut measurements = Vec::new();
    match options.mode {
        ProbeMode::Standard => {
            if options.should_run("lcs") {
                measurements.extend(run_lcs(&options, &mut sys));
            }
            if options.should_run("needleman_wunsch") {
                measurements.extend(run_nw(&options, &mut sys));
            }
            if options.should_run("smith_waterman") {
                measurements.extend(run_smith_waterman(&options, &mut sys));
            }
            if options.should_run("needleman_wunsch_affine") {
                measurements.extend(run_affine_nw(&options, &mut sys));
            }
            if options.should_run("edit_distance") {
                measurements.extend(run_edit_distance(&options, &mut sys));
            }
            if options.should_run("semiglobal") {
                measurements.extend(run_semiglobal(&options, &mut sys));
            }
        }
        ProbeMode::EditDistanceDeep => {
            measurements.extend(run_edit_distance_deep(&options, &mut sys));
        }
    }

    if measurements.is_empty() {
        eprintln!("scale_probe: no measurements selected");
        std::process::exit(2);
    }

    if measurements
        .iter()
        .any(|m| matches!(m.status, VerificationStatus::Failed))
    {
        print_summary(&measurements);
        let _ = options.format.write(&measurements);
        std::process::exit(1);
    }

    print_summary(&measurements);
    if let Err(err) = options.format.write(&measurements) {
        eprintln!("scale_probe output error: {err}");
        std::process::exit(1);
    }
}

struct Options {
    format: OutputFormat,
    verify_limit: usize,
    scenario: Option<String>,
    max_size: Option<usize>,
    mode: ProbeMode,
    engine: Option<DeepEngine>,
}

impl Options {
    fn parse<I, T>(mut args: I) -> Result<Self, String>
    where
        I: Iterator<Item = T>,
        T: Into<String>,
    {
        let mut format = OutputFormat::Csv;
        let mut verify_limit = 512;
        let mut scenario = None;
        let mut max_size = None;
        let mut mode = ProbeMode::Standard;
        let mut engine = None;

        while let Some(arg) = args.next() {
            let arg = arg.into();
            if arg == "-h" || arg == "--help" {
                Options::print_help();
                std::process::exit(0);
            } else if arg == "--format" {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value after --format".to_string())?
                    .into();
                format = OutputFormat::from_str(&value)?;
            } else if let Some(value) = arg.strip_prefix("--format=") {
                format = OutputFormat::from_str(value)?;
            } else if arg == "--verify-limit" {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value after --verify-limit".to_string())?
                    .into();
                verify_limit = parse_limit(&value)?;
            } else if let Some(value) = arg.strip_prefix("--verify-limit=") {
                verify_limit = parse_limit(value)?;
            } else if arg == "--scenario" {
                scenario = Some(
                    args.next()
                        .ok_or_else(|| "missing value after --scenario".to_string())?
                        .into(),
                );
            } else if let Some(value) = arg.strip_prefix("--scenario=") {
                scenario = Some(value.to_string());
            } else if arg == "--max-size" {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value after --max-size".to_string())?
                    .into();
                max_size = Some(parse_limit(&value)?);
            } else if let Some(value) = arg.strip_prefix("--max-size=") {
                max_size = Some(parse_limit(value)?);
            } else if arg == "--mode" {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value after --mode".to_string())?
                    .into();
                mode = ProbeMode::from_str(&value)?;
            } else if let Some(value) = arg.strip_prefix("--mode=") {
                mode = ProbeMode::from_str(value)?;
            } else if arg == "--engine" {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value after --engine".to_string())?
                    .into();
                engine = Some(DeepEngine::from_str(&value)?);
            } else if let Some(value) = arg.strip_prefix("--engine=") {
                engine = Some(DeepEngine::from_str(value)?);
            } else {
                return Err(format!("unrecognized argument '{arg}'"));
            }
        }

        Ok(Self {
            format,
            verify_limit,
            scenario,
            max_size,
            mode,
            engine,
        })
    }

    fn should_run(&self, scenario: &str) -> bool {
        self.scenario
            .as_deref()
            .is_none_or(|filter| filter == scenario)
    }

    fn sizes<'a>(&self, sizes: &'a [usize]) -> impl Iterator<Item = usize> + 'a {
        let max_size = self.max_size;
        sizes
            .iter()
            .copied()
            .filter(move |size| max_size.is_none_or(|limit| *size <= limit))
    }

    fn print_help() {
        println!(
            "\
Usage: cargo run --bin scale_probe -- [options]

Options:
  --format <csv|table|json>  Output format (default: csv)
  --verify-limit <N>         Largest size checked against full-table baselines (default: 512)
  --scenario <name>          Run only one scenario: lcs, needleman_wunsch,
                             smith_waterman, needleman_wunsch_affine,
                             edit_distance, semiglobal
  --max-size <N>             Skip scenario sizes larger than N
  --mode <standard|edit-distance-deep>
                             Probe mode (default: standard)
  --engine <hcp|hcp-linear|adaptive-banded-path|full-table|linear-space|adaptive-banded|myers-u64|edlib>
                             Engine filter for --mode edit-distance-deep
  -h, --help                 Print this help
"
        );
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ProbeMode {
    Standard,
    EditDistanceDeep,
}

impl ProbeMode {
    fn from_str(value: &str) -> Result<Self, String> {
        match value {
            "standard" => Ok(Self::Standard),
            "edit-distance-deep" => Ok(Self::EditDistanceDeep),
            other => Err(format!("unknown mode '{other}'")),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::EditDistanceDeep => "edit-distance-deep",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DeepEngine {
    Hcp,
    HcpLinear,
    AdaptiveBandedPath,
    FullTable,
    LinearSpace,
    AdaptiveBanded,
    MyersU64,
    Edlib,
}

impl DeepEngine {
    fn from_str(value: &str) -> Result<Self, String> {
        match value {
            "hcp" => Ok(Self::Hcp),
            "hcp-linear" => Ok(Self::HcpLinear),
            "adaptive-banded-path" => Ok(Self::AdaptiveBandedPath),
            "full-table" => Ok(Self::FullTable),
            "hirschberg" | "linear-space" => Ok(Self::LinearSpace),
            "adaptive-banded" => Ok(Self::AdaptiveBanded),
            "myers-u64" => Ok(Self::MyersU64),
            "edlib" => Ok(Self::Edlib),
            other => Err(format!("unknown engine '{other}'")),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Hcp => "hcp",
            Self::HcpLinear => "hcp-linear",
            Self::AdaptiveBandedPath => "adaptive-banded-path",
            Self::FullTable => "full-table",
            Self::LinearSpace => "linear-space",
            Self::AdaptiveBanded => "adaptive-banded",
            Self::MyersU64 => "myers-u64",
            Self::Edlib => "edlib",
        }
    }
}

fn parse_limit(value: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| "verify limit must be a non-negative integer".to_string())
}

#[derive(Clone, Copy)]
enum OutputFormat {
    Csv,
    Table,
    Json,
}

impl OutputFormat {
    fn from_str(value: &str) -> Result<Self, String> {
        match value {
            "csv" => Ok(Self::Csv),
            "table" => Ok(Self::Table),
            "json" => Ok(Self::Json),
            other => Err(format!("unknown format '{other}'")),
        }
    }

    fn write(self, measurements: &[Measurement]) -> Result<(), String> {
        match self {
            Self::Csv => write_csv(measurements),
            Self::Table => write_table(measurements),
            Self::Json => write_json(measurements),
        }
    }
}

#[derive(Serialize)]
struct Measurement {
    scenario: &'static str,
    case: &'static str,
    engine: &'static str,
    size: usize,
    query_len: usize,
    target_len: usize,
    wall_s: f64,
    rss_delta_bytes: u64,
    peak_rss_bytes: u64,
    status: VerificationStatus,
    detail: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    distance: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    path_score: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    path_len: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_distance: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    summary_build_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reconstruction_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    verification_ms: Option<f64>,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum VerificationStatus {
    Passed,
    Failed,
    NotChecked,
}

impl VerificationStatus {
    fn label(self) -> &'static str {
        match self {
            Self::Passed => "passed",
            Self::Failed => "failed",
            Self::NotChecked => "not_checked",
        }
    }
}

fn run_lcs(options: &Options, sys: &mut System) -> Vec<Measurement> {
    const SIZES: &[usize] = &[64, 128, 256, 512, 1024, 2048, 4096];
    options
        .sizes(SIZES)
        .map(|len| {
            measure("lcs", len, sys, || {
                let s = deterministic_dna(len);
                let t = deterministic_dna_offset(len, 1);
                let problem = LcsProblem::new(&s, &t);
                let (cost, path) = HcpEngine::new(problem.clone()).run();
                if len <= options.verify_limit {
                    let baseline = full_lcs_len(&s, &t);
                    let path_score = problem.score_path(&path);
                    if baseline == cost && path_score == Some(cost) {
                        (VerificationStatus::Passed, format!("cost={cost}"))
                    } else {
                        (
                            VerificationStatus::Failed,
                            format!("baseline={baseline}, cost={cost}, path_score={path_score:?}"),
                        )
                    }
                } else {
                    let path_score = problem.score_path(&path);
                    if path_score == Some(cost) {
                        (
                            VerificationStatus::NotChecked,
                            format!("cost={cost}, path_len={}", path.len()),
                        )
                    } else {
                        (
                            VerificationStatus::Failed,
                            format!("cost={cost}, path_score={path_score:?}"),
                        )
                    }
                }
            })
        })
        .collect()
}

fn run_nw(options: &Options, sys: &mut System) -> Vec<Measurement> {
    const SIZES: &[usize] = &[64, 128, 256, 512, 1024, 2048, 4096];
    const MATCH_SCORE: i32 = 2;
    const MISMATCH_PENALTY: i32 = 1;
    const GAP_PENALTY: i32 = -2;
    options
        .sizes(SIZES)
        .map(|len| {
            measure("needleman_wunsch", len, sys, || {
                let s = deterministic_dna(len);
                let t = deterministic_dna_offset(len, 2);
                let problem = NwProblem::new(&s, &t, MATCH_SCORE, MISMATCH_PENALTY, GAP_PENALTY);
                let (cost, path) = HcpEngine::new(problem.clone()).run();
                if len <= options.verify_limit {
                    let baseline =
                        full_nw_score(&s, &t, MATCH_SCORE, MISMATCH_PENALTY, GAP_PENALTY);
                    let path_score = problem.score_path(&path);
                    if baseline == cost && path_score == Some(cost) {
                        (VerificationStatus::Passed, format!("cost={cost}"))
                    } else {
                        (
                            VerificationStatus::Failed,
                            format!("baseline={baseline}, cost={cost}, path_score={path_score:?}"),
                        )
                    }
                } else {
                    let path_score = problem.score_path(&path);
                    if path_score == Some(cost) {
                        (
                            VerificationStatus::NotChecked,
                            format!("cost={cost}, path_len={}", path.len()),
                        )
                    } else {
                        (
                            VerificationStatus::Failed,
                            format!("cost={cost}, path_score={path_score:?}"),
                        )
                    }
                }
            })
        })
        .collect()
}

fn run_smith_waterman(options: &Options, sys: &mut System) -> Vec<Measurement> {
    const SIZES: &[usize] = &[64, 128, 256, 512, 1024, 2048];
    const MATCH_SCORE: i32 = 2;
    const MISMATCH_PENALTY: i32 = 1;
    const GAP_PENALTY: i32 = -2;
    options
        .sizes(SIZES)
        .map(|len| {
            measure("smith_waterman", len, sys, || {
                let s = deterministic_dna(len);
                let t = deterministic_dna_offset(len, 1);
                let problem =
                    SmithWatermanProblem::new(&s, &t, MATCH_SCORE, MISMATCH_PENALTY, GAP_PENALTY);
                let (cost, path) = HcpEngine::new(problem.clone()).run();
                if len <= options.verify_limit {
                    let baseline =
                        full_sw_score(&s, &t, MATCH_SCORE, MISMATCH_PENALTY, GAP_PENALTY);
                    let path_score = problem.score_path(&path);
                    if baseline == cost && path_score == Some(cost) {
                        (VerificationStatus::Passed, format!("cost={cost}"))
                    } else {
                        (
                            VerificationStatus::Failed,
                            format!("baseline={baseline}, cost={cost}, path_score={path_score:?}"),
                        )
                    }
                } else {
                    let path_score = problem.score_path(&path);
                    if path_score == Some(cost) {
                        (
                            VerificationStatus::NotChecked,
                            format!("cost={cost}, path_len={}", path.len()),
                        )
                    } else {
                        (
                            VerificationStatus::Failed,
                            format!("cost={cost}, path_score={path_score:?}"),
                        )
                    }
                }
            })
        })
        .collect()
}

fn run_affine_nw(options: &Options, sys: &mut System) -> Vec<Measurement> {
    const SIZES: &[usize] = &[32, 64, 128, 256, 512];
    const MATCH_SCORE: i32 = 2;
    const MISMATCH_PENALTY: i32 = 1;
    const GAP_OPEN: i32 = -3;
    const GAP_EXTEND: i32 = -1;
    options
        .sizes(SIZES)
        .map(|len| {
            measure("needleman_wunsch_affine", len, sys, || {
                let s = deterministic_dna(len);
                let t = deterministic_dna_offset(len, 3);
                let problem = NwAffineProblem::new(
                    &s,
                    &t,
                    MATCH_SCORE,
                    MISMATCH_PENALTY,
                    GAP_OPEN,
                    GAP_EXTEND,
                );
                let (cost, path) = HcpEngine::new(problem.clone()).run();
                if len <= options.verify_limit {
                    let baseline = full_affine_nw_score(
                        &s,
                        &t,
                        MATCH_SCORE,
                        MISMATCH_PENALTY,
                        GAP_OPEN,
                        GAP_EXTEND,
                    );
                    let path_score = problem.score_path(&path);
                    if baseline == cost && path_score == Some(cost) {
                        (VerificationStatus::Passed, format!("cost={cost}"))
                    } else {
                        (
                            VerificationStatus::Failed,
                            format!("baseline={baseline}, cost={cost}, path_score={path_score:?}"),
                        )
                    }
                } else {
                    let path_score = problem.score_path(&path);
                    if path_score == Some(cost) {
                        (
                            VerificationStatus::NotChecked,
                            format!("cost={cost}, path_len={}", path.len()),
                        )
                    } else {
                        (
                            VerificationStatus::Failed,
                            format!("cost={cost}, path_score={path_score:?}"),
                        )
                    }
                }
            })
        })
        .collect()
}

fn run_edit_distance(options: &Options, sys: &mut System) -> Vec<Measurement> {
    const SIZES: &[usize] = &[64, 128, 256, 512, 1024, 2048, 4096];
    options
        .sizes(SIZES)
        .map(|len| {
            measure("edit_distance", len, sys, || {
                let s = deterministic_dna(len);
                let t = deterministic_dna_offset(len, 1);
                let problem = EditDistanceProblem::new(&s, &t);
                let (distance, path) = HcpEngine::new(problem.clone()).run();
                if len <= options.verify_limit {
                    let baseline = full_edit_distance(&s, &t);
                    let path_score = problem.score_path(&path);
                    if baseline == distance && path_score == Some(distance) {
                        (VerificationStatus::Passed, format!("distance={distance}"))
                    } else {
                        (
                            VerificationStatus::Failed,
                            format!(
                                "baseline={baseline}, distance={distance}, path_score={path_score:?}"
                            ),
                        )
                    }
                } else {
                    let path_score = problem.score_path(&path);
                    if path_score == Some(distance) {
                        (
                            VerificationStatus::NotChecked,
                            format!("distance={distance}, path_len={}", path.len()),
                        )
                    } else {
                        (
                            VerificationStatus::Failed,
                            format!("distance={distance}, path_score={path_score:?}"),
                        )
                    }
                }
            })
        })
        .collect()
}

struct EditDistanceCase {
    name: &'static str,
    query: Vec<u8>,
    target: Vec<u8>,
}

struct EditDistanceOutcome {
    status: VerificationStatus,
    detail: String,
    distance: Option<u32>,
    path_score: Option<u32>,
    path_len: Option<usize>,
    expected_distance: Option<u32>,
    stats: Option<HcpRunStats>,
    verification_ms: Option<f64>,
}

fn run_edit_distance_deep(options: &Options, sys: &mut System) -> Vec<Measurement> {
    let engines = deep_engines(options.engine);
    let mut measurements = Vec::new();
    for case in edit_distance_cases() {
        let size = case.query.len().max(case.target.len());
        if options.max_size.is_some_and(|max_size| size > max_size) {
            continue;
        }
        let expected = edit_distance_linear_space(&case.query, &case.target);
        for engine in &engines {
            measurements.push(measure_edit_distance_engine(&case, *engine, expected, sys));
        }
    }
    measurements
}

fn deep_engines(filter: Option<DeepEngine>) -> Vec<DeepEngine> {
    match filter {
        Some(engine) => vec![engine],
        None => vec![
            DeepEngine::Hcp,
            DeepEngine::HcpLinear,
            DeepEngine::AdaptiveBandedPath,
            DeepEngine::FullTable,
            DeepEngine::LinearSpace,
            DeepEngine::AdaptiveBanded,
            DeepEngine::MyersU64,
            DeepEngine::Edlib,
        ],
    }
}

fn edit_distance_cases() -> Vec<EditDistanceCase> {
    vec![
        EditDistanceCase {
            name: "small_regression",
            query: b"kitten".to_vec(),
            target: b"sitting".to_vec(),
        },
        EditDistanceCase {
            name: "short_myers_window",
            query: deterministic_dna(64),
            target: deterministic_dna_offset(64, 1),
        },
        EditDistanceCase {
            name: "exact_match",
            query: deterministic_dna(256),
            target: deterministic_dna(256),
        },
        EditDistanceCase {
            name: "long_low_edit",
            query: deterministic_dna(2048),
            target: low_edit_variant(2048),
        },
        EditDistanceCase {
            name: "all_mismatch",
            query: vec![b'A'; 192],
            target: vec![b'T'; 192],
        },
        EditDistanceCase {
            name: "single_long_insertion",
            query: b"ACGTACGT".to_vec(),
            target: [b"ACGT".as_slice(), vec![b'A'; 192].as_slice(), b"ACGT"].concat(),
        },
        EditDistanceCase {
            name: "single_long_deletion",
            query: [b"ACGT".as_slice(), vec![b'C'; 192].as_slice(), b"ACGT"].concat(),
            target: b"ACGTACGT".to_vec(),
        },
        EditDistanceCase {
            name: "homopolymer",
            query: vec![b'A'; 256],
            target: vec![b'A'; 128],
        },
        EditDistanceCase {
            name: "repeats",
            query: repeat_pattern(b"AT", 160),
            target: repeat_pattern(b"TA", 160),
        },
        EditDistanceCase {
            name: "skewed_lengths",
            query: deterministic_dna(96),
            target: deterministic_dna_offset(320, 2),
        },
        EditDistanceCase {
            name: "random_dna",
            query: deterministic_dna(384),
            target: deterministic_dna_offset(384, 3),
        },
    ]
}

fn repeat_pattern(pattern: &[u8], len: usize) -> Vec<u8> {
    (0..len).map(|idx| pattern[idx % pattern.len()]).collect()
}

fn low_edit_variant(len: usize) -> Vec<u8> {
    let mut data = deterministic_dna(len);
    for idx in [17usize, 513, 1027, 1531] {
        if let Some(byte) = data.get_mut(idx) {
            *byte = match *byte {
                b'A' => b'C',
                b'C' => b'G',
                b'G' => b'T',
                _ => b'A',
            };
        }
    }
    data
}

fn measure_edit_distance_engine(
    case: &EditDistanceCase,
    engine: DeepEngine,
    expected: u32,
    sys: &mut System,
) -> Measurement {
    let before = rss_bytes(sys);
    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicU64::new(before));
    let sampler = spawn_rss_sampler(Arc::clone(&stop), Arc::clone(&peak));
    let start = Instant::now();
    let outcome = run_edit_distance_engine(case, engine, expected);
    let wall_s = start.elapsed().as_secs_f64();
    stop.store(true, Ordering::Relaxed);
    let _ = sampler.join();
    let after = rss_bytes(sys);
    let peak_rss_bytes = peak.load(Ordering::Relaxed).max(after).max(before);
    Measurement {
        scenario: "edit_distance_deep",
        case: case.name,
        engine: engine.label(),
        size: case.query.len().max(case.target.len()),
        query_len: case.query.len(),
        target_len: case.target.len(),
        wall_s,
        rss_delta_bytes: after.saturating_sub(before),
        peak_rss_bytes,
        status: outcome.status,
        detail: outcome.detail,
        distance: outcome.distance,
        path_score: outcome.path_score,
        path_len: outcome.path_len,
        expected_distance: outcome.expected_distance,
        summary_build_ms: outcome.stats.map(|stats| stats.summary_build_ms),
        reconstruction_ms: outcome.stats.map(|stats| stats.reconstruction_ms),
        verification_ms: outcome.verification_ms,
    }
}

fn run_edit_distance_engine(
    case: &EditDistanceCase,
    engine: DeepEngine,
    expected: u32,
) -> EditDistanceOutcome {
    match engine {
        DeepEngine::Hcp => {
            let problem = EditDistanceProblem::new(&case.query, &case.target);
            let (distance, path, stats) = HcpEngine::new(problem.clone()).run_with_stats();
            let verify_start = Instant::now();
            let path_score = problem.score_path(&path);
            let verification_ms = verify_start.elapsed().as_secs_f64() * 1000.0;
            let passed = distance == expected && path_score == Some(distance);
            EditDistanceOutcome {
                status: if passed {
                    VerificationStatus::Passed
                } else {
                    VerificationStatus::Failed
                },
                detail: format!("distance={distance}, path_len={}", path.len()),
                distance: Some(distance),
                path_score,
                path_len: Some(path.len()),
                expected_distance: Some(expected),
                stats: Some(stats),
                verification_ms: Some(verification_ms),
            }
        }
        DeepEngine::HcpLinear => {
            let problem = EditDistanceProblem::new(&case.query, &case.target);
            let (distance, path, stats) = HcpEngine::linear_space(problem.clone()).run_with_stats();
            let verify_start = Instant::now();
            let path_score = problem.score_path(&path);
            let verification_ms = verify_start.elapsed().as_secs_f64() * 1000.0;
            let passed = distance == expected && path_score == Some(distance);
            EditDistanceOutcome {
                status: if passed {
                    VerificationStatus::Passed
                } else {
                    VerificationStatus::Failed
                },
                detail: format!("distance={distance}, block_size=1, path_len={}", path.len()),
                distance: Some(distance),
                path_score,
                path_len: Some(path.len()),
                expected_distance: Some(expected),
                stats: Some(stats),
                verification_ms: Some(verification_ms),
            }
        }
        DeepEngine::AdaptiveBandedPath => {
            let problem = EditDistanceProblem::new(&case.query, &case.target);
            let trace_start = Instant::now();
            let trace = trace_adaptive_banded(&case.query, &case.target);
            let trace_ms = trace_start.elapsed().as_secs_f64() * 1000.0;
            let verify_start = Instant::now();
            let path_score = problem.score_path(&trace.path);
            let verification_ms = verify_start.elapsed().as_secs_f64() * 1000.0;
            let passed = trace.distance == expected && path_score == Some(trace.distance);
            EditDistanceOutcome {
                status: if passed {
                    VerificationStatus::Passed
                } else {
                    VerificationStatus::Failed
                },
                detail: format!(
                    "distance={}, band={}, path_len={}",
                    trace.distance,
                    trace.band,
                    trace.path.len()
                ),
                distance: Some(trace.distance),
                path_score,
                path_len: Some(trace.path.len()),
                expected_distance: Some(expected),
                stats: Some(HcpRunStats {
                    summary_build_ms: 0.0,
                    reconstruction_ms: trace_ms,
                }),
                verification_ms: Some(verification_ms),
            }
        }
        DeepEngine::FullTable => {
            let distance = edit_distance_full_table(&case.query, &case.target);
            edit_distance_baseline_outcome("full_table", distance, expected)
        }
        DeepEngine::LinearSpace => {
            let distance = distance_linear_space(&case.query, &case.target);
            edit_distance_baseline_outcome("linear_space", distance, expected)
        }
        DeepEngine::AdaptiveBanded => {
            let distance = distance_adaptive_banded(&case.query, &case.target);
            edit_distance_baseline_outcome("adaptive_banded", distance, expected)
        }
        DeepEngine::MyersU64 => match distance_myers_u64(&case.query, &case.target) {
            Some(distance) => edit_distance_baseline_outcome("myers_u64", distance, expected),
            None => EditDistanceOutcome {
                status: VerificationStatus::NotChecked,
                detail: "pattern length exceeds 64 symbols".to_string(),
                distance: None,
                path_score: None,
                path_len: None,
                expected_distance: Some(expected),
                stats: None,
                verification_ms: None,
            },
        },
        DeepEngine::Edlib => match edlib_distance(&case.query, &case.target) {
            Ok(distance) => edit_distance_baseline_outcome("edlib", distance, expected),
            Err(err) => EditDistanceOutcome {
                status: VerificationStatus::NotChecked,
                detail: err,
                distance: None,
                path_score: None,
                path_len: None,
                expected_distance: Some(expected),
                stats: None,
                verification_ms: None,
            },
        },
    }
}

fn edit_distance_baseline_outcome(
    label: &'static str,
    distance: u32,
    expected: u32,
) -> EditDistanceOutcome {
    EditDistanceOutcome {
        status: if distance == expected {
            VerificationStatus::Passed
        } else {
            VerificationStatus::Failed
        },
        detail: format!("{label}_distance={distance}"),
        distance: Some(distance),
        path_score: None,
        path_len: None,
        expected_distance: Some(expected),
        stats: None,
        verification_ms: None,
    }
}

fn edit_distance_linear_space(s: &[u8], t: &[u8]) -> u32 {
    distance_linear_space(s, t)
}

fn edit_distance_full_table(s: &[u8], t: &[u8]) -> u32 {
    let cols = t.len() + 1;
    let mut dp = vec![0u32; (s.len() + 1) * cols];
    for row in 1..=s.len() {
        dp[row * cols] = row as u32;
    }
    for (col, cell) in dp.iter_mut().take(cols).enumerate().skip(1) {
        *cell = col as u32;
    }
    for row in 1..=s.len() {
        for col in 1..=t.len() {
            let subst = u32::from(s[row - 1] != t[col - 1]);
            let diag = dp[(row - 1) * cols + col - 1] + subst;
            let delete = dp[(row - 1) * cols + col] + 1;
            let insert = dp[row * cols + col - 1] + 1;
            dp[row * cols + col] = diag.min(delete).min(insert);
        }
    }
    dp[s.len() * cols + t.len()]
}

fn edlib_distance(s: &[u8], t: &[u8]) -> Result<u32, String> {
    let query = std::str::from_utf8(s).map_err(|err| err.to_string())?;
    let target = std::str::from_utf8(t).map_err(|err| err.to_string())?;
    let output = Command::new("python3")
        .arg("-c")
        .arg(
            "import edlib,sys; print(edlib.align(sys.argv[1], sys.argv[2], task='distance')['editDistance'])",
        )
        .arg(query)
        .arg(target)
        .output()
        .map_err(|err| format!("edlib unavailable: {err}"))?;
    if !output.status.success() {
        return Err("edlib unavailable or failed".to_string());
    }
    let text = String::from_utf8(output.stdout).map_err(|err| err.to_string())?;
    text.trim()
        .parse::<u32>()
        .map_err(|err| format!("invalid edlib distance: {err}"))
}

fn run_semiglobal(options: &Options, sys: &mut System) -> Vec<Measurement> {
    const SIZES: &[usize] = &[64, 128, 256, 512, 1024, 2048];
    const MATCH_SCORE: i32 = 2;
    const MISMATCH_PENALTY: i32 = 1;
    const GAP_PENALTY: i32 = -2;
    options
        .sizes(SIZES)
        .map(|len| {
            measure("semiglobal", len, sys, || {
                let s = deterministic_dna(len);
                let t = deterministic_dna_offset(len + len / 4, 2);
                let problem =
                    SemiGlobalProblem::new(&s, &t, MATCH_SCORE, MISMATCH_PENALTY, GAP_PENALTY);
                let (cost, path) = HcpEngine::new(problem.clone()).run();
                if len <= options.verify_limit {
                    let baseline =
                        full_semiglobal_score(&s, &t, MATCH_SCORE, MISMATCH_PENALTY, GAP_PENALTY);
                    let path_score = problem.score_path(&path);
                    if baseline == cost && path_score == Some(cost) {
                        (VerificationStatus::Passed, format!("cost={cost}"))
                    } else {
                        (
                            VerificationStatus::Failed,
                            format!("baseline={baseline}, cost={cost}, path_score={path_score:?}"),
                        )
                    }
                } else {
                    let path_score = problem.score_path(&path);
                    if path_score == Some(cost) {
                        (
                            VerificationStatus::NotChecked,
                            format!("cost={cost}, path_len={}", path.len()),
                        )
                    } else {
                        (
                            VerificationStatus::Failed,
                            format!("cost={cost}, path_score={path_score:?}"),
                        )
                    }
                }
            })
        })
        .collect()
}

fn measure<F>(scenario: &'static str, size: usize, sys: &mut System, run: F) -> Measurement
where
    F: FnOnce() -> (VerificationStatus, String),
{
    let before = rss_bytes(sys);
    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicU64::new(before));
    let sampler = spawn_rss_sampler(Arc::clone(&stop), Arc::clone(&peak));
    let start = Instant::now();
    let (status, detail) = run();
    let wall_s = start.elapsed().as_secs_f64();
    stop.store(true, Ordering::Relaxed);
    let _ = sampler.join();
    let after = rss_bytes(sys);
    let peak_rss_bytes = peak.load(Ordering::Relaxed).max(after).max(before);
    Measurement {
        scenario,
        case: "deterministic",
        engine: "hcp",
        size,
        query_len: size,
        target_len: size,
        wall_s,
        rss_delta_bytes: after.saturating_sub(before),
        peak_rss_bytes,
        status,
        detail,
        distance: None,
        path_score: None,
        path_len: None,
        expected_distance: None,
        summary_build_ms: None,
        reconstruction_ms: None,
        verification_ms: None,
    }
}

fn spawn_rss_sampler(stop: Arc<AtomicBool>, peak: Arc<AtomicU64>) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut sys = System::new();
        while !stop.load(Ordering::Relaxed) {
            let current = rss_bytes(&mut sys);
            let mut observed = peak.load(Ordering::Relaxed);
            while current > observed {
                match peak.compare_exchange_weak(
                    observed,
                    current,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(next) => observed = next,
                }
            }
            thread::sleep(Duration::from_millis(1));
        }
    })
}

fn rss_bytes(sys: &mut System) -> u64 {
    sys.refresh_processes_specifics(ProcessRefreshKind::new().with_memory());
    get_current_pid()
        .ok()
        .and_then(|pid| sys.process(pid).map(|process| process.memory()))
        .unwrap_or(0)
}

fn deterministic_dna(len: usize) -> Vec<u8> {
    const ALPHABET: &[u8] = b"ACGT";
    (0..len).map(|i| ALPHABET[(i * 7 + 3) % 4]).collect()
}

fn deterministic_dna_offset(len: usize, offset: usize) -> Vec<u8> {
    const ALPHABET: &[u8] = b"ACGT";
    (0..len).map(|i| ALPHABET[(i * 5 + offset) % 4]).collect()
}

fn full_lcs_len(s: &[u8], t: &[u8]) -> u32 {
    let mut prev = vec![0; t.len() + 1];
    let mut curr = vec![0; t.len() + 1];
    for &a in s {
        curr[0] = 0;
        for j in 1..=t.len() {
            let diag = prev[j - 1] + u32::from(a == t[j - 1]);
            curr[j] = prev[j].max(curr[j - 1]).max(diag);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[t.len()]
}

fn full_nw_score(s: &[u8], t: &[u8], match_score: i32, mismatch_penalty: i32, gap: i32) -> i32 {
    let mut prev = Vec::with_capacity(t.len() + 1);
    let mut curr = vec![0; t.len() + 1];
    prev.push(0);
    for j in 1..=t.len() {
        prev.push(prev[j - 1] + gap);
    }
    for &a in s {
        curr[0] = prev[0] + gap;
        for j in 1..=t.len() {
            let pair = if a == t[j - 1] {
                match_score
            } else {
                -mismatch_penalty
            };
            let diag = prev[j - 1] + pair;
            let up = prev[j] + gap;
            let left = curr[j - 1] + gap;
            curr[j] = diag.max(up).max(left);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[t.len()]
}

fn full_sw_score(s: &[u8], t: &[u8], match_score: i32, mismatch_penalty: i32, gap: i32) -> i32 {
    let mut prev = vec![0; t.len() + 1];
    let mut curr = vec![0; t.len() + 1];
    let mut best = 0;
    for &a in s {
        curr[0] = 0;
        for j in 1..=t.len() {
            let pair = if a == t[j - 1] {
                match_score
            } else {
                -mismatch_penalty
            };
            let diag = prev[j - 1] + pair;
            let up = prev[j] + gap;
            let left = curr[j - 1] + gap;
            curr[j] = 0.max(diag).max(up).max(left);
            best = best.max(curr[j]);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    best
}

fn full_edit_distance(s: &[u8], t: &[u8]) -> u32 {
    let mut prev: Vec<u32> = (0..=t.len() as u32).collect();
    let mut curr = vec![0; t.len() + 1];
    for (row, &a) in s.iter().enumerate() {
        curr[0] = row as u32 + 1;
        for col in 1..=t.len() {
            let subst = u32::from(a != t[col - 1]);
            let diag = prev[col - 1] + subst;
            let delete = prev[col] + 1;
            let insert = curr[col - 1] + 1;
            curr[col] = diag.min(delete).min(insert);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[t.len()]
}

fn full_semiglobal_score(
    s: &[u8],
    t: &[u8],
    match_score: i32,
    mismatch_penalty: i32,
    gap: i32,
) -> i32 {
    let mut prev = vec![0; t.len() + 1];
    let mut curr = vec![0; t.len() + 1];
    for &a in s {
        curr[0] = prev[0] + gap;
        for col in 1..=t.len() {
            let pair = if a == t[col - 1] {
                match_score
            } else {
                -mismatch_penalty
            };
            let diag = prev[col - 1] + pair;
            let up = prev[col] + gap;
            let left = curr[col - 1] + gap;
            curr[col] = diag.max(up).max(left);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev.into_iter().max().unwrap_or(0)
}

fn full_affine_nw_score(
    s: &[u8],
    t: &[u8],
    match_score: i32,
    mismatch_penalty: i32,
    gap_open: i32,
    gap_extend: i32,
) -> i32 {
    let mut prev_m = vec![AFFINE_NEG_INF; t.len() + 1];
    let mut prev_gap_t = vec![AFFINE_NEG_INF; t.len() + 1];
    let mut prev_gap_s = vec![AFFINE_NEG_INF; t.len() + 1];
    prev_m[0] = 0;
    for j in 1..=t.len() {
        prev_gap_s[j] = affine_best_gap(
            prev_m[j - 1],
            prev_gap_t[j - 1],
            prev_gap_s[j - 1],
            false,
            gap_open,
            gap_extend,
        );
    }

    for &a in s {
        let mut curr_m = vec![AFFINE_NEG_INF; t.len() + 1];
        let mut curr_gap_t = vec![AFFINE_NEG_INF; t.len() + 1];
        let mut curr_gap_s = vec![AFFINE_NEG_INF; t.len() + 1];

        curr_gap_t[0] = affine_best_gap(
            prev_m[0],
            prev_gap_t[0],
            prev_gap_s[0],
            true,
            gap_open,
            gap_extend,
        );

        for j in 1..=t.len() {
            let pair = if a == t[j - 1] {
                match_score
            } else {
                -mismatch_penalty
            };
            curr_m[j] = affine_add(prev_m[j - 1], pair)
                .max(affine_add(prev_gap_t[j - 1], pair))
                .max(affine_add(prev_gap_s[j - 1], pair));
            curr_gap_t[j] = affine_best_gap(
                prev_m[j],
                prev_gap_t[j],
                prev_gap_s[j],
                true,
                gap_open,
                gap_extend,
            );
            curr_gap_s[j] = affine_best_gap(
                curr_m[j - 1],
                curr_gap_t[j - 1],
                curr_gap_s[j - 1],
                false,
                gap_open,
                gap_extend,
            );
        }

        prev_m = curr_m;
        prev_gap_t = curr_gap_t;
        prev_gap_s = curr_gap_s;
    }

    prev_m[t.len()]
        .max(prev_gap_t[t.len()])
        .max(prev_gap_s[t.len()])
}

fn affine_best_gap(
    match_score: i32,
    gap_in_t: i32,
    gap_in_s: i32,
    target_gap_in_t: bool,
    gap_open: i32,
    gap_extend: i32,
) -> i32 {
    let continue_score = if target_gap_in_t { gap_in_t } else { gap_in_s };
    affine_add(match_score, gap_open + gap_extend).max(affine_add(continue_score, gap_extend))
}

fn affine_add(base: i32, delta: i32) -> i32 {
    if base <= AFFINE_NEG_INF / 2 {
        AFFINE_NEG_INF
    } else {
        base.saturating_add(delta).max(AFFINE_NEG_INF)
    }
}

fn print_summary(measurements: &[Measurement]) {
    for measurement in measurements {
        eprintln!(
            "{},{},{},{},{:.4},{},{},{}",
            measurement.scenario,
            measurement.case,
            measurement.engine,
            measurement.size,
            measurement.wall_s,
            measurement.rss_delta_bytes,
            measurement.peak_rss_bytes,
            measurement.status.label()
        );
    }
}

fn write_csv(measurements: &[Measurement]) -> Result<(), String> {
    println!("scenario,case,engine,size,query_len,target_len,wall_s,rss_delta_bytes,peak_rss_bytes,status,distance,path_score,path_len,expected_distance,summary_build_ms,reconstruction_ms,verification_ms,detail");
    for measurement in measurements {
        println!(
            "{},{},{},{},{},{},{:.6},{},{},{},{},{},{},{},{},{},{},{}",
            measurement.scenario,
            measurement.case,
            measurement.engine,
            measurement.size,
            measurement.query_len,
            measurement.target_len,
            measurement.wall_s,
            measurement.rss_delta_bytes,
            measurement.peak_rss_bytes,
            measurement.status.label(),
            optional_u32(measurement.distance),
            optional_u32(measurement.path_score),
            optional_usize(measurement.path_len),
            optional_u32(measurement.expected_distance),
            optional_f64(measurement.summary_build_ms),
            optional_f64(measurement.reconstruction_ms),
            optional_f64(measurement.verification_ms),
            escape_csv(&measurement.detail)
        );
    }
    Ok(())
}

fn write_table(measurements: &[Measurement]) -> Result<(), String> {
    println!(
        "{:<20} {:<22} {:<12} {:>8} {:>10} {:>14} {:>14} {:<12} detail",
        "scenario", "case", "engine", "size", "wall_s", "rss_delta_b", "peak_rss_b", "status"
    );
    for measurement in measurements {
        println!(
            "{:<20} {:<22} {:<12} {:>8} {:>10.4} {:>14} {:>14} {:<12} {}",
            measurement.scenario,
            measurement.case,
            measurement.engine,
            measurement.size,
            measurement.wall_s,
            measurement.rss_delta_bytes,
            measurement.peak_rss_bytes,
            measurement.status.label(),
            measurement.detail
        );
    }
    Ok(())
}

fn write_json(measurements: &[Measurement]) -> Result<(), String> {
    let json = serde_json::to_string_pretty(measurements).map_err(|err| err.to_string())?;
    println!("{json}");
    Ok(())
}

fn escape_csv(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn optional_u32(value: Option<u32>) -> String {
    value.map_or_else(String::new, |value| value.to_string())
}

fn optional_usize(value: Option<usize>) -> String {
    value.map_or_else(String::new, |value| value.to_string())
}

fn optional_f64(value: Option<f64>) -> String {
    value.map_or_else(String::new, |value| format!("{value:.6}"))
}
