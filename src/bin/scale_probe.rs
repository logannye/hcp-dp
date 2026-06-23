use std::{
    env,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use hcp_dp::{
    problems::{
        edit_distance::EditDistanceProblem, lcs::LcsProblem, nw_affine::NwAffineProblem,
        nw_align::NwProblem, semiglobal::SemiGlobalProblem, smith_waterman::SmithWatermanProblem,
    },
    HcpEngine,
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
    if let Some(scenario) = &options.scenario {
        eprintln!("Scenario filter: {scenario}");
    }

    let mut sys = System::new();
    let mut measurements = Vec::new();
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
            } else {
                return Err(format!("unrecognized argument '{arg}'"));
            }
        }

        Ok(Self {
            format,
            verify_limit,
            scenario,
        })
    }

    fn should_run(&self, scenario: &str) -> bool {
        self.scenario
            .as_deref()
            .is_none_or(|filter| filter == scenario)
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
  -h, --help                 Print this help
"
        );
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
    size: usize,
    wall_s: f64,
    rss_delta_bytes: u64,
    peak_rss_bytes: u64,
    status: VerificationStatus,
    detail: String,
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
    SIZES
        .iter()
        .map(|&len| {
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
    SIZES
        .iter()
        .map(|&len| {
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
    SIZES
        .iter()
        .map(|&len| {
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
    SIZES
        .iter()
        .map(|&len| {
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
    SIZES
        .iter()
        .map(|&len| {
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

fn run_semiglobal(options: &Options, sys: &mut System) -> Vec<Measurement> {
    const SIZES: &[usize] = &[64, 128, 256, 512, 1024, 2048];
    const MATCH_SCORE: i32 = 2;
    const MISMATCH_PENALTY: i32 = 1;
    const GAP_PENALTY: i32 = -2;
    SIZES
        .iter()
        .map(|&len| {
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
        size,
        wall_s,
        rss_delta_bytes: after.saturating_sub(before),
        peak_rss_bytes,
        status,
        detail,
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
            "{},{},{:.4},{},{},{}",
            measurement.scenario,
            measurement.size,
            measurement.wall_s,
            measurement.rss_delta_bytes,
            measurement.peak_rss_bytes,
            measurement.status.label()
        );
    }
}

fn write_csv(measurements: &[Measurement]) -> Result<(), String> {
    println!("scenario,size,wall_s,rss_delta_bytes,peak_rss_bytes,status,detail");
    for measurement in measurements {
        println!(
            "{},{},{:.6},{},{},{},{}",
            measurement.scenario,
            measurement.size,
            measurement.wall_s,
            measurement.rss_delta_bytes,
            measurement.peak_rss_bytes,
            measurement.status.label(),
            escape_csv(&measurement.detail)
        );
    }
    Ok(())
}

fn write_table(measurements: &[Measurement]) -> Result<(), String> {
    println!(
        "{:<26} {:>8} {:>10} {:>14} {:>14} {:<12} detail",
        "scenario", "size", "wall_s", "rss_delta_b", "peak_rss_b", "status"
    );
    for measurement in measurements {
        println!(
            "{:<26} {:>8} {:>10.4} {:>14} {:>14} {:<12} {}",
            measurement.scenario,
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
