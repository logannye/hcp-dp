use std::env;
use std::time::Instant;

use hcp_dp::problems::{
    dag_sp::DagLayered,
    lcs::LcsProblem,
    nw_affine::NwAffineProblem,
    nw_align::NwProblem,
    viterbi::{Hmm, ViterbiProblem},
};
use hcp_dp::HcpEngine;
use sysinfo::{get_current_pid, ProcessRefreshKind, System};

fn main() {
    let options = match Options::parse(env::args().skip(1)) {
        Ok(opts) => opts,
        Err(err) => {
            eprintln!("scale_probe: {err}");
            Options::print_help();
            std::process::exit(2);
        }
    };

    // Print header explaining the test suite
    eprintln!("\n{}", "=".repeat(80));
    eprintln!("HCP-DP Scaling Probe: Performance and Correctness Testing");
    eprintln!("{}", "=".repeat(80));
    eprintln!();
    eprintln!("This script tests the Height-Compressed Dynamic Programming engine");
    eprintln!("across multiple problem types and input sizes to verify:");
    eprintln!("  • Correctness: Results match full-table DP baselines (up to size {})", options.verify_limit);
    eprintln!("  • Performance: Wall-clock time and memory usage scale appropriately");
    eprintln!("  • Scalability: Engine handles large inputs (up to 65536 elements)");
    eprintln!();
    eprintln!("Metrics explained:");
    eprintln!("  • wall_s: Wall-clock time in seconds (lower is better)");
    eprintln!("  • rss_delta_kib: Memory delta in KiB (measures memory efficiency)");
    eprintln!("  • status: 'passed' = matches baseline, 'not_checked' = too large to verify");
    eprintln!();
    eprintln!("{}", "=".repeat(80));
    eprintln!();

    let mut sys = System::new();
    let mut measurements = Vec::new();

    eprintln!("[1/5] Testing Longest Common Subsequence (LCS)...");
    eprintln!("      Testing sequence alignment with increasing input sizes.");
    measurements.extend(run_lcs(&options, &mut sys));
    eprintln!();

    eprintln!("[2/5] Testing Needleman-Wunsch Linear Gap Alignment...");
    eprintln!("      Testing global sequence alignment with linear gap penalties.");
    measurements.extend(run_nw_linear(&options, &mut sys));
    eprintln!();

    eprintln!("[3/5] Testing Needleman-Wunsch Affine Gap Alignment...");
    eprintln!("      Testing global sequence alignment with affine gap penalties.");
    measurements.extend(run_nw_affine(&options, &mut sys));
    eprintln!();

    eprintln!("[4/5] Testing Viterbi Decoding (HMM)...");
    eprintln!("      Testing Hidden Markov Model state sequence decoding.");
    measurements.extend(run_viterbi(&options, &mut sys));
    eprintln!();

    eprintln!("[5/5] Testing Layered DAG Shortest Path...");
    eprintln!("      Testing shortest path computation in layered directed acyclic graphs.");
    measurements.extend(run_dag_shortest_path(&options, &mut sys));
    eprintln!();

    // Print detailed summary
    print_summary(&measurements, &options);

    // Write structured output
    if let Err(err) = options.format.write(&measurements) {
        eprintln!("scale_probe output error: {err}");
        std::process::exit(1);
    }
}

struct Options {
    format: OutputFormat,
    verify_limit: usize,
}

impl Options {
    fn parse<I, T>(mut args: I) -> Result<Self, String>
    where
        I: Iterator<Item = T>,
        T: Into<String>,
    {
        let mut format = OutputFormat::Csv;
        let mut verify_limit = 512usize;

        while let Some(arg) = args.next() {
            let arg = arg.into();
            if arg == "--help" || arg == "-h" {
                Options::print_help();
                std::process::exit(0);
            } else if let Some(value) = arg.strip_prefix("--format=") {
                format = OutputFormat::from_str(value)?;
            } else if arg == "--format" {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value after --format".to_string())?
                    .into();
                format = OutputFormat::from_str(&value)?;
            } else if let Some(value) = arg.strip_prefix("--verify-limit=") {
                verify_limit = value
                    .parse::<usize>()
                    .map_err(|_| "verify limit must be a positive integer".to_string())?;
            } else if arg == "--verify-limit" {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value after --verify-limit".to_string())?
                    .into();
                verify_limit = value
                    .parse::<usize>()
                    .map_err(|_| "verify limit must be a positive integer".to_string())?;
            } else {
                return Err(format!("unrecognized argument '{arg}'"));
            }
        }

        Ok(Self {
            format,
            verify_limit,
        })
    }

    fn print_help() {
        println!(
            "\
Usage: cargo run --bin scale_probe [-- <options>]

Options:
  --format <csv|table|json>     Output format (default: csv)
  --verify-limit <N>            Maximum sequence length (or layer count) to verify via baseline (default: 512)
  -h, --help                    Print this help message

Examples:
  cargo run --bin scale_probe
  cargo run --bin scale_probe -- --format table --verify-limit 256
"
        );
    }
}

#[derive(Copy, Clone)]
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
            OutputFormat::Csv => write_csv(measurements),
            OutputFormat::Table => write_table(measurements),
            OutputFormat::Json => write_json(measurements),
        }
    }
}

#[derive(Clone)]
struct Measurement {
    scenario: &'static str,
    size_desc: String,
    wall_s: f64,
    rss_delta_kib: u64,
    verification_status: VerificationStatus,
    verification_detail: Option<String>,
}

#[derive(Clone, Copy)]
enum VerificationStatus {
    NotChecked,
    Passed,
    Failed,
}

impl VerificationStatus {
    fn label(&self) -> &'static str {
        match self {
            VerificationStatus::NotChecked => "not_checked",
            VerificationStatus::Passed => "passed",
            VerificationStatus::Failed => "failed",
        }
    }
}

fn run_lcs(options: &Options, sys: &mut System) -> Vec<Measurement> {
    const SIZES: &[usize] = &[256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536];
    let total = SIZES.len();
    SIZES
        .iter()
        .enumerate()
        .map(|(idx, &len)| {
            eprint!("      [{}/{}] Testing size {}... ", idx + 1, total, len);
            let mut cost_result = 0u32;
            let mut path_len_result = 0;
            let m = measure("lcs", format!("len={len}"), sys, || {
                let seq_a = deterministic_dna(len);
                let seq_b = deterministic_dna_offset(len, 1);
                let problem = LcsProblem::new(&seq_a, &seq_b);
                let engine = HcpEngine::new(problem);
                let (cost, path) = engine.run();
                cost_result = cost;
                path_len_result = path.len();

                let verification = if len <= options.verify_limit {
                    let baseline = full_lcs_len(&seq_a, &seq_b);
                    if baseline == cost {
                        (VerificationStatus::Passed, None)
                    } else {
                        (
                            VerificationStatus::Failed,
                            Some(format!("expected {baseline}, got {cost}")),
                        )
                    }
                } else {
                    (VerificationStatus::NotChecked, None)
                };

                verification
            });
            let status_icon = match m.verification_status {
                VerificationStatus::Passed => "✓",
                VerificationStatus::Failed => "✗",
                VerificationStatus::NotChecked => "○",
            };
            eprintln!(
                "{} LCS length={}, path_len={}, time={:.3}s, status={}",
                status_icon,
                cost_result,
                path_len_result,
                m.wall_s,
                m.verification_status.label()
            );
            m
        })
        .collect()
}

fn run_nw_linear(options: &Options, sys: &mut System) -> Vec<Measurement> {
    const SIZES: &[usize] = &[256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536];
    const MATCH_SCORE: i32 = 2;
    const MISMATCH_PENALTY: i32 = 1;
    const GAP_PENALTY: i32 = -2;
    let total = SIZES.len();

    SIZES
        .iter()
        .enumerate()
        .map(|(idx, &len)| {
            eprint!("      [{}/{}] Testing size {}... ", idx + 1, total, len);
            let mut score_result = 0i32;
            let mut path_len_result = 0;
            let m = measure("needleman_wunsch_linear", format!("len={len}"), sys, || {
                let seq_a = deterministic_dna(len);
                let seq_b = deterministic_dna_offset(len, 2);
                let problem =
                    NwProblem::new(&seq_a, &seq_b, MATCH_SCORE, MISMATCH_PENALTY, GAP_PENALTY);
                let engine = HcpEngine::new(problem);
                let (cost, path) = engine.run();
                score_result = cost;
                path_len_result = path.len();

                let verification = if len <= options.verify_limit {
                    let baseline =
                        full_nw_score(&seq_a, &seq_b, MATCH_SCORE, MISMATCH_PENALTY, GAP_PENALTY);
                    if baseline == cost {
                        (VerificationStatus::Passed, None)
                    } else {
                        (
                            VerificationStatus::Failed,
                            Some(format!("expected {baseline}, got {cost}")),
                        )
                    }
                } else {
                    (VerificationStatus::NotChecked, None)
                };

                verification
            });
            let status_icon = match m.verification_status {
                VerificationStatus::Passed => "✓",
                VerificationStatus::Failed => "✗",
                VerificationStatus::NotChecked => "○",
            };
            eprintln!(
                "{} score={}, path_len={}, time={:.3}s, status={}",
                status_icon,
                score_result,
                path_len_result,
                m.wall_s,
                m.verification_status.label()
            );
            m
        })
        .collect()
}

fn run_nw_affine(options: &Options, sys: &mut System) -> Vec<Measurement> {
    const SIZES: &[usize] = &[256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536];
    const MATCH_SCORE: i32 = 2;
    const MISMATCH_PENALTY: i32 = 1;
    const GAP_OPEN: i32 = -3;
    const GAP_EXTEND: i32 = -1;
    let total = SIZES.len();

    SIZES
        .iter()
        .enumerate()
        .map(|(idx, &len)| {
            eprint!("      [{}/{}] Testing size {}... ", idx + 1, total, len);
            let mut score_result = 0i32;
            let mut path_len_result = 0;
            let m = measure("needleman_wunsch_affine", format!("len={len}"), sys, || {
                let seq_a = deterministic_dna(len);
                let seq_b = deterministic_dna_offset(len, 3);
                let problem = NwAffineProblem::new(
                    &seq_a,
                    &seq_b,
                    MATCH_SCORE,
                    MISMATCH_PENALTY,
                    GAP_OPEN,
                    GAP_EXTEND,
                );
                let engine = HcpEngine::new(problem);
                let (cost, path) = engine.run();
                score_result = cost;
                path_len_result = path.len();

                let verification = if len <= options.verify_limit {
                    let baseline = full_affine_score(
                        &seq_a,
                        &seq_b,
                        MATCH_SCORE,
                        MISMATCH_PENALTY,
                        GAP_OPEN,
                        GAP_EXTEND,
                    );
                    if baseline == cost {
                        (VerificationStatus::Passed, None)
                    } else {
                        (
                            VerificationStatus::Failed,
                            Some(format!("expected {baseline}, got {cost}")),
                        )
                    }
                } else {
                    (VerificationStatus::NotChecked, None)
                };

                verification
            });
            let status_icon = match m.verification_status {
                VerificationStatus::Passed => "✓",
                VerificationStatus::Failed => "✗",
                VerificationStatus::NotChecked => "○",
            };
            eprintln!(
                "{} score={}, path_len={}, time={:.3}s, status={}",
                status_icon,
                score_result,
                path_len_result,
                m.wall_s,
                m.verification_status.label()
            );
            m
        })
        .collect()
}

fn run_viterbi(options: &Options, sys: &mut System) -> Vec<Measurement> {
    const LENGTHS: &[usize] = &[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536];
    let hmm = demo_hmm();
    let total = LENGTHS.len();

    LENGTHS
        .iter()
        .enumerate()
        .map(|(idx, &len)| {
            eprint!("      [{}/{}] Testing obs_len {}... ", idx + 1, total, len);
            let mut logp_result = 0.0f64;
            let mut path_len_result = 0;
            let m = measure("viterbi", format!("obs_len={len}"), sys, || {
                let observations = alternating_observations(len);
                let problem = ViterbiProblem::new(hmm.clone(), observations.clone());
                let engine = HcpEngine::new(problem);
                let (cost, path) = engine.run();
                logp_result = cost.0;
                path_len_result = path.len();

                let verification = if len <= options.verify_limit {
                    let (baseline_logp, baseline_path) = viterbi_baseline(&hmm, &observations);
                    let logp = cost.0;
                    let passed = (logp - baseline_logp).abs() <= 1e-6 && path.len() == baseline_path.len();
                    if passed {
                        (VerificationStatus::Passed, None)
                    } else {
                        (
                            VerificationStatus::Failed,
                            Some(format!(
                                "baseline logp={baseline_logp:.6}, got={logp:.6}, path_len={}, baseline_len={}",
                                path.len(),
                                baseline_path.len()
                            )),
                        )
                    }
                } else {
                    (VerificationStatus::NotChecked, None)
                };

                verification
            });
            let status_icon = match m.verification_status {
                VerificationStatus::Passed => "✓",
                VerificationStatus::Failed => "✗",
                VerificationStatus::NotChecked => "○",
            };
            eprintln!(
                "{} logp={:.3}, path_len={}, time={:.3}s, status={}",
                status_icon,
                logp_result,
                path_len_result,
                m.wall_s,
                m.verification_status.label()
            );
            m
        })
        .collect()
}

fn run_dag_shortest_path(options: &Options, sys: &mut System) -> Vec<Measurement> {
    const CONFIGS: &[(usize, usize)] = &[
        (64, 8),
        (128, 8),
        (256, 16),
        (384, 16),
        (512, 24),
        (1024, 32),
        (2048, 48),
        (4096, 64),
        (8192, 96),
        (16384, 128),
        (32768, 192),
        (65536, 256),
    ];
    let total = CONFIGS.len();

    CONFIGS
        .iter()
        .enumerate()
        .map(|(idx, &(layers, width))| {
            eprint!("      [{}/{}] Testing layers={}, width={}... ", idx + 1, total, layers, width);
            let mut cost_result = 0i64;
            let mut path_len_result = 0;
            let m = measure(
                "dag_shortest_path",
                format!("layers={layers},width={width}"),
                sys,
                || {
                    let (problem, adjacency, widths) = build_dag(layers, width);
                    let engine = HcpEngine::new(problem);
                    let (cost, path) = engine.run();
                    cost_result = cost;
                    path_len_result = path.len();

                    let verification = if layers <= options.verify_limit {
                        let baseline = dag_relax_baseline(&adjacency, &widths);
                        let best = baseline.into_iter().min().unwrap_or(i64::MAX / 4);
                        if best == cost {
                            (VerificationStatus::Passed, None)
                        } else {
                            (
                                VerificationStatus::Failed,
                                Some(format!("baseline={best}, got={cost}")),
                            )
                        }
                    } else {
                        (VerificationStatus::NotChecked, None)
                    };

                    verification
                },
            );
            let status_icon = match m.verification_status {
                VerificationStatus::Passed => "✓",
                VerificationStatus::Failed => "✗",
                VerificationStatus::NotChecked => "○",
            };
            eprintln!(
                "{} cost={}, path_len={}, time={:.3}s, status={}",
                status_icon,
                cost_result,
                path_len_result,
                m.wall_s,
                m.verification_status.label()
            );
            m
        })
        .collect()
}

fn print_summary(measurements: &[Measurement], options: &Options) {
    eprintln!("\n{}", "=".repeat(80));
    eprintln!("Test Summary");
    eprintln!("{}", "=".repeat(80));
    eprintln!();

    // Count verification statuses
    let mut passed = 0;
    let mut failed = 0;
    let mut not_checked = 0;
    for m in measurements {
        match m.verification_status {
            VerificationStatus::Passed => passed += 1,
            VerificationStatus::Failed => failed += 1,
            VerificationStatus::NotChecked => not_checked += 1,
        }
    }

    let total = measurements.len();
    eprintln!("Verification Results:");
    eprintln!("  Total tests: {}", total);
    eprintln!("  ✓ Passed: {} ({:.1}%)", passed, 100.0 * passed as f64 / total as f64);
    eprintln!("  ✗ Failed: {} ({:.1}%)", failed, 100.0 * failed as f64 / total as f64);
    eprintln!("  ○ Not checked (size > {}): {} ({:.1}%)", options.verify_limit, not_checked, 100.0 * not_checked as f64 / total as f64);
    eprintln!();

    // Show failures if any
    if failed > 0 {
        eprintln!("Failed Tests:");
        for m in measurements {
            if matches!(m.verification_status, VerificationStatus::Failed) {
                eprintln!("  ✗ {} ({})", m.scenario, m.size_desc);
                if let Some(ref detail) = m.verification_detail {
                    eprintln!("     Error: {}", detail);
                }
            }
        }
        eprintln!();
    }

    // Performance statistics by scenario
    eprintln!("Performance Statistics by Scenario:");
    eprintln!();
    
    use std::collections::HashMap;
    let mut by_scenario: HashMap<&str, Vec<&Measurement>> = HashMap::new();
    for m in measurements {
        by_scenario.entry(m.scenario).or_insert_with(Vec::new).push(m);
    }

    for (scenario, ms) in by_scenario.iter() {
        let times: Vec<f64> = ms.iter().map(|m| m.wall_s).collect();
        let min_time = times.iter().copied().fold(f64::INFINITY, f64::min);
        let max_time = times.iter().copied().fold(0.0, f64::max);
        let avg_time = times.iter().sum::<f64>() / times.len() as f64;
        
        let mems: Vec<u64> = ms.iter().map(|m| m.rss_delta_kib).collect();
        let max_mem = mems.iter().copied().max().unwrap_or(0);
        let avg_mem = mems.iter().sum::<u64>() as f64 / mems.len() as f64;

        eprintln!("  {}:", scenario);
        eprintln!("    Tests: {}", ms.len());
        eprintln!("    Time: min={:.3}s, max={:.3}s, avg={:.3}s", min_time, max_time, avg_time);
        eprintln!("    Memory: max_delta={} KiB, avg_delta={:.1} KiB", max_mem, avg_mem);
        
        // Show scaling behavior
        if ms.len() >= 2 {
            let first = ms.first().unwrap();
            let last = ms.last().unwrap();
            let size_ratio = if first.wall_s > 0.0 {
                last.wall_s / first.wall_s
            } else {
                0.0
            };
            eprintln!("    Scaling: {}x slower from smallest to largest", size_ratio);
        }
        eprintln!();
    }

    // Overall assessment
    eprintln!("{}", "=".repeat(80));
    if failed == 0 {
        eprintln!("✓ All verified tests passed! The HCP-DP engine is working correctly.");
    } else {
        eprintln!("✗ {} test(s) failed. Please review the errors above.", failed);
    }
    eprintln!();
    eprintln!("Interpretation:");
    eprintln!("  • 'passed' tests match full-table DP baselines (correctness verified)");
    eprintln!("  • 'not_checked' tests are too large for baseline verification but ran successfully");
    eprintln!("  • Time scaling should be sub-quadratic for HCP-DP (better than O(n²))");
    eprintln!("  • Memory usage should remain bounded even for large inputs");
    eprintln!("{}", "=".repeat(80));
    eprintln!();
}

fn measure<F>(
    scenario: &'static str,
    size_desc: String,
    sys: &mut System,
    compute: F,
) -> Measurement
where
    F: FnOnce() -> (VerificationStatus, Option<String>),
{
    let before = rss_kib(sys);
    let start = Instant::now();
    let (status, detail) = compute();
    let duration = start.elapsed();
    let after = rss_kib(sys);

    Measurement {
        scenario,
        size_desc,
        wall_s: duration.as_secs_f64(),
        rss_delta_kib: after.saturating_sub(before),
        verification_status: status,
        verification_detail: detail,
    }
}

fn write_csv(measurements: &[Measurement]) -> Result<(), String> {
    println!("scenario,size_desc,wall_s,rss_delta_kib,verification_status,verification_detail");
    for m in measurements {
        let detail = m
            .verification_detail
            .as_ref()
            .map(|s| s.replace('"', "'"))
            .unwrap_or_default();
        println!(
            "{},{},{:.3},{},{},\"{}\"",
            m.scenario,
            m.size_desc,
            m.wall_s,
            m.rss_delta_kib,
            m.verification_status.label(),
            detail
        );
    }
    Ok(())
}

fn write_table(measurements: &[Measurement]) -> Result<(), String> {
    let mut col1 = "scenario".len();
    let mut col2 = "size".len();
    for m in measurements {
        col1 = col1.max(m.scenario.len());
        col2 = col2.max(m.size_desc.len());
    }

    println!(
        "{:<col1$}  {:<col2$}  {:>12}  {:>14}  {:>12}  {}",
        "scenario",
        "size",
        "wall_s",
        "rss_delta_kib",
        "status",
        "detail",
        col1 = col1,
        col2 = col2
    );
    println!(
        "{:-<col1$}  {:-<col2$}  {:-<12}  {:-<14}  {:-<12}  {:-<12}",
        "",
        "",
        "",
        "",
        "",
        "",
        col1 = col1,
        col2 = col2
    );
    for m in measurements {
        println!(
            "{:<col1$}  {:<col2$}  {:>12.3}  {:>14}  {:>12}  {}",
            m.scenario,
            m.size_desc,
            m.wall_s,
            m.rss_delta_kib,
            m.verification_status.label(),
            m.verification_detail
                .as_ref()
                .map(|s| s.as_str())
                .unwrap_or(""),
            col1 = col1,
            col2 = col2
        );
    }
    Ok(())
}

fn write_json(measurements: &[Measurement]) -> Result<(), String> {
    println!("[");
    for (idx, m) in measurements.iter().enumerate() {
        let detail = m.verification_detail.as_ref().map(|s| s.replace('"', "'"));
        println!(
            "  {{\"scenario\":\"{}\",\"size\":\"{}\",\"wall_s\":{:.3},\"rss_delta_kib\":{},\"verification\":{{\"status\":\"{}\",\"detail\":{}}}}}{}",
            m.scenario,
            m.size_desc,
            m.wall_s,
            m.rss_delta_kib,
            m.verification_status.label(),
            match detail {
                Some(ref d) => format!("\"{d}\""),
                None => "null".to_string(),
            },
            if idx + 1 == measurements.len() { "" } else { "," }
        );
    }
    println!("]");
    Ok(())
}

fn rss_kib(sys: &mut System) -> u64 {
    sys.refresh_processes_specifics(ProcessRefreshKind::new());
    if let Some(process) = get_current_pid().ok().and_then(|pid| sys.process(pid)) {
        process.memory()
    } else {
        0
    }
}

fn deterministic_dna(len: usize) -> Vec<u8> {
    const ALPHABET: &[u8] = b"ACGT";
    (0..len).map(|i| ALPHABET[i % ALPHABET.len()]).collect()
}

fn deterministic_dna_offset(len: usize, offset: usize) -> Vec<u8> {
    const ALPHABET: &[u8] = b"ACGT";
    (0..len)
        .map(|i| ALPHABET[(i + offset) % ALPHABET.len()])
        .collect()
}

fn alternating_observations(len: usize) -> Vec<usize> {
    (0..len).map(|i| i % 2).collect()
}

fn full_lcs_len(s: &[u8], t: &[u8]) -> u32 {
    let n = s.len();
    let m = t.len();
    let mut dp = vec![vec![0u32; m + 1]; n + 1];
    for i in 1..=n {
        for j in 1..=m {
            let up = dp[i - 1][j];
            let left = dp[i][j - 1];
            let diag = dp[i - 1][j - 1] + u32::from(s[i - 1] == t[j - 1]);
            dp[i][j] = up.max(left).max(diag);
        }
    }
    dp[n][m]
}

fn full_nw_score(
    s: &[u8],
    t: &[u8],
    match_score: i32,
    mismatch_penalty: i32,
    gap_penalty: i32,
) -> i32 {
    let n = s.len();
    let m = t.len();
    let mut dp = vec![vec![0i32; m + 1]; n + 1];
    for i in 1..=n {
        dp[i][0] = dp[i - 1][0] + gap_penalty;
    }
    for j in 1..=m {
        dp[0][j] = dp[0][j - 1] + gap_penalty;
    }
    for i in 1..=n {
        for j in 1..=m {
            let score = if s[i - 1] == t[j - 1] {
                match_score
            } else {
                -mismatch_penalty
            };
            let diag = dp[i - 1][j - 1] + score;
            let up = dp[i - 1][j] + gap_penalty;
            let left = dp[i][j - 1] + gap_penalty;
            dp[i][j] = diag.max(up).max(left);
        }
    }
    dp[n][m]
}

fn full_affine_score(
    s: &[u8],
    t: &[u8],
    match_score: i32,
    mismatch_penalty: i32,
    gap_open_penalty: i32,
    gap_extend_penalty: i32,
) -> i32 {
    let n = s.len();
    let m = t.len();
    let neg_inf = i32::MIN / 4;
    let mut m_dp = vec![vec![neg_inf; m + 1]; n + 1];
    let mut ix_dp = vec![vec![neg_inf; m + 1]; n + 1];
    let mut iy_dp = vec![vec![neg_inf; m + 1]; n + 1];
    m_dp[0][0] = 0;
    for j in 1..=m {
        if j == 1 {
            iy_dp[0][j] = gap_open_penalty + gap_extend_penalty;
        } else {
            iy_dp[0][j] = iy_dp[0][j - 1] + gap_extend_penalty;
        }
    }
    for i in 1..=n {
        if i == 1 {
            ix_dp[i][0] = gap_open_penalty + gap_extend_penalty;
        } else {
            ix_dp[i][0] = ix_dp[i - 1][0] + gap_extend_penalty;
        }
    }
    for i in 1..=n {
        for j in 1..=m {
            let score = if s[i - 1] == t[j - 1] {
                match_score
            } else {
                -mismatch_penalty
            };
            m_dp[i][j] = (m_dp[i - 1][j - 1]
                .max(ix_dp[i - 1][j - 1])
                .max(iy_dp[i - 1][j - 1]))
                + score;
            ix_dp[i][j] = (m_dp[i - 1][j] + gap_open_penalty + gap_extend_penalty)
                .max(ix_dp[i - 1][j] + gap_extend_penalty);
            iy_dp[i][j] = (m_dp[i][j - 1] + gap_open_penalty + gap_extend_penalty)
                .max(iy_dp[i][j - 1] + gap_extend_penalty);
        }
    }
    *[m_dp[n][m], ix_dp[n][m], iy_dp[n][m]].iter().max().unwrap()
}

fn demo_hmm() -> Hmm {
    Hmm {
        n_states: 2,
        log_pi: vec![(0.5f64).ln(), (0.5f64).ln()],
        log_a: vec![
            vec![(0.9f64).ln(), (0.1f64).ln()],
            vec![(0.2f64).ln(), (0.8f64).ln()],
        ],
        log_b: vec![
            vec![(0.8f64).ln(), (0.2f64).ln()],
            vec![(0.3f64).ln(), (0.7f64).ln()],
        ],
    }
}

fn viterbi_baseline(hmm: &Hmm, obs: &[usize]) -> (f64, Vec<usize>) {
    if obs.is_empty() {
        return (0.0, Vec::new());
    }
    let n = hmm.n_states;
    let t = obs.len();
    let mut dp = vec![vec![f64::NEG_INFINITY; n]; t];
    let mut back = vec![vec![0usize; n]; t];

    for s in 0..n {
        dp[0][s] = hmm.log_pi[s] + hmm.log_b[s][obs[0]];
    }
    for time in 1..t {
        for s_to in 0..n {
            let emit = hmm.log_b[s_to][obs[time]];
            let mut best = f64::NEG_INFINITY;
            let mut arg = 0usize;
            for s_from in 0..n {
                let cand = dp[time - 1][s_from] + hmm.log_a[s_from][s_to] + emit;
                if cand > best {
                    best = cand;
                    arg = s_from;
                }
            }
            dp[time][s_to] = best;
            back[time][s_to] = arg;
        }
    }
    let mut best_state = 0usize;
    let mut best = f64::NEG_INFINITY;
    for s in 0..n {
        if dp[t - 1][s] > best {
            best = dp[t - 1][s];
            best_state = s;
        }
    }
    let mut path = vec![0usize; t];
    path[t - 1] = best_state;
    for time in (1..t).rev() {
        let next = path[time];
        path[time - 1] = back[time][next];
    }
    (best, path)
}

fn build_dag(layers: usize, width: usize) -> (DagLayered, Vec<Vec<Vec<(usize, i64)>>>, Vec<usize>) {
    let mut widths = Vec::with_capacity(layers + 1);
    widths.push(1);
    for _ in 0..layers {
        widths.push(width);
    }
    let mut adjacency = Vec::with_capacity(layers);
    for layer in 0..layers {
        let current_width = widths[layer];
        let next_width = widths[layer + 1];
        let mut layer_edges = Vec::with_capacity(current_width);
        for u in 0..current_width {
            let mut edges = Vec::with_capacity(next_width);
            for v in 0..next_width {
                let weight = ((layer + u + v) % 13 + 1) as i64;
                edges.push((v, weight));
            }
            layer_edges.push(edges);
        }
        adjacency.push(layer_edges);
    }
    let problem = DagLayered::new(adjacency.clone(), widths.clone());
    (problem, adjacency, widths)
}

fn dag_relax_baseline(adjacency: &[Vec<Vec<(usize, i64)>>], widths: &[usize]) -> Vec<i64> {
    let mut dist = vec![i64::MAX / 4; widths[0]];
    if !dist.is_empty() {
        dist[0] = 0;
    }
    for (layer, edges) in adjacency.iter().enumerate() {
        let mut next = vec![i64::MAX / 4; widths[layer + 1]];
        for (u, &du) in dist.iter().enumerate() {
            if du >= i64::MAX / 8 {
                continue;
            }
            for &(v, w) in &edges[u] {
                let cand = du.saturating_add(w);
                if cand < next[v] {
                    next[v] = cand;
                }
            }
        }
        dist = next;
    }
    dist
}
