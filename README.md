# HCP-DP

HCP-DP is an alpha-stage Rust crate and CLI for exact sequence-alignment
traceback from composable height-compressed dynamic-programming summaries.

The primary live-use surface is `hcp-align`, a command-line alignment tool for
biosequence-style workloads. The library remains intentionally small while the
height-compressed contract is hardened problem by problem.

The current public proof point is exact edit-distance traceback with a
reproducible report comparing HCP-DP against full-table, linear-space, and
optional Edlib baselines.

This repository was reset around a correctness-first contract. The public API is
intentionally small until each built-in problem proves:

- exact objective value,
- returned path realizes that objective,
- summaries are boundary-independent interval operators,
- merged summaries behave like the direct interval.

## Current Public Surface

The crate currently exports:

- `alignment::AlignmentTrace`
- `HcpEngine`
- `HcpEngineBuilder`
- `HcpProblem`
- `SummaryApply`
- `problems::edit_distance::EditDistanceProblem`
- `problems::lcs::LcsProblem`
- `problems::nw_align::NwProblem`
- `problems::nw_affine::NwAffineProblem`
- `problems::semiglobal::SemiGlobalProblem`
- `problems::smith_waterman::SmithWatermanProblem`

The workspace also includes the `hcp-align` binary.

Former examples for banded LCS, Viterbi, DAG shortest path, and matrix-chain
multiplication were removed from the public surface. They should be reintroduced
only after passing the same contract harness.

## Capability Matrix

| Problem | Exact cost | Exact path | Summary laws | CLI | External score validation | Report coverage | Caveat |
|---------|------------|------------|--------------|-----|---------------------------|-----------------|--------|
| LCS | yes | yes | yes | no | no | scale probe | library-only |
| Needleman-Wunsch, linear gap | yes | yes | yes | yes | Parasail optional | scale probe | no SIMD baseline yet |
| Needleman-Wunsch, affine gap | yes | yes | yes | yes | Parasail optional after gap calibration | scale probe | slower than linear modes |
| Smith-Waterman, linear gap | yes | yes | yes | yes | Parasail optional | scale probe | selected local traceback only |
| Edit distance | yes | yes | yes | yes | Edlib optional | scale probe | exact Levenshtein only |
| Semi-global, linear gap | yes | yes | yes | yes | no | scale probe | full query vs target interval |

Performance budgets are not enforced yet. Correctness comes first. See
[docs/capabilities.md](docs/capabilities.md) for details.

## Core Contract

An admissible problem implements `HcpProblem`:

```rust
pub trait HcpProblem {
    type State: Clone + PartialEq;
    type Frontier: Clone;
    type Summary: Clone + SummaryApply<Self::Frontier>;
    type Boundary: Clone + PartialEq;
    type Cost: Copy + Ord;

    fn num_layers(&self) -> usize;
    fn init_frontier(&self) -> Self::Frontier;
    fn forward_step(&self, layer: usize, frontier: &Self::Frontier) -> Self::Frontier;
    fn summarize_interval(&self, a: usize, b: usize) -> Self::Summary;
    fn merge_summary(&self, left: &Self::Summary, right: &Self::Summary) -> Self::Summary;
    fn initial_boundary(&self) -> Self::Boundary;
    fn terminal_boundary(&self, frontier_t: &Self::Frontier) -> Self::Boundary;
    fn choose_split(
        &self,
        a: usize,
        m: usize,
        c: usize,
        beta_a: &Self::Boundary,
        beta_c: &Self::Boundary,
        sigma_left: &Self::Summary,
        sigma_right: &Self::Summary,
    ) -> Self::Boundary;
    fn reconstruct_leaf(
        &self,
        a: usize,
        b: usize,
        beta_a: &Self::Boundary,
        beta_b: &Self::Boundary,
    ) -> Vec<Self::State>;
    fn extract_cost(&self, frontier_t: &Self::Frontier, beta_t: &Self::Boundary) -> Self::Cost;
}
```

The important rules:

- `summarize_interval(a, b)` must not depend on a particular input frontier.
- `SummaryApply::apply` must behave like replaying `forward_step` over that interval.
- `merge_summary(left, right)` must represent the adjacent union.
- `choose_split` must honor `beta_a` and `beta_c`.
- `reconstruct_leaf` must return a segment that starts at `beta_a`, ends at
  `beta_b`, and realizes the local optimum.

## Quickstart

```bash
cargo install --path .
cargo run --example lcs
cargo run --example align
hcp-align global-linear --query GATTACA --target GCATGCU --match 1 --mismatch-penalty 1 --gap -1 --verify
bash scripts/check.sh
cargo run --bin scale_probe -- --format table --verify-limit 512
python3 scripts/perf_report.py --scenario edit_distance --verify-limit 128
```

GitHub alpha binaries are produced by the manual `Release Alpha` workflow. Each
artifact contains `hcp-align`, README, license, and a SHA-256 checksum. This
repository is not published to crates.io yet.

Example:

```rust
use hcp_dp::{problems::lcs::LcsProblem, HcpEngine};

let problem = LcsProblem::new(b"CCA", b"C");
let (cost, path) = HcpEngine::new(problem.clone()).run();

assert_eq!(cost, 1);
assert_eq!(problem.score_path(&path), Some(cost));
```

Affine-gap alignment uses Gotoh state semantics. The first position in a gap
costs `gap_open + gap_extend`; each continued gap position costs `gap_extend`.
Smith-Waterman returns the selected local alignment path only. If no positive
local alignment exists, it returns cost `0` with an empty path.
Semi-global alignment consumes the full query against any target interval; target
prefix and suffix are free. Swap query and target if you need the opposite
orientation.

## CLI

`hcp-align` supports raw inline sequences and multi-record FASTA/FASTQ files.
File inputs use pairwise zip mode: query record `i` is aligned to target record
`i`, and mismatched record counts are rejected.

```bash
hcp-align global-linear \
  --query GATTACA --target GCATGCU \
  --match 1 --mismatch-penalty 1 --gap -1 \
  --verify --format json

hcp-align global-affine \
  --query ACB --target A \
  --match 2 --mismatch-penalty 1 --gap-open -3 --gap-extend -1 \
  --verify --format json

hcp-align local-linear \
  --query ACACACTA --target AGCACACA \
  --match 2 --mismatch-penalty 1 --gap -2 \
  --verify --format json

hcp-align edit-distance \
  --query kitten --target sitting \
  --verify --format json

hcp-align semiglobal-linear \
  --query ACGT --target TTACGTTT \
  --match 2 --mismatch-penalty 1 --gap -2 \
  --verify --format json

hcp-align edit-distance \
  --query-file reads.fa --target-file references.fa \
  --verify --format jsonl
```

Output formats are `text`, `json`, `jsonl`, `tsv`, and `cigar`. Every pair
reports record ids, score or distance, independently scored path value,
verification status, query/target coordinates, CIGAR-like operations using `=`,
`X`, `D`, and `I`, block size, path length, timing fields, and elapsed
milliseconds. `--show-alignment` adds aligned strings.

Structured output defaults to compact operation counts. Use
`--operation-detail full` to emit every alignment step, or
`--operation-detail none` for large batch runs where only score, coordinates,
and CIGAR are needed. Use `--output <PATH>` to write results to a file.

`--verify` checks the returned objective against a full-table baseline when
`max(query_len, target_len) <= --verify-limit`. The default limit is `2048`; use
`--verify-limit 0` for no limit. Larger pairs still get independent path scoring
and report `verification_status = "path_only"`.

Full CLI reference: [docs/cli.md](docs/cli.md).

## Development Checks

| Purpose | Command |
|---------|---------|
| Full local smoke check | `bash scripts/check.sh` |
| Unit and integration tests | `cargo test --lib --tests` |
| Feature compile checks | `cargo test --features tracing --lib --tests` and `cargo test --features parallel --lib --tests` |
| Scaling smoke probe | `cargo run --bin scale_probe -- --format table` |
| One probe scenario | `cargo run --bin scale_probe -- --scenario semiglobal --format json` |
| Bounded probe sizes | `cargo run --bin scale_probe -- --max-size 1024 --format table` |
| Edit-distance deep proof | `cargo run --bin scale_probe -- --mode edit-distance-deep --format json` |
| Optional external validation | `python3 scripts/validate_external.py` |
| Local report | `python3 scripts/perf_report.py` |
| Optional benchmarks | `RUN_BENCH=1 bash scripts/check.sh` |

CI mirrors these checks on stable Rust and the declared MSRV.
External validation against Parasail and Edlib is available as a manual GitHub
Actions workflow and is not part of default CI. The manual workflow uploads
`target/hcp-dp-report/` as an artifact.

The alpha release checklist is in
[docs/alpha-release-checklist.md](docs/alpha-release-checklist.md).

## Adding A New Problem

Use LCS or linear Needleman-Wunsch as the template.

Before exporting a new problem from `problems`, add tests that prove:

- summary apply equals direct replay,
- summary merge equals the direct combined interval,
- split boundaries are feasible,
- reconstructed segments join exactly,
- final path-realized cost equals the reported objective,
- known counterexamples remain fixed.

If any of these are missing, keep the module private or behind an experimental
feature.

## License

MIT.
