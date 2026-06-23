# Height-Compressed Dynamic Programming (HCP-DP)

HCP-DP is an alpha-stage Rust crate for experimenting with exact dynamic
programming over layered problems using composable interval summaries.

It now also ships `hcp-align`, a small command-line alignment tool for trying
the correctness-tested sequence kernels without writing Rust code.

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

| Problem | Exact cost | Exact path | Summary laws | Parallel optimized | Benchmarked |
|---------|------------|------------|--------------|--------------------|-------------|
| LCS | yes | yes | yes | no | smoke only |
| Needleman-Wunsch, linear gap | yes | yes | yes | no | smoke only |
| Needleman-Wunsch, affine gap | yes | yes | yes | no | smoke only |
| Smith-Waterman, linear gap | yes | yes | yes | no | smoke only |
| Edit distance | yes | yes | yes | no | smoke only |
| Semi-global, linear gap | yes | yes | yes | no | smoke only |

Performance baselines are not enforced yet. Correctness comes first.

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
cargo run --example lcs
cargo run --example align
cargo run --bin hcp-align -- global-linear --query GATTACA --target GCATGCU --match 1 --mismatch-penalty 1 --gap -1 --verify
bash scripts/check.sh
cargo run --bin scale_probe -- --format table --verify-limit 512
```

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

`hcp-align` supports raw sequence text, single-record FASTA, and single-record
FASTQ inputs:

```bash
cargo run --bin hcp-align -- global-linear \
  --query GATTACA --target GCATGCU \
  --match 1 --mismatch-penalty 1 --gap -1 \
  --verify --format json

cargo run --bin hcp-align -- global-affine \
  --query ACB --target A \
  --match 2 --mismatch-penalty 1 --gap-open -3 --gap-extend -1 \
  --verify --format json

cargo run --bin hcp-align -- local-linear \
  --query ACACACTA --target AGCACACA \
  --match 2 --mismatch-penalty 1 --gap -2 \
  --verify --format json

cargo run --bin hcp-align -- edit-distance \
  --query kitten --target sitting \
  --verify --format json

cargo run --bin hcp-align -- semiglobal-linear \
  --query ACGT --target TTACGTTT \
  --match 2 --mismatch-penalty 1 --gap -2 \
  --verify --format json
```

JSON output includes the reported score or distance, independent path score,
verification status, query/target coordinates, CIGAR-like operations using
`=`, `X`, `D`, and `I`, and optional aligned strings when `--show-alignment` is
set.

## Development Checks

| Purpose | Command |
|---------|---------|
| Full local smoke check | `bash scripts/check.sh` |
| Unit and integration tests | `cargo test --lib --tests` |
| Feature compile checks | `cargo test --features tracing --lib --tests` and `cargo test --features parallel --lib --tests` |
| Scaling smoke probe | `cargo run --bin scale_probe -- --format table` |
| One probe scenario | `cargo run --bin scale_probe -- --scenario semiglobal --format json` |
| Optional external validation | `python3 scripts/validate_external.py` |
| Local report | `python3 scripts/perf_report.py` |
| Optional benchmarks | `RUN_BENCH=1 bash scripts/check.sh` |

CI mirrors these checks on stable Rust and the declared MSRV.
External validation against Parasail and Edlib is available as a manual GitHub
Actions workflow and is not part of default CI.

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
