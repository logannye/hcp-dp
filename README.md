# Height-Compressed Dynamic Programming (HCP-DP)

[![Rust](https://img.shields.io/badge/rust-1.88%2B-b7410e?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Open Source: MIT](https://img.shields.io/badge/open%20source-MIT-0b7285.svg)](LICENSE)

> Height-Compressed Dynamic Programming (HCP-DP) in Rust — run massive dynamic programs with exact answers and dramatically reduced memory.

This repository packages height-compression algorithmic optimizations into a production-quality Rust crate, complete with reference problems, property tests, and performance baselines.

---

## Table of Contents

- [At a Glance](#at-a-glance)
- [What is this?](#what-is-this)
- [Why is this significant?](#why-is-this-significant)
- [High-level idea](#high-level-idea)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Testing & QA](#testing--qa)
- [Optional features](#optional-features)
- [Core API: how it works](#core-api-how-it-works)
- [Using the built-in problems](#using-the-built-in-problems)
- [How to adapt this to your own DP](#how-to-adapt-this-to-your-own-dp)
- [Design notes & limitations](#design-notes--limitations)
- [Roadmap (ideas)](#roadmap-ideas)
- [Documentation & Support](#documentation--support)
- [License](#license)

---

## At a Glance

- **Generic engine**: drop-in `HcpEngine` orchestrates height-compressed DP for any problem that implements the `HcpProblem` trait.
- **Reference implementations**: LCS (full and banded), Needleman–Wunsch (linear and affine gaps), Viterbi decoding, layered DAG shortest paths, matrix-chain multiplication.
- **Safety and correctness**: rich unit, integration, and property tests; optional heavy stress and deterministic parallel equivalence suites.
- **Performance tooling**: Criterion benchmarks with RSS tracking, perf baseline enforcement, scaling probe (tests up to 65536 elements), and scheduled GitHub Actions runs.
- **Developer ergonomics**: builder API, feature flags (`parallel`, `tracing`, `heavy`), comprehensive CI, and contributor guide.

## What is this?

Many important dynamic programs:

* sequence alignment (bio/NLP),
* LCS / edit distance,
* Viterbi decoding,
* layered shortest paths,
* time-indexed DPs,
* some interval / chain-structured optimization problems,

have:

1. a **layered** structure: states arranged in layers ( 0, 1, \dots, T ),
2. **local dependencies**: each layer depends only on the previous one (or a bounded window),
3. an **associative composition** over intervals,
4. the ability to **reconstruct optimal paths** from local information.

Naively, we store full DP tables: (O(T \cdot W)) memory. For large instances (e.g. genome-scale alignment, ultra-long sequences, crazy-deep networks), this is the barrier.

The HCP-DP framework exploits the algebraic + locality structure to:

* store only **block-level summaries** instead of the full table,
* recursively reconstruct an optimal path using **local recomputation**,
* achieve strong time–space tradeoffs (often near-linear time, sublinear memory in (T)).

This crate:

* codifies that pattern as a **generic engine** (`HcpEngine`),
* defines a **trait interface** (`HcpProblem`) for “height-compressible” DPs,
* ships **reference implementations** (LCS, Needleman–Wunsch, Viterbi, etc.) as templates.

Think of it as: *“Hirschberg, generalized and packaged.”*

---

## Why is this significant?

Because the bottleneck in modern large-scale DP is often **memory**, not math.

Examples of where this matters:

* **Genomics**: exact global/local alignment on chromosomes or pan-genome graphs.
* **Search / NLP**: aligning very long strings / logs / traces without quadratic RAM.
* **Probabilistic models**: Viterbi / forward-backward on very long sequences.
* **Time-indexed optimization**: inventory / resource / scheduling DPs over long horizons.

We want:

* exact answers (no lossy sketching),
* predictable and tunable memory usage,
* a reusable structure instead of bespoke one-off hacks.

HCP-DP gives you:

* a precise abstraction for “height-compressible” problems,
* a library implementation that:

  * separates problem-specific logic from the compression mechanism,
  * is written in safe Rust,
  * is designed for clarity, extensibility, and serious use.

---

## High-level idea

At a conceptual level, the algorithm has two phases:

1. **Forward: build block summaries**

   * Partition layers (0..T) into blocks of size (b).
   * For each block ([a,b)), run the DP forward with only O(W) working memory.
   * Emit a compact summary (\Sigma[a,b]) that captures:

     * how boundary data at `a` maps to frontier values at `b`,
     * enough structure to choose consistent midpoints later.
   * Store these summaries; discard internal DP rows.

2. **Backward / recursive: reconstruct optimal path**

   * Treat block summaries as nodes in a higher-level chain.
   * Recursively split the sequence of blocks:

     * merge summaries on the left/right,
     * choose a boundary state at the midpoint that lies on some optimal path (using summaries + local recomputation),
     * recurse on left and right subranges.
   * For a single block, locally recompute and reconstruct the optimal subpath between its entry/exit boundaries.

The magic ingredients:

* **Associativity**: summaries compose: (\Sigma[a,c] = \Sigma[a,b] \oplus \Sigma[b,c]).
* **Local verifiability**: checking consistency at block boundaries uses O(1)–O(W) data, not the whole history.
* **Deterministic replay**: given entry/exit constraints for a block, you can recompute the DP inside using only O(W) space.

The engine handles the orchestration; you just tell it how your DP behaves.

---

## Repository layout

```text
hcp-dp/
├─ Cargo.toml
├─ README.md
├─ CONTRIBUTING.md
├─ perf/
│   └─ baseline.json          # Criterion median baselines
├─ scripts/
│   ├─ check.sh               # primary smoke script
│   └─ check_bench.py         # compares benches to baselines
├─ src/
│   ├─ lib.rs                 # crate entry, exports traits + engine + problems
│   ├─ traits.rs              # HcpProblem trait and core interface
│   ├─ engine.rs              # generic height-compressed DP engine
│   ├─ blocks.rs              # block summary struct
│   ├─ utils.rs               # helpers (default block size, etc.)
│   └─ problems/
│       ├─ mod.rs
│       ├─ lcs.rs             # Full LCS via Hirschberg-style compression
│       ├─ lcs_banded.rs      # Banded LCS template
│       ├─ nw_align.rs        # Needleman–Wunsch global alignment (linear gap)
│       ├─ nw_affine.rs       # Gotoh affine-gap alignment
│       ├─ matrix_chain.rs    # Matrix-chain multiplication
│       ├─ dag_sp.rs          # Layered DAG shortest path
│       └─ viterbi.rs         # Viterbi decoding on HMMs
├─ tests/
│   ├─ engine_invariants.rs
│   ├─ lcs_banded_property.rs
│   ├─ nw_affine_property.rs
│   ├─ viterbi_degenerate.rs
│   ├─ dag_sp_random.rs
│   └─ parallel_equivalence.rs (cfg(feature = "parallel"))
├─ examples/
│   ├─ lcs.rs
│   ├─ align.rs
│   ├─ matrix_chain.rs
│   └─ viterbi.rs
├─ benches/
│   └─ large_align.rs
└─ src/bin/
    └─ scale_probe.rs        # Scaling performance probe (tests up to 65536 elements)
```

---

## Installation

### Use as a dependency

```toml
[dependencies]
hcp-dp = { git = "https://github.com/logannye/hcp-dp", package = "hcp-dp" }
```

### Develop locally

```bash
git clone https://github.com/logannye/hcp-dp.git
cd hcp-dp
cargo build
```

### Requirements

- Rust (stable toolchain; CI also checks a declared MSRV)
- `python3` (used by `scripts/check_bench.py` during perf comparisons)
- Optional: `cargo install grcov` + LLVM tools if you want local coverage reports

---

## Quickstart

```bash
git clone https://github.com/logannye/hcp-dp.git
cd hcp-dp
cargo run --example lcs            # try the LCS demo
bash scripts/check.sh              # run fmt, clippy, tests, examples
cargo run --bin scale_probe        # test scaling performance (256 to 65536 elements)
```

Need an alignment example instead? Swap in `cargo run --example align` for Needleman–Wunsch or `cargo run --example viterbi` for HMM decoding.

---

## Testing & QA

| Purpose | Command | Notes |
|---------|---------|-------|
| Smoke + lint + unit/integration | `bash scripts/check.sh` | runs fmt, clippy, build, tests, examples |
| Scaling performance probe | `cargo run --bin scale_probe` | tests all problem types from 256 to 65536 elements; verifies correctness against baselines and reports timing/memory; supports `--format csv\|table\|json` and `--verify-limit N` |
| Heavy regression suite | `cargo test --features heavy` | expect long runtime / high memory |
| Parallel determinism | `cargo test --features parallel --test parallel_equivalence` | checks Rayon feature for identical outputs |
| Property/randomized tests | `cargo test --test lcs_banded_property`<br>`cargo test --test nw_affine_property`<br>`cargo test --test viterbi_degenerate`<br>`cargo test --test dag_sp_random` | powered by `proptest` |
| Benchmarks | `RUN_BENCH=1 bash scripts/check.sh` | Criterion suites for throughput (alignment) and summary micro-ops (`summary_ops`). Baselines live in `perf/baseline.json`; set `PERF_ENFORCE=1` to gate CI, `PERF_TOLERANCE` to tune thresholds |
| Coverage (optional) | `RUSTFLAGS="-Zinstrument-coverage" LLVM_PROFILE_FILE="target/coverage/%p-%m.profraw" cargo test` | pair with `grcov` for HTML reports |

The CI workflows in `.github/workflows/` mirror these checks across Linux, macOS, Windows, multiple toolchains, and feature combinations. See `CONTRIBUTING.md` for the full contributor checklist.

> Benchmarks also record summary-operator latency. After hardware or algorithmic changes, run `RUN_BENCH=1 PERF_ENFORCE=1 bash scripts/check.sh` locally and copy the resulting `perf/baseline.json` if you want new gate values.

### Optional features

- `parallel`: enables Rayon-based block summary reductions. Deterministic and tested against the serial engine.
- `tracing`: emits tracing spans around the forward/reconstruct phases (useful for profiling/instrumentation).
- `heavy`: unlocks large-scale regression/stress tests (long runtime / high memory expected).
- Features can be combined, e.g. `cargo test --features "parallel tracing"`.

---

## Core API: how it works

### `HcpProblem`

To plug in your DP, implement:

```rust
pub trait HcpProblem {
    type State: Clone;
    type Frontier: Clone;
    type Summary: Clone;
    type Boundary: Clone;
    type Cost: Copy + Ord;

    fn num_layers(&self) -> usize;
    fn init_frontier(&self) -> Self::Frontier;
    fn forward_step(&self, layer: usize, frontier_i: &Self::Frontier) -> Self::Frontier;
    fn summarize_block(&self, a: usize, b: usize, frontier_a: &Self::Frontier)
        -> (Self::Frontier, Self::Summary);
    fn merge_summary(&self, left: &Self::Summary, right: &Self::Summary) -> Self::Summary;

    fn initial_boundary(&self) -> Self::Boundary;
    fn terminal_boundary(&self, frontier_T: &Self::Frontier) -> Self::Boundary;

    fn choose_boundary(
        &self,
        a: usize,
        m: usize,
        c: usize,
        sigma_left: &Self::Summary,
        sigma_right: &Self::Summary,
        beta_a: &Self::Boundary,
        beta_c: &Self::Boundary,
    ) -> Self::Boundary;

    fn reconstruct_block(
        &self,
        a: usize,
        b: usize,
        beta_a: &Self::Boundary,
        beta_b: &Self::Boundary,
    ) -> Vec<Self::State>;

    fn extract_cost(&self, frontier_T: &Self::Frontier, beta_T: &Self::Boundary) -> Self::Cost;
}
```

You choose:

* how to represent a “frontier” (e.g., DP row),
* what a summary Σ[a,b] contains,
* how boundaries are encoded,
* how to recompute inside a block.

This is where your problem-specific logic lives.

### `HcpEngine`

Once you implement `HcpProblem`:

```rust
use hcp_dp::{HcpEngine, HcpProblem};

let problem = MyDpProblem::new(...);
let engine = HcpEngine::new(problem);              // uses ~sqrt(T) block size
// or: HcpEngine::with_block_size(problem, b);

let (cost, path) = engine.run();

println!("Optimal cost = {:?}", cost);
println!("Path length = {}", path.len());
```

The engine:

* runs the **forward block summarization** (Phase I),
* picks the terminal boundary,
* runs **recursive reconstruction** (Phase II),
* returns an exact optimal path as `Vec<State>`.

---

## Using the built-in problems

These are both real and serve as templates.

### 1. LCS (Hirschberg-style)

```rust
use hcp_dp::{HcpEngine, problems::lcs::LcsProblem};

let s = b"ACCGGTCGAGTGCGCGGAAGCCGGCCGAA";
let t = b"GTCGTTCGGAATGCCGTTGCTCTGTAAA";

let problem = LcsProblem::new(s, t);
let engine = HcpEngine::new(problem);
let (len_lcs, path) = engine.run();
```

* Layers: positions in `s`.
* Frontier: DP row over `t`.
* Summaries + `choose_boundary`: implement a Hirschberg-like mid-split.
* `reconstruct_block`: does local DP inside a sub-block.
* Configuration helper:
  ```rust
  use hcp_dp::{HcpEngineBuilder, problems::lcs::LcsProblem};
  let engine = HcpEngineBuilder::new(LcsProblem::new(b"AA", b"A"))
      .with_block_size(1)
      .build();
  ```

Result: exact LCS and path with controlled memory.

### 2. Banded LCS

```rust
use hcp_dp::{HcpEngine, problems::lcs_banded::LcsBandedProblem};

let engine = HcpEngine::new(LcsBandedProblem::new(b"ACCG", b"ACGC", 32));
let (len, path) = engine.run();
```

- Restricts computation to a diagonal band (width `band`) for near-diagonal alignments.
- Property tests compare banded results to full LCS when `band >= |n-m| + slack`.

### 3. Needleman–Wunsch global alignment (linear gap)

```rust
use hcp_dp::{HcpEngine, problems::nw_align::NwProblem};

let s = b"GATTACA";
let t = b"GCATGCU";
let problem = NwProblem::new(s, t, 1, 1, -1);
let engine = HcpEngine::new(problem);
let (score, path) = engine.run();
```

* Supports full optimal global alignment reconstruction.
* Demonstrates how affine-like DPs can be plugged in with suitable summaries.

### 4. Needleman–Wunsch affine gaps (Gotoh)

```rust
use hcp_dp::{HcpEngine, problems::nw_affine::NwAffineProblem};

let engine = HcpEngine::new(NwAffineProblem::new(b"GATTACA", b"GCATGCU", 1, 1, -3, -1));
let (score, path) = engine.run();
```

- Three-state Gotoh recurrence (match, gap in `s`, gap in `t`).
- Property tests cross-check against a straightforward Gotoh baseline on small strings.

### 5. Matrix-chain multiplication

```rust
use hcp_dp::{HcpEngine, problems::matrix_chain::MatrixChainProblem};

let p = vec![30, 35, 15, 5, 10, 20, 25];
let problem = MatrixChainProblem::new(p);
let engine = HcpEngine::new(problem);
let (cost, splits) = engine.run();
```

* Shows interval DP embedding.
* Not aggressively optimized; it’s a didactic reference.

### 6. Viterbi decoding

```rust
use hcp_dp::{HcpEngine, problems::viterbi::{Hmm, ViterbiProblem}};

// define your HMM + obs...
let problem = ViterbiProblem::new(hmm, obs);
let engine = HcpEngine::new(problem);
let (log_prob, path) = engine.run();
```

* Layers: time steps.
* Frontier: best-path scores per state.
* Path: decoded most likely sequence.

### 7. Layered DAG shortest path

```rust
use hcp_dp::{HcpEngine, problems::dag_sp::DagLayered};

let adjacency = vec![
    vec![vec![(0, 1), (1, 5)]],
    vec![vec![(0, 1)], vec![(1, 1)]],
];
let widths = vec![1, 2, 2];
let (cost, path) = HcpEngine::new(DagLayered::new(adjacency, widths)).run();
```

- Works on layered DAGs (edges only between consecutive layers).
- Randomized tests compare against a simple topological relaxation baseline.

---

## How to adapt this to your own DP

Rough recipe:

1. **Define layers**

   Decide how to index your DP so that transitions are between consecutive layers:
   time steps, positions in a sequence, diagonals, etc.

2. **Define frontier**

   Choose a compact representation of “all relevant state at layer i”:

   * row vector,
   * slice of table,
   * a set of active states.

3. **Define summaries**

   For a block [a,b):

   * run your recurrence from `a` to `b` with minimal memory,
   * create a `Summary` that:

     * allows composition (`merge_summary`),
     * supports your `choose_boundary` logic.

   For many problems:

   * summary = “frontier at end of block” (+ sometimes a backward profile).

4. **Define boundaries**

   Figure out what it means to “pin” an optimal path through a layer:

   * a coordinate,
   * a state index,
   * a small set of choices.

5. **Implement reconstruction**

   Implement `reconstruct_block(a,b,beta_a,beta_b)`:

   * recompute the DP in [a,b],
   * backtrack under the boundary constraints,
   * return the local path segment.

6. **Plug into `HcpEngine`**

   Build, run, and verify your results against your baseline DP.

If your problem matches the height-compressible pattern (local dependencies, associative composition, local boundary checks), this will give you a principled, reusable, and often asymptotically optimal time–space tradeoff.

---

## Design notes & limitations

* This crate is **exact**, not approximate.
* It assumes:

  * your DP can be layered,
  * dependencies are local (no arbitrary global lookups),
  * you can define a meaningful summary monoid and reconstruction.
* For some problems (e.g. very irregular graphs, arbitrary pointer-chasing), the HCP assumptions do not hold; you won’t get benefits or correctness without more work.
* Reference problems favor clarity over micro-optimizations; you can:

  * tune block sizes,
  * use custom summaries,
  * integrate SIMD / parallelization on top.

---

## Roadmap (ideas)

Potential extensions (PRs welcome if you end up using this):

* Stronger generic helpers for:

  * forward/backward profiles,
  * Hirschberg-like splitting,
  * banded / sparse frontiers.
* Better instantiated algorithms for:

  * affine-gap alignment at genome scale,
  * large HMMs/CRFs,
  * DAG shortest paths with layered constraints.
* Optional `no_std` + embedded / streaming support.
* Integrations with bioinformatics / ML / verification ecosystems.

---

## Documentation & Support

- **API & architecture**: start with this `README.md`, then explore inline rustdoc in `src/` (`cargo doc --open`).
- **Contributor workflow**: refer to `CONTRIBUTING.md` for environment setup, testing expectations, and CI details.
- **Issue tracking**: open GitHub issues for bugs, feature requests, or performance regressions. Include repro steps and DP dimensions.
- **Discussions**: for research collaborations or integration questions, feel free to start a GitHub Discussion or reach out via issues; we welcome contributions.

---

## License

MIT. Use it in research, production, or experiments.
If you build something cool (or break it in an interesting way), contributions and issues are very welcome.
