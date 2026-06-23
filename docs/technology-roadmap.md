# Technology Roadmap

This roadmap orders the highest-value enhancements in the sequence that should
compound best if completed one after another. Each item should preserve the
project rule that public capabilities must report exact objectives and, when a
traceback is produced, an independently scoreable exact path.

## 1. Proof-Carrying Alignment Records

Add compact result certificates to structured output. The certificate should
hash normalized inputs, scoring parameters, traceback operations, result fields,
and the certificate payload itself. This gives every later backend a stable
reproducibility envelope for downstream pipelines.

Status: initial `hcp-align --certificate` support is implemented for JSON and
JSONL records.

## 2. Scoring Matrices And Protein Alignment

Generalize scoring beyond match/mismatch to named substitution matrices and
user-provided matrices. This should support protein workflows and give future
affine and wavefront backends a realistic scoring surface.

Status: initial named BLOSUM62 and custom matrix-file support is implemented for
the score-based CLI modes and the reusable problem constructors.

## 3. Interoperability Exports

Add PAF first, then SAM where the coordinate and CIGAR semantics are clear.
These formats make HCP-DP usable inside existing read-mapping and comparative
genomics pipelines.

Status: initial PAF and SAM export is implemented for traceback-producing CLI
records. SAM currently emits primary forward-strand records with generated
headers and unknown mapping qualities.

## 4. Wavefront Affine Traceback Backend

Implement an exact wavefront-style affine-gap backend for low-difference global
and extension workloads. It should emit the same path, verification status, and
certificate fields as the existing HCP and adaptive-banded engines.

Status: initial exact diagonal-band affine traceback is implemented for
`global-affine --engine wavefront`, with `--engine auto` fallback to HCP
linear-space traceback.

## 5. Minimizer-Seeded Exact Extension

Add minimizer seeding and chaining to find candidate windows, then run exact
traceback inside those windows. This turns the library from pairwise alignment
into a practical exact-extension engine for long reads and near-neighbor search.

Status: initial deterministic minimizer selection, monotone seed chaining, and
`seeded-global-linear` exact windowed traceback are implemented.

## 6. Sparse/DAG Alignment

Extend the HCP contract to sparse layered graphs and DAG alignment. This is a
better showcase for compressed frontier operators than dense rectangular DP and
opens the path toward pangenome-style workloads.

Status: initial `LayeredDagProblem` sparse longest-path proof point is
implemented with summary-law and all-block-size contract tests.

## 7. Viterbi And Profile-HMM Traceback

Add a non-sequence-grid probabilistic DP proof point with exact traceback. A
profile-HMM or generic Viterbi module would demonstrate that HCP-DP is a layered
DP framework, not only an aligner.

Status: initial dense `ViterbiProblem` is implemented with exact traceback and
contract tests.

## 8. Optimized Algebraic Summaries

Replace replay-only summaries in selected modules with true algebraic or
bit-parallel summary operators where they are mathematically justified. Keep the
contract tests as the admission gate for each optimized summary.

Status: initial bit-parallel LCS length scoring is implemented for targets up to
128 symbols and wired into `LcsProblem::full_table_len`; HCP path summaries
remain contract-backed.

## 9. Language And Runtime Bindings

Expose the stable CLI/library surface through Python and WASM bindings after
the result schema, scoring model, and core backends are less volatile.

Status: initial Python subprocess binding over `hcp-align --format json` is
implemented under `bindings/python`. WASM remains deferred.

## 10. Enforced Performance Budgets

Populate `perf/baseline.json`, make nightly performance regressions actionable,
and keep benchmark claims tied to reproducible report commands and artifacts.

Status: representative Criterion median budgets are populated in
`perf/baseline.json`, `scripts/check_bench.py` enforces slowdowns when
`PERF_ENFORCE=1`, and scheduled nightly perf runs enforce by default.
