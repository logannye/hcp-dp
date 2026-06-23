# Changelog

All notable user-facing changes should be recorded here.

This project is pre-1.0. Breaking API, CLI, and JSON changes are allowed before
the alpha release when they improve correctness or clarity.

## Unreleased

- Added `hcp-align --certificate` for opt-in SHA-256 result certificates in
  JSON/JSONL output, covering normalized inputs, scoring parameters, trace
  operations, result fields, and certificate payloads.
- Added built-in BLOSUM62 and custom substitution-matrix scoring for
  `global-linear`, `global-affine`, `local-linear`, and `semiglobal-linear`.
- Added `--format paf` for traceback-producing alignments, including standard
  PAF coordinates, match/block counts, `cg`, `NM`, verification, and pair-index
  tags.
- Added `--format sam` with `@SQ` headers, reference-oriented CIGAR, soft
  clipping for local alignments, and score/edit-distance tags.
- Added `global-affine --engine wavefront` for exact diagonal-band affine
  traceback, plus `--engine auto` fallback to HCP linear-space traceback.
- Added `seeded-global-linear`, a minimizer-seeded exact extension mode that
  chains shared k-mer seeds, expands the candidate window, and runs exact
  Needleman-Wunsch traceback inside that window.
- Added `problems::dag::LayeredDagProblem`, a sparse-frontier longest-path
  proof point with HCP contract tests.
- Added `problems::viterbi::ViterbiProblem`, a dense HMM-style Viterbi
  traceback proof point with HCP contract tests.
- Added exact bit-parallel LCS length scoring for targets up to 128 symbols,
  with exhaustive small-case tests and use in `LcsProblem::full_table_len`.
- Added `bindings/python/hcp_align.py`, a lightweight Python wrapper around the
  `hcp-align --format json` CLI contract.
- Populated `perf/baseline.json` and tightened `scripts/check_bench.py` so
  nightly perf runs enforce representative Criterion median budgets.
- Added exact adaptive-banded edit-distance traceback for low-edit-distance
  inputs, exposed through `hcp-align edit-distance --engine adaptive-banded`.
- Added default `hcp-align edit-distance --engine auto` selection, which tries a
  bounded exact banded traceback before falling back to HCP linear-space
  traceback.
- Added `hcp-align edit-distance --score-only`, backed by exact
  arbitrary-length Myers bit-vector distance.
- Added `hcp-linear`, adaptive-banded, arbitrary-length Myers, full-table, and
  linear-space edit-distance comparison modes to
  `scale_probe --mode edit-distance-deep`.
- Added `scripts/perf_report.py --max-size` and `--sample-output`, plus a
  checked-in bounded sample report under `docs/sample-report.md`.
- Added `problems::dtw::DtwProblem`, the first non-sequence-alignment public
  problem, with contract tests and `scale_probe` coverage.
- Added `docs/problem-author-guide.md` for implementing new problems against
  the interval-summary and endpoint-constrained traceback contract.
- Added public `contract` validation helpers for downstream `HcpProblem`
  implementations.
- Added a runnable custom-problem example showing min-plus interval summaries
  and public contract checks.
- Added `HcpEngine::linear_space(problem)` for explicit linear-space exact
  traceback runs.
- Polished README, CLI docs, capability matrix, output schema, and package
  metadata for production-facing users.

## 0.1.0-alpha.1 - 2026-06-23

- Polished public README, project health files, output schema documentation, and
  sample input data.
- Added strict rustdoc warning checks to CI and local smoke checks.
- Added release-binary smoke testing before alpha artifact upload.
- First GitHub alpha binary release for `hcp-align`.
- Exact alignment modes:
  - global Needleman-Wunsch with linear gaps,
  - global Gotoh affine-gap alignment,
  - Smith-Waterman local linear-gap alignment,
  - Levenshtein edit distance,
  - semi-global linear-gap alignment.
- Multi-record FASTA/FASTQ pairwise batch input.
- Text, JSON, JSONL, TSV, and CIGAR-style output.
- Independent path scoring and bounded full-table verification.
- Reproducible validation and performance report workflow.
