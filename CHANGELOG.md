# Changelog

All notable user-facing changes should be recorded here.

This project is pre-1.0. Breaking API, CLI, and JSON changes are allowed before
the alpha release when they improve correctness or clarity.

## Unreleased

- Added exact adaptive-banded edit-distance traceback for low-edit-distance
  inputs, exposed through `hcp-align edit-distance --engine adaptive-banded`.
- Added default `hcp-align edit-distance --engine auto` selection, which tries a
  bounded exact banded traceback before falling back to HCP linear-space
  traceback.
- Added `hcp-linear`, adaptive-banded, arbitrary-length Myers, full-table, and
  linear-space edit-distance comparison modes to
  `scale_probe --mode edit-distance-deep`.
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
