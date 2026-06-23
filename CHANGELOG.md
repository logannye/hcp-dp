# Changelog

All notable user-facing changes should be recorded here.

This project is pre-1.0. Breaking API, CLI, and JSON changes are allowed before
the alpha release when they improve correctness or clarity.

## Unreleased

- Polished public README, project health files, output schema documentation, and
  sample input data.
- Added strict rustdoc warning checks to CI and local smoke checks.
- Added release-binary smoke testing before alpha artifact upload.

## 0.1.0-alpha.1 - Planned

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
