# hcp-align Output Schema

Structured `hcp-align` output uses schema version `hcp-align.v1`.

The formal JSON Schema is available at
[`schemas/hcp-align.v1.schema.json`](../schemas/hcp-align.v1.schema.json).

## Formats

| Format | Shape |
|---|---|
| `json` | One object for one pair; an array of objects for multiple pairs. |
| `jsonl` | One object per line. |
| `tsv` | Header row plus one row per pair. |
| `cigar` | Compact tabular output with ids, objective, coordinates, and CIGAR. |
| `paf` | Pairwise mApping Format rows for traceback-producing alignments. |
| `sam` | SAM records with headers for traceback-producing alignments. |
| `text` | Human-readable summaries; not intended as a stable parser target. |

The formal JSON Schema applies to `json` and `jsonl` records. PAF, TSV, CIGAR,
SAM, and text outputs are line-oriented renderings of the same per-pair reports.

## Required Per-Pair Fields

| Field | Meaning |
|---|---|
| `schema_version` | Currently `hcp-align.v1`. |
| `engine` | Currently `hcp-dp`. |
| `pair_index` | Zero-based pair index after pairwise zip. |
| `query_id`, `target_id` | Record ids; inline raw inputs use `query` and `target`. |
| `mode` | One of `global-linear`, `global-affine`, `local-linear`, `edit-distance`, `semiglobal-linear`, `seeded-global-linear`. |
| `score` | Signed alignment score for score-based modes; `null` for edit distance. |
| `distance` | Edit distance for `edit-distance`; `null` for score-based modes. |
| `path_score` | Objective value computed independently from the returned path; `null` for score-only edit distance. |
| `verification_status` | `full`, `path_only`, `score_only`, or `failed`. |
| `verified` | Compatibility boolean for full verification pass/fail. Prefer `verification_status`. |
| `query_start`, `query_end` | Half-open query coordinates covered by the returned trace. |
| `target_start`, `target_end` | Half-open target coordinates covered by the returned trace. |
| `cigar` | Run-length encoded operations using `=`, `X`, `D`, and `I`. |
| `block_size` | HCP summary-tree block size used for the run; `0` means a non-HCP exact backend. |
| `path_length` | Number of DP states in the returned path. |
| `summary_build_ms` | Summary construction time in milliseconds. |
| `reconstruction_ms` | Path reconstruction time in milliseconds. |
| `verification_ms` | Independent verification time in milliseconds. |
| `elapsed_ms` | Total per-pair elapsed time in milliseconds. |

## Optional Fields

| Field | When Present |
|---|---|
| `backend` | Selected non-default per-problem backend, currently edit-distance backends (`hcp-linear`, `adaptive-banded`, `myers`), affine `wavefront-affine`, or `minimizer-seeded`. |
| `operation_counts` | Default for JSON/JSONL with `--operation-detail summary`. |
| `operations` | Only with `--operation-detail full`. |
| `certificate` | Only with `--certificate`; SHA-256 hashes for the normalized inputs, scoring parameters, result fields, and trace. |
| `aligned_query`, `aligned_target` | Only with `--show-alignment`. |
| `error` | Failed pair record emitted under `--continue-on-error`. |

## Operation Semantics

| Operation | CIGAR symbol | Meaning |
|---|---|---|
| `match` | `=` | Query and target consume one equal base. |
| `mismatch` | `X` | Query and target consume one unequal base. |
| `gap_in_target` | `D` | Query consumes one base; target has a gap. |
| `gap_in_query` | `I` | Target consumes one base; query has a gap. |

Coordinates are zero-based and half-open. `query_end - query_start` is the number
of query bases covered by non-`I` operations. `target_end - target_start` is the
number of target bases covered by non-`D` operations.

## Certificates

`--certificate` adds a `certificate` object to JSON and JSONL records:

| Field | Meaning |
|---|---|
| `version` | Currently `hcp-align.certificate.v1`. |
| `hash_algorithm` | Currently `sha256`. |
| `query_sha256`, `target_sha256` | Hashes of the normalized query and target bytes. |
| `parameters_sha256` | Hash of the scoring-parameter object used by the mode. |
| `trace_sha256` | Hash of emitted traceback operations; `null` for score-only records. |
| `result_sha256` | Hash of objective, coordinates, CIGAR, verification status, backend, and trace hash. |
| `certificate_sha256` | Hash of the certificate payload fields above. |

The certificate is a compact reproducibility envelope for downstream pipelines.
It complements, but does not replace, independent path scoring and bounded
full-table validation.

## Versioning

Before `1.0`, schema-breaking changes may occur when they improve correctness or
clarity. Such changes should update `schema_version`, the CLI docs, this file,
and `CHANGELOG.md`.
