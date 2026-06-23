# hcp-align CLI Reference

`hcp-align` is the alpha command-line surface for the verified sequence
alignment kernels.

## Install From Source

```bash
cargo install --path .
hcp-align --help
```

During development, the same commands can be run through Cargo:

```bash
cargo run --bin hcp-align -- global-linear --query ACGT --target ACGT
```

GitHub alpha builds are produced by the manual `Release Alpha` workflow. Each
artifact includes the `hcp-align` binary, README, license, and a SHA-256
checksum.

## Commands

| Command | Meaning |
|---|---|
| `global-linear` | Global Needleman-Wunsch alignment with a linear gap penalty. |
| `global-affine` | Global Gotoh alignment with affine gap penalties. |
| `local-linear` | Smith-Waterman local alignment with a linear gap penalty. |
| `edit-distance` | Exact Levenshtein edit distance. |
| `semiglobal-linear` | Full query aligned against any target interval; target prefix/suffix are free. |

## Inputs

Each run requires exactly one query source and exactly one target source:

```bash
--query <SEQ>
--target <SEQ>
--query-file <PATH>
--target-file <PATH>
```

Raw inline input is always a single record. File input may be raw sequence text,
FASTA, or FASTQ.

FASTA supports wrapped sequence lines. FASTQ supports standard four-line records;
wrapped FASTQ is intentionally out of scope for this alpha. Sequences are
normalized by stripping whitespace and uppercasing ASCII.

For multi-record files, `hcp-align` uses pairwise zip mode: query record `i` is
aligned to target record `i`. Mismatched record counts return a nonzero error.
One-vs-many and all-vs-all modes are deferred.

## Output Formats

| Format | Behavior |
|---|---|
| `text` | Human-readable output. Multiple pairs print concise repeated summaries. |
| `json` | One JSON object for one pair; a JSON array for multiple pairs. |
| `jsonl` | One JSON object per pair. Preferred for batch workflows. |
| `tsv` | Stable tabular output with detailed fields. |
| `cigar` | Compact tabular output focused on ids, score/distance, coordinates, and CIGAR. |

Per-pair structured output includes:

- `schema_version`
- `engine`
- `pair_index`
- `query_id`, `target_id`
- `mode`
- `score` or `distance`
- `path_score`
- `verification_status`
- `verified`
- `query_start`, `query_end`, `target_start`, `target_end`
- `cigar`
- `operation_counts` by default
- `operations` only with `--operation-detail full`
- `block_size`
- `path_length`
- `summary_build_ms`, `reconstruction_ms`, `verification_ms`
- `elapsed_ms`
- `aligned_query`, `aligned_target` when `--show-alignment` is set
- `error` for failed per-pair records when `--continue-on-error` is used

`verified` remains for simple compatibility with earlier JSON output. New
consumers should use `verification_status`.

Trace detail is controlled with:

```bash
--operation-detail none|summary|full
```

The default is `summary`, which emits compact operation counts instead of a
full per-step operation array. Use `full` when downstream tooling needs every
match, mismatch, insertion, and deletion step.

Write output to a file with:

```bash
--output <PATH>
```

Progress is controlled with:

```bash
--progress auto|always|never
```

Progress is always written to stderr. The default `auto` only prints progress
for multi-record runs when stderr is attached to a terminal.

Batch execution can continue after per-pair failures:

```bash
--continue-on-error
```

Without this flag, per-pair errors fail fast. With it, failed pairs are emitted
as records with `verification_status = "failed"` and an `error` field; the
process still exits nonzero if any record failed.

Threaded batch execution:

```bash
--threads <N>
```

Values above `1` require building with the crate's `parallel` feature. Without
that feature, only `--threads 1` is accepted.

## Verification

`hcp-align` always computes an independent path score from the returned path.

With `--verify`, it also runs a full-table baseline when:

```text
max(query_len, target_len) <= --verify-limit
```

The default `--verify-limit` is `2048`. Use `--verify-limit 0` to remove the
limit.

Verification statuses:

| Status | Meaning |
|---|---|
| `full` | Path score and full-table baseline both matched the reported objective. |
| `path_only` | Path score matched, but full-table baseline was not requested or was above the limit. |
| `failed` | Path scoring or full-table baseline disagreed; process exits nonzero. |

## Scoring

Linear modes:

```bash
--match <N>
--mismatch-penalty <N>
--gap <NEGATIVE_N>
```

Affine mode:

```bash
--match <N>
--mismatch-penalty <N>
--gap-open <NEGATIVE_N>
--gap-extend <NEGATIVE_N>
```

Affine convention: the first position in a gap costs
`gap_open + gap_extend`; each continued gap position costs `gap_extend`.

Edit distance has fixed unit costs: substitution, insertion, and deletion cost
`1`.

## Examples

```bash
hcp-align global-linear \
  --query GATTACA --target GCATGCU \
  --match 1 --mismatch-penalty 1 --gap -1 \
  --verify --format json
```

```bash
hcp-align global-affine \
  --query ACB --target A \
  --match 2 --mismatch-penalty 1 --gap-open -3 --gap-extend -1 \
  --verify --format json
```

```bash
hcp-align edit-distance \
  --query-file reads.fa --target-file references.fa \
  --verify --format jsonl --output results.jsonl
```

```bash
hcp-align edit-distance \
  --query-file reads.fa --target-file references.fa \
  --format jsonl --operation-detail none --progress always
```

```bash
hcp-align semiglobal-linear \
  --query ACGT --target TTACGTTT \
  --match 2 --mismatch-penalty 1 --gap -2 \
  --verify --show-alignment --format text
```

## Known Limitations

- Batch mode is pairwise zip only.
- Wrapped FASTQ is not supported.
- No SAM/BAM/PAF export yet.
- Protein substitution matrices are not implemented; scoring is match/mismatch.
- Multi-threaded batch mode requires the optional `parallel` feature.
- Performance is still reported conservatively until release artifacts include
  larger reproducible benchmarks.
