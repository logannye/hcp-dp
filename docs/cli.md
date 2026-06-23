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

- `pair_index`
- `query_id`, `target_id`
- `mode`
- `score` or `distance`
- `path_score`
- `verification_status`
- `verified`
- `query_start`, `query_end`, `target_start`, `target_end`
- `cigar`
- `operations`
- `block_size`
- `elapsed_ms`
- `aligned_query`, `aligned_target` when `--show-alignment` is set

`verified` remains for simple compatibility with earlier JSON output. New
consumers should use `verification_status`.

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
  --verify --format jsonl
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
- Performance is still reported conservatively until release artifacts include
  larger reproducible benchmarks.
