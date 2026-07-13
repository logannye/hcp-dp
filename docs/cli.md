# `hcp-align` v0.2 CLI

## `solve`

```text
hcp-align solve <MODE> [INPUT] [SCORING] [OUTPUT]
```

Modes are `edit-distance`, `global-linear`, `global-affine`, `local-linear`, and
`semiglobal-linear`. Provide exactly one query source and one target source with
`--query`/`--target` or `--query-file`/`--target-file`.

Important options:

- `--scope alignment|score`
- `--backend auto|hcp|banded-affine`
- `--band N`
- `--cross-check`
- `--operations`
- `--integrity`
- `--format text|json|jsonl|paf|sam`

Common batch options on `solve`, `verify`, and `compare`:

- `--threads N` (default `min(available CPUs, 8)`)
- `--max-in-flight N` (default `2 × threads`)
- `--max-in-flight-bases N` (default `64,000,000`)
- `--fail-fast`
- `--errors PATH` (JSONL sidecar; conflicts with `--fail-fast`)
- `--temp-dir PATH`

Affine scoring applies `gap_open + gap_extend` to the first gap position and
`gap_extend` to each continuation.

## `verify`

```text
hcp-align verify --alignment PATH --format json|jsonl|paf|sam INPUT [OPTIONS]
```

`--level optimal` is the default. `--max-cells` defaults to 25,000,000 and `0`
means unlimited. PAF and SAM additionally require `--mode` and applicable
scoring flags. JSON v2 records embed their problem definitions.

Imported records resolve query and target sequences by ID. Duplicate or missing
IDs are errors. PAF/SAM reverse-strand records are rejected in v0.2 rather than
being silently mis-normalized.

JSON and JSONL are framed incrementally. JSONL, PAF, and SAM have recoverable
line boundaries; JSON-array syntax corruption is fatal. JSONL continues after
record failures by default and emits ordered v2 error records inline. Other
output formats require `--errors` to continue.

## `compare`

```text
hcp-align compare \
  --left LEFT --left-format FORMAT \
  --right RIGHT --right-format FORMAT \
  INPUT
```

Both records are exactly verified. Different traces with the same valid optimum
are reported as tie-breaking differences and remain equivalent. JSON comparison
artifacts embed normalized sequence text and omit timings and expanded operation
arrays so they can be attached to an issue and replayed without the original
FASTA/FASTQ files.

## Exit codes

| Code | Meaning |
|---:|---|
| 0 | Requested assurance passed for every record. |
| 1 | Invalid, incorrectly scored, or non-optimal record. |
| 2 | Any error record, or a usage, parsing, source-resolution, or configuration error. |
| 3 | Inconclusive verification or internal verification error. |

Precedence is `2`, then `1`, then `3`, then `0`.
