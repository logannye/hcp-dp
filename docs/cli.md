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
| 2 | Usage, parsing, source-resolution, or configuration error. |
| 3 | Inconclusive verification or internal verification error. |
