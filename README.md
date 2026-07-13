# HCP-DP / `hcp-align`

[![CI](https://github.com/logannye/hcp-dp/actions/workflows/ci.yml/badge.svg)](https://github.com/logannye/hcp-dp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/github/license/logannye/hcp-dp.svg)](LICENSE)

`hcp-align` is a pure-Rust exact alignment oracle. It generates trusted reference
alignments, verifies JSON/PAF/SAM records against their source sequences, and
compares alignment backends without treating different but equally optimal
tracebacks as failures.

It is designed for aligner maintainers, bioinformatics infrastructure engineers,
and pipeline QA teams investigating CIGAR, coordinate, scoring, and optimality
problems. It is not a read mapper, reference index, variant caller, or claim to
be the fastest alignment implementation.

## Why use it?

- Produce an exact reference result for a sequence pair without depending on a
  production mapper's heuristics or tie-breaking policy.
- Find malformed CIGARs, spans, tags, digests, scoring conventions, and false
  optimality claims with named, auditable checks.
- Compare backends by validity and objective first, preserving different but
  equally optimal alignments as legitimate results.
- Export a compact disagreement artifact with normalized inputs, scoring, and
  backend provenance for regression tests and bug reports.

## Install

```bash
cargo install --git https://github.com/logannye/hcp-dp --bin hcp-align
```

The v0.2 series is a breaking pre-1.0 reset; see the migration guide before
upgrading an existing CLI, JSON, or Rust integration.

## Generate an exact reference

```bash
hcp-align solve edit-distance \
  --query kitten \
  --target sitting \
  --cross-check \
  --integrity \
  --format json
```

The result is an `hcp-align.v2` alignment record containing normalized sequence
identities and digests, the complete problem definition, exact objective,
canonical SAM-oriented CIGAR, named checks, backend provenance, optional
metrics, and an optional integrity fingerprint.

Supported solve modes:

- `edit-distance`
- `global-linear`
- `global-affine`
- `local-linear`
- `semiglobal-linear`

FASTA and FASTQ sources may be plain text or gzip-compressed (`.gz`).

Use `--scope score` for an objective without traceback. Every traceback is
rescored independently. `--cross-check` additionally compares the objective
with the rolling-frontier reference implementation.

## Verify external records

Native JSON embeds the alignment mode and scoring definition:

```bash
hcp-align verify \
  --alignment result.json \
  --format json \
  --query-file queries.fa \
  --target-file references.fa \
  --output-format json
```

PAF and SAM require an explicit problem definition because those formats do not
fully encode alignment scoring:

```bash
hcp-align verify \
  --alignment records.sam \
  --format sam \
  --query-file queries.fa \
  --target-file references.fa \
  --mode global-affine \
  --match 2 \
  --mismatch-penalty 1 \
  --gap-open -6 \
  --gap-extend -1
```

Verification levels:

| Level | Checks |
|---|---|
| `structure` | Schema, identifiers, lengths, spans, CIGAR legality and consumption. |
| `score` | Structure plus bases and independently realized objective. |
| `optimal` | Score plus agreement with an exact rolling-frontier optimum. Default. |

Exact verification refuses work above 25,000,000 DP cells by default and reports
`inconclusive`; it never silently weakens the requested assurance. Change the
limit with `--max-cells`, use `--max-cells 0` for unlimited work, or explicitly
request `--level score`.

Exit codes are stable:

- `0`: every record satisfies the requested level;
- `1`: at least one record is invalid or non-optimal;
- `2`: command, input, or configuration error;
- `3`: at least one record is inconclusive or verification encountered an
  internal error.

## Compare results

```bash
hcp-align compare \
  --left backend-a.json --left-format json \
  --right backend-b.json --right-format json \
  --query-file queries.fa \
  --target-file references.fa \
  --output-format json
```

Comparison first verifies both records exactly. Equal optimal objectives are
considered equivalent even when tie-breaking produces different CIGARs.

## Canonical semantics

Public CIGAR follows SAM orientation:

- `=` and `X` consume query and target;
- `I` consumes query only;
- `D` consumes target only.

JSON and PAF coordinates are zero-based and half-open. SAM conversion is
isolated at the format boundary. The verifier currently accepts primary
forward-strand SAM/PAF records; reverse-strand normalization and BAM/CRAM are
out of scope for v0.2.

## Rust API

The supported library surface includes:

- `AlignmentRecord`, `AlignmentMode`, `AlignmentSpan`, `Cigar`, and `CigarOp`;
- `ScoringScheme`, `GapModel`, and `SubstitutionScheme`;
- `SolveRequest`, `SolveOptions`, and `solve`;
- `VerificationRequest`, `VerificationLevel`, `VerificationReport`, and `verify`;
- JSON, JSONL, PAF, and SAM readers/writers under `hcp_dp::oracle`.

The generic height-compressed DP machinery is intentionally unstable and only
available with:

```toml
hcp-dp = { version = "0.2.0-alpha.1", features = ["experimental-hcp"] }
```

It lives under `hcp_dp::experimental::hcp` and carries no v0.2 compatibility
promise.

## Integrity fingerprints

`--integrity` hashes deterministic record fields. Timings are excluded. This
detects mutation when the expected digest is trusted separately; it is not a
signature, identity proof, or authenticity guarantee.

## Formats and scope

Supported contracts are JSON, JSONL, PAF, SAM, and human-readable text. BAM,
CRAM, TSV, custom CIGAR tables, read mapping, reference indexing, reverse-strand
mapping, and mapping-quality estimation are not supported.

## Development

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
bash scripts/check.sh
```

See:

- [CLI reference](docs/cli.md)
- [v1 to v2 migration](docs/migration-v2.md)
- [verification-first decision record](docs/adr/0001-verification-first.md)
- [output schema](docs/output-schema.md)
- [capability boundary](docs/capabilities.md)

## License

MIT.
