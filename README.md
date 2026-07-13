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

The latest published alpha is distributed as platform wheels, a crates.io
package, and signed GitHub release archives. Pin it explicitly. The `main`
branch now identifies itself as the unreleased `0.2.0-rc.1` candidate.

Python wheels bundle the matching `hcp-align` executable:

```bash
python -m pip install 'hcp-align==0.2.0a2'
```

Install the Rust binary from crates.io:

```bash
cargo install hcp-dp --version '=0.2.0-alpha.2' --locked
```

Download a platform archive from the
[v0.2.0-alpha.2 GitHub release](https://github.com/logannye/hcp-dp/releases/tag/v0.2.0-alpha.2),
verify it against `SHA256SUMS`, extract it, and run `hcp-align --version`.

To build the exact signed source tag instead:

```bash
cargo install --git https://github.com/logannye/hcp-dp \
  --tag v0.2.0-alpha.2 --locked --bin hcp-align
```

The v0.2 series is a breaking pre-1.0 reset; see the migration guide before
upgrading an existing CLI, JSON, or Rust integration.

Rust integrations should use the documented crate-root API. Storage,
scheduling, spooling, and CLI implementation types are deliberately private;
see [`docs/rust-api.md`](docs/rust-api.md) for the supported v0.2 surface.

Maintainers and pipeline engineers can use the concise
[`docs/pilot-guide.md`](docs/pilot-guide.md) to validate a real corpus or report
a sanitized backend disagreement as stable-release evidence.

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
Parsing is incremental, including wrapped FASTA, multiline FASTQ, and
multi-member gzip streams. Quality text is length-checked and discarded.

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
- `2`: at least one recoverable record error, or a command/input/configuration
  error;
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

## Bounded batches and record errors

All commands support `--threads`, `--max-in-flight`, and
`--max-in-flight-bases`. Defaults are at most eight workers, twice that many
in-flight records, and 64 million active bases. Output order is deterministic;
submission cannot advance beyond the bounded window when an early pair is
slow. A pair larger than the base budget runs alone.

JSONL is the recommended batch format. A malformed JSONL/PAF/SAM line or
record-scoped sequence failure produces an ordered `record_type: "error"` and
processing continues. Use `--fail-fast` to stop. JSON, PAF, SAM, and text remain
fail-fast unless `--errors errors.jsonl` provides a sidecar. Corrupt gzip,
duplicate source IDs, malformed JSON arrays, and temporary-store failures are
fatal because later results cannot be trusted.

`verify` and `compare` resolve arbitrary IDs through a temporary disk-backed
store with a 16 MiB cache (`--temp-dir` selects its parent). Temporary disk may
scale with source data; resident memory does not. This store is deleted on drop
and is not a persistent reference index.

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
hcp-dp = { version = "0.2.0-alpha.2", features = ["experimental-hcp"] }
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

## Differential development tooling

The standalone package under `tools/differential` pins Rust-Bio 4.0.1 and
WFA2-lib 2.3.6 without adding either to the runtime crate. It validates every
converted path with the oracle, emits `hcp-align.differential.v1` artifacts,
and deterministically reduces disagreements while preserving their exact
category set. External engines remain development and release-report tooling.

## Python

Platform wheels use the distribution name `hcp-align` and import name
`hcp_align`. `solve_batch`, `verify_batch`, and `compare_batch` are generators
over JSONL. Wheels bundle the matching binary and reject version mismatches;
resolution order is explicit constructor path, bundled binary,
`HCP_ALIGN_BIN`, then `PATH`.

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
