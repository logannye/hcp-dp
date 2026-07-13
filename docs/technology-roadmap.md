# v0.2 Delivery Roadmap

This roadmap is ordered by end-user evidence. A later milestone does not widen
the supported product until its gate is met.

## Alpha 1: trustworthy oracle contract

Status: implemented on the v0.2 development branch.

- Canonical scoring, span, provenance, objective, and SAM-oriented CIGAR types.
- Exact `solve` for the five supported modes with unconditional path rescoring.
- Strict JSON v2, PAF, and SAM import/export.
- Structure, score, and exact-optimal verification with a 25,000,000-cell
  default refusal budget and explicit `inconclusive` status.
- Objective-first, tie-aware `compare` with portable compact artifacts.
- Mutation-specific checks, exhaustive small binary pairs, and a versioned
  differential corpus covering empty inputs, repeats, homopolymers, skew,
  long gaps, local resets, free ends, and ambiguous optima.
- Pure-Rust runtime, gzip sequence input, Python subprocess wrapper, observed
  solver/verifier/parser baselines, and fixed Criterion registration.
- Generic HCP machinery isolated under `experimental-hcp`.

Release candidate: `0.2.0-alpha.1` after every GitHub workflow passes from the
same commit.

## Alpha 2: bounded batch operation

Status: next implementation milestone.

- Incremental FASTA/FASTQ decoding rather than whole-file accumulation.
- Bounded in-flight parallel work with deterministic ordered JSONL output.
- Recoverable per-record `error` records and continuation after pair failures.
- Peak-RSS scenarios and checked 15% memory budgets.
- A 100,000-pair batch gate proving memory is proportional to active pairs, not
  total record count.
- Disagreement-preserving corpus reduction for internal and external adapters.

No alpha-1 documentation claims these operational properties.

## 0.2 release: distribution and adoption

Status: requires external systems and users.

- Publish the crate and generated API documentation after API review.
- Build Linux, macOS, and Windows artifacts from one tag with checksums, SBOM,
  signed provenance, and the v2 schema/migration guide.
- Package the Python client with batch APIs and deterministic binary resolution.
- Publish reproducible differential reports for Edlib, WFA2, Parasail, and
  Rust-Bio where their scoring conventions can be translated exactly.
- Complete discovery with five aligner/library maintainers and hands-on pilots
  with at least three; turn permitted failures into regression fixtures.

The release gate is three external users independently verifying real records
or reproducing a backend bug.

## Experimental HCP SDK

Status: demand-gated; no promotion work begins before oracle adoption.

Promotion requires a non-replay algebraic summary used during reconstruction,
a measured benefit, reduced authoring burden, and two external implementers.
Until then, every HCP authoring API may change without migration support.

## Admission rule

No algorithm, format, binding, or generic DP demonstration enters the stable
surface unless it directly improves exact solving, record verification, or
backend comparison, or answers a validated external-user request.
