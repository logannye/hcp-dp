# v0.2 Delivery Roadmap

This roadmap distinguishes implemented code from gates requiring scheduled or
external evidence.

## `0.2.0-alpha.1`: trustworthy oracle contract

Status: implementation complete; superseded by `0.2.0-alpha.2` before
publication.

- Canonical scoring, spans, provenance, objectives, and SAM-oriented CIGAR.
- Exact solve, structure/score/optimal verification, tie-aware comparison, and
  strict JSON v2/PAF/SAM boundaries.
- Independent rolling-frontier optimality checks and explicit work refusal.
- Generic HCP machinery isolated under `experimental-hcp`.

## `0.2.0-alpha.2`: bounded operation

Status: published and publicly install-verified on 2026-07-13. Scheduled
evidence continues to accumulate for the RC/stable gates.

- Incremental wrapped FASTA and multiline FASTQ parsing over plain or
  multi-member gzip streams.
- Deterministic bounded worker windows with count and active-base limits.
- A transient 16 MiB-cache redb store for arbitrary-ID resolution and a
  disk-backed SAM body/reference spool.
- Ordered recoverable error records, JSONL continuation, sidecars for other
  formats, and stable exit precedence.
- Deterministic 1,000/100,000-pair solve and arbitrary-order verify RSS gates:
  at most 16 MiB growth, 128 MiB absolute RSS, and 15% over the checked Linux
  baseline.

Gate: four consecutive daily scheduled RSS/performance runs must pass before
final `0.2.0`; manual dispatches never count. The generator and fixtures are
ephemeral, not stored in Git.

## `0.2.0-rc.1`: differential tooling

Status: implementation and hardening complete; the tag remains blocked until
four consecutive daily scheduled RSS and Criterion runs pass.

- Standalone Rust-Bio 4.0.1 adapter for all five modes, including affine gap
  convention translation.
- Pinned WFA2-lib 2.3.6 adapter for edit, global linear/affine, and semiglobal
  linear modes, with heuristics disabled and explicit unsupported cases.
- Exact verification after every adapter conversion and explicit WFA indel
  orientation normalization.
- `hcp-align.differential.v1` artifacts and deterministic delta reduction with
  evaluation, per-call, and wall-clock limits.
- A checked discrepancy allowlist requiring backend/version/signature/reason
  and an issue URL.
- Crate-private storage/scheduler/spooling internals, bounded Python diagnostic
  capture, release Criterion reports, and pinned Edlib/Parasail evidence.

Gate: adapter contract tests pass and every external difference is resolved or
represented by a reviewed allowlist entry.

## `0.2.0`: distribution and adoption

Status: publishing automation, both trusted publishers, release environments,
branch protection, alpha artifact validation, and the initial crates.io
publication are complete. Scheduled history, RC observation, and adoption
evidence remain. Five approved pilot invitations were opened on 2026-07-14;
the adoption count remains 0/3 until public completion evidence is available.

- Dry-run workflow builds and smoke-tests Linux x86_64/aarch64, macOS
  x86_64/arm64, and Windows x86_64 archives and platform wheels.
- Tag workflow checks Rust/Python/tag equality, reuses tested artifacts,
  generates SPDX SBOMs and consolidated checksums, keyless-signs the checksum
  manifest, attests artifacts, and publishes GitHub/PyPI releases.
- The first `0.2.0-alpha.2` crates.io publication was completed manually from
  its clean signed commit; later releases use trusted publishing.
- The `hcp-align` PyPI pending trusted publisher is registered for
  `logannye/hcp-dp`, `.github/workflows/release.yml`, and the `pypi`
  environment.
- Python generators stream solve/verify/compare JSONL and enforce an exact
  bundled-binary version.

Gate: protected `release` and `pypi` environments, required branch checks, four
daily scheduled performance passes, no untriaged discrepancies, clean install
smoke tests, and three external users independently validating records or
reproducing an alignment bug. Stable also requires seven days of RC observation
without an open `release-blocker`. Automation must not create the final tag
before the adoption gate is documented.

## Experimental HCP SDK

Status: demand-gated; no promotion work begins before oracle adoption.

Promotion requires a non-replay algebraic summary used during reconstruction,
a measured benefit, reduced authoring burden, and two external implementers.

## Admission rule

No algorithm, format, binding, or generic DP demonstration enters the stable
surface unless it directly improves exact solving, record verification, or
backend comparison, or answers a validated external-user request.
