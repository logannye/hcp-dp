# ADR 0001: Verification-First Alignment Oracle

Status: accepted for `0.2.0-alpha.1`.

## Decision

HCP-DP is positioned as an exact alignment reference generator, verifier, and
backend comparison tool. The public product is `hcp-align`; height-compressed DP
is one implementation technique and an experimental SDK.

The runtime remains pure Rust. Edlib, WFA2, Parasail, and other engines may be
used by optional differential-test and benchmark tooling but are not runtime
dependencies.

Public CIGAR uses SAM semantics. JSON/PAF coordinates are zero-based and
half-open. Exact optimality is the default verifier assurance and never silently
downgrades when its work budget is exceeded.

## Consequences

- The CLI is organized as `solve`, `verify`, and `compare`.
- JSON uses the new `hcp-align.v2` record family.
- Mapper-like and non-alignment proof points are removed from the supported
  codebase.
- The generic HCP authoring API is feature-gated and explicitly unstable.
- Integrity hashes are described as mutation-detection fingerprints, not
  cryptographic proof or authenticity.
