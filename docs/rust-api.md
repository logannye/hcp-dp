# Supported Rust API for v0.2

The supported product API is exposed from the `hcp_dp` crate root. It is
organized around canonical alignment records and the same exact operations as
the CLI:

- canonical alignment types: `AlignmentRecord`, `AlignmentMode`,
  `AlignmentSpan`, `Cigar`, `CigarOp`, and their directly contained record
  types;
- scoring types: `ScoringScheme`, `GapModel`, `SubstitutionScheme`, and
  `SubstitutionMatrix`;
- exact solving: `SolveRequest`, `SolveOptions`, `SolveScope`, and `solve`;
- verification: `VerificationRequest`, `VerificationLevel`,
  `VerificationReport`, and `verify`;
- comparison: `ComparisonInputs`, `ComparisonReport`, and `compare`;
- JSON, JSONL, PAF, and SAM conversion functions re-exported from the crate
  root;
- batch error/summary types and incremental sequence readers.

The transient redb sequence index, temporary SAM spool, worker pool, ordered
scheduler, and CLI implementation are crate-private. They may change without a
migration notice and are intentionally absent from generated public API docs.

The height-compressed implementation is available only with the
`experimental-hcp` feature under `experimental::hcp`. Nothing in that module
has a v0.2 stability guarantee.

HCP-DP is pre-1.0, so a future minor release may still make intentional
breaking changes. Within the v0.2 release line, changes to the documented root
surface, canonical semantics, or serialized contracts require a changelog and
migration note.
