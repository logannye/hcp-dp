# `hcp-align.v2` Output Schema

The formal JSON Schema is
[`schemas/hcp-align.v2.schema.json`](../schemas/hcp-align.v2.schema.json).
Version 1 remains checked in only for historical-record identification.

Version 2 is a tagged record family:

| `record_type` | Producer |
|---|---|
| `alignment` | `hcp-align solve` or imported external alignment |
| `verification` | `hcp-align verify` |
| `comparison` | `hcp-align compare` |
| `error` | Reserved for recoverable streaming-batch failures. |

Alignment records contain `producer`, `pair`, `problem`, `alignment`, and
`verification`; `metrics` and `integrity` are optional. Problem scoring is
complete and deterministic. Timing metrics are never covered by integrity
fingerprints.

Comparison records embed normalized query and target sequence text so a
disagreement artifact is portable. Their nested alignment records omit timing,
expanded operations, and integrity fields to keep the artifact compact; backend
producer metadata, scoring, objectives, spans, and CIGAR remain present.

`error` is part of the v2 family so streaming producers can preserve ordering,
but `0.2.0-alpha.1` still treats parse and source-resolution failures as command
errors with exit code 2 rather than emitting recoverable error records.

Status is one of `verified`, `invalid`, `inconclusive`, or `error`. Assurance is
one of `structure`, `score`, or `optimal`. Consumers must inspect both instead
of treating a missing boolean as success.

CIGAR uses SAM orientation: `I` consumes query and `D` consumes target. JSON
coordinates are zero-based and half-open.
