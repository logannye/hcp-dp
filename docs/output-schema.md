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
| `error` | Recoverable streaming-batch failure. |

Alignment records contain `producer`, `pair`, `problem`, `alignment`, and
`verification`; `metrics` and `integrity` are optional. Problem scoring is
complete and deterministic. Timing metrics are never covered by integrity
fingerprints.

Comparison records embed normalized query and target sequence text so a
disagreement artifact is portable. Their nested alignment records omit timing,
expanded operations, and integrity fields to keep the artifact compact; backend
producer metadata, scoring, objectives, spans, and CIGAR remain present.

Error records contain producer metadata, optional pair/ID/line context, and a
stable `stage`, `code`, `message`, and `recoverable` flag. Stages are
`input_parse`, `record_parse`, `resolve`, `solve`, `verify`, `compare`,
`serialize`, and `worker`. JSONL emits recoverable errors in input order;
non-JSONL formats require an error sidecar to continue.

The development-only disagreement/reduction contract has a distinct schema,
[`schemas/hcp-align.differential.v1.schema.json`](../schemas/hcp-align.differential.v1.schema.json).
It includes backend versions, the preserved category set, and a deterministic
SHA-256 fingerprint over the exact normalized disagreement. It must not be
confused with stable oracle records.

Pinned Edlib/Parasail release evidence has another distinct contract,
[`schemas/hcp-align.external-validation.v1.schema.json`](../schemas/hcp-align.external-validation.v1.schema.json).
It records exact validator versions and treats malformed translations or
untriaged objective differences as release failures while keeping competitor
results informational.

Status is one of `verified`, `invalid`, `inconclusive`, or `error`. Assurance is
one of `structure`, `score`, or `optimal`. Consumers must inspect both instead
of treating a missing boolean as success.

CIGAR uses SAM orientation: `I` consumes query and `D` consumes target. JSON
coordinates are zero-based and half-open.
