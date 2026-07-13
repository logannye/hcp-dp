# Migrating from `hcp-align.v1` to `hcp-align.v2`

Version 0.2 is an intentional pre-1.0 breaking reset.

## CLI

Prefix former alignment modes with `solve`:

```text
hcp-align edit-distance ...
    -> hcp-align solve edit-distance ...
```

- `--certificate` is now `--integrity`.
- `--verify` is removed. Tracebacks are always rescored; use `--cross-check` for
  an independent optimum comparison.
- `--score-only` is now `--scope score`.
- affine backend `wavefront` is now `banded-affine`.
- TSV, custom CIGAR output, and `seeded-global-linear` are removed.
- DTW, Viterbi, and layered-DAG modules are no longer product capabilities.

## CIGAR

Version 1 described operations as gaps in query/target. Version 2 uses SAM
orientation consistently: `I` consumes query and `D` consumes target. Consumers
must not reuse v1 indel-direction assumptions.

## JSON

The flat v1 object is replaced by nested sections:

```text
producer, pair, problem, alignment, verification, metrics, integrity
```

`verified: bool|null` and `verification_status` are replaced by explicit status
values: `verified`, `invalid`, `inconclusive`, and `error`. Scoring parameters
are now required deterministic data rather than certificate-only input.

The old `schemas/hcp-align.v1.schema.json` remains available only to identify
historical records. New output validates against `hcp-align.v2`.

## Rust

Use the stable oracle types and functions re-exported from `hcp_dp`. Existing
`HcpEngine`, `HcpProblem`, and problem modules move under
`hcp_dp::experimental::hcp` and require the `experimental-hcp` feature.
