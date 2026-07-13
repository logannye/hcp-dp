# Contributing

HCP-DP is an alpha-stage project, but the public surface is expected to remain
truthful and well-tested. Prefer a small, proven capability over a broader API
whose exact score or traceback cannot be independently verified.

## Required Checks

Run before sending changes:

```bash
bash scripts/check.sh
```

Useful focused checks:

```bash
cargo test --lib --tests
cargo test --features tracing --lib --tests
cargo test --features tracing --lib --tests
cargo test --features experimental-hcp --lib --tests
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
cargo run --bin hcp-align -- solve edit-distance --query kitten --target sitting --cross-check
cargo run --bin hcp-align -- verify --alignment result.json --format json --query kitten --target sitting
python3 scripts/validate_external.py
cargo bench --bench oracle_perf
python3 scripts/perf_report.py
```

Coverage in CI uses `cargo-llvm-cov`.
External validation skips missing Parasail/Edlib packages by default; pass
`--required` only in an environment where those packages are installed.

## Capability Admission Rule

Do not add a mode, format, backend, or binding unless it directly improves
`solve`, `verify`, or `compare`, or responds to validated external demand.
Every alignment backend must prove:

- path-realized objective equals the reported objective,
- its objective agrees with the independent reference scorer,
- its canonical CIGAR round-trips through applicable formats,
- ambiguous optimal tracebacks are compared by objective equivalence.

Generic DP authoring APIs remain under `experimental-hcp` and have no stability
guarantee.

## Documentation Standards

Public docs should state capability boundaries directly. When adding a new mode
or backend, update the README, CLI reference, capability matrix, output schema
if needed, and changelog in the same change.
