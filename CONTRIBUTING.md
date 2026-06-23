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
cargo test --features parallel --lib --tests
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
cargo run --bin scale_probe -- --format table --verify-limit 512
cargo run --bin hcp-align -- edit-distance --query kitten --target sitting --verify
python3 scripts/validate_external.py
python3 scripts/perf_report.py --scenario edit_distance
```

Coverage in CI uses `cargo-llvm-cov`.
External validation skips missing Parasail/Edlib packages by default; pass
`--required` only in an environment where those packages are installed.

## Problem Admission Rule

Do not export a new `problems::*` module until tests prove:

- summary apply equals direct recurrence replay,
- merged summaries equal the direct combined interval,
- split boundaries are endpoint-constrained and feasible,
- reconstructed segments join exactly,
- path-realized objective equals reported objective.

If the proof is incomplete, keep the module private or behind an experimental
feature.

## Documentation Standards

Public docs should state capability boundaries directly. When adding a new mode
or backend, update the README, CLI reference, capability matrix, output schema
if needed, and changelog in the same change.
