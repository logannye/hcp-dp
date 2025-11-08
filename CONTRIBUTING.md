# Contributing

## Prerequisites
- Rust toolchain (stable + MSRV defined in CI)
- `python3` for perf checks
- `cargo install grcov` if you want local coverage reports

## Quality Gates
- Run `bash scripts/check.sh` before opening a PR.
- For benches: `RUN_BENCH=1 bash scripts/check.sh` (updates are compared against `perf/baseline.json`).
- Feature-specific:
  - Parallel: `cargo test --features parallel --test parallel_equivalence`
  - Heavy scale: `cargo test --features heavy`
- Coverage: `RUSTFLAGS="-Zinstrument-coverage" LLVM_PROFILE_FILE="target/coverage/%p-%m.profraw" cargo test` then `grcov`.

## Workflow
1. Fork & clone the repo, create a branch.
2. Make changes with tests.
3. Update docs (README/CONTRIBUTING) if commands/configs change.
4. Ensure `perf/baseline.json` is updated only when benchmarking environment is stable. If you update numbers, explain the hardware/conditions in the PR.
5. Open a PR; CI runs lint/test matrices, coverage, and scheduled perf jobs.


