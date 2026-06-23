> Sample artifact: this checked-in report was generated from a small bounded run on one machine. It demonstrates report structure and verification fields; it is not a universal performance claim. Regenerate local artifacts with `python3 scripts/perf_report.py`.

# HCP-DP Correctness And Performance Report

## Environment

- Timestamp: `example run`
- Commit: `local checkout used to generate the sample`
- Rust: `local Rust toolchain`
- Cargo: `local Cargo toolchain`
- OS: `local operating system`
- CPU: `local CPU`
- Command: `scripts/perf_report.py --scenario edit_distance --verify-limit 128 --max-size 128 --skip-external --sample-output docs/sample-report.md`

## Scale Probe

Raw JSON: `target/hcp-dp-report/scale_probe.json`

Status counts: `{"not_checked": 2, "passed": 18}`

| Scenario | Size | Wall s | RSS delta bytes | Peak RSS bytes | Status | Detail |
|---|---:|---:|---:|---:|---|---|
| edit_distance | 64 | 0.000992 | 4554752 | 11960320 | passed | distance=32 |
| edit_distance | 128 | 0.004723 | 720896 | 12730368 | passed | distance=64 |

## Edit Distance Deep Comparison

This section compares exact HCP edit-distance traceback engines and exact adaptive-banded traceback against dense, rolling-row, Myers bit-vector, and optional Edlib score baselines on fixed scenario families.

- `hcp`, `hcp-linear`, and `adaptive-banded-path` return exact reconstructed paths and independently scored path distances.
- `adaptive-banded`, `myers`, `myers-u64`, `linear-space`, `full-table`, and `edlib` are exact distance-only engines used to expose speed/memory regimes.

| Case | Engine | Query | Target | Wall s | Peak RSS bytes | Distance | Expected | Path score | Path len | Summary ms | Reconstruct ms | Verify ms | Status | Detail |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| small_regression | hcp | 6 | 7 | 0.000144 | 11354112 | 3 | 3 | 3 | 8 | 0.031 | 0.053 | 0.003 | passed | distance=3, path_len=8 |
| small_regression | hcp-linear | 6 | 7 | 0.000051 | 11468800 | 3 | 3 | 3 | 8 | 0.015 | 0.030 | 0.001 | passed | distance=3, block_size=1, path_len=8 |
| small_regression | adaptive-banded-path | 6 | 7 | 0.000023 | 11468800 | 3 | 3 | 3 | 8 | 0.000 | 0.018 | 0.002 | passed | distance=3, band=3, path_len=8 |
| small_regression | full-table | 6 | 7 | 0.000013 | 11534336 | 3 | 3 |  |  |  |  |  | passed | full_table_distance=3 |
| small_regression | linear-space | 6 | 7 | 0.000023 | 11534336 | 3 | 3 |  |  |  |  |  | passed | linear_space_distance=3 |
| small_regression | adaptive-banded | 6 | 7 | 0.000040 | 11567104 | 3 | 3 |  |  |  |  |  | passed | adaptive_banded_distance=3 |
| small_regression | myers | 6 | 7 | 0.000007 | 11583488 | 3 | 3 |  |  |  |  |  | passed | myers_distance=3 |
| small_regression | myers-u64 | 6 | 7 | 0.000007 | 11599872 | 3 | 3 |  |  |  |  |  | passed | myers_u64_distance=3 |
| small_regression | edlib | 6 | 7 | 0.102083 | 12484608 |  | 3 |  |  |  |  |  | not_checked | edlib unavailable or failed |
| short_myers_window | hcp | 64 | 64 | 0.000659 | 13107200 | 32 | 32 | 32 | 65 | 0.207 | 0.440 | 0.004 | passed | distance=32, path_len=65 |
| short_myers_window | hcp-linear | 64 | 64 | 0.001625 | 13254656 | 32 | 32 | 32 | 65 | 0.367 | 1.031 | 0.006 | passed | distance=32, block_size=1, path_len=65 |
| short_myers_window | adaptive-banded-path | 64 | 64 | 0.000915 | 13271040 | 32 | 32 | 32 | 65 | 0.000 | 0.900 | 0.006 | passed | distance=32, band=63, path_len=65 |
| short_myers_window | full-table | 64 | 64 | 0.005029 | 13336576 | 32 | 32 |  |  |  |  |  | passed | full_table_distance=32 |
| short_myers_window | linear-space | 64 | 64 | 0.000194 | 13975552 | 32 | 32 |  |  |  |  |  | passed | linear_space_distance=32 |
| short_myers_window | adaptive-banded | 64 | 64 | 0.001361 | 14041088 | 32 | 32 |  |  |  |  |  | passed | adaptive_banded_distance=32 |
| short_myers_window | myers | 64 | 64 | 0.000013 | 14041088 | 32 | 32 |  |  |  |  |  | passed | myers_distance=32 |
| short_myers_window | myers-u64 | 64 | 64 | 0.000008 | 14041088 | 32 | 32 |  |  |  |  |  | passed | myers_u64_distance=32 |
| short_myers_window | edlib | 64 | 64 | 0.074178 | 14123008 |  | 32 |  |  |  |  |  | not_checked | edlib unavailable or failed |

## External Validation

External validation was skipped.

Raw JSON: `target/hcp-dp-report/external-validation.json`

Status counts: `{}`

| Tool | Mode | Case | Status | HCP | External | Detail |
|---|---|---|---|---:|---:|---|

## Criterion

Criterion benches were not run. Pass `--benches` to include them.
