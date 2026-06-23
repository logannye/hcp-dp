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
| edit_distance | 64 | 0.000863 | 3751936 | 11239424 | passed | distance=32 |
| edit_distance | 128 | 0.002084 | 114688 | 11370496 | passed | distance=64 |

## Edit Distance Deep Comparison

This section compares exact HCP edit-distance traceback engines and exact adaptive-banded traceback against dense, rolling-row, Myers bit-vector, and optional Edlib score baselines on fixed scenario families.

- `hcp`, `hcp-linear`, and `adaptive-banded-path` return exact reconstructed paths and independently scored path distances.
- `adaptive-banded`, `myers`, `myers-u64`, `linear-space`, `full-table`, and `edlib` are exact distance-only engines used to expose speed/memory regimes.

| Case | Engine | Query | Target | Wall s | Peak RSS bytes | Distance | Expected | Path score | Path len | Summary ms | Reconstruct ms | Verify ms | Status | Detail |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| small_regression | hcp | 6 | 7 | 0.000095 | 11206656 | 3 | 3 | 3 | 8 | 0.020 | 0.037 | 0.002 | passed | distance=3, path_len=8 |
| small_regression | hcp-linear | 6 | 7 | 0.000030 | 11255808 | 3 | 3 | 3 | 8 | 0.009 | 0.017 | 0.001 | passed | distance=3, block_size=1, path_len=8 |
| small_regression | adaptive-banded-path | 6 | 7 | 0.000031 | 11386880 | 3 | 3 | 3 | 8 | 0.000 | 0.028 | 0.001 | passed | distance=3, band=3, path_len=8 |
| small_regression | full-table | 6 | 7 | 0.000012 | 11419648 | 3 | 3 |  |  |  |  |  | passed | full_table_distance=3 |
| small_regression | linear-space | 6 | 7 | 0.000008 | 11436032 | 3 | 3 |  |  |  |  |  | passed | linear_space_distance=3 |
| small_regression | adaptive-banded | 6 | 7 | 0.000011 | 11436032 | 3 | 3 |  |  |  |  |  | passed | adaptive_banded_distance=3 |
| small_regression | myers | 6 | 7 | 0.000004 | 11501568 | 3 | 3 |  |  |  |  |  | passed | myers_distance=3 |
| small_regression | myers-u64 | 6 | 7 | 0.000011 | 11534336 | 3 | 3 |  |  |  |  |  | passed | myers_u64_distance=3 |
| small_regression | edlib | 6 | 7 | 0.059652 | 12156928 |  | 3 |  |  |  |  |  | not_checked | edlib unavailable or failed |
| short_myers_window | hcp | 64 | 64 | 0.000374 | 12435456 | 32 | 32 | 32 | 65 | 0.119 | 0.244 | 0.003 | passed | distance=32, path_len=65 |
| short_myers_window | hcp-linear | 64 | 64 | 0.000484 | 12500992 | 32 | 32 | 32 | 65 | 0.123 | 0.355 | 0.003 | passed | distance=32, block_size=1, path_len=65 |
| short_myers_window | adaptive-banded-path | 64 | 64 | 0.000606 | 12533760 | 32 | 32 | 32 | 65 | 0.000 | 0.602 | 0.002 | passed | distance=32, band=63, path_len=65 |
| short_myers_window | full-table | 64 | 64 | 0.000140 | 12582912 | 32 | 32 |  |  |  |  |  | passed | full_table_distance=32 |
| short_myers_window | linear-space | 64 | 64 | 0.000122 | 12582912 | 32 | 32 |  |  |  |  |  | passed | linear_space_distance=32 |
| short_myers_window | adaptive-banded | 64 | 64 | 0.000530 | 12582912 | 32 | 32 |  |  |  |  |  | passed | adaptive_banded_distance=32 |
| short_myers_window | myers | 64 | 64 | 0.000007 | 12550144 | 32 | 32 |  |  |  |  |  | passed | myers_distance=32 |
| short_myers_window | myers-u64 | 64 | 64 | 0.000006 | 12566528 | 32 | 32 |  |  |  |  |  | passed | myers_u64_distance=32 |
| short_myers_window | edlib | 64 | 64 | 0.046921 | 12615680 |  | 32 |  |  |  |  |  | not_checked | edlib unavailable or failed |

## External Validation

External validation was skipped.

Raw JSON: `target/hcp-dp-report/external-validation.json`

Status counts: `{}`

| Tool | Mode | Case | Status | HCP | External | Detail |
|---|---|---|---|---:|---:|---|

## Criterion

Criterion benches were not run. Pass `--benches` to include them.
