# Capability Matrix

This matrix is the public claim boundary for the current alpha. A row should not
be expanded until the module passes the contract harness and has user-facing
validation where applicable.

| Problem | Exact objective | Exact path | Summary laws | CLI support | External score validation | Benchmark/report coverage | Known caveat |
|---|---|---|---|---|---|---|---|
| LCS | yes | yes | yes | no | no | `scale_probe` | Library-only in this alpha. |
| Needleman-Wunsch, linear gap | yes | yes | yes | yes, `global-linear` | Parasail optional | `scale_probe`, report workflow | No SIMD runtime path; Parasail is validation only. |
| Needleman-Wunsch, affine gap | yes | yes | yes | yes, `global-affine` | Parasail optional after affine calibration | `scale_probe`, report workflow | Boundary state is explicit; still slower than linear modes. |
| Smith-Waterman, linear gap | yes | yes | yes | yes, `local-linear` | Parasail optional | `scale_probe`, report workflow | Returns the selected local traceback, not flanking unaligned regions. |
| Edit distance, auto backend | yes | yes | n/a | yes, default `edit-distance` | Edlib optional | CLI smoke tests, deep comparison report | Deterministically selects exact adaptive-banded traceback or HCP linear-space fallback. |
| Edit distance, HCP traceback | yes | yes | yes | yes, `edit-distance --engine hcp` | Edlib optional | deep comparison report, `hcp` and `hcp-linear` engines | Generic summary-tree traceback. |
| Edit distance, adaptive banded | yes | yes | n/a | yes, `edit-distance --engine adaptive-banded` | checked against linear-space baseline; Edlib optional | deep comparison report, `adaptive-banded-path` engine | Exact specialized traceback; fastest when final edit distance is small. |
| Edit distance, Myers bit-vector | yes | no | n/a | yes, `edit-distance --score-only` | checked against linear-space baseline | deep comparison report, `myers` engine | Exact distance only; arbitrary pattern length. |
| Edit distance, Myers u64 | yes | no | n/a | report tool only | checked against linear-space baseline | deep comparison report, `myers-u64` engine | Exact distance only; pattern length must be at most 64 symbols. |
| Semi-global, linear gap | yes | yes | yes | yes, `semiglobal-linear` | no external anchor yet | `scale_probe`, report workflow | Full query against any target interval; swap inputs for the opposite orientation. |

## Contract Harness

Exported problems are expected to satisfy:

- summary apply equals direct interval replay,
- merged summary equals direct combined interval,
- split boundary is feasible for both recursive halves,
- reconstructed path joins exactly,
- independent path-realized objective equals reported objective,
- full-table baseline equals HCP result where applicable.

## Validation Artifacts

Default CI is network-independent. Optional external validation is available via:

```bash
python3 scripts/validate_external.py
python3 scripts/validate_external.py --required
```

The validator writes:

```text
target/hcp-dp-report/external-validation.json
```

Performance/correctness reports are generated with:

```bash
python3 scripts/perf_report.py
python3 scripts/perf_report.py --scenario edit_distance --verify-limit 128
```

The report directory contains:

```text
target/hcp-dp-report/report.md
target/hcp-dp-report/scale_probe.json
target/hcp-dp-report/external-validation.json
```

Manual GitHub release validation uploads the same directory as an artifact.

The edit-distance deep report compares:

- `hcp`: default square-root checkpoint exact traceback,
- `hcp-linear`: exact traceback with `block_size = 1`,
- `adaptive-banded-path`: exact traceback for low-edit regimes,
- `full-table`: dense score baseline,
- `linear-space`: rolling-row score baseline,
- `adaptive-banded`: exact score-only baseline for low-edit regimes,
- `myers`: exact bit-vector score baseline for arbitrary pattern lengths,
- `myers-u64`: exact bit-vector score baseline for short patterns,
- `edlib`: optional external exact score anchor.

Use:

```bash
cargo run --bin scale_probe -- --mode edit-distance-deep --format table
```
