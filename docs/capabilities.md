# Capability Matrix

This matrix describes what the current alpha can claim. A row should not be
expanded until the module passes the contract harness and has user-facing
validation where applicable.

| Problem | Exact objective | Exact path | Summary laws | CLI support | External score validation | Benchmark/report coverage | Known caveat |
|---|---|---|---|---|---|---|---|
| LCS | yes | yes | yes | no | no | `scale_probe` | Library-only in this alpha. |
| Needleman-Wunsch, linear gap | yes | yes | yes | yes, `global-linear` | Parasail optional | `scale_probe`, report workflow | No SIMD runtime path; Parasail is validation only. |
| Needleman-Wunsch, affine gap | yes | yes | yes | yes, `global-affine` | Parasail optional after affine calibration | `scale_probe`, report workflow | Boundary state is explicit; still slower than linear modes. |
| Smith-Waterman, linear gap | yes | yes | yes | yes, `local-linear` | Parasail optional | `scale_probe`, report workflow | Returns the selected local traceback, not flanking unaligned regions. |
| Edit distance | yes | yes | yes | yes, `edit-distance` | Edlib optional | `scale_probe`, report workflow | Levenshtein distance only. |
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
```

The report directory contains:

```text
target/hcp-dp-report/report.md
target/hcp-dp-report/scale_probe.json
target/hcp-dp-report/external-validation.json
```

Manual GitHub release validation uploads the same directory as an artifact.
