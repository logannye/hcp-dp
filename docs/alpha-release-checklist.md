# 0.1 Alpha Release Checklist

This alpha is a source-install and GitHub-binary release. Do not publish to
crates.io and do not rename the repository in this round.

## Required Local Gates

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
bash scripts/check.sh
FEATURES=parallel bash scripts/check.sh
FEATURES=tracing bash scripts/check.sh
FEATURES='parallel tracing' bash scripts/check.sh
cargo +1.88.0 check --workspace --all-targets
cargo bench --no-run
python3 scripts/validate_external.py
python3 scripts/perf_report.py --scenario edit_distance --verify-limit 128
python3 scripts/perf_report.py --scenario edit_distance --verify-limit 128 --max-size 128 --skip-external --sample-output docs/sample-report.md
cargo run --bin scale_probe -- --mode edit-distance-deep --engine hcp --max-size 128 --format json
cargo run --bin scale_probe -- --scenario dtw --max-size 128 --verify-limit 128 --format json
```

## Required GitHub Gates

- default CI green,
- coverage green,
- MSRV green,
- macOS, Windows, and Linux green,
- feature checks green,
- manual release validation workflow available,
- manual release alpha workflow available.

## Manual Release Validation

Run the `Release Validation Report` workflow from GitHub Actions. It installs
Parasail and Edlib, runs required external validation, generates the report, and
uploads:

```text
target/hcp-dp-report/
```

## Manual Alpha Binary Release

Run the `Release Alpha` workflow from GitHub Actions. It builds release-mode
`hcp-align` binaries for Linux, macOS, and Windows, uploads SHA-256 checksums,
and uploads the same correctness/performance report artifact.

## Alpha Readiness Criteria

- `hcp-align` supports single-record inline inputs and multi-record FASTA/FASTQ
  file inputs.
- Batch file mode rejects mismatched record counts.
- `json`, `jsonl`, `tsv`, `cigar`, `paf`, and `sam` outputs are tested.
- Independent path scoring is always computed.
- `--verify` reports `full`, `path_only`, or `failed` clearly.
- JSON/JSONL includes `schema_version` and `engine`.
- JSON/JSONL defaults to compact operation counts; full operations are opt-in.
- `--output`, `--progress`, `--continue-on-error`, and `--threads` are tested.
- `edit-distance --engine auto` is tested and reports the selected backend.
- Edit-distance deep reports compare HCP, full-table, linear-space, and optional
  Edlib engines, plus adaptive-banded traceback and Myers score baselines.
- Dynamic time warping is available as a library/report proof point for
  non-sequence DP generality and passes the same contract harness.
- `docs/sample-report.md` is generated from a small bounded run and clearly
  labeled as an example artifact, not a benchmark claim.
- Windows CI sets `SCALE_PROBE_MAX_SIZE=512` to keep smoke runtime comfortably
  below the job budget while local probes keep the larger default sizes.
- Smith-Waterman and semi-global traceback avoid repeated full selected-path
  reconstruction during split selection.
- README and CLI docs state scoring conventions, verification semantics, and
  limitations conservatively.

## Deferred

- crates.io publishing,
- repository rename,
- one-vs-many and all-vs-all batch modes,
- wrapped FASTQ parsing,
- SAM/BAM/PAF output,
- protein substitution matrices,
- enforced performance budgets,
- public claims beyond the rows in `docs/capabilities.md`.
