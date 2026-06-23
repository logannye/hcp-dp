# 0.1 Alpha Release Checklist

This alpha is a source-install and GitHub-artifact release. Do not publish to
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
```

## Required GitHub Gates

- default CI green,
- coverage green,
- MSRV green,
- macOS, Windows, and Linux green,
- feature checks green,
- manual release validation workflow available.

## Manual Release Validation

Run the `Release Validation Report` workflow from GitHub Actions. It installs
Parasail and Edlib, runs required external validation, generates the report, and
uploads:

```text
target/hcp-dp-report/
```

## Alpha Readiness Criteria

- `hcp-align` supports single-record inline inputs and multi-record FASTA/FASTQ
  file inputs.
- Batch file mode rejects mismatched record counts.
- `json`, `jsonl`, `tsv`, and `cigar` outputs are tested.
- Independent path scoring is always computed.
- `--verify` reports `full`, `path_only`, or `failed` clearly.
- Windows CI may set `SCALE_PROBE_MAX_SIZE` to keep smoke runtime comfortably
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
