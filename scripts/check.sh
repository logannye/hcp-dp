#!/usr/bin/env bash
set -euo pipefail

# Options
RUN_BENCH="${RUN_BENCH:-}"       # set to any non-empty to run benches
RELEASE_FLAG="${RELEASE:+--release}"
VERBOSE="${VERBOSE:-}"
say() { printf "\n[check] %s\n" "$*"; }

FEATURES="${FEATURES:-}"
if [[ -n "${FEATURES}" ]]; then
  say() { printf "\n[check %s] %s\n" "${FEATURES}" "$*"; }
  say "Enabling features: ${FEATURES}"
else
  say "Enabling features: (default)"
fi

SCALE_PROBE_MAX_SIZE="${SCALE_PROBE_MAX_SIZE:-}"
if [[ -z "${SCALE_PROBE_MAX_SIZE}" && "${RUNNER_OS:-}" == "Windows" ]]; then
  SCALE_PROBE_MAX_SIZE=512
fi

# Repo root
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR%/scripts}"
cd "${ROOT_DIR}"

require() { command -v "$1" >/dev/null 2>&1 || { echo "Missing command: $1"; exit 1; }; }

cargo_with_features() {
  local cmd="$1"
  shift || true
  local -a before
  local -a after
  before=()
  after=()
  while (($#)); do
    if [[ "$1" == "--" ]]; then
      after=("$@")
      break
    fi
    before+=("$1")
    shift || true
  done
  if [[ -n "${FEATURES}" ]]; then
    before+=("--features" "${FEATURES}")
  fi
  if [[ ${#after[@]} -gt 0 ]]; then
    cargo "$cmd" "${before[@]}" "${after[@]}"
  else
    cargo "$cmd" "${before[@]}"
  fi
}

parse_num() { # grep first integer from stdin
  grep -Eo '[-+]?[0-9]+' | head -n1 || true
}

parse_float() { # grep first float-like from stdin
  grep -Eo '[-+]?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?' | head -n1 || true
}

json_field() {
  local field="$1"
  python3 -c 'import json,sys; value=json.load(sys.stdin)[sys.argv[1]]; print(str(value).lower() if isinstance(value, bool) else value)' "$field"
}

jsonl_field() {
  local row="$1"
  local field="$2"
  python3 -c 'import json,sys; rows=[json.loads(line) for line in sys.stdin if line.strip()]; value=rows[int(sys.argv[1])][sys.argv[2]]; print(str(value).lower() if isinstance(value, bool) else value)' "$row" "$field"
}

assert_eq() { local got="$1" exp="$2" name="$3"; if [[ "$got" != "$exp" ]]; then echo "FAIL: $name (got=$got expected=$exp)"; exit 1; fi; }
assert_nonempty() { local v="$1" name="$2"; if [[ -z "$v" ]]; then echo "FAIL: $name empty"; exit 1; fi; }

require cargo
require rustc
require python3

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

say "Toolchain versions"
cargo --version
rustc --version

say "Formatting check"
cargo fmt --all -- --check

say "Clippy (deny warnings)"
cargo_with_features clippy --workspace --all-targets ${VERBOSE:+-v} -- -D warnings

say "Rustdoc (deny warnings)"
RUSTDOCFLAGS="-D warnings" cargo_with_features doc --workspace --no-deps --quiet

say "Build (${RELEASE:+release} build)"
cargo_with_features build --workspace --all-targets ${RELEASE_FLAG:-}

say "Unit tests"
cargo_with_features test --workspace --lib --tests --quiet ${RELEASE_FLAG:-}
# Guard: fail if no tests detected unless explicitly allowed.
if [[ -z "${ALLOW_ZERO_TESTS:-}" ]]; then
  TOTAL_TESTS=$(cargo_with_features test --workspace --lib --tests -- --list 2>/dev/null | grep -E '^[^ ]' | wc -l | tr -d ' ')
  if [[ "${TOTAL_TESTS}" == "0" ]]; then
    echo "FAIL: No tests detected. Set ALLOW_ZERO_TESTS=1 to bypass."
    exit 1
  fi
fi

say "Run example: lcs"
LCS_OUT=$(cargo_with_features run --quiet --example lcs ${RELEASE_FLAG:-})
LCS_LEN=$(printf '%s' "$LCS_OUT" | grep -E "^LCS length:" | parse_num)
assert_nonempty "$LCS_LEN" "LCS length"
# Classic pair has known LCS length 20
assert_eq "$LCS_LEN" "20" "LCS length"

say "Run example: align (Needleman-Wunsch)"
NW_OUT=$(cargo_with_features run --quiet --example align ${RELEASE_FLAG:-})
NW_SCORE=$(printf '%s' "$NW_OUT" | grep -E "^Global alignment score:" | parse_num)
assert_nonempty "$NW_SCORE" "NW score"
# For GATTACA vs GCATGCU with (+1,-1,-1), an optimal score is 0
assert_eq "$NW_SCORE" "0" "NW global alignment score"

say "Run example: smith_waterman"
SW_OUT=$(cargo_with_features run --quiet --example smith_waterman ${RELEASE_FLAG:-})
SW_SCORE=$(printf '%s' "$SW_OUT" | grep -E "^Local alignment score:" | parse_num)
assert_nonempty "$SW_SCORE" "Smith-Waterman score"
assert_eq "$SW_SCORE" "10" "Smith-Waterman local alignment score"

say "Run CLI: global-linear"
CLI_NW=$(cargo_with_features run --quiet --bin hcp-align ${RELEASE_FLAG:-} -- global-linear --query GATTACA --target GCATGCU --match 1 --mismatch-penalty 1 --gap -1 --verify --format json)
assert_eq "$(printf '%s' "$CLI_NW" | json_field score)" "0" "CLI global-linear score"
assert_eq "$(printf '%s' "$CLI_NW" | json_field verification_status)" "full" "CLI global-linear verification status"
assert_eq "$(printf '%s' "$CLI_NW" | json_field verified)" "true" "CLI global-linear verified"

say "Run CLI: global-affine"
CLI_AFFINE=$(cargo_with_features run --quiet --bin hcp-align ${RELEASE_FLAG:-} -- global-affine --query ACB --target A --match 2 --mismatch-penalty 1 --gap-open -3 --gap-extend -1 --verify --format json)
assert_eq "$(printf '%s' "$CLI_AFFINE" | json_field score)" "-3" "CLI global-affine score"
assert_eq "$(printf '%s' "$CLI_AFFINE" | json_field verified)" "true" "CLI global-affine verified"

say "Run CLI: local-linear"
CLI_SW=$(cargo_with_features run --quiet --bin hcp-align ${RELEASE_FLAG:-} -- local-linear --query ACACACTA --target AGCACACA --match 2 --mismatch-penalty 1 --gap -2 --verify --format json)
assert_eq "$(printf '%s' "$CLI_SW" | json_field score)" "10" "CLI local-linear score"
assert_eq "$(printf '%s' "$CLI_SW" | json_field verified)" "true" "CLI local-linear verified"

say "Run CLI: edit-distance"
CLI_EDIT=$(cargo_with_features run --quiet --bin hcp-align ${RELEASE_FLAG:-} -- edit-distance --query kitten --target sitting --verify --format json)
assert_eq "$(printf '%s' "$CLI_EDIT" | json_field schema_version)" "hcp-align.v1" "CLI schema version"
assert_eq "$(printf '%s' "$CLI_EDIT" | json_field distance)" "3" "CLI edit-distance distance"
assert_eq "$(printf '%s' "$CLI_EDIT" | json_field backend)" "adaptive-banded" "CLI edit-distance auto backend"
assert_eq "$(printf '%s' "$CLI_EDIT" | json_field verified)" "true" "CLI edit-distance verified"

say "Run CLI: edit-distance score-only"
CLI_EDIT_SCORE=$(cargo_with_features run --quiet --bin hcp-align ${RELEASE_FLAG:-} -- edit-distance --score-only --query kitten --target sitting --verify --format json)
assert_eq "$(printf '%s' "$CLI_EDIT_SCORE" | json_field distance)" "3" "CLI score-only edit-distance distance"
assert_eq "$(printf '%s' "$CLI_EDIT_SCORE" | json_field backend)" "myers" "CLI score-only edit-distance backend"
assert_eq "$(printf '%s' "$CLI_EDIT_SCORE" | json_field verification_status)" "full" "CLI score-only edit-distance verification"

say "Run CLI: edit-distance adaptive-banded"
CLI_EDIT_BANDED=$(cargo_with_features run --quiet --bin hcp-align ${RELEASE_FLAG:-} -- edit-distance --engine adaptive-banded --query kitten --target sitting --verify --format json)
assert_eq "$(printf '%s' "$CLI_EDIT_BANDED" | json_field distance)" "3" "CLI adaptive-banded edit-distance distance"
assert_eq "$(printf '%s' "$CLI_EDIT_BANDED" | json_field path_score)" "3" "CLI adaptive-banded edit-distance path score"
assert_eq "$(printf '%s' "$CLI_EDIT_BANDED" | json_field backend)" "adaptive-banded" "CLI adaptive-banded backend"

say "Run CLI: compact/full operation detail and output file"
CLI_EDIT_FULL=$(cargo_with_features run --quiet --bin hcp-align ${RELEASE_FLAG:-} -- edit-distance --query ACGT --target ACGA --operation-detail full --format json)
FULL_OP_COUNT=$(printf '%s' "$CLI_EDIT_FULL" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["operations"]))')
assert_eq "$FULL_OP_COUNT" "4" "CLI full operation count"
cargo_with_features run --quiet --bin hcp-align ${RELEASE_FLAG:-} -- edit-distance --query kitten --target sitting --verify --format json --output "${TMP_DIR}/edit.json"
assert_eq "$(cat "${TMP_DIR}/edit.json" | json_field distance)" "3" "CLI output file distance"

say "Run CLI: semiglobal-linear"
CLI_SEMI=$(cargo_with_features run --quiet --bin hcp-align ${RELEASE_FLAG:-} -- semiglobal-linear --query ACGT --target TTACGTTT --match 2 --mismatch-penalty 1 --gap -2 --verify --format json)
assert_eq "$(printf '%s' "$CLI_SEMI" | json_field score)" "8" "CLI semiglobal-linear score"
assert_eq "$(printf '%s' "$CLI_SEMI" | json_field target_start)" "2" "CLI semiglobal target start"
assert_eq "$(printf '%s' "$CLI_SEMI" | json_field target_end)" "6" "CLI semiglobal target end"
assert_eq "$(printf '%s' "$CLI_SEMI" | json_field verified)" "true" "CLI semiglobal-linear verified"

say "Run CLI: batch FASTA JSONL"
cat >"${TMP_DIR}/query.fa" <<'EOF'
>q1
AC
GT
>q2
AAAA
EOF
cat >"${TMP_DIR}/target.fa" <<'EOF'
>t1
ACGT
>t2
TTTT
EOF
CLI_BATCH_JSONL=$(cargo_with_features run --quiet --bin hcp-align ${RELEASE_FLAG:-} -- edit-distance --query-file "${TMP_DIR}/query.fa" --target-file "${TMP_DIR}/target.fa" --verify --format jsonl)
assert_eq "$(printf '%s\n' "$CLI_BATCH_JSONL" | wc -l | tr -d ' ')" "2" "CLI batch JSONL row count"
assert_eq "$(printf '%s' "$CLI_BATCH_JSONL" | jsonl_field 0 query_id)" "q1" "CLI batch JSONL query id"
assert_eq "$(printf '%s' "$CLI_BATCH_JSONL" | jsonl_field 1 distance)" "4" "CLI batch JSONL second distance"

say "Run CLI: batch FASTQ cigar"
cat >"${TMP_DIR}/query.fq" <<'EOF'
@fq1
AC
+
!!
@fq2
AA
+
!!
EOF
cat >"${TMP_DIR}/target.fq" <<'EOF'
@ft1
AC
+
!!
@ft2
AT
+
!!
EOF
CLI_BATCH_CIGAR=$(cargo_with_features run --quiet --bin hcp-align ${RELEASE_FLAG:-} -- global-linear --query-file "${TMP_DIR}/query.fq" --target-file "${TMP_DIR}/target.fq" --match 1 --mismatch-penalty 1 --gap -1 --format cigar)
assert_eq "$(printf '%s\n' "$CLI_BATCH_CIGAR" | wc -l | tr -d ' ')" "3" "CLI batch cigar row count"
printf '%s' "$CLI_BATCH_CIGAR" | grep -q 'pair_index	query_id	target_id' || { echo "FAIL: CLI cigar header missing"; exit 1; }

say "Run CLI: TSV and verify limit"
CLI_TSV=$(cargo_with_features run --quiet --bin hcp-align ${RELEASE_FLAG:-} -- local-linear --query ACACACTA --target AGCACACA --show-alignment --format tsv)
printf '%s' "$CLI_TSV" | grep -q 'pair_index	query_id	target_id' || { echo "FAIL: CLI TSV header missing"; exit 1; }
CLI_PATH_ONLY=$(cargo_with_features run --quiet --bin hcp-align ${RELEASE_FLAG:-} -- global-linear --query ACGT --target ACGT --verify --verify-limit 1 --format json)
assert_eq "$(printf '%s' "$CLI_PATH_ONLY" | json_field verification_status)" "path_only" "CLI verify-limit path-only status"

say "Run scale probe smoke"
SCALE_PROBE_ARGS=(--format table --verify-limit 128)
if [[ -n "${SCALE_PROBE_MAX_SIZE}" ]]; then
  SCALE_PROBE_ARGS+=(--max-size "${SCALE_PROBE_MAX_SIZE}")
fi
cargo_with_features run --quiet --bin scale_probe ${RELEASE_FLAG:-} -- "${SCALE_PROBE_ARGS[@]}" >/dev/null
cargo_with_features run --quiet --bin scale_probe ${RELEASE_FLAG:-} -- --mode edit-distance-deep --engine hcp --max-size 128 --format json >/dev/null
cargo_with_features run --quiet --bin scale_probe ${RELEASE_FLAG:-} -- --mode edit-distance-deep --engine adaptive-banded --max-size 2048 --format json >/dev/null
cargo_with_features run --quiet --bin scale_probe ${RELEASE_FLAG:-} -- --mode edit-distance-deep --engine adaptive-banded-path --max-size 2048 --format json >/dev/null
cargo_with_features run --quiet --bin scale_probe ${RELEASE_FLAG:-} -- --mode edit-distance-deep --engine myers --max-size 2048 --format json >/dev/null

if [[ -n "${RUN_BENCH}" ]]; then
  say "Benches (dev mode, faster)"
  export CRITERION_DEV_MODE=1
  cargo_with_features bench ${RELEASE_FLAG:-} || { echo "Bench failed"; exit 1; }
  python3 scripts/check_bench.py || exit 1
else
  say "Skipping benches (set RUN_BENCH=1 to enable)"
fi

say "All checks passed"
