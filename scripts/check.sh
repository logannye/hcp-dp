#!/usr/bin/env bash
set -euo pipefail

RUN_BENCH="${RUN_BENCH:-}"
RELEASE_FLAG="${RELEASE:+--release}"
FEATURES="${FEATURES:-}"
VERBOSE="${VERBOSE:-}"

say() { printf '\n[check] %s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }
require() { command -v "$1" >/dev/null 2>&1 || fail "missing command: $1"; }

cargo_with_features() {
  local subcommand="$1"
  shift
  local -a before=()
  local -a after=()
  local separator=0
  for argument in "$@"; do
    if [[ "${argument}" == "--" && "${separator}" == "0" ]]; then
      separator=1
      after+=(--)
    elif [[ "${separator}" == "0" ]]; then
      before+=("${argument}")
    else
      after+=("${argument}")
    fi
  done
  if [[ -n "${FEATURES}" ]]; then
    before+=(--features "${FEATURES}")
  fi
  if (( ${#after[@]} > 0 )); then
    cargo "${subcommand}" "${before[@]}" "${after[@]}"
  else
    cargo "${subcommand}" "${before[@]}"
  fi
}

json_value() {
  local path="$1"
  python3 -c '
import json, sys
value = json.load(sys.stdin)
for component in sys.argv[1].split("."):
    value = value[component]
if isinstance(value, bool):
    print(str(value).lower())
elif value is None:
    print("null")
else:
    print(value)
' "${path}"
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR%/scripts}"
cd "${ROOT_DIR}"

require cargo
require rustc
require python3

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

say "Toolchain versions"
cargo --version
rustc --version

say "Formatting"
cargo fmt --all -- --check

say "Public Rust API boundary"
python3 scripts/check_public_api.py

say "Clippy"
cargo_with_features clippy --workspace --all-targets ${VERBOSE:+-v} -- -D warnings

say "Rustdoc"
RUSTDOCFLAGS="-D warnings" cargo_with_features doc --workspace --no-deps --quiet

say "Build"
cargo_with_features build --workspace --all-targets ${RELEASE_FLAG:-}

say "Tests"
cargo_with_features test --workspace --all-targets --quiet ${RELEASE_FLAG:-}
python3 -m unittest discover -s scripts/tests -v

say "Solve, verify, and compare smoke workflow"
cargo_with_features run --quiet --bin hcp-align ${RELEASE_FLAG:-} -- \
  solve edit-distance --query kitten --target sitting --cross-check --integrity \
  --operations --format json --output "${TMP_DIR}/alignment.json"

[[ "$(json_value schema_version < "${TMP_DIR}/alignment.json")" == "hcp-align.v2" ]] || fail "wrong schema version"
[[ "$(json_value alignment.objective.value < "${TMP_DIR}/alignment.json")" == "3" ]] || fail "wrong edit distance"
[[ "$(json_value verification.status < "${TMP_DIR}/alignment.json")" == "verified" ]] || fail "solve cross-check failed"

cargo_with_features run --quiet --bin hcp-align ${RELEASE_FLAG:-} -- \
  verify --alignment "${TMP_DIR}/alignment.json" --format json \
  --query kitten --target sitting --output-format json --output "${TMP_DIR}/verification.json"
[[ "$(json_value verification.status < "${TMP_DIR}/verification.json")" == "verified" ]] || fail "JSON verification failed"

cargo_with_features run --quiet --bin hcp-align ${RELEASE_FLAG:-} -- \
  compare --left "${TMP_DIR}/alignment.json" --left-format json \
  --right "${TMP_DIR}/alignment.json" --right-format json \
  --query kitten --target sitting --output-format json --output "${TMP_DIR}/comparison.json"
[[ "$(json_value equivalent < "${TMP_DIR}/comparison.json")" == "true" ]] || fail "comparison failed"

say "Recoverable JSONL error contract"
{
  cat "${TMP_DIR}/alignment.json"
  printf '%s\n' '{"schema_version":"hcp-align.v2","record_type":"broken"}'
  cat "${TMP_DIR}/alignment.json"
} > "${TMP_DIR}/records.jsonl"
set +e
cargo_with_features run --quiet --bin hcp-align ${RELEASE_FLAG:-} -- \
  verify --alignment "${TMP_DIR}/records.jsonl" --format jsonl \
  --query kitten --target sitting --output-format jsonl \
  --output "${TMP_DIR}/recoverable.jsonl"
status=$?
set -e
[[ "${status}" == "2" ]] || fail "recoverable record errors must return exit code 2"
python3 - "${TMP_DIR}/recoverable.jsonl" "${TMP_DIR}/error.json" <<'PY'
import json, sys
records = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
errors = [record for record in records if record.get("record_type") == "error"]
assert len(records) == 3 and len(errors) == 1
json.dump(errors[0], open(sys.argv[2], "w", encoding="utf-8"))
PY

say "PAF and SAM round trips"
for format in paf sam; do
  cargo_with_features run --quiet --bin hcp-align ${RELEASE_FLAG:-} -- \
    solve edit-distance --query kitten --target sitting --format "${format}" \
    --output "${TMP_DIR}/alignment.${format}"
  cargo_with_features run --quiet --bin hcp-align ${RELEASE_FLAG:-} -- \
    verify --alignment "${TMP_DIR}/alignment.${format}" --format "${format}" \
    --query kitten --target sitting --mode edit-distance --output-format json \
    --output "${TMP_DIR}/${format}-verification.json"
  [[ "$(json_value verification.status < "${TMP_DIR}/${format}-verification.json")" == "verified" ]] || fail "${format} verification failed"
done

say "JSON v2 schema"
python3 scripts/validate_schema.py schemas/hcp-align.v2.schema.json \
  "${TMP_DIR}/alignment.json" "${TMP_DIR}/verification.json" \
  "${TMP_DIR}/comparison.json" "${TMP_DIR}/error.json"

if [[ -n "${RUN_BENCH}" ]]; then
  say "Criterion scenarios and checked baselines"
  CRITERION_DEV_MODE="${CRITERION_DEV_MODE:-1}" cargo_with_features bench --bench oracle_perf
  python3 scripts/check_bench.py
fi

say "All checks passed"
