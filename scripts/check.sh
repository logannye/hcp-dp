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

assert_eq() { local got="$1" exp="$2" name="$3"; if [[ "$got" != "$exp" ]]; then echo "FAIL: $name (got=$got expected=$exp)"; exit 1; fi; }
assert_nonempty() { local v="$1" name="$2"; if [[ -z "$v" ]]; then echo "FAIL: $name empty"; exit 1; fi; }

require cargo
require rustc
require python3

say "Toolchain versions"
cargo --version
rustc --version

say "Formatting check"
cargo fmt --all -- --check

say "Clippy (deny warnings)"
cargo_with_features clippy --workspace --all-targets ${VERBOSE:+-v} -- -D warnings

say "Build (${RELEASE:+release} build)"
cargo_with_features build --workspace --all-targets ${RELEASE_FLAG:-}

say "Unit tests"
cargo_with_features test --workspace --all-targets --quiet ${RELEASE_FLAG:-}
# Guard: fail if no tests detected unless explicitly allowed.
if [[ -z "${ALLOW_ZERO_TESTS:-}" ]]; then
  TOTAL_TESTS=$(cargo_with_features test --workspace --all-targets -- --list 2>/dev/null | grep -E '^[^ ]' | wc -l | tr -d ' ')
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

say "Run example: matrix_chain"
MC_OUT=$(cargo_with_features run --quiet --example matrix_chain ${RELEASE_FLAG:-})
MC_COST=$(printf '%s' "$MC_OUT" | grep -E "^Optimal multiplication cost:" | parse_num)
assert_nonempty "$MC_COST" "Matrix-chain cost"
# CLRS example expected cost
assert_eq "$MC_COST" "15125" "Matrix-chain optimal cost"

say "Run example: align (Needleman–Wunsch)"
NW_OUT=$(cargo_with_features run --quiet --example align ${RELEASE_FLAG:-})
NW_SCORE=$(printf '%s' "$NW_OUT" | grep -E "^Global alignment score:" | parse_num)
assert_nonempty "$NW_SCORE" "NW score"
# For GATTACA vs GCATGCU with (+1,-1,-1), an optimal score is 0
assert_eq "$NW_SCORE" "0" "NW global alignment score"

say "Run example: viterbi"
VIT_OUT=$(cargo_with_features run --quiet --example viterbi ${RELEASE_FLAG:-})
VIT_LOGP=$(printf '%s' "$VIT_OUT" | grep -E "^Best path log-probability:" | parse_float)
assert_nonempty "$VIT_LOGP" "Viterbi log-prob"
# Count number of printed states (one per line)
VIT_LEN=$(printf '%s' "$VIT_OUT" | grep -c "state = ")
# Example obs length = 7
assert_eq "$VIT_LEN" "7" "Viterbi path length"

if [[ -n "${RUN_BENCH}" ]]; then
  say "Benches (dev mode, faster)"
  export CRITERION_DEV_MODE=1
  cargo_with_features bench ${RELEASE_FLAG:-} || { echo "Bench failed"; exit 1; }
  python3 scripts/check_bench.py || exit 1
else
  say "Skipping benches (set RUN_BENCH=1 to enable)"
fi

say "All checks passed"


