#!/usr/bin/env bash
set -euo pipefail

readonly expected_commit="bcf473a6561fa297934a80eaa7c04b4ee412360c"
wfa_dir="${1:?usage: build-wfa2-adapter.sh WFA2_DIR [OUTPUT]}"
output="${2:-target/wfa2-adapter}"

actual_commit="$(git -C "$wfa_dir" rev-parse HEAD)"
if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "WFA2 checkout is $actual_commit; expected $expected_commit" >&2
  exit 2
fi

make -C "$wfa_dir" BUILD_TOOLS=0 BUILD_EXAMPLES=0 -j"${JOBS:-2}"
mkdir -p "$(dirname "$output")"
"${CXX:-c++}" -std=c++17 -O2 -I"$wfa_dir" \
  tools/differential/wfa2_adapter.cpp "$wfa_dir/lib/libwfa.a" -lm -o "$output"
"$output" </dev/null
