#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

BASELINE_PATH = Path("perf/baseline.json")
CRITERION_DIR = Path("target/criterion")

def load_baseline():
    if not BASELINE_PATH.exists():
        print(f"[perf] baseline file {BASELINE_PATH} not found; skipping comparison", file=sys.stderr)
        return {}
    with BASELINE_PATH.open() as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            print(f"[perf] failed to parse baseline: {exc}", file=sys.stderr)
            return {}
    if not isinstance(data, dict):
        print("[perf] baseline not a dict; skipping", file=sys.stderr)
        return {}
    return data

def load_estimate(bench_key):
    parts = bench_key.split("/")
    estimates = CRITERION_DIR.joinpath(*parts, "Base", "estimates.json")
    if not estimates.exists():
        print(f"[perf] estimates missing for {bench_key} at {estimates}", file=sys.stderr)
        return None
    with estimates.open() as fh:
        return json.load(fh)

def compare():
    baseline = load_baseline()
    if not baseline:
        return 0

    tolerance = float(os.environ.get("PERF_TOLERANCE", "0.10"))
    enforce = os.environ.get("PERF_ENFORCE", "0") == "1"
    failures = []

    for bench_key, metrics in baseline.items():
        estimate = load_estimate(bench_key)
        if estimate is None:
            continue
        new_median = estimate.get("median", {}).get("point_estimate")
        if new_median is None:
            print(f"[perf] median not found for {bench_key}", file=sys.stderr)
            continue
        base_median = metrics.get("median")
        if not base_median:
            print(f"[perf] baseline median missing for {bench_key}", file=sys.stderr)
            continue
        diff = abs(new_median - base_median) / base_median
        if diff > tolerance:
            msg = f"{bench_key}: median {new_median:.3e} vs baseline {base_median:.3e} (diff {diff:.2%})"
            failures.append(msg)
        else:
            print(f"[perf] OK {bench_key}: {diff:.2%} within tolerance", file=sys.stderr)

    if failures:
        print("[perf] regressions detected:", file=sys.stderr)
        for msg in failures:
            print(f"  {msg}", file=sys.stderr)
        if enforce:
            return 1
        print("[perf] PERF_ENFORCE not set; reporting success with warnings", file=sys.stderr)
    return 0

if __name__ == "__main__":
    sys.exit(compare())


