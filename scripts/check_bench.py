#!/usr/bin/env python3
"""Enforce checked, scenario-specific Criterion runtime baselines."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any


BASELINE_PATH = Path("perf/baseline.json")
CRITERION_DIR = Path("target/criterion")


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error


def criterion_directory(benchmark: str) -> Path:
    expected = re.sub(r"[^A-Za-z0-9_\-.]", "_", benchmark)
    direct = CRITERION_DIR / expected
    if direct.exists():
        return direct
    for metadata_path in CRITERION_DIR.glob("*/new/benchmark.json"):
        metadata = load_json(metadata_path)
        if metadata.get("full_id") == benchmark:
            return metadata_path.parents[1]
    return direct


def estimate_for(benchmark: str) -> float:
    directory = criterion_directory(benchmark)
    for generation in ("new", "base"):
        estimate_path = directory / generation / "estimates.json"
        if estimate_path.exists():
            data = load_json(estimate_path)
            median = data.get("median", {}).get("point_estimate")
            if isinstance(median, (int, float)):
                return float(median)
            raise ValueError(f"median estimate missing in {estimate_path}")
    raise ValueError(f"Criterion estimate missing for {benchmark} under {directory}")


def main() -> int:
    enforce = os.environ.get("PERF_ENFORCE", "0") == "1"
    try:
        baseline = load_json(BASELINE_PATH)
    except ValueError as error:
        print(f"[perf] {error}", file=sys.stderr)
        return 1 if enforce else 0
    if not isinstance(baseline, dict) or not baseline:
        print("[perf] baseline must be a non-empty object", file=sys.stderr)
        return 1 if enforce else 0

    relative_limit = float(os.environ.get("PERF_TOLERANCE", "0.25"))
    absolute_grace_ns = float(os.environ.get("PERF_ABSOLUTE_GRACE_NS", "5000000"))
    failures: list[str] = []

    for benchmark, budget in baseline.items():
        try:
            measured = estimate_for(benchmark)
        except ValueError as error:
            failures.append(str(error))
            continue
        expected = budget.get("median_ns")
        if not isinstance(expected, (int, float)) or expected <= 0:
            failures.append(f"{benchmark}: positive median_ns baseline is required")
            continue
        delta = measured - float(expected)
        slowdown = delta / float(expected)
        regression = slowdown > relative_limit and delta > absolute_grace_ns
        state = "REGRESSION" if regression else "OK"
        print(
            f"[perf] {state} {benchmark}: {measured:.0f} ns vs {expected:.0f} ns "
            f"({slowdown:+.1%}, delta {delta / 1_000_000:+.3f} ms)",
            file=sys.stderr,
        )
        if regression:
            failures.append(
                f"{benchmark}: exceeded baseline by both {relative_limit:.0%} "
                f"and {absolute_grace_ns / 1_000_000:.1f} ms"
            )

    if failures:
        print("[perf] benchmark checks failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1 if enforce else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
