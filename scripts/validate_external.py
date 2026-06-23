#!/usr/bin/env python3
"""Optional score validation against external alignment libraries."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_hcp(args: list[str]) -> dict:
    cmd = ["cargo", "run", "--quiet", "--bin", "hcp-align", "--", *args, "--verify", "--format", "json"]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"hcp-align failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return json.loads(proc.stdout)


def load_optional(name: str, required: bool):
    try:
        return importlib.import_module(name)
    except ImportError:
        if required:
            raise RuntimeError(f"required Python package '{name}' is not installed")
        print(f"[external] skipping {name}: package not installed", file=sys.stderr)
        return None


def parasail_matrix(parasail, alphabet: str, match: int, mismatch_penalty: int):
    return parasail.matrix_create(alphabet, match, -mismatch_penalty)


def validate_parasail(required: bool) -> list[str]:
    parasail = load_optional("parasail", required)
    if parasail is None:
        return []

    failures: list[str] = []

    linear = run_hcp([
        "global-linear",
        "--query",
        "GATTACA",
        "--target",
        "GCATGCU",
        "--match",
        "1",
        "--mismatch-penalty",
        "1",
        "--gap",
        "-1",
    ])
    matrix = parasail_matrix(parasail, "ACGTU", 1, 1)
    parasail_score = parasail.nw_scan_16("GATTACA", "GCATGCU", 0, 1, matrix).score
    compare("parasail global-linear", linear["score"], parasail_score, failures)

    affine = run_hcp([
        "global-affine",
        "--query",
        "ACB",
        "--target",
        "A",
        "--match",
        "2",
        "--mismatch-penalty",
        "1",
        "--gap-open",
        "-3",
        "--gap-extend",
        "-1",
    ])
    matrix = parasail_matrix(parasail, "ABC", 2, 1)
    parasail_score = parasail.nw_scan_16("ACB", "A", 3, 1, matrix).score
    compare("parasail global-affine", affine["score"], parasail_score, failures)

    local = run_hcp([
        "local-linear",
        "--query",
        "ACACACTA",
        "--target",
        "AGCACACA",
        "--match",
        "2",
        "--mismatch-penalty",
        "1",
        "--gap",
        "-2",
    ])
    matrix = parasail_matrix(parasail, "ACGT", 2, 1)
    parasail_score = parasail.sw_scan_16("ACACACTA", "AGCACACA", 0, 2, matrix).score
    compare("parasail local-linear", local["score"], parasail_score, failures)

    return failures


def validate_edlib(required: bool) -> list[str]:
    edlib = load_optional("edlib", required)
    if edlib is None:
        return []

    failures: list[str] = []
    hcp = run_hcp([
        "edit-distance",
        "--query",
        "kitten",
        "--target",
        "sitting",
    ])
    edlib_distance = edlib.align("kitten", "sitting")["editDistance"]
    compare("edlib edit-distance", hcp["distance"], edlib_distance, failures)
    return failures


def compare(label: str, hcp_value: int, external_value: int, failures: list[str]) -> None:
    if hcp_value == external_value:
        print(f"[external] OK {label}: {hcp_value}")
    else:
        failures.append(f"{label}: hcp={hcp_value}, external={external_value}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--required",
        action="store_true",
        help="fail if parasail or edlib is unavailable",
    )
    args = parser.parse_args()

    failures: list[str] = []
    try:
        failures.extend(validate_parasail(args.required))
        failures.extend(validate_edlib(args.required))
    except RuntimeError as exc:
        print(f"[external] {exc}", file=sys.stderr)
        return 2

    if failures:
        print("[external] validation failures:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print("[external] validation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
