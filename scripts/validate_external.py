#!/usr/bin/env python3
"""Optional score validation against external alignment libraries."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "target" / "hcp-dp-report"
REPORT_PATH = REPORT_DIR / "external-validation.json"


@dataclass(frozen=True)
class LinearCase:
    name: str
    query: str
    target: str
    match: int = 2
    mismatch: int = 1
    gap: int = -2


@dataclass(frozen=True)
class AffineCase:
    name: str
    query: str
    target: str
    match: int = 2
    mismatch: int = 1
    gap_open: int = -3
    gap_extend: int = -1


@dataclass(frozen=True)
class EditCase:
    name: str
    query: str
    target: str


LINEAR_CASES = [
    LinearCase("exact_match", "ACGTACGT", "ACGTACGT"),
    LinearCase("all_mismatches", "AAAAAA", "TTTTTT"),
    LinearCase("long_deletion", "ACGTACGTACGT", "ACGT"),
    LinearCase("long_insertion", "ACGT", "ACGTACGTACGT"),
    LinearCase("homopolymer", "AAAAAAA", "AAA"),
    LinearCase("repeats", "ATATATAT", "TATATA"),
    LinearCase("skewed_lengths", "ACGTAC", "ACGTACGTACGT"),
    LinearCase("tie_heavy", "AAAA", "AA", 1, 1, -1),
    LinearCase("classic_global", "GATTACA", "GCATGCU", 1, 1, -1),
]

LOCAL_CASES = [
    LinearCase("classic_local", "ACACACTA", "AGCACACA"),
    LinearCase("starts_after_split", "TTTTACGT", "ACGT"),
    LinearCase("ends_before_split", "ACGTAAAA", "ACGT"),
    LinearCase("repeats", "ATATATAT", "TATATA"),
    LinearCase("tie_heavy", "AAAA", "AA", 1, 1, -1),
]

AFFINE_CASES = [
    AffineCase("regression_two_gap", "ACB", "A"),
    AffineCase("one_position_gap", "AC", "A"),
    AffineCase("two_position_gap", "ACB", "A"),
    AffineCase("long_deletion", "ACGTACGTACGT", "ACGT"),
    AffineCase("long_insertion", "ACGT", "ACGTACGTACGT"),
    AffineCase("homopolymer", "AAAAAAA", "AAA"),
    AffineCase("repeats", "ATATATAT", "TATATA"),
    AffineCase("tie_heavy", "AAAA", "AA", 1, 1, -3, -1),
]

EDIT_CASES = [
    EditCase("kitten_sitting", "kitten", "sitting"),
    EditCase("exact_match", "ACGTACGT", "ACGTACGT"),
    EditCase("all_mismatches", "AAAAAA", "TTTTTT"),
    EditCase("long_deletion", "ACGTACGTACGT", "ACGT"),
    EditCase("long_insertion", "ACGT", "ACGTACGTACGT"),
    EditCase("homopolymer", "AAAAAAA", "AAA"),
    EditCase("repeats", "ATATATAT", "TATATA"),
    EditCase("skewed_lengths", "ACGTAC", "ACGTACGTACGT"),
]


def run_hcp(args: list[str]) -> dict[str, Any]:
    cmd = [
        "cargo",
        "run",
        "--quiet",
        "--bin",
        "hcp-align",
        "--",
        *args,
        "--verify",
        "--format",
        "json",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"hcp-align failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    parsed = json.loads(proc.stdout)
    if isinstance(parsed, list):
        raise RuntimeError("validate_external scenarios must produce single-record JSON objects")
    return parsed


def load_optional(name: str, required: bool):
    try:
        return importlib.import_module(name)
    except ImportError:
        if required:
            raise RuntimeError(f"required Python package '{name}' is not installed")
        print(f"[external] skipping {name}: package not installed", file=sys.stderr)
        return None


def result(
    *,
    tool: str,
    mode: str,
    case: str,
    status: str,
    hcp: int | None = None,
    external: int | None = None,
    detail: str = "",
) -> dict[str, Any]:
    return {
        "tool": tool,
        "mode": mode,
        "case": case,
        "status": status,
        "hcp": hcp,
        "external": external,
        "detail": detail,
    }


def compare(tool: str, mode: str, case: str, hcp_value: int, external_value: int) -> dict[str, Any]:
    if hcp_value == external_value:
        print(f"[external] OK {tool} {mode} {case}: {hcp_value}")
        return result(
            tool=tool,
            mode=mode,
            case=case,
            status="passed",
            hcp=hcp_value,
            external=external_value,
        )
    return result(
        tool=tool,
        mode=mode,
        case=case,
        status="failed",
        hcp=hcp_value,
        external=external_value,
        detail="score mismatch",
    )


def alphabet_for(*values: str) -> str:
    alphabet = "".join(sorted(set("".join(values))))
    return alphabet or "ACGT"


def parasail_matrix(parasail, case: LinearCase | AffineCase):
    return parasail.matrix_create(alphabet_for(case.query, case.target), case.match, -case.mismatch)


def parasail_linear_gap(case: LinearCase) -> tuple[int, int]:
    """Translate HCP's linear per-position gap score to Parasail open/extend penalties."""
    gap = abs(case.gap)
    return gap, gap


def parasail_affine_gap(case: AffineCase) -> tuple[int, int]:
    """Translate HCP's first-gap open+extend convention to Parasail's open-only first gap."""
    return abs(case.gap_open + case.gap_extend), abs(case.gap_extend)


def hcp_linear(case: LinearCase, mode: str) -> int:
    payload = run_hcp(
        [
            mode,
            "--query",
            case.query,
            "--target",
            case.target,
            "--match",
            str(case.match),
            "--mismatch-penalty",
            str(case.mismatch),
            "--gap",
            str(case.gap),
        ]
    )
    return int(payload["score"])


def hcp_affine(case: AffineCase) -> int:
    payload = run_hcp(
        [
            "global-affine",
            "--query",
            case.query,
            "--target",
            case.target,
            "--match",
            str(case.match),
            "--mismatch-penalty",
            str(case.mismatch),
            "--gap-open",
            str(case.gap_open),
            "--gap-extend",
            str(case.gap_extend),
        ]
    )
    return int(payload["score"])


def validate_parasail(required: bool) -> list[dict[str, Any]]:
    parasail = load_optional("parasail", required)
    if parasail is None:
        return [
            result(tool="parasail", mode=mode, case="all", status="skipped", detail="package not installed")
            for mode in ("global-linear", "global-affine", "local-linear")
        ]

    results: list[dict[str, Any]] = []
    for case in LINEAR_CASES:
        matrix = parasail_matrix(parasail, case)
        open_penalty, extend_penalty = parasail_linear_gap(case)
        hcp_score = hcp_linear(case, "global-linear")
        external_score = parasail.nw_scan_16(
            case.query, case.target, open_penalty, extend_penalty, matrix
        ).score
        results.append(compare("parasail", "global-linear", case.name, hcp_score, external_score))

    affine_ready, calibration = calibrate_parasail_affine(parasail)
    results.extend(calibration)
    if affine_ready:
        for case in AFFINE_CASES:
            matrix = parasail_matrix(parasail, case)
            open_penalty, extend_penalty = parasail_affine_gap(case)
            hcp_score = hcp_affine(case)
            external_score = parasail.nw_scan_16(
                case.query,
                case.target,
                open_penalty,
                extend_penalty,
                matrix,
            ).score
            results.append(compare("parasail", "global-affine", case.name, hcp_score, external_score))
    else:
        status = "failed" if required else "skipped"
        for case in AFFINE_CASES:
            results.append(
                result(
                    tool="parasail",
                    mode="global-affine",
                    case=case.name,
                    status=status,
                    detail="affine gap convention did not match calibration cases",
                )
            )

    for case in LOCAL_CASES:
        matrix = parasail_matrix(parasail, case)
        open_penalty, extend_penalty = parasail_linear_gap(case)
        hcp_score = hcp_linear(case, "local-linear")
        external_score = parasail.sw_scan_16(
            case.query, case.target, open_penalty, extend_penalty, matrix
        ).score
        results.append(compare("parasail", "local-linear", case.name, hcp_score, external_score))

    return results


def calibrate_parasail_affine(parasail) -> tuple[bool, list[dict[str, Any]]]:
    cases = [
        AffineCase("calibration_one_position_gap", "AC", "A"),
        AffineCase("calibration_two_position_gap", "ACB", "A"),
    ]
    results = []
    ok = True
    for case in cases:
        matrix = parasail_matrix(parasail, case)
        open_penalty, extend_penalty = parasail_affine_gap(case)
        hcp_score = hcp_affine(case)
        external_score = parasail.nw_scan_16(
            case.query,
            case.target,
            open_penalty,
            extend_penalty,
            matrix,
        ).score
        item = compare("parasail", "global-affine-calibration", case.name, hcp_score, external_score)
        ok = ok and item["status"] == "passed"
        results.append(item)
    return ok, results


def validate_edlib(required: bool) -> list[dict[str, Any]]:
    edlib = load_optional("edlib", required)
    if edlib is None:
        return [
            result(tool="edlib", mode="edit-distance", case="all", status="skipped", detail="package not installed")
        ]

    results = []
    for case in EDIT_CASES:
        hcp = run_hcp(["edit-distance", "--query", case.query, "--target", case.target])
        external_distance = edlib.align(case.query, case.target)["editDistance"]
        results.append(
            compare("edlib", "edit-distance", case.name, int(hcp["distance"]), int(external_distance))
        )
    return results


def write_results(results: list[dict[str, Any]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--required",
        action="store_true",
        help="fail if parasail or edlib is unavailable or cannot match required scoring conventions",
    )
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    try:
        results.extend(validate_parasail(args.required))
        results.extend(validate_edlib(args.required))
    except RuntimeError as exc:
        results.append(
            result(
                tool="external",
                mode="setup",
                case="required",
                status="failed",
                detail=str(exc),
            )
        )
        write_results(results)
        print(f"[external] {exc}", file=sys.stderr)
        return 2

    write_results(results)
    failures = [item for item in results if item["status"] == "failed"]
    if failures:
        print("[external] validation failures:", file=sys.stderr)
        for failure in failures:
            print(
                f"  {failure['tool']} {failure['mode']} {failure['case']}: "
                f"hcp={failure['hcp']} external={failure['external']} {failure['detail']}",
                file=sys.stderr,
            )
        return 1

    passed = sum(1 for item in results if item["status"] == "passed")
    skipped = sum(1 for item in results if item["status"] == "skipped")
    print(f"[external] validation complete: passed={passed}, skipped={skipped}")
    print(REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
