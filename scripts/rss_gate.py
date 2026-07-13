#!/usr/bin/env python3
"""Generate deterministic batches, measure peak RSS, and enforce bounded growth."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path


RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")


def sequence(index: int, length: int, salt: int) -> str:
    state = ((index + 1) * 0x9E3779B1) ^ salt
    alphabet = "ACGT"
    bases: list[str] = []
    for _ in range(length):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        bases.append(alphabet[state & 3])
    return "".join(bases)


def write_fasta(path: Path, count: int, length: int, prefix: str, salt: int, reverse: bool = False) -> None:
    indices = range(count - 1, -1, -1) if reverse else range(count)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            for index in indices:
                stream.write(f">{prefix}{index:06d}\n{sequence(index, length, salt)}\n".encode())


def run_timed(command: list[str], report: Path) -> int:
    completed = subprocess.run(
        ["/usr/bin/time", "-v", "-o", str(report), *command],
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"command exited {completed.returncode}: {' '.join(command)}")
    match = RSS_RE.search(report.read_text(encoding="utf-8"))
    if not match:
        raise RuntimeError(f"could not find peak RSS in {report}")
    return int(match.group(1))


def solve_measure(binary: Path, directory: Path, count: int, length: int) -> int:
    query = directory / f"query-{count}.fa.gz"
    target = directory / f"target-{count}.fa.gz"
    write_fasta(query, count, length, "q", 0xA5A5A5A5)
    write_fasta(target, count, length, "t", 0x5A5A5A5A)
    return run_timed(
        [
            str(binary),
            "solve",
            "edit-distance",
            "--query-file",
            str(query),
            "--target-file",
            str(target),
            "--scope",
            "score",
            "--format",
            "jsonl",
            "--threads",
            "4",
            "--max-in-flight",
            "8",
            "--output",
            os.devnull,
        ],
        directory / f"solve-{count}.time",
    )


def ordered_solve_count(binary: Path, directory: Path, count: int) -> tuple[bool, int]:
    process = subprocess.Popen(
        [
            str(binary),
            "solve",
            "edit-distance",
            "--query-file",
            str(directory / f"query-{count}.fa.gz"),
            "--target-file",
            str(directory / f"target-{count}.fa.gz"),
            "--scope",
            "score",
            "--format",
            "jsonl",
            "--threads",
            "4",
            "--max-in-flight",
            "8",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None and process.stderr is not None
    seen = 0
    ordered = True
    for line in process.stdout:
        record = json.loads(line)
        ordered &= record.get("pair", {}).get("index") == seen
        seen += 1
    error = process.stderr.read()
    status = process.wait()
    if status != 0:
        raise RuntimeError(f"ordered-count solve exited {status}: {error}")
    return ordered and seen == count, seen


def verify_measure(binary: Path, directory: Path, count: int, length: int) -> int:
    query = directory / "verify-query.fa.gz"
    target = directory / "verify-target.fa.gz"
    records = directory / "verify.paf"
    # Identical sequences make a valid exact edit-distance record. Source order
    # and alignment order intentionally differ to exercise arbitrary-ID lookup.
    write_fasta(query, count, length, "q", 0xC3C3C3C3)
    write_fasta(target, count, length, "t", 0xC3C3C3C3, reverse=True)
    step = 7_919
    if math.gcd(step, count) != 1:
        step = 7_921
    with records.open("w", encoding="ascii") as stream:
        for ordinal in range(count):
            index = (ordinal * step) % count
            stream.write(
                f"q{index:06d}\t{length}\t0\t{length}\t+\t"
                f"t{index:06d}\t{length}\t0\t{length}\t{length}\t{length}\t255\t"
                f"cg:Z:{length}=\tNM:i:0\n"
            )
    return run_timed(
        [
            str(binary),
            "verify",
            "--alignment",
            str(records),
            "--format",
            "paf",
            "--query-file",
            str(query),
            "--target-file",
            str(target),
            "--mode",
            "edit-distance",
            "--level",
            "score",
            "--output-format",
            "jsonl",
            "--threads",
            "4",
            "--max-in-flight",
            "8",
            "--output",
            os.devnull,
        ],
        directory / "verify.time",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("target/release/hcp-align"))
    parser.add_argument("--baseline", type=Path, default=Path("perf/rss-baseline.json"))
    parser.add_argument("--output", type=Path, default=Path("rss-report.json"))
    parser.add_argument("--large-count", type=int, default=100_000)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--no-enforce", action="store_true")
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="hcp-align-rss-") as temp:
        directory = Path(temp)
        small = solve_measure(args.binary, directory, 1_000, args.length)
        large = solve_measure(args.binary, directory, args.large_count, args.length)
        output_count_ok, output_count = ordered_solve_count(
            args.binary, directory, args.large_count
        )
        verify = verify_measure(args.binary, directory, args.large_count, args.length)

    absolute_limit = 128 * 1024
    delta_limit = 16 * 1024
    regression_limit = int(baseline["solve_100000_peak_rss_kib"] * 1.15)
    checks = {
        "bounded_growth": large <= small + delta_limit,
        "absolute_solve_budget": large <= absolute_limit,
        "baseline_regression": large <= regression_limit,
        "absolute_verify_budget": verify <= absolute_limit,
        "ordered_output_count": output_count_ok,
    }
    report = {
        "schema_version": "hcp-align.rss.v1",
        "scenario": {"pairs": args.large_count, "length": args.length, "threads": 4, "max_in_flight": 8},
        "peak_rss_kib": {"solve_1000": small, "solve_100000": large, "verify_100000": verify},
        "observed_output_records": output_count,
        "limits_kib": {
            "growth": delta_limit,
            "absolute": absolute_limit,
            "baseline_plus_15_percent": regression_limit,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if args.no_enforce or report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
