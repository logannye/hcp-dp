#!/usr/bin/env python3
"""Generate a local HCP-DP correctness/performance report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "target" / "hcp-dp-report"
REPORT_PATH = REPORT_DIR / "report.md"


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    print(f"[report] {' '.join(cmd)}", file=sys.stderr)
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)


def run_scale_probe(verify_limit: int, scenario: str | None) -> list[dict]:
    cmd = [
        "cargo",
        "run",
        "--quiet",
        "--bin",
        "scale_probe",
        "--",
        "--format",
        "json",
        "--verify-limit",
        str(verify_limit),
    ]
    if scenario:
        cmd.extend(["--scenario", scenario])
    proc = run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"scale_probe failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return json.loads(proc.stdout)


def maybe_run_benches(enabled: bool) -> str:
    if not enabled:
        return "Criterion benches were not run. Pass `--benches` to include them."
    proc = run(["cargo", "bench"])
    if proc.returncode != 0:
        return f"Criterion benches failed.\n\n```text\n{proc.stderr[-4000:]}\n```"
    return "Criterion benches completed successfully. Raw Criterion output is under `target/criterion/`."


def write_report(measurements: list[dict], bench_note: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# HCP-DP Correctness And Performance Report",
        "",
        f"Generated: {generated}",
        "",
        "## Scale Probe",
        "",
        "| Scenario | Size | Wall s | RSS delta bytes | Peak RSS bytes | Status | Detail |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for item in measurements:
        lines.append(
            "| {scenario} | {size} | {wall_s:.6f} | {rss_delta_bytes} | {peak_rss_bytes} | {status} | {detail} |".format(
                scenario=item["scenario"],
                size=item["size"],
                wall_s=item["wall_s"],
                rss_delta_bytes=item["rss_delta_bytes"],
                peak_rss_bytes=item["peak_rss_bytes"],
                status=item["status"],
                detail=str(item["detail"]).replace("|", "\\|"),
            )
        )
    lines.extend(["", "## Criterion", "", bench_note, ""])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-limit", type=int, default=512)
    parser.add_argument("--scenario")
    parser.add_argument("--benches", action="store_true")
    args = parser.parse_args()

    try:
        measurements = run_scale_probe(args.verify_limit, args.scenario)
        bench_note = maybe_run_benches(args.benches)
        write_report(measurements, bench_note)
    except RuntimeError as exc:
        print(f"[report] {exc}", file=sys.stderr)
        return 1

    print(REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
