#!/usr/bin/env python3
"""Summarize the registered oracle benchmarks into portable JSON and Markdown."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CRITERION = ROOT / "target" / "criterion"
OUTPUT = ROOT / "target" / "hcp-dp-report"


def command(*args: str) -> str:
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def collect() -> list[dict[str, object]]:
    measurements: list[dict[str, object]] = []
    for metadata_path in sorted(CRITERION.glob("*/new/benchmark.json")):
        estimate_path = metadata_path.with_name("estimates.json")
        if not estimate_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        estimates = json.loads(estimate_path.read_text(encoding="utf-8"))
        measurements.append(
            {
                "benchmark": metadata["full_id"],
                "median_ns": estimates["median"]["point_estimate"],
                "median_confidence_interval_ns": [
                    estimates["median"]["confidence_interval"]["lower_bound"],
                    estimates["median"]["confidence_interval"]["upper_bound"],
                ],
            }
        )
    return measurements


def main() -> int:
    measurements = collect()
    if not measurements:
        raise SystemExit("no Criterion estimates found; run cargo bench --bench oracle_perf first")
    report = {
        "schema_version": "hcp-align.performance.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": command("git", "rev-parse", "HEAD"),
        "rustc": command("rustc", "--version"),
        "platform": platform.platform(),
        "benchmarks": measurements,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# hcp-align performance report",
        "",
        f"- Commit: `{report['commit']}`",
        f"- Rust: `{report['rustc']}`",
        f"- Platform: `{report['platform']}`",
        "",
        "| Scenario | Median | 95% confidence interval |",
        "|---|---:|---:|",
    ]
    for measurement in measurements:
        low, high = measurement["median_confidence_interval_ns"]
        lines.append(
            f"| `{measurement['benchmark']}` | {measurement['median_ns'] / 1_000_000:.3f} ms | "
            f"{low / 1_000_000:.3f}–{high / 1_000_000:.3f} ms |"
        )
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
