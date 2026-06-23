#!/usr/bin/env python3
"""Generate a local HCP-DP correctness/performance report."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "target" / "hcp-dp-report"
REPORT_PATH = REPORT_DIR / "report.md"
SCALE_PROBE_PATH = REPORT_DIR / "scale_probe.json"
EXTERNAL_PATH = REPORT_DIR / "external-validation.json"


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    print(f"[report] {' '.join(cmd)}", file=sys.stderr)
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)


def run_text(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return "unavailable"
    return proc.stdout.strip()


def cpu_model() -> str:
    if sys.platform == "darwin":
        value = run_text(["sysctl", "-n", "machdep.cpu.brand_string"])
        if value != "unavailable":
            return value
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unavailable"


def metadata(argv: list[str]) -> dict[str, str]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": run_text(["git", "rev-parse", "HEAD"]),
        "rustc": run_text(["rustc", "--version"]),
        "cargo": run_text(["cargo", "--version"]),
        "os": platform.platform(),
        "cpu": cpu_model(),
        "command": " ".join(argv),
    }


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
    measurements = json.loads(proc.stdout)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SCALE_PROBE_PATH.write_text(json.dumps(measurements, indent=2) + "\n", encoding="utf-8")
    return measurements


def run_external_validation(required: bool, skip: bool) -> tuple[list[dict], str]:
    if skip:
        return [], "External validation was skipped."
    cmd = ["python3", "scripts/validate_external.py"]
    if required:
        cmd.append("--required")
    proc = run(cmd)
    if EXTERNAL_PATH.exists():
        data = json.loads(EXTERNAL_PATH.read_text(encoding="utf-8"))
    else:
        data = []
    if proc.returncode != 0:
        raise RuntimeError(
            f"external validation failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return data, "External validation completed."


def maybe_run_benches(enabled: bool) -> str:
    if not enabled:
        return "Criterion benches were not run. Pass `--benches` to include them."
    proc = run(["cargo", "bench"])
    if proc.returncode != 0:
        return f"Criterion benches failed.\n\n```text\n{proc.stderr[-4000:]}\n```"
    return "Criterion benches completed successfully. Raw Criterion output is under `target/criterion/`."


def status_counts(items: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        status = str(item.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def write_report(
    meta: dict[str, str],
    measurements: list[dict],
    external: list[dict],
    external_note: str,
    bench_note: str,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scale_counts = status_counts(measurements)
    external_counts = status_counts(external)

    lines = [
        "# HCP-DP Correctness And Performance Report",
        "",
        "## Environment",
        "",
        f"- Timestamp: `{meta['timestamp_utc']}`",
        f"- Commit: `{meta['commit']}`",
        f"- Rust: `{meta['rustc']}`",
        f"- Cargo: `{meta['cargo']}`",
        f"- OS: `{meta['os']}`",
        f"- CPU: `{meta['cpu']}`",
        f"- Command: `{meta['command']}`",
        "",
        "## Scale Probe",
        "",
        f"Raw JSON: `{SCALE_PROBE_PATH.relative_to(ROOT)}`",
        "",
        f"Status counts: `{json.dumps(scale_counts, sort_keys=True)}`",
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

    lines.extend(
        [
            "",
            "## External Validation",
            "",
            external_note,
            "",
            f"Raw JSON: `{EXTERNAL_PATH.relative_to(ROOT)}`" if EXTERNAL_PATH.exists() else "Raw JSON: not generated",
            "",
            f"Status counts: `{json.dumps(external_counts, sort_keys=True)}`",
            "",
            "| Tool | Mode | Case | Status | HCP | External | Detail |",
            "|---|---|---|---|---:|---:|---|",
        ]
    )
    for item in external:
        lines.append(
            "| {tool} | {mode} | {case} | {status} | {hcp} | {external} | {detail} |".format(
                tool=item.get("tool", ""),
                mode=item.get("mode", ""),
                case=item.get("case", ""),
                status=item.get("status", ""),
                hcp="" if item.get("hcp") is None else item.get("hcp"),
                external="" if item.get("external") is None else item.get("external"),
                detail=str(item.get("detail", "")).replace("|", "\\|"),
            )
        )

    lines.extend(["", "## Criterion", "", bench_note, ""])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-limit", type=int, default=512)
    parser.add_argument("--scenario")
    parser.add_argument("--benches", action="store_true")
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--external-required", action="store_true")
    args = parser.parse_args(argv[1:])

    try:
        meta = metadata(argv)
        measurements = run_scale_probe(args.verify_limit, args.scenario)
        external, external_note = run_external_validation(args.external_required, args.skip_external)
        bench_note = maybe_run_benches(args.benches)
        write_report(meta, measurements, external, external_note, bench_note)
    except RuntimeError as exc:
        print(f"[report] {exc}", file=sys.stderr)
        return 1

    print(REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
