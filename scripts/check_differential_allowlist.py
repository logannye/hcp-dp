#!/usr/bin/env python3
"""Validate the checked differential discrepancy allowlist."""

from __future__ import annotations

import json
import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allowlist", type=Path, default=Path("tools/differential/allowlist.json"))
    parser.add_argument("--report", type=Path, action="append", default=[])
    args = parser.parse_args()
    path = args.allowlist
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != "hcp-align.differential-allowlist.v1":
        raise SystemExit("invalid differential allowlist schema_version")
    required = {"backend", "version", "signature", "reason", "issue_url"}
    for index, entry in enumerate(document.get("entries", [])):
        missing = required - set(entry)
        if missing:
            raise SystemExit(f"allowlist entry {index} is missing {sorted(missing)}")
        if not str(entry["issue_url"]).startswith("https://github.com/"):
            raise SystemExit(f"allowlist entry {index} requires a GitHub issue URL")
    entries = document.get("entries", [])
    for report_path in args.report:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        reports = payload if isinstance(payload, list) else [payload]
        for report_index, report in enumerate(reports):
            signature = report.get("signature")
            if not signature or not signature.get("categories"):
                continue
            if report.get("original_comparison", {}).get("equivalent") is True:
                continue
            backends = report.get("backends", [])
            triaged = any(
                entry["signature"] == signature
                and any(
                    entry["backend"] in {backend.get("name"), backend.get("specification")}
                    and entry["version"] == backend.get("version")
                    for backend in backends
                )
                for entry in entries
            )
            if not triaged:
                raise SystemExit(
                    f"untriaged differential discrepancy in {report_path} at index {report_index}"
                )
    print(
        f"validated {len(entries)} differential allowlist entries "
        f"against {len(args.report)} reports"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
