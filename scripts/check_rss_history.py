#!/usr/bin/env python3
"""Require four scheduled RSS and performance successes for RC/stable tags."""

from __future__ import annotations

import argparse
import json
import os
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"))
    args = parser.parse_args()
    if args.tag not in {"v0.2.0-rc.1", "v0.2.0"}:
        print(f"scheduled performance history gate does not apply to {args.tag}")
        return 0
    if not args.repository or not args.token:
        raise SystemExit("GITHUB_REPOSITORY and GITHUB_TOKEN are required for the RSS history gate")

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {args.token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    for workflow in ("rss.yml", "nightly-perf.yml"):
        url = (
            f"https://api.github.com/repos/{args.repository}/actions/workflows/{workflow}/runs"
            "?event=schedule&status=completed&per_page=4"
        )
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            runs = json.load(response).get("workflow_runs", [])
        if len(runs) < 4:
            raise SystemExit(
                f"{args.tag} requires four scheduled {workflow} runs; found {len(runs)}"
            )
        failures = [run for run in runs if run.get("conclusion") != "success"]
        if failures:
            identifiers = ", ".join(str(run.get("id")) for run in failures)
            raise SystemExit(
                f"the latest four scheduled {workflow} runs are not all successful: {identifiers}"
            )
        print(f"latest four completed scheduled {workflow} runs all succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
