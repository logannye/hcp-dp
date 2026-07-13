#!/usr/bin/env python3
"""Require a seven-day, blocker-free RC observation before stable v0.2.0."""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any


RC_TAG = "v0.2.0-rc.1"
MINIMUM_AGE = timedelta(days=7)


def request_json(url: str, token: str) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def parse_github_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def validate(repository: str, token: str, now: datetime) -> None:
    release = request_json(
        f"https://api.github.com/repos/{repository}/releases/tags/{RC_TAG}", token
    )
    published_at = release.get("published_at")
    if not isinstance(published_at, str):
        raise SystemExit(f"{RC_TAG} has no published_at timestamp")
    age = now - parse_github_time(published_at)
    if age < MINIMUM_AGE:
        remaining = MINIMUM_AGE - age
        raise SystemExit(
            f"{RC_TAG} must be observed for seven days; {remaining} remains"
        )

    query = urllib.parse.urlencode(
        {"state": "open", "labels": "release-blocker", "per_page": 1}
    )
    blockers = request_json(
        f"https://api.github.com/repos/{repository}/issues?{query}", token
    )
    if blockers:
        number = blockers[0].get("number", "unknown")
        raise SystemExit(f"open release-blocking issue remains: #{number}")
    print(f"{RC_TAG} has seven days of observation and no open release blockers")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"))
    args = parser.parse_args()
    if args.tag != "v0.2.0":
        print(f"RC observation gate does not apply to {args.tag}")
        return 0
    if not args.repository or not args.token:
        raise SystemExit(
            "GITHUB_REPOSITORY and GITHUB_TOKEN are required for the RC observation gate"
        )
    validate(args.repository, args.token, datetime.now(timezone.utc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
