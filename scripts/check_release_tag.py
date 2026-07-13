#!/usr/bin/env python3
"""Require a verified annotated GitHub tag targeting the release commit."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


def validate_tag(
    ref: dict[str, Any],
    tag_object: dict[str, Any] | None,
    *,
    tag: str,
    expected_commit: str,
) -> list[str]:
    failures: list[str] = []
    target = ref.get("object") if isinstance(ref, dict) else None
    if not isinstance(target, dict) or target.get("type") != "tag":
        return [f"{tag} must be an annotated tag, not a lightweight tag"]
    if not isinstance(tag_object, dict):
        return [f"annotated tag object for {tag} is unavailable"]
    if tag_object.get("tag") != tag:
        failures.append(
            f"annotated tag name {tag_object.get('tag')!r} does not match {tag!r}"
        )
    verification = tag_object.get("verification")
    if not isinstance(verification, dict) or verification.get("verified") is not True:
        reason = verification.get("reason") if isinstance(verification, dict) else "missing"
        failures.append(f"{tag} does not have a verified signature: {reason}")
    release_target = tag_object.get("object")
    if not isinstance(release_target, dict) or release_target.get("type") != "commit":
        failures.append(f"{tag} must target a commit directly")
    elif release_target.get("sha") != expected_commit:
        failures.append(
            f"{tag} targets {release_target.get('sha')}, expected {expected_commit}"
        )
    return failures


def github_json(url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "hcp-dp-release-tag-check",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        raise SystemExit(f"GitHub tag lookup failed with HTTP {error.code}") from error
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise SystemExit(f"GitHub tag lookup failed: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit("GitHub tag lookup returned a non-object response")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--tag", default=os.environ.get("GITHUB_REF_NAME"))
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"))
    args = parser.parse_args()
    if not args.repository or not args.tag or not args.token:
        raise SystemExit("repository, tag, and GitHub token are required")

    encoded_tag = urllib.parse.quote(args.tag, safe="")
    root = f"https://api.github.com/repos/{args.repository}"
    ref = github_json(f"{root}/git/ref/tags/{encoded_tag}", args.token)
    target = ref.get("object")
    tag_object = None
    if isinstance(target, dict) and target.get("type") == "tag" and target.get("sha"):
        tag_object = github_json(f"{root}/git/tags/{target['sha']}", args.token)
    failures = validate_tag(
        ref,
        tag_object,
        tag=args.tag,
        expected_commit=args.expected_commit,
    )
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"verified signed annotated tag {args.tag} targets {args.expected_commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
