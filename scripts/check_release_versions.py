#!/usr/bin/env python3
"""Require tag, Cargo, Python, and binary-contract versions to agree."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path


def python_version(rust_version: str) -> str:
    match = re.fullmatch(r"(\d+\.\d+\.\d+)(?:-(alpha|rc)\.(\d+))?", rust_version)
    if not match:
        raise ValueError(f"unsupported Rust release version {rust_version!r}")
    base, kind, number = match.groups()
    if kind == "alpha":
        return f"{base}a{number}"
    if kind == "rc":
        return f"{base}rc{number}"
    return base


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    cargo = tomllib.loads(Path("Cargo.toml").read_text(encoding="utf-8"))
    python = tomllib.loads(Path("bindings/python/pyproject.toml").read_text(encoding="utf-8"))
    rust_version = cargo["package"]["version"]
    expected_tag = f"v{rust_version}"
    expected_python = python_version(rust_version)
    failures: list[str] = []
    if args.tag != expected_tag:
        failures.append(f"tag {args.tag!r} != {expected_tag!r}")
    if python["project"]["version"] != expected_python:
        failures.append(
            f"Python {python['project']['version']!r} != {expected_python!r}"
        )
    source = Path("bindings/python/src/hcp_align/__init__.py").read_text(encoding="utf-8")
    if f'__version__ = "{expected_python}"' not in source:
        failures.append("Python __version__ does not match pyproject")
    if f'_BINARY_VERSION = "{rust_version}"' not in source:
        failures.append("Python binary contract does not match Cargo")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"release versions agree: Rust {rust_version}, Python {expected_python}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
