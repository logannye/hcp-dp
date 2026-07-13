"""Streaming subprocess client for the hcp-align v2 JSONL contract."""

from __future__ import annotations

import importlib.resources
import json
import os
import shutil
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

__version__ = "0.2.0a2"
_BINARY_VERSION = "0.2.0-alpha.2"


class HcpAlignError(RuntimeError):
    def __init__(self, returncode: int, stderr: str) -> None:
        super().__init__(stderr.strip() or f"hcp-align exited with {returncode}")
        self.returncode = returncode
        self.stderr = stderr


class Client:
    """Resolve one compatible binary and expose streaming batch generators."""

    def __init__(self, executable: str | os.PathLike[str] | None = None) -> None:
        self.executable = self._resolve(executable)
        self.version = self._check_version()

    def _resolve(self, explicit: str | os.PathLike[str] | None) -> str:
        if explicit is not None:
            return str(explicit)
        binary_name = "hcp-align.exe" if os.name == "nt" else "hcp-align"
        bundled = importlib.resources.files("hcp_align").joinpath("bin", binary_name)
        if bundled.is_file():
            return os.fspath(bundled)
        if configured := os.environ.get("HCP_ALIGN_BIN"):
            return configured
        if discovered := shutil.which("hcp-align"):
            return discovered
        raise FileNotFoundError(
            "hcp-align not found in this wheel, HCP_ALIGN_BIN, or PATH"
        )

    def _check_version(self) -> str:
        completed = subprocess.run(
            [self.executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise HcpAlignError(completed.returncode, completed.stderr)
        version = completed.stdout.strip().split()[-1]
        if version != _BINARY_VERSION:
            raise RuntimeError(
                f"hcp-align binary {version!r} does not match Python package "
                f"binary contract {_BINARY_VERSION!r}"
            )
        return version

    def _jsonl(self, args: Sequence[str]) -> Iterator[dict[str, Any]]:
        process = subprocess.Popen(
            [self.executable, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        try:
            for line in process.stdout:
                if line.strip():
                    yield json.loads(line)
            stderr = process.stderr.read()
            returncode = process.wait()
            if returncode != 0:
                raise HcpAlignError(returncode, stderr)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
            process.stderr.close()

    def solve_batch(
        self,
        mode: str,
        *,
        query_file: str | os.PathLike[str],
        target_file: str | os.PathLike[str],
        extra_args: Sequence[str] = (),
    ) -> Iterator[dict[str, Any]]:
        return self._jsonl(
            [
                "solve",
                mode,
                "--query-file",
                str(query_file),
                "--target-file",
                str(target_file),
                "--format",
                "jsonl",
                *extra_args,
            ]
        )

    def verify_batch(
        self,
        alignment: str | os.PathLike[str],
        *,
        format: str,
        query_file: str | os.PathLike[str],
        target_file: str | os.PathLike[str],
        extra_args: Sequence[str] = (),
    ) -> Iterator[dict[str, Any]]:
        return self._jsonl(
            [
                "verify",
                "--alignment",
                str(alignment),
                "--format",
                format,
                "--query-file",
                str(query_file),
                "--target-file",
                str(target_file),
                "--output-format",
                "jsonl",
                *extra_args,
            ]
        )

    def compare_batch(
        self,
        *,
        left: str | os.PathLike[str],
        left_format: str,
        right: str | os.PathLike[str],
        right_format: str,
        query_file: str | os.PathLike[str],
        target_file: str | os.PathLike[str],
        extra_args: Sequence[str] = (),
    ) -> Iterator[dict[str, Any]]:
        return self._jsonl(
            [
                "compare",
                "--left",
                str(left),
                "--left-format",
                left_format,
                "--right",
                str(right),
                "--right-format",
                right_format,
                "--query-file",
                str(query_file),
                "--target-file",
                str(target_file),
                "--output-format",
                "jsonl",
                *extra_args,
            ]
        )


HcpAlign = Client


def solve_batch(mode: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
    return Client().solve_batch(mode, **kwargs)


def verify_batch(alignment: str | os.PathLike[str], **kwargs: Any) -> Iterator[dict[str, Any]]:
    return Client().verify_batch(alignment, **kwargs)


def compare_batch(**kwargs: Any) -> Iterator[dict[str, Any]]:
    return Client().compare_batch(**kwargs)


__all__ = [
    "Client",
    "HcpAlign",
    "HcpAlignError",
    "compare_batch",
    "solve_batch",
    "verify_batch",
]
