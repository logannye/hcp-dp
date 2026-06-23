"""Small Python wrapper for the hcp-align CLI JSON interface."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


class HcpAlignError(RuntimeError):
    """Raised when hcp-align exits unsuccessfully."""

    def __init__(self, returncode: int, stderr: str) -> None:
        super().__init__(stderr.strip() or f"hcp-align exited with {returncode}")
        self.returncode = returncode
        self.stderr = stderr


@dataclass(frozen=True)
class HcpAlign:
    """Convenience client for hcp-align.

    The executable is resolved from the explicit constructor argument, then
    ``HCP_ALIGN_BIN``, then ``PATH``.
    """

    executable: str | None = None

    def run_json(self, args: Sequence[str]) -> Any:
        completed = subprocess.run(
            [self._executable(), *args, "--format", "json"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise HcpAlignError(completed.returncode, completed.stderr)
        return json.loads(completed.stdout)

    def edit_distance(
        self,
        query: str,
        target: str,
        *,
        verify: bool = False,
        engine: str | None = None,
        score_only: bool = False,
    ) -> Mapping[str, Any]:
        args = ["edit-distance", "--query", query, "--target", target]
        if engine is not None:
            args.extend(["--engine", engine])
        if score_only:
            args.append("--score-only")
        if verify:
            args.append("--verify")
        return self.run_json(args)

    def global_linear(
        self,
        query: str,
        target: str,
        *,
        match: int = 2,
        mismatch_penalty: int = 1,
        gap: int = -2,
        verify: bool = False,
        extra_args: Iterable[str] = (),
    ) -> Mapping[str, Any]:
        args = [
            "global-linear",
            "--query",
            query,
            "--target",
            target,
            "--match",
            str(match),
            "--mismatch-penalty",
            str(mismatch_penalty),
            "--gap",
            str(gap),
            *extra_args,
        ]
        if verify:
            args.append("--verify")
        return self.run_json(args)

    def _executable(self) -> str:
        if self.executable:
            return self.executable
        if env_path := os.environ.get("HCP_ALIGN_BIN"):
            return env_path
        if path := shutil.which("hcp-align"):
            return path
        raise FileNotFoundError(
            "hcp-align executable not found; set HCP_ALIGN_BIN or install hcp-align"
        )


def edit_distance(query: str, target: str, **kwargs: Any) -> Mapping[str, Any]:
    return HcpAlign().edit_distance(query, target, **kwargs)


def global_linear(query: str, target: str, **kwargs: Any) -> Mapping[str, Any]:
    return HcpAlign().global_linear(query, target, **kwargs)
