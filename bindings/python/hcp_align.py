"""Subprocess client for the hcp-align v0.2 JSON contract."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class HcpAlignError(RuntimeError):
    def __init__(self, returncode: int, stderr: str) -> None:
        super().__init__(stderr.strip() or f"hcp-align exited with {returncode}")
        self.returncode = returncode
        self.stderr = stderr


@dataclass(frozen=True)
class HcpAlign:
    executable: str | None = None

    def run_json(self, args: Sequence[str], *, accepted_codes: tuple[int, ...] = (0,)) -> Any:
        completed = subprocess.run(
            [self._executable(), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode not in accepted_codes:
            raise HcpAlignError(completed.returncode, completed.stderr)
        return json.loads(completed.stdout)

    def solve_edit_distance(
        self,
        query: str,
        target: str,
        *,
        cross_check: bool = True,
        integrity: bool = False,
        score_only: bool = False,
    ) -> Mapping[str, Any]:
        args = [
            "solve",
            "edit-distance",
            "--query",
            query,
            "--target",
            target,
            "--format",
            "json",
        ]
        if cross_check:
            args.append("--cross-check")
        if integrity:
            args.append("--integrity")
        if score_only:
            args.extend(["--scope", "score"])
        return self.run_json(args)

    def solve_global_linear(
        self,
        query: str,
        target: str,
        *,
        match: int = 2,
        mismatch_penalty: int = 1,
        gap: int = -2,
        cross_check: bool = True,
    ) -> Mapping[str, Any]:
        args = [
            "solve",
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
            "--format",
            "json",
        ]
        if cross_check:
            args.append("--cross-check")
        return self.run_json(args)

    def verify_json(
        self,
        alignment: str | Path,
        *,
        query: str,
        target: str,
        level: str = "optimal",
        max_cells: int = 25_000_000,
    ) -> Mapping[str, Any]:
        return self.run_json(
            [
                "verify",
                "--alignment",
                str(alignment),
                "--format",
                "json",
                "--query",
                query,
                "--target",
                target,
                "--level",
                level,
                "--max-cells",
                str(max_cells),
                "--output-format",
                "json",
            ],
            accepted_codes=(0, 1, 3),
        )

    def _executable(self) -> str:
        if self.executable:
            return self.executable
        if path := os.environ.get("HCP_ALIGN_BIN"):
            return path
        if path := shutil.which("hcp-align"):
            return path
        raise FileNotFoundError("hcp-align not found; set HCP_ALIGN_BIN or install it")
