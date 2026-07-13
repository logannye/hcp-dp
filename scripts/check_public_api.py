#!/usr/bin/env python3
"""Guard the documented v0.2 Rust API from implementation-type leakage."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIB = (ROOT / "src/lib.rs").read_text(encoding="utf-8")

PRIVATE_MODULES = ("alignment", "batch", "oracle", "record_io", "scoring", "sequence_store")
REQUIRED_ROOT_EXPORTS = (
    "AlignmentMode",
    "AlignmentRecord",
    "AlignmentSpan",
    "BatchOptions",
    "BatchSummary",
    "Cigar",
    "CigarOp",
    "ComparisonReport",
    "ErrorRecord",
    "GapModel",
    "ScoringScheme",
    "SolveOptions",
    "SolveRequest",
    "SubstitutionMatrix",
    "VerificationLevel",
    "VerificationReport",
    "VerificationRequest",
    "compare",
    "solve",
    "verify",
)
FORBIDDEN_PUBLIC_ITEMS = {
    "src/batch.rs": ("BatchItem", "run_ordered"),
    "src/record_io.rs": ("AlignmentRecordReader", "SamSpool"),
    "src/sequence_store.rs": ("SequenceKind", "SequenceStore"),
}


def public_declaration(source: str, item: str) -> bool:
    pattern = rf"(?m)^pub\s+(?:struct|enum|fn|type|trait)\s+{re.escape(item)}\b"
    return re.search(pattern, source) is not None


def main() -> int:
    failures: list[str] = []
    for module in PRIVATE_MODULES:
        if re.search(rf"(?m)^pub\s+mod\s+{module}\s*;", LIB):
            failures.append(f"implementation module src::{module} must remain private")
        if not re.search(rf"(?m)^mod\s+{module}\s*;", LIB):
            failures.append(f"expected private module declaration for src::{module}")

    for item in REQUIRED_ROOT_EXPORTS:
        if not re.search(rf"\b{re.escape(item)}\b", LIB):
            failures.append(f"documented root export {item} is missing")

    for relative_path, items in FORBIDDEN_PUBLIC_ITEMS.items():
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        for item in items:
            if public_declaration(source, item):
                failures.append(f"implementation type {item} is public in {relative_path}")

    if "pub use crate::cli::main as __hcp_align_main" not in LIB:
        failures.append("binary entry point must remain a doc-hidden root shim")

    if failures:
        raise SystemExit("public API audit failed:\n- " + "\n- ".join(failures))
    print("public API audit passed: stable root surface contains no storage or scheduler types")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
