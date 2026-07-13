#!/usr/bin/env python3
"""Keep the stable v0.2.0 tag gated on documented external use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--evidence", type=Path, default=Path("release/adoption.json"))
    args = parser.parse_args()
    if args.tag != "v0.2.0":
        print(f"external-adoption gate does not apply to {args.tag}")
        return 0

    document = json.loads(args.evidence.read_text(encoding="utf-8"))
    if document.get("schema_version") != "hcp-align.adoption.v1":
        raise SystemExit("invalid adoption evidence schema_version")
    pilots = document.get("pilots")
    if not isinstance(pilots, list) or len(pilots) < 3:
        raise SystemExit("v0.2.0 requires evidence from at least three external pilots")
    required = {"pilot", "job", "summary", "evidence_url", "validated_at"}
    identities: set[str] = set()
    for index, pilot in enumerate(pilots):
        if not isinstance(pilot, dict) or required - set(pilot):
            raise SystemExit(f"pilot {index} is missing required adoption evidence")
        identity = str(pilot["pilot"]).strip()
        if not identity or identity in identities:
            raise SystemExit(f"pilot {index} has an empty or duplicate identity")
        identities.add(identity)
        if pilot["job"] not in {"validated_records", "reproduced_alignment_bug"}:
            raise SystemExit(f"pilot {index} has an unsupported job result")
        if not str(pilot["evidence_url"]).startswith("https://"):
            raise SystemExit(f"pilot {index} requires an HTTPS evidence URL")
        if len(str(pilot["summary"]).strip()) < 20:
            raise SystemExit(f"pilot {index} requires a concrete summary")
    print(f"validated external-adoption evidence for {len(identities)} pilots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
