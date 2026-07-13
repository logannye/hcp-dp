from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from check_release_tag import validate_tag


class ReleaseTagPolicyTests(unittest.TestCase):
    tag = "v0.2.0-alpha.2"
    commit = "a" * 40

    def ref(self, kind: str = "tag") -> dict:
        return {"object": {"type": kind, "sha": "b" * 40}}

    def annotated(self, *, verified: bool = True, target: str | None = None) -> dict:
        return {
            "tag": self.tag,
            "object": {"type": "commit", "sha": target or self.commit},
            "verification": {
                "verified": verified,
                "reason": "valid" if verified else "unsigned",
            },
        }

    def validate(self, ref: dict, tag_object: dict | None) -> list[str]:
        return validate_tag(
            ref,
            tag_object,
            tag=self.tag,
            expected_commit=self.commit,
        )

    def test_signed_annotated_tag_is_accepted(self) -> None:
        self.assertEqual(self.validate(self.ref(), self.annotated()), [])

    def test_lightweight_tag_is_rejected(self) -> None:
        failures = self.validate(self.ref("commit"), None)
        self.assertIn("annotated tag", failures[0])

    def test_unsigned_annotated_tag_is_rejected(self) -> None:
        failures = self.validate(self.ref(), self.annotated(verified=False))
        self.assertTrue(any("verified signature" in failure for failure in failures))

    def test_wrong_target_commit_is_rejected(self) -> None:
        failures = self.validate(self.ref(), self.annotated(target="c" * 40))
        self.assertTrue(any("expected" in failure for failure in failures))

    def test_non_commit_target_is_rejected(self) -> None:
        tag_object = self.annotated()
        tag_object["object"]["type"] = "tag"
        failures = self.validate(self.ref(), tag_object)
        self.assertTrue(any("target a commit" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
