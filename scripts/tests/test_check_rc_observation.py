from __future__ import annotations

import importlib.util
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "check_rc_observation.py"
SPEC = importlib.util.spec_from_file_location("check_rc_observation", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RcObservationTests(unittest.TestCase):
    now = datetime(2026, 8, 20, tzinfo=timezone.utc)

    def responses(self, published_at: datetime, blockers: list[dict[str, object]]):
        return [
            {"published_at": published_at.isoformat().replace("+00:00", "Z")},
            blockers,
        ]

    def test_accepts_old_rc_without_blockers(self) -> None:
        with patch.object(
            MODULE,
            "request_json",
            side_effect=self.responses(self.now - timedelta(days=8), []),
        ):
            MODULE.validate("logannye/hcp-dp", "token", self.now)

    def test_rejects_young_rc(self) -> None:
        with patch.object(
            MODULE,
            "request_json",
            side_effect=self.responses(self.now - timedelta(days=6), []),
        ):
            with self.assertRaises(SystemExit):
                MODULE.validate("logannye/hcp-dp", "token", self.now)

    def test_rejects_open_release_blocker(self) -> None:
        with patch.object(
            MODULE,
            "request_json",
            side_effect=self.responses(
                self.now - timedelta(days=8), [{"number": 42}]
            ),
        ):
            with self.assertRaises(SystemExit) as raised:
                MODULE.validate("logannye/hcp-dp", "token", self.now)
        self.assertIn("#42", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
