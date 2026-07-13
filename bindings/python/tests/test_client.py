from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from hcp_align import Client


class ClientTests(unittest.TestCase):
    def test_streams_jsonl_and_checks_version(self) -> None:
        if os.name == "nt":
            self.skipTest("shell fixture is Unix-only")
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "hcp-align"
            binary.write_text(
                "#!/bin/sh\n"
                "if [ \"$1\" = --version ]; then echo 'hcp-align 0.2.0-alpha.2'; exit 0; fi\n"
                "echo '{\"record_type\":\"alignment\",\"pair\":{\"index\":0}}'\n",
                encoding="utf-8",
            )
            binary.chmod(0o755)
            client = Client(binary)
            records = list(
                client.solve_batch(
                    "edit-distance", query_file="query.fa", target_file="target.fa"
                )
            )
            self.assertEqual(client.version, "0.2.0-alpha.2")
            self.assertEqual(records[0]["pair"]["index"], 0)

    def test_rejects_mismatched_binary_version(self) -> None:
        if os.name == "nt":
            self.skipTest("shell fixture is Unix-only")
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "hcp-align"
            binary.write_text("#!/bin/sh\necho 'hcp-align 9.9.9'\n", encoding="utf-8")
            binary.chmod(0o755)
            with self.assertRaises(RuntimeError):
                Client(binary)


if __name__ == "__main__":
    unittest.main()
