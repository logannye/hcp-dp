from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from hcp_align import Client, HcpAlignError, _MAX_DIAGNOSTIC_BYTES


class ClientTests(unittest.TestCase):
    def make_script(self, directory: str, behavior: str) -> Path:
        script = Path(directory) / "fake_hcp_align.py"
        script.write_text(f"import sys\n{behavior}\n", encoding="utf-8")
        return script

    def make_client(self) -> Client:
        version = subprocess.CompletedProcess(
            ["fake-hcp-align", "--version"],
            returncode=0,
            stdout="hcp-align 0.2.0-rc.1\n",
            stderr=None,
        )
        with patch("hcp_align.subprocess.run", return_value=version):
            return Client("fake-hcp-align")

    @contextmanager
    def run_script(
        self,
        script: Path,
        processes: list[subprocess.Popen[str]] | None = None,
    ):
        real_popen = subprocess.Popen

        def launch(command, *args, **kwargs):
            process = real_popen(
                [sys.executable, str(script), *command[1:]], *args, **kwargs
            )
            if processes is not None:
                processes.append(process)
            return process

        with patch("hcp_align.subprocess.Popen", side_effect=launch):
            yield

    def test_streams_jsonl_and_checks_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            script = self.make_script(
                directory,
                "print('{\"record_type\":\"alignment\",\"pair\":{\"index\":0}}')\n"
                "print('{\"record_type\":\"alignment\",\"pair\":{\"index\":1}}')",
            )
            client = self.make_client()
            with self.run_script(script):
                records = list(
                    client.solve_batch(
                        "edit-distance", query_file="query.fa", target_file="target.fa"
                    )
                )
            self.assertEqual(client.version, "0.2.0-rc.1")
            self.assertEqual(records[0]["pair"]["index"], 0)
            self.assertEqual(records[1]["pair"]["index"], 1)

    def test_rejects_mismatched_binary_version(self) -> None:
        version = subprocess.CompletedProcess(
            ["fake-hcp-align", "--version"],
            returncode=0,
            stdout="hcp-align 9.9.9\n",
            stderr=None,
        )
        with patch("hcp_align.subprocess.run", return_value=version):
            with self.assertRaises(RuntimeError):
                Client("fake-hcp-align")

    def test_large_stderr_is_bounded_and_preserves_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            script = self.make_script(
                directory,
                "print('{\"record_type\":\"alignment\",\"pair\":{\"index\":0}}')\n"
                "sys.stdout.flush()\n"
                f"sys.stderr.write('x' * ({_MAX_DIAGNOSTIC_BYTES} + 4096) + 'TAIL')\n"
                "raise SystemExit(7)",
            )
            client = self.make_client()
            with self.run_script(script):
                with self.assertRaises(HcpAlignError) as raised:
                    list(
                        client.solve_batch(
                            "edit-distance",
                            query_file="query.fa",
                            target_file="target.fa",
                        )
                    )
            error = raised.exception
            self.assertEqual(error.returncode, 7)
            self.assertIn("stderr truncated", error.stderr)
            self.assertTrue(error.stderr.endswith("TAIL"))
            self.assertLessEqual(len(error.stderr), _MAX_DIAGNOSTIC_BYTES + 80)

    def test_nonzero_exit_returns_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            script = self.make_script(
                directory,
                "sys.stderr.write('specific failure\\n')\nraise SystemExit(9)",
            )
            client = self.make_client()
            with self.run_script(script):
                with self.assertRaises(HcpAlignError) as raised:
                    list(
                        client.solve_batch(
                            "edit-distance",
                            query_file="query.fa",
                            target_file="target.fa",
                        )
                    )
            self.assertEqual(raised.exception.returncode, 9)
            self.assertEqual(raised.exception.stderr, "specific failure\n")

    def test_early_generator_close_terminates_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            script = self.make_script(
                directory,
                "import time\n"
                "print('{\"record_type\":\"alignment\",\"pair\":{\"index\":0}}', flush=True)\n"
                "while True:\n"
                "    sys.stderr.write('still running\\n')\n"
                "    sys.stderr.flush()\n"
                "    time.sleep(0.05)",
            )
            client = self.make_client()
            processes: list[subprocess.Popen[str]] = []
            with self.run_script(script, processes):
                records = client.solve_batch(
                    "edit-distance", query_file="query.fa", target_file="target.fa"
                )
                self.assertEqual(next(records)["pair"]["index"], 0)
                records.close()

            self.assertEqual(len(processes), 1)
            self.assertIsNotNone(processes[0].poll())

    def test_temporary_diagnostic_sink_is_closed_after_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            script = self.make_script(
                directory,
                "sys.stderr.write('informational\\n')\n"
                "print('{\"record_type\":\"alignment\",\"pair\":{\"index\":0}}')",
            )
            client = self.make_client()
            with self.run_script(script):
                records = list(
                    client.solve_batch(
                        "edit-distance", query_file="query.fa", target_file="target.fa"
                    )
                )
            self.assertEqual(len(records), 1)


if __name__ == "__main__":
    unittest.main()
