"""Compiler-gate tests using a fake local compiler (no Oreslang compiler claim)."""
import os
import tempfile
import unittest
from pathlib import Path

from scripts.eval_compiler import evaluate


class CompilerGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.compiler = root / "compiler"
        self.compiler.write_text(
            "#!/usr/bin/env python3\nimport pathlib,sys\n"
            "p=pathlib.Path(sys.argv[-1])\n"
            "sys.exit(0 if p.read_text().startswith('ok') else 3)\n",
            encoding="utf-8",
        )
        self.compiler.chmod(0o755)
        self.inputs = root / "candidates"
        self.inputs.mkdir()
        (self.inputs / "good.ores").write_text("ok source\n", encoding="utf-8")
        (self.inputs / "bad.ores").write_text("bad source\n", encoding="utf-8")

    def test_compiler_acceptance_and_rejection(self):
        result = evaluate(self.compiler, self.inputs, ["{source}"])
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["accepted"], 1)
        self.assertEqual(result["acceptance_rate"], 0.5)
        self.assertEqual({r["status"] for r in result["results"]},
                         {"accepted", "rejected"})

    def test_reject_shell_injection_and_invalid_placeholder(self):
        with self.assertRaises(ValueError):
            evaluate(self.compiler, self.inputs, ["{source}", "{source}"])
        with self.assertRaises(ValueError):
            evaluate(self.compiler, self.inputs, ["--check"])

    def test_file_budget(self):
        with self.assertRaises(ValueError):
            evaluate(self.compiler, self.inputs, ["{source}"], max_files=1)


if __name__ == "__main__":
    unittest.main()
