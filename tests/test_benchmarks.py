"""Dependency-free tests of Mac benchmark integrity and comparability."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_benchmarks import compare


def report(chip, cores, rate):
    return {
        "status": "gpu_smoke_pass",
        "source_commit": "1" * 40,
        "tracked_source_clean": True,
        "hardware": {"chipset": chip, "gpu_cores": cores, "device": "gpu"},
        "model": {
            "status": "completed", "device": "gpu", "machine": "arm64",
            "corpus_sha256": "a" * 64, "seed": 42, "train_chars": 331,
            "valid_chars": 37, "valid_scored_targets": 36,
            "steps": 100, "batch_size": 8, "mlx_version": "0.32.4",
            "model": {"vocab_size": 32, "context": 16, "dim": 128,
                      "heads": 4, "layers": 2},
            "gpu_training_tokens_per_second": rate,
            "best_valid_loss": 2.3,
        }
    }


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.a = Path(self.dir.name) / "m4.json"
        self.b = Path(self.dir.name) / "m5.json"
        self.m4 = report("Apple M4 Max", 40, 8000.0)
        self.m5 = report("Apple M5 Pro", 20, 4000.0)

    def check(self):
        self.a.write_text(json.dumps(self.m4), encoding="utf-8")
        self.b.write_text(json.dumps(self.m5), encoding="utf-8")
        return compare(self.a, self.b)

    def test_valid_equivalent_reports(self):
        output = self.check()
        self.assertEqual(output["status"], "comparable_smoke_reports")
        self.assertEqual(output["m4_max_over_m5_pro_ratio"], 2.0)

    def test_reject_different_source_commits(self):
        self.m5["source_commit"] = "2" * 40
        with self.assertRaisesRegex(ValueError, "different Git source commits"):
            self.check()

    def test_reject_dirty_checkout(self):
        self.m4["tracked_source_clean"] = False
        with self.assertRaisesRegex(ValueError, "dirty source checkout"):
            self.check()

    def test_reject_wrong_hardware(self):
        self.m5["hardware"]["gpu_cores"] = 40
        with self.assertRaisesRegex(ValueError, "hardware tag"):
            self.check()

    def test_reject_cpu_fallback(self):
        self.m4["model"]["device"] = "cpu"
        with self.assertRaises(ValueError):
            self.check()

    def test_reject_mismatched_training_configuration(self):
        for field, value in [("seed", 7), ("batch_size", 16),
                             ("mlx_version", "0.99"), ("corpus_sha256", "b" * 64)]:
            with self.subTest(field=field):
                original = self.m5["model"][field]
                self.m5["model"][field] = value
                with self.assertRaisesRegex(ValueError, "mismatched"):
                    self.check()
                self.m5["model"][field] = original

    def test_reject_invalid_throughput(self):
        for rate in (0, -1, float("nan"), float("inf")):
            with self.subTest(rate=rate):
                self.m4["model"]["gpu_training_tokens_per_second"] = rate
                with self.assertRaises(ValueError):
                    self.check()

    def test_reject_failed_acceptance(self):
        self.m4["status"] = "failed"
        with self.assertRaises(ValueError):
            self.check()


if __name__ == "__main__":
    unittest.main()
