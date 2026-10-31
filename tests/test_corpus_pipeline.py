"""Pure-stdlib security, split, revision and LoRA preflight tests (no GPU)."""
import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.build_corpus import ROOT, build
from scripts.lora_local import checked_dataset, command


def git(path, *argv):
    return subprocess.check_output(["git", "-C", str(path), *argv],
                                   text=True, stderr=subprocess.DEVNULL).strip()


class CorpusTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "artifacts").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "artifacts")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sources = []
        for index in range(3):
            src = self.root / f"repo{index}"
            src.mkdir()
            git(src, "init", "-q")
            (src / "lesson.ores").write_text(
                f"define actor Demo{index} as\n"
                f"  receive(Message m) -> void ::\n"
                f"    // example {index}, independent fixture group\n"
                f"  end\nend\n", encoding="utf-8")
            git(src, "add", "lesson.ores")
            git(src, "-c", "user.name=Fixture", "-c",
                "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
            self.sources.append({"id": f"repo-{index}", "path": str(src),
                                 "revision": git(src, "rev-parse", "HEAD"),
                                 "authorized": True, "license": "MIT"})
        self.manifest = {"version": 1, "sources": self.sources}
        self.counter = 0

    def output(self):
        self.counter += 1
        return self.root / f"build-{self.counter}"

    def test_repository_disjoint_splits_and_hashes(self):
        out = self.output()
        report = build(self.manifest, out)
        self.assertEqual({s["split"] for s in report["sources"]},
                         {"train", "valid", "test"})
        self.assertEqual(sum(x["count"] for x in report["splits"].values()), 3)
        self.assertEqual(checked_dataset(out), report)
        for name in ("train", "valid", "test"):
            self.assertEqual(len((out / f"{name}.jsonl").read_text().splitlines()), 1)

    def test_rejects_missing_consent_and_license(self):
        self.sources[0]["authorized"] = False
        with self.assertRaisesRegex(ValueError, "rights approval"):
            build(self.manifest, self.output())
        self.sources[0]["authorized"] = True
        self.sources[0]["license"] = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "license"):
            build(self.manifest, self.output())

    def test_rejects_forged_revision_and_dirty_repo(self):
        orig = self.sources[0]["revision"]
        self.sources[0]["revision"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "revision"):
            build(self.manifest, self.output())
        self.sources[0]["revision"] = orig
        p = Path(self.sources[0]["path"]) / "lesson.ores"
        p.write_text(p.read_text() + "// dirty\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dirty tracked"):
            build(self.manifest, self.output())

    def test_rejects_secret(self):
        src = Path(self.sources[0]["path"])
        with (src / "lesson.ores").open("a", encoding="utf-8") as file:
            file.write("secret_key = 'AAAAAAAAAAAAAAAAAAAAAAAAAAA'\n")
        git(src, "add", "lesson.ores")
        git(src, "-c", "user.name=Fixture", "-c",
            "user.email=fixture@example.invalid", "commit", "-qm", "secret fixture")
        self.sources[0]["revision"] = git(src, "rev-parse", "HEAD")
        with self.assertRaisesRegex(ValueError, "secret"):
            build(self.manifest, self.output())

    def test_rejects_cross_repo_duplicate(self):
        src = Path(self.sources[2]["path"])
        data = (Path(self.sources[0]["path"]) / "lesson.ores").read_text(encoding="utf-8")
        (src / "lesson.ores").write_text(data, encoding="utf-8")
        git(src, "add", "lesson.ores")
        git(src, "-c", "user.name=Fixture", "-c",
            "user.email=fixture@example.invalid", "commit", "-qm", "duplicate")
        self.sources[2]["revision"] = git(src, "rev-parse", "HEAD")
        with self.assertRaisesRegex(ValueError, "cross-repository duplicate"):
            build(self.manifest, self.output())

    def test_ignores_untracked_sources(self):
        src = Path(self.sources[0]["path"])
        (src / "untracked.ores").write_text(
            "A private untracked source file should never be copied.\n", encoding="utf-8")
        out = self.output()
        build(self.manifest, out)
        all_text = "".join((out / f"{split}.jsonl").read_text()
                           for split in ("train", "valid", "test"))
        self.assertNotIn("private untracked source", all_text)

    def test_rejects_output_outside_ignored_directory(self):
        with self.assertRaisesRegex(ValueError, "artifacts"):
            build(self.manifest, self.root.parent.parent / "outside")

    def test_detects_tampered_dataset(self):
        out = self.output()
        build(self.manifest, out)
        (out / "train.jsonl").write_text('{"text":"changed"}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            checked_dataset(out)

    def test_lora_cli_requires_local_model_and_explicit_rights(self):
        out = self.output()
        build(self.manifest, out)
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text("{}\n", encoding="utf-8")
        args = argparse.Namespace(model=model, data=out,
                                  adapter=self.root / "adapter",
                                  mode="train", iters=10, batch_size=1,
                                  num_layers=2, max_seq_length=128,
                                  lr=1e-5, seed=42, confirm_model_rights=True)
        argv = command(args)
        self.assertIn("mlx_lm", argv)
        self.assertIn("--train", argv)
        args.confirm_model_rights = False
        with self.assertRaisesRegex(ValueError, "model license"):
            command(args)


if __name__ == "__main__":
    unittest.main()
