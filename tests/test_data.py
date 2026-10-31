import tempfile
import unittest
from pathlib import Path
import numpy as np
from ores_llm.data import CharTokenizer, batch, split_text
from ores_llm.model import ModelConfig


class DataTests(unittest.TestCase):
    def test_tokenizer_roundtrip(self):
        t = CharTokenizer.fit("a🙂bcab")
        self.assertEqual(t.decode(t.encode("🙂abc")), "🙂abc")
        with self.assertRaises(ValueError):
            t.encode("unseen")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "vocab.json"
            t.save(path)
            self.assertEqual(CharTokenizer.load(path).decode(t.encode("🙂")), "🙂")

    def test_split_and_sample(self):
        train, valid = split_text("abcdefghij", .6)
        self.assertEqual((train, valid), ("abcdef", "ghij"))
        x, y = batch(np.arange(24), 4, 3, np.random.default_rng(1))
        self.assertEqual(x.shape, (3, 4))
        np.testing.assert_array_equal(x[:, 1:], y[:, :-1])

    def test_configuration_validation(self):
        with self.assertRaises(ValueError):
            ModelConfig(10, 8, 15, 4, 2).validate()


if __name__ == "__main__":
    unittest.main()
