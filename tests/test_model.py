"""Executable MLX invariants: causality, gradients, checkpointing and validation."""
import math
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from ores_llm.data import CharTokenizer, split_text
from ores_llm.model import ModelConfig, TinyLM
from scripts.train import evaluate_all


def small_model():
    mx.random.seed(11)
    return TinyLM(ModelConfig(vocab_size=16, context=8, dim=16, heads=4, layers=1))


class ModelTests(unittest.TestCase):
    def test_causal_attention_never_sees_future(self):
        model = small_model()
        model.eval()
        a = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        b = mx.array([[1, 2, 3, 9]], dtype=mx.int32)
        first = model(a)
        second = model(b)
        mx.eval(first, second)
        earlier_delta = mx.max(mx.abs(first[:, :3, :] - second[:, :3, :]))
        self.assertLess(float(earlier_delta.item()), 1e-5)

    def test_finite_gradients_and_weight_update(self):
        model = small_model()
        optimizer = optim.AdamW(learning_rate=1e-3)
        x = mx.array([[0, 1, 2, 3], [2, 3, 4, 5]], dtype=mx.int32)
        y = mx.array([[1, 2, 3, 4], [3, 4, 5, 6]], dtype=mx.int32)

        def loss_fn(m, inputs, targets):
            return nn.losses.cross_entropy(m(inputs), targets, reduction="mean")

        before = [(name, np.array(value)) for name, value in tree_flatten(model.parameters())]
        loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
        self.assertTrue(math.isfinite(float(loss.item())))
        for _, gradient in tree_flatten(grads):
            self.assertTrue(bool(mx.all(mx.isfinite(gradient)).item()))
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        after = dict(tree_flatten(model.parameters()))
        changed = any(
            np.any(saved != np.asarray(after[name])) for name, saved in before
        )
        self.assertTrue(changed, "gradient step must change at least one weight")

    def test_checkpoint_roundtrip_preserves_predictions(self):
        model = small_model()
        model.eval()
        input_ids = mx.array([[1, 2, 3]], dtype=mx.int32)
        before = model(input_ids)
        mx.eval(before)
        with tempfile.TemporaryDirectory() as d:
            checkpoint = str(Path(d) / "model.safetensors")
            model.save_weights(checkpoint)
            reloaded = small_model()
            reloaded.load_weights(checkpoint)
            reloaded.eval()
            after = reloaded(input_ids)
            mx.eval(after)
            self.assertLess(float(mx.max(mx.abs(before - after)).item()), 1e-5)

    def test_validation_scores_all_targets_and_is_deterministic(self):
        model = small_model()
        values = np.arange(23, dtype=np.int32) % 16
        loss1, count1 = evaluate_all(model, values, 8)
        loss2, count2 = evaluate_all(model, values, 8)
        self.assertEqual(count1, 22)
        self.assertEqual(count2, 22)
        self.assertTrue(math.isfinite(loss1))
        self.assertAlmostEqual(loss1, loss2, places=6)

    def test_sample_has_no_unknown_validation_character(self):
        text = (Path(__file__).resolve().parents[1] / "data/sample.txt").read_text(encoding="utf-8")
        train, valid = split_text(text)
        self.assertTrue(CharTokenizer.fit(train).encode(valid))

    def test_invalid_token_shape_is_rejected(self):
        model = small_model()
        with self.assertRaises(ValueError):
            model(mx.array([1, 2, 3], dtype=mx.int32))
        with self.assertRaises(ValueError):
            model(mx.zeros((1, 9), dtype=mx.int32))


if __name__ == "__main__":
    unittest.main()
