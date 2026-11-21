import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from essay_scoring.data import Essay
from essay_scoring.models.han import (
    HANRegressor, _encode, _pad_batch, _sentences, fit_han, predict_han, predict_checkpoint,
)


class HANTrainingTest(unittest.TestCase):
    def test_padding_masks_and_backward(self):
        torch.manual_seed(42)
        model = HANRegressor(6).eval()
        tokens = _pad_batch([[[2, 3], [4]], [[5]]])
        padded = F.pad(tokens, (0, 3, 0, 2))
        predicted = model(tokens)
        self.assertEqual(predicted.shape, (2,))
        self.assertTrue(torch.all((predicted >= 0) & (predicted <= 1)))
        torch.testing.assert_close(predicted, model(padded), atol=1e-6, rtol=1e-6)
        loss = F.mse_loss(predicted, torch.tensor([0.2, 0.8]))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertEqual(_encode(_sentences("...", 48, 64), {}), [[1]])

    def test_best_checkpoint_history_and_reload(self):
        train = [Essay(1, "A useful first essay.", 6), Essay(2, "Another longer essay. Good ideas!", 10)]
        validation = [Essay(3, "Validationonly essay.", 7), Essay(4, "Unseenword text.", 9)]
        checkpoint, metrics = fit_han(train, validation, (2, 12), epochs=2, batch_size=2, seed=42)
        history = checkpoint["training"]["history"]
        best = min(history, key=lambda row: row["validation_mse"])
        self.assertEqual(checkpoint["training"]["best_epoch"], best["epoch"])
        self.assertEqual(metrics["mse"], best["validation_mse"])
        self.assertIn("another", checkpoint["vocabulary"])
        self.assertNotIn("validationonly", checkpoint["vocabulary"])
        texts = [essay.text for essay in train]
        before = predict_checkpoint(checkpoint, texts, (2, 12))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(checkpoint, path)
            self.assertEqual(before, predict_han(path, texts, (2, 12)))
            with self.assertRaisesRegex(ValueError, "bounds"):
                predict_han(path, texts, (0, 4))


if __name__ == "__main__":
    unittest.main()
