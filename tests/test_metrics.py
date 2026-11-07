import unittest

from essay_scoring.metrics import evaluate_scores


class EssayMetricsTest(unittest.TestCase):
    def test_perfect_predictions(self):
        metrics = evaluate_scores([2, 5, 12], [2.0, 5.0, 12.0], (2, 12))
        self.assertEqual(metrics, {"qwk": 1.0, "mse": 0.0})

    def test_mse_uses_clipped_scores_before_rounding(self):
        metrics = evaluate_scores([2, 5, 12], [-3.0, 4.5, 15.0], (2, 12))
        self.assertEqual(metrics["qwk"], 1.0)
        self.assertAlmostEqual(metrics["mse"], 0.25 / 3)

    def test_qwk_is_undefined_for_identical_constant_scores(self):
        metrics = evaluate_scores([3, 3], [3.0, 3.0], (0, 4))
        self.assertEqual(metrics, {"qwk": None, "mse": 0.0})

    def test_qwk_preserves_distance_between_missing_score_levels(self):
        metrics = evaluate_scores([0, 1, 4], [0.0, 4.0, 4.0], (0, 4))
        self.assertAlmostEqual(metrics["qwk"], 40 / 67)
