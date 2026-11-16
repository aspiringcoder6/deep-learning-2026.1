"""QWK and MSE on the original rubric scale, without model dependencies."""

import math
from collections.abc import Sequence


def _checked(actual: Sequence[int], predictions: Sequence[float], bounds: tuple[int, int]):
    low, high = bounds
    if low >= high:
        raise ValueError("Score bounds must satisfy min < max.")
    if not actual or len(actual) != len(predictions):
        raise ValueError("Actual and predicted scores must have the same nonzero length.")
    if any(isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high for value in actual):
        raise ValueError("Actual scores must be integers inside the rubric range.")
    return clip_scores(predictions, bounds)


def clip_scores(predictions: Sequence[float], bounds: tuple[int, int]) -> list[float]:
    low, high = bounds
    scores = [float(value) for value in predictions]
    if not all(math.isfinite(value) for value in scores):
        raise ValueError("Predicted scores must be finite.")
    return [min(high, max(low, value)) for value in scores]


def round_scores(scores: Sequence[float]) -> list[int]:
    """Round nonnegative rubric scores to the nearest integer, halves upward."""
    return [math.floor(value + 0.5) for value in scores]


def mse(actual: Sequence[int], predictions: Sequence[float], bounds: tuple[int, int]) -> float:
    """Mean squared error of continuous predictions after rubric clipping."""
    clipped = _checked(actual, predictions, bounds)
    return sum((truth - guess) ** 2 for truth, guess in zip(actual, clipped, strict=True)) / len(actual)


def qwk(actual: Sequence[int], predictions: Sequence[float], bounds: tuple[int, int]) -> float | None:
    """Quadratic weighted kappa of clipped, rounded rubric scores.

    Returns None if both series are the same constant: expected disagreement is zero.
    """
    rounded = round_scores(_checked(actual, predictions, bounds))
    n = len(actual)
    observed = sum((truth - guess) ** 2 for truth, guess in zip(actual, rounded, strict=True))
    actual_counts = {}
    predicted_counts = {}
    for truth, guess in zip(actual, rounded, strict=True):
        actual_counts[truth] = actual_counts.get(truth, 0) + 1
        predicted_counts[guess] = predicted_counts.get(guess, 0) + 1
    expected = sum(
        count_a * count_b * (a - b) ** 2
        for a, count_a in actual_counts.items()
        for b, count_b in predicted_counts.items()
    )
    return None if expected == 0 else 1.0 - observed * n / expected


def evaluate_scores(
    actual: Sequence[int], predictions: Sequence[float], bounds: tuple[int, int]
) -> dict[str, float | None]:
    return {"qwk": qwk(actual, predictions, bounds), "mse": mse(actual, predictions, bounds)}
