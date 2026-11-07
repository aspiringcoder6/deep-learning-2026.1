"""Evaluate predictions on the original score scale for one ASAP prompt."""

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_squared_error


def clip_scores(predictions: Sequence[float], bounds: tuple[int, int]) -> np.ndarray:
    scores = np.asarray(predictions, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Predicted scores must be finite.")
    return np.clip(scores, *bounds)


def round_scores(scores: np.ndarray) -> np.ndarray:
    return np.floor(scores + 0.5).astype(int)


def evaluate_scores(
    actual: Sequence[int], predictions: Sequence[float], bounds: tuple[int, int]
) -> dict[str, float | None]:
    clipped = clip_scores(predictions, bounds)
    mse = float(mean_squared_error(actual, clipped))
    rounded = round_scores(clipped)
    if len(set(actual).union(rounded)) == 1:
        # Identical constant ratings have zero expected disagreement.
        return {"qwk": None, "mse": mse}

    low, high = bounds
    qwk = cohen_kappa_score(actual, rounded, labels=list(range(low, high + 1)), weights="quadratic")
    return {"qwk": float(qwk), "mse": mse}
