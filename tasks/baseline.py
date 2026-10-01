# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# History:
#   2026-09-09  Created. Logistic-regression baselines, pure NumPy, ported from
#               old task_sentiment.py.

"""Classical logistic-regression baselines (full-batch GD, L2)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .metrics import classification_metrics


def logreg(
    x_tr: NDArray,
    y_tr: NDArray,
    x_te: NDArray,
    y_te: NDArray,
    *,
    epochs: int = 400,
    lr: float = 1.0,
    l2: float = 1e-4,
) -> dict[str, float]:
    """Train L2 logistic regression and return test-set metrics.

    Args:
        x_tr: Training features ``(n_train, d)``.
        y_tr: Training labels.
        x_te: Test features ``(n_test, d)``.
        y_te: Test labels.
        epochs: Full-batch gradient steps.
        lr: Learning rate.
        l2: L2 penalty.

    Returns:
        The :func:`~vqc_scorecard.tasks.metrics.classification_metrics` dict on
        the test set.
    """
    w = np.zeros(x_tr.shape[1])
    b = 0.0
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-np.clip(x_tr @ w + b, -30, 30)))
        g = p - y_tr
        w -= lr * (x_tr.T @ g / len(y_tr) + l2 * w)
        b -= lr * float(np.mean(g))
    pred = ((x_te @ w + b) >= 0).astype(int)
    return classification_metrics(y_te, pred)
