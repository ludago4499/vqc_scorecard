# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# History:
#   2026-09-09  Created. Classification metrics incl. Matthews correlation
#               (the official CoLA metric; robust to class imbalance).
#   2026-09-23  Added macro_f1 (mean of the two per-class F1) and
#               balanced_accuracy, the second headline of the fair protocol:
#               unlike the positive-class F1, neither rewards a majority-class
#               collapse on CoLA. Existing keys unchanged.

"""Binary classification metrics, pure NumPy."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def classification_metrics(y_true: NDArray, y_pred: NDArray) -> dict[str, float]:
    """Return accuracy, precision, recall, F1 and Matthews correlation.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        A dict with keys ``accuracy``, ``precision``, ``recall``, ``f1``, ``mcc``,
        ``macro_f1`` and ``balanced_accuracy``. Matthews correlation (``mcc``)
        is the headline metric for CoLA; it is 0 for a degenerate constant
        predictor and handles class imbalance. ``macro_f1`` averages the F1 of
        both classes, so a constant predictor scores at most about 0.5.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0
    f1_neg = 2 * tn / (2 * tn + fn + fp) if tn + fn + fp else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {
        "accuracy": float(np.mean(y_pred == y_true)),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "mcc": float(mcc),
        "macro_f1": float((f1 + f1_neg) / 2.0),
        "balanced_accuracy": float((rec + spec) / 2.0),
    }
