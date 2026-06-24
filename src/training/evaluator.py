from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    metrics: dict[str, float] = {
        "brier_score": float(brier_score_loss(y_true, y_score)),
    }
    if len(set(y_true.tolist())) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["average_precision"] = float(average_precision_score(y_true, y_score))
        metrics["log_loss"] = float(log_loss(y_true, y_score, labels=[0, 1]))
    else:
        metrics["roc_auc"] = 0.5
        metrics["average_precision"] = float(np.mean(y_true))
        metrics["log_loss"] = 0.0
    return metrics


def per_release_metrics(y_true: np.ndarray, y_score: np.ndarray, source_dates: list[str]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, source_date in enumerate(source_dates):
        grouped[source_date].append(index)
    return {
        source_date: classification_metrics(y_true[indexes], y_score[indexes])
        for source_date, indexes in grouped.items()
        if indexes
    }


def summarize_evaluation(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    source_dates = [str(item.get("source_date", "")) for item in metadata]
    return {
        "overall": classification_metrics(y_true, y_score),
        "per_release": per_release_metrics(y_true, y_score, source_dates),
    }
