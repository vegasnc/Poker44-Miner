from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.inference.predictor import BotRiskPredictor

# Primary: rightmost-gap scan parameters
_MIN_GAP_SCAN = 0.010   # minimum gap to trigger dynamic threshold
_MIN_CHUNKS = 10        # only use dynamic threshold when batch is large enough
_MAX_HUMAN_FRAC = 0.30  # at most 30% of chunks can be classified as human
_MIN_HUMAN_FRAC = 0.005 # require at least 1 human (max(1, int(0.005*100)) = 1)

# Secondary: IQR outlier detection parameters (catches humans with no clear gap)
_IQR_MULTIPLIER = 1.5   # Tukey fence — chunks below Q1 - 1.5*IQR are outliers
_IQR_MAX_HUMAN_FRAC = 0.20  # IQR method capped at 20% to avoid over-detection


def _dynamic_threshold(scores: list[float], base_threshold: float = 0.5) -> float:
    """Detect the human/bot boundary in the score distribution.

    Two-stage approach:
    1. Rightmost-gap scan: scans from highest scores downward, finds first gap
       >= _MIN_GAP_SCAN within the 1-30% human fraction range. Avoids the trap
       of a large gap WITHIN the human cluster being mistaken for the boundary.
    2. IQR outlier fallback: when no clear gap exists, uses Tukey fences to
       detect statistical outliers on the low end (likely human chunks in an
       otherwise tight bot cluster). Catches cases like 2 humans scoring 0.905
       in a batch where all bots score 0.920-0.965.

    Falls back to base_threshold if neither method finds a boundary (all-bot batch).
    """
    if len(scores) < _MIN_CHUNKS:
        return base_threshold

    arr = np.sort(scores)
    n = len(arr)
    gaps = np.diff(arr)

    # --- Stage 1: Rightmost-gap scan ---
    max_human_idx = int(_MAX_HUMAN_FRAC * n)
    min_human_idx = max(0, int(_MIN_HUMAN_FRAC * n) - 1)

    for idx in range(n - 2, min_human_idx - 1, -1):
        n_human = idx + 1
        if n_human > max_human_idx:
            continue
        gap = gaps[idx]
        if gap >= _MIN_GAP_SCAN:
            n_bot = n - n_human
            if n_bot < int(0.50 * n):
                continue
            return float((arr[idx] + arr[idx + 1]) / 2)

    # --- Stage 2: IQR outlier detection ---
    # Detects human chunks that have no sharp gap but are statistical outliers.
    # Only triggers when the overall score level is high (mean > 0.85),
    # meaning we are in a predominantly-bot batch where any human would score low.
    mean_score = float(np.mean(arr))
    if mean_score > 0.85:
        q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        iqr = q3 - q1
        if iqr > 0:
            lower_fence = q1 - _IQR_MULTIPLIER * iqr
            n_outliers = int(np.sum(arr < lower_fence))
            min_outliers = max(1, int(_MIN_HUMAN_FRAC * n))
            max_outliers = int(_IQR_MAX_HUMAN_FRAC * n)
            if min_outliers <= n_outliers <= max_outliers:
                # Midpoint between the highest outlier and lowest non-outlier
                boundary = float((arr[n_outliers - 1] + arr[n_outliers]) / 2)
                n_bot = n - n_outliers
                if n_bot >= int(0.50 * n):
                    return boundary

    return base_threshold


def _remap_scores(scores: list[float], threshold: float) -> list[float]:
    """Remap raw model scores so humans land below 0.5 and bots above 0.5.

    The validator uses np.round(risk_scores) to make binary predictions.
    Our model outputs scores in 0.73-0.89 range — all round to 1 (bot),
    making FPR=100% and reward=0. This remapping fixes that:
      - score < threshold  → maps linearly to [0.0, 0.5)
      - score >= threshold → maps linearly to [0.5, 1.0]
    Relative ranking within each class is preserved, so AP stays high.
    """
    remapped = []
    lo = min(scores) if scores else 0.0
    hi = max(scores) if scores else 1.0
    for s in scores:
        if s < threshold:
            # human side: map [lo, threshold) → [0.0, 0.5)
            span = threshold - lo
            r = (0.5 * (s - lo) / span) if span > 0 else 0.25
        else:
            # bot side: map [threshold, hi] → [0.5, 1.0]
            span = hi - threshold
            r = (0.5 + 0.5 * (s - threshold) / span) if span > 0 else 0.75
        remapped.append(round(max(0.0, min(1.0, r)), 6))
    return remapped


class InferencePipeline:
    def __init__(self, predictor: BotRiskPredictor | None = None, threshold: float = 0.5) -> None:
        self.predictor = predictor or BotRiskPredictor()
        self.threshold = threshold

    def score_synapse_chunks(self, chunks: list[Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        scores = self.predictor.predict_chunks(chunks)
        latency_ms = (time.perf_counter() - t0) * 1000
        threshold = _dynamic_threshold(scores, self.threshold)
        remapped = _remap_scores(scores, threshold)
        return {
            "risk_scores": remapped,
            "raw_scores_mean": float(np.mean(scores)) if scores else 0.5,
            "predictions": [score >= threshold for score in scores],
            "model_manifest": self.predictor.manifest(),
            "inference_latency_ms": latency_ms,
            "threshold_used": threshold,
        }
