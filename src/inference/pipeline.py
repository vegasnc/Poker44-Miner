from __future__ import annotations

import time
from typing import Any

from src.inference.predictor import BotRiskPredictor


class InferencePipeline:
    def __init__(self, predictor: BotRiskPredictor | None = None, threshold: float = 0.5) -> None:
        self.predictor = predictor or BotRiskPredictor()
        self.threshold = threshold

    def score_synapse_chunks(self, chunks: list[Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        scores = self.predictor.predict_chunks(chunks)
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "risk_scores": scores,
            "predictions": [score >= self.threshold for score in scores],
            "model_manifest": self.predictor.manifest(),
            "inference_latency_ms": latency_ms,
        }
