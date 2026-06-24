from __future__ import annotations

from typing import Any

from src.inference.predictor import BotRiskPredictor


class InferencePipeline:
    def __init__(self, predictor: BotRiskPredictor | None = None, threshold: float = 0.5) -> None:
        self.predictor = predictor or BotRiskPredictor()
        self.threshold = threshold

    def score_synapse_chunks(self, chunks: list[Any]) -> dict[str, Any]:
        scores = self.predictor.predict_chunks(chunks)
        return {
            "risk_scores": scores,
            "predictions": [score >= self.threshold for score in scores],
            "model_manifest": self.predictor.manifest(),
        }
