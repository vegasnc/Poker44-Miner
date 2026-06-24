from __future__ import annotations

import logging
from typing import Any

from src.inference.pipeline import InferencePipeline
from src.inference.predictor import BotRiskPredictor
from src.utils.helpers import clamp_probability, load_yaml

LOGGER = logging.getLogger(__name__)


class Poker44BotDetectionMiner:
    """Chunk-level miner adapter for Poker44 DetectionSynapse payloads."""

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        config = load_yaml(config_path, default={})
        inference_config = config.get("inference", {})
        predictor = BotRiskPredictor(
            model_path=inference_config.get("model_path", "models/saved/model.joblib"),
            feature_names_path=inference_config.get("feature_names_path", "models/saved/feature_names.json"),
            config_path=config_path,
            fallback_score=float(inference_config.get("fallback_score", 0.5)),
        )
        self.pipeline = InferencePipeline(
            predictor=predictor,
            threshold=float(inference_config.get("prediction_threshold", 0.5)),
        )

    def forward(self, synapse: Any) -> Any:
        chunks = getattr(synapse, "chunks", None)
        if chunks is None and isinstance(synapse, dict):
            chunks = synapse.get("chunks")
        chunks = chunks or []

        try:
            result = self.pipeline.score_synapse_chunks(chunks)
        except Exception as exc:
            LOGGER.exception("Poker44 inference failed; returning neutral scores: %s", exc)
            result = {
                "risk_scores": [0.5] * len(chunks),
                "predictions": [False] * len(chunks),
                "model_manifest": {},
            }

        result["risk_scores"] = [clamp_probability(score) for score in result["risk_scores"]]
        if isinstance(synapse, dict):
            synapse.update(result)
            return synapse
        for key, value in result.items():
            setattr(synapse, key, value)
        return synapse


def score_chunks(chunks: list[Any], config_path: str = "config/config.yaml") -> dict[str, Any]:
    return Poker44BotDetectionMiner(config_path=config_path).pipeline.score_synapse_chunks(chunks)
