from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from src.features.engineering import FeaturePipeline
from src.utils.helpers import clamp_probability, load_json, load_yaml


class BotRiskPredictor:
    def __init__(
        self,
        model_path: str | Path = "models/saved/model.joblib",
        feature_names_path: str | Path = "models/saved/feature_names.json",
        config_path: str | Path = "config/config.yaml",
        fallback_score: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path)
        self.feature_names_path = Path(feature_names_path)
        self.config_path = Path(config_path)
        self.config = load_yaml(self.config_path, default={})
        self.fallback_score = float(fallback_score)
        self.model: Any | None = None
        self.features: FeaturePipeline | None = None

    def load(self) -> "BotRiskPredictor":
        if not self.model_path.exists() or not self.feature_names_path.exists():
            raise FileNotFoundError(
                f"Missing model artifacts: {self.model_path} and/or {self.feature_names_path}. "
                "Run scripts/train_model.py first."
            )
        artifact = joblib.load(self.model_path)
        # Support both StackedEnsemble (direct pickle) and LightGBMBotDetector (dict wrapper)
        if isinstance(artifact, dict):
            from src.models.baseline import LightGBMBotDetector
            detector = LightGBMBotDetector()
            detector.model = artifact["model"]
            detector.backend = artifact.get("backend", "lightgbm")
            self.model = detector
        else:
            self.model = artifact
        self.features = FeaturePipeline.load(self.feature_names_path, config=self.config)
        return self

    def predict_chunk(self, chunk_group: Any) -> float:
        return self.predict_chunks([chunk_group])[0]

    def predict_chunks(self, chunk_groups: list[Any]) -> list[float]:
        if not chunk_groups:
            return []
        if self.model is None or self.features is None:
            self.load()
        assert self.model is not None and self.features is not None
        try:
            from src.data.preprocessor import normalize_chunk_group
            from src.features.per_hand import extract_hand_matrix
            normalized = [normalize_chunk_group(cg) for cg in chunk_groups]
            matrix = self.features.transform(normalized)
            # Pass hand matrices for Set Transformer if model supports it
            hand_matrices = [extract_hand_matrix(ng) for ng in normalized]
            import inspect
            sig = inspect.signature(self.model.predict_proba)
            if "hand_matrices" in sig.parameters:
                scores = self.model.predict_proba(matrix, hand_matrices=hand_matrices)
            else:
                scores = self.model.predict_proba(matrix)
            return [clamp_probability(float(score)) for score in scores]
        except Exception:
            return [clamp_probability(self.fallback_score)] * len(chunk_groups)

    def manifest(self, manifest_path: str | Path = "config/manifest.json") -> dict[str, Any]:
        return load_json(manifest_path, default={}) or {}
