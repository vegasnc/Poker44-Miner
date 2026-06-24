from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class LightGBMBotDetector:
    def __init__(self, params: dict[str, Any] | None = None, random_seed: int = 42) -> None:
        self.params = dict(params or {})
        self.random_seed = random_seed
        self.model: Any | None = None
        self.backend = "lightgbm"

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        x_valid: pd.DataFrame | None = None,
        y_valid: np.ndarray | None = None,
    ) -> "LightGBMBotDetector":
        try:
            from lightgbm import LGBMClassifier

            params = dict(self.params)
            early_stopping_rounds = int(params.pop("early_stopping_rounds", 0) or 0)
            params.setdefault("random_state", self.random_seed)
            self.model = LGBMClassifier(**params)
            fit_kwargs: dict[str, Any] = {}
            if x_valid is not None and y_valid is not None and len(set(y_valid.tolist())) > 1:
                fit_kwargs["eval_set"] = [(x_valid, y_valid)]
                fit_kwargs["eval_metric"] = params.get("metric", "auc")
                if early_stopping_rounds > 0:
                    from lightgbm import early_stopping, log_evaluation

                    fit_kwargs["callbacks"] = [
                        early_stopping(early_stopping_rounds, verbose=False),
                        log_evaluation(period=0),
                    ]
            self.model.fit(x_train, y_train, **fit_kwargs)
            self.backend = "lightgbm"
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier

            self.model = GradientBoostingClassifier(random_state=self.random_seed)
            self.model.fit(x_train, y_train)
            self.backend = "sklearn-gradient-boosting"
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not fitted")
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)
            if probabilities.shape[1] == 1:
                return probabilities[:, 0]
            return probabilities[:, 1]
        raw = self.model.predict(features)
        return 1.0 / (1.0 + np.exp(-raw))

    def feature_importance(self, feature_names: list[str]) -> list[dict[str, float | str]]:
        if self.model is None:
            return []
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            return []
        pairs = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)
        return [{"feature": name, "importance": float(score)} for name, score in pairs]

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise ValueError("Model is not fitted")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"backend": self.backend, "params": self.params, "model": self.model}, path)

    @classmethod
    def load(cls, path: str | Path) -> "LightGBMBotDetector":
        payload = joblib.load(path)
        detector = cls(params=payload.get("params", {}))
        detector.model = payload["model"]
        detector.backend = payload.get("backend", "lightgbm")
        return detector
