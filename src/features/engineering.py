from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.preprocessor import normalize_chunk_group
from src.features.advanced import extract_advanced_features
from src.features.behavioral import (
    extract_behavioral_features,
    extract_bet_sizing_features,
    extract_timing_features,
)
from src.features.advanced_tells import extract_advanced_tell_features
from src.features.gto_tells import extract_gto_tell_features
from src.features.bot_tells import extract_bot_tell_features
from src.features.gto_engine_features import extract_gto_engine_features
from src.features.per_hand import extract_per_hand_features
from src.features.statistical import extract_statistical_features
from src.utils.helpers import load_json, save_json


class FeaturePipeline:
    def __init__(self, feature_names: list[str] | None = None, config: dict[str, Any] | None = None) -> None:
        self.feature_names = feature_names
        self.config = config or {}

    def extract_one(self, chunk_group: Any) -> dict[str, float]:
        normalized = normalize_chunk_group(chunk_group)
        features: dict[str, float] = {}

        enabled = self.config.get("features", self.config)
        if enabled.get("action_patterns", True) or enabled.get("aggression_metrics", True):
            features.update(extract_behavioral_features(normalized))
        if enabled.get("bet_sizing", True) or enabled.get("consistency_metrics", True):
            features.update(extract_bet_sizing_features(normalized))
        if enabled.get("statistical_features", True) or enabled.get("positional_play", True):
            features.update(extract_statistical_features(normalized))
        if enabled.get("timing_patterns", False):
            features.update(extract_timing_features(normalized))
        if enabled.get("advanced_patterns", True):
            features.update(extract_advanced_features(normalized))
        if enabled.get("per_hand_aggregation", True):
            features.update(extract_per_hand_features(normalized))
        if enabled.get("bot_tell_features", True):
            features.update(extract_bot_tell_features(normalized))
        if enabled.get("advanced_tell_features", True):
            features.update(extract_advanced_tell_features(normalized))
        if enabled.get("gto_tell_features", True):
            features.update(extract_gto_tell_features(normalized))
        if enabled.get("gto_engine_features", True):
            features.update(extract_gto_engine_features(normalized))

        if not features:
            features["hand_count"] = float(len(normalized))

        # Normalize cumulative count features by hand count to create per-hand rates.
        # Live validator sends 100 hands/chunk vs 30-40 in benchmark — without this,
        # raw counts are ~2.5x higher in live data, causing everything to look "bot-like".
        # Normalize any feature containing "count" (cumulative totals), ending in "_hands"
        # (position/situation hand tallies), or matching other known cumulative patterns.
        # ph_* features are already per-hand aggregates and must NOT be normalized again.
        n_hands = float(max(1, len(normalized)))
        # Features that contain "count" but are already per-hand means/stds:
        # mean_player_count, std_street_count, etc. → skip via prefix check.
        # ph_* features are per-hand aggregates, mean_/std_ features are per-hand stats.
        _SKIP_PREFIXES = ("ph_", "mean_", "std_")
        _SKIP_EXACT = {"hand_count"}
        # ph_chunk_unique_raise_to is a per-chunk cumulative count despite the ph_ prefix
        _PH_EXCEPTIONS = {"ph_chunk_unique_raise_to"}
        # Additional cumulative patterns that don't contain "count" in their name
        _CUMULATIVE_PATTERNS = ("_hands", "_opportunities", "_bets_faced_by_villain",
                                "_postflop_faced_aggression")
        for key in list(features.keys()):
            if key in _SKIP_EXACT:
                continue
            if key in _PH_EXCEPTIONS or (
                "count" in key or any(p in key for p in _CUMULATIVE_PATTERNS)
            ):
                if not any(key.startswith(p) for p in _SKIP_PREFIXES) or key in _PH_EXCEPTIONS:
                    features[key] = features[key] / n_hands

        return {key: _finite(value) for key, value in features.items()}

    def transform(self, chunk_groups: list[Any], fit: bool = False) -> pd.DataFrame:
        rows = [self.extract_one(chunk_group) for chunk_group in chunk_groups]
        if fit or self.feature_names is None:
            names = sorted({name for row in rows for name in row})
            self.feature_names = names
        assert self.feature_names is not None
        matrix = [{name: row.get(name, 0.0) for name in self.feature_names} for row in rows]
        return pd.DataFrame(matrix, columns=self.feature_names, dtype=float)

    def fit_transform(self, chunk_groups: list[Any]) -> pd.DataFrame:
        return self.transform(chunk_groups, fit=True)

    def save_feature_names(self, path: str | Path) -> None:
        if self.feature_names is None:
            raise ValueError("Feature names are not fitted")
        save_json(path, self.feature_names)

    @classmethod
    def load(cls, path: str | Path, config: dict[str, Any] | None = None) -> "FeaturePipeline":
        names = load_json(path, default=[])
        return cls(feature_names=list(names or []), config=config)


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(number):
        return 0.0
    return number
