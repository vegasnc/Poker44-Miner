from __future__ import annotations

from pathlib import Path

import numpy as np

from miners.custom_miner import Poker44BotDetectionMiner
from src.features.engineering import FeaturePipeline
from src.models.baseline import LightGBMBotDetector
from src.inference.predictor import BotRiskPredictor
from src.utils.helpers import save_json


def test_predictor_scores_one_value_per_chunk(tmp_path: Path, sample_chunk_group):
    chunks = [sample_chunk_group, [], sample_chunk_group, []]
    labels = np.asarray([1, 0, 1, 0], dtype=int)
    pipeline = FeaturePipeline()
    features = pipeline.fit_transform(chunks)

    model = LightGBMBotDetector(params={"n_estimators": 5})
    model.fit(features, labels)
    model_path = tmp_path / "model.joblib"
    features_path = tmp_path / "feature_names.json"
    config_path = tmp_path / "config.yaml"
    model.save(model_path)
    pipeline.save_feature_names(features_path)
    config_path.write_text("features:\n  action_patterns: true\n", encoding="utf-8")

    predictor = BotRiskPredictor(model_path=model_path, feature_names_path=features_path, config_path=config_path)
    scores = predictor.predict_chunks(chunks)

    assert len(scores) == len(chunks)
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_miner_forward_handles_missing_artifacts_with_neutral_scores(tmp_path: Path, sample_chunk_group):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "inference:\n"
        f"  model_path: '{tmp_path / 'missing.joblib'}'\n"
        f"  feature_names_path: '{tmp_path / 'missing_features.json'}'\n"
        "  fallback_score: 0.5\n",
        encoding="utf-8",
    )
    miner = Poker44BotDetectionMiner(config_path=str(config_path))
    synapse = {"chunks": [sample_chunk_group]}

    result = miner.forward(synapse)

    assert result["risk_scores"] == [0.5]
    assert result["predictions"] == [False]
