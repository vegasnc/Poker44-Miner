from __future__ import annotations

from src.features.engineering import FeaturePipeline


def test_feature_pipeline_extracts_expected_core_features(sample_chunk_group):
    pipeline = FeaturePipeline()
    matrix = pipeline.fit_transform([sample_chunk_group])

    assert matrix.shape[0] == 1
    assert matrix.loc[0, "hand_count"] == 1.0
    assert matrix.loc[0, "raise_rate"] > 0.0
    assert matrix.loc[0, "aggression_frequency"] > 0.0
    assert matrix.loc[0, "vpip"] == 1.0
    assert matrix.loc[0, "line_cbet_rate"] == 1.0
    assert matrix.loc[0, "bet_bucket_half_pot_rate"] > 0.0
    assert list(matrix.columns) == sorted(matrix.columns)


def test_empty_chunk_is_supported():
    pipeline = FeaturePipeline()
    features = pipeline.fit_transform([[]])

    assert features.loc[0, "hand_count"] == 0.0
    assert features.loc[0, "action_count"] == 0.0
