from __future__ import annotations

import numpy as np

from src.features.engineering import FeaturePipeline
from src.models.baseline import LightGBMBotDetector


def test_model_prediction_shape_and_range(sample_chunk_group):
    chunks = [sample_chunk_group, [], sample_chunk_group, []]
    labels = np.asarray([1, 0, 1, 0], dtype=int)
    features = FeaturePipeline().fit_transform(chunks)

    model = LightGBMBotDetector(params={"n_estimators": 5, "learning_rate": 0.1})
    model.fit(features, labels)
    scores = model.predict_proba(features)

    assert scores.shape == (4,)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)
