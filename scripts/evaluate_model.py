#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import numpy as np

from src.data.loader import BenchmarkClient, iter_examples_from_records
from src.inference.predictor import BotRiskPredictor
from src.training.evaluator import summarize_evaluation
from src.utils.helpers import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved Poker44 bot detection model.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--source-date", required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    config = load_yaml(args.config, default={})
    client = BenchmarkClient(
        base_url=config.get("data", {}).get("base_url", "https://api.poker44.net/api/v1/benchmark"),
        cache_dir=config.get("data", {}).get("cache_dir", "data/cache"),
    )
    examples = iter_examples_from_records(client.chunks(args.source_date, limit=args.limit))
    predictor = BotRiskPredictor(
        model_path=config.get("inference", {}).get("model_path", "models/saved/model.joblib"),
        feature_names_path=config.get("inference", {}).get("feature_names_path", "models/saved/feature_names.json"),
        config_path=args.config,
    )
    scores = predictor.predict_chunks([example.chunk for example in examples])
    metrics = summarize_evaluation(
        np.asarray([example.label for example in examples], dtype=int),
        np.asarray(scores, dtype=float),
        [{"source_date": example.source_date} for example in examples],
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
