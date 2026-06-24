from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from src.data.loader import BenchmarkClient, DatasetExample, load_benchmark_examples
from src.features.engineering import FeaturePipeline
from src.features.per_hand import extract_hand_matrix
from src.models.baseline import LightGBMBotDetector
from src.models.ensemble import StackedEnsemble
from src.training.evaluator import summarize_evaluation
from src.utils.helpers import load_json, load_yaml, save_json, sha256_file, sha256_files


class TrainingPipeline:
    def __init__(self, config_path: str = "config/config.yaml", manifest_path: str = "config/manifest.json") -> None:
        self.config_path = Path(config_path)
        self.manifest_path = Path(manifest_path)
        self.config = load_yaml(self.config_path, default={})

    def run(self, source_dates: list[str] | None = None, limit_per_day: int | None = None) -> dict[str, Any]:
        data_config = self.config.get("data", {})
        training_config = self.config.get("training", {})
        model_config = self.config.get("model", {})
        inference_config = self.config.get("inference", {})

        client = BenchmarkClient(
            base_url=data_config.get("base_url", "https://api.poker44.net/api/v1/benchmark"),
            cache_dir=data_config.get("cache_dir", "data/cache"),
            timeout=inference_config.get("timeout_seconds", 30),
            max_retries=inference_config.get("max_retries", 3),
        )
        dates = source_dates or self._latest_dates(client, int(training_config.get("min_releases", 3)))
        limit = int(limit_per_day or data_config.get("max_chunks_per_day", 100))
        examples = load_benchmark_examples(dates, client, limit_per_day=limit)
        if not examples:
            raise ValueError("No benchmark examples were loaded")

        train_examples, valid_examples, test_examples = split_by_release(examples)
        feature_pipeline = FeaturePipeline(config=self.config)
        x_train = feature_pipeline.fit_transform([example.chunk for example in train_examples])
        y_train = np.asarray([example.label for example in train_examples], dtype=int)
        x_valid = feature_pipeline.transform([example.chunk for example in valid_examples]) if valid_examples else None
        y_valid = np.asarray([example.label for example in valid_examples], dtype=int) if valid_examples else None
        x_test = feature_pipeline.transform([example.chunk for example in test_examples])
        y_test = np.asarray([example.label for example in test_examples], dtype=int)

        # Per-hand matrices for Set Transformer
        mats_train = _extract_matrices(train_examples)
        mats_valid = _extract_matrices(valid_examples) if valid_examples else None
        mats_test = _extract_matrices(test_examples)

        seed = int(training_config.get("random_seed", 42))
        model = _build_model(model_config, seed)
        model.fit(x_train, y_train, x_valid, y_valid,
                  hand_matrices_train=mats_train, hand_matrices_valid=mats_valid)
        test_scores = model.predict_proba(x_test, hand_matrices=mats_test)
        metrics = summarize_evaluation(y_test, test_scores, [_metadata(example) for example in test_examples])

        # Final production model retrained on all available data
        final_feature_pipeline = FeaturePipeline(config=self.config)
        x_all = final_feature_pipeline.fit_transform([example.chunk for example in examples])
        y_all = np.asarray([example.label for example in examples], dtype=int)

        # Prune zero-importance features before final training
        x_all, final_feature_pipeline = _prune_zero_importance(x_all, y_all, final_feature_pipeline, model_config, seed)

        mats_all = _extract_matrices(examples)
        final_model = _build_model(model_config, seed)
        final_model.fit(x_all, y_all, hand_matrices_train=mats_all)

        model_path = Path(inference_config.get("model_path", "models/saved/model.joblib"))
        feature_names_path = Path(inference_config.get("feature_names_path", "models/saved/feature_names.json"))
        metrics_path = model_path.parent / "metrics.json"
        importances_path = model_path.parent / "feature_importance.json"

        final_model.save(model_path)
        final_feature_pipeline.save_feature_names(feature_names_path)
        save_json(metrics_path, metrics)
        feat_names = final_feature_pipeline.feature_names or []
        if hasattr(final_model, "feature_importance"):
            save_json(importances_path, final_model.feature_importance(feat_names))
        self._update_manifest(model_path, dates)

        return {
            "dates": dates,
            "train_examples": len(train_examples),
            "validation_examples": len(valid_examples),
            "test_examples": len(test_examples),
            "model_path": str(model_path),
            "feature_names_path": str(feature_names_path),
            "metrics": metrics,
        }

    def _latest_dates(self, client: BenchmarkClient, minimum: int) -> list[str]:
        data_config = self.config.get("data", {})
        start_date = data_config.get("start_date")
        end_date = data_config.get("end_date")
        releases = client.releases(limit=120)
        dates = [release["sourceDate"] for release in releases if release.get("sourceDate")]
        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]
        dates = sorted(dates)
        if len(dates) < minimum:
            raise ValueError(f"Need at least {minimum} benchmark releases, found {len(dates)}")
        return dates

    def _update_manifest(self, model_path: Path, dates: list[str]) -> None:
        manifest = load_json(self.manifest_path, default={})
        implementation_files = manifest.get("implementation_files") or [
            "neurons/miner.py",
            "miners/custom_miner.py",
            "src/inference/predictor.py",
            "src/inference/pipeline.py",
            "src/features/engineering.py",
            "src/models/baseline.py",
            "src/models/ensemble.py",
        ]
        manifest["repo_commit"] = _git_commit()
        manifest["artifact_sha256"] = sha256_file(model_path)
        manifest["implementation_sha256"] = sha256_files(implementation_files)
        manifest["training_data_statement"] = (
            f"Trained on Poker44 public benchmark releases from {min(dates)} to {max(dates)}"
        )
        manifest["training_data_sources"] = [
            f"https://api.poker44.net/api/v1/benchmark/chunks?sourceDate={date}" for date in dates
        ]
        save_json(self.manifest_path, manifest)


def _extract_matrices(examples: list[DatasetExample]) -> list[np.ndarray]:
    """Extract per-hand feature matrices from a list of DatasetExamples."""
    from src.data.preprocessor import normalize_chunk_group
    return [extract_hand_matrix(normalize_chunk_group(ex.chunk)) for ex in examples]


def _prune_zero_importance(
    x_all: Any,
    y_all: Any,
    pipeline: FeaturePipeline,
    model_config: dict[str, Any],
    seed: int,
) -> tuple[Any, FeaturePipeline]:
    """Remove features with zero importance using a quick LightGBM probe."""
    import logging
    from lightgbm import LGBMClassifier

    probe = LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05, verbose=-1, random_state=seed)
    probe.fit(x_all, y_all)
    importances = probe.feature_importances_
    feat_names = list(x_all.columns) if hasattr(x_all, "columns") else (pipeline.feature_names or [])
    keep = [name for name, imp in zip(feat_names, importances) if imp > 0]
    if len(keep) < 10:
        return x_all, pipeline
    pruned_pipeline = FeaturePipeline(feature_names=keep, config=pipeline.config)
    x_pruned = x_all[keep] if hasattr(x_all, "__getitem__") else x_all
    logging.getLogger(__name__).info(
        "Feature pruning: %d → %d features (removed %d zero-importance)",
        len(feat_names), len(keep), len(feat_names) - len(keep),
    )
    return x_pruned, pruned_pipeline


def _build_model(model_config: dict[str, Any], seed: int) -> Any:
    model_type = model_config.get("type", "lightgbm")
    if model_type == "stacked_ensemble":
        ensemble_cfg = model_config.get("ensemble", {})
        return StackedEnsemble(
            lgbm_params=model_config.get("lgbm_params", {}),
            xgb_params=model_config.get("xgb_params", {}),
            catboost_params=model_config.get("catboost_params", {}),
            set_transformer_params=model_config.get("set_transformer_params", {}),
            n_folds=int(ensemble_cfg.get("n_folds", 5)),
            random_seed=seed,
            calibrator_blend=float(ensemble_cfg.get("calibrator_blend", 0.7)),
            human_weight=float(ensemble_cfg.get("human_weight", 2.0)),
            target_fpr=float(ensemble_cfg.get("target_fpr", 0.04)),
            use_set_transformer=bool(ensemble_cfg.get("use_set_transformer", True)),
        )
    return LightGBMBotDetector(
        params=model_config.get("params", {}),
        random_seed=seed,
    )


def split_by_release(
    examples: list[DatasetExample],
) -> tuple[list[DatasetExample], list[DatasetExample], list[DatasetExample]]:
    dates = sorted({example.source_date for example in examples})
    if len(dates) >= 3:
        test_date = dates[-1]
        valid_date = dates[-2]
        train_dates = set(dates[:-2])
        train = [example for example in examples if example.source_date in train_dates]
        valid = [example for example in examples if example.source_date == valid_date]
        test = [example for example in examples if example.source_date == test_date]
        return train, valid, test
    if len(dates) == 2:
        train = [example for example in examples if example.source_date == dates[0]]
        test = [example for example in examples if example.source_date == dates[1]]
        return train, [], test
    midpoint = max(1, int(len(examples) * 0.8))
    return examples[:midpoint], [], examples[midpoint:]


def _metadata(example: DatasetExample) -> dict[str, Any]:
    return {
        "source_date": example.source_date,
        "chunk_id": example.chunk_id,
        "chunk_hash": example.chunk_hash,
        "split": example.split,
    }


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "uncommitted"
