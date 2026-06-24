from __future__ import annotations

from typing import Any

from src.training.trainer import TrainingPipeline


def train_model(
    config_path: str = "config/config.yaml",
    manifest_path: str = "config/manifest.json",
    source_dates: list[str] | None = None,
    limit_per_day: int | None = None,
) -> dict[str, Any]:
    return TrainingPipeline(config_path=config_path, manifest_path=manifest_path).run(
        source_dates=source_dates,
        limit_per_day=limit_per_day,
    )
