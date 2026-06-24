from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.helpers import load_json, validate_manifest


def validate_scores(scores: list[float], expected_count: int) -> None:
    if len(scores) != expected_count:
        raise ValueError(f"Expected {expected_count} scores, got {len(scores)}")
    invalid = [score for score in scores if score < 0.0 or score > 1.0]
    if invalid:
        raise ValueError(f"Scores must be in [0, 1], got {invalid[:3]}")


def validate_manifest_file(path: str | Path) -> list[str]:
    manifest: dict[str, Any] = load_json(path, default={})
    return validate_manifest(manifest)
