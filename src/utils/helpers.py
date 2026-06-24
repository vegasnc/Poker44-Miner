from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_json(path: str | Path, default: Any | None = None) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        return default
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: Any) -> None:
    file_path = Path(path)
    ensure_dir(file_path.parent)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_yaml(path: str | Path, default: Any | None = None) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        return default
    with file_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or default


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_files(paths: Iterable[str | Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(p) for p in paths):
        digest.update(str(path).encode("utf-8"))
        if path.exists():
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return default if denominator == 0 else numerator / denominator


def clamp_probability(value: float) -> float:
    if value != value:
        return 0.5
    return max(0.0, min(1.0, float(value)))


def flatten_dict(prefix: str, value: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, item in value.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(item, dict):
            flat.update(flatten_dict(name, item))
        elif isinstance(item, bool):
            flat[name] = float(item)
        elif isinstance(item, (int, float)) and item == item:
            flat[name] = float(item)
    return flat


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    required_fields = [
        "schema_version",
        "open_source",
        "repo_url",
        "repo_commit",
        "model_name",
        "model_version",
        "training_data_statement",
        "private_data_attestation",
        "implementation_files",
        "implementation_sha256",
    ]
    return [field for field in required_fields if not manifest.get(field)]
