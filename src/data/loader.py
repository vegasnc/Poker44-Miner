from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from src.utils.helpers import ensure_dir, load_json, save_json


@dataclass(frozen=True)
class DatasetExample:
    chunk: list[dict[str, Any]]
    label: int
    source_date: str
    split: str | None = None
    chunk_id: str | None = None
    chunk_hash: str | None = None
    release_version: str | None = None
    schema_version: str | None = None


class BenchmarkClient:
    def __init__(
        self,
        base_url: str = "https://api.poker44.net/api/v1/benchmark",
        cache_dir: str | Path = "data/cache",
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_dir = ensure_dir(cache_dir)
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    def status(self, use_cache: bool = True) -> dict[str, Any]:
        return self._get_json(self.base_url, cache_name="status.json", use_cache=use_cache)

    def releases(self, limit: int = 30, before: str | None = None, use_cache: bool = True) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if before:
            params["before"] = before
        payload = self._get_json(f"{self.base_url}/releases", params=params, use_cache=use_cache)
        data = payload.get("data", payload)
        return list(data.get("releases", []))

    def chunks(
        self,
        source_date: str,
        limit: int = 100,
        split: str | None = None,
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"sourceDate": source_date, "limit": limit}
            if split:
                params["split"] = split
            if cursor:
                params["cursor"] = cursor
            cache_name = f"chunks_{source_date}_{split or 'all'}_{cursor or 'first'}_{limit}.json"
            payload = self._get_json(f"{self.base_url}/chunks", params=params, cache_name=cache_name, use_cache=use_cache)
            data = payload.get("data", payload)
            page_records = data.get("chunks", data if isinstance(data, list) else [])
            records.extend(page_records)
            cursor = data.get("nextCursor") or data.get("cursor") or data.get("next_cursor")
            if not cursor:
                break
        return records

    def _get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        cache_name: str | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        cache_file = self.cache_dir / (cache_name or self._cache_name(url, params))
        if use_cache:
            cached = load_json(cache_file)
            if cached is not None:
                return cached

        last_error: Exception | None = None
        for _ in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
                save_json(cache_file, payload)
                return payload
            except requests.RequestException as exc:
                last_error = exc
        raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error

    @staticmethod
    def _cache_name(url: str, params: dict[str, Any] | None) -> str:
        raw = json.dumps({"url": url, "params": params or {}}, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest() + ".json"


def iter_examples_from_records(records: list[dict[str, Any]]) -> list[DatasetExample]:
    examples: list[DatasetExample] = []
    for record in records:
        chunk_groups = record.get("chunks") or []
        labels = record.get("groundTruth") or record.get("ground_truth") or []
        if len(chunk_groups) != len(labels):
            raise ValueError(
                f"Label count mismatch for {record.get('chunkId')}: "
                f"{len(chunk_groups)} chunks vs {len(labels)} labels"
            )
        for index, chunk_group in enumerate(chunk_groups):
            examples.append(
                DatasetExample(
                    chunk=list(chunk_group or []),
                    label=int(labels[index]),
                    source_date=str(record.get("sourceDate") or ""),
                    split=record.get("split"),
                    chunk_id=record.get("chunkId"),
                    chunk_hash=record.get("chunkHash"),
                    release_version=record.get("releaseVersion"),
                    schema_version=record.get("schemaVersion"),
                )
            )
    return examples


def load_benchmark_examples(
    source_dates: list[str],
    client: BenchmarkClient,
    limit_per_day: int = 100,
    split: str | None = None,
) -> list[DatasetExample]:
    examples: list[DatasetExample] = []
    for source_date in source_dates:
        records = client.chunks(source_date=source_date, limit=limit_per_day, split=split)
        examples.extend(iter_examples_from_records(records))
    return examples
