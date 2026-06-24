#!/usr/bin/env python3
from __future__ import annotations

import argparse

from src.data.loader import BenchmarkClient
from src.utils.helpers import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and cache Poker44 benchmark releases.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--limit", type=int, default=5, help="Number of recent releases to inspect.")
    parser.add_argument("--chunks-per-day", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config, default={})
    data_config = config.get("data", {})
    client = BenchmarkClient(
        base_url=data_config.get("base_url", "https://api.poker44.net/api/v1/benchmark"),
        cache_dir=data_config.get("cache_dir", "data/cache"),
    )
    releases = client.releases(limit=args.limit, use_cache=False)
    limit = args.chunks_per_day or int(data_config.get("max_chunks_per_day", 100))
    for release in releases:
        source_date = release["sourceDate"]
        records = client.chunks(source_date=source_date, limit=limit, use_cache=False)
        print(f"{source_date}: cached {len(records)} benchmark records")


if __name__ == "__main__":
    main()
