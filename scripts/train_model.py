#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from src.models.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Poker44 bot detection baseline.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--manifest", default="config/manifest.json")
    parser.add_argument("--source-date", action="append", dest="source_dates")
    parser.add_argument("--limit-per-day", type=int, default=None)
    args = parser.parse_args()

    result = train_model(
        config_path=args.config,
        manifest_path=args.manifest,
        source_dates=args.source_dates,
        limit_per_day=args.limit_per_day,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
