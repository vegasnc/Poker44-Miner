#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from miners.custom_miner import score_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a local JSON file of Poker44 chunk groups.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--chunks-json", required=True, help="Path to JSON list of chunk groups or {'chunks': [...]} payload.")
    args = parser.parse_args()

    payload = json.loads(Path(args.chunks_json).read_text(encoding="utf-8"))
    chunks = payload.get("chunks", payload) if isinstance(payload, dict) else payload
    result = score_chunks(chunks, config_path=args.config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
