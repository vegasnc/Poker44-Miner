# Poker44 Bot Detection Miner

Production-ready baseline miner for Bittensor Subnet 126 (Poker44). It trains a chunk-level bot detector on the public Poker44 benchmark and returns one risk score per miner-visible chunk group.

Scores are probabilities in `[0, 1]`: values closer to `1` are more bot-like and values closer to `0` are more human-like.

## What Is Included

- Benchmark API loader with JSON caching.
- Robust feature pipeline for action patterns, bet sizing, aggression, position, consistency, showdown, VPIP, PFR, 3-bet, and steal metrics.
- LightGBM baseline model with a local sklearn fallback for development environments.
- Date-aware training split to reduce overfitting to a single release.
- Batch inference pipeline and Poker44 miner adapter.
- Model manifest template and post-training hash updates.
- Unit tests for data parsing, features, model scoring, and miner integration.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Download Benchmark Data

```bash
python scripts/download_data.py --limit 5 --chunks-per-day 100
```

## Train

Use the latest configured releases:

```bash
python scripts/train_model.py
```

Or specify release dates explicitly:

```bash
python scripts/train_model.py \
  --source-date 2026-06-22 \
  --source-date 2026-06-23 \
  --source-date 2026-06-24
```

Training writes:

- `models/saved/model.joblib`
- `models/saved/feature_names.json`
- `models/saved/metrics.json`
- `models/saved/feature_importance.json`
- updated `config/manifest.json`

## Evaluate

```bash
python scripts/evaluate_model.py --source-date 2026-06-24 --limit 100
```

## Run Local Inference

Pass a JSON list of chunk groups, or an object with a top-level `chunks` field:

```bash
python scripts/run_miner.py --chunks-json sample_chunks.json
```

## Miner Integration

Import `Poker44BotDetectionMiner` from `miners/custom_miner.py` and call `forward(synapse)`. The adapter accepts either a `DetectionSynapse`-like object with a `chunks` attribute or a dictionary with a `chunks` key.

It returns/mutates:

- `risk_scores`: one score per input chunk group.
- `predictions`: boolean predictions using the configured threshold.
- `model_manifest`: transparency metadata from `config/manifest.json`.

## Transparency

Before production use, update `config/manifest.json` with your public repository URL, artifact URL, model card URL, and a real public commit. The training script fills in artifact and implementation hashes automatically after a model is trained.

## Tests

```bash
pytest
```
