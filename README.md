# Poker44 Bot Detection Miner

Production-ready baseline miner for Bittensor Subnet 126 (Poker44). It trains a chunk-level bot detector on the public Poker44 benchmark and returns one risk score per miner-visible chunk group.

Scores are probabilities in `[0, 1]`: values closer to `1` are more bot-like and values closer to `0` are more human-like.

## What Is Included

- Benchmark API loader with JSON caching.
- Robust feature pipeline for action patterns, bet sizing, aggression, position, consistency, showdown, VPIP, PFR, 3-bet, and steal metrics.
- LightGBM baseline model with a local sklearn fallback for development environments.
- Date-aware training split to reduce overfitting to a single release.
- Batch inference pipeline, local miner adapter, and production `neurons/miner.py` entrypoint.
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

## Run As A Live Poker44 Miner

The production entrypoint is `neurons/miner.py`, aligned with the official Poker44 subnet [`neurons` folder](https://github.com/Poker44/Poker44-subnet/tree/main/neurons).

First install the official Poker44 subnet runtime so `poker44.base.miner`, `poker44.validator.synapse`, and `bittensor` are importable. Then train or place model artifacts at `models/saved/model.joblib` and `models/saved/feature_names.json`.

Direct run:

```bash
python neurons/miner.py \
  --netuid 126 \
  --wallet.name "$WALLET_NAME" \
  --wallet.hotkey "$HOTKEY" \
  --subtensor.network finney \
  --axon.port "$AXON_PORT" \
  --blacklist.allowed_validator_hotkeys $ALLOWED_VALIDATOR_HOTKEYS
```

Or use the helper:

```bash
WALLET_NAME=my_cold \
HOTKEY=my_poker44_hotkey \
AXON_PORT=8091 \
ALLOWED_VALIDATOR_HOTKEYS="validator_hotkey_1 validator_hotkey_2" \
bash scripts/run_live_miner.sh
```

`neurons/miner.py` receives `DetectionSynapse(chunks=...)`, calls the local model-backed predictor, and returns `risk_scores`, `predictions`, and `model_manifest`.

## Run With PM2

The project includes `ecosystem.config.js` for running the miner as a PM2 service. Runtime values are loaded from `.env`.

Current `.env` variables:

```bash
NETUID=126
WALLET_NAME=pierre
HOTKEY=pierre1hotkey
SUBTENSOR_NETWORK=finney
AXON_PORT=7091
AXON_EXTERNAL_IP=
AXON_EXTERNAL_PORT=
ALLOWED_VALIDATOR_HOTKEYS=
POKER44_LOG_DIR=logs
POKER44_LOG_LEVEL=INFO
POKER44_LOG_TEE_STDIO=1
```

```bash
pm2 start ecosystem.config.js
pm2 logs poker44-miner
pm2 status
```

You can also override settings for one PM2 start by setting shell variables:

```bash
WALLET_NAME=pierre \
HOTKEY=pierre1hotkey \
AXON_PORT=7091 \
SUBTENSOR_NETWORK=finney \
pm2 start ecosystem.config.js
```

For boot persistence:

```bash
pm2 save
pm2 startup
```

Common operations:

```bash
pm2 restart poker44-miner
pm2 stop poker44-miner
pm2 delete poker44-miner
```

Project logs are recorded under `logs/`:

```bash
logs/poker44-miner.log      # all Python, synapse, raw Axon HTTP, exceptions, mirrored stdout/stderr
logs/poker44-miner.out.log  # PM2 stdout capture
logs/poker44-miner.err.log  # PM2 stderr capture
```

Watch the unified miner log:

```bash
pm2 logs poker44-miner
tail -f logs/poker44-miner.log
```

## Library Integration

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
