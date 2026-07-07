#!/usr/bin/env bash
set -euo pipefail
cd /home/poker44

# Update end_date to today so training always uses fresh data
TODAY=$(date -u +%Y-%m-%d)
sed -i "s/end_date: \"[0-9-]*\"/end_date: \"${TODAY}\"/" config/config.yaml

# Download new data, train, restart miner
/home/poker44/venv/bin/python scripts/train_model.py >> logs/retrain.log 2>&1
pm2 restart poker44-miner poker44-miner2 --update-env
