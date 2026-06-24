#!/usr/bin/env bash
set -euo pipefail

python neurons/miner.py \
  --netuid "${NETUID:-126}" \
  --wallet.name "${WALLET_NAME:?Set WALLET_NAME}" \
  --wallet.hotkey "${HOTKEY:?Set HOTKEY}" \
  --subtensor.network "${SUBTENSOR_NETWORK:-finney}" \
  --axon.port "${AXON_PORT:?Set AXON_PORT}"
