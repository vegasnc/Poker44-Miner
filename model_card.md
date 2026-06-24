# Model Card: poker44-bot-detector

## Overview

This is a baseline LightGBM bot detection model for Poker44 Subnet 126. It scores benchmark or validator-provided poker hand chunk groups and returns one bot-risk probability per chunk group.

## Intended Use

The model is intended for Poker44 miner inference where the validator sends `DetectionSynapse(chunks=...)`. It should not be used as a player disciplinary system or identity-level fraud detector without additional validation.

## Training Data

The training pipeline uses only official public Poker44 benchmark releases from `https://api.poker44.net/api/v1/benchmark`. Labels are consumed only from the benchmark `groundTruth` array and are not expected at production inference time.

## Features

Feature groups include action rates by street, aggression metrics, bet sizing distributions, positional behavior, VPIP, PFR, 3-bet frequency, steal attempts, showdown percentage, win rate, player count, stack statistics, and pot growth.

Identifiers such as `chunkId`, `chunkHash`, dates, and hand IDs are intentionally excluded from model features.

## Model

The baseline is LightGBM binary classification. The code includes an sklearn gradient boosting fallback for local environments that have not installed LightGBM, but production training should use LightGBM.

## Evaluation

Primary metric: ROC AUC.

Secondary metrics: average precision, log loss, Brier score, and per-release metrics. Use held-out release dates to estimate generalization to unseen data.

## Limitations

The public benchmark is a development surface and may not perfectly match live validator data. Feature quality and date-separated validation matter more than fitting one release tightly. Timing features are disabled by default because timing fields may be absent.
