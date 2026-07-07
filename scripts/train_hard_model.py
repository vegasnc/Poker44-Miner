#!/usr/bin/env python3
"""Train the Hard-Case Model from hard_regime chunk data + benchmark data.

The Hard-Case Model is a SEPARATE model saved to models/saved/model_hard.joblib.
It never modifies model.joblib (the General Model).

Usage:
    python scripts/train_hard_model.py                     # use all saved hard_regime batches
    python scripts/train_hard_model.py --min-humans 200    # wait until 200 hard humans collected
    python scripts/train_hard_model.py --dry-run           # report data stats without training
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

os.chdir(Path(__file__).resolve().parents[1])
sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hard_model")


def load_hard_regime_examples(hard_dir: str = "data/hard_regime") -> tuple[list, list, list, list]:
    """Load raw chunks from hard_regime JSON files.

    Returns (human_chunks, bot_chunks, human_scores, bot_scores).
    Each chunk is a list of 100 hand dicts — ready for FeaturePipeline.
    Labels are from our model predictions (pseudo-labels).
    """
    from src.data.loader import DatasetExample

    human_examples, bot_examples = [], []
    human_scores, bot_scores = [], []

    files = sorted(glob.glob(f"{hard_dir}/*.json"))
    if not files:
        raise FileNotFoundError(f"No hard_regime files in {hard_dir}/")

    for fpath in files:
        with open(fpath) as f:
            d = json.load(f)
        for record in d["chunks"]:
            label = record["label"]
            chunk = record["chunk"]
            score = record["score"]
            if not chunk:
                continue
            ex = DatasetExample(chunk=chunk, label=label, source_date="hard_regime", split="train")
            if label == 0:
                human_examples.append(ex)
                human_scores.append(score)
            else:
                bot_examples.append(ex)
                bot_scores.append(score)

    return human_examples, bot_examples, human_scores, bot_scores


def print_data_report(human_ex, bot_ex, human_scores, bot_scores):
    log.info("=== HARD REGIME DATA REPORT ===")
    log.info(f"  Hard humans : {len(human_ex)} chunks | score range {min(human_scores):.3f}–{max(human_scores):.3f} | mean {np.mean(human_scores):.3f}")
    log.info(f"  Hard bots   : {len(bot_ex)} chunks | score range {min(bot_scores):.3f}–{max(bot_scores):.3f} | mean {np.mean(bot_scores):.3f}")
    log.info(f"  Gap         : {min(bot_scores) - max(human_scores):.3f}")
    hands_h = [len(e.chunk) for e in human_ex]
    hands_b = [len(e.chunk) for e in bot_ex]
    log.info(f"  Hands/chunk : humans {min(hands_h)}–{max(hands_h)} | bots {min(hands_b)}–{max(hands_b)}")


def train(args):
    from src.data.loader import BenchmarkClient, load_benchmark_examples
    from src.features.engineering import FeaturePipeline
    from src.features.per_hand import extract_hand_matrix
    from src.data.preprocessor import normalize_chunk_group
    from src.models.ensemble import StackedEnsemble
    from src.training.trainer import _augment_concatenated_chunks, split_by_release
    from src.utils.helpers import load_yaml, save_json, sha256_file

    import yaml

    # ── 1. Load hard_regime data ──────────────────────────────────────────────
    log.info("Loading hard_regime chunks …")
    human_ex, bot_ex, human_scores, bot_scores = load_hard_regime_examples(args.hard_dir)
    print_data_report(human_ex, bot_ex, human_scores, bot_scores)

    if len(human_ex) < args.min_humans:
        log.error(f"Only {len(human_ex)} hard humans — need {args.min_humans}. Run again later.")
        sys.exit(1)

    if args.dry_run:
        log.info("--dry-run: stopping here, no training performed.")
        return

    # ── 2. Load benchmark data (adds variety, especially for bots) ────────────
    log.info("Loading benchmark data …")
    config = load_yaml("config/config.yaml", default={})
    data_cfg = config.get("data", {})
    client = BenchmarkClient(
        base_url=data_cfg.get("base_url", "https://api.poker44.net/api/v1/benchmark"),
        cache_dir=data_cfg.get("cache_dir", "data/cache"),
    )

    # Use last 7 benchmark dates
    releases = client.releases(limit=30)
    dates = sorted({r["sourceDate"] for r in releases if r.get("sourceDate")})[-7:]
    bench_examples = load_benchmark_examples(dates, client, limit_per_day=48)
    log.info(f"Benchmark: {len(bench_examples)} examples from {dates[0]} to {dates[-1]}")

    # ── 3. Augment benchmark data to 100-hand chunks ──────────────────────────
    bench_aug = _augment_concatenated_chunks(
        list(bench_examples),
        n_per_class=300,
        target_hands=100,
        seed=42,
    )
    log.info(f"Benchmark augmented: +{len(bench_aug)} 100-hand chunks")

    # ── 4. Combine: hard_regime (primary) + benchmark (secondary) ────────────
    # Hard humans are capped at their actual count to avoid over-representing
    # pseudo-labeled data. Bots: use hard_regime bots + benchmark bots.
    hard_humans_capped = human_ex[:min(len(human_ex), 300)]
    hard_bots_capped   = bot_ex[:min(len(bot_ex), 500)]

    all_examples = hard_humans_capped + hard_bots_capped + bench_aug
    log.info(
        f"Training set: {len(hard_humans_capped)} hard-humans + "
        f"{len(hard_bots_capped)} hard-bots + {len(bench_aug)} bench-aug = "
        f"{len(all_examples)} total"
    )

    # ── 5. Build features ──────────────────────────────────────────────────────
    log.info("Extracting features …")
    fp = FeaturePipeline(config=config)
    X = fp.fit_transform([e.chunk for e in all_examples])
    y = np.array([e.label for e in all_examples])
    mats = [extract_hand_matrix(normalize_chunk_group(e.chunk)) for e in all_examples]
    log.info(f"Feature matrix: {X.shape}")

    # ── 6. Train Hard-Case Model ───────────────────────────────────────────────
    # Key difference from General Model:
    #   human_weight = 20.0  (was 10.0) — more aggressively push hard humans toward 0
    #   target_fpr   = 0.0   (same)
    model_cfg = config.get("model", {})
    ensemble_cfg = model_cfg.get("ensemble", {})

    log.info("Training Hard-Case StackedEnsemble (human_weight=20, target_fpr=0.0) …")
    model = StackedEnsemble(
        lgbm_params=model_cfg.get("lgbm_params", {}),
        xgb_params=model_cfg.get("xgb_params", {}),
        catboost_params=model_cfg.get("catboost_params", {}),
        set_transformer_params=model_cfg.get("set_transformer_params", {}),
        n_folds=int(ensemble_cfg.get("n_folds", 5)),
        random_seed=42,
        calibrator_blend=float(ensemble_cfg.get("calibrator_blend", 0.7)),
        human_weight=20.0,
        target_fpr=0.0,
        use_set_transformer=bool(ensemble_cfg.get("use_set_transformer", True)),
    )
    model.fit(X, y, hand_matrices_train=mats)

    # ── 7. Quick evaluation on hard humans ────────────────────────────────────
    log.info("Evaluating on hard regime data …")
    hard_X = fp.transform([e.chunk for e in hard_humans_capped + hard_bots_capped])
    hard_y = np.array([e.label for e in hard_humans_capped + hard_bots_capped])
    hard_mats = [extract_hand_matrix(normalize_chunk_group(e.chunk)) for e in hard_humans_capped + hard_bots_capped]
    hard_scores = model.predict_proba(hard_X, hand_matrices=hard_mats)

    h_scores = [s for s, l in zip(hard_scores, hard_y) if l == 0]
    b_scores = [s for s, l in zip(hard_scores, hard_y) if l == 1]
    log.info(f"  Hard humans  : min={min(h_scores):.3f}  max={max(h_scores):.3f}  mean={np.mean(h_scores):.3f}")
    log.info(f"  Hard bots    : min={min(b_scores):.3f}  max={max(b_scores):.3f}  mean={np.mean(b_scores):.3f}")
    log.info(f"  Gap          : {min(b_scores) - max(h_scores):.3f}  (was {min(bot_scores) - max(human_scores):.3f} before training)")
    fpr = sum(1 for s, l in zip(hard_scores, hard_y) if s >= 0.5 and l == 0) / max(sum(hard_y == 0), 1)
    log.info(f"  FPR (thr=0.5): {fpr:.3f}")

    # ── 8. Save to separate file — never touches model.joblib ─────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    fp.save_feature_names(out_path.parent / "feature_names_hard.json")

    metrics = {
        "hard_humans": len(h_scores),
        "hard_bots": len(b_scores),
        "human_score_min": round(min(h_scores), 4),
        "human_score_max": round(max(h_scores), 4),
        "human_score_mean": round(float(np.mean(h_scores)), 4),
        "bot_score_min": round(min(b_scores), 4),
        "gap": round(min(b_scores) - max(h_scores), 4),
        "fpr_at_0.5": round(fpr, 4),
        "training_examples": len(all_examples),
    }
    save_json(out_path.parent / "metrics_hard.json", metrics)

    log.info(f"Hard-Case Model saved → {out_path}")
    log.info(f"General Model (model.joblib) was NOT modified.")
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Train Hard-Case Model for UID 3 batches.")
    parser.add_argument("--hard-dir",    default="data/hard_regime",       help="Directory with hard_regime JSON files")
    parser.add_argument("--output",      default="models/saved/model_hard.joblib", help="Output path (never overwrites model.joblib)")
    parser.add_argument("--min-humans",  type=int, default=50,             help="Minimum hard human chunks required")
    parser.add_argument("--dry-run",     action="store_true",              help="Report data stats without training")
    args = parser.parse_args()

    if Path(args.output).name == "model.joblib":
        print("ERROR: output cannot be model.joblib — that is the General Model.")
        sys.exit(1)

    train(args)


if __name__ == "__main__":
    main()
