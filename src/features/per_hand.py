"""Per-hand feature extraction with chunk-level aggregation.

Key insight from top miners: compute features per individual hand, then aggregate
with full statistical moments (mean/std/min/max/p10/p25/p50/p75/p90). This captures
the DISTRIBUTION and CONSISTENCY of behavior across hands — bots are highly consistent,
humans are variable.
"""
from __future__ import annotations

from collections import Counter
from math import log2
from typing import Any

import numpy as np

from src.data.preprocessor import numeric
from src.utils.helpers import safe_div

AGGRESSIVE = {"bet", "raise", "allin"}
VOLUNTARY_PREFLOP = {"call", "bet", "raise", "allin"}
STANDARD_POT_FRACTIONS = [0.25, 0.33, 0.50, 0.67, 0.75, 1.00, 1.25, 1.50, 2.00]
SNAP_TOLERANCE = 0.05  # within 5% of standard fraction counts as "snapped"


def extract_per_hand_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    """Compute per-hand features then aggregate statistically across the chunk."""
    hero_seat = _hero_seat(chunk_group)
    per_hand: list[dict[str, float]] = [_hand_features(hand, hero_seat) for hand in chunk_group]
    return _aggregate(per_hand)


def extract_hand_matrix(chunk_group: list[dict[str, Any]]) -> np.ndarray:
    """Return (num_hands, num_features) matrix for the Set Transformer.

    Each row is one hand's feature vector in a consistent, fixed-order layout.
    Columns are sorted feature names from _hand_features() to ensure reproducibility.
    """
    hero_seat = _hero_seat(chunk_group)
    per_hand = [_hand_features(hand, hero_seat) for hand in chunk_group]
    if not per_hand:
        return np.zeros((1, 33), dtype=np.float32)
    keys = sorted(per_hand[0].keys())
    matrix = np.array([[h.get(k, 0.0) for k in keys] for h in per_hand], dtype=np.float32)
    # Replace non-finite values with 0
    matrix = np.where(np.isfinite(matrix), matrix, 0.0)
    return matrix


# Expose the feature key order so the Set Transformer knows input_dim at init time
HAND_FEATURE_KEYS: list[str] = []  # populated lazily on first call

def hand_feature_dim() -> int:
    """Return the number of features per hand (input_dim for Set Transformer)."""
    from src.data.preprocessor import normalize_chunk_group
    if not HAND_FEATURE_KEYS:
        dummy = [{"actions": [], "metadata": {}, "players": [], "outcome": {}}]
        sample = _hand_features(dummy[0], None)
        HAND_FEATURE_KEYS.extend(sorted(sample.keys()))
    return len(HAND_FEATURE_KEYS)


def _hero_seat(chunk_group: list[dict[str, Any]]) -> int | None:
    for hand in chunk_group:
        meta = hand.get("metadata", {})
        if isinstance(meta, dict) and meta.get("hero_seat") is not None:
            try:
                return int(meta["hero_seat"])
            except (TypeError, ValueError):
                pass
        for player in (hand.get("players") or []):
            if isinstance(player, dict) and (player.get("is_hero") or player.get("hero")):
                try:
                    return int(player.get("seat") or player.get("seat_id") or 0)
                except (TypeError, ValueError):
                    pass
    return None


def _hand_features(hand: dict[str, Any], hero_seat: int | None) -> dict[str, float]:
    actions = hand.get("actions") or []
    hero_actions = [a for a in actions if a.get("actor_seat") == hero_seat] if hero_seat is not None else []

    total = len(actions)
    hero_total = len(hero_actions)

    action_types = [str(a.get("action_type", "unknown")) for a in actions]
    hero_types = [str(a.get("action_type", "unknown")) for a in hero_actions]
    counts = Counter(action_types)
    hero_counts = Counter(hero_types)

    # Hero bet sizing analysis
    hero_bets = [a for a in hero_actions if a.get("action_type") in AGGRESSIVE]
    hero_bet_ratios = [
        safe_div(numeric(a.get("amount")), numeric(a.get("pot_before")))
        for a in hero_bets if numeric(a.get("pot_before")) > 0 and a.get("action_type") != "allin"
    ]
    hero_bb_amounts = [
        numeric(a.get("normalized_amount_bb"))
        for a in hero_bets if numeric(a.get("normalized_amount_bb")) > 0
    ]

    # Snap-to-fraction: bots bet exact standard fractions, humans vary
    snap_count = sum(
        1 for ratio in hero_bet_ratios
        if any(abs(ratio - frac) <= SNAP_TOLERANCE * frac for frac in STANDARD_POT_FRACTIONS)
    )

    # Unique bet sizes: bots reuse same amounts
    hero_raise_to_values = [
        round(numeric(a.get("raise_to")), 4) for a in hero_actions
        if a.get("raise_to") and numeric(a.get("raise_to")) > 0
    ]

    # Actor patterns
    actors = [a.get("actor_seat") for a in actions]
    actor_switches = sum(1 for i in range(1, len(actors)) if actors[i] != actors[i - 1])
    actor_set = {x for x in actors if x is not None}

    # Pot dynamics
    pots = [numeric(a.get("pot_before")) for a in actions if numeric(a.get("pot_before")) > 0]
    pot_increases = sum(1 for i in range(1, len(pots)) if pots[i] >= pots[i - 1]) if len(pots) > 1 else 0
    pot_start = pots[0] if pots else 0.0
    pot_end = numeric(actions[-1].get("pot_after")) if actions else 0.0

    # Street reach
    streets_seen = {str(a.get("street", "")) for a in actions if a.get("street")}
    streets_seen.discard("unknown")

    # Hero preflop behaviour
    hero_preflop = [a for a in hero_actions if a.get("street") == "preflop"]
    hero_vpip = float(any(a.get("action_type") in VOLUNTARY_PREFLOP for a in hero_preflop))
    hero_pfr = float(any(a.get("action_type") in {"raise", "allin"} for a in hero_preflop))

    # Facing aggression: when villain bets/raises, does hero fold/call/raise?
    villain_agg_count = sum(
        1 for a in actions
        if a.get("actor_seat") != hero_seat and a.get("action_type") in AGGRESSIVE
    )
    hero_fold_vs_agg = float(
        villain_agg_count > 0 and any(a.get("action_type") == "fold" for a in hero_actions)
    )

    return {
        "action_count": float(total),
        "hero_action_count": float(hero_total),
        "hero_action_share": safe_div(hero_total, total),
        "player_count": float(len({p.get("seat") for p in (hand.get("players") or [])} - {None})),
        "street_count": float(len(streets_seen)),

        # Action composition
        "action_entropy": _entropy(Counter(action_types)),
        "actor_entropy": _entropy(Counter(str(x) for x in actors if x is not None)),
        "aggression_rate": safe_div(sum(counts.get(a, 0) for a in AGGRESSIVE), total),
        "fold_rate": safe_div(counts.get("fold", 0), total),
        "call_rate": safe_div(counts.get("call", 0), total),
        "check_rate": safe_div(counts.get("check", 0), total),

        # Hero-specific action composition
        "hero_aggression_rate": safe_div(sum(hero_counts.get(a, 0) for a in AGGRESSIVE), hero_total),
        "hero_fold": float("fold" in hero_types),
        "hero_call": float("call" in hero_types),
        "hero_bet_count": float(sum(hero_counts.get(a, 0) for a in AGGRESSIVE)),
        "hero_vpip": hero_vpip,
        "hero_pfr": hero_pfr,
        "hero_fold_vs_aggression": hero_fold_vs_agg,

        # Hero bet sizing
        "hero_bet_ratio_mean": float(np.mean(hero_bet_ratios)) if hero_bet_ratios else 0.0,
        "hero_bet_ratio_std": float(np.std(hero_bet_ratios)) if len(hero_bet_ratios) > 1 else 0.0,
        "hero_bet_ratio_max": float(max(hero_bet_ratios)) if hero_bet_ratios else 0.0,
        "hero_bet_bb_mean": float(np.mean(hero_bb_amounts)) if hero_bb_amounts else 0.0,
        "hero_bet_bb_std": float(np.std(hero_bb_amounts)) if len(hero_bb_amounts) > 1 else 0.0,

        # Precision/quantization: bots snap to standard fractions
        "hero_bet_snap_rate": safe_div(snap_count, len(hero_bet_ratios)),
        "hero_unique_raise_to_count": float(len(set(hero_raise_to_values))),
        "hero_bet_count_nonzero": float(len(hero_bet_ratios)),

        # Actor dynamics
        "actor_switch_rate": safe_div(actor_switches, max(total - 1, 0)),
        "unique_actor_count": float(len(actor_set)),
        "actor_count_ratio": safe_div(len(actor_set), max(total, 1)),

        # Pot dynamics
        "pot_monotonicity": safe_div(pot_increases, max(len(pots) - 1, 0)),
        "pot_growth": max(0.0, pot_end - pot_start),
        "pot_growth_ratio": safe_div(pot_end - pot_start, max(pot_start, 1e-9)),

        # Amount nonzero
        "nonzero_amount_rate": safe_div(
            sum(1 for a in actions if numeric(a.get("amount")) > 0 or numeric(a.get("raise_to")) > 0),
            total,
        ),
    }


def _aggregate(per_hand: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-hand features with full statistical moments."""
    if not per_hand:
        return {}

    all_keys = list(per_hand[0].keys())
    result: dict[str, float] = {}

    for key in all_keys:
        values = np.array([h[key] for h in per_hand], dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            for stat in ("mean", "std", "min", "max", "p10", "p25", "p50", "p75", "p90"):
                result[f"ph_{key}_{stat}"] = 0.0
            continue

        result[f"ph_{key}_mean"] = float(np.mean(values))
        result[f"ph_{key}_std"] = float(np.std(values)) if len(values) > 1 else 0.0
        result[f"ph_{key}_min"] = float(np.min(values))
        result[f"ph_{key}_max"] = float(np.max(values))
        result[f"ph_{key}_p10"] = float(np.percentile(values, 10))
        result[f"ph_{key}_p25"] = float(np.percentile(values, 25))
        result[f"ph_{key}_p50"] = float(np.percentile(values, 50))
        result[f"ph_{key}_p75"] = float(np.percentile(values, 75))
        result[f"ph_{key}_p90"] = float(np.percentile(values, 90))

    # Cross-hand consistency metrics (autocorrelation proxies)
    for key in ("hero_bet_ratio_mean", "hero_bet_bb_mean", "hero_aggression_rate", "hero_action_share"):
        values = np.array([h[key] for h in per_hand], dtype=float)
        values = values[np.isfinite(values)]
        if len(values) > 2:
            # Coefficient of variation across hands — bots have near-zero CV
            result[f"ph_{key}_cv"] = safe_div(float(np.std(values)), abs(float(np.mean(values))))
            # First-order autocorrelation
            if len(values) > 3:
                try:
                    v1, v2 = values[:-1], values[1:]
                    if np.std(v1) > 1e-9 and np.std(v2) > 1e-9:
                        result[f"ph_{key}_autocorr"] = float(np.corrcoef(v1, v2)[0, 1])
                    else:
                        result[f"ph_{key}_autocorr"] = 0.0
                except Exception:
                    result[f"ph_{key}_autocorr"] = 0.0

    # Chunk-level snap precision: across all hero bets in all hands
    all_snap_rates = [h["hero_bet_snap_rate"] for h in per_hand if h["hero_bet_count_nonzero"] > 0]
    result["ph_chunk_snap_rate"] = float(np.mean(all_snap_rates)) if all_snap_rates else 0.0

    # Hero unique raise-to sizes across entire chunk
    result["ph_chunk_unique_raise_to"] = float(sum(h["hero_unique_raise_to_count"] for h in per_hand))

    return result


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * log2(c / total) for c in counts.values() if c > 0)
