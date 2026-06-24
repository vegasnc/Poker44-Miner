from __future__ import annotations

from collections import Counter, defaultdict
from math import log2
from statistics import mean, pstdev
from typing import Any

import numpy as np

from src.data.preprocessor import numeric
from src.utils.helpers import safe_div

STREETS = ("preflop", "flop", "turn", "river")
ACTION_TYPES = ("fold", "check", "call", "bet", "raise", "allin", "blind")
AGGRESSIVE_ACTIONS = {"bet", "raise", "allin"}
PASSIVE_ACTIONS = {"call", "check"}


def action_entropy(action_types: list[str]) -> float:
    if not action_types:
        return 0.0
    counts = Counter(action_types)
    total = len(action_types)
    return -sum((count / total) * log2(count / total) for count in counts.values())


def extract_behavioral_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    actions: list[dict[str, Any]] = [action for hand in chunk_group for action in hand.get("actions", [])]
    action_types = [str(action.get("action_type", "unknown")) for action in actions]
    action_counts = Counter(action_types)
    street_counts: dict[str, Counter[str]] = defaultdict(Counter)
    actor_counts: Counter[Any] = Counter()
    sequence_repeats = 0

    last_action: str | None = None
    for action in actions:
        action_type = str(action.get("action_type", "unknown"))
        street = str(action.get("street", "unknown"))
        street_counts[street][action_type] += 1
        actor_counts[action.get("actor_seat")] += 1
        if last_action == action_type:
            sequence_repeats += 1
        last_action = action_type

    total_actions = len(actions)
    aggressive = sum(action_counts[action] for action in AGGRESSIVE_ACTIONS)
    passive = sum(action_counts[action] for action in PASSIVE_ACTIONS)
    calls = action_counts["call"]
    checks = action_counts["check"]
    raises = action_counts["raise"] + action_counts["allin"]
    bets = action_counts["bet"]

    features: dict[str, float] = {
        "hand_count": float(len(chunk_group)),
        "action_count": float(total_actions),
        "unique_actor_count": float(len([actor for actor in actor_counts if actor is not None])),
        "actions_per_hand": safe_div(total_actions, len(chunk_group)),
        "action_entropy": action_entropy(action_types),
        "repeat_action_rate": safe_div(sequence_repeats, max(total_actions - 1, 0)),
        "aggression_factor": safe_div(bets + raises, calls, default=float(bets + raises)),
        "aggression_frequency": safe_div(aggressive, total_actions),
        "passive_action_frequency": safe_div(passive, total_actions),
        "check_raise_rate": _check_raise_rate(chunk_group),
    }

    for action_type in ACTION_TYPES:
        features[f"{action_type}_count"] = float(action_counts[action_type])
        features[f"{action_type}_rate"] = safe_div(action_counts[action_type], total_actions)

    for street in STREETS:
        street_total = sum(street_counts[street].values())
        features[f"{street}_action_count"] = float(street_total)
        features[f"{street}_aggression_frequency"] = safe_div(
            sum(street_counts[street][action] for action in AGGRESSIVE_ACTIONS),
            street_total,
        )
        for action_type in ("fold", "check", "call", "bet", "raise"):
            features[f"{street}_{action_type}_rate"] = safe_div(street_counts[street][action_type], street_total)

    actor_action_counts = list(actor_counts.values())
    features["actor_action_count_mean"] = float(mean(actor_action_counts)) if actor_action_counts else 0.0
    features["actor_action_count_std"] = float(pstdev(actor_action_counts)) if len(actor_action_counts) > 1 else 0.0
    return features


def extract_bet_sizing_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    actions = [action for hand in chunk_group for action in hand.get("actions", [])]
    bet_actions = [action for action in actions if action.get("action_type") in {"bet", "raise", "call", "allin"}]
    # Chip amounts (raw): use amount/raise_to/call_to but NOT normalized_amount_bb (different unit).
    chip_amounts = [_chip_amount(action) for action in bet_actions]
    pot_ratios = [
        safe_div(_chip_amount(action), numeric(action.get("pot_before")))
        for action in bet_actions
        if numeric(action.get("pot_before")) > 0
    ]
    # BB-normalised amounts: use the dedicated field only.
    bb_amounts = [numeric(action.get("normalized_amount_bb")) for action in bet_actions if numeric(action.get("normalized_amount_bb")) > 0]

    features = _distribution_features("bet_amount", chip_amounts)
    features.update(_distribution_features("bet_pot_ratio", pot_ratios))
    features.update(_distribution_features("bet_amount_bb", bb_amounts))

    for street in STREETS:
        street_chip = [_chip_amount(action) for action in bet_actions if action.get("street") == street]
        features.update(_distribution_features(f"{street}_bet_amount", street_chip, compact=True))

    return features


def extract_timing_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    times: list[float] = []
    for hand in chunk_group:
        for action in hand.get("actions", []):
            for key in ("response_time", "decision_time", "elapsed_ms", "time_ms", "timestamp_delta"):
                value = numeric(action.get(key), default=-1.0)
                if value >= 0:
                    times.append(value)
                    break
    return _distribution_features("decision_time", times)


def _chip_amount(action: dict[str, Any]) -> float:
    """Bet size in chips (raw currency units), excluding BB-normalised field."""
    return max(
        numeric(action.get("amount")),
        numeric(action.get("raise_to")),
        numeric(action.get("call_to")),
    )


def _effective_amount(action: dict[str, Any]) -> float:
    """Bet size in chips. Kept for backward compatibility with callers that don't use BB units."""
    return _chip_amount(action)


def _distribution_features(prefix: str, values: list[float], compact: bool = False) -> dict[str, float]:
    values = [float(value) for value in values if value == value]
    if not values:
        base = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        if not compact:
            base.update({"median": 0.0, "p90": 0.0, "cv": 0.0})
        return {f"{prefix}_{key}": value for key, value in base.items()}

    array = np.asarray(values, dtype=float)
    base = {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }
    if not compact:
        base.update(
            {
                "median": float(np.median(array)),
                "p25": float(np.percentile(array, 25)),
                "p75": float(np.percentile(array, 75)),
                "p90": float(np.percentile(array, 90)),
                "cv": safe_div(float(np.std(array)), abs(float(np.mean(array)))),
            }
        )
    return {f"{prefix}_{key}": value for key, value in base.items()}


def _check_raise_rate(chunk_group: list[dict[str, Any]]) -> float:
    opportunities = 0
    check_raises = 0
    for hand in chunk_group:
        by_actor_street: dict[tuple[Any, str], list[str]] = defaultdict(list)
        for action in hand.get("actions", []):
            key = (action.get("actor_seat"), str(action.get("street", "unknown")))
            by_actor_street[key].append(str(action.get("action_type", "unknown")))
        for sequence in by_actor_street.values():
            if "check" in sequence:
                opportunities += 1
                check_index = sequence.index("check")
                if any(action in {"raise", "allin"} for action in sequence[check_index + 1 :]):
                    check_raises += 1
    return safe_div(check_raises, opportunities)
