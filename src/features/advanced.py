from __future__ import annotations

from collections import Counter, defaultdict
from math import log2
from statistics import mean, pstdev
from typing import Any

import numpy as np

from src.data.preprocessor import numeric
from src.utils.helpers import safe_div

AGGRESSIVE_ACTIONS = {"bet", "raise", "allin"}
VOLUNTARY_PREFLOP_ACTIONS = {"call", "bet", "raise", "allin"}
STREETS = ("preflop", "flop", "turn", "river")
STATIC_NGRAMS = (
    "raise_call",
    "raise_fold",
    "call_check",
    "check_bet",
    "check_raise",
    "bet_fold",
    "bet_call",
    "bet_raise",
    "raise_call_check",
    "call_check_fold",
    "check_bet_fold",
    "raise_call_check_bet",
)


def extract_advanced_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    features: dict[str, float] = {}
    features.update(_sequence_features(chunk_group))
    features.update(_actor_consistency_features(chunk_group))
    features.update(_line_features(chunk_group))
    features.update(_bet_geometry_features(chunk_group))
    features.update(_signature_features(chunk_group))
    features.update(_pot_dynamics_features(chunk_group))
    features.update(_hand_classification_features(chunk_group))
    return features


def _sequence_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    bigrams: Counter[str] = Counter()
    trigrams: Counter[str] = Counter()
    street_bigrams: Counter[str] = Counter()
    total_bigrams = 0
    total_trigrams = 0

    for hand in chunk_group:
        actions = [str(action.get("action_type", "unknown")) for action in hand.get("actions", [])]
        streets = [str(action.get("street", "unknown")) for action in hand.get("actions", [])]
        for index in range(len(actions) - 1):
            bigrams[f"{actions[index]}_{actions[index + 1]}"] += 1
            street_bigrams[f"{streets[index]}_{actions[index]}__{streets[index + 1]}_{actions[index + 1]}"] += 1
            total_bigrams += 1
        for index in range(len(actions) - 2):
            trigrams[f"{actions[index]}_{actions[index + 1]}_{actions[index + 2]}"] += 1
            total_trigrams += 1

    features = {
        "seq_bigram_entropy": _entropy(bigrams),
        "seq_trigram_entropy": _entropy(trigrams),
        "seq_unique_bigram_rate": safe_div(len(bigrams), total_bigrams),
        "seq_unique_trigram_rate": safe_div(len(trigrams), total_trigrams),
        "seq_street_bigram_entropy": _entropy(street_bigrams),
    }
    for ngram in STATIC_NGRAMS:
        features[f"seq_{ngram}_rate"] = safe_div(bigrams[ngram] + trigrams[ngram], total_bigrams + total_trigrams)
    return features


def _actor_consistency_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    # Track per-hand actions per actor so we can compute true per-hand rates.
    # actor_hand_actions[seat][hand_index] = list of actions
    actor_hand_actions: dict[Any, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for hand_index, hand in enumerate(chunk_group):
        for action in hand.get("actions", []):
            actor_hand_actions[action.get("actor_seat")][hand_index].append(action)

    action_counts: list[float] = []
    aggression_rates: list[float] = []
    fold_rates: list[float] = []
    call_rates: list[float] = []
    vpip_rates: list[float] = []
    pfr_rates: list[float] = []
    amount_cvs: list[float] = []
    repeated_bucket_rates: list[float] = []

    for hand_actions in actor_hand_actions.values():
        actions = [action for hand in hand_actions.values() for action in hand]
        total = len(actions)
        if total == 0:
            continue
        counts = Counter(str(action.get("action_type", "unknown")) for action in actions)
        action_counts.append(float(total))
        aggression_rates.append(safe_div(sum(counts[action] for action in AGGRESSIVE_ACTIONS), total))
        fold_rates.append(safe_div(counts["fold"], total))
        call_rates.append(safe_div(counts["call"], total))

        # Per-hand VPIP/PFR: fraction of hands where actor voluntarily entered / raised preflop.
        vpip_hand_count = 0
        pfr_hand_count = 0
        for hand in hand_actions.values():
            preflop = [a for a in hand if a.get("street") == "preflop"]
            if any(a.get("action_type") in VOLUNTARY_PREFLOP_ACTIONS for a in preflop):
                vpip_hand_count += 1
            if any(a.get("action_type") in {"raise", "allin"} for a in preflop):
                pfr_hand_count += 1
        hand_count = len(hand_actions)
        vpip_rates.append(safe_div(vpip_hand_count, hand_count))
        pfr_rates.append(safe_div(pfr_hand_count, hand_count))

        amounts = [_effective_amount(action) for action in actions if _effective_amount(action) > 0]
        if amounts:
            amount_cvs.append(_cv(amounts))
            buckets = [_amount_bucket(action) for action in actions if _effective_amount(action) > 0]
            most_common = Counter(buckets).most_common(1)[0][1] if buckets else 0
            repeated_bucket_rates.append(safe_div(most_common, len(buckets)))

    features: dict[str, float] = {}
    features.update(_dist("actor_action_count", action_counts))
    features.update(_dist("actor_aggression_rate", aggression_rates))
    features.update(_dist("actor_fold_rate", fold_rates))
    features.update(_dist("actor_call_rate", call_rates))
    features.update(_dist("actor_vpip_flag", vpip_rates))
    features.update(_dist("actor_pfr_flag", pfr_rates))
    features.update(_dist("actor_amount_cv", amount_cvs))
    features.update(_dist("actor_repeated_bet_bucket_rate", repeated_bucket_rates))
    features["actor_count"] = float(len(actor_hand_actions))
    return features


def _line_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    hand_count = len(chunk_group)
    preflop_raise = 0
    cbet = 0
    delayed_turn_aggression = 0
    river_aggression = 0
    check_call_lines = 0
    check_fold_lines = 0
    bet_bet_lines = 0

    for hand in chunk_group:
        by_street: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for action in hand.get("actions", []):
            by_street[str(action.get("street", "unknown"))].append(action)

        preflop_aggressors = {
            action.get("actor_seat")
            for action in by_street["preflop"]
            if action.get("action_type") in {"raise", "allin"}
        }
        if preflop_aggressors:
            preflop_raise += 1
        if any(action.get("actor_seat") in preflop_aggressors and action.get("action_type") in AGGRESSIVE_ACTIONS for action in by_street["flop"]):
            cbet += 1
        if not any(action.get("action_type") in AGGRESSIVE_ACTIONS for action in by_street["flop"]) and any(
            action.get("action_type") in AGGRESSIVE_ACTIONS for action in by_street["turn"]
        ):
            delayed_turn_aggression += 1
        if any(action.get("action_type") in AGGRESSIVE_ACTIONS for action in by_street["river"]):
            river_aggression += 1
        if _street_line_contains(by_street, ("check", "call")):
            check_call_lines += 1
        if _street_line_contains(by_street, ("check", "fold")):
            check_fold_lines += 1
        if _street_line_contains(by_street, ("bet", "bet")):
            bet_bet_lines += 1

    return {
        "line_preflop_raise_rate": safe_div(preflop_raise, hand_count),
        "line_cbet_rate": safe_div(cbet, preflop_raise),
        "line_delayed_turn_aggression_rate": safe_div(delayed_turn_aggression, hand_count),
        "line_river_aggression_rate": safe_div(river_aggression, hand_count),
        "line_check_call_rate": safe_div(check_call_lines, hand_count),
        "line_check_fold_rate": safe_div(check_fold_lines, hand_count),
        "line_bet_bet_rate": safe_div(bet_bet_lines, hand_count),
    }


def _bet_geometry_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    bucket_counts: Counter[str] = Counter()
    all_buckets = ("tiny", "small", "half_pot", "pot", "overbet", "allin")
    transitions: Counter[str] = Counter()
    previous_bucket: str | None = None
    total = 0

    for hand in chunk_group:
        for action in hand.get("actions", []):
            if action.get("action_type") not in {"bet", "raise", "call", "allin"}:
                continue
            bucket = _amount_bucket(action)
            bucket_counts[bucket] += 1
            total += 1
            if previous_bucket is not None:
                transitions[f"{previous_bucket}_to_{bucket}"] += 1
            previous_bucket = bucket

    features = {f"bet_bucket_{bucket}_rate": safe_div(bucket_counts[bucket], total) for bucket in all_buckets}
    features["bet_bucket_entropy"] = _entropy(bucket_counts)
    features["bet_bucket_repeat_rate"] = safe_div(max(bucket_counts.values()) if bucket_counts else 0, total)
    for transition in ("small_to_small", "half_pot_to_half_pot", "pot_to_pot", "small_to_pot", "pot_to_overbet"):
        features[f"bet_transition_{transition}_rate"] = safe_div(transitions[transition], max(total - 1, 0))
    return features


def _street_line_contains(by_street: dict[str, list[dict[str, Any]]], pattern: tuple[str, str]) -> bool:
    for street in STREETS:
        actions = [str(action.get("action_type", "unknown")) for action in by_street[street]]
        for index in range(len(actions) - 1):
            if (actions[index], actions[index + 1]) == pattern:
                return True
    return False


def _effective_amount(action: dict[str, Any]) -> float:
    """Bet size in chips (raw currency units), excluding BB-normalised field."""
    return max(
        numeric(action.get("amount")),
        numeric(action.get("raise_to")),
        numeric(action.get("call_to")),
    )


def _amount_bucket(action: dict[str, Any]) -> str:
    if action.get("action_type") == "allin":
        return "allin"
    amount = _effective_amount(action)
    pot_before = numeric(action.get("pot_before"))
    ratio = safe_div(amount, pot_before)
    if ratio <= 0.20:
        return "tiny"
    if ratio <= 0.45:
        return "small"
    if ratio <= 0.75:
        return "half_pot"
    if ratio <= 1.20:
        return "pot"
    return "overbet"


def _entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * log2(count / total) for count in counts.values())


def _dist(prefix: str, values: list[float]) -> dict[str, float]:
    values = [float(value) for value in values if value == value]
    if not values:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(mean(values)),
        f"{prefix}_std": float(pstdev(values)) if len(values) > 1 else 0.0,
        f"{prefix}_min": float(min(values)),
        f"{prefix}_max": float(max(values)),
    }


def _cv(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    return safe_div(float(np.std(array)), abs(float(np.mean(array))))


def _max_run_share(values: list[Any]) -> float:
    """Longest consecutive run of identical values as a fraction of total length."""
    if not values:
        return 0.0
    best = 1
    run = 1
    for index in range(1, len(values)):
        if values[index] == values[index - 1]:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best / len(values)


def _signature_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    """Frequency-based signature analysis: detect repetitive action/amount-bucket patterns."""
    action_sigs: Counter[str] = Counter()
    bucket_sigs: Counter[str] = Counter()
    actor_sigs: Counter[str] = Counter()

    action_runs: list[float] = []
    bucket_runs: list[float] = []
    actor_runs: list[float] = []

    for hand in chunk_group:
        actions = [str(a.get("action_type", "?")) for a in hand.get("actions", [])]
        buckets = [_amount_bucket(a) for a in hand.get("actions", [])
                   if a.get("action_type") in {"bet", "raise", "call", "allin"}]
        actors = [str(a.get("actor_seat", "?")) for a in hand.get("actions", [])]

        sig_a = "_".join(actions)
        sig_b = "_".join(buckets)
        sig_c = "_".join(actors)
        if sig_a:
            action_sigs[sig_a] += 1
        if sig_b:
            bucket_sigs[sig_b] += 1
        if sig_c:
            actor_sigs[sig_c] += 1

        action_runs.append(_max_run_share(actions))
        bucket_runs.append(_max_run_share(buckets))
        actor_runs.append(_max_run_share(actors))

    hand_count = len(chunk_group)
    top_action_sig_rate = safe_div(action_sigs.most_common(1)[0][1], hand_count) if action_sigs else 0.0
    top_bucket_sig_rate = safe_div(bucket_sigs.most_common(1)[0][1], hand_count) if bucket_sigs else 0.0
    top_actor_sig_rate = safe_div(actor_sigs.most_common(1)[0][1], hand_count) if actor_sigs else 0.0

    return {
        "sig_action_seq_entropy": _entropy(action_sigs),
        "sig_bucket_seq_entropy": _entropy(bucket_sigs),
        "sig_actor_seq_entropy": _entropy(actor_sigs),
        "sig_action_unique_rate": safe_div(len(action_sigs), hand_count),
        "sig_bucket_unique_rate": safe_div(len(bucket_sigs), hand_count),
        "sig_top_action_sig_rate": top_action_sig_rate,
        "sig_top_bucket_sig_rate": top_bucket_sig_rate,
        "sig_top_actor_sig_rate": top_actor_sig_rate,
        "sig_action_run_mean": float(mean(action_runs)) if action_runs else 0.0,
        "sig_action_run_max": float(max(action_runs)) if action_runs else 0.0,
        "sig_bucket_run_mean": float(mean(bucket_runs)) if bucket_runs else 0.0,
        "sig_actor_run_mean": float(mean(actor_runs)) if actor_runs else 0.0,
    }


def _pot_dynamics_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    """Pot growth monotonicity and delta consistency across hands."""
    monotonicity_vals: list[float] = []
    pot_delta_cvs: list[float] = []

    for hand in chunk_group:
        pots = [numeric(a.get("pot_before")) for a in hand.get("actions", [])
                if numeric(a.get("pot_before")) > 0]
        if len(pots) < 2:
            continue
        increases = sum(1 for i in range(1, len(pots)) if pots[i] >= pots[i - 1])
        monotonicity_vals.append(increases / (len(pots) - 1))
        deltas = [pots[i] - pots[i - 1] for i in range(1, len(pots))]
        pot_delta_cvs.append(_cv([abs(d) for d in deltas if d != 0.0]))

    return {
        "pot_monotonicity_mean": float(mean(monotonicity_vals)) if monotonicity_vals else 0.0,
        "pot_monotonicity_std": float(pstdev(monotonicity_vals)) if len(monotonicity_vals) > 1 else 0.0,
        "pot_delta_cv_mean": float(mean(pot_delta_cvs)) if pot_delta_cvs else 0.0,
    }


def _hand_classification_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    """Classification flags: rates of aggressive, low-entropy, high-actor-count hands."""
    hand_count = len(chunk_group)
    aggressive_hands = 0
    low_entropy_hands = 0
    high_actor_hands = 0
    long_sequence_hands = 0

    for hand in chunk_group:
        actions = [str(a.get("action_type", "?")) for a in hand.get("actions", [])]
        actors = {a.get("actor_seat") for a in hand.get("actions", []) if a.get("actor_seat") is not None}
        agg_count = sum(1 for a in actions if a in AGGRESSIVE_ACTIONS)

        if agg_count >= 3:
            aggressive_hands += 1
        counts = Counter(actions)
        total = len(actions)
        if total > 0:
            entropy = -sum((c / total) * log2(c / total) for c in counts.values() if c > 0)
            if entropy < 0.8:
                low_entropy_hands += 1
        if len(actors) >= 4:
            high_actor_hands += 1
        if total >= 10:
            long_sequence_hands += 1

    return {
        "cls_aggressive_hand_rate": safe_div(aggressive_hands, hand_count),
        "cls_low_entropy_hand_rate": safe_div(low_entropy_hands, hand_count),
        "cls_high_actor_hand_rate": safe_div(high_actor_hands, hand_count),
        "cls_long_sequence_hand_rate": safe_div(long_sequence_hands, hand_count),
    }
