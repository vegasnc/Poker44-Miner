"""GTO-based features derived from bet sizing and pot odds analysis.

These features don't require hole cards. They exploit the fact that GTO bots
use solver-derived bet sizes that cluster tightly around standard fractions
(1/3, 1/2, 2/3, 3/4, 1x, 4/3, 3/2 pot), while humans deviate more randomly.

Prefix: ge_
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

# Standard GTO bet size fractions (as proportion of pot)
_GTO_FRACTIONS = [0.25, 0.33, 0.50, 0.67, 0.75, 1.00, 1.25, 1.33, 1.50, 2.00]

# Street index for ordering
_STREET_ORDER = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 1e-9 else 0.0


def _nearest_gto_deviation(fraction: float) -> float:
    """Absolute deviation from the nearest standard GTO bet fraction."""
    return min(abs(fraction - f) for f in _GTO_FRACTIONS)


def _pot_odds(call_amount: float, pot_before: float) -> float:
    """Equity needed to break even on a call: call / (pot + call)."""
    total = pot_before + call_amount
    return _safe_div(call_amount, total)


def extract_gto_engine_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    """
    Extract GTO-based features from a chunk (list of hand dicts).
    Returns a flat dict with prefix 'ge_'.
    """
    if not chunk_group:
        return {}

    # ---- Collect per-hand metrics ----
    bet_fractions: list[float] = []          # hero bet/raise as fraction of pot
    gto_deviations: list[float] = []         # deviation from nearest GTO fraction
    pot_odds_calls: list[float] = []         # pot odds faced on hero calls
    pot_odds_folds: list[float] = []         # pot odds faced on hero folds (vs villain bet)
    overbet_flags: list[float] = []          # 1 if hero bet > 1.2× pot
    minraise_flags: list[float] = []         # 1 if hero raise is near minimum
    sizing_by_street: dict[str, list[float]] = {s: [] for s in _STREET_ORDER}

    # Per-street pot-odds compliance
    po_correct: list[float] = []    # 1=hero called when pot odds favourable (equity > 33%)
    po_incorrect: list[float] = []  # 1=hero called when pot odds unfavourable

    # Villain response to hero bets
    villain_fold_after_hero_bet: list[float] = []
    hero_continuation_bet: list[float] = []  # 1 if hero bets flop after preflop aggressor

    for hand in chunk_group:
        actions: list[dict] = hand.get("actions") or []
        meta: dict = hand.get("metadata") or {}
        bb = float(meta.get("bb") or 0.02)
        hero_seat = meta.get("hero_seat")

        # ---- Classify actions ----
        hero_preflop_raised = False
        last_hero_bet_street: str | None = None
        last_villain_action_after_hero: str | None = None

        for i, act in enumerate(actions):
            atype = act.get("action_type", "")
            seat = act.get("actor_seat")
            street = act.get("street", "preflop")
            pot_before = float(act.get("pot_before") or 0.0)
            amount = float(act.get("amount") or 0.0)
            raise_to = float(act.get("raise_to") or 0.0)
            call_to = float(act.get("call_to") or 0.0)

            is_hero = (seat == hero_seat)

            # Track hero preflop aggression for c-bet detection
            if street == "preflop" and is_hero and atype in ("raise", "bet"):
                hero_preflop_raised = True

            # --- Hero bet/raise: compute sizing fraction ---
            if is_hero and atype in ("bet", "raise") and raise_to > 0 and pot_before > bb * 0.5:
                fraction = _safe_div(raise_to, pot_before)
                if 0.05 < fraction < 5.0:  # sanity bounds
                    bet_fractions.append(fraction)
                    gto_deviations.append(_nearest_gto_deviation(fraction))
                    overbet_flags.append(float(fraction > 1.2))
                    minraise_flags.append(float(fraction < 0.4 and atype == "raise"))
                    if street in sizing_by_street:
                        sizing_by_street[street].append(fraction)
                last_hero_bet_street = street

                # Look ahead: did villain fold to this bet?
                for future in actions[i + 1:]:
                    if future.get("street") != street:
                        break
                    if future.get("actor_seat") != hero_seat:
                        villain_fold_after_hero_bet.append(
                            float(future.get("action_type") == "fold")
                        )
                        break

                # C-bet detection: hero bets flop after preflop raise
                if street == "flop" and hero_preflop_raised:
                    hero_continuation_bet.append(1.0)

            # --- Hero call: pot odds ---
            if is_hero and atype == "call" and call_to > 0 and pot_before > 0:
                call_amount = call_to - float(act.get("amount") or 0.0)
                if call_amount < 0:
                    call_amount = amount
                po = _pot_odds(call_amount, pot_before)
                pot_odds_calls.append(po)
                # Pot odds < 0.33 means we need < 33% equity → generally profitable call
                po_correct.append(float(po < 0.40))
                po_incorrect.append(float(po >= 0.40))

            # --- Hero fold: pot odds they folded to ---
            if is_hero and atype == "fold":
                # Find the villain bet that forced this fold
                for prev in reversed(actions[:i]):
                    if prev.get("actor_seat") != hero_seat and prev.get("action_type") in ("bet", "raise"):
                        prev_pot = float(prev.get("pot_before") or 0.0)
                        prev_raise = float(prev.get("raise_to") or 0.0)
                        call_needed = prev_raise
                        po = _pot_odds(call_needed, prev_pot)
                        pot_odds_folds.append(po)
                        break

    # ---- Aggregate features ----
    feats: dict[str, float] = {}

    # Bet sizing distribution
    feats["ge_bet_fraction_mean"] = float(np.mean(bet_fractions)) if bet_fractions else 0.0
    feats["ge_bet_fraction_std"] = float(np.std(bet_fractions)) if len(bet_fractions) > 1 else 0.0
    feats["ge_bet_fraction_median"] = float(np.median(bet_fractions)) if bet_fractions else 0.0
    feats["ge_bet_count"] = float(len(bet_fractions))

    # GTO deviation metrics
    feats["ge_gto_deviation_mean"] = float(np.mean(gto_deviations)) if gto_deviations else 0.0
    feats["ge_gto_deviation_std"] = float(np.std(gto_deviations)) if len(gto_deviations) > 1 else 0.0
    feats["ge_gto_deviation_max"] = float(max(gto_deviations)) if gto_deviations else 0.0
    # Low deviation = tight clustering = more bot-like
    feats["ge_gto_low_deviation_rate"] = _safe_div(
        sum(1 for d in gto_deviations if d < 0.08), len(gto_deviations)
    ) if gto_deviations else 0.0

    # Bet size entropy (low entropy = bot-like clustering)
    if len(bet_fractions) > 2:
        # Bucket into 20 bins [0, 2.5] pot
        hist, _ = np.histogram(bet_fractions, bins=20, range=(0.0, 2.5))
        hist = hist.astype(float)
        hist_norm = hist / hist.sum() if hist.sum() > 0 else hist
        entropy = -sum(p * math.log(p + 1e-10) for p in hist_norm if p > 0)
        feats["ge_bet_size_entropy"] = entropy
    else:
        feats["ge_bet_size_entropy"] = 0.0

    # Overbet / minraise tendencies
    feats["ge_overbet_rate"] = _safe_div(sum(overbet_flags), len(overbet_flags)) if overbet_flags else 0.0
    feats["ge_minraise_rate"] = _safe_div(sum(minraise_flags), len(minraise_flags)) if minraise_flags else 0.0

    # Pot odds on calls
    feats["ge_call_pot_odds_mean"] = float(np.mean(pot_odds_calls)) if pot_odds_calls else 0.0
    feats["ge_call_pot_odds_std"] = float(np.std(pot_odds_calls)) if len(pot_odds_calls) > 1 else 0.0
    feats["ge_call_count"] = float(len(pot_odds_calls))

    # Pot odds on folds
    feats["ge_fold_pot_odds_mean"] = float(np.mean(pot_odds_folds)) if pot_odds_folds else 0.0
    feats["ge_fold_pot_odds_std"] = float(np.std(pot_odds_folds)) if len(pot_odds_folds) > 1 else 0.0

    # Pot-odds compliance: bots fold/call mechanically based on pot odds threshold
    total_po_decisions = len(po_correct) + len(po_incorrect)
    feats["ge_po_compliance_rate"] = _safe_div(
        sum(po_correct), total_po_decisions
    ) if total_po_decisions else 0.0

    # Fold to unfavorable pot odds (>40%) - bots do this more consistently
    folds_vs_unfav = sum(1 for po in pot_odds_folds if po > 0.40)
    feats["ge_fold_unfav_po_rate"] = _safe_div(folds_vs_unfav, len(pot_odds_folds)) if pot_odds_folds else 0.0

    # C-bet frequency
    feats["ge_cbet_rate"] = _safe_div(sum(hero_continuation_bet), len(hero_continuation_bet)) if hero_continuation_bet else 0.0

    # Villain fold to hero bet rate
    feats["ge_villain_fold_to_bet_rate"] = _safe_div(
        sum(villain_fold_after_hero_bet), len(villain_fold_after_hero_bet)
    ) if villain_fold_after_hero_bet else 0.0

    # Per-street bet sizing consistency
    for street in ("preflop", "flop", "turn", "river"):
        fracs = sizing_by_street[street]
        feats[f"ge_bet_std_{street}"] = float(np.std(fracs)) if len(fracs) > 1 else 0.0
        feats[f"ge_bet_mean_{street}"] = float(np.mean(fracs)) if fracs else 0.0
        feats[f"ge_gto_dev_{street}"] = float(np.mean([_nearest_gto_deviation(f) for f in fracs])) if fracs else 0.0

    # Ratio of flop to preflop bet size (GTO bots have consistent ratio)
    pf_mean = feats["ge_bet_mean_preflop"]
    flop_mean = feats["ge_bet_mean_flop"]
    feats["ge_flop_to_pf_bet_ratio"] = _safe_div(flop_mean, pf_mean) if pf_mean > 0 else 0.0

    # Bet size quantization: fraction of bets that land near a GTO size (within 8%)
    feats["ge_quantized_bet_rate"] = feats["ge_gto_low_deviation_rate"]

    return feats
