"""Bot-specific behavioral tell features.

Based on real commercial bot behavioral analysis:
- Bots fold immediately to postflop aggression (low WWSF, low AF postflop)
- Bots have anomalous 3bet/VPIP ratios (high 3bet relative to VPIP)
- Bots have high preflop aggression but passive postflop (AF mismatch)
- Bots use consistent, snapped-to-GTO bet sizing across all streets
- Bots show zero tilt/variance — identical responses to similar board textures
"""
from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Any

from src.data.preprocessor import numeric
from src.utils.helpers import safe_div

AGGRESSIVE = {"bet", "raise", "allin"}
STREETS = ("preflop", "flop", "turn", "river")


def extract_bot_tell_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    features: dict[str, float] = {}
    hero_seat = _hero_seat(chunk_group)

    features.update(_fold_to_aggression_features(chunk_group, hero_seat))
    features.update(_aggression_factor_features(chunk_group, hero_seat))
    features.update(_vpip_3bet_ratio_features(chunk_group, hero_seat))
    features.update(_postflop_surrender_features(chunk_group, hero_seat))
    features.update(_bet_size_consistency_features(chunk_group, hero_seat))
    features.update(_street_transition_features(chunk_group, hero_seat))
    return features


# ── Hero seat ──────────────────────────────────────────────────────────────

def _hero_seat(chunk_group: list[dict[str, Any]]) -> int | None:
    for hand in chunk_group:
        meta = hand.get("metadata", {})
        if isinstance(meta, dict) and meta.get("hero_seat") is not None:
            try:
                return int(meta["hero_seat"])
            except (TypeError, ValueError):
                pass
    return None


# ── 1. Fold-to-aggression by street ───────────────────────────────────────

def _fold_to_aggression_features(chunk_group: list[dict[str, Any]], hero: int | None) -> dict[str, float]:
    """
    Key bot tell: fold immediately when facing a bet/raise on flop/turn/river.
    Commercial bots (e.g. 3UpGaming) had WWSF ~15-17% — they almost never
    win when they see a flop, folding to any postflop aggression.
    """
    faced: dict[str, int] = {s: 0 for s in STREETS}
    folded: dict[str, int] = {s: 0 for s in STREETS}

    for hand in chunk_group:
        actions = hand.get("actions", [])
        by_street: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for a in actions:
            by_street[str(a.get("street", "unknown"))].append(a)

        for street in STREETS:
            street_actions = by_street[street]
            hero_acted = False
            villain_aggressed = False
            for i, a in enumerate(street_actions):
                seat = a.get("actor_seat")
                atype = str(a.get("action_type", ""))
                if seat != hero and atype in AGGRESSIVE:
                    villain_aggressed = True
                if seat == hero and villain_aggressed and not hero_acted:
                    faced[street] += 1
                    if atype == "fold":
                        folded[street] += 1
                    hero_acted = True

    features: dict[str, float] = {}
    for street in STREETS:
        features[f"bt_fold_to_{street}_aggression"] = safe_div(folded[street], faced[street])
        features[f"bt_faced_{street}_aggression_count"] = float(faced[street])

    # Overall postflop fold-to-aggression (flop+turn+river combined)
    post_faced = sum(faced[s] for s in ("flop", "turn", "river"))
    post_folded = sum(folded[s] for s in ("flop", "turn", "river"))
    features["bt_postflop_fold_to_aggression"] = safe_div(post_folded, post_faced)
    features["bt_postflop_faced_aggression"] = float(post_faced)
    return features


# ── 2. Aggression Factor (AF) by street ───────────────────────────────────

def _aggression_factor_features(chunk_group: list[dict[str, Any]], hero: int | None) -> dict[str, float]:
    """
    AF = (bets + raises) / calls.
    Bot tell: high preflop AF (3-bets to steal) but very low postflop AF (~0.5).
    Normal human: more balanced AF across streets.
    """
    hero_agg: dict[str, int] = {s: 0 for s in STREETS}
    hero_calls: dict[str, int] = {s: 0 for s in STREETS}

    for hand in chunk_group:
        for a in hand.get("actions", []):
            if a.get("actor_seat") != hero:
                continue
            street = str(a.get("street", "unknown"))
            atype = str(a.get("action_type", ""))
            if atype in AGGRESSIVE:
                hero_agg[street] = hero_agg.get(street, 0) + 1
            elif atype == "call":
                hero_calls[street] = hero_calls.get(street, 0) + 1

    features: dict[str, float] = {}
    for street in STREETS:
        features[f"bt_hero_af_{street}"] = safe_div(hero_agg.get(street, 0), hero_calls.get(street, 0))

    preflop_af = safe_div(hero_agg.get("preflop", 0), hero_calls.get("preflop", 0))
    postflop_agg = sum(hero_agg.get(s, 0) for s in ("flop", "turn", "river"))
    postflop_calls = sum(hero_calls.get(s, 0) for s in ("flop", "turn", "river"))
    postflop_af = safe_div(postflop_agg, postflop_calls)

    features["bt_hero_af_preflop"] = preflop_af
    features["bt_hero_af_postflop"] = postflop_af
    # Mismatch: high preflop - low postflop is the bot tell
    features["bt_hero_af_mismatch"] = preflop_af - postflop_af
    features["bt_hero_af_ratio"] = safe_div(preflop_af, max(postflop_af, 0.01))
    return features


# ── 3. VPIP / 3bet ratio anomaly ──────────────────────────────────────────

def _vpip_3bet_ratio_features(chunk_group: list[dict[str, Any]], hero: int | None) -> dict[str, float]:
    """
    Bot tell: VPIP=22 but 3bet=8 → 3bet/VPIP ratio ~0.36 (bots 3bet a huge
    fraction of their VPIP range). Normal humans: 3bet/VPIP ~0.10-0.20.
    Also: high 3bet with low postflop AF is a contradiction for humans.
    """
    vpip_hands = 0
    threebet_hands = 0
    open_raise_hands = 0
    hand_count = len(chunk_group)

    for hand in chunk_group:
        preflop = [a for a in hand.get("actions", []) if a.get("street") == "preflop"]
        hero_pre = [a for a in preflop if a.get("actor_seat") == hero]

        if any(a.get("action_type") in {"call", "bet", "raise", "allin"} for a in hero_pre):
            vpip_hands += 1

        # 3-bet: hero raises after another player already raised
        raises_before_hero = 0
        for a in preflop:
            if a.get("actor_seat") != hero and a.get("action_type") in {"raise", "allin"}:
                raises_before_hero += 1
            if a.get("actor_seat") == hero and a.get("action_type") in {"raise", "allin"} and raises_before_hero >= 1:
                threebet_hands += 1
                break

        # Open raise: hero raises with no prior raise
        prior_raise = False
        for a in preflop:
            if a.get("actor_seat") != hero and a.get("action_type") in {"raise", "allin"}:
                prior_raise = True
            if a.get("actor_seat") == hero and a.get("action_type") in {"raise", "allin"} and not prior_raise:
                open_raise_hands += 1
                break

    vpip_rate = safe_div(vpip_hands, hand_count)
    threebet_rate = safe_div(threebet_hands, hand_count)
    open_raise_rate = safe_div(open_raise_hands, hand_count)

    return {
        "bt_hero_vpip": vpip_rate,
        "bt_hero_3bet_rate": threebet_rate,
        "bt_hero_open_raise_rate": open_raise_rate,
        # Key ratio: bots have abnormally high 3bet/VPIP
        "bt_hero_3bet_vpip_ratio": safe_div(threebet_rate, max(vpip_rate, 0.01)),
        # And high 3bet/open_raise ratio
        "bt_hero_3bet_open_ratio": safe_div(threebet_rate, max(open_raise_rate, 0.01)),
    }


# ── 4. Postflop surrender pattern (WWSF proxy) ────────────────────────────

def _postflop_surrender_features(chunk_group: list[dict[str, Any]], hero: int | None) -> dict[str, float]:
    """
    WWSF (Won When Saw Flop): bots had 15-17%, humans typically 40-50%.
    Bots c-bet the flop but immediately fold to any counter-aggression,
    so they rarely win postflop pots they don't win on the flop c-bet.
    """
    saw_flop = 0
    won_postflop = 0
    cbet_then_folded = 0
    cbet_opportunities = 0
    check_fold_flop = 0
    check_flop_count = 0

    for hand in chunk_group:
        actions = hand.get("actions", [])
        by_street: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for a in actions:
            by_street[str(a.get("street", "unknown"))].append(a)

        flop_actions = by_street.get("flop", [])
        if not flop_actions:
            continue

        hero_on_flop = any(a.get("actor_seat") == hero for a in flop_actions)
        if not hero_on_flop:
            continue
        saw_flop += 1

        # Win: hand ended and hero is in winners
        outcome = hand.get("outcome", {})
        winners = outcome.get("winners") or []
        if hero in winners:
            won_postflop += 1

        # C-bet: hero raised preflop AND bets first on flop
        preflop_actions = by_street.get("preflop", [])
        hero_raised_preflop = any(
            a.get("actor_seat") == hero and a.get("action_type") in {"raise", "allin"}
            for a in preflop_actions
        )
        if hero_raised_preflop:
            cbet_opportunities += 1
            hero_flop_actions = [a for a in flop_actions if a.get("actor_seat") == hero]
            if hero_flop_actions and hero_flop_actions[0].get("action_type") in {"bet", "raise", "allin"}:
                # Hero c-bet — did they then fold to a raise?
                subsequent = [a for a in flop_actions if a.get("actor_seat") == hero and flop_actions.index(a) > flop_actions.index(hero_flop_actions[0])]
                if any(a.get("action_type") == "fold" for a in subsequent):
                    cbet_then_folded += 1

        # Check-fold on flop
        hero_flop = [a for a in flop_actions if a.get("actor_seat") == hero]
        if hero_flop and hero_flop[0].get("action_type") == "check":
            check_flop_count += 1
            if any(a.get("action_type") == "fold" for a in hero_flop[1:]):
                check_fold_flop += 1

    return {
        "bt_wwsf_proxy": safe_div(won_postflop, saw_flop),
        "bt_saw_flop_count": float(saw_flop),
        "bt_cbet_fold_rate": safe_div(cbet_then_folded, cbet_opportunities),
        "bt_cbet_opportunities": float(cbet_opportunities),
        "bt_check_fold_flop_rate": safe_div(check_fold_flop, check_flop_count),
    }


# ── 5. Bet size consistency (GTO snapping) ────────────────────────────────

def _bet_size_consistency_features(chunk_group: list[dict[str, Any]], hero: int | None) -> dict[str, float]:
    """
    Bots snap to standard GTO fractions on every street with near-zero variance.
    Measure: std of hero's bet/pot ratios per street, and overall uniqueness.
    """
    STANDARD_FRACTIONS = [0.25, 0.33, 0.50, 0.67, 0.75, 1.00, 1.25, 1.50, 2.00]

    street_ratios: dict[str, list[float]] = {s: [] for s in STREETS}
    all_raise_to: list[float] = []

    for hand in chunk_group:
        for a in hand.get("actions", []):
            if a.get("actor_seat") != hero:
                continue
            if a.get("action_type") not in AGGRESSIVE:
                continue
            street = str(a.get("street", "unknown"))
            pot = numeric(a.get("pot_before"))
            amount = numeric(a.get("amount"))
            if pot > 0 and amount > 0 and a.get("action_type") != "allin":
                ratio = amount / pot
                if street in street_ratios:
                    street_ratios[street].append(ratio)
            rt = numeric(a.get("raise_to"))
            if rt > 0:
                all_raise_to.append(round(rt, 3))

    features: dict[str, float] = {}

    all_ratios: list[float] = []
    for street in STREETS:
        ratios = street_ratios[street]
        all_ratios.extend(ratios)
        if ratios:
            features[f"bt_hero_{street}_bet_ratio_std"] = float(pstdev(ratios)) if len(ratios) > 1 else 0.0
            features[f"bt_hero_{street}_bet_ratio_mean"] = float(mean(ratios))
            snapped = sum(
                1 for r in ratios
                if any(abs(r - f) <= 0.05 * f for f in STANDARD_FRACTIONS)
            )
            features[f"bt_hero_{street}_snap_rate"] = safe_div(snapped, len(ratios))
        else:
            features[f"bt_hero_{street}_bet_ratio_std"] = 0.0
            features[f"bt_hero_{street}_bet_ratio_mean"] = 0.0
            features[f"bt_hero_{street}_snap_rate"] = 0.0

    if all_ratios:
        features["bt_hero_overall_bet_ratio_std"] = float(pstdev(all_ratios)) if len(all_ratios) > 1 else 0.0
        features["bt_hero_overall_snap_rate"] = safe_div(
            sum(1 for r in all_ratios if any(abs(r - f) <= 0.05 * f for f in STANDARD_FRACTIONS)),
            len(all_ratios),
        )
    else:
        features["bt_hero_overall_bet_ratio_std"] = 0.0
        features["bt_hero_overall_snap_rate"] = 0.0

    # Unique raise_to amounts: bots reuse same sizes across hands
    features["bt_hero_unique_raise_to"] = float(len(set(all_raise_to)))
    features["bt_hero_raise_to_count"] = float(len(all_raise_to))
    features["bt_hero_raise_to_uniqueness"] = safe_div(len(set(all_raise_to)), len(all_raise_to))

    return features


# ── 6. Street transition aggression patterns ──────────────────────────────

def _street_transition_features(chunk_group: list[dict[str, Any]], hero: int | None) -> dict[str, float]:
    """
    Bot pattern: aggressive preflop → passive/folding postflop.
    Measure hero's aggression transition across streets.
    """
    street_agg: dict[str, list[float]] = {s: [] for s in STREETS}

    for hand in chunk_group:
        by_street: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for a in hand.get("actions", []):
            by_street[str(a.get("street", "unknown"))].append(a)

        for street in STREETS:
            hero_acts = [a for a in by_street[street] if a.get("actor_seat") == hero]
            if hero_acts:
                agg = sum(1 for a in hero_acts if a.get("action_type") in AGGRESSIVE)
                street_agg[street].append(safe_div(agg, len(hero_acts)))

    features: dict[str, float] = {}
    for street in STREETS:
        vals = street_agg[street]
        features[f"bt_hero_agg_rate_{street}"] = float(mean(vals)) if vals else 0.0

    pre = features.get("bt_hero_agg_rate_preflop", 0.0)
    flop = features.get("bt_hero_agg_rate_flop", 0.0)
    turn = features.get("bt_hero_agg_rate_turn", 0.0)
    river = features.get("bt_hero_agg_rate_river", 0.0)
    postflop = mean([flop, turn, river]) if any([flop, turn, river]) else 0.0

    # The key bot tell: high preflop, low postflop
    features["bt_preflop_to_postflop_agg_drop"] = max(0.0, pre - postflop)
    features["bt_agg_trend_flop_to_river"] = flop - river  # bots drop off quickly
    features["bt_postflop_agg_consistency"] = float(pstdev([flop, turn, river])) if all([flop, turn, river]) else 0.0

    return features
