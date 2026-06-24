"""GTO-pattern features derived from bot decision-making research.

Key insight: bots in <20-hand sessions are locked in pure GTO mode.
GTO has precise, solver-trained frequencies that differ from human play:
  - Fixed check/bet ratios per street
  - Stable probe-bet frequency when checked to
  - Near-zero behavioral drift across the session
  - PFR/VPIP ratio stays unnaturally constant hand-over-hand
  - Responses to villain aggression are deterministic (same action every time)
"""
from __future__ import annotations

from collections import Counter, defaultdict
from math import log2
from statistics import mean, pstdev
from typing import Any

from src.data.preprocessor import numeric
from src.utils.helpers import safe_div

AGGRESSIVE = {"bet", "raise", "allin"}
STREETS = ("preflop", "flop", "turn", "river")
VOLUNTARY_PREFLOP = {"call", "bet", "raise", "allin"}


def extract_gto_tell_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    features: dict[str, float] = {}
    hero = _hero_seat(chunk_group)
    features.update(_behavioral_drift_features(chunk_group, hero))
    features.update(_check_bet_balance_features(chunk_group, hero))
    features.update(_probe_bet_features(chunk_group, hero))
    features.update(_pfr_vpip_stability_features(chunk_group, hero))
    features.update(_villain_response_consistency_features(chunk_group, hero))
    return features


def _hero_seat(chunk_group: list[dict[str, Any]]) -> int | None:
    for hand in chunk_group:
        meta = hand.get("metadata", {})
        if isinstance(meta, dict) and meta.get("hero_seat") is not None:
            try:
                return int(meta["hero_seat"])
            except (TypeError, ValueError):
                pass
    return None


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * log2(c / total) for c in counts.values() if c > 0)


# ── 1. First-half vs second-half behavioral drift ─────────────────────────

def _behavioral_drift_features(chunk_group: list[dict[str, Any]], hero: int | None) -> dict[str, float]:
    """
    Bots are locked in pure GTO for <20 hands — no adaptation.
    Humans naturally adjust: they warm up, loosen up, adapt to table dynamics.
    Split chunk into first vs second half and measure behavioral differences.
    Large drift = human; near-zero drift = bot.
    """
    n = len(chunk_group)
    if n < 4:
        return {
            "gto_drift_vpip": 0.0, "gto_drift_pfr": 0.0, "gto_drift_agg": 0.0,
            "gto_drift_fold": 0.0, "gto_drift_bet_size": 0.0,
            "gto_drift_composite": 0.0,
        }

    mid = n // 2
    first_half = chunk_group[:mid]
    second_half = chunk_group[mid:]

    def half_stats(hands: list[dict]) -> dict[str, float]:
        vpip_hands, pfr_hands, hand_count = 0, 0, len(hands)
        agg_actions, total_actions, fold_actions = 0, 0, 0
        bet_ratios: list[float] = []
        for hand in hands:
            preflop = [a for a in hand.get("actions", []) if a.get("street") == "preflop"]
            hero_pre = [a for a in preflop if a.get("actor_seat") == hero]
            if any(a.get("action_type") in VOLUNTARY_PREFLOP for a in hero_pre):
                vpip_hands += 1
            if any(a.get("action_type") in {"raise", "allin"} for a in hero_pre):
                pfr_hands += 1
            for a in hand.get("actions", []):
                if a.get("actor_seat") != hero:
                    continue
                total_actions += 1
                atype = a.get("action_type", "")
                if atype in AGGRESSIVE:
                    agg_actions += 1
                    pot = numeric(a.get("pot_before"))
                    amt = numeric(a.get("amount"))
                    if pot > 0 and amt > 0 and atype != "allin":
                        bet_ratios.append(amt / pot)
                elif atype == "fold":
                    fold_actions += 1
        return {
            "vpip": safe_div(vpip_hands, hand_count),
            "pfr": safe_div(pfr_hands, hand_count),
            "agg": safe_div(agg_actions, total_actions),
            "fold": safe_div(fold_actions, total_actions),
            "bet_size": float(mean(bet_ratios)) if bet_ratios else 0.0,
        }

    f = half_stats(first_half)
    s = half_stats(second_half)

    diffs = {k: abs(s[k] - f[k]) for k in f}
    composite = float(mean(diffs.values()))

    return {
        "gto_drift_vpip": diffs["vpip"],
        "gto_drift_pfr": diffs["pfr"],
        "gto_drift_agg": diffs["agg"],
        "gto_drift_fold": diffs["fold"],
        "gto_drift_bet_size": diffs["bet_size"],
        "gto_drift_composite": composite,
        # Direction of drift: positive = loosening (human tilt), negative = tightening
        "gto_drift_vpip_signed": s["vpip"] - f["vpip"],
        "gto_drift_agg_signed": s["agg"] - f["agg"],
    }


# ── 2. Check/bet balance per street ───────────────────────────────────────

def _check_bet_balance_features(chunk_group: list[dict[str, Any]], hero: int | None) -> dict[str, float]:
    """
    GTO solvers prescribe specific check/bet frequency per street.
    Bots hit these frequencies precisely; humans deviate significantly.
    GTO typical: flop ~55% bet when in position, ~35% OOP.
    Key signal: hero's check/bet ratio deviates very little from a fixed value.
    Also measure balance between value bets and bluffs via bet-sizing consistency.
    """
    street_check: dict[str, int] = {s: 0 for s in STREETS}
    street_bet: dict[str, int] = {s: 0 for s in STREETS}
    # Track per-hand check/bet decisions to measure consistency
    per_hand_bet_rate: dict[str, list[float]] = {s: [] for s in STREETS}

    for hand in chunk_group:
        by_street: dict[str, list] = defaultdict(list)
        for a in hand.get("actions", []):
            by_street[str(a.get("street",""))].append(a)

        for street in ("flop", "turn", "river"):
            hero_acts = [a for a in by_street[street] if a.get("actor_seat") == hero]
            if not hero_acts:
                continue
            n_bet = sum(1 for a in hero_acts if a.get("action_type") in AGGRESSIVE)
            n_check = sum(1 for a in hero_acts if a.get("action_type") == "check")
            street_bet[street] += n_bet
            street_check[street] += n_check
            total = n_bet + n_check
            if total > 0:
                per_hand_bet_rate[street].append(safe_div(n_bet, total))

    features: dict[str, float] = {}
    for street in ("flop", "turn", "river"):
        total = street_bet[street] + street_check[street]
        bet_rate = safe_div(street_bet[street], total)
        features[f"gto_bet_rate_{street}"] = bet_rate
        # Distance from 0.5 balance: pure GTO aims for a mix; measure deviation
        features[f"gto_bet_imbalance_{street}"] = abs(bet_rate - 0.5)
        # Consistency of bet/check decisions across hands (bots: near-zero std)
        rates = per_hand_bet_rate[street]
        features[f"gto_bet_rate_std_{street}"] = float(pstdev(rates)) if len(rates) > 1 else 0.0
        features[f"gto_bet_rate_cv_{street}"] = safe_div(
            float(pstdev(rates)) if len(rates) > 1 else 0.0,
            max(abs(float(mean(rates)) if rates else 0.0), 1e-9)
        )

    return features


# ── 3. Probe-bet (donk-bet) frequency when checked to ────────────────────

def _probe_bet_features(chunk_group: list[dict[str, Any]], hero: int | None) -> dict[str, float]:
    """
    GTO probing: when villain checks to hero, the solver prescribes a specific
    bet frequency (often ~30-50% on each street). Bots hit this exactly.
    Humans probe inconsistently — sometimes never bet when checked to, sometimes always.
    Also: when villain bets and hero is last to act, hero's raise frequency.
    """
    checked_to_hero: dict[str, int] = {s: 0 for s in STREETS}
    hero_bet_after_check: dict[str, int] = {s: 0 for s in STREETS}
    villain_bet_hero_raised: dict[str, int] = {s: 0 for s in STREETS}
    villain_bet_hero_faced: dict[str, int] = {s: 0 for s in STREETS}

    # Per-hand rates for consistency measurement
    ph_probe_rates: dict[str, list[float]] = {s: [] for s in STREETS}
    ph_raise_rates: dict[str, list[float]] = {s: [] for s in STREETS}

    for hand in chunk_group:
        by_street: dict[str, list] = defaultdict(list)
        for a in hand.get("actions", []):
            by_street[str(a.get("street",""))].append(a)

        for street in ("flop", "turn", "river"):
            acts = by_street[street]
            if not acts:
                continue

            hand_checked_to = 0
            hand_bet_after = 0
            hand_villain_bet = 0
            hand_raised = 0

            for i, a in enumerate(acts):
                seat = a.get("actor_seat")
                atype = a.get("action_type", "")

                # Was hero checked to? (villain checked, next meaningful action is hero's)
                if seat != hero and atype == "check":
                    # Check if hero acts next and what they do
                    for j in range(i + 1, len(acts)):
                        b = acts[j]
                        if b.get("actor_seat") == hero:
                            hand_checked_to += 1
                            checked_to_hero[street] += 1
                            if b.get("action_type") in AGGRESSIVE:
                                hand_bet_after += 1
                                hero_bet_after_check[street] += 1
                            break
                        break  # another non-hero acted first

                # Villain bet/raised → hero faces it
                if seat != hero and atype in AGGRESSIVE:
                    for j in range(i + 1, len(acts)):
                        b = acts[j]
                        if b.get("actor_seat") == hero:
                            hand_villain_bet += 1
                            villain_bet_hero_faced[street] += 1
                            if b.get("action_type") in AGGRESSIVE:
                                hand_raised += 1
                                villain_bet_hero_raised[street] += 1
                            break
                        break

            if hand_checked_to > 0:
                ph_probe_rates[street].append(safe_div(hand_bet_after, hand_checked_to))
            if hand_villain_bet > 0:
                ph_raise_rates[street].append(safe_div(hand_raised, hand_villain_bet))

    features: dict[str, float] = {}
    for street in ("flop", "turn", "river"):
        probe_rate = safe_div(hero_bet_after_check[street], checked_to_hero[street])
        raise_rate = safe_div(villain_bet_hero_raised[street], villain_bet_hero_faced[street])
        features[f"gto_probe_rate_{street}"] = probe_rate
        features[f"gto_raise_vs_bet_rate_{street}"] = raise_rate
        features[f"gto_checked_to_count_{street}"] = float(checked_to_hero[street])

        # Consistency (bots: very stable per-hand probe frequency)
        rates = ph_probe_rates[street]
        features[f"gto_probe_rate_std_{street}"] = float(pstdev(rates)) if len(rates) > 1 else 0.0

    return features


# ── 4. PFR/VPIP ratio stability across hands ──────────────────────────────

def _pfr_vpip_stability_features(chunk_group: list[dict[str, Any]], hero: int | None) -> dict[str, float]:
    """
    "Too consistent numbers = suspicious" — rooms flag stable VPIP/PFR/3bet.
    Per-hand PFR/VPIP ratio variance is near-zero for GTO bots; humans show
    natural session-to-session and hand-to-hand variance.
    """
    per_hand_vpip: list[float] = []
    per_hand_pfr: list[float] = []
    per_hand_pfr_vpip_ratio: list[float] = []
    per_hand_3bet: list[float] = []

    for hand in chunk_group:
        preflop = [a for a in hand.get("actions", []) if a.get("street") == "preflop"]
        hero_pre = [a for a in preflop if a.get("actor_seat") == hero]

        vpip = float(any(a.get("action_type") in VOLUNTARY_PREFLOP for a in hero_pre))
        pfr = float(any(a.get("action_type") in {"raise", "allin"} for a in hero_pre))

        # 3bet: hero raises after villain already raised
        prior_raise = False
        three_bet = 0.0
        for a in preflop:
            if a.get("actor_seat") != hero and a.get("action_type") in {"raise", "allin"}:
                prior_raise = True
            if a.get("actor_seat") == hero and a.get("action_type") in {"raise", "allin"} and prior_raise:
                three_bet = 1.0
                break

        per_hand_vpip.append(vpip)
        per_hand_pfr.append(pfr)
        per_hand_3bet.append(three_bet)
        # Ratio: meaningful only when vpip > 0
        if vpip > 0:
            per_hand_pfr_vpip_ratio.append(pfr / vpip)

    n = len(chunk_group)
    vpip_mean = float(mean(per_hand_vpip)) if per_hand_vpip else 0.0
    pfr_mean = float(mean(per_hand_pfr)) if per_hand_pfr else 0.0

    features: dict[str, float] = {
        "gto_vpip_mean": vpip_mean,
        "gto_pfr_mean": pfr_mean,
        "gto_3bet_mean": float(mean(per_hand_3bet)) if per_hand_3bet else 0.0,
        "gto_pfr_vpip_ratio": safe_div(pfr_mean, max(vpip_mean, 1e-9)),
        # Variance across hands (key bot signal: near-zero)
        "gto_vpip_std": float(pstdev(per_hand_vpip)) if len(per_hand_vpip) > 1 else 0.0,
        "gto_pfr_std": float(pstdev(per_hand_pfr)) if len(per_hand_pfr) > 1 else 0.0,
        "gto_pfr_vpip_ratio_std": float(pstdev(per_hand_pfr_vpip_ratio)) if len(per_hand_pfr_vpip_ratio) > 1 else 0.0,
        "gto_3bet_std": float(pstdev(per_hand_3bet)) if len(per_hand_3bet) > 1 else 0.0,
        # Coefficient of variation
        "gto_vpip_cv": safe_div(
            float(pstdev(per_hand_vpip)) if len(per_hand_vpip) > 1 else 0.0,
            max(vpip_mean, 1e-9)
        ),
        "gto_pfr_cv": safe_div(
            float(pstdev(per_hand_pfr)) if len(per_hand_pfr) > 1 else 0.0,
            max(pfr_mean, 1e-9)
        ),
    }
    return features


# ── 5. Villain aggression → hero response consistency ─────────────────────

def _villain_response_consistency_features(chunk_group: list[dict[str, Any]], hero: int | None) -> dict[str, float]:
    """
    Bots respond to the same situation the same way every time (deterministic GTO).
    When a villain bets on a specific street, a bot folds/calls/raises at fixed freqs.
    Measure: hero's response distribution when facing villain bets, per street.
    Low entropy in that distribution = bot.
    Also: does hero's response change based on pot size? (Humans: yes; bots: less so)
    """
    response_actions: dict[str, list[str]] = {s: [] for s in STREETS}
    response_small_pot: dict[str, list[str]] = {s: [] for s in STREETS}
    response_big_pot: dict[str, list[str]] = {s: [] for s in STREETS}

    for hand in chunk_group:
        actions = hand.get("actions", [])
        meta = hand.get("metadata", {})
        bb = numeric(meta.get("bb")) or 0.02

        for i, a in enumerate(actions):
            if a.get("actor_seat") == hero:
                continue
            if a.get("action_type") not in AGGRESSIVE:
                continue
            street = str(a.get("street", ""))
            if street not in STREETS:
                continue

            pot = numeric(a.get("pot_before"))
            pot_in_bb = pot / bb if bb > 0 else 0

            # Next hero action
            for j in range(i + 1, len(actions)):
                b = actions[j]
                if b.get("street") != street:
                    break
                if b.get("actor_seat") == hero:
                    atype = str(b.get("action_type", ""))
                    response_actions[street].append(atype)
                    if pot_in_bb < 10:
                        response_small_pot[street].append(atype)
                    else:
                        response_big_pot[street].append(atype)
                    break
                break

    features: dict[str, float] = {}
    for street in STREETS:
        acts = response_actions[street]
        ent = _entropy(Counter(acts))
        features[f"gto_response_entropy_{street}"] = ent
        features[f"gto_response_count_{street}"] = float(len(acts))

        # Fold rate when facing villain bet per street
        features[f"gto_fold_rate_vs_bet_{street}"] = safe_div(
            acts.count("fold"), len(acts)
        )
        features[f"gto_call_rate_vs_bet_{street}"] = safe_div(
            acts.count("call"), len(acts)
        )
        features[f"gto_raise_rate_vs_bet_{street}"] = safe_div(
            sum(1 for a in acts if a in {"raise","bet","allin"}), len(acts)
        )

        # Pot-size sensitivity: does hero respond differently by pot size?
        small_ent = _entropy(Counter(response_small_pot[street]))
        big_ent = _entropy(Counter(response_big_pot[street]))
        # Bots: same entropy regardless of pot; humans: more cautious in big pots
        features[f"gto_response_pot_sensitivity_{street}"] = abs(big_ent - small_ent)

    # Overall response entropy across all postflop streets
    all_post = []
    for street in ("flop", "turn", "river"):
        all_post.extend(response_actions[street])
    features["gto_postflop_response_entropy"] = _entropy(Counter(all_post))

    return features
