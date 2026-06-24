"""Advanced bot-tell features derived from research literature.

Implements:
1. 4-bet/5-bet frequency ratios (GTO bots have precise, consistent 4bet frequencies)
2. SPR-dependent aggression (bots play identically at all SPRs; humans adjust)
3. Multiway vs heads-up aggression difference (bots apply HU GTO in multiway pots)
4. Per-street action entropy (bots show lower entropy — more deterministic choices)
5. Villain fold-to-hero-bet rate (indirect measure of hero's credibility/exploitation)
6. Positional aggression differential (bots have perfect positional awareness or none)
7. Preflop raise sequence depth distribution (bots trigger deeper raise wars at correct freqs)
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


def extract_advanced_tell_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    features: dict[str, float] = {}
    features.update(_fourbet_features(chunk_group))
    features.update(_spr_features(chunk_group))
    features.update(_multiway_hu_features(chunk_group))
    features.update(_street_entropy_features(chunk_group))
    features.update(_villain_fold_features(chunk_group))
    features.update(_positional_features(chunk_group))
    features.update(_raise_depth_features(chunk_group))
    return features


# ── helpers ────────────────────────────────────────────────────────────────

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


# ── 1. 4-bet / 5-bet frequency ─────────────────────────────────────────────

def _fourbet_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    """
    Bots trained on GTO solvers have precise 4bet frequencies (~10-15% vs 3bets).
    Humans rarely 4-bet and do so inconsistently.
    Key bot tell: a bot will 4bet exactly the solver-optimal % of the time.
    """
    hero = _hero_seat(chunk_group)
    threebet_faced = 0    # times hero faces a 3bet (villain raised after hero's raise)
    fourbet_by_hero = 0
    fourbet_faced = 0     # times hero faces a 4bet
    fivebet_by_hero = 0
    total_hands = len(chunk_group)

    for hand in chunk_group:
        preflop = [a for a in hand.get("actions", []) if a.get("street") == "preflop"]
        raise_number = 0  # running count of total raises so far
        hero_raised = False
        hero_raise_at = -1

        for a in preflop:
            atype = a.get("action_type", "")
            seat = a.get("actor_seat")

            if atype in AGGRESSIVE:
                raise_number += 1
                if seat == hero:
                    hero_raised = True
                    hero_raise_at = raise_number
                    if raise_number == 3:  # hero is 4-betting
                        fourbet_by_hero += 1
                    elif raise_number == 5:
                        fivebet_by_hero += 1
                else:
                    if raise_number == 3 and hero_raised and hero_raise_at == 2:
                        # hero raised (2bet or 3bet), villain now 4bets
                        threebet_faced += 1  # actually villain 3bet hero's open
                    if raise_number == 2 and hero_raised and hero_raise_at == 1:
                        threebet_faced += 1
                    if raise_number == 4 and hero_raise_at == 3:
                        fourbet_faced += 1

    return {
        "at_hero_4bet_rate": safe_div(fourbet_by_hero, max(threebet_faced, 1)),
        "at_hero_5bet_rate": safe_div(fivebet_by_hero, max(fourbet_faced, 1)),
        "at_hero_4bet_count": float(fourbet_by_hero),
        "at_hero_5bet_count": float(fivebet_by_hero),
        "at_3bet_faced_count": float(threebet_faced),
        "at_4bet_faced_count": float(fourbet_faced),
        # Normalised per hand
        "at_hero_4bet_per_hand": safe_div(fourbet_by_hero, total_hands),
        "at_hero_5bet_per_hand": safe_div(fivebet_by_hero, total_hands),
    }


# ── 2. SPR-dependent aggression ────────────────────────────────────────────

def _spr_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    """
    SPR (Stack-to-Pot Ratio) dramatically changes optimal strategy.
    Humans play very differently at SPR<3 vs SPR>10; bots apply the same GTO
    template. Measure: hero's aggression rate stratified by SPR bucket.
    Also: variance of hero's aggression across SPR buckets (bots=near-zero).
    """
    hero = _hero_seat(chunk_group)
    # SPR buckets: shallow (<3), medium (3-10), deep (>10)
    bucket_agg: dict[str, list[float]] = {"shallow": [], "medium": [], "deep": []}
    bucket_call: dict[str, list[float]] = {"shallow": [], "medium": [], "deep": []}

    for hand in chunk_group:
        meta = hand.get("metadata", {})
        bb = numeric(meta.get("bb")) or 0.02
        players = hand.get("players", [])
        hero_player = next((p for p in players if p.get("seat") == hero), None)
        if not hero_player:
            continue

        hero_stack = numeric(hero_player.get("starting_stack"))
        if hero_stack <= 0:
            continue

        # Find the pot at the start of the flop to compute SPR
        actions = hand.get("actions", [])
        flop_actions = [a for a in actions if a.get("street") == "flop"]
        if not flop_actions:
            continue

        pot_at_flop = numeric(flop_actions[0].get("pot_before"))
        if pot_at_flop <= 0:
            continue

        spr = hero_stack / pot_at_flop
        bucket = "shallow" if spr < 3 else ("medium" if spr < 10 else "deep")

        hero_flop_acts = [a for a in flop_actions if a.get("actor_seat") == hero]
        if not hero_flop_acts:
            continue

        n_agg = sum(1 for a in hero_flop_acts if a.get("action_type") in AGGRESSIVE)
        n_call = sum(1 for a in hero_flop_acts if a.get("action_type") == "call")
        agg_rate = safe_div(n_agg, len(hero_flop_acts))
        call_rate = safe_div(n_call, len(hero_flop_acts))
        bucket_agg[bucket].append(agg_rate)
        bucket_call[bucket].append(call_rate)

    features: dict[str, float] = {}
    agg_means = []
    for bkt in ("shallow", "medium", "deep"):
        vals = bucket_agg[bkt]
        features[f"at_hero_agg_{bkt}"] = float(mean(vals)) if vals else 0.0
        features[f"at_hero_{bkt}_hand_count"] = float(len(vals))
        if vals:
            agg_means.append(float(mean(vals)))

    # Variance across SPR buckets: bots show near-zero variance (same play regardless of SPR)
    features["at_spr_agg_variance"] = float(pstdev(agg_means)) if len(agg_means) > 1 else 0.0
    features["at_spr_agg_range"] = (max(agg_means) - min(agg_means)) if len(agg_means) > 1 else 0.0

    return features


# ── 3. Multiway vs heads-up aggression ────────────────────────────────────

def _multiway_hu_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    """
    GTO solvers are optimised for heads-up. In multiway pots, humans drastically
    tighten their range; bots may continue with HU frequencies or have poorly
    calibrated multiway play. Measure the difference in hero aggression HU vs MW.
    """
    hero = _hero_seat(chunk_group)
    hu_agg, hu_total = 0, 0
    mw_agg, mw_total = 0, 0

    for hand in chunk_group:
        actions = hand.get("actions", [])
        by_street: dict[str, list] = defaultdict(list)
        for a in actions:
            by_street[str(a.get("street", ""))].append(a)

        for street in ("flop", "turn", "river"):
            street_acts = by_street[street]
            if not street_acts:
                continue
            active_seats = {a.get("actor_seat") for a in street_acts
                            if a.get("action_type") not in {"fold"}}
            is_hu = len(active_seats) <= 2
            hero_acts = [a for a in street_acts if a.get("actor_seat") == hero]
            if not hero_acts:
                continue

            n_agg = sum(1 for a in hero_acts if a.get("action_type") in AGGRESSIVE)
            if is_hu:
                hu_agg += n_agg
                hu_total += len(hero_acts)
            else:
                mw_agg += n_agg
                mw_total += len(hero_acts)

    hu_rate = safe_div(hu_agg, hu_total)
    mw_rate = safe_div(mw_agg, mw_total)
    return {
        "at_hero_hu_postflop_agg": hu_rate,
        "at_hero_mw_postflop_agg": mw_rate,
        # Humans: big positive diff (less aggressive MW); bots: near-zero diff
        "at_hero_hu_mw_agg_diff": hu_rate - mw_rate,
        "at_hero_hu_hands": float(hu_total),
        "at_hero_mw_hands": float(mw_total),
        "at_hero_mw_ratio": safe_div(mw_total, max(hu_total + mw_total, 1)),
    }


# ── 4. Per-street action entropy ───────────────────────────────────────────

def _street_entropy_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    """
    Entropy of hero's action distribution per street.
    Bots: low entropy (nearly always choose the same action in a given spot).
    Humans: higher entropy (mix folds, calls, raises with more variety).
    Also measures entropy of BET SIZES (bots repeat same sizes).
    """
    hero = _hero_seat(chunk_group)
    street_actions: dict[str, list[str]] = {s: [] for s in STREETS}
    street_bet_sizes: dict[str, list[float]] = {s: [] for s in STREETS}

    for hand in chunk_group:
        for a in hand.get("actions", []):
            if a.get("actor_seat") != hero:
                continue
            street = str(a.get("street", "unknown"))
            if street not in STREETS:
                continue
            atype = str(a.get("action_type", ""))
            street_actions[street].append(atype)
            if atype in AGGRESSIVE:
                pot = numeric(a.get("pot_before"))
                amt = numeric(a.get("amount"))
                if pot > 0 and amt > 0:
                    street_bet_sizes[street].append(round(amt / pot, 2))

    features: dict[str, float] = {}
    all_entropy = []
    for street in STREETS:
        acts = street_actions[street]
        ent = _entropy(Counter(acts)) if acts else 0.0
        features[f"at_hero_action_entropy_{street}"] = ent
        features[f"at_hero_action_count_{street}"] = float(len(acts))
        if acts:
            all_entropy.append(ent)

        # Bet size entropy: bots snap to same sizes → very low entropy
        sizes = street_bet_sizes[street]
        if sizes:
            # Quantise to 0.05 buckets for entropy
            quantised = [round(s / 0.05) * 0.05 for s in sizes]
            features[f"at_hero_bet_size_entropy_{street}"] = _entropy(Counter(
                f"{q:.2f}" for q in quantised
            ))
        else:
            features[f"at_hero_bet_size_entropy_{street}"] = 0.0

    features["at_hero_action_entropy_mean"] = float(mean(all_entropy)) if all_entropy else 0.0
    features["at_hero_action_entropy_std"] = float(pstdev(all_entropy)) if len(all_entropy) > 1 else 0.0
    return features


# ── 5. Villain fold-to-hero-bet rate ──────────────────────────────────────

def _villain_fold_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    """
    Measure how often opponents fold to hero's bets/raises.
    Bots betting GTO fractions often get called more (opponents can't be exploited);
    human bluffers get different fold equity. Also captures whether hero is bluffing
    in spots where humans normally wouldn't.
    """
    hero = _hero_seat(chunk_group)
    hero_bet_then_villain_fold = 0
    hero_bet_then_villain_call = 0
    hero_bet_then_villain_raise = 0
    total_hero_bets_faced = 0

    for hand in chunk_group:
        actions = hand.get("actions", [])
        for i, a in enumerate(actions):
            if a.get("actor_seat") != hero:
                continue
            if a.get("action_type") not in AGGRESSIVE:
                continue
            # Look at the next action by a non-hero player on the same street
            street = a.get("street")
            for j in range(i + 1, len(actions)):
                b = actions[j]
                if b.get("street") != street:
                    break
                if b.get("actor_seat") == hero:
                    continue
                total_hero_bets_faced += 1
                btype = b.get("action_type", "")
                if btype == "fold":
                    hero_bet_then_villain_fold += 1
                elif btype == "call":
                    hero_bet_then_villain_call += 1
                elif btype in AGGRESSIVE:
                    hero_bet_then_villain_raise += 1
                break

    return {
        "at_villain_fold_to_hero_bet": safe_div(hero_bet_then_villain_fold, total_hero_bets_faced),
        "at_villain_call_to_hero_bet": safe_div(hero_bet_then_villain_call, total_hero_bets_faced),
        "at_villain_raise_to_hero_bet": safe_div(hero_bet_then_villain_raise, total_hero_bets_faced),
        "at_hero_bets_faced_by_villain": float(total_hero_bets_faced),
    }


# ── 6. Positional aggression differential ─────────────────────────────────

def _positional_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    """
    In position (IP) players should be more aggressive postflop than OOP.
    Bots trained on GTO have very precise positional awareness; others may over- or
    under-adjust. Measure hero's postflop aggression when IP vs OOP.

    IP approximation: hero_seat is ≥ button_seat (acts after button in postflop)
    or hero is the button. Simplified: hero is IP when button_seat < hero_seat or
    hero_seat == button_seat.
    """
    hero = _hero_seat(chunk_group)
    ip_agg, ip_total = 0, 0
    oop_agg, oop_total = 0, 0

    for hand in chunk_group:
        meta = hand.get("metadata", {})
        button_seat = meta.get("button_seat")
        max_seats = int(meta.get("max_seats", 6))
        if button_seat is None or hero is None:
            continue

        # Determine if hero is IP: hero acts after all remaining players postflop.
        # In a 6-max game, postflop action starts left of button.
        # Hero is last to act (IP) if hero == button_seat or closest clockwise before btn.
        # Simplified: compute "distance from button" — smaller = closer to button = more IP.
        def seats_from_btn(seat: int) -> int:
            return (button_seat - seat) % max_seats

        hero_dist = seats_from_btn(hero)
        # hero_dist=0: hero is button (most IP postflop)
        # hero_dist=1: hero is cutoff
        is_ip = hero_dist <= 1  # button or cutoff approximation

        for a in hand.get("actions", []):
            if a.get("actor_seat") != hero:
                continue
            if a.get("street") not in ("flop", "turn", "river"):
                continue
            atype = a.get("action_type", "")
            if is_ip:
                ip_total += 1
                if atype in AGGRESSIVE:
                    ip_agg += 1
            else:
                oop_total += 1
                if atype in AGGRESSIVE:
                    oop_agg += 1

    ip_rate = safe_div(ip_agg, ip_total)
    oop_rate = safe_div(oop_agg, oop_total)

    return {
        "at_hero_ip_postflop_agg": ip_rate,
        "at_hero_oop_postflop_agg": oop_rate,
        # Humans: positively skewed (more aggressive IP); bots: very consistent or perfectly calibrated
        "at_hero_ip_oop_agg_diff": ip_rate - oop_rate,
        "at_hero_ip_hands": float(ip_total),
        "at_hero_oop_hands": float(oop_total),
        "at_hero_ip_ratio": safe_div(ip_total, max(ip_total + oop_total, 1)),
    }


# ── 7. Preflop raise sequence depth distribution ──────────────────────────

def _raise_depth_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    """
    Measure the distribution of preflop raise depths (2bet/3bet/4bet/5bet).
    Bots trigger 4bet/5bet wars at GTO-precise frequencies; humans are inconsistent.
    Also: bots always respond to given raise depths the same way (low entropy per depth).
    """
    hero = _hero_seat(chunk_group)
    depth_counts: Counter = Counter()  # depth at which the hand ended preflop
    hero_response_at_depth: dict[int, list[str]] = defaultdict(list)  # depth→hero's actions

    for hand in chunk_group:
        preflop = [a for a in hand.get("actions", []) if a.get("street") == "preflop"]
        raise_depth = 0
        for a in preflop:
            atype = a.get("action_type", "")
            if atype in AGGRESSIVE:
                raise_depth += 1
            if a.get("actor_seat") == hero and raise_depth > 0:
                hero_response_at_depth[raise_depth].append(atype)
        depth_counts[raise_depth] += 1

    total = sum(depth_counts.values()) or 1
    features: dict[str, float] = {
        "at_preflop_2bet_rate": depth_counts.get(1, 0) / total,
        "at_preflop_3bet_rate": depth_counts.get(2, 0) / total,
        "at_preflop_4bet_rate": depth_counts.get(3, 0) / total,
        "at_preflop_5bet_plus_rate": sum(v for k, v in depth_counts.items() if k >= 4) / total,
        "at_preflop_max_depth": float(max(depth_counts.keys(), default=0)),
        "at_preflop_depth_entropy": _entropy(Counter({str(k): v for k, v in depth_counts.items()})),
    }

    # Hero's action entropy at each raise depth (low = bot-like determinism)
    for depth in (1, 2, 3):
        acts = hero_response_at_depth[depth]
        features[f"at_hero_action_entropy_at_{depth}bet"] = _entropy(Counter(acts)) if acts else 0.0
        features[f"at_hero_count_at_{depth}bet"] = float(len(acts))

    return features
