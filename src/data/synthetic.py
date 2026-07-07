"""Synthetic poker hand history generator using pypokerengine.

Generates labeled chunks (bot=1 / human=0) in the same format as the
benchmark dataset. Used to augment training data.

GTO bot:  tight ranges, solver-aligned bet sizes, pot-odds-correct calls/folds,
          very low variance in sizing.
Human bot: loose ranges, emotionally-influenced sizing, inconsistent pot-odds,
           occasional tilt patterns (overbets after losses).
"""
from __future__ import annotations

import random
import uuid
from typing import Any

import numpy as np

try:
    from pypokerengine.api.emulator import Emulator
    from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
    from pypokerengine.engine.hand_evaluator import HandEvaluator
    _PPE_AVAILABLE = True
except ImportError:
    _PPE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_GTO_BET_FRACTIONS = [0.33, 0.50, 0.67, 0.75, 1.00, 1.33]
_BB = 0.02
_SB = 0.01
_STARTING_STACK = 100 * _BB   # 100 BB

_STREETS = ["preflop", "flop", "turn", "river"]

# Position-based opening ranges (fraction of hands played preflop)
_GTO_VPIP = {0: 0.15, 1: 0.18, 2: 0.22, 3: 0.28, 4: 0.35, 5: 0.45}   # BTN most liberal
_HUMAN_VPIP = {0: 0.25, 1: 0.28, 2: 0.30, 3: 0.35, 4: 0.40, 5: 0.55}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _rand_stack(base: float = _STARTING_STACK, spread: float = 0.3) -> float:
    return round(base * random.uniform(1 - spread, 1 + spread), 4)


def _gto_bet_size(pot: float) -> float:
    """Pick a GTO bet size: one of the standard fractions with low noise."""
    fraction = random.choice(_GTO_BET_FRACTIONS)
    noise = random.gauss(0, 0.02)   # ±2% noise around the fraction
    fraction = max(0.25, fraction + noise)
    return round(pot * fraction, 4)


def _human_bet_size(pot: float, tilt: float = 0.0) -> float:
    """Human bet size: more variable, influenced by tilt."""
    base_fraction = random.uniform(0.30, 1.50)
    tilt_boost = tilt * random.uniform(0.5, 1.5)   # tilt → overbetting
    fraction = base_fraction + tilt_boost
    noise = random.gauss(0, 0.12)
    fraction = max(0.20, fraction + noise)
    return round(pot * fraction, 4)


def _action_id_gen():
    i = 0
    while True:
        i += 1
        yield str(i)


# ---------------------------------------------------------------------------
# Single hand simulator (no pypokerengine engine state, pure probabilistic)
# ---------------------------------------------------------------------------

def _simulate_hand(
    n_players: int,
    hero_seat: int,
    is_bot: bool,
    stacks: list[float],
    tilt: float = 0.0,
) -> dict[str, Any]:
    """
    Simulate one poker hand probabilistically (no card dealing).
    Returns a hand dict matching the benchmark format.
    """
    rng = random.Random()
    aid = _action_id_gen()

    bb = _BB
    sb = _SB
    pot = sb + bb
    actions: list[dict] = []

    # Decide if hero plays this hand
    pos = hero_seat % n_players
    vpip = _GTO_VPIP.get(pos, 0.25) if is_bot else _HUMAN_VPIP.get(pos, 0.35)

    hero_plays = rng.random() < vpip
    hero_stack = stacks[hero_seat]

    # ---- PREFLOP ----
    street = "preflop"
    active_players = list(range(n_players))
    current_pot = pot

    # Pre-action folds by early positions
    for seat in range(n_players):
        if seat == hero_seat:
            continue
        fold_prob = 0.65 if is_bot else 0.55
        if rng.random() < fold_prob:
            a = {
                "action_id": next(aid),
                "action_type": "fold",
                "actor_seat": seat + 1,
                "amount": 0,
                "call_to": None,
                "normalized_amount_bb": 0,
                "pot_after": round(current_pot, 4),
                "pot_before": round(current_pot, 4),
                "raise_to": None,
                "street": street,
            }
            actions.append(a)
            active_players.remove(seat)

    if not hero_plays:
        # Hero folds preflop
        a = {
            "action_id": next(aid),
            "action_type": "fold",
            "actor_seat": hero_seat + 1,
            "amount": 0,
            "call_to": None,
            "normalized_amount_bb": 0,
            "pot_after": round(current_pot, 4),
            "pot_before": round(current_pot, 4),
            "raise_to": None,
            "street": street,
        }
        actions.append(a)
        return _build_hand_dict(hero_seat, n_players, stacks, bb, sb, actions, showdown=False)

    # Hero raises preflop
    pot_before = current_pot
    raise_size = _gto_bet_size(pot_before) if is_bot else _human_bet_size(pot_before, tilt)
    raise_size = max(raise_size, bb * 2)
    raise_to = round(raise_size, 4)
    current_pot = round(pot_before + raise_size, 4)
    a = {
        "action_id": next(aid),
        "action_type": "raise",
        "actor_seat": hero_seat + 1,
        "amount": round(raise_size, 4),
        "call_to": None,
        "normalized_amount_bb": round(raise_size / bb, 2),
        "pot_after": current_pot,
        "pot_before": pot_before,
        "raise_to": raise_to,
        "street": street,
    }
    actions.append(a)

    # Remaining players respond
    callers = 0
    for seat in active_players:
        if seat == hero_seat:
            continue
        call_prob = 0.35 if is_bot else 0.45
        if rng.random() < call_prob:
            pot_before_call = current_pot
            current_pot = round(current_pot + raise_to * 0.5, 4)
            a = {
                "action_id": next(aid),
                "action_type": "call",
                "actor_seat": seat + 1,
                "amount": round(raise_to * 0.5, 4),
                "call_to": raise_to,
                "normalized_amount_bb": round(raise_to / bb, 2),
                "pot_after": current_pot,
                "pot_before": pot_before_call,
                "raise_to": None,
                "street": street,
            }
            actions.append(a)
            callers += 1

    # If no callers → hero wins preflop
    if callers == 0:
        return _build_hand_dict(hero_seat, n_players, stacks, bb, sb, actions, showdown=False)

    # ---- POSTFLOP streets ----
    hero_raised_last_street = True
    for street in ("flop", "turn", "river"):
        if rng.random() < 0.35:  # Chance street is not reached
            break

        # Hero continuation bet or check
        cbet_prob = 0.65 if is_bot else 0.50
        if hero_raised_last_street and rng.random() < cbet_prob:
            pot_before = current_pot
            bet_size = _gto_bet_size(pot_before) if is_bot else _human_bet_size(pot_before, tilt)
            current_pot = round(pot_before + bet_size, 4)
            a = {
                "action_id": next(aid),
                "action_type": "bet",
                "actor_seat": hero_seat + 1,
                "amount": round(bet_size, 4),
                "call_to": None,
                "normalized_amount_bb": round(bet_size / bb, 2),
                "pot_after": current_pot,
                "pot_before": pot_before,
                "raise_to": round(bet_size, 4),
                "street": street,
            }
            actions.append(a)
            hero_raised_last_street = True

            # Villain response
            villain_fold_prob = 0.55 if is_bot else 0.45
            for _ in range(callers):
                pot_before_v = current_pot
                if rng.random() < villain_fold_prob:
                    a = {
                        "action_id": next(aid),
                        "action_type": "fold",
                        "actor_seat": (hero_seat + 1) % n_players + 1,
                        "amount": 0,
                        "call_to": None,
                        "normalized_amount_bb": 0,
                        "pot_after": round(pot_before_v, 4),
                        "pot_before": round(pot_before_v, 4),
                        "raise_to": None,
                        "street": street,
                    }
                    actions.append(a)
                    callers -= 1
                    if callers == 0:
                        break
                else:
                    call_amt = round(bet_size, 4)
                    current_pot = round(current_pot + call_amt, 4)
                    a = {
                        "action_id": next(aid),
                        "action_type": "call",
                        "actor_seat": (hero_seat + 1) % n_players + 1,
                        "amount": call_amt,
                        "call_to": round(bet_size, 4),
                        "normalized_amount_bb": round(bet_size / bb, 2),
                        "pot_after": current_pot,
                        "pot_before": pot_before_v,
                        "raise_to": None,
                        "street": street,
                    }
                    actions.append(a)
        else:
            hero_raised_last_street = False

        if callers == 0:
            break

    showdown = callers > 0 and street == "river"
    return _build_hand_dict(hero_seat, n_players, stacks, bb, sb, actions, showdown=showdown)


def _build_hand_dict(
    hero_seat: int,
    n_players: int,
    stacks: list[float],
    bb: float,
    sb: float,
    actions: list[dict],
    showdown: bool,
) -> dict[str, Any]:
    players = [
        {
            "hole_cards": None,
            "player_uid": f"seat_{i + 1}",
            "seat": i + 1,
            "showed_hand": False,
            "starting_stack": round(stacks[i], 4),
        }
        for i in range(n_players)
    ]
    return {
        "actions": actions,
        "hand_id": f"syn_{uuid.uuid4().hex[:20]}",
        "metadata": {
            "ante": 0,
            "bb": bb,
            "button_seat": 0,
            "game_type": "Hold'em",
            "hand_ended_on_street": "",
            "hero_seat": hero_seat + 1,
            "limit_type": "No Limit",
            "max_seats": n_players,
            "rng_seed_commitment": None,
            "sb": sb,
        },
        "outcome": {
            "payouts": {},
            "rake": 0,
            "result_reason": "",
            "showdown": showdown,
            "total_pot": 0,
            "winners": [],
        },
        "players": players,
        "streets": [],
    }


# ---------------------------------------------------------------------------
# Chunk (session) generator
# ---------------------------------------------------------------------------

def generate_chunk(
    is_bot: bool,
    n_hands: int = 35,
    n_players: int = 6,
) -> list[dict[str, Any]]:
    """
    Generate one chunk (session) of n_hands hands for a single player.
    is_bot=True → GTO-style behaviour; False → human-style behaviour.
    """
    hero_seat = random.randint(0, n_players - 1)
    stacks = [_rand_stack() for _ in range(n_players)]
    tilt = 0.0   # humans accumulate tilt after losses
    hands = []

    for _ in range(n_hands):
        hand = _simulate_hand(
            n_players=n_players,
            hero_seat=hero_seat,
            is_bot=is_bot,
            stacks=stacks,
            tilt=tilt if not is_bot else 0.0,
        )
        hands.append(hand)

        # Human tilt: small chance of tilt after a big pot lost (proxy)
        if not is_bot:
            lost = random.random() < 0.15
            tilt = max(0.0, tilt + (0.3 if lost else -0.1))

        # Rotate hero seat occasionally (simulates table change)
        if random.random() < 0.05:
            hero_seat = random.randint(0, n_players - 1)

    return hands


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    n_chunks_per_class: int = 200,
    n_hands_per_chunk: int = 35,
    n_players: int = 6,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Returns a list of dicts: {'chunk': [...hands...], 'label': 0/1, 'source': 'synthetic'}.
    label=1 → bot, label=0 → human.
    """
    random.seed(seed)
    np.random.seed(seed)

    records = []
    for _ in range(n_chunks_per_class):
        chunk = generate_chunk(is_bot=True, n_hands=n_hands_per_chunk, n_players=n_players)
        records.append({"chunk": chunk, "label": 1, "source": "synthetic"})

    for _ in range(n_chunks_per_class):
        chunk = generate_chunk(is_bot=False, n_hands=n_hands_per_chunk, n_players=n_players)
        records.append({"chunk": chunk, "label": 0, "source": "synthetic"})

    random.shuffle(records)
    return records
