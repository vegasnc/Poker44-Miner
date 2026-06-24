from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, pstdev
from typing import Any

from src.data.preprocessor import numeric
from src.utils.helpers import safe_div


def extract_statistical_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    hand_count = len(chunk_group)
    player_counts: list[int] = []
    street_counts: list[int] = []
    starting_stacks: list[float] = []
    pot_growth_values: list[float] = []
    hero_seats: list[float] = []

    vpip_hands = 0
    pfr_hands = 0
    three_bet_hands = 0
    steal_attempt_hands = 0
    showdown_hands = 0
    winning_hands = 0

    position_actions: dict[str, Counter[str]] = defaultdict(Counter)
    blind_actions: Counter[str] = Counter()
    late_actions: Counter[str] = Counter()
    early_actions: Counter[str] = Counter()

    for hand in chunk_group:
        players = hand.get("players", [])
        actions = hand.get("actions", [])
        player_counts.append(len(players))
        street_counts.append(len(hand.get("streets", [])) or _observed_street_count(actions))
        starting_stacks.extend(_starting_stacks(players))
        hero_seat = _hero_seat(players)
        if hero_seat is not None:
            hero_seats.append(float(hero_seat))

        preflop_actions = [action for action in actions if action.get("street") == "preflop"]
        if any(action.get("action_type") in {"call", "bet", "raise", "allin"} for action in preflop_actions):
            vpip_hands += 1
        if any(action.get("action_type") in {"raise", "allin"} for action in preflop_actions):
            pfr_hands += 1
        if _has_three_bet(preflop_actions):
            three_bet_hands += 1
        if _has_steal_attempt(preflop_actions, players):
            steal_attempt_hands += 1
        if _went_to_showdown(hand):
            showdown_hands += 1
        if _won_hand(hand):
            winning_hands += 1

        pot_growth_values.append(_pot_growth(actions))
        _collect_position_actions(players, actions, position_actions, blind_actions, late_actions, early_actions)

    features = {
        "mean_player_count": _mean(player_counts),
        "std_player_count": _std(player_counts),
        "mean_street_count": _mean(street_counts),
        "std_street_count": _std(street_counts),
        "mean_starting_stack": _mean(starting_stacks),
        "std_starting_stack": _std(starting_stacks),
        "mean_pot_growth": _mean(pot_growth_values),
        "std_pot_growth": _std(pot_growth_values),
        "hero_seat_mean": _mean(hero_seats),
        "vpip": safe_div(vpip_hands, hand_count),
        "pfr": safe_div(pfr_hands, hand_count),
        "three_bet_frequency": safe_div(three_bet_hands, hand_count),
        "steal_attempt_frequency": safe_div(steal_attempt_hands, hand_count),
        "went_to_showdown_percentage": safe_div(showdown_hands, hand_count),
        "win_rate": safe_div(winning_hands, hand_count),
    }

    features.update(_position_summary("blind", blind_actions))
    features.update(_position_summary("late", late_actions))
    features.update(_position_summary("early", early_actions))
    features["position_bucket_count"] = float(len(position_actions))
    return features


def _starting_stacks(players: list[dict[str, Any]]) -> list[float]:
    stacks = []
    for player in players:
        stack = numeric(player.get("starting_stack") or player.get("stack") or player.get("chips"))
        if stack > 0:
            stacks.append(stack)
    return stacks


def _hero_seat(players: list[dict[str, Any]]) -> int | None:
    for player in players:
        if player.get("is_hero") or player.get("hero") or player.get("target"):
            seat = player.get("seat") or player.get("seat_id") or player.get("position_index")
            try:
                return int(seat)
            except (TypeError, ValueError):
                return None
    return None


def _observed_street_count(actions: list[dict[str, Any]]) -> int:
    streets = {action.get("street") for action in actions if action.get("street") and action.get("street") != "unknown"}
    return len(streets)


def _has_three_bet(preflop_actions: list[dict[str, Any]]) -> bool:
    raise_count = 0
    for action in preflop_actions:
        if action.get("action_type") in {"raise", "allin"}:
            raise_count += 1
        if raise_count >= 2:
            return True
    return False


def _has_steal_attempt(preflop_actions: list[dict[str, Any]], players: list[dict[str, Any]]) -> bool:
    seat_to_position = {_seat(player): _position_bucket(player) for player in players}
    for action in preflop_actions:
        if action.get("action_type") not in {"raise", "allin"}:
            continue
        if seat_to_position.get(action.get("actor_seat")) == "late":
            return True
    return False


def _went_to_showdown(hand: dict[str, Any]) -> bool:
    outcome = hand.get("outcome", {})
    if outcome.get("showdown") is not None:
        return bool(outcome.get("showdown"))
    actions = hand.get("actions", [])
    return any(action.get("street") == "river" for action in actions) and bool(outcome)


def _won_hand(hand: dict[str, Any]) -> bool:
    outcome = hand.get("outcome", {})
    for key in ("won", "is_winner", "hero_won"):
        if key in outcome:
            return bool(outcome[key])
    winners = outcome.get("winners") or outcome.get("winner_seats") or []
    hero = _hero_seat(hand.get("players", []))
    return hero is not None and hero in winners


def _pot_growth(actions: list[dict[str, Any]]) -> float:
    if not actions:
        return 0.0
    first = numeric(actions[0].get("pot_before"))
    last = numeric(actions[-1].get("pot_after"))
    return max(0.0, last - first)


def _collect_position_actions(
    players: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    position_actions: dict[str, Counter[str]],
    blind_actions: Counter[str],
    late_actions: Counter[str],
    early_actions: Counter[str],
) -> None:
    seat_to_position = {_seat(player): _position_bucket(player) for player in players}
    for action in actions:
        action_type = str(action.get("action_type", "unknown"))
        bucket = seat_to_position.get(action.get("actor_seat"), "unknown")
        position_actions[bucket][action_type] += 1
        if bucket == "blinds":
            blind_actions[action_type] += 1
        elif bucket == "late":
            late_actions[action_type] += 1
        elif bucket == "early":
            early_actions[action_type] += 1


def _seat(player: dict[str, Any]) -> Any:
    return player.get("seat") or player.get("seat_id") or player.get("position_index")


def _position_bucket(player: dict[str, Any]) -> str:
    raw = str(player.get("position") or player.get("pos") or player.get("role") or "").lower()
    if raw in {"sb", "bb", "small_blind", "big_blind", "blind"}:
        return "blinds"
    if raw in {"button", "btn", "co", "cutoff", "hj", "lojack"}:
        return "late"
    if raw in {"utg", "utg1", "utg2", "mp", "mp1", "mp2"}:
        return "early"
    return "unknown"


def _position_summary(prefix: str, counts: Counter[str]) -> dict[str, float]:
    total = sum(counts.values())
    return {
        f"{prefix}_action_count": float(total),
        f"{prefix}_fold_rate": safe_div(counts["fold"], total),
        f"{prefix}_call_rate": safe_div(counts["call"], total),
        f"{prefix}_raise_rate": safe_div(counts["raise"] + counts["allin"], total),
        f"{prefix}_aggression_rate": safe_div(counts["bet"] + counts["raise"] + counts["allin"], total),
    }


def _mean(values: list[float] | list[int]) -> float:
    return float(mean(values)) if values else 0.0


def _std(values: list[float] | list[int]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0
