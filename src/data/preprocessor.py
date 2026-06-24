from __future__ import annotations

from typing import Any


ACTION_ALIASES = {
    "all-in": "allin",
    "all_in": "allin",
    "small blind": "blind",
    "big blind": "blind",
}


def normalize_action_type(action_type: Any) -> str:
    normalized = str(action_type or "unknown").strip().lower().replace(" ", "_")
    return ACTION_ALIASES.get(normalized, normalized)


def numeric(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        number = float(value)
        return number if number == number else default
    except (TypeError, ValueError):
        return default


def normalize_players(players: Any) -> list[dict[str, Any]]:
    if isinstance(players, dict):
        iterable = players.values()
    elif isinstance(players, list):
        iterable = players
    else:
        iterable = []
    return [player for player in iterable if isinstance(player, dict)]


def normalize_actions(hand: dict[str, Any]) -> list[dict[str, Any]]:
    actions = hand.get("actions") or []
    if not isinstance(actions, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        item = dict(action)
        item["action_type"] = normalize_action_type(item.get("action_type") or item.get("type"))
        item["street"] = str(item.get("street") or "unknown").strip().lower()
        item["amount"] = numeric(item.get("amount"))
        item["raise_to"] = numeric(item.get("raise_to"))
        item["call_to"] = numeric(item.get("call_to"))
        item["pot_before"] = numeric(item.get("pot_before"))
        item["pot_after"] = numeric(item.get("pot_after"))
        item["normalized_amount_bb"] = numeric(item.get("normalized_amount_bb"))
        item["_order"] = index
        normalized.append(item)
    return normalized


def normalize_hand(hand: Any) -> dict[str, Any]:
    if not isinstance(hand, dict):
        return {"metadata": {}, "players": [], "streets": [], "actions": [], "outcome": {}}
    return {
        "metadata": hand.get("metadata") if isinstance(hand.get("metadata"), dict) else {},
        "players": normalize_players(hand.get("players")),
        "streets": hand.get("streets") if isinstance(hand.get("streets"), list) else [],
        "actions": normalize_actions(hand),
        "outcome": hand.get("outcome") if isinstance(hand.get("outcome"), dict) else {},
    }


def normalize_chunk_group(chunk_group: Any) -> list[dict[str, Any]]:
    if not isinstance(chunk_group, list):
        return []
    return [normalize_hand(hand) for hand in chunk_group]
