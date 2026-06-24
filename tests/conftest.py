from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture
def sample_chunk_group() -> list[dict[str, Any]]:
    return [
        {
            "metadata": {"table": "test"},
            "players": [
                {"seat": 0, "position": "btn", "starting_stack": 100, "is_hero": True},
                {"seat": 1, "position": "bb", "starting_stack": 100},
            ],
            "streets": ["preflop", "flop"],
            "actions": [
                {"street": "preflop", "actor_seat": 0, "action_type": "raise", "amount": 3, "pot_before": 1.5, "pot_after": 4.5},
                {"street": "preflop", "actor_seat": 1, "action_type": "call", "amount": 2, "pot_before": 4.5, "pot_after": 6.5},
                {"street": "flop", "actor_seat": 1, "action_type": "check", "pot_before": 6.5, "pot_after": 6.5},
                {"street": "flop", "actor_seat": 0, "action_type": "bet", "amount": 4, "pot_before": 6.5, "pot_after": 10.5},
                {"street": "flop", "actor_seat": 1, "action_type": "fold", "pot_before": 10.5, "pot_after": 10.5},
            ],
            "outcome": {"hero_won": True},
        }
    ]
