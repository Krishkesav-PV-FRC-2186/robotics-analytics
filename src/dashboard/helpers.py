"""Pure helpers for the Streamlit scouting dashboard."""

from __future__ import annotations

import math


def parse_team_list(raw: str) -> list[int]:
    """Parse a comma/space-separated list of FRC team numbers."""
    tokens = raw.replace(";", ",").replace(" ", ",").split(",")
    teams: list[int] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        teams.append(int(token))
    return teams


def win_probabilities(red_synergy: float, blue_synergy: float) -> tuple[float, float]:
    """Convert two 0–100 synergy scores into complementary win probabilities."""
    scale = 20.0
    red_weight = math.exp(red_synergy / scale)
    blue_weight = math.exp(blue_synergy / scale)
    total = red_weight + blue_weight
    if total == 0:
        return 0.5, 0.5
    return red_weight / total, blue_weight / total
