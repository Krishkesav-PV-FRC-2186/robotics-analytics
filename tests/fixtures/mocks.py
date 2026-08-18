"""Shared test fixtures and mock implementations."""

from __future__ import annotations

import json
from typing import Any

from src.ingestion.api_client import MatchData, MatchVideo
from src.llm.report_generator import LLMClient
from src.storage.neo4j_client import Neo4jAllianceClient
from src.vision.types import TrackState, TrackingResult


MOCK_TBA_TEAM = {
    "key": "frc254",
    "team_number": 254,
    "nickname": "The Cheesy Poofs",
    "city": "San Jose",
    "state_prov": "California",
    "country": "USA",
    "website": "https://team254.com",
}

MOCK_TBA_MATCHES = [
    {
        "key": "2024caln_qm12",
        "event_key": "2024caln",
        "comp_level": "qm",
        "set_number": 1,
        "match_number": 12,
        "winning_alliance": "red",
        "time": 1700000000,
        "red_score": 85,
        "blue_score": 72,
        "alliances": {
            "red": {"team_keys": ["frc254", "frc1678", "frc118"]},
            "blue": {"team_keys": ["frc971", "frc1323", "frc4414"]},
        },
        "videos": [{"type": "youtube", "key": "abc123"}],
    },
    {
        "key": "2024caln_qm25",
        "event_key": "2024caln",
        "comp_level": "qm",
        "set_number": 1,
        "match_number": 25,
        "winning_alliance": "blue",
        "time": 1700001000,
        "red_score": 60,
        "blue_score": 78,
        "alliances": {
            "red": {"team_keys": ["frc254", "frc1678", "frc118"]},
            "blue": {"team_keys": ["frc2056", "frc1114", "frc1241"]},
        },
        "videos": [],
    },
]


def parse_mock_matches() -> list[MatchData]:
    from src.ingestion.api_client import TBAClient

    client = TBAClient(api_key="test-key")
    return [client._parse_match(m) for m in MOCK_TBA_MATCHES]


class MockLLMClient(LLMClient):
    """Returns deterministic JSON responses for each report type."""

    def __init__(self) -> None:
        super().__init__(api_key="mock-key")

    def complete(self, system: str, user: str) -> str:
        if "SWOT" in user or '"strengths"' in user:
            return json.dumps(
                {
                    "team_number": 254,
                    "title": "Team 254 SWOT Analysis",
                    "strengths": ["Fast cycle times", "Strong endgame"],
                    "weaknesses": ["Vulnerable to heavy defense"],
                    "opportunities": ["Alliance captain potential"],
                    "threats": ["Opponent counter-defense"],
                    "auto_assessment": "Consistent auto scoring",
                    "teleop_assessment": "High teleop throughput",
                    "cycle_speed_assessment": "Above average CPM",
                    "defense_vulnerability": "Moderate — struggles against swerve defense",
                    "recommended_role": "primary cycle",
                    "confidence": 0.85,
                }
            )
        if "Pre-Match" in user or "alliance_strengths" in user:
            return json.dumps(
                {
                    "title": "Pre-Match Strategy: Alliance 254/1678/118",
                    "executive_summary": "Strong cycling alliance with endgame insurance.",
                    "alliance_strengths": ["Complementary roles", "High synergy score"],
                    "alliance_weaknesses": ["Limited defense depth"],
                    "game_plan": {
                        "auto_strategy": "Score preload and exit zone",
                        "teleop_priorities": ["Maintain cycle lanes", "Avoid defense"],
                        "endgame_plan": "Climb in final 20 seconds",
                    },
                    "counter_defense_mitigations": [
                        "Spread field positioning",
                        "Designate escape routes",
                    ],
                    "key_metrics_to_watch": ["CPM", "Defense time lost"],
                    "confidence": 0.88,
                }
            )
        return json.dumps(
            {
                "title": "Picklist Recommendation",
                "recommended_pick": 1678,
                "executive_summary": "Team 1678 adds defense and chemistry.",
                "synergy_justification": "Highest synergy score among candidates.",
                "role_fit_explanation": "Defense role complements existing cyclers.",
                "risk_factors": ["Limited endgame capability"],
                "alternative_picks": [
                    {"team_number": 118, "reason": "Strong endgame backup"}
                ],
                "confidence": 0.82,
            }
        )


class InMemoryNeo4jClient(Neo4jAllianceClient):
    """Neo4j client backed by in-memory dicts for testing."""

    def __init__(self) -> None:
        super().__init__(uri="bolt://localhost:7687", user="neo4j", password="test")
        self._teams: dict[int, str | None] = {}
        self._alliances: list[dict[str, Any]] = []

    def connect(self) -> None:
        self._neo4j_connected = True

    def close(self) -> None:
        self._neo4j_connected = False

    def ensure_schema(self) -> None:
        pass

    def upsert_team(self, team_number: int, nickname: str | None = None) -> None:
        self._teams[team_number] = nickname

    def record_alliance(
        self,
        team_numbers: list[int],
        match_key: str,
        alliance_color: str,
        nicknames: dict[int, str] | None = None,
    ) -> None:
        nicknames = nicknames or {}
        for num in team_numbers:
            self.upsert_team(num, nicknames.get(num))
        for i, a in enumerate(team_numbers):
            for b in team_numbers[i + 1 :]:
                self._alliances.append(
                    {
                        "team_a": a,
                        "team_b": b,
                        "match_key": match_key,
                        "color": alliance_color,
                        "weight": 1.0,
                    }
                )

    def get_alliance_partners(
        self, team_number: int, min_weight: float = 1.0
    ) -> list[dict[str, Any]]:
        partners: dict[int, float] = {}
        for edge in self._alliances:
            if edge["team_a"] == team_number:
                partners[edge["team_b"]] = partners.get(edge["team_b"], 0) + edge["weight"]
            elif edge["team_b"] == team_number:
                partners[edge["team_a"]] = partners.get(edge["team_a"], 0) + edge["weight"]
        return [
            {
                "team_number": num,
                "nickname": self._teams.get(num),
                "total_alliances": weight,
            }
            for num, weight in sorted(partners.items(), key=lambda x: -x[1])
            if weight >= min_weight
        ]

    def get_alliance_edges(self, team_numbers: list[int]) -> list:
        from src.storage.neo4j_client import AllianceEdge

        edges = []
        for e in self._alliances:
            if e["team_a"] in team_numbers and e["team_b"] in team_numbers:
                edges.append(
                    AllianceEdge(
                        team_a=e["team_a"],
                        team_b=e["team_b"],
                        match_key=e["match_key"],
                        alliance_color=e["color"],
                        weight=e["weight"],
                    )
                )
        return edges

    def compute_alliance_chemistry(self, team_numbers: list[int]) -> float:
        if len(team_numbers) < 2:
            return 0.0
        edges = self.get_alliance_edges(team_numbers)
        max_pairs = len(team_numbers) * (len(team_numbers) - 1) // 2
        if max_pairs == 0:
            return 0.0
        pairs = len({(min(e.team_a, e.team_b), max(e.team_a, e.team_b)) for e in edges})
        return min(1.0, pairs / max_pairs)


class MockTrackingProvider:
    """Generates synthetic tracking results simulating a scoring cycle."""

    def generate_tracking(
        self, match: MatchData, team_number: int
    ) -> list[TrackingResult]:
        frames: list[TrackingResult] = []
        path = [
            (150, 600), (150, 600), (150, 600), (150, 600), (150, 600),
            (300, 600), (500, 600), (700, 600),
            (850, 150), (850, 150), (850, 150), (850, 150), (850, 150),
        ]
        for i, (x, y) in enumerate(path):
            prev = path[i - 1] if i > 0 else (x, y)
            track = TrackState(
                track_id=1,
                bbox=(x - 20, y - 20, x + 20, y + 20),
                confidence=0.9,
                class_id=0,
                class_name="robot",
                movement_vector=(x - prev[0], y - prev[1]),
                centroid=(float(x), float(y)),
            )
            frames.append(TrackingResult(frame_index=i, tracks=[track]))
        return frames


class MockTBAClient:
    """Async mock for TBAClient used in orchestrator tests."""

    def __init__(self, api_key: str = "test-key") -> None:
        self.api_key = api_key

    async def __aenter__(self) -> MockTBAClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    async def get_team(self, team_number: int) -> dict[str, Any]:
        return {**MOCK_TBA_TEAM, "team_number": team_number, "key": f"frc{team_number}"}

    async def get_team_event_matches(
        self, team_number: int, event_key: str
    ) -> list[MatchData]:
        matches = parse_mock_matches()
        return [
            m
            for m in matches
            if team_number in m.red_teams or team_number in m.blue_teams
        ]
