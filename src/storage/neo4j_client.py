"""Neo4j graph client for FRC alliance relationship modeling."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from neo4j import GraphDatabase, Driver, Session


@dataclass
class AllianceEdge:
    """An ALLIED_WITH relationship between two teams."""

    team_a: int
    team_b: int
    match_key: str
    alliance_color: str
    weight: float = 1.0


@dataclass
class TeamNode:
    """Team node in the alliance graph."""

    team_number: int
    nickname: str | None = None
    alliance_count: int = 0


class Neo4jAllianceClient:
    """
    Neo4j driver for modeling FRC alliance relationships.

    Graph schema:
        (Team {team_number, nickname})-[:ALLIED_WITH {match_key, color, weight}]->(Team)

    Teams that compete on the same alliance in a match receive a
    bidirectional ALLIED_WITH edge weighted by co-appearance frequency.
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str = "neo4j",
    ) -> None:
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "password123")
        self.database = database
        self._driver: Driver | None = None

    def connect(self) -> None:
        self._driver = GraphDatabase.driver(
            self.uri, auth=(self.user, self.password)
        )

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> Neo4jAllianceClient:
        self.connect()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _session(self) -> Session:
        if self._driver is None:
            raise RuntimeError("Neo4jAllianceClient is not connected. Call connect() first.")
        return self._driver.session(database=self.database)

    def ensure_schema(self) -> None:
        """Create uniqueness constraints and indexes."""
        with self._session() as session:
            session.run(
                "CREATE CONSTRAINT team_number IF NOT EXISTS "
                "FOR (t:Team) REQUIRE t.team_number IS UNIQUE"
            )

    def upsert_team(self, team_number: int, nickname: str | None = None) -> None:
        """Create or update a Team node."""
        with self._session() as session:
            session.run(
                """
                MERGE (t:Team {team_number: $team_number})
                SET t.nickname = COALESCE($nickname, t.nickname)
                """,
                team_number=team_number,
                nickname=nickname,
            )

    def record_alliance(
        self,
        team_numbers: list[int],
        match_key: str,
        alliance_color: str,
        nicknames: dict[int, str] | None = None,
    ) -> None:
        """
        Record an alliance roster, creating ALLIED_WITH edges between
        every pair of co-allied teams.

        Args:
            team_numbers: Three (or fewer) team numbers on the alliance.
            match_key: TBA match key for provenance.
            alliance_color: 'red' or 'blue'.
            nicknames: Optional mapping of team_number → nickname.
        """
        nicknames = nicknames or {}
        for team_num in team_numbers:
            self.upsert_team(team_num, nicknames.get(team_num))

        with self._session() as session:
            for i, team_a in enumerate(team_numbers):
                for team_b in team_numbers[i + 1 :]:
                    session.run(
                        """
                        MATCH (a:Team {team_number: $team_a})
                        MATCH (b:Team {team_number: $team_b})
                        MERGE (a)-[r:ALLIED_WITH {match_key: $match_key}]->(b)
                        SET r.color = $color,
                            r.weight = COALESCE(r.weight, 0) + 1
                        MERGE (b)-[r2:ALLIED_WITH {match_key: $match_key}]->(a)
                        SET r2.color = $color,
                            r2.weight = COALESCE(r2.weight, 0) + 1
                        """,
                        team_a=team_a,
                        team_b=team_b,
                        match_key=match_key,
                        color=alliance_color,
                    )

    def get_alliance_partners(
        self, team_number: int, min_weight: float = 1.0
    ) -> list[dict[str, Any]]:
        """Return teams that have allied with the given team, sorted by weight."""
        with self._session() as session:
            result = session.run(
                """
                MATCH (t:Team {team_number: $team_number})-[r:ALLIED_WITH]-(partner:Team)
                WHERE r.weight >= $min_weight
                RETURN partner.team_number AS team_number,
                       partner.nickname AS nickname,
                       sum(r.weight) AS total_alliances
                ORDER BY total_alliances DESC
                """,
                team_number=team_number,
                min_weight=min_weight,
            )
            return [dict(record) for record in result]

    def get_alliance_edges(self, team_numbers: list[int]) -> list[AllianceEdge]:
        """Return ALLIED_WITH edges among a set of teams."""
        with self._session() as session:
            result = session.run(
                """
                MATCH (a:Team)-[r:ALLIED_WITH]->(b:Team)
                WHERE a.team_number IN $teams AND b.team_number IN $teams
                RETURN a.team_number AS team_a,
                       b.team_number AS team_b,
                       r.match_key AS match_key,
                       r.color AS alliance_color,
                       r.weight AS weight
                """,
                teams=team_numbers,
            )
            return [
                AllianceEdge(
                    team_a=rec["team_a"],
                    team_b=rec["team_b"],
                    match_key=rec["match_key"],
                    alliance_color=rec["alliance_color"],
                    weight=rec["weight"],
                )
                for rec in result
            ]

    def get_synergy_subgraph(self, team_numbers: list[int]) -> dict[str, Any]:
        """
        Extract the induced subgraph for a proposed alliance.

        Returns nodes and edges suitable for feeding into AllianceSynergyEngine.
        """
        with self._session() as session:
            nodes_result = session.run(
                """
                MATCH (t:Team)
                WHERE t.team_number IN $teams
                OPTIONAL MATCH (t)-[r:ALLIED_WITH]-()
                RETURN t.team_number AS team_number,
                       t.nickname AS nickname,
                       count(r) AS alliance_count
                """,
                teams=team_numbers,
            )
            nodes = [dict(rec) for rec in nodes_result]

            edges_result = session.run(
                """
                MATCH (a:Team)-[r:ALLIED_WITH]->(b:Team)
                WHERE a.team_number IN $teams AND b.team_number IN $teams
                RETURN a.team_number AS source,
                       b.team_number AS target,
                       sum(r.weight) AS weight
                """,
                teams=team_numbers,
            )
            edges = [dict(rec) for rec in edges_result]

        return {"nodes": nodes, "edges": edges}

    def compute_alliance_chemistry(self, team_numbers: list[int]) -> float:
        """
        Quick chemistry score based on historical co-alliance frequency.

        Returns 0.0–1.0 representing how often these teams have allied before.
        """
        if len(team_numbers) < 2:
            return 0.0

        pairs = 0
        total_weight = 0.0
        edges = self.get_alliance_edges(team_numbers)
        seen: set[tuple[int, int]] = set()

        for edge in edges:
            key = (min(edge.team_a, edge.team_b), max(edge.team_a, edge.team_b))
            if key not in seen:
                seen.add(key)
                pairs += 1
                total_weight += edge.weight

        max_pairs = len(team_numbers) * (len(team_numbers) - 1) // 2
        if max_pairs == 0:
            return 0.0

        coverage = pairs / max_pairs
        avg_weight = total_weight / pairs if pairs > 0 else 0.0
        return min(1.0, coverage * min(1.0, avg_weight / 5.0))
