"""Integration tests for FastAPI routes."""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestAnalyzeTeamEndpoint:
    def test_analyze_team_returns_swot_report(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/analyze/team",
            json={
                "team_number": 254,
                "event_key": "2024caln",
                "max_matches": 2,
                "process_video": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["team_number"] == 254
        assert data["matches_processed"] == 2
        assert "report" in data
        assert data["report"]["team_number"] == 254


class TestAnalyzeMatchStrategyEndpoint:
    def test_match_strategy_returns_report(self, client: TestClient) -> None:
        client.post(
            "/api/v1/analyze/team",
            json={"team_number": 254, "event_key": "2024caln", "max_matches": 1},
        )
        response = client.post(
            "/api/v1/analyze/match-strategy",
            json={
                "alliance_teams": [254, 1678, 118],
                "opponent_teams": [971, 1323, 4414],
                "event_key": "2024caln",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "synergy" in data
        assert "report" in data
        assert data["report"]["executive_summary"]


class TestAnalyzePicklistEndpoint:
    def test_picklist_returns_recommendation(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/analyze/picklist",
            json={
                "event_key": "2024caln",
                "current_alliance": [254],
                "candidates": [1678, 118],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["recommended_pick"] in [1678, 118]
        assert len(data["candidate_scores"]) == 2


class TestTeamGraphEndpoint:
    def test_team_graph_returns_nodes_and_edges(self, client: TestClient) -> None:
        client.post(
            "/api/v1/analyze/team",
            json={"team_number": 254, "event_key": "2024caln", "max_matches": 1},
        )
        response = client.get("/api/v1/teams/254/graph")
        assert response.status_code == 200
        data = response.json()
        assert data["team_number"] == 254
        assert len(data["nodes"]) >= 1
        assert any(n["relationship"] == "self" for n in data["nodes"])


class TestHealthEndpoint:
    def test_health_check(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
