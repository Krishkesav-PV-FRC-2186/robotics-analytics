"""Unit tests for dashboard helpers and database seeding."""

from __future__ import annotations

from src.dashboard.helpers import parse_team_list, win_probabilities
from src.ingestion.api_client import TBAClient
from src.storage.postgres_models import Event, Match, Team
from tests.fixtures.mocks import MOCK_TBA_MATCHES, MOCK_TBA_TEAM, InMemoryNeo4jClient
from scripts.seed_db import persist_event_data


def test_parse_team_list() -> None:
    assert parse_team_list("254, 1678, 118") == [254, 1678, 118]
    assert parse_team_list("254 1678") == [254, 1678]


def test_win_probabilities_sum_to_one() -> None:
    red, blue = win_probabilities(90.0, 60.0)
    assert red > blue
    assert abs((red + blue) - 1.0) < 1e-9


def test_seed_persists_teams_matches_and_alliances(tmp_path) -> None:
    client = TBAClient(api_key="test-key")
    matches = [client._parse_match(m) for m in MOCK_TBA_MATCHES]
    tba_event = {
        "key": "2024caln",
        "name": "Central Valley Regional",
        "year": 2024,
        "event_type": "1",
        "city": "Fresno",
        "state_prov": "CA",
        "country": "USA",
    }
    neo4j = InMemoryNeo4jClient()
    neo4j.connect()
    db_url = f"sqlite:///{tmp_path / 'seed.db'}"
    stats = persist_event_data(db_url, neo4j, tba_event, [MOCK_TBA_TEAM], matches)
    assert stats["teams"] == 1
    assert stats["matches"] == 2

    from src.storage.postgres_models import create_session_factory

    session_factory, _ = create_session_factory(db_url)
    with session_factory() as session:
        assert session.query(Event).count() == 1
        assert session.query(Match).count() == 2
        assert session.query(Team).count() >= 1

    partners = neo4j.get_alliance_partners(254)
    assert any(p["team_number"] == 1678 for p in partners)
