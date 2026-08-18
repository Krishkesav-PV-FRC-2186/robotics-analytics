#!/usr/bin/env python3
"""Seed PostgreSQL and Neo4j with TBA match metadata for an FRC event."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any

from src.ingestion.api_client import MatchData, TBAClient
from src.storage.neo4j_client import Neo4jAllianceClient
from src.storage.postgres_models import Event, Match, Team, create_session_factory


def _upsert_team(session, tba_team: dict[str, Any]) -> Team:
    team_number = tba_team["team_number"]
    existing = session.query(Team).filter_by(team_number=team_number).first()
    if existing:
        existing.nickname = tba_team.get("nickname", existing.nickname)
        existing.city = tba_team.get("city", existing.city)
        existing.state_prov = tba_team.get("state_prov", existing.state_prov)
        existing.country = tba_team.get("country", existing.country)
        existing.website = tba_team.get("website", existing.website)
        return existing
    team = Team.from_tba(tba_team)
    session.add(team)
    session.flush()
    return team


def _ensure_team(session, team_number: int, nicknames: dict[int, str]) -> Team:
    existing = session.query(Team).filter_by(team_number=team_number).first()
    if existing:
        return existing
    team = Team(team_number=team_number, nickname=nicknames.get(team_number))
    session.add(team)
    session.flush()
    return team


def persist_event_data(
    database_url: str,
    neo4j: Neo4jAllianceClient,
    tba_event: dict[str, Any],
    tba_teams: list[dict[str, Any]],
    matches: list[MatchData],
) -> dict[str, int]:
    session_factory, _engine = create_session_factory(database_url)
    nicknames = {t["team_number"]: t.get("nickname") or "" for t in tba_teams}

    teams_written = 0
    matches_written = 0

    with session_factory() as session:
        event = session.query(Event).filter_by(event_key=tba_event["key"]).first()
        if event is None:
            event = Event.from_tba(tba_event)
            session.add(event)
            session.flush()

        for tba_team in tba_teams:
            team = _upsert_team(session, tba_team)
            if team not in event.teams:
                event.teams.append(team)
            teams_written += 1

        for match_data in matches:
            existing = session.query(Match).filter_by(match_key=match_data.key).first()
            if existing is None:
                db_match = Match.from_tba(match_data.raw, event.id)
                db_match.red_score = match_data.red_score
                db_match.blue_score = match_data.blue_score
                session.add(db_match)
                session.flush()
                matches_written += 1
            else:
                db_match = existing

            for team_num in match_data.red_teams:
                team = _ensure_team(session, team_num, nicknames)
                if team not in db_match.red_alliance:
                    db_match.red_alliance.append(team)
            for team_num in match_data.blue_teams:
                team = _ensure_team(session, team_num, nicknames)
                if team not in db_match.blue_alliance:
                    db_match.blue_alliance.append(team)

            if match_data.red_teams:
                neo4j.record_alliance(
                    match_data.red_teams, match_data.key, "red", nicknames
                )
            if match_data.blue_teams:
                neo4j.record_alliance(
                    match_data.blue_teams, match_data.key, "blue", nicknames
                )

        session.commit()

    return {"teams": teams_written, "matches": matches_written}


async def seed_event(event_key: str) -> dict[str, int]:
    api_key = os.environ.get("TBA_API_KEY", "")
    if not api_key:
        raise SystemExit("TBA_API_KEY is required to seed the database.")

    database_url = os.environ.get(
        "DATABASE_URL", "postgresql://robotics:robotics@localhost:5432/robotics"
    )

    async with TBAClient(api_key=api_key) as tba:
        tba_event = await tba.get_event(event_key)
        tba_teams = await tba.get_event_teams(event_key)
        matches = await tba.get_event_matches(event_key)

    with Neo4jAllianceClient() as neo4j:
        neo4j.ensure_schema()
        stats = persist_event_data(database_url, neo4j, tba_event, tba_teams, matches)

    stats["event_key"] = event_key  # type: ignore[assignment]
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed Postgres and Neo4j with TBA event match metadata."
    )
    parser.add_argument(
        "--event-key",
        default="2024caln",
        help="TBA event key to ingest (default: 2024caln)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stats = asyncio.run(seed_event(args.event_key))
    print(
        f"Seeded {stats['event_key']}: "
        f"{stats['teams']} teams, {stats['matches']} matches written."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
