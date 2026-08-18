"""Unified async pipeline orchestrating all analytics subsystems."""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Callable, Protocol

from sqlalchemy.orm import Session, sessionmaker

from src.analytics.synergy_engine import (
    AllianceSynergyEngine,
    ScoringRole,
    TeamProfile,
)
from src.api.schemas import (
    AnalyzeMatchStrategyResponse,
    AnalyzePicklistResponse,
    AnalyzeTeamResponse,
    TeamGraphEdge,
    TeamGraphNode,
    TeamGraphResponse,
)
from src.ingestion.api_client import MatchData, TBAClient
from src.llm.report_generator import (
    PicklistContext,
    PreMatchContext,
    ReportGenerator,
    TeamSWOTContext,
)
from src.storage.neo4j_client import Neo4jAllianceClient
from src.storage.postgres_models import Base, Event, Match, Team, create_session_factory
from src.vision.event_engine import (
    CycleEvent,
    DefenseEvent,
    EndgameEvent,
    VisionStateMachine,
)
from src.vision.types import TrackingResult


class TrackingProvider(Protocol):
    """Protocol for supplying tracking results (real tracker or test mock)."""

    def generate_tracking(
        self, match: MatchData, team_number: int
    ) -> list[TrackingResult]:
        ...


class AnalyticsOrchestrator:
    """
    Connects TBAClient, vision processing, Postgres, Neo4j, synergy engine,
    and ReportGenerator into a unified execution flow.
    """

    def __init__(
        self,
        database_url: str | None = None,
        tba_api_key: str | None = None,
        neo4j_client: Neo4jAllianceClient | None = None,
        report_generator: ReportGenerator | None = None,
        tracking_provider: TrackingProvider | None = None,
        synergy_engine: AllianceSynergyEngine | None = None,
    ) -> None:
        self.database_url = database_url or os.environ.get(
            "DATABASE_URL", "sqlite:///./robotics_analytics.db"
        )
        self.tba_api_key = tba_api_key or os.environ.get("TBA_API_KEY", "")
        self.neo4j = neo4j_client or Neo4jAllianceClient()
        self.reports = report_generator or ReportGenerator()
        self.synergy = synergy_engine or AllianceSynergyEngine()
        self.tracking_provider = tracking_provider

        self._session_factory, self._engine = create_session_factory(self.database_url)
        Base.metadata.create_all(self._engine)

        self._neo4j_connected = False

    def _ensure_neo4j(self) -> None:
        if not self._neo4j_connected:
            self.neo4j.connect()
            self.neo4j.ensure_schema()
            self._neo4j_connected = True

    def _session(self) -> Session:
        return self._session_factory()

    @staticmethod
    def _event_to_dict(event: CycleEvent | DefenseEvent | EndgameEvent) -> dict[str, Any]:
        data = asdict(event)
        if "event_type" in data:
            data["event_type"] = data["event_type"].value
        if "action" in data and hasattr(data["action"], "value"):
            data["action"] = data["action"].value
        return data

    def _persist_team(self, session: Session, tba_team: dict[str, Any]) -> Team:
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

    def _persist_event(self, session: Session, event_key: str) -> Event:
        existing = session.query(Event).filter_by(event_key=event_key).first()
        if existing:
            return existing
        year = int(event_key[:4]) if event_key[:4].isdigit() else None
        event = Event(event_key=event_key, year=year, name=event_key)
        session.add(event)
        session.flush()
        return event

    def _persist_match(
        self,
        session: Session,
        match_data: MatchData,
        event: Event,
        nicknames: dict[int, str],
    ) -> Match:
        existing = session.query(Match).filter_by(match_key=match_data.key).first()
        if existing:
            return existing

        db_match = Match.from_tba(match_data.raw, event.id)
        session.add(db_match)
        session.flush()

        for team_num in match_data.red_teams:
            team = session.query(Team).filter_by(team_number=team_num).first()
            if team is None:
                team = Team(team_number=team_num, nickname=nicknames.get(team_num))
                session.add(team)
                session.flush()
            if team not in db_match.red_alliance:
                db_match.red_alliance.append(team)

        for team_num in match_data.blue_teams:
            team = session.query(Team).filter_by(team_number=team_num).first()
            if team is None:
                team = Team(team_number=team_num, nickname=nicknames.get(team_num))
                session.add(team)
                session.flush()
            if team not in db_match.blue_alliance:
                db_match.blue_alliance.append(team)

        self._ensure_neo4j()
        if match_data.red_teams:
            self.neo4j.record_alliance(
                match_data.red_teams, match_data.key, "red", nicknames
            )
        if match_data.blue_teams:
            self.neo4j.record_alliance(
                match_data.blue_teams, match_data.key, "blue", nicknames
            )

        return db_match

    def _process_vision_events(
        self,
        tracking_results: list[TrackingResult],
        alliance_track_ids: set[int] | None = None,
        opponent_track_ids: set[int] | None = None,
    ) -> tuple[list[CycleEvent], list[DefenseEvent], list[EndgameEvent]]:
        state_machine = VisionStateMachine(
            zones=VisionStateMachine.default_field_zones(),
            alliance_track_ids=alliance_track_ids,
            opponent_track_ids=opponent_track_ids,
            endgame_start_time_s=0.0,
        )
        for result in tracking_results:
            state_machine.process_frame(result)
        return (
            state_machine.get_cycle_events(),
            state_machine.get_defense_events(),
            state_machine.get_endgame_events(),
        )

    @staticmethod
    def _derive_team_profile(
        team_number: int,
        cycles: list[CycleEvent],
        defenses: list[DefenseEvent],
        endgames: list[EndgameEvent],
    ) -> dict[str, Any]:
        cycle_durations = [c.cycle_duration_s for c in cycles if c.cycle_duration_s > 0]
        avg_cycle = (
            sum(cycle_durations) / len(cycle_durations) if cycle_durations else 15.0
        )
        teleop_cpm = (len(cycles) / max(avg_cycle, 1.0)) * 60.0 if cycles else 0.0
        defense_rating = min(1.0, len(defenses) * 0.15) if defenses else 0.0
        endgame_rate = min(1.0, len(endgames) * 0.25) if endgames else 0.0

        if teleop_cpm >= 4.0:
            role = ScoringRole.CYCLE.value
        elif defense_rating >= 0.5:
            role = ScoringRole.DEFENSE.value
        elif endgame_rate >= 0.5:
            role = ScoringRole.ENDGAME.value
        else:
            role = ScoringRole.HYBRID.value

        return {
            "team_number": team_number,
            "auto_points": 0.0,
            "teleop_cpm": round(teleop_cpm, 2),
            "avg_cycle_time": round(avg_cycle, 2),
            "endgame_success_rate": round(endgame_rate, 2),
            "defense_rating": round(defense_rating, 2),
            "primary_role": role,
        }

    def _get_opponents_from_db(self, team_number: int) -> list[dict[str, Any]]:
        with self._session() as session:
            team = session.query(Team).filter_by(team_number=team_number).first()
            if not team:
                return []

            opponents: dict[int, dict[str, Any]] = {}
            for match in team.red_matches:
                for opp in match.blue_alliance:
                    if opp.team_number != team_number:
                        opponents[opp.team_number] = {
                            "team_number": opp.team_number,
                            "nickname": opp.nickname,
                            "match_key": match.match_key,
                        }
            for match in team.blue_matches:
                for opp in match.red_alliance:
                    if opp.team_number != team_number:
                        opponents[opp.team_number] = {
                            "team_number": opp.team_number,
                            "nickname": opp.nickname,
                            "match_key": match.match_key,
                        }
            return list(opponents.values())

    def _build_team_profiles(
        self, team_numbers: list[int]
    ) -> list[TeamProfile]:
        profiles: list[TeamProfile] = []
        for num in team_numbers:
            role = ScoringRole.HYBRID
            profiles.append(
                TeamProfile(team_number=num, primary_role=role)
            )
        return profiles

    async def analyze_team(
        self,
        team_number: int,
        event_key: str,
        *,
        max_matches: int = 5,
        process_video: bool = True,
    ) -> AnalyzeTeamResponse:
        if not self.tba_api_key:
            raise ValueError("TBA_API_KEY is required for team analysis")

        async with TBAClient(api_key=self.tba_api_key) as tba:
            tba_team = await tba.get_team(team_number)
            matches = await tba.get_team_event_matches(team_number, event_key)
            matches = matches[:max_matches]

        nicknames: dict[int, str] = {team_number: tba_team.get("nickname", "")}
        all_cycles: list[CycleEvent] = []
        all_defenses: list[DefenseEvent] = []
        all_endgames: list[EndgameEvent] = []
        recent_match_summaries: list[dict[str, Any]] = []

        with self._session() as session:
            self._persist_team(session, tba_team)
            event = self._persist_event(session, event_key)

            for match_data in matches:
                for num in match_data.red_teams + match_data.blue_teams:
                    if num not in nicknames:
                        nicknames[num] = f"Team {num}"
                self._persist_match(session, match_data, event, nicknames)

                on_red = team_number in match_data.red_teams
                alliance = match_data.red_teams if on_red else match_data.blue_teams
                opponents = match_data.blue_teams if on_red else match_data.red_teams
                won = (
                    (on_red and match_data.winning_alliance == "red")
                    or (not on_red and match_data.winning_alliance == "blue")
                )
                score = (
                    match_data.red_score if on_red else match_data.blue_score
                )
                recent_match_summaries.append(
                    {
                        "match_key": match_data.key,
                        "won": won,
                        "score": score,
                    }
                )

                if process_video and self.tracking_provider:
                    tracking = self.tracking_provider.generate_tracking(
                        match_data, team_number
                    )
                    alliance_ids = set(range(1, len(alliance) + 1))
                    opponent_ids = set(range(len(alliance) + 1, len(alliance) + len(opponents) + 1))
                    cycles, defenses, endgames = self._process_vision_events(
                        tracking, alliance_ids, opponent_ids
                    )
                    all_cycles.extend(cycles)
                    all_defenses.extend(defenses)
                    all_endgames.extend(endgames)

            session.commit()

        profile = self._derive_team_profile(
            team_number, all_cycles, all_defenses, all_endgames
        )
        swot_context = ReportGenerator.build_swot_from_profile(
            team_number,
            profile,
            recent_matches=recent_match_summaries,
            vision_events={
                "cycles": [self._event_to_dict(e) for e in all_cycles],
                "defense": [self._event_to_dict(e) for e in all_defenses],
                "endgame": [self._event_to_dict(e) for e in all_endgames],
            },
        )
        report = self.reports.generate_team_swot(swot_context)

        return AnalyzeTeamResponse(
            team_number=team_number,
            event_key=event_key,
            matches_processed=len(matches),
            vision_events={
                "cycles": len(all_cycles),
                "defense": len(all_defenses),
                "endgame": len(all_endgames),
            },
            report=report,
            metadata={"nickname": tba_team.get("nickname"), "profile": profile},
        )

    async def analyze_match_strategy(
        self,
        alliance_teams: list[int],
        opponent_teams: list[int] | None = None,
        *,
        event_key: str = "",
        match_key: str = "",
        comp_level: str = "qm",
    ) -> AnalyzeMatchStrategyResponse:
        opponent_teams = opponent_teams or []
        self._ensure_neo4j()

        profiles = self._build_team_profiles(alliance_teams)
        chemistry = self.neo4j.compute_alliance_chemistry(alliance_teams)
        synergy_score = self.synergy.compute_synergy(profiles, chemistry)

        team_roles = {p.team_number: p.primary_role.value for p in profiles}
        opponent_summary = [
            {"team_number": n, "defense_rating": 0.5, "teleop_cpm": 3.0}
            for n in opponent_teams
        ]
        defense_risks = []
        if opponent_teams:
            defense_risks.append(
                f"Opponents {opponent_teams} may deploy counter-defense against primary cyclers"
            )

        context = ReportGenerator.build_pre_match_from_synergy(
            alliance_teams,
            synergy_score.to_dict(),
            team_roles,
            event_key=event_key,
            match_key=match_key,
            opponents=opponent_summary,
            defense_risks=defense_risks,
        )
        report = self.reports.generate_pre_match_strategy(context)

        return AnalyzeMatchStrategyResponse(
            alliance_teams=alliance_teams,
            opponent_teams=opponent_teams,
            synergy=synergy_score.to_dict(),
            report=report,
        )

    async def analyze_picklist(
        self,
        event_key: str,
        current_alliance: list[int],
        candidates: list[int],
        *,
        target_team: int | None = None,
        synergy_threshold: float = 70.0,
    ) -> AnalyzePicklistResponse:
        self._ensure_neo4j()
        candidate_scores: list[dict[str, Any]] = []

        for candidate in candidates:
            proposed = current_alliance + [candidate]
            profiles = self._build_team_profiles(proposed)
            chemistry = self.neo4j.compute_alliance_chemistry(proposed)
            score = self.synergy.compute_synergy(profiles, chemistry)
            candidate_scores.append(
                {
                    "team_number": candidate,
                    "synergy_score": score.overall,
                    "role": profiles[-1].primary_role.value,
                    "rank": 0,
                    **score.to_dict(),
                }
            )

        candidate_scores.sort(key=lambda c: c["synergy_score"], reverse=True)
        for rank, entry in enumerate(candidate_scores, start=1):
            entry["rank"] = rank

        recommended = target_team or candidate_scores[0]["team_number"]
        pick_order = [c["team_number"] for c in candidate_scores]

        context = PicklistContext(
            event_key=event_key,
            pick_order=pick_order,
            candidates=candidate_scores,
            current_alliance=current_alliance,
            remaining_picks=1,
            synergy_threshold=synergy_threshold,
            scoring_rationale=candidate_scores,
            recommended_team=recommended,
        )
        report = self.reports.generate_picklist_explanation(context)

        return AnalyzePicklistResponse(
            event_key=event_key,
            recommended_pick=recommended,
            candidate_scores=candidate_scores,
            report=report,
        )

    def get_team_graph(self, team_number: int) -> TeamGraphResponse:
        self._ensure_neo4j()

        with self._session() as session:
            team = session.query(Team).filter_by(team_number=team_number).first()

        allies = self.neo4j.get_alliance_partners(team_number)
        opponents = self._get_opponents_from_db(team_number)

        nodes: list[TeamGraphNode] = [
            TeamGraphNode(
                team_number=team_number,
                nickname=team.nickname if team else None,
                relationship="self",
            )
        ]
        edges: list[TeamGraphEdge] = []
        chemistry_scores: dict[str, float] = {}

        seen_nodes: set[int] = {team_number}
        for ally in allies:
            ally_num = ally["team_number"]
            if ally_num not in seen_nodes:
                nodes.append(
                    TeamGraphNode(
                        team_number=ally_num,
                        nickname=ally.get("nickname"),
                        alliance_count=ally.get("total_alliances", 0),
                        relationship="ally",
                    )
                )
                seen_nodes.add(ally_num)
            weight = float(ally.get("total_alliances", 1))
            edges.append(
                TeamGraphEdge(
                    source=team_number,
                    target=ally_num,
                    relationship="allied_with",
                    weight=weight,
                )
            )
            proposed = sorted([team_number, ally_num])
            chemistry_scores[f"{proposed[0]}-{proposed[1]}"] = (
                self.neo4j.compute_alliance_chemistry(proposed)
            )

        for opp in opponents:
            opp_num = opp["team_number"]
            if opp_num not in seen_nodes:
                nodes.append(
                    TeamGraphNode(
                        team_number=opp_num,
                        nickname=opp.get("nickname"),
                        relationship="opponent",
                    )
                )
                seen_nodes.add(opp_num)
            edges.append(
                TeamGraphEdge(
                    source=team_number,
                    target=opp_num,
                    relationship="opposed_with",
                    weight=1.0,
                    match_key=opp.get("match_key"),
                )
            )

        return TeamGraphResponse(
            team_number=team_number,
            nodes=nodes,
            edges=edges,
            chemistry_scores=chemistry_scores,
        )

    def close(self) -> None:
        if self._neo4j_connected:
            self.neo4j.close()
            self._neo4j_connected = False
