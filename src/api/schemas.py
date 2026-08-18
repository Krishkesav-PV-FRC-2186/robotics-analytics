"""Request and response schemas for the analytics API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.llm.report_generator import (
    PicklistExplanationReport,
    PreMatchStrategyReport,
    TeamSWOTReport,
)


class AnalyzeTeamRequest(BaseModel):
    team_number: int = Field(..., ge=1, description="FRC team number")
    event_key: str = Field(..., min_length=1, description="TBA event key, e.g. 2024caln")
    max_matches: int = Field(default=5, ge=1, le=50, description="Max recent matches to process")
    process_video: bool = Field(
        default=True,
        description="Run vision event processing on available match footage",
    )


class AnalyzeTeamResponse(BaseModel):
    team_number: int
    event_key: str
    matches_processed: int
    vision_events: dict[str, int]
    report: TeamSWOTReport
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalyzeMatchStrategyRequest(BaseModel):
    alliance_teams: list[int] = Field(..., min_length=1, max_length=3)
    opponent_teams: list[int] = Field(default_factory=list)
    event_key: str = ""
    match_key: str = ""
    comp_level: str = "qm"


class AnalyzeMatchStrategyResponse(BaseModel):
    alliance_teams: list[int]
    opponent_teams: list[int]
    synergy: dict[str, Any]
    report: PreMatchStrategyReport


class AnalyzePicklistRequest(BaseModel):
    event_key: str
    current_alliance: list[int] = Field(default_factory=list)
    candidates: list[int] = Field(..., min_length=1)
    target_team: int | None = None
    synergy_threshold: float = Field(default=70.0, ge=0.0, le=100.0)


class AnalyzePicklistResponse(BaseModel):
    event_key: str
    recommended_pick: int
    candidate_scores: list[dict[str, Any]]
    report: PicklistExplanationReport


class TeamGraphNode(BaseModel):
    team_number: int
    nickname: str | None = None
    alliance_count: int = 0
    relationship: str = "self"  # self, ally, opponent


class TeamGraphEdge(BaseModel):
    source: int
    target: int
    relationship: str  # allied_with, opposed_with
    weight: float = 1.0
    match_key: str | None = None


class TeamGraphResponse(BaseModel):
    team_number: int
    nodes: list[TeamGraphNode]
    edges: list[TeamGraphEdge]
    chemistry_scores: dict[str, float] = Field(default_factory=dict)
