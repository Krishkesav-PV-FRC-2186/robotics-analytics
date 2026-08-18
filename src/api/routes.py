"""FastAPI route handlers for the robotics analytics platform."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.schemas import (
    AnalyzeMatchStrategyRequest,
    AnalyzeMatchStrategyResponse,
    AnalyzePicklistRequest,
    AnalyzePicklistResponse,
    AnalyzeTeamRequest,
    AnalyzeTeamResponse,
    TeamGraphResponse,
)
from src.pipeline.orchestrator import AnalyticsOrchestrator

router = APIRouter(prefix="/api/v1")


def get_orchestrator(request: Request) -> AnalyticsOrchestrator:
    orchestrator: AnalyticsOrchestrator | None = getattr(
        request.app.state, "orchestrator", None
    )
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Analytics orchestrator not initialized")
    return orchestrator


@router.post("/analyze/team", response_model=AnalyzeTeamResponse)
async def analyze_team(
    body: AnalyzeTeamRequest,
    orchestrator: AnalyticsOrchestrator = Depends(get_orchestrator),
) -> AnalyzeTeamResponse:
    """
    Fetch TBA metadata, run video event processing, update Postgres/Neo4j,
    and return a Team SWOT Report.
    """
    try:
        return await orchestrator.analyze_team(
            team_number=body.team_number,
            event_key=body.event_key,
            max_matches=body.max_matches,
            process_video=body.process_video,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/analyze/match-strategy", response_model=AnalyzeMatchStrategyResponse)
async def analyze_match_strategy(
    body: AnalyzeMatchStrategyRequest,
    orchestrator: AnalyticsOrchestrator = Depends(get_orchestrator),
) -> AnalyzeMatchStrategyResponse:
    """
    Compute alliance synergy from stored stats and graph data,
    then return a Pre-Match Strategy Report.
    """
    try:
        return await orchestrator.analyze_match_strategy(
            alliance_teams=body.alliance_teams,
            opponent_teams=body.opponent_teams,
            event_key=body.event_key,
            match_key=body.match_key,
            comp_level=body.comp_level,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/analyze/picklist", response_model=AnalyzePicklistResponse)
async def analyze_picklist(
    body: AnalyzePicklistRequest,
    orchestrator: AnalyticsOrchestrator = Depends(get_orchestrator),
) -> AnalyzePicklistResponse:
    """
    Score candidate teams against the current alliance using Neo4j synergy
    and return a Picklist Explanation Report.
    """
    try:
        return await orchestrator.analyze_picklist(
            event_key=body.event_key,
            current_alliance=body.current_alliance,
            candidates=body.candidates,
            target_team=body.target_team,
            synergy_threshold=body.synergy_threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/teams/{team_number}/graph", response_model=TeamGraphResponse)
async def get_team_graph(
    team_number: int,
    orchestrator: AnalyticsOrchestrator = Depends(get_orchestrator),
) -> TeamGraphResponse:
    """Return the team's alliance sub-graph with chemistry scores."""
    try:
        return orchestrator.get_team_graph(team_number)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
