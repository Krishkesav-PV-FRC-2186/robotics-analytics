"""Integration tests for AnalyticsOrchestrator."""

from __future__ import annotations

import pytest

from src.pipeline.orchestrator import AnalyticsOrchestrator


@pytest.mark.asyncio
async def test_analyze_team_persists_data_and_returns_swot(
    orchestrator: AnalyticsOrchestrator,
) -> None:
    result = await orchestrator.analyze_team(
        team_number=254,
        event_key="2024caln",
        max_matches=2,
        process_video=True,
    )

    assert result.team_number == 254
    assert result.event_key == "2024caln"
    assert result.matches_processed == 2
    assert result.report.team_number == 254
    assert len(result.report.strengths) > 0


@pytest.mark.asyncio
async def test_analyze_match_strategy_returns_synergy_and_report(
    orchestrator: AnalyticsOrchestrator,
) -> None:
    await orchestrator.analyze_team(254, "2024caln", max_matches=1, process_video=False)

    result = await orchestrator.analyze_match_strategy(
        alliance_teams=[254, 1678, 118],
        opponent_teams=[971, 1323, 4414],
        event_key="2024caln",
        match_key="2024caln_qm12",
    )

    assert result.alliance_teams == [254, 1678, 118]
    assert "overall" in result.synergy
    assert result.report.title
    assert len(result.report.alliance_strengths) > 0


@pytest.mark.asyncio
async def test_analyze_picklist_ranks_candidates(
    orchestrator: AnalyticsOrchestrator,
) -> None:
    await orchestrator.analyze_team(254, "2024caln", max_matches=1, process_video=False)

    result = await orchestrator.analyze_picklist(
        event_key="2024caln",
        current_alliance=[254],
        candidates=[1678, 118, 971],
    )

    assert result.event_key == "2024caln"
    assert result.recommended_pick in [1678, 118, 971]
    assert len(result.candidate_scores) == 3
    assert result.candidate_scores[0]["rank"] == 1


def test_get_team_graph_returns_allies_and_opponents(
    orchestrator: AnalyticsOrchestrator,
) -> None:
    import asyncio

    asyncio.run(
        orchestrator.analyze_team(254, "2024caln", max_matches=1, process_video=False)
    )

    graph = orchestrator.get_team_graph(254)

    assert graph.team_number == 254
    relationships = {n.relationship for n in graph.nodes}
    assert "self" in relationships
    edge_types = {e.relationship for e in graph.edges}
    assert "allied_with" in edge_types or "opposed_with" in edge_types
