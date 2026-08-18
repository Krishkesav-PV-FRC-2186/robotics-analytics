"""Streamlit scouting dashboard for the robotics analytics platform."""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

from src.dashboard.helpers import parse_team_list, win_probabilities

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = 120.0


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{API_URL}{path}"
    try:
        response = httpx.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        st.error(f"API error {exc.response.status_code}: {detail}")
        raise
    except httpx.RequestError as exc:
        st.error(f"Cannot reach API at {url}: {exc}")
        raise


def api_get(path: str) -> dict[str, Any]:
    url = f"{API_URL}{path}"
    try:
        response = httpx.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        st.error(f"API error {exc.response.status_code}: {detail}")
        raise
    except httpx.RequestError as exc:
        st.error(f"Cannot reach API at {url}: {exc}")
        raise


def render_swot_report(report: dict[str, Any], metadata: dict[str, Any]) -> None:
    profile = metadata.get("profile") or {}
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Auto points (avg)", f"{profile.get('auto_points', 0):.1f}")
    col2.metric("Teleop CPM", f"{profile.get('teleop_cpm', 0):.2f}")
    col3.metric("Avg cycle time (s)", f"{profile.get('avg_cycle_time', 0):.1f}")
    col4.metric("Endgame success", f"{float(profile.get('endgame_success_rate', 0)) * 100:.0f}%")

    st.caption(f"Primary role: **{profile.get('primary_role', 'unknown')}**")
    st.subheader(report.get("title", "Team SWOT"))

    left, right = st.columns(2)
    with left:
        st.markdown("**Strengths**")
        for item in report.get("strengths", []):
            st.write(f"- {item}")
        st.markdown("**Opportunities**")
        for item in report.get("opportunities", []):
            st.write(f"- {item}")
    with right:
        st.markdown("**Weaknesses**")
        for item in report.get("weaknesses", []):
            st.write(f"- {item}")
        st.markdown("**Threats**")
        for item in report.get("threats", []):
            st.write(f"- {item}")

    st.markdown("**Assessments**")
    st.write(f"**Auto:** {report.get('auto_assessment', '')}")
    st.write(f"**Teleop:** {report.get('teleop_assessment', '')}")
    st.write(f"**Cycle speed:** {report.get('cycle_speed_assessment', '')}")
    st.write(f"**Defense vulnerability:** {report.get('defense_vulnerability', '')}")
    st.info(f"Recommended role: {report.get('recommended_role', 'n/a')}")


def render_strategy_report(report: dict[str, Any]) -> None:
    st.subheader(report.get("title", "Pre-Match Strategy"))
    st.write(report.get("executive_summary", ""))
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Alliance strengths**")
        for item in report.get("alliance_strengths", []):
            st.write(f"- {item}")
    with col2:
        st.markdown("**Alliance weaknesses**")
        for item in report.get("alliance_weaknesses", []):
            st.write(f"- {item}")

    game_plan = report.get("game_plan") or {}
    st.markdown("**Game plan**")
    st.write(f"**Auto:** {game_plan.get('auto_strategy', '')}")
    st.write(f"**Endgame:** {game_plan.get('endgame_plan', '')}")
    for item in game_plan.get("teleop_priorities", []):
        st.write(f"- {item}")

    st.markdown("**Counter-defense mitigations**")
    for item in report.get("counter_defense_mitigations", []):
        st.write(f"- {item}")


def tab_team_swot() -> None:
    st.header("Team SWOT Explorer")
    st.caption("Fetch TBA metadata, process recent matches, and generate an LLM SWOT report.")

    with st.form("swot_form"):
        team_number = st.number_input("Team number", min_value=1, value=254, step=1)
        event_key = st.text_input("Event key", value="2024caln")
        max_matches = st.slider("Matches to analyze", min_value=1, max_value=20, value=5)
        process_video = st.checkbox("Run vision event processing", value=False)
        submitted = st.form_submit_button("Generate SWOT")

    if not submitted:
        return

    with st.spinner("Analyzing team..."):
        data = api_post(
            "/api/v1/analyze/team",
            {
                "team_number": int(team_number),
                "event_key": event_key,
                "max_matches": int(max_matches),
                "process_video": process_video,
            },
        )

    events = data.get("vision_events") or {}
    st.success(
        f"Processed {data.get('matches_processed', 0)} matches — "
        f"cycles={events.get('cycles', 0)}, defense={events.get('defense', 0)}, "
        f"endgame={events.get('endgame', 0)}"
    )
    render_swot_report(data.get("report") or {}, data.get("metadata") or {})


def tab_match_strategy() -> None:
    st.header("Match Strategy Room")
    st.caption("Compare Red vs Blue alliances for synergy, win probability, and pre-match strategy.")

    with st.form("strategy_form"):
        event_key = st.text_input("Event key", value="2024caln")
        match_key = st.text_input("Match key (optional)", value="")
        red_raw = st.text_input("Red alliance teams", value="254, 1678, 118")
        blue_raw = st.text_input("Blue alliance teams", value="971, 1323, 4414")
        submitted = st.form_submit_button("Generate strategy")

    if not submitted:
        return

    red_teams = parse_team_list(red_raw)
    blue_teams = parse_team_list(blue_raw)
    if not red_teams or not blue_teams:
        st.warning("Enter at least one team number on each alliance.")
        return

    with st.spinner("Scoring both alliances..."):
        red_result = api_post(
            "/api/v1/analyze/match-strategy",
            {
                "alliance_teams": red_teams,
                "opponent_teams": blue_teams,
                "event_key": event_key,
                "match_key": match_key,
            },
        )
        blue_result = api_post(
            "/api/v1/analyze/match-strategy",
            {
                "alliance_teams": blue_teams,
                "opponent_teams": red_teams,
                "event_key": event_key,
                "match_key": match_key,
            },
        )

    red_synergy = float((red_result.get("synergy") or {}).get("overall", 0))
    blue_synergy = float((blue_result.get("synergy") or {}).get("overall", 0))
    red_win, blue_win = win_probabilities(red_synergy, blue_synergy)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Red synergy", f"{red_synergy:.1f}")
        st.metric("Red win probability", f"{red_win * 100:.1f}%")
        st.progress(min(1.0, red_win))
    with col2:
        st.metric("Blue synergy", f"{blue_synergy:.1f}")
        st.metric("Blue win probability", f"{blue_win * 100:.1f}%")
        st.progress(min(1.0, blue_win))

    st.divider()
    st.markdown("### Red alliance report")
    render_strategy_report(red_result.get("report") or {})
    st.divider()
    st.markdown("### Blue alliance report")
    render_strategy_report(blue_result.get("report") or {})


def tab_picklist() -> None:
    st.header("Alliance Picklist Builder")
    st.caption("Rank playoff partner candidates by Neo4j chemistry / synergy and explain the pick.")

    with st.form("picklist_form"):
        event_key = st.text_input("Event key", value="2024caln")
        target_team = st.number_input("Target team (alliance captain)", min_value=1, value=254, step=1)
        extra_alliance = st.text_input("Already-picked partners (optional)", value="")
        candidates_raw = st.text_input("Candidate teams", value="1678, 118, 971, 1323")
        threshold = st.slider("Synergy threshold", min_value=0, max_value=100, value=70)
        submitted = st.form_submit_button("Build picklist")

    if not submitted:
        return

    current = [int(target_team)] + parse_team_list(extra_alliance)
    candidates = parse_team_list(candidates_raw)
    if not candidates:
        st.warning("Enter at least one candidate team number.")
        return

    with st.spinner("Ranking candidates..."):
        data = api_post(
            "/api/v1/analyze/picklist",
            {
                "event_key": event_key,
                "current_alliance": current,
                "candidates": candidates,
                "target_team": int(target_team),
                "synergy_threshold": float(threshold),
            },
        )

    st.success(f"Recommended pick: Team {data.get('recommended_pick')}")
    scores = data.get("candidate_scores") or []
    if scores:
        st.dataframe(scores, use_container_width=True)

    report = data.get("report") or {}
    st.subheader(report.get("title", "Picklist explanation"))
    st.write(report.get("executive_summary", ""))
    st.write(f"**Synergy:** {report.get('synergy_justification', '')}")
    st.write(f"**Role fit:** {report.get('role_fit_explanation', '')}")
    st.markdown("**Risk factors**")
    for item in report.get("risk_factors", []):
        st.write(f"- {item}")
    st.markdown("**Alternatives**")
    for alt in report.get("alternative_picks", []):
        st.write(f"- Team {alt.get('team_number')}: {alt.get('reason')}")

    with st.expander("Target team alliance graph"):
        graph = api_get(f"/api/v1/teams/{int(target_team)}/graph")
        st.json(graph.get("chemistry_scores") or {})
        st.dataframe(graph.get("nodes") or [], use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="FRC Robotics Analytics",
        page_icon="🤖",
        layout="wide",
    )
    st.title("FRC Robotics Analytics")
    st.caption(f"Backend: `{API_URL}`")

    swot, strategy, picklist = st.tabs(
        ["Team SWOT Explorer", "Match Strategy Room", "Alliance Picklist Builder"]
    )
    with swot:
        tab_team_swot()
    with strategy:
        tab_match_strategy()
    with picklist:
        tab_picklist()


if __name__ == "__main__":
    main()
