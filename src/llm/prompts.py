"""Prompt templates for FRC analytics LLM report generation."""

from __future__ import annotations

SYSTEM_ANALYST = """\
You are an expert FRC (FIRST Robotics Competition) analytics coach. \
You analyze alliance data, match statistics, and scouting reports to produce \
actionable, concise strategy documents for drive teams and alliance captains.

Rules:
- Ground every claim in the provided data; do not invent statistics.
- Use FRC terminology correctly (cycles, CPM, auto, teleop, endgame, picklist).
- Be specific about team numbers when referencing strengths or risks.
- Keep recommendations practical for a 15-second drive-team briefing.
- Output valid JSON matching the requested schema exactly.
"""

PRE_MATCH_STRATEGY_USER = """\
Generate a Pre-Match Strategy Report for the upcoming alliance.

## Alliance
Teams: {team_numbers}
Synergy Score: {synergy_overall}/100
  - Cycle Overlap: {cycle_overlap_score}/100
  - Role Complementarity: {role_complementarity_score}/100
  - Historical Chemistry: {historical_chemistry_score}/100

## Team Roles
{team_roles}

## Opponent Intel
{opponent_summary}

## Counter-Defense Risks
{counter_defense_risks}

## Match Context
Event: {event_key}
Match: {match_key}
Comp Level: {comp_level}

Respond with JSON matching this schema:
{{
  "title": "string",
  "executive_summary": "string (2-3 sentences)",
  "alliance_strengths": ["string"],
  "alliance_weaknesses": ["string"],
  "game_plan": {{
    "auto_strategy": "string",
    "teleop_priorities": ["string"],
    "endgame_plan": "string"
  }},
  "counter_defense_mitigations": ["string"],
  "key_metrics_to_watch": ["string"],
  "confidence": 0.0-1.0
}}
"""

TEAM_SWOT_USER = """\
Generate a Team SWOT Report for FRC Team {team_number}.

## Performance Metrics
- Auto Points (avg): {auto_points}
- Teleop CPM: {teleop_cpm}
- Avg Cycle Time: {avg_cycle_time}s
- Endgame Success Rate: {endgame_success_rate}%
- Defense Rating: {defense_rating}/1.0
- Primary Role: {primary_role}

## Recent Match Data
{recent_matches}

## Vision-Derived Events (last 5 matches)
{cycle_events_summary}
{defense_events_summary}
{endgame_events_summary}

## Scouting Notes
{scouting_notes}

Respond with JSON matching this schema:
{{
  "team_number": {team_number},
  "title": "string",
  "strengths": ["string"],
  "weaknesses": ["string"],
  "opportunities": ["string"],
  "threats": ["string"],
  "auto_assessment": "string",
  "teleop_assessment": "string",
  "cycle_speed_assessment": "string",
  "defense_vulnerability": "string",
  "recommended_role": "string",
  "confidence": 0.0-1.0
}}
"""

PICKLIST_EXPLANATION_USER = """\
Explain the picklist recommendation for {event_key}.

## Recommended Pick Order
{pick_order}

## Candidate Pool Analysis
{candidate_analysis}

## Alliance Build Context
Current alliance: {current_alliance}
Remaining picks: {remaining_picks}
Target synergy threshold: {synergy_threshold}/100

## Scoring Rationale
{scoring_rationale}

Respond with JSON matching this schema:
{{
  "title": "string",
  "recommended_pick": {recommended_team},
  "executive_summary": "string (2-3 sentences)",
  "synergy_justification": "string",
  "role_fit_explanation": "string",
  "risk_factors": ["string"],
  "alternative_picks": [
    {{"team_number": 0, "reason": "string"}}
  ],
  "confidence": 0.0-1.0
}}
"""


def format_team_roles(roles: dict[int, str]) -> str:
    lines = [f"- Team {num}: {role}" for num, role in sorted(roles.items())]
    return "\n".join(lines) if lines else "No role data available."


def format_opponent_summary(opponents: list[dict]) -> str:
    if not opponents:
        return "No opponent data available."
    lines = []
    for opp in opponents:
        lines.append(
            f"- Team {opp.get('team_number', '?')}: "
            f"defense rating {opp.get('defense_rating', 'N/A')}, "
            f"CPM {opp.get('teleop_cpm', 'N/A')}"
        )
    return "\n".join(lines)


def format_counter_defense_risks(risks: list[str]) -> str:
    if not risks:
        return "No elevated counter-defense risks identified."
    return "\n".join(f"- {r}" for r in risks)


def format_recent_matches(matches: list[dict]) -> str:
    if not matches:
        return "No recent match data."
    lines = []
    for m in matches:
        lines.append(
            f"- {m.get('match_key', '?')}: "
            f"{'W' if m.get('won') else 'L'} "
            f"score {m.get('score', '?')}"
        )
    return "\n".join(lines)


def format_event_summary(events: list[dict], label: str) -> str:
    if not events:
        return f"{label}: None recorded."
    lines = [f"{label}:"]
    for e in events[:10]:
        lines.append(f"  - {e}")
    return "\n".join(lines)


def format_candidate_analysis(candidates: list[dict]) -> str:
    if not candidates:
        return "No candidates analyzed."
    lines = []
    for c in candidates:
        lines.append(
            f"- Team {c['team_number']}: synergy {c.get('synergy_score', '?')}/100, "
            f"role={c.get('role', '?')}, rank={c.get('rank', '?')}"
        )
    return "\n".join(lines)


def format_scoring_rationale(scores: list[dict]) -> str:
    if not scores:
        return "No scoring breakdown available."
    lines = []
    for s in scores:
        lines.append(
            f"- Team {s['team_number']}: overall={s.get('overall', '?')}, "
            f"cycle={s.get('cycle_overlap_score', '?')}, "
            f"role={s.get('role_complementarity_score', '?')}, "
            f"chemistry={s.get('historical_chemistry_score', '?')}"
        )
    return "\n".join(lines)
