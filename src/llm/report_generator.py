"""Structured LLM pipeline for FRC analytics report generation."""

from __future__ import annotations

import json
import os
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src.llm import prompts


# ---------------------------------------------------------------------------
# Pydantic output schemas
# ---------------------------------------------------------------------------


class GamePlan(BaseModel):
    auto_strategy: str
    teleop_priorities: list[str]
    endgame_plan: str


class PreMatchStrategyReport(BaseModel):
    title: str
    executive_summary: str
    alliance_strengths: list[str]
    alliance_weaknesses: list[str]
    game_plan: GamePlan
    counter_defense_mitigations: list[str]
    key_metrics_to_watch: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class TeamSWOTReport(BaseModel):
    team_number: int
    title: str
    strengths: list[str]
    weaknesses: list[str]
    opportunities: list[str]
    threats: list[str]
    auto_assessment: str
    teleop_assessment: str
    cycle_speed_assessment: str
    defense_vulnerability: str
    recommended_role: str
    confidence: float = Field(ge=0.0, le=1.0)


class AlternativePick(BaseModel):
    team_number: int
    reason: str


class PicklistExplanationReport(BaseModel):
    title: str
    recommended_pick: int
    executive_summary: str
    synergy_justification: str
    role_fit_explanation: str
    risk_factors: list[str]
    alternative_picks: list[AlternativePick]
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Input context models
# ---------------------------------------------------------------------------


class PreMatchContext(BaseModel):
    team_numbers: list[int]
    synergy_overall: float
    cycle_overlap_score: float
    role_complementarity_score: float
    historical_chemistry_score: float
    team_roles: dict[int, str]
    opponent_summary: list[dict[str, Any]] = Field(default_factory=list)
    counter_defense_risks: list[str] = Field(default_factory=list)
    event_key: str = ""
    match_key: str = ""
    comp_level: str = "qm"


class TeamSWOTContext(BaseModel):
    team_number: int
    auto_points: float = 0.0
    teleop_cpm: float = 0.0
    avg_cycle_time: float = 15.0
    endgame_success_rate: float = 0.0
    defense_rating: float = 0.0
    primary_role: str = "hybrid"
    recent_matches: list[dict[str, Any]] = Field(default_factory=list)
    cycle_events: list[dict[str, Any]] = Field(default_factory=list)
    defense_events: list[dict[str, Any]] = Field(default_factory=list)
    endgame_events: list[dict[str, Any]] = Field(default_factory=list)
    scouting_notes: str = ""


class PicklistContext(BaseModel):
    event_key: str
    pick_order: list[int]
    candidates: list[dict[str, Any]]
    current_alliance: list[int] = Field(default_factory=list)
    remaining_picks: int = 1
    synergy_threshold: float = 70.0
    scoring_rationale: list[dict[str, Any]] = Field(default_factory=list)
    recommended_team: int = 0


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------


class ReportGraphState(TypedDict):
    report_type: str
    system_prompt: str
    user_prompt: str
    raw_response: str
    parsed_json: dict[str, Any]
    report: dict[str, Any]
    error: str | None


# ---------------------------------------------------------------------------
# LLM client wrapper
# ---------------------------------------------------------------------------


class LLMClient:
    """Thin wrapper around Google Gemini via LangChain ChatGoogleGenerativeAI."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        temperature: float = 0.3,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get(
            "GOOGLE_API_KEY", ""
        )
        self.temperature = temperature
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "Gemini API key required. Set GEMINI_API_KEY or GOOGLE_API_KEY, "
                    "or pass api_key to LLMClient."
                )
            os.environ.setdefault("GOOGLE_API_KEY", self.api_key)
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_google_genai import ChatGoogleGenerativeAI

            self._message_types = (SystemMessage, HumanMessage)
            self._client = ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=self.api_key,
                temperature=self.temperature,
            )
        return self._client

    @staticmethod
    def _extract_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
            return "".join(parts)
        return str(content or "")

    @staticmethod
    def _strip_markdown_fence(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[-1]
            if stripped.rstrip().endswith("```"):
                stripped = stripped.rsplit("```", 1)[0]
        return stripped.strip()

    def complete(self, system: str, user: str) -> str:
        client = self._get_client()
        system_message, human_message = self._message_types
        response = client.invoke(
            [
                system_message(content=system),
                human_message(content=f"{user}\n\nRespond with valid JSON only."),
            ]
        )
        return self._strip_markdown_fence(self._extract_text(response.content)) or "{}"


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------


class ReportGenerator:
    """
    Structured LLM pipeline producing three report types via LangGraph:

    1. Pre-Match Strategy Reports
    2. Team SWOT Reports
    3. Picklist Recommendation Explanations
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        model: str = "gemini-2.0-flash",
    ) -> None:
        self.llm = llm or LLMClient(model=model)
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        graph: StateGraph = StateGraph(ReportGraphState)

        graph.add_node("call_llm", self._node_call_llm)
        graph.add_node("parse_response", self._node_parse_response)
        graph.add_node("validate_report", self._node_validate_report)

        graph.add_edge(START, "call_llm")
        graph.add_edge("call_llm", "parse_response")
        graph.add_edge("parse_response", "validate_report")
        graph.add_edge("validate_report", END)

        return graph.compile()

    def _node_call_llm(self, state: ReportGraphState) -> ReportGraphState:
        try:
            raw = self.llm.complete(state["system_prompt"], state["user_prompt"])
            return {**state, "raw_response": raw, "error": None}
        except Exception as exc:
            return {**state, "raw_response": "{}", "error": str(exc)}

    @staticmethod
    def _node_parse_response(state: ReportGraphState) -> ReportGraphState:
        if state.get("error"):
            return state
        try:
            parsed = json.loads(state["raw_response"])
            return {**state, "parsed_json": parsed, "error": None}
        except json.JSONDecodeError as exc:
            return {**state, "parsed_json": {}, "error": f"JSON parse error: {exc}"}

    @staticmethod
    def _node_validate_report(state: ReportGraphState) -> ReportGraphState:
        if state.get("error"):
            return state
        schema_map: dict[str, type[BaseModel]] = {
            "pre_match": PreMatchStrategyReport,
            "swot": TeamSWOTReport,
            "picklist": PicklistExplanationReport,
        }
        model_cls = schema_map.get(state["report_type"])
        if model_cls is None:
            return {**state, "error": f"Unknown report type: {state['report_type']}"}
        try:
            validated = model_cls.model_validate(state["parsed_json"])
            return {**state, "report": validated.model_dump(), "error": None}
        except Exception as exc:
            return {**state, "error": f"Validation error: {exc}"}

    def _run_pipeline(
        self,
        report_type: Literal["pre_match", "swot", "picklist"],
        system: str,
        user: str,
    ) -> dict[str, Any]:
        initial: ReportGraphState = {
            "report_type": report_type,
            "system_prompt": system,
            "user_prompt": user,
            "raw_response": "",
            "parsed_json": {},
            "report": {},
            "error": None,
        }
        final = self._graph.invoke(initial)
        if final.get("error"):
            raise RuntimeError(f"Report generation failed: {final['error']}")
        return final["report"]

    def generate_pre_match_strategy(
        self, context: PreMatchContext
    ) -> PreMatchStrategyReport:
        user_prompt = prompts.PRE_MATCH_STRATEGY_USER.format(
            team_numbers=context.team_numbers,
            synergy_overall=context.synergy_overall,
            cycle_overlap_score=context.cycle_overlap_score,
            role_complementarity_score=context.role_complementarity_score,
            historical_chemistry_score=context.historical_chemistry_score,
            team_roles=prompts.format_team_roles(context.team_roles),
            opponent_summary=prompts.format_opponent_summary(context.opponent_summary),
            counter_defense_risks=prompts.format_counter_defense_risks(
                context.counter_defense_risks
            ),
            event_key=context.event_key,
            match_key=context.match_key,
            comp_level=context.comp_level,
        )
        report = self._run_pipeline("pre_match", prompts.SYSTEM_ANALYST, user_prompt)
        return PreMatchStrategyReport.model_validate(report)

    def generate_team_swot(self, context: TeamSWOTContext) -> TeamSWOTReport:
        user_prompt = prompts.TEAM_SWOT_USER.format(
            team_number=context.team_number,
            auto_points=context.auto_points,
            teleop_cpm=context.teleop_cpm,
            avg_cycle_time=context.avg_cycle_time,
            endgame_success_rate=context.endgame_success_rate * 100,
            defense_rating=context.defense_rating,
            primary_role=context.primary_role,
            recent_matches=prompts.format_recent_matches(context.recent_matches),
            cycle_events_summary=prompts.format_event_summary(
                context.cycle_events, "Cycle Events"
            ),
            defense_events_summary=prompts.format_event_summary(
                context.defense_events, "Defense Events"
            ),
            endgame_events_summary=prompts.format_event_summary(
                context.endgame_events, "Endgame Events"
            ),
            scouting_notes=context.scouting_notes or "No scouting notes provided.",
        )
        report = self._run_pipeline("swot", prompts.SYSTEM_ANALYST, user_prompt)
        return TeamSWOTReport.model_validate(report)

    def generate_picklist_explanation(
        self, context: PicklistContext
    ) -> PicklistExplanationReport:
        recommended = context.recommended_team or (
            context.pick_order[0] if context.pick_order else 0
        )
        user_prompt = prompts.PICKLIST_EXPLANATION_USER.format(
            event_key=context.event_key,
            pick_order=context.pick_order,
            candidate_analysis=prompts.format_candidate_analysis(context.candidates),
            current_alliance=context.current_alliance,
            remaining_picks=context.remaining_picks,
            synergy_threshold=context.synergy_threshold,
            scoring_rationale=prompts.format_scoring_rationale(
                context.scoring_rationale
            ),
            recommended_team=recommended,
        )
        report = self._run_pipeline("picklist", prompts.SYSTEM_ANALYST, user_prompt)
        return PicklistExplanationReport.model_validate(report)

    @staticmethod
    def build_pre_match_from_synergy(
        team_numbers: list[int],
        synergy: dict[str, Any],
        team_roles: dict[int, str],
        *,
        event_key: str = "",
        match_key: str = "",
        opponents: list[dict[str, Any]] | None = None,
        defense_risks: list[str] | None = None,
    ) -> PreMatchContext:
        """Helper to build PreMatchContext from a synergy engine result dict."""
        return PreMatchContext(
            team_numbers=team_numbers,
            synergy_overall=synergy.get("overall", 0.0),
            cycle_overlap_score=synergy.get("cycle_overlap_score", 0.0),
            role_complementarity_score=synergy.get("role_complementarity_score", 0.0),
            historical_chemistry_score=synergy.get("historical_chemistry_score", 0.0),
            team_roles=team_roles,
            opponent_summary=opponents or [],
            counter_defense_risks=defense_risks or [],
            event_key=event_key,
            match_key=match_key,
        )

    @staticmethod
    def build_swot_from_profile(
        team_number: int,
        profile: dict[str, Any],
        *,
        recent_matches: list[dict[str, Any]] | None = None,
        vision_events: dict[str, list[dict[str, Any]]] | None = None,
        scouting_notes: str = "",
    ) -> TeamSWOTContext:
        """Helper to build TeamSWOTContext from analytics profile data."""
        vision_events = vision_events or {}
        return TeamSWOTContext(
            team_number=team_number,
            auto_points=profile.get("auto_points", 0.0),
            teleop_cpm=profile.get("teleop_cpm", 0.0),
            avg_cycle_time=profile.get("avg_cycle_time", 15.0),
            endgame_success_rate=profile.get("endgame_success_rate", 0.0),
            defense_rating=profile.get("defense_rating", 0.0),
            primary_role=profile.get("primary_role", "hybrid"),
            recent_matches=recent_matches or [],
            cycle_events=vision_events.get("cycles", []),
            defense_events=vision_events.get("defense", []),
            endgame_events=vision_events.get("endgame", []),
            scouting_notes=scouting_notes,
        )
