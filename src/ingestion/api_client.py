"""Async client for The Blue Alliance (TBA) API v3."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx

TBA_BASE_URL = "https://www.thebluealliance.com/api/v3"


@dataclass
class AllianceRoster:
    """Three-team alliance roster for a qualification or playoff match."""

    match_key: str
    alliance_color: str  # "red" or "blue"
    team_keys: list[str] = field(default_factory=list)
    team_numbers: list[int] = field(default_factory=list)


@dataclass
class MatchVideo:
    """YouTube video associated with a match."""

    match_key: str
    video_type: str
    youtube_key: str
    youtube_url: str


@dataclass
class MatchData:
    """Structured match record from TBA."""

    key: str
    event_key: str
    comp_level: str
    set_number: int
    match_number: int
    winning_alliance: str | None
    time: int | None
    red_score: int | None
    blue_score: int | None
    red_teams: list[int]
    blue_teams: list[int]
    videos: list[MatchVideo] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


class TBAClient:
    """Async HTTP client for fetching FRC event data from The Blue Alliance."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = TBA_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("TBA_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "TBA API key is required. Set TBA_API_KEY environment variable "
                "or pass api_key to TBAClient."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> TBAClient:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"X-TBA-Auth-Key": self.api_key},
            timeout=self.timeout,
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                "TBAClient must be used as an async context manager: "
                "async with TBAClient() as client: ..."
            )
        return self._client

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        client = self._ensure_client()
        response = await client.get(path, params=params)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _team_key_to_number(team_key: str) -> int:
        """Convert 'frc254' → 254."""
        return int(team_key.replace("frc", ""))

    @staticmethod
    def _number_to_team_key(team_number: int) -> str:
        """Convert 254 → 'frc254'."""
        return f"frc{team_number}"

    @staticmethod
    def _parse_videos(match_key: str, raw_match: dict[str, Any]) -> list[MatchVideo]:
        videos: list[MatchVideo] = []
        for video in raw_match.get("videos", []) or []:
            youtube_key = video.get("key", "")
            if not youtube_key:
                continue
            videos.append(
                MatchVideo(
                    match_key=match_key,
                    video_type=video.get("type", "unknown"),
                    youtube_key=youtube_key,
                    youtube_url=f"https://www.youtube.com/watch?v={youtube_key}",
                )
            )
        return videos

    def _parse_match(self, raw: dict[str, Any]) -> MatchData:
        alliances = raw.get("alliances", {})
        red_team_keys = alliances.get("red", {}).get("team_keys", [])
        blue_team_keys = alliances.get("blue", {}).get("team_keys", [])

        match_key = raw["key"]
        return MatchData(
            key=match_key,
            event_key=raw.get("event_key", ""),
            comp_level=raw.get("comp_level", ""),
            set_number=raw.get("set_number", 0),
            match_number=raw.get("match_number", 0),
            winning_alliance=raw.get("winning_alliance"),
            time=raw.get("time"),
            red_score=raw.get("red_score"),
            blue_score=raw.get("blue_score"),
            red_teams=[self._team_key_to_number(k) for k in red_team_keys],
            blue_teams=[self._team_key_to_number(k) for k in blue_team_keys],
            videos=self._parse_videos(match_key, raw),
            raw=raw,
        )

    async def get_event_matches(
        self,
        event_key: str,
        *,
        comp_level: str | None = None,
    ) -> list[MatchData]:
        """
        Fetch all matches for an event.

        Args:
            event_key: TBA event key (e.g. '2024caln').
            comp_level: Optional filter — 'qm', 'qf', 'sf', or 'f'.
        """
        params: dict[str, Any] = {}
        if comp_level:
            params["comp_level"] = comp_level

        raw_matches: list[dict[str, Any]] = await self._get(
            f"/event/{event_key}/matches", params=params or None
        )
        return [self._parse_match(m) for m in raw_matches]

    async def get_match(self, match_key: str) -> MatchData:
        """Fetch a single match by key (e.g. '2024caln_qm12')."""
        raw = await self._get(f"/match/{match_key}")
        return self._parse_match(raw)

    async def get_alliance_rosters(self, match_key: str) -> list[AllianceRoster]:
        """
        Extract red and blue alliance rosters from a match.

        Returns two AllianceRoster objects (red first, then blue).
        """
        match = await self.get_match(match_key)
        return [
            AllianceRoster(
                match_key=match.key,
                alliance_color="red",
                team_keys=[self._number_to_team_key(n) for n in match.red_teams],
                team_numbers=match.red_teams,
            ),
            AllianceRoster(
                match_key=match.key,
                alliance_color="blue",
                team_keys=[self._number_to_team_key(n) for n in match.blue_teams],
                team_numbers=match.blue_teams,
            ),
        ]

    async def get_event_alliance_rosters(
        self, event_key: str, *, comp_level: str | None = None
    ) -> list[AllianceRoster]:
        """Fetch alliance rosters for every match in an event."""
        matches = await self.get_event_matches(event_key, comp_level=comp_level)
        rosters: list[AllianceRoster] = []
        for match in matches:
            rosters.extend(
                [
                    AllianceRoster(
                        match_key=match.key,
                        alliance_color="red",
                        team_keys=[
                            self._number_to_team_key(n) for n in match.red_teams
                        ],
                        team_numbers=match.red_teams,
                    ),
                    AllianceRoster(
                        match_key=match.key,
                        alliance_color="blue",
                        team_keys=[
                            self._number_to_team_key(n) for n in match.blue_teams
                        ],
                        team_numbers=match.blue_teams,
                    ),
                ]
            )
        return rosters

    async def get_match_videos(self, match_key: str) -> list[MatchVideo]:
        """Return YouTube video URLs associated with a match."""
        match = await self.get_match(match_key)
        return match.videos

    async def get_event_videos(self, event_key: str) -> list[MatchVideo]:
        """Return all YouTube video URLs for every match in an event."""
        matches = await self.get_event_matches(event_key)
        videos: list[MatchVideo] = []
        for match in matches:
            videos.extend(match.videos)
        return videos

    async def get_event(self, event_key: str) -> dict[str, Any]:
        """Fetch metadata for a single event."""
        return await self._get(f"/event/{event_key}")

    async def get_event_teams(self, event_key: str) -> list[dict[str, Any]]:
        """Fetch team metadata for all teams attending an event."""
        return await self._get(f"/event/{event_key}/teams")

    async def get_team(self, team_number: int) -> dict[str, Any]:
        """Fetch metadata for a single team."""
        return await self._get(f"/team/frc{team_number}")

    async def get_team_event_matches(
        self, team_number: int, event_key: str
    ) -> list[MatchData]:
        """Fetch all matches for a team at a specific event."""
        all_matches = await self.get_event_matches(event_key)
        return [
            m
            for m in all_matches
            if team_number in m.red_teams or team_number in m.blue_teams
        ]
