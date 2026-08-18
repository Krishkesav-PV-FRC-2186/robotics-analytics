"""Alliance synergy scoring based on cycle overlap and role complementarity."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


class ScoringRole(str, Enum):
    """Primary scoring role classification for an FRC robot."""

    CYCLE = "cycle"  # Fast game-piece cycling
    DEFENSE = "defense"  # Defensive disruption
    ENDGAME = "endgame"  # End-game climb / trap / park
    SUPPORT = "support"  # Feeder, passer, or utility
    HYBRID = "hybrid"  # Balanced multi-role


@dataclass
class CycleWindow:
    """Time interval during which a team is actively scoring."""

    team_number: int
    start_time: float  # seconds from match start
    end_time: float
    points_contributed: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def overlaps(self, other: CycleWindow) -> float:
        """Return overlap duration in seconds with another cycle window."""
        overlap_start = max(self.start_time, other.start_time)
        overlap_end = min(self.end_time, other.end_time)
        return max(0.0, overlap_end - overlap_start)


@dataclass
class TeamProfile:
    """Analytics profile for a single team used in synergy computation."""

    team_number: int
    primary_role: ScoringRole
    secondary_role: ScoringRole | None = None
    cycle_windows: list[CycleWindow] = field(default_factory=list)
    avg_cycle_time: float = 15.0  # seconds
    auto_points: float = 0.0
    teleop_cpm: float = 0.0  # cycles per minute
    endgame_success_rate: float = 0.0
    defense_rating: float = 0.0  # 0-1 scale


@dataclass
class SynergyScore:
    """Computed alliance synergy breakdown (all sub-scores 0–100)."""

    overall: float
    cycle_overlap_score: float
    role_complementarity_score: float
    historical_chemistry_score: float
    team_numbers: list[int]
    details: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "overall": round(self.overall, 2),
            "cycle_overlap_score": round(self.cycle_overlap_score, 2),
            "role_complementarity_score": round(self.role_complementarity_score, 2),
            "historical_chemistry_score": round(self.historical_chemistry_score, 2),
            "team_numbers": self.team_numbers,
            "details": {k: round(v, 2) for k, v in self.details.items()},
        }


# Role complementarity matrix: higher values = better pairing
ROLE_COMPATIBILITY: dict[tuple[ScoringRole, ScoringRole], float] = {
    (ScoringRole.CYCLE, ScoringRole.DEFENSE): 1.0,
    (ScoringRole.CYCLE, ScoringRole.ENDGAME): 0.9,
    (ScoringRole.CYCLE, ScoringRole.SUPPORT): 0.85,
    (ScoringRole.CYCLE, ScoringRole.HYBRID): 0.7,
    (ScoringRole.DEFENSE, ScoringRole.ENDGAME): 0.8,
    (ScoringRole.DEFENSE, ScoringRole.SUPPORT): 0.75,
    (ScoringRole.DEFENSE, ScoringRole.HYBRID): 0.65,
    (ScoringRole.ENDGAME, ScoringRole.SUPPORT): 0.7,
    (ScoringRole.ENDGAME, ScoringRole.HYBRID): 0.6,
    (ScoringRole.SUPPORT, ScoringRole.HYBRID): 0.55,
    (ScoringRole.CYCLE, ScoringRole.CYCLE): 0.3,
    (ScoringRole.DEFENSE, ScoringRole.DEFENSE): 0.2,
    (ScoringRole.ENDGAME, ScoringRole.ENDGAME): 0.25,
    (ScoringRole.SUPPORT, ScoringRole.SUPPORT): 0.4,
    (ScoringRole.HYBRID, ScoringRole.HYBRID): 0.5,
}


class AllianceSynergyEngine:
    """
    Compute Alliance Synergy Scores (0–100) for a proposed three-team alliance.

    The score combines three weighted factors:
      1. Cycle overlap — penalizes teams whose scoring cycles conflict in time.
      2. Role complementarity — rewards diverse, non-redundant scoring roles.
      3. Historical chemistry — optional boost from prior co-alliance data.
    """

    CYCLE_WEIGHT = 0.40
    ROLE_WEIGHT = 0.40
    CHEMISTRY_WEIGHT = 0.20

    def __init__(
        self,
        cycle_weight: float = CYCLE_WEIGHT,
        role_weight: float = ROLE_WEIGHT,
        chemistry_weight: float = CHEMISTRY_WEIGHT,
    ) -> None:
        total = cycle_weight + role_weight + chemistry_weight
        self.cycle_weight = cycle_weight / total
        self.role_weight = role_weight / total
        self.chemistry_weight = chemistry_weight / total

    @staticmethod
    def _role_pair_score(role_a: ScoringRole, role_b: ScoringRole) -> float:
        if role_a == role_b:
            key = (role_a, role_b)
        else:
            key = (min(role_a, role_b, key=lambda r: r.value),
                   max(role_a, role_b, key=lambda r: r.value))
        return ROLE_COMPATIBILITY.get(key, ROLE_COMPATIBILITY.get((key[1], key[0]), 0.5))

    def compute_cycle_overlap_score(
        self, profiles: Sequence[TeamProfile]
    ) -> tuple[float, dict[str, float]]:
        """
        Score cycle timing across the alliance.

        Lower temporal overlap between primary cyclers yields a higher score,
        because simultaneous field congestion reduces effective throughput.
        """
        if not profiles:
            return 0.0, {}

        all_windows: list[CycleWindow] = []
        for profile in profiles:
            all_windows.extend(profile.cycle_windows)

        if len(all_windows) < 2:
            avg_cycle = sum(p.avg_cycle_time for p in profiles) / len(profiles)
            stagger_potential = min(1.0, avg_cycle / 20.0)
            return stagger_potential * 100.0, {"stagger_potential": stagger_potential * 100}

        total_overlap = 0.0
        total_possible = 0.0
        pair_count = 0

        for i, window_a in enumerate(all_windows):
            for window_b in all_windows[i + 1 :]:
                if window_a.team_number == window_b.team_number:
                    continue
                overlap = window_a.overlaps(window_b)
                possible = min(window_a.duration, window_b.duration)
                total_overlap += overlap
                total_possible += possible
                pair_count += 1

        if pair_count == 0 or total_possible == 0:
            overlap_ratio = 0.0
        else:
            overlap_ratio = total_overlap / total_possible

        non_overlap = 1.0 - overlap_ratio
        score = non_overlap * 100.0

        cycle_times = [p.avg_cycle_time for p in profiles]
        if len(cycle_times) >= 2:
            spread = max(cycle_times) - min(cycle_times)
            stagger_bonus = min(15.0, spread * 1.5)
            score = min(100.0, score + stagger_bonus)

        return score, {
            "overlap_ratio": overlap_ratio * 100,
            "non_overlap_score": non_overlap * 100,
        }

    def compute_role_complementarity_score(
        self, profiles: Sequence[TeamProfile]
    ) -> tuple[float, dict[str, float]]:
        """
        Score how well team roles complement each other.

        Diverse role assignments (e.g. cycle + defense + endgame) score highest.
        """
        if len(profiles) < 2:
            return 50.0, {}

        roles = [p.primary_role for p in profiles]
        unique_roles = len(set(roles))
        diversity_bonus = (unique_roles / len(roles)) * 100.0

        pair_scores: list[float] = []
        for i, profile_a in enumerate(profiles):
            for profile_b in profiles[i + 1 :]:
                pair_scores.append(
                    self._role_pair_score(profile_a.primary_role, profile_b.primary_role)
                )

        avg_pair = sum(pair_scores) / len(pair_scores) if pair_scores else 0.5
        raw_score = 0.6 * (avg_pair * 100.0) + 0.4 * diversity_bonus

        secondary_bonus = 0.0
        for profile in profiles:
            if profile.secondary_role and profile.secondary_role != profile.primary_role:
                for other in profiles:
                    if other.team_number != profile.team_number:
                        secondary_bonus += (
                            self._role_pair_score(profile.secondary_role, other.primary_role)
                            * 5.0
                        )
        secondary_bonus = min(10.0, secondary_bonus)
        final = min(100.0, raw_score + secondary_bonus)

        return final, {
            "role_diversity": diversity_bonus,
            "avg_pair_compatibility": avg_pair * 100,
            "secondary_bonus": secondary_bonus,
        }

    def compute_historical_chemistry_score(
        self, chemistry: float
    ) -> float:
        """Convert a 0–1 historical chemistry value to a 0–100 score."""
        return min(100.0, max(0.0, chemistry * 100.0))

    def compute_synergy(
        self,
        profiles: Sequence[TeamProfile],
        historical_chemistry: float = 0.0,
    ) -> SynergyScore:
        """
        Compute the overall Alliance Synergy Score for a set of teams.

        Args:
            profiles: TeamProfile for each alliance member (typically 3).
            historical_chemistry: 0–1 value from Neo4jAllianceClient.compute_alliance_chemistry.

        Returns:
            SynergyScore with overall and component scores on a 0–100 scale.
        """
        cycle_score, cycle_details = self.compute_cycle_overlap_score(profiles)
        role_score, role_details = self.compute_role_complementarity_score(profiles)
        chemistry_score = self.compute_historical_chemistry_score(historical_chemistry)

        overall = (
            self.cycle_weight * cycle_score
            + self.role_weight * role_score
            + self.chemistry_weight * chemistry_score
        )

        team_numbers = [p.team_number for p in profiles]
        details = {**cycle_details, **role_details}
        details["chemistry_input"] = historical_chemistry * 100

        return SynergyScore(
            overall=round(min(100.0, max(0.0, overall)), 2),
            cycle_overlap_score=round(cycle_score, 2),
            role_complementarity_score=round(role_score, 2),
            historical_chemistry_score=round(chemistry_score, 2),
            team_numbers=team_numbers,
            details=details,
        )

    def rank_alliances(
        self,
        candidate_alliances: list[list[TeamProfile]],
        chemistry_values: list[float] | None = None,
    ) -> list[SynergyScore]:
        """Score and rank multiple candidate alliances, highest synergy first."""
        if chemistry_values is None:
            chemistry_values = [0.0] * len(candidate_alliances)

        scores = [
            self.compute_synergy(profiles, chem)
            for profiles, chem in zip(candidate_alliances, chemistry_values)
        ]
        return sorted(scores, key=lambda s: s.overall, reverse=True)

    @staticmethod
    def build_profiles_from_tracking(
        team_numbers: list[int],
        cycle_data: dict[int, list[tuple[float, float]]],
        roles: dict[int, ScoringRole] | None = None,
    ) -> list[TeamProfile]:
        """
        Construct TeamProfile objects from vision-tracker cycle timing data.

        Args:
            team_numbers: Alliance team numbers.
            cycle_data: Mapping of team_number → list of (start, end) cycle times.
            roles: Optional mapping of team_number → ScoringRole.
        """
        roles = roles or {}
        profiles: list[TeamProfile] = []

        for team_num in team_numbers:
            windows_raw = cycle_data.get(team_num, [])
            windows = [
                CycleWindow(team_number=team_num, start_time=s, end_time=e)
                for s, e in windows_raw
            ]
            durations = [w.duration for w in windows if w.duration > 0]
            avg_cycle = sum(durations) / len(durations) if durations else 15.0

            profiles.append(
                TeamProfile(
                    team_number=team_num,
                    primary_role=roles.get(team_num, ScoringRole.HYBRID),
                    cycle_windows=windows,
                    avg_cycle_time=avg_cycle,
                )
            )

        return profiles
