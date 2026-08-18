"""Convert raw robot tracking data into discrete match events."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Sequence

import numpy as np

from src.vision.types import TrackState, TrackingResult


class EventType(str, Enum):
    CYCLE = "cycle"
    DEFENSE = "defense"
    ENDGAME = "endgame"


class EndgameAction(str, Enum):
    CLIMB = "climb"
    DOCK = "dock"
    PARK = "park"


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def distance_to(self, other: Point) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class FieldZone:
    """Rectangular field region defined in pixel coordinates."""

    name: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    zone_type: str  # "intake", "scoring", "endgame_climb", "endgame_dock", "endgame_park"

    def contains(self, point: Point) -> bool:
        return (
            self.x_min <= point.x <= self.x_max
            and self.y_min <= point.y <= self.y_max
        )

    @property
    def centroid(self) -> Point:
        return Point((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)


@dataclass
class CycleEvent:
    track_id: int
    event_type: EventType = EventType.CYCLE
    start_frame: int = 0
    end_frame: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    intake_zone: str = ""
    scoring_zone: str = ""
    traverse_distance_m: float = 0.0
    cycle_duration_s: float = 0.0
    confidence: float = 1.0


@dataclass
class DefenseEvent:
    defender_track_id: int
    target_track_id: int
    event_type: EventType = EventType.DEFENSE
    start_frame: int = 0
    end_frame: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    min_proximity_m: float = 0.0
    avg_proximity_m: float = 0.0
    velocity_reduction_pct: float = 0.0
    duration_s: float = 0.0
    confidence: float = 1.0


@dataclass
class EndgameEvent:
    track_id: int
    event_type: EventType = EventType.ENDGAME
    action: EndgameAction = EndgameAction.PARK
    start_frame: int = 0
    end_frame: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    zone_name: str = ""
    dwell_duration_s: float = 0.0
    confidence: float = 1.0


MatchEvent = CycleEvent | DefenseEvent | EndgameEvent


class _CyclePhase(str, Enum):
    IDLE = "idle"
    INTAKE_DWELL = "intake_dwell"
    TRAVERSING = "traversing"
    SCORING_DWELL = "scoring_dwell"


@dataclass
class _TrackSnapshot:
    frame_index: int
    timestamp: float
    centroid: Point
    speed_px: float
    speed_m: float
    zone: str | None


@dataclass
class _CycleTracker:
    phase: _CyclePhase = _CyclePhase.IDLE
    intake_zone: str = ""
    scoring_zone: str = ""
    intake_start_frame: int = 0
    intake_start_time: float = 0.0
    traverse_start_frame: int = 0
    traverse_start_time: float = 0.0
    traverse_start_point: Point | None = None
    traverse_distance_m: float = 0.0
    scoring_start_frame: int = 0
    scoring_start_time: float = 0.0
    intake_dwell_frames: int = 0
    scoring_dwell_frames: int = 0


@dataclass
class _DefenseTracker:
    active: bool = False
    defender_id: int = -1
    start_frame: int = 0
    start_time: float = 0.0
    proximity_samples: list[float] = field(default_factory=list)
    baseline_speed_m: float = 0.0
    speed_samples: list[float] = field(default_factory=list)


@dataclass
class _EndgameTracker:
    active: bool = False
    action: EndgameAction = EndgameAction.PARK
    zone_name: str = ""
    start_frame: int = 0
    start_time: float = 0.0
    dwell_frames: int = 0


class VisionStateMachine:
    """
    Stateful processor that converts per-frame robot tracks into discrete
    match events: cycles, defensive plays, and endgame actions.

    Operates on pixel-space centroids with a configurable pixels-per-meter
    scale factor.  Field zones are rectangular regions supplied at init.
    """

    DEFAULT_INTAKE_DWELL_FRAMES = 15
    DEFAULT_SCORING_DWELL_FRAMES = 10
    DEFAULT_ENDGAME_DWELL_FRAMES = 45
    DEFAULT_DEFENSE_PROXIMITY_M = 2.0
    DEFAULT_VELOCITY_REDUCTION_THRESHOLD = 0.35
    DEFAULT_MIN_DEFENSE_FRAMES = 20
    DEFAULT_ENDGAME_START_TIME_S = 120.0

    def __init__(
        self,
        zones: Sequence[FieldZone],
        *,
        pixels_per_meter: float = 50.0,
        frame_rate: float = 30.0,
        alliance_track_ids: set[int] | None = None,
        opponent_track_ids: set[int] | None = None,
        intake_dwell_frames: int = DEFAULT_INTAKE_DWELL_FRAMES,
        scoring_dwell_frames: int = DEFAULT_SCORING_DWELL_FRAMES,
        endgame_dwell_frames: int = DEFAULT_ENDGAME_DWELL_FRAMES,
        defense_proximity_m: float = DEFAULT_DEFENSE_PROXIMITY_M,
        velocity_reduction_threshold: float = DEFAULT_VELOCITY_REDUCTION_THRESHOLD,
        min_defense_frames: int = DEFAULT_MIN_DEFENSE_FRAMES,
        endgame_start_time_s: float = DEFAULT_ENDGAME_START_TIME_S,
    ) -> None:
        self.zones = list(zones)
        self.pixels_per_meter = pixels_per_meter
        self.frame_rate = frame_rate
        self.alliance_track_ids = alliance_track_ids or set()
        self.opponent_track_ids = opponent_track_ids or set()
        self.intake_dwell_frames = intake_dwell_frames
        self.scoring_dwell_frames = scoring_dwell_frames
        self.endgame_dwell_frames = endgame_dwell_frames
        self.defense_proximity_m = defense_proximity_m
        self.velocity_reduction_threshold = velocity_reduction_threshold
        self.min_defense_frames = min_defense_frames
        self.endgame_start_time_s = endgame_start_time_s

        self._intake_zones = [z for z in zones if z.zone_type == "intake"]
        self._scoring_zones = [z for z in zones if z.zone_type.startswith("scoring")]
        self._endgame_zones = [z for z in zones if z.zone_type.startswith("endgame")]

        self._cycle: dict[int, _CycleTracker] = {}
        self._defense: dict[tuple[int, int], _DefenseTracker] = {}
        self._endgame: dict[int, _EndgameTracker] = {}
        self._history: dict[int, list[_TrackSnapshot]] = {}
        self._events: list[MatchEvent] = []
        self._current_frame: int = 0
        self._current_time: float = 0.0

    @property
    def events(self) -> list[MatchEvent]:
        return list(self._events)

    def reset(self) -> None:
        self._cycle.clear()
        self._defense.clear()
        self._endgame.clear()
        self._history.clear()
        self._events.clear()
        self._current_frame = 0
        self._current_time = 0.0

    def _px_to_m(self, distance_px: float) -> float:
        return distance_px / self.pixels_per_meter

    def _speed_px_to_m(self, speed_px: float) -> float:
        return speed_px * self.frame_rate / self.pixels_per_meter

    def _frame_to_time(self, frame: int) -> float:
        return frame / self.frame_rate

    def _zone_at(self, point: Point) -> FieldZone | None:
        for zone in self.zones:
            if zone.contains(point):
                return zone
        return None

    def _intake_zone_at(self, point: Point) -> FieldZone | None:
        for zone in self._intake_zones:
            if zone.contains(point):
                return zone
        return None

    def _scoring_zone_at(self, point: Point) -> FieldZone | None:
        for zone in self._scoring_zones:
            if zone.contains(point):
                return zone
        return None

    def _endgame_zone_at(self, point: Point) -> FieldZone | None:
        for zone in self._endgame_zones:
            if zone.contains(point):
                return zone
        return None

    @staticmethod
    def _speed_from_vector(movement: tuple[float, float]) -> float:
        return math.hypot(movement[0], movement[1])

    def _snapshot(self, track: TrackState) -> _TrackSnapshot:
        point = Point(track.centroid[0], track.centroid[1])
        speed_px = self._speed_from_vector(track.movement_vector)
        zone = self._zone_at(point)
        return _TrackSnapshot(
            frame_index=self._current_frame,
            timestamp=self._current_time,
            centroid=point,
            speed_px=speed_px,
            speed_m=self._speed_px_to_m(speed_px),
            zone=zone.name if zone else None,
        )

    def _update_cycle(self, track_id: int, snap: _TrackSnapshot) -> CycleEvent | None:
        tracker = self._cycle.setdefault(track_id, _CycleTracker())
        intake = self._intake_zone_at(snap.centroid)
        scoring = self._scoring_zone_at(snap.centroid)

        if tracker.phase == _CyclePhase.IDLE:
            if intake:
                tracker.phase = _CyclePhase.INTAKE_DWELL
                tracker.intake_zone = intake.name
                tracker.intake_start_frame = snap.frame_index
                tracker.intake_start_time = snap.timestamp
                tracker.intake_dwell_frames = 1
            return None

        if tracker.phase == _CyclePhase.INTAKE_DWELL:
            if intake and intake.name == tracker.intake_zone:
                tracker.intake_dwell_frames += 1
                if tracker.intake_dwell_frames >= self.intake_dwell_frames:
                    tracker.phase = _CyclePhase.TRAVERSING
                    tracker.traverse_start_frame = snap.frame_index
                    tracker.traverse_start_time = snap.timestamp
                    tracker.traverse_start_point = snap.centroid
                    tracker.traverse_distance_m = 0.0
            else:
                tracker.phase = _CyclePhase.IDLE
                tracker.intake_dwell_frames = 0
            return None

        if tracker.phase == _CyclePhase.TRAVERSING:
            if tracker.traverse_start_point:
                step_px = snap.centroid.distance_to(tracker.traverse_start_point)
                tracker.traverse_distance_m += self._px_to_m(step_px)
                tracker.traverse_start_point = snap.centroid
            if scoring:
                tracker.phase = _CyclePhase.SCORING_DWELL
                tracker.scoring_zone = scoring.name
                tracker.scoring_start_frame = snap.frame_index
                tracker.scoring_start_time = snap.timestamp
                tracker.scoring_dwell_frames = 1
            elif intake:
                tracker.phase = _CyclePhase.IDLE
            return None

        if tracker.phase == _CyclePhase.SCORING_DWELL:
            if scoring and scoring.name == tracker.scoring_zone:
                tracker.scoring_dwell_frames += 1
                if tracker.scoring_dwell_frames >= self.scoring_dwell_frames:
                    event = CycleEvent(
                        track_id=track_id,
                        start_frame=tracker.intake_start_frame,
                        end_frame=snap.frame_index,
                        start_time=tracker.intake_start_time,
                        end_time=snap.timestamp,
                        intake_zone=tracker.intake_zone,
                        scoring_zone=tracker.scoring_zone,
                        traverse_distance_m=tracker.traverse_distance_m,
                        cycle_duration_s=snap.timestamp - tracker.intake_start_time,
                        confidence=min(
                            1.0,
                            tracker.traverse_distance_m / 5.0,
                        ),
                    )
                    self._cycle[track_id] = _CycleTracker()
                    return event
            else:
                tracker.phase = _CyclePhase.IDLE
            return None

        return None

    def _is_opponent(self, track_id: int) -> bool:
        if self.opponent_track_ids:
            return track_id in self.opponent_track_ids
        if self.alliance_track_ids:
            return track_id not in self.alliance_track_ids
        return False

    def _is_alliance(self, track_id: int) -> bool:
        if self.alliance_track_ids:
            return track_id in self.alliance_track_ids
        if self.opponent_track_ids:
            return track_id not in self.opponent_track_ids
        return True

    def _update_defense(
        self, tracks: dict[int, _TrackSnapshot]
    ) -> list[DefenseEvent]:
        emitted: list[DefenseEvent] = []
        proximity_px = self.defense_proximity_m * self.pixels_per_meter

        defenders = [tid for tid in tracks if self._is_opponent(tid)]
        targets = [tid for tid in tracks if self._is_alliance(tid)]

        for defender_id in defenders:
            for target_id in targets:
                if defender_id == target_id:
                    continue
                key = (defender_id, target_id)
                dt = self._defense.setdefault(key, _DefenseTracker())
                dist_px = tracks[defender_id].centroid.distance_to(
                    tracks[target_id].centroid
                )
                dist_m = self._px_to_m(dist_px)
                target_speed = tracks[target_id].speed_m

                if dist_m <= self.defense_proximity_m:
                    if not dt.active:
                        dt.active = True
                        dt.defender_id = defender_id
                        dt.start_frame = self._current_frame
                        dt.start_time = self._current_time
                        dt.proximity_samples = [dist_m]
                        dt.speed_samples = [target_speed]
                        history = self._history.get(target_id, [])
                        recent = [s.speed_m for s in history[-30:] if s.speed_m > 0]
                        dt.baseline_speed_m = (
                            float(np.mean(recent)) if recent else max(target_speed, 0.5)
                        )
                    else:
                        dt.proximity_samples.append(dist_m)
                        dt.speed_samples.append(target_speed)
                elif dt.active:
                    duration_frames = self._current_frame - dt.start_frame
                    if duration_frames >= self.min_defense_frames:
                        avg_speed = float(np.mean(dt.speed_samples)) if dt.speed_samples else 0.0
                        reduction = 0.0
                        if dt.baseline_speed_m > 0:
                            reduction = max(
                                0.0,
                                (dt.baseline_speed_m - avg_speed) / dt.baseline_speed_m,
                            )
                        if reduction >= self.velocity_reduction_threshold:
                            emitted.append(
                                DefenseEvent(
                                    defender_track_id=defender_id,
                                    target_track_id=target_id,
                                    start_frame=dt.start_frame,
                                    end_frame=self._current_frame,
                                    start_time=dt.start_time,
                                    end_time=self._current_time,
                                    min_proximity_m=min(dt.proximity_samples),
                                    avg_proximity_m=float(np.mean(dt.proximity_samples)),
                                    velocity_reduction_pct=reduction * 100.0,
                                    duration_s=self._current_time - dt.start_time,
                                    confidence=min(1.0, reduction / 0.5),
                                )
                            )
                    self._defense[key] = _DefenseTracker()

        return emitted

    def _endgame_action_for_zone(self, zone: FieldZone) -> EndgameAction:
        mapping = {
            "endgame_climb": EndgameAction.CLIMB,
            "endgame_dock": EndgameAction.DOCK,
            "endgame_park": EndgameAction.PARK,
        }
        return mapping.get(zone.zone_type, EndgameAction.PARK)

    def _update_endgame(
        self, track_id: int, snap: _TrackSnapshot
    ) -> EndgameEvent | None:
        if snap.timestamp < self.endgame_start_time_s:
            return None

        zone = self._endgame_zone_at(snap.centroid)
        tracker = self._endgame.setdefault(track_id, _EndgameTracker())

        if zone:
            action = self._endgame_action_for_zone(zone)
            if not tracker.active:
                tracker.active = True
                tracker.action = action
                tracker.zone_name = zone.name
                tracker.start_frame = snap.frame_index
                tracker.start_time = snap.timestamp
                tracker.dwell_frames = 1
            elif tracker.zone_name == zone.name:
                tracker.dwell_frames += 1
                if tracker.dwell_frames == self.endgame_dwell_frames:
                    return EndgameEvent(
                        track_id=track_id,
                        action=tracker.action,
                        start_frame=tracker.start_frame,
                        end_frame=snap.frame_index,
                        start_time=tracker.start_time,
                        end_time=snap.timestamp,
                        zone_name=tracker.zone_name,
                        dwell_duration_s=snap.timestamp - tracker.start_time,
                        confidence=min(1.0, tracker.dwell_frames / self.endgame_dwell_frames),
                    )
            else:
                self._endgame[track_id] = _EndgameTracker()
        elif tracker.active:
            self._endgame[track_id] = _EndgameTracker()

        return None

    def process_frame(self, result: TrackingResult) -> list[MatchEvent]:
        """Process one TrackingResult and return newly detected events."""
        self._current_frame = result.frame_index
        self._current_time = self._frame_to_time(result.frame_index)
        frame_events: list[MatchEvent] = []

        snapshots: dict[int, _TrackSnapshot] = {}
        for track in result.tracks:
            snap = self._snapshot(track)
            snapshots[track.track_id] = snap
            self._history.setdefault(track.track_id, []).append(snap)

            cycle_event = self._update_cycle(track.track_id, snap)
            if cycle_event:
                frame_events.append(cycle_event)

            endgame_event = self._update_endgame(track.track_id, snap)
            if endgame_event:
                frame_events.append(endgame_event)

        frame_events.extend(self._update_defense(snapshots))
        self._events.extend(frame_events)
        return frame_events

    def process_stream(
        self, results: Iterator[TrackingResult]
    ) -> list[MatchEvent]:
        """Process a stream of tracking results and return all detected events."""
        for result in results:
            self.process_frame(result)
        return self.events

    def get_cycle_events(self) -> list[CycleEvent]:
        return [e for e in self._events if isinstance(e, CycleEvent)]

    def get_defense_events(self) -> list[DefenseEvent]:
        return [e for e in self._events if isinstance(e, DefenseEvent)]

    def get_endgame_events(self) -> list[EndgameEvent]:
        return [e for e in self._events if isinstance(e, EndgameEvent)]

    @classmethod
    def default_field_zones(
        cls,
        field_width: float = 1920.0,
        field_height: float = 1080.0,
    ) -> list[FieldZone]:
        """Generate a standard set of placeholder zones for a 1920×1080 frame."""
        w, h = field_width, field_height
        return [
            FieldZone("red_intake", 0, h * 0.3, w * 0.15, h * 0.7, "intake"),
            FieldZone("blue_intake", w * 0.85, h * 0.3, w, h * 0.7, "intake"),
            FieldZone("red_speaker", w * 0.35, 0, w * 0.5, h * 0.25, "scoring"),
            FieldZone("blue_speaker", w * 0.35, h * 0.75, w * 0.5, h, "scoring"),
            FieldZone("red_amp", w * 0.1, h * 0.1, w * 0.25, h * 0.3, "scoring"),
            FieldZone("blue_amp", w * 0.75, h * 0.7, w * 0.9, h * 0.9, "scoring"),
            FieldZone("red_climb", w * 0.4, h * 0.35, w * 0.55, h * 0.55, "endgame_climb"),
            FieldZone("blue_climb", w * 0.4, h * 0.45, w * 0.55, h * 0.65, "endgame_climb"),
            FieldZone("red_dock", w * 0.55, h * 0.35, w * 0.7, h * 0.55, "endgame_dock"),
            FieldZone("red_park", w * 0.05, h * 0.05, w * 0.2, h * 0.2, "endgame_park"),
            FieldZone("blue_park", w * 0.8, h * 0.8, w * 0.95, h * 0.95, "endgame_park"),
        ]
