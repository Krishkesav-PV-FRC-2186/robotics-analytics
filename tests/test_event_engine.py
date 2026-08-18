"""Unit tests for VisionStateMachine event detection."""

from __future__ import annotations

import pytest

from src.vision.event_engine import VisionStateMachine
from src.vision.types import TrackState, TrackingResult


def _track(tid: int, x: float, y: float, dx: float = 0, dy: float = 0) -> TrackState:
    return TrackState(
        track_id=tid,
        bbox=(x - 20, y - 20, x + 20, y + 20),
        confidence=0.9,
        class_id=0,
        class_name="robot",
        movement_vector=(dx, dy),
        centroid=(x, y),
    )


@pytest.fixture
def state_machine() -> VisionStateMachine:
    return VisionStateMachine(
        zones=VisionStateMachine.default_field_zones(),
        pixels_per_meter=50.0,
        frame_rate=30.0,
        alliance_track_ids={1},
        opponent_track_ids={2},
        intake_dwell_frames=5,
        scoring_dwell_frames=5,
        endgame_dwell_frames=5,
        min_defense_frames=5,
        endgame_start_time_s=0.0,
    )


class TestCycleEventDetection:
    def test_cycle_event_triggered_on_intake_traverse_score(
        self, state_machine: VisionStateMachine
    ) -> None:
        path = [
            (150, 600), (150, 600), (150, 600), (150, 600), (150, 600),
            (300, 600), (500, 600), (700, 600),
            (850, 150), (850, 150), (850, 150), (850, 150), (850, 150),
        ]
        events = []
        for i, (x, y) in enumerate(path):
            prev = path[i - 1] if i > 0 else (x, y)
            result = TrackingResult(
                frame_index=i,
                tracks=[_track(1, x, y, x - prev[0], y - prev[1])],
            )
            events.extend(state_machine.process_frame(result))

        cycles = state_machine.get_cycle_events()
        assert len(cycles) >= 1
        assert cycles[0].intake_zone == "red_intake"
        assert "speaker" in cycles[0].scoring_zone or "amp" in cycles[0].scoring_zone


class TestDefenseEventDetection:
    def test_defense_event_on_proximity_and_velocity_drop(
        self, state_machine: VisionStateMachine
    ) -> None:
        for i in range(8):
            speed = max(0.0, 8.0 - i * 1.0)
            state_machine.process_frame(
                TrackingResult(
                    frame_index=i,
                    tracks=[
                        _track(1, 500, 500, speed, 0),
                        _track(2, 515, 508, 0, 0),
                    ],
                )
            )
        state_machine.process_frame(
            TrackingResult(
                frame_index=8,
                tracks=[
                    _track(1, 500, 500, 1, 0),
                    _track(2, 800, 800, 0, 0),
                ],
            )
        )
        defenses = state_machine.get_defense_events()
        assert len(defenses) >= 1
        assert defenses[0].defender_track_id == 2
        assert defenses[0].target_track_id == 1


class TestEndgameEventDetection:
    def test_endgame_climb_detected_in_zone(
        self, state_machine: VisionStateMachine
    ) -> None:
        for i in range(8):
            state_machine.process_frame(
                TrackingResult(
                    frame_index=i,
                    tracks=[_track(3, 900, 486)],
                )
            )
        endgames = state_machine.get_endgame_events()
        assert len(endgames) >= 1
        assert endgames[0].action.value == "climb"
