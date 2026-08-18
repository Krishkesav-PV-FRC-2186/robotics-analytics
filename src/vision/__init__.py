"""Vision pipeline — detection, tracking, and event extraction."""

from src.vision.types import TrackState, TrackingResult
from src.vision.event_engine import (
    VisionStateMachine,
    CycleEvent,
    DefenseEvent,
    EndgameEvent,
    FieldZone,
    MatchEvent,
)

__all__ = [
    "RobotTracker",
    "TrackingResult",
    "TrackState",
    "VisionStateMachine",
    "CycleEvent",
    "DefenseEvent",
    "EndgameEvent",
    "FieldZone",
    "MatchEvent",
]


def __getattr__(name: str):
    if name == "RobotTracker":
        from src.vision.tracker import RobotTracker

        return RobotTracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
