"""Lightweight tracking dataclasses shared across vision modules."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrackState:
    """Per-robot tracking state for a single frame."""

    track_id: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    movement_vector: tuple[float, float]  # (dx, dy) pixels since last frame
    centroid: tuple[float, float] = field(default=(0.0, 0.0))

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass
class TrackingResult:
    """Tracking output for one video frame."""

    frame_index: int
    tracks: list[TrackState] = field(default_factory=list)

    @property
    def num_tracks(self) -> int:
        return len(self.tracks)
