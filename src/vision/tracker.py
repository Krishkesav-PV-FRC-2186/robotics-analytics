"""Robot detection and multi-object tracking for FRC match video."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from src.vision.types import TrackState, TrackingResult


class RobotTracker:
    """
    Detect and track robots in match video using YOLO object detection
    and ByteTrack multi-object tracking.

    Outputs per-frame bounding boxes, persistent track IDs, and inter-frame
    movement vectors derived from centroid displacement.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
        target_class_ids: list[int] | None = None,
    ) -> None:
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.target_class_ids = target_class_ids

        self.byte_tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

        self._prev_centroids: dict[int, tuple[float, float]] = {}

    @staticmethod
    def _centroid(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _movement_vector(
        self, track_id: int, centroid: tuple[float, float]
    ) -> tuple[float, float]:
        prev = self._prev_centroids.get(track_id)
        if prev is None:
            return (0.0, 0.0)
        return (centroid[0] - prev[0], centroid[1] - prev[1])

    def _detections_to_tracks(
        self, detections: sv.Detections, class_names: dict[int, str]
    ) -> list[TrackState]:
        tracks: list[TrackState] = []
        for i in range(len(detections)):
            track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
            if track_id < 0:
                continue

            x1, y1, x2, y2 = detections.xyxy[i]
            bbox = (float(x1), float(y1), float(x2), float(y2))
            centroid = self._centroid(bbox)
            movement = self._movement_vector(track_id, centroid)
            self._prev_centroids[track_id] = centroid

            class_id = int(detections.class_id[i])
            tracks.append(
                TrackState(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=float(detections.confidence[i]),
                    class_id=class_id,
                    class_name=class_names.get(class_id, str(class_id)),
                    movement_vector=movement,
                    centroid=centroid,
                )
            )
        return tracks

    def process_frame(
        self, frame: np.ndarray, frame_index: int = 0
    ) -> TrackingResult:
        """
        Run detection + tracking on a single BGR frame.

        Args:
            frame: OpenCV BGR image (H, W, 3).
            frame_index: Sequential frame number for output labeling.

        Returns:
            TrackingResult with bounding boxes, track IDs, and movement vectors.
        """
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        if self.target_class_ids is not None and len(detections) > 0:
            mask = np.isin(detections.class_id, self.target_class_ids)
            detections = detections[mask]

        tracked = self.byte_tracker.update_with_detections(detections)
        class_names: dict[int, str] = results.names or {}
        tracks = self._detections_to_tracks(tracked, class_names)

        return TrackingResult(frame_index=frame_index, tracks=tracks)

    def process_video(
        self,
        video_path: str | Path,
        *,
        start_frame: int = 0,
        max_frames: int | None = None,
    ) -> Iterator[TrackingResult]:
        """
        Stream tracking results frame-by-frame from a video file.

        Args:
            video_path: Path to a video file readable by OpenCV.
            start_frame: Frame index to begin processing.
            max_frames: Optional cap on frames processed.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_index = start_frame
        frames_processed = 0

        try:
            while True:
                if max_frames is not None and frames_processed >= max_frames:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                yield self.process_frame(frame, frame_index)
                frame_index += 1
                frames_processed += 1
        finally:
            cap.release()

    def reset(self) -> None:
        """Clear tracker state between videos."""
        self.byte_tracker.reset()
        self._prev_centroids.clear()

    def annotate_frame(
        self,
        frame: np.ndarray,
        result: TrackingResult,
        *,
        show_vectors: bool = True,
    ) -> np.ndarray:
        """Draw bounding boxes, track IDs, and movement vectors on a frame."""
        annotated = frame.copy()
        for track in result.tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"#{track.track_id} {track.class_name}"
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
            if show_vectors and track.movement_vector != (0.0, 0.0):
                cx, cy = map(int, track.centroid)
                dx, dy = track.movement_vector
                end = (int(cx + dx * 3), int(cy + dy * 3))
                cv2.arrowedLine(annotated, (cx, cy), end, (0, 0, 255), 2)
        return annotated
