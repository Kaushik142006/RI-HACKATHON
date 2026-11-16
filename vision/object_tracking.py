"""
ByteTrack Object Tracking Module.
Maintains continuous object identity and track IDs across sequential frames.
"""

from typing import List, Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ByteTrackTracker:
    """ByteTrack-based multi-object tracker with deterministic simulation fallback."""

    def __init__(self, track_thresh: float = 0.5, match_thresh: float = 0.8, simulation_mode: bool = True):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.simulation_mode = simulation_mode
        self._next_id = 1
        self._tracks: Dict[str, int] = {}

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Assign track IDs to detections based on spatial proximity / class continuity.
        Each detection dict is enriched with "track_id".
        """
        tracked_results = []
        for det in detections:
            cls = det["class"]
            bbox = det["bbox"]
            center = bbox.center

            # Simple nearest-track assignment
            assigned_id = None
            best_dist = float("inf")

            for key, t_id in self._tracks.items():
                t_cls, t_cx, t_cy = key.split(":")
                if t_cls == cls:
                    dist = np.hypot(center[0] - float(t_cx), center[1] - float(t_cy))
                    if dist < 60.0 and dist < best_dist:
                        assigned_id = t_id
                        best_dist = dist

            if assigned_id is None:
                assigned_id = self._next_id
                self._next_id += 1

            # Update track registry
            new_key = f"{cls}:{center[0]:.1f}:{center[1]:.1f}"
            self._tracks[new_key] = assigned_id

            enriched_det = dict(det)
            enriched_det["track_id"] = assigned_id
            tracked_results.append(enriched_det)

        return tracked_results
