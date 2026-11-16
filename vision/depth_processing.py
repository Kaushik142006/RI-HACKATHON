"""
Depth Processing Module.
Extracts, filters, and scales RealSense depth data for accurate 3D spatial reasoning.
"""

from typing import Tuple, Optional
import logging
import numpy as np
from vision.vision_state import BoundingBox

logger = logging.getLogger(__name__)


class DepthProcessor:
    """Processes raw depth buffers with spatial and median filtering."""

    def __init__(self, depth_scale: float = 0.001, min_depth_m: float = 0.15, max_depth_m: float = 3.0):
        self.depth_scale = depth_scale
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m

    def get_object_depth(self, depth_map: np.ndarray, bbox: BoundingBox, mask: Optional[np.ndarray] = None) -> float:
        """
        Extract representative depth (in meters) for an object region.
        Applies median filtering and trims zeros/outliers.
        """
        h, w = depth_map.shape[:2]
        x1 = max(0, int(bbox.xmin))
        y1 = max(0, int(bbox.ymin))
        x2 = min(w, int(bbox.xmax))
        y2 = min(h, int(bbox.ymax))

        if x2 <= x1 or y2 <= y1:
            return 0.75  # Safe tabletop default

        # Center patch (inner 50% to avoid boundary background bleed)
        cx1 = int(x1 + 0.25 * (x2 - x1))
        cx2 = int(x2 - 0.25 * (x2 - x1))
        cy1 = int(y1 + 0.25 * (y2 - y1))
        cy2 = int(y2 - 0.25 * (y2 - y1))

        depth_patch = depth_map[cy1:cy2, cx1:cx2].astype(np.float32)

        # Apply depth scale if input is raw uint16 mm
        if depth_patch.dtype == np.uint16 or depth_patch.max() > 50.0:
            depth_patch = depth_patch * self.depth_scale

        # Filter out invalid readings
        valid_mask = (depth_patch >= self.min_depth_m) & (depth_patch <= self.max_depth_m)
        valid_depths = depth_patch[valid_mask]

        if len(valid_depths) == 0:
            logger.warning("No valid depth pixels found in bbox patch. Using default depth 0.75m")
            return 0.75

        # Use median for robustness against noise
        median_depth = float(np.median(valid_depths))
        return median_depth
