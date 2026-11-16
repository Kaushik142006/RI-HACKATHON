"""
3D Coordinate Estimation Module.
Deprojects 2D pixel coordinates and depth to 3D metric camera coordinates.
"""

from typing import Tuple, Dict, Any
import numpy as np
from vision.vision_state import Position3D

class CoordinateEstimator:
    """Computes 3D coordinates from pixel (u, v) and depth using pinhole camera model."""

    def __init__(self, fx: float = 615.0, fy: float = 615.0, cx: float = 640.0, cy: float = 360.0):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def pixel_to_camera_point(self, u: float, v: float, depth_m: float) -> Tuple[float, float, float]:
        """
        Deproject 2D pixel (u, v) and depth (Z) to 3D point (X, Y, Z) in Camera Frame.
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = Z
        """
        if depth_m <= 0:
            depth_m = 0.5  # Fallback minimum distance

        x_c = (u - self.cx) * depth_m / self.fx
        y_c = (v - self.cy) * depth_m / self.fy
        z_c = depth_m
        return float(x_c), float(y_c), float(z_c)

    def estimate_bbox_3d(self, u_center: float, v_center: float, depth_m: float) -> Position3D:
        """Estimate 3D position of an object centroid."""
        xc, yc, zc = self.pixel_to_camera_point(u_center, v_center, depth_m)
        return Position3D(x=round(xc, 3), y=round(yc, 3), z=round(zc, 3))
