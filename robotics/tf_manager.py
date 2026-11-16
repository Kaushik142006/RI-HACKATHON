"""
TF2 Coordinate Frame Transformation Manager.
Manages spatial transforms between base_link, camera_color_optical_frame, tool0_flange, and world.
"""

from typing import Dict, List, Optional
import numpy as np


class TFManager:
    """Maintains and computes coordinate transformations between robot frames."""

    def __init__(self, base_frame: str = "base_link"):
        self.base_frame = base_frame
        self._transforms: Dict[str, np.ndarray] = {}

        # Default transforms (4x4 homogeneous matrices)
        self.register_transform("world", "base_link", np.eye(4))
        # Camera mounted at (0.0, 0.45, 0.85) looking down at 45 deg
        T_cam = np.eye(4)
        T_cam[:3, 3] = [0.0, 0.45, 0.85]
        self.register_transform("base_link", "camera_link", T_cam)

    def register_transform(self, parent_frame: str, child_frame: str, matrix_4x4: np.ndarray) -> None:
        """Register a known transformation matrix."""
        key = f"{parent_frame}->{child_frame}"
        self._transforms[key] = matrix_4x4

    def transform_point(self, point_xyz: List[float], from_frame: str, to_frame: str) -> List[float]:
        """Transform a 3D point from source frame to target frame."""
        if from_frame == to_frame:
            return point_xyz

        key = f"{from_frame}->{to_frame}"
        if key in self._transforms:
            T = self._transforms[key]
        else:
            inv_key = f"{to_frame}->{from_frame}"
            if inv_key in self._transforms:
                T = np.linalg.inv(self._transforms[inv_key])
            else:
                # Identity fallback
                T = np.eye(4)

        p = np.array([point_xyz[0], point_xyz[1], point_xyz[2], 1.0])
        p_trans = T @ p
        return [round(float(p_trans[0]), 3), round(float(p_trans[1]), 3), round(float(p_trans[2]), 3)]
