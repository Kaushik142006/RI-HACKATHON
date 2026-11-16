"""
Camera Calibration & Coordinate Transformations Module.
Transforms 3D points between Camera Optical Frame and Robot Base Frame.
"""

from typing import Tuple, List
import numpy as np
from vision.vision_state import Position3D


class CameraCalibration:
    """Manages hand-eye calibration and frame transformations."""

    def __init__(self, translation: List[float] = None, euler_deg: List[float] = None):
        """
        Extrinsics: Position of camera with respect to robot base.
        translation: [x, y, z] in meters
        euler_deg: [roll, pitch, yaw] in degrees
        """
        self.translation = np.array(translation if translation else [0.0, 0.45, 0.85], dtype=np.float32)
        euler = euler_deg if euler_deg else [0.0, 45.0, -90.0]

        # Compute rotation matrix from Euler angles (extrinsic XYZ)
        r_rad = np.radians(euler[0])
        p_rad = np.radians(euler[1])
        y_rad = np.radians(euler[2])

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(r_rad), -np.sin(r_rad)],
            [0, np.sin(r_rad), np.cos(r_rad)]
        ])
        Ry = np.array([
            [np.cos(p_rad), 0, np.sin(p_rad)],
            [0, 1, 0],
            [-np.sin(p_rad), 0, np.cos(p_rad)]
        ])
        Rz = np.array([
            [np.cos(y_rad), -np.sin(y_rad), 0],
            [np.sin(y_rad), np.cos(y_rad), 0],
            [0, 0, 1]
        ])

        self.R = Rz @ Ry @ Rx

        # Construct 4x4 Homogeneous Transformation Matrix T_base_camera
        self.T_base_cam = np.eye(4, dtype=np.float32)
        self.T_base_cam[:3, :3] = self.R
        self.T_base_cam[:3, 3] = self.translation

    def camera_to_base(self, point_camera: Position3D) -> Position3D:
        """Transform a 3D coordinate from camera optical frame to robot base frame."""
        p_cam = np.array([point_camera.x, point_camera.y, point_camera.z, 1.0], dtype=np.float32)
        p_base = self.T_base_cam @ p_cam

        return Position3D(
            x=round(float(p_base[0]), 3),
            y=round(float(p_base[1]), 3),
            z=round(float(p_base[2]), 3)
        )

    def base_to_camera(self, point_base: Position3D) -> Position3D:
        """Transform a 3D coordinate from robot base frame back to camera optical frame."""
        T_cam_base = np.linalg.inv(self.T_base_cam)
        p_b = np.array([point_base.x, point_base.y, point_base.z, 1.0], dtype=np.float32)
        p_cam = T_cam_base @ p_b

        return Position3D(
            x=round(float(p_cam[0]), 3),
            y=round(float(p_cam[1]), 3),
            z=round(float(p_cam[2]), 3)
        )
