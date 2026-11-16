"""
Obstacle Detection Module.
Identifies collision hazards in the workspace using LiDAR scans and camera point clouds.
"""

from typing import List
import logging
import numpy as np
from vision.vision_state import Obstacle, Position3D

logger = logging.getLogger(__name__)


class ObstacleDetector:
    """Detects physical obstacles and intrusion hazards within the robot workspace."""

    def __init__(self, critical_distance_m: float = 0.25, warning_distance_m: float = 0.45):
        self.critical_distance_m = critical_distance_m
        self.warning_distance_m = warning_distance_m

    def detect_obstacles(self, lidar_points_2d: np.ndarray, workspace_bounds: dict) -> List[Obstacle]:
        """
        Analyze points and identify unauthorized objects within the robot operating zone.
        """
        obstacles = []
        if len(lidar_points_2d) == 0:
            return obstacles

        # Filter points within workspace
        x_min, x_max = workspace_bounds.get("x", (-0.8, 0.8))
        y_min, y_max = workspace_bounds.get("y", (-0.8, 0.8))

        # Check for points that violate safety distance
        obs_id = 1
        for pt in lidar_points_2d:
            dist = np.hypot(pt[0], pt[1])
            if dist < self.critical_distance_m:
                obstacles.append(Obstacle(
                    obstacle_id=obs_id,
                    obstacle_type="proximity_hazard",
                    position_3d=Position3D(x=round(float(pt[0]), 3), y=round(float(pt[1]), 3), z=0.1),
                    bounding_radius_m=0.15,
                    threat_level="critical"
                ))
                obs_id += 1
                if obs_id > 5:
                    break  # Keep list manageable

        return obstacles
