"""
Collision Checker Module.
Validates workspace bounding boundaries, obstacle separation distances, and self-collision constraints.
"""

from typing import List, Dict, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


class CollisionChecker:
    """Verifies that Cartesian paths and joint configurations maintain safe clearance from obstacles and table."""

    def __init__(self, workspace_bounds: Dict = None, min_obstacle_clearance_m: float = 0.05):
        self.workspace_bounds = workspace_bounds or {
            "x": (-0.8, 0.8),
            "y": (-0.8, 0.8),
            "z": (0.0, 1.2)  # Z cannot be negative (table penetration)
        }
        self.min_clearance = min_obstacle_clearance_m

    def check_cartesian_point(self, x: float, y: float, z: float, obstacles: List[Dict] = None) -> Tuple[bool, str]:
        """Check if a Cartesian point respects workspace boundaries and obstacle margins."""
        x_min, x_max = self.workspace_bounds["x"]
        y_min, y_max = self.workspace_bounds["y"]
        z_min, z_max = self.workspace_bounds["z"]

        if not (x_min <= x <= x_max):
            return False, f"X coordinate {x:.3f}m violates workspace limits [{x_min}, {x_max}]."
        if not (y_min <= y <= y_max):
            return False, f"Y coordinate {y:.3f}m violates workspace limits [{y_min}, {y_max}]."
        if not (z_min <= z <= z_max):
            return False, f"Z coordinate {z:.3f}m violates workspace limits [{z_min}, {z_max}]. Table penetration hazard!"

        if obstacles:
            for obs in obstacles:
                pos = obs.get("position_3d", {})
                ox = pos.get("x", 0.0)
                oy = pos.get("y", 0.0)
                oz = pos.get("z", 0.0)
                dist = np.sqrt((x - ox)**2 + (y - oy)**2 + (z - oz)**2)
                rad = obs.get("bounding_radius_m", 0.1)
                if dist < (rad + self.min_clearance):
                    return False, f"Proximity violation: distance {dist:.3f}m to obstacle {obs.get('obstacle_id')} < minimum safe margin {rad + self.min_clearance:.3f}m."

        return True, "Cartesian point clear of collisions."
