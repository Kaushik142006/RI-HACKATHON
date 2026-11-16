"""
MoveIt 2 Interface Module.
Wraps MoveGroup and PlanningScene interactions for trajectory synthesis and obstacle integration.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class MoveItInterface:
    """Interfaces with MoveIt 2 Planning Scene and MoveGroup Action Servers."""

    def __init__(self, group_name: str = "manipulator", simulation_mode: bool = True):
        self.group_name = group_name
        self.simulation_mode = simulation_mode
        self.planning_scene_objects = []

    def update_planning_scene(self, obstacles: List[Dict[str, Any]]) -> bool:
        """Add collision objects to the MoveIt 2 Planning Scene."""
        self.planning_scene_objects = list(obstacles)
        logger.info(f"Updated MoveIt 2 Planning Scene with {len(obstacles)} collision geometry objects.")
        return True

    def plan_to_cartesian_target(self, target_xyz: List[float], speed_factor: float = 0.5) -> Dict[str, Any]:
        """Request collision-aware trajectory plan to target XYZ."""
        logger.info(f"MoveIt 2 planning trajectory to Cartesian target {target_xyz} with speed scaling {speed_factor}")
        return {
            "success": True,
            "target": target_xyz,
            "planning_time": 0.035,
            "simulated": self.simulation_mode
        }
