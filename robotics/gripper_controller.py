"""
Gripper Controller Module.
Controls parallel jaw gripper stroke, grasping force, and release cycles.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class GripperController:
    """Controls gripper open/close operations and reads grasp effort sensors."""

    def __init__(self, max_stroke_m: float = 0.08, max_force_n: float = 100.0, simulation_mode: bool = True):
        self.max_stroke_m = max_stroke_m
        self.max_force_n = max_force_n
        self.simulation_mode = simulation_mode
        self.current_position = max_stroke_m  # Initially open
        self.is_holding_object = False

    def open_gripper(self) -> Dict[str, Any]:
        """Open gripper to maximum stroke."""
        self.current_position = self.max_stroke_m
        self.is_holding_object = False
        logger.info(f"Gripper opened to {self.max_stroke_m * 100:.1f} cm.")
        return {"position": self.current_position, "holding": False, "status": "OPEN"}

    def close_gripper(self, target_position: float = 0.02, effort: float = 50.0) -> Dict[str, Any]:
        """Close gripper until resistance or target stroke reached."""
        self.current_position = max(0.002, target_position)
        # If position is around object width (~0.02m-0.06m), grasp is solid
        self.is_holding_object = (0.01 <= self.current_position <= 0.07)
        logger.info(f"Gripper closed to {self.current_position * 100:.1f} cm with {effort} N force. Object grasped: {self.is_holding_object}")
        return {
            "position": self.current_position,
            "force_applied_n": min(effort, self.max_force_n),
            "holding": self.is_holding_object,
            "status": "GRASPED" if self.is_holding_object else "CLOSED_EMPTY"
        }
