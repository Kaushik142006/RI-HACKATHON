"""
Simulated Environment Module.
Coordinates mock sensors, tabletop objects, and simulated robot arm execution.
"""

from typing import Dict, Any, List
import logging
from simulation.mock_camera import MockCamera
from simulation.mock_lidar import MockLidar
from simulation.mock_robot import MockRobotArm

logger = logging.getLogger(__name__)


class SimulatedEnvironment:
    """Integrated virtual tabletop test environment."""

    def __init__(self):
        self.camera = MockCamera()
        self.lidar = MockLidar()
        self.robot = MockRobotArm()
        self.tabletop_objects = [
            {"id": 1, "class": "bottle", "color": "red", "pose": {"x": 0.32, "y": 0.18, "z": 0.75}},
            {"id": 2, "class": "box", "color": "blue", "pose": {"x": 0.45, "y": -0.25, "z": 0.78}},
            {"id": 3, "class": "cup", "color": "green", "pose": {"x": 0.25, "y": 0.35, "z": 0.80}}
        ]

    def reset(self) -> None:
        """Reset environment to initial state."""
        self.robot = MockRobotArm()
        logger.info("Simulated environment reset.")

    def get_ground_truth(self) -> Dict[str, Any]:
        """Return ground truth state for unit and integration testing."""
        return {
            "objects": list(self.tabletop_objects),
            "robot": self.robot.get_state()
        }
