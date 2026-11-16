"""
Mock Robot Arm & Gripper Simulation Module.
Maintains simulated joint states, forward kinematics, gripper position, and payload state.
"""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MockRobotArm:
    """Simulated 6-DOF robotic manipulator and parallel jaw gripper."""

    def __init__(self):
        self.joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        self.joint_positions = [0.0, -0.785, 1.57, -0.785, 1.57, 0.0]
        self.gripper_position = 0.08  # Open
        self.is_holding = False
        self.attached_object = None

    def execute_joint_trajectory(self, trajectory: Dict[str, Any]) -> bool:
        """Play through trajectory points updating arm joint state."""
        points = trajectory.get("points", [])
        if not points:
            return False

        last_pt = points[-1]
        self.joint_positions = list(last_pt.get("positions", self.joint_positions))
        logger.info(f"[MockRobotArm] Completed trajectory. Final joint state: {self.joint_positions}")
        return True

    def set_gripper(self, position_m: float, effort_n: float) -> bool:
        """Update simulated gripper jaw spacing."""
        self.gripper_position = position_m
        # Grasped if closed to between 0.01 and 0.05m
        if 0.01 <= position_m <= 0.05:
            self.is_holding = True
            logger.info(f"[MockRobotArm] Gripper grasped object at {position_m*100:.1f} cm.")
        else:
            self.is_holding = False
            self.attached_object = None
            logger.info(f"[MockRobotArm] Gripper released. Position: {position_m*100:.1f} cm.")
        return True

    def get_state(self) -> Dict[str, Any]:
        """Return current simulated robot telemetry."""
        return {
            "joint_positions": list(self.joint_positions),
            "gripper_position": self.gripper_position,
            "is_holding_object": self.is_holding,
            "attached_object": self.attached_object
        }
