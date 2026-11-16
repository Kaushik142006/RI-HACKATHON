"""
Hardware Interface Module.
Abstraction layer over ros2_control HardwareComponentInterface, managing joint actuators and read/write cycles.
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class HardwareInterface:
    """Manages command and state interfaces for arm actuators and gripper."""

    def __init__(self, joint_names: Optional[List[str]] = None, simulation_mode: bool = True):
        self.joint_names = joint_names or [f"joint_{i+1}" for i in range(6)]
        self.simulation_mode = simulation_mode
        self.joint_positions = [0.0] * len(self.joint_names)
        self.joint_velocities = [0.0] * len(self.joint_names)
        self.joint_efforts = [0.0] * len(self.joint_names)
        self.is_active = False

    def on_init(self) -> bool:
        """Initialize communication with motor drives."""
        self.is_active = True
        logger.info(f"Hardware Interface initialized in {'SIMULATION' if self.simulation_mode else 'PHYSICAL'} mode.")
        return True

    def read_state(self) -> Dict[str, List[float]]:
        """Read joint positions, velocities, and currents."""
        return {
            "positions": list(self.joint_positions),
            "velocities": list(self.joint_velocities),
            "efforts": list(self.joint_efforts)
        }

    def write_commands(self, target_positions: List[float]) -> bool:
        """Write position setpoints to actuators."""
        if not self.is_active:
            logger.error("Cannot write commands: Hardware interface not active.")
            return False

        self.joint_positions = list(target_positions)
        return True
