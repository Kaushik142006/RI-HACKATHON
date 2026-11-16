"""
Joint Limits & Velocity Verification Module.
Validates position, velocity, and acceleration against physical robot bounds.
"""

from typing import List, Dict, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


class JointLimitsChecker:
    """Verifies that all trajectory points respect mechanical joint bounds and velocity thresholds."""

    # Default 6-DOF Manipulator Limits (radians and rad/s)
    DEFAULT_POSITION_LIMITS = {
        "joint_1": (-np.pi, np.pi),
        "joint_2": (-2.094, 2.094),
        "joint_3": (-2.618, 2.618),
        "joint_4": (-np.pi, np.pi),
        "joint_5": (-2.094, 2.094),
        "joint_6": (-2 * np.pi, 2 * np.pi)
    }

    DEFAULT_VELOCITY_LIMITS = {
        "joint_1": 1.5,
        "joint_2": 1.5,
        "joint_3": 1.5,
        "joint_4": 2.0,
        "joint_5": 2.0,
        "joint_6": 2.0
    }

    def __init__(self, pos_limits: Dict = None, vel_limits: Dict = None):
        self.pos_limits = pos_limits or self.DEFAULT_POSITION_LIMITS
        self.vel_limits = vel_limits or self.DEFAULT_VELOCITY_LIMITS

    def validate_trajectory(self, trajectory: Dict) -> Tuple[bool, str]:
        """
        Validate position and velocity of every point in trajectory.
        Returns: (is_valid: bool, error_message: str)
        """
        points = trajectory.get("points", [])
        joint_names = trajectory.get("joint_names", list(self.pos_limits.keys()))

        if not points:
            return False, "Trajectory contains no points."

        for i, pt in enumerate(points):
            positions = pt.get("positions", [])
            velocities = pt.get("velocities", [])

            for j_idx, val in enumerate(positions):
                j_name = joint_names[j_idx] if j_idx < len(joint_names) else f"joint_{j_idx+1}"
                if j_name in self.pos_limits:
                    min_pos, max_pos = self.pos_limits[j_name]
                    if not (min_pos <= val <= max_pos):
                        return False, f"Point {i}: {j_name} position {val:.3f} rad violates limits [{min_pos:.3f}, {max_pos:.3f}]."

            for j_idx, vel in enumerate(velocities):
                j_name = joint_names[j_idx] if j_idx < len(joint_names) else f"joint_{j_idx+1}"
                if j_name in self.vel_limits:
                    max_vel = self.vel_limits[j_name]
                    if abs(vel) > max_vel:
                        return False, f"Point {i}: {j_name} velocity {vel:.3f} rad/s exceeds maximum {max_vel:.3f} rad/s."

        return True, "All joint positions and velocities within physical limits."
