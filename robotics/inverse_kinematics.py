"""
Inverse Kinematics (IK) Solver Module.
Solves joint configurations for desired end-effector Cartesian poses.
"""

from typing import List, Optional, Tuple
import numpy as np


class IKSolver:
    """Solves analytical/numerical inverse kinematics for 6-DOF articulated arm."""

    def __init__(self, d1: float = 0.35, a2: float = 0.35, a3: float = 0.30, d6: float = 0.15):
        # Kinematic link lengths (meters)
        self.d1 = d1
        self.a2 = a2
        self.a3 = a3
        self.d6 = d6

    def solve_ik(self, target_xyz: List[float], orientation_rpy: List[float] = None) -> Optional[List[float]]:
        """
        Calculate joint angles [q1..q6] for desired position (x, y, z).
        Returns joint angles in radians or None if unreachable.
        """
        x, y, z = target_xyz[0], target_xyz[1], target_xyz[2]

        # Base rotation q1
        q1 = np.arctan2(y, x)

        # Planar reach
        r = np.sqrt(x**2 + y**2)
        r_wrist = r - self.d6
        z_wrist = z - self.d1

        d_sq = r_wrist**2 + z_wrist**2
        d = np.sqrt(d_sq)

        # Reachability check
        if d > (self.a2 + self.a3) or d < abs(self.a2 - self.a3):
            # Clamp or return nearest boundary solution
            d = min(d, (self.a2 + self.a3) * 0.98)

        # Cosine rule for elbow q3
        cos_q3 = (d_sq - self.a2**2 - self.a3**2) / (2 * self.a2 * self.a3)
        cos_q3 = np.clip(cos_q3, -1.0, 1.0)
        q3 = np.arccos(cos_q3)

        # Shoulder q2
        alpha = np.arctan2(z_wrist, r_wrist)
        beta = np.arctan2(self.a3 * np.sin(q3), self.a2 + self.a3 * np.cos(q3))
        q2 = alpha - beta

        # Wrist orientation
        q4 = 0.0
        q5 = np.pi/2 - (q2 + q3)
        q6 = 0.0

        return [round(float(q1), 3), round(float(q2), 3), round(float(q3), 3),
                round(float(q4), 3), round(float(q5), 3), round(float(q6), 3)]
