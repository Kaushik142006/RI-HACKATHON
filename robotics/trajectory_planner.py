"""
Trajectory Planner Module.
Interpolates time-parameterized joint trajectories using cubic/quintic splines with velocity and acceleration profiles.
"""

from typing import List, Dict, Any
import numpy as np


class TrajectoryPlanner:
    """Computes smooth trajectory points with trapezoidal / polynomial velocity profiles."""

    def __init__(self, max_vel: float = 1.0, max_acc: float = 0.5):
        self.max_vel = max_vel
        self.max_acc = max_acc

    def interpolate_joints(self, start_joints: List[float], goal_joints: List[float], num_steps: int = 10) -> List[Dict[str, Any]]:
        """Generate smooth linearly and polynomial interpolated joint trajectory steps."""
        points = []
        duration = 2.0  # seconds
        dt = duration / float(num_steps)

        for step in range(num_steps + 1):
            s = step / float(num_steps)
            # Quintic polynomial interpolation factor: 10s^3 - 15s^4 + 6s^5
            s_curve = 10 * (s**3) - 15 * (s**4) + 6 * (s**5)

            curr_positions = [
                round(float(start + (goal - start) * s_curve), 3)
                for start, goal in zip(start_joints, goal_joints)
            ]
            curr_velocities = [
                round(float((goal - start) * (30 * s**2 - 60 * s**3 + 30 * s**4) / duration), 3)
                for start, goal in zip(start_joints, goal_joints)
            ]

            points.append({
                "time_from_start_sec": round(step * dt, 2),
                "positions": curr_positions,
                "velocities": curr_velocities
            })

        return points
