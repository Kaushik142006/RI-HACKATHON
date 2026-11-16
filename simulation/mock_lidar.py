"""
Mock LiDAR Simulation Module.
Generates synthetic 2D range scans and obstacle profiles for testing safety mechanisms.
"""

from typing import Tuple
import numpy as np


class MockLidar:
    """Simulated 2D laser scanner."""

    def __init__(self, num_points: int = 360, range_max_m: float = 10.0):
        self.num_points = num_points
        self.range_max_m = range_max_m
        self.angles = np.linspace(-np.pi, np.pi, num_points)

    def scan(self, inject_critical_obstacle: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Produce range measurements (in meters).
        inject_critical_obstacle: if True, injects an obstacle at 0.15m (critical stop zone).
        """
        ranges = np.full(self.num_points, 3.0, dtype=np.float32)
        ranges += np.random.uniform(-0.01, 0.01, self.num_points)

        if inject_critical_obstacle:
            # Place an obstacle directly in front at 0.18m
            front_indices = range(175, 185)
            for idx in front_indices:
                ranges[idx] = 0.18

        return ranges, self.angles
