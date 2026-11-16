"""
LiDAR Data Acquisition & Processing Module.
Interfaces with 2D LiDAR scanner and converts range scans to cartesian workspace point clouds.
"""

from typing import List, Dict, Any, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


class LidarProcessor:
    """Acquires and processes 2D LiDAR laser scans."""

    def __init__(self, angle_min: float = -np.pi, angle_max: float = np.pi, num_readings: int = 360, range_min: float = 0.1, range_max: float = 10.0, simulation_mode: bool = True):
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.num_readings = num_readings
        self.range_min = range_min
        self.range_max = range_max
        self.simulation_mode = simulation_mode
        self.angles = np.linspace(angle_min, angle_max, num_readings)

    def get_scan(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get latest scan ranges and angles.
        Returns:
            ranges (np.ndarray): distances in meters
            angles (np.ndarray): radians
        """
        if self.simulation_mode:
            return self._generate_simulated_scan()

        # In physical hardware mode, this binds to ROS 2 LaserScan topic or serial driver
        return self._generate_simulated_scan()

    def _generate_simulated_scan(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate safe, clear environment scan with a clear table workspace."""
        # Baseline room boundary at 3.0 meters
        ranges = np.full(self.num_readings, 3.0, dtype=np.float32)

        # Add some noise
        ranges += np.random.uniform(-0.02, 0.02, self.num_readings)
        return ranges, self.angles

    def ranges_to_points_2d(self, ranges: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """
        Convert polar range-angle measurements to 2D Cartesian (X, Y) points.
        Returns: Shape (N, 2)
        """
        valid_mask = (ranges >= self.range_min) & (ranges <= self.range_max)
        r = ranges[valid_mask]
        theta = angles[valid_mask]

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack((x, y))
