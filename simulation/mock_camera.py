"""
Mock Camera Simulation Module.
Generates synthetic RGB and Depth frames representing a realistic tabletop scenario.
"""

from typing import Tuple, Dict, Any
import numpy as np


class MockCamera:
    """Simulated RealSense D435i RGB-D sensor."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.intrinsics = {
            "fx": 615.0,
            "fy": 615.0,
            "cx": width / 2.0,
            "cy": height / 2.0,
            "depth_scale": 0.001
        }

    def capture_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Produce synthetic RGB and Depth numpy arrays.
        Renders a tabletop containing:
        1. Red bottle
        2. Blue storage box
        3. Green cup
        """
        # Gray tabletop
        rgb = np.full((self.height, self.width, 3), 200, dtype=np.uint8)
        depth = np.full((self.height, self.width), 850, dtype=np.uint16)

        # Draw simulated red bottle (BGR format)
        rgb[290:470, 365:435] = [20, 20, 220]
        depth[290:470, 365:435] = 750

        # Draw simulated blue storage box
        rgb[310:490, 740:960] = [210, 40, 30]
        depth[310:490, 740:960] = 780

        # Draw simulated green cup
        rgb[310:390, 570:630] = [30, 200, 30]
        depth[310:390, 570:630] = 800

        return rgb, depth
