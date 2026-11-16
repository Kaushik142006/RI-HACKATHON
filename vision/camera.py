"""
Camera Interface Module.
Handles RealSense D435i RGB-D streaming and simulation fallback.
"""

import time
import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class CameraInterface:
    """Acquires synchronized RGB and Depth frames from RealSense D435i or Mock Camera."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30, simulation_mode: bool = True):
        self.width = width
        self.height = height
        self.fps = fps
        self.simulation_mode = simulation_mode
        self.is_connected = False
        self.pipeline = None
        self.align = None
        self._frame_count = 0

        # Intrinsics defaults
        self.intrinsics: Dict[str, float] = {
            "fx": 615.0,
            "fy": 615.0,
            "cx": width / 2.0,
            "cy": height / 2.0,
            "depth_scale": 0.001  # millimeters to meters
        }

    def start(self) -> bool:
        """Initialize and start camera stream."""
        if not self.simulation_mode:
            try:
                import pyrealsense2 as rs
                self.pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
                config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
                profile = self.pipeline.start(config)

                # Get active depth scale
                depth_sensor = profile.get_device().first_depth_sensor()
                self.intrinsics["depth_scale"] = depth_sensor.get_depth_scale()

                # Align depth to color
                self.align = rs.align(rs.stream.color)
                self.is_connected = True
                logger.info("RealSense D435i camera initialized successfully.")
                return True
            except Exception as e:
                logger.warning(f"Failed to connect physical RealSense camera: {e}. Falling back to simulation mode.")
                self.simulation_mode = True

        self.is_connected = True
        logger.info("Initialized Camera in SIMULATION mode.")
        return True

    def get_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve aligned RGB and Depth frames.
        Returns:
            rgb_image (np.ndarray): Shape (H, W, 3), uint8
            depth_image (np.ndarray): Shape (H, W), uint16 or float32 (in mm or meters)
        """
        if not self.is_connected:
            self.start()

        self._frame_count += 1

        if not self.simulation_mode and self.pipeline is not None:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=3000)
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    raise RuntimeError("Received empty frame from RealSense.")

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                return color_image, depth_image
            except Exception as e:
                logger.error(f"Error reading RealSense hardware frame: {e}. Generating simulated frame.")

        return self._generate_simulated_frame()

    def _generate_simulated_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic RGB-D frame containing a tabletop scene with red bottle, blue cup, storage box."""
        # 1. Background tabletop (gray)
        rgb = np.full((self.height, self.width, 3), 210, dtype=np.uint8)
        # Depth map initialized to table plane ~0.85m (850mm)
        depth = np.full((self.height, self.width), 850, dtype=np.uint16)

        # Draw a simulated Red Bottle: center ~ (400, 380), width 70, height 180
        # Color in BGR: Red = [30, 30, 220]
        bx1, by1, bx2, by2 = 365, 290, 435, 470
        rgb[by1:by2, bx1:bx2] = [30, 30, 220]
        depth[by1:by2, bx1:bx2] = 750  # 0.75m distance

        # Draw a simulated Blue Storage Box: center ~ (850, 400), width 220, height 180
        # Color in BGR: Blue = [200, 50, 40]
        sx1, sy1, sx2, sy2 = 740, 310, 960, 490
        rgb[sy1:sy2, sx1:sx2] = [200, 50, 40]
        depth[sy1:sy2, sx1:sx2] = 780  # 0.78m distance

        # Draw a simulated Green Cup: center ~ (600, 350), width 60, height 80
        # Color in BGR: Green = [40, 190, 40]
        gx1, gy1, gx2, gy2 = 570, 310, 630, 390
        rgb[gy1:gy2, gx1:gx2] = [40, 190, 40]
        depth[gy1:gy2, gx1:gx2] = 800  # 0.80m distance

        return rgb, depth

    def get_intrinsics(self) -> Dict[str, float]:
        """Return camera intrinsic parameters."""
        return self.intrinsics

    def stop(self) -> None:
        """Stop camera streams."""
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception as e:
                logger.warning(f"Error stopping camera: {e}")
        self.is_connected = False
        logger.info("Camera stream stopped.")
