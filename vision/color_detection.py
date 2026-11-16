"""
OpenCV Color Identification Module.
Classifies object color using HSV color space thresholding and histogram analysis.
"""

from typing import Tuple, Dict
import logging
import cv2
import numpy as np
from vision.vision_state import BoundingBox

logger = logging.getLogger(__name__)


class ColorDetector:
    """Identifies the dominant color of a detected object region."""

    COLOR_RANGES = {
        "red": [
            (np.array([0, 100, 100]), np.array([10, 255, 255])),
            (np.array([170, 100, 100]), np.array([180, 255, 255]))
        ],
        "blue": [
            (np.array([95, 100, 80]), np.array([130, 255, 255]))
        ],
        "green": [
            (np.array([35, 70, 70]), np.array([85, 255, 255]))
        ],
        "yellow": [
            (np.array([20, 100, 100]), np.array([35, 255, 255]))
        ],
        "orange": [
            (np.array([11, 100, 100]), np.array([19, 255, 255]))
        ],
        "black": [
            (np.array([0, 0, 0]), np.array([180, 255, 40]))
        ],
        "white": [
            (np.array([0, 0, 200]), np.array([180, 40, 255]))
        ]
    }

    def identify_color(self, image: np.ndarray, bbox: BoundingBox, mask: np.ndarray = None) -> Tuple[str, float]:
        """
        Identify dominant color within bounding box or segmentation mask.
        Returns:
            Tuple of (color_name: str, confidence: float)
        """
        h, w = image.shape[:2]
        x1 = max(0, int(bbox.xmin))
        y1 = max(0, int(bbox.ymin))
        x2 = min(w, int(bbox.xmax))
        y2 = min(h, int(bbox.ymax))

        if x2 <= x1 or y2 <= y1:
            return "unknown", 0.0

        crop = image[y1:y2, x1:x2]
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        total_pixels = crop.shape[0] * crop.shape[1]
        if total_pixels == 0:
            return "unknown", 0.0

        best_color = "unknown"
        max_matches = 0

        for color_name, ranges in self.COLOR_RANGES.items():
            color_mask = np.zeros(hsv_crop.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                sub_mask = cv2.inRange(hsv_crop, lower, upper)
                color_mask = cv2.bitwise_or(color_mask, sub_mask)

            matched_pixels = cv2.countNonZero(color_mask)
            if matched_pixels > max_matches:
                max_matches = matched_pixels
                best_color = color_name

        confidence = float(max_matches / total_pixels) if total_pixels > 0 else 0.0
        # If very low confidence, default to nearest hue
        if confidence < 0.15:
            # Fallback to mean HSV
            mean_hsv = cv2.mean(hsv_crop)[:3]
            h_val = mean_hsv[0]
            if h_val < 10 or h_val > 170:
                best_color = "red"
            elif 95 <= h_val <= 130:
                best_color = "blue"
            elif 35 <= h_val <= 85:
                best_color = "green"
            elif 20 <= h_val <= 35:
                best_color = "yellow"
            confidence = 0.50

        return best_color, min(1.0, confidence)
