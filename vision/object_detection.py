"""
YOLOv8 Object Detection Module.
Detects objects and extracts bounding boxes, confidences, and CNN class labels.
"""

from typing import List, Dict, Any, Optional
import logging
import numpy as np
from vision.vision_state import BoundingBox

logger = logging.getLogger(__name__)


class ObjectDetector:
    """Wrapper for YOLOv8 Object Detection with simulation fallback."""

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.5, iou_threshold: float = 0.45, simulation_mode: bool = True):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.simulation_mode = simulation_mode
        self.model = None

        if not self.simulation_mode:
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                logger.info(f"Loaded YOLOv8 detection model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load YOLOv8 model ({e}). Using simulation detection engine.")
                self.simulation_mode = True

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run object detection on an RGB/BGR image.
        Returns:
            List of dicts: [{"class": str, "confidence": float, "bbox": BoundingBox, "class_id": int}]
        """
        if not self.simulation_mode and self.model is not None:
            try:
                results = self.model.predict(
                    source=image,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
                detections = []
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        coords = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        cls_name = r.names[cls_id]
                        detections.append({
                            "class": cls_name,
                            "confidence": conf,
                            "bbox": BoundingBox(xmin=float(coords[0]), ymin=float(coords[1]), xmax=float(coords[2]), ymax=float(coords[3])),
                            "class_id": cls_id
                        })
                return detections
            except Exception as e:
                logger.error(f"Error during YOLO detection: {e}. Falling back to simulation.")

        # Simulation Detections matching simulated camera frame
        return [
            {
                "class": "bottle",
                "confidence": 0.94,
                "bbox": BoundingBox(xmin=365.0, ymin=290.0, xmax=435.0, ymax=470.0),
                "class_id": 39
            },
            {
                "class": "box",
                "confidence": 0.91,
                "bbox": BoundingBox(xmin=740.0, ymin=310.0, xmax=960.0, ymax=490.0),
                "class_id": 40
            },
            {
                "class": "cup",
                "confidence": 0.88,
                "bbox": BoundingBox(xmin=570.0, ymin=310.0, xmax=630.0, ymax=390.0),
                "class_id": 41
            }
        ]
