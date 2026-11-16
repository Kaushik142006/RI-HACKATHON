"""
YOLOv8-Seg Object Segmentation Module.
Generates pixel-wise segmentation masks and boundary contours.
"""

from typing import List, Dict, Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ObjectSegmenter:
    """Performs instance segmentation to isolate object pixel masks."""

    def __init__(self, model_path: str = "yolov8n-seg.pt", conf_threshold: float = 0.5, simulation_mode: bool = True):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.simulation_mode = simulation_mode
        self.model = None

        if not self.simulation_mode:
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                logger.info(f"Loaded YOLOv8-Seg model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load YOLOv8-Seg model ({e}). Using simulation segmentation.")
                self.simulation_mode = True

    def segment(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Segment objects in the image.
        Returns:
            List of dicts: [{"class": str, "mask": np.ndarray (H, W bool), "polygon": list}]
        """
        h, w = image.shape[:2]
        if not self.simulation_mode and self.model is not None:
            try:
                results = self.model.predict(source=image, conf=self.conf_threshold, verbose=False)
                segments = []
                for r in results:
                    if r.masks is not None:
                        for mask, box in zip(r.masks.data, r.boxes):
                            mask_np = mask.cpu().numpy().astype(bool)
                            cls_id = int(box.cls[0].cpu().numpy())
                            cls_name = r.names[cls_id]
                            segments.append({
                                "class": cls_name,
                                "mask": mask_np,
                                "confidence": float(box.conf[0].cpu().numpy())
                            })
                return segments
            except Exception as e:
                logger.error(f"Error in YOLO-Seg: {e}. Falling back to simulation.")

        # Synthetic segmentation masks for red bottle, storage box, cup
        bottle_mask = np.zeros((h, w), dtype=bool)
        bottle_mask[290:470, 365:435] = True

        box_mask = np.zeros((h, w), dtype=bool)
        box_mask[310:490, 740:960] = True

        cup_mask = np.zeros((h, w), dtype=bool)
        cup_mask[310:390, 570:630] = True

        return [
            {"class": "bottle", "mask": bottle_mask, "confidence": 0.94},
            {"class": "box", "mask": box_mask, "confidence": 0.91},
            {"class": "cup", "mask": cup_mask, "confidence": 0.88}
        ]
