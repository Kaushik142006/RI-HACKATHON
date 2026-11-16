"""
Unified Vision Pipeline.
Coordinates all vision modules: RGB-D capture, YOLO detection, segmentation,
tracking, color identification, 3D localization, and obstacle estimation.
"""

import time
import logging
from typing import Optional, Dict, Any
from vision.camera import CameraInterface
from vision.object_detection import ObjectDetector
from vision.object_segmentation import ObjectSegmenter
from vision.object_tracking import ByteTrackTracker
from vision.color_detection import ColorDetector
from vision.depth_processing import DepthProcessor
from vision.coordinate_estimation import CoordinateEstimator
from vision.camera_calibration import CameraCalibration
from vision.lidar_processing import LidarProcessor
from vision.obstacle_detection import ObstacleDetector
from vision.vision_state import UnifiedVisionState, DetectedObject, EnvironmentStatus, Position3D

logger = logging.getLogger(__name__)


class VisionPipeline:
    """Master pipeline producing structured UnifiedVisionState."""

    def __init__(self, simulation_mode: bool = True):
        self.simulation_mode = simulation_mode
        self.camera = CameraInterface(simulation_mode=simulation_mode)
        self.detector = ObjectDetector(simulation_mode=simulation_mode)
        self.segmenter = ObjectSegmenter(simulation_mode=simulation_mode)
        self.tracker = ByteTrackTracker(simulation_mode=simulation_mode)
        self.color_detector = ColorDetector()
        self.depth_processor = DepthProcessor()
        self.coord_estimator = CoordinateEstimator()
        self.calibration = CameraCalibration()
        self.lidar = LidarProcessor(simulation_mode=simulation_mode)
        self.obstacle_detector = ObstacleDetector()
        self._frame_count = 0

    def start(self) -> None:
        """Start hardware sensors or simulation mocks."""
        self.camera.start()
        logger.info("Vision Pipeline started.")

    def process_frame(self) -> UnifiedVisionState:
        """
        Execute full vision perception cycle:
        1. Capture RGB-D frames.
        2. Detect objects & bounding boxes.
        3. Segment masks.
        4. Track objects across frames.
        5. Extract color.
        6. Compute 3D coordinates in robot base frame.
        7. Scan LiDAR and detect obstacles.
        8. Evaluate environment status.
        """
        self._frame_count += 1
        rgb, depth = self.camera.get_frames()

        # Step 1: Detect objects
        raw_detections = self.detector.detect(rgb)

        # Step 2: Track objects
        tracked_detections = self.tracker.update(raw_detections)

        # Step 3: Segment objects
        segmented = self.segmenter.segment(rgb)

        detected_objects = []
        obj_id = 1
        for det in tracked_detections:
            bbox = det["bbox"]
            cls_name = det["class"]
            conf = det["confidence"]
            track_id = det.get("track_id", obj_id)

            # Determine color
            color_name, color_conf = self.color_detector.identify_color(rgb, bbox)

            # Depth in meters
            depth_m = self.depth_processor.get_object_depth(depth, bbox)

            # 3D coordinates in camera frame
            cx, cy = bbox.center
            cam_pos = self.coord_estimator.estimate_bbox_3d(cx, cy, depth_m)

            # Transform to robot base frame
            base_pos = self.calibration.camera_to_base(cam_pos)

            detected_objects.append(DetectedObject(
                object_id=obj_id,
                track_id=track_id,
                class_name=cls_name,
                color=color_name,
                confidence=round(conf, 2),
                bbox=bbox,
                position_3d=base_pos,
                dimensions_3d={"width": 0.08, "height": 0.20, "depth": 0.08},
                graspable=(cls_name in ["bottle", "cup", "can", "apple"])
            ))
            obj_id += 1

        # Step 4: LiDAR obstacles
        ranges, angles = self.lidar.get_scan()
        lidar_pts = self.lidar.ranges_to_points_2d(ranges, angles)
        obstacles = self.obstacle_detector.detect_obstacles(lidar_pts, {"x": (-0.8, 0.8), "y": (-0.8, 0.8)})

        # Step 5: Evaluate Environment Suitability
        num_objects = len(detected_objects)
        num_obstacles = len(obstacles)
        clutter_ratio = min(1.0, num_objects / 10.0)

        if any(obs.threat_level == "critical" for obs in obstacles):
            env_status = "unsuitable"
            is_safe = False
            reason = "Critical obstacle inside manipulator workspace."
        elif num_objects > 8:
            env_status = "unsuitable"
            is_safe = False
            reason = "Extreme clutter detected in workspace."
        elif num_objects > 0:
            env_status = "manageable"
            is_safe = True
            reason = None
        else:
            env_status = "clean"
            is_safe = True
            reason = None

        status_details = EnvironmentStatus(
            status=env_status,
            is_safe=is_safe,
            clutter_ratio=round(clutter_ratio, 2),
            detected_object_count=num_objects,
            obstacle_count=num_obstacles,
            rejection_reason=reason
        )

        return UnifiedVisionState(
            timestamp=time.time(),
            frame_id=self._frame_count,
            objects=detected_objects,
            obstacles=obstacles,
            environment_status=env_status,
            details=status_details,
            table_plane={"z": 0.0, "normal": [0.0, 0.0, 1.0]}
        )

    def stop(self) -> None:
        """Stop sensors."""
        self.camera.stop()
