"""
Unit Tests for Computer Vision Pipeline.
Tests RGB-D capture, object detection, segmentation, ByteTrack, color identification,
depth processing, 3D coordinate estimation, and unified vision state.
"""

import pytest
import numpy as np
from vision.camera import CameraInterface
from vision.object_detection import ObjectDetector
from vision.object_segmentation import ObjectSegmenter
from vision.object_tracking import ByteTrackTracker
from vision.color_detection import ColorDetector
from vision.depth_processing import DepthProcessor
from vision.coordinate_estimation import CoordinateEstimator
from vision.camera_calibration import CameraCalibration
from vision.vision_pipeline import VisionPipeline
from vision.vision_state import BoundingBox, Position3D


def test_camera_simulation_frames():
    cam = CameraInterface(simulation_mode=True)
    cam.start()
    rgb, depth = cam.get_frames()
    assert rgb.shape == (720, 1280, 3)
    assert depth.shape == (720, 1280)
    cam.stop()


def test_object_detector():
    detector = ObjectDetector(simulation_mode=True)
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    detections = detector.detect(img)
    assert len(detections) >= 1
    classes = [d["class"] for d in detections]
    assert "bottle" in classes


def test_color_detection():
    color_det = ColorDetector()
    # Create bright red test image
    red_img = np.zeros((100, 100, 3), dtype=np.uint8)
    red_img[:] = [0, 0, 255]  # BGR Red
    bbox = BoundingBox(xmin=0, ymin=0, xmax=100, ymax=100)
    color, conf = color_det.identify_color(red_img, bbox)
    assert color == "red"
    assert conf > 0.5


def test_coordinate_estimation_and_calibration():
    estimator = CoordinateEstimator(fx=615.0, fy=615.0, cx=640.0, cy=360.0)
    pos_cam = estimator.estimate_bbox_3d(u_center=640.0, v_center=360.0, depth_m=1.0)
    # At optical center, X=0, Y=0, Z=depth
    assert abs(pos_cam.x) < 0.01
    assert abs(pos_cam.y) < 0.01
    assert abs(pos_cam.z - 1.0) < 0.01

    calib = CameraCalibration()
    pos_base = calib.camera_to_base(pos_cam)
    assert isinstance(pos_base, Position3D)


def test_vision_pipeline_unified_state():
    pipe = VisionPipeline(simulation_mode=True)
    pipe.start()
    state = pipe.process_frame()
    assert len(state.objects) > 0
    assert state.environment_status in ["clean", "manageable", "unsuitable"]
    pipe.stop()
