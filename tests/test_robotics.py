"""
Unit Tests for Robotics Manipulation Subsystem.
Tests TF transformations, analytical IK solver, trajectory interpolator, gripper controller, and hardware interface.
"""

import pytest
import numpy as np
from robotics.tf_manager import TFManager
from robotics.inverse_kinematics import IKSolver
from robotics.trajectory_planner import TrajectoryPlanner
from robotics.gripper_controller import GripperController
from robotics.hardware_interface import HardwareInterface


def test_tf_manager():
    tf = TFManager()
    pt = [0.0, 0.0, 0.0]
    # base_link to camera_link
    pt_cam = tf.transform_point(pt, "base_link", "camera_link")
    assert isinstance(pt_cam, list)
    assert len(pt_cam) == 3


def test_ik_solver():
    solver = IKSolver()
    target_xyz = [0.35, 0.20, 0.70]
    joints = solver.solve_ik(target_xyz)
    assert joints is not None
    assert len(joints) == 6
    # Verify q1 corresponds to azimuth angle
    expected_q1 = np.arctan2(0.20, 0.35)
    assert abs(joints[0] - expected_q1) < 0.05


def test_trajectory_planner_interpolation():
    planner = TrajectoryPlanner()
    start_q = [0.0]*6
    goal_q = [0.5]*6
    points = planner.interpolate_joints(start_q, goal_q, num_steps=5)
    assert len(points) == 6  # 0 to 5 steps
    assert points[0]["positions"] == [0.0]*6
    assert points[-1]["positions"] == [0.5]*6


def test_gripper_controller():
    gripper = GripperController(simulation_mode=True)
    res_open = gripper.open_gripper()
    assert res_open["position"] == 0.08
    assert not res_open["holding"]

    res_close = gripper.close_gripper(target_position=0.03, effort=40.0)
    assert res_close["holding"]
    assert res_close["status"] == "GRASPED"


def test_hardware_interface():
    hw = HardwareInterface(simulation_mode=True)
    assert hw.on_init()
    assert hw.write_commands([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    state = hw.read_state()
    assert state["positions"][0] == 0.1
