"""
Unit Tests for Independent Safety Mechanism.
Tests E-Stop, joint limits, velocity thresholds, workspace boundaries, and trajectory auditing.
"""

import pytest
import numpy as np
from robotics.safety.emergency_stop import EmergencyStopManager
from robotics.safety.joint_limits import JointLimitsChecker
from robotics.safety.collision_checker import CollisionChecker
from robotics.safety.safety_gate import IndependentSafetyGate


def test_emergency_stop():
    estop = EmergencyStopManager()
    assert not estop.is_active()

    estop.trigger_estop("Test E-Stop Trip")
    assert estop.is_active()
    assert estop.get_trip_reason() == "Test E-Stop Trip"

    estop.reset_estop()
    assert not estop.is_active()


def test_joint_limits_checker_valid():
    checker = JointLimitsChecker()
    valid_traj = {
        "joint_names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
        "points": [
            {"positions": [0.0, 0.5, 0.5, 0.0, 0.5, 0.0], "velocities": [0.1]*6}
        ]
    }
    is_valid, msg = checker.validate_trajectory(valid_traj)
    assert is_valid, msg


def test_joint_limits_checker_exceeded():
    checker = JointLimitsChecker()
    # joint_2 limit is [-2.094, 2.094]; test 3.5 rad
    invalid_traj = {
        "joint_names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
        "points": [
            {"positions": [0.0, 3.5, 0.5, 0.0, 0.5, 0.0], "velocities": [0.1]*6}
        ]
    }
    is_valid, msg = checker.validate_trajectory(invalid_traj)
    assert not is_valid
    assert "violates limits" in msg


def test_collision_checker_workspace_bounds():
    checker = CollisionChecker()
    # Point inside safe workspace
    valid, _ = checker.check_cartesian_point(0.3, 0.2, 0.5)
    assert valid

    # Point with negative Z (table penetration)
    valid, msg = checker.check_cartesian_point(0.3, 0.2, -0.05)
    assert not valid
    assert "Table penetration" in msg


def test_independent_safety_gate_audit():
    gate = IndependentSafetyGate(simulation_mode=True)
    valid_traj = {
        "joint_names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
        "points": [
            {"positions": [0.0, 0.5, 0.5, 0.0, 0.5, 0.0], "velocities": [0.1]*6}
        ]
    }
    waypoints = [{"x": 0.3, "y": 0.2, "z": 0.5}]
    is_approved, reason, telemetry = gate.audit_motion_plan(valid_traj, waypoints)
    assert is_approved
    assert telemetry["joint_limits_passed"]
