"""
Unit Tests for LangGraph Multi-Agent Orchestration.
Tests StateGraph routing, prompt chaining, tool functions, and recovery policies.
"""

import pytest
from orchestration.state import AgentState
from orchestration.routing import (
    route_after_environment_check,
    route_after_nlp,
    route_after_safety_gate,
    route_after_verification
)
from orchestration.tool_registry import (
    compute_euclidean_distance,
    check_gripper_affordance,
    validate_workspace_boundaries
)
from orchestration.error_recovery import ErrorRecoveryPolicy


def test_routing_conditions():
    # Environment Check Router
    state = AgentState(environment_status="unsuitable")
    assert route_after_environment_check(state) == "agent_16_error_handling"

    state.environment_status = "manageable"
    assert route_after_environment_check(state) == "agent_3_vision_context"

    # Safety Gate Router
    state.safety_approval = False
    assert route_after_safety_gate(state) == "agent_16_error_handling"

    state.safety_approval = True
    assert route_after_safety_gate(state) == "agent_13_hardware_control"


def test_tool_registry():
    # Distance
    dist = compute_euclidean_distance({"x": 0.0, "y": 0.0, "z": 0.0}, {"x": 3.0, "y": 4.0, "z": 0.0})
    assert abs(dist - 5.0) < 0.001

    # Gripper affordance
    assert check_gripper_affordance({"width": 0.05}, max_gripper_opening=0.08)
    assert not check_gripper_affordance({"width": 0.12}, max_gripper_opening=0.08)

    # Workspace bounds
    assert validate_workspace_boundaries(0.3, 0.2, 0.5, {"x": [-0.8, 0.8], "y": [-0.8, 0.8], "z": [0.0, 1.2]})
    assert not validate_workspace_boundaries(1.5, 0.2, 0.5, {"x": [-0.8, 0.8], "y": [-0.8, 0.8], "z": [0.0, 1.2]})


def test_error_recovery_policy():
    policy = ErrorRecoveryPolicy(max_retries=3)
    assert policy.can_retry(1, "TrajectoryDeviation")
    assert not policy.can_retry(3, "TrajectoryDeviation")
    assert not policy.can_retry(0, "TablePenetration")
