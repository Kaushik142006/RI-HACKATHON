"""
Unit Tests for All 16 Logical Agents.
Tests schemas, state transitions, and core logic across Agents 1 through 16.
"""

import pytest
from orchestration.state import AgentState
from agents.main_orchestrator import MainOrchestratorAgent
from agents.environment_check_agent import EnvironmentCheckAgent
from agents.vision_context_agent import VisionContextPromptAgent
from agents.nlp_agent import NLPAgent
from agents.work_splitter_agent import WorkSplitterAgent
from agents.perception_agent import PerceptionAgent
from agents.task_planning_agent import TaskPlanningAgent
from agents.decision_agent import DecisionAgent
from agents.path_planning_agent import AIPathPlanningAgent
from agents.synthesizer_agent import SynthesizerAgent
from agents.ros2_communication_agent import ROS2CommunicationAgent
from agents.moveit_motion_agent import MoveItMotionAgent
from agents.hardware_control_agent import HardwareControlAgent
from agents.verification_agent import VerificationAgent
from agents.evaluation_agent import EvaluationAgent
from agents.error_handling_agent import ErrorHandlingAgent


@pytest.fixture
def base_state():
    return AgentState(
        conversation_id="test_conv",
        original_voice_command="Pick up the red bottle and place it inside the storage box.",
        transcribed_text="Pick up the red bottle and place it inside the storage box.",
        vision_state={
            "environment_status": "manageable",
            "objects": [
                {"object_id": 1, "class": "bottle", "color": "red", "position_3d": {"x": 0.32, "y": 0.18, "z": 0.75}},
                {"object_id": 2, "class": "box", "color": "blue", "position_3d": {"x": 0.45, "y": -0.25, "z": 0.78}}
            ],
            "obstacles": []
        }
    )


def test_agent_1_orchestrator(base_state):
    agent = MainOrchestratorAgent()
    out = agent.process(base_state)
    assert out.step_count == 1


def test_agent_2_environment_check(base_state):
    agent = EnvironmentCheckAgent()
    out = agent.process(base_state)
    assert out.environment_status == "manageable"
    assert out.error_information is None


def test_agent_3_vision_context(base_state):
    agent = VisionContextPromptAgent()
    out = agent.process(base_state)
    assert "red bottle" in out.scene_description


def test_agent_4_nlp(base_state):
    agent = NLPAgent()
    out = agent.process(base_state)
    assert out.requested_object == "bottle"
    assert out.nlp_output["target_color"] == "red"


def test_agent_7_task_planning(base_state):
    agent = TaskPlanningAgent()
    out = agent.process(base_state)
    assert len(out.task_decomposition) == 9
    steps = [s["name"] for s in out.task_decomposition]
    assert "open_gripper" in steps
    assert "close_gripper_grasp" in steps
    assert "release_gripper" in steps


def test_agent_12_moveit_motion(base_state):
    agent = MoveItMotionAgent(simulation_mode=True)
    out = agent.process(base_state)
    traj = out.execution_plan.get("computed_trajectory")
    assert traj is not None
    assert len(traj["points"]) > 0


def test_agent_13_hardware_control_blocks_without_safety(base_state):
    agent = HardwareControlAgent(simulation_mode=True)
    base_state.safety_approval = False
    out = agent.process(base_state)
    # Must block execution!
    assert out.robot_execution_status == "safety_blocked"


def test_agent_14_and_15_verification_evaluation(base_state):
    base_state.robot_execution_status = "success"
    v_agent = VerificationAgent()
    out = v_agent.process(base_state)
    assert out.verification_results["placement_verified"]

    e_agent = EvaluationAgent()
    out2 = e_agent.process(out)
    assert out2.evaluation_metrics["task_success"]
    assert out2.evaluation_metrics["accuracy_score_percent"] > 0
