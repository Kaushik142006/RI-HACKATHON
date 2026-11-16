"""
Agent 7 – Task Planning Agent.
Decomposes high-level commands into sequential atomic robot manipulation primitives:
Locate object -> Approach -> Grasp -> Lift -> Move -> Place -> Release -> Retract.
"""

from typing import List, Dict, Any
import logging
from orchestration.state import AgentState

logger = logging.getLogger(__name__)


class TaskPlanningAgent:
    """Agent 7: Decomposes manipulation tasks into atomic robotic action steps."""

    def __init__(self):
        self.agent_id = "agent_7_task_planning"

    def plan_sequence(self, grasp_pose: Dict[str, float], place_pose: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Generate sequential atomic manipulation primitives.
        """
        gx, gy, gz = grasp_pose.get("x", 0.3), grasp_pose.get("y", 0.2), grasp_pose.get("z", 0.7)
        px, py, pz = place_pose.get("x", 0.4), place_pose.get("y", -0.2), place_pose.get("z", 0.75)

        return [
            {
                "step": 1,
                "name": "open_gripper",
                "primitive": "gripper_command",
                "params": {"position": 0.08, "effort": 30.0}
            },
            {
                "step": 2,
                "name": "approach_pre_grasp",
                "primitive": "cartesian_move",
                "params": {"target_position": {"x": gx, "y": gy, "z": gz + 0.12}, "speed_factor": 0.5}
            },
            {
                "step": 3,
                "name": "descend_to_grasp",
                "primitive": "cartesian_move",
                "params": {"target_position": {"x": gx, "y": gy, "z": gz}, "speed_factor": 0.2}
            },
            {
                "step": 4,
                "name": "close_gripper_grasp",
                "primitive": "gripper_command",
                "params": {"position": 0.02, "effort": 80.0}
            },
            {
                "step": 5,
                "name": "lift_object",
                "primitive": "cartesian_move",
                "params": {"target_position": {"x": gx, "y": gy, "z": gz + 0.15}, "speed_factor": 0.3}
            },
            {
                "step": 6,
                "name": "transit_to_destination",
                "primitive": "cartesian_move",
                "params": {"target_position": {"x": px, "y": py, "z": pz + 0.15}, "speed_factor": 0.5}
            },
            {
                "step": 7,
                "name": "descend_to_place",
                "primitive": "cartesian_move",
                "params": {"target_position": {"x": px, "y": py, "z": pz + 0.05}, "speed_factor": 0.2}
            },
            {
                "step": 8,
                "name": "release_gripper",
                "primitive": "gripper_command",
                "params": {"position": 0.08, "effort": 30.0}
            },
            {
                "step": 9,
                "name": "retract_home",
                "primitive": "joint_move",
                "params": {"target_named_pose": "home", "speed_factor": 0.5}
            }
        ]

    def process(self, state: AgentState) -> AgentState:
        """Process agent node in LangGraph."""
        logger.info(f"[{self.agent_id}] Decomposing task into executable action primitives...")

        reqs = state.path_planning_requirements or {}
        grasp_pose = reqs.get("grasp_pose", {"x": 0.32, "y": 0.18, "z": 0.75})
        place_pose = reqs.get("place_pose", {"x": 0.45, "y": -0.25, "z": 0.78})

        sequence = self.plan_sequence(grasp_pose, place_pose)
        state.task_decomposition = sequence

        logger.info(f"[{self.agent_id}] Generated {len(sequence)} atomic action primitives.")
        return state
