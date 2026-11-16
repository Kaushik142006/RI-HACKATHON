"""
Agent 5 – Work Splitter Agent.
Divides the validated task into sub-goals and assigns tasks to Perception,
Task Planning, Decision, and Path Planning agents.
"""

from typing import Dict, Any, List
import logging
from orchestration.state import AgentState

logger = logging.getLogger(__name__)


class WorkSplitterAgent:
    """Agent 5: Work Splitter dividing tasks across specialised planning and perception modules."""

    def __init__(self):
        self.agent_id = "agent_5_work_splitter"

    def process(self, state: AgentState) -> AgentState:
        """Process agent node in LangGraph."""
        logger.info(f"[{self.agent_id}] Dividing validated task into sub-tasks...")

        nlp = state.nlp_output or {}
        action = nlp.get("action", "pick_and_place")
        target_obj = state.requested_object or "bottle"
        target_color = nlp.get("target_color", "red")
        destination = state.requested_destination or "storage box"

        # Define sub-tasks for downstream specialists
        assignments = {
            "perception_task": {
                "target_class": target_obj,
                "target_color": target_color,
                "destination_class": destination
            },
            "task_planning_task": {
                "macro_action": action,
                "phases": ["locate", "approach", "grasp", "lift", "move", "place", "release"]
            },
            "decision_task": {
                "check_reachability": True,
                "check_graspability": True,
                "handle_ambiguity": True
            },
            "path_planning_task": {
                "collision_avoidance": True,
                "coordinate_frame": "base_link"
            }
        }

        # Store sub-task specification in state
        if not state.decision_output:
            state.decision_output = {}
        state.decision_output["work_division"] = assignments

        logger.info(f"[{self.agent_id}] Assigned tasks to Perception, Planning, Decision, and Motion agents.")
        return state
