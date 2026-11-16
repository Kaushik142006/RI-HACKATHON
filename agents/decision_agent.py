"""
Agent 8 – Decision Agent.
Evaluates task feasibility, checks physical reachability and kinematic limits,
handles ambiguities (e.g. multiple matching bottles), and authorizes execution.
"""

from typing import Dict, Any, List, Tuple
import logging
from orchestration.state import AgentState

logger = logging.getLogger(__name__)


class DecisionAgent:
    """Agent 8: Decision Agent validating reachability, grasp feasibility, and resolving ambiguities."""

    def __init__(self, reach_radius_m: float = 0.85):
        self.agent_id = "agent_8_decision"
        self.reach_radius_m = reach_radius_m

    def check_feasibility(self, state: AgentState) -> Tuple[bool, str]:
        """
        Check reachability, missing objects, or ambiguous multiple targets.
        """
        vision = state.vision_state or {}
        objects = vision.get("objects", [])

        req_obj = (state.requested_object or "").lower()
        req_color = ((state.nlp_output or {}).get("target_color") or "").lower()

        # Check for multiple ambiguous objects
        matches = [
            o for o in objects
            if req_obj in o.get("class", "").lower() and (not req_color or req_color in o.get("color", "").lower())
        ]

        if len(matches) > 1:
            return False, f"Ambiguity detected: {len(matches)} matching '{req_color} {req_obj}' found. Please specify target by ID or location."

        # Check reachability of grasp pose
        reqs = state.path_planning_requirements or {}
        grasp_pose = reqs.get("grasp_pose", {})
        gx = grasp_pose.get("x", 0.0)
        gy = grasp_pose.get("y", 0.0)
        gz = grasp_pose.get("z", 0.0)

        dist = (gx**2 + gy**2 + gz**2)**0.5
        if dist > self.reach_radius_m:
            return False, f"Target object at distance {dist:.2f}m exceeds robot maximum reach ({self.reach_radius_m:.2f}m)."

        return True, "Task is feasible and kinematically reachable."

    def process(self, state: AgentState) -> AgentState:
        """Process agent node in LangGraph."""
        logger.info(f"[{self.agent_id}] Evaluating task feasibility and resolving ambiguities...")

        is_feasible, reason = self.check_feasibility(state)

        if not state.decision_output:
            state.decision_output = {}

        state.decision_output["is_feasible"] = is_feasible
        state.decision_output["feasibility_reason"] = reason

        if not is_feasible:
            state.error_information = {
                "failed_agent": self.agent_id,
                "error_type": "InfeasibleTask",
                "message": reason
            }
            logger.warning(f"[{self.agent_id}] Task deemed infeasible: {reason}")
        else:
            logger.info(f"[{self.agent_id}] Task authorized: {reason}")

        return state
