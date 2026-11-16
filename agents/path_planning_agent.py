"""
Agent 9 – AI Path Planning Agent.
Coordinates high-level collision-aware path planning, translates tasks into geometric waypoints,
and interfaces with the ROS 2 / MoveIt 2 motion planning pipeline.
"""

from typing import Dict, Any, List
import logging
from orchestration.state import AgentState

logger = logging.getLogger(__name__)


class AIPathPlanningAgent:
    """Agent 9: AI Path Planning Agent synthesizing spatial waypoints and collision constraints."""

    def __init__(self):
        self.agent_id = "agent_9_path_planning"

    def generate_waypoints(self, state: AgentState) -> List[Dict[str, Any]]:
        """Compute Cartesian waypoints for the manipulator tool center point (TCP)."""
        reqs = state.path_planning_requirements or {}
        gp = reqs.get("grasp_pose", {"x": 0.32, "y": 0.18, "z": 0.75})
        pp = reqs.get("place_pose", {"x": 0.45, "y": -0.25, "z": 0.78})

        # Generate smooth intermediate transit points avoiding direct straight collision with table
        waypoints = [
            {"name": "pre_grasp", "x": gp["x"], "y": gp["y"], "z": gp["z"] + 0.15, "roll": 180.0, "pitch": 0.0, "yaw": 0.0},
            {"name": "grasp", "x": gp["x"], "y": gp["y"], "z": gp["z"], "roll": 180.0, "pitch": 0.0, "yaw": 0.0},
            {"name": "lift", "x": gp["x"], "y": gp["y"], "z": gp["z"] + 0.20, "roll": 180.0, "pitch": 0.0, "yaw": 0.0},
            {"name": "transit", "x": (gp["x"] + pp["x"]) / 2, "y": (gp["y"] + pp["y"]) / 2, "z": max(gp["z"], pp["z"]) + 0.25, "roll": 180.0, "pitch": 0.0, "yaw": 0.0},
            {"name": "pre_place", "x": pp["x"], "y": pp["y"], "z": pp["z"] + 0.15, "roll": 180.0, "pitch": 0.0, "yaw": 0.0},
            {"name": "place", "x": pp["x"], "y": pp["y"], "z": pp["z"] + 0.04, "roll": 180.0, "pitch": 0.0, "yaw": 0.0},
            {"name": "retract", "x": pp["x"], "y": pp["y"], "z": pp["z"] + 0.20, "roll": 180.0, "pitch": 0.0, "yaw": 0.0}
        ]
        return waypoints

    def process(self, state: AgentState) -> AgentState:
        """Process agent node in LangGraph."""
        logger.info(f"[{self.agent_id}] Generating high-level geometric path waypoints...")

        waypoints = self.generate_waypoints(state)

        if not state.path_planning_requirements:
            state.path_planning_requirements = {}

        state.path_planning_requirements["waypoints"] = waypoints
        state.path_planning_requirements["planner_id"] = "RRTConnect"
        state.path_planning_requirements["allowed_planning_time"] = 5.0

        logger.info(f"[{self.agent_id}] Generated {len(waypoints)} trajectory waypoints.")
        return state
