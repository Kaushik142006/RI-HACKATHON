"""
Agent 6 – Perception Agent.
Extracts spatial coordinates, bounding limits, and depth for target object and destination.
"""

from typing import Dict, Any, Optional
import logging
from orchestration.state import AgentState

logger = logging.getLogger(__name__)


class PerceptionAgent:
    """Agent 6: Analyzes visual information and isolates target object and destination 3D coordinates."""

    def __init__(self):
        self.agent_id = "agent_6_perception"

    def process(self, state: AgentState) -> AgentState:
        """Process agent node in LangGraph."""
        logger.info(f"[{self.agent_id}] Extracting spatial data for requested target and destination...")

        vision = state.vision_state or {}
        objects = vision.get("objects", [])

        req_obj = (state.requested_object or "bottle").lower()
        req_color = (state.nlp_output or {}).get("target_color", "").lower()
        dest_name = (state.requested_destination or "box").lower()

        target_match = None
        dest_match = None

        for obj in objects:
            cls = obj.get("class", "").lower()
            col = obj.get("color", "").lower()

            # Target object matching
            if req_obj in cls:
                if not req_color or req_color in col:
                    target_match = obj

            # Destination matching
            if any(d in cls for d in ["box", "bin", "tray"]) or any(d in dest_name for d in [cls]):
                dest_match = obj

        if not target_match:
            # Fallback to first object or generate simulated tabletop position
            logger.warning(f"[{self.agent_id}] Target '{req_color} {req_obj}' not directly matched. Using first available object.")
            if objects:
                target_match = objects[0]
            else:
                target_match = {
                    "object_id": 1,
                    "class": req_obj,
                    "color": req_color or "red",
                    "position_3d": {"x": 0.32, "y": 0.18, "z": 0.75}
                }

        if not dest_match:
            # Destination fallback position
            dest_match = {
                "object_id": 2,
                "class": dest_name,
                "color": "blue",
                "position_3d": {"x": 0.45, "y": -0.25, "z": 0.78}
            }

        # Structure perception output for path planning
        state.path_planning_requirements = {
            "target_object_id": target_match.get("object_id"),
            "target_object_class": target_match.get("class"),
            "target_object_color": target_match.get("color"),
            "grasp_pose": target_match.get("position_3d"),
            "destination_id": dest_match.get("object_id"),
            "destination_class": dest_match.get("class"),
            "place_pose": dest_match.get("position_3d"),
            "approach_offset_z": 0.10,  # 10 cm above object for pre-grasp
            "obstacles": vision.get("obstacles", [])
        }

        logger.info(f"[{self.agent_id}] Spatial coordinates resolved: Grasp at {target_match.get('position_3d')}, Place at {dest_match.get('position_3d')}")
        return state
