"""
Agent 3 – Vision Context Prompt Agent.
Generates rich structured scene descriptions from vision state for downstream planning agents.
"""

from typing import Dict, Any, List
import logging
from orchestration.state import AgentState

logger = logging.getLogger(__name__)


class VisionContextPromptAgent:
    """Agent 3: Transforms numerical vision detections into structured linguistic scene descriptions."""

    def __init__(self):
        self.agent_id = "agent_3_vision_context"

    def build_scene_description(self, vision_state: Dict[str, Any]) -> str:
        """Construct natural language scene summary from objects and obstacles."""
        if not vision_state or "objects" not in vision_state:
            return "No vision data currently available."

        objects: List[Dict[str, Any]] = vision_state.get("objects", [])
        obstacles: List[Dict[str, Any]] = vision_state.get("obstacles", [])
        env_status = vision_state.get("environment_status", "unknown")

        lines = [f"Workspace Status: {env_status.upper()}."]
        lines.append(f"Detected {len(objects)} object(s) on the tabletop:")

        for obj in objects:
            oid = obj.get("object_id")
            cls = obj.get("class", "object")
            col = obj.get("color", "unknown color")
            pos = obj.get("position_3d", {})
            conf = obj.get("confidence", 0.0)
            x = pos.get("x", 0.0)
            y = pos.get("y", 0.0)
            z = pos.get("z", 0.0)
            lines.append(f" - [ID {oid}] {col} {cls} (confidence: {conf:.2f}) at position (x={x:.2f}m, y={y:.2f}m, z={z:.2f}m)")

        if obstacles:
            lines.append(f"Detected {len(obstacles)} obstacle(s):")
            for obs in obstacles:
                lines.append(f" - Hazard {obs.get('obstacle_type')} at position {obs.get('position_3d')} [threat: {obs.get('threat_level')}]")
        else:
            lines.append("No collision hazards detected in arm trajectory path.")

        return "\n".join(lines)

    def process(self, state: AgentState) -> AgentState:
        """Process agent node in LangGraph."""
        logger.info(f"[{self.agent_id}] Generating structured vision context prompt...")
        scene_desc = self.build_scene_description(state.vision_state or {})
        state.scene_description = scene_desc
        logger.info(f"[{self.agent_id}] Scene description generated:\n{scene_desc}")
        return state
