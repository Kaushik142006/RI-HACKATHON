"""
Agent 2 – Initial Environment Check Agent.
Evaluates workspace suitability, distinguishes clean, manageable, and unsuitable environments,
approves execution or rejects unsafe clutter, and generates user cleanup notifications.
"""

from typing import Dict, Any, Tuple
import logging
from orchestration.state import AgentState

logger = logging.getLogger(__name__)


class EnvironmentCheckAgent:
    """Agent 2: Evaluates workspace clutter and physical safety conditions."""

    def __init__(self, max_clutter_threshold: int = 8):
        self.agent_id = "agent_2_environment_check"
        self.max_clutter_threshold = max_clutter_threshold

    def evaluate_environment(self, vision_state: Dict[str, Any]) -> Tuple[str, bool, str]:
        """
        Evaluate workspace safety.
        Returns:
            Tuple of (status: str, is_approved: bool, message: str)
        """
        if not vision_state:
            return "manageable", True, "Default simulated environment approved."

        status = vision_state.get("environment_status", "manageable")
        obstacles = vision_state.get("obstacles", [])
        objects = vision_state.get("objects", [])

        # Check for critical obstacles
        for obs in obstacles:
            if obs.get("threat_level") == "critical":
                return "unsuitable", False, f"Critical hazard detected at position {obs.get('position_3d')}."

        # Check for extreme clutter
        if len(objects) > self.max_clutter_threshold:
            return "unsuitable", False, f"Excessive clutter: {len(objects)} items detected in workspace."

        if status == "unsuitable":
            return "unsuitable", False, "Vision system reported workspace as unsuitable."

        if len(objects) == 0:
            return "clean", True, "Workspace is clean and ready."

        return "manageable", True, f"Environment manageable with {len(objects)} objects and {len(obstacles)} obstacles."

    def process(self, state: AgentState) -> AgentState:
        """Process agent node in LangGraph."""
        logger.info(f"[{self.agent_id}] Checking environment suitability...")
        status, is_approved, message = self.evaluate_environment(state.vision_state or {})

        state.environment_status = status
        if not is_approved:
            state.error_information = {
                "failed_agent": self.agent_id,
                "error_type": "EnvironmentUnsuitable",
                "message": message,
                "action_required": "Please clear workspace clutter or remove obstacles, then re-scan."
            }
            logger.warning(f"[{self.agent_id}] REJECTED workspace: {message}")
        else:
            logger.info(f"[{self.agent_id}] APPROVED workspace: {message}")

        return state
