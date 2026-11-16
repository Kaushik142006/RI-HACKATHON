"""
Agent 1 – Main / Orchestrator Agent.
Central controller: coordinates agents, retrieves conversation context,
validates object and destination availability, and synthesizes NLP with vision.
"""

from typing import Dict, Any, Optional
import logging
from orchestration.state import AgentState

logger = logging.getLogger(__name__)


class MainOrchestratorAgent:
    """Agent 1: Central controller coordinating AI workflow and state initialization."""

    def __init__(self, db_client=None):
        self.agent_id = "agent_1_orchestrator"
        self.db_client = db_client

    def process(self, state: AgentState) -> AgentState:
        """
        Execute Orchestrator logic:
        1. Retrieve previous conversation context if available.
        2. Validate inputs.
        3. Match requested object & destination against vision state if present.
        4. Advance step count and prepare workflow state.
        """
        logger.info(f"[{self.agent_id}] Running Main Orchestrator...")
        state.step_count += 1

        # 1. Fetch conversation history reference from database if available
        if self.db_client and state.conversation_id:
            try:
                history = self.db_client.get_recent_history(state.conversation_id, limit=3)
                if history:
                    state.conversation_history_reference = f"Retrieved {len(history)} recent turns."
            except Exception as e:
                logger.warning(f"[{self.agent_id}] Could not retrieve history: {e}")

        # 2. Extract requested objects if NLP has already run
        if state.nlp_output:
            state.requested_object = state.nlp_output.get("target_object")
            state.requested_destination = state.nlp_output.get("destination")

        # 3. If vision state exists, validate availability
        if state.vision_state and state.requested_object:
            objects = state.vision_state.get("objects", [])
            req_obj = state.requested_object.lower()
            req_color = (state.nlp_output or {}).get("target_color", "").lower()

            found = False
            for obj in objects:
                cls_match = req_obj in obj.get("class", "").lower()
                color_match = not req_color or req_color in obj.get("color", "").lower()
                if cls_match and color_match:
                    found = True
                    break

            state.object_availability = found
            if not found:
                logger.warning(f"[{self.agent_id}] Target object '{req_color} {req_obj}' not found in vision state.")

        logger.info(f"[{self.agent_id}] Orchestrator processed state: object_avail={state.object_availability}")
        return state
