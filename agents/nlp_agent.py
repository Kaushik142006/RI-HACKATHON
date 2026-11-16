"""
Agent 4 – NLP Agent.
Processes transcribed human commands using NLTK preprocessing, intent classification,
and entity extraction to yield structured command schemas.
"""

from typing import Dict, Any, Optional
import logging
from orchestration.state import AgentState
from nlp.command_parser import CommandParser

logger = logging.getLogger(__name__)


class NLPAgent:
    """Agent 4: NLP Agent parsing natural language command and intent."""

    def __init__(self, classifier_type: str = "svm"):
        self.agent_id = "agent_4_nlp"
        self.parser = CommandParser(classifier_type=classifier_type)

    def process(self, state: AgentState) -> AgentState:
        """Process agent node in LangGraph."""
        logger.info(f"[{self.agent_id}] Running NLP Agent...")

        text_to_process = state.transcribed_text or state.original_voice_command
        if not text_to_process:
            logger.warning(f"[{self.agent_id}] No input command text found.")
            state.error_information = {
                "failed_agent": self.agent_id,
                "error_type": "MissingCommand",
                "message": "No voice or text command received."
            }
            return state

        parsed = self.parser.parse(text_to_process)
        state.nlp_output = parsed.model_dump()
        state.requested_object = parsed.target_object
        state.requested_destination = parsed.destination

        if not parsed.is_valid:
            state.error_information = {
                "failed_agent": self.agent_id,
                "error_type": "InvalidCommandFormat",
                "message": parsed.validation_error or "Command could not be structured into actionable goals."
            }
            logger.warning(f"[{self.agent_id}] Command parsing invalid: {parsed.validation_error}")
        else:
            logger.info(f"[{self.agent_id}] Command parsed successfully: action={parsed.action}, obj={parsed.target_object}, dest={parsed.destination}")

        return state
