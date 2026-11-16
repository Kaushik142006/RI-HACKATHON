"""
Agent 16 – Error Handling Agent.
Handles execution anomalies, grasp slips, trajectory replanning, and initiates recovery loops.
"""

from typing import Dict, Any, Tuple
import logging
from orchestration.state import AgentState

logger = logging.getLogger(__name__)


class ErrorHandlingAgent:
    """Agent 16: Diagnoses faults and formulates recovery procedures or triggers replanning."""

    def __init__(self, max_allowed_retries: int = 3):
        self.agent_id = "agent_16_error_handling"
        self.max_allowed_retries = max_allowed_retries

    def formulate_recovery_strategy(self, state: AgentState) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Analyze error information and determine if automatic recovery or replanning is possible.
        Returns:
            Tuple of (can_retry: bool, recovery_action: str, recovery_details: dict)
        """
        err_info = state.error_information or {}
        err_type = err_info.get("error_type", "UnknownError")
        err_msg = err_info.get("message", "No specific error message provided.")

        if state.retry_count >= self.max_allowed_retries:
            return False, "ABORT", {
                "reason": f"Maximum retry limit ({self.max_allowed_retries}) exceeded.",
                "human_intervention_required": True
            }

        # Specific failure recovery logic
        if err_type == "SafetyGateRejection":
            return True, "REPLAN_MOTION", {
                "strategy": "Adjust trajectory velocity scaling and recalculate inverse kinematics avoiding obstacle boundaries.",
                "next_node": "agent_9_path_planning"
            }
        elif err_type == "VerificationFailed" or err_type == "GraspFailure":
            return True, "RESCAN_AND_RETRY_GRASP", {
                "strategy": "Trigger new camera perception scan to re-locate shifted object, then attempt modified approach angle.",
                "next_node": "agent_1_orchestrator"
            }
        elif err_type == "EnvironmentUnsuitable":
            return False, "NOTIFY_USER_CLEANUP", {
                "strategy": "Workspace requires physical user cleanup before system can safely resume.",
                "human_intervention_required": True
            }
        elif err_type == "MissingCommand" or err_type == "InvalidCommandFormat":
            return False, "PROMPT_USER_CLARIFICATION", {
                "strategy": "Prompt user for clear command phrasing.",
                "human_intervention_required": True
            }

        # Default fallback replan
        return True, "REPLAN", {
            "strategy": "General retry via main orchestrator.",
            "next_node": "agent_1_orchestrator"
        }

    def process(self, state: AgentState) -> AgentState:
        """Process agent node in LangGraph."""
        logger.info(f"[{self.agent_id}] Processing system error and determining recovery procedure...")

        can_retry, action, details = self.formulate_recovery_strategy(state)

        if can_retry:
            state.retry_count += 1
            logger.info(f"[{self.agent_id}] Recovery strategy: {action} (Attempt {state.retry_count}/{self.max_allowed_retries})")
            state.final_response = f"Recovery in progress ({action}): {details.get('strategy')}"
        else:
            logger.critical(f"[{self.agent_id}] Recovery cannot proceed automatically: {action}. Action: {details.get('strategy')}")
            state.final_response = f"Execution halted: {details.get('reason', details.get('strategy'))}"

        if not state.error_information:
            state.error_information = {}
        state.error_information["recovery_decision"] = {
            "can_retry": can_retry,
            "action": action,
            "details": details
        }

        return state
