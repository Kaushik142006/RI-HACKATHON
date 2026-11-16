"""
Conditional Routing Functions for LangGraph Multi-Agent Workflow.
Evaluates agent state and dynamically routes to downstream nodes or error recovery.
"""

import logging
from orchestration.state import AgentState

logger = logging.getLogger(__name__)


def route_after_environment_check(state: AgentState) -> str:
    """Route after Agent 2 (Environment Check)."""
    env_status = (state.environment_status or "").lower()
    err_info = state.error_information or {}

    if env_status == "unsuitable" or err_info.get("error_type") == "EnvironmentUnsuitable":
        logger.warning("[Router] Routing to Error Handling due to unsuitable environment.")
        return "agent_16_error_handling"

    return "agent_3_vision_context"


def route_after_nlp(state: AgentState) -> str:
    """Route after Agent 4 (NLP)."""
    err_info = state.error_information or {}
    if err_info.get("failed_agent") == "agent_4_nlp":
        logger.warning("[Router] Routing to Error Handling due to NLP parsing failure.")
        return "agent_16_error_handling"

    return "agent_5_work_splitter"


def route_after_decision(state: AgentState) -> str:
    """Route after Agent 8 (Decision)."""
    dec_out = state.decision_output or {}
    is_feasible = dec_out.get("is_feasible", False)

    if not is_feasible:
        logger.warning("[Router] Routing to Error Handling: task deemed infeasible or ambiguous.")
        return "agent_16_error_handling"

    return "agent_9_path_planning"


def route_after_safety_gate(state: AgentState) -> str:
    """Route after Independent Safety Gate (between Agent 12 and Agent 13)."""
    if not state.safety_approval:
        logger.critical("[Router] Trajectory REJECTED by Independent Safety Gate! Routing to Error Handling.")
        return "agent_16_error_handling"

    return "agent_13_hardware_control"


def route_after_verification(state: AgentState) -> str:
    """Route after Agent 14 (Verification)."""
    v_results = state.verification_results or {}
    grasp_ok = v_results.get("grasp_verified", False)
    place_ok = v_results.get("placement_verified", False)

    if not (grasp_ok and place_ok):
        logger.warning("[Router] Verification failed: Routing to Error Handling.")
        return "agent_16_error_handling"

    return "agent_15_evaluation"


def route_after_error_handling(state: AgentState) -> str:
    """Route after Agent 16 (Error Handling): decide whether to replan or abort."""
    err_info = state.error_information or {}
    recovery = err_info.get("recovery_decision", {})
    can_retry = recovery.get("can_retry", False)
    action = recovery.get("action", "ABORT")

    if can_retry:
        if action == "REPLAN_MOTION":
            logger.info("[Router] Looping back to Agent 9 (AI Path Planning) for motion replanning.")
            return "agent_9_path_planning"
        else:
            logger.info("[Router] Looping back to Agent 1 (Orchestrator) for full rescan & replan.")
            return "agent_1_orchestrator"

    logger.info("[Router] Terminating execution: user intervention required.")
    return "end"
