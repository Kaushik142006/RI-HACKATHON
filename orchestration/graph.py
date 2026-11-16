"""
LangGraph Multi-Agent Workflow StateGraph.
Integrates all 16 logical agents, the Independent Safety Gate, conditional routing,
and PostgreSQL logging into a unified execution pipeline.
"""

from typing import Dict, Any, Callable
import logging
import time
from orchestration.state import AgentState
from orchestration.routing import (
    route_after_environment_check,
    route_after_nlp,
    route_after_decision,
    route_after_safety_gate,
    route_after_verification,
    route_after_error_handling
)

# Import all 16 agents
from agents.main_orchestrator import MainOrchestratorAgent
from agents.environment_check_agent import EnvironmentCheckAgent
from agents.vision_context_agent import VisionContextPromptAgent
from agents.nlp_agent import NLPAgent
from agents.work_splitter_agent import WorkSplitterAgent
from agents.perception_agent import PerceptionAgent
from agents.task_planning_agent import TaskPlanningAgent
from agents.decision_agent import DecisionAgent
from agents.path_planning_agent import AIPathPlanningAgent
from agents.synthesizer_agent import SynthesizerAgent
from agents.ros2_communication_agent import ROS2CommunicationAgent
from agents.moveit_motion_agent import MoveItMotionAgent
from agents.hardware_control_agent import HardwareControlAgent
from agents.verification_agent import VerificationAgent
from agents.evaluation_agent import EvaluationAgent
from agents.error_handling_agent import ErrorHandlingAgent

# Independent Safety Mechanism
from robotics.safety.safety_gate import IndependentSafetyGate

# Database Logging
from database.chat_history import ChatHistoryRepository
from database.agent_logs import AgentLogsRepository
from database.task_history import TaskHistoryRepository
from database.error_logs import ErrorLogsRepository

logger = logging.getLogger(__name__)


class VLAMultiAgentGraph:
    """Orchestrates all 16 agents using LangGraph StateGraph and Independent Safety Gate."""

    def __init__(self, simulation_mode: bool = True):
        self.simulation_mode = simulation_mode

        # Initialize all 16 agents
        self.agent_1 = MainOrchestratorAgent()
        self.agent_2 = EnvironmentCheckAgent()
        self.agent_3 = VisionContextPromptAgent()
        self.agent_4 = NLPAgent()
        self.agent_5 = WorkSplitterAgent()
        self.agent_6 = PerceptionAgent()
        self.agent_7 = TaskPlanningAgent()
        self.agent_8 = DecisionAgent()
        self.agent_9 = AIPathPlanningAgent()
        self.agent_10 = SynthesizerAgent()
        self.agent_11 = ROS2CommunicationAgent(simulation_mode=simulation_mode)
        self.agent_12 = MoveItMotionAgent(simulation_mode=simulation_mode)
        self.agent_13 = HardwareControlAgent(simulation_mode=simulation_mode)
        self.agent_14 = VerificationAgent()
        self.agent_15 = EvaluationAgent()
        self.agent_16 = ErrorHandlingAgent()

        # Independent Safety Gate (Non-AI Layer)
        self.safety_gate = IndependentSafetyGate(simulation_mode=simulation_mode)

        # Database Repositories
        self.chat_repo = ChatHistoryRepository()
        self.agent_log_repo = AgentLogsRepository()
        self.task_repo = TaskHistoryRepository()
        self.error_repo = ErrorLogsRepository()

        # Build Graph
        self.compiled_graph = self._build_langgraph()

    def _execute_agent_with_logging(self, agent_instance, agent_name: str, state: AgentState) -> AgentState:
        """Helper to run an agent and log execution metrics to PostgreSQL."""
        t0 = time.time()
        status = "started"
        try:
            state = agent_instance.process(state)
            status = "success"
        except Exception as e:
            status = "failed"
            logger.error(f"Agent {agent_name} raised exception: {e}")
            state.error_information = {
                "failed_agent": agent_name,
                "error_type": type(e).__name__,
                "message": str(e)
            }
        finally:
            elapsed_ms = (time.time() - t0) * 1000.0
            self.agent_log_repo.log_agent_execution(
                conversation_id=state.conversation_id,
                agent_id=agent_name,
                step_number=state.step_count,
                status=status,
                input_state={"cmd": state.original_voice_command},
                output_state={"error": state.error_information},
                execution_time_ms=elapsed_ms
            )
        return state

    # Node functions
    def node_agent_1(self, state: AgentState) -> AgentState:
        return self._execute_agent_with_logging(self.agent_1, "agent_1_orchestrator", state)

    def node_agent_2(self, state: AgentState) -> AgentState:
        return self._execute_agent_with_logging(self.agent_2, "agent_2_environment_check", state)

    def node_agent_3(self, state: AgentState) -> AgentState:
        return self._execute_agent_with_logging(self.agent_3, "agent_3_vision_context", state)

    def node_agent_4(self, state: AgentState) -> AgentState:
        return self._execute_agent_with_logging(self.agent_4, "agent_4_nlp", state)

    def node_agent_5(self, state: AgentState) -> AgentState:
        return self._execute_agent_with_logging(self.agent_5, "agent_5_work_splitter", state)

    def node_agent_6(self, state: AgentState) -> AgentState:
        return self._execute_agent_with_logging(self.agent_6, "agent_6_perception", state)

    def node_agent_7(self, state: AgentState) -> AgentState:
        return self._execute_agent_with_logging(self.agent_7, "agent_7_task_planning", state)

    def node_agent_8(self, state: AgentState) -> AgentState:
        return self._execute_agent_with_logging(self.agent_8, "agent_8_decision", state)

    def node_agent_9(self, state: AgentState) -> AgentState:
        return self._execute_agent_with_logging(self.agent_9, "agent_9_path_planning", state)

    def node_agent_10(self, state: AgentState) -> AgentState:
        return self._execute_agent_with_logging(self.agent_10, "agent_10_synthesizer", state)

    def node_agent_11(self, state: AgentState) -> AgentState:
        return self._execute_agent_with_logging(self.agent_11, "agent_11_ros2_comm", state)

    def node_agent_12(self, state: AgentState) -> AgentState:
        return self._execute_agent_with_logging(self.agent_12, "agent_12_moveit_motion", state)

    def node_safety_gate(self, state: AgentState) -> AgentState:
        """Independent non-AI safety check sitting strictly between MoveIt 2 and Hardware Agent."""
        logger.info("[Independent Safety Gate] Intercepting trajectory for rigorous safety audit...")
        traj = (state.execution_plan or {}).get("computed_trajectory", {})
        waypoints = (state.path_planning_requirements or {}).get("waypoints", [])
        obstacles = (state.vision_state or {}).get("obstacles", [])

        is_approved, reason, telemetry = self.safety_gate.audit_motion_plan(traj, waypoints, obstacles)
        state.safety_approval = is_approved
        state.safety_details = telemetry

        if not is_approved:
            state.error_information = {
                "failed_agent": "independent_safety_gate",
                "error_type": "SafetyGateRejection",
                "message": reason
            }
            self.error_repo.log_error(
                conversation_id=state.conversation_id,
                agent_id="safety_gate",
                error_type="SafetyGateRejection",
                error_message=reason
            )
        return state

    def node_agent_13(self, state: AgentState) -> AgentState:
        return self._execute_agent_with_logging(self.agent_13, "agent_13_hardware_control", state)

    def node_agent_14(self, state: AgentState) -> AgentState:
        return self._execute_agent_with_logging(self.agent_14, "agent_14_verification", state)

    def node_agent_15(self, state: AgentState) -> AgentState:
        return self._execute_agent_with_logging(self.agent_15, "agent_15_evaluation", state)

    def node_agent_16(self, state: AgentState) -> AgentState:
        return self._execute_agent_with_logging(self.agent_16, "agent_16_error_handling", state)

    def _build_langgraph(self):
        """Construct LangGraph StateGraph with conditional edges."""
        try:
            from langgraph.graph import StateGraph, END
            workflow = StateGraph(AgentState)

            # Register Nodes
            workflow.add_node("agent_1_orchestrator", self.node_agent_1)
            workflow.add_node("agent_2_environment_check", self.node_agent_2)
            workflow.add_node("agent_3_vision_context", self.node_agent_3)
            workflow.add_node("agent_4_nlp", self.node_agent_4)
            workflow.add_node("agent_5_work_splitter", self.node_agent_5)
            workflow.add_node("agent_6_perception", self.node_agent_6)
            workflow.add_node("agent_7_task_planning", self.node_agent_7)
            workflow.add_node("agent_8_decision", self.node_agent_8)
            workflow.add_node("agent_9_path_planning", self.node_agent_9)
            workflow.add_node("agent_10_synthesizer", self.node_agent_10)
            workflow.add_node("agent_11_ros2_comm", self.node_agent_11)
            workflow.add_node("agent_12_moveit_motion", self.node_agent_12)
            workflow.add_node("safety_gate", self.node_safety_gate)
            workflow.add_node("agent_13_hardware_control", self.node_agent_13)
            workflow.add_node("agent_14_verification", self.node_agent_14)
            workflow.add_node("agent_15_evaluation", self.node_agent_15)
            workflow.add_node("agent_16_error_handling", self.node_agent_16)

            # Define Edges
            workflow.set_entry_point("agent_1_orchestrator")
            workflow.add_edge("agent_1_orchestrator", "agent_2_environment_check")

            # Route after Agent 2
            workflow.add_conditional_edges(
                "agent_2_environment_check",
                route_after_environment_check,
                {
                    "agent_3_vision_context": "agent_3_vision_context",
                    "agent_16_error_handling": "agent_16_error_handling"
                }
            )

            workflow.add_edge("agent_3_vision_context", "agent_4_nlp")

            # Route after Agent 4
            workflow.add_conditional_edges(
                "agent_4_nlp",
                route_after_nlp,
                {
                    "agent_5_work_splitter": "agent_5_work_splitter",
                    "agent_16_error_handling": "agent_16_error_handling"
                }
            )

            workflow.add_edge("agent_5_work_splitter", "agent_6_perception")
            workflow.add_edge("agent_6_perception", "agent_7_task_planning")
            workflow.add_edge("agent_7_task_planning", "agent_8_decision")

            # Route after Agent 8
            workflow.add_conditional_edges(
                "agent_8_decision",
                route_after_decision,
                {
                    "agent_9_path_planning": "agent_9_path_planning",
                    "agent_16_error_handling": "agent_16_error_handling"
                }
            )

            workflow.add_edge("agent_9_path_planning", "agent_10_synthesizer")
            workflow.add_edge("agent_10_synthesizer", "agent_11_ros2_comm")
            workflow.add_edge("agent_11_ros2_comm", "agent_12_moveit_motion")
            workflow.add_edge("agent_12_moveit_motion", "safety_gate")

            # Route after Independent Safety Gate
            workflow.add_conditional_edges(
                "safety_gate",
                route_after_safety_gate,
                {
                    "agent_13_hardware_control": "agent_13_hardware_control",
                    "agent_16_error_handling": "agent_16_error_handling"
                }
            )

            workflow.add_edge("agent_13_hardware_control", "agent_14_verification")

            # Route after Agent 14
            workflow.add_conditional_edges(
                "agent_14_verification",
                route_after_verification,
                {
                    "agent_15_evaluation": "agent_15_evaluation",
                    "agent_16_error_handling": "agent_16_error_handling"
                }
            )

            workflow.add_edge("agent_15_evaluation", END)

            # Route after Agent 16 (Recovery loop)
            workflow.add_conditional_edges(
                "agent_16_error_handling",
                route_after_error_handling,
                {
                    "agent_1_orchestrator": "agent_1_orchestrator",
                    "agent_9_path_planning": "agent_9_path_planning",
                    "end": END
                }
            )

            return workflow.compile()
        except Exception as e:
            logger.warning(f"LangGraph compile note: {e}. Using resilient StateGraph engine.")
            return None

    def execute(self, initial_state: AgentState) -> AgentState:
        """
        Execute full workflow starting from entry point.
        Guarantees deterministic execution across all 16 agents and safety gate.
        """
        logger.info("=== Starting VLA Multi-Agent Graph Execution ===")
        # Save initial command to chat history
        cmd = initial_state.original_voice_command or initial_state.transcribed_text or "No command"
        self.chat_repo.add_message(initial_state.conversation_id, "user", cmd)

        current_state = initial_state

        if self.compiled_graph is not None:
            try:
                res = self.compiled_graph.invoke(current_state)
                if isinstance(res, AgentState):
                    return res
                elif isinstance(res, dict):
                    return AgentState(**res)
            except Exception as e:
                logger.warning(f"Error in compiled LangGraph execution: {e}. Executing sequence via direct pipeline.")

        # Resilient direct execution pipeline honoring the exact same graph flow
        # 1 -> 2
        current_state = self.node_agent_1(current_state)
        current_state = self.node_agent_2(current_state)

        if route_after_environment_check(current_state) == "agent_16_error_handling":
            return self.node_agent_16(current_state)

        # 3 -> 4
        current_state = self.node_agent_3(current_state)
        current_state = self.node_agent_4(current_state)

        if route_after_nlp(current_state) == "agent_16_error_handling":
            return self.node_agent_16(current_state)

        # 5 -> 6 -> 7 -> 8
        current_state = self.node_agent_5(current_state)
        current_state = self.node_agent_6(current_state)
        current_state = self.node_agent_7(current_state)
        current_state = self.node_agent_8(current_state)

        if route_after_decision(current_state) == "agent_16_error_handling":
            return self.node_agent_16(current_state)

        # 9 -> 10 -> 11 -> 12
        current_state = self.node_agent_9(current_state)
        current_state = self.node_agent_10(current_state)
        current_state = self.node_agent_11(current_state)
        current_state = self.node_agent_12(current_state)

        # Mandatory Independent Safety Gate Check
        current_state = self.node_safety_gate(current_state)
        if route_after_safety_gate(current_state) == "agent_16_error_handling":
            return self.node_agent_16(current_state)

        # 13 -> 14
        current_state = self.node_agent_13(current_state)
        current_state = self.node_agent_14(current_state)

        if route_after_verification(current_state) == "agent_16_error_handling":
            return self.node_agent_16(current_state)

        # 15 (Evaluation) -> END
        current_state = self.node_agent_15(current_state)

        # Log completion
        resp = current_state.final_response or "Task complete."
        self.chat_repo.add_message(current_state.conversation_id, "robot", resp)

        logger.info("=== Completed VLA Multi-Agent Graph Execution ===")
        return current_state
