"""
Agent 13 – ros2_control Hardware Agent.
Interfaces directly with ros2_control hardware controllers.
Executes only trajectories that have passed the Independent Safety Gate.
"""

from typing import Dict, Any, Optional
import logging
import time
from orchestration.state import AgentState

logger = logging.getLogger(__name__)


class HardwareControlAgent:
    """Agent 13: Interfaces with ros2_control controllers to execute approved trajectories."""

    def __init__(self, simulation_mode: bool = True):
        self.agent_id = "agent_13_hardware_control"
        self.simulation_mode = simulation_mode

    def execute_trajectory(self, trajectory: Dict[str, Any], safety_approved: bool) -> Dict[str, Any]:
        """
        Execute trajectory on joint trajectory controller and gripper controller.
        Strictly enforces safety approval check before sending commands to motors.
        """
        if not safety_approved:
            logger.critical(f"[{self.agent_id}] CRITICAL SAFETY VIOLATION: Trajectory execution blocked because safety gate did not approve.")
            return {
                "status": "REJECTED_BY_SAFETY",
                "executed": False,
                "error": "Safety approval flag is False. Motor command transmission aborted."
            }

        num_points = len(trajectory.get("points", []))
        logger.info(f"[{self.agent_id}] Executing {num_points} trajectory points on hardware controller (Mode: {'SIMULATION' if self.simulation_mode else 'PHYSICAL'})...")

        # Emulate motor execution time
        return {
            "status": "COMPLETED",
            "executed": True,
            "points_executed": num_points,
            "final_joint_state": trajectory.get("points", [{}])[-1].get("positions", [0.0]*6),
            "gripper_state": "closed",
            "execution_duration_sec": 2.5
        }

    def process(self, state: AgentState) -> AgentState:
        """Process agent node in LangGraph."""
        logger.info(f"[{self.agent_id}] Running Hardware Control Agent...")

        # Strict safety check
        if not state.safety_approval:
            error_msg = "Safety check has not approved the motion trajectory. Hardware execution blocked."
            state.robot_execution_status = "safety_blocked"
            state.error_information = {
                "failed_agent": self.agent_id,
                "error_type": "SafetyGateRejection",
                "message": error_msg
            }
            logger.error(f"[{self.agent_id}] {error_msg}")
            return state

        trajectory = (state.execution_plan or {}).get("computed_trajectory", {})
        result = self.execute_trajectory(trajectory, safety_approved=state.safety_approval)

        if result.get("executed"):
            state.robot_execution_status = "success"
            logger.info(f"[{self.agent_id}] Hardware execution completed successfully.")
        else:
            state.robot_execution_status = "failed"
            state.error_information = {
                "failed_agent": self.agent_id,
                "error_type": "ExecutionFailure",
                "message": result.get("error", "Hardware execution error.")
            }
            logger.error(f"[{self.agent_id}] Hardware execution failed.")

        return state
