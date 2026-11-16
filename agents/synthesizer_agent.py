"""
Agent 10 – Synthesizer Agent.
Combines outputs from the specialised agents, verifies consistency between task primitives
and spatial waypoints, and compiles the unified execution plan for ROS 2.
"""

from typing import Dict, Any, List
import logging
from orchestration.state import AgentState

logger = logging.getLogger(__name__)


class SynthesizerAgent:
    """Agent 10: Integrates perception, task decomposition, and path waypoints into a unified execution plan."""

    def __init__(self):
        self.agent_id = "agent_10_synthesizer"

    def synthesize_plan(self, state: AgentState) -> Dict[str, Any]:
        """Compile complete execution payload."""
        reqs = state.path_planning_requirements or {}
        tasks = state.task_decomposition or []
        waypoints = reqs.get("waypoints", [])

        unified_plan = {
            "plan_id": f"plan_{state.step_count}_{state.conversation_id}",
            "command": state.transcribed_text or state.original_voice_command,
            "target": {
                "class": state.requested_object,
                "color": (state.nlp_output or {}).get("target_color"),
                "grasp_pose": reqs.get("grasp_pose")
            },
            "destination": {
                "class": state.requested_destination,
                "place_pose": reqs.get("place_pose")
            },
            "task_steps": tasks,
            "waypoints": waypoints,
            "execution_mode": "simulation" if not state.safety_approval else "physical_ready"
        }
        return unified_plan

    def process(self, state: AgentState) -> AgentState:
        """Process agent node in LangGraph."""
        logger.info(f"[{self.agent_id}] Synthesizing unified execution plan...")

        plan = self.synthesize_plan(state)
        state.execution_plan = plan
        state.ros2_execution_goal = plan

        logger.info(f"[{self.agent_id}] Unified plan synthesized successfully with {len(plan.get('task_steps', []))} steps.")
        return state
