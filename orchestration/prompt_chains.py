"""
Prompt Chaining Module.
Formats structured representations from upstream stages to compose context-rich prompts for downstream reasoning.
"""

from typing import Dict, Any


class PromptChainManager:
    """Manages sequential structured prompt synthesis between agents."""

    @staticmethod
    def format_planning_prompt(scene_desc: str, command: str, requested_obj: str, destination: str) -> str:
        """Chain scene description and user command into planning instructions."""
        return (
            f"=== ROBOTIC TASK PLANNING CONTEXT ===\n"
            f"Visual Scene Observations:\n{scene_desc}\n\n"
            f"User Voice Command: \"{command}\"\n"
            f"Target Object: {requested_obj}\n"
            f"Target Destination: {destination}\n\n"
            f"Goal: Formulate collision-free pick-and-place primitive sequence.\n"
        )

    @staticmethod
    def format_verification_prompt(command: str, initial_pose: Dict[str, float], final_pose: Dict[str, float]) -> str:
        """Chain task execution telemetry into verification summary."""
        return (
            f"Task: {command}\n"
            f"Initial Object Coordinate: {initial_pose}\n"
            f"Final Object Coordinate: {final_pose}\n"
            f"Requirement: Verify spatial displacement conforms to requested target placement."
        )
