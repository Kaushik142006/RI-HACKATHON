"""
Central Shared State Definition for LangGraph Multi-Agent Orchestration.
Contains all fields specified in the system architecture.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class AgentState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Session & Conversation Metadata
    conversation_id: str = "conv_default"
    step_count: int = 0
    conversation_history_reference: Optional[str] = None

    # Input & NLP
    original_voice_command: Optional[str] = None
    transcribed_text: Optional[str] = None
    nlp_output: Optional[Dict[str, Any]] = None
    requested_object: Optional[str] = None
    requested_destination: Optional[str] = None

    # Vision & Environment
    vision_state: Optional[Dict[str, Any]] = None
    environment_status: Optional[str] = None  # clean, manageable, unsuitable
    scene_description: Optional[str] = None
    object_availability: bool = False

    # AI Decision & Planning
    task_decomposition: List[Dict[str, Any]] = Field(default_factory=list)
    decision_output: Optional[Dict[str, Any]] = None
    path_planning_requirements: Optional[Dict[str, Any]] = None
    execution_plan: Optional[Dict[str, Any]] = None

    # Safety & ROS 2 Execution
    safety_approval: bool = False
    safety_details: Optional[Dict[str, Any]] = None
    ros2_execution_goal: Optional[Dict[str, Any]] = None
    robot_execution_status: str = "idle"  # idle, executing, success, failed

    # Verification, Evaluation & Error Recovery
    verification_results: Optional[Dict[str, Any]] = None
    evaluation_metrics: Optional[Dict[str, Any]] = None
    error_information: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    max_retries: int = 3
    final_response: Optional[str] = None
