"""
Integration & End-to-End Test for Complete VLA Multi-Agent System.
Executes benchmark scenario across all 16 agents, Independent Safety Gate, Computer Vision, and NLP.
"""

import pytest
from orchestration.graph import VLAMultiAgentGraph
from orchestration.state import AgentState
from vision.vision_pipeline import VisionPipeline
from nlp.speech_to_text import SpeechTranscriber


def test_full_end_to_end_benchmark_execution():
    """
    Test complete sequence:
    Voice Command -> Transcription -> NLP Preprocessing -> Vision Capture ->
    Environment Inspection -> Scene Context -> Work Splitting -> Perception ->
    Task Planning -> Feasibility Decision -> MoveIt 2 Motion Planning ->
    Independent Safety Audit -> Hardware Control -> Verification -> Evaluation.
    """
    # 1. Pipeline components
    vision_pipeline = VisionPipeline(simulation_mode=True)
    vision_pipeline.start()
    graph = VLAMultiAgentGraph(simulation_mode=True)
    transcriber = SpeechTranscriber(simulation_mode=True)

    # 2. Transcribe command
    command_text = transcriber.transcribe()
    assert "red bottle" in command_text.lower()

    # 3. Capture vision state
    v_state = vision_pipeline.process_frame()
    assert len(v_state.objects) > 0

    # 4. Construct initial state
    initial_state = AgentState(
        conversation_id="test_e2e_benchmark",
        original_voice_command=command_text,
        transcribed_text=command_text,
        vision_state=v_state.model_dump()
    )

    # 5. Execute 16-agent graph
    final_state = graph.execute(initial_state)

    # 6. Verify end-to-end outcome
    assert final_state.step_count > 0
    assert final_state.requested_object == "bottle"
    assert final_state.environment_status in ["manageable", "clean"]
    assert len(final_state.task_decomposition) == 9
    assert final_state.safety_approval is True
    assert final_state.robot_execution_status == "success"
    assert final_state.verification_results["placement_verified"] is True
    assert final_state.evaluation_metrics["task_success"] is True
    assert final_state.evaluation_metrics["accuracy_score_percent"] > 0
    assert "Successfully completed task" in final_state.final_response

    vision_pipeline.stop()
