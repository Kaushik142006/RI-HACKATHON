"""
Master System Runner Script: End-to-End Demonstration Scenario.
Executes the benchmark command through all 16 logical agents, the Independent Safety Gate,
the Computer Vision pipeline, NLP parser, and logs results to PostgreSQL.
"""

import sys
import json
import logging
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestration.graph import VLAMultiAgentGraph
from orchestration.state import AgentState
from vision.vision_pipeline import VisionPipeline
from nlp.speech_to_text import SpeechTranscriber
from database.connection import db_manager

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)
logger = logging.getLogger("MASTER_RUNNER")


def run_demonstration():
    print("\n" + "="*80)
    print("VISION-LANGUAGE-ACTION INTELLIGENT ROBOT SYSTEM")
    print("END-TO-END DEMONSTRATION SCENARIO")
    print("="*80 + "\n")

    # Step 1: Ensure database schema
    logger.info("Initializing database connectivity...")
    db_manager.init_schema()

    # Step 2: Initialize Vision & Multi-Agent Graph
    logger.info("Initializing Computer Vision Pipeline and 16-Agent LangGraph Orchestrator...")
    vision_pipeline = VisionPipeline(simulation_mode=True)
    vision_pipeline.start()
    graph = VLAMultiAgentGraph(simulation_mode=True)

    # Step 3: Simulated Voice Input
    transcriber = SpeechTranscriber(simulation_mode=True)
    voice_command = transcriber.transcribe()
    print(f"\n>>> [HUMAN VOICE COMMAND]: \"{voice_command}\"\n")

    # Step 4: Capture Real-time / Simulated Vision State
    logger.info("Triggering real-time workspace perception scan...")
    vision_state = vision_pipeline.process_frame()
    logger.info(f"Vision detected {len(vision_state.objects)} tabletop objects and {len(vision_state.obstacles)} obstacles.")

    # Step 5: Compose Initial State
    initial_state = AgentState(
        conversation_id="conv_demo_benchmark",
        original_voice_command=voice_command,
        transcribed_text=voice_command,
        vision_state=vision_state.model_dump()
    )

    # Step 6: Execute LangGraph Orchestrator through all 16 agents
    logger.info("Executing 16-Agent Workflow through LangGraph StateGraph...\n")
    final_state = graph.execute(initial_state)

    # Step 7: Print Complete Execution Trace
    print("\n" + "="*80)
    print("COMPLETE END-TO-END SYSTEM EXECUTION TRACE")
    print("="*80)
    print(f"1. Transcribed Command      : {final_state.transcribed_text}")
    print(f"2. Intent Classified        : {final_state.nlp_output.get('intent')} (conf: {final_state.nlp_output.get('intent_confidence')})")
    print(f"3. Extracted Target Object  : {final_state.nlp_output.get('target_color')} {final_state.requested_object}")
    print(f"4. Extracted Destination    : {final_state.requested_destination}")
    print(f"5. Environment Status       : {final_state.environment_status.upper()}")
    print(f"6. Object Available In Scene: {final_state.object_availability}")
    print(f"7. Task Decomposition Steps : {len(final_state.task_decomposition)} manipulation primitives")
    for step in final_state.task_decomposition:
        print(f"     - Step {step['step']}: {step['name']} ({step['primitive']})")
    print(f"8. Independent Safety Gate  : {'APPROVED' if final_state.safety_approval else 'REJECTED'}")
    print(f"9. Hardware Controller Status: {final_state.robot_execution_status.upper()}")
    print(f"10. Verification Outcome     : {'VERIFIED' if final_state.verification_results.get('placement_verified') else 'FAILED'}")
    print(f"11. Evaluation Metrics       : Accuracy Score = {final_state.evaluation_metrics.get('accuracy_score_percent')}%")
    print(f"12. Final System Response   : {final_state.final_response}")
    print("="*80 + "\n")

    vision_pipeline.stop()
    logger.info("End-to-End demonstration run successfully concluded.")


if __name__ == "__main__":
    run_demonstration()
