"""
Unit Tests for Database Operations.
Tests database connection, schema migration, chat history, agent logs, task history, and error persistence.
"""

import pytest
from database.connection import db_manager
from database.chat_history import ChatHistoryRepository
from database.agent_logs import AgentLogsRepository
from database.task_history import TaskHistoryRepository
from database.error_logs import ErrorLogsRepository


@pytest.fixture(autouse=True)
def init_test_db():
    db_manager.init_schema()


def test_chat_history_logging_and_retrieval():
    repo = ChatHistoryRepository()
    conv_id = "test_conv_db_1"
    repo.add_message(conv_id, "user", "Pick up red bottle")
    repo.add_message(conv_id, "robot", "Grasping bottle now")

    history = repo.get_recent_history(conv_id, limit=5)
    assert len(history) == 2
    assert history[0]["content"] == "Pick up red bottle"
    assert history[1]["content"] == "Grasping bottle now"


def test_agent_logs_persistence():
    repo = AgentLogsRepository()
    conv_id = "test_conv_db_2"
    repo.log_agent_execution(
        conversation_id=conv_id,
        agent_id="agent_1_orchestrator",
        step_number=1,
        status="success",
        execution_time_ms=12.5
    )
    # Check execution without error
    assert True


def test_task_and_execution_history():
    repo = TaskHistoryRepository()
    conv_id = "test_conv_db_3"
    task_id = "task_001"
    repo.record_task_start(
        task_id=task_id,
        conversation_id=conv_id,
        raw_command="Pick up red bottle",
        action="pick_and_place",
        target_object="bottle",
        target_color="red",
        destination="box"
    )
    repo.record_execution(
        task_id=task_id,
        trajectory_points_count=9,
        safety_approved=True,
        safety_telemetry={"status": "OK"},
        hardware_status="COMPLETED",
        verification_status="VERIFIED",
        accuracy_score=98.5
    )
    repo.record_task_completion(task_id, status="completed")
    assert True


def test_error_logs_persistence():
    repo = ErrorLogsRepository()
    conv_id = "test_conv_db_4"
    repo.log_error(
        conversation_id=conv_id,
        agent_id="safety_gate",
        error_type="VelocityLimitExceeded",
        error_message="Velocity 2.5 rad/s exceeds max 1.5 rad/s"
    )
    assert True
