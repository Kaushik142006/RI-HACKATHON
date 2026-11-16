"""
Task History Database Module.
Records high-level tasks, decompositions, execution records, and verification outcomes.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from database.connection import db_manager
from database.models import TaskHistoryModel, ExecutionHistoryModel
from database.chat_history import ChatHistoryRepository

logger = logging.getLogger(__name__)


class TaskHistoryRepository:
    """Manages lifecycle records of robotic manipulation tasks and trajectory executions."""

    def __init__(self, db=db_manager):
        self.db = db
        self.chat_repo = ChatHistoryRepository(db)

    def record_task_start(
        self,
        task_id: str,
        conversation_id: str,
        raw_command: str,
        action: str,
        target_object: str = None,
        target_color: str = None,
        destination: str = None,
        decomposition: List[Dict[str, Any]] = None
    ) -> None:
        """Record newly initiated robotic task."""
        self.chat_repo.ensure_conversation(conversation_id)
        session = self.db.get_session()
        try:
            task = TaskHistoryModel(
                task_id=task_id,
                conversation_id=conversation_id,
                raw_command=raw_command,
                action=action,
                target_object=target_object,
                target_color=target_color,
                destination=destination,
                task_status="executing",
                decomposition=decomposition or []
            )
            session.add(task)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error recording task start: {e}")
        finally:
            session.close()

    def record_task_completion(self, task_id: str, status: str = "completed") -> None:
        """Mark task completed or failed."""
        session = self.db.get_session()
        try:
            task = session.query(TaskHistoryModel).filter_by(task_id=task_id).first()
            if task:
                task.task_status = status
                task.completed_at = datetime.utcnow()
                session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error marking task completion: {e}")
        finally:
            session.close()

    def record_execution(
        self,
        task_id: str,
        trajectory_points_count: int,
        safety_approved: bool,
        safety_telemetry: Dict[str, Any],
        hardware_status: str,
        verification_status: str,
        accuracy_score: float
    ) -> None:
        """Record low-level trajectory execution and verification telemetry."""
        session = self.db.get_session()
        try:
            exec_rec = ExecutionHistoryModel(
                task_id=task_id,
                trajectory_points_count=trajectory_points_count,
                safety_approved=safety_approved,
                safety_telemetry=safety_telemetry or {},
                hardware_status=hardware_status,
                verification_status=verification_status,
                accuracy_score=accuracy_score
            )
            session.add(exec_rec)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error recording execution: {e}")
        finally:
            session.close()
