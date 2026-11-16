"""
Error Logs Database Module.
Persists system anomalies, agent failures, safety violations, and recovery outcomes.
"""

from typing import Optional
import logging
from database.connection import db_manager
from database.models import ErrorLogModel
from database.chat_history import ChatHistoryRepository

logger = logging.getLogger(__name__)


class ErrorLogsRepository:
    """Logs runtime anomalies and recovery strategies."""

    def __init__(self, db=db_manager):
        self.db = db
        self.chat_repo = ChatHistoryRepository(db)

    def log_error(
        self,
        conversation_id: str,
        agent_id: Optional[str],
        error_type: str,
        error_message: str,
        recovery_action: Optional[str] = None,
        resolved: bool = False
    ) -> None:
        """Save error record to PostgreSQL."""
        self.chat_repo.ensure_conversation(conversation_id)
        session = self.db.get_session()
        try:
            err = ErrorLogModel(
                conversation_id=conversation_id,
                agent_id=agent_id,
                error_type=error_type,
                error_message=error_message,
                recovery_action=recovery_action,
                resolved=resolved
            )
            session.add(err)
            session.commit()
            logger.info(f"Recorded error event: [{error_type}] from {agent_id}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error persisting error log: {e}")
        finally:
            session.close()
