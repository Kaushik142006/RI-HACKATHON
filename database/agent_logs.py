"""
Agent Logs Database Module.
Persists execution logs, step counts, execution times, and state transitions for each agent.
"""

from typing import Dict, Any, Optional
import logging
from database.connection import db_manager
from database.models import AgentLogModel
from database.chat_history import ChatHistoryRepository

logger = logging.getLogger(__name__)


class AgentLogsRepository:
    """Logs individual agent execution cycles into PostgreSQL."""

    def __init__(self, db=db_manager):
        self.db = db
        self.chat_repo = ChatHistoryRepository(db)

    def log_agent_execution(
        self,
        conversation_id: str,
        agent_id: str,
        step_number: int,
        status: str,
        input_state: Dict[str, Any] = None,
        output_state: Dict[str, Any] = None,
        execution_time_ms: float = 0.0
    ) -> None:
        """Record an agent's execution node lifecycle."""
        self.chat_repo.ensure_conversation(conversation_id)
        session = self.db.get_session()
        try:
            log_entry = AgentLogModel(
                conversation_id=conversation_id,
                agent_id=agent_id,
                step_number=step_number,
                status=status,
                input_state=input_state or {},
                output_state=output_state or {},
                execution_time_ms=execution_time_ms
            )
            session.add(log_entry)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving agent log: {e}")
        finally:
            session.close()
