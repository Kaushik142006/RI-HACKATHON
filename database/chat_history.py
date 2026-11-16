"""
Chat History Database Module.
Stores and retrieves conversation turns and messages with context query helpers for Agent 1.
"""

from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from database.connection import db_manager
from database.models import ConversationModel, MessageModel

logger = logging.getLogger(__name__)


class ChatHistoryRepository:
    """Handles conversation records and context retrieval."""

    def __init__(self, db=db_manager):
        self.db = db

    def ensure_conversation(self, conversation_id: str, title: str = "Robot Interaction") -> None:
        """Create conversation record if it does not exist."""
        session = self.db.get_session()
        try:
            conv = session.query(ConversationModel).filter_by(conversation_id=conversation_id).first()
            if not conv:
                conv = ConversationModel(conversation_id=conversation_id, title=title)
                session.add(conv)
                session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error ensuring conversation: {e}")
        finally:
            session.close()

    def add_message(self, conversation_id: str, sender: str, content: str, metadata: Dict[str, Any] = None) -> None:
        """Record user command or robot reply."""
        self.ensure_conversation(conversation_id)
        session = self.db.get_session()
        try:
            msg = MessageModel(
                conversation_id=conversation_id,
                sender=sender,
                content=content,
                extra_metadata=metadata or {}
            )
            session.add(msg)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error logging message: {e}")
        finally:
            session.close()

    def get_recent_history(self, conversation_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve recent conversation context for the Orchestrator."""
        session = self.db.get_session()
        try:
            msgs = (
                session.query(MessageModel)
                .filter_by(conversation_id=conversation_id)
                .order_by(MessageModel.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {"sender": m.sender, "content": m.content, "timestamp": str(m.created_at)}
                for m in reversed(msgs)
            ]
        except Exception as e:
            logger.error(f"Error retrieving history: {e}")
            return []
        finally:
            session.close()
