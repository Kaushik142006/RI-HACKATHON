"""
SQLAlchemy Models for PostgreSQL Database.
Defines schemas for conversations, messages, agent execution logs, task history,
execution records, and error traces with cross-dialect JSON/JSONB support.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy import Column, String, Integer, Float, Boolean, Text, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

# Uses JSONB on PostgreSQL and JSON on SQLite in-memory fallback
JsonType = JSON().with_variant(JSONB, "postgresql")


class ConversationModel(Base):
    __tablename__ = "conversations"

    conversation_id = Column(String(64), primary_key=True)
    title = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    extra_metadata = Column("metadata", JsonType, default=dict)

    messages = relationship("MessageModel", back_populates="conversation", cascade="all, delete-orphan")
    agent_logs = relationship("AgentLogModel", back_populates="conversation", cascade="all, delete-orphan")
    tasks = relationship("TaskHistoryModel", back_populates="conversation", cascade="all, delete-orphan")


class MessageModel(Base):
    __tablename__ = "messages"

    message_id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String(64), ForeignKey("conversations.conversation_id", ondelete="CASCADE"), nullable=False)
    sender = Column(String(32), nullable=False)
    content = Column(Text, nullable=False)
    audio_path = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    extra_metadata = Column("metadata", JsonType, default=dict)

    conversation = relationship("ConversationModel", back_populates="messages")


class AgentLogModel(Base):
    __tablename__ = "agent_logs"

    log_id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String(64), ForeignKey("conversations.conversation_id", ondelete="CASCADE"), nullable=False)
    agent_id = Column(String(64), nullable=False)
    step_number = Column(Integer, default=1)
    status = Column(String(32), nullable=False)
    input_state = Column(JsonType, default=dict)
    output_state = Column(JsonType, default=dict)
    execution_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("ConversationModel", back_populates="agent_logs")


class TaskHistoryModel(Base):
    __tablename__ = "task_history"

    task_id = Column(String(64), primary_key=True)
    conversation_id = Column(String(64), ForeignKey("conversations.conversation_id", ondelete="CASCADE"), nullable=False)
    raw_command = Column(Text, nullable=False)
    action = Column(String(64), nullable=False)
    target_object = Column(String(64), nullable=True)
    target_color = Column(String(32), nullable=True)
    destination = Column(String(64), nullable=True)
    task_status = Column(String(32), nullable=False)
    decomposition = Column(JsonType, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    conversation = relationship("ConversationModel", back_populates="tasks")
    executions = relationship("ExecutionHistoryModel", back_populates="task", cascade="all, delete-orphan")


class ExecutionHistoryModel(Base):
    __tablename__ = "execution_history"

    execution_id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(64), ForeignKey("task_history.task_id", ondelete="CASCADE"), nullable=False)
    trajectory_points_count = Column(Integer, default=0)
    safety_approved = Column(Boolean, default=False)
    safety_telemetry = Column(JsonType, default=dict)
    hardware_status = Column(String(32), nullable=False)
    verification_status = Column(String(32), nullable=False)
    accuracy_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    task = relationship("TaskHistoryModel", back_populates="executions")


class ErrorLogModel(Base):
    __tablename__ = "error_logs"

    error_id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String(64), ForeignKey("conversations.conversation_id", ondelete="CASCADE"), nullable=False)
    agent_id = Column(String(64), nullable=True)
    error_type = Column(String(64), nullable=False)
    error_message = Column(Text, nullable=False)
    recovery_action = Column(String(64), nullable=True)
    resolved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
