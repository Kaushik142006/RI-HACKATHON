"""
Database Connection and Session Manager.
Connects to PostgreSQL using psycopg3 and SQLAlchemy, with safe in-memory fallback.
"""

from typing import Optional, Any
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from config.settings import settings
from config.database_config import db_config
from database.models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database engine, sessions, schema initialization, and fallback cache."""

    def __init__(self):
        self.dsn = db_config.dsn
        self.engine = None
        self.SessionLocal = None
        self.is_connected = False
        self.in_memory_mode = False

        # Fallback in-memory dictionaries
        self.fallback_conversations = {}
        self.fallback_messages = []
        self.fallback_agent_logs = []
        self.fallback_tasks = {}
        self.fallback_executions = []
        self.fallback_error_logs = []

        self._initialize_connection()

    def _initialize_connection(self) -> None:
        """Attempt connection to PostgreSQL, falling back to SQLite in-memory if offline."""
        try:
            self.engine = create_engine(
                self.dsn,
                pool_size=db_config.min_pool_size,
                max_overflow=db_config.max_pool_size,
                connect_args={"connect_timeout": int(db_config.connect_timeout)}
            )
            # Test connection
            with self.engine.connect() as conn:
                self.is_connected = True
                self.in_memory_mode = False
                logger.info(f"Connected to PostgreSQL database at {db_config.host}:{db_config.port}/{db_config.database}")
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        except Exception as e:
            logger.warning(f"Could not connect to PostgreSQL ({e}). Using temporary SQLite in-memory database.")
            self.in_memory_mode = True
            self.engine = create_engine("sqlite:///:memory:", echo=False)
            Base.metadata.create_all(bind=self.engine)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            self.is_connected = True

    def init_schema(self) -> None:
        """Create all tables in database if not existing."""
        if self.engine:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database schema verified/created.")

    def get_session(self) -> Session:
        """Get database session."""
        if self.SessionLocal is None:
            self._initialize_connection()
        return self.SessionLocal()


db_manager = DatabaseManager()
