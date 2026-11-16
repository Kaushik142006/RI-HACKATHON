"""
Script: Initialize PostgreSQL Database.
Reads schema.sql and creates tables for conversations, messages, logs, tasks, and errors.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.connection import db_manager
from database.models import Base
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("init_db")


def main():
    logger.info("Initializing VLA Robot PostgreSQL Database schema...")
    try:
        db_manager.init_schema()
        logger.info("Database initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
