"""
Central System Configuration using Pydantic Settings.
Loads environment variables, YAML config files, and runtime flags.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import os
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # General System
    APP_ENV: str = "development"
    SIMULATION_MODE: bool = True
    LOG_LEVEL: str = "INFO"

    # Robot Physical Parameters
    ROBOT_NAME: str = "vla_manipulator_6dof"
    ROBOT_BASE_FRAME: str = "base_link"
    ROBOT_EE_FRAME: str = "tool0_flange"
    GRIPPER_MAX_EFFORT: float = 100.0
    GRIPPER_OPEN_WIDTH: float = 0.08
    GRIPPER_CLOSE_WIDTH: float = 0.005

    # Workspace Limits (meters)
    WORKSPACE_X_MIN: float = -0.8
    WORKSPACE_X_MAX: float = 0.8
    WORKSPACE_Y_MIN: float = -0.8
    WORKSPACE_Y_MAX: float = 0.8
    WORKSPACE_Z_MIN: float = 0.0
    WORKSPACE_Z_MAX: float = 1.2
    MAX_JOINT_VELOCITY: float = 1.5
    MAX_ACCELERATION: float = 1.0

    # Camera & Vision
    CAMERA_TYPE: str = "realsense_d435i"
    CAMERA_WIDTH: int = 1280
    CAMERA_HEIGHT: int = 720
    CAMERA_FPS: int = 30
    ENABLE_DEPTH: bool = True
    YOLO_MODEL_PATH: str = "yolov8n.pt"
    YOLO_SEG_MODEL_PATH: str = "yolov8n-seg.pt"
    CONFIDENCE_THRESHOLD: float = 0.5
    IOU_THRESHOLD: float = 0.45

    # NLP & Speech
    WHISPER_MODEL_SIZE: str = "base"
    SPACY_MODEL: str = "en_core_web_sm"
    OPENAI_API_KEY: Optional[str] = None
    LLM_PROVIDER: str = "mock_or_openai"
    LLM_MODEL: str = "gpt-4o-mini"

    # Database (PostgreSQL)
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "vla_robot_db"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "postgres"
    DB_POOL_MIN: int = 1
    DB_POOL_MAX: int = 10
    DB_TIMEOUT: float = 5.0

    # ROS 2 & Middleware
    ROS_DOMAIN_ID: int = 0
    ROS_NAMESPACE: str = "/vla_robot"
    ENABLE_ROS2: bool = False

    # Web Dashboard
    WEB_HOST: str = "0.0.0.0"
    WEB_PORT: int = 8000

    def get_postgres_dsn(self) -> str:
        """Construct PostgreSQL connection string."""
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @classmethod
    def load_yaml(cls, filename: str) -> Dict[str, Any]:
        """Utility to load external YAML config files."""
        filepath = CONFIG_DIR / filename
        if not filepath.exists():
            return {}
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}


settings = AppSettings()
