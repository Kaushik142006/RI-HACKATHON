"""
Database configuration and connection pooling specifications.
"""

from dataclasses import dataclass
from config.settings import settings


@dataclass
class DatabaseConfig:
    host: str = settings.DB_HOST
    port: int = settings.DB_PORT
    database: str = settings.DB_NAME
    user: str = settings.DB_USER
    password: str = settings.DB_PASSWORD
    min_pool_size: int = settings.DB_POOL_MIN
    max_pool_size: int = settings.DB_POOL_MAX
    connect_timeout: float = settings.DB_TIMEOUT

    @property
    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def async_dsn(self) -> str:
        return f"postgresql+psycopg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


db_config = DatabaseConfig()
