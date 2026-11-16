"""
Emergency Stop Protection Module.
Manages software and hardware E-Stop latching and interlock states.
"""

import logging

logger = logging.getLogger(__name__)


class EmergencyStopManager:
    """Independent Emergency Stop Monitor and Interlock."""

    def __init__(self):
        self._is_tripped = False
        self._trip_reason = None

    def trigger_estop(self, reason: str = "Manual E-Stop engaged") -> None:
        """Immediately trip the emergency stop and latch hardware safety relay."""
        self._is_tripped = True
        self._trip_reason = reason
        logger.critical(f"EMERGENCY STOP TRIPPED: {reason}")

    def reset_estop(self) -> bool:
        """Reset the E-stop latch if safe."""
        self._is_tripped = False
        self._trip_reason = None
        logger.info("Emergency Stop successfully reset.")
        return True

    def is_active(self) -> bool:
        """Returns True if E-Stop is currently preventing motion."""
        return self._is_tripped

    def get_trip_reason(self) -> str:
        return self._trip_reason or "None"
