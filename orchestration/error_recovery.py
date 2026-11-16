"""
Error Recovery and Retry Policy Module.
Defines automated retry counters, error classifications, and fallback policies.
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ErrorRecoveryPolicy:
    """Manages retry allowances and safe state rollbacks during execution anomalies."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def can_retry(self, current_retry_count: int, error_type: str) -> bool:
        """Determine if an error is recoverable."""
        # Non-recoverable fatal errors
        fatal_errors = {"FatalHardwareTrip", "TablePenetration", "EStopManualEngaged"}
        if error_type in fatal_errors:
            return False

        return current_retry_count < self.max_retries
