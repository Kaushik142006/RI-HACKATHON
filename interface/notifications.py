"""
User Notification & Clarification Generator Module.
Formats system alerts, missing object notices, clarification requests, and cleanup prompts.
"""

from typing import Dict, Any, Optional
from datetime import datetime


class NotificationManager:
    """Creates structured user-facing messages and banners for the web UI."""

    @staticmethod
    def create_notification(level: str, title: str, message: str, action_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        level: 'info', 'success', 'warning', 'error', 'cleanup_required', 'clarification'
        """
        return {
            "level": level,
            "title": title,
            "message": message,
            "action_prompt": action_prompt,
            "timestamp": datetime.utcnow().strftime("%H:%M:%S")
        }

    @staticmethod
    def format_missing_object(target_obj: str, target_color: str) -> Dict[str, Any]:
        return NotificationManager.create_notification(
            level="warning",
            title="Object Not Found",
            message=f"Could not locate '{target_color} {target_obj}' in the workspace camera field of view.",
            action_prompt="Please place the requested item on the tabletop and request a re-scan."
        )

    @staticmethod
    def format_cleanup_prompt(clutter_count: int) -> Dict[str, Any]:
        return NotificationManager.create_notification(
            level="cleanup_required",
            title="Workspace Clutter Rejection",
            message=f"Initial environment inspection failed: {clutter_count} items detected. The arm path is blocked.",
            action_prompt="Please remove unnecessary items and confirm once the tabletop is clean."
        )

    @staticmethod
    def format_clarification_request(ambiguous_count: int, target_name: str) -> Dict[str, Any]:
        return NotificationManager.create_notification(
            level="clarification",
            title="Target Ambiguity Detected",
            message=f"Found {ambiguous_count} objects matching '{target_name}'.",
            action_prompt="Please specify which one (e.g. 'the red bottle on the left')."
        )
