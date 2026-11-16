"""
Live Task Monitor Module.
Maintains live execution telemetry, agent progress, and status streaming for the web interface.
"""

from typing import Dict, Any, List
import time


class TaskMonitor:
    """Stores active execution metrics, step progression, and pipeline logs."""

    def __init__(self):
        self._current_task: Dict[str, Any] = {
            "status": "idle",
            "active_agent": "None",
            "progress_percent": 0,
            "command": "Waiting for command...",
            "last_log": "System initialized in Simulation Mode.",
            "pipeline_stages": [
                {"name": "Speech / NLP", "status": "pending"},
                {"name": "Vision Perception", "status": "pending"},
                {"name": "Environment Audit", "status": "pending"},
                {"name": "Task & Path Planning", "status": "pending"},
                {"name": "Independent Safety Gate", "status": "pending"},
                {"name": "Robot Hardware Execution", "status": "pending"},
                {"name": "Verification & Evaluation", "status": "pending"}
            ]
        }
        self._logs: List[str] = ["System ready."]

    def update_stage(self, stage_name: str, status: str, log_message: str = None) -> None:
        """Update status of a pipeline stage ('pending', 'running', 'completed', 'failed')."""
        for s in self._current_task["pipeline_stages"]:
            if s["name"] == stage_name:
                s["status"] = status
                break
        if log_message:
            self._logs.append(f"[{time.strftime('%H:%M:%S')}] {log_message}")
            self._current_task["last_log"] = log_message

    def get_status(self) -> Dict[str, Any]:
        """Return snapshot of monitor telemetry."""
        return {
            "task": self._current_task,
            "logs": self._logs[-20:]
        }

    def reset(self) -> None:
        """Reset task monitor to idle."""
        self.__init__()


task_monitor = TaskMonitor()
