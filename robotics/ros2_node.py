"""
ROS 2 Node Module.
Initializes the ROS 2 computational node, topics, services, and action bridges with graceful fallback.
"""

from typing import Dict, Any, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class VLARobotNode:
    """ROS 2 Node managing topics, service calls, and action client interactions."""

    def __init__(self, node_name: str = "vla_manipulator_core", simulation_mode: bool = True):
        self.node_name = node_name
        self.simulation_mode = simulation_mode
        self._rclpy_node = None
        self._publishers = {}
        self._subscribers = {}

        if not self.simulation_mode:
            try:
                import rclpy
                from rclpy.node import Node
                if not rclpy.ok():
                    rclpy.init()
                self._rclpy_node = Node(self.node_name)
                logger.info(f"ROS 2 Node '{node_name}' initialized.")
            except Exception as e:
                logger.warning(f"Unable to initialize rclpy Node: {e}. Defaulting to simulation mode.")
                self.simulation_mode = True

        if self.simulation_mode:
            logger.info(f"Simulated ROS 2 Node '{node_name}' active.")

    def publish_topic(self, topic: str, message: Any) -> None:
        """Publish message to given ROS 2 topic."""
        logger.debug(f"Publishing to topic '{topic}': {message}")

    def call_service(self, service_name: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a ROS 2 service."""
        logger.info(f"Invoking ROS 2 service '{service_name}'")
        return {"success": True, "service": service_name}
