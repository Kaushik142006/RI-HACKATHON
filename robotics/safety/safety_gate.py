"""
Independent Safety Gate Module.
The mandatory non-AI security interlock layer positioned strictly between
MoveIt 2 Motion Planning (Agent 12) and ros2_control Hardware (Agent 13).
"""

from typing import Dict, Any, Tuple, List
import logging
from robotics.safety.emergency_stop import EmergencyStopManager
from robotics.safety.joint_limits import JointLimitsChecker
from robotics.safety.collision_checker import CollisionChecker

logger = logging.getLogger(__name__)


class IndependentSafetyGate:
    """
    Independent Safety Gate.
    Verifies E-Stop status, trajectory feasibility, joint limits, velocity limits,
    workspace bounding box, and collision clearance before authorizing hardware motion.
    """

    def __init__(self, simulation_mode: bool = True):
        self.simulation_mode = simulation_mode
        self.estop = EmergencyStopManager()
        self.joint_checker = JointLimitsChecker()
        self.collision_checker = CollisionChecker()

    def audit_motion_plan(self, trajectory: Dict[str, Any], waypoints: List[Dict[str, Any]] = None, obstacles: List[Dict[str, Any]] = None) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Execute full safety audit:
        1. E-Stop check
        2. Trajectory joint limit check
        3. Cartesian workspace boundary check
        4. Obstacle clearance check
        Returns:
            Tuple of (is_approved: bool, reason: str, audit_telemetry: dict)
        """
        # Check 1: E-Stop
        if self.estop.is_active():
            msg = f"HARDWARE INTERLOCK: Emergency Stop is active ({self.estop.get_trip_reason()})."
            logger.critical(msg)
            return False, msg, {"estop": True}

        # Check 2: Joint & Velocity Limits
        joints_valid, joint_err = self.joint_checker.validate_trajectory(trajectory)
        if not joints_valid:
            logger.error(f"SAFETY GATE REJECTION: Joint limits violated: {joint_err}")
            return False, joint_err, {"joint_limits_passed": False}

        # Check 3: Cartesian and Workspace checks on waypoints
        if waypoints:
            for wp in waypoints:
                x, y, z = wp.get("x", 0.0), wp.get("y", 0.0), wp.get("z", 0.0)
                cart_valid, cart_err = self.collision_checker.check_cartesian_point(x, y, z, obstacles)
                if not cart_valid:
                    logger.error(f"SAFETY GATE REJECTION: Cartesian collision check failed: {cart_err}")
                    return False, cart_err, {"collision_clearance_passed": False}

        telemetry = {
            "estop_status": "NORMAL",
            "joint_limits_passed": True,
            "collision_clearance_passed": True,
            "trajectory_points_audited": len(trajectory.get("points", [])),
            "mode": "SIMULATION" if self.simulation_mode else "HARDWARE"
        }
        logger.info("INDEPENDENT SAFETY GATE: Trajectory AUDIT PASSED. Motor execution authorized.")
        return True, "Trajectory verified safe for physical execution.", telemetry
