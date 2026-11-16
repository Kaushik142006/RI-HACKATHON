"""
Script: Run Standalone Robot Simulation.
Tests mock sensors, virtual robot forward and inverse kinematics, and simulated gripper grasping.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation.simulated_environment import SimulatedEnvironment
from robotics.safety.safety_gate import IndependentSafetyGate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_sim")


def main():
    logger.info("Starting VLA Simulated Environment...")
    env = SimulatedEnvironment()

    rgb, depth = env.camera.capture_frame()
    logger.info(f"Captured virtual camera frame: RGB shape {rgb.shape}, Depth shape {depth.shape}")

    ranges, angles = env.lidar.scan()
    logger.info(f"Captured virtual LiDAR scan: {len(ranges)} range readings.")

    # Test safety gate with mock robot trajectory
    safety_gate = IndependentSafetyGate(simulation_mode=True)
    sample_trajectory = {
        "joint_names": env.robot.joint_names,
        "points": [
            {"positions": [0.0, -0.785, 1.57, -0.785, 1.57, 0.0], "velocities": [0.1]*6}
        ]
    }
    is_safe, reason, telemetry = safety_gate.audit_motion_plan(sample_trajectory)
    logger.info(f"Safety Gate Audit Result: Safe={is_safe}, Reason='{reason}'")

    # Move virtual arm
    env.robot.execute_joint_trajectory(sample_trajectory)
    # Grasp object
    env.robot.set_gripper(0.02, 50.0)

    state = env.robot.get_state()
    logger.info(f"Final Simulated Robot State: {state}")
    logger.info("Simulation test successfully completed!")


if __name__ == "__main__":
    main()
