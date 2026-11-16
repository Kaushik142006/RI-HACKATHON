"""
Python Tools and Callable Function Registry.
Modular helper tools exposed to LangGraph agents for spatial transformations,
sensor queries, and validation.
"""

from typing import Dict, Any, List
import numpy as np


def compute_euclidean_distance(pt1: Dict[str, float], pt2: Dict[str, float]) -> float:
    """Tool: Calculate 3D Euclidean distance between two points."""
    dx = pt1.get("x", 0.0) - pt2.get("x", 0.0)
    dy = pt1.get("y", 0.0) - pt2.get("y", 0.0)
    dz = pt1.get("z", 0.0) - pt2.get("z", 0.0)
    return float(np.sqrt(dx**2 + dy**2 + dz**2))


def check_gripper_affordance(object_dimensions: Dict[str, float], max_gripper_opening: float = 0.08) -> bool:
    """Tool: Verify whether object diameter/width fits inside gripper jaws."""
    width = object_dimensions.get("width", 0.05)
    return width <= max_gripper_opening


def validate_workspace_boundaries(x: float, y: float, z: float, bounds: Dict[str, List[float]]) -> bool:
    """Tool: Verify point lies within permitted workspace volume."""
    bx = bounds.get("x", [-0.8, 0.8])
    by = bounds.get("y", [-0.8, 0.8])
    bz = bounds.get("z", [0.0, 1.2])
    return (bx[0] <= x <= bx[1]) and (by[0] <= y <= by[1]) and (bz[0] <= z <= bz[1])
