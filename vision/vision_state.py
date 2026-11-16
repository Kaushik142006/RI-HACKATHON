"""
Vision state data models and schemas.
Pydantic schemas for detected objects, spatial coordinates, obstacles, and unified vision state.
"""

from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class Position3D(BaseModel):
    x: float = Field(..., description="X coordinate in robot base frame (meters)")
    y: float = Field(..., description="Y coordinate in robot base frame (meters)")
    z: float = Field(..., description="Z coordinate in robot base frame (meters)")


class BoundingBox(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    @property
    def area(self) -> float:
        return max(0.0, self.xmax - self.xmin) * max(0.0, self.ymax - self.ymin)


class DetectedObject(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    object_id: int
    track_id: Optional[int] = None
    class_name: str = Field(..., alias="class")
    color: str
    confidence: float
    bbox: Optional[BoundingBox] = None
    segmentation_mask: Optional[List[List[int]]] = None
    position_3d: Position3D
    dimensions_3d: Optional[Dict[str, float]] = None
    graspable: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Obstacle(BaseModel):
    obstacle_id: int
    obstacle_type: str
    position_3d: Position3D
    bounding_radius_m: float
    threat_level: str = "low"  # low, medium, critical


class EnvironmentStatus(BaseModel):
    status: str = Field(..., description="clean, manageable, or unsuitable")
    is_safe: bool = True
    clutter_ratio: float = 0.0
    detected_object_count: int = 0
    obstacle_count: int = 0
    rejection_reason: Optional[str] = None


class UnifiedVisionState(BaseModel):
    timestamp: float
    frame_id: int
    objects: List[DetectedObject] = Field(default_factory=list)
    obstacles: List[Obstacle] = Field(default_factory=list)
    environment_status: str = "manageable"  # clean, manageable, unsuitable
    details: Optional[EnvironmentStatus] = None
    table_plane: Optional[Dict[str, float]] = None

    def find_object(self, target_class: Optional[str] = None, target_color: Optional[str] = None) -> Optional[DetectedObject]:
        """Helper to find an object matching class and/or color."""
        for obj in self.objects:
            match_class = target_class is None or obj.class_name.lower() == target_class.lower()
            match_color = target_color is None or obj.color.lower() == target_color.lower()
            if match_class and match_color:
                return obj
        return None
