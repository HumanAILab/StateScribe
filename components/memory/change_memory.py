# components/memory/change_memory.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import logging
import numpy as np

logger = logging.getLogger(__name__)


def _to_optional_ndarray(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    return np.asarray(value)


@dataclass
class ObjectSnapshot:
    """Represents a single snapshot of an object at a specific time."""

    change_type: str  # "appear", "disappear", "change"
    timestamp: datetime
    description: str
    bbox_3d: np.ndarray  # [xmin, ymin, zmin, xmax, ymax, zmax] in world coordinates
    object_description: str = ""
    change_description: str = ""
    context_description: str = ""
    bbox_2d_image: Optional[np.ndarray] = None
    dino_feature: Optional[np.ndarray] = None
    description_embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "change_type": self.change_type,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "object_description": self.object_description,
            "change_description": self.change_description,
            "context_description": self.context_description,
            "bbox_3d": self.bbox_3d.tolist() if isinstance(self.bbox_3d, np.ndarray) else self.bbox_3d,
        }

    def to_persist_dict(self) -> Dict[str, Any]:
        return {
            "change_type": self.change_type,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "object_description": self.object_description,
            "change_description": self.change_description,
            "context_description": self.context_description,
            "bbox_3d": _to_optional_ndarray(self.bbox_3d),
            "bbox_2d_image": _to_optional_ndarray(self.bbox_2d_image),
            "dino_feature": _to_optional_ndarray(self.dino_feature),
            "description_embedding": _to_optional_ndarray(self.description_embedding),
        }

    @classmethod
    def from_persist_dict(cls, data: Dict[str, Any]) -> "ObjectSnapshot":
        return cls(
            change_type=str(data["change_type"]),
            timestamp=datetime.fromisoformat(str(data["timestamp"])),
            description=str(data["description"]),
            bbox_3d=np.asarray(data["bbox_3d"]),
            object_description=str(data.get("object_description", "")),
            change_description=str(data.get("change_description", "")),
            context_description=str(data.get("context_description", "")),
            bbox_2d_image=_to_optional_ndarray(data.get("bbox_2d_image")),
            dino_feature=_to_optional_ndarray(data.get("dino_feature")),
            description_embedding=_to_optional_ndarray(data.get("description_embedding")),
        )


@dataclass
class TrackedObject:
    """Represents a tracked object with its history of snapshots."""

    object_id: str
    snapshots: List[ObjectSnapshot] = field(default_factory=list)

    def get_latest_snapshot(self) -> Optional[ObjectSnapshot]:
        """Get the most recent snapshot."""
        return self.snapshots[-1] if self.snapshots else None

    def get_latest_snapshot_by_type(self, change_type: str) -> Optional[ObjectSnapshot]:
        """Get the latest snapshot of a specific change type."""
        for snap in reversed(self.snapshots):
            if snap.change_type == change_type:
                return snap
        return None

    def get_latest_snapshot_with_dino(self) -> Optional[ObjectSnapshot]:
        """Get the latest snapshot that has a DINO feature."""
        for snap in reversed(self.snapshots):
            if snap.dino_feature is not None:
                return snap
        return None

    def has_snapshot_type(self, change_type: str) -> bool:
        """Check whether this object already has a snapshot of the given type."""
        return any(snap.change_type == change_type for snap in self.snapshots)

    def is_disappeared(self) -> bool:
        """Check if the object has disappeared."""
        latest = self.get_latest_snapshot()
        return bool(latest and latest.change_type == "disappear")

    def add_snapshot(self, snapshot: ObjectSnapshot) -> None:
        """Add a new snapshot to the history."""
        self.snapshots.append(snapshot)
        logger.debug(
            "Added snapshot to object %s: %s at %s",
            self.object_id,
            snapshot.change_type,
            snapshot.timestamp,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "object_id": self.object_id,
            "snapshots": [s.to_dict() for s in self.snapshots],
        }

    def to_persist_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "snapshots": [s.to_persist_dict() for s in self.snapshots],
        }

    @classmethod
    def from_persist_dict(cls, data: Dict[str, Any]) -> "TrackedObject":
        return cls(
            object_id=str(data["object_id"]),
            snapshots=[ObjectSnapshot.from_persist_dict(item) for item in data.get("snapshots", [])],
        )


class ChangeMemory:
    """
    Manages the memory of object changes in the scene.
    Tracks objects and their snapshots over time.
    """

    def __init__(self) -> None:
        self.objects: Dict[str, TrackedObject] = {}
        self._next_object_id = 0

    def generate_object_id(self) -> str:
        """Generate a unique object ID."""
        obj_id = f"obj_{self._next_object_id:04d}"
        self._next_object_id += 1
        return obj_id

    def add_new_object(self, snapshot: ObjectSnapshot) -> str:
        """Add a new tracked object with its initial snapshot."""
        object_id = self.generate_object_id()
        self.objects[object_id] = TrackedObject(object_id=object_id, snapshots=[snapshot])
        logger.debug("Created new object %s: %s", object_id, snapshot.description)
        return object_id

    def update_object(self, object_id: str, snapshot: ObjectSnapshot) -> None:
        """Add a new snapshot to an existing object."""
        if object_id not in self.objects:
            logger.debug("Object %s not found, creating new", object_id)
            self.objects[object_id] = TrackedObject(object_id=object_id, snapshots=[])
        self.objects[object_id].add_snapshot(snapshot)

    def get_object(self, object_id: str) -> Optional[TrackedObject]:
        """Retrieve a tracked object by ID."""
        return self.objects.get(object_id)

    def get_all_objects(self) -> List[TrackedObject]:
        """Get all tracked objects."""
        return list(self.objects.values())

    def get_active_objects(self) -> List[TrackedObject]:
        """Get objects that have not disappeared."""
        return [obj for obj in self.objects.values() if not obj.is_disappeared()]

    def get_disappeared_objects(self, max_age_seconds: Optional[float] = None) -> List[TrackedObject]:
        """Get objects that have disappeared."""
        disappeared = [obj for obj in self.objects.values() if obj.is_disappeared()]
        if max_age_seconds is None:
            return disappeared

        now = datetime.now()
        filtered: List[TrackedObject] = []
        for obj in disappeared:
            latest = obj.get_latest_snapshot()
            if latest is None:
                continue
            age = (now - latest.timestamp).total_seconds()
            if age <= max_age_seconds:
                filtered.append(obj)
        return filtered

    def get_matching_pool(self, include_disappeared: bool) -> List[TrackedObject]:
        """Get objects that can be used as merge candidates."""
        if include_disappeared:
            return self.get_all_objects()
        return self.get_active_objects()

    def find_iou_candidates(
        self,
        bbox_3d: np.ndarray,
        iou_threshold: float,
        window_size: int,
        iou_mode: str = "3d",
        bbox_expansion_ratio: float = 0.0,
        include_disappeared: bool = True,
        max_snapshot_timestamp: Optional[datetime] = None,
    ) -> List[Tuple[TrackedObject, float]]:
        """
        Find candidate objects whose recent 3D boxes pass the IoU threshold.

        Returns candidates sorted by IoU descending.
        """
        if bbox_3d is None:
            return []

        candidates: List[Tuple[TrackedObject, float]] = []
        for obj in self.get_matching_pool(include_disappeared=include_disappeared):
            iou = self._max_iou_recent(
                obj,
                bbox_3d,
                window_size,
                iou_mode=iou_mode,
                bbox_expansion_ratio=bbox_expansion_ratio,
                max_snapshot_timestamp=max_snapshot_timestamp,
            )
            if iou >= iou_threshold:
                candidates.append((obj, iou))

        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates

    def _max_iou_recent(
        self,
        obj: TrackedObject,
        bbox_3d: np.ndarray,
        window_size: int,
        iou_mode: str = "3d",
        bbox_expansion_ratio: float = 0.0,
        max_snapshot_timestamp: Optional[datetime] = None,
    ) -> float:
        if window_size <= 0:
            return 0.0
        best = 0.0
        eligible = 0
        for snap in reversed(obj.snapshots):
            if max_snapshot_timestamp is not None and snap.timestamp >= max_snapshot_timestamp:
                continue
            if snap.bbox_3d is None:
                continue
            iou = self._compute_bbox_iou(
                bbox_3d,
                snap.bbox_3d,
                mode=iou_mode,
                expansion_ratio=bbox_expansion_ratio,
            )
            if iou > best:
                best = iou
            eligible += 1
            if eligible >= window_size:
                break
        return best

    @staticmethod
    def _expand_bbox_3d(bbox: np.ndarray, expansion_ratio: float) -> np.ndarray:
        box = np.asarray(bbox, dtype=np.float64)
        if expansion_ratio <= 0:
            return box

        mins = box[:3]
        maxs = box[3:]
        extents = np.maximum(maxs - mins, 0.0)
        center = (mins + maxs) * 0.5
        half_extents = extents * (1.0 + float(expansion_ratio)) * 0.5
        return np.concatenate([center - half_extents, center + half_extents])

    @classmethod
    def _compute_bbox_iou(
        cls,
        bbox1: np.ndarray,
        bbox2: np.ndarray,
        mode: str = "3d",
        expansion_ratio: float = 0.0,
    ) -> float:
        mode_normalized = str(mode).strip().lower()
        if mode_normalized not in {"3d", "xz"}:
            raise ValueError(f"Unsupported IoU mode: {mode}")

        box1 = cls._expand_bbox_3d(bbox1, expansion_ratio)
        box2 = cls._expand_bbox_3d(bbox2, expansion_ratio)

        axes = (0, 2) if mode_normalized == "xz" else (0, 1, 2)

        intersection = 1.0
        vol1 = 1.0
        vol2 = 1.0
        for axis in axes:
            axis_min = max(float(box1[axis]), float(box2[axis]))
            axis_max = min(float(box1[axis + 3]), float(box2[axis + 3]))
            if axis_min >= axis_max:
                return 0.0

            intersection *= axis_max - axis_min
            vol1 *= max(0.0, float(box1[axis + 3]) - float(box1[axis]))
            vol2 *= max(0.0, float(box2[axis + 3]) - float(box2[axis]))

        union = vol1 + vol2 - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    def to_dict(self) -> Dict[str, Any]:
        """Convert the entire memory to a dictionary."""
        return {
            "objects": {obj_id: obj.to_dict() for obj_id, obj in self.objects.items()},
            "next_object_id": self._next_object_id,
        }

    def to_persist_dict(self) -> Dict[str, Any]:
        return {
            "objects": {obj_id: obj.to_persist_dict() for obj_id, obj in self.objects.items()},
            "next_object_id": self._next_object_id,
        }

    def load_from_persist_dict(self, data: Dict[str, Any]) -> None:
        objects_data = data["objects"]
        self.objects = {str(obj_id): TrackedObject.from_persist_dict(obj_data) for obj_id, obj_data in objects_data.items()}
        self._next_object_id = int(data["next_object_id"])

    def clear(self) -> None:
        """Clear all tracked objects."""
        self.objects.clear()
        self._next_object_id = 0
        logger.debug("Change memory cleared")
