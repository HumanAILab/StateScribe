import json
import logging
from typing import Dict, List, Optional

import cv2
import numpy as np
from google.genai import types

from components.memory.change_memory import ChangeMemory, TrackedObject
from components.memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class AgentTools:
    """Tools exposed to the QA agent."""

    def __init__(self, memory_manager: MemoryManager, change_memory: ChangeMemory):
        self.memory_manager = memory_manager
        self.change_memory = change_memory

    def get_object_distance_and_direction(self) -> str:
        """Gets the current distance and clock direction for all tracked objects.

        Returns:
            A JSON string with one entry per tracked object, including distance, direction, latest description, and status.
        """
        pose = self._latest_user_pose()
        items: List[Dict[str, object]] = []

        for tracked in self._all_objects_sorted():
            latest = tracked.get_latest_snapshot()
            if latest is None:
                continue

            item: Dict[str, object] = {
                "object_id": tracked.object_id,
                "status": "disappeared" if tracked.is_disappeared() else "active",
                "latest_change_type": latest.change_type,
                "latest_description": latest.description,
                "latest_timestamp": latest.timestamp.isoformat() if latest.timestamp else "",
            }

            if latest.bbox_3d is None:
                item["warning"] = "object_has_no_3d_bbox"
                items.append(item)
                continue

            center_world = self._bbox_center(latest.bbox_3d)
            if center_world is None:
                item["warning"] = "invalid_bbox_3d"
                items.append(item)
                continue

            item["world_center_xyz"] = [round(float(v), 3) for v in center_world]

            if pose is None:
                item["warning"] = "latest_user_pose_unavailable"
                items.append(item)
                continue

            relative = self._world_to_user_relative(center_world, pose)
            if relative is None:
                item["warning"] = "failed_to_project_to_user_frame"
                items.append(item)
                continue

            distance = float(np.linalg.norm(relative))
            clock = self._clock_direction(relative)
            item["relative_xyz"] = [round(float(v), 3) for v in relative]
            item["distance_feet"] = round(distance * 3.28084, 2)
            item["clock_direction"] = f"{clock} o'clock"
            items.append(item)

        payload = {
            "returned_count": len(items),
            "objects": items,
        }
        return json.dumps(payload, ensure_ascii=True)

    def retrieve_recent_images(self, limit: int = 1) -> tuple[List[types.Part], str]:
        """Retrieves the most recent frame images from memory for the agent to see.

        Args:
            limit: Number of latest frames to retrieve (default 1).

        Returns:
            A tuple of (list of image Parts for the model, summary text with timestamps).
        """
        try:
            count = int(limit)
        except (TypeError, ValueError):
            count = 1
        if count <= 0:
            count = 1

        frames = self.memory_manager.get_memory()
        if not frames:
            return ([], "No images found in memory.")

        frames_sorted = sorted(frames, key=lambda frame: frame.timestamp, reverse=True)[:count]
        parts: List[types.Part] = []
        lines: List[str] = []

        for idx, frame in enumerate(frames_sorted, start=1):
            rgb = getattr(frame, "rgb_image", None)
            if rgb is None:
                continue
            try:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                ok, enc = cv2.imencode(".jpg", bgr)
                if not ok:
                    continue
                parts.append(types.Part.from_bytes(data=enc.tobytes(), mime_type="image/jpeg"))
                lines.append(f"[{idx}] timestamp={frame.timestamp.isoformat()}")
            except Exception:
                logger.warning("Failed to encode frame image.", exc_info=True)
                continue

        if not parts:
            return ([], "No encodable images found in memory.")

        text = f"Retrieved {len(parts)} latest image(s), newest first.\n" + "\n".join(lines)
        return (parts, text)

    def get_recent_change_snapshots(self, limit: int = 3) -> str:
        """Gets the N most recent scene-change snapshots (appear/disappear/change) with distance and direction.

        Args:
            limit: Number of snapshots to return, newest first (default 3).

        Returns:
            A JSON string with recent_change_snapshots (each with snapshot and distance_and_direction).
        """
        try:
            count = int(limit)
        except (TypeError, ValueError):
            count = 3
        if count <= 0:
            count = 3

        all_rows = []
        for obj in self._all_objects_sorted():
            for snap in obj.snapshots:
                all_rows.append((snap.timestamp, obj.object_id, snap))

        all_rows.sort(key=lambda row: row[0], reverse=True)
        selected = all_rows[:count]

        pose = self._latest_user_pose()
        items: List[Dict[str, object]] = []
        for idx, (_, object_id, snap) in enumerate(selected, start=1):
            item: Dict[str, object] = {
                "rank": idx,
                "object_id": object_id,
                "snapshot": {
                    "change_type": snap.change_type,
                    "description": snap.description,
                    "timestamp": snap.timestamp.isoformat() if snap.timestamp else "",
                },
            }

            if snap.bbox_3d is None:
                item["distance_and_direction"] = {"warning": "snapshot_has_no_3d_bbox"}
                items.append(item)
                continue

            center_world = self._bbox_center(snap.bbox_3d)
            if center_world is None:
                item["distance_and_direction"] = {"warning": "invalid_bbox_3d"}
                items.append(item)
                continue

            spatial: Dict[str, object] = {
                "world_center_xyz": [round(float(v), 3) for v in center_world],
            }
            if pose is None:
                spatial["warning"] = "latest_user_pose_unavailable"
                item["distance_and_direction"] = spatial
                items.append(item)
                continue

            relative = self._world_to_user_relative(center_world, pose)
            if relative is None:
                spatial["warning"] = "failed_to_project_to_user_frame"
                item["distance_and_direction"] = spatial
                items.append(item)
                continue

            distance = float(np.linalg.norm(relative))
            clock = self._clock_direction(relative)
            spatial["relative_xyz"] = [round(float(v), 3) for v in relative]
            spatial["distance_feet"] = round(distance * 3.28084, 2)
            spatial["clock_direction"] = f"{clock} o'clock"
            item["distance_and_direction"] = spatial
            items.append(item)

        payload = {
            "requested_count": count,
            "returned_count": len(items),
            "order": "newest_to_oldest",
            "recent_change_snapshots": items,
        }
        return json.dumps(payload, ensure_ascii=True)

    def _resolve_object(self, reference: str) -> Optional[TrackedObject]:
        normalized_id = self._normalize_object_id(reference)
        if normalized_id is None:
            return None
        return self.change_memory.get_object(normalized_id)

    @staticmethod
    def _normalize_object_id(reference: str) -> Optional[str]:
        ref = (reference or "").strip().lower()
        if not ref:
            return None

        if ref.startswith("obj_"):
            suffix = ref[4:]
            if not suffix.isdigit():
                return None
            return f"obj_{int(suffix):04d}"

        if ref.startswith("#"):
            ref = ref[1:]

        if ref.isdigit():
            return f"obj_{int(ref):04d}"

        return None

    def _available_objects_preview(self, limit: int = 12) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        for obj in self._all_objects_sorted():
            latest = obj.get_latest_snapshot()
            rows.append(
                {
                    "object_id": obj.object_id,
                    "description": (latest.description if latest else ""),
                }
            )
            if len(rows) >= limit:
                break
        return rows

    @staticmethod
    def _bbox_center(bbox_3d: np.ndarray) -> Optional[np.ndarray]:
        arr = np.asarray(bbox_3d, dtype=np.float64).reshape(-1)
        if arr.size != 6:
            return None
        return (arr[:3] + arr[3:]) / 2.0

    def _latest_user_pose(self) -> Optional[np.ndarray]:
        frames = self.memory_manager.get_memory()
        if not frames:
            return None
        latest = frames[-1]
        pose = getattr(latest, "pose_matrix", None)
        if pose is None:
            return None
        return np.asarray(pose, dtype=np.float64)

    @staticmethod
    def _world_to_user_relative(world_point: np.ndarray, pose_matrix: np.ndarray) -> Optional[np.ndarray]:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        cam_pos = pose_matrix[:3, 3]
        cam_fwd_world = pose_matrix[:3, 2]

        forward = cam_fwd_world - np.dot(cam_fwd_world, world_up) * world_up
        if np.linalg.norm(forward) < 1e-6:
            cam_x_world = pose_matrix[:3, 0]
            forward = cam_x_world - np.dot(cam_x_world, world_up) * world_up
        if np.linalg.norm(forward) < 1e-6:
            forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        z_axis = forward / (np.linalg.norm(forward) + 1e-9)
        y_axis = world_up
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-9)

        human_pose = np.eye(4, dtype=np.float64)
        human_pose[:3, 0] = x_axis
        human_pose[:3, 1] = y_axis
        human_pose[:3, 2] = z_axis
        human_pose[:3, 3] = cam_pos

        inv_pose = np.linalg.inv(human_pose)
        point_h = np.array([world_point[0], world_point[1], world_point[2], 1.0], dtype=np.float64)
        return (inv_pose @ point_h)[:3]

    @staticmethod
    def _clock_direction(relative_xyz: np.ndarray) -> int:
        x, _, z = relative_xyz.tolist()
        angle_deg = float(np.degrees(np.arctan2(-x, z)))
        clock_index = int(np.round((angle_deg % 360.0) / 30.0)) % 12
        return 12 if clock_index == 0 else clock_index

    def _all_objects_sorted(self) -> List[TrackedObject]:
        return sorted(self.change_memory.get_all_objects(), key=lambda item: item.object_id)
