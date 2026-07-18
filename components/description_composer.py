from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np

from components.memory.change_memory import ChangeMemory
from config import (
    DESCRIPTION_REPLACEMENT_IOU_THRESHOLD,
    LOCATION_PHRASE_UNIT_SYSTEM,
)


class ChangeDescriptionComposer:
    LOCATION_TOKEN_PATTERN = re.compile(r"\[\[(DIR|DIS):([A-Za-z0-9_:\-]+)\]\]")

    def __init__(
        self,
        change_memory: ChangeMemory,
        latest_pose_matrix_getter: Optional[Callable[[], Optional[np.ndarray]]] = None,
        replacement_iou_threshold: float = DESCRIPTION_REPLACEMENT_IOU_THRESHOLD,
    ) -> None:
        self.change_memory = change_memory
        self.latest_pose_matrix_getter = latest_pose_matrix_getter
        self.replacement_iou_threshold = replacement_iou_threshold
        unit_system = (LOCATION_PHRASE_UNIT_SYSTEM or "").strip().lower()
        self.location_unit_system = "imperial" if unit_system == "imperial" else "metric"

    @staticmethod
    def _ensure_sentence(text: str) -> str:
        clean = text.strip()
        if not clean:
            return ""
        if clean[-1] in ".!?":
            return clean
        return f"{clean}."

    @staticmethod
    def _normalize_object_name(text: str) -> str:
        clean = text.strip()
        if not clean:
            return "object"

        lower = clean.lower()
        for article in ("the ", "an ", "a "):
            if lower.startswith(article):
                clean = clean[len(article):].strip()
                break
        return clean if clean else "object"

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        total = max(0, int(round(seconds)))
        if total < 60:
            return f"{total} second{'s' if total != 1 else ''}"

        minutes = total // 60
        if minutes < 60:
            return f"{minutes} minute{'s' if minutes != 1 else ''}"

        hours = minutes // 60
        if hours < 24:
            return f"{hours} hour{'s' if hours != 1 else ''}"

        days = hours // 24
        return f"{days} day{'s' if days != 1 else ''}"

    @staticmethod
    def _bbox_center(bbox: np.ndarray) -> Optional[np.ndarray]:
        arr = np.asarray(bbox, dtype=np.float64).reshape(-1)
        if arr.size != 6:
            return None
        return (arr[:3] + arr[3:]) / 2.0

    @staticmethod
    def _clock_direction(relative_xyz: np.ndarray) -> int:
        x, _, z = relative_xyz.tolist()
        angle_deg = float(np.degrees(np.arctan2(-x, z)))
        clock_idx = int(np.round((angle_deg % 360.0) / 30.0)) % 12
        return 12 if clock_idx == 0 else clock_idx

    def _direction_phrase_from_clock(self, clock: int) -> str:
        return f"at your {clock} o'clock"

    @staticmethod
    def _distance_phrase_metric(distance_m: float) -> str:
        if distance_m < 0.35:
            return "within arm's reach"
        if distance_m < 0.75:
            return "about half a meter away"
        if distance_m < 1.5:
            return "about one meter away"
        if distance_m < 2.5:
            return "about two meters away"
        if distance_m < 3.5:
            return "about three meters away"

        meters = max(1, int(round(distance_m)))
        unit = "meter" if meters == 1 else "meters"
        return f"about {meters} {unit} away"

    @staticmethod
    def _distance_phrase_imperial(distance_m: float) -> str:
        feet = distance_m * 3.28084
        if feet < 1.5:
            return "within arm's reach"
        if feet < 2.5:
            return "about 2 feet away"
        if feet < 4.0:
            return "about 3 feet away"
        if feet < 6.0:
            return "about 5 feet away"
        if feet < 10.0:
            return "about 8 feet away"
        if feet < 15.0:
            rounded_feet = max(1, int(round(feet)))
            unit = "foot" if rounded_feet == 1 else "feet"
            return f"about {rounded_feet} {unit} away"

        yards = max(1, int(round(feet / 3.0)))
        unit = "yard" if yards == 1 else "yards"
        return f"about {yards} {unit} away"

    def _distance_phrase(self, distance_m: float) -> str:
        if not np.isfinite(distance_m) or distance_m < 0:
            return ""
        if self.location_unit_system == "imperial":
            return self._distance_phrase_imperial(distance_m)
        return self._distance_phrase_metric(distance_m)

    @staticmethod
    def _compose_direction_distance(
        direction_phrase: str,
        distance_phrase: str,
        distance_m: float,
        clock: int,
    ) -> str:
        direction = (direction_phrase or "").strip()
        distance = (distance_phrase or "").strip()
        if direction and not distance:
            return direction
        if distance and not direction:
            return distance
        if not direction and not distance:
            return ""

        template_idx = (int(round(max(0.0, distance_m) * 10.0)) + int(clock)) % 3
        if template_idx == 0:
            return f"{direction}, {distance}"
        if template_idx == 1:
            return f"{distance}, {direction}"
        return f"{direction} and {distance}"

    def _location_parts(self, change: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pose = self._get_latest_pose_matrix()
        if pose is None:
            return None
        bbox = self._location_bbox_for_change(change)
        if bbox is None:
            return None
        center = self._bbox_center(bbox)
        if center is None:
            return None
        relative = self._world_to_user_relative(center, pose)
        if relative is None:
            return None

        distance_m = float(np.linalg.norm(relative))
        clock = self._clock_direction(relative)
        direction_phrase = self._direction_phrase_from_clock(clock)
        distance_phrase = self._distance_phrase(distance_m)
        parsed_now = self._compose_direction_distance(
            direction_phrase=direction_phrase,
            distance_phrase=distance_phrase,
            distance_m=distance_m,
            clock=clock,
        )
        return {
            "parsed_now": parsed_now,
            "direction": direction_phrase,
            "distance": distance_phrase,
            "clock": clock,
            "distance_m": round(distance_m, 3),
        }

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

    def _get_latest_pose_matrix(self) -> Optional[np.ndarray]:
        if self.latest_pose_matrix_getter is None:
            return None
        pose = self.latest_pose_matrix_getter()
        if pose is None:
            return None
        arr = np.asarray(pose, dtype=np.float64)
        if arr.shape != (4, 4):
            return None
        return arr

    def _location_bbox_for_change(self, change: Dict[str, Any]) -> Optional[np.ndarray]:
        ctype = change.get("change_type", "")
        b0 = change.get("bbox_3d_t0")
        b1 = change.get("bbox_3d_t1")
        if ctype == "appear":
            return b1 if b1 is not None else b0
        if ctype == "disappear":
            return b0 if b0 is not None else b1
        return b1 if b1 is not None else b0

    def _location_phrase(self, change: Dict[str, Any]) -> str:
        parts = self._location_parts(change)
        if not parts:
            return ""
        return str(parts.get("parsed_now", "") or "")

    def location_parse_for_object_id(self, object_id: str) -> Dict[str, Any]:
        clean_id = (object_id or "").strip()
        if not clean_id:
            return {}
        tracked = self.change_memory.get_object(clean_id)
        if tracked is None:
            return {}
        latest = tracked.get_latest_snapshot()
        if latest is None or latest.bbox_3d is None:
            return {}
        parts = self._location_parts(
            {
                "change_type": latest.change_type or "change",
                "bbox_3d_t0": latest.bbox_3d,
                "bbox_3d_t1": latest.bbox_3d,
            }
        )
        return parts if parts is not None else {}

    def direction_phrase_for_object_id(self, object_id: str) -> str:
        parsed = self.location_parse_for_object_id(object_id)
        return str(parsed.get("direction", "") or "")

    def distance_phrase_for_object_id(self, object_id: str) -> str:
        parsed = self.location_parse_for_object_id(object_id)
        return str(parsed.get("distance", "") or "")

    def resolve_location_tokens(self, text: str) -> str:
        raw = text or ""

        def _replace(match: re.Match[str]) -> str:
            token_type = (match.group(1) or "").strip().upper()
            object_id = (match.group(2) or "").strip()
            if token_type == "DIR":
                rendered = self.direction_phrase_for_object_id(object_id)
            else:
                rendered = self.distance_phrase_for_object_id(object_id)
            return rendered if rendered else ""

        resolved = self.LOCATION_TOKEN_PATTERN.sub(_replace, raw)
        return " ".join(resolved.split())

    def _compose_with_location(self, change: Dict[str, Any], content: str) -> str:
        location = self._location_phrase(change)
        if not location:
            return self._ensure_sentence(content)
        return self._ensure_sentence(f"{location}, {content}")

    @staticmethod
    def _bbox_iou_3d(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        x_min = max(float(bbox1[0]), float(bbox2[0]))
        y_min = max(float(bbox1[1]), float(bbox2[1]))
        z_min = max(float(bbox1[2]), float(bbox2[2]))
        x_max = min(float(bbox1[3]), float(bbox2[3]))
        y_max = min(float(bbox1[4]), float(bbox2[4]))
        z_max = min(float(bbox1[5]), float(bbox2[5]))

        if x_min >= x_max or y_min >= y_max or z_min >= z_max:
            return 0.0

        intersection = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        vol1 = (float(bbox1[3]) - float(bbox1[0])) * (float(bbox1[4]) - float(bbox1[1])) * (float(bbox1[5]) - float(bbox1[2]))
        vol2 = (float(bbox2[3]) - float(bbox2[0])) * (float(bbox2[4]) - float(bbox2[1])) * (float(bbox2[5]) - float(bbox2[2]))
        union = vol1 + vol2 - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    def _collect_active_bbox_snapshots(self) -> List[Dict[str, Any]]:
        snapshots: List[Dict[str, Any]] = []
        for obj in self.change_memory.get_active_objects():
            latest = obj.get_latest_snapshot()
            if latest is None or latest.bbox_3d is None:
                continue
            snapshots.append({
                "object_id": obj.object_id,
                "bbox_3d": latest.bbox_3d,
            })
        return snapshots

    def _get_object_name(self, change: Dict[str, Any]) -> str:
        obj = change.get("object_description", "").strip()
        if obj:
            return self._normalize_object_name(obj)

        object_id = change.get("object_id")
        if isinstance(object_id, str):
            tracked = self.change_memory.get_object(object_id)
            if tracked:
                latest = tracked.get_latest_snapshot()
                if latest:
                    latest_object = str(getattr(latest, "object_description", "") or "").strip()
                    if latest_object:
                        return self._normalize_object_name(latest_object)
                    if latest.description:
                        return self._normalize_object_name(latest.description)

        return "object"

    @staticmethod
    def _context_phrase(change: Dict[str, Any]) -> str:
        context = str(change.get("context_description", "")).strip()
        if not context:
            return ""

        lower = context.lower()
        prepositional_prefixes = (
            "on ",
            "in ",
            "at ",
            "near ",
            "by ",
            "under ",
            "behind ",
            "beside ",
            "next to ",
            "against ",
            "inside ",
            "outside ",
            "around ",
            "between ",
        )
        if lower.startswith(prepositional_prefixes):
            return context
        return f"around {context}"

    @staticmethod
    def _append_context(base: str, context_phrase: str, disappeared: bool = False) -> str:
        if not context_phrase:
            return base
        if disappeared:
            return f"{base}, previously {context_phrase}"
        return f"{base}, {context_phrase}"

    @staticmethod
    def _match_bbox_for_change(change: Dict[str, Any]) -> Optional[np.ndarray]:
        ctype = change.get("change_type", "")
        if ctype == "appear":
            return change.get("bbox_3d_t1")
        if ctype == "disappear":
            return change.get("bbox_3d_t0")
        return None

    def _should_ignore_appear_disappear(
        self,
        change: Dict[str, Any],
        active_bbox_snapshots: List[Dict[str, Any]],
        transient_object_ids: Set[str],
    ) -> bool:
        ctype = change.get("change_type", "")
        if ctype not in {"appear", "disappear"}:
            return False

        if ctype == "appear" and change.get("memory_action") in {"update_existing_object", "merge_existing_object"}:
            return True

        bbox = self._match_bbox_for_change(change)
        if bbox is None:
            return False

        current_id = change.get("object_id")
        for active in active_bbox_snapshots:
            active_bbox = active.get("bbox_3d")
            if active_bbox is None:
                continue
            active_id = active.get("object_id")
            if isinstance(active_id, str) and active_id in transient_object_ids:
                continue
            if isinstance(current_id, str) and active_id == current_id:
                continue
            if self._bbox_iou_3d(bbox, active_bbox) >= self.replacement_iou_threshold:
                return True
        return False

    def _find_replacement_pairs(self, changes: List[Dict[str, Any]]) -> List[tuple[int, int]]:
        disappear_indices = [
            idx for idx, change in enumerate(changes)
            if change.get("change_type") == "disappear" and change.get("bbox_3d_t0") is not None
        ]
        appear_indices = [
            idx for idx, change in enumerate(changes)
            if change.get("change_type") == "appear" and change.get("bbox_3d_t1") is not None
        ]

        used_appear: Set[int] = set()
        pairs: List[tuple[int, int]] = []

        for dis_idx in disappear_indices:
            dis_bbox = changes[dis_idx].get("bbox_3d_t0")
            if dis_bbox is None:
                continue

            best_idx = None
            best_iou = 0.0
            for app_idx in appear_indices:
                if app_idx in used_appear:
                    continue

                app_bbox = changes[app_idx].get("bbox_3d_t1")
                if app_bbox is None:
                    continue

                iou = self._bbox_iou_3d(dis_bbox, app_bbox)
                if iou < self.replacement_iou_threshold:
                    continue

                if iou > best_iou:
                    best_iou = iou
                    best_idx = app_idx

            if best_idx is not None:
                used_appear.add(best_idx)
                pairs.append((dis_idx, best_idx))

        return pairs

    def _compose_replacement_description(
        self, disappear_change: Dict[str, Any], appear_change: Dict[str, Any]
    ) -> str:
        old_name = self._get_object_name(disappear_change)
        new_name = self._get_object_name(appear_change)
        base = f"There was {old_name} here before, now it is {new_name}"
        context_phrase = self._context_phrase(appear_change) or self._context_phrase(disappear_change)
        base = self._append_context(base, context_phrase)
        return self._compose_with_location(appear_change, base)

    def _compose_change_description(self, change: Dict[str, Any]) -> str:
        context_phrase = self._context_phrase(change)
        raw_desc = change.get("change_description", "").strip()
        if raw_desc:
            return self._compose_with_location(change, self._append_context(raw_desc, context_phrase))
        obj = self._get_object_name(change)
        return self._compose_with_location(change, self._append_context(f"{obj} changed", context_phrase))

    def _compose_appear_description(self, change: Dict[str, Any]) -> str:
        obj = self._get_object_name(change)
        context_phrase = self._context_phrase(change)
        return self._compose_with_location(change, self._append_context(f"{obj} appeared", context_phrase))

    def _compose_disappear_description(self, change: Dict[str, Any]) -> str:
        obj = self._get_object_name(change)
        context_phrase = self._context_phrase(change)
        return self._compose_with_location(
            change,
            self._append_context(f"{obj} disappeared", context_phrase, disappeared=True),
        )

    def build(
        self,
        changes: List[Dict[str, Any]],
        ref_timestamp: datetime,
        cur_timestamp: datetime,
    ) -> str:
        if not changes:
            return ""

        active_bbox_snapshots = self._collect_active_bbox_snapshots()

        filtered_appear_disappear: List[Dict[str, Any]] = []
        change_only: List[Dict[str, Any]] = []
        transient_object_ids: Set[str] = {
            change.get("object_id")
            for change in changes
            if change.get("change_type", "") in {"appear", "disappear"}
            and isinstance(change.get("object_id"), str)
        }
        for change in changes:
            ctype = change.get("change_type", "")
            if ctype in {"appear", "disappear"}:
                if self._should_ignore_appear_disappear(
                    change,
                    active_bbox_snapshots,
                    transient_object_ids,
                ):
                    continue
                filtered_appear_disappear.append(change)
                continue
            change_only.append(change)

        parts: List[str] = []
        consumed: Set[int] = set()
        for dis_idx, app_idx in self._find_replacement_pairs(filtered_appear_disappear):
            parts.append(
                self._compose_replacement_description(
                    filtered_appear_disappear[dis_idx],
                    filtered_appear_disappear[app_idx],
                )
            )
            consumed.add(dis_idx)
            consumed.add(app_idx)

        for idx, change in enumerate(filtered_appear_disappear):
            if idx in consumed:
                continue
            ctype = change.get("change_type", "")
            if ctype == "appear":
                parts.append(self._compose_appear_description(change))
            elif ctype == "disappear":
                parts.append(self._compose_disappear_description(change))

        for change in change_only:
            parts.append(self._compose_change_description(change))

        if not parts:
            return ""

        elapsed = max(0.0, (cur_timestamp - ref_timestamp).total_seconds())
        if elapsed < 1.0:
            prefix = "Compared to just now,"
        else:
            prefix = f"Compared to {self._format_elapsed(elapsed)} ago,"

        return f"{prefix} {' '.join(parts)}"
