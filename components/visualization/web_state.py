from __future__ import annotations

import base64
import copy
import json
import threading
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

import cv2
import numpy as np


def _to_iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None:
        return ""
    return str(value)


def _to_uint8_rgb(image: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.ndim != 3:
        return None
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0) * 255.0
    else:
        arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


def _encode_rgb_data_url(image_rgb: Optional[np.ndarray], ext: str = ".jpg", quality: int = 85) -> str:
    img = _to_uint8_rgb(image_rgb)
    if img is None:
        return ""
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    params: List[int] = []
    mime = "image/jpeg"
    if ext == ".jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, int(max(30, min(100, quality)))]
    elif ext == ".png":
        mime = "image/png"
    ok, encoded = cv2.imencode(ext, bgr, params)
    if not ok:
        return ""
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _encode_gray_data_url(image_gray: Optional[np.ndarray], ext: str = ".png") -> str:
    if image_gray is None:
        return ""
    arr = np.asarray(image_gray)
    if arr.ndim != 2:
        return ""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(ext, arr)
    if not ok:
        return ""
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _colorize_scalar_map(array_like: Optional[np.ndarray], is_depth: bool) -> Optional[np.ndarray]:
    if array_like is None:
        return None
    arr = np.asarray(array_like)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim > 2:
        arr = arr[..., 0]
    if arr.ndim != 2:
        return None

    arr = arr.astype(np.float32)
    valid = np.isfinite(arr)
    if is_depth:
        valid &= arr > 0

    if not np.any(valid):
        norm = np.zeros_like(arr, dtype=np.uint8)
    else:
        values = arr[valid]
        if values.size > 1000:
            vmin = float(np.percentile(values, 1.0))
            vmax = float(np.percentile(values, 99.0))
        else:
            vmin = float(np.min(values))
            vmax = float(np.max(values))
        if not np.isfinite(vmax - vmin) or (vmax - vmin) < 1e-6:
            vmax = vmin + 1e-6
        norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
        norm = (norm * 255.0).astype(np.uint8)

    colored = cv2.applyColorMap(norm, getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET))
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _overlay_mask(image_rgb: Optional[np.ndarray], mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    img = _to_uint8_rgb(image_rgb)
    if img is None:
        return None
    out = img.copy()
    if mask is None:
        return out
    m = np.asarray(mask)
    if m.ndim != 2:
        return out
    if m.shape[:2] != out.shape[:2]:
        m = cv2.resize(m.astype(np.uint8), (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
    binary = m > 0
    color = np.zeros_like(out)
    color[binary] = (0, 255, 80)
    out = cv2.addWeighted(out, 0.72, color, 0.28, 0.0)
    return out


def _serialize_feature(vector: Optional[np.ndarray], preview_limit: int = 32) -> Optional[Dict[str, Any]]:
    if vector is None:
        return None
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    norm = float(np.linalg.norm(arr))
    preview = [round(float(v), 5) for v in arr[:preview_limit].tolist()]
    return {
        "dim": int(arr.size),
        "norm": round(norm, 6),
        "preview": preview,
    }


def _serialize_bbox(bbox: Optional[np.ndarray]) -> List[float]:
    if bbox is None:
        return []
    arr = np.asarray(bbox, dtype=np.float64).reshape(-1)
    return [round(float(v), 4) for v in arr.tolist()]


class WebStateStore:
    def __init__(
        self,
        realtime_limit: Optional[int] = None,
        change_limit: Optional[int] = None,
        log_limit: int = 1500,
        speech_limit: int = 600,
    ) -> None:
        self._lock = threading.Lock()
        self._realtime_limit = self._normalize_limit(realtime_limit)
        self._change_limit = self._normalize_limit(change_limit)

        self._realtime_history: List[Dict[str, Any]] = []
        self._change_history: List[Dict[str, Any]] = []
        self._latest_realtime_key: Optional[str] = None

        self._change_memory_summary: List[Dict[str, Any]] = []
        self._change_memory_details: Dict[str, Dict[str, Any]] = {}
        self._live_describing: Dict[str, str] = {"timestamp": "", "image": ""}

        self._mesh_payload: Dict[str, Any] = {
            "mesh": {"vertices": [], "triangles": []},
            "bboxes": [],
            "updated_at": "",
        }

        self._logs: Dict[str, Deque[Dict[str, Any]]] = {
            "app": deque(maxlen=max(200, int(log_limit))),
            "speech": deque(maxlen=max(200, int(log_limit))),
            "frame": deque(maxlen=max(100, int(log_limit))),
        }
        self._speech_timeline: Deque[Dict[str, Any]] = deque(maxlen=max(80, int(speech_limit)))

        self._versions = {
            "realtime": 0,
            "change": 0,
            "change_memory": 0,
            "mesh": 0,
            "app_log": 0,
            "speech_log": 0,
            "frame_log": 0,
            "speech_timeline": 0,
        }

    @staticmethod
    def _normalize_limit(limit: Optional[int]) -> Optional[int]:
        if limit is None:
            return None
        value = int(limit)
        if value <= 0:
            return None
        return max(20, value)

    def ingest_latest_frame(self, frame: Any) -> None:
        if frame is None:
            return
        ts = _to_iso(getattr(frame, "timestamp", None))
        if not ts:
            return
        with self._lock:
            if self._latest_realtime_key == ts:
                return

        rgb = _to_uint8_rgb(getattr(frame, "rgb_image", None))
        if rgb is None:
            return

        depth = getattr(frame, "depth_map", None)
        conf = getattr(frame, "confidence_map", None)
        depth_vis = _colorize_scalar_map(depth, is_depth=True)
        conf_vis = _colorize_scalar_map(conf, is_depth=False) if conf is not None else None

        if depth_vis is not None and depth_vis.shape[:2] != rgb.shape[:2]:
            depth_vis = cv2.resize(depth_vis, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        if conf_vis is not None and conf_vis.shape[:2] != rgb.shape[:2]:
            conf_vis = cv2.resize(conf_vis, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        entry = {
            "timestamp": ts,
            "rgb": _encode_rgb_data_url(rgb, ext=".jpg", quality=86),
            "depth": _encode_rgb_data_url(depth_vis, ext=".jpg", quality=86) if depth_vis is not None else "",
            "confidence": _encode_rgb_data_url(conf_vis, ext=".jpg", quality=86) if conf_vis is not None else "",
            "describing_image": "",
            "describing_timestamp": "",
            "shape": [int(rgb.shape[0]), int(rgb.shape[1])],
        }

        with self._lock:
            if self._live_describing.get("timestamp") == ts:
                entry["describing_image"] = self._live_describing.get("image", "")
                entry["describing_timestamp"] = ts
            self._latest_realtime_key = ts
            self._realtime_history.append(entry)
            if self._realtime_limit is not None and len(self._realtime_history) > self._realtime_limit:
                self._realtime_history = self._realtime_history[-self._realtime_limit :]
            self._versions["realtime"] += 1

    def set_live_describing_frame(self, frame: Any) -> None:
        if frame is None:
            return
        ts = _to_iso(getattr(frame, "timestamp", None))
        if not ts:
            return
        rgb = _to_uint8_rgb(getattr(frame, "rgb_image", None))
        if rgb is None:
            return
        encoded = _encode_rgb_data_url(rgb, ext=".jpg", quality=86)
        if not encoded:
            return

        with self._lock:
            self._live_describing = {"timestamp": ts, "image": encoded}
            for idx in range(len(self._realtime_history) - 1, -1, -1):
                row = self._realtime_history[idx]
                if row.get("timestamp") != ts:
                    continue
                row["describing_image"] = encoded
                row["describing_timestamp"] = ts
                break
            self._versions["realtime"] += 1

    def add_change_event(
        self,
        current_image: Optional[np.ndarray],
        reference_image: Optional[np.ndarray],
        mask_t1: Optional[np.ndarray],
        mask_t0: Optional[np.ndarray],
        description: str,
        timestamp: Any,
        vlm_annotated_t0: Optional[np.ndarray],
        vlm_annotated_t1: Optional[np.ndarray],
    ) -> None:
        current = _to_uint8_rgb(current_image)
        reference = _to_uint8_rgb(reference_image)
        if current is None or reference is None:
            return

        current_mask_overlay = _overlay_mask(current, mask_t1)
        reference_mask_overlay = _overlay_mask(reference, mask_t0)
        vlm_t0 = _to_uint8_rgb(vlm_annotated_t0)
        vlm_t1 = _to_uint8_rgb(vlm_annotated_t1)

        entry = {
            "timestamp": _to_iso(timestamp),
            "description": str(description or ""),
            "panels": {
                "current_frame": _encode_rgb_data_url(current, ext=".jpg", quality=86),
                "reference_frame": _encode_rgb_data_url(reference, ext=".jpg", quality=86),
                "current_gemini": _encode_rgb_data_url(vlm_t1, ext=".jpg", quality=86) if vlm_t1 is not None else "",
                "reference_gemini": _encode_rgb_data_url(vlm_t0, ext=".jpg", quality=86) if vlm_t0 is not None else "",
                "current_mask": _encode_rgb_data_url(current_mask_overlay, ext=".jpg", quality=86)
                if current_mask_overlay is not None
                else "",
                "reference_mask": _encode_rgb_data_url(reference_mask_overlay, ext=".jpg", quality=86)
                if reference_mask_overlay is not None
                else "",
                "raw_mask_t1": _encode_gray_data_url(mask_t1),
                "raw_mask_t0": _encode_gray_data_url(mask_t0),
            },
            "shape_current": [int(current.shape[0]), int(current.shape[1])],
            "shape_reference": [int(reference.shape[0]), int(reference.shape[1])],
        }

        with self._lock:
            self._change_history.append(entry)
            if self._change_limit is not None and len(self._change_history) > self._change_limit:
                self._change_history = self._change_history[-self._change_limit :]
            self._versions["change"] += 1

    def update_change_memory(self, change_memory: Any) -> None:
        if change_memory is None:
            return
        all_objects = change_memory.get_all_objects()
        summaries: List[Dict[str, Any]] = []
        details: Dict[str, Dict[str, Any]] = {}

        for obj in sorted(all_objects, key=lambda item: item.object_id):
            latest = obj.get_latest_snapshot()
            latest_desc = latest.description if latest is not None else ""
            latest_ts = _to_iso(latest.timestamp if latest is not None else None)
            status = "disappeared" if obj.is_disappeared() else "active"

            summaries.append(
                {
                    "object_id": obj.object_id,
                    "status": status,
                    "snapshot_count": len(obj.snapshots),
                    "latest_description": latest_desc,
                    "latest_timestamp": latest_ts,
                    "latest_change_type": latest.change_type if latest is not None else "",
                }
            )

            snapshots: List[Dict[str, Any]] = []
            for snap in obj.snapshots:
                snapshots.append(
                    {
                        "change_type": snap.change_type,
                        "timestamp": _to_iso(snap.timestamp),
                        "description": snap.description,
                        "bbox_3d": _serialize_bbox(snap.bbox_3d),
                        "bbox_image": _encode_rgb_data_url(snap.bbox_2d_image, ext=".jpg", quality=82)
                        if snap.bbox_2d_image is not None
                        else "",
                        "dino_feature": _serialize_feature(snap.dino_feature),
                        "description_embedding": _serialize_feature(snap.description_embedding),
                    }
                )

            details[obj.object_id] = {
                "object_id": obj.object_id,
                "status": status,
                "latest_description": latest_desc,
                "latest_timestamp": latest_ts,
                "snapshots": snapshots,
            }

        with self._lock:
            self._change_memory_summary = summaries
            self._change_memory_details = details
            self._versions["change_memory"] += 1

    def set_mesh_payload(self, mesh_payload: Dict[str, Any]) -> None:
        payload = mesh_payload or {}
        with self._lock:
            self._mesh_payload = {
                "mesh": payload.get("mesh", {"vertices": [], "triangles": []}),
                "bboxes": payload.get("bboxes", []),
                "updated_at": payload.get("updated_at", ""),
            }
            self._versions["mesh"] += 1

    def add_log_entry(self, kind: str, entry: Dict[str, Any]) -> None:
        target = "app"
        if kind == "speech":
            target = "speech"
        elif kind == "frame":
            target = "frame"

        payload = copy.deepcopy(entry)
        with self._lock:
            self._logs[target].append(payload)
            self._versions[f"{target}_log"] += 1

        if target == "speech":
            tag = str(payload.get("tag", "")).strip().lower()
            if tag == "broker_emit":
                spoken = {
                    "timestamp": payload.get("timestamp", ""),
                    "source": payload.get("fields", {}).get("source", ""),
                    "text": payload.get("fields", {}).get("text", payload.get("message", "")),
                }
                with self._lock:
                    self._speech_timeline.append(spoken)
                    self._versions["speech_timeline"] += 1

    def get_meta(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "versions": copy.deepcopy(self._versions),
                "counts": {
                    "realtime": len(self._realtime_history),
                    "change": len(self._change_history),
                    "change_memory_objects": len(self._change_memory_summary),
                    "speech_timeline": len(self._speech_timeline),
                },
                "server_time": datetime.now().isoformat(),
            }

    def get_realtime_entry(self, index: Optional[int] = None) -> Dict[str, Any]:
        with self._lock:
            if not self._realtime_history:
                return {}
            idx = len(self._realtime_history) - 1 if index is None else int(index)
            idx = max(0, min(idx, len(self._realtime_history) - 1))
            payload = copy.deepcopy(self._realtime_history[idx])
            payload["live_describing_image"] = self._live_describing.get("image", "")
            payload["live_describing_timestamp"] = self._live_describing.get("timestamp", "")
            payload["index"] = idx
            payload["count"] = len(self._realtime_history)
            return payload

    def get_change_entry(self, index: Optional[int] = None) -> Dict[str, Any]:
        with self._lock:
            if not self._change_history:
                return {}
            idx = len(self._change_history) - 1 if index is None else int(index)
            idx = max(0, min(idx, len(self._change_history) - 1))
            payload = copy.deepcopy(self._change_history[idx])
            payload["index"] = idx
            payload["count"] = len(self._change_history)
            return payload

    def get_change_memory_summary(self) -> List[Dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._change_memory_summary)

    def get_object_detail(self, object_id: str) -> Dict[str, Any]:
        oid = (object_id or "").strip()
        if not oid:
            return {}
        with self._lock:
            return copy.deepcopy(self._change_memory_details.get(oid, {}))

    def get_mesh_payload(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._mesh_payload)

    def get_logs(self, kind: str, limit: int = 200) -> List[Dict[str, Any]]:
        target = "app"
        if kind == "speech":
            target = "speech"
        elif kind == "frame":
            target = "frame"
        cap = max(1, min(2000, int(limit)))
        with self._lock:
            items = list(self._logs[target])[-cap:]
            return copy.deepcopy(items)

    def get_speech_timeline(self, limit: int = 100) -> List[Dict[str, Any]]:
        cap = max(1, min(800, int(limit)))
        with self._lock:
            items = list(self._speech_timeline)[-cap:]
            return copy.deepcopy(items)

    @staticmethod
    def parse_speech_message(message: str) -> Dict[str, Any]:
        raw = str(message or "").strip()
        parsed = {"tag": "", "fields": {}, "message": raw}
        if not raw.startswith("["):
            return parsed
        close = raw.find("]")
        if close <= 1:
            return parsed
        tag = raw[1:close].strip()
        rest = raw[close + 1 :].strip()
        fields: Dict[str, str] = {}

        if " text=" in f" {rest}":
            prefix, text_value = rest.split(" text=", 1)
            for token in prefix.split():
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                fields[key.strip()] = value.strip()
            fields["text"] = text_value.strip()
        else:
            for token in rest.split():
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                fields[key.strip()] = value.strip()

        parsed["tag"] = tag
        parsed["fields"] = fields
        return parsed

    @staticmethod
    def safe_json_bytes(payload: Any) -> bytes:
        return json.dumps(payload, ensure_ascii=True).encode("utf-8")
