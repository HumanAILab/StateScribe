from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np
import open3d as o3d

from components.logging.frame_logger import FrameLogger
from components.memory.change_memory import ChangeMemory
from components.memory.memory_manager import MemoryManager
from components.visualization.web_state import WebStateStore
from config import (
    VISUALIZATION_ENABLE_3D_RECONSTRUCTION,
    WEB_VIS_HOST,
    WEB_VIS_MESH_MAX_TRIANGLES,
    WEB_VIS_PORT,
)

logger = logging.getLogger(__name__)


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None:
        return ""
    return str(value)


class _MeshTracker:
    def __init__(self, memory_manager: MemoryManager, change_memory: ChangeMemory, max_triangles: int) -> None:
        self.memory_manager = memory_manager
        self.change_memory = change_memory
        self.max_triangles = max(1000, int(max_triangles))
        self.enable_reconstruction = bool(VISUALIZATION_ENABLE_3D_RECONSTRUCTION)
        self.integrated_frames: set[str] = set()
        self.volume = None
        if self.enable_reconstruction:
            self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=0.03,
                sdf_trunc=0.09,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            )
        self._cached_mesh: Dict[str, Any] = {"vertices": [], "triangles": [], "vertex_colors": []}

    def _integrate_frame(self, frame: Any) -> None:
        if self.volume is None:
            return

        rgb = getattr(frame, "rgb_image", None)
        depth = getattr(frame, "depth_map", None)
        intrinsics = getattr(frame, "intrinsics", None)
        pose = getattr(frame, "pose_matrix", None)
        if rgb is None or depth is None or intrinsics is None or pose is None:
            return

        rgb_arr = np.asarray(rgb)
        if rgb_arr.dtype != np.uint8:
            rgb_arr = np.clip(rgb_arr, 0.0, 1.0) * 255.0
            rgb_arr = rgb_arr.astype(np.uint8)
        depth_arr = np.asarray(depth, dtype=np.float32)
        if depth_arr.shape[:2] != rgb_arr.shape[:2]:
            return

        rgb_o3d = o3d.geometry.Image(rgb_arr)
        depth_o3d = o3d.geometry.Image(depth_arr)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_scale=1.0,
            depth_trunc=5.0,
            convert_rgb_to_intensity=False,
        )
        h, w, _ = rgb_arr.shape
        intr = o3d.camera.PinholeCameraIntrinsic(
            int(w),
            int(h),
            float(intrinsics[0, 0]),
            float(intrinsics[1, 1]),
            float(intrinsics[0, 2]),
            float(intrinsics[1, 2]),
        )
        extrinsic = np.linalg.inv(np.asarray(pose, dtype=np.float64))
        self.volume.integrate(rgbd, intr, extrinsic)

    def _refresh_mesh(self) -> None:
        if self.volume is None:
            self._cached_mesh = {"vertices": [], "triangles": [], "vertex_colors": []}
            return

        mesh = self.volume.extract_triangle_mesh()
        if not mesh.has_triangles() or not mesh.has_vertices():
            self._cached_mesh = {"vertices": [], "triangles": [], "vertex_colors": []}
            return
        mesh.compute_vertex_normals()
        if len(mesh.triangles) > self.max_triangles:
            mesh = mesh.simplify_quadric_decimation(self.max_triangles)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        triangles = np.asarray(mesh.triangles, dtype=np.int32)
        vertex_colors: List[List[float]] = []
        if mesh.has_vertex_colors():
            colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
            if colors.shape[0] == vertices.shape[0]:
                vertex_colors = np.clip(colors, 0.0, 1.0).tolist()
        self._cached_mesh = {
            "vertices": np.round(vertices, 4).tolist(),
            "triangles": triangles.tolist(),
            "vertex_colors": vertex_colors,
        }

    def _serialize_bboxes(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        def _append(obj: Any, status: str) -> None:
            latest = obj.get_latest_snapshot()
            if latest is None or latest.bbox_3d is None:
                return
            bbox = np.asarray(latest.bbox_3d, dtype=np.float64).reshape(-1)
            if bbox.size != 6:
                return
            rows.append(
                {
                    "object_id": obj.object_id,
                    "status": status,
                    "description": latest.description,
                    "timestamp": _iso(latest.timestamp),
                    "min": [round(float(v), 4) for v in bbox[:3].tolist()],
                    "max": [round(float(v), 4) for v in bbox[3:].tolist()],
                    "color": [0.10, 0.78, 0.32] if status == "active" else [0.88, 0.25, 0.25],
                }
            )

        for obj in self.change_memory.get_active_objects():
            _append(obj, "active")
        for obj in self.change_memory.get_disappeared_objects():
            _append(obj, "disappeared")
        return rows

    def build_payload(self) -> Dict[str, Any]:
        if self.enable_reconstruction:
            frames = self.memory_manager.get_memory()
            missing: List[Any] = []
            for frame in reversed(frames):
                key = _iso(getattr(frame, "timestamp", None))
                if not key:
                    continue
                if key in self.integrated_frames:
                    break
                missing.append(frame)

            for frame in reversed(missing):
                try:
                    self._integrate_frame(frame)
                    key = _iso(getattr(frame, "timestamp", None))
                    if key:
                        self.integrated_frames.add(key)
                except Exception:
                    logger.exception("Failed to integrate frame for web mesh.")

            if missing:
                try:
                    self._refresh_mesh()
                except Exception:
                    logger.exception("Failed to refresh web mesh.")
        else:
            self._cached_mesh = {"vertices": [], "triangles": [], "vertex_colors": []}

        return {
            "mesh": self._cached_mesh,
            "bboxes": self._serialize_bboxes(),
            "updated_at": datetime.now().isoformat(),
        }


class _WebLogHandler(logging.Handler):
    def __init__(self, store: WebStateStore, kind: str) -> None:
        super().__init__(level=logging.DEBUG)
        self.store = store
        self.kind = kind

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
            entry: Dict[str, Any] = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "logger": record.name,
                "level": record.levelname,
                "message": message,
            }
            if self.kind == "speech":
                parsed = self.store.parse_speech_message(message)
                entry["tag"] = parsed.get("tag", "")
                entry["fields"] = parsed.get("fields", {})
            self.store.add_log_entry(self.kind, entry)
        except Exception:
            return


class _WebServerThread(threading.Thread):
    def __init__(self, store: WebStateStore, host: str, port: int, static_dir: str) -> None:
        super().__init__(daemon=True, name="statescribe-web-ui")
        self.store = store
        self.host = host
        self.port = int(port)
        self.static_dir = static_dir
        self.httpd: Optional[ThreadingHTTPServer] = None

    def run(self) -> None:
        handler_cls = self._build_handler()
        self.httpd = ThreadingHTTPServer((self.host, self.port), handler_cls)
        url = f"http://{self.host}:{self.port}"
        print(f"StateScribe Web UI: {url}", flush=True)
        logger.info("Web visualizer listening at %s", url)
        self.httpd.serve_forever(poll_interval=0.5)

    def stop(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()

    def _build_handler(self):
        store = self.store
        static_dir = self.static_dir

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path
                params = parse_qs(parsed.query)

                if path == "/api/meta":
                    self._send_json(store.get_meta())
                    return
                if path == "/api/realtime":
                    idx = _int_param(params, "index")
                    self._send_json(store.get_realtime_entry(index=idx))
                    return
                if path == "/api/change":
                    idx = _int_param(params, "index")
                    self._send_json(store.get_change_entry(index=idx))
                    return
                if path == "/api/change-memory":
                    self._send_json(store.get_change_memory_summary())
                    return
                if path == "/api/object":
                    object_id = _str_param(params, "id")
                    self._send_json(store.get_object_detail(object_id))
                    return
                if path == "/api/mesh":
                    self._send_json(store.get_mesh_payload())
                    return
                if path == "/api/logs":
                    kind = _str_param(params, "kind") or "app"
                    limit = _int_param(params, "limit", default=200) or 200
                    self._send_json(store.get_logs(kind, limit=limit))
                    return
                if path == "/api/speech-timeline":
                    limit = _int_param(params, "limit", default=120) or 120
                    self._send_json(store.get_speech_timeline(limit=limit))
                    return

                if path == "/":
                    path = "/index.html"
                self._serve_static(path)

            def _serve_static(self, route_path: str) -> None:
                target = route_path.lstrip("/")
                if target not in {"index.html", "app.js", "styles.css"}:
                    self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
                    return
                file_path = os.path.join(static_dir, target)
                if not os.path.exists(file_path):
                    self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
                    return

                content_type = "text/plain; charset=utf-8"
                if target.endswith(".html"):
                    content_type = "text/html; charset=utf-8"
                elif target.endswith(".css"):
                    content_type = "text/css; charset=utf-8"
                elif target.endswith(".js"):
                    content_type = "application/javascript; charset=utf-8"

                with open(file_path, "rb") as f:
                    body = f.read()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_json(self, payload: Any) -> None:
                body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                return

        return Handler


def _int_param(params: Dict[str, List[str]], key: str, default: Optional[int] = None) -> Optional[int]:
    values = params.get(key)
    if not values:
        return default
    try:
        return int(values[0])
    except Exception:
        return default


def _str_param(params: Dict[str, List[str]], key: str) -> str:
    values = params.get(key)
    if not values:
        return ""
    return str(values[0] or "")


class WebVisualizer:
    def __init__(
        self,
        memory_manager: MemoryManager,
        change_memory: ChangeMemory,
        state_store: Optional[WebStateStore] = None,
        host: str = WEB_VIS_HOST,
        port: int = WEB_VIS_PORT,
        max_triangles: int = WEB_VIS_MESH_MAX_TRIANGLES,
    ) -> None:
        self.memory_manager = memory_manager
        self.change_memory = change_memory
        self.state_store = state_store if state_store is not None else WebStateStore()
        self.host = host
        self.port = int(port)
        self.mesh_tracker = _MeshTracker(memory_manager, change_memory, max_triangles=max_triangles)

        self._app_handler = _WebLogHandler(self.state_store, kind="app")
        self._speech_handler = _WebLogHandler(self.state_store, kind="speech")
        self._server = _WebServerThread(
            self.state_store,
            host=self.host,
            port=self.port,
            static_dir=os.path.join(os.path.dirname(__file__), "web_ui"),
        )
        self._last_mesh_update_at = 0.0
        self._mesh_interval_sec = 0.8
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        logging.getLogger().addHandler(self._app_handler)
        logging.getLogger("statescribe.speech").addHandler(self._speech_handler)
        FrameLogger.subscribe(self._on_frame_log)

        self._server.start()

    def update(self) -> bool:
        if not self._initialized:
            return True

        now = time.perf_counter()
        if now - self._last_mesh_update_at >= self._mesh_interval_sec:
            payload = self.mesh_tracker.build_payload()
            self.state_store.set_mesh_payload(payload)
            self._last_mesh_update_at = now
        return True

    def destroy(self) -> None:
        if not self._initialized:
            return
        self._initialized = False
        try:
            self._server.stop()
        except Exception:
            logger.exception("Failed to stop web visualizer server.")

        try:
            FrameLogger.unsubscribe(self._on_frame_log)
        except Exception:
            logger.exception("Failed to detach frame log subscriber.")

        try:
            logging.getLogger().removeHandler(self._app_handler)
            logging.getLogger("statescribe.speech").removeHandler(self._speech_handler)
        except Exception:
            logger.exception("Failed to detach web log handlers.")

    def _on_frame_log(self, payload: Dict[str, Any]) -> None:
        self.state_store.add_log_entry("frame", payload)
