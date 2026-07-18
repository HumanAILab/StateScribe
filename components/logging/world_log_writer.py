from __future__ import annotations

from datetime import datetime
import json
import logging
import os
import threading
from typing import Any, Callable, Dict, Optional, TextIO, Tuple

from config import LONG_TERM_MEMORY_BASE_PATH
from components.logging.frame_logger import FrameLogger
from components.visualization.web_state import WebStateStore


_LOG_FILENAMES = {
    "app": "app.jsonl",
    "speech": "speech.jsonl",
    "frame": "frame.jsonl",
}


class _StructuredLogHandler(logging.Handler):
    def __init__(self, writer: "WorldStructuredLogWriter", kind: str) -> None:
        super().__init__(level=logging.DEBUG)
        self._writer = writer
        self._kind = kind

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
            entry: Dict[str, Any] = {
                "kind": self._kind,
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "logger": record.name,
                "level": record.levelname,
                "message": message,
            }
            if self._kind == "speech":
                parsed = WebStateStore.parse_speech_message(message)
                entry["tag"] = parsed.get("tag", "")
                entry["fields"] = parsed.get("fields", {})
            self._writer.write(self._kind, entry)
        except Exception:
            return


class WorldStructuredLogWriter:
    def __init__(
        self,
        current_world_name_getter: Callable[[], Optional[str]],
        base_path: str = LONG_TERM_MEMORY_BASE_PATH,
    ) -> None:
        self._current_world_name_getter = current_world_name_getter
        self._base_path = base_path
        self._lock = threading.Lock()
        self._files: Dict[Tuple[str, str], TextIO] = {}
        self._last_world_name: Optional[str] = None
        self._started = False
        self._app_handler = _StructuredLogHandler(self, "app")
        self._speech_handler = _StructuredLogHandler(self, "speech")

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        logging.getLogger().addHandler(self._app_handler)
        logging.getLogger("statescribe.speech").addHandler(self._speech_handler)
        FrameLogger.subscribe(self._on_frame_log)

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        try:
            FrameLogger.unsubscribe(self._on_frame_log)
        except Exception:
            pass
        try:
            logging.getLogger().removeHandler(self._app_handler)
        except Exception:
            pass
        try:
            logging.getLogger("statescribe.speech").removeHandler(self._speech_handler)
        except Exception:
            pass

        with self._lock:
            for handle in self._files.values():
                try:
                    handle.close()
                except Exception:
                    pass
            self._files.clear()

    def write(self, kind: str, entry: Dict[str, Any]) -> None:
        target_kind = kind if kind in _LOG_FILENAMES else "app"
        payload = dict(entry or {})
        payload["kind"] = target_kind

        world_name = self._resolve_world_name(payload)
        if not world_name:
            return
        payload["world_name"] = world_name

        line = json.dumps(payload, ensure_ascii=True)
        with self._lock:
            self._last_world_name = world_name
            handle = self._get_handle(world_name, target_kind)
            handle.write(line)
            handle.write("\n")
            handle.flush()

    def _resolve_world_name(self, entry: Dict[str, Any]) -> str:
        world_name = str(entry.get("world_name", "") or "").strip()
        if world_name:
            return world_name

        current_world = self._current_world_name_getter()
        if isinstance(current_world, str) and current_world.strip():
            return current_world.strip()

        return (self._last_world_name or "").strip()

    def _get_handle(self, world_name: str, kind: str) -> TextIO:
        key = (world_name, kind)
        handle = self._files.get(key)
        if handle is not None and not handle.closed:
            return handle

        world_dir = os.path.join(self._base_path, world_name, "logs")
        os.makedirs(world_dir, exist_ok=True)
        file_path = os.path.join(world_dir, _LOG_FILENAMES[kind])
        handle = open(file_path, "a", encoding="utf-8", buffering=1)
        self._files[key] = handle
        return handle

    def _on_frame_log(self, payload: Dict[str, Any]) -> None:
        self.write("frame", payload)
