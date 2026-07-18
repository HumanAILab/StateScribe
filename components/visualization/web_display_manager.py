from __future__ import annotations

from datetime import datetime
from queue import Queue
from typing import Any, Dict, Optional, Set

import numpy as np

from components.memory.change_memory import ChangeMemory
from components.memory.memory_manager import MemoryManager
from components.visualization.web_state import WebStateStore


class WebDisplayManager:
    def __init__(self, memory_manager: MemoryManager, state_store: Optional[WebStateStore] = None) -> None:
        self.memory_manager = memory_manager
        self.state_store = state_store if state_store is not None else WebStateStore()
        self._pending: Queue[Dict[str, Any]] = Queue()
        self._seen_frame_timestamps: Set[datetime] = set()
        self._last_world_name: Optional[str] = None

    def initialize(self) -> None:
        return

    def add_description(self, description: str, timestamp: Optional[datetime] = None) -> None:
        if not description:
            return
        self._pending.put(
            {
                "description_only": True,
                "description": str(description),
                "timestamp": timestamp or datetime.now(),
            }
        )

    def update_detection_result(
        self,
        current_image: np.ndarray,
        reference_image: np.ndarray,
        mask_t1: np.ndarray,
        mask_t0: np.ndarray,
        description: str,
        timestamp: datetime,
        change_memory: Optional[ChangeMemory] = None,
        vlm_annotated_t0: Optional[np.ndarray] = None,
        vlm_annotated_t1: Optional[np.ndarray] = None,
    ) -> None:
        self._pending.put(
            {
                "current_image": current_image.copy() if current_image is not None else None,
                "reference_image": reference_image.copy() if reference_image is not None else None,
                "mask_t1": mask_t1.copy() if mask_t1 is not None else None,
                "mask_t0": mask_t0.copy() if mask_t0 is not None else None,
                "vlm_annotated_t0": vlm_annotated_t0.copy() if vlm_annotated_t0 is not None else None,
                "vlm_annotated_t1": vlm_annotated_t1.copy() if vlm_annotated_t1 is not None else None,
                "description": str(description or ""),
                "timestamp": timestamp,
                "change_memory": change_memory,
            }
        )

    def update_memory_display(self, change_memory: ChangeMemory) -> None:
        self.state_store.update_change_memory(change_memory)

    def set_live_describing_frame(self, frame: Any) -> None:
        self.state_store.set_live_describing_frame(frame)

    def pump(self) -> None:
        world_name = self.memory_manager.current_world_name
        if world_name != self._last_world_name:
            self._last_world_name = world_name
            self._seen_frame_timestamps.clear()

        while not self._pending.empty():
            item = self._pending.get_nowait()
            if item.get("description_only"):
                continue
            self.state_store.add_change_event(
                current_image=item.get("current_image"),
                reference_image=item.get("reference_image"),
                mask_t1=item.get("mask_t1"),
                mask_t0=item.get("mask_t0"),
                description=item.get("description", ""),
                timestamp=item.get("timestamp"),
                vlm_annotated_t0=item.get("vlm_annotated_t0"),
                vlm_annotated_t1=item.get("vlm_annotated_t1"),
            )
            maybe_change_memory = item.get("change_memory")
            if maybe_change_memory is not None:
                self.state_store.update_change_memory(maybe_change_memory)

        frames = self.memory_manager.get_memory()
        for frame in frames:
            timestamp = getattr(frame, "timestamp", None)
            if timestamp is None:
                continue
            if timestamp in self._seen_frame_timestamps:
                continue
            self.state_store.ingest_latest_frame(frame)
            self._seen_frame_timestamps.add(timestamp)

    def clear(self) -> None:
        return

    def close(self) -> None:
        return
