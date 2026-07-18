# components/memory/memory_manager.py
import os
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any

import numpy as np

from .frame import Frame
from memo_utils.common import compress_object, decompress_object
from config import (
    LONG_TERM_MEMORY_BASE_PATH,
    LTM_LOAD_LATEST_CAPTURE_ONLY,
    LTM_CAPTURE_GAP_SECONDS,
)

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages a single in-memory frame list plus on-disk persistence.
    Background workers handle asynchronous processing while preserving
    final storage order.
    """

    # ---------- life-cycle --------------------------------------------------

    def __init__(self, max_workers: int = 8):
        # memory buffer (single source of truth)
        self.memory: List[Frame] = []

        # session-related paths
        self.current_world_name: Optional[str] = None
        self.ltm_path: Optional[str] = None
        self.frames_path: Optional[str] = None

        # parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._submit_index: int = 0                # index assigned when frame is queued
        self._next_index: int = 0                  # next index expected to flush in order
        self._pending_results: Dict[int, Frame] = {}  # completed frames awaiting order
        self._max_workers = max_workers

    # ---------- session management -----------------------------------------

    def start_session(self, world_name: str):
        """
        Start / resume a world session; load memory from disk if present.
        """
        if self.current_world_name == world_name:
            logger.debug(f"Resuming session for world: {world_name}")
            return

        self.end_session()  # clear previous state
        self.executor = ThreadPoolExecutor(max_workers=self._max_workers)
        self.current_world_name = world_name

        # reset ordering counters for the new session
        self._submit_index = 0
        self._next_index = 0
        self._pending_results.clear()

        # prepare paths and load memory from disk
        self.ltm_path = os.path.join(LONG_TERM_MEMORY_BASE_PATH, world_name)
        self.frames_path = os.path.join(self.ltm_path, "frames")
        self._ensure_ltm_path_exists()
        self._load_memory_from_disk()

        # align indices with loaded memory
        self._submit_index = len(self.memory)
        self._next_index = len(self.memory)
        logger.debug(
            f"Session started for world: {world_name}. "
            f"Loaded {len(self.memory)} frames from disk."
        )

    def end_session(self):
        """
        Clear all in-memory data and shut down background workers.
        Safe to call multiple times.
        """
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=False)
            self.executor = None

        # reset runtime state
        self.memory.clear()
        self._pending_results.clear()
        self.current_world_name = None
        self.ltm_path = None
        self.frames_path = None
        self._submit_index = 0
        self._next_index = 0
        logger.debug("Session ended. All memory cleared.")

    # ---------- public API --------------------------------------------------

    def add_frame(self, frame: Frame):
        """
        Add frame to memory immediately and queue asynchronous processing.
        Disk persistence remains in submission order.
        """
        if self.current_world_name is None:
            self.start_session(frame.world_name)

        logger.debug("Adding frame to memory: %s", frame.world_name)
        # store in memory right away
        idx = self._submit_index
        self._submit_index += 1
        self.memory.append(frame)

        # queue async enhancement
        self.executor.submit(self._enhance_and_collect_frame, idx, frame)

        logger.debug(
            f"Queued frame {frame.timestamp} (idx {idx}) for async enhancement."
        )

    def get_memory(self) -> List[Frame]:
        """Return all frames (may lag behind if enhancements still running)."""
        return self.memory

    # ---------- async processing helpers -----------------------------------

    def _enhance_and_collect_frame(self, index: int, frame: Frame):
        """
        Background worker: process frame, then flush results in order.
        """
        enhanced = self._compute_clip_for_frame(frame)

        with self._lock:
            # store completed result
            self._pending_results[index] = enhanced

            # flush in-order frames to disk
            while self._next_index in self._pending_results:
                ready_frame = self._pending_results.pop(self._next_index)
                if self._next_index < len(self.memory):
                    self.memory[self._next_index] = ready_frame
                else:
                    self.memory.append(ready_frame)
                self._save_frame_to_disk(ready_frame)
                logger.debug(
                    f"Flushed frame idx {self._next_index} "
                    f"(ts {ready_frame.timestamp}) to disk."
                )
                self._next_index += 1

    # ---------- disk I/O ----------------------------------------------------

    def _ensure_ltm_path_exists(self):
        """Create directory for current world if needed."""
        if self.frames_path:
            os.makedirs(self.frames_path, exist_ok=True)

    def _save_frame_to_disk(self, frame: Frame):
        """Compress and save frame object to disk."""
        if not self.frames_path:
            logger.debug("Cannot save frame: frames path not set.")
            return

        ts_str = frame.timestamp.strftime("%Y%m%d_%H%M%S_%f")
        file_path = os.path.join(self.frames_path, f"{ts_str}.frame")
        compressed = compress_object(frame)
        with open(file_path, "wb") as f:
            f.write(compressed)

    def _load_memory_from_disk(self):
        """Load frames from disk on session start."""
        if not self.frames_path or not os.path.exists(self.frames_path):
            return

        files = sorted(
            f for f in os.listdir(self.frames_path) if f.endswith(".frame")
        )
        loaded_frames: List[Frame] = []
        for fname in files:
            with open(os.path.join(self.frames_path, fname), "rb") as f:
                frame = decompress_object(f.read())
            if isinstance(frame, Frame):
                loaded_frames.append(frame)
            else:
                logger.debug(
                    f"Skipping {fname}: not a valid Frame object."
                )

        if not loaded_frames:
            return

        if LTM_LOAD_LATEST_CAPTURE_ONLY and len(loaded_frames) > 1:
            gap_seconds = max(0.0, float(LTM_CAPTURE_GAP_SECONDS))
            start_idx = 0
            for idx in range(len(loaded_frames) - 1, 0, -1):
                prev_ts = getattr(loaded_frames[idx - 1], "timestamp", None)
                cur_ts = getattr(loaded_frames[idx], "timestamp", None)
                if prev_ts is None or cur_ts is None:
                    continue
                gap = (cur_ts - prev_ts).total_seconds()
                if gap > gap_seconds:
                    start_idx = idx
                    break
            if start_idx > 0:
                dropped = start_idx
                loaded_frames = loaded_frames[start_idx:]
                logger.debug(
                    "LTM latest-capture load enabled: kept %d frames, dropped %d older frames (gap threshold=%.1fs).",
                    len(loaded_frames),
                    dropped,
                    gap_seconds,
                )

        self.memory.extend(loaded_frames)

    # ---------- frame processing -------------------------------------------

    def _compute_clip_for_frame(self, frame: Frame) -> Frame:
        return frame
