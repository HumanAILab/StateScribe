from __future__ import annotations

from datetime import datetime, timedelta
from threading import Lock
from time import perf_counter
from typing import Optional


class BenchmarkClock:
    def __init__(self, mode: str, start_timestamp: datetime) -> None:
        self.mode = mode
        self.start_timestamp = start_timestamp
        self._run_start_perf: Optional[float] = None
        self._latest_frame_timestamp: datetime = start_timestamp
        self._anchor_perf: Optional[float] = None
        self._anchor_timestamp: datetime = start_timestamp
        self._lock = Lock()

    def start(self, run_start_perf: Optional[float] = None) -> float:
        value = perf_counter() if run_start_perf is None else float(run_start_perf)
        with self._lock:
            self._run_start_perf = value
            self._anchor_perf = value
            self._anchor_timestamp = self.start_timestamp
        return value

    def set_latest_frame_timestamp(self, timestamp: Optional[datetime]) -> None:
        if timestamp is None:
            return
        with self._lock:
            if timestamp >= self._latest_frame_timestamp:
                self._latest_frame_timestamp = timestamp
                now_perf = perf_counter()
                self._anchor_perf = now_perf
                self._anchor_timestamp = timestamp

    def now(self) -> datetime:
        with self._lock:
            latest_frame_timestamp = self._latest_frame_timestamp
            anchor_perf = self._anchor_perf
            anchor_timestamp = self._anchor_timestamp
            mode = self.mode

        if mode == "realtime" and anchor_perf is not None:
            elapsed = max(0.0, perf_counter() - anchor_perf)
            return anchor_timestamp + timedelta(seconds=elapsed)
        return latest_frame_timestamp
