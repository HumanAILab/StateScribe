import logging
import threading
from collections import deque
from time import perf_counter
from typing import Callable, Deque, Optional, Union

from components.speech_timing import estimate_tts_duration_sec
from config import (
    BROKER_MESSAGE_GAP_SECONDS,
    DEFAULT_TTS_RATE,
    DEFAULT_TTS_WPM_AT_HALF,
)

logger = logging.getLogger(__name__)
speech_logger = logging.getLogger("statescribe.speech")

DescriptionPayload = Union[str, Callable[[], str]]


class DescriptionBroker:
    accepts_deferred_payload = True

    def __init__(
        self,
        emit_func: Callable[[str], None],
        tts_rate: float = DEFAULT_TTS_RATE,
        tts_wpm_at_half: float = DEFAULT_TTS_WPM_AT_HALF,
        message_gap_seconds: float = BROKER_MESSAGE_GAP_SECONDS,
    ):
        self.emit_func = emit_func
        self.tts_rate = float(tts_rate)
        self.tts_wpm_at_half = float(tts_wpm_at_half)
        self.message_gap_seconds = max(0.0, float(message_gap_seconds))

        self._lock = threading.Lock()
        self._wake_event = threading.Event()
        self._change_queue: Deque[DescriptionPayload] = deque()
        self._agent_queue: Deque[DescriptionPayload] = deque()
        self._agent_waiting = False
        self._busy_until = 0.0
        self._running = True
        self._worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="description-broker",
        )
        self._worker.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False
        self._wake_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)

    def begin_agent_response(self) -> None:
        with self._lock:
            self._agent_waiting = True
        self._wake_event.set()

    def cancel_agent_response(self) -> None:
        with self._lock:
            self._agent_waiting = False
        self._wake_event.set()

    def end_agent_response(self) -> None:
        with self._lock:
            self._agent_waiting = False
        self._wake_event.set()

    def get_busy_until(self) -> float:
        with self._lock:
            return float(self._busy_until)

    def project_release_time(self, text: str) -> float:
        normalized = self._normalize_text(text)
        now = perf_counter()
        if not normalized:
            return now
        with self._lock:
            available_at = max(now, self._busy_until)
        return available_at + self._estimate_tts_duration_sec(normalized) + self.message_gap_seconds

    def publish_agent(self, payload: DescriptionPayload) -> bool:
        if self._is_empty_payload(payload):
            return False
        with self._lock:
            self._agent_waiting = False
            self._agent_queue.append(payload)
        self._wake_event.set()
        return True

    def publish_change(self, payload: DescriptionPayload) -> bool:
        if self._is_empty_payload(payload):
            return False
        with self._lock:
            self._change_queue.append(payload)
        self._wake_event.set()
        return True

    def publish_live(self, text: str) -> bool:
        normalized = self._normalize_text(text)
        if not normalized:
            return False

        with self._lock:
            now = perf_counter()
            if self._agent_waiting:
                return False
            if self._agent_queue or self._change_queue:
                return False
            if now < self._busy_until:
                return False
            self._reserve_speech_window_locked(normalized, now, source="live")

        if self._emit_text(normalized, source="live"):
            return True
        self._clear_busy_if_idle()
        return False

    @staticmethod
    def _is_empty_payload(payload: DescriptionPayload) -> bool:
        if callable(payload):
            return False
        return not (payload or "").strip()

    @staticmethod
    def _normalize_text(text: str) -> str:
        clean = (text or "").strip()
        if not clean:
            return ""
        return " ".join(clean.split())

    def _resolve_payload(self, payload: DescriptionPayload) -> str:
        try:
            value = payload() if callable(payload) else payload
        except Exception:
            logger.exception("Failed to resolve description payload.")
            return ""
        return self._normalize_text(value)

    def _estimate_tts_duration_sec(self, text: str) -> float:
        return estimate_tts_duration_sec(
            text,
            rate=self.tts_rate,
            wpm_at_half=self.tts_wpm_at_half,
        )

    def _reserve_speech_window_locked(self, text: str, now: float, source: str) -> None:
        tts_sec = self._estimate_tts_duration_sec(text)
        speech_logger.debug(f"[tts_est] source={source} sec={tts_sec:.2f} text={text}")
        self._busy_until = max(self._busy_until, now + tts_sec + self.message_gap_seconds)

    def _clear_busy_if_idle(self) -> None:
        now = perf_counter()
        with self._lock:
            if self._agent_queue:
                return
            if self._change_queue:
                return
            self._busy_until = min(self._busy_until, now)

    def _emit_text(self, text: str, source: str) -> bool:
        started = perf_counter()
        try:
            self.emit_func(text)
            elapsed = perf_counter() - started
            speech_logger.debug(f"[broker_emit] source={source} sec={elapsed:.3f} text={text}")
            return True
        except Exception:
            logger.exception("Failed to emit description text.")
            return False

    def _worker_loop(self) -> None:
        while True:
            source: Optional[str] = None
            payload: Optional[DescriptionPayload] = None
            wait_timeout = 0.2

            with self._lock:
                if not self._running:
                    return

                now = perf_counter()
                if now < self._busy_until:
                    wait_timeout = min(0.2, max(0.01, self._busy_until - now))
                elif self._agent_queue:
                    source = "agent"
                    payload = self._agent_queue.popleft()
                elif self._agent_waiting:
                    wait_timeout = 0.2
                elif self._change_queue:
                    source = "change"
                    payload = self._change_queue.popleft()

            if payload is None:
                self._wake_event.wait(timeout=wait_timeout)
                self._wake_event.clear()
                continue

            text = self._resolve_payload(payload)
            if not text:
                continue

            should_wait = False
            with self._lock:
                if not self._running:
                    return

                now = perf_counter()
                blocked_by_busy = now < self._busy_until
                blocked_by_agent = source != "agent" and self._agent_waiting
                if blocked_by_busy or blocked_by_agent:
                    should_wait = True
                    if source == "agent":
                        self._agent_queue.appendleft(payload)
                    else:
                        self._change_queue.appendleft(payload)
                else:
                    self._reserve_speech_window_locked(text, now, source=source or "unknown")

            if should_wait:
                self._wake_event.wait(timeout=0.05)
                self._wake_event.clear()
                continue

            if not self._emit_text(text, source=source or "unknown"):
                self._clear_busy_if_idle()
