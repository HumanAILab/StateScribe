import json
import logging
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from collections import deque
from datetime import datetime
from time import perf_counter
from typing import Any, Callable, Deque, Dict, List, Optional

from google import genai
from google.genai import types

from components.description_composer import ChangeDescriptionComposer
from components.memory.change_memory import ChangeMemory
from components.memory.memory_manager import MemoryManager
from config import (
    AI_PARAPHRASE_API_LEAD_SECONDS,
    AI_PARAPHRASE_DESCRIPTION_ENABLED,
    AI_PARAPHRASE_DESCRIPTION_MAX_TOKENS,
    AI_PARAPHRASE_DESCRIPTION_MODEL,
    AI_PARAPHRASE_DESCRIPTION_TEMPERATURE,
    AI_PARAPHRASE_MAX_CHANGES_PER_SUMMARY,
    AI_PARAPHRASE_PREVIOUS_OUTPUT_COUNT,
    GEMINI_API_KEY,
    GEMINI_API_TIMEOUT_MS,
    MAX_CONCURRENT_AI_SUMMARY,
)
from prompts import (
    AI_PARAPHRASE_OUTPUT_SCHEMA,
    AI_PARAPHRASE_SYSTEM_PROMPT,
    AI_PARAPHRASE_USER_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)


class AIParaphraseDescriptionPipeline(threading.Thread):
    def __init__(
        self,
        memory_manager: MemoryManager,
        change_memory: ChangeMemory,
        on_description: Optional[Callable[[Any], Any]] = None,
        speech_busy_until_getter: Optional[Callable[[], float]] = None,
        previous_output_count: int = AI_PARAPHRASE_PREVIOUS_OUTPUT_COUNT,
        max_changes_per_summary: int = AI_PARAPHRASE_MAX_CHANGES_PER_SUMMARY,
    ):
        super().__init__(daemon=True)
        self.memory_manager = memory_manager
        self.change_memory = change_memory
        self.on_description = on_description
        self.speech_busy_until_getter = speech_busy_until_getter
        self.previous_output_count = max(0, int(previous_output_count))
        self.max_changes_per_summary = max(1, int(max_changes_per_summary))
        self.api_lead_seconds = max(0.0, float(AI_PARAPHRASE_API_LEAD_SECONDS))

        self.enabled = bool(
            AI_PARAPHRASE_DESCRIPTION_ENABLED
            and GEMINI_API_KEY
            and self.on_description is not None
        )
        self.client = genai.Client(api_key=GEMINI_API_KEY, http_options={"timeout": GEMINI_API_TIMEOUT_MS}) if self.enabled else None
        self.description_composer = ChangeDescriptionComposer(
            change_memory,
            latest_pose_matrix_getter=self._latest_pose_matrix,
        )

        history_cap = max(8, self.previous_output_count * 4)
        self._recent_outputs: Deque[str] = deque(maxlen=history_cap)

        self._pending_live: Optional[Dict[str, Any]] = None
        self._pending_change_snapshots: Deque[Dict[str, Any]] = deque()
        self._buffer_started_at: Optional[float] = None

        self._lock = threading.Lock()
        self._wake_event = threading.Event()
        self.running = False
        self._paused = False
        self._summary_submit_seq = 0
        self._latest_completed_summary_seq = 0
        self._summary_executor: Optional[ThreadPoolExecutor] = None

        if self.enabled:
            self._summary_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_AI_SUMMARY)

        if not AI_PARAPHRASE_DESCRIPTION_ENABLED:
            logger.debug("AIParaphraseDescriptionPipeline disabled by config.")
        elif not GEMINI_API_KEY:
            logger.debug("AIParaphraseDescriptionPipeline disabled: GEMINI_API_KEY not set.")
        elif self.on_description is None:
            logger.debug("AIParaphraseDescriptionPipeline disabled: on_description callback is missing.")
        else:
            logger.debug("AIParaphraseDescriptionPipeline initialized with model=%s", AI_PARAPHRASE_DESCRIPTION_MODEL)

    def stop(self) -> None:
        self.running = False
        self._wake_event.set()

    def set_paused(self, paused: bool) -> None:
        next_state = bool(paused)
        if self._paused == next_state:
            return
        self._paused = next_state
        logger.debug("AIParaphraseDescriptionPipeline %s for VQA", "paused" if next_state else "resumed")
        self._wake_event.set()

    def add_live_description(self, text: str) -> bool:
        if not self.enabled or self._paused:
            return False
        normalized = self._normalize_text(text)
        if not normalized:
            return False
        now_iso = datetime.now().isoformat()
        with self._lock:
            self._pending_live = {
                "text": normalized,
                "timestamp": now_iso,
            }
            if self._buffer_started_at is None:
                self._buffer_started_at = perf_counter()
        self._wake_event.set()
        return True

    def add_change_snapshot(self, snapshot: Dict[str, Any]) -> bool:
        if not self.enabled or self._paused:
            return False
        normalized = self._normalize_snapshot(snapshot)
        if normalized is None:
            return False
        split_snapshots = self._split_snapshot_changes(normalized)
        if not split_snapshots:
            return False
        with self._lock:
            self._pending_change_snapshots.extend(split_snapshots)
            if self._buffer_started_at is None:
                self._buffer_started_at = perf_counter()
        self._wake_event.set()
        return True

    def run(self) -> None:
        if not self.enabled:
            return

        self.running = True
        logger.debug("AIParaphraseDescriptionPipeline started.")
        pending_jobs: Dict[Future, int] = {}

        try:
            while self.running:
                if self._paused:
                    self._wake_event.wait(timeout=0.1)
                    self._wake_event.clear()
                    continue
                payload: Optional[Dict[str, Any]] = None
                wait_timeout = 0.2

                with self._lock:
                    if not pending_jobs and self._has_pending_locked():
                        now = perf_counter()
                        started_at = self._buffer_started_at if self._buffer_started_at is not None else now
                        due_at = started_at + self._buffer_window_seconds_locked()
                        remaining = due_at - now
                        if remaining <= 0.0:
                            payload = self._drain_payload_locked()
                        else:
                            wait_timeout = min(0.2, max(0.01, remaining))

                if payload is not None and self._summary_executor is not None:
                    with self._lock:
                        self._summary_submit_seq += 1
                        seq = self._summary_submit_seq
                    future = self._summary_executor.submit(self._generate_summary, payload)
                    pending_jobs[future] = seq

                handled = self._collect_summary_results(pending_jobs)
                if handled:
                    continue

                if pending_jobs:
                    wait_timeout = min(wait_timeout, 0.05)

                self._wake_event.wait(timeout=wait_timeout)
                self._wake_event.clear()
        finally:
            if self._summary_executor is not None:
                self._summary_executor.shutdown(wait=False)
                self._summary_executor = None
            logger.debug("AIParaphraseDescriptionPipeline stopped.")

    def _collect_summary_results(self, pending_jobs: Dict[Future, int]) -> bool:
        if not pending_jobs:
            return False
        done, _ = wait(tuple(pending_jobs.keys()), timeout=0.0, return_when=FIRST_COMPLETED)
        if not done:
            return False

        completed: List[tuple[int, str]] = []
        for future in done:
            seq = pending_jobs.pop(future, None)
            if seq is None:
                continue
            try:
                summary = future.result()
            except Exception:
                logger.exception("Failed to resolve AI paraphrase summary future.")
                continue

            completed.append((int(seq), summary or ""))

        completed.sort(key=lambda item: item[0], reverse=True)
        for seq, summary in completed:
            if seq < self._latest_completed_summary_seq:
                logger.debug(
                    "Dropping stale AI summary result: seq=%s latest_completed=%s",
                    seq,
                    self._latest_completed_summary_seq,
                )
                continue

            self._latest_completed_summary_seq = max(self._latest_completed_summary_seq, seq)
            if summary:
                self._publish_summary(summary)
        return True

    def _has_pending_locked(self) -> bool:
        return self._pending_live is not None or bool(self._pending_change_snapshots)

    def _drain_payload_locked(self) -> Dict[str, Any]:
        change_snapshots: List[Dict[str, Any]] = []
        while self._pending_change_snapshots and len(change_snapshots) < self.max_changes_per_summary:
            change_snapshots.append(self._pending_change_snapshots.popleft())

        payload = {
            "latest_live_description": (self._pending_live or {}).get("text", ""),
            "live_timestamp": (self._pending_live or {}).get("timestamp", ""),
            "current_time": datetime.now().isoformat(),
            "change_snapshots": change_snapshots,
            "previous_outputs": self._context_tail(list(self._recent_outputs)),
        }
        self._pending_live = None
        self._buffer_started_at = perf_counter() if self._pending_change_snapshots else None
        return payload

    def _context_tail(self, items: List[str]) -> List[str]:
        if self.previous_output_count <= 0:
            return []
        return items[-self.previous_output_count:]

    def _buffer_window_seconds_locked(self) -> float:
        started = self._buffer_started_at if self._buffer_started_at is not None else perf_counter()
        fire_at = self._current_speech_busy_until() - self.api_lead_seconds
        return max(0.0, fire_at - started)

    def _generate_summary(self, payload: Dict[str, Any]) -> str:
        summary = ""
        if self.client is not None:
            try:
                logger.debug(f"Generating AI paraphrase summary with payload: {payload}")

                prompt = AI_PARAPHRASE_USER_PROMPT_TEMPLATE.format(
                    buffer_json=json.dumps(payload, ensure_ascii=True)
                )
                response = self.client.models.generate_content(
                    model=AI_PARAPHRASE_DESCRIPTION_MODEL,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        temperature=AI_PARAPHRASE_DESCRIPTION_TEMPERATURE,
                        max_output_tokens=AI_PARAPHRASE_DESCRIPTION_MAX_TOKENS,
                        system_instruction=AI_PARAPHRASE_SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        response_schema=AI_PARAPHRASE_OUTPUT_SCHEMA,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                parsed = json.loads(response.text or "{}")
                summary = self._normalize_text(parsed.get("summary", ""))
            except Exception:
                logger.exception("Failed to generate AI paraphrase summary.")

        if not summary:
            summary = self._fallback_summary(payload)
        if not summary:
            return ""

        summary_cmp = self._comparison_text(summary)
        if not summary_cmp:
            return ""

        previous_cmp = {
            self._comparison_text(item)
            for item in payload.get("previous_outputs", [])
            if item
        }
        if summary_cmp in previous_cmp:
            return ""

        return summary

    def _publish_summary(self, summary: str) -> None:
        if self.on_description is None:
            return

        normalized = self._normalize_text(summary)
        if not normalized:
            return

        def _deferred_payload(raw_text: str = normalized) -> str:
            rendered = self._render_for_emit(raw_text)
            return rendered

        callback_target = getattr(self.on_description, "__self__", None)
        supports_deferred = getattr(callback_target, "accepts_deferred_payload", False)
        payload: Any = _deferred_payload if supports_deferred else _deferred_payload()

        accepted = self.on_description(payload)
        if isinstance(accepted, bool) and not accepted:
            return

        with self._lock:
            self._recent_outputs.append(normalized)

    def _render_for_emit(self, text: str) -> str:
        resolved = self.description_composer.resolve_location_tokens(text)
        normalized = self._normalize_text(resolved)
        if not normalized:
            return ""
        if normalized[-1] not in ".!?":
            return f"{normalized}."
        return normalized

    def _fallback_summary(self, payload: Dict[str, Any]) -> str:
        snapshots = payload.get("change_snapshots", [])
        if snapshots:
            latest_changes = snapshots[-1].get("changes", []) if isinstance(snapshots[-1], dict) else []
            if latest_changes:
                change = latest_changes[-1]
                obj = self._normalize_text(str(change.get("object_description", "")))
                desc = self._normalize_text(str(change.get("change_description", "")))
                context = self._normalize_text(str(change.get("context_description", "")))
                object_id = self._normalize_text(str(change.get("object_id", "")))
                pieces = []
                if obj:
                    pieces.append(obj)
                if desc:
                    pieces.append(desc)
                if context:
                    pieces.append(f"around {context}")
                fallback = " ".join(pieces).strip()
                if fallback and object_id:
                    fallback = f"{fallback} [[DIR:{object_id}]] [[DIS:{object_id}]]"
                if fallback:
                    return fallback

        live_text = self._normalize_text(payload.get("latest_live_description", ""))
        return live_text

    @staticmethod
    def _normalize_bbox_3d(raw: Any) -> Optional[tuple[float, float, float, float, float, float]]:
        if not isinstance(raw, (list, tuple)) or len(raw) != 6:
            return None
        try:
            x1, y1, z1, x2, y2, z2 = [float(v) for v in raw]
        except Exception:
            return None
        if x2 <= x1 or y2 <= y1 or z2 <= z1:
            return None
        return (x1, y1, z1, x2, y2, z2)

    @staticmethod
    def _bbox_overlap_3d(
        a: tuple[float, float, float, float, float, float],
        b: tuple[float, float, float, float, float, float],
    ) -> bool:
        ax1, ay1, az1, ax2, ay2, az2 = a
        bx1, by1, bz1, bx2, by2, bz2 = b
        return (
            max(ax1, bx1) < min(ax2, bx2)
            and max(ay1, by1) < min(ay2, by2)
            and max(az1, bz1) < min(az2, bz2)
        )

    @classmethod
    def _changes_overlap_by_bbox(cls, change_a: Dict[str, Any], change_b: Dict[str, Any]) -> bool:
        boxes3d_a = [
            cls._normalize_bbox_3d(change_a.get("bbox_3d_t0")),
            cls._normalize_bbox_3d(change_a.get("bbox_3d_t1")),
        ]
        boxes3d_b = [
            cls._normalize_bbox_3d(change_b.get("bbox_3d_t0")),
            cls._normalize_bbox_3d(change_b.get("bbox_3d_t1")),
        ]
        for box_a in boxes3d_a:
            if box_a is None:
                continue
            for box_b in boxes3d_b:
                if box_b is None:
                    continue
                if cls._bbox_overlap_3d(box_a, box_b):
                    return True
        return False

    @classmethod
    def _split_snapshot_changes(cls, snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        changes = snapshot.get("changes", [])
        if not isinstance(changes, list):
            return []
        normalized_changes = [change for change in changes if isinstance(change, dict)]
        if not normalized_changes:
            return []

        base = {k: v for k, v in snapshot.items() if k != "changes"}
        count = len(normalized_changes)
        neighbors: Dict[int, List[int]] = {idx: [] for idx in range(count)}
        for i in range(count):
            for j in range(i + 1, count):
                if cls._changes_overlap_by_bbox(normalized_changes[i], normalized_changes[j]):
                    neighbors[i].append(j)
                    neighbors[j].append(i)

        groups: List[List[int]] = []
        visited: set[int] = set()
        for start in range(count):
            if start in visited:
                continue
            stack = [start]
            visited.add(start)
            group: List[int] = []
            while stack:
                current = stack.pop()
                group.append(current)
                for nxt in neighbors[current]:
                    if nxt in visited:
                        continue
                    visited.add(nxt)
                    stack.append(nxt)
            groups.append(sorted(group))

        groups.sort(key=lambda group: group[0])
        split: List[Dict[str, Any]] = []
        for group in groups:
            row = dict(base)
            row["changes"] = [normalized_changes[idx] for idx in group]
            split.append(row)
        return split

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join((text or "").split()).strip()

    def _comparison_text(self, text: str) -> str:
        no_tokens = ChangeDescriptionComposer.LOCATION_TOKEN_PATTERN.sub("", text or "")
        return self._normalize_text(no_tokens).lower()

    def _normalize_snapshot(self, snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(snapshot, dict):
            return None
        cleaned = self._to_json_compatible(snapshot)
        if not isinstance(cleaned, dict):
            return None
        changes = cleaned.get("changes", [])
        if not isinstance(changes, list) or not changes:
            return None
        return cleaned

    def _latest_pose_matrix(self):
        frames = self.memory_manager.get_memory()
        if not frames:
            return None
        latest = frames[-1]
        return getattr(latest, "pose_matrix", None)

    def _current_speech_busy_until(self) -> float:
        if self.speech_busy_until_getter is None:
            return 0.0
        try:
            return float(self.speech_busy_until_getter())
        except Exception:
            logger.exception("Failed to read speech busy window from broker.")
            return 0.0

    def _to_json_compatible(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return {
                str(k): self._to_json_compatible(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple, deque)):
            return [self._to_json_compatible(v) for v in value]

        to_list = getattr(value, "tolist", None)
        if callable(to_list):
            try:
                return self._to_json_compatible(to_list())
            except Exception:
                pass

        return str(value)
