from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import re
from time import perf_counter, sleep
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from google import genai
from google.genai import types
from google.genai import _transformers as genai_transformers
from google.genai import errors as genai_errors

from benchmark.config import (
    DEFAULT_FAST_POLL_INTERVAL_SEC,
    DEFAULT_FRAME_STRIDE,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REALTIME_POLL_INTERVAL_SEC,
    DEFAULT_USE_ALIGNED_METADATA,
)
from benchmark.dataset_loader import CaptureFrameRecord, load_dataset_records
from benchmark.recorder import BenchmarkOutputRecorder
from benchmark.baselines.online_prompts import (
    ONLINE_BASELINE_EVAL_TURN_INTRO,
    ONLINE_BASELINE_RESPONSE_SCHEMA,
    ONLINE_BASELINE_WARMUP_TURN_INTRO,
)
from config import GEMINI_API_KEY

LOGGER = logging.getLogger("benchmark.baselines.online")

DEFAULT_ONLINE_GEMINI_MODEL = "gemini-3-flash-preview"
DEFAULT_ONLINE_GEMINI_TIMEOUT_MS = 60000
DEFAULT_ONLINE_MIN_FRAMES_PER_REQUEST = 5
DEFAULT_ONLINE_MAX_FRAMES_PER_REQUEST = 10
DEFAULT_ONLINE_HISTORY_TURN_LIMIT = 6


@dataclass(frozen=True)
class OnlineBaselineChange:
    evidence_frame_id: str
    change_type: str
    object_description: str
    change_description: str
    context_description: str
    clock_direction: int
    distance_feet: float


@dataclass(frozen=True)
class OnlineBaselineTurnResult:
    raw_response_text: str
    changes: List[OnlineBaselineChange]


@dataclass(frozen=True)
class CompletedTurn:
    result: OnlineBaselineTurnResult
    request_start_wall: datetime
    request_end_wall: datetime
    request_start_perf: float
    request_end_perf: float


@dataclass
class PendingTurn:
    future: Future
    frames: List[CaptureFrameRecord]
    frame_id_to_record: Dict[str, CaptureFrameRecord]
    record_outputs: bool


class OnlineBaselineBackend(Protocol):
    method_slug: str
    provider_name: str
    model_name: str
    description_mode: str

    def start_session(self, session_name: str) -> None:
        ...

    def preload_frames(self, frames: Sequence[CaptureFrameRecord]) -> None:
        ...

    def process_frames(
        self,
        frames: Sequence[CaptureFrameRecord],
        *,
        warmup: bool,
    ) -> OnlineBaselineTurnResult:
        ...

    def close(self) -> None:
        ...


class GeminiOnlineBaselineBackend:
    method_slug = "baseline_online_gemini_flash"
    provider_name = "gemini"
    description_mode = "baseline_online_gemini"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_ONLINE_GEMINI_MODEL,
        timeout_ms: int = DEFAULT_ONLINE_GEMINI_TIMEOUT_MS,
    ) -> None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set.")
        self.model_name = str(model_name or DEFAULT_ONLINE_GEMINI_MODEL)
        self._timeout_ms = max(1, int(timeout_ms))
        self._client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"timeout": self._timeout_ms},
        )
        self._history_turns: Optional[List[Tuple[Any, Any]]] = None
        self._uploaded_files: Dict[str, Any] = {}

    def start_session(self, session_name: str) -> None:
        del session_name
        self._history_turns = []

    def preload_frames(self, frames: Sequence[CaptureFrameRecord]) -> None:
        seen_paths = set()
        for record in frames:
            rgb_path = record.rgb_path
            key = str(rgb_path.resolve())
            if key in seen_paths:
                continue
            seen_paths.add(key)
            self._upload_frame(rgb_path)

    def process_frames(
        self,
        frames: Sequence[CaptureFrameRecord],
        *,
        warmup: bool,
    ) -> OnlineBaselineTurnResult:
        if self._history_turns is None:
            raise RuntimeError("Gemini online baseline session has not been started.")
        if not frames:
            return OnlineBaselineTurnResult(raw_response_text="", changes=[])

        message = self._build_turn_message(frames, warmup=warmup)
        input_content = genai_transformers.t_content(message)
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=self._request_contents(input_content),
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=ONLINE_BASELINE_RESPONSE_SCHEMA,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        self._record_history(input_content, response)
        raw_text = str(response.text or "").strip()
        payload = _safe_json_loads(raw_text)
        return OnlineBaselineTurnResult(
            raw_response_text=raw_text,
            changes=_parse_online_changes(payload),
        )

    def close(self) -> None:
        self._uploaded_files.clear()
        self._history_turns = None

    def _build_turn_message(
        self,
        frames: Sequence[CaptureFrameRecord],
        *,
        warmup: bool,
    ) -> List[Any]:
        intro = ONLINE_BASELINE_WARMUP_TURN_INTRO if warmup else ONLINE_BASELINE_EVAL_TURN_INTRO
        parts: List[Any] = [intro]
        parts.append("Frames in this turn:")
        for offset, record in enumerate(frames, start=1):
            frame_id = _frame_id(offset)
            parts.append(
                (
                    f"{frame_id}: capture={record.capture_name}, "
                    f"frame_index={record.frame_index}, "
                    f"timestamp={record.timestamp.isoformat()}"
                )
            )
            parts.append(self._upload_frame(record.rgb_path))
        return parts

    def _upload_frame(self, rgb_path: Path) -> Any:
        key = str(rgb_path.resolve())
        cached = self._uploaded_files.get(key)
        if cached is not None:
            return cached
        for attempt in range(5):
            try:
                uploaded = self._client.files.upload(file=str(rgb_path))
                self._uploaded_files[key] = uploaded
                return uploaded
            except genai_errors.ServerError:
                if attempt == 2:
                    raise
                LOGGER.warning("Retrying frame upload after ServerError: %s", rgb_path)
                sleep(1.0)

    def _request_contents(self, input_content: Any) -> List[Any]:
        contents: List[Any] = []
        if self._history_turns:
            for user_content, model_content in self._history_turns:
                contents.append(user_content)
                contents.append(model_content)
        contents.append(input_content)
        return contents

    def _record_history(self, input_content: Any, response: Any) -> None:
        if self._history_turns is None:
            return
        model_content = _first_model_content(response)
        if model_content is None or not _is_valid_content(model_content):
            return
        self._history_turns.append((input_content, model_content))
        if len(self._history_turns) > DEFAULT_ONLINE_HISTORY_TURN_LIMIT:
            self._history_turns = self._history_turns[-DEFAULT_ONLINE_HISTORY_TURN_LIMIT :]


class OnlineBaselineRunner:
    def __init__(
        self,
        *,
        backend: OnlineBaselineBackend,
        dataset_dir: Path,
        frame_stride: int = DEFAULT_FRAME_STRIDE,
        max_captures: Optional[int] = None,
        output_root: Path = DEFAULT_OUTPUT_ROOT,
        use_aligned_metadata: bool = DEFAULT_USE_ALIGNED_METADATA,
        require_confidence: bool = False,
        world_name_prefix: str = "baseline",
    ) -> None:
        self.backend = backend
        self.dataset_dir = dataset_dir
        self.frame_stride = max(1, int(frame_stride))
        self.max_captures = int(max_captures) if max_captures is not None and max_captures > 0 else None
        self.output_root = output_root
        self.use_aligned_metadata = bool(use_aligned_metadata)
        self.require_confidence = bool(require_confidence)

        self.records: List[CaptureFrameRecord] = load_dataset_records(
            dataset_dir=self.dataset_dir,
            frame_stride=self.frame_stride,
            use_aligned_metadata=self.use_aligned_metadata,
            require_confidence=self.require_confidence,
            max_captures=self.max_captures,
        )
        if not self.records:
            raise ValueError("No valid frame record found for online baseline run.")

        dataset_slug = _safe_slug(self.dataset_dir.name or "dataset")
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix_slug = _safe_slug(world_name_prefix)
        self.world_name = f"{prefix_slug}_{backend.method_slug}_{dataset_slug}_{run_stamp}"
        self.output_dir = self.output_root / dataset_slug / backend.method_slug / run_stamp

        self.output_recorder = BenchmarkOutputRecorder()

        self.feed_records: List[Dict[str, Any]] = []
        self._feed_row_by_timestamp: Dict[str, Dict[str, Any]] = {}
        self._feed_perf_by_timestamp: Dict[str, Tuple[float, float]] = {}
        self._frame_processing_rows: Dict[str, Dict[str, Any]] = {}
        self.skipped_records: List[Dict[str, Any]] = []

        self._session_history: List[Tuple[str, CaptureFrameRecord]] = []
        self._session_index_by_timestamp: Dict[str, int] = {}
        self._run_start_wall: Optional[datetime] = None
        self._run_start_perf: Optional[float] = None
        self._request_count = 0
        self._warmup_request_count = 0

    def run(self) -> Dict[str, Path]:
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            self.backend.preload_frames(self.records)
            self.backend.start_session(self.world_name)
            self._run_start_wall = datetime.now()
            self._run_start_perf = perf_counter()
            self._session_history = []
            self._session_index_by_timestamp = {}
            self._run_record_batch(
                executor=executor,
                records=self.records,
                record_outputs=True,
                respect_realtime=True,
            )

            return self._write_outputs()
        finally:
            executor.shutdown(wait=False)
            self.backend.close()

    def _run_record_batch(
        self,
        *,
        executor: ThreadPoolExecutor,
        records: Sequence[CaptureFrameRecord],
        record_outputs: bool,
        respect_realtime: bool,
    ) -> None:
        if not records:
            return

        schedule_offsets = _build_schedule_offsets(list(records)) if respect_realtime else [0.0] * len(records)
        batch_start_perf = perf_counter()
        pending_frames: List[CaptureFrameRecord] = []
        next_index = 0
        inflight: Optional[PendingTurn] = None

        while next_index < len(records) or pending_frames or inflight is not None:
            now_perf = perf_counter()
            while next_index < len(records) and self._ready_to_feed(
                now_perf,
                batch_start_perf,
                schedule_offsets,
                next_index,
                respect_realtime=respect_realtime,
            ):
                record = records[next_index]
                pending_frames.append(record)
                self._observe_record(record, record_outputs=record_outputs)
                next_index += 1

            if inflight is not None and inflight.future.done():
                completed = inflight.future.result()
                self._finalize_turn(
                    completed=completed,
                    pending_turn=inflight,
                )
                inflight = None
                continue

            should_submit = (
                len(pending_frames) >= DEFAULT_ONLINE_MIN_FRAMES_PER_REQUEST
                or (next_index >= len(records) and bool(pending_frames))
            )
            if inflight is None and should_submit:
                frames = _sample_turn_frames(
                    pending_frames,
                    max_count=DEFAULT_ONLINE_MAX_FRAMES_PER_REQUEST,
                )
                pending_frames.clear()
                frame_id_to_record = {
                    _frame_id(offset): record
                    for offset, record in enumerate(frames, start=1)
                }
                future = executor.submit(
                    self._run_backend_turn,
                    frames,
                    not record_outputs,
                )
                inflight = PendingTurn(
                    future=future,
                    frames=frames,
                    frame_id_to_record=frame_id_to_record,
                    record_outputs=record_outputs,
                )
                self._request_count += 1
                if not record_outputs:
                    self._warmup_request_count += 1

            poll = DEFAULT_REALTIME_POLL_INTERVAL_SEC if respect_realtime else DEFAULT_FAST_POLL_INTERVAL_SEC
            sleep(poll)

    def _ready_to_feed(
        self,
        now_perf: float,
        batch_start_perf: float,
        schedule_offsets: Sequence[float],
        record_index: int,
        *,
        respect_realtime: bool,
    ) -> bool:
        if not respect_realtime:
            return True
        if record_index >= len(schedule_offsets):
            return True
        return now_perf >= batch_start_perf + float(schedule_offsets[record_index])

    def _observe_record(self, record: CaptureFrameRecord, *, record_outputs: bool) -> None:
        feed_start_wall = datetime.now()
        feed_start_perf = perf_counter()
        feed_end_perf = perf_counter()
        feed_end_wall = datetime.now()
        frame_timestamp = record.timestamp.isoformat()

        self._session_index_by_timestamp[frame_timestamp] = len(self._session_history)
        self._session_history.append((frame_timestamp, record))

        if not record_outputs:
            return

        feed_row = {
            "capture_name": record.capture_name,
            "frame_index": record.frame_index,
            "frame_timestamp": frame_timestamp,
            "feed_start_wall": feed_start_wall.isoformat(),
            "feed_end_wall": feed_end_wall.isoformat(),
            "feed_duration_s": round(max(0.0, feed_end_perf - feed_start_perf), 6),
            "feed_elapsed_s": _elapsed_since_run_start(self._run_start_perf, feed_start_perf),
        }
        self.feed_records.append(feed_row)
        self._feed_row_by_timestamp[frame_timestamp] = feed_row
        self._feed_perf_by_timestamp[frame_timestamp] = (feed_start_perf, feed_end_perf)

    def _run_backend_turn(
        self,
        frames: Sequence[CaptureFrameRecord],
        warmup: bool,
    ) -> CompletedTurn:
        request_start_wall = datetime.now()
        request_start_perf = perf_counter()
        result = self.backend.process_frames(frames, warmup=warmup)
        request_end_perf = perf_counter()
        request_end_wall = datetime.now()
        return CompletedTurn(
            result=result,
            request_start_wall=request_start_wall,
            request_end_wall=request_end_wall,
            request_start_perf=request_start_perf,
            request_end_perf=request_end_perf,
        )

    def _finalize_turn(
        self,
        *,
        completed: CompletedTurn,
        pending_turn: PendingTurn,
    ) -> None:
        request_duration_s = max(0.0, completed.request_end_perf - completed.request_start_perf)
        for record in pending_turn.frames:
            frame_timestamp = record.timestamp.isoformat()
            if not pending_turn.record_outputs:
                continue
            feed_perf = self._feed_perf_by_timestamp.get(frame_timestamp)
            feed_row = self._feed_row_by_timestamp.get(frame_timestamp)
            if feed_perf is None or feed_row is None:
                continue
            feed_start_perf, feed_end_perf = feed_perf
            queue_wait_s = max(0.0, completed.request_start_perf - feed_end_perf)
            total_duration_s = max(0.0, completed.request_end_perf - feed_start_perf)
            previous_ts = self._reference_timestamp_for(frame_timestamp)
            self._frame_processing_rows[frame_timestamp] = {
                "frame_timestamp": frame_timestamp,
                "reference_timestamp": previous_ts if previous_ts != frame_timestamp else "",
                "reference_found": bool(previous_ts and previous_ts != frame_timestamp),
                "total_duration_s": round(total_duration_s, 6),
                "end_to_end_duration_s": round(total_duration_s, 6),
                "processing_duration_s": round(request_duration_s, 6),
                "queue_wait_total_s": round(queue_wait_s, 6),
                "pre_pipeline_duration_s": 0.0,
                "frame_queue_wait_s": round(queue_wait_s, 6),
                "vlm_result_queue_wait_s": 0.0,
                "frame_start_wall": completed.request_start_wall.isoformat(),
                "frame_end_wall": completed.request_end_wall.isoformat(),
                "ingress_wall": feed_row.get("feed_start_wall", ""),
                "pipeline_enqueue_wall": feed_row.get("feed_end_wall", ""),
                "pipeline_dequeue_wall": completed.request_start_wall.isoformat(),
                "stage_timings_ordered": [
                    {"stage": "local_queue_wait", "duration_s": round(queue_wait_s, 6)},
                    {"stage": "vlm_request", "duration_s": round(request_duration_s, 6)},
                ],
            }

        if not pending_turn.record_outputs:
            return

        for change in completed.result.changes:
            detection_record = pending_turn.frame_id_to_record.get(change.evidence_frame_id)
            if detection_record is None:
                detection_record = pending_turn.frames[-1] if pending_turn.frames else None
            if detection_record is None:
                continue
            current_timestamp = detection_record.timestamp.isoformat()
            reference_timestamp = self._reference_timestamp_for(current_timestamp)
            structured_change = {
                "change_type": change.change_type,
                "object_description": change.object_description,
                "change_description": change.change_description,
                "context_description": change.context_description,
                "clock_direction": change.clock_direction,
                "distance_feet": round(float(change.distance_feet), 3),
                "evidence_frame_id": change.evidence_frame_id,
                "capture_name": detection_record.capture_name,
                "frame_index": detection_record.frame_index,
                "frame_timestamp": current_timestamp,
            }
            self.output_recorder.record_prediction(
                text=_render_prediction_text(change),
                description_mode=self.backend.description_mode,
                current_timestamp=current_timestamp,
                reference_timestamp=reference_timestamp,
                changes=[structured_change],
                recorded_from=self.backend.method_slug,
            )

    def _reference_timestamp_for(self, frame_timestamp: str) -> str:
        index = self._session_index_by_timestamp.get(frame_timestamp)
        if index is None or index <= 0:
            return frame_timestamp
        return self._session_history[index - 1][0]

    def _write_outputs(self) -> Dict[str, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        predictions_path = self.output_dir / "predictions.json"
        frames_path = self.output_dir / "frames.jsonl"
        manifest_path = self.output_dir / "manifest.json"

        predictions_payload = self.output_recorder.build_predictions_payload(
            dataset=str(self.dataset_dir),
            world_name=self.world_name,
            mode="online",
            frame_stride=self.frame_stride,
            live_descriptions_enabled=False,
            description_output_mode=self.backend.method_slug,
            run_start_wall=self._run_start_wall,
            feed_records=self.feed_records,
        )
        self.output_recorder.write_json(predictions_path, predictions_payload)

        with frames_path.open("w", encoding="utf-8") as handle:
            for feed_row in self.feed_records:
                key = str(feed_row.get("frame_timestamp", ""))
                processing_row = self._frame_processing_rows.get(key, {})
                output_row = {
                    "capture_name": feed_row.get("capture_name", ""),
                    "frame_index": feed_row.get("frame_index"),
                    "frame_timestamp": key,
                    "feed_start_wall": feed_row.get("feed_start_wall", ""),
                    "feed_end_wall": feed_row.get("feed_end_wall", ""),
                    "feed_duration_s": feed_row.get("feed_duration_s"),
                    "feed_elapsed_s": feed_row.get("feed_elapsed_s"),
                    "processing_recorded": bool(processing_row),
                    "reference_timestamp": processing_row.get("reference_timestamp", ""),
                    "reference_found": bool(processing_row.get("reference_found", False)),
                    "total_duration_s": processing_row.get("total_duration_s"),
                    "end_to_end_duration_s": processing_row.get("end_to_end_duration_s"),
                    "processing_duration_s": processing_row.get("processing_duration_s"),
                    "queue_wait_total_s": processing_row.get("queue_wait_total_s"),
                    "pre_pipeline_duration_s": processing_row.get("pre_pipeline_duration_s"),
                    "frame_queue_wait_s": processing_row.get("frame_queue_wait_s"),
                    "vlm_result_queue_wait_s": processing_row.get("vlm_result_queue_wait_s"),
                    "frame_start_wall": processing_row.get("frame_start_wall", ""),
                    "frame_end_wall": processing_row.get("frame_end_wall", ""),
                    "ingress_wall": processing_row.get("ingress_wall", ""),
                    "pipeline_enqueue_wall": processing_row.get("pipeline_enqueue_wall", ""),
                    "pipeline_dequeue_wall": processing_row.get("pipeline_dequeue_wall", ""),
                    "stage_timings_ordered": processing_row.get("stage_timings_ordered", []),
                }
                handle.write(json.dumps(output_row, ensure_ascii=True))
                handle.write("\n")

        manifest = {
            "dataset": str(self.dataset_dir),
            "mode": "online",
            "runner_kind": "baseline",
            "baseline_kind": "online",
            "baseline_method": self.backend.method_slug,
            "provider": self.backend.provider_name,
            "model": self.backend.model_name,
            "frame_stride": self.frame_stride,
            "max_captures": self.max_captures,
            "live_descriptions_enabled": False,
            "description_output_mode": self.backend.method_slug,
            "latency_time_axis": "wall_elapsed_seconds",
            "world_name": self.world_name,
            "session_scope": "dataset",
            "records_total": len(self.records),
            "frames_fed": len(self.feed_records),
            "feed_records_count": len(self.feed_records),
            "frame_log_count": len(self._frame_processing_rows),
            "spoken_emit_count": 0,
            "prediction_count": len(predictions_payload.get("records", [])),
            "request_count": self._request_count,
            "warmup_request_count": self._warmup_request_count,
            "skipped_records": list(self.skipped_records),
            "run_start_wall": self._run_start_wall.isoformat() if self._run_start_wall is not None else "",
            "simulated_start_time": self.records[0].timestamp.isoformat(),
            "simulated_end_time": self.records[-1].timestamp.isoformat(),
            "pipeline_timeout_reached": False,
            "broker_timeout_reached": False,
        }
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=True, indent=2)

        LOGGER.info("Online baseline output: %s", self.output_dir)
        return {
            "output_dir": self.output_dir,
            "predictions": predictions_path,
            "frames": frames_path,
            "manifest": manifest_path,
        }
def _frame_id(offset: int) -> str:
    return f"frame_{int(offset):03d}"


def _first_model_content(response: Any) -> Optional[Any]:
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return None
    first = candidates[0]
    return getattr(first, "content", None)


def _is_valid_content(content: Any) -> bool:
    parts = getattr(content, "parts", None)
    if not parts:
        return False
    for part in parts:
        if part == types.Part():
            return False
        text = getattr(part, "text", None)
        if text is not None and text == "":
            return False
    return True


def _parse_online_changes(payload: Dict[str, Any]) -> List[OnlineBaselineChange]:
    rows = payload.get("changes")
    if not isinstance(rows, list):
        return []
    parsed: List[OnlineBaselineChange] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        change_type = str(row.get("change_type", "") or "").strip().lower()
        if change_type not in {"appear", "disappear", "change"}:
            continue
        evidence_frame_id = str(row.get("evidence_frame_id", "") or "").strip()
        if not evidence_frame_id:
            continue
        clock = _normalize_clock(row.get("clock_direction"))
        distance_feet = _normalize_distance(row.get("distance_feet"))
        if clock is None or distance_feet is None:
            continue
        parsed.append(
            OnlineBaselineChange(
                evidence_frame_id=evidence_frame_id,
                change_type=change_type,
                object_description=_clean_text(row.get("object_description", "")),
                change_description=_clean_text(row.get("change_description", "")),
                context_description=_clean_text(row.get("context_description", "")),
                clock_direction=clock,
                distance_feet=distance_feet,
            )
        )
    return parsed


def _normalize_clock(value: Any) -> Optional[int]:
    try:
        clock = int(value)
    except Exception:
        return None
    if clock < 1 or clock > 12:
        return None
    return clock


def _normalize_distance(value: Any) -> Optional[float]:
    try:
        distance = float(value)
    except Exception:
        return None
    if distance < 0.0:
        return None
    return round(distance, 3)


def _render_prediction_text(change: OnlineBaselineChange) -> str:
    subject = _clean_text(change.object_description) or "something"
    detail = _clean_text(change.change_description)
    context = _clean_text(change.context_description)

    if not detail:
        if change.change_type == "appear":
            detail = "appeared"
        elif change.change_type == "disappear":
            detail = "disappeared"
        else:
            detail = "changed"

    sentence = f"I notice {subject} {detail}."
    if context:
        sentence = f"{sentence[:-1]} {context}."
    distance_text = _format_distance_feet(change.distance_feet)
    return (
        f"{sentence} "
        f"It's at your {int(change.clock_direction)} o'clock, about {distance_text} feet away."
    )


def _format_distance_feet(value: float) -> str:
    rounded = round(max(0.0, float(value)), 1)
    if abs(rounded - round(rounded)) < 1e-6:
        return str(int(round(rounded)))
    return f"{rounded:.1f}".rstrip("0").rstrip(".")


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _safe_json_loads(text: str) -> Dict[str, Any]:
    stripped = str(text or "").strip()
    if not stripped:
        return {}
    try:
        payload = json.loads(stripped)
    except Exception:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match is None:
            return {}
        try:
            payload = json.loads(match.group(0))
        except Exception:
            return {}
    return payload if isinstance(payload, dict) else {}


def _safe_slug(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "")).strip("_")
    return cleaned or "baseline"


def _sample_turn_frames(
    frames: Sequence[CaptureFrameRecord],
    *,
    max_count: int,
) -> List[CaptureFrameRecord]:
    rows = list(frames)
    limit = max(1, int(max_count))
    if len(rows) <= limit:
        return rows
    stride = max(1, (len(rows) + limit - 1) // limit)
    return rows[::stride][:limit]


def _build_schedule_offsets(records: List[CaptureFrameRecord]) -> List[float]:
    if not records:
        return []
    offsets: List[float] = [0.0]
    total = 0.0
    previous = records[0]
    for current in records[1:]:
        delta = max(0.0, (current.timestamp - previous.timestamp).total_seconds())
        if current.capture_name != previous.capture_name:
            delta = 0.0
        total += delta
        offsets.append(total)
        previous = current
    return offsets
def _elapsed_since_run_start(run_start_perf: Optional[float], now_perf: float) -> Optional[float]:
    if run_start_perf is None:
        return None
    try:
        return round(max(0.0, float(now_perf) - float(run_start_perf)), 6)
    except Exception:
        return None
