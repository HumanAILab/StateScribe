from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import re
import shutil
import tempfile
from time import perf_counter, sleep
from typing import Any, Dict, List, Optional, Sequence

import cv2
from google import genai
from google.genai import types

from benchmark.baselines.offline_prompts import (
    OFFLINE_BASELINE_ANALYSIS_PROMPT,
    OFFLINE_BASELINE_RESPONSE_SCHEMA,
    OFFLINE_BASELINE_SYSTEM_PROMPT,
)
from benchmark.config import DEFAULT_FRAME_STRIDE, DEFAULT_OUTPUT_ROOT, DEFAULT_USE_ALIGNED_METADATA
from benchmark.dataset_loader import CaptureFrameRecord, load_dataset_records
from benchmark.recorder import BenchmarkOutputRecorder
from config import GEMINI_API_KEY

LOGGER = logging.getLogger("benchmark.baselines.offline")

DEFAULT_OFFLINE_GEMINI_MODEL = "gemini-3.1-pro-preview"
DEFAULT_OFFLINE_GEMINI_TIMEOUT_MS = 30000000
DEFAULT_OFFLINE_VIDEO_FPS = 1.0
DEFAULT_OFFLINE_FILE_ACTIVE_TIMEOUT_S = 300.0
DEFAULT_OFFLINE_FILE_ACTIVE_POLL_INTERVAL_S = 1.0


@dataclass(frozen=True)
class OfflineBaselineChange:
    evidence_time_seconds: float
    change_type: str
    object_description: str
    change_description: str
    context_description: str
    clock_direction: int
    distance_feet: float


@dataclass(frozen=True)
class PreparedVideo:
    records: List[CaptureFrameRecord]
    local_path: Path
    uploaded_file: Any
    duration_seconds: float


@dataclass(frozen=True)
class OfflineBaselineResult:
    raw_response_text: str
    changes: List[OfflineBaselineChange]
    processing_duration_s: float
    request_count: int
    stage_timings: List[Dict[str, Any]]


class GeminiOfflineBaselineBackend:
    method_slug = "baseline_offline_gemini_flash"
    provider_name = "gemini"
    description_mode = "baseline_offline_gemini"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_OFFLINE_GEMINI_MODEL,
        timeout_ms: int = DEFAULT_OFFLINE_GEMINI_TIMEOUT_MS,
        video_fps: float = DEFAULT_OFFLINE_VIDEO_FPS,
    ) -> None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set.")
        self.model_name = str(model_name or DEFAULT_OFFLINE_GEMINI_MODEL)
        self._timeout_ms = max(1, int(timeout_ms))
        self._video_fps = max(1e-6, float(video_fps))
        self._client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"timeout": self._timeout_ms},
        )
        self._temp_dir: Optional[Path] = None
        self._prepared_video: Optional[PreparedVideo] = None

    def prepare_inputs(self, records: Sequence[CaptureFrameRecord]) -> None:
        ordered_records = list(records)
        self._cleanup_local_temp_dir()
        self._prepared_video = None
        if not ordered_records:
            return
        temp_dir = Path(tempfile.mkdtemp(prefix="offline_baseline_video_"))
        self._temp_dir = temp_dir
        local_path = temp_dir / "environment.mp4"
        _write_video_clip(ordered_records, local_path, fps=self._video_fps)
        uploaded_file = self._client.files.upload(file=str(local_path))
        uploaded_file = self._wait_for_file_active(uploaded_file)
        self._prepared_video = PreparedVideo(
            records=list(ordered_records),
            local_path=local_path,
            uploaded_file=uploaded_file,
            duration_seconds=_video_duration_seconds(len(ordered_records), self._video_fps),
        )

    def process_records(self, records: Sequence[CaptureFrameRecord]) -> OfflineBaselineResult:
        ordered_records = list(records)
        if not ordered_records:
            return OfflineBaselineResult(
                raw_response_text="",
                changes=[],
                processing_duration_s=0.0,
                request_count=0,
                stage_timings=[],
            )
        if self._prepared_video is None:
            self.prepare_inputs(ordered_records)

        request_started = perf_counter()
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=self._build_request_contents(),
            config=types.GenerateContentConfig(
                system_instruction=OFFLINE_BASELINE_SYSTEM_PROMPT,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=OFFLINE_BASELINE_RESPONSE_SCHEMA,
                thinking_config=types.ThinkingConfig(thinking_level="high"),
            ),
        )
        duration_s = max(0.0, perf_counter() - request_started)
        response_text = str(response.text or "").strip()
        payload = _safe_json_loads(response_text)
        return OfflineBaselineResult(
            raw_response_text=response_text,
            changes=_parse_offline_changes(payload),
            processing_duration_s=round(duration_s, 6),
            request_count=1,
            stage_timings=[
                {
                    "stage": "video_analysis",
                    "duration_s": round(duration_s, 6),
                    "frame_count": len(ordered_records),
                    "video_count": 1,
                }
            ],
        )

    def resolve_record(self, evidence_time_seconds: float) -> Optional[CaptureFrameRecord]:
        video = self._prepared_video
        if video is None or not video.records:
            return None
        target_index = int(max(0.0, float(evidence_time_seconds)) * self._video_fps + 1e-6)
        target_index = min(max(target_index, 0), len(video.records) - 1)
        return video.records[target_index]

    def close(self) -> None:
        self._prepared_video = None
        self._cleanup_local_temp_dir()

    def _build_request_contents(self) -> List[Any]:
        video = self._prepared_video
        if video is None or not video.records:
            raise ValueError("Offline video input has not been prepared.")
        first = video.records[0]
        last = video.records[-1]
        lines = [
            OFFLINE_BASELINE_ANALYSIS_PROMPT,
            f"The video is encoded at {self._video_fps:g} video frame(s) per second.",
            "Use evidence_time_seconds as the playback time within the video.",
        ]
        if abs(self._video_fps - 1.0) < 1e-6:
            lines.append("At this encoding, each one-second step corresponds to one sampled frame.")
        else:
            lines.append(f"Each sampled frame advances the clip by {1.0 / self._video_fps:g} seconds.")
        lines.append(
            (
                f"Video duration_seconds={_format_seconds(video.duration_seconds)}, "
                f"sampled_frame_count={len(video.records)}, "
                f"start_capture={first.capture_name}, start_frame_index={first.frame_index}, "
                f"end_capture={last.capture_name}, end_frame_index={last.frame_index}"
            )
        )
        contents: List[Any] = ["\n".join(lines)]
        contents.append(video.uploaded_file)
        return contents

    def _wait_for_file_active(self, uploaded_file: Any) -> Any:
        file_name = str(getattr(uploaded_file, "name", "") or "").strip()
        if not file_name:
            return uploaded_file
        deadline = perf_counter() + DEFAULT_OFFLINE_FILE_ACTIVE_TIMEOUT_S
        current = uploaded_file
        while True:
            state = _file_state_name(current)
            if state == "ACTIVE":
                return current
            if state == "FAILED":
                raise RuntimeError(f"Uploaded video file failed to become ACTIVE: {file_name}")
            if perf_counter() >= deadline:
                raise TimeoutError(f"Timed out waiting for uploaded video file to become ACTIVE: {file_name}")
            sleep(DEFAULT_OFFLINE_FILE_ACTIVE_POLL_INTERVAL_S)
            current = self._client.files.get(name=file_name)

    def _cleanup_local_temp_dir(self) -> None:
        if self._temp_dir is None:
            return
        try:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        finally:
            self._temp_dir = None


class OfflineBaselineRunner:
    def __init__(
        self,
        *,
        backend: GeminiOfflineBaselineBackend,
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
            raise ValueError("No valid frame record found for offline baseline run.")

        dataset_slug = _safe_slug(self.dataset_dir.name or "dataset")
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix_slug = _safe_slug(world_name_prefix)
        self.world_name = f"{prefix_slug}_{backend.method_slug}_{dataset_slug}_{run_stamp}"
        self.output_dir = self.output_root / dataset_slug / backend.method_slug / run_stamp

        self.output_recorder = BenchmarkOutputRecorder()
        self.feed_records: List[Dict[str, Any]] = []
        self._frame_processing_rows: Dict[str, Dict[str, Any]] = {}
        self._ordered_timestamps = [record.timestamp.isoformat() for record in self.records]
        self.skipped_records: List[Dict[str, Any]] = []
        self._run_start_wall: Optional[datetime] = None
        self._run_start_perf: Optional[float] = None

    def run(self) -> Dict[str, Path]:
        try:
            self.backend.prepare_inputs(self.records)
            self._run_start_wall = datetime.now()
            self._run_start_perf = perf_counter()
            request_start_wall = datetime.now()
            request_start_perf = perf_counter()
            self._record_feed(request_start_wall, request_start_perf)
            result = self.backend.process_records(self.records)
            request_end_wall = datetime.now()
            self._record_processing_rows(
                request_start_wall=request_start_wall,
                request_end_wall=request_end_wall,
                result=result,
            )
            self._record_predictions(result)
            return self._write_outputs(result)
        finally:
            self.backend.close()

    def _record_feed(self, request_start_wall: datetime, request_start_perf: float) -> None:
        feed_elapsed_s = _elapsed_since_run_start(self._run_start_perf, request_start_perf)
        for record in self.records:
            self.feed_records.append(
                {
                    "capture_name": record.capture_name,
                    "frame_index": record.frame_index,
                    "frame_timestamp": record.timestamp.isoformat(),
                    "feed_start_wall": request_start_wall.isoformat(),
                    "feed_end_wall": request_start_wall.isoformat(),
                    "feed_duration_s": 0.0,
                    "feed_elapsed_s": feed_elapsed_s,
                }
            )

    def _record_processing_rows(
        self,
        *,
        request_start_wall: datetime,
        request_end_wall: datetime,
        result: OfflineBaselineResult,
    ) -> None:
        for frame_timestamp in self._ordered_timestamps:
            reference_timestamp = self._reference_timestamp_for(frame_timestamp)
            self._frame_processing_rows[frame_timestamp] = {
                "frame_timestamp": frame_timestamp,
                "reference_timestamp": reference_timestamp if reference_timestamp != frame_timestamp else "",
                "reference_found": bool(reference_timestamp and reference_timestamp != frame_timestamp),
                "total_duration_s": result.processing_duration_s,
                "end_to_end_duration_s": result.processing_duration_s,
                "processing_duration_s": result.processing_duration_s,
                "queue_wait_total_s": 0.0,
                "pre_pipeline_duration_s": 0.0,
                "frame_queue_wait_s": 0.0,
                "vlm_result_queue_wait_s": 0.0,
                "frame_start_wall": request_start_wall.isoformat(),
                "frame_end_wall": request_end_wall.isoformat(),
                "ingress_wall": request_start_wall.isoformat(),
                "pipeline_enqueue_wall": request_start_wall.isoformat(),
                "pipeline_dequeue_wall": request_start_wall.isoformat(),
                "stage_timings_ordered": list(result.stage_timings),
            }

    def _record_predictions(self, result: OfflineBaselineResult) -> None:
        for change in result.changes:
            detection_record = self.backend.resolve_record(change.evidence_time_seconds)
            if detection_record is None:
                continue
            current_timestamp = detection_record.timestamp.isoformat()
            structured_change = {
                "change_type": change.change_type,
                "object_description": change.object_description,
                "change_description": change.change_description,
                "context_description": change.context_description,
                "clock_direction": change.clock_direction,
                "distance_feet": round(float(change.distance_feet), 3),
                "evidence_time_seconds": round(float(change.evidence_time_seconds), 3),
                "capture_name": detection_record.capture_name,
                "frame_index": detection_record.frame_index,
                "frame_timestamp": current_timestamp,
            }
            self.output_recorder.record_prediction(
                text=_render_prediction_text(change),
                description_mode=self.backend.description_mode,
                current_timestamp=current_timestamp,
                reference_timestamp=self._reference_timestamp_for(current_timestamp),
                changes=[structured_change],
                recorded_from=self.backend.method_slug,
            )

    def _reference_timestamp_for(self, frame_timestamp: str) -> str:
        try:
            index = self._ordered_timestamps.index(frame_timestamp)
        except ValueError:
            return frame_timestamp
        if index <= 0:
            return frame_timestamp
        return self._ordered_timestamps[index - 1]

    def _write_outputs(self, result: OfflineBaselineResult) -> Dict[str, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        predictions_path = self.output_dir / "predictions.json"
        frames_path = self.output_dir / "frames.jsonl"
        manifest_path = self.output_dir / "manifest.json"

        predictions_payload = self.output_recorder.build_predictions_payload(
            dataset=str(self.dataset_dir),
            world_name=self.world_name,
            mode="offline",
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
            "mode": "offline",
            "runner_kind": "baseline",
            "baseline_kind": "offline",
            "baseline_method": self.backend.method_slug,
            "provider": self.backend.provider_name,
            "model": self.backend.model_name,
            "frame_stride": self.frame_stride,
            "max_captures": self.max_captures,
            "live_descriptions_enabled": False,
            "description_output_mode": self.backend.method_slug,
            "latency_time_axis": "offline_processing_only",
            "latency_semantics": "processing_only",
            "evidence_time_axis": "clip_playback_seconds",
            "world_name": self.world_name,
            "session_scope": "dataset",
            "request_count": int(result.request_count),
            "warmup_request_count": 0,
            "records_total": len(self.records),
            "frames_fed": len(self.feed_records),
            "feed_records_count": len(self.feed_records),
            "frame_log_count": len(self._frame_processing_rows),
            "spoken_emit_count": 0,
            "prediction_count": len(predictions_payload.get("records", [])),
            "video_clip_count": 1,
            "video_fps": round(float(self.backend._video_fps), 6),
            "skipped_records": list(self.skipped_records),
            "run_start_wall": self._run_start_wall.isoformat() if self._run_start_wall is not None else "",
            "simulated_start_time": self.records[0].timestamp.isoformat(),
            "simulated_end_time": self.records[-1].timestamp.isoformat(),
            "pipeline_timeout_reached": False,
            "broker_timeout_reached": False,
        }
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=True, indent=2)

        LOGGER.info("Offline baseline output: %s", self.output_dir)
        return {
            "output_dir": self.output_dir,
            "predictions": predictions_path,
            "frames": frames_path,
            "manifest": manifest_path,
        }


def _write_video_clip(records: Sequence[CaptureFrameRecord], output_path: Path, *, fps: float) -> None:
    if not records:
        raise ValueError("Cannot build a video clip from zero records.")
    first_frame = cv2.imread(str(records[0].rgb_path), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise ValueError(f"Failed to read image: {records[0].rgb_path}")
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(max(1e-6, fps)),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    try:
        writer.write(first_frame)
        for record in records[1:]:
            frame = cv2.imread(str(record.rgb_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Failed to read image: {record.rgb_path}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            writer.write(frame)
    finally:
        writer.release()


def _video_duration_seconds(frame_count: int, fps: float) -> float:
    return round(max(0.0, float(frame_count) / max(1e-6, float(fps))), 6)


def _file_state_name(file_obj: Any) -> str:
    state = getattr(file_obj, "state", None)
    if state is None:
        return ""
    value = getattr(state, "value", state)
    return str(value or "").strip().upper()


def _parse_offline_changes(payload: Dict[str, Any]) -> List[OfflineBaselineChange]:
    rows = payload.get("changes")
    if not isinstance(rows, list):
        return []
    parsed: List[OfflineBaselineChange] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        change_type = str(row.get("change_type", "") or "").strip().lower()
        if change_type not in {"appear", "disappear", "change"}:
            continue
        evidence_time_seconds = _normalize_non_negative_float(row.get("evidence_time_seconds"))
        clock = _normalize_clock(row.get("clock_direction"))
        distance_feet = _normalize_distance(row.get("distance_feet"))
        if evidence_time_seconds is None or clock is None or distance_feet is None:
            continue
        parsed.append(
            OfflineBaselineChange(
                evidence_time_seconds=evidence_time_seconds,
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


def _normalize_non_negative_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return None
    if number < 0.0:
        return None
    return round(number, 3)


def _normalize_distance(value: Any) -> Optional[float]:
    distance = _normalize_non_negative_float(value)
    if distance is None:
        return None
    return round(distance, 3)


def _render_prediction_text(change: OfflineBaselineChange) -> str:
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


def _format_seconds(value: float) -> str:
    rounded = round(max(0.0, float(value)), 3)
    if abs(rounded - round(rounded)) < 1e-6:
        return str(int(round(rounded)))
    return f"{rounded:.3f}".rstrip("0").rstrip(".")


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


def _elapsed_since_run_start(run_start_perf: Optional[float], now_perf: float) -> Optional[float]:
    if run_start_perf is None:
        return None
    try:
        return round(max(0.0, float(now_perf) - float(run_start_perf)), 6)
    except Exception:
        return None
