# components/logging/frame_logger.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional
import json
import logging
import threading

import numpy as np

STAGE_ORDER = [
    "ingress_to_enqueue",
    "frame_queue_wait",
    "frame_matching",
    "vlm_call",
    "vlm_result_queue_wait",
    "prepare_changes",
    "visibility_masks",
    "filter_changes",
    "snapshot_features",
    "fastsam_segmentation",
    "compute_3d_bboxes",
    "update_change_memory",
    "display_update",
]


@dataclass
class StageTiming:
    start_wall: Optional[datetime] = None
    end_wall: Optional[datetime] = None
    duration_s: Optional[float] = None


@dataclass
class FrameLogRecord:
    frame_timestamp: datetime
    world_name: Optional[str] = None
    reference_timestamp: Optional[datetime] = None
    reference_found: bool = False
    ingress_wall: Optional[datetime] = None
    pipeline_enqueue_wall: Optional[datetime] = None
    pipeline_dequeue_wall: Optional[datetime] = None
    frame_start_wall: datetime = field(default_factory=datetime.now)
    frame_end_wall: Optional[datetime] = None
    total_duration_s: Optional[float] = None
    pre_pipeline_duration_s: Optional[float] = None
    frame_queue_wait_s: Optional[float] = None
    vlm_result_queue_wait_s: Optional[float] = None
    queue_wait_total_s: Optional[float] = None
    processing_duration_s: Optional[float] = None
    end_to_end_duration_s: Optional[float] = None
    stage_timings: Dict[str, StageTiming] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)
    vlm_raw: Optional[Dict[str, Any]] = None
    prepared_changes: List[Dict[str, Any]] = field(default_factory=list)
    filter_trace: List[Dict[str, Any]] = field(default_factory=list)
    kept_changes: List[Dict[str, Any]] = field(default_factory=list)
    memory_actions: List[Dict[str, Any]] = field(default_factory=list)
    change_memory_snapshot: Optional[Dict[str, Any]] = None
    final_description: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    _ingress_perf: Optional[float] = field(default=None, repr=False)
    _frame_start_perf: float = field(default_factory=perf_counter, repr=False)
    _stage_start_perf: Dict[str, float] = field(default_factory=dict, repr=False)

    def start_stage(self, name: str) -> None:
        timing = self.stage_timings.get(name)
        if timing is None:
            timing = StageTiming()
            self.stage_timings[name] = timing
        timing.start_wall = datetime.now()
        self._stage_start_perf[name] = perf_counter()

    def end_stage(self, name: str) -> None:
        timing = self.stage_timings.get(name)
        if timing is None:
            timing = StageTiming()
            self.stage_timings[name] = timing
        timing.end_wall = datetime.now()
        start_perf = self._stage_start_perf.get(name)
        if start_perf is not None:
            timing.duration_s = perf_counter() - start_perf

    def set_stage_timing(
        self,
        name: str,
        start_wall: Optional[datetime],
        end_wall: Optional[datetime],
        duration_s: Optional[float]
    ) -> None:
        self.stage_timings[name] = StageTiming(
            start_wall=start_wall,
            end_wall=end_wall,
            duration_s=duration_s
        )

    def finish(self) -> None:
        end_perf = perf_counter()
        if self.frame_end_wall is None:
            self.frame_end_wall = datetime.now()
            self.total_duration_s = end_perf - self._frame_start_perf

        if self.pre_pipeline_duration_s is None and self.ingress_wall and self.pipeline_enqueue_wall:
            self.pre_pipeline_duration_s = max(
                0.0,
                (self.pipeline_enqueue_wall - self.ingress_wall).total_seconds(),
            )
        if self.frame_queue_wait_s is None and self.pipeline_enqueue_wall and self.pipeline_dequeue_wall:
            self.frame_queue_wait_s = max(
                0.0,
                (self.pipeline_dequeue_wall - self.pipeline_enqueue_wall).total_seconds(),
            )

        queue_parts = [
            value
            for value in (self.frame_queue_wait_s, self.vlm_result_queue_wait_s)
            if isinstance(value, (int, float))
        ]
        if queue_parts:
            self.queue_wait_total_s = sum(float(v) for v in queue_parts)
        elif self.queue_wait_total_s is None:
            self.queue_wait_total_s = 0.0

        if self.processing_duration_s is None and isinstance(self.total_duration_s, (int, float)):
            queue_inside_pipeline = float(self.vlm_result_queue_wait_s or 0.0)
            self.processing_duration_s = max(0.0, float(self.total_duration_s) - queue_inside_pipeline)

        if self.end_to_end_duration_s is None:
            if isinstance(self._ingress_perf, (int, float)):
                self.end_to_end_duration_s = max(0.0, end_perf - float(self._ingress_perf))
            elif self.ingress_wall is not None and self.frame_end_wall is not None:
                self.end_to_end_duration_s = max(
                    0.0,
                    (self.frame_end_wall - self.ingress_wall).total_seconds(),
                )


class FrameLogger:
    _subscribers: List[Callable[[Dict[str, Any]], None]] = []
    _sub_lock = threading.Lock()

    def __init__(self) -> None:
        self._logger = logging.getLogger("statescribe.frame")

    @classmethod
    def subscribe(cls, callback: Callable[[Dict[str, Any]], None]) -> None:
        with cls._sub_lock:
            if callback not in cls._subscribers:
                cls._subscribers.append(callback)

    @classmethod
    def unsubscribe(cls, callback: Callable[[Dict[str, Any]], None]) -> None:
        with cls._sub_lock:
            cls._subscribers = [item for item in cls._subscribers if item != callback]

    def write(self, record: FrameLogRecord) -> None:
        record.finish()
        payload = serialize_frame_record(record)
        with self._sub_lock:
            subscribers = list(self._subscribers)
        for callback in subscribers:
            try:
                callback(payload)
            except Exception:
                continue
        if self._logger.isEnabledFor(logging.DEBUG):
            report = format_frame_report(record)
            self._logger.debug(report)


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if np is not None and isinstance(value, np.ndarray):
        if value.size <= 24:
            return value.tolist()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype)
        }
    if isinstance(value, dict):
        return {str(k): _sanitize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(v) for v in value]
    return value


def _json_block(value: Any) -> List[str]:
    cleaned = _sanitize_value(value)
    text = json.dumps(cleaned, indent=2, ensure_ascii=True)
    return text.splitlines()


def _format_stage_timing(name: str, timing: StageTiming) -> str:
    start = timing.start_wall.isoformat() if timing.start_wall else "n/a"
    end = timing.end_wall.isoformat() if timing.end_wall else "n/a"
    if timing.duration_s is None:
        duration = "n/a"
    else:
        duration = f"{timing.duration_s:.3f}s"
    return f"- {name}: start={start}, end={end}, duration={duration}"


def _format_change_memory(snapshot: Optional[Dict[str, Any]]) -> List[str]:
    if not snapshot:
        return ["(empty)"]

    objects = snapshot.get("objects", {})
    if not objects:
        return ["(empty)"]

    lines: List[str] = []
    for obj_id in sorted(objects.keys()):
        obj = objects[obj_id]
        snaps = obj.get("snapshots", [])
        status = "active"
        if snaps:
            last_type = snaps[-1].get("change_type")
            if last_type == "disappear":
                status = "disappeared"
        lines.append(f"- {obj_id} ({status}), snapshots={len(snaps)}")
        for snap in snaps:
            ts = snap.get("timestamp", "")
            change_type = snap.get("change_type", "")
            desc = snap.get("description", "")
            bbox = snap.get("bbox_3d", None)
            bbox_str = json.dumps(_sanitize_value(bbox), ensure_ascii=True) if bbox is not None else "null"
            lines.append(f"  - {ts} | {change_type} | {desc} | bbox_3d={bbox_str}")
    return lines


def _serialize_stage_timing(timing: StageTiming) -> Dict[str, Any]:
    return {
        "start_wall": timing.start_wall.isoformat() if timing.start_wall else "",
        "end_wall": timing.end_wall.isoformat() if timing.end_wall else "",
        "duration_s": round(float(timing.duration_s), 6) if timing.duration_s is not None else None,
    }


def serialize_frame_record(record: FrameLogRecord) -> Dict[str, Any]:
    stage_timings = {
        name: _serialize_stage_timing(timing)
        for name, timing in record.stage_timings.items()
    }
    ordered_stage_timings: List[Dict[str, Any]] = []
    for name in STAGE_ORDER:
        if name in stage_timings:
            ordered_stage_timings.append({"name": name, **stage_timings[name]})
    for name in sorted(stage_timings.keys()):
        if name not in STAGE_ORDER:
            ordered_stage_timings.append({"name": name, **stage_timings[name]})

    return {
        "frame_timestamp": record.frame_timestamp.isoformat() if record.frame_timestamp else "",
        "world_name": record.world_name or "",
        "reference_timestamp": record.reference_timestamp.isoformat() if record.reference_timestamp else "",
        "reference_found": bool(record.reference_found),
        "ingress_wall": record.ingress_wall.isoformat() if record.ingress_wall else "",
        "pipeline_enqueue_wall": record.pipeline_enqueue_wall.isoformat() if record.pipeline_enqueue_wall else "",
        "pipeline_dequeue_wall": record.pipeline_dequeue_wall.isoformat() if record.pipeline_dequeue_wall else "",
        "frame_start_wall": record.frame_start_wall.isoformat() if record.frame_start_wall else "",
        "frame_end_wall": record.frame_end_wall.isoformat() if record.frame_end_wall else "",
        "pre_pipeline_duration_s": round(float(record.pre_pipeline_duration_s), 6) if record.pre_pipeline_duration_s is not None else None,
        "frame_queue_wait_s": round(float(record.frame_queue_wait_s), 6) if record.frame_queue_wait_s is not None else None,
        "vlm_result_queue_wait_s": round(float(record.vlm_result_queue_wait_s), 6) if record.vlm_result_queue_wait_s is not None else None,
        "queue_wait_total_s": round(float(record.queue_wait_total_s), 6) if record.queue_wait_total_s is not None else None,
        "processing_duration_s": round(float(record.processing_duration_s), 6) if record.processing_duration_s is not None else None,
        "end_to_end_duration_s": round(float(record.end_to_end_duration_s), 6) if record.end_to_end_duration_s is not None else None,
        "total_duration_s": round(float(record.total_duration_s), 6) if record.total_duration_s is not None else None,
        "stage_timings": stage_timings,
        "stage_timings_ordered": ordered_stage_timings,
        "notes": _sanitize_value(record.notes),
        "vlm_raw": _sanitize_value(record.vlm_raw),
        "prepared_changes": _sanitize_value(record.prepared_changes),
        "filter_trace": _sanitize_value(record.filter_trace),
        "kept_changes": _sanitize_value(record.kept_changes),
        "memory_actions": _sanitize_value(record.memory_actions),
        "change_memory_snapshot": _sanitize_value(record.change_memory_snapshot),
        "final_description": record.final_description or "",
        "errors": _sanitize_value(record.errors),
    }


def format_frame_report(record: FrameLogRecord) -> str:
    lines: List[str] = []
    lines.append("=" * 96)
    lines.append(
        f"FRAME {record.frame_timestamp.isoformat()} | world={record.world_name or 'unknown'}"
    )
    if record.reference_found:
        ref_ts = record.reference_timestamp.isoformat() if record.reference_timestamp else "unknown"
        lines.append(f"Reference frame: FOUND ({ref_ts})")
    else:
        lines.append("Reference frame: NOT FOUND")

    total = f"{record.total_duration_s:.3f}s" if record.total_duration_s is not None else "n/a"
    start = record.frame_start_wall.isoformat() if record.frame_start_wall else "n/a"
    end = record.frame_end_wall.isoformat() if record.frame_end_wall else "n/a"
    lines.append(f"Frame processing: start={start}, end={end}, duration={total}")
    ingress = record.ingress_wall.isoformat() if record.ingress_wall else "n/a"
    pre_pipeline = f"{record.pre_pipeline_duration_s:.3f}s" if record.pre_pipeline_duration_s is not None else "n/a"
    frame_queue_wait = f"{record.frame_queue_wait_s:.3f}s" if record.frame_queue_wait_s is not None else "n/a"
    vlm_result_wait = f"{record.vlm_result_queue_wait_s:.3f}s" if record.vlm_result_queue_wait_s is not None else "n/a"
    queue_total = f"{record.queue_wait_total_s:.3f}s" if record.queue_wait_total_s is not None else "n/a"
    processing = f"{record.processing_duration_s:.3f}s" if record.processing_duration_s is not None else "n/a"
    end_to_end = f"{record.end_to_end_duration_s:.3f}s" if record.end_to_end_duration_s is not None else "n/a"
    lines.append(
        "Timing breakdown: "
        f"ingress={ingress}, pre_pipeline={pre_pipeline}, frame_queue_wait={frame_queue_wait}, "
        f"vlm_result_queue_wait={vlm_result_wait}, queue_wait_total={queue_total}, "
        f"processing={processing}, end_to_end={end_to_end}"
    )

    if record.notes:
        lines.append("Notes:")
        for key, value in record.notes.items():
            lines.append(f"- {key}: {json.dumps(_sanitize_value(value), ensure_ascii=True)}")

    if record.stage_timings:
        lines.append("Stage timings:")
        for name in STAGE_ORDER:
            if name in record.stage_timings:
                lines.append(_format_stage_timing(name, record.stage_timings[name]))
        for name in sorted(record.stage_timings.keys()):
            if name not in STAGE_ORDER:
                lines.append(_format_stage_timing(name, record.stage_timings[name]))

    if record.vlm_raw is not None:
        lines.append("VLM raw result:")
        lines.extend(_json_block(record.vlm_raw))

    if record.prepared_changes:
        lines.append("Prepared changes (raw + normalized bboxes):")
        for item in record.prepared_changes:
            lines.append(json.dumps(_sanitize_value(item), ensure_ascii=True))

    if record.filter_trace:
        lines.append("Filter decisions:")
        for item in record.filter_trace:
            lines.append(json.dumps(_sanitize_value(item), ensure_ascii=True))

    if record.kept_changes:
        lines.append("Kept changes summary:")
        for item in record.kept_changes:
            lines.append(json.dumps(_sanitize_value(item), ensure_ascii=True))

    if record.memory_actions:
        lines.append("Memory updates:")
        for item in record.memory_actions:
            lines.append(json.dumps(_sanitize_value(item), ensure_ascii=True))

    lines.append("Change memory snapshot:")
    lines.extend(_format_change_memory(record.change_memory_snapshot))

    desc = record.final_description if record.final_description else "(none)"
    lines.append(f"Final description: {desc}")

    if record.errors:
        lines.append("Errors:")
        for err in record.errors:
            lines.append(f"- {err}")

    lines.append("")
    return "\n".join(lines)
