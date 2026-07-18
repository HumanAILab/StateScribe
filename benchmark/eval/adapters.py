from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from benchmark.dataset_loader import extract_pose, list_capture_dirs, load_metadata_entries, resolve_metadata_path


@dataclass(frozen=True)
class GroundTruthEvent:
    event_id: str
    segment_id: str
    change_type: str
    object_description: str
    change_description: str
    capture_index: int
    capture_name: str
    frame_index: int
    timestamp: str
    bbox_3d_t0: Optional[Tuple[float, float, float, float, float, float]]
    bbox_3d_t1: Optional[Tuple[float, float, float, float, float, float]]
    raw: Dict[str, Any]


@dataclass(frozen=True)
class PredictionRecord:
    prediction_index: int
    recorded_from: str
    description_mode: str
    text: str
    current_timestamp: str
    reference_timestamp: str
    detection_frame: Dict[str, Any]
    reference_frame: Dict[str, Any]
    changes: List[Dict[str, Any]]


@dataclass(frozen=True)
class FrameRecord:
    capture_name: str
    frame_index: int
    frame_timestamp: str
    feed_elapsed_s: Optional[float]
    processing_duration_s: Optional[float]
    raw: Dict[str, Any]


@dataclass
class EvaluationBundle:
    annotations_path: Path
    benchmark_output_dir: Path
    annotations: Dict[str, Any]
    manifest: Dict[str, Any]
    predictions_payload: Dict[str, Any]
    frames: List[FrameRecord]
    ground_truth_events: List[GroundTruthEvent]
    predictions: List[PredictionRecord]
    capture_index_by_name: Dict[str, int]
    capture_dir_by_name: Dict[str, Path]
    relevant_capture_names: List[str]


@dataclass(frozen=True)
class RepeatedCaptureRef:
    capture_index: int
    capture_name: str
    capture_dir: Optional[Path]
    repeat_index: int
    source_capture_index: int
    source_capture_name: str


REPEATED_CAPTURE_NAME_PATTERN = re.compile(
    r"^(?P<repeat_index>\d+)_(?P<source_capture_index>\d+)_(?P<source_capture_name>.+)_(?P<synthetic_stamp>\d{8}_\d{6})$"
)
TRANSITION_SEGMENT_PATTERN = re.compile(r"^(?P<start>\d+)_to_(?P<end>\d+)$")


class FrameLookup:
    def __init__(self, frames: Sequence[FrameRecord]) -> None:
        self._rows_by_capture: Dict[str, List[FrameRecord]] = {}
        self._rows_by_key: Dict[Tuple[str, int], FrameRecord] = {}
        for row in frames:
            self._rows_by_capture.setdefault(row.capture_name, []).append(row)
            self._rows_by_key[(row.capture_name, row.frame_index)] = row
        for rows in self._rows_by_capture.values():
            rows.sort(key=lambda item: item.frame_index)

    def get_exact(self, capture_name: str, frame_index: int) -> Optional[FrameRecord]:
        return self._rows_by_key.get((capture_name, int(frame_index)))

    def first_at_or_after(self, capture_name: str, frame_index: int) -> Optional[FrameRecord]:
        rows = self._rows_by_capture.get(capture_name, [])
        for row in rows:
            if row.frame_index >= int(frame_index):
                return row
        return None


class PoseLookup:
    def __init__(self, capture_dir_by_name: Mapping[str, Path]) -> None:
        self._capture_dir_by_name = dict(capture_dir_by_name)
        self._metadata_cache: Dict[str, List[Dict[str, Any]]] = {}

    def get_pose(self, capture_name: str, frame_index: int) -> Optional[np.ndarray]:
        capture_dir = self._capture_dir_by_name.get(capture_name)
        if capture_dir is None:
            return None
        entries = self._metadata_cache.get(capture_name)
        if entries is None:
            metadata_path = resolve_metadata_path(capture_dir, use_aligned_metadata=True)
            entries = load_metadata_entries(metadata_path)
            self._metadata_cache[capture_name] = entries
        if frame_index < 0 or frame_index >= len(entries):
            return None
        return extract_pose((entries[frame_index] or {}).get("cameraPose"))


def load_evaluation_bundle(
    annotations_path: Path,
    benchmark_output_dir: Path,
    *,
    max_captures: Optional[int] = None,
) -> EvaluationBundle:
    annotations = _load_json(annotations_path)
    manifest = _load_json(benchmark_output_dir / "manifest.json")
    predictions_payload = _load_json(benchmark_output_dir / "predictions.json")
    all_frames = _load_frames(benchmark_output_dir / "frames.jsonl")

    source_capture_manifest = _ordered_capture_manifest(((annotations.get("manifest") or {}).get("captures") or []))
    benchmark_capture_manifest = _load_benchmark_capture_manifest(manifest, all_frames, benchmark_output_dir)
    repeated_eval = _looks_like_repeated_benchmark(source_capture_manifest, benchmark_capture_manifest)

    ordered_captures = benchmark_capture_manifest if repeated_eval else source_capture_manifest
    if max_captures is not None and max_captures > 0:
        ordered_captures = ordered_captures[: int(max_captures)]
    selected_capture_names = [
        str(row.get("capture_name", "") or "")
        for row in ordered_captures
        if row.get("capture_name")
    ]
    selected_capture_set = set(selected_capture_names)

    capture_index_by_name = {
        str(row.get("capture_name", "")): int(row.get("capture_index", -1))
        for row in ordered_captures
        if row.get("capture_name")
    }
    capture_dir_by_name = {
        str(row.get("capture_name", "")): _resolve_manifest_path(
            annotations_path.parent,
            str(row.get("capture_dir", "")),
        )
        for row in ordered_captures
        if row.get("capture_name") and row.get("capture_dir")
    }

    frames = [
        row
        for row in all_frames
        if row.capture_name in selected_capture_set
    ]
    present_capture_names = {row.capture_name for row in frames if row.capture_name}
    relevant_capture_names = [
        capture_name
        for capture_name in selected_capture_names
        if capture_name in present_capture_names
    ]
    raw_ground_truth_events = (
        _expand_ground_truth_events_for_repeated(
            annotations=annotations,
            source_capture_manifest=source_capture_manifest,
            benchmark_capture_manifest=ordered_captures,
        )
        if repeated_eval
        else _load_ground_truth_events(annotations)
    )
    ground_truth_events = [
        event
        for event in raw_ground_truth_events
        if event.capture_name in relevant_capture_names
    ]
    predictions = [
        prediction
        for prediction in _load_predictions(predictions_payload)
        if _prediction_in_selected_captures(prediction, selected_capture_set)
    ]

    return EvaluationBundle(
        annotations_path=annotations_path,
        benchmark_output_dir=benchmark_output_dir,
        annotations=annotations,
        manifest=manifest,
        predictions_payload=predictions_payload,
        frames=frames,
        ground_truth_events=ground_truth_events,
        predictions=predictions,
        capture_index_by_name=capture_index_by_name,
        capture_dir_by_name=capture_dir_by_name,
        relevant_capture_names=relevant_capture_names,
    )


def segment_candidates_for_capture(capture_index: int) -> List[str]:
    values = [str(capture_index)]
    if capture_index > 0:
        values.append(f"{capture_index - 1}_to_{capture_index}")
    return values


def _ordered_capture_manifest(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    indexed_rows: List[Tuple[int, int, Mapping[str, Any]]] = []
    for offset, row in enumerate(rows):
        if not isinstance(row, Mapping) or not row.get("capture_name"):
            continue
        raw_index = row.get("capture_index", -1)
        try:
            capture_index = int(raw_index)
        except Exception:
            capture_index = 10**9
        indexed_rows.append((capture_index, offset, row))
    indexed_rows.sort(key=lambda item: (item[0], item[1]))
    return [row for _, _, row in indexed_rows]


def _load_benchmark_capture_manifest(
    manifest: Mapping[str, Any],
    frames: Sequence[FrameRecord],
    benchmark_output_dir: Path,
) -> List[Mapping[str, Any]]:
    dataset_value = str(manifest.get("dataset", "") or "")
    rows: List[Mapping[str, Any]] = []
    if dataset_value:
        dataset_dir = _resolve_manifest_path(benchmark_output_dir, dataset_value)
        if dataset_dir.exists():
            for capture_index, capture_dir in enumerate(list_capture_dirs(dataset_dir)):
                rows.append(
                    {
                        "capture_index": int(capture_index),
                        "capture_name": capture_dir.name,
                        "capture_dir": str(capture_dir.resolve()),
                    }
                )
    if rows:
        return rows

    seen = set()
    fallback_rows: List[Mapping[str, Any]] = []
    for frame in frames:
        capture_name = str(frame.capture_name or "")
        if not capture_name or capture_name in seen:
            continue
        seen.add(capture_name)
        fallback_rows.append(
            {
                "capture_index": len(fallback_rows),
                "capture_name": capture_name,
            }
        )
    return fallback_rows


def _looks_like_repeated_benchmark(
    source_capture_manifest: Sequence[Mapping[str, Any]],
    benchmark_capture_manifest: Sequence[Mapping[str, Any]],
) -> bool:
    if len(benchmark_capture_manifest) <= len(source_capture_manifest) or len(source_capture_manifest) <= 1:
        return False
    parsed = _parse_repeated_capture_manifest(
        benchmark_capture_manifest,
        source_capture_manifest,
        require_multiple_repeats=True,
    )
    return bool(parsed)


def _parse_repeated_capture_manifest(
    benchmark_capture_manifest: Sequence[Mapping[str, Any]],
    source_capture_manifest: Sequence[Mapping[str, Any]],
    *,
    require_multiple_repeats: bool,
) -> List[RepeatedCaptureRef]:
    if not benchmark_capture_manifest or not source_capture_manifest:
        return []

    source_by_index = {
        int(row.get("capture_index", -1)): row
        for row in source_capture_manifest
        if row.get("capture_name")
    }
    parsed: List[RepeatedCaptureRef] = []
    for row in benchmark_capture_manifest:
        capture_name = str(row.get("capture_name", "") or "")
        match = REPEATED_CAPTURE_NAME_PATTERN.fullmatch(capture_name)
        if match is None:
            return []
        source_capture_index = int(match.group("source_capture_index"))
        source_row = source_by_index.get(source_capture_index)
        if source_row is None:
            return []
        source_capture_name = str(match.group("source_capture_name") or "")
        expected_source_name = str(source_row.get("capture_name", "") or "")
        if source_capture_name != expected_source_name:
            return []
        capture_dir = str(row.get("capture_dir", "") or "")
        parsed.append(
            RepeatedCaptureRef(
                capture_index=int(row.get("capture_index", -1)),
                capture_name=capture_name,
                capture_dir=Path(capture_dir) if capture_dir else None,
                repeat_index=int(match.group("repeat_index")),
                source_capture_index=source_capture_index,
                source_capture_name=source_capture_name,
            )
        )

    if require_multiple_repeats:
        repeat_indices = {item.repeat_index for item in parsed}
        if len(parsed) <= len(source_capture_manifest) or len(repeat_indices) < 2:
            return []
    return parsed


def _expand_ground_truth_events_for_repeated(
    *,
    annotations: Mapping[str, Any],
    source_capture_manifest: Sequence[Mapping[str, Any]],
    benchmark_capture_manifest: Sequence[Mapping[str, Any]],
) -> List[GroundTruthEvent]:
    repeated_refs = _parse_repeated_capture_manifest(
        benchmark_capture_manifest,
        source_capture_manifest,
        require_multiple_repeats=False,
    )
    if not repeated_refs:
        return _load_ground_truth_events(annotations)

    source_events = _load_ground_truth_events(annotations)
    source_capture_count = len(source_capture_manifest)
    source_capture_name_by_index = {
        int(row.get("capture_index", -1)): str(row.get("capture_name", "") or "")
        for row in source_capture_manifest
        if row.get("capture_name")
    }
    source_capture_index_by_name = {
        str(row.get("capture_name", "") or ""): int(row.get("capture_index", -1))
        for row in source_capture_manifest
        if row.get("capture_name")
    }
    segment_by_id = {
        str(item.get("segment_id", "")): item
        for item in ((annotations.get("manifest") or {}).get("segments") or [])
        if isinstance(item, dict)
    }
    ref_by_repeat_source = {
        (item.repeat_index, item.source_capture_index): item
        for item in repeated_refs
    }

    output: List[GroundTruthEvent] = []
    for event in source_events:
        output.extend(
            _expand_ground_truth_event_for_repeated(
                event=event,
                source_capture_count=source_capture_count,
                source_capture_name_by_index=source_capture_name_by_index,
                source_capture_index_by_name=source_capture_index_by_name,
                segment_meta=segment_by_id.get(event.segment_id, {}),
                repeated_refs=repeated_refs,
                ref_by_repeat_source=ref_by_repeat_source,
            )
        )
    return output


def _expand_ground_truth_event_for_repeated(
    *,
    event: GroundTruthEvent,
    source_capture_count: int,
    source_capture_name_by_index: Mapping[int, str],
    source_capture_index_by_name: Mapping[str, int],
    segment_meta: Mapping[str, Any],
    repeated_refs: Sequence[RepeatedCaptureRef],
    ref_by_repeat_source: Mapping[Tuple[int, int], RepeatedCaptureRef],
) -> List[GroundTruthEvent]:
    if source_capture_count <= 0:
        return []

    segment_id = str(event.segment_id or "")
    if segment_id.isdigit():
        source_segment_index = int(segment_id)
        return [
            _build_repeated_ground_truth_event(
                event=event,
                event_id=f"{event.event_id}__cap_{ref.capture_index:03d}",
                segment_id=str(ref.capture_index),
                evidence_ref=_map_within_capture_evidence_ref(
                    event=event,
                    default_ref=ref,
                    source_capture_index_by_name=source_capture_index_by_name,
                    ref_by_repeat_source=ref_by_repeat_source,
                ),
                source_event_id=event.event_id,
                repeat_index=ref.repeat_index,
            )
            for ref in repeated_refs
            if ref.source_capture_index == source_segment_index
        ]

    match = TRANSITION_SEGMENT_PATTERN.fullmatch(segment_id)
    if match is None:
        return []

    source_start_index = int(match.group("start"))
    raw_end_index = int(match.group("end"))
    wrap_target = segment_meta.get("wrap_to_capture_index")
    if wrap_target is None and raw_end_index == source_capture_count:
        wrap_target = 0
    logical_end_index = int(wrap_target) if wrap_target is not None else raw_end_index

    output: List[GroundTruthEvent] = []
    for start_ref in repeated_refs:
        if start_ref.source_capture_index != source_start_index:
            continue
        if wrap_target is None:
            end_ref = ref_by_repeat_source.get((start_ref.repeat_index, logical_end_index))
        else:
            end_ref = ref_by_repeat_source.get((start_ref.repeat_index + 1, logical_end_index))
        if end_ref is None:
            continue
        evidence_ref = _map_transition_evidence_ref(
            event=event,
            source_start_index=source_start_index,
            logical_end_index=logical_end_index,
            start_ref=start_ref,
            end_ref=end_ref,
            source_capture_name_by_index=source_capture_name_by_index,
            source_capture_index_by_name=source_capture_index_by_name,
        )
        output.append(
            _build_repeated_ground_truth_event(
                event=event,
                event_id=f"{event.event_id}__seg_{start_ref.capture_index:03d}_to_{end_ref.capture_index:03d}",
                segment_id=f"{start_ref.capture_index}_to_{end_ref.capture_index}",
                evidence_ref=evidence_ref,
                source_event_id=event.event_id,
                repeat_index=start_ref.repeat_index,
            )
        )
    return output


def _map_within_capture_evidence_ref(
    *,
    event: GroundTruthEvent,
    default_ref: RepeatedCaptureRef,
    source_capture_index_by_name: Mapping[str, int],
    ref_by_repeat_source: Mapping[Tuple[int, int], RepeatedCaptureRef],
) -> RepeatedCaptureRef:
    evidence_source_index = _source_capture_index_for_event(
        event=event,
        source_capture_index_by_name=source_capture_index_by_name,
    )
    if evidence_source_index is None:
        return default_ref
    return ref_by_repeat_source.get((default_ref.repeat_index, evidence_source_index), default_ref)


def _map_transition_evidence_ref(
    *,
    event: GroundTruthEvent,
    source_start_index: int,
    logical_end_index: int,
    start_ref: RepeatedCaptureRef,
    end_ref: RepeatedCaptureRef,
    source_capture_name_by_index: Mapping[int, str],
    source_capture_index_by_name: Mapping[str, int],
) -> RepeatedCaptureRef:
    evidence_source_index = _source_capture_index_for_event(
        event=event,
        source_capture_index_by_name=source_capture_index_by_name,
    )
    if evidence_source_index == source_start_index:
        return start_ref
    if evidence_source_index == logical_end_index:
        return end_ref
    if event.capture_name == source_capture_name_by_index.get(source_start_index, ""):
        return start_ref
    if event.capture_name == source_capture_name_by_index.get(logical_end_index, ""):
        return end_ref
    return end_ref


def _source_capture_index_for_event(
    *,
    event: GroundTruthEvent,
    source_capture_index_by_name: Mapping[str, int],
) -> Optional[int]:
    if int(event.capture_index) >= 0:
        return int(event.capture_index)
    capture_name = str(event.capture_name or "")
    if capture_name in source_capture_index_by_name:
        return int(source_capture_index_by_name[capture_name])
    return None


def _build_repeated_ground_truth_event(
    *,
    event: GroundTruthEvent,
    event_id: str,
    segment_id: str,
    evidence_ref: RepeatedCaptureRef,
    source_event_id: str,
    repeat_index: int,
) -> GroundTruthEvent:
    raw = dict(event.raw)
    first_evidence = dict(raw.get("first_evidence") or {})
    first_evidence["capture_index"] = int(evidence_ref.capture_index)
    first_evidence["capture_name"] = str(evidence_ref.capture_name)
    raw["event_id"] = str(event_id)
    raw["segment_id"] = str(segment_id)
    raw["first_evidence"] = first_evidence
    raw["source_event_id"] = str(source_event_id)
    raw["source_segment_id"] = str(event.segment_id)
    raw["repeat_index"] = int(repeat_index)
    return GroundTruthEvent(
        event_id=str(event_id),
        segment_id=str(segment_id),
        change_type=event.change_type,
        object_description=event.object_description,
        change_description=event.change_description,
        capture_index=int(evidence_ref.capture_index),
        capture_name=str(evidence_ref.capture_name),
        frame_index=int(event.frame_index),
        timestamp=event.timestamp,
        bbox_3d_t0=event.bbox_3d_t0,
        bbox_3d_t1=event.bbox_3d_t1,
        raw=raw,
    )


def _prediction_in_selected_captures(prediction: PredictionRecord, selected_capture_set: set[str]) -> bool:
    if not selected_capture_set:
        return True
    capture_name = str((prediction.detection_frame or {}).get("capture_name", "") or "")
    if not capture_name:
        return True
    return capture_name in selected_capture_set


def _resolve_manifest_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(str(raw_path or ""))
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_ground_truth_events(payload: Mapping[str, Any]) -> List[GroundTruthEvent]:
    rows = payload.get("events")
    if not isinstance(rows, list):
        return []
    events: List[GroundTruthEvent] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        first = row.get("first_evidence") or {}
        if not isinstance(first, dict):
            continue
        capture_name = str(first.get("capture_name", "") or "")
        if not capture_name:
            continue
        events.append(
            GroundTruthEvent(
                event_id=str(row.get("event_id", "") or ""),
                segment_id=str(row.get("segment_id", "") or ""),
                change_type=str(row.get("change_type", "") or ""),
                object_description=str(row.get("object_description", "") or ""),
                change_description=str(row.get("change_description", "") or ""),
                capture_index=_to_int(first.get("capture_index"), default=-1),
                capture_name=capture_name,
                frame_index=_to_int(first.get("frame_index"), default=-1),
                timestamp=str(first.get("timestamp", "") or ""),
                bbox_3d_t0=_normalize_bbox(row.get("bbox_3d_t0")),
                bbox_3d_t1=_normalize_bbox(row.get("bbox_3d_t1")),
                raw=dict(row),
            )
        )
    return events


def _load_predictions(payload: Mapping[str, Any]) -> List[PredictionRecord]:
    rows = payload.get("records")
    if not isinstance(rows, list):
        return []
    records: List[PredictionRecord] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        records.append(
            PredictionRecord(
                prediction_index=_to_int(row.get("prediction_index"), default=0),
                recorded_from=str(row.get("recorded_from", "") or ""),
                description_mode=str(row.get("description_mode", "") or ""),
                text=str(row.get("text", "") or ""),
                current_timestamp=str(row.get("current_timestamp", "") or ""),
                reference_timestamp=str(row.get("reference_timestamp", "") or ""),
                detection_frame=dict(row.get("detection_frame") or {}),
                reference_frame=dict(row.get("reference_frame") or {}),
                changes=[dict(item) for item in (row.get("changes") or []) if isinstance(item, dict)],
            )
        )
    return records


def _load_frames(path: Path) -> List[FrameRecord]:
    rows: List[FrameRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            raw = json.loads(text)
            rows.append(
                FrameRecord(
                    capture_name=str(raw.get("capture_name", "") or ""),
                    frame_index=_to_int(raw.get("frame_index"), default=-1),
                    frame_timestamp=str(raw.get("frame_timestamp", "") or ""),
                    feed_elapsed_s=_to_optional_float(raw.get("feed_elapsed_s")),
                    processing_duration_s=_to_optional_float(raw.get("processing_duration_s")),
                    raw=raw,
                )
            )
    return rows


def _normalize_bbox(raw: Any) -> Optional[Tuple[float, float, float, float, float, float]]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 6:
        return None
    try:
        values = tuple(float(item) for item in raw)
    except Exception:
        return None
    if values[3] <= values[0] or values[4] <= values[1] or values[5] <= values[2]:
        return None
    return values


def _to_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _to_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object payload in {path}")
    return data
