from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import logging
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from benchmark.eval.adapters import (
    EvaluationBundle,
    FrameLookup,
    GroundTruthEvent,
    PoseLookup,
    PredictionRecord,
    segment_candidates_for_capture,
)


LOCATION_TOKEN_PATTERN = re.compile(r"\[\[(DIR|DIS):([A-Za-z0-9_:\-]+)\]\]")
CLOCK_DISTANCE_PATTERN = re.compile(
    r"at your\s+(\d{1,2})\s*o'?clock[^0-9]{0,24}(?:about\s+)?(\d+(?:\.\d+)?)\s*(?:foot|feet)\s+away",
    re.IGNORECASE,
)
DISTANCE_CLOCK_PATTERN = re.compile(
    r"(?:about\s+)?(\d+(?:\.\d+)?)\s*(?:foot|feet)\s+away[^0-9]{0,24}at your\s+(\d{1,2})\s*o'?clock",
    re.IGNORECASE,
)
LOGGER = logging.getLogger("benchmark.eval.scorer")


def evaluate_bundle(
    bundle: EvaluationBundle,
    judge: Any,
    *,
    fps: float = 30.0,
    max_clock_error_hours: int = 1,
    max_distance_error_ft: float = 2.0,
    judge_max_workers: int = 1,
) -> Dict[str, Any]:
    frame_lookup = FrameLookup(bundle.frames)
    pose_lookup = PoseLookup(bundle.capture_dir_by_name)
    gt_by_id = {event.event_id: event for event in bundle.ground_truth_events}
    offline_processing_only = _offline_processing_only_latency(bundle.manifest)

    warnings = _build_warnings(bundle.manifest)
    if any(not prediction.changes or not (prediction.detection_frame or {}).get("capture_name") for prediction in bundle.predictions):
        warnings.append("Predictions appear to come from an older benchmark schema; rerun benchmark for reliable latency and spatial scoring.")
    prediction_rows = _score_predictions(
        bundle=bundle,
        judge=judge,
        gt_by_id=gt_by_id,
        max_workers=judge_max_workers,
    )
    skipped_prediction_rows = [row for row in prediction_rows if bool(row.get("judge_skipped", False))]
    if skipped_prediction_rows:
        warnings.append(
            f"Skipped {len(skipped_prediction_rows)} prediction(s) because judge requests still failed after retries."
        )

    prediction_rows.sort(key=lambda row: _prediction_sort_key(bundle, row))

    first_prediction_by_gt: Dict[str, int] = {}
    matched_predictions_by_gt: Dict[str, List[int]] = {}
    for row in prediction_rows:
        row_idx = int(row["prediction_index"])
        for event_id in row["matched_event_ids"]:
            matched_predictions_by_gt.setdefault(event_id, []).append(row_idx)
            first_prediction_by_gt.setdefault(event_id, row_idx)

    correct_prediction_ids = set(first_prediction_by_gt.values())
    for row in prediction_rows:
        matched_ids = row["matched_event_ids"]
        if bool(row.get("judge_skipped", False)):
            classification = "judge_failed"
        elif not matched_ids:
            classification = "hallucination"
        elif int(row["prediction_index"]) in correct_prediction_ids:
            classification = "correct"
        else:
            classification = "duplicate"
        row["classification"] = classification

    scored_prediction_rows = [
        row
        for row in prediction_rows
        if not bool(row.get("judge_skipped", False))
    ]

    gt_rows: List[Dict[str, Any]] = []
    latency_values: List[float] = []
    spatial_clock_errors: List[float] = []
    spatial_distance_errors: List[float] = []
    spatial_correct_count = 0
    spatial_scored_count = 0

    row_by_prediction_id = {
        int(row["prediction_index"]): row
        for row in prediction_rows
    }
    for event in bundle.ground_truth_events:
        matched_prediction_ids = matched_predictions_by_gt.get(event.event_id, [])
        first_prediction_id = first_prediction_by_gt.get(event.event_id)
        first_row = row_by_prediction_id.get(first_prediction_id) if first_prediction_id is not None else None
        latency = _compute_latency(
            event,
            first_row,
            frame_lookup,
            fps=fps,
            processing_only=offline_processing_only,
        )
        spatial = _score_spatial(
            event,
            first_row,
            pose_lookup,
            max_clock_error_hours=max_clock_error_hours,
            max_distance_error_ft=max_distance_error_ft,
        )
        if latency is not None:
            latency_values.append(latency["latency_seconds"])
        if spatial is not None and spatial.get("scored"):
            spatial_scored_count += 1
            spatial_clock_errors.append(float(spatial["clock_error_hours"]))
            spatial_distance_errors.append(float(spatial["distance_error_ft"]))
            if bool(spatial.get("correct")):
                spatial_correct_count += 1

        gt_rows.append(
            {
                "event_id": event.event_id,
                "segment_id": event.segment_id,
                "change_type": event.change_type,
                "object_description": event.object_description,
                "change_description": event.change_description,
                "first_evidence": {
                    "capture_index": event.capture_index,
                    "capture_name": event.capture_name,
                    "frame_index": event.frame_index,
                    "timestamp": event.timestamp,
                },
                "matched_prediction_ids": matched_prediction_ids,
                "matched": bool(matched_prediction_ids),
                "first_prediction_id": first_prediction_id,
                "latency": latency,
                "spatial": spatial,
            }
        )
    summary = {
        "ground_truth_total": len(bundle.ground_truth_events),
        "ground_truth_matched": sum(1 for row in gt_rows if row["matched"]),
        "ground_truth_missed": sum(1 for row in gt_rows if not row["matched"]),
        "ground_truth_recall": _safe_rate(
            sum(1 for row in gt_rows if row["matched"]),
            len(bundle.ground_truth_events),
        ),
        "prediction_total": len(scored_prediction_rows),
        "prediction_skipped": len(skipped_prediction_rows),
        "prediction_correct": sum(1 for row in scored_prediction_rows if row["classification"] == "correct"),
        "prediction_duplicate": sum(1 for row in scored_prediction_rows if row["classification"] == "duplicate"),
        "prediction_hallucination": sum(1 for row in scored_prediction_rows if row["classification"] == "hallucination"),
        "prediction_precision": _safe_rate(
            sum(1 for row in scored_prediction_rows if row["classification"] == "correct"),
            len(scored_prediction_rows),
        ),
        "prediction_precision_with_duplicates": _safe_rate(
            sum(1 for row in scored_prediction_rows if row["classification"] in {"correct", "duplicate"}),
            len(scored_prediction_rows),
        ),
        "latency": _summarize_numeric(latency_values),
        "spatial": {
            "scored_count": spatial_scored_count,
            "correct_count": spatial_correct_count,
            "accuracy": _safe_rate(spatial_correct_count, spatial_scored_count),
            "mean_clock_error_hours": _safe_mean(spatial_clock_errors),
            "mean_distance_error_ft": _safe_mean(spatial_distance_errors),
        },
    }

    return {
        "metadata": {
            "annotations_path": str(bundle.annotations_path),
            "benchmark_output_dir": str(bundle.benchmark_output_dir),
            "judge_model": getattr(judge, "model", ""),
            "fps": float(fps),
            "max_clock_error_hours": int(max_clock_error_hours),
            "max_distance_error_ft": float(max_distance_error_ft),
            "latency_definition": (
                "detection_frame_processing_duration_s"
                if offline_processing_only
                else "max(0, (detection_frame_index - gt_evidence_frame_index) / fps + detection_frame_processing_duration_s)"
            ),
        },
        "warnings": warnings,
        "summary": summary,
        "ground_truth": gt_rows,
        "predictions": prediction_rows,
    }


def _candidate_events(bundle: EvaluationBundle, prediction: PredictionRecord) -> List[GroundTruthEvent]:
    capture_name = str((prediction.detection_frame or {}).get("capture_name", "") or "")
    capture_index = bundle.capture_index_by_name.get(capture_name, -1)
    if capture_index < 0:
        return list(bundle.ground_truth_events)
    allowed_segments = set(segment_candidates_for_capture(capture_index))
    return [
        event
        for event in bundle.ground_truth_events
        if event.segment_id in allowed_segments
    ]


def _score_predictions(
    *,
    bundle: EvaluationBundle,
    judge: Any,
    gt_by_id: Mapping[str, GroundTruthEvent],
    max_workers: int,
) -> List[Dict[str, Any]]:
    worker_count = max(1, int(max_workers))
    if worker_count <= 1 or len(bundle.predictions) <= 1:
        return [
            _score_prediction_row(
                bundle=bundle,
                judge=judge,
                gt_by_id=gt_by_id,
                prediction=prediction,
            )
            for prediction in bundle.predictions
        ]

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(
            executor.map(
                lambda prediction: _score_prediction_row(
                    bundle=bundle,
                    judge=judge,
                    gt_by_id=gt_by_id,
                    prediction=prediction,
                ),
                bundle.predictions,
            )
        )


def _score_prediction_row(
    *,
    bundle: EvaluationBundle,
    judge: Any,
    gt_by_id: Mapping[str, GroundTruthEvent],
    prediction: PredictionRecord,
) -> Dict[str, Any]:
    candidates = _candidate_events(bundle, prediction)
    base_row = {
        "prediction_index": prediction.prediction_index,
        "text": prediction.text,
        "recorded_from": prediction.recorded_from,
        "description_mode": prediction.description_mode,
        "current_timestamp": prediction.current_timestamp,
        "reference_timestamp": prediction.reference_timestamp,
        "detection_frame": dict(prediction.detection_frame),
        "reference_frame": dict(prediction.reference_frame),
        "changes": [dict(item) for item in prediction.changes],
        "candidate_event_ids": [item.event_id for item in candidates],
    }
    try:
        judge_result = judge.match_prediction(
            prediction_text=prediction.text,
            candidates=candidates,
        )
    except Exception as exc:
        error_summary = _score_error_summary(exc)
        LOGGER.warning(
            "Skipping prediction %s during evaluation because judge failed after retries: %s",
            prediction.prediction_index,
            error_summary,
        )
        return {
            **base_row,
            "matched_event_ids": [],
            "judge_reason": "",
            "judge_skipped": True,
            "judge_error": error_summary,
        }

    matched_ids = [
        event_id
        for event_id in judge_result.get("matched_event_ids", [])
        if event_id in gt_by_id
    ]
    return {
        **base_row,
        "matched_event_ids": matched_ids,
        "judge_reason": str(judge_result.get("reason", "") or ""),
        "judge_skipped": False,
        "judge_error": "",
    }


def _score_error_summary(exc: Exception) -> str:
    code = getattr(exc, "code", None)
    status = getattr(exc, "status", None)
    message = getattr(exc, "message", None)
    parts = [str(part) for part in (code, status, message) if part not in {None, ""}]
    if parts:
        return " | ".join(parts)
    return f"{exc.__class__.__name__}: {exc}"


def _prediction_sort_key(bundle: EvaluationBundle, row: Mapping[str, Any]) -> Tuple[int, int, int]:
    detection_frame = row.get("detection_frame") or {}
    capture_name = str(detection_frame.get("capture_name", "") or "")
    capture_index = bundle.capture_index_by_name.get(capture_name, 10**9)
    frame_index = _to_int(detection_frame.get("frame_index"), default=10**9)
    return (capture_index, frame_index, _to_int(row.get("prediction_index"), default=10**9))


def _compute_latency(
    event: GroundTruthEvent,
    row: Optional[Mapping[str, Any]],
    frame_lookup: FrameLookup,
    *,
    fps: float,
    processing_only: bool = False,
) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    detection_frame = row.get("detection_frame") or {}
    capture_name = str(detection_frame.get("capture_name", "") or "")
    detection_frame_index = _to_int(detection_frame.get("frame_index"), default=-1)
    detection_frame_row = frame_lookup.get_exact(capture_name, detection_frame_index)
    processing_duration_s = (
        detection_frame_row.processing_duration_s
        if detection_frame_row is not None
        else None
    )
    if detection_frame_index >= 0 and event.frame_index >= 0 and capture_name == event.capture_name:
        frame_gap_seconds = float(detection_frame_index - event.frame_index) / max(1e-9, float(fps))
    else:
        frame_gap_seconds = _fallback_latency_gap_seconds(event, detection_frame, frame_lookup)
    if frame_gap_seconds is None:
        return None
    processing = max(0.0, float(processing_duration_s or 0.0))
    latency_seconds = processing if processing_only else max(0.0, frame_gap_seconds + processing)
    return {
        "frame_gap_seconds": round(frame_gap_seconds, 6),
        "processing_duration_s": round(processing, 6),
        "latency_seconds": round(latency_seconds, 6),
        "detection_capture_name": capture_name,
        "detection_frame_index": detection_frame_index,
    }


def _offline_processing_only_latency(manifest: Mapping[str, Any]) -> bool:
    if str(manifest.get("latency_semantics", "") or "").strip().lower() == "processing_only":
        return True
    if str(manifest.get("baseline_kind", "") or "").strip().lower() == "offline":
        return True
    return str(manifest.get("mode", "") or "").strip().lower() == "offline"


def _fallback_latency_gap_seconds(
    event: GroundTruthEvent,
    detection_frame: Mapping[str, Any],
    frame_lookup: FrameLookup,
) -> Optional[float]:
    detection_capture = str(detection_frame.get("capture_name", "") or "")
    detection_frame_index = _to_int(detection_frame.get("frame_index"), default=-1)
    detection_row = frame_lookup.get_exact(detection_capture, detection_frame_index)
    evidence_row = frame_lookup.first_at_or_after(event.capture_name, event.frame_index)
    if detection_row is not None and evidence_row is not None:
        det_elapsed = detection_row.feed_elapsed_s
        ev_elapsed = evidence_row.feed_elapsed_s
        if det_elapsed is not None and ev_elapsed is not None:
            return float(det_elapsed) - float(ev_elapsed)
    det_ts = _parse_iso(str(detection_frame.get("frame_timestamp", "") or ""))
    gt_ts = _parse_iso(event.timestamp)
    if det_ts is not None and gt_ts is not None:
        return (det_ts - gt_ts).total_seconds()
    return None


def _score_spatial(
    event: GroundTruthEvent,
    row: Optional[Mapping[str, Any]],
    pose_lookup: PoseLookup,
    *,
    max_clock_error_hours: int,
    max_distance_error_ft: float,
) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    detection_frame = row.get("detection_frame") or {}
    capture_name = str(detection_frame.get("capture_name", "") or "")
    frame_index = _to_int(detection_frame.get("frame_index"), default=-1)
    if not capture_name or frame_index < 0:
        return None
    pose = pose_lookup.get_pose(capture_name, frame_index)
    if pose is None:
        return None

    gt_bbox = _bbox_for_event(event)
    gt_location = _location_from_bbox(gt_bbox, pose)
    if gt_location is None:
        return {
            "scored": False,
            "reason": "Missing or invalid ground-truth bbox.",
        }

    prediction_locations = _prediction_location_candidates(row, pose)
    if not prediction_locations:
        return {
            "scored": False,
            "reason": "No parsable predicted direction-distance claim.",
            "gt_clock": gt_location["clock"],
            "gt_distance_ft": gt_location["distance_ft"],
        }

    best = min(
        prediction_locations,
        key=lambda item: (_clock_error_hours(item["clock"], gt_location["clock"]), abs(item["distance_ft"] - gt_location["distance_ft"])),
    )
    clock_error = _clock_error_hours(best["clock"], gt_location["clock"])
    distance_error = abs(best["distance_ft"] - gt_location["distance_ft"])
    return {
        "scored": True,
        "correct": clock_error <= int(max_clock_error_hours) and distance_error <= float(max_distance_error_ft),
        "gt_clock": gt_location["clock"],
        "gt_distance_ft": round(gt_location["distance_ft"], 3),
        "predicted_clock": int(best["clock"]),
        "predicted_distance_ft": round(float(best["distance_ft"]), 3),
        "clock_error_hours": int(clock_error),
        "distance_error_ft": round(float(distance_error), 3),
    }


def _prediction_location_candidates(row: Mapping[str, Any], pose: np.ndarray) -> List[Dict[str, float]]:
    text = str(row.get("text", "") or "")
    changes = row.get("changes") or []
    candidates: List[Dict[str, float]] = []
    seen_token_ids = set()
    for _, object_id in LOCATION_TOKEN_PATTERN.findall(text):
        if object_id in seen_token_ids:
            continue
        seen_token_ids.add(object_id)
        change = next(
            (
                item
                for item in changes
                if isinstance(item, dict) and str(item.get("object_id", "") or "") == object_id
            ),
            None,
        )
        bbox = _bbox_for_change(change) if change is not None else None
        location = _location_from_bbox(bbox, pose)
        if location is not None:
            candidates.append(location)

    for match in CLOCK_DISTANCE_PATTERN.finditer(text):
        candidates.append(
            {
                "clock": int(match.group(1)),
                "distance_ft": float(match.group(2)),
            }
        )
    for match in DISTANCE_CLOCK_PATTERN.finditer(text):
        candidates.append(
            {
                "clock": int(match.group(2)),
                "distance_ft": float(match.group(1)),
            }
        )
    return candidates


def _bbox_for_event(event: GroundTruthEvent) -> Optional[Tuple[float, float, float, float, float, float]]:
    if event.change_type == "appear":
        return event.bbox_3d_t1 or event.bbox_3d_t0
    if event.change_type == "disappear":
        return event.bbox_3d_t0 or event.bbox_3d_t1
    return event.bbox_3d_t1 or event.bbox_3d_t0


def _bbox_for_change(change: Optional[Mapping[str, Any]]) -> Optional[Tuple[float, float, float, float, float, float]]:
    if not isinstance(change, Mapping):
        return None
    change_type = str(change.get("change_type", "") or "")
    b0 = _normalize_bbox(change.get("bbox_3d_t0"))
    b1 = _normalize_bbox(change.get("bbox_3d_t1"))
    if change_type == "appear":
        return b1 or b0
    if change_type == "disappear":
        return b0 or b1
    return b1 or b0


def _location_from_bbox(
    bbox: Optional[Tuple[float, float, float, float, float, float]],
    pose: np.ndarray,
) -> Optional[Dict[str, float]]:
    if bbox is None:
        return None
    center = (np.asarray(bbox[:3], dtype=np.float64) + np.asarray(bbox[3:], dtype=np.float64)) / 2.0
    relative = _world_to_user_relative(center, pose)
    if relative is None:
        return None
    distance_m = float(np.linalg.norm(relative))
    return {
        "clock": int(_clock_direction(relative)),
        "distance_ft": float(distance_m * 3.28084),
    }


def _world_to_user_relative(world_point: np.ndarray, pose_matrix: np.ndarray) -> Optional[np.ndarray]:
    if pose_matrix is None or np.asarray(pose_matrix).shape != (4, 4):
        return None
    pose = np.asarray(pose_matrix, dtype=np.float64)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    cam_pos = pose[:3, 3]
    cam_fwd_world = pose[:3, 2]
    forward = cam_fwd_world - np.dot(cam_fwd_world, world_up) * world_up
    if np.linalg.norm(forward) < 1e-6:
        cam_x_world = pose[:3, 0]
        forward = cam_x_world - np.dot(cam_x_world, world_up) * world_up
    if np.linalg.norm(forward) < 1e-6:
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    z_axis = forward / (np.linalg.norm(forward) + 1e-9)
    y_axis = world_up
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-9)

    human_pose = np.eye(4, dtype=np.float64)
    human_pose[:3, 0] = x_axis
    human_pose[:3, 1] = y_axis
    human_pose[:3, 2] = z_axis
    human_pose[:3, 3] = cam_pos

    inv_pose = np.linalg.inv(human_pose)
    point_h = np.array([world_point[0], world_point[1], world_point[2], 1.0], dtype=np.float64)
    return (inv_pose @ point_h)[:3]


def _clock_direction(relative_xyz: np.ndarray) -> int:
    x, _, z = relative_xyz.tolist()
    angle_deg = float(np.degrees(np.arctan2(-x, z)))
    clock_idx = int(np.round((angle_deg % 360.0) / 30.0)) % 12
    return 12 if clock_idx == 0 else clock_idx


def _clock_error_hours(predicted: int, ground_truth: int) -> int:
    a = int(predicted) % 12
    b = int(ground_truth) % 12
    diff = abs(a - b)
    return min(diff, 12 - diff)


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


def _build_warnings(manifest: Mapping[str, Any]) -> List[str]:
    warnings: List[str] = []
    if bool(manifest.get("pipeline_timeout_reached", False)):
        warnings.append("Pipeline drain timeout reached during benchmark run.")
    if bool(manifest.get("broker_timeout_reached", False)):
        warnings.append("Broker drain timeout reached during benchmark run.")
    return warnings


def _summarize_numeric(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean_s": None,
            "median_s": None,
            "p90_s": None,
        }
    arr = np.asarray(list(values), dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean_s": round(float(arr.mean()), 6),
        "median_s": round(float(np.median(arr)), 6),
        "p90_s": round(float(np.percentile(arr, 90)), 6),
    }


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    arr = np.asarray(list(values), dtype=np.float64)
    return round(float(arr.mean()), 6)


def _safe_rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _parse_iso(value: str) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None
