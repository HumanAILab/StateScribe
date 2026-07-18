from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Mapping, Optional

from components.description_broker import DescriptionBroker


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None:
        return ""
    return str(value)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


class BenchmarkDescriptionBroker(DescriptionBroker):
    def __init__(self, *args: Any, on_emit: Optional[Any] = None, **kwargs: Any) -> None:
        self._on_emit = on_emit
        super().__init__(*args, **kwargs)

    def _emit_text(self, text: str, source: str) -> bool:
        success = super()._emit_text(text, source)
        if not success or self._on_emit is None:
            return success
        try:
            self._on_emit(text=text, source=source)
        except Exception:
            return success
        return success


class BenchmarkOutputRecorder:
    def __init__(self) -> None:
        self._spoken_lock = Lock()
        self._change_lock = Lock()
        self._prediction_lock = Lock()
        self.spoken_records: List[Dict[str, Any]] = []
        self.change_records: List[Dict[str, Any]] = []
        self.prediction_records: List[Dict[str, Any]] = []

    def record_spoken_event(
        self,
        *,
        text: str,
        source: str,
        emit_wall_time: datetime,
        emit_simulated_time: datetime,
        emit_elapsed_s: Optional[float],
        latest_frame_timestamp: Optional[datetime],
    ) -> Dict[str, Any]:
        record = {
            "spoken_index": 0,
            "source": _clean_text(source).lower(),
            "text": _clean_text(text),
            "emit_wall_time": _iso(emit_wall_time),
            "emit_simulated_time": _iso(emit_simulated_time),
            "emit_elapsed_s": _round_optional_seconds(emit_elapsed_s),
            "latest_frame_timestamp": _iso(latest_frame_timestamp),
        }
        with self._spoken_lock:
            record["spoken_index"] = len(self.spoken_records) + 1
            self.spoken_records.append(record)
        return dict(record)

    def record_change_snapshot(
        self,
        payload: Mapping[str, Any],
        *,
        description_mode: str,
        final_description: str = "",
        recorded_from: str = "change_snapshot",
    ) -> Optional[Dict[str, Any]]:
        changes = payload.get("changes")
        if not isinstance(changes, list) or not changes:
            return None

        record = {
            "change_event_index": 0,
            "recorded_from": _clean_text(recorded_from) or "change_snapshot",
            "description_mode": _clean_text(description_mode),
            "reference_timestamp": _iso(payload.get("reference_timestamp")),
            "current_timestamp": _iso(payload.get("current_timestamp")),
            "final_description": _clean_text(final_description),
            "changes": _clone_json_compatible(changes),
        }
        with self._change_lock:
            record["change_event_index"] = len(self.change_records) + 1
            self.change_records.append(record)
        return dict(record)

    def record_prediction(
        self,
        *,
        text: str,
        description_mode: str,
        current_timestamp: Any,
        reference_timestamp: Any,
        changes: Any,
        recorded_from: str,
    ) -> Optional[Dict[str, Any]]:
        clean_text = _clean_text(text)
        if not clean_text:
            return None
        if not isinstance(changes, list) or not changes:
            return None

        record = {
            "prediction_index": 0,
            "recorded_from": _clean_text(recorded_from) or "unknown",
            "description_mode": _clean_text(description_mode),
            "text": clean_text,
            "current_timestamp": _iso(current_timestamp),
            "reference_timestamp": _iso(reference_timestamp),
            "changes": _clone_json_compatible(changes),
        }
        with self._prediction_lock:
            record["prediction_index"] = len(self.prediction_records) + 1
            self.prediction_records.append(record)
        return dict(record)

    def record_change_from_frame_log(self, payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        kept_changes = payload.get("kept_changes")
        if not isinstance(kept_changes, list) or not kept_changes:
            return None

        return self.record_change_snapshot(
            {
                "reference_timestamp": payload.get("reference_timestamp", ""),
                "current_timestamp": payload.get("frame_timestamp", ""),
                "changes": kept_changes,
            },
            description_mode=str((payload.get("notes") or {}).get("description_mode", "")),
            final_description=str(payload.get("final_description", "")),
            recorded_from="frame_log",
        )

    def update_change_description(self, *, current_timestamp: str, final_description: str) -> bool:
        target_timestamp = _clean_text(current_timestamp)
        if not target_timestamp:
            return False
        clean_description = _clean_text(final_description)
        if not clean_description:
            return False
        with self._change_lock:
            for record in reversed(self.change_records):
                if str(record.get("current_timestamp", "")) != target_timestamp:
                    continue
                record["final_description"] = clean_description
                return True
        return False

    def build_spoken_payload(
        self,
        *,
        dataset: str,
        world_name: str,
        mode: str,
        frame_stride: int,
        run_start_wall: Optional[datetime],
        feed_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        feed_index = _build_feed_index(feed_records)
        with self._spoken_lock:
            records = [_enrich_spoken_record(record, feed_index) for record in self.spoken_records]
        return {
            "dataset": dataset,
            "world_name": world_name,
            "mode": mode,
            "frame_stride": frame_stride,
            "run_start_wall": _iso(run_start_wall),
            "latency_time_axis": "wall_elapsed_seconds",
            "records": records,
        }

    def build_predictions_payload(
        self,
        *,
        dataset: str,
        world_name: str,
        mode: str,
        frame_stride: int,
        live_descriptions_enabled: bool,
        description_output_mode: str,
        run_start_wall: Optional[datetime],
        feed_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        feed_index = _build_feed_index(feed_records)
        with self._prediction_lock:
            prediction_records = [
                _to_prediction_record(record, feed_index=feed_index)
                for record in self.prediction_records
            ]
        return {
            "dataset": dataset,
            "world_name": world_name,
            "mode": mode,
            "frame_stride": frame_stride,
            "live_descriptions_enabled": bool(live_descriptions_enabled),
            "description_output_mode": _clean_text(description_output_mode),
            "run_start_wall": _iso(run_start_wall),
            "latency_time_axis": "wall_elapsed_seconds",
            "records": prediction_records,
        }

    def build_change_payload(
        self,
        *,
        dataset: str,
        world_name: str,
        mode: str,
        frame_stride: int,
        run_start_wall: Optional[datetime],
        feed_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        feed_index = _build_feed_index(feed_records)
        with self._change_lock:
            records = [_enrich_change_record(record, feed_index) for record in self.change_records]
        return {
            "dataset": dataset,
            "world_name": world_name,
            "mode": mode,
            "frame_stride": frame_stride,
            "run_start_wall": _iso(run_start_wall),
            "latency_time_axis": "wall_elapsed_seconds",
            "records": records,
        }

    @staticmethod
    def write_json(path: Path, payload: Mapping[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(_to_json_compatible(payload), handle, ensure_ascii=True, indent=2)


def _clone_json_compatible(value: Any) -> Any:
    return json.loads(json.dumps(_to_json_compatible(value), ensure_ascii=True))


def _to_json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {
            str(key): _to_json_compatible(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]

    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        try:
            return _to_json_compatible(to_list())
        except Exception:
            pass

    return str(value)


def _build_feed_index(feed_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        str(row.get("frame_timestamp", "")): row
        for row in feed_records
        if row.get("frame_timestamp")
    }


def _feed_brief(feed_row: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not feed_row:
        return {}
    return {
        "capture_name": str(feed_row.get("capture_name", "")),
        "frame_index": feed_row.get("frame_index"),
        "frame_timestamp": str(feed_row.get("frame_timestamp", "")),
        "feed_start_wall": str(feed_row.get("feed_start_wall", "")),
        "feed_end_wall": str(feed_row.get("feed_end_wall", "")),
        "feed_elapsed_s": feed_row.get("feed_elapsed_s"),
    }


def _enrich_spoken_record(record: Mapping[str, Any], feed_index: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    enriched = dict(record)
    latest_key = str(record.get("latest_frame_timestamp", ""))
    latest_feed = feed_index.get(latest_key)
    if latest_feed is not None:
        enriched["capture_name"] = latest_feed.get("capture_name", "")
        enriched["frame_index"] = latest_feed.get("frame_index")
        enriched["latest_frame"] = _feed_brief(latest_feed)
    return enriched


def _enrich_change_record(record: Mapping[str, Any], feed_index: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    enriched = dict(record)
    current_key = str(record.get("current_timestamp", ""))
    reference_key = str(record.get("reference_timestamp", ""))
    current_feed = feed_index.get(current_key)
    reference_feed = feed_index.get(reference_key)
    if current_feed is not None:
        enriched["current_frame"] = _feed_brief(current_feed)
    if reference_feed is not None:
        enriched["reference_frame"] = _feed_brief(reference_feed)
    return enriched


def _to_prediction_record(
    record: Mapping[str, Any],
    *,
    feed_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    current_key = str(record.get("current_timestamp", ""))
    reference_key = str(record.get("reference_timestamp", ""))
    current_feed = feed_index.get(current_key)
    reference_feed = feed_index.get(reference_key)
    payload = {
        "prediction_index": int(record.get("prediction_index", 0) or 0),
        "recorded_from": _clean_text(record.get("recorded_from", "")),
        "description_mode": _clean_text(record.get("description_mode", "")),
        "text": _clean_text(record.get("text", "")),
        "current_timestamp": current_key,
        "reference_timestamp": reference_key,
        "detection_frame": _feed_brief(current_feed),
        "reference_frame": _feed_brief(reference_feed),
        "changes": _clone_json_compatible(record.get("changes", [])),
    }
    if not payload["detection_frame"]:
        payload["detection_frame"] = {
            "capture_name": "",
            "frame_index": None,
            "frame_timestamp": current_key,
            "feed_start_wall": "",
            "feed_end_wall": "",
            "feed_elapsed_s": None,
        }
    return payload


def _round_optional_seconds(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), 6)
    except Exception:
        return None
