from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from components.feature_factory import FeatureFactory
from components.memory.frame import Frame
from config import ORIGINAL_RESOLUTION, TARGET_RESOLUTION


@dataclass(frozen=True)
class CaptureFrameRecord:
    capture_name: str
    capture_dir: Path
    frame_index: int
    timestamp_seconds: float
    timestamp: datetime
    metadata: Dict[str, Any]
    rgb_path: Path
    depth_path: Path
    confidence_path: Optional[Path]


def list_capture_dirs(dataset_dir: Path) -> List[Path]:
    if not dataset_dir.exists():
        return []
    dirs = [path for path in dataset_dir.iterdir() if path.is_dir() and (path / "metadata.jsonl").exists()]
    return sorted(dirs, key=_capture_sort_key)


def resolve_metadata_path(capture_dir: Path, use_aligned_metadata: bool) -> Path:
    aligned = capture_dir / "metadata_aligned.jsonl"
    if use_aligned_metadata and aligned.exists():
        return aligned
    return capture_dir / "metadata.jsonl"


def load_metadata_entries(metadata_path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            entries.append(json.loads(text))
    return entries


def load_capture_records(
    capture_dir: Path,
    frame_stride: int,
    use_aligned_metadata: bool,
    require_confidence: bool,
) -> List[CaptureFrameRecord]:
    stride = max(1, int(frame_stride))
    metadata_path = resolve_metadata_path(capture_dir, use_aligned_metadata=use_aligned_metadata)
    metadata_entries = load_metadata_entries(metadata_path)
    if not metadata_entries:
        return []
    capture_start = parse_capture_start_time(capture_dir.name)
    capture_raw_base = _first_numeric_timestamp(metadata_entries)

    rgb_dir = capture_dir / "rgb"
    depth_dir = capture_dir / "depth"
    confidence_dir = capture_dir / "confidence"

    rgb_map = _collect_indexed_files(rgb_dir, allowed_suffixes={".jpg", ".jpeg", ".png"})
    depth_map = _collect_indexed_files(depth_dir, allowed_suffixes={".bin"})
    confidence_map = _collect_indexed_files(confidence_dir, allowed_suffixes={".bin"})

    common = set(rgb_map.keys()) & set(depth_map.keys())
    if require_confidence:
        common &= set(confidence_map.keys())

    valid_indices = sorted(idx for idx in common if idx < len(metadata_entries))
    sampled_indices = valid_indices[::stride]

    records: List[CaptureFrameRecord] = []
    for frame_index in sampled_indices:
        metadata = metadata_entries[frame_index]
        timestamp, timestamp_seconds = metadata_timestamp_to_datetime(
            metadata.get("timestamp"),
            fallback_seconds=float(frame_index),
            capture_start=capture_start,
            capture_raw_base=capture_raw_base,
        )

        confidence_path = confidence_map.get(frame_index)
        if confidence_path is None and require_confidence:
            continue

        records.append(
            CaptureFrameRecord(
                capture_name=capture_dir.name,
                capture_dir=capture_dir,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                timestamp=timestamp,
                metadata=metadata,
                rgb_path=rgb_map[frame_index],
                depth_path=depth_map[frame_index],
                confidence_path=confidence_path,
            )
        )
    return records


def load_dataset_records(
    dataset_dir: Path,
    frame_stride: int,
    use_aligned_metadata: bool,
    require_confidence: bool,
    max_captures: Optional[int] = None,
) -> List[CaptureFrameRecord]:
    records: List[CaptureFrameRecord] = []
    capture_dirs = list_capture_dirs(dataset_dir)
    if max_captures is not None and max_captures > 0:
        capture_dirs = capture_dirs[:max_captures]

    for capture_dir in capture_dirs:
        capture_records = load_capture_records(
            capture_dir=capture_dir,
            frame_stride=frame_stride,
            use_aligned_metadata=use_aligned_metadata,
            require_confidence=require_confidence,
        )
        records.extend(capture_records)
    return _normalize_monotonic_timestamps(records)


def metadata_timestamp_to_datetime(
    raw: Any,
    fallback_seconds: float,
    capture_start: Optional[datetime] = None,
    capture_raw_base: Optional[float] = None,
) -> Tuple[datetime, float]:
    if isinstance(raw, (int, float)):
        seconds = float(raw)
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            seconds = float(fallback_seconds)
        else:
            try:
                dt = datetime.fromisoformat(text)
                return dt, dt.timestamp()
            except ValueError:
                try:
                    seconds = float(text)
                except ValueError:
                    seconds = float(fallback_seconds)
    else:
        seconds = float(fallback_seconds)

    if not math.isfinite(seconds):
        seconds = float(fallback_seconds)
    if capture_start is not None and capture_raw_base is not None:
        offset = max(0.0, seconds - capture_raw_base)
        derived = capture_start + timedelta(seconds=offset)
        return derived, derived.timestamp()
    return datetime.fromtimestamp(seconds), seconds


def build_frame(
    record: CaptureFrameRecord,
    world_name: str,
    feature_factory: Optional[FeatureFactory] = None,
) -> Optional[Frame]:
    metadata = record.metadata
    camera_pose = metadata.get("cameraPose")

    rgb_image = _decode_rgb(record.rgb_path)
    if rgb_image is None:
        return None

    depth_meta = metadata.get("depth") or {}
    confidence_meta = metadata.get("confidence") or {}

    depth_width = _to_int(depth_meta.get("width"), default=256)
    depth_height = _to_int(depth_meta.get("height"), default=192)
    conf_width = _to_int(confidence_meta.get("width"), default=depth_width)
    conf_height = _to_int(confidence_meta.get("height"), default=depth_height)

    depth_map = _decode_depth(record.depth_path, width=depth_width, height=depth_height)
    confidence_map: Optional[np.ndarray]
    if record.confidence_path is None:
        confidence_map = None
    else:
        confidence_map = _decode_confidence(record.confidence_path, width=conf_width, height=conf_height)
    intrinsics = extract_intrinsics(camera_pose)
    pose_matrix = extract_pose(camera_pose)

    if depth_map is None or intrinsics is None or pose_matrix is None:
        return None

    if confidence_map is None:
        confidence_map = np.ones_like(depth_map, dtype=np.uint8)

    rgb_resized = cv2.resize(rgb_image, TARGET_RESOLUTION, interpolation=cv2.INTER_LINEAR)
    depth_resized = cv2.resize(depth_map, TARGET_RESOLUTION, interpolation=cv2.INTER_NEAREST)
    confidence_resized = cv2.resize(confidence_map, TARGET_RESOLUTION, interpolation=cv2.INTER_NEAREST)
    depth_resized[confidence_resized == 0] = 0.0

    dino_feature = None
    if feature_factory is not None:
        dino_feature = feature_factory.compute_dino_feature(rgb_resized)

    return Frame(
        timestamp=record.timestamp,
        world_name=world_name,
        rgb_image=rgb_resized,
        depth_map=depth_resized,
        depth_map_original=depth_resized.copy(),
        confidence_map=confidence_map,
        pose_matrix=pose_matrix,
        intrinsics=intrinsics,
        clip_embedding=dino_feature,
    )


def extract_intrinsics(camera_pose: Any) -> Optional[np.ndarray]:
    if not isinstance(camera_pose, dict):
        return None
    intrinsics = camera_pose.get("intrinsics")
    if not isinstance(intrinsics, Sequence) or len(intrinsics) < 8:
        return None

    fx = float(intrinsics[0])
    fy = float(intrinsics[4])
    cx = float(intrinsics[7])
    cy = float(intrinsics[6])

    scale_x = TARGET_RESOLUTION[0] / ORIGINAL_RESOLUTION[0]
    scale_y = TARGET_RESOLUTION[1] / ORIGINAL_RESOLUTION[1]
    fx_scaled = fx * scale_x
    fy_scaled = fy * scale_y
    cx_scaled = cx * scale_x
    cy_scaled = cy * scale_y

    return np.array(
        [
            [fx_scaled, 0.0, cx_scaled],
            [0.0, fy_scaled, cy_scaled],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def extract_pose(camera_pose: Any) -> Optional[np.ndarray]:
    if not isinstance(camera_pose, dict):
        return None
    transform_values = camera_pose.get("transform")
    if not isinstance(transform_values, Sequence) or len(transform_values) != 16:
        return None

    transform = np.asarray(transform_values, dtype=np.float64).reshape(4, 4, order="F")
    rotation = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    invert = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return transform @ rotation @ invert


def _collect_indexed_files(directory: Path, allowed_suffixes: set[str]) -> Dict[int, Path]:
    output: Dict[int, Path] = {}
    if not directory.exists():
        return output

    for path in directory.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed_suffixes:
            continue
        index = _stem_to_index(path.stem)
        if index is None:
            continue
        output[index] = path
    return output


def _decode_rgb(path: Path) -> Optional[np.ndarray]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _decode_depth(path: Path, width: int, height: int) -> Optional[np.ndarray]:
    expected = int(width) * int(height)
    values = _decode_bin(path, dtype=np.float32, expected_count=expected)
    if values is None:
        return None
    image = values.reshape((height, width))
    return np.rot90(image, k=-1)


def _decode_confidence(path: Path, width: int, height: int) -> Optional[np.ndarray]:
    expected = int(width) * int(height)
    values = _decode_bin(path, dtype=np.uint8, expected_count=expected)
    if values is None:
        return None
    image = values.reshape((height, width))
    return np.rot90(image, k=-1)


def _decode_bin(path: Path, dtype: Any, expected_count: int) -> Optional[np.ndarray]:
    raw = path.read_bytes()
    values = np.frombuffer(raw, dtype=dtype)
    if values.size < expected_count:
        return None
    if values.size > expected_count:
        values = values[:expected_count]
    return values


def _capture_sort_key(path: Path) -> Tuple[int, str]:
    name = path.name
    suffix = name.rsplit("_", 2)
    if len(suffix) >= 2:
        candidate = "_".join(suffix[-2:])
        try:
            dt = datetime.strptime(candidate, "%Y%m%d_%H%M%S")
            return int(dt.timestamp()), name
        except ValueError:
            pass
    return 2**31 - 1, name


def _stem_to_index(stem: str) -> Optional[int]:
    if not stem.isdigit():
        return None
    return int(stem)


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def parse_capture_start_time(capture_name: str) -> Optional[datetime]:
    match = re.search(r"(\d{8}_\d{6})$", capture_name)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def _first_numeric_timestamp(entries: Sequence[Dict[str, Any]]) -> Optional[float]:
    for row in entries:
        value = row.get("timestamp")
        if isinstance(value, (int, float)):
            numeric = float(value)
            if math.isfinite(numeric):
                return numeric
        if isinstance(value, str):
            text = value.strip()
            if not text:
                continue
            try:
                numeric = float(text)
            except ValueError:
                continue
            if math.isfinite(numeric):
                return numeric
    return None


def _normalize_monotonic_timestamps(records: Sequence[CaptureFrameRecord]) -> List[CaptureFrameRecord]:
    output: List[CaptureFrameRecord] = []
    last_timestamp: Optional[datetime] = None
    for row in records:
        timestamp = row.timestamp
        if last_timestamp is not None and timestamp <= last_timestamp:
            timestamp = last_timestamp + timedelta(microseconds=1)
            row = replace(row, timestamp=timestamp, timestamp_seconds=timestamp.timestamp())
        output.append(row)
        last_timestamp = timestamp
    return output
