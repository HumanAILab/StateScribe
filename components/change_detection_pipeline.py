# components/change_detection_pipeline.py
import logging
import threading
import os
import json
from queue import Empty, Queue
from typing import Optional, Set, List, Dict, Any, Callable, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import time
from time import perf_counter

import numpy as np
import cv2
import torch
from ultralytics import FastSAM

from components.memory.frame import Frame
from components.memory.memory_manager import MemoryManager
from components.memory.change_memory import ChangeMemory, ObjectSnapshot, TrackedObject
from components.feature_factory import FeatureFactory
from components.description_composer import ChangeDescriptionComposer
from components.vlm.change_analyzer import ChangeAnalyzer
from components.visualization.display_manager import DisplayManager
from components.logging.frame_logger import FrameLogger, FrameLogRecord
from memo_utils.common import compress_object, decompress_object
from memo_utils.frame_matching import find_and_select_reference_frame
from memo_utils.visibility_utils import compute_visibility_mask, compute_occlusion_mask
from config import (
    FRAME_QUEUE_MAXSIZE,
    VLM_RESULT_QUEUE_MAXSIZE,
    MAX_CONCURRENT_VLM_DIFF,
    FASTSAM_MODEL_PATH,
    FASTSAM_BBOX_SHRINK_RATIO,
    BBOX_3D_OUTLIER_STD_THRESHOLD,
    CHANGE_MEMORY_IOU_MODE,
    CHANGE_MEMORY_IOU_THRESHOLD,
    CHANGE_MEMORY_IOU_BBOX_EXPANSION_RATIO,
    CHANGE_MEMORY_BBOX_WINDOW,
    CHANGE_MEMORY_DINO_SIM_THRESHOLD,
    CHANGE_MEMORY_TEXT_SIM_THRESHOLD,
    CHANGE_DETECTION_ALWAYS_TAG_REFERENCES,
    CHANGE_DETECTION_REPLAY_RECOVERED_HISTORY,
    DIFF_VLM_DISCARD_IF_SLOWER_THAN_SECONDS,
    BBOX_MIN_AREA_RATIO,
    VISIBILITY_MASK_KERNEL_SIZE,
    VISIBILITY_MASK_DOWNSAMPLE_RATIO,
    VISIBILITY_MASK_EDGE_TRIM_RATIO,
    VISIBILITY_MASK_MAX_OUTSIDE_AREA_RATIO,
    OCCLUSION_RELIEF_DEPTH_TOLERANCE,
    OCCLUSION_RELIEF_MIN_OCCLUDED_RATIO,
    MASK_ERODE_KERNEL_SIZE,
    BBOX_OUTLIER_TRIM_RATIO,
    VLM_RAW_BBOX_DEBUG_ENABLED,
    VLM_RAW_BBOX_DEBUG_DIR,
    REJECTED_BBOX_DEBUG_ENABLED,
    REJECTED_BBOX_DEBUG_DIR,
    FRAME_CLUSTER_REQUIRE_LIVE_DESCRIBED_TAG,
    FRAME_MATCHING_BRISQUE_MAX_SCORE,
    LONG_TERM_MEMORY_BASE_PATH,
)

logger = logging.getLogger(__name__)
CHANGE_MEMORY_STATE_FILENAME = "change_memory.state"


def norm_box_to_xyxy(box: list[int], shape: tuple[int, int, int]) -> Optional[List[int]]:
    if len(box) != 4:
        return None
    h, w = shape[:2]
    y1, x1, y2, x2 = [int(v) for v in box]
    y1 = max(0, min(1000, y1))
    x1 = max(0, min(1000, x1))
    y2 = max(0, min(1000, y2))
    x2 = max(0, min(1000, x2))
    if y1 == y2 or x1 == x2:
        return None
    if y1 > y2:
        y1, y2 = y2, y1
    if x1 > x2:
        x1, x2 = x2, x1
    px1 = int(x1 / 1000.0 * w)
    px2 = int(x2 / 1000.0 * w)
    py1 = int(y1 / 1000.0 * h)
    py2 = int(y2 / 1000.0 * h)
    if px1 >= px2 or py1 >= py2:
        return None
    return [px1, py1, px2, py2]


def draw_boxes(
    image: np.ndarray,
    boxes: List[List[int]],
    color: tuple[int, int, int],
    labels: Optional[List[str]] = None
) -> np.ndarray:
    out = image.copy()
    for i, box in enumerate(boxes, 1):
        x1, y1, x2, y2 = box
        label = labels[i - 1] if labels else i
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, str(label), (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def wrap_text(text: str, max_len: int) -> List[str]:
    words = text.split()
    lines = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 <= max_len:
            current = (current + " " + word).strip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def build_description_lines(descriptions: List[str], max_len: int = 60) -> List[str]:
    if not descriptions:
        return ["no confident changes"]
    lines = []
    for desc in descriptions:
        lines.extend(wrap_text(desc, max_len))
    return lines


def draw_text_block(image: np.ndarray, lines: List[str]) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    padding = 10
    line_h = 22
    block_h = padding * 2 + line_h * len(lines)
    y0 = max(0, h - block_h)
    cv2.rectangle(out, (0, y0), (w, h), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        y = y0 + padding + (i + 1) * line_h - 6
        cv2.putText(out, line, (padding, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return out


def run_fastsam(model, image: np.ndarray, boxes: List[List[int]]) -> List[np.ndarray]:
    if not boxes:
        return []
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    shrink_ratio = min(max(float(FASTSAM_BBOX_SHRINK_RATIO), 0.0), 0.99)
    fastsam_boxes = [_shrink_box_for_fastsam(box, image.shape, shrink_ratio) for box in boxes]
    results = model(bgr, bboxes=fastsam_boxes, verbose=False, save=False)
    if not results:
        raise RuntimeError("FastSAM returned no results")
    res = results[0]
    if res.masks is None or res.masks.data is None:
        raise RuntimeError("FastSAM returned no masks")
    masks = res.masks.data
    if hasattr(masks, "cpu"):
        masks = masks.cpu().numpy()
    if masks.shape[0] != len(boxes):
        raise RuntimeError(
            f"FastSAM mask count mismatch: expected {len(boxes)}, got {int(masks.shape[0])}"
        )
    out = []
    for mask in masks:
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        out.append((mask > 0.5).astype(np.uint8) * 255)
    return out


def _shrink_box_for_fastsam(
    box: List[int],
    shape: tuple[int, int, int],
    shrink_ratio: float,
) -> List[int]:
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = shape[:2]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if shrink_ratio <= 0.0 or x2 <= x1 or y2 <= y1:
        return [x1, y1, x2, y2]

    box_w = x2 - x1
    box_h = y2 - y1
    dx = min(int(round(box_w * shrink_ratio * 0.5)), max(0, (box_w - 1) // 2))
    dy = min(int(round(box_h * shrink_ratio * 0.5)), max(0, (box_h - 1) // 2))
    return [x1 + dx, y1 + dy, x2 - dx, y2 - dy]


def box_area_ratio(box: List[int], shape: tuple[int, int, int]) -> float:
    x1, y1, x2, y2 = box
    h, w = shape[:2]
    area = max(0, x2 - x1) * max(0, y2 - y1)
    return area / float(max(1, w * h))


def outside_mask_ratio(box: List[int], mask: np.ndarray) -> float:
    x1, y1, x2, y2 = box
    h, w = mask.shape[:2]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return 1.0
    region = mask[y1:y2, x1:x2]
    total = region.size
    if total == 0:
        return 1.0
    inside = int(np.count_nonzero(region > 0))
    outside = total - inside
    return outside / float(total)


def erode_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.erode(mask, kernel, iterations=1)


def confidence_short_label(value: Any) -> str:
    confidence = str(value or "").strip()
    if confidence == "low":
        return "L"
    if confidence == "high":
        return "H"
    if confidence == "med":
        return "M"
    return "?"




def format_change_descriptions(change_items: List[Dict[str, Any]]) -> List[str]:
    descriptions = []
    for idx, change in enumerate(change_items, 1):
        obj = change.get("object_description", "").strip()
        desc = change.get("change_description", "").strip()
        conf = str(change.get("confidence", "")).strip()
        text = f"{idx}. [{conf}]"
        if obj and desc:
            text = f"{idx}. [{conf}] {obj}: {desc}"
        elif desc:
            text = f"{idx}. [{conf}] {desc}"
        elif obj:
            text = f"{idx}. [{conf}] {obj}"
        descriptions.append(text)
    return descriptions


class ChangeDetectionPipeline(threading.Thread):
    """
    Pipeline stages:
    1. Frame matching + parallel VLM diff detection
    2. FastSAM segmentation + memory update (sequential)
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        display_manager: Optional[DisplayManager] = None,
        on_description: Optional[Callable[[Any], Any]] = None,
        on_change_snapshot: Optional[Callable[[Dict[str, Any]], Any]] = None,
        feature_factory: Optional[FeatureFactory] = None,
        use_ai_paraphrase: bool = False,
    ):
        super().__init__(daemon=True)
        self.memory_manager = memory_manager
        self.change_memory = ChangeMemory()
        self.description_composer = ChangeDescriptionComposer(
            self.change_memory,
            latest_pose_matrix_getter=self._get_latest_pose_matrix,
        )
        self.display_manager = display_manager
        self.on_description = on_description
        self.on_change_snapshot = on_change_snapshot
        self.feature_factory = feature_factory if feature_factory is not None else FeatureFactory()
        self._owns_feature_factory = feature_factory is None
        self.use_ai_paraphrase = bool(use_ai_paraphrase)

        self.frame_queue = Queue(maxsize=FRAME_QUEUE_MAXSIZE)
        self.vlm_result_queue = Queue(maxsize=VLM_RESULT_QUEUE_MAXSIZE)

        self.change_analyzer = ChangeAnalyzer()
        self.seg_model = FastSAM(FASTSAM_MODEL_PATH)
        self.frame_logger = FrameLogger()

        self.described_frames: Set[datetime] = set()
        self.live_described_frames: Set[datetime] = set()
        self.processed_frames: Set[datetime] = set()
        self.current_world_name: Optional[str] = None
        self.change_state_path: Optional[str] = None
        self._state_lock = threading.Lock()
        self._queued_frame_timestamps: Set[datetime] = set()
        self._inflight_frame_timestamps: Set[datetime] = set()
        self._inflight_reference_counts: Dict[datetime, int] = {}
        self._frame_reference_tags: Dict[datetime, datetime] = {}
        self._frame_ingress_walls: Dict[datetime, datetime] = {}
        self._frame_ingress_perfs: Dict[datetime, float] = {}
        self._frame_enqueue_walls: Dict[datetime, datetime] = {}
        self._frame_enqueue_perfs: Dict[datetime, float] = {}
        self._recovery_queued_for_world = False
        self._paused = False
        self.running = False

        self.vlm_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_VLM_DIFF)

        logger.debug("ChangeDetectionPipeline initialized")

    def set_paused(self, paused: bool) -> None:
        next_state = bool(paused)
        if self._paused == next_state:
            return
        self._paused = next_state
        logger.debug("ChangeDetectionPipeline %s for VQA", "paused" if next_state else "resumed")

    def _get_latest_pose_matrix(self) -> Optional[np.ndarray]:
        frames = self.memory_manager.get_memory()
        if not frames:
            return None
        latest = frames[-1]
        pose = getattr(latest, "pose_matrix", None)
        if pose is None:
            return None
        arr = np.asarray(pose, dtype=np.float64)
        if arr.shape != (4, 4):
            return None
        return arr

    @staticmethod
    def _timestamps_to_iso(values: Set[datetime]) -> List[str]:
        return sorted(ts.isoformat() for ts in values)

    @staticmethod
    def _iso_to_timestamps(values: List[str]) -> Set[datetime]:
        return {datetime.fromisoformat(str(item)) for item in values}

    def _ensure_world_state(self, world_name: Optional[str]) -> None:
        clean_world = (world_name or "").strip()
        if not clean_world:
            return

        with self._state_lock:
            if self.current_world_name == clean_world:
                return

            if self.current_world_name:
                self._save_state_locked()

            self.current_world_name = clean_world
            self.change_state_path = os.path.join(
                LONG_TERM_MEMORY_BASE_PATH,
                clean_world,
                CHANGE_MEMORY_STATE_FILENAME,
            )
            self.change_memory.clear()
            self.described_frames.clear()
            self.live_described_frames.clear()
            self.processed_frames.clear()
            self._queued_frame_timestamps.clear()
            self._inflight_frame_timestamps.clear()
            self._inflight_reference_counts.clear()
            self._frame_reference_tags.clear()
            self._frame_ingress_walls.clear()
            self._frame_ingress_perfs.clear()
            self._frame_enqueue_walls.clear()
            self._frame_enqueue_perfs.clear()
            self._recovery_queued_for_world = False
            self._load_state_locked()
            if self.display_manager is not None:
                self.display_manager.update_memory_display(self.change_memory)

    def _load_state_locked(self) -> None:
        if not self.change_state_path or not os.path.exists(self.change_state_path):
            return

        with open(self.change_state_path, "rb") as f:
            payload = decompress_object(f.read())

        self.change_memory.load_from_persist_dict(payload["change_memory"])
        self.described_frames = self._iso_to_timestamps(payload["described_frames"])
        self.live_described_frames = self._iso_to_timestamps(payload["live_described_frames"])
        self.processed_frames = self._iso_to_timestamps(payload["processed_frames"])

    def _save_state_locked(self) -> None:
        if not self.current_world_name or not self.change_state_path:
            return

        os.makedirs(os.path.dirname(self.change_state_path), exist_ok=True)
        payload = {
            "world_name": self.current_world_name,
            "change_memory": self.change_memory.to_persist_dict(),
            "described_frames": self._timestamps_to_iso(self.described_frames),
            "live_described_frames": self._timestamps_to_iso(self.live_described_frames),
            "processed_frames": self._timestamps_to_iso(self.processed_frames),
        }
        compressed = compress_object(payload)
        temp_path = f"{self.change_state_path}.tmp"
        with open(temp_path, "wb") as f:
            f.write(compressed)
        os.replace(temp_path, self.change_state_path)

    def _queue_frame_locked(
        self,
        frame: Frame,
        ingress_wall: Optional[datetime] = None,
        ingress_perf: Optional[float] = None,
    ) -> None:
        timestamp = frame.timestamp
        if timestamp is None:
            return
        if timestamp in self.processed_frames:
            return
        if timestamp in self._queued_frame_timestamps:
            return
        if timestamp in self._inflight_frame_timestamps:
            return

        self.frame_queue.put_nowait(frame)
        if isinstance(ingress_wall, datetime):
            self._frame_ingress_walls.setdefault(timestamp, ingress_wall)
        if isinstance(ingress_perf, (int, float)):
            self._frame_ingress_perfs.setdefault(timestamp, float(ingress_perf))
        self._frame_enqueue_walls[timestamp] = datetime.now()
        self._frame_enqueue_perfs[timestamp] = perf_counter()
        self._queued_frame_timestamps.add(timestamp)

    def _enqueue_unprocessed_frames_locked(self) -> None:
        candidates = [
            frame
            for frame in self.memory_manager.get_memory()
            if frame.timestamp is not None and frame.timestamp not in self.processed_frames
        ]
        candidates.sort(key=lambda item: item.timestamp)
        for frame in candidates:
            self._queue_frame_locked(frame)

    def _mark_frame_inflight(self, timestamp: Optional[datetime]) -> None:
        if timestamp is None:
            return
        with self._state_lock:
            self._queued_frame_timestamps.discard(timestamp)
            self._inflight_frame_timestamps.add(timestamp)

    def _register_pending_reference(self, frame_timestamp: Optional[datetime], reference_timestamp: Optional[datetime]) -> None:
        if frame_timestamp is None or reference_timestamp is None:
            return
        with self._state_lock:
            previous_ref = self._frame_reference_tags.get(frame_timestamp)
            if previous_ref is not None:
                prev_count = self._inflight_reference_counts.get(previous_ref, 0)
                if prev_count <= 1:
                    self._inflight_reference_counts.pop(previous_ref, None)
                else:
                    self._inflight_reference_counts[previous_ref] = prev_count - 1

            self._frame_reference_tags[frame_timestamp] = reference_timestamp
            self._inflight_reference_counts[reference_timestamp] = self._inflight_reference_counts.get(reference_timestamp, 0) + 1

    def _complete_frame_processing(
        self,
        timestamp: Optional[datetime],
        keep_reference_tag: bool = True,
    ) -> None:
        if timestamp is None:
            return
        with self._state_lock:
            self.processed_frames.add(timestamp)
            self._queued_frame_timestamps.discard(timestamp)
            self._inflight_frame_timestamps.discard(timestamp)
            self._frame_ingress_walls.pop(timestamp, None)
            self._frame_ingress_perfs.pop(timestamp, None)
            self._frame_enqueue_walls.pop(timestamp, None)
            self._frame_enqueue_perfs.pop(timestamp, None)

            reference_timestamp = self._frame_reference_tags.pop(timestamp, None)
            if reference_timestamp is not None:
                if keep_reference_tag:
                    self.described_frames.add(reference_timestamp)
                ref_count = self._inflight_reference_counts.get(reference_timestamp, 0)
                if ref_count <= 1:
                    self._inflight_reference_counts.pop(reference_timestamp, None)
                else:
                    self._inflight_reference_counts[reference_timestamp] = ref_count - 1

            self._save_state_locked()

    def add_frame(
        self,
        frame: Frame,
        ingress_wall: Optional[datetime] = None,
        ingress_perf: Optional[float] = None,
    ):
        if self._paused:
            return
        self._ensure_world_state(frame.world_name)
        with self._state_lock:
            if not self._recovery_queued_for_world:
                if CHANGE_DETECTION_REPLAY_RECOVERED_HISTORY:
                    self._enqueue_unprocessed_frames_locked()
                self._recovery_queued_for_world = True
            self._queue_frame_locked(
                frame,
                ingress_wall=ingress_wall,
                ingress_perf=ingress_perf,
            )
        logger.debug(f"Frame {frame.timestamp} added to pipeline")

    def mark_live_described_frame(self, timestamp: Optional[datetime]) -> None:
        if timestamp is None:
            return
        self._ensure_world_state(self.memory_manager.current_world_name)
        with self._state_lock:
            self.live_described_frames.add(timestamp)
            self._save_state_locked()
        logger.debug("Marked live-described frame: %s", timestamp)

    def run(self):
        self.running = True
        logger.debug("ChangeDetectionPipeline started")

        vlm_thread = threading.Thread(target=self._vlm_detection_stage, daemon=True)
        memory_thread = threading.Thread(target=self._segmentation_memory_stage, daemon=True)

        vlm_thread.start()
        memory_thread.start()

        vlm_thread.join()
        memory_thread.join()

        if self._owns_feature_factory:
            self.feature_factory.close()
        logger.debug("ChangeDetectionPipeline stopped")

    def stop(self):
        self.running = False
        self.vlm_executor.shutdown(wait=False, cancel_futures=False)
        with self._state_lock:
            self._save_state_locked()

    def _run_vlm_with_timing(self, ref_image: np.ndarray, cur_image: np.ndarray) -> Dict[str, Any]:
        start_wall = datetime.now()
        start_perf = perf_counter()
        result = self.change_analyzer.detect_changes(ref_image, cur_image)
        end_perf = perf_counter()
        end_wall = datetime.now()
        duration_s = end_perf - start_perf
        return {
            "result": result,
            "start_wall": start_wall,
            "end_wall": end_wall,
            "duration_s": duration_s,
            "start_perf": start_perf,
            "end_perf": end_perf,
        }

    def _vlm_detection_stage(self):
        pending: Dict[Any, Dict[str, Any]] = {}

        while self.running:
            if self.frame_queue.empty():
                time.sleep(0.1)
                frame = None
                dequeue_wall = None
                dequeue_perf = None
            else:
                frame = self.frame_queue.get_nowait()
                dequeue_wall = datetime.now()
                dequeue_perf = perf_counter()

            if frame is not None:
                self._ensure_world_state(frame.world_name)
                ingress_wall = None
                ingress_perf = None
                enqueue_wall = None
                enqueue_perf = None
                with self._state_lock:
                    if frame.timestamp is not None:
                        self._queued_frame_timestamps.discard(frame.timestamp)
                        self._inflight_frame_timestamps.add(frame.timestamp)
                        ingress_wall = self._frame_ingress_walls.get(frame.timestamp)
                        ingress_perf = self._frame_ingress_perfs.get(frame.timestamp)
                        enqueue_wall = self._frame_enqueue_walls.get(frame.timestamp)
                        enqueue_perf = self._frame_enqueue_perfs.get(frame.timestamp)
                    described_snapshot = set(self.described_frames)
                    described_snapshot.update(self._inflight_reference_counts.keys())
                    live_described_snapshot = set(self.live_described_frames)

                log_record = FrameLogRecord(
                    frame_timestamp=frame.timestamp,
                    world_name=frame.world_name,
                    frame_start_wall=dequeue_wall or datetime.now(),
                )
                if isinstance(dequeue_perf, (int, float)):
                    log_record._frame_start_perf = float(dequeue_perf)
                log_record.ingress_wall = ingress_wall
                log_record.pipeline_enqueue_wall = enqueue_wall
                log_record.pipeline_dequeue_wall = dequeue_wall
                if isinstance(ingress_perf, (int, float)):
                    log_record._ingress_perf = float(ingress_perf)
                if isinstance(ingress_perf, (int, float)) and isinstance(enqueue_perf, (int, float)):
                    pre_pipeline = max(0.0, float(enqueue_perf) - float(ingress_perf))
                    log_record.pre_pipeline_duration_s = pre_pipeline
                    log_record.set_stage_timing(
                        "ingress_to_enqueue",
                        ingress_wall,
                        enqueue_wall,
                        pre_pipeline,
                    )
                if isinstance(enqueue_perf, (int, float)) and isinstance(dequeue_perf, (int, float)):
                    frame_queue_wait = max(0.0, float(dequeue_perf) - float(enqueue_perf))
                    log_record.frame_queue_wait_s = frame_queue_wait
                    log_record.set_stage_timing(
                        "frame_queue_wait",
                        enqueue_wall,
                        dequeue_wall,
                        frame_queue_wait,
                    )
                log_record.start_stage("frame_matching")
                ltm_frames = self.memory_manager.get_memory()
                reference_frame = find_and_select_reference_frame(
                    frame,
                    ltm_frames,
                    described_frames=described_snapshot,
                    live_described_frames=live_described_snapshot,
                    require_live_described_cluster=FRAME_CLUSTER_REQUIRE_LIVE_DESCRIBED_TAG,
                    brisque_max_score=FRAME_MATCHING_BRISQUE_MAX_SCORE,
                )
                log_record.end_stage("frame_matching")
                log_record.reference_found = reference_frame is not None
                log_record.reference_timestamp = reference_frame.timestamp if reference_frame else None
                log_record.notes["ltm_frame_count"] = len(ltm_frames)
                log_record.notes["live_described_frame_count"] = len(live_described_snapshot)

                if reference_frame is None:
                    log_record.final_description = "no reference frame found"
                    log_record.change_memory_snapshot = self.change_memory.to_dict()
                    self.frame_logger.write(log_record)
                    self._complete_frame_processing(frame.timestamp)
                    continue

                if (
                    not FRAME_CLUSTER_REQUIRE_LIVE_DESCRIBED_TAG
                    and reference_frame is not None
                    and reference_frame.timestamp is not None
                ):
                    self._register_pending_reference(
                        frame.timestamp,
                        reference_frame.timestamp,
                    )
                future = self.vlm_executor.submit(
                    self._run_vlm_with_timing,
                    reference_frame.rgb_image,
                    frame.rgb_image
                )
                pending[future] = {
                    "frame": frame,
                    "reference_frame": reference_frame,
                    "log_record": log_record,
                }

            if not pending:
                continue

            done, _ = wait(pending.keys(), timeout=0.0, return_when=FIRST_COMPLETED)
            for future in done:
                data = pending.pop(future)
                log_record = data.get("log_record")
                result_payload = future.result()
                frame = data.get("frame")

                if log_record is not None and isinstance(result_payload, dict):
                    log_record.set_stage_timing(
                        "vlm_call",
                        result_payload.get("start_wall"),
                        result_payload.get("end_wall"),
                        result_payload.get("duration_s"),
                    )

                vlm_duration_s = result_payload.get("duration_s") if isinstance(result_payload, dict) else None
                slow_vlm_threshold_s = float(DIFF_VLM_DISCARD_IF_SLOWER_THAN_SECONDS)
                if (
                    log_record is not None
                    and frame is not None
                    and slow_vlm_threshold_s > 0.0
                    and isinstance(vlm_duration_s, (int, float))
                    and float(vlm_duration_s) > slow_vlm_threshold_s
                ):
                    log_record.final_description = (
                        f"discarded: vlm call exceeded {slow_vlm_threshold_s:.1f}s"
                    )
                    log_record.vlm_raw = (
                        result_payload.get("result") if isinstance(result_payload, dict) else result_payload
                    )
                    log_record.notes["vlm_discard_threshold_s"] = slow_vlm_threshold_s
                    log_record.change_memory_snapshot = self.change_memory.to_dict()
                    self.frame_logger.write(log_record)
                    self._complete_frame_processing(
                        frame.timestamp,
                        keep_reference_tag=CHANGE_DETECTION_ALWAYS_TAG_REFERENCES,
                    )
                    continue

                data["vlm_result"] = result_payload.get("result") if isinstance(result_payload, dict) else result_payload
                data["timestamp"] = datetime.now()
                data["vlm_done_wall"] = result_payload.get("end_wall") if isinstance(result_payload, dict) else None
                data["vlm_done_perf"] = result_payload.get("end_perf") if isinstance(result_payload, dict) else None
                data["log_record"] = log_record
                self.vlm_result_queue.put(data)

    def _segmentation_memory_stage(self):
        while self.running:
            try:
                item = self.vlm_result_queue.get(timeout=0.1)
            except Empty:
                continue
            segmentation_start_wall = datetime.now()
            segmentation_start_perf = perf_counter()

            frame = item["frame"]
            reference_frame = item["reference_frame"]
            result = item["vlm_result"]

            log_record = item.get("log_record")
            if log_record is None:
                log_record = FrameLogRecord(
                    frame_timestamp=frame.timestamp,
                    world_name=frame.world_name
                )
                log_record.reference_found = reference_frame is not None
                log_record.reference_timestamp = reference_frame.timestamp if reference_frame else None

            vlm_done_wall = item.get("vlm_done_wall")
            vlm_done_perf = item.get("vlm_done_perf")
            if isinstance(vlm_done_perf, (int, float)):
                result_queue_wait = max(0.0, float(segmentation_start_perf) - float(vlm_done_perf))
                log_record.vlm_result_queue_wait_s = result_queue_wait
                log_record.set_stage_timing(
                    "vlm_result_queue_wait",
                    vlm_done_wall if isinstance(vlm_done_wall, datetime) else None,
                    segmentation_start_wall,
                    result_queue_wait,
                )

            log_record.vlm_raw = result
            changes_raw = result.get("changes", []) if isinstance(result, dict) else []
            log_record.notes["vlm_change_count"] = len(changes_raw)
            if not changes_raw:
                log_record.final_description = "no changes from vlm"
                log_record.change_memory_snapshot = self.change_memory.to_dict()
                self.frame_logger.write(log_record)
                self._complete_frame_processing(
                    frame.timestamp,
                    keep_reference_tag=CHANGE_DETECTION_ALWAYS_TAG_REFERENCES,
                )
                continue

            prepare_trace: List[Dict[str, Any]] = []
            log_record.start_stage("prepare_changes")
            change_items = self._prepare_changes(
                changes_raw,
                reference_frame.rgb_image.shape,
                frame.rgb_image.shape,
                trace=prepare_trace
            )
            log_record.end_stage("prepare_changes")
            log_record.prepared_changes = prepare_trace

            if not change_items:
                log_record.final_description = "no valid changes after normalization"
                log_record.change_memory_snapshot = self.change_memory.to_dict()
                self.frame_logger.write(log_record)
                self._complete_frame_processing(
                    frame.timestamp,
                    keep_reference_tag=CHANGE_DETECTION_ALWAYS_TAG_REFERENCES,
                )
                continue

            log_record.notes["confidence_counts_raw"] = self._confidence_counts(change_items)
            vlm_view_t0, vlm_view_t1 = self._build_vlm_views(
                reference_frame.rgb_image,
                frame.rgb_image,
                change_items,
            )

            need_t0 = any(c["bbox_t0"] is not None for c in change_items)
            need_t1 = any(c["bbox_t1"] is not None for c in change_items)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            vis_mask_t0 = None
            vis_mask_t1 = None
            occlusion_mask = None
            has_appear = any(c.get("change_type") == "appear" for c in change_items)
            log_record.start_stage("visibility_masks")
            if need_t0:
                vis_mask_t0 = compute_visibility_mask(
                    frame,
                    reference_frame,
                    device,
                    VISIBILITY_MASK_KERNEL_SIZE,
                    VISIBILITY_MASK_DOWNSAMPLE_RATIO,
                    edge_trim_ratio=VISIBILITY_MASK_EDGE_TRIM_RATIO,
                )
            if need_t1:
                vis_mask_t1 = compute_visibility_mask(
                    reference_frame,
                    frame,
                    device,
                    VISIBILITY_MASK_KERNEL_SIZE,
                    VISIBILITY_MASK_DOWNSAMPLE_RATIO,
                    edge_trim_ratio=VISIBILITY_MASK_EDGE_TRIM_RATIO,
                )
            if has_appear:
                occlusion_mask = compute_occlusion_mask(
                    frame,
                    reference_frame,
                    device,
                    VISIBILITY_MASK_DOWNSAMPLE_RATIO,
                    depth_tolerance=OCCLUSION_RELIEF_DEPTH_TOLERANCE,
                )
            log_record.end_stage("visibility_masks")
            log_record.notes["visibility_masks"] = {
                "need_t0": need_t0,
                "need_t1": need_t1,
                "has_appear": has_appear,
                "device": device,
            }

            self._save_vlm_raw_bbox_debug(
                reference_frame.rgb_image,
                frame.rgb_image,
                change_items,
                reference_frame.timestamp,
                frame.timestamp,
                vis_mask_t0,
                vis_mask_t1,
                occlusion_mask=occlusion_mask,
            )

            filter_trace: List[Dict[str, Any]] = []
            log_record.start_stage("filter_changes")
            change_items = self._filter_changes(
                change_items,
                reference_frame.rgb_image.shape,
                frame.rgb_image.shape,
                vis_mask_t0,
                vis_mask_t1,
                occlusion_mask=occlusion_mask,
                trace=filter_trace,
            )
            log_record.end_stage("filter_changes")
            log_record.filter_trace = filter_trace
            self._save_rejected_bbox_debug(
                reference_frame.rgb_image,
                frame.rgb_image,
                filter_trace,
                reference_frame.timestamp,
                frame.timestamp,
                vis_mask_t0,
                vis_mask_t1,
            )
            log_record.notes["confidence_counts_after_gate"] = self._confidence_counts(change_items)
            if not change_items:
                log_record.final_description = "no changes after filtering"
                if self.display_manager:
                    self.display_manager.update_detection_result(
                        current_image=frame.rgb_image,
                        reference_image=reference_frame.rgb_image,
                        mask_t1=np.zeros(frame.rgb_image.shape[:2], dtype=np.uint8),
                        mask_t0=np.zeros(reference_frame.rgb_image.shape[:2], dtype=np.uint8),
                        description=log_record.final_description,
                        timestamp=frame.timestamp,
                        change_memory=self.change_memory,
                        vlm_annotated_t0=vlm_view_t0,
                        vlm_annotated_t1=vlm_view_t1,
                    )
                log_record.change_memory_snapshot = self.change_memory.to_dict()
                self.frame_logger.write(log_record)
                self._complete_frame_processing(
                    frame.timestamp,
                    keep_reference_tag=CHANGE_DETECTION_ALWAYS_TAG_REFERENCES,
                )
                continue

            boxes_t0 = [(idx, c["bbox_t0"]) for idx, c in enumerate(change_items) if c["bbox_t0"] is not None]
            boxes_t1 = [(idx, c["bbox_t1"]) for idx, c in enumerate(change_items) if c["bbox_t1"] is not None]

            box_list_t0 = [box for _, box in boxes_t0]
            box_list_t1 = [box for _, box in boxes_t1]

            log_record.start_stage("snapshot_features")
            self._attach_snapshot_features(
                change_items=change_items,
                ref_image=reference_frame.rgb_image,
                cur_image=frame.rgb_image,
            )
            log_record.end_stage("snapshot_features")

            log_record.start_stage("fastsam_segmentation")
            try:
                masks_t0 = run_fastsam(self.seg_model, reference_frame.rgb_image, box_list_t0)
                masks_t1 = run_fastsam(self.seg_model, frame.rgb_image, box_list_t1)
            except Exception as exc:
                log_record.end_stage("fastsam_segmentation")
                error_message = str(exc).strip() or exc.__class__.__name__
                logger.warning(
                    "FastSAM segmentation failed; skipping frame. frame=%s reference=%s error=%s",
                    frame.timestamp.isoformat() if frame.timestamp else "n/a",
                    reference_frame.timestamp.isoformat() if reference_frame.timestamp else "n/a",
                    error_message,
                )
                log_record.errors.append(error_message)
                log_record.notes["fastsam_error"] = error_message
                log_record.final_description = "fastsam segmentation failed; skipped frame"
                log_record.change_memory_snapshot = self.change_memory.to_dict()
                self.frame_logger.write(log_record)
                self._complete_frame_processing(
                    frame.timestamp,
                    keep_reference_tag=False,
                )
                continue

            for (change_idx, _), mask in zip(boxes_t0, masks_t0):
                change_items[change_idx]["mask_t0"] = erode_mask(mask, MASK_ERODE_KERNEL_SIZE)

            for (change_idx, _), mask in zip(boxes_t1, masks_t1):
                change_items[change_idx]["mask_t1"] = erode_mask(mask, MASK_ERODE_KERNEL_SIZE)
            log_record.end_stage("fastsam_segmentation")

            log_record.start_stage("compute_3d_bboxes")
            mask_t0 = np.zeros(reference_frame.rgb_image.shape[:2], dtype=np.uint8)
            mask_t1 = np.zeros(frame.rgb_image.shape[:2], dtype=np.uint8)

            for change in change_items:
                if change["mask_t0"] is not None:
                    mask_t0 = np.maximum(mask_t0, change["mask_t0"])
                if change["mask_t1"] is not None:
                    mask_t1 = np.maximum(mask_t1, change["mask_t1"])

                if change["mask_t0"] is not None:
                    change["bbox_3d_t0"] = self._compute_3d_bbox(
                        change["mask_t0"],
                        reference_frame.depth_map_original,
                        reference_frame.intrinsics,
                        reference_frame.pose_matrix,
                        BBOX_OUTLIER_TRIM_RATIO
                    )
                if change["mask_t1"] is not None:
                    change["bbox_3d_t1"] = self._compute_3d_bbox(
                        change["mask_t1"],
                        frame.depth_map_original,
                        frame.intrinsics,
                        frame.pose_matrix,
                        BBOX_OUTLIER_TRIM_RATIO
                    )
            log_record.end_stage("compute_3d_bboxes")

            memory_trace: List[Dict[str, Any]] = []
            log_record.start_stage("update_change_memory")
            kept_changes = self._update_change_memory(
                change_items,
                reference_frame.timestamp,
                frame.timestamp,
                reference_frame.rgb_image,
                frame.rgb_image,
                trace=memory_trace
            )
            log_record.end_stage("update_change_memory")
            log_record.memory_actions = memory_trace
            log_record.kept_changes = self._summarize_changes_for_log(kept_changes)
            if not kept_changes:
                log_record.final_description = "no changes kept after memory update"
                log_record.change_memory_snapshot = self.change_memory.to_dict()
                self.frame_logger.write(log_record)
                self._complete_frame_processing(frame.timestamp)
                continue

            description = ""
            if self.use_ai_paraphrase and self.on_change_snapshot is not None:
                self._emit_change_snapshot(
                    self._build_change_snapshot_payload(
                        kept_changes,
                        reference_frame.timestamp,
                        frame.timestamp,
                    )
                )
                description = self._build_ai_mode_description(kept_changes)
                log_record.notes["description_mode"] = "ai_paraphrase"
            else:
                description = self.description_composer.build(
                    kept_changes,
                    reference_frame.timestamp,
                    frame.timestamp,
                )
                if not description:
                    log_record.final_description = "no user description after memory update"
                    log_record.change_memory_snapshot = self.change_memory.to_dict()
                    self.frame_logger.write(log_record)
                    self._complete_frame_processing(frame.timestamp)
                    continue

                self._emit_description(
                    immediate_text=description,
                    deferred_payload=lambda changes=tuple(kept_changes), ref_ts=reference_frame.timestamp, cur_ts=frame.timestamp: self.description_composer.build(
                        list(changes),
                        ref_ts,
                        cur_ts,
                    ),
                )
                log_record.notes["description_mode"] = "legacy"

            timestamp = frame.timestamp
            log_record.final_description = description

            log_record.start_stage("display_update")
            if self.display_manager:
                self.display_manager.update_detection_result(
                    current_image=frame.rgb_image,
                    reference_image=reference_frame.rgb_image,
                    mask_t1=mask_t1,
                    mask_t0=mask_t0,
                    description=description,
                    timestamp=timestamp,
                    change_memory=self.change_memory,
                    vlm_annotated_t0=vlm_view_t0,
                    vlm_annotated_t1=vlm_view_t1
                )
            log_record.end_stage("display_update")
            log_record.change_memory_snapshot = self.change_memory.to_dict()
            self.frame_logger.write(log_record)
            self._complete_frame_processing(frame.timestamp)

    @staticmethod
    def _serialize_snapshot_for_ai(
        snapshot: Optional[ObjectSnapshot],
        hide_change_description: bool = False,
    ) -> Dict[str, Any]:
        if snapshot is None:
            return {}

        object_description = str(getattr(snapshot, "object_description", "") or "").strip()
        if not object_description:
            object_description = str(snapshot.description or "").strip()

        change_description = ""
        if not hide_change_description:
            change_description = str(getattr(snapshot, "change_description", "") or "").strip()
            if not change_description:
                change_description = str(snapshot.description or "").strip()

        return {
            "change_type": snapshot.change_type,
            "timestamp": snapshot.timestamp.isoformat() if snapshot.timestamp else "",
            "object_description": object_description,
            "change_description": change_description,
            "context_description": snapshot.context_description,
        }

    @staticmethod
    def _normalize_bbox_3d_for_ai(raw: Any) -> Optional[Tuple[float, float, float, float, float, float]]:
        if not isinstance(raw, (list, tuple)) or len(raw) != 6:
            return None
        try:
            x1, y1, z1, x2, y2, z2 = [float(v) for v in raw]
        except Exception:
            return None
        if x2 <= x1 or y2 <= y1 or z2 <= z1:
            return None
        return (x1, y1, z1, x2, y2, z2)

    @classmethod
    def _bbox_3d_overlap_for_ai(cls, raw_a: Any, raw_b: Any) -> bool:
        box_a = cls._normalize_bbox_3d_for_ai(raw_a)
        box_b = cls._normalize_bbox_3d_for_ai(raw_b)
        if box_a is None or box_b is None:
            return False

        ax1, ay1, az1, ax2, ay2, az2 = box_a
        bx1, by1, bz1, bx2, by2, bz2 = box_b
        return (
            max(ax1, bx1) < min(ax2, bx2)
            and max(ay1, by1) < min(ay2, by2)
            and max(az1, bz1) < min(az2, bz2)
        )

    @classmethod
    def _collapse_replaced_change_for_ai(
        cls,
        serialized_changes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if len(serialized_changes) != 2:
            return serialized_changes

        appear_row = next((row for row in serialized_changes if str(row.get("change_type", "")).strip() == "appear"), None)
        disappear_row = next((row for row in serialized_changes if str(row.get("change_type", "")).strip() == "disappear"), None)
        if appear_row is None or disappear_row is None:
            return serialized_changes

        overlap = cls._bbox_3d_overlap_for_ai(disappear_row.get("bbox_3d_t0"), appear_row.get("bbox_3d_t1"))
        if not overlap:
            return serialized_changes

        old_obj = str(disappear_row.get("object_description", "")).strip()
        new_obj = str(appear_row.get("object_description", "")).strip()
        context = str(appear_row.get("context_description", "")).strip() or str(disappear_row.get("context_description", "")).strip()
        if old_obj and new_obj:
            change_description = f"{old_obj} was replaced by {new_obj}."
        elif new_obj:
            change_description = f"A previous object was replaced by {new_obj}."
        elif old_obj:
            change_description = f"{old_obj} was replaced by a different object."
        else:
            change_description = "One object was replaced by another in the same location."

        previous_snapshot = disappear_row.get("previous_snapshot") or disappear_row.get("current_snapshot") or {}
        current_snapshot = appear_row.get("current_snapshot") or {}
        object_description = new_obj or old_obj
        object_id = str(appear_row.get("object_id", "") or disappear_row.get("object_id", "")).strip()

        return [{
            "object_id": object_id,
            "change_type": "replaced",
            "object_description": object_description,
            "change_description": change_description,
            "context_description": context,
            "bbox_3d_t0": disappear_row.get("bbox_3d_t0"),
            "bbox_3d_t1": appear_row.get("bbox_3d_t1"),
            "current_snapshot": current_snapshot,
            "previous_snapshot": previous_snapshot,
        }]

    def _build_change_snapshot_payload(
        self,
        kept_changes: List[Dict[str, Any]],
        ref_timestamp: datetime,
        cur_timestamp: datetime,
    ) -> Dict[str, Any]:
        serialized_changes: List[Dict[str, Any]] = []
        for change in kept_changes:
            object_id = change.get("object_id")
            object_id_text = object_id if isinstance(object_id, str) else ""

            current_snapshot = {}
            previous_snapshot = {}
            if object_id_text:
                tracked = self.change_memory.get_object(object_id_text)
                latest = tracked.get_latest_snapshot() if tracked else None
                previous = tracked.snapshots[-2] if tracked and len(tracked.snapshots) >= 2 else None
                current_snapshot = self._serialize_snapshot_for_ai(
                    latest,
                    hide_change_description=False,
                )
                previous_snapshot = self._serialize_snapshot_for_ai(
                    previous,
                    hide_change_description=True,
                )

            serialized_changes.append(
                {
                    "object_id": object_id_text,
                    "change_type": str(change.get("change_type", "")),
                    "object_description": str(change.get("object_description", "")),
                    "change_description": str(change.get("change_description", "")),
                    "context_description": str(change.get("context_description", "")),
                    "bbox_3d_t0": change.get("bbox_3d_t0"),
                    "bbox_3d_t1": change.get("bbox_3d_t1"),
                    "current_snapshot": current_snapshot,
                    "previous_snapshot": previous_snapshot,
                }
            )

        serialized_changes = self._collapse_replaced_change_for_ai(serialized_changes)

        return {
            "reference_timestamp": ref_timestamp.isoformat() if ref_timestamp else "",
            "current_timestamp": cur_timestamp.isoformat() if cur_timestamp else "",
            "changes": serialized_changes,
        }

    @staticmethod
    def _build_ai_mode_description(kept_changes: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for change in kept_changes:
            change_type = str(change.get("change_type", "")).strip()
            obj = str(change.get("object_description", "")).strip()
            desc = str(change.get("change_description", "")).strip()
            context = str(change.get("context_description", "")).strip()
            context_suffix = f" (around {context})" if context else ""
            if obj and desc:
                parts.append(f"{obj}: {desc}{context_suffix}")
            elif desc:
                parts.append(f"{desc}{context_suffix}")
            elif obj:
                parts.append(f"{obj} {change_type}".strip() + context_suffix)
            elif change_type:
                parts.append(f"{change_type}{context_suffix}")
        if not parts:
            return "change snapshot queued"
        return " | ".join(parts)

    def _emit_change_snapshot(self, payload: Dict[str, Any]) -> None:
        if not self.on_change_snapshot:
            return
        try:
            self.on_change_snapshot(payload)
        except Exception:
            logger.exception("Failed to emit change snapshot.")

    def _emit_description(
        self,
        immediate_text: str,
        deferred_payload: Optional[Callable[[], str]] = None,
    ) -> None:
        if not self.on_description:
            return
        payload: Any = immediate_text
        callback_target = getattr(self.on_description, "__self__", None)
        if deferred_payload is not None and getattr(callback_target, "accepts_deferred_payload", False):
            payload = deferred_payload
        self.on_description(payload)

    @staticmethod
    def _cosine_similarity(vec_a: Optional[np.ndarray], vec_b: Optional[np.ndarray]) -> Optional[float]:
        if vec_a is None or vec_b is None:
            return None
        a = np.asarray(vec_a, dtype=np.float32).reshape(-1)
        b = np.asarray(vec_b, dtype=np.float32).reshape(-1)
        if a.shape != b.shape:
            return None
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a <= 1e-9 or norm_b <= 1e-9:
            return None
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def _clip_box_to_image(box: List[int], shape: tuple[int, int, int]) -> Optional[List[int]]:
        if box is None or len(box) != 4:
            return None
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = shape[:2]
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    def _crop_box_region(self, image: np.ndarray, box: Optional[List[int]]) -> Optional[np.ndarray]:
        if box is None:
            return None
        clipped = self._clip_box_to_image(box, image.shape)
        if clipped is None:
            return None
        x1, y1, x2, y2 = clipped
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    @staticmethod
    def _text_for_embedding(change: Dict[str, Any]) -> str:
        desc = change.get("change_description", "").strip()
        if desc:
            return desc
        return change.get("object_description", "").strip()

    def _attach_snapshot_features(
        self,
        change_items: List[Dict[str, Any]],
        ref_image: np.ndarray,
        cur_image: np.ndarray,
    ) -> None:
        for change in change_items:
            change["dino_feat_t0"] = None
            change["dino_feat_t1"] = None
            change["description_embedding"] = None

        dino_targets: List[Tuple[int, str]] = []
        dino_crops: List[np.ndarray] = []
        for idx, change in enumerate(change_items):
            crop_t0 = self._crop_box_region(ref_image, change.get("bbox_t0"))
            if crop_t0 is not None:
                dino_targets.append((idx, "dino_feat_t0"))
                dino_crops.append(crop_t0)

            crop_t1 = self._crop_box_region(cur_image, change.get("bbox_t1"))
            if crop_t1 is not None:
                dino_targets.append((idx, "dino_feat_t1"))
                dino_crops.append(crop_t1)

        if dino_crops:
            dino_feats = self.feature_factory.compute_dino_features(dino_crops)
            for (idx, output_key), feat in zip(dino_targets, dino_feats):
                change_items[idx][output_key] = feat

        text_to_indices: Dict[str, List[int]] = {}
        for idx, change in enumerate(change_items):
            text = self._text_for_embedding(change)
            if not text:
                continue
            if text not in text_to_indices:
                text_to_indices[text] = []
            text_to_indices[text].append(idx)

        if not text_to_indices:
            return

        unique_texts = list(text_to_indices.keys())
        text_feats = self.feature_factory.compute_text_features(unique_texts)
        for text, feat in zip(unique_texts, text_feats):
            for idx in text_to_indices[text]:
                change_items[idx]["description_embedding"] = np.asarray(feat, dtype=np.float32).copy()

    def _select_merge_bbox_and_feature(
        self, change: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        change_type = change.get("change_type", "")
        b0 = change.get("bbox_3d_t0")
        b1 = change.get("bbox_3d_t1")
        d0 = change.get("dino_feat_t0")
        d1 = change.get("dino_feat_t1")

        if change_type == "appear":
            return b1, d1
        if change_type == "disappear":
            return b0, d0
        if b1 is not None:
            return b1, d1 if d1 is not None else d0
        return b0, d0 if d0 is not None else d1

    def _select_snapshot_bbox_and_feature(
        self, change: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        change_type = change.get("change_type", "")
        b0 = change.get("bbox_3d_t0")
        b1 = change.get("bbox_3d_t1")
        d0 = change.get("dino_feat_t0")
        d1 = change.get("dino_feat_t1")

        if change_type == "appear":
            return b1, d1
        if change_type == "disappear":
            return b0, d0
        if b1 is not None:
            return b1, d1 if d1 is not None else d0
        return b0, d0 if d0 is not None else d1

    def _find_best_merge_candidate(
        self,
        bbox_3d: Optional[np.ndarray],
        dino_feature: Optional[np.ndarray],
        max_snapshot_timestamp: Optional[datetime] = None,
    ) -> Tuple[Optional[TrackedObject], float, float]:
        if bbox_3d is None:
            return None, 0.0, -1.0

        candidates = self.change_memory.find_iou_candidates(
            bbox_3d=bbox_3d,
            iou_threshold=CHANGE_MEMORY_IOU_THRESHOLD,
            window_size=CHANGE_MEMORY_BBOX_WINDOW,
            iou_mode=CHANGE_MEMORY_IOU_MODE,
            bbox_expansion_ratio=CHANGE_MEMORY_IOU_BBOX_EXPANSION_RATIO,
            include_disappeared=True,
            max_snapshot_timestamp=max_snapshot_timestamp,
        )
        if not candidates:
            return None, 0.0, -1.0

        best_obj: Optional[TrackedObject] = None
        best_iou = 0.0
        best_dino = -1.0
        for obj, iou in candidates:
            candidate_best_dino = -1.0
            for snap in obj.snapshots:
                if max_snapshot_timestamp is not None and snap.timestamp >= max_snapshot_timestamp:
                    continue
                if snap.dino_feature is None:
                    continue
                dino_sim = self._cosine_similarity(dino_feature, snap.dino_feature)
                if dino_sim is None:
                    continue
                candidate_best_dino = max(candidate_best_dino, dino_sim)

            if candidate_best_dino < CHANGE_MEMORY_DINO_SIM_THRESHOLD:
                continue
            if candidate_best_dino > best_dino or (np.isclose(candidate_best_dino, best_dino) and iou > best_iou):
                best_obj = obj
                best_iou = iou
                best_dino = candidate_best_dino
        return best_obj, best_iou, best_dino

    @staticmethod
    def _stale_reference_reason(
        obj: TrackedObject,
        ref_timestamp: datetime,
    ) -> Optional[str]:
        latest = obj.get_latest_snapshot()
        if latest is None or latest.timestamp is None:
            return None
        if latest.timestamp > ref_timestamp:
            return (
                "stale_reference:"
                f"ref={ref_timestamp.isoformat()};"
                f"latest={latest.timestamp.isoformat()}"
            )
        return None

    def _duplicate_reason(
        self,
        obj: TrackedObject,
        change_type: str,
        text_embedding: Optional[np.ndarray],
    ) -> Optional[str]:
        latest = obj.get_latest_snapshot()
        latest_type = latest.change_type if latest is not None else ""

        # State-machine dedup for appear/disappear:
        # - appear is allowed only after disappear (or for a new object)
        # - disappear is allowed only when object is currently active
        if change_type == "appear":
            if latest is not None and latest_type != "disappear":
                return f"duplicate_appear:last={latest_type or 'none'}"
            return None
        if change_type == "disappear":
            if latest is not None and latest_type == "disappear":
                return "duplicate_disappear:last=disappear"
            return None

        if change_type != "change":
            return None

        previous_change = obj.get_latest_snapshot_by_type("change")
        if previous_change is None:
            return None
        sim = self._cosine_similarity(text_embedding, previous_change.description_embedding)
        if sim is None:
            return None
        if sim >= CHANGE_MEMORY_TEXT_SIM_THRESHOLD:
            return f"duplicate_change_text:{sim:.3f}"
        return None

    def _prepare_changes(
        self,
        changes_raw: List[Dict[str, Any]],
        ref_shape: tuple[int, int, int],
        cur_shape: tuple[int, int, int],
        trace: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        change_items: List[Dict[str, Any]] = []
        valid_types = {"appear", "disappear", "change"}

        for idx, change in enumerate(changes_raw):
            raw_type = str(change.get("change_type", "")).strip().lower()
            ctype = "change" if raw_type == "move" else raw_type
            obj = change.get("object_description", "").strip()
            desc = change.get("change_description", "").strip()
            context_desc = change.get("context_description", "").strip()
            confidence = str(change.get("confidence", "")).strip()

            raw_b0 = change.get("bbox_t0", [])
            raw_b1 = change.get("bbox_t1", [])
            b0 = norm_box_to_xyxy(raw_b0, ref_shape)
            b1 = norm_box_to_xyxy(raw_b1, cur_shape)

            if ctype not in valid_types:
                if trace is not None:
                    trace.append({
                        "index": idx,
                        "change_type_raw": raw_type,
                        "change_type_norm": ctype,
                        "object_description": obj,
                        "change_description": desc,
                        "context_description": context_desc,
                        "confidence": confidence,
                        "bbox_t0_raw": raw_b0,
                        "bbox_t1_raw": raw_b1,
                        "bbox_t0_norm": b0,
                        "bbox_t1_norm": b1,
                        "dropped": True,
                        "reason": "invalid_change_type",
                    })
                continue

            if trace is not None:
                trace.append({
                    "index": idx,
                    "change_type_raw": raw_type,
                    "change_type_norm": ctype,
                    "object_description": obj,
                    "change_description": desc,
                    "context_description": context_desc,
                    "confidence": confidence,
                    "bbox_t0_raw": raw_b0,
                    "bbox_t1_raw": raw_b1,
                    "bbox_t0_norm": b0,
                    "bbox_t1_norm": b1,
                    "dropped": False,
                })

            change_items.append({
                "change_type": ctype,
                "object_description": obj,
                "change_description": desc,
                "context_description": context_desc,
                "confidence": confidence,
                "bbox_t0": b0,
                "bbox_t1": b1,
                "mask_t0": None,
                "mask_t1": None,
                "bbox_3d_t0": None,
                "bbox_3d_t1": None,
                "dino_feat_t0": None,
                "dino_feat_t1": None,
                "description_embedding": None,
            })

        return change_items

    def _filter_changes(
        self,
        change_items: List[Dict[str, Any]],
        ref_shape: tuple[int, int, int],
        cur_shape: tuple[int, int, int],
        vis_mask_t0: Optional[np.ndarray],
        vis_mask_t1: Optional[np.ndarray],
        occlusion_mask: Optional[np.ndarray] = None,
        trace: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        filtered = []
        for idx, change in enumerate(change_items):
            ctype = change.get("change_type", "")
            confidence = str(change.get("confidence", "")).strip()
            b0 = change.get("bbox_t0")
            b1 = change.get("bbox_t1")

            reasons = []

            if confidence in {"low", "med"}:
                reasons.append("non_high_confidence")

            if ctype not in {"appear", "disappear", "change"}:
                reasons.append("invalid_change_type")
            if ctype == "appear" and b1 is None:
                reasons.append("appear_missing_bbox_t1")
            if ctype == "disappear" and b0 is None:
                reasons.append("disappear_missing_bbox_t0")
            if ctype == "change" and (b0 is None or b1 is None):
                reasons.append("change_missing_bbox")
            if b0 is None and b1 is None:
                reasons.append("both_bboxes_missing")

            if b0 is not None:
                if box_area_ratio(b0, ref_shape) < BBOX_MIN_AREA_RATIO:
                    reasons.append("bbox_t0_area_too_small")
                if vis_mask_t0 is not None:
                    if outside_mask_ratio(b0, vis_mask_t0) > VISIBILITY_MASK_MAX_OUTSIDE_AREA_RATIO:
                        reasons.append("bbox_t0_outside_visibility")

            if b1 is not None:
                if box_area_ratio(b1, cur_shape) < BBOX_MIN_AREA_RATIO:
                    reasons.append("bbox_t1_area_too_small")
                if vis_mask_t1 is not None:
                    if outside_mask_ratio(b1, vis_mask_t1) > VISIBILITY_MASK_MAX_OUTSIDE_AREA_RATIO:
                        reasons.append("bbox_t1_outside_visibility")

            if ctype == "appear" and b1 is not None and occlusion_mask is not None:
                occluded_ratio = 1.0 - outside_mask_ratio(b1, occlusion_mask)
                if occluded_ratio > OCCLUSION_RELIEF_MIN_OCCLUDED_RATIO:
                    reasons.append("appear_occlusion_relief")

            kept = len(reasons) == 0
            if trace is not None:
                trace.append({
                    "index": idx,
                    "change_type": ctype,
                    "confidence": confidence,
                    "object_description": change.get("object_description", ""),
                    "change_description": change.get("change_description", ""),
                    "bbox_t0": b0,
                    "bbox_t1": b1,
                    "kept": kept,
                    "reasons": reasons,
                })

            if kept:
                filtered.append(change)

        return filtered

    def _build_vlm_views(
        self,
        ref_image: np.ndarray,
        cur_image: np.ndarray,
        change_items: List[Dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        ref_boxes: List[List[int]] = []
        cur_boxes: List[List[int]] = []
        ref_labels: List[str] = []
        cur_labels: List[str] = []
        for idx, change in enumerate(change_items, 1):
            tag = confidence_short_label(change.get("confidence", ""))
            b0 = change.get("bbox_t0")
            b1 = change.get("bbox_t1")
            if b0 is not None:
                ref_boxes.append(b0)
                ref_labels.append(f"{idx}-{tag}")
            if b1 is not None:
                cur_boxes.append(b1)
                cur_labels.append(f"{idx}-{tag}")

        ref_annot = draw_boxes(ref_image, ref_boxes, (255, 0, 0), ref_labels)
        cur_annot = draw_boxes(cur_image, cur_boxes, (0, 255, 0), cur_labels)

        lines = build_description_lines(format_change_descriptions(change_items))
        ref_annot = draw_text_block(ref_annot, lines)
        cur_annot = draw_text_block(cur_annot, lines)

        return ref_annot, cur_annot

    def _save_vlm_raw_bbox_debug(
        self,
        ref_image: np.ndarray,
        cur_image: np.ndarray,
        change_items: List[Dict[str, Any]],
        ref_timestamp: Optional[datetime],
        cur_timestamp: Optional[datetime],
        vis_mask_t0: Optional[np.ndarray],
        vis_mask_t1: Optional[np.ndarray],
        occlusion_mask: Optional[np.ndarray] = None,
    ) -> None:
        if not VLM_RAW_BBOX_DEBUG_ENABLED:
            return
        os.makedirs(VLM_RAW_BBOX_DEBUG_DIR, exist_ok=True)

        ref = ref_image.copy()
        cur = cur_image.copy()

        def _overlay_mask(
            image: np.ndarray,
            mask: Optional[np.ndarray],
            color: tuple[int, int, int],
            alpha: float = 0.25,
        ) -> np.ndarray:
            if mask is None:
                return image
            if mask.shape[:2] != image.shape[:2]:
                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
            else:
                mask_resized = mask
            overlay = image.copy()
            c = np.array(color, dtype=np.uint8)
            hit = mask_resized > 0
            overlay[hit] = (overlay[hit] * (1.0 - alpha) + c * alpha).astype(np.uint8)
            return overlay

        ref = _overlay_mask(ref, vis_mask_t0, (0, 180, 255))
        cur = _overlay_mask(cur, vis_mask_t1, (0, 180, 255))
        cur = _overlay_mask(cur, occlusion_mask, (255, 50, 50), alpha=0.35)

        boxes_t0 = []
        labels_t0: List[str] = []
        boxes_t1 = []
        labels_t1: List[str] = []
        for idx, change in enumerate(change_items, 1):
            b0 = change.get("bbox_t0")
            b1 = change.get("bbox_t1")
            tag = confidence_short_label(change.get("confidence", ""))
            if b0 is not None:
                boxes_t0.append(b0)
                labels_t0.append(f"{idx}-{tag}")
            if b1 is not None:
                boxes_t1.append(b1)
                labels_t1.append(f"{idx}-{tag}")

        if boxes_t0:
            ref = draw_boxes(ref, boxes_t0, (255, 0, 0), labels_t0)
        if boxes_t1:
            cur = draw_boxes(cur, boxes_t1, (0, 255, 0), labels_t1)

        def _label_panel(image: np.ndarray, text: str) -> np.ndarray:
            out = image.copy()
            cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
            cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            return out

        ref_count = len(boxes_t0)
        cur_count = len(boxes_t1)
        confidence_counts = self._confidence_counts(change_items)
        conf_suffix = (
            f"L:{confidence_counts['low']} "
            f"M:{confidence_counts['med']} "
            f"H:{confidence_counts['high']}"
        )
        ref = _label_panel(ref, f"t0 raw bboxes: {ref_count} | {conf_suffix}")
        cur = _label_panel(cur, f"t1 raw bboxes: {cur_count} | {conf_suffix}")
        lines = build_description_lines(format_change_descriptions(change_items), max_len=54)
        ref = draw_text_block(ref, lines)
        cur = draw_text_block(cur, lines)

        if ref.shape != cur.shape:
            cur = cv2.resize(cur, (ref.shape[1], ref.shape[0]))

        combined = np.hstack([ref, cur])
        ts = cur_timestamp or ref_timestamp or datetime.now()
        ts_name = ts.strftime("%Y%m%d_%H%M%S_%f")
        out_path = os.path.join(VLM_RAW_BBOX_DEBUG_DIR, f"vlm_raw_bbox_{ts_name}.png")
        cv2.imwrite(out_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    @staticmethod
    def _overlay_debug_mask(
        image: np.ndarray,
        mask: Optional[np.ndarray],
        color: tuple[int, int, int],
        alpha: float = 0.25,
    ) -> np.ndarray:
        if mask is None:
            return image.copy()
        if mask.shape[:2] != image.shape[:2]:
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask
        out = image.copy()
        color_arr = np.asarray(color, dtype=np.uint8)
        hit = mask_resized > 0
        out[hit] = (out[hit] * (1.0 - alpha) + color_arr * alpha).astype(np.uint8)
        return out

    @staticmethod
    def _draw_single_box(
        image: np.ndarray,
        box: Optional[List[int]],
        color: tuple[int, int, int],
    ) -> np.ndarray:
        out = image.copy()
        if box is None or len(box) != 4:
            return out
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        return out

    @staticmethod
    def _write_debug_image(path: str, image: np.ndarray) -> None:
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def _save_rejected_bbox_debug(
        self,
        ref_image: np.ndarray,
        cur_image: np.ndarray,
        filter_trace: List[Dict[str, Any]],
        ref_timestamp: Optional[datetime],
        cur_timestamp: Optional[datetime],
        vis_mask_t0: Optional[np.ndarray],
        vis_mask_t1: Optional[np.ndarray],
    ) -> None:
        if not REJECTED_BBOX_DEBUG_ENABLED:
            return

        rejected_rows = [
            row
            for row in filter_trace
            if isinstance(row, dict) and not bool(row.get("kept", False))
        ]
        if not rejected_rows:
            return

        os.makedirs(REJECTED_BBOX_DEBUG_DIR, exist_ok=True)
        cur_tag = (cur_timestamp or ref_timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")
        ref_base = ref_image.copy()
        cur_base = cur_image.copy()
        ref_mask = self._overlay_debug_mask(ref_base, vis_mask_t0, (0, 180, 255))
        cur_mask = self._overlay_debug_mask(cur_base, vis_mask_t1, (0, 180, 255))

        for offset, row in enumerate(rejected_rows, start=1):
            idx = int(row.get("index", -1))
            folder_name = f"rejected_{cur_tag}_change_{idx:02d}_reject_{offset:02d}"
            folder_path = os.path.join(REJECTED_BBOX_DEBUG_DIR, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            bbox_t0 = row.get("bbox_t0")
            bbox_t1 = row.get("bbox_t1")
            ref_mask_box = self._draw_single_box(ref_mask, bbox_t0, (255, 214, 88))
            cur_mask_box = self._draw_single_box(cur_mask, bbox_t1, (255, 214, 88))

            self._write_debug_image(os.path.join(folder_path, "ref_raw.jpg"), ref_base)
            self._write_debug_image(os.path.join(folder_path, "cur_raw.jpg"), cur_base)
            self._write_debug_image(os.path.join(folder_path, "ref_visibility.jpg"), ref_mask)
            self._write_debug_image(os.path.join(folder_path, "cur_visibility.jpg"), cur_mask)
            self._write_debug_image(os.path.join(folder_path, "ref_visibility_bbox.jpg"), ref_mask_box)
            self._write_debug_image(os.path.join(folder_path, "cur_visibility_bbox.jpg"), cur_mask_box)

            metadata = {
                "change_index": idx,
                "change_type": str(row.get("change_type", "") or ""),
                "object_description": str(row.get("object_description", "") or ""),
                "change_description": str(row.get("change_description", "") or ""),
                "bbox_t0": row.get("bbox_t0"),
                "bbox_t1": row.get("bbox_t1"),
                "rejection_reasons": list(row.get("reasons", []) or []),
                "reference_timestamp": ref_timestamp.isoformat() if ref_timestamp else "",
                "current_timestamp": cur_timestamp.isoformat() if cur_timestamp else "",
            }
            with open(os.path.join(folder_path, "metadata.json"), "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, ensure_ascii=True, indent=2)

    def _compute_3d_bbox(
        self,
        mask: np.ndarray,
        depth_map: np.ndarray,
        intrinsics: np.ndarray,
        pose_matrix: np.ndarray,
        outlier_trim_ratio: float
    ) -> Optional[np.ndarray]:
        rows, cols = np.where(mask > 0)
        if len(rows) == 0:
            return None

        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        depth_values = depth_map[rows, cols]
        valid_depth = depth_values > 0
        if int(np.count_nonzero(valid_depth)) < 10:
            return None

        rows = rows[valid_depth].astype(np.float64, copy=False)
        cols = cols[valid_depth].astype(np.float64, copy=False)
        depth_values = depth_values[valid_depth].astype(np.float64, copy=False)

        x = (cols - cx) * depth_values / fx
        y = (rows - cy) * depth_values / fy
        points_camera_h = np.column_stack([x, y, depth_values, np.ones_like(depth_values)])
        points_3d = (pose_matrix @ points_camera_h.T).T[:, :3]

        mean = np.mean(points_3d, axis=0)
        std = np.std(points_3d, axis=0)

        if np.any(std > 0.01):
            mask_inliers = np.all(
                np.abs(points_3d - mean) < BBOX_3D_OUTLIER_STD_THRESHOLD * std,
                axis=1
            )
            points_filtered = points_3d[mask_inliers]
            if len(points_filtered) >= 5:
                points_3d = points_filtered

        if outlier_trim_ratio > 0:
            keep_n = int(len(points_3d) * (1.0 - outlier_trim_ratio))
            if keep_n >= 3 and keep_n < len(points_3d):
                center = np.median(points_3d, axis=0)
                dist = np.linalg.norm(points_3d - center, axis=1)
                keep_idx = np.argsort(dist)[:keep_n]
                points_3d = points_3d[keep_idx]

        if len(points_3d) < 3:
            return None

        bbox_min = np.min(points_3d, axis=0)
        bbox_max = np.max(points_3d, axis=0)
        extent = bbox_max - bbox_min

        if np.any(extent <= 0):
            for i in range(3):
                if extent[i] <= 0:
                    bbox_min[i] -= 0.01
                    bbox_max[i] += 0.01

        return np.concatenate([bbox_min, bbox_max])

    def _snapshot_description(self, change: Dict[str, Any]) -> str:
        obj = change.get("object_description", "").strip()
        desc = change.get("change_description", "").strip()
        return desc if desc else obj

    @staticmethod
    def _snapshot_context(change: Dict[str, Any]) -> str:
        return str(change.get("context_description", "")).strip()

    @staticmethod
    def _confidence_counts(changes: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = {"low": 0, "med": 0, "high": 0}
        for change in changes:
            confidence = str(change.get("confidence", "")).strip()
            if confidence in counts:
                counts[confidence] += 1
        return counts

    def _summarize_changes_for_log(self, changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        summary = []
        for idx, change in enumerate(changes):
            summary.append({
                "index": idx,
                "object_id": change.get("object_id"),
                "memory_action": change.get("memory_action"),
                "change_type": change.get("change_type", ""),
                "confidence": str(change.get("confidence", "")).strip(),
                "object_description": change.get("object_description", ""),
                "change_description": change.get("change_description", ""),
                "context_description": change.get("context_description", ""),
                "bbox_t0": change.get("bbox_t0"),
                "bbox_t1": change.get("bbox_t1"),
                "bbox_3d_t0": change.get("bbox_3d_t0"),
                "bbox_3d_t1": change.get("bbox_3d_t1"),
            })
        return summary

    def _snapshot_crop_for_change(
        self,
        change: Dict[str, Any],
        ref_image: np.ndarray,
        cur_image: np.ndarray,
    ) -> Optional[np.ndarray]:
        ctype = change.get("change_type", "")
        crop: Optional[np.ndarray] = None
        if ctype == "appear":
            crop = self._crop_box_region(cur_image, change.get("bbox_t1"))
        elif ctype == "disappear":
            crop = self._crop_box_region(ref_image, change.get("bbox_t0"))
        else:
            crop = self._crop_box_region(cur_image, change.get("bbox_t1"))
            if crop is None:
                crop = self._crop_box_region(ref_image, change.get("bbox_t0"))

        if crop is None:
            return None
        h, w = crop.shape[:2]
        max_edge = max(h, w)
        if max_edge > 224:
            scale = 224.0 / float(max_edge)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return crop.copy()

    def _update_change_memory(
        self,
        changes: List[Dict[str, Any]],
        ref_timestamp: datetime,
        cur_timestamp: datetime,
        ref_image: np.ndarray,
        cur_image: np.ndarray,
        trace: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        kept: List[Dict[str, Any]] = []
        for idx, change in enumerate(changes):
            change_type = change.get("change_type", "")
            object_desc = str(change.get("object_description", "")).strip()
            change_desc = str(change.get("change_description", "")).strip()
            desc = self._snapshot_description(change)
            context_desc = self._snapshot_context(change)
            text_emb = change.get("description_embedding")

            merge_bbox, merge_dino = self._select_merge_bbox_and_feature(change)
            snapshot_bbox, snapshot_dino = self._select_snapshot_bbox_and_feature(change)
            if merge_bbox is None and snapshot_bbox is None:
                if trace is not None:
                    trace.append(
                        {
                            "index": idx,
                            "action": "skipped_no_usable_bbox",
                            "change_type": change_type,
                            "description": desc,
                        }
                    )
                continue

            matched_obj, matched_iou, matched_dino = self._find_best_merge_candidate(
                bbox_3d=merge_bbox,
                dino_feature=merge_dino,
                max_snapshot_timestamp=cur_timestamp,
            )

            target_obj: Optional[TrackedObject] = matched_obj
            action = "merge_existing_object" if matched_obj is not None else "create_new_object"
            seeded_with_appear = False

            if target_obj is None and change_type in {"change", "disappear"}:
                seed_bbox = change.get("bbox_3d_t0")
                if seed_bbox is not None:
                    seed_snapshot = ObjectSnapshot(
                        change_type="appear",
                        timestamp=ref_timestamp,
                        description=desc,
                        bbox_3d=seed_bbox,
                        object_description=object_desc if object_desc else desc,
                        change_description="",
                        context_description=context_desc,
                        bbox_2d_image=None,
                        dino_feature=change.get("dino_feat_t0"),
                        description_embedding=text_emb,
                    )
                    obj_id = self.change_memory.add_new_object(seed_snapshot)
                    target_obj = self.change_memory.get_object(obj_id)
                    action = "create_new_object_with_appear_seed"
                    seeded_with_appear = True

            if target_obj is not None:
                stale_reference = None
                if not seeded_with_appear:
                    stale_reference = self._stale_reference_reason(target_obj, ref_timestamp)
                if stale_reference is not None:
                    if trace is not None:
                        trace.append(
                            {
                                "index": idx,
                                "action": "discard_stale_reference_update",
                                "object_id": target_obj.object_id,
                                "change_type": change_type,
                                "description": desc,
                                "reason": stale_reference,
                                "matched_iou": matched_iou,
                                "matched_dino": matched_dino,
                            }
                        )
                    continue

                duplicate = self._duplicate_reason(target_obj, change_type, text_emb)
                if duplicate is not None:
                    if trace is not None:
                        trace.append(
                            {
                                "index": idx,
                                "action": "discard_duplicate_snapshot",
                                "object_id": target_obj.object_id,
                                "change_type": change_type,
                                "description": desc,
                                "reason": duplicate,
                                "matched_iou": matched_iou,
                                "matched_dino": matched_dino,
                            }
                        )
                    continue

            write_bbox = snapshot_bbox if snapshot_bbox is not None else merge_bbox
            write_dino = snapshot_dino if snapshot_dino is not None else merge_dino
            if write_bbox is None:
                if trace is not None:
                    trace.append(
                        {
                            "index": idx,
                            "action": "skipped_missing_write_bbox",
                            "change_type": change_type,
                            "description": desc,
                        }
                    )
                continue

            snapshot = ObjectSnapshot(
                change_type=change_type,
                timestamp=cur_timestamp,
                description=desc,
                bbox_3d=write_bbox,
                object_description=object_desc if object_desc else desc,
                change_description=change_desc,
                context_description=context_desc,
                bbox_2d_image=self._snapshot_crop_for_change(change, ref_image, cur_image),
                dino_feature=write_dino,
                description_embedding=text_emb,
            )

            if target_obj is None:
                obj_id = self.change_memory.add_new_object(snapshot)
            else:
                obj_id = target_obj.object_id
                self.change_memory.update_object(obj_id, snapshot)

            if trace is not None:
                trace.append(
                    {
                        "index": idx,
                        "action": action,
                        "object_id": obj_id,
                        "change_type": change_type,
                        "description": desc,
                        "seeded_with_appear": seeded_with_appear,
                        "matched_iou": matched_iou,
                        "matched_dino": matched_dino,
                    }
                )

            change["object_id"] = obj_id
            change["memory_action"] = action
            kept.append(change)

        return kept
