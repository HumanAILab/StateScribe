from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime
import json
import logging
from pathlib import Path
import re
from threading import Lock
import sys
from time import perf_counter, sleep
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark.clock import BenchmarkClock
from benchmark.config import (
    DEFAULT_BROKER_DRAIN_TIMEOUT_SEC,
    DEFAULT_DATASET_DIR,
    DEFAULT_FAST_POLL_INTERVAL_SEC,
    DEFAULT_FRAME_STRIDE,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PIPELINE_DRAIN_TIMEOUT_SEC,
    DEFAULT_REALTIME_POLL_INTERVAL_SEC,
    DEFAULT_REQUIRE_CONFIDENCE,
    DEFAULT_USE_ALIGNED_METADATA,
)
from benchmark.dataset_loader import CaptureFrameRecord, build_frame, load_dataset_records
from benchmark.recorder import BenchmarkDescriptionBroker, BenchmarkOutputRecorder
from components.ai_description_pipeline import AIParaphraseDescriptionPipeline
from components.change_detection_pipeline import ChangeDetectionPipeline
from components.feature_factory import FeatureFactory
from components.live_description_pipeline import LiveDescriptionPipeline
from components.logging.frame_logger import FrameLogger
from components.memory.memory_manager import MemoryManager
from components.visualization.display_manager import DisplayManager
from components.visualization.web_display_manager import WebDisplayManager
from components.visualization.web_state import WebStateStore
from components.visualization.web_visualizer import WebVisualizer
from components.visualization.simple_visualizer import Open3DVisualizer
from config import (
    DESCRIPTION_OUTPUT_MODE,
    VISUALIZATION_BACKEND,
    VISUALIZATION_ENABLED,
)
from logging_config import setup_logging
from memo_utils.common import compress_object

LOGGER = logging.getLogger("benchmark.run")
OBJECT_MEMORY_SCHEMA_VERSION = 1


class BenchmarkMemoryManager(MemoryManager):
    def start_session(self, world_name: str):
        if self.current_world_name == world_name:
            LOGGER.debug("Resuming benchmark session for world: %s", world_name)
            return

        self.end_session()
        self.executor = ThreadPoolExecutor(max_workers=self._max_workers)
        self.current_world_name = world_name
        self._submit_index = 0
        self._next_index = 0
        self._pending_results.clear()
        self.ltm_path = None
        self.frames_path = None
        LOGGER.debug("Benchmark session started for world: %s (disk persistence disabled).", world_name)

    def _ensure_ltm_path_exists(self):
        return

    def _save_frame_to_disk(self, frame):
        return

    def _load_memory_from_disk(self):
        return


class BenchmarkChangeDetectionPipeline(ChangeDetectionPipeline):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._change_memory_seed: Optional[Dict[str, Any]] = None
        self._change_memory_seed_source_world: Optional[str] = None

    def seed_change_memory(
        self,
        payload: Optional[Dict[str, Any]],
        *,
        source_world_name: Optional[str] = None,
    ) -> None:
        if not isinstance(payload, dict):
            return
        objects = payload.get("objects")
        if not isinstance(objects, dict):
            return
        with self._state_lock:
            self._change_memory_seed = payload
            self._change_memory_seed_source_world = str(source_world_name or "").strip() or None

    def _ensure_world_state(self, world_name: Optional[str]) -> None:
        clean_world = (world_name or "").strip()
        if not clean_world:
            return

        with self._state_lock:
            if self.current_world_name == clean_world:
                return

            self.current_world_name = clean_world
            self.change_state_path = None
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
            seed_payload = self._change_memory_seed
            seed_source_world = self._change_memory_seed_source_world
            self._change_memory_seed = None
            self._change_memory_seed_source_world = None
            if isinstance(seed_payload, dict):
                try:
                    self.change_memory.load_from_persist_dict(seed_payload)
                    LOGGER.info(
                        "Seeded benchmark OTM for %s from %s with %d object(s).",
                        clean_world,
                        seed_source_world or "previous capture",
                        len(self.change_memory.get_all_objects()),
                    )
                except Exception:
                    LOGGER.exception("Failed to seed benchmark OTM for world %s.", clean_world)
                    self.change_memory.clear()
            if self.display_manager is not None:
                self.display_manager.update_memory_display(self.change_memory)

    def _load_state_locked(self) -> None:
        return

    def _save_state_locked(self) -> None:
        return


class BenchmarkWebStateStore(WebStateStore):
    def __init__(self, clock: BenchmarkClock) -> None:
        super().__init__()
        self._clock = clock

    def get_meta(self) -> Dict[str, Any]:
        payload = super().get_meta()
        payload["server_time"] = self._clock.now().isoformat()
        return payload

    def add_log_entry(self, kind: str, entry: Dict[str, Any]) -> None:
        payload = dict(entry)
        if kind == "speech":
            payload["timestamp"] = self._clock.now().isoformat()
        super().add_log_entry(kind, payload)


class BenchmarkAIParaphraseDescriptionPipeline(AIParaphraseDescriptionPipeline):
    def __init__(
        self,
        *args: Any,
        on_summary_ready: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._benchmark_on_summary_ready = on_summary_ready
        self._benchmark_pending_job_count = 0

    def _collect_summary_results(self, pending_jobs: Dict[Future, Dict[str, Any]]) -> bool:
        if not pending_jobs:
            return False
        done, _ = wait(tuple(pending_jobs.keys()), timeout=0.0, return_when=FIRST_COMPLETED)
        if not done:
            return False

        completed: List[Tuple[int, str, Dict[str, Any]]] = []
        for future in done:
            meta = pending_jobs.pop(future, None)
            if not isinstance(meta, dict):
                continue
            seq = int(meta.get("sequence", -1))
            try:
                summary = future.result()
            except Exception:
                LOGGER.exception("Failed to resolve AI paraphrase summary future.")
                continue
            completed.append((seq, summary or "", dict(meta.get("payload") or {})))

        completed.sort(key=lambda item: item[0], reverse=True)
        for seq, summary, payload in completed:
            if seq < self._latest_completed_summary_seq:
                LOGGER.debug(
                    "Dropping stale AI summary result: seq=%s latest_completed=%s",
                    seq,
                    self._latest_completed_summary_seq,
                )
                continue
            self._latest_completed_summary_seq = max(self._latest_completed_summary_seq, seq)
            if summary:
                self._publish_summary(summary, payload)
        self._benchmark_pending_job_count = len(pending_jobs)
        return True

    def _publish_summary(self, summary: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if self.on_description is None:
            return

        normalized = self._normalize_text(summary)
        if not normalized:
            return
        summary_payload = dict(payload or {})

        def _deferred_payload(raw_text: str = normalized) -> str:
            return self._render_for_emit(raw_text)

        callback_target = getattr(self.on_description, "__self__", None)
        supports_deferred = getattr(callback_target, "accepts_deferred_payload", False)
        broker_payload: Any = _deferred_payload if supports_deferred else _deferred_payload()

        accepted = self.on_description(broker_payload)
        if isinstance(accepted, bool) and not accepted:
            return

        with self._lock:
            self._recent_outputs.append(normalized)

        if self._benchmark_on_summary_ready is not None:
            try:
                self._benchmark_on_summary_ready(normalized, summary_payload)
            except Exception:
                LOGGER.exception("Failed to notify benchmark AI summary callback.")

    def run(self) -> None:
        if not self.enabled:
            return

        self.running = True
        LOGGER.debug("BenchmarkAIParaphraseDescriptionPipeline started.")
        pending_jobs: Dict[Future, Dict[str, Any]] = {}

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
                    pending_jobs[future] = {
                        "sequence": seq,
                        "payload": payload,
                    }
                    self._benchmark_pending_job_count = len(pending_jobs)

                handled = self._collect_summary_results(pending_jobs)
                if handled:
                    continue

                if pending_jobs:
                    wait_timeout = min(wait_timeout, 0.05)

                self._wake_event.wait(timeout=wait_timeout)
                self._wake_event.clear()
        finally:
            self._benchmark_pending_job_count = 0
            if self._summary_executor is not None:
                self._summary_executor.shutdown(wait=False)
                self._summary_executor = None
            LOGGER.debug("BenchmarkAIParaphraseDescriptionPipeline stopped.")

    def is_idle(self) -> bool:
        with self._lock:
            has_buffered_inputs = self._has_pending_locked()
        return (not has_buffered_inputs) and self._benchmark_pending_job_count == 0


class BenchmarkRunner:
    def __init__(
        self,
        dataset_dir: Path,
        mode: str,
        enable_live_descriptions: Optional[bool],
        feed_interval_sec: Optional[float],
        frame_stride: int,
        max_captures: Optional[int],
        output_root: Path,
        use_aligned_metadata: bool,
        require_confidence: bool,
        pipeline_drain_timeout_sec: float,
        broker_drain_timeout_sec: float,
        world_name_prefix: str,
        share_object_memory_across_captures: bool,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.enable_live_descriptions = bool(enable_live_descriptions) if enable_live_descriptions is not None else self.mode != "fast"
        self.feed_interval_sec = float(feed_interval_sec) if feed_interval_sec is not None and float(feed_interval_sec) > 0 else None
        self.frame_stride = max(1, int(frame_stride))
        self.max_captures = max_captures if (max_captures is not None and max_captures > 0) else None
        self.output_root = output_root
        self.use_aligned_metadata = bool(use_aligned_metadata)
        self.require_confidence = bool(require_confidence)
        self.pipeline_drain_timeout_sec = max(1.0, float(pipeline_drain_timeout_sec))
        self.broker_drain_timeout_sec = max(1.0, float(broker_drain_timeout_sec))
        self.share_object_memory_across_captures = bool(share_object_memory_across_captures)

        self.records: List[CaptureFrameRecord] = load_dataset_records(
            dataset_dir=self.dataset_dir,
            frame_stride=self.frame_stride,
            use_aligned_metadata=self.use_aligned_metadata,
            require_confidence=self.require_confidence,
            max_captures=self.max_captures,
        )
        if not self.records:
            raise ValueError("No valid frame record found for benchmark run.")

        dataset_slug = _safe_slug(self.dataset_dir.name or "dataset")
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.world_name = f"{_safe_slug(world_name_prefix)}_{dataset_slug}_{run_stamp}"
        self.output_dir = self.output_root / dataset_slug / "statescribe" / run_stamp

        self.clock = BenchmarkClock(mode=self.mode, start_timestamp=self.records[0].timestamp)
        self.capture_record_groups = _group_records_by_capture(self.records)
        self._first_timestamp = self.records[0].timestamp
        self._last_timestamp = self.records[-1].timestamp
        self._run_start_perf: Optional[float] = None
        self._run_start_wall: Optional[datetime] = None
        self._latest_fed_timestamp: Optional[datetime] = None
        self._pipeline_timeout_reached = False
        self._broker_timeout_reached = False

        self._emit_lock = Lock()
        self._frame_lock = Lock()
        self._feed_lock = Lock()

        self.emit_records: List[Dict[str, Any]] = []
        self.frame_log_records: List[Dict[str, Any]] = []
        self.feed_records: List[Dict[str, Any]] = []
        self.skipped_records: List[Dict[str, Any]] = []

        self._fed_frame_count = 0
        self._pipeline_fed_frame_count = 0
        self._frame_logger_subscribed = False
        self._recorded_frame_keys: set[Tuple[str, str]] = set()
        self._known_world_names: set[str] = set()
        self._active_session_world_name: Optional[str] = None
        self._shared_change_memory_state: Optional[Dict[str, Any]] = None
        self._shared_change_memory_source_world: Optional[str] = None
        self._latest_object_memory_state: Optional[Dict[str, Any]] = None
        self._latest_object_memory_world_name: str = ""

        self.output_recorder = BenchmarkOutputRecorder()

        self.visualization_backend = (VISUALIZATION_BACKEND or "desktop").strip().lower()
        if self.visualization_backend not in {"desktop", "web"}:
            self.visualization_backend = "desktop"

        self.memory_manager: Optional[BenchmarkMemoryManager] = None
        self.feature_factory: Optional[FeatureFactory] = None
        self.description_broker: Optional[BenchmarkDescriptionBroker] = None
        self.display_manager: Optional[Any] = None
        self.change_pipeline: Optional[BenchmarkChangeDetectionPipeline] = None
        self.ai_description_pipeline: Optional[BenchmarkAIParaphraseDescriptionPipeline] = None
        self.live_pipeline: Optional[LiveDescriptionPipeline] = None
        self.visualizer: Optional[Any] = None

        self._build_runtime_components()

    def _build_runtime_components(self) -> None:
        self.memory_manager = BenchmarkMemoryManager()
        self.feature_factory = FeatureFactory()
        self.description_broker = BenchmarkDescriptionBroker(
            emit_func=self._emit_description_text_only,
            on_emit=self._on_description_emitted,
        )

        if self.visualization_backend == "web":
            state_store = BenchmarkWebStateStore(self.clock)
            self.display_manager = WebDisplayManager(self.memory_manager, state_store=state_store)
        else:
            self.display_manager = DisplayManager()

        self.change_pipeline = BenchmarkChangeDetectionPipeline(
            self.memory_manager,
            self.display_manager,
            on_description=self.description_broker.publish_change,
            feature_factory=self.feature_factory,
        )
        if self.share_object_memory_across_captures and self._shared_change_memory_state is not None:
            self.change_pipeline.seed_change_memory(
                self._shared_change_memory_state,
                source_world_name=self._shared_change_memory_source_world,
            )

        self.ai_description_pipeline = None
        live_description_callback = self.description_broker.publish_live
        if DESCRIPTION_OUTPUT_MODE == "ai_paraphrase":
            candidate = BenchmarkAIParaphraseDescriptionPipeline(
                memory_manager=self.memory_manager,
                change_memory=self.change_pipeline.change_memory,
                on_description=self.description_broker.publish_change,
                speech_busy_until_getter=self.description_broker.get_busy_until,
                on_summary_ready=self._on_ai_summary_ready,
            )
            if candidate.enabled:
                self.ai_description_pipeline = candidate
                live_description_callback = candidate.add_live_description
                self.change_pipeline.on_description = None
                self.change_pipeline.on_change_snapshot = self._build_change_snapshot_forwarder(candidate)
                self.change_pipeline.use_ai_paraphrase = True

        if self.enable_live_descriptions:
            self.live_pipeline = LiveDescriptionPipeline(
                on_description=live_description_callback,
                feature_factory=self.feature_factory,
                on_emit_success=self._on_live_description_emitted,
                downstream_manages_final_emit=self.ai_description_pipeline is not None,
            )
        else:
            self.live_pipeline = None

        self.visualizer = None
        if VISUALIZATION_ENABLED:
            if self.visualization_backend == "web":
                state_store = getattr(self.display_manager, "state_store", None)
                self.visualizer = WebVisualizer(
                    self.memory_manager,
                    self.change_pipeline.change_memory,
                    state_store=state_store,
                )
            else:
                self.visualizer = Open3DVisualizer(self.memory_manager, self.change_pipeline.change_memory)

    def run(self) -> Dict[str, Path]:
        self._run_start_wall = datetime.now()
        self._run_start_perf = self.clock.start()

        try:
            for capture_index, eval_records in enumerate(self.capture_record_groups):
                session_world_name = self._session_world_name(capture_index)
                self._known_world_names.add(session_world_name)
                self._active_session_world_name = session_world_name
                self._pipeline_fed_frame_count = 0
                self._start_components()

                if capture_index > 0:
                    self._run_record_batch(
                        self.capture_record_groups[capture_index - 1],
                        world_name=session_world_name,
                        record_outputs=False,
                        respect_realtime=False,
                    )
                    if not self._wait_for_pipeline_drain():
                        self._pipeline_timeout_reached = True
                        LOGGER.warning("Pipeline drain timeout reached during warmup capture %s.", capture_index)
                        self._snapshot_object_memory(session_world_name=session_world_name)
                        break

                self._run_record_batch(
                    eval_records,
                    world_name=session_world_name,
                    record_outputs=True,
                    respect_realtime=(self.mode == "realtime"),
                )
                if not self._wait_for_pipeline_drain():
                    self._pipeline_timeout_reached = True
                    LOGGER.warning("Pipeline drain timeout reached during evaluated capture %s.", capture_index)
                    self._snapshot_object_memory(session_world_name=session_world_name)
                    break

                if not self._wait_for_ai_summary_drain():
                    self._pipeline_timeout_reached = True
                    LOGGER.warning("AI summary drain timeout reached during evaluated capture %s.", capture_index)
                self._broker_timeout_reached = (not self._drain_broker()) or self._broker_timeout_reached
                self._pump()
                self._snapshot_object_memory(session_world_name=session_world_name)
                self._shutdown_components()
                if capture_index + 1 < len(self.capture_record_groups):
                    self._build_runtime_components()

            if self.change_pipeline is not None:
                self._shutdown_components()
            output_files = self._write_outputs()
            self._wait_for_post_run_inspection()
            return output_files
        finally:
            self._shutdown_components()

    def _start_components(self) -> None:
        if self.display_manager is None or self.change_pipeline is None or self.description_broker is None:
            return
        self.display_manager.initialize()
        if self.visualizer is not None:
            self.visualizer.initialize()
        if not self._frame_logger_subscribed:
            FrameLogger.subscribe(self._on_frame_log)
            self._frame_logger_subscribed = True
        if self.ai_description_pipeline is not None:
            self.ai_description_pipeline.start()
        self.change_pipeline.start()
        if self.live_pipeline is not None:
            self.live_pipeline.start()

    def _shutdown_components(self) -> None:
        try:
            if self.live_pipeline is not None:
                self.live_pipeline.stop()
        except Exception:
            LOGGER.exception("Failed to stop live pipeline.")
        try:
            if self.change_pipeline is not None:
                self.change_pipeline.stop()
        except Exception:
            LOGGER.exception("Failed to stop change pipeline.")
        if self.ai_description_pipeline is not None:
            try:
                self.ai_description_pipeline.stop()
            except Exception:
                LOGGER.exception("Failed to stop AI paraphrase pipeline.")
        try:
            if self.description_broker is not None:
                self.description_broker.stop()
        except Exception:
            LOGGER.exception("Failed to stop description broker.")

        if self.visualizer is not None:
            try:
                self.visualizer.destroy()
            except Exception:
                LOGGER.exception("Failed to destroy visualizer.")

        try:
            if self.display_manager is not None:
                self.display_manager.close()
        except Exception:
            LOGGER.exception("Failed to close display manager.")

        try:
            if self.live_pipeline is not None:
                self.live_pipeline.join(timeout=5)
        except Exception:
            LOGGER.exception("Failed while joining live pipeline.")
        try:
            if self.change_pipeline is not None:
                self.change_pipeline.join(timeout=10)
        except Exception:
            LOGGER.exception("Failed while joining change pipeline.")
        if self.ai_description_pipeline is not None:
            try:
                self.ai_description_pipeline.join(timeout=5)
            except Exception:
                LOGGER.exception("Failed while joining AI paraphrase pipeline.")

        if self._frame_logger_subscribed:
            FrameLogger.unsubscribe(self._on_frame_log)
            self._frame_logger_subscribed = False

        try:
            if self.feature_factory is not None:
                self.feature_factory.close()
        except Exception:
            LOGGER.exception("Failed to close feature factory.")

        try:
            if self.memory_manager is not None:
                self.memory_manager.end_session()
        except Exception:
            LOGGER.exception("Failed to end memory session.")

        self.live_pipeline = None
        self.ai_description_pipeline = None
        self.change_pipeline = None
        self.description_broker = None
        self.display_manager = None
        self.visualizer = None
        self.feature_factory = None
        self.memory_manager = None
        self._active_session_world_name = None

    def _run_record_batch(
        self,
        records: Sequence[CaptureFrameRecord],
        *,
        world_name: str,
        record_outputs: bool,
        respect_realtime: bool,
    ) -> None:
        if not records:
            return
        respect_schedule = bool(record_outputs and self.feed_interval_sec is not None) or respect_realtime
        if record_outputs and self.feed_interval_sec is not None:
            schedule_offsets = [idx * self.feed_interval_sec for idx in range(len(records))]
        else:
            schedule_offsets = _build_schedule_offsets(list(records)) if respect_realtime else [0.0] * len(records)
        batch_start_perf = perf_counter()
        next_index = 0
        total_records = len(records)
        while next_index < total_records:
            now_perf = perf_counter()
            if self._ready_to_feed(
                now_perf,
                batch_start_perf,
                schedule_offsets,
                next_index,
                respect_schedule=respect_schedule,
            ):
                self._feed_record(
                    records[next_index],
                    world_name=world_name,
                    record_outputs=record_outputs,
                )
                next_index += 1
            self._pump()
            poll = DEFAULT_FAST_POLL_INTERVAL_SEC if self.mode == "fast" or not respect_realtime else DEFAULT_REALTIME_POLL_INTERVAL_SEC
            sleep(poll)

    def _ready_to_feed(
        self,
        now_perf: float,
        batch_start_perf: float,
        schedule_offsets: Sequence[float],
        record_index: int,
        *,
        respect_schedule: bool,
    ) -> bool:
        if self.change_pipeline is None:
            return False
        if respect_schedule and record_index < len(schedule_offsets):
            if now_perf < batch_start_perf + float(schedule_offsets[record_index]):
                return False
        return not self.change_pipeline.frame_queue.full()

    def _feed_record(
        self,
        record: CaptureFrameRecord,
        *,
        world_name: str,
        record_outputs: bool,
    ) -> None:
        if self.feature_factory is None or self.memory_manager is None or self.change_pipeline is None:
            return
        feed_start_wall = datetime.now()
        feed_start_perf = perf_counter()

        frame = build_frame(
            record=record,
            world_name=world_name,
            feature_factory=self.feature_factory,
        )
        feed_duration_s = perf_counter() - feed_start_perf
        feed_end_wall = datetime.now()

        if frame is None:
            if record_outputs:
                with self._feed_lock:
                    self.skipped_records.append(
                        {
                            "capture_name": record.capture_name,
                            "frame_index": record.frame_index,
                            "frame_timestamp": record.timestamp.isoformat(),
                            "reason": "failed_to_decode_or_build_frame",
                        }
                    )
            return

        self.clock.set_latest_frame_timestamp(frame.timestamp)
        self._latest_fed_timestamp = frame.timestamp

        with self._feed_lock:
            self._pipeline_fed_frame_count += 1
            if record_outputs:
                frame_timestamp = frame.timestamp.isoformat()
                self._fed_frame_count += 1
                self._recorded_frame_keys.add((str(world_name or ""), frame_timestamp))
                self.feed_records.append(
                    {
                        "capture_name": record.capture_name,
                        "frame_index": record.frame_index,
                        "frame_timestamp": frame_timestamp,
                        "feed_start_wall": feed_start_wall.isoformat(),
                        "feed_end_wall": feed_end_wall.isoformat(),
                        "feed_duration_s": round(feed_duration_s, 6),
                        "feed_elapsed_s": _elapsed_since_run_start(self._run_start_perf, feed_start_perf),
                    }
                )

        self.memory_manager.add_frame(frame)
        self.change_pipeline.add_frame(
            frame,
            ingress_wall=feed_start_wall,
            ingress_perf=feed_start_perf,
        )
        if record_outputs and self.live_pipeline is not None:
            self.live_pipeline.add_frame(frame)

    def _pump(self) -> None:
        if self.display_manager is None:
            return
        self.display_manager.pump()
        if self.visualizer is None:
            return
        try:
            alive = self.visualizer.update()
        except Exception:
            LOGGER.exception("Visualizer update failed.")
            alive = False

        if alive is False:
            try:
                self.visualizer.destroy()
            except Exception:
                LOGGER.exception("Visualizer destroy failed after close.")
            self.visualizer = None

    def _pipeline_idle(self) -> bool:
        with self._feed_lock:
            expected = self._pipeline_fed_frame_count

        if self.change_pipeline is None:
            return True

        with self.change_pipeline._state_lock:
            processed_count = len(self.change_pipeline.processed_frames)
            inflight_count = len(self.change_pipeline._inflight_frame_timestamps)
            queued_count = len(self.change_pipeline._queued_frame_timestamps)

        if processed_count < expected:
            return False
        if inflight_count != 0 or queued_count != 0:
            return False
        if not self.change_pipeline.frame_queue.empty():
            return False
        if not self.change_pipeline.vlm_result_queue.empty():
            return False
        return True

    def _wait_for_pipeline_drain(self) -> bool:
        deadline = perf_counter() + self.pipeline_drain_timeout_sec
        poll = DEFAULT_FAST_POLL_INTERVAL_SEC if self.mode == "fast" else DEFAULT_REALTIME_POLL_INTERVAL_SEC
        while perf_counter() < deadline:
            if self._pipeline_idle():
                return True
            self._pump()
            sleep(poll)
        return False

    def _broker_idle(self) -> bool:
        if self.description_broker is None:
            return True
        now_perf = perf_counter()
        broker = self.description_broker
        with broker._lock:
            has_pending = bool(broker._change_queue) or bool(broker._agent_queue) or broker._agent_waiting
            busy_until = broker._busy_until
        return (not has_pending) and now_perf >= busy_until

    def _drain_broker(self) -> bool:
        deadline = perf_counter() + self.broker_drain_timeout_sec
        poll = DEFAULT_FAST_POLL_INTERVAL_SEC if self.mode == "fast" else DEFAULT_REALTIME_POLL_INTERVAL_SEC
        while perf_counter() < deadline:
            if self._broker_idle():
                return True
            self._pump()
            sleep(poll)
        LOGGER.warning("Broker drain timeout reached.")
        return False

    def _ai_summary_idle(self) -> bool:
        if self.ai_description_pipeline is None:
            return True
        return self.ai_description_pipeline.is_idle()

    def _wait_for_ai_summary_drain(self) -> bool:
        deadline = perf_counter() + self.pipeline_drain_timeout_sec
        poll = DEFAULT_FAST_POLL_INTERVAL_SEC if self.mode == "fast" else DEFAULT_REALTIME_POLL_INTERVAL_SEC
        while perf_counter() < deadline:
            if self._ai_summary_idle():
                return True
            self._pump()
            sleep(poll)
        return False

    def _wait_for_post_run_inspection(self) -> None:
        if self.visualization_backend != "web" or self.visualizer is None:
            return

        host = getattr(self.visualizer, "host", "127.0.0.1")
        port = getattr(self.visualizer, "port", "")
        LOGGER.info(
            "Benchmark completed. Web visualizer remains available at http://%s:%s . Press Ctrl+C to exit.",
            host,
            port,
        )
        try:
            while True:
                self._pump()
                sleep(DEFAULT_REALTIME_POLL_INTERVAL_SEC)
        except KeyboardInterrupt:
            LOGGER.info("Stopping benchmark inspection mode.")

    def _build_change_snapshot_forwarder(self, candidate: AIParaphraseDescriptionPipeline):
        def _forward(payload: Dict[str, Any]) -> bool:
            current_timestamp = str(payload.get("current_timestamp", "") or "")
            session_world_name = self._active_session_world_name or ""
            if not self._should_record_frame(session_world_name, current_timestamp):
                return False
            try:
                self._record_change_snapshot(payload, final_description="", world_name=session_world_name)
            except Exception:
                LOGGER.exception("Failed to record benchmark change snapshot.")
            return candidate.add_change_snapshot(payload)

        return _forward

    def _emit_description_text_only(self, description: str) -> None:
        text = " ".join((description or "").split()).strip()
        if not text or self.display_manager is None:
            return

        simulated_time = self.clock.now()
        self.display_manager.add_description(text, simulated_time)
        LOGGER.info("[benchmark_emit] %s", text)

    def _on_description_emitted(self, *, text: str, source: str) -> None:
        emitted_wall = datetime.now()
        emitted_perf = perf_counter()
        simulated_time = self.clock.now()
        latest_frame_timestamp = self._latest_fed_timestamp
        latest_key = latest_frame_timestamp.isoformat() if latest_frame_timestamp is not None else ""
        session_world_name = self._active_session_world_name or ""
        if not self._should_record_frame(session_world_name, latest_key):
            return

        spoken_record = self.output_recorder.record_spoken_event(
            text=text,
            source=source,
            emit_wall_time=emitted_wall,
            emit_simulated_time=simulated_time,
            emit_elapsed_s=_elapsed_since_run_start(self._run_start_perf, emitted_perf),
            latest_frame_timestamp=latest_frame_timestamp,
        )
        emit_record = {
            "emit_index": spoken_record["spoken_index"],
            "source": spoken_record["source"],
            "text": spoken_record["text"],
            "emit_wall_time": spoken_record["emit_wall_time"],
            "emit_simulated_time": spoken_record["emit_simulated_time"],
            "emit_elapsed_s": spoken_record["emit_elapsed_s"],
            "latest_frame_timestamp": spoken_record["latest_frame_timestamp"],
        }
        with self._emit_lock:
            self.emit_records.append(emit_record)

    def _record_change_snapshot(self, payload: Dict[str, Any], final_description: str, *, world_name: str) -> None:
        current_timestamp = str(payload.get("current_timestamp", "") or "")
        if not self._should_record_frame(world_name, current_timestamp):
            return
        if self.change_pipeline is None:
            return
        mode = "ai_paraphrase" if self.change_pipeline.use_ai_paraphrase else "legacy"
        record = self.output_recorder.record_change_snapshot(
            payload,
            description_mode=mode,
            final_description=final_description,
        )
        if record is None:
            return

    def _on_ai_summary_ready(self, summary: str, payload: Dict[str, Any]) -> None:
        snapshots = payload.get("change_snapshots")
        if not isinstance(snapshots, list) or not snapshots:
            return

        latest_snapshot = next(
            (row for row in reversed(snapshots) if isinstance(row, dict) and isinstance(row.get("changes"), list) and row.get("changes")),
            None,
        )
        if latest_snapshot is None:
            return
        session_world_name = self._active_session_world_name or ""
        current_timestamp = str(latest_snapshot.get("current_timestamp", "") or "")
        if not self._should_record_frame(session_world_name, current_timestamp):
            return

        self.output_recorder.record_prediction(
            text=summary,
            description_mode="ai_paraphrase",
            current_timestamp=current_timestamp,
            reference_timestamp=latest_snapshot.get("reference_timestamp"),
            changes=latest_snapshot.get("changes"),
            recorded_from="ai_summary",
        )

    def _on_live_description_emitted(self, frame: Any) -> None:
        if self.change_pipeline is None:
            return
        timestamp = getattr(frame, "timestamp", None)
        self.change_pipeline.mark_live_described_frame(timestamp)
        setter = getattr(self.display_manager, "set_live_describing_frame", None) if self.display_manager is not None else None
        if callable(setter):
            try:
                setter(frame)
            except Exception:
                LOGGER.exception("Failed to update live describing frame in display manager.")

    def _on_frame_log(self, payload: Dict[str, Any]) -> None:
        world_name = str(payload.get("world_name", "") or "")
        if world_name not in self._known_world_names:
            return
        frame_timestamp = str(payload.get("frame_timestamp", "") or "")
        if not self._should_record_frame(world_name, frame_timestamp):
            return
        with self._frame_lock:
            self.frame_log_records.append(dict(payload))
        if self.change_pipeline is not None and self.change_pipeline.use_ai_paraphrase:
            self.output_recorder.update_change_description(
                current_timestamp=frame_timestamp,
                final_description=str(payload.get("final_description", "")),
            )
            return
        self.output_recorder.record_prediction(
            text=str(payload.get("final_description", "")),
            description_mode=str((payload.get("notes") or {}).get("description_mode", "")),
            current_timestamp=frame_timestamp,
            reference_timestamp=payload.get("reference_timestamp", ""),
            changes=payload.get("kept_changes", []),
            recorded_from="frame_log",
        )
        self.output_recorder.record_change_from_frame_log(payload)

    def _should_record_frame(self, world_name: str, frame_timestamp: str) -> bool:
        clean_world_name = str(world_name or "").strip()
        clean_timestamp = str(frame_timestamp or "").strip()
        if not clean_world_name or not clean_timestamp:
            return False
        with self._feed_lock:
            return (clean_world_name, clean_timestamp) in self._recorded_frame_keys

    def _snapshot_object_memory(self, *, session_world_name: str) -> None:
        if self.change_pipeline is None:
            return
        try:
            with self.change_pipeline._state_lock:
                payload = self.change_pipeline.change_memory.to_persist_dict()
        except Exception:
            LOGGER.exception("Failed to snapshot final OTM state for world %s.", session_world_name)
            return

        self._latest_object_memory_state = payload
        self._latest_object_memory_world_name = session_world_name
        if self.share_object_memory_across_captures:
            self._shared_change_memory_state = payload
            self._shared_change_memory_source_world = session_world_name

    def _session_world_name(self, capture_index: int) -> str:
        return f"{self.world_name}_capture_{int(capture_index):03d}"

    def _write_outputs(self) -> Dict[str, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        predictions_path = self.output_dir / "predictions.json"
        frames_path = self.output_dir / "frames.jsonl"
        manifest_path = self.output_dir / "manifest.json"
        object_memory_state_path = self.output_dir / "object_based_memory.state"
        object_memory_summary_path = self.output_dir / "object_based_memory_summary.json"

        predictions_payload = self.output_recorder.build_predictions_payload(
            dataset=str(self.dataset_dir),
            world_name=self.world_name,
            mode=self.mode,
            frame_stride=self.frame_stride,
            live_descriptions_enabled=self.enable_live_descriptions,
            description_output_mode=DESCRIPTION_OUTPUT_MODE,
            run_start_wall=self._run_start_wall,
            feed_records=self.feed_records,
        )
        self.output_recorder.write_json(predictions_path, predictions_payload)

        with self._feed_lock:
            feed_rows = [
                dict(row)
                for row in self.feed_records
            ]

        with self._frame_lock:
            frame_log_by_timestamp = {
                str(row.get("frame_timestamp", "")): dict(row)
                for row in self.frame_log_records
                if row.get("frame_timestamp")
            }

        with frames_path.open("w", encoding="utf-8") as handle:
            for feed_row in feed_rows:
                key = str(feed_row.get("frame_timestamp", ""))
                row = frame_log_by_timestamp.get(key, {})
                output_row: Dict[str, Any] = {
                    "capture_name": feed_row.get("capture_name", ""),
                    "frame_index": feed_row.get("frame_index"),
                    "frame_timestamp": key,
                    "feed_start_wall": feed_row.get("feed_start_wall", ""),
                    "feed_end_wall": feed_row.get("feed_end_wall", ""),
                    "feed_duration_s": feed_row.get("feed_duration_s"),
                    "feed_elapsed_s": feed_row.get("feed_elapsed_s"),
                    "processing_recorded": bool(row),
                    "reference_timestamp": row.get("reference_timestamp", ""),
                    "reference_found": bool(row.get("reference_found", False)),
                    "total_duration_s": row.get("total_duration_s"),
                    "end_to_end_duration_s": row.get("end_to_end_duration_s"),
                    "processing_duration_s": row.get("processing_duration_s"),
                    "queue_wait_total_s": row.get("queue_wait_total_s"),
                    "pre_pipeline_duration_s": row.get("pre_pipeline_duration_s"),
                    "frame_queue_wait_s": row.get("frame_queue_wait_s"),
                    "vlm_result_queue_wait_s": row.get("vlm_result_queue_wait_s"),
                    "frame_start_wall": row.get("frame_start_wall", ""),
                    "frame_end_wall": row.get("frame_end_wall", ""),
                    "ingress_wall": row.get("ingress_wall", ""),
                    "pipeline_enqueue_wall": row.get("pipeline_enqueue_wall", ""),
                    "pipeline_dequeue_wall": row.get("pipeline_dequeue_wall", ""),
                    "stage_timings_ordered": row.get("stage_timings_ordered", []),
                }
                handle.write(json.dumps(output_row, ensure_ascii=True))
                handle.write("\n")

        final_change_memory_state = self._latest_object_memory_state or {
            "objects": {},
            "next_object_id": 0,
        }
        object_memory_payload = {
            "schema_version": OBJECT_MEMORY_SCHEMA_VERSION,
            "dataset": str(self.dataset_dir),
            "world_name": self._latest_object_memory_world_name or self.world_name,
            "object_memory_scope": "dataset" if self.share_object_memory_across_captures else "capture",
            "share_object_memory_across_captures": self.share_object_memory_across_captures,
            "source_world_name": self._shared_change_memory_source_world if self.share_object_memory_across_captures else "",
            "change_memory": final_change_memory_state,
        }
        compressed_object_memory_payload = compress_object(object_memory_payload)
        with object_memory_state_path.open("wb") as handle:
            handle.write(compressed_object_memory_payload)

        final_object_memory_stats = _summarize_object_memory_payload(final_change_memory_state)
        final_object_memory_stats["compressed_bytes"] = len(compressed_object_memory_payload)

        object_memory_summary = {
            "schema_version": OBJECT_MEMORY_SCHEMA_VERSION,
            "dataset": str(self.dataset_dir),
            "world_name": self._latest_object_memory_world_name or self.world_name,
            "object_memory_scope": "dataset" if self.share_object_memory_across_captures else "capture",
            "share_object_memory_across_captures": self.share_object_memory_across_captures,
            **dict(final_object_memory_stats),
        }
        with object_memory_summary_path.open("w", encoding="utf-8") as handle:
            json.dump(object_memory_summary, handle, ensure_ascii=True, indent=2)

        with self._feed_lock:
            fed_count = self._fed_frame_count
            skipped = list(self.skipped_records)
            feed_count = len(self.feed_records)
        with self._frame_lock:
            frame_log_count = len(self.frame_log_records)
        with self._emit_lock:
            emit_count = len(self.emit_records)

        manifest = {
            "dataset": str(self.dataset_dir),
            "mode": self.mode,
            "frame_stride": self.frame_stride,
            "max_captures": self.max_captures,
            "live_descriptions_enabled": self.enable_live_descriptions,
            "description_output_mode": DESCRIPTION_OUTPUT_MODE,
            "latency_time_axis": "wall_elapsed_seconds",
            "world_name": self.world_name,
            "records_total": len(self.records),
            "frames_fed": fed_count,
            "feed_records_count": feed_count,
            "frame_log_count": frame_log_count,
            "spoken_emit_count": emit_count,
            "prediction_count": len(predictions_payload.get("records", [])),
            "skipped_records": skipped,
            "run_start_wall": self._run_start_wall.isoformat() if self._run_start_wall is not None else "",
            "simulated_start_time": self._first_timestamp.isoformat(),
            "simulated_end_time": self.clock.now().isoformat(),
            "share_object_memory_across_captures": self.share_object_memory_across_captures,
            "object_memory_scope": "dataset" if self.share_object_memory_across_captures else "capture",
            "object_memory_final_stats": dict(final_object_memory_stats),
            "object_memory_artifacts": {
                "state": object_memory_state_path.name,
                "summary": object_memory_summary_path.name,
            },
            "pipeline_timeout_reached": self._pipeline_timeout_reached,
            "broker_timeout_reached": self._broker_timeout_reached,
        }
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=True, indent=2)

        LOGGER.info("Benchmark output: %s", self.output_dir)
        return {
            "output_dir": self.output_dir,
            "predictions": predictions_path,
            "frames": frames_path,
            "object_memory_state": object_memory_state_path,
            "object_memory_summary": object_memory_summary_path,
            "manifest": manifest_path,
        }


def _safe_slug(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return cleaned or "benchmark"


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


def _group_records_by_capture(records: Sequence[CaptureFrameRecord]) -> List[List[CaptureFrameRecord]]:
    groups: List[List[CaptureFrameRecord]] = []
    current_group: List[CaptureFrameRecord] = []
    current_capture_name = ""
    for record in records:
        if not current_group or record.capture_name == current_capture_name:
            current_group.append(record)
            current_capture_name = record.capture_name
            continue
        groups.append(current_group)
        current_group = [record]
        current_capture_name = record.capture_name
    if current_group:
        groups.append(current_group)
    return groups


def _elapsed_since_run_start(run_start_perf: Optional[float], now_perf: float) -> Optional[float]:
    if run_start_perf is None:
        return None
    try:
        return round(max(0.0, float(now_perf) - float(run_start_perf)), 6)
    except Exception:
        return None


def _summarize_object_memory_payload(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    memory_payload = payload if isinstance(payload, dict) else {
        "objects": {},
        "next_object_id": 0,
    }
    objects = memory_payload.get("objects", {})
    if not isinstance(objects, dict):
        objects = {}

    object_count = 0
    active_object_count = 0
    disappeared_object_count = 0
    snapshot_count = 0

    for obj in objects.values():
        if not isinstance(obj, dict):
            continue
        object_count += 1
        snapshots = obj.get("snapshots", [])
        if not isinstance(snapshots, list):
            snapshots = []
        snapshot_count += len(snapshots)
        last_snapshot = snapshots[-1] if snapshots else {}
        if isinstance(last_snapshot, dict) and str(last_snapshot.get("change_type", "")).strip().lower() == "disappear":
            disappeared_object_count += 1
        else:
            active_object_count += 1

    return {
        "object_count": object_count,
        "active_object_count": active_object_count,
        "disappeared_object_count": disappeared_object_count,
        "snapshot_count": snapshot_count,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline benchmark with dataset captures.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR, help="Dataset directory containing capture folders.")
    parser.add_argument("--mode", choices=("realtime", "fast"), default="realtime", help="Feed mode for benchmark frames.")
    parser.add_argument(
        "--enable-live-descriptions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable live scene descriptions; defaults to on in realtime mode and off in fast mode.",
    )
    parser.add_argument("--feed-interval-sec", type=float, default=None, help="Optional fixed feed interval in seconds; applies only to recorded evaluation frames.")
    parser.add_argument("--frame-stride", type=int, default=DEFAULT_FRAME_STRIDE, help="Take every N valid frame.")
    parser.add_argument(
        "--max-captures",
        type=int,
        default=None,
        help="Read only first N captures in dataset; default reads all.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root output directory for benchmark runs.")
    parser.add_argument(
        "--use-aligned-metadata",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_ALIGNED_METADATA,
        help="Use metadata_aligned.jsonl when available.",
    )
    parser.add_argument(
        "--require-confidence",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_REQUIRE_CONFIDENCE,
        help="Require confidence frame files to exist when collecting valid frames.",
    )
    parser.add_argument(
        "--pipeline-drain-timeout-sec",
        type=float,
        default=DEFAULT_PIPELINE_DRAIN_TIMEOUT_SEC,
        help="Timeout waiting for change pipeline completion.",
    )
    parser.add_argument(
        "--broker-drain-timeout-sec",
        type=float,
        default=DEFAULT_BROKER_DRAIN_TIMEOUT_SEC,
        help="Timeout waiting for broker queue drain.",
    )
    parser.add_argument("--world-name-prefix", type=str, default="benchmark", help="Prefix for generated world name.")
    parser.add_argument(
        "--share-object-memory-across-captures",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Carry the previous capture's object-based memory into the next capture while keeping frame history limited to the current warmup/eval window.",
    )
    return parser


def main() -> int:
    setup_logging()
    parser = _build_parser()
    args = parser.parse_args()

    try:
        runner = BenchmarkRunner(
            dataset_dir=args.dataset,
            mode=args.mode,
            enable_live_descriptions=args.enable_live_descriptions,
            feed_interval_sec=args.feed_interval_sec,
            frame_stride=args.frame_stride,
            max_captures=args.max_captures,
            output_root=args.output_root,
            use_aligned_metadata=bool(args.use_aligned_metadata),
            require_confidence=bool(args.require_confidence),
            pipeline_drain_timeout_sec=args.pipeline_drain_timeout_sec,
            broker_drain_timeout_sec=args.broker_drain_timeout_sec,
            world_name_prefix=args.world_name_prefix,
            share_object_memory_across_captures=bool(args.share_object_memory_across_captures),
        )
        output_files = runner.run()
    except Exception:
        LOGGER.exception("Benchmark run failed.")
        return 1

    LOGGER.info("Benchmark finished.")
    for key, path in output_files.items():
        LOGGER.info("%s: %s", key, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
